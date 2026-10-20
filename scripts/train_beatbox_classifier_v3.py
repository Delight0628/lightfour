"""
Stage A v3 — beatbox 分类器（mel 域增强 + 稳定化）
v1: test 0.5008（Kick 95% / Snare 40% / 闭镲 43% / 开镲 8%）
v2: 类权重+波形增强 -> val 剧烈震荡(0.53~0.63)，发现致命缺陷：
    波形 pitch_shift 重采样后按原长裁剪 -> 截掉 hi-hat 衰减尾巴
    而"闭镲 vs 开镲"的唯一判别特征就是衰减时长 -> 增强直接毁掉关键特征
v3 修正：
  1. 频率增强改在【mel 域】做（沿 mel 轴 warp，时间轴完全不动）-> 保留衰减线索
     变调用 mel 域时间轴缩放（不改绝对时长太多）
  2. BatchNorm -> GroupNorm（消除 batch 间不稳定，val 震荡主因）
  3. LR 2e-3 -> 1e-3 + 权重 EMA（评估用 EMA，抹平 epoch 抖动）
  4. 早停 patience 8
运行： py -3.12 -u scripts/train_beatbox_classifier_v3.py
"""
import os, json, random, time, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(ROOT, "data", "sft", "beatbox_clips")
OUT_DIR = os.path.join(ROOT, "checkpoints", "beatbox_cls_v3")
os.makedirs(OUT_DIR, exist_ok=True)

SR, N_MELS, N_FFT, HOP = 22050, 64, 1024, 256
MAX_FRAMES = int(1.5 * SR / HOP) + 1
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}
EPOCHS, PATIENCE, EMA_DECAY = 26, 8, 0.995

random.seed(42); np.random.seed(42); torch.manual_seed(42)
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=40, f_max=SR // 2)
G_MEAN, G_STD = -4.0, 3.0

items = []
for fn in os.listdir(CLIP_DIR):
    if fn.endswith(".wav"):
        p = fn.split("_")[0]; c = fn[:-4].split("_")[-1]
        if c in CLASSES: items.append((os.path.join(CLIP_DIR, fn), p, c))
def split_of(p):
    if p in TEST_PARTS: return "test"
    if p in VAL_PARTS: return "val"
    return "train"
splits = defaultdict(list)
for path, p, c in items: splits[split_of(p)].append((path, c))
for k in ("train", "val", "test"):
    print(f"{k}: {len(splits[k])} {dict(Counter(c for _, c in splits[k]))}", flush=True)
cnt = Counter(c for _, c in splits["train"]); total = len(splits["train"])
class_weight = torch.tensor([total / (len(CLASSES) * cnt[c]) for c in CLASSES], dtype=torch.float32)

# ---------------- 增强（mel 域为主，保时间轴） ----------------
def mel_freq_warp(lm, lo=0.88, hi=1.14):
    """沿 mel 频率轴缩放：模拟不同嗓音/音高，时间轴（衰减）完全不动"""
    f = np.random.uniform(lo, hi)
    F_, T = lm.shape
    n_new = max(8, int(F_ * f))
    dst = np.linspace(0, F_ - 1, n_new)
    i0 = np.clip(np.floor(dst).astype(int), 0, F_ - 1)
    i1 = np.clip(i0 + 1, 0, F_ - 1)
    frac = (dst - i0)[:, None]
    out = lm[i0] * (1.0 - frac) + lm[i1] * frac
    if n_new >= F_: out = out[:F_]
    else: out = np.pad(out, ((0, F_ - n_new), (0, 0)))
    return out

def mel_time_scale(lm, lo=0.92, hi=1.09):
    """轻微时间缩放（只改整体时长一点点，避免抹掉衰减差异）"""
    f = np.random.uniform(lo, hi)
    F_, T = lm.shape
    n_new = max(8, int(T * f))
    dst = np.linspace(0, T - 1, n_new)
    i0 = np.clip(np.floor(dst).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    frac = (dst - i0)[None, :]
    out = lm[:, i0] * (1.0 - frac) + lm[:, i1] * frac
    if n_new >= T: out = out[:, :T]
    else: out = np.pad(out, ((0, 0), (0, T - n_new)))
    return out

def spec_augment(lm):
    T, F_ = lm.shape
    if np.random.rand() < 0.5:
        t = np.random.randint(0, 16); t0 = np.random.randint(0, max(1, T - t)); lm[:, t0:t0 + t] = 0
    if np.random.rand() < 0.5:
        f = np.random.randint(0, 7); f0 = np.random.randint(0, max(1, F_ - f)); lm[f0:f0 + f, :] = 0
    return lm

def wav_aug(wav):
    wav = wav * np.random.uniform(0.7, 1.35)
    if np.random.rand() < 0.6:
        snr = np.random.uniform(20, 45)
        p = np.mean(wav ** 2) + 1e-10
        wav = wav + np.random.randn(len(wav)).astype("float32") * np.sqrt(p / (10 ** (snr / 10)))
    return wav.astype("float32")

def load_logmel(path, train=False):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        n = int(len(wav) * SR / sr)
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    if train: wav = wav_aug(wav)
    t = torch.from_numpy(np.ascontiguousarray(wav))
    if t.numel() < 16: t = F.pad(t, (0, 16 - t.numel()))
    lm = torch.log(mel_tf(t) + 1e-6).numpy()
    fr = lm.shape[1]
    if fr < MAX_FRAMES: lm = np.pad(lm, ((0, 0), (0, MAX_FRAMES - fr)))
    else: lm = lm[:, :MAX_FRAMES]
    if train:
        lm = mel_freq_warp(lm)
        lm = mel_time_scale(lm)
        lm = spec_augment(lm)
    lm = (lm - G_MEAN) / G_STD
    return torch.from_numpy(lm.astype("float32"))

class DS(Dataset):
    def __init__(self, s, train): self.s = s; self.train = train
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        path, c = self.s[i]
        return load_logmel(path, self.train), CLASSES.index(c)

# ---------------- 模型（GroupNorm 稳定） ----------------
def gn(c): return nn.GroupNorm(min(8, c), c)
class MelCNN(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), gn(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), gn(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), gn(128), nn.ReLU())
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)), nn.AdaptiveMaxPool2d((1, 1))])
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(256, n))
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        z = self.f(x)
        return self.head(torch.cat([p(z).flatten(1) for p in self.pool], 1))

def evaluate(model, loader):
    model.eval(); cm = np.zeros((4, 4), dtype=int); confs = []
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(model(x), 1); pr = p.argmax(1)
            for t, q in zip(y.tolist(), pr.tolist()): cm[t, q] += 1
            confs += p.max(1).values.tolist()
    return np.trace(cm) / max(cm.sum(), 1), cm, float(np.mean(confs))

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(DS(splits["train"], True), batch_size=64, shuffle=True, num_workers=3, drop_last=True)
    va = DataLoader(DS(splits["val"], False), batch_size=64, num_workers=0)
    te = DataLoader(DS(splits["test"], False), batch_size=64, num_workers=0)
    model = MelCNN().to(dev)
    ema = copy.deepcopy(model); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=class_weight.to(dev), label_smoothing=0.05)

    best_val, best_ep, bad = 0.0, 0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = nb = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            lam, perm = 0.8, torch.randperm(x.size(0))
            mix = np.random.rand() < 0.5
            if mix: x = lam * x + (1 - lam) * x[perm]
            opt.zero_grad(); out = model(x)
            loss = (lam * crit(out, y) + (1 - lam) * crit(out, y[perm])) if mix else crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            tot += loss.item(); nb += 1
        sched.step()
        vacc, vcm, _ = evaluate(ema, va)
        pc = [vcm[i, i] / max(vcm[i].sum(), 1) for i in range(4)]
        if vacc > best_val:
            best_val, best_ep, bad = vacc, ep, 0
            torch.save(ema.state_dict(), os.path.join(OUT_DIR, "best.pt"))
        else: bad += 1
        print(f"ep{ep:02d} loss {tot/max(nb,1):.4f} val {vacc:.4f} best {best_val:.4f}(ep{best_ep}) "
              f"[{' '.join(f'{CLASS_CN[CLASSES[i]]}:{pc[i]:.2f}' for i in range(4))}] {time.time()-t0:.0f}s", flush=True)
        if bad >= PATIENCE:
            print(f"早停：连续 {PATIENCE} epoch 无提升", flush=True); break

    ema.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt")))
    tacc, cm, tconf = evaluate(ema, te); vacc, vcm, _ = evaluate(ema, va)
    print("\n" + "=" * 62, flush=True)
    print(f"最佳 epoch={best_ep}  val={vacc:.4f}  test={tacc:.4f}  (随机 0.25, v1=0.5008)", flush=True)
    for i, c in enumerate(CLASSES):
        print(f"  {CLASS_CN[c]:4s} {cm[i,i]/max(cm[i].sum(),1):.4f} ({cm[i,i]}/{cm[i].sum()})", flush=True)
    print("混淆矩阵(行=真,列=预测):", flush=True); print(cm, flush=True)
    ema.eval()
    with torch.no_grad():
        zp = F.softmax(ema(torch.zeros(1, 1, N_MELS, MAX_FRAMES)), 1).squeeze()
    print(f"静音 -> {CLASS_CN[CLASSES[int(zp.argmax())]]} 置信 {zp.max():.3f} (test 均值 {tconf:.3f})", flush=True)
    rep = {"best_epoch": best_ep, "val_acc": vacc, "test_acc": tacc, "random_baseline": 0.25,
           "v1_test_acc": 0.5008,
           "per_class_acc": {CLASS_CN[c]: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
           "confusion_matrix": cm.tolist(), "class_order": CLASSES,
           "silence_pred": CLASS_CN[CLASSES[int(zp.argmax())]], "silence_conf": float(zp.max()),
           "test_mean_conf": tconf}
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"-> {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
