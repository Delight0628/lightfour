"""
Stage A v2 — beatbox 分类器（抗跨人泛化版）
v1 结果：test 0.5008（随机 0.25），Kick 95% / Snare 40.5% / 闭镲 42.8% / 开镲 7.8%
根因：①类别偏置（偏预测 Kick）②跨人泛化差（val 0.72 vs test 0.50）③严重过拟合（epoch5 后 val 崩）
对策：类权重 + 波形增强(增益/噪/变速/变调) + SpecAugment + Mixup + label smoothing
      + 全局固定归一化(保留响度线索) + Avg&Max 池化 + 早停
运行： py -3.12 -u scripts/train_beatbox_classifier_v2.py
"""
import os, sys, json, random, time
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
OUT_DIR = os.path.join(ROOT, "checkpoints", "beatbox_cls_v2")
os.makedirs(OUT_DIR, exist_ok=True)

SR, N_MELS, N_FFT, HOP = 22050, 64, 1024, 256
MAX_FRAMES = int(1.5 * SR / HOP) + 1
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}
EPOCHS, PATIENCE = 22, 6

random.seed(42); np.random.seed(42); torch.manual_seed(42)
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=40, f_max=SR // 2)
G_MEAN, G_STD = -4.0, 3.0   # log-mel 全局归一化常数（保留绝对响度线索，区分 Snare/HiHat）

# ---------------- 数据 ----------------
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

cnt = Counter(c for _, c in splits["train"])
total = len(splits["train"])
class_weight = torch.tensor([total / (len(CLASSES) * cnt[c]) for c in CLASSES], dtype=torch.float32)
print(f"class_weight: {[round(float(w),3) for w in class_weight]}", flush=True)

# ---------------- 增强 ----------------
def pitch_shift(wav, factor):
    n = len(wav); n_new = max(8, int(n / factor))
    sh = np.interp(np.linspace(0, 1, n_new, endpoint=False),
                   np.linspace(0, 1, n, endpoint=False), wav)
    if len(sh) >= n: sh = sh[:n]
    else: sh = np.pad(sh, (0, n - len(sh)))
    return sh.astype("float32")

def augment(wav):
    wav = wav * np.random.uniform(0.65, 1.4)                      # 增益
    if np.random.rand() < 0.6:                                     # 加噪
        snr = np.random.uniform(18, 45)
        p = np.mean(wav ** 2) + 1e-10
        wav = wav + np.random.randn(len(wav)).astype("float32") * np.sqrt(p / (10 ** (snr / 10)))
    if np.random.rand() < 0.55:                                    # 变调（保持时长，跨嗓音鲁棒）
        wav = pitch_shift(wav, np.random.uniform(0.84, 1.19))
    if np.random.rand() < 0.3:                                     # 变速（改时长）
        rate = np.random.uniform(0.85, 1.15)
        n_new = max(8, int(len(wav) * rate))
        wav = np.interp(np.linspace(0, 1, n_new, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    if np.random.rand() < 0.4 and len(wav) > 64:                   # 时间平移
        wav = np.roll(wav, np.random.randint(-len(wav) // 8, len(wav) // 8))
    return wav.astype("float32")

def spec_augment(lm):
    T, F = lm.shape
    if np.random.rand() < 0.5:
        t = np.random.randint(0, 21); t0 = np.random.randint(0, max(1, T - t)); lm[:, t0:t0 + t] = 0
    if np.random.rand() < 0.5:
        f = np.random.randint(0, 9); f0 = np.random.randint(0, max(1, F - f)); lm[f0:f0 + f, :] = 0
    return lm

def load_logmel(path, train=False):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        n = int(len(wav) * SR / sr)
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    if train: wav = augment(wav)
    t = torch.from_numpy(np.ascontiguousarray(wav))
    if t.numel() < 16: t = F.pad(t, (0, 16 - t.numel()))
    lm = torch.log(mel_tf(t) + 1e-6)
    fr = lm.shape[-1]
    lm = F.pad(lm, (0, MAX_FRAMES - fr)) if fr < MAX_FRAMES else lm[:, :MAX_FRAMES]
    if train: lm = spec_augment(lm)
    return (lm - G_MEAN) / G_STD

class DS(Dataset):
    def __init__(self, s, train): self.s = s; self.train = train
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        path, c = self.s[i]
        return load_logmel(path, self.train), CLASSES.index(c)

# ---------------- 模型 ----------------
class MelCNN(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)), nn.AdaptiveMaxPool2d((1, 1))])
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, n))
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        z = self.f(x)
        return self.head(torch.cat([p(z).flatten(1) for p in self.pool], 1))

def evaluate(model, loader):
    model.eval(); cm = np.zeros((4, 4), dtype=int); confs = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(model(x.to(dev)), 1); pr = p.argmax(1).cpu()
            for t, q in zip(y.tolist(), pr.tolist()): cm[t, q] += 1
            confs += p.max(1).values.tolist()
    acc = np.trace(cm) / max(cm.sum(), 1)
    return acc, cm, float(np.mean(confs))

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    tr = DataLoader(DS(splits["train"], True), batch_size=64, shuffle=True, num_workers=3, drop_last=True)
    va = DataLoader(DS(splits["val"], False), batch_size=64, num_workers=0)
    te = DataLoader(DS(splits["test"], False), batch_size=64, num_workers=0)

    model = MelCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=class_weight.to(dev), label_smoothing=0.05)

    best_val, best_ep, bad, best_cm = 0.0, 0, 0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = nb = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            lam, perm = 0.7, torch.randperm(x.size(0))
            if np.random.rand() < 0.5:                     # Mixup
                x = lam * x + (1 - lam) * x[perm]
            opt.zero_grad()
            out = model(x)
            if lam != 1.0:
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y[perm])
            else:
                loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        sched.step()
        vacc, vcm, vconf = evaluate(model, va)
        per_cls = [vcm[i, i] / max(vcm[i].sum(), 1) for i in range(4)]
        if vacc > best_val:
            best_val, best_ep, bad = vacc, ep, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
            best_cm = vcm.copy()
        else:
            bad += 1
        print(f"ep{ep:02d} loss {tot/max(nb,1):.4f} val {vacc:.4f} best {best_val:.4f}(ep{best_ep}) "
              f"percls[{' '.join(f'{CLASS_CN[CLASSES[i]]}:{per_cls[i]:.2f}' for i in range(4))}] "
              f"vconf {vconf:.3f} {time.time()-t0:.0f}s", flush=True)
        if bad >= PATIENCE:
            print(f"早停：连续 {PATIENCE} epoch 无提升", flush=True); break

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt")))
    tacc, cm, tconf = evaluate(model, te)
    vacc, vcm, vconf = evaluate(model, va)
    print("\n" + "=" * 62, flush=True)
    print(f"最佳 epoch={best_ep}  val_acc={vacc:.4f}  test_acc={tacc:.4f}  (随机 0.25)", flush=True)
    print("每类 test 准确率:", flush=True)
    for i, c in enumerate(CLASSES):
        print(f"  {CLASS_CN[c]:4s} {cm[i,i]/max(cm[i].sum(),1):.4f}  ({cm[i,i]}/{cm[i].sum()})", flush=True)
    print(f"混淆矩阵 (行=真,列=预测 {CLASSES}):", flush=True)
    print(cm, flush=True)
    print(f"test 平均置信度 {tconf:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        zp = F.softmax(model(torch.zeros(1, 1, N_MELS, MAX_FRAMES, device=dev)), 1).squeeze().cpu()
    print(f"静音输入 -> {CLASS_CN[CLASSES[int(zp.argmax())]]} 置信 {zp.max():.3f} (应显著低于 test 均值 {tconf:.3f})", flush=True)

    rep = {"best_epoch": best_ep, "val_acc": vacc, "test_acc": tacc, "random_baseline": 0.25,
           "per_class_acc": {CLASS_CN[c]: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
           "confusion_matrix": cm.tolist(), "class_order": CLASSES,
           "silence_pred": CLASS_CN[CLASSES[int(zp.argmax())]], "silence_conf": float(zp.max()),
           "test_mean_conf": tconf, "v1_test_acc": 0.5008}
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"-> {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
