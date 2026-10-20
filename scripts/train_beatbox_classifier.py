"""
Stage A — 专用 beatbox 音色分类器（log-mel CNN）
目标：在 AVP-LVT 单音切片上真正“听音频”分类 Kick/Snare/闭镲/开镲，报真实准确率 vs 随机基线。
- 划分按参与者留出（防同人泄漏），test 对齐论文评测集 P9/P11/P19/P24
- 验收：test 准确率 ≥ 70%（4 类随机 25%）+ 音频敏感性（零输入/静音置信度应低）
运行： py -3.12 scripts/train_beatbox_classifier.py
"""
import os, sys, json, math, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 防 Windows OpenMP 多线程死锁
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(ROOT, "data", "sft", "beatbox_clips")
OUT_DIR = os.path.join(ROOT, "checkpoints", "beatbox_cls")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 22050
N_MELS = 64
N_FFT = 1024
HOP = 256
MAX_SEC = 1.5
MAX_FRAMES = int(MAX_SEC * SR / HOP) + 1  # ~130
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓(Kick)", "sd": "军鼓(Snare)", "hhc": "闭镲(Closed Hi-Hat)", "hho": "开镲(Open Hi-Hat)"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}  # 从训练参与者里留作验证

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def parse_name(fn):
    base = os.path.splitext(fn)[0]
    parts = base.split("_")
    participant = parts[0]
    cls = parts[-1]
    return participant, cls

# ---------- 收集样本 ----------
items = []
for fn in os.listdir(CLIP_DIR):
    if not fn.endswith(".wav"):
        continue
    p, c = parse_name(fn)
    if c not in CLASSES:
        continue
    items.append((os.path.join(CLIP_DIR, fn), p, c))
print(f"总样本: {len(items)}  类别分布: {Counter(c for _,_,c in items)}")

def split_of(participant):
    if participant in TEST_PARTS: return "test"
    if participant in VAL_PARTS: return "val"
    return "train"

splits = defaultdict(list)
for path, p, c in items:
    splits[split_of(p)].append((path, c))
for k in ["train", "val", "test"]:
    print(f"{k}: {len(splits[k])}  分布 {Counter(c for _,c in splits[k])}")

# ---------- 特征 ----------
mel = torchaudio_mel = None
import torchaudio
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=40, f_max=SR//2
)

def load_logmel(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        # 线性重采样（无需 librosa）
        n_new = int(len(wav) * SR / sr)
        x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
        x_new = np.linspace(0, 1, num=n_new, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype("float32")
    t = torch.from_numpy(wav)
    if t.numel() < 16:
        t = F.pad(t, (0, 16 - t.numel()))
    mel = mel_tf(t)  # (n_mels, frames)
    logmel = torch.log(mel + 1e-6)
    # 固定帧长（pad/截断）
    frames = logmel.shape[-1]
    if frames < MAX_FRAMES:
        logmel = F.pad(logmel, (0, MAX_FRAMES - frames))
    else:
        logmel = logmel[:, :MAX_FRAMES]
    # 全局标准化
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel

class BeatboxDS(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, c = self.samples[i]
        x = load_logmel(path)
        y = CLASSES.index(c)
        return x, y

# ---------- 模型 ----------
class MelCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        f = self.features(x).flatten(1)
        return self.classifier(f)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    conf_mat = np.zeros((4,4), dtype=int)
    all_true, all_pred, all_conf = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(1)
            correct += (pred == y).sum().item(); total += y.numel()
            for t, p in zip(y.tolist(), pred.tolist()):
                conf_mat[t, p] += 1
            all_true += y.tolist(); all_pred += pred.tolist(); all_conf += prob.max(1).values.tolist()
    acc = correct / max(total, 1)
    return acc, conf_mat, all_true, all_pred, all_conf

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    train_loader = DataLoader(BeatboxDS(splits["train"]), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(BeatboxDS(splits["val"]), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(BeatboxDS(splits["test"]), batch_size=64, shuffle=False, num_workers=0)

    model = MelCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    t0 = time.time()
    for epoch in range(1, 26):
        model.train()
        run_loss = 0.0; nb = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            run_loss += loss.item(); nb += 1
        sched.step()
        val_acc, _, _, _, _ = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:2d}  train_loss {run_loss/max(nb,1):.4f}  val_acc {val_acc:.4f}  best {best_val:.4f}  ({time.time()-t0:.0f}s)")

    # 最终测试
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt")))
    test_acc, conf_mat, true_l, pred_l, conf_l = evaluate(model, test_loader)
    val_acc, _, _, _, _ = evaluate(model, val_loader)

    print("\n" + "="*60)
    print(f"训练完成  ({time.time()-t0:.0f}s)")
    print(f"val_acc  = {val_acc:.4f}")
    print(f"test_acc = {test_acc:.4f}   (随机基线 0.2500)  提升 {test_acc-0.25:+.4f}")
    print(f"每类准确率:")
    for i, c in enumerate(CLASSES):
        row_sum = conf_mat[i].sum()
        per_cls = conf_mat[i,i]/row_sum if row_sum else 0
        print(f"  {CLASS_CN[c]:22s}  {per_cls:.4f}  ({conf_mat[i,i]}/{row_sum})")
    print(f"混淆矩阵 (行=真实, 列=预测, 序 {CLASSES}):")
    print(conf_mat)

    # 音频敏感性：零输入（静音）应低置信，确认模型依赖音频内容
    model.eval()
    with torch.no_grad():
        zero_in = torch.zeros(1, 1, N_MELS, MAX_FRAMES)
        logits = model(zero_in)
        prob = F.softmax(logits, 1).squeeze()
        zero_conf = prob.max().item(); zero_pred = CLASSES[prob.argmax().item()]
    print(f"\n音频敏感性检查:")
    print(f"  纯静音输入 -> 预测 {CLASS_CN.get(zero_pred, zero_pred)}  置信度 {zero_conf:.4f} (应明显低于正常样本平均 {np.mean(conf_l):.4f})")
    print(f"  正常样本平均置信度 {np.mean(conf_l):.4f}")

    # 保存结果
    report = {
        "val_acc": val_acc, "test_acc": test_acc, "random_baseline": 0.25,
        "per_class_acc": {CLASS_CN[c]: float(conf_mat[i,i]/conf_mat[i].sum()) if conf_mat[i].sum() else 0 for i,c in enumerate(CLASSES)},
        "confusion_matrix": conf_mat.tolist(), "class_order": CLASSES,
        "zero_input_pred": zero_pred, "zero_input_conf": float(zero_conf),
        "normal_mean_conf": float(np.mean(conf_l)),
        "splits": {k: len(v) for k, v in splits.items()},
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n模型与报告已保存到 {OUT_DIR}")

if __name__ == "__main__":
    main()
