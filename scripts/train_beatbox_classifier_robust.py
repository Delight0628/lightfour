"""
Stage A robust：在 v2 基础上加入真实视频的 no_vocals 背景，做域随机化训练。

动机：
  AVP-LVT 是安静环境下的单音；真实 B 站视频带伴奏/欢呼/房间感。
  v2 在真实切片上置信度只有 0.45，几乎无法伪标注。
  用 Demucs 的 no_vocals.wav 当背景，按随机 SNR 混入训练片段，
  让模型学会在真实干扰下抓鼓点。

运行：
  /root/miniconda3/bin/python scripts/train_beatbox_classifier_robust.py \
    --bg-dir data/bbox_raw/demucs_srv --epochs 15
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(8)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(ROOT, "data", "sft", "beatbox_clips")
OUT_DIR = os.path.join(ROOT, "checkpoints", "beatbox_cls_robust")
os.makedirs(OUT_DIR, exist_ok=True)

SR, N_MELS, N_FFT, HOP = 22050, 64, 1024, 256
MAX_FRAMES = int(1.5 * SR / HOP) + 1
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}
G_MEAN, G_STD = -4.0, 3.0
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=40, f_max=SR // 2)


def split_of(p):
    if p in TEST_PARTS:
        return "test"
    if p in VAL_PARTS:
        return "val"
    return "train"


def load_bg(path, n):
    info = sf.info(path)
    dur = info.frames / max(info.samplerate, 1)
    start = random.uniform(0, max(0.0, dur - n / SR))
    b, bsr = sf.read(path, start=int(start * bsr) if False else int(start * info.samplerate),
                     frames=n, dtype="float32", always_2d=False)
    if b.ndim > 1:
        b = b.mean(axis=1)
    if bsr != SR:
        m = int(len(b) * SR / bsr)
        b = np.interp(np.linspace(0, 1, m, endpoint=False),
                      np.linspace(0, 1, len(b), endpoint=False), b).astype("float32")
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    return b[:n]


def augment(wav, bg_files):
    wav = wav * random.uniform(0.75, 1.35)
    if random.random() < 0.5:
        snr = random.uniform(20, 45)
        p = np.mean(wav ** 2) + 1e-10
        wav = wav + np.random.randn(len(wav)).astype("float32") * np.sqrt(p / (10 ** (snr / 10)))
    if bg_files and random.random() < 0.7:
        bg = load_bg(random.choice(bg_files), len(wav))
        snr = random.uniform(0, 20)
        rw = np.sqrt(np.mean(wav ** 2) + 1e-10)
        rb = np.sqrt(np.mean(bg ** 2) + 1e-10)
        wav = wav + bg * (rw / rb) * (10 ** (-snr / 20))
    if random.random() < 0.3 and len(wav) > 64:
        wav = np.roll(wav, random.randint(-len(wav) // 10, len(wav) // 10))
    return wav.astype("float32")


def spec_augment(lm):
    T, F_ = lm.shape
    if random.random() < 0.5:
        t = random.randint(0, 18)
        t0 = random.randint(0, max(1, T - t))
        lm[:, t0:t0 + t] = 0
    if random.random() < 0.5:
        f = random.randint(0, 8)
        f0 = random.randint(0, max(1, F_ - f))
        lm[f0:f0 + f, :] = 0
    return lm


def load_logmel(path, train, bg_files):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        n = int(len(wav) * SR / sr)
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    if train:
        wav = augment(wav, bg_files)
    t = torch.from_numpy(np.ascontiguousarray(wav))
    if t.numel() < 16:
        t = F.pad(t, (0, 16 - t.numel()))
    lm = torch.log(mel_tf(t) + 1e-6)
    fr = lm.shape[-1]
    lm = F.pad(lm, (0, MAX_FRAMES - fr)) if fr < MAX_FRAMES else lm[:, :MAX_FRAMES]
    if train:
        lm = torch.from_numpy(spec_augment(lm.numpy()))
    return (lm - G_MEAN) / G_STD


class DS(Dataset):
    def __init__(self, items, train, bg_files):
        self.items, self.train, self.bg_files = items, train, bg_files

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, c = self.items[i]
        return load_logmel(path, self.train, self.bg_files), CLASSES.index(c)


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class MelCNN(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), gn(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), gn(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), gn(128), nn.ReLU())
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)), nn.AdaptiveMaxPool2d((1, 1))])
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.f(x)
        return self.head(torch.cat([p(z).flatten(1) for p in self.pool], 1))


def evaluate(model, loader):
    model.eval()
    cm = np.zeros((4, 4), dtype=int)
    confs = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(model(x.to(dev)), 1)
            pr = p.argmax(1).cpu()
            for t, q in zip(y.tolist(), pr.tolist()):
                cm[t, q] += 1
            confs += p.max(1).values.tolist()
    return np.trace(cm) / max(cm.sum(), 1), cm, float(np.mean(confs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-dir", default=os.path.join(ROOT, "data", "bbox_raw", "demucs_srv"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    bg_files = []
    for root, _dirs, files in os.walk(args.bg_dir):
        for f in files:
            if f == "no_vocals.wav":
                bg_files.append(os.path.join(root, f))
    print(f"bg_files={len(bg_files)}", flush=True)

    items = []
    for fn in os.listdir(CLIP_DIR):
        if fn.endswith(".wav"):
            p = fn.split("_")[0]; c = fn[:-4].split("_")[-1]
            if c in CLASSES:
                items.append((os.path.join(CLIP_DIR, fn), p, c))
    splits = defaultdict(list)
    for path, p, c in items:
        splits[split_of(p)].append((path, c))
    for k in ("train", "val", "test"):
        print(f"{k}: {len(splits[k])} {dict(Counter(c for _, c in splits[k]))}", flush=True)

    cnt = Counter(c for _, c in splits["train"])
    total = len(splits["train"])
    class_weight = torch.tensor([total / (len(CLASSES) * cnt[c]) for c in CLASSES], dtype=torch.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    tr = DataLoader(DS(splits["train"], True, bg_files), batch_size=args.batch_size,
                    shuffle=True, num_workers=3, drop_last=True)
    va = DataLoader(DS(splits["val"], False, []), batch_size=64, num_workers=0)
    te = DataLoader(DS(splits["test"], False, []), batch_size=64, num_workers=0)

    model = MelCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=class_weight.to(dev), label_smoothing=0.05)

    best_val, best_ep, bad = 0.0, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = nb = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        vacc, vcm, vconf = evaluate(model, va)
        pc = [vcm[i, i] / max(vcm[i].sum(), 1) for i in range(4)]
        if vacc > best_val:
            best_val, best_ep, bad = vacc, ep, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
        else:
            bad += 1
        print(f"ep{ep:02d} loss {tot/max(nb,1):.4f} val {vacc:.4f} best {best_val:.4f}(ep{best_ep}) "
              f"[{' '.join(f'{CLASS_CN[CLASSES[i]]}:{pc[i]:.2f}' for i in range(4))}] "
              f"{time.time()-t0:.0f}s", flush=True)
        if bad >= args.patience:
            print("early stop", flush=True)
            break

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt")))
    tacc, cm, tconf = evaluate(model, te)
    vacc, vcm, _ = evaluate(model, va)
    print(f"\nBEST ep={best_ep} val={vacc:.4f} test={tacc:.4f} random=0.25", flush=True)
    for i, c in enumerate(CLASSES):
        print(f"  {CLASS_CN[c]} {cm[i,i]/max(cm[i].sum(),1):.4f} ({cm[i,i]}/{cm[i].sum()})", flush=True)
    rep = {"best_epoch": best_ep, "val_acc": float(vacc), "test_acc": float(tacc),
           "per_class": {CLASS_CN[c]: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
           "confusion_matrix": cm.tolist(), "bg_files": len(bg_files),
           "test_mean_conf": tconf}
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
