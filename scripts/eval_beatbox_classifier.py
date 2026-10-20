"""
通用评测：加载 beatbox 分类器 checkpoint，在 participant-held-out test 集上评测
用法： py -3.12 -u scripts/eval_beatbox_classifier.py --ckpt checkpoints/beatbox_cls_v3/best.pt
"""
import os, json, argparse
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
SR, N_MELS, N_FFT, HOP = 22050, 64, 1024, 256
MAX_FRAMES = int(1.5 * SR / HOP) + 1
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}
G_MEAN, G_STD = -4.0, 3.0
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=40, f_max=SR // 2)

def load_logmel(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR:
        n = int(len(wav) * SR / sr)
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    t = torch.from_numpy(np.ascontiguousarray(wav))
    if t.numel() < 16: t = F.pad(t, (0, 16 - t.numel()))
    lm = torch.log(mel_tf(t) + 1e-6)
    fr = lm.shape[-1]
    lm = F.pad(lm, (0, MAX_FRAMES - fr)) if fr < MAX_FRAMES else lm[:, :MAX_FRAMES]
    return (lm - G_MEAN) / G_STD

class DS(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        path, c = self.s[i]; return load_logmel(path), CLASSES.index(c)

USE_BN = False
def gn(c): return nn.BatchNorm2d(c) if USE_BN else nn.GroupNorm(min(8, c), c)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    splits = defaultdict(list)
    for fn in os.listdir(CLIP_DIR):
        if not fn.endswith(".wav"): continue
        p = fn.split("_")[0]; c = fn[:-4].split("_")[-1]
        if c not in CLASSES: continue
        sp = "test" if p in TEST_PARTS else ("val" if p in VAL_PARTS else "train")
        splits[sp].append((os.path.join(CLIP_DIR, fn), c))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=dev)
    global USE_BN
    USE_BN = any(k.endswith("running_mean") for k in sd)
    print(f"arch={'BatchNorm(v1/v2)' if USE_BN else 'GroupNorm(v3)'}", flush=True)
    model = MelCNN().to(dev)
    model.load_state_dict(sd); model.eval()
    res = {}
    for sp in ("val", "test"):
        dl = DataLoader(DS(splits[sp]), batch_size=64, num_workers=0)
        cm = np.zeros((4, 4), dtype=int); confs = []
        with torch.no_grad():
            for x, y in dl:
                p = F.softmax(model(x.to(dev)), 1); pr = p.argmax(1).cpu()
                for t, q in zip(y.tolist(), pr.tolist()): cm[t, q] += 1
                confs += p.max(1).values.tolist()
        acc = np.trace(cm) / max(cm.sum(), 1)
        res[sp] = {"acc": float(acc), "cm": cm.tolist(),
                   "per_class": {CLASS_CN[c]: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
                   "mean_conf": float(np.mean(confs)), "n": int(cm.sum())}
        print(f"[{sp}] acc={acc:.4f}  分布 {dict(Counter(c for _, c in splits[sp]))}", flush=True)
        for i, c in enumerate(CLASSES):
            print(f"   {CLASS_CN[c]:4s} {cm[i,i]/max(cm[i].sum(),1):.4f} ({cm[i,i]}/{cm[i].sum()})", flush=True)
        print("   混淆矩阵:", flush=True); print(cm, flush=True)
    with torch.no_grad():
        zp = F.softmax(model(torch.zeros(1, 1, N_MELS, MAX_FRAMES, device=dev)), 1).squeeze()
    print(f"静音 -> {CLASS_CN[CLASSES[int(zp.argmax())]]} 置信 {zp.max():.3f} "
          f"(test 均值 {res['test']['mean_conf']:.3f})", flush=True)
    res["silence"] = {"pred": CLASS_CN[CLASSES[int(zp.argmax())]], "conf": float(zp.max())}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}", flush=True)

if __name__ == "__main__":
    main()
