"""
对 onset 切片做伪标注（兼容 v2 BatchNorm 架构）。

与旧 pseudo_label.py 的区别：
  - 旧版是 v1 架构（features.* / BatchNorm / 128 维 AvgPool / per-clip 归一化）
  - 本版复用 v2 架构（f.* / Avg+Max 256 维），归一化用全局 (x - (-4)) / 3
  - 自动检测 checkpoint 是 v1/v2（BatchNorm）还是 v3（GroupNorm）

用法：
  PYTHONPATH=/gemini/code/lightfour-train/pylibs \
  /root/miniconda3/bin/python scripts/pseudo_label_v2.py \
    --manifest /tmp/bbox_segments/segments_manifest.jsonl \
    --ckpt checkpoints/beatbox_cls_v2/best.pt \
    --conf 0.85 --out data/real_bbox/pseudo
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

torch.set_num_threads(8)
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
SR, N_MELS, N_FFT, HOP = 22050, 64, 1024, 256
MAX_FRAMES = int(1.5 * SR / HOP) + 1
G_MEAN, G_STD = -4.0, 3.0
mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=40, f_max=SR // 2)


def gn(c, use_bn):
    return nn.BatchNorm2d(c) if use_bn else nn.GroupNorm(min(8, c), c)


class MelCNN(nn.Module):
    def __init__(self, n=4, use_bn=False):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), gn(32, use_bn), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), gn(64, use_bn), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), gn(128, use_bn), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), gn(128, use_bn), nn.ReLU())
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)), nn.AdaptiveMaxPool2d((1, 1))])
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.f(x)
        return self.head(torch.cat([p(z).flatten(1) for p in self.pool], 1))


def load_logmel(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        n = int(len(wav) * SR / sr)
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    t = torch.from_numpy(np.ascontiguousarray(wav))
    if t.numel() < 16:
        t = F.pad(t, (0, 16 - t.numel()))
    lm = torch.log(mel_tf(t) + 1e-6)
    fr = lm.shape[-1]
    lm = F.pad(lm, (0, MAX_FRAMES - fr)) if fr < MAX_FRAMES else lm[:, :MAX_FRAMES]
    return (lm - G_MEAN) / G_STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conf", type=float, default=0.85)
    ap.add_argument("--out", default="data/real_bbox/pseudo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--audit", type=float, default=0.05)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=dev)
    use_bn = any(k.endswith("running_mean") for k in sd)
    model = MelCNN(4, use_bn=use_bn).to(dev)
    model.load_state_dict(sd)
    model.eval()
    print(f"ckpt={args.ckpt} arch={'BatchNorm(v1/v2)' if use_bn else 'GroupNorm(v3)'} device={dev}", flush=True)

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], 0
    for i, r in enumerate(rows, 1):
        wp = r.get("wav")
        if not wp or not os.path.exists(wp):
            continue
        x = load_logmel(wp).unsqueeze(0).to(dev)
        with torch.no_grad():
            prob = F.softmax(model(x), 1).squeeze().cpu()
        conf, idx = float(prob.max()), int(prob.argmax())
        label = CLASSES[idx]
        rec = {**r, "pseudo_label": label, "label_cn": CLASS_CN[label],
               "confidence": round(conf, 4),
               "probs": {CLASSES[j]: round(float(prob[j]), 4) for j in range(4)}}
        if conf >= args.conf:
            kept.append(rec)
        else:
            dropped += 1
        if i % 500 == 0:
            print(f"  {i}/{len(rows)} kept={len(kept)} dropped={dropped}", flush=True)

    with open(out / "pseudo_labels.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 抽检样本
    import random
    random.seed(42)
    random.shuffle(kept)
    n_audit = max(1, int(len(kept) * args.audit))
    with open(out / "audit_sample.jsonl", "w", encoding="utf-8") as f:
        for r in kept[:n_audit]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter(r["pseudo_label"] for r in kept)
    print(f"\ninput={len(rows)} kept={len(kept)} dropped={dropped} "
          f"keep_rate={len(kept)/max(len(rows),1):.1%}", flush=True)
    print(f"分布: {{k: v for k, v in dist.items()}}".replace("{k: v for k, v in dist.items()}", str(dict(dist))), flush=True)
    print(f"平均置信度: {np.mean([r['confidence'] for r in kept]) if kept else 0:.4f}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
