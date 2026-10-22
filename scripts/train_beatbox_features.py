"""
Stage A 备选：手工特征 + RandomForest（更快、对跨人/真实录音可能更稳）。

特征：duration、RMS、peak、ZCR、谱质心/带宽/rolloff/flatness、
低/中/高频能量占比、衰减时间（峰值到 -20dB）。
划分与 v2 一致：test=P9/P11/P19/P24，val=P20/P21。

运行：
  /root/miniconda3/bin/python scripts/train_beatbox_features.py
"""
import json
import os
from collections import Counter, defaultdict

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(ROOT, "data", "sft", "beatbox_clips")
OUT_DIR = os.path.join(ROOT, "checkpoints", "beatbox_feat")
os.makedirs(OUT_DIR, exist_ok=True)
CLASSES = ["kd", "sd", "hhc", "hho"]
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}
SR = 22050
N_FFT = 1024
HOP = 256


def split_of(p):
    if p in TEST_PARTS:
        return "test"
    if p in VAL_PARTS:
        return "val"
    return "train"


def feats(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < 64:
        y = np.pad(y, (0, 64 - len(y)))
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    total = float(S.sum() + 1e-9)
    low = float(S[freqs < 150].sum() / total)
    mid = float(S[(freqs >= 150) & (freqs < 2000)].sum() / total)
    high = float(S[freqs >= 4000].sum() / total)
    centroid = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=SR).mean())
    rolloff = float(librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85).mean())
    flatness = float(librosa.feature.spectral_flatness(S=S).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y, hop_length=HOP).mean())
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    frame = librosa.util.frame(y, frame_length=HOP, hop_length=HOP)
    frms = np.sqrt((frame ** 2).mean(axis=0) + 1e-12)
    pk = int(np.argmax(frms))
    thr = max(frms[pk] * 0.1, 1e-4)
    above = np.where(frms > thr)[0]
    decay = float((above[-1] - pk) * HOP / SR) if len(above) else 0.0
    return [
        len(y) / SR, rms, peak, zcr, centroid, bandwidth, rolloff, flatness,
        low, mid, high, decay, float(frms.max()), float(frms.mean()),
        float(np.percentile(frms, 90)),
    ]


def main():
    splits = defaultdict(list)
    for fn in sorted(os.listdir(CLIP_DIR)):
        if not fn.endswith(".wav"):
            continue
        p = fn.split("_")[0]
        c = fn[:-4].split("_")[-1]
        if c not in CLASSES:
            continue
        splits[split_of(p)].append((os.path.join(CLIP_DIR, fn), c))

    data = {}
    for sp, items in splits.items():
        X, y = [], []
        for i, (path, c) in enumerate(items):
            try:
                X.append(feats(path))
                y.append(CLASSES.index(c))
            except Exception as exc:
                print("skip", path, exc)
        data[sp] = (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
        print(sp, data[sp][0].shape, Counter(y))

    scaler = StandardScaler().fit(data["train"][0])
    Xtr = scaler.transform(data["train"][0])
    Xva = scaler.transform(data["val"][0])
    Xte = scaler.transform(data["test"][0])
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=18, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=16, random_state=42,
    )
    clf.fit(Xtr, data["train"][1])
    vacc = accuracy_score(data["val"][1], clf.predict(Xva))
    pred = clf.predict(Xte)
    tacc = accuracy_score(data["test"][1], pred)
    cm = confusion_matrix(data["test"][1], pred, labels=list(range(4)))
    print(f"val={vacc:.4f} test={tacc:.4f} (random 0.25)")
    for i, c in enumerate(CLASSES):
        print(c, round(cm[i, i] / max(cm[i].sum(), 1), 4), cm[i].tolist())
    joblib.dump({"scaler": scaler, "clf": clf, "classes": CLASSES},
                os.path.join(OUT_DIR, "rf.joblib"))
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"val_acc": float(vacc), "test_acc": float(tacc),
                   "per_class": {CLASSES[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(4)},
                   "confusion_matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)
    print("->", OUT_DIR)


if __name__ == "__main__":
    main()
