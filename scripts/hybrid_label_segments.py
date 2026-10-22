"""
混合弱标：模型标签 + 镲类规则重平衡。

背景：
  robust 模型在真实切片上把 64% 判成军鼓，开镲高置信为 0。
  但镲类的物理特征很明确：高频占比高、谱质心高、闭镲衰减短/开镲衰延长。
  因此对高频段用规则重判 hhc/hho，其余沿用模型标签。

输出：
  labels.jsonl（全部）
  labels_highconf.jsonl（confidence >= 0.7）
  stats.json
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

CLASSES = ["kd", "sd", "hhc", "hho"]
CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
SR, N_FFT, HOP = 22050, 1024, 256


def feats(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    fr = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    tot = float(S.sum() + 1e-9)
    low = float(S[fr < 150].sum() / tot)
    mid = float(S[(fr >= 150) & (fr < 2000)].sum() / tot)
    high = float(S[fr >= 4000].sum() / tot)
    cen = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y, hop_length=HOP).mean())
    frame = librosa.util.frame(y, frame_length=HOP, hop_length=HOP)
    rms = np.sqrt((frame ** 2).mean(axis=0) + 1e-12)
    pk = int(np.argmax(rms)); thr = max(rms[pk] * 0.1, 1e-4)
    above = np.where(rms > thr)[0]
    dec = float((above[-1] - pk) * HOP / SR) if len(above) else 0.0
    return low, mid, high, cen, zcr, dec


def work(args):
    row, model_label, model_conf = args
    try:
        low, mid, high, cen, zcr, dec = feats(row["wav"])
    except Exception:
        return None
    label, conf, source = model_label, model_conf, "model"
    if high > 0.55 and cen > 3400:
        rule = "hhc" if dec < 0.12 else "hho"
        hat_conf = min(0.95, 0.55 + max(0.0, high - 0.55) * 1.5)
        if hat_conf >= conf:
            label, conf, source = rule, hat_conf, "rule_hat"
        else:
            label, source = rule, "rule_hat"
    return {**row, "pseudo_label": label, "label_cn": CN[label],
            "confidence": round(float(conf), 4), "label_source": source,
            "feat": {"low": round(low, 4), "mid": round(mid, 4), "high": round(high, 4),
                     "centroid": round(cen, 1), "zcr": round(zcr, 4), "decay": round(dec, 4)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pseudo = {}
    for l in open(args.pseudo, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            pseudo[r["id"]] = r
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    tasks = []
    for r in rows:
        p = pseudo.get(r["id"])
        if not p:
            continue
        tasks.append((r, p["pseudo_label"], p["confidence"]))
    print(f"tasks={len(tasks)} workers={args.workers}", flush=True)

    out_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                out_rows.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(futs)}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hi = [r for r in out_rows if r["confidence"] >= 0.7]
    with open(out.with_name(out.stem + "_highconf.jsonl"), "w", encoding="utf-8") as f:
        for r in hi:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "total": len(out_rows),
        "highconf_0.7": len(hi),
        "dist_all": dict(Counter(r["pseudo_label"] for r in out_rows)),
        "dist_highconf": dict(Counter(r["pseudo_label"] for r in hi)),
        "source_counts": dict(Counter(r["label_source"] for r in out_rows)),
        "mean_conf": round(float(np.mean([r["confidence"] for r in out_rows])), 4),
    }
    out.with_name("stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
