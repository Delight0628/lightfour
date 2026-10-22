"""
用真实 B 站高置信切片组装 2 秒「鼓谱 -> 真实音频」生成对。

与 hit_bank 合成数据的区别：
  - 音频直接来自真实 vocals，不是 4,872 个 hit 重排；
  - 鼓谱来自 hybrid 弱标（含置信度），可过滤；
  - 输出与 gen_synth 同格式：wav/ + mel/ + scores.jsonl + omni_pairs.jsonl。

用法：
  python scripts/build_real_gen_pairs.py \
    --labels data/real_bbox/hybrid_labels_78v_highconf.jsonl \
    --vocals data/bbox_raw/demucs_srv \
    --out data/real_gen
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
WIN = 2.0
N_FFT, HOP, N_MELS = 1024, 256, 64
CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}


def log_mel(x):
    n = len(x)
    if n < N_FFT:
        x = np.pad(x, (0, N_FFT - n))
    frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(N_FFT), axis=1)) ** 2
    f = np.fft.rfftfreq(N_FFT, 1 / SR)
    mel_min, mel_max = 2595 * np.log10(1 + 40 / 700), 2595 * np.log10(1 + (SR / 2) / 700)
    m = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz = 700 * (10 ** (m / 2595) - 1)
    bins = np.floor((N_FFT + 1) * hz / SR).astype(int)
    fb = np.zeros((N_MELS, len(f)))
    for i in range(N_MELS):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        r = min(r, len(f) - 1); c = min(c, r)
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return np.log(S @ fb.T + 1e-6).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--vocals", default="data/bbox_raw/demucs_srv")
    ap.add_argument("--out", default="data/real_gen")
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--step", type=float, default=2.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "mel").mkdir(parents=True, exist_ok=True)

    scores, pairs = [], []
    cls_cnt = Counter()
    n_win = 0
    for src, rs in sorted(by_src.items()):
        vpath = Path(args.vocals) / "htdemucs" / src / "vocals.wav"
        if not vpath.exists():
            print("missing", vpath)
            continue
        y, sr = sf.read(str(vpath), dtype="float32")
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != SR:
            n = int(len(y) * SR / sr)
            y = np.interp(np.linspace(0, 1, n, endpoint=False),
                          np.linspace(0, 1, len(y), endpoint=False), y).astype("float32")
        rs.sort(key=lambda r: float(r["start"]))
        dur = len(y) / SR
        t = 0.0
        while t + WIN <= dur:
            hits = [r for r in rs if t <= float(r["start"]) < t + WIN]
            if len(hits) >= args.min_hits:
                seg = y[int(t * SR):int((t + WIN) * SR)]
                if len(seg) < int(WIN * SR):
                    seg = np.pad(seg, (0, int(WIN * SR) - len(seg)))
                peak = float(np.max(np.abs(seg)) + 1e-8)
                seg = seg / peak * 0.85
                sid = f"real_{n_win:06d}"
                wp = out / "wav" / f"{sid}.wav"
                sf.write(str(wp), seg, SR, subtype="PCM_16")
                mel = log_mel(seg)
                np.save(out / "mel" / f"{sid}.npy", mel)
                notes = [{"onset": round(float(r["start"]) - t, 3),
                          "cls": r["pseudo_label"],
                          "confidence": float(r["confidence"])} for r in hits]
                for n in notes: cls_cnt[n["cls"]] += 1
                drum_text = "、".join(CN[n["cls"]] for n in notes)
                rec = {"id": sid, "source": src, "wav": str(wp),
                       "window": [round(t, 3), round(t + WIN, 3)],
                       "score": notes, "drum_text": drum_text,
                       "mel": f"mel/{sid}.npy"}
                scores.append(rec)
                pairs.append({"id": sid, "messages": [
                    {"role": "user", "content": f"用beatbox演绎这段鼓谱：{drum_text}（真实录音）"},
                    {"role": "assistant", "content": drum_text, "audios": [str(wp)]}]})
                n_win += 1
            t += args.step
        print(f"{src}: windows={n_win}", flush=True)

    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for r in scores: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "omni_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in pairs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"windows": len(scores), "hours": round(len(scores) * WIN / 3600, 4),
             "sources": len(by_src), "class_counts": dict(cls_cnt),
             "min_hits": args.min_hits, "step": args.step}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
