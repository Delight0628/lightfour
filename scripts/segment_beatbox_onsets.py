"""
把 Demucs 分离后的 vocals.wav 切成“单次击打”片段（onset 级）。

相比 preprocess_video.py 的能量活跃区切片，本脚本用 librosa onset_detect
检测每次起音，按 onset -> 下一次 onset（或最大 1.0s）切窗，保住衰减尾。

输出：
  out/<source_stem>/<source_stem>_0000.wav
  out/<source_stem>/manifest.jsonl
  out/segments_manifest.jsonl（合并）

用法：
  PYTHONPATH=/gemini/code/lightfour-train/pylibs \
  python scripts/segment_beatbox_onsets.py \
    --input /gemini/code/lightfour-train/data/bbox_raw/demucs \
    --out /tmp/bbox_segments --workers 8
"""
import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22050
HOP = 160


def process_one(args):
    wav_path, out_root, min_dur, max_dur, max_tail, min_rms, min_peak = args
    wav_path = Path(wav_path)
    out_root = Path(out_root)
    stem = wav_path.parent.name if wav_path.name == "vocals.wav" else wav_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    if len(y) < SR * 0.2:
        return stem, []
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP, backtrack=True, units="time",
    )
    onsets = np.asarray(onsets, dtype=np.float64)
    records = []
    for i, t in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else t + max_tail
        start = max(0.0, t - 0.02)
        end = min(len(y) / sr, min(nxt, t + max_tail))
        dur = end - start
        if dur < min_dur or dur > max_dur:
            continue
        s0, s1 = int(start * sr), int(end * sr)
        seg = y[s0:s1]
        if len(seg) < int(min_dur * sr):
            continue
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        peak = float(np.max(np.abs(seg)) + 1e-12)
        if rms < min_rms or peak < min_peak:
            continue
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
        seg_id = f"{stem}_{i:04d}"
        wp = out_dir / f"{seg_id}.wav"
        sf.write(str(wp), seg, sr, subtype="PCM_16")
        records.append({
            "id": seg_id, "source": stem, "wav": str(wp),
            "start": round(start, 4), "end": round(end, 4),
            "dur": round(dur, 4), "rms": round(rms, 5),
            "peak": round(peak, 5), "zcr": round(zcr, 4),
        })
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stem, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="vocals.wav 文件或目录")
    ap.add_argument("--out", default="/tmp/bbox_segments")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-dur", type=float, default=0.06)
    ap.add_argument("--max-dur", type=float, default=1.20)
    ap.add_argument("--max-tail", type=float, default=1.00)
    ap.add_argument("--min-rms", type=float, default=0.008)
    ap.add_argument("--min-peak", type=float, default=0.02)
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        files = sorted(inp.glob("**/vocals.wav"))
    else:
        files = [inp]
    if args.limit:
        files = files[: args.limit]
    print(f"input files={len(files)} out={args.out} workers={args.workers}", flush=True)

    tasks = [(str(p), args.out, args.min_dur, args.max_dur, args.max_tail,
              args.min_rms, args.min_peak) for p in files]
    all_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                stem, recs = fut.result()
                all_records.extend(recs)
                print(f"[{i}/{len(futs)}] {stem}: {len(recs)} segments", flush=True)
            except Exception:
                print(f"[FAIL] {traceback.format_exc()}", flush=True)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "segments_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTOTAL segments={len(all_records)} -> {out_root/'segments_manifest.jsonl'}")


if __name__ == "__main__":
    main()
