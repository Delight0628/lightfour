"""
Route B 数据准备：把 gen_synth 的 tar 分片整理成可随机访问的训练张量。

背景：
  render_synth_tar.py 为了规避 seaweedfs 写小文件慢，把 90,000 个窗口打成了 45 个 tar。
  但训练时若每次随机打开 tar 取成员，需要顺序扫描，速度不可接受。
  本脚本把每个 tar 转成两个“大文件”：
    mel/shard_XXXXX.npy   (2000, 122, 64) float32，可直接 np.load(mmap_mode='r')
    wav/shard_XXXXX.bin   原始 wav 字节顺序拼接，索引记录 offset/len

  => 45 个 tar -> 90 个大文件，保持顺序大块写，seaweedfs 友好；
  => 训练用 memmap 随机读取，不产生 18 万个小文件。

输出：
  data/gen_synth_routeb/
    mel/shard_00000.npy ...
    wav/shard_00000.bin ...
    index.jsonl        全量索引（鼓谱 + 张量位置）
    train.jsonl        训练集（id % 50 != 0）
    val.jsonl          验证集（id % 50 == 0，约 2%）
    routeb_stats.json  mel 均值/方差、类别分布、时长统计

运行：
  /root/miniconda3/bin/python scripts/prepare_gen_routeb.py
  # 快速验证：--limit-shards 1
"""
import argparse
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import tarfile

FRAMES = 122
N_MELS = 64
CLASSES = ["kd", "sd", "hhc", "hho"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fix_mel(mel):
    """统一成 (122, 64) float32；兼容转置/轻微长度差异。"""
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        mel = mel.reshape(FRAMES, N_MELS)
    if mel.shape == (N_MELS, FRAMES):
        mel = mel.T
    if mel.shape[0] < FRAMES:
        mel = np.pad(mel, ((0, FRAMES - mel.shape[0]), (0, 0)))
    elif mel.shape[0] > FRAMES:
        mel = mel[:FRAMES]
    if mel.shape[1] < N_MELS:
        mel = np.pad(mel, ((0, 0), (0, N_MELS - mel.shape[1])))
    elif mel.shape[1] > N_MELS:
        mel = mel[:, :N_MELS]
    return np.ascontiguousarray(mel, dtype=np.float32)


def compact_score(score):
    """只保留训练条件需要的 onset + 鼓类，减小索引体积。"""
    out = []
    for n in score:
        out.append({"onset": float(n["onset"]), "cls": n["cls"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gemini/code/lightfour-train/data")
    ap.add_argument("--shards", default="", help="默认 {root}/gen_synth_tar/shards")
    ap.add_argument("--scores", default="", help="默认 {root}/gen_synth_tar/scores.jsonl")
    ap.add_argument("--out", default="", help="默认 {root}/gen_synth_routeb")
    ap.add_argument("--limit-shards", type=int, default=0, help="只处理前 N 个分片（调试）")
    ap.add_argument("--no-wav", action="store_true", help="只导出 mel（wav 仍留在 tar）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    shards_dir = Path(args.shards) if args.shards else root / "gen_synth_tar" / "shards"
    scores_path = Path(args.scores) if args.scores else root / "gen_synth_tar" / "scores.jsonl"
    out = Path(args.out) if args.out else root / "gen_synth_routeb"

    if out.exists() and not args.overwrite:
        raise SystemExit(f"输出目录已存在：{out}；加 --overwrite 或换 --out")
    (out / "mel").mkdir(parents=True, exist_ok=True)
    (out / "wav").mkdir(parents=True, exist_ok=True)

    print(f"[load] {scores_path}", flush=True)
    scores = {r["id"]: r for r in read_jsonl(scores_path)}
    print(f"[load] scores={len(scores)}", flush=True)

    shard_names = sorted(p.name for p in shards_dir.glob("shard_*.tar"))
    if args.limit_shards:
        shard_names = shard_names[: args.limit_shards]
    print(f"[plan] shards={len(shard_names)}  out={out}", flush=True)

    index = []
    mel_n = 0
    mel_sum = 0.0
    mel_sq = 0.0
    mel_min = float("inf")
    mel_max = float("-inf")
    wav_total = 0
    durations = []
    class_cnt = Counter()
    t0 = time.time()

    for si, shard_name in enumerate(shard_names, 1):
        shard_id = shard_name[:-4]
        tar_path = shards_dir / shard_name
        mel_path = out / "mel" / f"{shard_id}.npy"
        wav_path = out / "wav" / f"{shard_id}.bin"
        rows = []
        mel_rows = []

        with tarfile.open(tar_path, "r") as tf:
            members = tf.getnames()
            ids = sorted(
                n.split("/")[-1][:-4]
                for n in members
                if n.startswith("wav/") and n.endswith(".wav")
            )
            wf = None if args.no_wav else open(wav_path, "wb")
            try:
                for row, rid in enumerate(ids):
                    if wf is not None:
                        wb = tf.extractfile(f"wav/{rid}.wav").read()
                        off = wf.tell()
                        wf.write(wb)
                        wav_total += len(wb)
                        if row % 500 == 0:
                            a, sr = sf.read(io.BytesIO(wb), dtype="float32")
                            durations.append({"id": rid, "sr": int(sr),
                                              "dur": round(len(a) / sr, 4)})
                    else:
                        wb = b""
                        off = 0

                    mb = tf.extractfile(f"mel/{rid}.npy").read()
                    mel = fix_mel(np.load(io.BytesIO(mb)))
                    mel_rows.append(mel)

                    meta = scores.get(rid, {})
                    score = compact_score(meta.get("score", []))
                    for n in score:
                        class_cnt[n["cls"]] += 1
                    rows.append({
                        "id": rid,
                        "shard": shard_id,
                        "row": row,
                        "mel": f"mel/{shard_id}.npy",
                        "wav": f"wav/{shard_id}.bin",
                        "wav_offset": off,
                        "wav_len": len(wb),
                        "tempo_bpm": float(meta.get("tempo_bpm", 0.0)),
                        "n_notes": int(meta.get("n_notes", len(score))),
                        "score": score,
                    })
            finally:
                if wf is not None:
                    wf.close()

        arr = np.stack(mel_rows, axis=0) if mel_rows else np.zeros((0, FRAMES, N_MELS), np.float32)
        np.save(mel_path, arr)

        mel_n += int(arr.size)
        mel_sum += float(arr.sum(dtype=np.float64))
        mel_sq += float(np.square(arr.astype(np.float64)).sum())
        mel_min = min(mel_min, float(arr.min()))
        mel_max = max(mel_max, float(arr.max()))
        index.extend(rows)

        print(
            f"[{si:02d}/{len(shard_names)}] {shard_id} rows={len(rows)} "
            f"mel={arr.shape} {time.time() - t0:.0f}s",
            flush=True,
        )

    mean = mel_sum / max(mel_n, 1)
    var = mel_sq / max(mel_n, 1) - mean * mean
    std = math.sqrt(max(var, 1e-8))

    write_jsonl(out / "index.jsonl", index)
    train = [r for r in index if int(r["id"].rsplit("_", 1)[1]) % 50 != 0]
    val = [r for r in index if int(r["id"].rsplit("_", 1)[1]) % 50 == 0]
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    stats = {
        "n_windows": len(index),
        "n_shards": len(shard_names),
        "frames": FRAMES,
        "n_mels": N_MELS,
        "mel_mean": round(mean, 6),
        "mel_std": round(std, 6),
        "mel_min": round(mel_min, 6),
        "mel_max": round(mel_max, 6),
        "wav_total_bytes": wav_total,
        "class_counts": dict(class_cnt),
        "duration_samples": durations[:20],
        "train": len(train),
        "val": len(val),
        "elapsed_sec": round(time.time() - t0, 1),
        "wav_exported": not args.no_wav,
    }
    with open(out / "routeb_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
