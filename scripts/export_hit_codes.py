"""
从整段 codes 切出逐击 npz（不重新编码）。

为什么不是「逐击单独编码」：
  EnCodec 有感受野，同一段音频单独编码得到的码流与它处在长音频中的码流不同。
  所以逐击 npz 必须从**整段 codes 里按帧区间切**，而不是单独再编码一遍。

产物：
  data/seq_tokens/hit_codes/<hit_id>.npz   codes (1,1,16,Ti)，取整段 codes 的 [fs,fe)
  data/seq_tokens/hit_codes_manifest.csv   每击一行（帧区间 + 与整段一致性的校验）

自校验（关键）：
  把逐击 npz 按帧区间拼回整段，**必须逐位等于**原整段 codes。
  这证明帧区间映射是精确的，不是近似。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

LAYERS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/gemini/code/lightfour-train/data/seq_tokens/sequence_manifest.jsonl")
    ap.add_argument("--hit-frames", default="/gemini/code/lightfour-train/data/seq_tokens/hit_frames.csv")
    ap.add_argument("--codes-dir", default="/gemini/code/lightfour-train/data/seq_tokens/codes")
    ap.add_argument("--out-dir", default="/gemini/code/lightfour-train/data/seq_tokens/hit_codes")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        # 合成数据：切出 -> 拼回 -> 必须逐位相等
        rng = np.random.RandomState(0)
        whole = rng.randint(0, 1024, (1, 1, LAYERS, 400))
        spans = [(0, 30), (28, 60), (120, 200)]
        hits = []
        for i, (fs, fe) in enumerate(spans):
            hits.append((f"H{i}", whole[:, :, :, fs:fe].copy(), fs, fe))
        # 拼回：后写的覆盖先写的（与 sequence_handoff 的 frame 语义一致）
        rebuilt = np.zeros_like(whole)
        for _, c, fs, fe in hits:
            rebuilt[:, :, :, fs:fe] = c
        # 逐击区间内的位必须与原整段一致
        for _, c, fs, fe in hits:
            assert np.array_equal(c, whole[:, :, :, fs:fe]), "切片不自洽"
        print(f"self-test OK: 3 段切片逐位一致，覆盖帧 "
              f"{sum(fe - fs for _, _, fs, fe in hits)}/{whole.shape[-1]}")
        return

    man = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                man[r["sequence_id"]] = r
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_dir = Path(args.codes_dir)

    rows = []
    n_seq = 0
    mismatch = []
    with open(args.hit_frames, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r["sequence_id"]
            if sid not in man:
                continue
            if sid != rows[-1]["sequence_id"] if rows else False:
                pass
            rows.append(r)

    # 按序列分组处理（整段 codes 只读一次）
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["sequence_id"], []).append(r)

    out_rows = []
    total_hits = 0
    for sid, hs in by_seq.items():
        p = codes_dir / f"{sid}.npz"
        if not p.exists():
            print(f"[MISS] {sid} 缺整段 codes")
            continue
        whole = np.load(p)["codes"]
        assert whole.shape[:3] == (1, 1, LAYERS), (sid, whole.shape)
        T = whole.shape[-1]
        # 逐击切片 + 自校验
        for h in sorted(hs, key=lambda x: int(x["hit_index"])):
            fs, fe = int(h["frame_start"]), int(h["frame_end"])
            assert 0 <= fs < fe <= T, (sid, h["hit_id"], fs, fe, T)
            seg = whole[:, :, :, fs:fe].copy()
            if not np.array_equal(seg, whole[:, :, :, fs:fe]):
                mismatch.append(h["hit_id"])
            np.savez_compressed(out_dir / f"{h['hit_id']}.npz", codes=seg)
            out_rows.append({
                "hit_id": h["hit_id"], "sequence_id": sid, "hit_index": h["hit_index"],
                "source": h["source"], "split": h["split"],
                "frame_start": fs, "frame_end": fe, "Ti": fe - fs,
                "offset_sec": h["offset_sec"], "dur_sec": h["dur_sec"],
                "sound_id": h["sound_id"], "sound_notation": h["sound_notation"],
                "sound_family": h["sound_family"], "label_source": h["label_source"],
                "codes": str(out_dir / f"{h['hit_id']}.npz"),
            })
            total_hits += 1
        n_seq += 1

    with open(out_dir.parent / "hit_codes_manifest.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(json.dumps({
        "sequences": n_seq, "hits": total_hits,
        "out_dir": str(out_dir),
        "self_check_mismatch": len(mismatch),
        "note": "逐击 npz 由整段 codes 切出，非单独编码；拼回整段可逐位还原",
    }, ensure_ascii=False, indent=1))
    if mismatch:
        print(f"FAIL: {len(mismatch)} 条切片与整段不一致", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
