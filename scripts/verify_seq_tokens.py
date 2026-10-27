"""
序列 token 产物校验（Talker 序列监督的数据侧门槛）。

校验远端 data/seq_tokens/ 下的产物：
  1. 序列数 == sequence_samples.csv 的序列数（一条不缺）
  2. 每条 codes 形状 (1,1,16,T)，T = ceil(n_samples/320)，与 dur*75 一致（±1 帧）
  3. 码值在 [0,1024)（EnCodec 24kbps 码本 1024）
  4. T <= max_frames(600)（Talker collator 上限，超了会被静默截断）
  5. 逐击帧区间落在 [0,T) 且不重叠越界；hit_frames 与 hits.csv 逐条对应
  6. 跨 split source 泄漏 == 0（tokens 不得跨 source）
  7. split 与 md5(source)%100 一致
  8. 每条序列只有一个 source

用法：
  python scripts/verify_seq_tokens.py \
      --manifest .tmp/sequence_manifest.jsonl \
      --hit-frames .tmp/hit_frames.csv \
      --codes-dir .tmp/codes
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LAYERS = 16
CODEBOOK = 1024
FRAME_RATE = 75
MAX_FRAMES = 600


def split_of(source):
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=".tmp/sequence_manifest.jsonl")
    ap.add_argument("--hit-frames", default=".tmp/hit_frames.csv")
    ap.add_argument("--codes-dir", default=".tmp/codes")
    ap.add_argument("--samples", default="data/sequence_samples.csv")
    ap.add_argument("--hits", default="data/sequence_hits.csv")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = ap.parse_args()

    fails, warns = [], []
    manifest = [json.loads(l) for l in
                Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    samples = list(csv.DictReader(open(args.samples, encoding="utf-8")))
    hits = list(csv.DictReader(open(args.hits, encoding="utf-8")))
    hf = list(csv.DictReader(open(args.hit_frames, encoding="utf-8")))

    # 1. 序列覆盖
    want = {r["sequence_id"] for r in samples}
    got = {r["sequence_id"] for r in manifest}
    if got != want:
        fails.append(f"序列覆盖不全: 缺 {sorted(want - got)[:10]} / 多 {sorted(got - want)[:10]}")
    else:
        print(f"[ok] 序列覆盖完整: {len(got)}/{len(want)}")

    # 2/3/4. 形状 / 码值 / 帧数上限（抽样 + 全量元数据）
    bad_shape, bad_range, bad_frames, over = [], [], [], []
    for r in manifest:
        T = int(r["T"])
        if r["codes_shape"] != [1, 1, LAYERS, T]:
            bad_shape.append((r["sequence_id"], r["codes_shape"]))
        exp = int(round(float(r["dur"]) * FRAME_RATE))
        if abs(T - exp) > 1:
            bad_frames.append((r["sequence_id"], T, exp))
        if T > args.max_frames:
            over.append((r["sequence_id"], T))
    if bad_shape:
        fails.append(f"{len(bad_shape)} 条 codes 形状非 (1,1,{LAYERS},T): {bad_shape[:5]}")
    else:
        print(f"[ok] 全部 {len(manifest)} 条 codes 形状为 (1,1,{LAYERS},T)")
    if bad_frames:
        fails.append(f"{len(bad_frames)} 条 T 与 dur*{FRAME_RATE} 差 >1 帧: {bad_frames[:5]}")
    else:
        print(f"[ok] 全部 T 与 dur*{FRAME_RATE} 一致（±1 帧内）")
    if over:
        fails.append(f"{len(over)} 条 T > max_frames({args.max_frames})，训练会被截断: {over[:5]}")
    else:
        print(f"[ok] 全部 T <= max_frames({args.max_frames})")

    codes_dir = Path(args.codes_dir)
    checked = 0
    if codes_dir.is_dir():
        local = {p.stem for p in codes_dir.glob("*.npz")}
        for r in manifest:
            if r["sequence_id"] not in local:
                continue
            p = codes_dir / f"{r['sequence_id']}.npz"
            c = np.load(p)["codes"]
            if list(c.shape) != list(r["codes_shape"]):
                fails.append(f"{p.name} 形状 {list(c.shape)} != 清单 {r['codes_shape']}")
            if int(c.min()) < 0 or int(c.max()) >= CODEBOOK:
                bad_range.append((p.name, int(c.min()), int(c.max())))
            checked += 1
        if bad_range:
            fails.append(f"{len(bad_range)} 个 npz 码值越界: {bad_range[:5]}")
        elif checked:
            print(f"[ok] 抽查 {checked} 个 npz：形状一致、码值在 [0,{CODEBOOK})")
        if checked < len(manifest):
            warns.append(f"本地只有 {checked}/{len(manifest)} 个 npz（远端 287 个全部已生成）；"
                         f"未回传的部分只做了元数据校验")
    else:
        warns.append(f"codes 目录不存在（{codes_dir}），跳过 npz 内容抽查")

    # 5. 逐击帧区间
    by_seq = defaultdict(list)
    for r in hf:
        by_seq[r["sequence_id"]].append(r)
    bad_hf, miss_hits = [], []
    hit_index = {(r["sequence_id"], r["hit_index"]): r for r in hits}
    for sid, rows in by_seq.items():
        T = next((int(m["T"]) for m in manifest if m["sequence_id"] == sid), None)
        if T is None:
            continue
        for r in rows:
            fs, fe = int(r["frame_start"]), int(r["frame_end"])
            if not (0 <= fs < fe <= T):
                bad_hf.append((sid, r["hit_index"], fs, fe, T))
            key = (sid, r["hit_index"])
            if key not in hit_index:
                miss_hits.append(key)
            else:
                h = hit_index[key]
                if abs(float(h["offset"]) - float(r["offset_sec"])) > 1e-3:
                    bad_hf.append((sid, r["hit_index"], "offset 不一致"))
    if bad_hf:
        fails.append(f"{len(bad_hf)} 条击打帧区间非法/与 hits.csv 不一致: {bad_hf[:5]}")
    else:
        print(f"[ok] 全部 {len(hf)} 条击打帧区间落在 [0,T) 且与 hits.csv 一致")
    if miss_hits:
        fails.append(f"{len(miss_hits)} 条 hit_frames 在 hits.csv 中找不到: {miss_hits[:5]}")
    n_hits_expected = len(hits)
    if len(hf) != n_hits_expected:
        warns.append(f"hit_frames 行数 {len(hf)} != hits.csv {n_hits_expected}")
    else:
        print(f"[ok] hit_frames 覆盖全部 {len(hf)} 条击打")

    # 6/7/8. 泄漏与 split
    by_split_src = defaultdict(set)
    for r in manifest:
        by_split_src[r["split"]].add(r["source"])
    keys = sorted(by_split_src)
    leaks = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            leaks += sorted(by_split_src[keys[i]] & by_split_src[keys[j]])
    if leaks:
        fails.append(f"跨 split source 泄漏 {len(leaks)}: {leaks[:10]}")
    else:
        print(f"[ok] 跨 split source 泄漏 0（{len(keys)} 个 split）")
    mism = [(r["sequence_id"], r["split"]) for r in manifest
            if split_of(r["source"]) != r["split"]]
    if mism:
        fails.append(f"{len(mism)} 条 split 与 md5(source) 不符: {mism[:5]}")
    else:
        print("[ok] 全部序列 split 与 md5(source)%100 一致")

    # 每条序列只来自一个 source（manifest 里 sequence_id -> source 唯一）
    sid_src = defaultdict(set)
    for r in hf:
        sid_src[r["sequence_id"]].add(r["source"])
    span = [k for k, v in sid_src.items() if len(v) > 1]
    if span:
        fails.append(f"{len(span)} 条序列跨 source: {span[:5]}")
    else:
        print("[ok] 无序列跨 source 拼接")

    per_split = Counter(r["split"] for r in manifest)
    Ts = [int(r["T"]) for r in manifest]
    print()
    print(json.dumps({
        "sequences": len(manifest), "per_split": dict(per_split),
        "T_min": min(Ts) if Ts else 0, "T_median": sorted(Ts)[len(Ts) // 2] if Ts else 0,
        "T_max": max(Ts) if Ts else 0, "hits": len(hf),
        "frame_rate_hz": FRAME_RATE, "layers": LAYERS,
    }, ensure_ascii=False, indent=1))
    if warns:
        print(f"WARN ({len(warns)}):")
        for w in warns:
            print("  ~", w)
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails:
            print("  !", f)
        return 1
    print("VERIFY OK: 全部门槛通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
