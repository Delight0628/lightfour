"""
生成 103 音色 seed 标注候选 + 试听包 + CSV 标注模板。

用法：
  python scripts/build_sound_seeds.py \
    --labels data/real_bbox/hybrid_labels_all_highconf_l2.jsonl \
    --out data/real_bbox/sound_seed_candidates.jsonl \
    --csv data/sound_seed_annotation.csv \
    --pack /tmp/sound_seed_pack --per-family 25 --listen 5

说明：
  - 每家族最多 25 条候选，来源去重（同一 BV 最多 3 条），时长 ≥0.06s。
  - candidate_notation 用种子最近邻给出，仅作提示；人耳确认后填 human_notation。
  - 试听包每家族 5 条，打包成 tar，供人工核对。
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

BASE = Path("/gemini/code/lightfour-train")


def split_of(source):
    """按 source(BV) 的 md5 分组切分；与 prepare_talker_data.py 同口径。"""
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(BASE / "data/real_bbox/hybrid_labels_all_highconf_l2.jsonl"))
    ap.add_argument("--out", default=str(BASE / "data/real_bbox/sound_seed_candidates.jsonl"))
    ap.add_argument("--csv", default=str(BASE / "data/sound_seed_annotation.csv"))
    ap.add_argument("--pack", default="/tmp/sound_seed_pack")
    ap.add_argument("--per-family", type=int, default=25)
    ap.add_argument("--listen", type=int, default=5)
    ap.add_argument("--min-dur", type=float, default=0.06)
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--segments-root", default=str(BASE / "data/real_bbox/segments_batch3"),
                    help="切片根目录；持久盘优先，缺失时回退 /tmp")
    ap.add_argument("--crosswalk", default=str(BASE / "data/beatbox_seed_crosswalk.csv"),
                    help="种子库记法 -> canonical 103 记法 的 crosswalk（由 build_seed_manifest.py 生成）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.labels).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    # canonical 103 体系 crosswalk：种子库记法（带 '<' 前缀）-> canonical 记法/家族
    canon_by_not = {}
    if Path(args.crosswalk).exists():
        import csv as _csv
        with open(args.crosswalk, encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                canon_by_not[row["seedlib_notation"]] = row
    else:
        print(f"[warn] crosswalk 不存在: {args.crosswalk}，canonical 字段将全部为 unknown",
              flush=True)
    seg_root = Path(args.segments_root)
    if not seg_root.exists():
        fallback = Path("/tmp/batch3_segments")
        print(f"[warn] {seg_root} 不存在，回退 {fallback}", flush=True)
        seg_root = fallback
    for r in rows:
        r["wav"] = r["wav"].replace(
            "/gemini/code/lightfour-train/data/real_bbox/segments_batch3/", str(seg_root) + "/")
    rows = [r for r in rows if Path(r["wav"]).exists()
            and r.get("dur", 0) >= args.min_dur
            and r.get("confidence", 0) >= args.min_conf]

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r.get("sound_family") or "unknown"].append(r)

    random.seed(20261023)
    picked = []
    for fam, items in sorted(by_fam.items()):
        # 来源去重 + 高置信优先，再随机打散
        per_src = defaultdict(int)
        pool = []
        for r in sorted(items, key=lambda x: -x.get("confidence", 0)):
            src = r.get("source", "")
            if per_src[src] >= 3:
                continue
            per_src[src] += 1
            pool.append(r)
            if len(pool) >= args.per_family * 3:
                break
        picked.extend(random.sample(pool, min(args.per_family, len(pool))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({
                "id": r["id"], "wav": r["wav"], "source": r.get("source"),
                "family": r.get("sound_family"), "candidate_notation": r.get("seed_notation"),
                "nearest_seed": r.get("nearest_seed"), "seed_dist": r.get("seed_dist"),
                "confidence": r.get("confidence"), "dur": r.get("dur"),
                "start": r.get("start"),
                # canonical 103 体系字段（由 seed 记法经 crosswalk 归一而来）
                "sound_id": canon_by_not.get(r.get("seed_notation"), {}).get("canonical_sound_id", ""),
                "sound_notation": canon_by_not.get(r.get("seed_notation"), {}).get("canonical_notation", "unknown"),
                "sound_en": canon_by_not.get(r.get("seed_notation"), {}).get("canonical_en", "unknown"),
                "sound_cn": canon_by_not.get(r.get("seed_notation"), {}).get("canonical_cn", "unknown"),
                "sound_family": canon_by_not.get(r.get("seed_notation"), {}).get("canonical_family", "unknown"),
                "label_source": "seed_match", "sound_confidence": 0.0,
            }, ensure_ascii=False) + "\n")
    with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "family", "candidate_notation", "nearest_seed", "wav",
                    "human_notation", "note", "source", "split",
                    "canonical_notation", "canonical_en", "canonical_family",
                    "sound_id", "label_source", "sound_confidence",
                    "annotate_status", "annotator"])
        for r in picked:
            c = canon_by_not.get(r.get("seed_notation"), {})
            w.writerow([r["id"], r.get("sound_family"), r.get("seed_notation"),
                        r.get("nearest_seed"), r["wav"], "", "",
                        r.get("source", ""), split_of(r.get("source") or r["id"]),
                        c.get("canonical_notation", "unknown"), c.get("canonical_en", "unknown"),
                        c.get("canonical_family", "unknown"), c.get("canonical_sound_id", ""),
                        "seed_match", 0.0, "pending", ""])

    pack = Path(args.pack)
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    listen = []
    for fam, items in sorted(by_fam.items()):
        sub = [r for r in picked if r.get("sound_family") == fam][: args.listen]
        for r in sub:
            dst = pack / f"{fam}_{r['id']}.wav"
            shutil.copy2(r["wav"], dst)
            listen.append({"file": dst.name, "family": fam, "id": r["id"],
                           "candidate_notation": r.get("seed_notation"),
                           "nearest_seed": r.get("nearest_seed")})
    (pack / "manifest.json").write_text(json.dumps(listen, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    tar = Path("/tmp/sound_seed_pack.tar")
    with tarfile.open(tar, "w") as t:
        t.add(pack, arcname=pack.name)
    print(f"families={len(by_fam)} picked={len(picked)} listen={len(listen)} "
          f"tar={tar} {tar.stat().st_size // 1024}KB", flush=True)
    print("->", out, args.csv, flush=True)


if __name__ == "__main__":
    main()
