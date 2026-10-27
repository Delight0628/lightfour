"""nasal / tongue 最小标注补充单（PI 裁决 139 第 4 项）。

现有 `data/annotation_worklist.csv` 已覆盖 17 family x 5 = 85 行（tongue 1 行），
缺的只有 nasal（0 行）。本脚本把当前语料里**音频真实存在**的 nasal / tongue
命中打成补充单，供 Delight 在 `annotate_ui.py` 队列里使用。

硬约束：
  - 只收 `data/real_bbox/segments_batch3/<source>/<hit_id>.wav` 已持久化的命中
    （/tmp 切片已被清理，不可作为标注源）
  - 本地副本落在 `data/bbox_raw/family_gap_candidates/`，逐文件校验字节数与时长
  - 不写 `data/sound_seed_annotation.csv`（UI 队列是冻结产物，合并需一句批准）

用法：python scripts/build_annotation_supplement.py
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sequence_samples import split_of  # noqa: E402

COLS = ["order", "id", "sound_family", "sound_id", "candidate_notation",
        "canonical_notation", "canonical_en", "slice_wav", "ref_wav", "ref_notation",
        "split", "source", "human_notation", "annotate_status", "annotator", "note"]

TARGETS = ["nasal", "tongue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/real_bbox/hybrid_labels_all_highconf_l2.jsonl")
    ap.add_argument("--taxonomy", default="data/beatbox_taxonomy.json")
    ap.add_argument("--crosswalk", default="data/beatbox_seed_crosswalk.csv")
    ap.add_argument("--cand-dir", default="data/bbox_raw/family_gap_candidates")
    ap.add_argument("--worklist", default="data/annotation_worklist.csv")
    ap.add_argument("--out", default="data/annotation_worklist_supplement.csv")
    args = ap.parse_args()

    tax = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))["sounds"]
    cw = {r["seedlib_notation"]: r for r in
          csv.DictReader(open(args.crosswalk, encoding="utf-8"))}
    labels = [json.loads(l) for l in Path(args.labels).read_text(encoding="utf-8").splitlines()
              if l.strip()]
    base = list(csv.DictReader(open(args.worklist, encoding="utf-8-sig")))
    start_order = max(int(r["order"]) for r in base) + 1

    rows, skipped = [], []
    for fam in TARGETS:
        picked = 0
        for r in labels:
            c = cw.get(r.get("seed_notation") or "")
            if not c:
                skipped.append((r["id"], f"no_crosswalk:{r.get('seed_notation')}"))
                continue
            t = tax[int(c["canonical_sound_id"])]
            if t["family"] != fam:
                continue
            local = Path(args.cand_dir) / f"{r['id']}.wav"
            if not local.exists():
                skipped.append((r["id"], "audio_not_persisted"))
                continue
            if picked >= 5:
                continue
            picked += 1
            rows.append({
                "order": start_order + len(rows),
                "id": r["id"],
                "sound_family": t["family"],
                "sound_id": c["canonical_sound_id"],
                "candidate_notation": c["seedlib_notation"],
                "canonical_notation": t["notation"],
                "canonical_en": t["en"],
                "slice_wav": local.as_posix(),
                "ref_wav": "",
                "ref_notation": "",
                "split": split_of(r["source"]),
                "source": r["source"],
                "human_notation": "",
                "annotate_status": "pending",
                "annotator": "",
                "note": (f"model 伪标签 conf={float(r['confidence']):.3f} "
                         f"seed_dist={float(r['seed_dist']):.2f}；无种子参考，仅凭听感确认"),
            })
        print(f"[{fam}] 收入 {picked} 条")

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"[out] {out} rows={len(rows)} order {start_order}..{start_order + len(rows) - 1}")
    print(f"[base] {args.worklist} rows={len(base)} "
          f"families={len({r['sound_family'] for r in base})}")
    print(f"[skip] {len(skipped)} 条未收：{Counter(x[1] for x in skipped)}")
    print("[merge] 需追加到 data/sound_seed_annotation.csv 才能进 UI 队列（未执行，等批准）")


if __name__ == "__main__":
    main()
