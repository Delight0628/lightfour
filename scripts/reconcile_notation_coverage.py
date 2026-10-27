"""记法覆盖对账（PI 裁决 136/137）。

产出 outputs/data_notation_reconcile.csv：
  notation / sound_ids / family / in_103 / in_seedlib / n_hits / label_source / action

口径（两个分母必须同时报，禁止只报一个）：
  - id 级：taxonomy 有 103 条 sounds，但只有 97 个唯一 notation（6 组同形异 id）
  - notation 级：97 个唯一记法
覆盖数一律写「id 级 x/103（notation 级 y/97）」。

in_103=N 的记法从分子剔除并标 unknown；本表实测 0 条为 N。
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT_COLS = ["notation", "sound_ids", "family", "in_103", "in_seedlib",
            "n_hits", "label_source", "action"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="Data/sequence_hits.csv")
    ap.add_argument("--seeds", default="Data/seed_manifest.csv")
    ap.add_argument("--taxonomy", default="Data/beatbox_taxonomy.json")
    ap.add_argument("--out", default="outputs/data_notation_reconcile.csv")
    args = ap.parse_args()

    tax = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))["sounds"]
    by_notation = defaultdict(list)
    for i, s in enumerate(tax):
        by_notation[s["notation"]].append(i)

    hits = list(csv.DictReader(open(args.hits, encoding="utf-8-sig")))
    seeds = list(csv.DictReader(open(args.seeds, encoding="utf-8-sig")))

    n_hits = Counter(r["sound_notation"] for r in hits)
    src = defaultdict(Counter)
    for r in hits:
        src[r["sound_notation"]][r["label_source"]] += 1
    seed_notations = {r["sound_notation"] for r in seeds}
    seed_ids = {int(r["sound_id"]) for r in seeds}
    hit_ids = {int(r["sound_id"]) for r in hits}

    rows = []
    for notation in sorted(set(n_hits) | seed_notations):
        ids = by_notation.get(notation, [])
        in_103 = "Y" if ids else "N"
        if not ids:
            action = "剔除分子并标 unknown"
        elif notation not in seed_notations:
            action = "seed 缺项：建议补种子"
        elif n_hits.get(notation, 0) == 0:
            action = "有种子无序列命中：查窗口/过滤"
        else:
            action = "保留"
        rows.append({
            "notation": notation,
            "sound_ids": "|".join(str(i) for i in ids),
            "family": "|".join(sorted({s["family"] for s in tax if s["notation"] == notation})),
            "in_103": in_103,
            "in_seedlib": "Y" if notation in seed_notations else "N",
            "n_hits": n_hits.get(notation, 0),
            "label_source": "|".join(f"{k}:{v}" for k, v in sorted(src[notation].items())),
            "action": action,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    n_tax_ids = len(tax)
    n_tax_notation = len(by_notation)
    out_of_103 = [r for r in rows if r["in_103"] == "N"]
    seed_notation_n = len(seed_notations)
    hit_notation_n = len(set(n_hits))

    print(f"[out] {out} rows={len(rows)}")
    print(f"[taxonomy] ids={n_tax_ids} unique_notation={n_tax_notation} "
          f"dup_notation_groups={sum(1 for v in by_notation.values() if len(v) > 1)}")
    print(f"[coverage] seed  : id 级 {len(seed_ids)}/{n_tax_ids} "
          f"(notation 级 {seed_notation_n}/{n_tax_notation})")
    print(f"[coverage] hits  : id 级 {len(hit_ids)}/{n_tax_ids} "
          f"(notation 级 {hit_notation_n}/{n_tax_notation})")
    print(f"[coverage] union : id 级 {len(seed_ids | hit_ids)}/{n_tax_ids} "
          f"(notation 级 {len(seed_notations | set(n_hits))}/{n_tax_notation})")
    print(f"[out-of-103] {len(out_of_103)} 条: {[r['notation'] for r in out_of_103]}")
    new = [r for r in rows if r["in_seedlib"] == "N"]
    print(f"[seed 缺项] {len(new)} 条记法 / {sum(int(r['n_hits']) for r in new)} hits")
    empty = [r["notation"] for r in rows if r["in_seedlib"] == "Y" and int(r["n_hits"]) == 0]
    print(f"[有种子无命中] {len(empty)} 条: {empty}")


if __name__ == "__main__":
    main()
