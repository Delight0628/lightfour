"""seed_manifest 契约校验（PI 裁决 135 的验收门槛）。

检查项（任一不过 exit 1）：
  1. 列数 28，且 `confidence_scope` 紧跟 `sound_confidence`，`na_reason` 在其后
  2. `confidence_scope` 只允许 crosswalk_exact / model / unavailable
  3. `sound_confidence == 0.0` 的行数 = 0（0.0 会被阈值当成"最不确定"，与"未知"混淆）
  4. crosswalk_exact -> sound_confidence == 1.0 且 na_reason 为空
  5. unavailable    -> sound_confidence 为空 且 na_reason 非空（100%）
  6. model          -> sound_confidence 是 (0,1] 概率；manifest 若无 hit_id 列则直接判 FAIL
                       （hit 级不可追溯的模型概率不得进 manifest）
  7. id 唯一；sound_id ∈ [0,102]；sound_notation 必须能在 taxonomy 中找到

用法：python scripts/validate_seed_manifest.py [--manifest Data/seed_manifest.csv]
"""
import argparse
import csv
import json
from pathlib import Path

SCOPES = {"crosswalk_exact", "model", "unavailable"}
EXPECTED_COLS = [
    "id", "source", "split", "wav", "wav_local", "in_listen_pack", "sound_id",
    "sound_notation", "sound_en", "sound_cn", "sound_family", "is_short_variant",
    "sound_confidence", "confidence_scope", "na_reason", "label_source",
    "candidate_notation_seedlib", "candidate_family_l2", "nearest_seed",
    "crosswalk_method", "crosswalk_score", "seed_dist", "model_confidence",
    "dur", "start", "annotate_status", "annotator", "note",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Data/seed_manifest.csv")
    ap.add_argument("--taxonomy", default="Data/beatbox_taxonomy.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    cols = list(rows[0].keys())
    fails = []

    if cols != EXPECTED_COLS:
        fails.append(f"列不合契约：{len(cols)} 列；差异 "
                     f"missing={set(EXPECTED_COLS) - set(cols)} extra={set(cols) - set(EXPECTED_COLS)}")

    if "hit_id" not in cols:
        pass  # 下面按 scope 判定

    tax = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    tax_notations = {s["notation"] for s in tax["sounds"]}
    tax_ids = {i for i in range(len(tax["sounds"]))}

    bad_scope, zero_conf, exact_bad, na_bad, model_bad = [], [], [], [], []
    for r in rows:
        scope = (r.get("confidence_scope") or "").strip()
        conf = (r.get("sound_confidence") or "").strip()
        na = (r.get("na_reason") or "").strip()

        if scope not in SCOPES:
            bad_scope.append((r["id"], scope))
            continue
        if conf:
            try:
                if float(conf) == 0.0:
                    zero_conf.append(r["id"])
            except ValueError:
                bad_scope.append((r["id"], f"non-numeric:{conf}"))

        if scope == "crosswalk_exact":
            if conf != "1.0" or na:
                exact_bad.append((r["id"], conf, na))
        elif scope == "unavailable":
            if conf or not na:
                na_bad.append((r["id"], conf, na))
        elif scope == "model":
            if "hit_id" not in cols or not (r.get("hit_id") or "").strip():
                model_bad.append((r["id"], "no hit_id column" if "hit_id" not in cols else "empty hit_id"))
            elif not conf or not (0.0 < float(conf) <= 1.0):
                model_bad.append((r["id"], conf))

    ids = [r["id"] for r in rows]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    bad_sid = [(r["id"], r["sound_id"]) for r in rows if not str(r["sound_id"]).strip().isdigit()
               or int(r["sound_id"]) not in tax_ids]
    bad_not = [(r["id"], r["sound_notation"]) for r in rows if r["sound_notation"] not in tax_notations]

    for name, bad in [("scope 非法", bad_scope), ("sound_confidence==0.0", zero_conf),
                      ("crosswalk_exact 值不符", exact_bad), ("unavailable 缺 na_reason", na_bad),
                      ("model 概率/hit_id 不合规", model_bad), ("id 重复", dup_ids),
                      ("sound_id 越界", bad_sid), ("sound_notation 不在 taxonomy", bad_not)]:
        if bad:
            fails.append(f"{name}: {len(bad)} 条 {bad[:5]}")

    print(f"[in] {args.manifest} rows={len(rows)} cols={len(cols)}")
    for k in sorted(SCOPES):
        print(f"  scope {k:16} = {sum(1 for r in rows if r['confidence_scope'] == k)}")
    print(f"  sound_confidence==0.0 = {len(zero_conf)}")
    print(f"  na_reason 非空 = {sum(1 for r in rows if (r['na_reason'] or '').strip())}")

    if fails:
        print("\n[FAIL]")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("\n[PASS] 8 项契约全过")


if __name__ == "__main__":
    main()
