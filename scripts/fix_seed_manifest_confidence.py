"""seed_manifest 置信度语义修复（PI 裁决 135，26 -> 28 列）。

问题：`sound_confidence` 408/408 全为 `0.0`，语义上是占位值而非概率；
下游任何按阈值/排序使用它的逻辑都会把「未知」误当成「最不确定」。

修复（列名保持兼容，不改 26 列中任何已有值）：
  - 新增 `confidence_scope`，三值枚举：
      crosswalk_exact : 记法字符串精确命中种子库（值恒 1.0，语义是"命中"，不是概率）
      model           : 模型概率（行内必须有 hit_id，本 manifest 无该列 -> 直接拒绝）
      unavailable     : 无法给出置信度，`na_reason` 必填，`sound_confidence` 留空
  - 新增 `na_reason`
  - `sound_confidence == 0.0` 的行数必须清零（0.0 不是"未知"，是"最不确定"）

禁止：用 `sequence_hits.confidence`（hit 级，median 0.7938）回填本表。
      种子级与 hit 级粒度不同，回填即伪造概率。

用法：
  python scripts/fix_seed_manifest_confidence.py            # 落盘（先自动备份）
  python scripts/fix_seed_manifest_confidence.py --dry-run  # 只报差异
"""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

SCOPE_EXACT = "crosswalk_exact"
SCOPE_MODEL = "model"
SCOPE_NA = "unavailable"

INSERT_AFTER = "sound_confidence"
NEW_COLS = ["confidence_scope", "na_reason"]


def md5_of(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def decide(row):
    """返回 (scope, sound_confidence, na_reason)。"""
    label_source = (row.get("label_source") or "").strip()
    score = (row.get("crosswalk_score") or "").strip()

    if label_source == "seed_match":
        # 种子库精确命中：值恒 1.0，语义是字符串命中而非概率
        try:
            exact = float(score) == 1.0
        except ValueError:
            exact = False
        if exact:
            return SCOPE_EXACT, "1.0", ""
        return SCOPE_NA, "", "crosswalk_score_not_exact"

    if label_source == "model":
        prob = (row.get("model_confidence") or "").strip()
        if not prob:
            return SCOPE_NA, "", "model_confidence_missing"
        if not (row.get("hit_id") or "").strip():
            # hit_id 不可追溯的模型概率不得进 manifest
            return SCOPE_NA, "", "hit_id_missing"
        return SCOPE_MODEL, prob, ""

    if label_source == "human":
        return SCOPE_EXACT, "1.0", ""
    return SCOPE_NA, "", f"label_source_{label_source or 'empty'}_has_no_probability"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Data/seed_manifest.csv")
    ap.add_argument("--backup-dir", default="backups")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.manifest)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    fieldnames = list(rows[0].keys())
    before_md5 = hashlib.md5(raw).hexdigest()
    print(f"[in] {path} rows={len(rows)} cols={len(fieldnames)} md5={before_md5}")

    if INSERT_AFTER not in fieldnames:
        raise SystemExit(f"ABORT: 找不到列 {INSERT_AFTER}")
    if any(c in fieldnames for c in NEW_COLS):
        raise SystemExit("ABORT: 新列已存在，本脚本只做一次性插入（防重复跑）")

    out_cols = list(fieldnames)
    pos = out_cols.index(INSERT_AFTER) + 1
    for offset, col in enumerate(NEW_COLS):
        out_cols.insert(pos + offset, col)

    stats = {"changed_conf": 0, "scope": {}}
    for r in rows:
        scope, conf, na = decide(r)
        old = (r.get("sound_confidence") or "").strip()
        if old != conf:
            stats["changed_conf"] += 1
        if scope == SCOPE_NA and not na:
            raise SystemExit(f"ABORT: {r['id']} unavailable 但 na_reason 为空")
        r["na_reason"] = na
        r["confidence_scope"] = scope
        r["sound_confidence"] = conf
        stats["scope"][scope] = stats["scope"].get(scope, 0) + 1

    zero = sum(1 for r in rows if (r["sound_confidence"] or "").strip() not in ("",) and
               float(r["sound_confidence"]) == 0.0)
    na_missing = sum(1 for r in rows if r["confidence_scope"] == SCOPE_NA and not r["na_reason"])
    print(f"[stats] scope={stats['scope']} changed_conf={stats['changed_conf']}")
    print(f"[check] sound_confidence==0.0 rows = {zero} (门槛 0)")
    print(f"[check] unavailable 且 na_reason 为空 = {na_missing} (门槛 0)")

    if args.dry_run:
        print("[dry-run] 不写盘")
        return
    if zero != 0 or na_missing != 0:
        raise SystemExit("ABORT: 门槛未过，不写盘")

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f"2026-10-27_seed_manifest_pre_135_{before_md5}.csv"
    if not backup.exists():
        shutil.copy2(path, backup)
    print(f"[backup] {backup} md5={md5_of(backup)}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    print(f"[out] {path} rows={len(rows)} cols={len(out_cols)} md5={md5_of(path)}")


if __name__ == "__main__":
    main()
