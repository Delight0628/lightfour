"""
裁决 82/83「重出门槛」唯一实现：用 **mel 列** 重出三层 + 13c/13p 双 CI + 四元组。

背景（agent.md §14.24 裁决 82/83）：
  - 列族钉 `mel*_vs_true`；`nmel*` 降为「跨列参考，禁引」（不进卡面数字栏、不进结论句、不与 mel 列并排）。
  - 旧「正确三层」整表作废，原因：逐条 min-arm raw 用 mel；「逐条 min-arm gm」实为 nmel 列 median
    （不是 gm）；组级 13p gm `120.4` 只能由 `nmel13p_align` 复现；组级 13c `101.6` 三种变体均复现不出。
  - `120.4` 的 CI 是 nmel 列产物 ⇒ **mel 列 CI 必须重算**，不得沿用 `[107.5,139.0]`。
  - 冻结：新表出来前禁写「镲类是否跨 100%」任一方向；本脚本产出待 @Codec 重出 + @Eval 独立复算双签。

三层定义（全部落在 mel 列，逐条退化率 = mel13X_vs_true / mel75_vs_true − 1）：
  L1 逐条 min-arm   : 每条 hit 取 min(d13c, d13p)（对 13Hz 最有利的逐条下界），聚合 = median
  L2 组级 13c 臂    : 每条 hit 只用 13c，聚合 = median（+ 均值另列）
  L3 组级 13p 臂    : 每条 hit 只用 13p（oracle 取帧），聚合 = median

注意：`per_hit.csv` **没有** 逐条 gain-matched mel 列（脚本内 `metric="gain_matched"` 实际读的是
`nmel*` 前缀，这正是 83 判定的「错标」来源）。因此三层里不出现 gm 层；任何 gm 数字都属 nmel
列，统一进 `cross_column_reference`。

CI 口径 = 按 source 聚类 bootstrap（同一 source 的击打不独立），统计量 = 退化率中位数，
n_boot=2000，seed=20261027，与 `codec_rate_ab.py` 冻结 estimator 同源。

四元组：每行强制打印 (列=mel*_vs_true, 聚合=median, 池=定义+过滤式, n)，缺任一格该行不得入卡。

用法：
  python scripts/mel_layers_reissue.py                      # 全量，写 outputs/mel_layers_reissue_v1/
  python scripts/mel_layers_reissue.py --check outputs/mel_layers_reissue_v1/mel_layers.json
      # 双签用：重算后与给定 JSON 逐行比对，全等 exit 0，否则打印差异行 exit 3
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPECTED_MD5 = "235158c7877f5620e5443790c19f30e8"
EXPECTED_BYTES = 1284083
CODEC = "encodec_24khz / 24kbps / layers 16 / codebook 1024"
ESTIMATOR = "degradation = mel13X_vs_true / mel75_vs_true - 1; agg = median; CI = cluster bootstrap by source"
N_BOOT = 2000
SEED = 20261027
COLUMN = "mel*_vs_true"

POOLS = [
    ("all_hits", "全部击打段（287 序列滑窗内所有 hit）", "全部行", lambda r: True),
    ("click_60_200ms", "时长桶 60–200ms（跨 12 family，**不是** clicks 家族）", "0.060 <= dur <= 0.200", lambda r: r["is_click_60_200"] == "1"),
    ("hihats", "sound_family == hihats", "sound_family == 'hihats'", lambda r: r["sound_family"] == "hihats"),
    ("cymbal", "sound_family == cymbal（镲类单列，禁与 hihats 合并）", "sound_family == 'cymbal'", lambda r: r["sound_family"] == "cymbal"),
]


def md5_file(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def degradation(rows, key):
    e75 = np.array([float(r["mel75_vs_true"]) for r in rows], dtype=float)
    e13 = np.array([float(r[key]) for r in rows], dtype=float)
    return (e13 - e75) / np.maximum(e75, 1e-9), e75, e13


def minarm_degr(rows):
    e75 = np.array([float(r["mel75_vs_true"]) for r in rows], dtype=float)
    dc = np.array([float(r["mel13c_vs_true"]) for r in rows], dtype=float)
    dp = np.array([float(r["mel13p_vs_true"]) for r in rows], dtype=float)
    d = np.minimum((dc - e75) / np.maximum(e75, 1e-9), (dp - e75) / np.maximum(e75, 1e-9))
    return d


def cluster_median_ci(rows, values):
    """按 source 聚类 bootstrap；返回 (point, ci_lo, ci_hi, n_sources)。"""
    by_src = defaultdict(list)
    for r, v in zip(rows, values):
        by_src[r["source"]].append(float(v))
    srcs = sorted(by_src)
    arrs = [np.array(by_src[s]) for s in srcs]
    point = float(np.median(np.concatenate(arrs)))
    if len(srcs) < 2:
        return point, None, None, len(srcs)
    rng = np.random.default_rng(SEED)
    meds = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(srcs), len(srcs))
        meds[b] = np.median(np.concatenate([arrs[i] for i in pick]))
    return point, float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), len(srcs)


def row_out(pool, definition, filt, layer, rows, values, nseq):
    d = np.asarray(values, dtype=float)
    point, lo, hi, nsrc = cluster_median_ci(rows, d)
    return {
        # ---- 四元组（每行强制，缺一不可入卡）----
        "column": COLUMN,
        "aggregate": "median",
        "pool": pool,
        "filter": filt,
        "n": len(rows),
        # ---- 补充 ----
        "layer": layer,
        "pool_definition": definition,
        "n_sequences": nseq,
        "n_sources": nsrc,
        "degr_median_pct": round(point * 100, 1),
        "degr_mean_pct": round(float(d.mean()) * 100, 1),
        "ci95_median_pct": None if lo is None else [round(lo * 100, 1), round(hi * 100, 1)],
        "ci_method": "cluster bootstrap by source" if lo is not None else "n_sources<2，只报点估计",
        "n_boot": N_BOOT if lo is not None else 0,
        "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-hit", default="outputs/codec_rate_ab/per_hit.csv")
    ap.add_argument("--out-dir", default="outputs/mel_layers_reissue_v1")
    ap.add_argument("--check", default=None, help="双签比对：与给定 JSON 逐行比 median/CI/n")
    args = ap.parse_args()

    ph = Path(args.per_hit)
    got_md5, got_bytes = md5_file(ph), ph.stat().st_size
    if got_md5 != EXPECTED_MD5 or got_bytes != EXPECTED_BYTES:
        raise SystemExit(
            f"[HARD FAIL] per_hit.csv 指纹不符：md5={got_md5} bytes={got_bytes} "
            f"期望 md5={EXPECTED_MD5} bytes={EXPECTED_BYTES}"
        )

    import csv
    with ph.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    nseq = len({r["sequence_id"] for r in rows})
    if len(rows) != 5472 or nseq != 287:
        raise SystemExit(f"[HARD FAIL] 行数/序列数异常：rows={len(rows)} seqs={nseq}")

    layers = []
    for pool, definition, filt, fn in POOLS:
        sel = [r for r in rows if fn(r)]
        nseq_pool = len({r["sequence_id"] for r in sel})
        layers.append(row_out(pool, definition, filt, "L1 逐条 min-arm（raw）", sel, minarm_degr(sel), nseq_pool))
        layers.append(row_out(pool, definition, filt, "L2 组级 13c 臂（raw）", sel, degradation(sel, "mel13c_vs_true")[0], nseq_pool))
        layers.append(row_out(pool, definition, filt, "L3 组级 13p 臂（raw）", sel, degradation(sel, "mel13p_vs_true")[0], nseq_pool))

    # ---------------- M1：16 family 分层表（287 序列 / 5,472 击打）
    fams = defaultdict(list)
    for r in rows:
        fams[r["sound_family"]].append(r)
    families = []
    for fam in sorted(fams, key=lambda k: (-len(fams[k]), k)):
        sel = fams[fam]
        seqs = len({r["sequence_id"] for r in sel})
        e75 = np.array([float(r["mel75_vs_true"]) for r in sel])
        l1 = row_out(f"family:{fam}", f"sound_family == '{fam}'", f"sound_family == '{fam}'", "L1 逐条 min-arm（raw）", sel, minarm_degr(sel), seqs)
        l2 = row_out(f"family:{fam}", f"sound_family == '{fam}'", f"sound_family == '{fam}'", "L2 组级 13c 臂（raw）", sel, degradation(sel, "mel13c_vs_true")[0], seqs)
        l3 = row_out(f"family:{fam}", f"sound_family == '{fam}'", f"sound_family == '{fam}'", "L3 组级 13p 臂（raw）", sel, degradation(sel, "mel13p_vs_true")[0], seqs)
        families.append({
            "family": fam, "n": len(sel), "n_sequences": seqs, "n_sources": l2["n_sources"],
            "mel75_median": round(float(np.median(e75)), 4),
            "minarm_median_pct": l1["degr_median_pct"], "minarm_ci95": l1["ci95_median_pct"],
            "arm13c_median_pct": l2["degr_median_pct"], "arm13c_ci95": l2["ci95_median_pct"],
            "arm13p_median_pct": l3["degr_median_pct"], "arm13p_ci95": l3["ci95_median_pct"],
            "column": COLUMN, "aggregate": "median",
            "filter": f"sound_family == '{fam}'",
            "confidence_note": "n<30 仅报数，不入结论句" if len(sel) < 30 else "",
        })

    # ---------------- 跨列参考：nmel 行（禁引）
    cross = []
    for pool, definition, filt, fn in POOLS:
        sel = [r for r in rows if fn(r)]
        for arm in ("13c", "13p"):
            key = f"nmel{arm}_vs_true"
            d = (np.array([float(r[key]) for r in sel]) - np.array([float(r["nmel75_vs_true"]) for r in sel])) / \
                np.maximum(np.array([float(r["nmel75_vs_true"]) for r in sel]), 1e-9)
            point, lo, hi, nsrc = cluster_median_ci(sel, d)
            cross.append({
                "column": "nmel*_vs_true", "aggregate": "median", "pool": pool, "filter": filt,
                "n": len(sel), "arm": arm, "n_sources": nsrc,
                "degr_median_pct": round(point * 100, 1),
                "ci95_median_pct": None if lo is None else [round(lo * 100, 1), round(hi * 100, 1)],
                "label": "跨列参考，禁引（不进卡面数字栏、不进结论句、不与 mel 列并排）",
                "quote_allowed": False,
            })

    out = {
        "spec": "mel_layers_reissue v1（agent.md 裁决 82/83 重出门槛）",
        "codec": CODEC,
        "estimator": ESTIMATOR,
        "seed": SEED, "n_boot": N_BOOT,
        "input": {"path": str(ph).replace("\\", "/"), "md5": got_md5, "bytes": got_bytes,
                  "n_sequences": nseq, "n_hits": len(rows),
                  "n_families": len(fams)},
        "code_sha256": sha256_file(__file__),
        "spec_sha256": hashlib.sha256(
            f"{ESTIMATOR}|{COLUMN}|{N_BOOT}|{SEED}|{EXPECTED_MD5}".encode()).hexdigest(),
        "layers": layers,
        "families_16": families,
        "cross_column_reference": cross,
        "notes": [
            "三层全部落在 mel 列；per_hit.csv 无逐条 gain-matched mel 列，故不出现 gm 层。",
            "13c/13p 双 CI 已按 mel 列重算；nmel 列旧 CI（如 120.4 [107.5,139.0]）不得沿用。",
            "镲类是否跨 100% 在双签前禁写任一方向（裁决 83 冻结）。",
            "每行四元组 = (column, aggregate, pool+filter, n)；缺任一格不得入卡。",
        ],
    }

    if args.check:
        prev = json.loads(Path(args.check).read_text(encoding="utf-8"))
        diffs = []
        for section in ("layers", "families_16", "cross_column_reference"):
            a, b = prev.get(section, []), out[section]
            if len(a) != len(b):
                diffs.append(f"{section}: 行数 {len(a)} vs {len(b)}")
                continue
            for i, (x, y) in enumerate(zip(a, b)):
                for k in ("degr_median_pct", "ci95_median_pct", "n", "n_sources"):
                    if k in x and x.get(k) != y.get(k):
                        diffs.append(f"{section}[{i}] {x.get('pool', x.get('family'))} {k}: {x.get(k)} vs {y.get(k)}")
        if diffs:
            print("DIFF（两造不同值）：")
            for d in diffs:
                print("  " + d)
            sys.exit(3)
        print(f"CHECK PASS：{__file__} 逐行同值（layers/families_16/cross_column_reference）")
        return

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "mel_layers.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    # ---------------- Markdown
    L = ["# mel 列三层重出（裁决 82/83 重出门槛）", ""]
    L.append(f"- 输入：`{out['input']['path']}` md5 `{got_md5}` / {got_bytes:,}B / "
             f"{nseq} 序列 / {len(rows):,} 击打 / {len(fams)} family")
    L.append(f"- 列族：`{COLUMN}`（`nmel*` 仅作跨列参考，禁引）")
    L.append(f"- estimator：`{ESTIMATOR}`；n_boot={N_BOOT}，seed={SEED}")
    L.append(f"- codec：{CODEC}")
    L.append(f"- `code_sha256`：`{out['code_sha256'][:16]}`；`spec_sha256`：`{out['spec_sha256'][:16]}`")
    L.append("")
    L.append("## 三层（mel 列，四个池）")
    L.append("")
    L.append("| 池 | 层 | n | 序列 | 源 | 退化中位 | 退化均值 | 95% CI（按 source 聚类） |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in layers:
        ci = r["ci95_median_pct"]
        cis = "n/a（源<2）" if ci is None else f"[{ci[0]}, {ci[1]}]"
        L.append(f"| {r['pool']} | {r['layer']} | {r['n']} | {r['n_sequences']} | {r['n_sources']} | "
                 f"{r['degr_median_pct']}% | {r['degr_mean_pct']}% | {cis} |")
    L.append("")
    L.append("四元组（每行强制）：列 `mel*_vs_true` / 聚合 `median` / 池定义与过滤式见下 / n 见上表。")
    L.append("")
    L.append("| 池 | 定义 | 过滤式 |")
    L.append("|---|---|---|")
    for pool, definition, filt, _ in POOLS:
        L.append(f"| {pool} | {definition} | `{filt}` |")
    L.append("")
    L.append("## M1：16 family 分层（同列族、同 estimator）")
    L.append("")
    L.append("| family | n | 序列 | 源 | mel75 中位 | min-arm 中位 | 13c 中位 [CI] | 13p 中位 [CI] |")
    L.append("|---|---:|---:|---:|---:|---:|---|---|")
    for f in families:
        c2 = "n/a" if f["arm13c_ci95"] is None else f"[{f['arm13c_ci95'][0]}, {f['arm13c_ci95'][1]}]"
        c3 = "n/a" if f["arm13p_ci95"] is None else f"[{f['arm13p_ci95'][0]}, {f['arm13p_ci95'][1]}]"
        note = " ⚠n<30" if f["confidence_note"] else ""
        L.append(f"| {f['family']}{note} | {f['n']} | {f['n_sequences']} | {f['n_sources']} | "
                 f"{f['mel75_median']} | {f['minarm_median_pct']}% | {f['arm13c_median_pct']}% {c2} | "
                 f"{f['arm13p_median_pct']}% {c3} |")
    L.append("")
    L.append("## 跨列参考（`nmel*` 列，禁引）")
    L.append("")
    L.append("| 池 | 臂 | n | 退化中位 | 95% CI | 状态 |")
    L.append("|---|---|---:|---:|---|---|")
    for c in cross:
        ci = "n/a" if c["ci95_median_pct"] is None else f"[{c['ci95_median_pct'][0]}, {c['ci95_median_pct'][1]}]"
        L.append(f"| {c['pool']} | {c['arm']} | {c['n']} | {c['degr_median_pct']}% | {ci} | 跨列参考，禁引 |")
    L.append("")
    L.append("## 双签用法")
    L.append("")
    L.append("```bash")
    L.append("python scripts/mel_layers_reissue.py --check outputs/mel_layers_reissue_v1/mel_layers.json")
    L.append("# 逐行同值 → exit 0 打印 CHECK PASS；有差异 → exit 3 逐行打印")
    L.append("```")
    L.append("")
    L.append("## 冻结（裁决 83）")
    L.append("")
    L.append("- 双签前禁写「镲类是否跨 100%」任一方向；本表只给数，不给结论。")
    L.append("- `nmel*` 行不得进卡面数字栏、不得进结论句、不得与 mel 列并排。")
    L.append("- 79 的 `176.2 / 145.6 / 100.6` 只对应 L1 层，不得与 L2/L3 混写成同一层。")
    (od / "MEL_LAYERS.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    print(f"wrote {od}/mel_layers.json 和 MEL_LAYERS.md")
    print(f"input md5 {got_md5} bytes {got_bytes}")
    print(f"code_sha256 {out['code_sha256']}")
    for r in layers:
        if r["pool"] in ("click_60_200ms", "hihats", "cymbal"):
            print(f"  {r['pool']:14s} {r['layer']:26s} n={r['n']:5d} med={r['degr_median_pct']:7.1f}% ci={r['ci95_median_pct']}")


if __name__ == "__main__":
    main()
