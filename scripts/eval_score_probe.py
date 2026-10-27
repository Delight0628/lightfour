#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightfour seedlib-consistency probe scorer (识别侧零伪标上界探针).

口径 v2（2026-10-26 锁定，Eval 维护）：
  * 评分单元 = sound_id（0-based，范围 0..102；不要假设 1..103）。
  * 测试集 = data/eval_refaudio_testset.csv，95 行 / 95 唯一 sound_id / 19 family，
    label_source 全部 seedlib_reference。每音色仅 1 条 rendition。
  * 该探针衡量的是「对种子库规则的一致性（师生一致）」，不是独立真值。
    任何分数不得写入 agent.md §0 识别准确率。
  * notation-level 存在硬天花板 97/103 = 94.2%（6 组记法碰撞）；sound_id-level 无天花板。
    碰撞组内命中按 pair-correct 计，不计错，但单独列出。
  * 随机基线 = 1 / n_sounds（当前 95-way = 1.05%）；通过线 top-1 >= 5x 随机基线，
    且 19 family 中 >= 15 个有非零命中。
  * n < 5 的 family 只报数，不判通过/不通过。
  * 音频敏感性一票否决：扰动下预测对输入不敏感（换音频输出全不变）直接不通过。

用法:
  python scripts/eval_score_probe.py --pred preds.csv --out card.md
  python scripts/eval_score_probe.py --pred preds.csv --perturb perturb_preds.csv --out card.md
  python scripts/eval_score_probe.py --self-test

preds.csv 必需列: ref_id, pred_sound_id
  可选列: pred_sound_id_top3（| 或 , 分隔）, pred_notation, pred_en
perturb_preds.csv 列: perturb_id, ref_id, pred_sound_id
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TESTSET = os.path.join(ROOT, "data", "eval_refaudio_testset.csv")
DEFAULT_TAXONOMY = os.path.join(ROOT, "data", "beatbox_taxonomy.json")
PROBE_NAME = "seedlib-consistency probe"


# ---- G5 口径（Eval 锁定 v1，2026-10-27）-------------------------------------
# 短变体是评测切片维度，不是第 104 类；只进分层统计，不进标签空间。
# 配对判据：taxonomy en 名称的 (short) ↔ (Longer/Long/基型) 配对，且 crosswalk
# variant_note 声明"保留独立音色 id，不并入基型"。每条断言见 _check_g5_pairs()。
SHORT_PAIRS = [
    # (短变体 sound_id, 基型 sound_id, 备注)
    (25, 23, "K` Inward K Rimshot (short) <-> K' Inward K Rimshot"),
    (2, 3, "Pw Fat Kick Short <-> Pww Fat Kick Long（种子库无中性 Fat Kick，按短/长配对）"),
    (10, 9, "T' Hi-Hat T (short) <-> T Hi-hat T"),
    (19, 18, "k' K palate snare (short) <-> k K palate snare"),
    (70, 71, "Ww* Inward Hollow Lip Roll (Short) <-> Ww*~ Inward Hollow Lip Roll (Longer)"),
]
SHORT_NO_BASE = [
    # 有短变体但标签空间里没有基型的孤儿：无法组成混淆对，只单列报数
    (65, "Z Vibrator Lip Bass (Short)：taxonomy/crosswalk 均无 Vibrator Lip Bass 基型"),
]


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_taxonomy(path: str):
    """sound_id = list index (0-based)."""
    with open(path, encoding="utf-8") as fh:
        sounds = json.load(fh)["sounds"]
    id2snd = {i: s for i, s in enumerate(sounds)}
    notation2ids = collections.defaultdict(list)
    for i, s in enumerate(sounds):
        notation2ids[s.get("notation", "")].append(i)
    # 记法碰撞组：同记法的 sound_id 集合
    collision_groups = [ids for ids in notation2ids.values() if len(ids) > 1]
    return id2snd, notation2ids, collision_groups


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def parse_top3(value: str):
    if not value:
        return []
    out = []
    for tok in value.replace(",", "|").split("|"):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def score(test_rows, pred_rows, id2snd, notation2ids, collision_groups):
    pred_by_ref = {}
    dup = 0
    for r in pred_rows:
        key = r.get("ref_id") or r.get("id")
        if key in pred_by_ref:
            dup += 1
        pred_by_ref[key] = r

    joined, missing = [], []
    for t in test_rows:
        key = t["ref_id"]
        if key not in pred_by_ref:
            missing.append(key)
            continue
        p = pred_by_ref[key]
        truth_id = int(t["sound_id"])
        pred_id = int(p["pred_sound_id"])
        top3 = parse_top3(p.get("pred_sound_id_top3", ""))
        pred_notation = (p.get("pred_notation") or id2snd.get(pred_id, {}).get("notation", "")).strip()
        truth_notation = t["sound_notation"]
        # pair-correct: 预测落在真值记法碰撞组内
        pair_ids = set(notation2ids.get(truth_notation, [truth_id]))
        pair_ok = pred_id in pair_ids
        joined.append({
            "ref_id": key,
            "truth_id": truth_id,
            "pred_id": pred_id,
            "truth_notation": truth_notation,
            "pred_notation": pred_notation,
            "truth_en": t["sound_en"],
            "pred_en": p.get("pred_en") or id2snd.get(pred_id, {}).get("en", ""),
            "family": t["sound_family"],
            "pred_family": id2snd.get(pred_id, {}).get("family", "unknown"),
            "top1_ok": pred_id == truth_id,
            "top3_ok": (not top3) or (truth_id in top3) or (pred_id == truth_id),
            "notation_raw_ok": pred_notation == truth_notation,
            "notation_pair_ok": pair_ok or pred_notation == truth_notation,
        })
    extra = [k for k in pred_by_ref if k not in {t["ref_id"] for t in test_rows}]
    return joined, missing, extra, dup


def _check_g5_pairs(id2snd, test_rows):
    """配对自检：短变体 is_short_variant=1、基型=0、两者都在测试集、记法不同。"""
    present = {int(t["sound_id"]) for t in test_rows}
    for sv, _, _ in SHORT_PAIRS:
        assert id2snd[sv].get("is_short_variant") in (1, "1", True), f"id {sv} 不是短变体"
    for sv, _ in SHORT_NO_BASE:
        assert id2snd[sv].get("is_short_variant") in (1, "1", True), f"id {sv} 不是短变体"
    for sv, base, _ in SHORT_PAIRS:
        assert id2snd[base].get("is_short_variant") in (0, "0", False, None), f"基型 id {base} 被标成短变体"
        assert sv in present and base in present, f"配对 {sv}<->{base} 有成员不在测试集"
        assert id2snd[sv]["notation"] != id2snd[base]["notation"], f"配对 {sv}<->{base} 记法相同"
    for sv, _ in SHORT_NO_BASE:
        assert sv in present


def g5_stats(joined, id2snd):
    """G5：基型/短变体/全体三栏 + 短变体逐条 + 成对混淆（不参与通过判定）。"""
    by_truth = {j["truth_id"]: j for j in joined}
    sv_ids = {sv for sv, _, _ in SHORT_PAIRS} | {sv for sv, _ in SHORT_NO_BASE}
    base_ids = {b for _, b, _ in SHORT_PAIRS}

    def grp(pred_ids):
        rows = [j for j in joined if j["truth_id"] in pred_ids]
        n = len(rows)
        hit = sum(j["top1_ok"] for j in rows)
        pair = sum(j["notation_pair_ok"] for j in rows)
        return {"n": n, "top1": hit, "not_pair": pair}

    groups = {
        "base": grp({j["truth_id"] for j in joined} - sv_ids),
        "short": grp(sv_ids),
        "all": grp({j["truth_id"] for j in joined}),
    }

    short_rows = []
    pair_by_sv = {sv: base for sv, base, _ in SHORT_PAIRS}
    for sv in sorted(sv_ids):
        j = by_truth.get(sv)
        if j is None:
            continue
        base = pair_by_sv.get(sv)
        if j["top1_ok"]:
            kind = "命中"
        elif base is not None and j["pred_id"] == base:
            kind = "互混→基型"
        else:
            kind = "其他错"
        short_rows.append({
            "ref_id": j["ref_id"], "truth_id": sv, "notation": j["truth_notation"],
            "pred_id": j["pred_id"], "pred_notation": j["pred_notation"], "kind": kind,
            "base_id": base, "base_notation": id2snd[base]["notation"] if base is not None else "",
        })

    pairs = []
    for sv, base, note in SHORT_PAIRS:
        a, b = by_truth.get(sv), by_truth.get(base)
        rec = {
            "sv_id": sv, "sv_notation": id2snd[sv]["notation"],
            "base_id": base, "base_notation": id2snd[base]["notation"], "note": note,
            "sv_pred": a["pred_id"] if a else None,
            "base_pred": b["pred_id"] if b else None,
            "sv_hit": bool(a and a["top1_ok"]),
            "base_hit": bool(b and b["top1_ok"]),
            "sv_as_base": bool(a and a["pred_id"] == base),
            "base_as_sv": bool(b and b["pred_id"] == sv),
        }
        if rec["sv_hit"] and rec["base_hit"]:
            rec["outcome"] = "双命中"
        elif rec["sv_as_base"] and rec["base_as_sv"]:
            rec["outcome"] = "双混淆"
        elif rec["sv_as_base"]:
            rec["outcome"] = "短混基"
        elif rec["base_as_sv"]:
            rec["outcome"] = "基混短"
        elif rec["sv_hit"] or rec["base_hit"]:
            rec["outcome"] = "半命中（另一侧错向非配对）"
        else:
            rec["outcome"] = "双错（错向非配对）"
        pairs.append(rec)

    return {
        "groups": groups, "short_rows": short_rows, "pairs": pairs,
        "orphan": SHORT_NO_BASE,
        "pair_confusions": sum(1 for r in pairs if r["sv_as_base"] or r["base_as_sv"]),
    }


def render_card(args, test_rows, joined, missing, extra, dup, id2snd, collision_groups,
                perturb_summary, verdict_lines):
    n = len(joined)
    n_sounds = len({int(t["sound_id"]) for t in test_rows})
    top1 = sum(j["top1_ok"] for j in joined)
    top3 = sum(j["top3_ok"] for j in joined)
    not_raw = sum(j["notation_raw_ok"] for j in joined)
    not_pair = sum(j["notation_pair_ok"] for j in joined)
    base = 1.0 / n_sounds
    fam = collections.defaultdict(lambda: [0, 0])
    for j in joined:
        fam[j["family"]][0] += 1
        fam[j["family"]][1] += j["top1_ok"]
    fam_hit = sum(1 for v in fam.values() if v[1] > 0)

    L = []
    L.append(f"# 评测卡 · {PROBE_NAME}")
    L.append("")
    L.append(f"- 实验 ID: {args.experiment_id or '(未填)'}")
    L.append(f"- 测试集: `{args.testset}`  md5=`{md5_file(args.testset)}`  样本数={len(test_rows)} / 音色数={n_sounds}")
    L.append(f"- 预测文件: `{args.pred}`  md5=`{md5_file(args.pred)}`")
    if args.perturb:
        L.append(f"- 扰动预测: `{args.perturb}`  md5=`{md5_file(args.perturb)}`")
    L.append(f"- 试听包路径: {args.listen_pack or '(待补)'}")
    L.append("")
    L.append("## 口径声明")
    L.append("")
    L.append(f"- 本卡为 **{PROBE_NAME}**（种子库规则一致性 / 师生一致），**不是独立真值准确率**，"
             "**不得写入 agent.md §0 识别准确率**。")
    L.append("- 评分单元 = sound_id（0-based 0..102）；notation-level 硬天花板 97/103 = 94.2%。")
    L.append(f"- 随机基线 = {base:.4f} ({n_sounds}-way)；通过线 = top-1 ≥ 5× 随机基线 = {5*base:.4f}，"
             f"且 family 非零命中 ≥ 15/19。")
    L.append("- n<5 的 family 只报数，不判通过/不通过。")
    L.append("")
    L.append("## 总体")
    L.append("")
    L.append("| 指标 | 命中/样本 | 值 | 95% CI |")
    L.append("|---|---|---|---|")
    lo, hi = wilson(top1, n)
    L.append(f"| sound_id top-1 | {top1}/{n} | {top1/n:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lo, hi = wilson(top3, n)
    L.append(f"| sound_id top-3 | {top3}/{n} | {top3/n:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lo, hi = wilson(not_raw, n)
    L.append(f"| notation raw | {not_raw}/{n} | {not_raw/n:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lo, hi = wilson(not_pair, n)
    L.append(f"| notation pair-aware（碰撞组内不计错） | {not_pair}/{n} | {not_pair/n:.4f} | [{lo:.4f}, {hi:.4f}] |")
    if dup or missing or extra:
        L.append(f"- ⚠️ 预测文件异常: 重复 ref_id={dup}, 缺失={len(missing)} {missing[:5]}, 多余={len(extra)} {extra[:5]}")
    L.append("")
    g5 = g5_stats(joined, id2snd)
    _check_g5_pairs(id2snd, test_rows)
    g, sr, pr = g5["groups"], g5["short_rows"], g5["pairs"]
    n_sv = g["short"]["n"]
    sv_hit = g["short"]["top1"]
    sv_as_base = sum(1 for r in sr if r["kind"] == "互混→基型")
    L.append("## G5 短变体 / 基型分层（只报数，不参与通过判定）")
    L.append("")
    L.append("> 口径锁定（Eval v1）：`is_short_variant` 只进**分层统计**，不进标签空间（仍 103 类）。"
             "每个短变体 sound_id 在本测试集 **n=1**，逐 id 只报数、不下通过/不通过；"
             "短变体组仅以「命中 x/6」形式给方向性证据。")
    L.append("")
    L.append("| 组 | n | top-1 | 95% CI | notation pair-aware |")
    L.append("|---|---|---|---|---|")
    for key, name in (("base", "基型组"), ("short", "短变体组"), ("all", "全体")):
        v = g[key]
        if v["n"] == 0:
            continue
        lo, hi = wilson(v["top1"], v["n"])
        L.append(f"| {name} | {v['n']} | {v['top1']}/{v['n']} = {v['top1']/v['n']:.4f} | "
                 f"[{lo:.4f}, {hi:.4f}] | {v['not_pair']}/{v['n']} = {v['not_pair']/v['n']:.4f} |")
    L.append("")
    L.append(f"短变体组命中 **{sv_hit}/{n_sv}**，其中误判为对应基型 **{sv_as_base}/{n_sv}**，"
             f"其他错 **{n_sv - sv_hit - sv_as_base}/{n_sv}**。")
    L.append("")
    L.append("### 短变体逐条（n=1/id，只报数）")
    L.append("")
    L.append("| ref_id | sound_id | 真值记法 | 预测 id | 预测记法 | 结果 | 对应基型 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sr:
        b = f"{r['base_id']} {r['base_notation']}" if r["base_id"] is not None else "无基型"
        L.append(f"| {r['ref_id']} | {r['truth_id']} | {r['notation']} | {r['pred_id']} | "
                 f"{r['pred_notation']} | {r['kind']} | {b} |")
    L.append("")
    L.append("### 成对混淆：短变体 ↔ 对应基型（逐对列出，不折算均值）")
    L.append("")
    L.append("| # | 短变体 | 基型 | 短样本→预测 | 基样本→预测 | 结果 |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(pr, 1):
        L.append(f"| {i} | {r['sv_id']} {r['sv_notation']} | {r['base_id']} {r['base_notation']} | "
                 f"{r['sv_pred']} | {r['base_pred']} | {r['outcome']} |")
    L.append("")
    L.append(f"- 成对混淆：**{g5['pair_confusions']}/{len(pr)} 对**出现短↔基互混"
             f"（短→基 {sum(1 for r in pr if r['sv_as_base'])} 对 / 基→短 {sum(1 for r in pr if r['base_as_sv'])} 对）。")
    for sv, note in g5["orphan"]:
        L.append(f"- 无基型孤儿：id {sv} —— {note}。无法组成混淆对，只单列报数。")
    L.append("- 置信度：每 id n=1，Wilson 区间仅具形式意义；不做组间显著性检验（样本不足）。"
             "G5 **不参与**通过/不通过判定，避免把 n=6 的切片写成结论。")
    L.append("")

    L.append("## 分 family")
    L.append("")
    L.append("| family | n | correct | acc | 备注 |")
    L.append("|---|---|---|---|---|")
    for name in sorted(fam):
        k, cnt = fam[name][1], fam[name][0]
        note = "n<5，只报数" if cnt < 5 else ""
        L.append(f"| {name} | {cnt} | {k} | {k/cnt:.4f} | {note} |")
    L.append("")
    L.append(f"family 非零命中: {fam_hit}/{len(fam)}")
    L.append("")
    L.append("## family 块状混淆矩阵（行为真值，列为预测）")
    L.append("")
    names = sorted({j["family"] for j in joined} | {j["pred_family"] for j in joined})
    cm = collections.Counter((j["family"], j["pred_family"]) for j in joined)
    L.append("| true\\pred | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for a in names:
        L.append("| " + a + " | " + " | ".join(str(cm.get((a, b), 0)) for b in names) + " |")
    L.append("")

    if perturb_summary:
        L.append("## 音频敏感性（扰动 P0–P7）")
        L.append("")
        L.append("| perturb_id | kind | n | acc | 与 P0 预测翻转率 |")
        L.append("|---|---|---|---|---|")
        for row in perturb_summary["rows"]:
            L.append(f"| {row['perturb_id']} | {row['kind']} | {row['n']} | {row['acc']:.4f} | {row['flip_rate']:.4f} |")
        L.append("")
        L.append(f"- 全部扰动预测完全不变的样本占比: {perturb_summary['invariant_rate']:.4f}"
                 f"（阈值 ≥0.99 触发一票否决）")
        L.append(f"- 敏感性判定: {perturb_summary['verdict']}")
        L.append("")

    fails = [j for j in joined if not j["top1_ok"]]
    L.append(f"## 失败案例（{len(fails)}/{n}，列前 20）")
    L.append("")
    L.append("| ref_id | 真值 | 预测 | family | 真值记法 | 预测记法 |")
    L.append("|---|---|---|---|---|---|")
    for j in fails[:20]:
        L.append(f"| {j['ref_id']} | {j['truth_en']} | {j['pred_en']} | {j['family']} | "
                 f"{j['truth_notation']} | {j['pred_notation']} |")
    L.append("")
    L.append("## 结论")
    L.append("")
    for line in verdict_lines:
        L.append(f"- {line}")
    L.append("")
    return "\n".join(L)


def evaluate(args):
    test_rows = read_csv(args.testset)
    pred_rows = read_csv(args.pred)
    id2snd, notation2ids, collision_groups = load_taxonomy(args.taxonomy)
    joined, missing, extra, dup = score(test_rows, pred_rows, id2snd, notation2ids, collision_groups)
    if not joined:
        raise SystemExit("预测与测试集没有可对齐的行，检查 ref_id")

    n = len(joined)
    n_sounds = len({int(t["sound_id"]) for t in test_rows})
    top1 = sum(j["top1_ok"] for j in joined) / n
    fam = collections.defaultdict(lambda: [0, 0])
    for j in joined:
        fam[j["family"]][0] += 1
        fam[j["family"]][1] += j["top1_ok"]
    fam_hit = sum(1 for v in fam.values() if v[1] > 0)
    base = 1.0 / n_sounds

    verdict = []
    perturb_summary = None
    if args.perturb:
        perturb_summary = score_perturbations(args, test_rows, id2snd)
        verdict.append(f"音频敏感性: {perturb_summary['verdict']}"
                       + ("（一票否决）" if perturb_summary["veto"] else ""))

    passed = True
    if perturb_summary and perturb_summary["veto"]:
        passed = False
        verdict.append("❌ 不通过：扰动下预测对输入不敏感（换音频输出不变），判定未学到音频映射。")
    if top1 < 5 * base:
        passed = False
        verdict.append(f"❌ 不通过：top-1 {top1:.4f} < 5×随机基线 {5*base:.4f}。")
    else:
        verdict.append(f"✓ top-1 {top1:.4f} ≥ 5×随机基线 {5*base:.4f}。")
    if fam_hit < 15:
        passed = False
        verdict.append(f"❌ 不通过：family 非零命中 {fam_hit}/19 < 15。")
    else:
        verdict.append(f"✓ family 非零命中 {fam_hit}/19 ≥ 15。")
    verdict.append("⚠️ 置信度：snares 样本占比高、12 个 family n<5，汇总值仅供方向参考；"
                   f"n<5 family 不计入通过判定。")
    g5 = g5_stats(joined, id2snd)
    sv_n = g5["groups"]["short"]["n"]
    sv_hit = g5["groups"]["short"]["top1"]
    sv_as_base = sum(1 for r in g5["short_rows"] if r["kind"] == "互混→基型")
    verdict.append(f"G5 短变体组：命中 {sv_hit}/{sv_n}，误判为对应基型 {sv_as_base}/{sv_n}"
                   "（方向性证据，n=1/id，不判通过/不通过）。")
    verdict.append(f"G5 成对混淆：{g5['pair_confusions']}/{len(g5['pairs'])} 对出现短↔基互混；"
                   "id 65 Z 无基型，只报数。")
    verdict.append(f"最终结论：**{'通过' if passed else '不通过'}**（{PROBE_NAME}，不进 §0）")

    card = render_card(args, test_rows, joined, missing, extra, dup, id2snd,
                       collision_groups, perturb_summary, verdict)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(card)
    print(card)
    print(f"\n[written] {args.out}")
    return passed


def score_perturbations(args, test_rows, id2snd):
    truth = {t["ref_id"]: int(t["sound_id"]) for t in test_rows}
    rows = read_csv(args.perturb)
    kinds = {}
    kind_csv = os.path.join(ROOT, "data", "eval_perturbations.csv")
    if os.path.exists(kind_csv):
        for r in read_csv(kind_csv):
            kinds[r["perturb_id"]] = r.get("kind", "")
        # 子变体（P1a/P2b ...）归并回 base kind；同时看 manifest 的 base_perturb_id
        mani = os.path.join(ROOT, "data", "eval_perturb_manifest.csv")
        if os.path.exists(mani):
            for r in read_csv(mani):
                pid, base = r.get("perturb_id", ""), r.get("base_perturb_id", "")
                if pid and base and pid not in kinds:
                    kinds[pid] = r.get("kind", kinds.get(base, ""))
    by_p = collections.defaultdict(dict)
    for r in rows:
        by_p[r["perturb_id"]][r["ref_id"]] = int(r["pred_sound_id"])
    base_id = "P0" if "P0" in by_p else sorted(by_p)[0]
    base_pred = by_p[base_id]
    out_rows = []
    for pid in sorted(by_p):
        preds = by_p[pid]
        common = [k for k in preds if k in truth]
        acc = sum(preds[k] == truth[k] for k in common) / max(1, len(common))
        flip = sum(preds.get(k) != base_pred.get(k) for k in base_pred) / max(1, len(base_pred))
        out_rows.append({"perturb_id": pid, "kind": kinds.get(pid, ""), "n": len(common),
                         "acc": acc, "flip_rate": flip})
    all_ids = sorted(by_p)
    refs = set(base_pred)
    invariant = sum(all(by_p[p].get(k) == base_pred.get(k) for p in all_ids) for k in refs)
    invariant_rate = invariant / max(1, len(refs))
    veto = invariant_rate >= 0.99
    verdict = ("❌ 对输入不敏感" if veto else "✓ 预测随扰动变化，未见“不听音频”迹象")
    return {"rows": out_rows, "invariant_rate": invariant_rate, "veto": veto, "verdict": verdict}


def self_test(args):
    import random
    test_rows = read_csv(args.testset)
    id2snd, notation2ids, _ = load_taxonomy(args.taxonomy)
    ids = sorted(id2snd)
    rng = random.Random(20261026)
    tmp = os.path.join(ROOT, ".ekko-tmp")
    os.makedirs(tmp, exist_ok=True)
    pred_path = os.path.join(tmp, "probe_selftest_pred.csv")
    pert_path = os.path.join(tmp, "probe_selftest_perturb.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref_id", "pred_sound_id", "pred_sound_id_top3"])
        for t in test_rows:
            truth = int(t["sound_id"])
            if rng.random() < 0.6:
                p = truth
            else:
                p = rng.choice([i for i in ids if i != truth])
            w.writerow([t["ref_id"], p, f"{p}|{rng.choice(ids)}|{truth}"])
    with open(pert_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["perturb_id", "ref_id", "pred_sound_id"])
        for t in test_rows:
            truth = int(t["sound_id"])
            for i in range(8):
                p = truth if (i == 0 and rng.random() < 0.6) else rng.choice(ids)
                w.writerow([f"P{i}", t["ref_id"], p])
    args.pred = pred_path
    args.perturb = pert_path
    args.out = os.path.join(tmp, "probe_selftest_card.md")
    args.experiment_id = "SELF-TEST"
    ok = evaluate(args)
    print(f"\n[self-test] ran end-to-end, verdict_passed={ok}（合成预测，不代表任何模型能力）")


def g5_unit_test(args):
    """确定性 G5 单元测试：手工构造互混预测，断言分层与成对表的计数正确。"""
    test_rows = read_csv(args.testset)
    id2snd, notation2ids, _ = load_taxonomy(args.taxonomy)
    truth = {int(t["sound_id"]): t["ref_id"] for t in test_rows}
    # 构造：25→23 短混基 / 23→25 基混短 / 19→18 短混基 / 70→71 短混基 / 65→58 其他错
    # 其余（含 2、10 及 3/9/18/71）全部预测正确
    forced = {25: 23, 23: 25, 19: 18, 70: 71, 65: 58}
    pred_rows = [{"ref_id": truth[t], "pred_sound_id": str(forced.get(t, t))} for t in truth]
    joined, missing, extra, dup = score(test_rows, pred_rows, id2snd, notation2ids, [])
    assert not missing and not extra and not dup, (missing, extra, dup)
    _check_g5_pairs(id2snd, test_rows)
    g5 = g5_stats(joined, id2snd)
    g = g5["groups"]
    assert (g["short"]["n"], g["short"]["top1"]) == (6, 2), g["short"]
    assert (g["base"]["n"], g["base"]["top1"]) == (89, 88), g["base"]
    assert g["all"]["n"] == 95 and g["all"]["top1"] == 90, g["all"]
    kinds = {r["truth_id"]: r["kind"] for r in g5["short_rows"]}
    assert kinds == {2: "命中", 10: "命中", 19: "互混→基型", 25: "互混→基型",
                     65: "其他错", 70: "互混→基型"}, kinds
    outcomes = {r["sv_id"]: r["outcome"] for r in g5["pairs"]}
    assert outcomes == {2: "双命中", 10: "双命中", 19: "短混基", 25: "双混淆", 70: "短混基"}, outcomes
    assert g5["pair_confusions"] == 3, g5["pair_confusions"]
    assert [sv for sv, _ in g5["orphan"]] == [65]
    print("[g5-unit-test] PASS: 分层计数 / 短变体逐条 / 5 对成对混淆 / Z 无基型 全部符合预期")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", default=DEFAULT_TESTSET)
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    ap.add_argument("--pred", default="")
    ap.add_argument("--perturb", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, ".ekko-tmp", "eval_probe_card.md"))
    ap.add_argument("--listen-pack", default="")
    ap.add_argument("--experiment-id", default="")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--g5-unit-test", action="store_true")
    args = ap.parse_args()
    if args.g5_unit_test:
        g5_unit_test(args)
    elif args.self_test:
        self_test(args)
    else:
        if not args.pred:
            ap.error("--pred is required unless --self-test")
        ok = evaluate(args)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
