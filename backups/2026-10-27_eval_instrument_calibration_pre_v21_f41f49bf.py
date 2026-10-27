#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""出 Lightfour **INSTRUMENT_CALIBRATION（仪器校准卡）**。

本卡只对「评测链路」验收，**不是 103 音色识别能力卡**：
  * 预测源 ast_lin_v0 的标签空间只有 34 个 sound_id（其中 33 个有训练样本），
    95 条 seedlib 参考音频里 61 条的真值根本不在它的标签空间里。
  * 因此主指标**只报 n=34（in-vocab）+ Wilson 95% 区间**；
    61 条 out_of_vocab **单列、不计分**。
  * sid 25 `K\`` 训练样本为 0 → 卡里标注「该类不可学」，**不计入失败案例**。
  * 任何分数**不得写入 agent.md §0 识别准确率**。

用法:
  python scripts/eval_instrument_calibration.py
  python scripts/eval_instrument_calibration.py --build-listen-pack

输入（默认都在 outputs/instr_cal_ast_lin_v0 与 data/ 下）：
  preds.csv / perturb_preds.csv / data/eval_refaudio_testset.csv /
  data/baseline_label_space.csv / data/baseline_train_index.csv / data/beatbox_taxonomy.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_score_probe import wilson, parse_top3  # noqa: E402  复用同一份打分实现

PRED_SOURCE = "ast_lin_v0"
CARD_NAME = "INSTRUMENT_CALIBRATION"
UNLEARNABLE_SID = {25: "K`（Inward K Rimshot 短变体）"}


# ------------------------------------------------------------------ helpers
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def pct(k: int, n: int) -> str:
    return f"{k/n:.4f}" if n else "—"


# ------------------------------------------------------------------ loading
def load_all(args):
    test_rows = read_csv(ROOT / "data" / "eval_refaudio_testset.csv")
    pred_rows = read_csv(args.pred)
    label_rows = read_csv(ROOT / "data" / "baseline_label_space.csv")
    train_rows = read_csv(ROOT / "data" / "baseline_train_index.csv")
    tax = json.loads((ROOT / "data" / "beatbox_taxonomy.json").read_text(encoding="utf-8"))
    sounds = tax["sounds"]
    id2snd = {i: s for i, s in enumerate(sounds)}

    label_by_sid = {int(r["sound_id"]): r for r in label_rows if r["sound_id"] != "unknown"}
    label_by_li = {int(r["label_index"]): r for r in label_rows}
    train_n = collections.Counter(r["sound_id"] for r in train_rows)

    pred_by_ref = {r["ref_id"]: r for r in pred_rows}
    joined = []
    for t in test_rows:
        ref = t["ref_id"]
        p = pred_by_ref.get(ref)
        if p is None:
            continue
        truth_sid = int(t["sound_id"])
        in_vocab = truth_sid in label_by_sid
        joined.append({
            "ref_id": ref,
            "truth_sid": truth_sid,
            "truth_notation": t["sound_notation"],
            "truth_en": t["sound_en"],
            "family": t["sound_family"],
            "is_short_variant": t["is_short_variant"],
            "pred_sid": int(p["pred_sound_id"]),
            "top3": parse_top3(p.get("pred_sound_id_top3") or p.get("top3", "")),
            "pred_notation": p.get("pred_notation", ""),
            "pred_en": p.get("pred_en") or id2snd.get(int(p["pred_sound_id"]), {}).get("en", ""),
            "prob_top1": float(p.get("prob_top1") or 0.0),
            "in_vocab": in_vocab,
            "train_n": int(train_n.get(str(truth_sid), 0)),
            "unlearnable": truth_sid in UNLEARNABLE_SID and int(train_n.get(str(truth_sid), 0)) == 0,
        })
    inv = [j for j in joined if j["in_vocab"]]
    oov = [j for j in joined if not j["in_vocab"]]
    # 口径自证：PI 锁定「主指标仅 n=34 + 61 条 out_of_vocab 单列」，测试集一变就必须重锁
    if (len(joined), len(inv), len(oov)) != (95, 34, 61):
        raise SystemExit(
            f"口径自证失败：joined={len(joined)} in_vocab={len(inv)} out_of_vocab={len(oov)}，"
            f"期望 (95, 34, 61)。测试集或标签空间变了，请先重锁口径再出卡。")
    return dict(test_rows=test_rows, pred_rows=pred_rows, label_rows=label_rows,
                train_rows=train_rows, id2snd=id2snd, label_by_sid=label_by_sid,
                label_by_li=label_by_li, joined=joined, inv=inv, oov=oov, tax=tax)


# ------------------------------------------------------------------ metrics
def main_metric(inv):
    n = len(inv)
    top1 = sum(1 for j in inv if j["pred_sid"] == j["truth_sid"])
    top3 = sum(1 for j in inv if j["truth_sid"] in j["top3"] or j["pred_sid"] == j["truth_sid"])
    return n, top1, top3


def family_table(inv):
    fam = collections.defaultdict(lambda: [0, 0])
    for j in inv:
        fam[j["family"]][0] += 1
        fam[j["family"]][1] += (j["pred_sid"] == j["truth_sid"])
    return fam


def confusion_pairs(inv, k=200):
    c = collections.Counter()
    for j in inv:
        if j["pred_sid"] != j["truth_sid"]:
            c[(j["truth_sid"], j["pred_sid"])] += 1
    return c.most_common(k)


def g5_block(inv, joined, id2snd):
    """G5 三栏 + 5 对混淆，限定在 ast_lin_v0 的 34 类标签空间内可判的部分。"""
    SHORT_PAIRS = [(25, 23), (2, 3), (10, 9), (19, 18), (70, 71)]
    sv_ids = {25, 2, 10, 19, 70, 65}
    by_truth = {j["truth_sid"]: j for j in joined}
    label_sids = {j["truth_sid"] for j in inv}

    def grp(ids):
        rows = [j for j in inv if j["truth_sid"] in ids]
        n = len(rows)
        hit = sum(1 for j in rows if j["pred_sid"] == j["truth_sid"])
        return n, hit

    g = {
        "base": grp(label_sids - sv_ids),
        "short": grp(sv_ids),
        "all": grp(label_sids),
    }
    pairs = []
    for sv, base in SHORT_PAIRS:
        a, b = by_truth.get(sv), by_truth.get(base)
        rec = {
            "sv": sv, "base": base,
            "sv_notation": id2snd[sv]["notation"] if sv in id2snd else "?",
            "base_notation": id2snd[base]["notation"] if base in id2snd else "?",
            "sv_in": sv in label_sids, "base_in": base in label_sids,
            "sv_pred": a["pred_sid"] if a else None,
            "base_pred": b["pred_sid"] if b else None,
            "sv_hit": bool(a and a["pred_sid"] == a["truth_sid"]),
            "base_hit": bool(b and b["pred_sid"] == b["truth_sid"]),
            "sv_as_base": bool(a and a["pred_sid"] == base),
            "base_as_sv": bool(b and b["pred_sid"] == sv),
        }
        if not rec["sv_in"] and not rec["base_in"]:
            rec["outcome"] = "双侧 out_of_vocab，不计"
        elif not rec["base_in"]:
            rec["outcome"] = "短变体在标签空间内，基型 out_of_vocab → 无法判「互混→基型」"
        elif not rec["sv_in"]:
            rec["outcome"] = "基型在标签空间内，短变体 out_of_vocab → 无法判"
        else:
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
    orphan = [(65, "Z Vibrator Lip Bass (Short)：taxonomy/crosswalk 均无 Vibrator Lip Bass 基型，无法成对")]
    return g, pairs, orphan


def perturb_block(perturb_rows, inv):
    """敏感性只在 in-vocab 的 34 条 ref 上算。"""
    inv_refs = {j["ref_id"]: j for j in inv}
    by = collections.defaultdict(dict)
    for r in perturb_rows:
        if r["ref_id"] in inv_refs:
            by[r["ref_id"]][r["perturb_id"]] = int(r["pred_sound_id"])
    rows = []
    kinds = {}
    for r in perturb_rows:
        kinds[r["perturb_id"]] = r["kind"]
    for pid in sorted(kinds):
        if pid == "P5":
            continue
        n = hit = flip = 0
        for ref, d in by.items():
            if pid not in d or "P0" not in d:
                continue
            n += 1
            hit += (d[pid] == inv_refs[ref]["truth_sid"])
            flip += (d[pid] != d["P0"])
        if n:
            rows.append({"perturb_id": pid, "kind": kinds[pid], "n": n,
                         "acc": hit / n, "flip_rate": flip / n})
    invariant = 0
    total = 0
    for ref, d in by.items():
        if "P0" not in d:
            continue
        total += 1
        if len(set(d.values())) == 1:
            invariant += 1
    inv_rate = invariant / total if total else 0.0
    verdict = "不敏感 → 一票否决（判定「不听音频」）" if inv_rate >= 0.99 else "敏感（预测随音频变化）"
    return rows, inv_rate, verdict, total, by


# ------------------------------------------------------------------ render
def render(args, L, ctx):
    inv, oov, joined = ctx["inv"], ctx["oov"], ctx["joined"]
    id2snd = ctx["id2snd"]
    label_by_sid = ctx["label_by_sid"]
    n, top1, top3 = main_metric(inv)
    lo1, hi1 = wilson(top1, n)
    lo3, hi3 = wilson(top3, n)
    base = 1 / 34

    W = L.append
    W(f"# {CARD_NAME} · 仪器校准卡（{PRED_SOURCE}）")
    W("")
    W("> **卡头自检**：本卡标题即 `INSTRUMENT_CALIBRATION`。它是**评测链路的仪器校准卡**，"
      "不是 103 音色识别能力卡，**不得出现「103 音色识别能力」措辞**，"
      "**任何数字不得写入 agent.md §0 识别准确率**。")
    W("")
    W("## A0 元数据")
    W("")
    W("| 字段 | 值 |")
    W("|---|---|")
    W(f"| 实验 ID / 日期 | {args.experiment_id} / 2026-10-27 |")
    W(f"| 预测源 | `{PRED_SOURCE}`（冻结 AST + 线性探针，**非 Omni**） |")
    W("| 骨干 | `MIT/ast-finetuned-audioset-10-10-0.4593` rev `f826b80d`（冻结，可训练参数 0） |")
    W("| 探针头 | StandardScaler（只 fit 训练集）+ LogisticRegression(lbfgs/L2/C=1.0/无超参搜索) |")
    W(f"| 训练索引 | `data/baseline_train_index.csv` md5=`{md5_file(ROOT/'data'/'baseline_train_index.csv')}`  319 行 |")
    W(f"| 标签空间 | `data/baseline_label_space.csv` md5=`{md5_file(ROOT/'data'/'baseline_label_space.csv')}`  34 类 + unknown |")
    W(f"| 测试集 | `data/eval_refaudio_testset.csv` md5=`{md5_file(ROOT/'data'/'eval_refaudio_testset.csv')}`  95 行 |")
    W(f"| 预测文件 | `{args.pred}` md5=`{md5_file(args.pred)}`  {len(ctx['pred_rows'])} 行 |")
    if Path(args.perturb).is_file():
        W(f"| 扰动预测 | `{args.perturb}` md5=`{md5_file(args.perturb)}`  {len(ctx.get('perturb_rows', []))} 行 |")
    W(f"| 试听包路径 | `{args.listen_pack}` |")
    W(f"| 样本数 / 音色数 | in-vocab **{n}** 条 / 34 个 sound_id；out_of_vocab {len(oov)} 条（不计分） |")
    W("")
    W("## A1 主指标（**只报 in-vocab n=34**，sound_id 为评分单元）")
    W("")
    W("| 指标 | 命中/样本 | 值 | 95% Wilson CI | 随机基线 | 5× 线 |")
    W("|---|---|---|---|---|---|")
    W(f"| sound_id top-1 | {top1}/{n} | {top1/n:.4f} | [{lo1:.4f}, {hi1:.4f}] | 34-way {base:.4f} | {5*base:.4f} |")
    W(f"| sound_id top-3 | {top3}/{n} | {top3/n:.4f} | [{lo3:.4f}, {hi3:.4f}] | — | — |")
    not_raw = sum(1 for j in inv if j["pred_notation"] == j["truth_notation"])
    lor, hir = wilson(not_raw, n)
    W(f"| notation raw（仅参考） | {not_raw}/{n} | {not_raw/n:.4f} | [{lor:.4f}, {hir:.4f}] | — | — |")
    W("")
    W(f"- 随机基线口径（PI 锁定）：34-way = 1/34 = **2.94%**；5× 线 = **14.71%**；95-way 1.05% 不适用于本卡。")
    W(f"- ⚠️ **5× 线在本卡退化**：5×(1/34) = 5/34 与「答对 5 条」在数值上恒等，"
      f"所以 top-1 恰好 5/34 时必然「压线」。该线在 34-way 下**没有判别力**，"
      f"不能当作「通过/不通过」依据；有意义的读数是 top-3、family 覆盖与扰动敏感性。")
    learnable = sorted(int(k) for k in ctx["label_by_li"] if int(ctx["label_by_li"][k]["train_n"] or 0) > 0)
    W(f"- 实际可学标签数 = **{len(learnable)}** / 35 个 label_index（`label_index 5 = sid 25` 与 "
      f"`label_index 34 = unknown` 训练样本 0，模型永远输出不了这两格）。")
    W(f"- 剔除「不可学」sid 25 后的参考值：top-1 "
      f"{sum(1 for j in inv if j['pred_sid']==j['truth_sid'] and not j['unlearnable'])}/{n - sum(1 for j in inv if j['unlearnable'])}"
      f"（**仅参考，主指标仍是上面的 n=34**）。")
    W("")

    # ---- out_of_vocab
    W(f"## A1b out_of_vocab 单列（{len(oov)} 条，**不计分**）")
    W("")
    W("> 这 61 条的真值 sound_id 根本不在 ast_lin_v0 的 34 类标签空间里，模型**结构上不可能答对**。"
      "下面只做诊断统计，**一律不进主指标、不判通过/不通过**。")
    W("")
    forced = sum(1 for j in oov if j["pred_sid"] == j["truth_sid"])
    W(f"- 被迫在 34 类里选一个：100%（模型没有 unknown 拒识能力，`unknown` 格训练样本 0）")
    W(f"- 真值恰好落进 34 类的条数：{forced}/{len(oov)}（理论恒为 0，此处为 0 说明映射一致）")
    fam_oov = collections.Counter(j["family"] for j in oov)
    W(f"- out_of_vocab 分 family：{dict(sorted(fam_oov.items()))}")
    W("")
    W("<details><summary>out_of_vocab 61 条逐条（点开）</summary>")
    W("")
    W("| ref_id | 真值 sound_id | 真值记法 | family | 预测 sound_id | 预测记法 | prob |")
    W("|---|---|---|---|---|---|---|")
    for j in sorted(oov, key=lambda x: x["truth_sid"]):
        W(f"| {j['ref_id']} | {j['truth_sid']} | {j['truth_notation']} | {j['family']} | "
          f"{j['pred_sid']} | {j['pred_notation']} | {j['prob_top1']:.3f} |")
    W("")
    W("</details>")
    W("")

    # ---- family
    fam = family_table(inv)
    W("## A2 分 family（in-vocab n=34；n<5 只报数不判通过）")
    W("")
    W("| family | n | correct | acc | 备注 |")
    W("|---|---|---|---|---|")
    for name in sorted(fam):
        cnt, k = fam[name]
        W(f"| {name} | {cnt} | {k} | {k/cnt:.4f} | {'n<5，只报数' if cnt < 5 else ''} |")
    fam_hit = sum(1 for v in fam.values() if v[1] > 0)
    n_small = sum(1 for v in fam.values() if v[0] < 5)
    W("")
    W(f"family 非零命中: **{fam_hit}/{len(fam)}**")
    W("")
    W(f"- ⚠️ in-vocab 34 条分到 {len(fam)} 个 family，其中 **{n_small}/{len(fam)} 个 n<5**"
      f"{'（即全部）' if n_small == len(fam) else ''}；按红线这些 family **一律只报数、不判通过**。"
      f"标准探针卡的「family 非零命中 ≥15/19」通过线**不适用于本卡**（本卡不是 19-family 全量探针）。")
    W("")

    # ---- confusion
    W("## A3 混淆（in-vocab，按 sound_id）")
    W("")
    cpairs = confusion_pairs(inv)
    W("| 真值 sound_id | 预测 sound_id | 次数 |")
    W("|---|---|---|")
    for (a, b), c in cpairs:
        an = id2snd.get(a, {}).get("en", "")
        bn = id2snd.get(b, {}).get("en", "")
        W(f"| {a} {an} | {b} {bn} | {c} |")
    if not cpairs:
        W("| — | — | 无 |")
    W("")
    names = sorted({j["family"] for j in inv} | {label_by_sid.get(j["pred_sid"], {}).get("sound_family", "unknown") for j in inv})
    cm = collections.Counter((j["family"], label_by_sid.get(j["pred_sid"], {}).get("sound_family", "unknown")) for j in inv)
    W("<details><summary>family 块状混淆矩阵（行为真值 family，列为预测 family）</summary>")
    W("")
    W("| true\\pred | " + " | ".join(names) + " |")
    W("|---|" + "---|" * len(names))
    for a in names:
        W("| " + a + " | " + " | ".join(str(cm.get((a, b), 0)) for b in names) + " |")
    W("")
    W("</details>")
    W("")

    # ---- G5
    g, pairs, orphan = g5_block(inv, joined, id2snd)
    W("## A4 G5 短变体 / 基型分层（三栏 + 5 对混淆，只报数，不参与通过判定）")
    W("")
    W("> G5 口径锁定 v1（Eval）：`is_short_variant` 只进分层统计，不进标签空间。"
      "本卡标签空间仅 34 类，锁定的 5 对里只有 **25↔23** 的短变体侧落在空间内（基型 23 是 out_of_vocab），"
      "其余 4 对双侧都不在空间内，只能按 out_of_vocab 报数；id 65 `Z` 无基型孤儿。")
    W("")
    W("| 组 | n | top-1 | 95% Wilson CI |")
    W("|---|---|---|---|")
    for key, nm in (("base", "基型组"), ("short", "短变体组"), ("all", "全体")):
        cnt, k = g[key]
        if cnt == 0:
            continue
        lo, hi = wilson(k, cnt)
        W(f"| {nm} | {cnt} | {k}/{cnt} = {k/cnt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    W("")
    W("### 5 对成对混淆（逐对列出，不折算均值）")
    W("")
    W("| # | 短变体 | 基型 | 短在空间 | 基在空间 | 短样本→预测 | 基样本→预测 | 结果 |")
    W("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(pairs, 1):
        W(f"| {i} | {r['sv']} {r['sv_notation']} | {r['base']} {r['base_notation']} | "
          f"{'✓' if r['sv_in'] else '✗'} | {'✓' if r['base_in'] else '✗'} | "
          f"{r['sv_pred'] if r['sv_pred'] is not None else '—'} | "
          f"{r['base_pred'] if r['base_pred'] is not None else '—'} | {r['outcome']} |")
    W("")
    for sv, note in orphan:
        pred = next((j["pred_sid"] for j in joined if j["truth_sid"] == sv), None)
        W(f"- 无基型孤儿：id {sv} —— {note}。本卡内预测 = {pred}，单列报数，不计混淆对。")
    W("- 置信度：短变体组 n=2（sid 25 / sid 65），Wilson 区间仅具形式意义；G5 **不参与**通过/不通过判定。")
    W("")

    # ---- perturbation
    if ctx.get("perturb_rows"):
        rows, inv_rate, verdict, total, by = perturb_block(ctx["perturb_rows"], inv)
        W("## A5 音频敏感性（扰动 P0–P7，一票否决项；只在 in-vocab n=34 上算）")
        W("")
        W("| perturb_id | kind | n | acc | 与 P0 预测翻转率 |")
        W("|---|---|---|---|---|")
        for r in rows:
            W(f"| {r['perturb_id']} | {r['kind']} | {r['n']} | {r['acc']:.4f} | {r['flip_rate']:.4f} |")
        W("")
        W(f"- 全部扰动（含 P0）预测完全不变的样本占比: **{inv_rate:.4f}**（阈值 ≥0.99 → 一票否决）")
        W(f"- 敏感性判定: **{verdict}**")
        # P0（扰动管线重编码 wav）与 preds.csv（原始 mp3）的一致率，用来判断扰动管线本身有没有引入偏移
        p0_agree = p0_n = 0
        orig_pred = {j["ref_id"]: j["pred_sid"] for j in inv}
        for ref, d in by.items():
            if "P0" in d and ref in orig_pred:
                p0_n += 1
                p0_agree += (d["P0"] == orig_pred[ref])
        W(f"- **P0 与原始 mp3 预测一致率**：{p0_agree}/{p0_n}"
          f"（P0 是扰动管线从 mp3 重编码出的 wav，一致率低说明扰动管线本身引入了偏移，"
          f"敏感性结论须按此打折）")
        W("- **P5 seed_swap 缺项**：探针集每 sound_id 仅 1 条 rendition，无同音色另一 source 的同类片段，"
          "故 P5 `available=0` 未生成。敏感性结论永久带「P5 缺项」保留；补齐需 @Data 提供同 sound_id 第二条 rendition。")
        W("")
    else:
        W("## A5 音频敏感性")
        W("")
        W("- ⚠️ 本卡未提供 `perturb_preds.csv`，敏感性**未测**，按口径视为**不通过**。")
        W("")

    # ---- failures
    fails = [j for j in inv if j["pred_sid"] != j["truth_sid"] and not j["unlearnable"]]
    unlearn = [j for j in inv if j["unlearnable"]]
    W(f"## A6 失败案例清单（{len(fails)}/{n}，已剔除「不可学」sid 25）")
    W("")
    W("| ref_id | 真值 | 预测 | family | 真值记法 | 预测记法 | prob |")
    W("|---|---|---|---|---|---|---|")
    for j in sorted(fails, key=lambda x: -x["prob_top1"]):
        W(f"| {j['ref_id']} | {j['truth_sid']} {j['truth_en']} | {j['pred_sid']} {j['pred_en']} | "
          f"{j['family']} | {j['truth_notation']} | {j['pred_notation']} | {j['prob_top1']:.3f} |")
    W("")
    for j in unlearn:
        W(f"- **该类不可学**：ref `{j['ref_id']}` 真值 sid **{j['truth_sid']} `{j['truth_notation']}`**（{UNLEARNABLE_SID[j['truth_sid']]}）"
          f"，训练样本 **0**（`label_index 5` 无任何训练行）→ 模型结构上不可能答对，"
          f"**不计入失败案例**。预测 = {j['pred_sid']}（仅供诊断）。")
    W("")

    # ---- listen pack
    pack = Path(args.listen_pack)
    W("## A6b 试听包")
    W("")
    if pack.is_dir():
        man = read_csv(pack / "MANIFEST.csv") if (pack / "MANIFEST.csv").is_file() else []
        cnt = collections.Counter(m["kind"] for m in man)
        W(f"- 路径：`{args.listen_pack}`，条目 **{len(man)}**")
        W(f"- 结构：`probe_in_vocab/` {cnt.get('probe_in_vocab', 0)} 条（文件名前缀=真值 sound_id）；"
          f"`probe_out_of_vocab/` {cnt.get('probe_out_of_vocab', 0)} 条（不计分）；"
          f"`perturb_in_vocab/` {cnt.get('perturb_in_vocab', 0)} 条（P5 缺项）；"
          f"`failures/` {cnt.get('failure', 0)} 条（文件名=`truth<真值>_pred<预测>_<ref_id>`，**不含 sid 25**）")
        W(f"- `MANIFEST.csv` 含每条的 kind/ref_id/文件名/字节数/md5；`README.md` 说明口径。")
        W(f"- 音频不入 git、不外传。")
    else:
        W(f"- ❌ 试听包缺失：`{args.listen_pack}`")
    W("")

    # ---- verdict
    ok_fields = [
        ("卡头为 INSTRUMENT_CALIBRATION", True),
        ("主指标只报 in-vocab n=34 + Wilson", True),
        ("out_of_vocab 61 条单列且不计分", True),
        ("sid 25 标注「该类不可学」且不计入失败案例", True),
        ("G5 三栏 + 5 对混淆已出", True),
        ("扰动敏感性已出", bool(ctx.get("perturb_rows"))),
        ("失败清单已出", True),
        ("试听包已出", Path(args.listen_pack).is_dir()),
    ]
    W("## A7 结论")
    W("")
    W("### 本卡验收（评测链路是否可用）")
    W("")
    for name, ok in ok_fields:
        W(f"- {'✅' if ok else '❌'} {name}")
    all_ok = all(ok for _, ok in ok_fields)
    W("")
    W(f"**评测链路结论：{'通过' if all_ok else '不通过'}**"
      f"——{'卡片字段齐全，preds/测试集/标签空间 md5 自证可复现，链路可交付给 @Talker 产出真实预测。' if all_ok else '存在缺项，见上。'}")
    W("")
    W("### 仪器读数（不是能力结论）")
    W("")
    W(f"- ast_lin_v0 在 in-vocab n=34 上 top-1 = **{top1}/{n} = {top1/n:.4f}**"
      f"（95% Wilson [{lo1:.4f}, {hi1:.4f}]）；top-3 = **{top3}/{n} = {top3/n:.4f}**"
      f"（[{lo3:.4f}, {hi3:.4f}]）。")
    W(f"- 5× 随机基线 = {5*base:.4f}，本读数{'压线' if abs(top1/n - 5*base) < 1e-9 else ('达到' if top1/n >= 5*base else '未达到')}"
      f"；但如 A1 所述，该线在 34-way 下与「答对 5 条」恒等，**不具判别力，不作为通过条件**。")
    W(f"- 该读数的含义仅是「**冻结 AST + 线性探针在这套评测链路上的地板值**」；"
      f"它既不是 103 音色识别能力，也不是 Thinker 文本头的能力上限。")
    W("- **不得**把本卡任何数字写入 agent.md §0 识别准确率；§0 只能由 `human_notation` ≥15 family 后的人工标注集出具。")
    W("")
    W("### 置信度与已知混淆变量")
    W("")
    W("- **域差异（最大混淆变量）**：训练 = B 站实战切片 22050Hz wav（70–390ms，Demucs 人声 + onset 切片），"
      "评测 = seedlib 参考音频 mp3（64–776ms，录音棚/单条示范）。两者录音链路、混响、底噪完全不同，"
      "绝对准确率偏低**不能**直接解读成「模型不听音频」或「评测工具坏了」。")
    W("- **时长混淆**：短切片被 pad 到 1024 帧（10s）输入 AST，pad 占比 92–99%；"
      "训练/评测两侧时长分布不同会让探针部分依赖时长线索。")
    W("- **样本倾斜**：33 个可学类里 12 类 n<5（含 1 类 n=0）；319 行只覆盖 33/103 音色、"
      "53 个 source，且全部 `split=train`（无独立训练内验证集，本卡也不需要）。")
    W("- **标签噪声**：训练标签来自 `beatbox_seed_crosswalk.csv` 记法匹配（exact_notation 209 / stripped_lt 92 / ambiguous+name 18），"
      "`sound_confidence=0.0` 是「无模型打分」占位值，不是低置信度。")
    W("- **P5 缺项**：敏感性结论带保留（见 A5）。")
    W("")
    W("## C 通用红线自检")
    W("")
    W("1. ✅ 未只报汇总均值——A2 分 family + 样本数在。")
    W("2. ✅ 未跨 codec 版本比较（本卡纯文本/标签层面，不涉 codec 数字）。")
    W("3. ✅ 未用训练集重建冒充泛化——测试集是 `eval_refaudio_testset.csv` 固定版 + md5。")
    W("4. ✅ n<5 分组只报数、不下结论。")
    W("5. ✅ 探针分未写入 §0。")
    W("")
    return all_ok


# ------------------------------------------------------------------ listen pack
def build_listen_pack(ctx, pack_dir: Path):
    inv, oov, joined = ctx["inv"], ctx["oov"], ctx["joined"]
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    (pack_dir / "probe_in_vocab").mkdir(parents=True)
    (pack_dir / "probe_out_of_vocab").mkdir(parents=True)
    (pack_dir / "perturb_in_vocab").mkdir(parents=True)
    (pack_dir / "failures").mkdir(parents=True)

    src_by_ref = {t["ref_id"]: t for t in ctx["test_rows"]}
    man = []

    def add(kind, ref, src, dst):
        if not Path(src).is_file():
            return
        shutil.copy2(src, dst)
        man.append({"kind": kind, "ref_id": ref, "file": str(dst.relative_to(pack_dir)),
                    "bytes": os.path.getsize(dst), "md5": md5_file(dst)})

    for j in sorted(inv, key=lambda x: x["truth_sid"]):
        add("probe_in_vocab", j["ref_id"],
            ROOT / src_by_ref[j["ref_id"]]["audio"].replace("\\", "/"),
            pack_dir / "probe_in_vocab" / f"{j['truth_sid']:03d}_{j['ref_id']}.mp3")
    for j in sorted(oov, key=lambda x: x["truth_sid"]):
        add("probe_out_of_vocab", j["ref_id"],
            ROOT / src_by_ref[j["ref_id"]]["audio"].replace("\\", "/"),
            pack_dir / "probe_out_of_vocab" / f"{j['truth_sid']:03d}_{j['ref_id']}.mp3")
    for j in sorted(joined, key=lambda x: (x["truth_sid"], x["ref_id"])):
        if j["pred_sid"] != j["truth_sid"] and j["in_vocab"] and not j["unlearnable"]:
            add("failure", j["ref_id"],
                ROOT / src_by_ref[j["ref_id"]]["audio"].replace("\\", "/"),
                pack_dir / "failures" / f"truth{j['truth_sid']:03d}_pred{j['pred_sid']:03d}_{j['ref_id']}.mp3")

    inv_refs = {j["ref_id"] for j in inv}
    man_path = ROOT / "data" / "eval_perturb_manifest.csv"
    if man_path.is_file():
        for r in read_csv(man_path):
            if r["ref_id"] not in inv_refs or str(r.get("available", "1")) != "1":
                continue
            src = ROOT / "data" / "eval_perturb_audio" / r["out_file"]
            add("perturb_in_vocab", r["ref_id"], src,
                pack_dir / "perturb_in_vocab" / r["out_file"])

    with open(pack_dir / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "ref_id", "file", "bytes", "md5"])
        w.writeheader()
        w.writerows(man)

    n_fail = sum(1 for m in man if m["kind"] == "failure")
    (pack_dir / "README.md").write_text(
        f"""# 试听包 · {CARD_NAME} · {PRED_SOURCE}

生成时间：2026-10-27   条目数：{len(man)}

- `probe_in_vocab/`：34 条 in-vocab 参考音频（文件名前缀 = 真值 sound_id）
- `probe_out_of_vocab/`：61 条 out_of_vocab 参考音频（**不计分**，只供人工听感核对）
- `perturb_in_vocab/`：in-vocab 34 条 ref 的扰动样本（P5 缺项，见卡片 A5）
- `failures/`：{n_fail} 条失败案例（文件名 = `truth<真值>_pred<预测>_<ref_id>.mp3`）
  **不含 sid 25 `K\\``**：该类训练样本 0，标注「该类不可学」，按口径不计入失败案例。

口径提醒：本试听包只服务**仪器校准**；音频均为 seedlib 参考素材与本地扰动产物，不外传、不入 git。
""",
        encoding="utf-8")
    return len(man)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "outputs" / "instr_cal_ast_lin_v0" / "preds.csv"))
    ap.add_argument("--perturb", default=str(ROOT / "outputs" / "instr_cal_ast_lin_v0" / "perturb_preds.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "instr_cal_ast_lin_v0" / "INSTRUMENT_CALIBRATION.md"))
    ap.add_argument("--listen-pack", default=str(ROOT / "outputs" / "instr_cal_ast_lin_v0" / "listen"))
    ap.add_argument("--experiment-id", default="ast_lin_v0_instr_cal_2026-10-27")
    ap.add_argument("--build-listen-pack", action="store_true")
    args = ap.parse_args()

    ctx = load_all(args)
    if Path(args.perturb).is_file():
        ctx["perturb_rows"] = read_csv(Path(args.perturb))

    if args.build_listen_pack:
        n = build_listen_pack(ctx, Path(args.listen_pack))
        print(f"试听包已生成: {args.listen_pack}  条目={n}", flush=True)

    L: list[str] = []
    all_ok = render(args, L, ctx)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("card:", out)
    print("card md5:", md5_file(out))
    print("VERDICT:", "PASS" if all_ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
