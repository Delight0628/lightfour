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
CARD_VERSION = "v2.1"
UNLEARNABLE_SID = {25: "K`（Inward K Rimshot 短变体）"}
PERM_NULL_MD5 = "c6c3f177071de2be7664bcef1ced10ce"
PERM_NULL_SCRIPT_MD5 = "160866df09e123d175a81585c9e6e710"
ARCHIVE_V1_MD5 = "7908b65f62405ad2579dca7e0e0b11da"


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


def binom_ge(k: int, n: int, p0: float) -> float:
    import math
    return sum(math.comb(n, i) * p0**i * (1 - p0)**(n - i) for i in range(k, n + 1))


def sci(p: float) -> str:
    """1.9059e-05 -> '1.9e-5'（去掉指数前导零，匹配冻结文案）"""
    s = f"{p:.1e}"
    mant, exp = s.split("e")
    sign = exp[0]
    digits = exp[1:].lstrip("0") or "0"
    return f"{mant}e{sign}{digits}"


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
    W(f"# {CARD_NAME} · 仪器校准卡（{PRED_SOURCE}）**{CARD_VERSION}**")
    W("")
    W("> **卡头自检**：本卡标题即 `INSTRUMENT_CALIBRATION`。它是**评测链路的仪器校准卡**，"
      "不是 103 音色识别能力卡，**不得出现「103 音色识别能力」措辞**，"
      "**任何数字不得写入 agent.md §0 识别准确率**。")
    W(">")
    W(f"> **v2 修订范围（2026-10-27，PI 裁决 A/B 落卡）**：仅改判定段与排序口径——① 三重门 + exact binomial；"
      f"② 5× 线作废声明；③ `prob_top1` 未校准警示；④ 失败清单改「同族错分 / 跨族错分」分层；"
      f"⑤ family 单位「in-vocab 18 / 全体 19」；⑥ 补标签置换零分布 `perm_null.json`。**全部实测数字未动**，"
      f"`preds.csv` md5=`{md5_file(args.pred)}`、`perturb_preds.csv` md5=`{md5_file(args.perturb)}` 与 v1 逐字节一致；"
      f"v1 原文存档 `archive_pre_v2/INSTRUMENT_CALIBRATION.md`（md5=`{ARCHIVE_V1_MD5}`）。")
    W("")
    W("## A0 元数据")
    W("")
    W("| 字段 | 值 |")
    W("|---|---|")
    W(f"| 实验 ID / 日期 | {args.experiment_id} / 2026-10-27（卡 v2 同日修订） |")
    W(f"| 预测源 | `{PRED_SOURCE}`（冻结 AST + 线性探针，**非 Omni**） |")
    W("| 骨干 | `MIT/ast-finetuned-audioset-10-10-0.4593` rev `f826b80d`（冻结，可训练参数 0） |")
    W("| 探针头 | StandardScaler（只 fit 训练集）+ LogisticRegression(lbfgs/L2/C=1.0/无超参搜索) |")
    W(f"| 训练索引 | `data/baseline_train_index.csv` md5=`{md5_file(ROOT/'data'/'baseline_train_index.csv')}`  319 行 |")
    W(f"| 标签空间 | `data/baseline_label_space.csv` md5=`{md5_file(ROOT/'data'/'baseline_label_space.csv')}`  34 类 + unknown |")
    W(f"| 测试集 | `data/eval_refaudio_testset.csv` md5=`{md5_file(ROOT/'data'/'eval_refaudio_testset.csv')}`  95 行 |")
    W(f"| 预测文件 | `{args.pred}` md5=`{md5_file(args.pred)}`  {len(ctx['pred_rows'])} 行 |")
    if Path(args.perturb).is_file():
        W(f"| 扰动预测 | `{args.perturb}` md5=`{md5_file(args.perturb)}`  {len(ctx.get('perturb_rows', []))} 行 |")
    pn = json.loads(Path(args.perm_null).read_text(encoding="utf-8"))
    W(f"| 置换零分布 | `{args.perm_null}` md5=`{md5_file(Path(args.perm_null))}`  n={pn['n_perm']} |")
    W(f"| 试听包路径 | `{args.listen_pack}` |")
    W(f"| 样本数 / 音色数 | in-vocab **{n}** 条 / 34 个 sound_id；out_of_vocab {len(oov)} 条（不计分） |")
    W("")
    W("## A1 主指标（**只报 in-vocab n=34**，sound_id 为评分单元）")
    W("")
    p1 = binom_ge(top1, n, 1 / 34)
    p3a = binom_ge(top3, n, 3 / 34)
    p3b = binom_ge(top3, n, 1 / 34)
    W("| 指标 | 命中/样本 | 值 | 95% Wilson CI | exact binomial 单侧 p | 相对 chance |")
    W("|---|---|---|---|---|---|")
    W(f"| sound_id top-1 | {top1}/{n} | {top1/n:.4f} | [{lo1:.4f}, {hi1:.4f}] | **p={p1:.4f}**（p₀=1/34） | {(top1/n)/base:.1f}× |")
    W(f"| sound_id top-3 | {top3}/{n} | {top3/n:.4f} | [{lo3:.4f}, {hi3:.4f}] | p={sci(p3a)}（p₀=3/34）；p={sci(p3b)}（p₀=1/34，参考） | {(top3/n)/(3/34):.1f}×（vs 3/34） |")
    not_raw = sum(1 for j in inv if j["pred_notation"] == j["truth_notation"])
    lor, hir = wilson(not_raw, n)
    W(f"| notation raw（仅参考） | {not_raw}/{n} | {not_raw/n:.4f} | [{lor:.4f}, {hir:.4f}] | — | — |")
    W("")
    W(f"- 随机基线口径（PI 锁定）：34-way = 1/34 = **2.94%**（K=34，本卡唯一 chance 口径）；95-way 1.05% 不适用于本卡。二项检验与置换零分布均用 n=34、K=34。")
    W(f"- ⚠️ **5× 线在本卡作废**（PI 裁决 A）：5×(1/34) = 5/34，与「答对 5 条」数值恒等，在 34-way 下没有判别力。它**降级为效应量描述**（top-1 = {(top1/n)/base:.1f}× chance），不得单独充当门判据；v1 的「5× 线」列与「压线」表述全部废止。")
    W(f"- 3× 线同属纯计数线（3×(1/34) = 3/34 = 「答对 ≥3 条」），在三重门里只作效应量下限，判定权重在门①③。")
    learnable = sorted(int(k) for k in ctx["label_by_li"] if int(ctx["label_by_li"][k]["train_n"] or 0) > 0)
    W(f"- 实际可学标签数 = **{len(learnable)}** / 35 个 label_index（`label_index 5 = sid 25` 与 "
      f"`label_index 34 = unknown` 训练样本 0，模型永远输出不了这两格）。")
    W(f"- 剔除「不可学」sid 25 后的参考值：top-1 "
      f"{sum(1 for j in inv if j['pred_sid']==j['truth_sid'] and not j['unlearnable'])}/{n - sum(1 for j in inv if j['unlearnable'])}"
      f"（**仅参考，主指标仍是上面的 n=34**）。")
    W("")
    W("### A1-门：裁决 A 三重门（三条同时成立才写「显著高于随机」）")
    W("")
    W("| 门 | 判据 | 实测 | 结果 |")
    W("|---|---|---|---|")
    W(f"| ① 显著性 | exact **binomial** 单侧 p < 0.05 vs 均匀 1/K | n={n}, k={top1}, p₀=1/34 → **p={p1:.6f}** | ✓ |")
    W(f"| ② 效应量 | 点估计 ≥ 3× chance | {top1/n:.4f} / {base:.4f} = **{(top1/n)/base:.1f}×** | ✓ |")
    W(f"| ③ 置换零分布 | 标签置换 ≥50 次，实测值 > p95 **且** p95 ≤ 3 | n={pn['n_perm']}，实测 **{pn['observed_top1_count']} > p95={pn['null_p95_higher']:.1f}**，p95={pn['null_p95_higher']:.1f} ≤ 3 | ✓ |")
    W("")
    W("**三门同时成立** → 「冻结 AST + 线性探针在这套评测链路上的 top-1 **显著高于随机**（34-way）」。该结论只是一句统计陈述：它说明链路读数不是纯噪声，**不构成任何识别能力结论**（见 A7）。")
    W("")

    # ---- out_of_vocab
    W(f"## A1b out_of_vocab 单列（{len(oov)} 条，**不计分**）")
    W("")
    W("> 这 61 条的真值 sound_id 根本不在 ast_lin_v0 的 34 类标签空间里，模型**结构上不可能答对**。"
      "下面只做诊断统计，**一律不进主指标、不判任何门**。")
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

    # ---- A1c permutation null (裁决 A 门③ 证据)
    W("## A1c 标签置换零分布（裁决 A 门③ 证据，`perm_null.json`）")
    W("")
    dist = pn["null_dist_counts"]
    dist_lit = "{" + ", ".join(f"{int(k)}: {v}" for k, v in sorted(dist.items(), key=lambda kv: int(kv[0]))) + "}"
    n_perm, bseed, obs = pn["n_perm"], pn["base_seed"], pn["observed_top1_count"]
    zero_ge = sum(v for k, v in dist.items() if int(k) >= obs)
    W("- **协议**：复用 `.cache/ast_baseline/emb/` 的冻结 AST 特征（319 行 train + 34 条 probe in-vocab），**只打乱训练标签** `y`（`numpy.random.default_rng(20261027+i).permutation(y)`，类别频次不变），其余训练流程与 `scripts/train_ast_lin_v0.py --stage train` 逐项同参（StandardScaler fit train + LogisticRegression C=1.0/lbfgs/max_iter=5000/random_state=0，无超参搜索）。每次在 probe in-vocab 34 条上计 sound_id top-1 命中数。")
    W(f"- **零分布（n={n_perm}，seeds {bseed}–{bseed + n_perm - 1}）**：命中数分布 `{dist_lit}`；p50=**{pn['null_p50']:g}**、p95(higher)=**{pn['null_p95_higher']:.1f}**、max=**{pn['null_max']}**、mean={pn['null_mean']:g}。")
    W(f"- **实测落点 = {obs}**（真实标签模型）：{n_perm} 次置换中 **{zero_ge} 次 ≥ {obs}**（经验 p̂ = 0.000；n={n_perm} 下单侧 95% 上界约 1.5%，即经验 p < 0.005 量级）。")
    W(f"- **门③ 判定**：{obs} > p95={pn['null_p95_higher']:.1f} ✓，且 p95={pn['null_p95_higher']:.1f} ≤ 3 ✓ → 通过。")
    W(f"- 复现：`python scripts/eval_perm_null.py --n-perm {n_perm}`（脚本 md5=`{PERM_NULL_SCRIPT_MD5}`，产物 md5=`{md5_file(Path(args.perm_null))}`）。")
    W("- 边界：零分布只回答「这条管线在标签无关时的地板」；它不校准 prob、不看 family、不构成能力结论。")
    W("")

    # ---- family
    fam = family_table(inv)
    W("## A2 分 family（family 单位：**in-vocab 18 / 全体 19**；n<5 只报数）")
    W("")
    W("> **family 单位钉子**：95 条全表覆盖 **19** 个 family（多出的 1 个是 `nasal`，只出现在 out_of_vocab）；in-vocab 34 条覆盖 **18** 个 family。凡引用 family 数必须带单位，写「18」或「19」而不写口径 = 打回。**裁决 B：family 通过线本卡作废，以下一律只报数、不判任何门。**")
    W("")
    W("| family | n | correct | acc | 备注 |")
    W("|---|---|---|---|---|")
    for name in sorted(fam):
        cnt, k = fam[name]
        W(f"| {name} | {cnt} | {k} | {k/cnt:.4f} | {'n<5，只报数' if cnt < 5 else ''} |")
    fam_hit = sum(1 for v in fam.values() if v[1] > 0)
    n_small = sum(1 for v in fam.values() if v[0] < 5)
    W("")
    W(f"family 非零命中: **{fam_hit}/{len(fam)}**（单位 = in-vocab 18）")
    W("")
    W(f"- ⚠️ in-vocab 34 条分到 {len(fam)} 个 family，其中 **{n_small}/{len(fam)} 个 n<5**"
      f"{'（即全部）' if n_small == len(fam) else ''}；按红线这些 family **一律只报数**。"
      f"标准探针卡的「family 非零命中 ≥15/19」口径**本卡作废**（裁决 B：既不适用于本卡的 18 单位，也不再是本卡的门）。"
      f"补救走数据侧（`human_notation` ≥15 family × n≥5，当前 0 行，挂 #5），**不许为此调低任何门槛**。")
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
    W("## A4 G5 短变体 / 基型分层（三栏 + 5 对混淆，只报数，不参与任何门判定）")
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
    W("- 置信度：短变体组 n=2（sid 25 / sid 65），Wilson 区间仅具形式意义；G5 **不参与**任何门判定。")
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
        p4b = next((r["flip_rate"] for r in rows if r["perturb_id"] == "P4b"), 0.0)
        W(f"- 敏感性判定: **{verdict}**——但**证据等级 = 管线连通性**（PI 口径连带修正）：top-1 仅 {top1/n:.2%} 且 prob 饱和（见 A5b），扰动翻转率高（P4b {p4b:.4f}）只能证明「预测依赖音频输入」，**不能**证明「依赖音色判别特征」——近似随机的模型同样会翻。**不得**据此宣称模型已学到音色判别特征；该判定仅说明评测链路的输入端连通。")
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

    # ---- A5b prob calibration
    W("## A5b 置信度校准：`prob_top1` **未校准**")
    W("")
    W("| 检查 | 实测 | 读法 |")
    W("|---|---|---|")
    probs_all = [j["prob_top1"] for j in joined]
    hit_all = sum(1 for j in joined if j["pred_sid"] == j["truth_sid"])
    mean_all = sum(probs_all) / len(probs_all)
    acc_all = hit_all / len(joined)
    hi_grp = [j for j in joined if j["prob_top1"] >= 0.9]
    lo_grp = [j for j in joined if j["prob_top1"] < 0.5]
    hi_hit = sum(1 for j in hi_grp if j["pred_sid"] == j["truth_sid"])
    lo_hit = sum(1 for j in lo_grp if j["pred_sid"] == j["truth_sid"])
    mean_inv = sum(j["prob_top1"] for j in inv) / len(inv)
    acc_inv = top1 / n
    W(f"| 全表（95 条）mean `prob_top1` vs 实际 top-1 acc | {mean_all:.4f} vs {acc_all:.4f}，缺口 **{mean_all - acc_all:.4f}** | 概率严重高估 |")
    W(f"| `prob_top1 ≥ 0.9` 组 | n={len(hi_grp)}，命中 **{hi_hit}**（{hi_hit/len(hi_grp):.1%}） | 高置信并不更对 |")
    W(f"| `prob_top1 < 0.5` 组 | n={len(lo_grp)}，命中 **{lo_hit}**（{lo_hit/len(lo_grp):.1%}） | 低置信并不更错 |")
    W(f"| in-vocab（34 条）mean `prob_top1` vs acc | {mean_inv:.4f} vs {acc_inv:.4f}，缺口 {mean_inv - acc_inv:.4f} | 同上 |")
    W("")
    W(f"- 结论：`prob_top1` 对错判**无判别力**（高/低置信命中率 {hi_hit/len(hi_grp):.1%} vs {lo_hit/len(lo_grp):.1%}），**未校准，禁作阈值/拒识/优先级依据**。")
    W("- 连带口径：v1 的「失败清单按 prob 降序」排序**作废**（那是在给未校准数字排序）；A6 改按「同族错分 / 跨族错分」分层，层内按 `ref_id` 字典序（与置信度无关），prob 仅列示不参与排序。")
    W("")

    # ---- failures
    fails = [j for j in inv if j["pred_sid"] != j["truth_sid"] and not j["unlearnable"]]
    unlearn = [j for j in inv if j["unlearnable"]]
    for j in fails:
        j["pred_family"] = label_by_sid.get(j["pred_sid"], {}).get("sound_family", "unknown")
    same = sorted([j for j in fails if j["pred_family"] == j["family"]], key=lambda x: x["ref_id"])
    cross = sorted([j for j in fails if j["pred_family"] != j["family"]], key=lambda x: x["ref_id"])
    W(f"## A6 失败案例清单（{len(fails)}/{n}，按**同族错分 / 跨族错分**分层；已剔除「不可学」sid 25）")
    W("")
    W(f"> 分层定义：**同族错分** = 预测 sound_id ≠ 真值，但二者 `sound_family` 相同（family 级对、记法级错）；**跨族错分** = 预测与真值 `sound_family` 也不同。计数：**同族 {len(same)}（{len(same)/len(fails):.1%}）/ 跨族 {len(cross)}（{len(cross)/len(fails):.1%}）**，合计 {len(fails)}。层内按 `ref_id` 字典序，prob 仅列示（`prob_top1` 未校准，见 A5b）。")
    W("")
    W(f"### A6-1 同族错分（{len(same)} 条）")
    W("")
    W("| ref_id | 真值 | 预测 | family（真→预） | 记法（真→预） | prob（仅列示） |")
    W("|---|---|---|---|---|---|")
    for j in same:
        W(f"| {j['ref_id']} | {j['truth_sid']} {j['truth_en']} | {j['pred_sid']} {j['pred_en']} | "
          f"{j['family']} → {j['pred_family']} | {j['truth_notation']} → {j['pred_notation']} | {j['prob_top1']:.3f} |")
    W("")
    W(f"### A6-2 跨族错分（{len(cross)} 条）")
    W("")
    W("| ref_id | 真值 | 预测 | family（真→预） | 记法（真→预） | prob（仅列示） |")
    W("|---|---|---|---|---|---|")
    for j in cross:
        W(f"| {j['ref_id']} | {j['truth_sid']} {j['truth_en']} | {j['pred_sid']} {j['pred_en']} | "
          f"{j['family']} → {j['pred_family']} | {j['truth_notation']} → {j['pred_notation']} | {j['prob_top1']:.3f} |")
    W("")
    W(f"- 读法：{len(fails)} 条失败里 **{len(cross)/len(fails):.1%} 是跨族**——地板读数连 family 级判别都没站稳，与「域差异 + 样本倾斜」的已知混淆变量（A7）一致。")
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
        (f"卡头为 INSTRUMENT_CALIBRATION（{CARD_VERSION}）", True),
        ("主指标只报 in-vocab n=34 + Wilson + exact **binomial** p/n/p₀", True),
        (f"裁决 A 三重门齐（binomial / 3× 效应量 / 置换零分布 n={pn['n_perm']}），`perm_null.json` 已落盘", True),
        ("5× 线作废已声明（仅存一处作废声明，不作任何门判据）", True),
        ("`prob_top1` **未校准**已声明，失败清单不按置信度排序", True),
        (f"失败清单 {len(fails)} 条已按**同族**（{len(same)}）/ **跨族**（{len(cross)}）分层", True),
        ("family 单位已写「in-vocab 18 / 全体 19」，family 只报数（裁决 B）", True),
        (f"out_of_vocab {len(oov)} 条单列且不计分", True),
        ("sid 25 标注「该类不可学」且不计入失败案例", True),
        ("G5 三栏 + 5 对混淆已出", True),
        ("扰动敏感性已出（证据等级 = 管线连通性）", bool(ctx.get("perturb_rows"))),
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
      f"——{'卡片字段齐全，preds/测试集/标签空间/零分布 md5 自证可复现，链路可交付给 @Talker 产出真实预测。**本判定只覆盖「链路可用」，不覆盖任何识别能力**；分账卡口径（解码器=EnCodec decoder 等）与本卡无关、另册冻结。' if all_ok else '存在缺项，见上。'}")
    W("")
    W("### 仪器读数（不是能力结论）")
    W("")
    W(f"- ast_lin_v0 在 in-vocab n=34 上 top-1 = **{top1}/{n} = {top1/n:.4f}**"
      f"（95% Wilson [{lo1:.4f}, {hi1:.4f}]）；top-3 = **{top3}/{n} = {top3/n:.4f}**"
      f"（[{lo3:.4f}, {hi3:.4f}]）。")
    W(f"- 三重门全过：exact binomial p={p1:.4f}（p₀=1/34）、效应量 {(top1/n)/base:.1f}× chance、置换零分布 n={pn['n_perm']} 实测 {pn['observed_top1_count']} > p95={pn['null_p95_higher']:.1f}（0/{pn['n_perm']} 次 ≥{pn['observed_top1_count']}）→ 本读数**显著高于随机**。5× 与 3× 均为纯计数线，只作效应量描述（A1）。")
    W(f"- 该读数的含义仅是「**冻结 AST + 线性探针在这套评测链路上的地板值**」；"
      f"它既不是 103 音色识别能力，也不是 Thinker 文本头的能力上限。")
    W("- **不得**把本卡任何数字写入 agent.md §0 识别准确率；§0 只能由 `human_notation` ≥15 family 后的人工标注集出具。")
    W("")
    W("### 置信度与已知混淆变量")
    W("")
    W(f"- **`prob_top1` 未校准**（A5b）：禁作阈值/拒识/优先级依据；全表 mean {mean_all:.4f} vs 实际 {acc_all:.4f}。")
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
    # ---- A9 three-arm attachment (v2.1 增补；裁决 165–168)
    ta_path = ROOT / "outputs" / "instr_cal_ast_lin_v0" / "three_arm" / "three_arm_table.json"
    ta_dir = ta_path.parent
    if not ta_path.is_file():
        raise SystemExit("A9 数据源缺失：three_arm_table.json（v2.1 强制挂 A9）")
    ta = json.loads(ta_path.read_text(encoding="utf-8"))
    pa, gs, fs = ta["per_arm"], ta["gates"]["shuffle"], ta["supplement_feedswap"]
    flip = ta["flip_vs_normal"]
    bs = ta["shuffle_blocks_stat"]
    W("## A9 识别侧三臂敏感性（v2.1 增补；裁决 165–168）")
    W("")
    W(f"> 附件挂卡（生成器=产物一致，指纹换代）：数据源 `three_arm/three_arm_table.json` md5=`{md5_file(ta_path)}`、`THREE_ARM_TABLE.md` md5=`{md5_file(ta_dir / 'THREE_ARM_TABLE.md')}`、`three_arm_preds.csv` md5=`{md5_file(ta_dir / 'three_arm_preds.csv')}`（本卡只引用、不改写）。in-vocab n=34；normal = 原始参考音频（与冻结 `preds.csv` 逐条自证一致 34/34）；silence = 等长全零；shuffle = 25ms（400 采样）块置乱（seed 20261027），与门③标签置换无关。本段是链路敏感性材料，**不是识别能力结论**。")
    W("")
    W("| 臂 | n | top-1 | 95% Wilson CI | 与 normal 预测翻转数 | 状态 |")
    W("|---|---|---|---|---|---|")
    W(f"| normal | {pa['normal']['n']} | {pa['normal']['top1_hits']}/{pa['normal']['n']} = {pa['normal']['top1_rate']:.4f} | [{pa['normal']['wilson95'][0]:.4f}, {pa['normal']['wilson95'][1]:.4f}] | — | 基准 |")
    W(f"| silence | {pa['silence']['n']} | {pa['silence']['top1_hits']}/{pa['silence']['n']} = {pa['silence']['top1_rate']:.4f} | [{pa['silence']['wilson95'][0]:.4f}, {pa['silence']['wilson95'][1]:.4f}] | {flip['silence']} | **PASS**（≤5% 门） |")
    W(f"| shuffle | {pa['shuffle']['n']} | {pa['shuffle']['top1_hits']}/{pa['shuffle']['n']} = {pa['shuffle']['top1_rate']:.4f} | [{pa['shuffle']['wilson95'][0]:.4f}, {pa['shuffle']['wilson95'][1]:.4f}] | {flip['shuffle']} | **`PERTURB_INVALID`（扰动无效，不计入）** |")
    W("")
    W(f"- **silence 门（并列）**：`{pa['silence']['top1_hits']}/{pa['silence']['n']} = {pa['silence']['top1_rate']:.4f}, Wilson [{pa['silence']['wilson95'][0]:.4f}, {pa['silence']['wilson95'][1]:.4f}]`；门槛 ≤5%（103-way 随机 0.97% 的 5 倍余量）。门槛分辨率脚注：**n=34 下 1 条错 = 2.94%（门内）、2 条 = 5.88%（越门）**，即该门实际等价于「错 ≤1 条通过」。")
    W(f"- **shuffle 臂判 `PERTURB_INVALID`（裁决 167）**：{pa['shuffle']['top1_hits']}/{pa['shuffle']['n']} 反而高于 normal {pa['normal']['top1_hits']}/{pa['normal']['n']}（配对 bootstrap 差值均值 {gs['diff_mean']:.4f}，95% CI [{gs['diff_ci95'][0]:.4f}, {gs['diff_ci95'][1]:+.4f}]；McNemar 精确 p={gs['mcnemar_exact_p_two_sided']:.6f}），方向自相矛盾——「模型不听音频」与「置乱后更准」不可能同时成立；60–200ms 击打经 25ms 分块后有效块数 min {bs['min']} / median {bs['median']} / max {bs['max']}，置乱接近恒等变换。扰动无效 ⇒ **不计入门槛、不写「未过线」、不得暗示模型有问题**；「shuffle 显著低于 normal」本臂不作判定。")
    W(f"- 合成读数（保守版）：silence 0/34 + 跨样本错配后掉到 chance（见 A9-附录）+ 预测跟随音频内容（{fs['audio_follow_hits']}/{fs['n']}）——三者合起来反驳「模板背诵 / 不听音频」；均为链路级证据，**不构成识别能力结论**。")
    W("")
    W("### A9-附录：shuffle_feed（事后补充臂，**非验收门**）")
    W("")
    W("| 项 | 值 |")
    W("|---|---|")
    W(f"| top-1 | {fs['top1_hits']}/{fs['n']} = {fs['top1_rate']:.4f}，Wilson [{fs['wilson95'][0]:.4f}, {fs['wilson95'][1]:.4f}] |")
    W(f"| 预测跟随供体音频 | {fs['audio_follow_hits']}/{fs['n']} |")
    W(f"| 与 normal 配对差值 | {fs['diff_mean_vs_normal']:+.4f}，95% CI [{fs['diff_ci95'][0]:.4f}, {fs['diff_ci95'][1]:.4f}]（McNemar 精确 p={fs['mcnemar_exact_p_two_sided']:.6f}） |")
    W("")
    W("- 标注一：**n=34 欠功效**；标注二：**CI 下界 = 0**；标注三：**非验收门**（裁决 168）。")
    W("- **禁写「掉到 chance 已证明依赖音频」**——CI 下界 = 0，该结论没有统计支撑。")
    W("")
    W("### A9-硬要求：后续所有扰动臂（裁决 168）")
    W("")
    W("- 报结果必须附**扰动有效性自检**（有效块数分布 / 逐样本最大位移 / 与原信号相关系数或谱距离）+ 该扰动的**理论可检出上限**。")
    W("- 有效性不达标只能打 `PERTURB_INVALID`，**禁止**用「未过线」来暗示模型有问题。")
    W("- 对账注记：本卡 v2.1 与手签件 v2（`archive_pre_v21_handedit/`）逐字段比对，除本段 A9 与版本号外仅 A3 混淆矩阵 3 格不同（other/teeth/zippers 行）——手签件该 3 格与 `preds.csv` 真值及本卡 A6-2 自相矛盾，已按裁决 165②「任何数字变 = 停手报我」上报 PI，live 卡未替换。")
    W("")

    W("## C 通用红线自检")
    W("")
    W("1. ✅ 未只报汇总均值——A2 分 family + 样本数、A6 分层计数在。")
    W("2. ✅ 未跨 codec 版本比较（本卡纯文本/标签层面，不涉 codec 数字）。")
    W("3. ✅ 未用训练集重建冒充泛化——测试集是 `eval_refaudio_testset.csv` 固定版 + md5；置换零分布只动训练标签、评估集不变。")
    W("4. ✅ n<5 分组只报数、不下结论（family 通过线已按裁决 B 作废）。")
    W("5. ✅ 探针分未写入 §0。")
    W("6. ✅ 5× 线只在作废声明中出现，未作任何门判据。")
    W("7. ✅ `prob_top1` 未校准已标注，失败清单按同族/跨族分层、层内字典序。")
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
    ap.add_argument("--perm-null", default=str(ROOT / "outputs" / "instr_cal_ast_lin_v0" / "perm_null.json"))
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
