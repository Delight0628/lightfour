"""
识别侧三臂敏感性表（Eval，PI 裁决 143/155 派工，2026-10-27）。

三臂（同一 testset 音频、同一冻结 AST + 线性探针 ast_lin_v0）：
  * normal  : 原始参考音频（data/eval_refaudio_testset.csv 的 audio 路径）
  * silence : 与 normal 等长的全零静音（16 kHz，长度 = 重采样后采样点数）
  * shuffle : 25 ms 块级时间置乱（16 kHz，块 = 400 采样，固定种子 20261027+i，
              块集合含尾部残块；保留能量/谱内容与总长度，破坏时间包络与起音）
              —— 本臂定义为**音频级**置乱，与三重门③的**标签置换**（perm_null.json）不是一回事。

口径（与 eval_instrument_calibration.py 逐项一致）：
  * in-vocab = truth_sound_id ∈ data/baseline_label_space.csv 的 34 个 sound_id（unknown 行除外），n=34；
  * 计分 = sound_id top-1 命中；prob_top1 未校准，不作阈值/拒识；
  * 本卡是**仪器校准**材料，不是识别能力结论，任何数字不得写入 agent.md §0。

门槛（PI 验收）：
  * silence 臂 top-1 ≤ 5%（随机 0.97% 的 5 倍余量；n=34 下即命中 ≤1），不过即 FAIL；
  * shuffle 臂须「显著低于 normal」：配对 bootstrap 差值（normal - shuffle）95% CI 下界 > 0
    （B=10000，seed=20261027），并附 McNemar 精确检验（discordant 对）。

红线：
  * 不覆盖任何冻结产物；本脚本只写 outputs/instr_cal_ast_lin_v0/three_arm/ 与
    .cache/ast_baseline/emb_three_arm/；
  * normal 臂预测必须与冻结 preds.csv（md5 efc821bf…）逐条一致，否则 ABORT——
    这是「我的链路 = 冻结链路」的自证，不一致就不许出表。

用法：
  python scripts/eval_three_arm_sensitivity.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "instr_cal_ast_lin_v0" / "three_arm"
CACHE = ROOT / ".cache" / "ast_baseline" / "emb_three_arm"
AST_DIR = ROOT / ".cache" / "ast_baseline"
LABEL_SPACE = ROOT / "data" / "baseline_label_space.csv"
TESTSET = ROOT / "data" / "eval_refaudio_testset.csv"
PREDS = ROOT / "outputs" / "instr_cal_ast_lin_v0" / "preds.csv"
MODEL = ROOT / "outputs" / "instr_cal_ast_lin_v0" / "model_ast_lin_v0.npz"

PREDS_MD5_FROZEN = "efc821bfe15b2a936d641cd4263e9c78"
CHUNK = 400           # 25 ms @ 16 kHz
SEED = 20261027
B_BOOT = 10000
SR = 16000


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p: Path):
    import csv
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wilson(k: int, n: int, z: float = 1.959963985):
    import math
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def load_train_module():
    """按路径 import 冻结脚本 train_ast_lin_v0.py，保证 featurize 代码同一份（不是拷贝）。"""
    path = ROOT / "scripts" / "train_ast_lin_v0.py"
    spec = importlib.util.spec_from_file_location("train_ast_lin_v0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import numpy as np
    import soundfile as sf

    # ---------- 0. 锚定与 in-vocab 口径
    assert md5_file(PREDS) == PREDS_MD5_FROZEN, f"preds.md5 {md5_file(PREDS)} != 冻结值，ABORT"
    labels = read_csv(LABEL_SPACE)
    sid2li = {}
    for r in labels:
        if r["sound_id"] == "unknown":
            continue
        sid2li[int(r["sound_id"])] = int(r["label_index"])
    assert len(sid2li) == 34, f"label space sids={len(sid2li)}"

    test = read_csv(TESTSET)
    inv = [r for r in test if int(r["sound_id"]) in sid2li]
    inv.sort(key=lambda r: r["ref_id"])
    assert len(test) == 95 and len(inv) == 34, f"joined={len(test)} inv={len(inv)}"
    assert len({int(r["sound_id"]) for r in inv}) == 34

    frozen = {r["ref_id"]: r for r in read_csv(PREDS)}

    # ---------- 1. 造臂音频（落盘为可审计 wav，featurize 只吃落盘件）
    mod = load_train_module()
    out_audio = OUT / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    import soxr
    arm_paths = {"normal": {}, "silence": {}, "shuffle": {}}
    shuffle_spec = {}
    for i, r in enumerate(inv):
        ref_id = r["ref_id"]
        src = ROOT / r["audio"].replace("\\", "/")
        y, sr = sf.read(str(src), always_2d=True, dtype="float32")
        y = y.mean(axis=1)
        if sr != SR:
            y = soxr.resample(y, sr, SR, quality="HQ")
        y = y.astype("float32")
        assert y.size > 0

        sil = np.zeros_like(y)
        blocks = [y[j:j + CHUNK] for j in range(0, y.size, CHUNK)]
        rng = np.random.default_rng(SEED + i)
        perm = rng.permutation(len(blocks))
        shuf = np.concatenate([blocks[j] for j in perm]) if len(blocks) > 1 else y.copy()
        assert shuf.size == y.size
        shuffle_spec[ref_id] = {"n_blocks": len(blocks), "perm_head": [int(x) for x in perm[:8]],
                                "seed": SEED + i, "chunk_samples": CHUNK}

        for arm, arr in (("silence", sil), ("shuffle", shuf)):
            p = out_audio / f"{ref_id}__{arm}.wav"
            sf.write(str(p), arr, SR, subtype="FLOAT")
            arm_paths[arm][ref_id] = p
        arm_paths["normal"][ref_id] = src

    # ---------- 2. featurize 三臂（同一函数；normal 另做缓存等价自证）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    torch.set_num_threads(6)
    fe = ASTFeatureExtractor.from_pretrained(str(AST_DIR))
    md = ASTForAudioClassification.from_pretrained(str(AST_DIR)).eval()

    def featurize(paths_by_ref, arm):
        vecs = {}
        items = sorted(paths_by_ref.items())
        for j in range(0, len(items), 8):
            chunk = items[j:j + 8]
            waves = [mod.load_audio_16k(p) for _, p in chunk]
            batch = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                emb = md.audio_spectrogram_transformer(input_values=batch["input_values"]).pooler_output
            emb = emb.to(torch.float32).cpu().numpy()
            for (ref_id, _), vec in zip(chunk, emb):
                vec = vec.astype(np.float32)
                np.save(CACHE / f"threearm__{arm}__{ref_id}.npy", vec)
                vecs[ref_id] = vec
        return vecs

    vecs = {arm: featurize(arm_paths[arm], arm) for arm in ("normal", "silence", "shuffle")}

    # normal 臂 = 冻结链路自证：与 .cache/ast_baseline/emb/probe__*.npy 对拍
    cache_probe = ROOT / ".cache" / "ast_baseline" / "emb"
    max_diff = 0.0
    n_match = 0
    for ref_id in vecs["normal"]:
        old = np.load(cache_probe / f"probe__{ref_id}.npy")
        d = float(np.max(np.abs(old - vecs["normal"][ref_id])))
        max_diff = max(max_diff, d)
        n_match += 1 if d == 0.0 else 0
    print(f"[identity] normal feats vs frozen probe cache: exact {n_match}/{len(vecs['normal'])} max_abs_diff={max_diff:.3e}")
    assert max_diff <= 1e-5, f"特征链路与冻结缓存漂移 max_abs_diff={max_diff:.3e}，ABORT"

    # ---------- 3. predict（与 stage_predict.predict_one 同式）
    z = np.load(MODEL)
    mean, scale = z["scaler_mean"], z["scaler_scale"]
    coef, intercept, classes = z["coef"], z["intercept"], z["classes"].astype(np.int64)
    li2 = {int(r["label_index"]): r for r in labels}

    def predict_one(vec):
        x = (vec.astype(np.float64) - mean) / scale
        logits = x @ coef.T + intercept
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        order = np.argsort(-p)[:3]
        return [(int(classes[k]), float(p[k])) for k in order]

    rows = []
    pred_sid = {}
    for r in inv:
        ref_id = r["ref_id"]
        truth = int(r["sound_id"])
        for arm in ("normal", "silence", "shuffle"):
            top = predict_one(vecs[arm][ref_id])
            li0, p0 = top[0]
            sid = int(li2[li0]["sound_id"]) if li0 in li2 else -1
            pred_sid[(arm, ref_id)] = sid
            rows.append({
                "ref_id": ref_id, "arm": arm, "truth_sound_id": truth,
                "truth_notation": r.get("sound_notation", ""), "truth_family": r.get("sound_family", ""),
                "pred_sound_id": sid, "pred_notation": li2[li0]["sound_notation"] if li0 in li2 else "",
                "top3_sound_id": "|".join(str(int(li2[li]["sound_id"])) for li, _ in top),
                "prob_top1": f"{p0:.6f}", "hit": int(sid == truth),
                "same_as_normal_pred": int(sid == pred_sid.get(("normal", ref_id), sid)) if arm != "normal" else 1,
            })

    # normal 臂预测 = 冻结 preds.csv 逐条一致，否则 ABORT
    bad = [r["ref_id"] for r in rows if r["arm"] == "normal"
           and int(frozen[r["ref_id"]]["pred_sound_id"]) != r["pred_sound_id"]]
    assert not bad, f"normal 臂预测与冻结 preds.csv 不一致: {bad[:5]}"
    print("[identity] normal preds == frozen preds.csv: 34/34")

    # ---------- 4. 三臂指标 + 门槛
    def arm_stats(arm):
        rr = [r for r in rows if r["arm"] == arm]
        k = sum(r["hit"] for r in rr)
        n = len(rr)
        lo, hi = wilson(k, n)
        return {"arm": arm, "n": n, "top1_hits": k, "top1_rate": k / n,
                "wilson95": [round(lo, 4), round(hi, 4)]}

    st = {a: arm_stats(a) for a in ("normal", "silence", "shuffle")}
    st["normal"]["top3_hits"] = sum(
        int(r["truth_sound_id"]) in [int(x) for x in r["top3_sound_id"].split("|")]
        for r in rows if r["arm"] == "normal")

    flip_sil = sum(pred_sid[("silence", r["ref_id"])] != pred_sid[("normal", r["ref_id"])] for r in inv)
    flip_shuf = sum(pred_sid[("shuffle", r["ref_id"])] != pred_sid[("normal", r["ref_id"])] for r in inv)

    # silence 门
    sil_rate = st["silence"]["top1_rate"]
    gate_silence = {"rule": "silence 臂 top-1 ≤ 0.05（103-way 随机 0.97% 的 5 倍余量；n=34 下命中 ≤1）",
                    "observed": round(sil_rate, 4), "hits": st["silence"]["top1_hits"],
                    "pass": bool(sil_rate <= 0.05)}
    # shuffle 门：配对 bootstrap + McNemar
    d = np.asarray([next(r["hit"] for r in rows if r["arm"] == "normal" and r["ref_id"] == x["ref_id"])
                    - next(r["hit"] for r in rows if r["arm"] == "shuffle" and r["ref_id"] == x["ref_id"])
                    for x in inv], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, d.size, size=(B_BOOT, d.size))
    means = d[idx].mean(axis=1)
    lo_b, hi_b = np.percentile(means, [2.5, 97.5])
    p_le0 = float((means <= 0).mean())
    b = int(((d == 1).sum()))   # normal 命中 / shuffle 未中
    c = int(((d == -1).sum()))  # normal 未中 / shuffle 命中
    try:
        from scipy.stats import binomtest
        mcnemar_p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0
    except Exception:
        mcnemar_p = float("nan")
    gate_shuffle = {"rule": "配对 bootstrap（normal-shuffle）95% CI 下界 > 0 才写「shuffle 显著低于 normal」",
                    "bootstrap_B": B_BOOT, "bootstrap_seed": SEED,
                    "diff_mean": round(float(d.mean()), 4),
                    "diff_ci95": [round(float(lo_b), 4), round(float(hi_b), 4)],
                    "P(mean<=0)": round(p_le0, 4),
                    "mcnemar_discordant": {"normal_hit_shuffle_miss": b, "normal_miss_shuffle_hit": c},
                    "mcnemar_exact_p_two_sided": round(mcnemar_p, 6),
                    "pass": bool(lo_b > 0)}

    # ---- 解释性补充臂 shuffle_feed（事后定义，不进 25ms 注册臂门槛）----
    # 「shuffle」有两种读法：①时间块置乱（上面的注册臂）；②跨样本错配音频（本臂）。
    # 本臂不跑模型：样本位 i 喂供体 j 的音频，其预测 = 供体 j 的 normal 预测（链路同源）。
    # 回答的问题：预测跟音频走，还是跟样本位/模板走。判据同 shuffle：显著低于 normal 才算过。
    perm_feed = np.random.default_rng(SEED).permutation(len(inv))
    feed_pred = {inv[i]["ref_id"]: pred_sid[("normal", inv[int(perm_feed[i])]["ref_id"])]
                 for i in range(len(inv))}
    feed_hits = int(sum(feed_pred[r["ref_id"]] == int(r["sound_id"]) for r in inv))
    audio_follow = int(sum(feed_pred[inv[i]["ref_id"]] == int(inv[int(perm_feed[i])]["sound_id"])
                           for i in range(len(inv))))
    d_feed = np.asarray([
        next(r["hit"] for r in rows if r["arm"] == "normal" and r["ref_id"] == x["ref_id"])
        - int(feed_pred[x["ref_id"]] == int(x["sound_id"])) for x in inv], dtype=np.float64)
    rng_f = np.random.default_rng(SEED)
    idx_f = rng_f.integers(0, d_feed.size, size=(B_BOOT, d_feed.size))
    means_f = d_feed[idx_f].mean(axis=1)
    lo_f, hi_f = np.percentile(means_f, [2.5, 97.5])
    b_f = int((d_feed == 1).sum())
    c_f = int((d_feed == -1).sum())
    try:
        mcnemar_f = float(binomtest(b_f, b_f + c_f, 0.5).pvalue) if (b_f + c_f) else 1.0
    except Exception:
        mcnemar_f = float("nan")
    feed_lo, feed_hi = wilson(feed_hits, len(inv))
    supp_feedswap = {
        "status": "事后补充臂（解释性，不替代 25ms 注册臂；PI 未裁前不写入任何门）",
        "definition": f"跨样本错配音频：固定置换 np.random.default_rng({SEED}).permutation({len(inv)})，"
                      "样本位 i 喂供体 j 音频，计分仍按样本位 i 真值",
        "n": len(inv), "top1_hits": feed_hits, "top1_rate": feed_hits / len(inv),
        "wilson95": [round(feed_lo, 4), round(feed_hi, 4)],
        "audio_follow_hits": audio_follow,
        "diff_mean_vs_normal": round(float(d_feed.mean()), 4),
        "diff_ci95": [round(float(lo_f), 4), round(float(hi_f), 4)],
        "mcnemar_discordant": {"normal_hit_feed_miss": b_f, "normal_miss_feed_hit": c_f},
        "mcnemar_exact_p_two_sided": round(mcnemar_f, 6),
        "pass_as_shuffle": bool(lo_f > 0),
    }

    # 分 family（不得只报汇总均值）
    fam = {}
    for r in rows:
        f = r["truth_family"] or "unknown"
        fam.setdefault(f, {"n": 0, "normal_hit": 0, "silence_hit": 0, "shuffle_hit": 0})
        fam[f]["n"] += 1
        fam[f][f"{r['arm']}_hit"] += r["hit"]

    table_placeholder_blocks = {"min": min(v["n_blocks"] for v in shuffle_spec.values()),
                               "median": sorted(v["n_blocks"] for v in shuffle_spec.values())[len(shuffle_spec)//2],
                               "max": max(v["n_blocks"] for v in shuffle_spec.values())}
    fail_detail = {
        "normal_hit_but_shuffle_hit": [r["ref_id"] for r in rows if r["arm"] == "normal" and r["hit"]
                                       and pred_sid[("shuffle", r["ref_id"])] == int(r["truth_sound_id"])],
        "normal_miss_but_shuffle_hit": [r["ref_id"] for r in rows if r["arm"] == "normal" and not r["hit"]
                                        and pred_sid[("shuffle", r["ref_id"])] == int(r["truth_sound_id"])],
        "silence_pred_equals_normal_pred": [x["ref_id"] for x in inv
                                            if pred_sid[("silence", x["ref_id"])] == pred_sid[("normal", x["ref_id"])]],
        "silence_hits": [r["ref_id"] for r in rows if r["arm"] == "silence" and r["hit"]],
    }

    # ---------- 5. 落盘
    import csv as _csv
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "three_arm_preds.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    table = {
        "artifact": "three_arm_sensitivity",
        "card": "INSTRUMENT_CALIBRATION (ast_lin_v0) · 识别侧三臂敏感性（裁决 143/155）",
        "arms": {
            "normal": "原始参考音频（= 冻结 preds.csv 链路，逐条自证一致）",
            "silence": "等长全零静音（16 kHz，长度与 normal 相同）",
            "shuffle": f"25 ms 块级时间置乱（块={CHUNK} 采样 @16 kHz，seed={SEED}+i，含尾部残块；"
                       "保留谱内容与长度，破坏时间包络/起音）",
        },
        "shuffle_note": "shuffle 为音频级置乱；三重门③（perm_null.json）是标签置换，两者不同源、不可混引。",
        "protocol": {
            "in_vocab": "truth_sound_id ∈ baseline_label_space.csv 的 34 个 sound_id（unknown 行除外），n=34",
            "metric": "sound_id top-1 命中；prob_top1 未校准不作阈值",
            "feature": "ASTModel.pooler_output（冻结 AST rev f826b80d）",
            "identity_check": {"normal_feats_vs_frozen_cache": f"exact {n_match}/34（判据 max_abs_diff ≤ 1e-5）",
                               "max_abs_diff": max_diff,
                               "normal_preds_vs_frozen_preds_csv": "34/34"},
            "featurize_source": "scripts/train_ast_lin_v0.py（按路径 import，非拷贝）",
        },
        "per_arm": st,
        "flip_vs_normal": {"silence": flip_sil, "shuffle": flip_shuf, "n": len(inv)},
        "gates": {"silence": gate_silence,
                  "shuffle": dict(gate_shuffle, status="PERTURB_INVALID",
                                  note="裁决 167：扰动无效（25ms 块置乱对 60–200ms 击打接近恒等变换，有效块数 min 3/median 11/max 33），不计入门槛、不写「未过线」")},
        "shuffle_blocks_stat": {"min": min(v["n_blocks"] for v in shuffle_spec.values()),
                                "median": sorted(v["n_blocks"] for v in shuffle_spec.values())[len(shuffle_spec)//2],
                                "max": max(v["n_blocks"] for v in shuffle_spec.values())},
        "supplement_feedswap": supp_feedswap,
        "per_family": fam,
        "fail_detail": fail_detail,
        "shuffle_spec_sample": {k: shuffle_spec[k] for k in list(sorted(shuffle_spec))[:3]},
        "inputs": {
            "preds_csv_md5": md5_file(PREDS),
            "testset_md5": md5_file(TESTSET),
            "label_space_md5": md5_file(LABEL_SPACE),
            "model_npz_md5": md5_file(MODEL),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "soundfile": __import__("soundfile").__version__},
        "redlines": ["仪器校准材料，禁写入 agent.md §0 识别准确率",
                     "silence 门按 103-way 随机 0.97%×5 余量 = ≤5%；n=34 命中 ≤1",
                     "样本不足（n=34）结论带置信度限幅，禁外推到 103 类",
                     "shuffle 臂判 PERTURB_INVALID（裁决 167）：扰动无效不计入门槛，禁止用「未过线」暗示模型有问题",
                     "后续所有扰动臂必须附扰动有效性自检（有效块数分布/逐样本最大位移/与原信号相关系数或谱距离）+ 理论可检出上限（裁决 168）"],
    }
    (OUT / "three_arm_table.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# 识别侧三臂敏感性表（ast_lin_v0，INSTRUMENT_CALIBRATION 附件）\n")
    md.append("> 仪器校准材料：说明评测链路读音频，**不是识别能力结论**；任何数字不得写入 agent.md §0。\n")
    md.append(f"> 口径：in-vocab n=**34**（sound_id top-1）；`prob_top1` 未校准；"
              f"shuffle = 音频级 {CHUNK} 采样块置乱（seed {SEED}），与门③标签置换无关。\n")
    md.append("\n## 三臂主表\n")
    md.append("| 臂 | n | top-1 命中 | top-1 | 95% Wilson CI | 与 normal 预测翻转数 |")
    md.append("|---|---:|---:|---:|---|---:|")
    md.append(f"| normal | 34 | {st['normal']['top1_hits']}/34 | {st['normal']['top1_rate']:.4f} | "
              f"[{st['normal']['wilson95'][0]:.4f}, {st['normal']['wilson95'][1]:.4f}] | — |")
    md.append(f"| silence | 34 | {st['silence']['top1_hits']}/34 | {st['silence']['top1_rate']:.4f} | "
              f"[{st['silence']['wilson95'][0]:.4f}, {st['silence']['wilson95'][1]:.4f}] | {flip_sil} |")
    md.append(f"| shuffle | 34 | {st['shuffle']['top1_hits']}/34 | {st['shuffle']['top1_rate']:.4f} | "
              f"[{st['shuffle']['wilson95'][0]:.4f}, {st['shuffle']['wilson95'][1]:.4f}] | {flip_shuf} |")
    md.append(f"\n（normal top-3 参考：{st['normal']['top3_hits']}/34，非门判据。）\n")
    md.append("\n## 门槛判定\n")
    md.append(f"- **silence 门**：top-1 {st['silence']['top1_hits']}/34 = {sil_rate:.4f} "
              f"{'≤' if gate_silence['pass'] else '>'} 0.05 → **{'PASS' if gate_silence['pass'] else 'FAIL'}**"
              f"（门槛：103-way 随机 0.97% 的 5 倍余量；n=34 下命中 ≤1）")
    md.append(f"  - 门槛分辨率脚注（裁决 168）：n=34 下 1 条错 = 2.94%（门内）、2 条 = 5.88%（越门），"
              f"即该门实际等价于「错 ≤1 条通过」。")
    md.append(f"- **shuffle 臂：`PERTURB_INVALID`（扰动无效，不计入）**（裁决 167）——"
              f"配对 bootstrap（normal-shuffle，B={B_BOOT}，seed={SEED}）差值均值 "
              f"{gate_shuffle['diff_mean']:.4f}，95% CI **[{lo_b:.4f}, {hi_b:.4f}]**，P(mean≤0)={p_le0:.4f}；"
              f"McNemar 精确（discordant b={b}, c={c}）p={mcnemar_p:.6f}。9/34 反而高于 normal 5/34，"
              f"方向自相矛盾；25ms 块置乱对 60–200ms 击打接近恒等变换（有效块数 min 3 / median 11 / max 33）。"
              f"扰动无效 ⇒ **不计入门槛、不写「未过线」、不得暗示模型有问题**；"
              f"「shuffle 显著低于 normal」本臂不作判定。")
    md.append("\n## 附录：解释性补充臂 shuffle_feed（事后定义，**非验收门**）\n")
    md.append("- 标注一：**n=34 欠功效**；标注二：**CI 下界 = 0**；标注三：**非验收门**（裁决 168）。")
    md.append(f"- 读法②「shuffle」= 跨样本错配音频（固定置换 seed={SEED}）：top-1 {feed_hits}/34 = "
              f"{feed_hits/34:.4f}，Wilson [{feed_lo:.4f}, {feed_hi:.4f}]；预测跟随供体音频 {audio_follow}/34。")
    md.append(f"- 与 normal 配对差值均值 {supp_feedswap['diff_mean_vs_normal']:.4f}，95% CI "
              f"[{lo_f:.4f}, {hi_f:.4f}]；McNemar（b={b_f}, c={c_f}）p={mcnemar_f:.6f} → "
              f"{'过「显著低于 normal」' if supp_feedswap['pass_as_shuffle'] else '未过「显著低于 normal」'}。"
              f"**禁写「掉到 chance 已证明依赖音频」**——CI 下界 = 0，该结论没有统计支撑。")
    md.append(f"- 25ms 注册臂的块数分布：min={table_placeholder_blocks['min']} / "
              f"median={table_placeholder_blocks['median']} / max={table_placeholder_blocks['max']}（短击打块数少，"
              "置乱强度有限，是该臂判 PERTURB_INVALID 的依据（扰动有效性自检，裁决 167/168）。")
    md.append("\n## 分 family（n 小，只报数不外推）\n")
    md.append("| truth family | n | normal 命中 | silence 命中 | shuffle 命中 |")
    md.append("|---|---:|---:|---:|---:|")
    for f in sorted(fam):
        v = fam[f]
        md.append(f"| {f} | {v['n']} | {v['normal_hit']} | {v['silence_hit']} | {v['shuffle_hit']} |")
    md.append("\n## 失败/异常清单\n")
    for k, v in fail_detail.items():
        md.append(f"- `{k}`（{len(v)}）：{', '.join(v) if v else '—'}")
    md.append("\n## 自证\n")
    md.append(f"- normal 特征 vs 冻结 `probe__*.npy`：exact {n_match}/34，max_abs_diff={max_diff:.3e}；")
    md.append("- normal 预测 vs 冻结 `preds.csv`：**34/34 逐条一致**（不一致即 ABORT，本表不产出）；")
    md.append(f"- 输入指纹：preds.md5=`{table['inputs']['preds_csv_md5']}` / "
              f"testset.md5=`{table['inputs']['testset_md5']}` / "
              f"label_space.md5=`{table['inputs']['label_space_md5']}` / "
              f"model.md5=`{table['inputs']['model_npz_md5']}`。")
    md.append("\n## 限幅\n")
    md.append("- n=34，family 级 n 多为个位数，**只报数不外推**；")
    md.append("- 本表只证明「链路读音频、对静音/置乱有反应」，不证明 103 音色识别能力；")
    md.append("- shuffle 臂已判 `PERTURB_INVALID`（扰动无效，不计入门槛，裁决 167）；"
              "无效扰动禁止用「未过线」暗示模型有问题。\n")
    md.append("- 后续所有扰动臂（硬要求，裁决 168）：报结果必须附**扰动有效性自检**"
              "（有效块数分布 / 逐样本最大位移 / 与原信号相关系数或谱距离）+ 该扰动的**理论可检出上限**；"
              "有效性不达标只能打 `PERTURB_INVALID`。\n")
    (OUT / "THREE_ARM_TABLE.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({"silence": gate_silence, "shuffle": gate_shuffle,
                      "per_arm": {k: (v.get("top1_hits"), v.get("top1_rate")) for k, v in st.items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
