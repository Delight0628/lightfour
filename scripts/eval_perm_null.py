#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标签置换零分布（ast_lin_v0 仪器校准卡 v2 · 裁决 A 第三门配套）。

协议（PI 裁决 A 三重门之③，Eval 执行 2026-10-27）：
  * 复用 .cache/ast_baseline/emb/ 的 AST 冻结特征（train__<row_id>.npy 319 条 +
    probe__<ref_id>.npy），**只打乱训练标签 y**，其余训练流程逐项与
    scripts/train_ast_lin_v0.py --stage train 相同：
    StandardScaler(fit on train X) + LogisticRegression(C=1.0, solver="lbfgs",
    max_iter=5000, random_state=0, 无 class_weight、无超参搜索)。
  * 每次置换后在 probe in-vocab 34 条上算 sound_id top-1 命中数（口径与
    eval_instrument_calibration.py 一致：truth_sid 在 baseline_label_space.csv
    的 34 个 sound_id 内即 in-vocab；sid 25 保留但结构上不可能命中）。
  * 判定线（PI）：实测 5 > p95 **且** p95 <= 3 才可写「显著高于随机」；
    否则结论降级为「不显著」，不许保留原表述。
  * 置换只动标签、不动特征/评估集；零分布是"这条管线在标签无关时的地板"。

产物（默认）：outputs/instr_cal_ast_lin_v0/perm_null.json
红线：本脚本只产出零分布证据，不改 preds.csv / perturb_preds.csv，不进 §0。

用法:
  python scripts/eval_perm_null.py --n-perm 50
  python scripts/eval_perm_null.py --n-perm 50 --jobs 4     # 多进程并行
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / ".cache" / "ast_baseline" / "emb"
DEFAULT_OUT = ROOT / "outputs" / "instr_cal_ast_lin_v0"
BASE_SEED = 20261027


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k)，X~Bin(n,p)。精确求和（n 小，不依赖 scipy）。"""
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def load_xy(cache: Path):
    """train X(319,768)/y、probe X/真值 sid、in-vocab mask，顺序固定。"""
    import numpy as np

    train_rows = read_csv(ROOT / "data" / "baseline_train_index.csv")
    Xt, yt, tkeys = [], [], []
    for r in sorted(train_rows, key=lambda x: x["row_id"]):
        p = cache / f"train__{r['row_id']}.npy"
        if not p.is_file():
            raise SystemExit(f"缺训练特征 {p}；先跑 train_ast_lin_v0.py --stage featurize --set train")
        vec = np.load(p)
        if vec.shape != (768,):
            raise SystemExit(f"训练特征形状异常 {p}: {vec.shape}")
        Xt.append(vec)
        yt.append(int(r["label_index"]))
        tkeys.append(f"train__{r['row_id']}")

    label_rows = read_csv(ROOT / "data" / "baseline_label_space.csv")
    label_by_sid = {int(r["sound_id"]): r for r in label_rows if r["sound_id"] != "unknown"}

    probe_rows = read_csv(ROOT / "data" / "eval_refaudio_testset.csv")
    Xp, psid, pkeys = [], [], []
    for r in sorted(probe_rows, key=lambda x: x["ref_id"]):
        p = cache / f"probe__{r['ref_id']}.npy"
        if not p.is_file():
            raise SystemExit(f"缺 probe 特征 {p}；先跑 train_ast_lin_v0.py --stage featurize --set probe")
        vec = np.load(p)
        if vec.shape != (768,):
            raise SystemExit(f"probe 特征形状异常 {p}: {vec.shape}")
        Xp.append(vec)
        psid.append(int(r["sound_id"]))
        pkeys.append(r["ref_id"])

    in_vocab = [sid in label_by_sid for sid in psid]
    return (np.asarray(Xt, dtype=np.float32), np.asarray(yt, dtype=np.int64), tkeys,
            np.asarray(Xp, dtype=np.float32), np.asarray(psid, dtype=np.int64), pkeys,
            np.asarray(in_vocab, dtype=bool), label_by_sid)


LI2SID = None


def li2sid_map():
    """label_index → sound_id（与 train_ast_lin_v0.stage_predict 同源；unknown 行无 sound_id）。"""
    global LI2SID
    if LI2SID is None:
        rows = read_csv(ROOT / "data" / "baseline_label_space.csv")
        LI2SID = {int(r["label_index"]): int(r["sound_id"])
                  for r in rows if r["sound_id"] != "unknown"}
    return LI2SID


def train_and_score(Xt, y, Xp, psid, inv_mask):
    """一次训练 + in-vocab top-1 命中数（协议 = train_ast_lin_v0.stage_train）。"""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xt)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=0)
    clf.fit(Xs, y)

    mean, scale = scaler.mean_, scaler.scale_
    coef, intercept, classes = clf.coef_, clf.intercept_, clf.classes_.astype(np.int64)
    x = (Xp.astype(np.float64) - mean) / scale
    logits = x @ coef.T + intercept
    pred_li = classes[np.argmax(logits, axis=1)]

    m = li2sid_map()
    pred_sid = np.asarray([m[int(li)] for li in pred_li], dtype=np.int64)

    hit = (pred_sid == psid) & inv_mask
    return int(hit.sum()), pred_sid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--observed", type=int, default=5,
                    help="实测 top-1 命中数（ast_lin_v0 真实标签模型 = 5/34），只作落点记录")
    args = ap.parse_args()

    import numpy as np

    cache, out_dir = Path(args.cache_dir), Path(args.out_dir)
    Xt, yt, tkeys, Xp, psid, pkeys, inv_mask, _ = load_xy(cache)
    n_inv = int(inv_mask.sum())
    print(f"train X={Xt.shape} y_classes={len(set(yt.tolist()))}  probe in-vocab n={n_inv}", flush=True)

    t0 = time.time()
    counts, top1_rate = [], []
    for i in range(args.n_perm):
        rng = np.random.default_rng(args.base_seed + i)
        y_perm = rng.permutation(yt)
        c, _ = train_and_score(Xt, y_perm, Xp, psid, inv_mask)
        counts.append(int(c))
        top1_rate.append(float(c / n_inv))
        print(f"[perm {i + 1:3d}/{args.n_perm}] seed={args.base_seed + i} top1={c}/{n_inv} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    arr = np.asarray(counts, dtype=np.int64)
    p95_linear = float(np.percentile(arr, 95))
    p95_higher = float(np.percentile(arr, 95, method="higher"))
    p50 = float(np.percentile(arr, 50))
    emp_p = float((arr >= args.observed).mean())

    obs = int(args.observed)
    gate_obs_gt_p95 = bool(obs > p95_higher)
    gate_p95_le3 = bool(p95_higher <= 3)
    gate_pass = bool(gate_obs_gt_p95 and gate_p95_le3)

    n = n_inv
    p0 = 1.0 / 34.0
    binom_p = binom_sf(obs, n, p0)
    binom_p_top3 = binom_sf(12, n, 3.0 / 34.0)

    dist = {str(k): int((arr == k).sum()) for k in sorted(set(arr.tolist()))}
    verdict = "显著高于随机" if gate_pass else "不显著（结论降级，不得保留原表述）"

    out = {
        "artifact": "perm_null",
        "card": "INSTRUMENT_CALIBRATION (ast_lin_v0) v2 · 裁决 A 三重门之③",
        "protocol": {
            "feature_cache": str(cache.relative_to(ROOT)),
            "feature_md5_sample": md5_file(cache / f"{tkeys[0]}.npy"),
            "n_train_rows": int(Xt.shape[0]),
            "n_train_classes": int(len(set(yt.tolist()))),
            "probe_in_vocab_n": n,
            "shuffle": "numpy.random.default_rng(base_seed+i).permutation(y)（只动训练标签，频次不变）",
            "trainer": "StandardScaler(fit train) + LogisticRegression(C=1.0, solver=lbfgs, "
                       "max_iter=5000, random_state=0, class_weight=None)",
            "trainer_note": "与 scripts/train_ast_lin_v0.py --stage train 逐项同参；无超参搜索",
            "metric": "probe in-vocab 34 条 sound_id top-1 命中数（sid 25 保留、结构上不可能命中）",
            "in_vocab_rule": "truth_sid ∈ baseline_label_space.csv 的 34 个 sound_id（同 eval_instrument_calibration.py）",
        },
        "n_perm": int(args.n_perm),
        "base_seed": int(args.base_seed),
        "seeds": [int(args.base_seed + i) for i in range(args.n_perm)],
        "null_top1_counts": [int(c) for c in counts],
        "null_top1_rate": top1_rate,
        "null_dist_counts": dist,
        "null_p50": p50,
        "null_p95_linear": p95_linear,
        "null_p95_higher": p95_higher,
        "null_max": int(arr.max()),
        "null_mean": float(arr.mean()),
        "observed_top1_count": obs,
        "observed_top1_rate": obs / n,
        "empirical_p_ge_observed": emp_p,
        "binomial": {
            "k": obs, "n": n, "p0": p0,
            "p_one_sided_ge_k": binom_p,
            "note": "exact binomial 单侧 P(X>=k)，p0=1/34（34-way）",
            "top3_reference": {"k": 12, "n": n, "p0": 3.0 / 34.0,
                               "p_one_sided_ge_k": binom_p_top3,
                               "note": "top-3 参考，p0=3/34；非第三门判据"},
        },
        "gate_A3": {
            "rule": "实测 > p95 且 p95 <= 3 才可写「显著高于随机」",
            "observed_gt_p95": gate_obs_gt_p95,
            "p95_le_3": gate_p95_le3,
            "pass": gate_pass,
            "p95_used": "null_p95_higher（保守口径）",
        },
        "verdict": verdict,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "perm_null.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", dst, flush=True)
    print(f"null counts={counts}  p95(higher)={p95_higher}  observed={obs}  verdict={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
