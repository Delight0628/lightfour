"""
Critic 独立复算脚本（不复用 mel_layers_reissue.py 的函数，避免自证）：
  (A) mel 列三层 L1/L2/L3 —— 按 82/83 定义，从 per_hit.csv 独立实现 estimator + cluster bootstrap CI
  (B) 58.1 / 100.6 / 170.0 逐数钉「池 + 聚合 + 列 + n」
  (C) 试听 12 击源包：两臂 RMS dB 差 + raw mel + 电平剥离(gain-matched) mel

口径依据：
  82/83：列 = mel*_vs_true；逐条退化率 = mel13X_vs_true / mel75_vs_true - 1
  L1 = 逐条 min(min(13c,13p) 退化率) 聚合 median
  L2 = 组级 13c 臂退化率 聚合 median
  L3 = 组级 13p 臂退化率 聚合 median
  CI = 按 source 聚类 bootstrap，n_boot=2000，seed=20261027（与冻结 estimator 同源）
  mel 实现唯一源 = scripts/mel_metric.py（240f6f18，冻结不改）
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
import mel_metric  # noqa: E402  冻结 log-mel 实现，240f6f18

PER_HIT = Path("outputs/codec_rate_ab/per_hit.csv")
EXPECT_MD5 = "235158c7877f5620e5443790c19f30e8"
EXPECT_BYTES = 1284083
SEED = 20261027
N_BOOT = 2000

POOLS = {
    "click_60_200ms": lambda r: r["is_click_60_200"] == "1",
    "hihats": lambda r: r["sound_family"] == "hihats",
    "cymbal": lambda r: r["sound_family"] == "cymbal",
}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def cluster_ci(rows, values):
    by_src = defaultdict(list)
    for r, v in zip(rows, values):
        by_src[r["source"]].append(float(v))
    srcs = sorted(by_src)
    arrs = [np.array(by_src[s]) for s in srcs]
    point = float(np.median(np.concatenate(arrs)))
    if len(srcs) < 2:
        return point, None, None, len(srcs)
    rng = np.random.default_rng(SEED)
    meds = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(srcs), len(srcs))
        meds[b] = np.median(np.concatenate([arrs[i] for i in pick]))
    return point, float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), len(srcs)


def degr(rows, key):
    e75 = np.array([float(r["mel75_vs_true"]) for r in rows])
    e13 = np.array([float(r[key]) for r in rows])
    return (e13 - e75) / np.maximum(e75, 1e-9)


def minarm_degr(rows):
    e75 = np.array([float(r["mel75_vs_true"]) for r in rows])
    dc = np.array([float(r["mel13c_vs_true"]) for r in rows])
    dp = np.array([float(r["mel13p_vs_true"]) for r in rows])
    return np.minimum((dc - e75) / np.maximum(e75, 1e-9), (dp - e75) / np.maximum(e75, 1e-9))


def part_a(rows):
    out = {"layers": [], "three_numbers": []}
    for pool, fn in POOLS.items():
        sel = [r for r in rows if fn(r)]
        nseq = len({r["sequence_id"] for r in sel})
        for layer, vals in (
            ("L1 逐条 min-arm", minarm_degr(sel)),
            ("L2 组级 13c 臂", degr(sel, "mel13c_vs_true")),
            ("L3 组级 13p 臂", degr(sel, "mel13p_vs_true")),
        ):
            p, lo, hi, nsrc = cluster_ci(sel, vals)
            v = np.asarray(vals, float)
            out["layers"].append({
                "column": "mel*_vs_true", "aggregate": "median", "pool": pool,
                "filter": {"click_60_200ms": "is_click_60_200==1",
                           "hihats": "sound_family=='hihats'",
                           "cymbal": "sound_family=='cymbal'"}[pool],
                "n": len(sel), "n_sequences": nseq, "n_sources": nsrc,
                "layer": layer,
                "min": round(float(v.min()) * 100, 1),
                "median_pct": round(p * 100, 1),
                "max": round(float(v.max()) * 100, 1),
                "ci95": None if lo is None else [round(lo * 100, 1), round(hi * 100, 1)],
            })
    # 58.1 / 100.6 / 170.0 归位
    sel = [r for r in rows if r["sound_family"] == "cymbal"]
    v = minarm_degr(sel)
    v = np.sort(v * 100)
    out["three_numbers"] = {
        "58.1": {"value": 58.1, "pool": "cymbal", "column": "mel*_vs_true",
                 "layer": "L1 逐条 min-arm", "statistic": "min", "n": len(sel),
                 "recomputed": round(float(v.min()), 1)},
        "100.6": {"value": 100.6, "pool": "cymbal", "column": "mel*_vs_true",
                  "layer": "L1 逐条 min-arm", "statistic": "median", "n": len(sel),
                  "recomputed": round(float(np.median(v)), 1)},
        "170.0": {"value": 170.0, "pool": "cymbal", "column": "mel*_vs_true",
                  "layer": "L1 逐条 min-arm", "statistic": "max", "n": len(sel),
                  "recomputed": round(float(v.max()), 1)},
    }
    return out


def part_c(pack_dir):
    """12 击源包：RMS dB 差 + raw mel + 电平剥离(gain-matched) mel。"""
    import soundfile as sf
    idx = json.loads((Path(pack_dir) / "index.json").read_text(encoding="utf-8"))
    rows = []
    for r in idx:
        base = f"{r['hit_id']}_h{r['hit_index']}"
        y = sf.read(str(Path(pack_dir) / "wav" / f"{base}_true.wav"), dtype="float32")[0]
        arms = {}
        for arm in ("75", "13c", "13p"):
            p = Path(pack_dir) / "wav" / f"{base}_{arm}.wav"
            if p.exists():
                arms[arm] = sf.read(str(p), dtype="float32")[0]
        tgt_rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))

        def gmatch(x):
            r = float(np.sqrt(np.mean(x ** 2) + 1e-12))
            return x * (tgt_rms / max(r, 1e-9))

        rec = {"hit_id": r["hit_id"], "hit_index": r["hit_index"],
               "sound_notation": r["sound_notation"], "dur_ms": r["dur_ms"]}
        for arm, x in arms.items():
            arm_rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
            raw = mel_metric.compare(y, x, "min")
            gmv = mel_metric.compare(y, gmatch(x), "min")
            rec[arm] = {
                "rms_db_vs_true": round(20 * np.log10(max(arm_rms, 1e-12) / tgt_rms), 2),
                "mel_raw": round(raw["value"], 5),
                "mel_gain_matched": round(gmv["value"], 5),
                "n_frames": raw["n_frames"], "frame_diff": raw["frame_diff"],
                "truncated": raw["truncated"],
            }
        rows.append(rec)
    agg = {}
    for arm in ("75", "13c", "13p"):
        d = np.array([x[arm]["rms_db_vs_true"] for x in rows])
        mr = np.array([x[arm]["mel_raw"] for x in rows])
        mg = np.array([x[arm]["mel_gain_matched"] for x in rows])
        agg[arm] = {
            "n": len(rows),
            "rms_db_median": round(float(np.median(d)), 2),
            "rms_db_min": round(float(d.min()), 2), "rms_db_max": round(float(d.max()), 2),
            "n_louder_than_1db": int((d > 1.0).sum()),
            "mel_raw_median": round(float(np.median(mr)), 4),
            "mel_gain_matched_median": round(float(np.median(mg)), 4),
            "mel_raw_median_over_gm_median": round(float(np.median(mr / np.maximum(mg, 1e-9))), 3),
        }
    return {"rows": rows, "agg": agg}


def main():
    got_md5, got_b = md5(PER_HIT), PER_HIT.stat().st_size
    if got_md5 != EXPECT_MD5 or got_b != EXPECT_BYTES:
        raise SystemExit(f"[HARD FAIL] per_hit 指纹不符 md5={got_md5} bytes={got_b}")
    with PER_HIT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 5472:
        raise SystemExit(f"[HARD FAIL] rows={len(rows)}")
    res = {"input_md5": got_md5, "input_bytes": got_b, "n_rows": len(rows)}
    res.update(part_a(rows))
    res["part_c_listen12"] = part_c("outputs/listen_hihats_v2")
    out = Path("outputs/critic_m6_recheck")
    out.mkdir(parents=True, exist_ok=True)
    (out / "critic_m6_recheck.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    print("=== (A) mel 列三层（列=mel*_vs_true，聚合=median，CI=按 source 聚类 bootstrap）===")
    for r in res["layers"]:
        ci = "n/a" if r["ci95"] is None else f"[{r['ci95'][0]}, {r['ci95'][1]}]"
        print(f"  {r['pool']:14s} {r['layer']:16s} n={r['n']:5d} seq={r['n_sequences']:4d} src={r['n_sources']:3d} "
              f"med={r['median_pct']:7.1f}% ci={ci} min={r['min']:6.1f} max={r['max']:6.1f}")
    print("=== (B) 58.1 / 100.6 / 170.0 归位 ===")
    for k, v in res["three_numbers"].items():
        print(f"  {k}: pool={v['pool']} column={v['column']} layer={v['layer']} "
              f"stat={v['statistic']} n={v['n']} recomputed={v['recomputed']}")
    print("=== (C) 试听 12 击源包 ===")
    for arm, a in res["part_c_listen12"]["agg"].items():
        print(f"  {arm:4s} n={a['n']} RMS {a['rms_db_median']:+.2f} dB [{a['rms_db_min']:+.2f},{a['rms_db_max']:+.2f}] "
              f" louder1db={a['n_louder_than_1db']}/12  mel_raw_med={a['mel_raw_median']:.4f} "
              f"mel_gm_med={a['mel_gain_matched_median']:.4f}")
    print(f"wrote {out}/critic_m6_recheck.json")


if __name__ == "__main__":
    main()
