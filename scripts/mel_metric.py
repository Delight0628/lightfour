"""Lightfour 唯一 log-mel 实现（评测红线：mel 数字只准出自本文件）。

背景（agent.md §5.2j 裁决① / §5.2g 依据四）：
  历史 mel 数字散落在 codec_rate_ab.py / codec_c1.py / eval_gen_audio.py 各自的复制品里，
  跨脚本绝对值不可比。本文件把 codec_rate_ab.py 的冻结实现抽出来，其余脚本改为 import，
  保证「同一批音频 → 同一个数」。

冻结口径：
  sr=24000, n_fft=1024, hop=256, n_mels=128, log(m + 1e-6)
  一帧 = hop/sr = 10.67ms（这条决定了「短击打只有 ~17 帧」的算术）

帧长口径（frame_policy）：
  min  默认。截断到较短的一侧，**但必须先过帧差断言**：|n_a - n_b| > max(1, ceil(2%·max))
       直接 raise FrameMismatch。禁止无断言的 min 截断（旧 bug 就是静默截断）。
  pad  只在敏感性附录用，不进决策表。零填充波形到等长再算 mel（不是填 mel 矩阵）。

为什么 pad 不能当主口径（@Data 实测 95 条短击打）：一条臂短 1 帧时 pad 会把 mel 距离
  从 p50 0.0087 抬到 0.0245（2.2×），37/95 条被抬 >0.05 绝对值 —— 这个伪差比 L16→L32
  的真实改善（0.066）还大，足够改写 #9(a) 结论。

用法：
  python scripts/mel_metric.py --self-test
  python scripts/mel_metric.py --a a.wav --b b.wav [--frame-policy min|pad]
  python scripts/mel_metric.py --reproduce-listen outputs/codec_rate_ab/listen
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SR = 24000
N_FFT = 1024
HOP = 256
N_MELS = 128
LOG_EPS = 1e-6
MEL_SPEC = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "log_eps": LOG_EPS}

FRAME_POLICIES = ("min", "pad")
DEFAULT_FRAME_POLICY = "min"
FRAME_DIFF_ABS_TOL = 1        # 至少容忍 1 帧（多臂解码长度会差 1 帧）
FRAME_DIFF_REL_TOL = 0.02     # 或 2% 帧数，取较大者


class FrameMismatch(ValueError):
    """两臂 mel 帧数差超容差。出现即说明有别的 bug，必须停下来查，不许静默截断。"""


def logmel(y: np.ndarray) -> np.ndarray:
    """(n_samples,) -> (n_mels, n_frames) log-mel，冻结口径。"""
    import librosa

    y = np.ascontiguousarray(y, dtype=np.float32).ravel()
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return np.log(m + LOG_EPS)


def n_frames(n_samples: int) -> int:
    """librosa center=True 的帧数（与 logmel(...).shape[1] 一致）。"""
    return 1 + int(n_samples) // HOP


def frame_tolerance(n_a: int, n_b: int) -> int:
    return max(FRAME_DIFF_ABS_TOL, int(math.ceil(FRAME_DIFF_REL_TOL * max(n_a, n_b))))


def compare(a, b, frame_policy: str = DEFAULT_FRAME_POLICY, strict: bool = True) -> dict:
    """两段波形的 log-mel 平均绝对距离。

    返回 dict（不要把 value 单独取出来丢掉帧信息，写卡时必须连 n_frames/frame_diff 一起写）：
      value         float，均值绝对距离
      frame_policy  min|pad
      n_frames      [n_a, n_b]（pad 口径下为 pad 后的帧数，pre_pad 见 pre_pad_n_frames）
      frame_diff    原始（padding 之前）帧数差
      frame_tol     本次允许的最大帧差
      truncated     被丢掉的最长一侧帧数（min 口径），pad 口径恒为 0
    """
    if frame_policy not in FRAME_POLICIES:
        raise ValueError(f"frame_policy 必须是 {FRAME_POLICIES}，收到 {frame_policy!r}")
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return {"value": None, "frame_policy": frame_policy, "n_frames": [0, 0],
                "frame_diff": 0, "frame_tol": 0, "truncated": 0, "reason": "empty"}
    if a.size < N_FFT or b.size < N_FFT:
        return {"value": None, "frame_policy": frame_policy,
                "n_frames": [n_frames(a.size), n_frames(b.size)],
                "frame_diff": abs(n_frames(a.size) - n_frames(b.size)),
                "frame_tol": frame_tolerance(n_frames(a.size), n_frames(b.size)),
                "truncated": 0, "reason": "shorter than n_fft"}

    na, nb = n_frames(a.size), n_frames(b.size)
    diff = abs(na - nb)
    tol = frame_tolerance(na, nb)
    if strict and diff > tol:
        raise FrameMismatch(
            f"mel 帧数差 {diff} > 容差 {tol}（n_a={na}, n_b={nb}, "
            f"samples={a.size}/{b.size}）；这不是正常解码长度抖动，先查上游")

    if frame_policy == "pad":
        n = max(a.size, b.size)
        a2 = np.pad(a, (0, n - a.size))
        b2 = np.pad(b, (0, n - b.size))
        A, B = logmel(a2), logmel(b2)
        m = min(A.shape[1], B.shape[1])
        value = float(np.mean(np.abs(A[:, :m] - B[:, :m]))) if m else None
        return {"value": value, "frame_policy": "pad",
                "n_frames": [A.shape[1], B.shape[1]], "pre_pad_n_frames": [na, nb],
                "frame_diff": diff, "frame_tol": tol, "truncated": 0}

    A, B = logmel(a), logmel(b)
    m = min(A.shape[1], B.shape[1])
    value = float(np.mean(np.abs(A[:, :m] - B[:, :m]))) if m else None
    return {"value": value, "frame_policy": "min", "n_frames": [A.shape[1], B.shape[1]],
            "frame_diff": diff, "frame_tol": tol, "truncated": int(abs(A.shape[1] - B.shape[1]))}


def mel_dist(a, b, frame_policy: str = DEFAULT_FRAME_POLICY) -> float | None:
    """兼容旧脚本的纯数值接口；要帧信息请直接用 compare()。"""
    return compare(a, b, frame_policy)["value"]


def compare_many(ref, arms: dict, frame_policy: str = DEFAULT_FRAME_POLICY,
                 strict: bool = True) -> dict:
    """一段真值 vs 多条臂，一次给出各臂数值 + **最大帧差**（自证必须报 max，不只报比例）。

    strict=False 只给诊断/自检用（要看到超阈帧差本身），不许进决策表。
    """
    out, frames = {}, {}
    diffs = {}
    for k, v in arms.items():
        r = compare(ref, v, frame_policy, strict=strict)
        out[k] = r["value"]
        frames[k] = r["n_frames"][1]
        diffs[k] = r["frame_diff"]
    n_ref = n_frames(len(np.asarray(ref).ravel()))
    return {
        "values": out,
        "n_frames_ref": n_ref,
        "n_frames_arm": frames,
        "frame_diff": diffs,
        "frame_diff_max": int(max(diffs.values())) if diffs else 0,
        "frame_policy": frame_policy,
        "mel_frames": "|".join(str(x) for x in [n_ref] + list(frames.values())),
    }


# ------------------------------------------------------------------ 自检
def self_test() -> None:
    rng = np.random.default_rng(20261027)
    dur = 0.30
    y = (rng.standard_normal(int(SR * dur)) * 0.1).astype(np.float32)
    y[-256:] += 0.9 * np.sin(2 * np.pi * 4000 * np.arange(256) / SR)

    # 1) 同信号 → 0，且逐位可复现
    r1, r2 = compare(y, y), compare(y, y)
    assert r1["value"] == 0.0 and r2["value"] == r1["value"], (r1, r2)
    assert r1["frame_diff"] == 0 and r1["truncated"] == 0

    # 2) 短 1 帧：min 不 raise，帧差 = 1（这是 C1 里「各臂差 1 帧」的常态）
    y1 = y[:-HOP]
    r = compare(y, y1)
    assert r["frame_diff"] == 1 and r["truncated"] == 1 and r["value"] > 0, r
    assert r["n_frames"] == [n_frames(y.size), n_frames(y1.size)]

    # 3) 短 3 帧：超容差必须 raise（禁止静默截断）
    y3 = y[:-3 * HOP]
    try:
        compare(y, y3)
    except FrameMismatch:
        pass
    else:
        raise AssertionError("帧差 3 > 容差 1 却没 raise，静默截断 bug 复现")

    # 4) pad 口径不 raise，但会把距离抬高（@Data：短 1 帧 p50 抬 2.2×）
    rp = compare(y, y1, "pad")
    rm = compare(y, y1, "min", strict=False)
    assert rp["frame_diff"] == 1 and rp["n_frames"][0] == rp["n_frames"][1]
    assert rp["value"] > rm["value"], (rp["value"], rm["value"])

    # 5) 容差随帧数增长（600 帧时容差 = 12）
    assert frame_tolerance(600, 600) == 12, frame_tolerance(600, 600)
    assert frame_tolerance(20, 19) == 1

    # 6) compare_many 报的是最大帧差，不是比例
    rm_ = compare_many(y, {"a": y, "b": y1, "c": y[:-2 * HOP]}, strict=False)
    assert rm_["frame_diff_max"] == 2, rm_
    assert rm_["n_frames_ref"] == n_frames(y.size)

    print("mel_metric self-test OK: 冻结口径/帧差断言/pad 抬升方向/max 帧差 "
          f"[n_fft={N_FFT} hop={HOP} 帧={1000*HOP/SR:.2f}ms]")


# ------------------------------------------------------------------ 复现试听包三臂
def reproduce_listen(pack_dir: Path) -> dict:
    """用本实现重算 listen/index.json 里每条的四臂 mel，和 pkl 里存的数对比。

    这条命令是「mel_metric 落地后 16/32 两臂重测」之前的最小自证：
    若 max |recomputed - stored| ≈ 0，说明抽出实现没有改变任何数字。
    """
    import soundfile as sf

    idx = json.loads((pack_dir / "index.json").read_text(encoding="utf-8"))
    per_row, fam_vals = [], {}
    max_frame_diff = 0
    frame_diff_hist = {}
    for r in idx:
        base = f"{r['hit_id']}_h{r['hit_index']}"
        y = sf.read(str(pack_dir / f"{base}_true.wav"), dtype="float32")[0]
        arms = {}
        for arm in ("75", "13c", "13p"):
            p = pack_dir / f"{base}_{arm}.wav"
            if p.exists():
                arms[arm] = sf.read(str(p), dtype="float32")[0]
        if len(arms) < 3:
            continue
        m = compare_many(y, arms, "min")
        max_frame_diff = max(max_frame_diff, m["frame_diff_max"])
        for k, d in m["frame_diff"].items():
            frame_diff_hist[str(d)] = frame_diff_hist.get(str(d), 0) + 1
        rec = {"hit_id": r["hit_id"], "hit_index": r["hit_index"],
               "sound_family": r["sound_family"], "sound_notation": r["sound_notation"],
               "sound_id": r["sound_id"], "source": r["source"],
               "frame_diff_max": m["frame_diff_max"], "mel_frames": m["mel_frames"]}
        for arm in ("75", "13c", "13p"):
            val = m["values"][arm]
            stored = r.get(f"mel{arm}_vs_true")
            rec[f"mel{arm}_vs_true"] = None if val is None else round(val, 5)
            rec[f"stored_mel{arm}_vs_true"] = stored
            rec[f"delta_{arm}"] = (None if (val is None or stored in (None, ""))
                                   else round(abs(val - float(stored)), 5))
            fam_vals.setdefault(r["sound_family"], {}).setdefault(arm, []).append(val)
        per_row.append(rec)

    fam = {}
    for f, d in fam_vals.items():
        fam[f] = {"n": len(next(iter(d.values())))}
        for arm, vals in d.items():
            v = np.array([x for x in vals if x is not None], dtype=float)
            fam[f][f"mel{arm}_mean"] = round(float(v.mean()), 4) if v.size else None
            fam[f][f"mel{arm}_median"] = round(float(np.median(v)), 4) if v.size else None

    deltas = [abs(pr[f"delta_{a}"]) for pr in per_row for a in ("75", "13c", "13p")
              if pr[f"delta_{a}"] is not None]
    return {
        "pack_dir": str(pack_dir), "n_rows": len(per_row),
        "frame_diff_max": max_frame_diff, "frame_diff_hist": frame_diff_hist,
        "max_abs_delta_vs_stored": round(max(deltas), 5) if deltas else None,
        "by_family": fam, "rows": per_row,
    }


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--frame-policy", default=DEFAULT_FRAME_POLICY, choices=FRAME_POLICIES)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--reproduce-listen", help="试听包目录（含 index.json + *_true.wav）")
    ap.add_argument("--json-out", help="把复现结果写到这个 json 路径")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.reproduce_listen:
        res = reproduce_listen(Path(args.reproduce_listen))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                         ensure_ascii=False, indent=1))
        return
    if not (args.a and args.b):
        ap.error("需要 --self-test 或 --reproduce-listen 或 --a/--b")

    import soundfile as sf

    a = sf.read(args.a, dtype="float32")[0]
    b = sf.read(args.b, dtype="float32")[0]
    print(json.dumps(compare(a, b, args.frame_policy), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
