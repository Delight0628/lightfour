"""
§10.1 #9(a) 最小判定实验：75Hz 直通重建 vs 13Hz 视图重建。

问题：Talker 本体是 13Hz / 16 组 / 2048 码本。若把 EnCodec 75Hz 的 16 层 code
压到 13Hz（每 77ms 一个 token），60–200ms 的单击还能不能保住？

做法（同一批音频、同一 codec、同一 mel 实现，唯一变量 = 帧率视图）：
  1. 读已有整段 codes (1,1,16,T) @75Hz（不重新编码，避免编码器造成的批次差）
  2. arm=75   : 直接解码 codes              -> w75   （上界参照）
  3. arm=13c  : 每 77ms 取「中心帧」，hold-repeat 回 75Hz -> w13c（确定性策略）
  4. arm=13p  : 每 77ms 取「能量最大帧」，hold-repeat 回 75Hz -> w13p（**oracle 上界**，
                对 13Hz 最有利；13Hz 若在 oracle 下都崩，结论无争议）
  5. 只在 逐击段 上算指标：log-mel 距离 与 高频能量占比(>=2kHz)
     分组：全体 / 60–200ms 单击 / hihats / cymbal（镲类必须单列）
  6. 判定：degradation = (e13 - e75) / e75，e* = mel(y_true, w*)；>10% → 13Hz 作废

为什么必须 hold-repeat 而不是抽帧后按 13Hz 解码：Talker 输出 13Hz token，解码器
仍然按 75Hz 跑，每一帧都要有码，所以只能用「一码管 77ms」的方式展开，这正是
真实推理时会发生的事（阶梯化 / 瞬态抹平）。

口径：
  - 16 层（RVQ 由粗到细），码本 1024，EnCodec 24kHz/24kbps，hop 320
  - 段内**不做** RMS 归一化（电平差算误差，刻意如此），两臂同等处理
  - mel: sr=24000, n_fft=1024, hop=256, n_mels=128, log(m+1e-6)
  - hf_ratio: stft n_fft=512 hop=128，>=2kHz 能量 / 总能量（与 eval_gen_audio 一致）
  - 音频只在服务器读，不落本地、不入 git

用法：
  python scripts/codec_rate_ab.py --limit 3 --out-dir /tmp/rate_ab_smoke   # smoke
  python scripts/codec_rate_ab.py                                          # 全量 287
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 24000
HOP = 320
SRC_FPS = 75.0
LAYERS = 16
CODEBOOK = 1024
MEL_SPEC = {"sr": SR, "n_fft": 1024, "hop": 256, "n_mels": 128, "log_eps": 1e-6}
HF_CUTOFF = 2000
BASE = Path("/gemini/code/lightfour-train")


# ---------------------------------------------------------------- 可复现指纹
def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _codes_digest(codes_dir, seq_ids):
    """对实际参与计算的 codes npz 取 name:md5 清单再做 sha256，避免整目录遍历的歧义。"""
    h = hashlib.sha256()
    for sid in sorted(seq_ids):
        p = Path(codes_dir) / f"{sid}.npz"
        if p.exists():
            h.update(f"{sid}:{_md5(p)}\n".encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------- 指标实现
def logmel(y):
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=MEL_SPEC["n_fft"],
                                       hop_length=MEL_SPEC["hop"],
                                       n_mels=MEL_SPEC["n_mels"])
    return np.log(m + MEL_SPEC["log_eps"])


def mel_dist(a, b):
    if len(a) < MEL_SPEC["n_fft"] or len(b) < MEL_SPEC["n_fft"]:
        return None
    A, B = logmel(a), logmel(b)
    m = min(A.shape[1], B.shape[1])
    if m == 0:
        return None
    return float(np.mean(np.abs(A[:, :m] - B[:, :m])))


def hf_ratio(y):
    import librosa
    if len(y) < 512:
        return None
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    return float(spec[freqs >= HF_CUTOFF].sum() / (spec.sum() + 1e-9))


def frame_rms(y, T):
    r = np.zeros(T, dtype=np.float64)
    for t in range(T):
        seg = y[t * HOP:(t + 1) * HOP]
        r[t] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return r


# ---------------------------------------------------------------- 13Hz 视图
def rate_view(codes, T, dst_fps, mode, rms=None):
    """codes: (1,1,L,T) int64 -> 同形状，每 75/dst_fps 帧只保留 1 帧并 hold-repeat。

    mode='center' 取区间中心帧（确定性、不看音频）
    mode='peak'   取区间内 rms 最大的帧（oracle 上界，对 13Hz 最有利）
    """
    step = SRC_FPS / float(dst_fps)
    out = np.empty_like(codes)
    n_bins = int(math.ceil(T / step))
    for i in range(n_bins):
        lo = int(round(i * step))
        hi = min(T, int(round((i + 1) * step)))
        if hi <= lo:
            hi = min(T, lo + 1)
        if lo >= T:
            break
        if mode == "center":
            pick = min(T - 1, max(0, (lo + hi - 1) // 2))
        elif mode == "peak":
            assert rms is not None
            pick = lo + int(np.argmax(rms[lo:hi]))
        else:
            raise ValueError(mode)
        out[..., lo:hi] = codes[..., pick:pick + 1]
    return out


# ---------------------------------------------------------------- 自检
def self_test():
    # 帧率换算：75 -> 13
    step = SRC_FPS / 13.0
    assert abs(step - 5.7692) < 1e-3, step
    T = 600
    codes = np.tile(np.arange(T, dtype=np.int64), (1, 1, LAYERS, 1))
    v = rate_view(codes, T, 13, "center")
    assert v.shape == codes.shape
    # hold-repeat：每个 13Hz bin 内码值恒定；源码单调则视图也单调
    # bin0 = 帧 [0,6)，中心帧 = 2 -> 前 6 帧全部为 2
    assert np.array_equal(v[0, 0, 0, :6], np.full(6, 2, dtype=np.int64)), v[0, 0, 0, :6]
    assert np.all(np.diff(v[0, 0, 0, :]) >= 0)  # 单调不降
    assert int(v[0, 0, 0, -1]) < T
    uniq = len(np.unique(v[0, 0, 0, :]))
    assert uniq == int(math.ceil(T / step)), (uniq, math.ceil(T / step))
    # 覆盖率：每个 75Hz 帧都被写入
    assert np.all(v[0, 0, 0, :] >= 0)
    # peak 版本：把能量峰值放在 bin0 的中心帧(2)，peak 应退化为 center
    r = np.ones(T)
    r[2] = 2.0
    vp = rate_view(codes, T, 13, "peak", rms=r)
    assert int(vp[0, 0, 0, 5]) == int(v[0, 0, 0, 5]), (vp[0, 0, 0, :6], v[0, 0, 0, :6])
    # 完全均匀音频下 13Hz 视图应无损（源码恒定则视图=源码）
    flat = np.zeros((1, 1, LAYERS, T), dtype=np.int64)
    vf = rate_view(flat, T, 13, "center")
    assert np.array_equal(vf, flat)
    print("self-test OK: 13Hz 视图帧数/覆盖率/hold-repeat/无损基线")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(BASE / "data/sequence_samples.csv"))
    ap.add_argument("--hits", default=str(BASE / "data/sequence_hits.csv"))
    ap.add_argument("--codes-dir", default=str(BASE / "data/seq_tokens/codes"))
    ap.add_argument("--vocals-root", default=str(BASE / "data/bbox_raw/demucs_srv/htdemucs"))
    ap.add_argument("--out-dir", default=str(BASE / "outputs/codec_rate_ab"))
    ap.add_argument("--model", default=str(BASE / "models/encodec_24khz"))
    ap.add_argument("--dst-fps", type=float, default=13.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--listen-n", type=int, default=12, help="试听包导出条数（0=不导出）")
    ap.add_argument("--filter-local-audio", action="store_true",
                    help="只做 audio_available=1 的序列。默认关：口径按 PI 裁决用全量 287 条"
                         "（audio_available 只表示本地切片在不在，不是训练可用性）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import librosa
    import soundfile as sf
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codec_tokens import load_model

    torch.set_num_threads(args.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.model, dev)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = list(csv.DictReader(open(args.samples, encoding="utf-8")))
    if args.filter_local_audio:
        samples = [r for r in samples if r["audio_available"] == "1"]
    hits_by_seq = defaultdict(list)
    for r in csv.DictReader(open(args.hits, encoding="utf-8")):
        if args.filter_local_audio and r["audio_available"] != "1":
            continue
        hits_by_seq[r["sequence_id"]].append(r)
    for v in hits_by_seq.values():
        v.sort(key=lambda x: float(x["offset"]))
    if args.limit:
        samples = samples[: args.limit]

    wav_cache = {}
    rows = []
    skipped = []
    for i, s in enumerate(samples, 1):
        sid = s["sequence_id"]
        npz = Path(args.codes_dir) / f"{sid}.npz"
        if not npz.exists():
            skipped.append({"sequence_id": sid, "reason": "codes missing"})
            continue
        src = s["source"]
        if src not in wav_cache:
            p = Path(args.vocals_root) / src / "vocals.wav"
            wav_cache[src] = librosa.load(str(p), sr=SR, mono=True)[0] if p.exists() else None
        y_all = wav_cache[src]
        if y_all is None:
            skipped.append({"sequence_id": sid, "reason": "vocals missing"})
            continue

        codes = np.load(npz)["codes"]
        assert codes.ndim == 4 and codes.shape[0] == 1 and codes.shape[1] == 1, codes.shape
        assert codes.shape[2] == LAYERS, codes.shape
        T = int(codes.shape[3])
        y = y_all[int(round(float(s["start"]) * SR)):int(round(float(s["end"]) * SR))]
        assert abs(T - math.ceil(len(y) / HOP)) <= 1, (sid, T, len(y))

        rms = frame_rms(y, T)
        c13c = rate_view(codes, T, args.dst_fps, "center")
        c13p = rate_view(codes, T, args.dst_fps, "peak", rms=rms)

        def dec(c):
            # transformers 5.16.1 的 EncodecModel.decode 只接受单个 frame（batch 必须=1），
            # 不能把三条臂堆 batch 一次解，只能逐臂调用。
            with torch.no_grad():
                o = model.decode(torch.from_numpy(c).to(dev), [None], None)
            return o.audio_values[0].mean(dim=0).cpu().numpy().astype("float32")

        w75, w13c, w13p = dec(codes), dec(c13c), dec(c13p)
        n = min(len(y), len(w75), len(w13c), len(w13p))
        y, w75, w13c, w13p = y[:n], w75[:n], w13c[:n], w13p[:n]

        for h in hits_by_seq.get(sid, []):
            off, hd = float(h["offset"]), float(h["dur"])
            a = int(round(off * SR))
            b = min(n, int(round((off + hd) * SR)))
            if b - a < MEL_SPEC["n_fft"]:
                continue
            ys, s75, s13c, s13p = y[a:b], w75[a:b], w13c[a:b], w13p[a:b]
            e75 = mel_dist(ys, s75)
            e13c = mel_dist(ys, s13c)
            e13p = mel_dist(ys, s13p)
            if e75 is None:
                continue
            # 增益对齐变体：把两臂都缩放到真值段的 RMS 再算 mel。
            # 目的是剥掉「hold-repeat 重复帧导致电平升高」这一非结构性偏差，
            # 只留下频谱形状/时间结构的差异 —— 对 13Hz 更有利的口径。
            tgt = float(np.sqrt(np.mean(ys ** 2) + 1e-12))

            def grms(x):
                r = np.sqrt(np.mean(x ** 2) + 1e-12)
                return x * (tgt / max(r, 1e-9))

            n75 = mel_dist(ys, grms(s75))
            n13c = mel_dist(ys, grms(s13c))
            n13p = mel_dist(ys, grms(s13p))

            # R1：13p 用真值 onset 做 ±1 个 13Hz 桶的时移对齐后再算 mel。
            # 1 桶 = SRC/dst 帧 = 6 帧 = 6*320 采样。整窗平移（在更长的解码波形上取窗），
            # 不做零填充、不改动波形本身；取 ±1 桶内最优值 = 对 13Hz 最有利的口径。
            shift_n = int(round(SRC_FPS / args.dst_fps)) * HOP
            best, best_bucket = None, None
            for sh in (0, shift_n, -shift_n):
                a2, b2 = a + sh, b + sh
                if a2 < 0 or b2 > n or b2 - a2 < MEL_SPEC["n_fft"]:
                    continue
                e = mel_dist(ys, grms(w13p[a2:b2]))
                if e is not None and (best is None or e < best):
                    best, best_bucket = e, sh // shift_n
            rows.append({
                "sequence_id": sid, "hit_index": h["hit_index"], "hit_id": h["hit_id"],
                "source": src, "split": s["split"],
                "sound_id": h["sound_id"], "sound_notation": h["sound_notation"],
                "sound_family": h["sound_family"],
                "is_short_variant": h["is_short_variant"],
                "dur_ms": round(hd * 1000, 1),
                "is_click_60_200": int(0.060 <= hd <= 0.200),
                "mel75_vs_true": round(e75, 5),
                "mel13c_vs_true": round(e13c, 5),
                "mel13p_vs_true": round(e13p, 5),
                "mel13c_vs_75": round(mel_dist(s75, s13c), 5),
                "mel13p_vs_75": round(mel_dist(s75, s13p), 5),
                "nmel75_vs_true": round(n75, 5),
                "nmel13c_vs_true": round(n13c, 5),
                "nmel13p_vs_true": round(n13p, 5),
                "hf_true": round(hf_ratio(ys), 5),
                "hf75": round(hf_ratio(s75), 5),
                "hf13c": round(hf_ratio(s13c), 5),
                "hf13p": round(hf_ratio(s13p), 5),
                # 语义：layer0 在 13c 视图里出现的**唯一码值个数**（中位 ~52），
                # 不是 13Hz 桶数（T/5.769），别当帧率证据用。
                "n13c_uniq_layer0_codes": int(len(np.unique(c13c[0, 0, 0, :]))),
                # R1 结果：±1 桶对齐后的 gain_matched mel 与获胜位移（桶）
                "nmel13p_align_vs_true": round(best, 5) if best is not None else "",
                "align_shift_bucket": best_bucket if best_bucket is not None else "",
                "mel_frames": "|".join(str(logmel(x).shape[1]) for x in (ys, s75, s13c, s13p)),
            })
        if i % 20 == 0:
            print(f"[{i}/{len(samples)}] hits={len(rows)}", flush=True)

    if not rows:
        raise SystemExit("没有任何逐击行，检查输入")

    # ------------------------------------------------ 汇总
    def agg(sel, arm, prefix=""):
        e75 = np.array([r[prefix + "mel75_vs_true"] for r in sel])
        e13 = np.array([r[prefix + f"mel13{arm}_vs_true"] for r in sel])
        d = (e13 - e75) / np.maximum(e75, 1e-9)
        hf75 = np.array([r["hf75"] for r in sel])
        hf13 = np.array([r[f"hf13{arm}"] for r in sel])
        hft = np.array([r["hf_true"] for r in sel])
        return {
            "n": len(sel), "arm": f"13{arm}",
            "mel75_mean": round(float(e75.mean()), 4),
            "mel75_median": round(float(np.median(e75)), 4),
            "mel13_mean": round(float(e13.mean()), 4),
            "mel13_median": round(float(np.median(e13)), 4),
            "degr_mean_pct": round(float(d.mean()) * 100, 1),
            "degr_median_pct": round(float(np.median(d)) * 100, 1),
            "degr_worse_share": round(float((e13 > e75).mean()), 4),
            "hf_true_mean": round(float(np.nanmean(hft)), 4),
            "hf75_mean": round(float(np.nanmean(hf75)), 4),
            "hf13_mean": round(float(np.nanmean(hf13)), 4),
            "hf_abs_err_75": round(float(np.nanmean(np.abs(hf75 - hft))), 5),
            "hf_abs_err_13": round(float(np.nanmean(np.abs(hf13 - hft))), 5),
        }

    groups = {
        "all_hits": rows,
        "click_60_200ms": [r for r in rows if r["is_click_60_200"] == 1],
        "hihats": [r for r in rows if r["sound_family"] == "hihats"],
        "cymbal": [r for r in rows if r["sound_family"] == "cymbal"],
        "short_variant": [r for r in rows if r["is_short_variant"] == "1"],
    }
    fams = defaultdict(list)
    for r in rows:
        fams[r["sound_family"]].append(r)
    for f, v in fams.items():
        if len(v) >= 25:
            groups[f"family:{f}"] = v

    table = []
    for g, sel in groups.items():
        if not sel:
            continue
        for arm in ("c", "p"):
            table.append({"group": g, "metric": "raw", **agg(sel, arm)})
            table.append({"group": g, "metric": "gain_matched", **agg(sel, arm, "n")})

    # ------------------------------------------------ R1：±1 桶 onset 对齐（只对 13p）
    def agg_align(sel):
        sel = [r for r in sel if r["nmel13p_align_vs_true"] != ""]
        if not sel:
            return None
        e75 = np.array([r["nmel75_vs_true"] for r in sel], dtype=float)
        e13 = np.array([r["nmel13p_align_vs_true"] for r in sel], dtype=float)
        d = (e13 - e75) / np.maximum(e75, 1e-9)
        shifts = defaultdict(int)
        for r in sel:
            shifts[int(r["align_shift_bucket"])] += 1
        return {
            "n": len(sel), "arm": "13p_align",
            "mel75_median": round(float(np.median(e75)), 4),
            "mel13_median": round(float(np.median(e13)), 4),
            "degr_median_pct": round(float(np.median(d)) * 100, 1),
            "degr_mean_pct": round(float(d.mean()) * 100, 1),
            "degr_worse_share": round(float((e13 > e75).mean()), 4),
            "shift_counts": dict(sorted(shifts.items())),
        }

    for g, sel in groups.items():
        a = agg_align(sel)
        if a:
            table.append({"group": g, "metric": "gain_matched_onset_aligned", **a})

    # ------------------------------------------------ R3：按 source 聚类 bootstrap CI
    def cluster_ci(sel, arm="p", prefix="n", n_boot=2000, seed=20261027):
        """退化中位数的 95% CI，**按 source 重采样**（同一 source 的击打不独立）。"""
        if not sel:
            return None
        by_src = defaultdict(list)
        for r in sel:
            e75 = float(r[prefix + "mel75_vs_true"])
            e13 = float(r[prefix + f"mel13{arm}_vs_true"])
            by_src[r["source"]].append((e13 - e75) / max(e75, 1e-9))
        srcs = sorted(by_src)
        arrs = [np.array(by_src[s], dtype=float) for s in srcs]
        point = float(np.median(np.concatenate(arrs)))
        out = {"n": len(sel), "n_sources": len(srcs),
               "point_median_pct": round(point * 100, 1)}
        if len(srcs) < 2:
            out.update({"n_boot": 0, "seed": seed, "ci95_median_pct": None,
                        "note": "source<2，聚类 bootstrap 无意义，只报点估计"})
            return out
        rng = np.random.default_rng(seed)
        meds = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            pick = rng.integers(0, len(srcs), len(srcs))
            meds[b] = np.median(np.concatenate([arrs[i] for i in pick]))
        out.update({"n_boot": n_boot, "seed": seed,
                    "ci95_median_pct": [round(float(np.percentile(meds, 2.5)) * 100, 1),
                                        round(float(np.percentile(meds, 97.5)) * 100, 1)]})
        return out

    cluster = {g: cluster_ci(groups[g]) for g in
               ("all_hits", "click_60_200ms", "hihats", "cymbal") if groups.get(g)}

    # 判定：取对 13Hz **最有利** 的口径与臂 —— 增益对齐 + 退化最小的那个 arm。
    clicks = [r for r in rows if r["is_click_60_200"] == 1]
    dec_rows = [t for t in table
                if t["group"] == "click_60_200ms" and t["metric"] == "gain_matched"]
    dec = min(dec_rows, key=lambda t: t["degr_median_pct"]) if dec_rows else None
    verdict = "NO_DATA"
    if dec:
        verdict = "KILL_13HZ" if dec["degr_median_pct"] > 10 else "13HZ_MAX_DEGRADATION_LE10"

    # 时间分辨率算术（与解码无关的硬约束）：13Hz = 76.9ms/格，
    # 任何两个 onset 间隔 < 76.9ms 的击打在 13Hz token 流里必然合并成一格。
    BWIN = 1000.0 / args.dst_fps
    gaps = []
    for sid, hs in hits_by_seq.items():
        for a_, b_ in zip(hs, hs[1:]):
            gaps.append((float(b_["offset"]) - float(a_["offset"])) * 1000.0)
    gaps = np.array(gaps)
    gap_stats = {
        "bin_ms": round(BWIN, 1),
        "n_pairs": int(len(gaps)),
        "share_gap_lt_bin": round(float((gaps < BWIN).mean()), 4),
        "share_gap_lt_2bin": round(float((gaps < 2 * BWIN).mean()), 4),
        "gap_p05_ms": round(float(np.percentile(gaps, 5)), 1),
        "gap_p25_ms": round(float(np.percentile(gaps, 25)), 1),
        "gap_median_ms": round(float(np.median(gaps)), 1),
    }

    summary = {
        "experiment": "§10.1 #9(a) 75Hz 直通 vs 13Hz 视图（hold-repeat）",
        "codec": "encodec_24khz / bandwidth 24kbps / layers 16 / codebook 1024",
        "frame_rate_src_hz": SRC_FPS, "frame_rate_dst_hz": args.dst_fps,
        "mel_spec": MEL_SPEC, "hf_cutoff_hz": HF_CUTOFF,
        "level_normalization": "两套口径：raw（不归一化，电平差算误差）+ gain_matched"
                               "（两臂都缩放到真值段 RMS，剥掉重复帧造成的电平增益，对 13Hz 更有利）",
        "n_sequences": len(samples), "n_hits": len(rows),
        "n_click_60_200ms": len(clicks),
        "n_hihats": len(groups["hihats"]), "n_cymbal": len(groups["cymbal"]),
        "median_dur_ms_click": round(float(np.median([r["dur_ms"] for r in clicks])), 1) if clicks else None,
        "arms": {
            "75": "直通 codes 解码",
            "13c": f"每 {round(SRC_FPS/args.dst_fps,3)} 帧取中心帧 + hold-repeat",
            "13p": f"每 {round(SRC_FPS/args.dst_fps,3)} 帧取能量最大帧（oracle 选帧，但非 mel-最优）",
        },
        "decision_rule": "click_60_200ms 的 gain_matched mel 退化中位 >10%（取退化更小的臂）→ 13Hz 作废",
        "decision_basis": dec,
        "r1_onset_aligned": {
            "question": "13p 解码后用真值 onset 做 ±1 个 13Hz 桶时移对齐（整窗平移，不许改码流），再算 gain_matched mel",
            "click_gain_matched_degr_median_pct": (agg_align(clicks) or {}).get("degr_median_pct"),
            "click_gain_matched_degr_mean_pct": (agg_align(clicks) or {}).get("degr_mean_pct"),
            "click_mel75_median": (agg_align(clicks) or {}).get("mel75_median"),
            "click_mel13_median": (agg_align(clicks) or {}).get("mel13_median"),
            "click_n": (agg_align(clicks) or {}).get("n"),
            "by_group": {g: agg_align(groups[g]) for g in
                         ("all_hits", "click_60_200ms", "hihats", "cymbal")
                         if groups.get(g) and agg_align(groups[g])},
        },
        "cluster_bootstrap": {
            "method": "按 source 重采样（cluster bootstrap），统计量为 gain_matched 退化中位数，arm=13p",
            "groups": cluster,
        },
        "column_semantics": {
            "hf_abs_err_*": "逐条 |hf_arm - hf_true| 的**平均绝对误差(MAE)**，不是均值差；"
                            "引用时必须写明 MAE",
            "n13c_uniq_layer0_codes": "13c 视图中 layer0 的**唯一码值个数**（中位 ~52），"
                                      "不是 13Hz 桶数（T/5.769），不可当帧率证据",
            "nmel13p_align_vs_true": "R1：13p 在 ±1 桶内最优位移后的 gain_matched mel",
        },
        "temporal_resolution": gap_stats,
        "verdict": verdict,
        "skipped": skipped,
        "generated_by": "scripts/codec_rate_ab.py",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_md5": {
            "sequence_samples": _md5(args.samples),
            "sequence_hits": _md5(args.hits),
            "codes_npz": _codes_digest(args.codes_dir, [s["sequence_id"] for s in samples]),
            "n_codes_hashed": len(samples),
        },
        "args": {k: v for k, v in vars(args).items()},
    }

    with open(out_dir / "per_hit.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "table": table}, f, ensure_ascii=False, indent=1)

    md = ["# 75Hz 直通 vs 13Hz 视图（hold-repeat）", "",
          f"- codec: {summary['codec']}",
          f"- 序列 {summary['n_sequences']} / 击打段 {summary['n_hits']}"
          f"（60–200ms {summary['n_click_60_200ms']}、hihats {summary['n_hihats']}、"
          f"cymbal {summary['n_cymbal']}）",
          f"- mel: {MEL_SPEC}；hf: >= {HF_CUTOFF}Hz；两套口径 raw / gain_matched",
          f"- 判定规则: {summary['decision_rule']} → **{verdict}**", "",
          "| group | metric | arm | n | mel75 med | mel13 med | 退化中位 | 退化均值 | 13更差占比 | hf_true | hf75 | hf13 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        md.append(f"| {t['group']} | {t['metric']} | {t['arm']} | {t['n']} | {t['mel75_median']} | "
                  f"{t['mel13_median']} | {t['degr_median_pct']}% | {t['degr_mean_pct']}% | "
                  f"{t['degr_worse_share']:.2f} | {t.get('hf_true_mean', '-')} | "
                  f"{t.get('hf75_mean', '-')} | {t.get('hf13_mean', '-')} |")
    ts = summary["temporal_resolution"]
    md += ["", "## 时间分辨率硬约束（与解码无关的算术）", "",
           f"- 13Hz = {ts['bin_ms']}ms/格；相邻 onset 间隔 < 1 格的占 "
           f"**{ts['share_gap_lt_bin']*100:.1f}%**，< 2 格的占 "
           f"**{ts['share_gap_lt_2bin']*100:.1f}%**（{ts['n_pairs']} 对）",
           f"- 间隔分位：p05={ts['gap_p05_ms']}ms p25={ts['gap_p25_ms']}ms "
           f"中位={ts['gap_median_ms']}ms", ""]

    r1 = summary["r1_onset_aligned"]
    md += ["## R1：13p 用真值 onset ±1 桶对齐后的退化", "",
           f"- **click_60_200ms gain_matched 退化中位 = {r1['click_gain_matched_degr_median_pct']}%**"
           f"（n={r1['click_n']}，mel75 中位 {r1['click_mel75_median']} → "
           f"mel13 中位 {r1['click_mel13_median']}）",
           f"- 均值退化 {r1['click_gain_matched_degr_mean_pct']}%；"
           "口径：整窗平移 ±1 桶（76.9ms），取三者最优，码流本身不动", ""]
    for g, a in r1["by_group"].items():
        md.append(f"- {g}: n={a['n']} 退化中位 {a['degr_median_pct']}% "
                  f"位移分布 {a['shift_counts']}")
    md += ["", "## 按 source 聚类的 95% CI（退化中位数，arm=13p，gain_matched）", "",
           "| group | n | n_sources | 点估计 | 95% CI |", "|---|---:|---:|---:|---|"]
    for g, c in summary["cluster_bootstrap"]["groups"].items():
        ci = c.get("ci95_median_pct")
        ci_s = f"[{ci[0]}%, {ci[1]}%]" if ci else "N/A（source<2）"
        md.append(f"| {g} | {c['n']} | {c['n_sources']} | {c['point_median_pct']}% | {ci_s} |")
    md += ["", "## 字段口径（避免误读）", ""]
    for k, v in summary["column_semantics"].items():
        md.append(f"- `{k}`: {v}")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------ 试听包（按 family 分层，不能只靠最差排序）
    if args.listen_n:
        lp = out_dir / "listen"
        lp.mkdir(parents=True, exist_ok=True)
        for stale in lp.glob("*.wav"):   # 清掉上一轮的 wav，避免换基数后残留旧样本
            stale.unlink()

        def fam(r):
            return r["sound_family"]

        def spread(sel, k):
            """在 family 内按退化排序后均匀取 k 条（最差/中位/最好都覆盖），
            不做「取 top-k」——那样镲类若整体退化更大就会把 hihats 全挤掉。"""
            sel = sorted(sel, key=lambda r: r["mel13p_vs_75"])
            if len(sel) <= k:
                return sel
            if k == 1:
                return [sel[len(sel) // 2]]
            idx = [round(i * (len(sel) - 1) / (k - 1)) for i in range(k)]
            return [sel[i] for i in idx]

        pools = {
            "cymbal": [r for r in rows if fam(r) == "cymbal"],
            "hihats": [r for r in rows if fam(r) == "hihats"],
            "click": [r for r in rows if r["is_click_60_200"]
                      and fam(r) not in ("hihats", "cymbal")],
        }
        per = max(1, args.listen_n // len(pools))
        picked = []
        for pool in pools.values():
            picked += spread(pool, per)
        if len(picked) < args.listen_n:
            used = {(r["sequence_id"], r["hit_index"]) for r in picked}
            rest = sorted([r for r in rows
                           if (r["sequence_id"], r["hit_index"]) not in used],
                          key=lambda r: -r["mel13p_vs_75"])
            picked += rest[: args.listen_n - len(picked)]
        picked = picked[: args.listen_n]
        comp = {k: sum(1 for r in picked if r in v) for k, v in pools.items()}
        comp["_per_family_quota"] = per
        (lp / "composition.json").write_text(
            json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
        # picked 存的是指标行（没有 offset），需要回查原始击打行
        orig = {(r["sequence_id"], r["hit_index"]): r
                for r in csv.DictReader(open(args.hits, encoding="utf-8"))}
        # 需要重解码：为省内存只重建被选序列
        by_seq = defaultdict(list)
        for r in picked:
            by_seq[r["sequence_id"]].append(r)
        for sid, hs in by_seq.items():
            s = next(x for x in samples if x["sequence_id"] == sid)
            y_all = wav_cache[s["source"]]
            y = y_all[int(round(float(s["start"]) * SR)):int(round(float(s["end"]) * SR))]
            codes = np.load(Path(args.codes_dir) / f"{sid}.npz")["codes"]
            T = int(codes.shape[3])
            rms = frame_rms(y, T)
            views = {
                "true": None,
                "75": codes,
                "13c": rate_view(codes, T, args.dst_fps, "center"),
                "13p": rate_view(codes, T, args.dst_fps, "peak", rms=rms),
            }
            auds = {}
            for k, v in views.items():
                if v is None:
                    auds[k] = y
                else:
                    with torch.no_grad():
                        o = model.decode(torch.from_numpy(v).to(dev), [None], None)
                    auds[k] = o.audio_values[0].mean(dim=0).cpu().numpy().astype("float32")
            n = min(len(a) for a in auds.values())
            for h in hs:
                oh = orig[(h["sequence_id"], h["hit_index"])]
                a = int(round(float(oh["offset"]) * SR))
                b = min(n, int(round((float(oh["offset"]) + float(oh["dur"])) * SR)))
                for k, au in auds.items():
                    sf.write(str(lp / f"{oh['hit_id']}_h{oh['hit_index']}_{k}.wav"),
                             au[a:b], SR)
        (lp / "index.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        with open(out_dir / "summary.md", "a", encoding="utf-8") as f:
            f.write("\n## 试听包构成（分层抽样，禁止当作 family 已验证的证据）\n\n")
            f.write(f"- 每 family 配额 {per}，实际构成 {comp}\n")
            f.write(f"- 包内 {len(picked)} 条，文件 {len(list(lp.glob('*.wav')))} 个"
                    f"（每条 4 臂：true/75/13c/13p）\n")

    print(json.dumps({"n_hits": len(rows), "verdict": verdict, "out": str(out_dir)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
