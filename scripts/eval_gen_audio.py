"""
生成音频的多维评测：不只 mel（mel 平滑，会掩盖瞬态丢失）。

对每一对 (gt.wav, gen.wav) 计算：
  1. 时长比          len(gen)/len(gt)
  2. onset 对齐误差   第一个能量起点的采样偏差（ms）
  3. 包络相关性       RMS 包络 Pearson 相关（衡量衰减形状）
  4. attack-to-decay  峰值到 -20dB 的衰减时长（ms）
  5. 高频能量占比     2kHz 以上能量 / 总能量（锤/镲类最敏感）
  6. 谱通量相关       spectral flux 序列相关（瞬态密度）
  7. mel 距离         仅作参考，不作主判据

用法：
  python scripts/eval_gen_audio.py --dir /tmp/tiny_codec_gen
  python scripts/eval_gen_audio.py --dir /tmp/tiny_codec_gen --json out.json
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def rms_env(y, hop=64):
    return librosa.feature.rms(y=y, frame_length=256, hop_length=hop)[0]


def first_onset(y, sr=24000, win_ms=2.0, rel=0.1):
    """首个能量起点，返回**样本级浮点位置**（线性插值到阈值交点）。

    旧实现用 RMS 帧索引（hop=64 -> 2.67ms 量化）且实测系统性低估 1 帧：
    真值 5.33/10/20/40ms -> 测得 2.67/5.33/16/37.33ms。改样本级 + 插值后
    校准误差 < 1ms（见 scripts/eval_gen_audio.py 自检）。
    """
    w = max(1, int(round(sr * win_ms / 1000.0)))
    env = np.convolve(np.abs(y), np.ones(w, dtype=np.float64) / w, mode="same")
    if env.size == 0:
        return None
    peak = float(env.max())
    if peak <= 0:
        return None
    thr = rel * peak
    idx = int(np.argmax(env > thr))
    if idx > 0:
        y0, y1 = float(env[idx - 1]), float(env[idx])
        frac = (thr - y0) / (y1 - y0) if y1 > y0 else 0.0
        idx = idx - 1 + max(0.0, min(1.0, frac))
    return float(idx)


def decay_ms(y, hop=64):
    """从峰值到 -20dB（相对峰值）的帧数换算成毫秒。"""
    env = rms_env(y, hop)
    if env.size == 0 or env.max() <= 0:
        return None
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    thr = peak * 10 ** (-20 / 20)
    after = np.where(env[peak_i:] < thr)[0]
    frames = int(after[0]) if after.size else len(env) - peak_i
    return round(frames * hop / 24000 * 1000, 1)


def hf_ratio(y, sr=24000, cutoff=2000):
    if len(y) < 512:
        return None
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    hi = spec[freqs >= cutoff]
    total = spec.sum() + 1e-9
    return float(hi.sum() / total)


def spectral_flux(y, hop=128):
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=hop))
    d = np.diff(spec, axis=1)
    d[d < 0] = 0
    return np.sqrt((d ** 2).sum(axis=0))


def logmel(y, sr=24000):
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    return np.log(m + 1e-6)


def norm_rms(y, target=0.1):
    r = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return y * (target / max(r, 1e-6))


def corr(a, b):
    n = min(len(a), len(b))
    if n < 8:
        return None
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load(p):
    y, _ = librosa.load(str(p), sr=24000, mono=True)
    return y


def compare(gt_path, gen_path):
    y, g = load(gt_path), load(gen_path)
    yn, gn = norm_rms(y), norm_rms(g)
    n = min(len(y), len(g))
    a, b = logmel(yn[:n]), logmel(gn[:n])
    m = min(a.shape[1], b.shape[1])
    mel = float(np.mean(np.abs(a[:, :m] - b[:, :m])))

    o1, o2 = first_onset(y), first_onset(g)
    onset_err = None
    if o1 is not None and o2 is not None:
        # first_onset 现在返回样本位置（浮点），直接换算 ms
        onset_err = round(abs(o1 - o2) / 24000 * 1000, 2)

    f1, f2 = spectral_flux(y), spectral_flux(g)
    return {
        "dur_ratio": round(len(g) / max(len(y), 1), 3),
        "onset_err_ms": onset_err,
        "env_corr": round(v, 3) if (v := corr(rms_env(y), rms_env(g))) is not None else None,
        "decay_gt_ms": decay_ms(y),
        "decay_gen_ms": decay_ms(g),
        "hf_ratio_gt": round(v, 3) if (v := hf_ratio(y)) is not None else None,
        "hf_ratio_gen": round(v, 3) if (v := hf_ratio(g)) is not None else None,
        "flux_corr": round(v, 3) if (v := corr(f1, f2)) is not None else None,
        "mel_dist": round(mel, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    gts = sorted(d.glob("*_gt.wav"))
    rows = []
    for gt in gts:
        gen = gt.with_name(gt.name.replace("_gt.wav", "_gen.wav"))
        if not gen.exists():
            continue
        rec = {
            "name": gt.stem.replace("_gt", ""),
            # 显式记录配对文件名，保证 report 行可 join 回磁盘 wav（配合 eval_pack_index.py）
            "gen_file": gen.name,
            "gt_file": gt.name,
        }
        rec.update(compare(gt, gen))
        rows.append(rec)

    if not rows:
        print("no pairs found in", d)
        return

    def agg(k):
        vs = [r[k] for r in rows if r[k] is not None]
        return None if not vs else (round(float(np.mean(vs)), 3), round(float(np.median(vs)), 3))

    keys = ["dur_ratio", "onset_err_ms", "env_corr", "decay_gt_ms", "decay_gen_ms",
            "hf_ratio_gt", "hf_ratio_gen", "flux_corr", "mel_dist"]  # gen/gt_file 不聚合
    print(f"{'metric':<16}{'mean':>10}{'median':>10}")
    for k in keys:
        v = agg(k)
        if v:
            print(f"{k:<16}{v[0]:>10}{v[1]:>10}")
    print(f"\n{'name':<28}{'dur':>7}{'onset':>8}{'env':>8}{'decayG':>8}{'hfG':>7}{'flux':>7}{'mel':>7}")
    for r in rows:
        print(f"{r['name']:<28}{r['dur_ratio']:>7}{str(r['onset_err_ms']):>8}"
              f"{str(r['env_corr']):>8}{str(r['decay_gen_ms']):>8}"
              f"{str(r['hf_ratio_gen']):>7}{str(r['flux_corr']):>7}{r['mel_dist']:>7}")

    out = args.json or str(d / "eval_report.json")
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n->", out)


if __name__ == "__main__":
    main()
