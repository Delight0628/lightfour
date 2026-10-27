"""隔离 codec 天花板：同一批最差家族，16 层 vs 32 层 EnCodec 重建 A/B。

目的：判断「小模型学不动高频镲类」是 (a) codec 层数不够，还是 (b) codec 空间本身装不下。
  32 层明显好于 16 层 -> 层数问题，加层数/扩模型有意义
  32 层也和 16 层一样差 -> codec 空间问题，换模型无用，应转 DAC 或原生 codec

同时重测 codec_tokens.py 修复后的往返 mel（旧 0.661 因 decode 截断无效）。

用法：
  python scripts/codec_layers_ab.py --families 4 --per-family 3
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, EncodecModel

BASE = Path("/gemini/code/lightfour-train")
SRC = BASE / "data/real_bbox/hybrid_labels_all_highconf_l2.jsonl"
SEG_ROOT = BASE / "data/real_bbox/segments_batch3"
OUT = Path("/tmp/codec_layers_ab")
SR = 24000


def relocate(p):
    """jsonl 里的 wav 分散在 /tmp 下的三个目录，实例重启即丢；
    持久盘是合并后的 segments_batch3，按尾部两段（BVxxx/xxx.wav）重定位。"""
    tail = Path(p).parts[-2:]
    return SEG_ROOT / tail[0] / tail[1]


def norm_rms(y, target=0.1):
    r = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return y * (target / max(r, 1e-6))


def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=128)
    return np.log(m + 1e-6)


def hf_ratio(y, cutoff=2000):
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    return float(spec[freqs >= cutoff].sum() / (spec.sum() + 1e-9))


def decay_ms(y, hop=64):
    env = librosa.feature.rms(y=y, frame_length=256, hop_length=hop)[0]
    if env.size == 0 or env.max() <= 0:
        return None
    i = int(np.argmax(env))
    thr = float(env[i]) * 10 ** (-20 / 20)
    after = np.where(env[i:] < thr)[0]
    return round(int(after[0]) * hop / SR * 1000, 1) if after.size else None


def mel_dist(a, b):
    n = min(len(a), len(b))
    if n < 1024:
        return None
    x, y = logmel(norm_rms(a[:n])), logmel(norm_rms(b[:n]))
    m = min(x.shape[1], y.shape[1])
    return round(float(np.mean(np.abs(x[:, :m] - y[:, :m]))), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=4,
                    help="取指标最差的几个家族（按 §7.3 的高频清单）")
    ap.add_argument("--per-family", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["wav"] = str(relocate(r["wav"]))
    rows = [r for r in rows if Path(r["wav"]).exists()]
    print("可用样本", len(rows), flush=True)
    if not rows:
        raise SystemExit(f"找不到可用 wav，检查 {SEG_ROOT}")

    # 小模型最差的一批：摩擦音/click/whistle/drag，按 family 名匹配
    targets = ["Clicks", "Liprolls", "Lip oscillations", "Drag", "Cymbal", "Hi-hats"]
    picked = []
    for fam in targets:
        sub = [r for r in rows if r.get("sound_family") == fam]
        sub.sort(key=lambda r: -r.get("confidence", 0))
        picked.extend(sub[: args.per_family])
        if len(picked) >= args.families * args.per_family:
            break
    picked = picked[: args.families * args.per_family]
    print(f"选中 {len(picked)} 条，家族：{sorted({r.get('sound_family') for r in picked})}",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.from_pretrained(str(BASE / "models/encodec_24khz")).to(dev).eval()
    model.config.target_bandwidths = [24.0]
    proc = AutoProcessor.from_pretrained(str(BASE / "models/encodec_24khz"))
    OUT.mkdir(parents=True, exist_ok=True)

    report = []
    for i, r in enumerate(picked, 1):
        y, _ = librosa.load(r["wav"], sr=SR, mono=True)
        inp = proc(raw_audio=y, sampling_rate=SR, return_tensors="pt")
        inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.no_grad():
            enc = model.encode(inp["input_values"], inp["padding_mask"], bandwidth=24.0)
            codes = enc.audio_codes
            rec = {"id": r["id"], "family": r.get("sound_family"),
                   "notation": r.get("human_notation") or r.get("pseudo_label"),
                   "dur": round(len(y) / SR, 3),
                   "raw_layers": int(codes.shape[2]),
                   "hf_ratio_gt": round(hf_ratio(y), 3),
                   "decay_gt_ms": decay_ms(y)}
            for layers in (16, 32):
                c = codes[:, :, :layers, :].contiguous()
                w = model.decode(c, [None], None).audio_values[0].mean(dim=0)
                w = w.cpu().numpy().astype("float32")
                sf.write(str(OUT / f"{r['id']}_L{layers}.wav"), w, SR)
                rec[f"mel_L{layers}"] = mel_dist(y, w)
                rec[f"hf_L{layers}"] = round(hf_ratio(w), 3)
                rec[f"decay_L{layers}"] = decay_ms(w)
            report.append(rec)
        print(f"[{i}/{len(picked)}] {r['id']} {r.get('sound_family'):<18} "
              f"mel16={rec['mel_L16']} mel32={rec['mel_L32']} "
              f"hfG={rec['hf_ratio_gt']} hf16={rec['hf_L16']} hf32={rec['hf_L32']}",
              flush=True)

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    ok = [x for x in report if x.get("mel_L16") and x.get("mel_L32")]
    if ok:
        m16 = float(np.median([x["mel_L16"] for x in ok]))
        m32 = float(np.median([x["mel_L32"] for x in ok]))
        d16 = float(np.median([abs(x["hf_L16"] - x["hf_ratio_gt"]) for x in ok
                               if x.get("hf_L16") is not None]))
        d32 = float(np.median([abs(x["hf_L32"] - x["hf_ratio_gt"]) for x in ok
                               if x.get("hf_L32") is not None]))
        print(f"\n汇总 n={len(ok)}  mel16={m16:.3f} mel32={m32:.3f} "
              f"（32 层改善 {100*(m16-m32)/m16:+.1f}%）", flush=True)
        print(f"高频占比偏差 hf16={d16:.3f} hf32={d32:.3f}", flush=True)
        verdict = ("32 层明显更好 → 是层数/带宽问题，扩模型有意义"
                   if m32 < m16 * 0.9 else
                   "32 层没有实质改善 → codec 空间本身装不下高频，换大模型无用")
        print("判定:", verdict, flush=True)
    print("->", OUT, flush=True)


if __name__ == "__main__":
    main()
