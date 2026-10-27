"""
EnCodec 24kHz / 24kbps 音频 <-> token 工具。

用法：
  # 单文件编码（wav -> .npy，内含 codes 与 scales）
  python scripts/codec_tokens.py encode --input a.wav --out a.npy

  # 单文件解码（.npy -> wav）
  python scripts/codec_tokens.py decode --input a.npy --out a.wav

  # 目录批量编码（每个 wav 生成同名 .npy + codes_manifest.jsonl）
  python scripts/codec_tokens.py encode --input-dir data/segments --out-dir data/tokens

  # 往返验证（encode -> decode -> 指标）
  python scripts/codec_tokens.py roundtrip --input-dir data/segments --out-dir /tmp/rt

说明：
  - 模型固定为 EnCodec 24kHz、24kbps（75Hz 帧率、码本 1024）。
  - token 形状：codes (n_q, 1, T)，scales (1, 1)；.npy 用 np.savez 保存。
  - 只做研究用途，音频不入 git。
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, EncodecModel

SR = 24000
BANDWIDTH = 24.0
DEFAULT_MODEL = "/gemini/code/lightfour-train/models/encodec_24khz"


def load_model(model_dir, device):
    model = EncodecModel.from_pretrained(model_dir).to(device).eval()
    model.config.target_bandwidths = [BANDWIDTH]
    proc = AutoProcessor.from_pretrained(model_dir)
    return model, proc


def encode_wav(path, model, proc, device):
    y, _ = librosa.load(path, sr=SR, mono=True)
    inp = proc(raw_audio=y, sampling_rate=SR, return_tensors="pt")
    inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.no_grad():
        enc = model.encode(inp["input_values"], inp["padding_mask"])
    codes = enc.audio_codes.cpu().numpy()
    return codes, y


def decode_codes(codes, model, device):
    """codes: (n_q, 1, T) -> 波形。

    注意：transformers 的 EncodecModel.decode 会把 padding_mask.shape[-1] 当作
    波形采样点上限来截断音频。这里传帧级 mask 会把 13 帧切成 13 个采样点，
    所以必须传 None；audio_scales 传 [None] 表示不做归一化缩放。
    """
    codes_t = torch.from_numpy(codes).to(device)
    with torch.no_grad():
        out = model.decode(codes_t, [None], None)
    audio = out.audio_values if hasattr(out, "audio_values") else out[0]
    w = audio[0].mean(dim=0).cpu().numpy().astype("float32")
    # 回归护栏：EnCodec 24kHz / hop 320 → 每帧 320 采样点。任何把帧级 mask 传进
    # decode 的写法都会把波形截成 T 个采样点，这里直接报错而不是静默通过。
    expected = int(codes.shape[-1]) * 320
    if abs(len(w) - expected) > 320:
        raise RuntimeError(
            f"decode 输出长度异常: {len(w)} 采样点，期望 ~{expected}"
            f"（{codes.shape[-1]} 帧 × 320）。检查是否给 decode 传了帧级 padding_mask。")
    return w


def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=128)
    return np.log(m + 1e-6)


def norm_rms(y, target=0.1):
    r = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return y * (target / max(r, 1e-6))


def cmd_encode(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)
    out = Path(args.out)
    codes, y = encode_wav(args.input, model, proc, device)
    np.savez_compressed(out, codes=codes)
    print(json.dumps({"input": args.input, "out": str(out), "codes": list(codes.shape),
                      "dur": round(len(y) / SR, 3)}, ensure_ascii=False))


def cmd_decode(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.model, device)
    z = np.load(args.input)
    w = decode_codes(z["codes"], model, device)
    sf.write(args.out, w, SR)
    print(json.dumps({"input": args.input, "out": args.out,
                      "dur": round(len(w) / SR, 3)}, ensure_ascii=False))


def cmd_encode_dir(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input_dir).rglob("*.wav"))
    manifest = []
    for i, p in enumerate(files, 1):
        try:
            codes, y = encode_wav(str(p), model, proc, device)
            np.savez_compressed(out_dir / f"{p.stem}.npz", codes=codes)
            manifest.append({"wav": str(p), "token": str(out_dir / f"{p.stem}.npz"),
                             "codes_shape": list(codes.shape), "dur": round(len(y) / SR, 3)})
            if i % 50 == 0:
                print(f"{i}/{len(files)}", flush=True)
        except Exception as exc:
            print(f"[FAIL] {p}: {exc}", flush=True)
    with open(out_dir / "codes_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"encoded {len(manifest)}/{len(files)} -> {out_dir}", flush=True)


def cmd_roundtrip(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input_dir).rglob("*.wav"))[: args.limit or None]
    rows = []
    for p in files:
        codes, y = encode_wav(str(p), model, proc, device)
        w = decode_codes(codes, model, device)
        n = min(len(y), len(w))
        a, b = logmel(norm_rms(y[:n])), logmel(norm_rms(w[:n]))
        m = min(a.shape[1], b.shape[1])
        d = float(np.mean(np.abs(a[:, :m] - b[:, :m])))
        sf.write(out_dir / f"{p.stem}.wav", w, SR)
        rows.append({"id": p.stem, "dur": round(len(y) / SR, 3),
                     "codes": list(codes.shape), "mel_dist_norm": round(d, 4)})
    with open(out_dir / "roundtrip_report.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    if rows:
        print(f"n={len(rows)} mel_mean={np.mean([r['mel_dist_norm'] for r in rows]):.3f} "
              f"mel_median={np.median([r['mel_dist_norm'] for r in rows]):.3f}", flush=True)
    print("->", out_dir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("encode")
    a.add_argument("--input", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_encode)

    b = sub.add_parser("decode")
    b.add_argument("--input", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_decode)

    c = sub.add_parser("encode-dir")
    c.add_argument("--input-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=cmd_encode_dir)

    d = sub.add_parser("roundtrip")
    d.add_argument("--input-dir", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--limit", type=int, default=20)
    d.set_defaults(func=cmd_roundtrip)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
