"""DAC 24kHz 往返对比（P2 一次性尝试，PI 裁决⑦/裁决⑥口径）。

目的：只回答一个问题——在**同样的 75Hz、同样的 16 组码本**预算下，
DAC 24kHz 的重建是否比已冻结的 EnCodec 24kHz/24kbps 16 层更好。
不回答「DAC 能不能替代」，也不改选型表结论。

三条 arm（同批音频、同一 logmel 实现、同一 RMS 对齐）：
  dac32 : DAC 24k 全 32 组
  dac16 : DAC 24k 前 16 组（与 EnCodec 16 层的组数对齐）
  enc16 : EnCodec 24kHz / 24kbps / 16 层（已冻结基线）

依赖说明（重要）：
  镜像源缺 `ffmpy` / `randomname`，`audiotools` 装不全，导致 `import dac` 失败。
  这里用 stub 注入 `audiotools` 与 `audiotools.ml.BaseModel`；已核 DAC 建模路径
  （dac/model/dac.py）只在 `__main__` 演示块里用 `AudioSignal`，`BaseModel` 仅作基类。
  → 不影响 encode/decode 数值。这是「一次性尝试」的合法降级，不写进任何生产路径。

用法：
  python scripts/dac_roundtrip.py --self-test
  python scripts/dac_roundtrip.py --input-csv data/eval_refaudio_testset.csv \
      --audio-prefix data/ --out-dir outputs/dac_rt
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec_c1 import (  # noqa: E402  同一实现，禁止另写 logmel
    HOP, SR, describe, hf_ratio, load_audio, mel_dist, norm_family,
    norm_rms, rows_from_csv,
)

DAC_DIR = Path("/gemini/code/lightfour-train/models/dac_24khz")
DAC_CFG = dict(encoder_dim=64, encoder_rates=[2, 4, 5, 8], latent_dim=1024,
               decoder_dim=1536, decoder_rates=[8, 5, 4, 2],
               n_codebooks=32, codebook_size=1024, codebook_dim=8, sample_rate=24000)


# ---------------------------------------------------------------- import shim
def install_audiotools_stub() -> None:
    """让 `import dac` 在缺 audiotools 全部依赖时仍可用。

    必须在 `import dac` 之前调用；只补两个符号，其余属性访问会 AttributeError，
    避免悄悄吞掉真实依赖缺失。
    """
    if "audiotools" in sys.modules:
        return
    at = types.ModuleType("audiotools")

    class AudioSignal:  # 仅用于 dac/model/dac.py 的 __main__ 演示块
        def __init__(self, *a, **k):
            raise RuntimeError("audiotools stub: AudioSignal 不可用")

    at.AudioSignal = AudioSignal

    class STFTParams:  # dac/nn/loss.py 只在 MultiScaleSTFTLoss 里用，本脚本不调用
        def __init__(self, window_type="hann", window_length=0.032,
                     hop_length=0.008, n_fft=0, **kw):
            self.window_type = window_type
            self.window_length = window_length
            self.hop_length = hop_length
            self.n_fft = n_fft

    at.STFTParams = STFTParams

    ml = types.ModuleType("audiotools.ml")

    class BaseModel(torch.nn.Module):
        # dac/__init__.py 依赖这个类属性做参数注册，真实 audiotools 里是 list
        INTERN: list = []
        EXTERN: list = []
        EXTERN_BUFFER: list = []
        INTERN_BUFFER: list = []

    ml.BaseModel = BaseModel

    class Accelerator:  # dac/utils/__init__.py 只做名字绑定，训练路径才实例化
        def __init__(self, *a, **k):
            raise RuntimeError("audiotools stub: Accelerator 不可用")

    ml.Accelerator = Accelerator
    at.ml = ml
    sys.modules["audiotools"] = at
    sys.modules["audiotools.ml"] = ml


# ---------------------------------------------------------------- model
def load_dac(device: str):
    install_audiotools_stub()
    from dac.model.dac import DAC
    from safetensors.torch import load_file

    model = DAC(**DAC_CFG)
    sd = load_file(str(DAC_DIR / "model.safetensors"))
    msg = model.load_state_dict(sd, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise SystemExit(f"FAIL state_dict: missing={msg.missing_keys[:5]} "
                         f"unexpected={msg.unexpected_keys[:5]} "
                         f"(n={len(msg.missing_keys)}/{len(msg.unexpected_keys)})")
    model.eval().to(device)
    # 结构性断言：帧率必须与 EnCodec 24k 相同，否则「省序列」的结论不成立
    assert model.hop_length == HOP, f"hop {model.hop_length} != {HOP}"
    assert model.sample_rate == SR
    assert model.n_codebooks == 32 and model.codebook_size == 1024
    return model


def dac_pad(y: np.ndarray) -> np.ndarray:
    """preprocess 口径：右 pad 到 hop 的整数倍（与 model.preprocess 一致）。"""
    right = math.ceil(len(y) / HOP) * HOP - len(y)
    return np.pad(y, (0, right)) if right else y


@torch.no_grad()
def dac_roundtrip(model, y: np.ndarray, n_q: int, device: str) -> np.ndarray:
    """y(24k mono) -> codes(1,n_q,T) -> z -> wav。返回裁剪到原始长度的波形。"""
    n = len(y)
    x = torch.from_numpy(dac_pad(y)).float()[None, None].to(device)
    _, codes, _, _, _ = model.encode(x, n_quantizers=n_q)
    z, _, _ = model.quantizer.from_codes(codes)
    w = model.decode(z)
    w = w[0, 0].detach().cpu().numpy().astype("float32")
    return w[:n], int(codes.shape[-1])


@torch.no_grad()
def enc_roundtrip(y: np.ndarray, model, proc, device: str, n_q: int) -> np.ndarray:
    """EnCodec 24k/24kbps 前 n_q 层。padding_mask 用采样点级（C1 修复后口径）。"""
    n = len(y)
    inp = proc(raw_audio=y, sampling_rate=SR, return_tensors="pt")
    inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    enc = model.encode(inp["input_values"], inp["padding_mask"], bandwidth=24.0)
    codes = enc.audio_codes
    codes = codes[0] if isinstance(codes, (list, tuple)) else codes
    out = model.decode(codes[:, :, :n_q], [None], padding_mask=None)
    w = out.audio_values if hasattr(out, "audio_values") else out[0]
    w = w[0].mean(dim=0).detach().cpu().numpy().astype("float32")
    return w[:n]


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- selftest
def self_test() -> int:
    install_audiotools_stub()
    from dac.model.dac import DAC
    m = DAC(**DAC_CFG)
    print("hop", m.hop_length, "sr", m.sample_rate, "n_q", m.n_codebooks,
          "cb", m.codebook_size, "params",
          round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M")
    ok = (m.hop_length == HOP == 320 and m.n_codebooks == 32 and m.codebook_size == 1024)
    # 随机权重下的形状/长度闭环（不判断音质）
    y = np.random.RandomState(0).randn(24000).astype("float32") * 0.1
    w, t = dac_roundtrip(m, y, 16, "cpu")
    print("frames", t, "expected", math.ceil(len(y) / HOP), "wav", len(w))
    ok = ok and t == math.ceil(len(y) / HOP) and abs(len(w) - len(y)) <= 1
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default="data/eval_refaudio_testset.csv")
    ap.add_argument("--audio-prefix", default="data/")
    ap.add_argument("--out-dir", default="outputs/dac_rt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="dac32,dac16,enc16")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = rows_from_csv(args.input_csv, args.audio_prefix)
    rows = [r for r in rows if (Path(r["audio"]).exists())]
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows {len(rows)} arms {arms} device {args.device}", flush=True)
    if not rows:
        raise SystemExit("FAIL: 没有可用音频")

    dac = load_dac(args.device) if any(a.startswith("dac") for a in arms) else None
    enc = proc = None
    if "enc16" in arms:
        from transformers import AutoProcessor, EncodecModel
        enc = EncodecModel.from_pretrained(
            "/gemini/code/lightfour-train/models/encodec_24khz").to(args.device).eval()
        proc = AutoProcessor.from_pretrained(
            "/gemini/code/lightfour-train/models/encodec_24khz")

    recs = []
    for i, r in enumerate(rows, 1):
        y = load_audio(Path(r["audio"]))
        rec = {"ref_id": r["ref_id"], "family": r["family"], "notation": r["notation"],
               "dur": round(len(y) / SR, 4)}
        if "dac32" in arms:
            w, t = dac_roundtrip(dac, y, 32, args.device)
            rec.update(dac32_mel=round(mel_dist(y, w), 4),
                       dac32_drift=round((len(w) / len(y) - 1) * 100, 4),
                       dac32_hf=round(hf_ratio(w), 4), dac32_T=t)
        if "dac16" in arms:
            w, t = dac_roundtrip(dac, y, 16, args.device)
            rec.update(dac16_mel=round(mel_dist(y, w), 4),
                       dac16_drift=round((len(w) / len(y) - 1) * 100, 4),
                       dac16_hf=round(hf_ratio(w), 4), dac16_T=t)
        if "enc16" in arms:
            w = enc_roundtrip(y, enc, proc, args.device, 16)
            rec.update(enc16_mel=round(mel_dist(y, w), 4),
                       enc16_drift=round((len(w) / len(y) - 1) * 100, 4),
                       enc16_hf=round(hf_ratio(w), 4))
        rec["true_hf"] = round(hf_ratio(y), 4)
        recs.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["ref_id", "family", "notation", "dur", "true_hf"] + [
        f"{a}_{k}" for a in arms for k in ("mel", "drift", "hf")]
    with open(out / "report.csv", "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wcsv.writeheader()
        for rec in recs:
            wcsv.writerow(rec)

    summary = {"input_csv": args.input_csv, "n": len(recs), "arms": arms,
               "script": sha16(Path(__file__)), "dac_config": DAC_CFG,
               "hop": HOP, "sr": SR, "overall": {}, "by_family": {}}
    for a in arms:
        summary["overall"][a] = {
            "mel": describe([r[f"{a}_mel"] for r in recs]),
            "drift_pct": describe([abs(r[f"{a}_drift"]) for r in recs]),
            "hf": describe([r[f"{a}_hf"] for r in recs]),
            "tokens_per_s": 75 * (16 if a == "enc16" or a == "dac16" else 32),
        }
    fams = sorted({r["family"] for r in recs})
    for fam in fams:
        sub = [r for r in recs if r["family"] == fam]
        summary["by_family"][fam] = {
            a: describe([r[f"{a}_mel"] for r in sub]) for a in arms}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(
        {"script": sha16(Path(__file__)), "input_csv": sha16(Path(args.input_csv)),
         "n": len(recs), "arms": arms, "audio_unmodified": True,
         "note": "mel 用 codec_c1.logmel（与 C1 卡同实现）；禁用 mel_metric.py 口径并表"},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# DAC 24k 往返（n={len(recs)}，{args.input_csv}）", "",
             "mel 为 codec_c1.logmel 口径，跨实现不可比。", "",
             "| arm | n | mel 中位 | mel 均值 | 时长漂移中位% | HF 中位 | tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        o = summary["overall"][a]
        lines.append(f"| {a} | {o['mel']['n']} | {o['mel']['median']} | {o['mel']['mean']} | "
                     f"{o['drift_pct']['median']} | {o['hf']['median']} | {o['tokens_per_s']} |")
    lines += ["", "## 按 family（mel 中位）", "",
              "| family | n | " + " | ".join(arms) + " |", "|---|---:|" + "---:|" * len(arms)]
    for fam in fams:
        d = summary["by_family"][fam]
        n = d[arms[0]]["n"]
        cells = " | ".join(str(d[a].get("median", "-")) for a in arms)
        lines.append(f"| {fam} | {n} | {cells} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
