#!/usr/bin/env python3
"""C1: EnCodec 24kHz/24kbps 同批往返对比（orig / L16 / L32），带 padding_mask 截断自证。

背景（必须遵守，2026-10-26 群内裁决）：
  transformers 5.16.1 的 `EncodecModel.decode(audio_codes, audio_scales, padding_mask=None)`
  会把 `padding_mask.shape[-1]` 当作**采样点**上限来截断波形。历史脚本误传**帧级**
  mask（长度 T，例如 71），decode 输出就被切成 71 个采样点（2.96ms），此后所有
  往返 mel / 时长指标全部作废。正确做法：decode 时 `padding_mask=None`，
  只传 `audio_scales=[None]`（不做 scale 反归一化），输出长度 = T*320 采样点，
  再按原始长度裁剪。

本脚本做三件事：
  1) 自证 mask 修复：同一 wav 分别用「帧级 mask（旧 bug 路径）」与「None（修复路径）」
     解码，逐条打印样本数；修复路径必须满足 len(decoded_raw) == T*320 且裁剪后
     逐条 == 原始样本数（不接受均值）。
  2) 同批三列：orig / L16（前 16 层码本，等效 12kbps）/ L32（全 32 层，24kbps），
     同一模型版本、同一 log-mel 实现、同一次运行。
  3) 分 family 汇总：hihats、cymbal 单独成行；n<5 只出区间不判通过。

用法（在训练服务器 /gemini/code/lightfour-train 下）：
  python scripts/codec_c1.py --input-csv data/eval_refaudio_testset.csv \
      --out-dir outputs/c1_encodec --dump-tokens 5

产物：report.csv / report.json / summary.md / manifest.json / selfcheck.json
      pack/{id}_{orig,L16,L32}.wav  （ASCII 文件名，可直接做 A/B）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, EncodecModel

SR = 24000
BANDWIDTH = 24.0
HOP = 320          # 24kHz / 75Hz
N_FFT = 1024
MEL_HOP = 256
N_MELS = 128
HF_CUTOFF = 2000
LAYERS = (16, 32)
MIN_N_FAMILY = 5   # n<5 只出区间


# ---------------------------------------------------------------- audio utils
def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(y, dtype=np.float32)


def norm_rms(y: np.ndarray, target: float = 0.1) -> np.ndarray:
    r = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return y * (target / max(r, 1e-6))


def logmel(y: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP, n_mels=N_MELS)
    return np.log(m + 1e-6)


def mel_dist(a: np.ndarray, b: np.ndarray) -> float:
    """RMS 归一化后的 log-mel 平均绝对差（帧数取小者）。"""
    n = min(len(a), len(b))
    x, y = logmel(norm_rms(a[:n])), logmel(norm_rms(b[:n]))
    m = min(x.shape[1], y.shape[1])
    return float(np.mean(np.abs(x[:, :m] - y[:, :m])))


def hf_ratio(y: np.ndarray) -> float:
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    return float(spec[freqs >= HF_CUTOFF].sum() / (spec.sum() + 1e-9))


def decay_ms(y: np.ndarray, hop: int = 64) -> float | None:
    env = librosa.feature.rms(y=y, frame_length=256, hop_length=hop)[0]
    if env.size == 0 or env.max() <= 0:
        return None
    i = int(np.argmax(env))
    thr = float(env[i]) * 10 ** (-20 / 20)
    after = np.where(env[i:] < thr)[0]
    return round(int(after[0]) * hop / SR * 1000, 1) if after.size else None


def spectral_flux(y: np.ndarray) -> float:
    s = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    d = np.diff(s, axis=1)
    return float(np.mean(np.sqrt((d ** 2).sum(axis=0)))) if d.size else 0.0


# ---------------------------------------------------------------- model utils
def encode(y: np.ndarray, model, proc, device: str) -> torch.Tensor:
    """返回 codes (1, 1, n_q, T)。padding_mask 用**采样点级**（与 input_values 同长）。"""
    inp = proc(raw_audio=y, sampling_rate=SR, return_tensors="pt")
    inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.no_grad():
        enc = model.encode(inp["input_values"], inp["padding_mask"], bandwidth=BANDWIDTH)
    codes = enc.audio_codes
    if isinstance(codes, (list, tuple)):
        codes = codes[0]
    return codes


def decode(codes: torch.Tensor, model, device: str, padding_mask=None) -> np.ndarray:
    """codes (1,1,n_q,T) -> mono float32。padding_mask 必须是 None 或**采样点级**。"""
    codes = codes.to(device)
    with torch.no_grad():
        out = model.decode(codes, [None], padding_mask=padding_mask)
    w = out.audio_values if hasattr(out, "audio_values") else out[0]
    return w[0].mean(dim=0).detach().cpu().numpy().astype("float32")


def script_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- input rows
FAMILY_ALIAS = {"hi-hats": "hihats", "hi_hats": "hihats", "cymbals": "cymbal",
                "lip_oscillations": "lip_oscillation", "pops": "pops"}


def norm_family(name: str | None) -> str:
    if not name:
        return "unknown"
    key = name.strip().lower().replace(" ", "_")
    return FAMILY_ALIAS.get(key, key)


def rows_from_csv(path: str, prefix: str) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out.append({
            "ref_id": r["ref_id"],
            "audio": prefix + (r.get("audio") or "").replace("\\", "/"),
            "sound_id": r.get("sound_id"),
            "notation": r.get("sound_notation"),
            "family": norm_family(r.get("sound_family")),
            "split": r.get("split"),
            "label_source": r.get("label_source"),
        })
    return out


def rows_from_jsonl(path: str, per_family: int, families: list[str],
                    wav_root: str) -> list[dict]:
    """真实域面板：按 family 等距抽（确定性，不用 rng），wav 重定位到持久盘。"""
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    picked: list[dict] = []
    missing = 0
    for fam in families:
        sub = [r for r in raw if norm_family(r.get("sound_family")) == fam]
        sub = [r for r in sub
               if (Path(wav_root) / str(r.get("source")) / Path(str(r["wav"])).name).exists()]
        missing += sum(1 for r in raw if norm_family(r.get("sound_family")) == fam) - len(sub)
        sub.sort(key=lambda r: str(r.get("id")))
        if not sub:
            continue
        if len(sub) <= per_family:
            sel = sub
        else:
            idx = np.linspace(0, len(sub) - 1, per_family).round().astype(int)
            sel = [sub[i] for i in sorted(set(idx.tolist()))]
        for r in sel:
            wav = Path(wav_root) / str(r.get("source")) / Path(str(r["wav"])).name
            picked.append({
                "ref_id": str(Path(str(r["wav"])).stem),
                "audio": str(wav),
                "sound_id": None,
                "notation": r.get("pseudo_label"),
                "family": fam,
                "split": r.get("batch") or r.get("label_source") or "unknown",
                "label_source": r.get("label_source"),
            })
    if missing:
        print(f"[warn] {missing} 条 jsonl 记录的 wav 不在持久盘，已在抽样前剔除", flush=True)
    return picked


# ---------------------------------------------------------------- stats
def describe(vals: list[float]) -> dict:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    d = {"n": int(v.size), "median": round(float(np.median(v)), 4),
         "mean": round(float(np.mean(v)), 4)}
    if v.size < MIN_N_FAMILY:
        d.update({"min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
                  "verdict_allowed": False})
    else:
        d.update({"q25": round(float(np.quantile(v, 0.25)), 4),
                  "q75": round(float(np.quantile(v, 0.75)), 4),
                  "verdict_allowed": True})
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default="data/eval_refaudio_testset.csv")
    ap.add_argument("--audio-prefix", default="", help="csv 内相对路径前缀，默认按当前目录解析")
    ap.add_argument("--input-jsonl", default="", help="真实域面板：hybrid_labels jsonl")
    ap.add_argument("--wav-root", default="data/real_bbox/segments_batch3")
    ap.add_argument("--per-family", type=int, default=12)
    ap.add_argument("--families", default="hihats,cymbal,snares,clicks,kicks,drag,liprolls,zippers")
    ap.add_argument("--out-dir", default="outputs/c1_encodec")
    ap.add_argument("--model", default="models/encodec_24khz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-tokens", type=int, default=0, help="前 N 条导出 token .npz 供形状校验")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.input_jsonl:
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
        rows = rows_from_jsonl(args.input_jsonl, args.per_family, fams, args.wav_root)
        src_label = args.input_jsonl
    else:
        rows = rows_from_csv(args.input_csv, args.audio_prefix)
        src_label = args.input_csv
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("空输入")

    out = Path(args.out_dir)
    pack = out / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    tok_dir = out / "tokens"
    if args.dump_tokens:
        tok_dir.mkdir(parents=True, exist_ok=True)

    model = EncodecModel.from_pretrained(args.model).to(device).eval()
    model.config.target_bandwidths = [BANDWIDTH]
    proc = AutoProcessor.from_pretrained(args.model)

    n_q_total = int(model.quantizer.num_quantizers)
    print(f"[env] device={device} transformers={__import__('transformers').__version__} "
          f"n_q={n_q_total} band={BANDWIDTH}kbps frame_rate={SR / HOP:.0f}Hz "
          f"n={len(rows)}", flush=True)

    report: list[dict] = []
    selfcheck: dict = {"bug_demo": None, "per_file": [], "asserts": {}}

    for idx, r in enumerate(rows, 1):
        ref_id = r["ref_id"]
        path = Path(r["audio"])
        if not path.exists():
            print(f"[MISS] {ref_id} {path}", flush=True)
            continue
        y = load_audio(path)
        n_orig = len(y)
        codes = encode(y, model, proc, device)
        if codes.ndim != 4 or codes.shape[2] != n_q_total:
            raise SystemExit(f"codes shape 意外: {tuple(codes.shape)}")
        T = int(codes.shape[-1])

        # --- mask 修复自证（仅第一条记录现场，含旧 bug 路径） ---
        if selfcheck["bug_demo"] is None:
            frame_mask = torch.ones((1, T), dtype=torch.long, device=device)
            buggy = decode(codes, model, device, padding_mask=frame_mask)
            fixed_raw = decode(codes, model, device, padding_mask=None)
            selfcheck["bug_demo"] = {
                "ref_id": ref_id,
                "n_orig": n_orig, "T": T, "expected_T_x_320": T * HOP,
                "buggy_frame_mask_len": int(frame_mask.shape[-1]),
                "buggy_decoded_samples": int(len(buggy)),
                "fixed_decoded_samples": int(len(fixed_raw)),
                "note": "帧级 mask 被当成采样点上限 -> 输出退化成 T 个采样点；None 才是正确路径",
            }
            print(f"[mask-demo] T={T} buggy_len={len(buggy)} fixed_len={len(fixed_raw)} "
                  f"expected={T * HOP}", flush=True)

        rec = {
            "ref_id": ref_id,
            "sound_id": r.get("sound_id"),
            "notation": r.get("notation"),
            "family": r.get("family"),
            "split": r.get("split"),
            "label_source": r.get("label_source"),
            "dur_s": round(n_orig / SR, 4),
            "n_orig": n_orig,
            "T": T,
            "hf_gt": round(hf_ratio(y), 4),
            "decay_gt_ms": decay_ms(y),
            "flux_gt": round(spectral_flux(y), 4),
        }

        for layers in LAYERS:
            c = codes[:, :, :layers, :].contiguous()
            w_raw = decode(c, model, device, padding_mask=None)
            w = w_raw[:n_orig]                       # 唯一允许的长度对齐动作
            rec[f"raw_samples_L{layers}"] = int(len(w_raw))
            rec[f"n_L{layers}"] = int(len(w))
            rec[f"dur_ratio_raw_L{layers}"] = round(len(w_raw) / n_orig, 5)
            rec[f"mel_L{layers}"] = round(mel_dist(y, w), 4)
            rec[f"hf_L{layers}"] = round(hf_ratio(w), 4)
            rec[f"hf_absdiff_L{layers}"] = round(abs(hf_ratio(w) - hf_ratio(y)), 4)
            rec[f"decay_L{layers}"] = decay_ms(w)
            d = decay_ms(w)
            g = rec["decay_gt_ms"]
            rec[f"decay_absdiff_L{layers}"] = (None if (d is None or g is None)
                                               else round(abs(d - g), 1))
            rec[f"flux_L{layers}"] = round(spectral_flux(w), 4)
            sf.write(str(pack / f"{ref_id}_L{layers}.wav"), w, SR)
        sf.write(str(pack / f"{ref_id}_orig.wav"), y, SR)

        # 长度自证：逐条、精确
        ok_raw = all(rec[f"raw_samples_L{k}"] == T * HOP for k in LAYERS)
        ok_trim = all(rec[f"n_L{k}"] == n_orig for k in LAYERS)
        rec["len_check"] = bool(ok_raw and ok_trim)
        selfcheck["per_file"].append({"ref_id": ref_id, "n_orig": n_orig, "T": T,
                                      "raw_L16": rec["raw_samples_L16"],
                                      "raw_L32": rec["raw_samples_L32"],
                                      "trim_L16": rec["n_L16"], "trim_L32": rec["n_L32"],
                                      "ok": rec["len_check"]})

        if idx <= args.dump_tokens:
            np.savez_compressed(tok_dir / f"{ref_id}.npz",
                                codes=codes[0, 0].cpu().numpy().astype("int16"),  # (n_q,T)
                                dur=np.float32(n_orig / SR))
        report.append(rec)
        if idx % 10 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] {ref_id} T={T} "
                  f"mel16={rec['mel_L16']} mel32={rec['mel_L32']}", flush=True)

    # ---- token 形状校验（round-trip .npz）
    tok_check = {"dumped": 0, "ok": True, "shape": None, "note": ""}
    if args.dump_tokens:
        files = sorted(tok_dir.glob("*.npz"))
        tok_check["dumped"] = len(files)
        for f in files:
            z = np.load(f)
            c = z["codes"]
            if c.shape[0] != n_q_total:
                tok_check["ok"] = False
                tok_check["note"] = f"{f.name} shape={c.shape}"
                break
            if int(c.min()) < 0 or int(c.max()) > 1023:
                tok_check["ok"] = False
                tok_check["note"] = f"{f.name} code out of range"
                break
        tok_check["shape"] = list(np.load(files[0])["codes"].shape) if files else None
        tok_check["savez_bytes_per_file"] = int(np.mean([f.stat().st_size for f in files])) if files else 0

    # ---- 汇总
    fams = sorted({r["family"] for r in report})
    per_family = {}
    for fam in fams:
        sub = [r for r in report if r["family"] == fam]
        per_family[fam] = {
            "n": len(sub),
            "mel_L16": describe([r["mel_L16"] for r in sub]),
            "mel_L32": describe([r["mel_L32"] for r in sub]),
            "hf_absdiff_L16": describe([r["hf_absdiff_L16"] for r in sub]),
            "hf_absdiff_L32": describe([r["hf_absdiff_L32"] for r in sub]),
        }
        m16, m32 = per_family[fam]["mel_L16"].get("median"), per_family[fam]["mel_L32"].get("median")
        if m16 and m32:
            per_family[fam]["delta_pct"] = round(100 * (m16 - m32) / m16, 1)

    all16 = describe([r["mel_L16"] for r in report])
    all32 = describe([r["mel_L32"] for r in report])
    overall = {
        "n": len(report),
        "mel_L16": all16, "mel_L32": all32,
        "delta_pct": round(100 * (all16["median"] - all32["median"]) / all16["median"], 1)
        if all16.get("median") and all32.get("median") else None,
        "mean_delta_pct": round(100 * (all16["mean"] - all32["mean"]) / all16["mean"], 1)
        if all16.get("mean") and all32.get("mean") else None,
        "dur_ratio_raw_median": round(float(np.median([r["dur_ratio_raw_L32"] for r in report])), 5),
        "dur_ratio_trim": 1.0,
        "len_check_all_ok": all(r["len_check"] for r in report),
        "bits_per_s_L16": 16 * 10 * 75,
        "bits_per_s_L32": 32 * 10 * 75,
    }

    meta = {
        "script_sha256_16": script_sha(),
        "model_dir": args.model,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "sr": SR, "hop": HOP, "frame_rate_hz": SR / HOP,
        "bandwidth_kbps": BANDWIDTH, "n_q_total": n_q_total,
        "codebook_size": int(model.quantizer.codebook_size),
        "mel": {"n_fft": N_FFT, "hop": MEL_HOP, "n_mels": N_MELS, "norm": "rms->0.1"},
        "decode_padding_mask": None,
        "input_csv": src_label,
        "input_sha256_16": hashlib.sha256(Path(src_label).read_bytes()).hexdigest()[:16],
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(
        {"overall": overall, "per_family": per_family, "meta": meta, "tokens": tok_check},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "selfcheck.json").write_text(json.dumps(selfcheck, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(
        {"meta": meta, "files": [{"ref_id": r["ref_id"], "sound_id": r["sound_id"],
                                  "family": r["family"], "split": r["split"],
                                  "T": r["T"], "n_orig": r["n_orig"]} for r in report]},
        ensure_ascii=False, indent=1), encoding="utf-8")

    cols = ["ref_id", "sound_id", "family", "notation", "dur_s", "T", "n_orig",
            "mel_L16", "mel_L32", "hf_gt", "hf_L16", "hf_L32",
            "hf_absdiff_L16", "hf_absdiff_L32", "decay_gt_ms", "decay_L16", "decay_L32",
            "dur_ratio_raw_L16", "dur_ratio_raw_L32", "len_check"]
    with open(out / "report.csv", "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wcsv.writeheader()
        for r in report:
            wcsv.writerow(r)

    # ---- markdown 摘要
    lines = []
    lines.append("# C1 — EnCodec 24kHz/24kbps 同批往返（orig / L16 / L32）\n")
    lines.append(f"- 脚本 sha256[:16] `{meta['script_sha256_16']}`，transformers {meta['transformers']}，"
                 f"torch {meta['torch']}，n_q={n_q_total}，帧率 {SR/HOP:.0f}Hz，码本 "
                 f"{meta['codebook_size']}")
    lines.append(f"- 输入 `{src_label}`（sha256[:16] `{meta['input_sha256_16']}`），n={len(report)}")
    lines.append(f"- decode 传 `padding_mask=None`；长度对齐只用尾部裁剪；"
                 f"逐条样本数校验 {'全部通过' if overall['len_check_all_ok'] else '存在失败'}\n")
    lines.append("## 总表\n")
    lines.append("| 指标 | L16(12kbps,16层) | L32(24kbps,32层) | 32层相对16层 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| log-mel L1 中位 | {all16['median']} | {all32['median']} | {overall['delta_pct']:+.1f}% |")
    lines.append(f"| log-mel L1 均值 | {all16['mean']} | {all32['mean']} | {overall['mean_delta_pct']:+.1f}% |")
    lines.append(f"| 解码原始/原长 | {np.median([r['dur_ratio_raw_L16'] for r in report]):.5f} "
                 f"| {np.median([r['dur_ratio_raw_L32'] for r in report]):.5f} | 裁剪后 1.00000 |")
    lines.append("\n## 分 family\n")
    lines.append("| family | n | mel_L16 中位 | mel_L32 中位 | Δ% | hf|Δ| L16 | hf|Δ| L32 | 结论口径 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for fam in fams:
        f_ = per_family[fam]
        gate = "可判" if f_["n"] >= MIN_N_FAMILY else "n<5 仅区间"
        lines.append(f"| {fam} | {f_['n']} | {f_['mel_L16'].get('median')} | "
                     f"{f_['mel_L32'].get('median')} | {f_.get('delta_pct')} | "
                     f"{f_['hf_absdiff_L16'].get('median')} | {f_['hf_absdiff_L32'].get('median')} | {gate} |")
    hi = per_family.get("hihats", {}); cy = per_family.get("cymbal", {})
    lines.append("\n## 镲类专项（必须单列）\n")
    for name, f_ in (("hihats", hi), ("cymbal", cy)):
        if not f_:
            lines.append(f"- {name}: 本批 0 条，不出结论")
            continue
        a, b = f_["mel_L16"], f_["mel_L32"]
        if a.get("verdict_allowed"):
            lines.append(f"- {name}: n={f_['n']}，mel16 中位 {a['median']}（IQR {a.get('q25')}–{a.get('q75')}），"
                         f"mel32 中位 {b['median']}（IQR {b.get('q25')}–{b.get('q75')}），Δ {f_.get('delta_pct')}%")
        else:
            lines.append(f"- {name}: n={f_['n']}，mel16 区间 [{a.get('min')}, {a.get('max')}]，"
                         f"mel32 区间 [{b.get('min')}, {b.get('max')}]，Δ {f_.get('delta_pct')}%"
                         f"（n<5 不判通过）")
    lines.append("\n## 边界\n")
    lines.append("- 本表只评 **codec 重建上限**，不含任何生成模型；gt 为有损重建，不是原始声源。")
    if args.input_jsonl:
        lines.append("- 本批为 **B 站实战切片**，标签来自伪标（AST/nearest-seed），"
                     "非人工真值；仅用于域内相对比较，不得当作识别准确率。")
    else:
        lines.append("- 本批为 seedlib 参考音（零伪标），域与 B 站实战切片不同；不得跨版本比较。")
    lines.append(f"- cymbal n={cy.get('n')}，hihats n={hi.get('n')}：镲类仍需独立人耳复核，"
                 f"本表不足以宣布 codec 过关。")
    lines.append(f"\n## token 形状与校验\n")
    lines.append(f"- codes: `(1, 1, {n_q_total}, T)` int64 → 存档 int16 `({n_q_total}, T)`；"
                 f"码本 0..{meta['codebook_size'] - 1}（校验 {'OK' if tok_check['ok'] else 'FAIL'}）")
    lines.append(f"- 75Hz × 16 层 = 1200 tok/s（12kbps）；75Hz × 32 层 = 2400 tok/s（24kbps）")
    lines.append(f"- Talker 目标 13Hz × 16 组：同段音频需 75→13Hz 重采样，T_talker ≈ T*13/75")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines), flush=True)
    print(f"\n[len-check] all_ok={overall['len_check_all_ok']} "
          f"n_fail={sum(1 for r in report if not r['len_check'])}", flush=True)
    return 0 if overall["len_check_all_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
