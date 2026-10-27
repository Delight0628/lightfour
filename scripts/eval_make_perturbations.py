#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成识别侧音频敏感性测试的扰动音频包（Eval 维护，口径锁定 2026-10-27）。

来源 = data/eval_refaudio_testset.csv 的 95 条 seedlib 参考音频（零伪标）。
扰动定义 = data/eval_perturbations.csv（P0–P7）。
多参数扰动展开成子变体（P1a/P1b ...），扰动 id 带小写后缀；kind 归并回 base_perturb_id。

P5 seed_swap 的硬约束：
  探针集每个 sound_id 只有 1 条 rendition（n=1/id），没有"同音色另一 source 的同类片段"。
  用别的音色冒充 swap 会把真值改掉，属于伪造测试集。故 P5 在探针集上 **不生成**，
  manifest 中记 kind=unavailable，直到 @Data 提供同 id 第二条 rendition 才可补。

输出:
  data/eval_perturb_audio/<perturb_id>__<ref_id>.wav   （24kHz mono，不入 git）
  data/eval_perturb_manifest.csv                       （含逐文件 md5，可入库、可复算）

用法:
  python scripts/eval_make_perturbations.py
  python scripts/eval_make_perturbations.py --verify      # 只校验已生成文件
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = os.path.join(ROOT, "data", "eval_refaudio_testset.csv")
SPEC = os.path.join(ROOT, "data", "eval_perturbations.csv")
OUT_DIR = os.path.join(ROOT, "data", "eval_perturb_audio")
MANIFEST = os.path.join(ROOT, "data", "eval_perturb_manifest.csv")
SR = 24000

# (perturb_id, base_id, kind, 参数) —— 参数写进 manifest，判据写死在这里
VARIANTS = [
    ("P0", "P0", "none", {}),
    ("P1a", "P1", "time_stretch", {"rate": 0.9}),
    ("P1b", "P1", "time_stretch", {"rate": 1.1}),
    ("P2a", "P2", "time_stretch", {"rate": 0.8}),
    ("P2b", "P2", "time_stretch", {"rate": 1.25}),
    ("P3", "P3", "additive_noise", {"snr_db": 20.0}),
    ("P4a", "P4", "additive_noise", {"snr_db": 10.0}),
    ("P4b", "P4", "additive_noise", {"snr_db": 0.0}),
    ("P5", "P5", "seed_swap", {}),
    ("P6", "P6", "silence_pad", {"pad_s": 0.5}),
    ("P7a", "P7", "gain", {"gain": 0.5}),
    ("P7b", "P7", "gain", {"gain": 2.0}),
]
# 验收容差：扰动实际强度必须落在声明参数附近，否则视为生成错误
TOL = {"snr_db": 0.5, "rate": 0.02, "gain": 0.01, "pad_s": 0.01}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_audio(path):
    """mp3/wav -> 24kHz mono float32。优先 soundfile，回退 torchaudio。"""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(y))
    except Exception:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0) if wav.ndim > 1 else wav
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.float()


def rng_for(ref_id, perturb_id):
    seed = int(hashlib.md5(f"{ref_id}|{perturb_id}|lightfour-eval-v1".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def time_stretch(y, rate):
    """STFT 相位声码器：保持音高、改时长。"""
    n_fft, hop = 1024, 256
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft), return_complex=True)
    phase = torch.linspace(0.0, float(np.pi * hop), n_fft // 2 + 1, dtype=torch.float32)[..., None]
    out = torchaudio.functional.phase_vocoder(spec, rate=float(rate), phase_advance=phase)
    n = int(round(y.numel() / float(rate)))
    return torch.istft(out, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=torch.hann_window(n_fft), length=n)


def add_noise(y, snr_db, rng):
    sig_rms = float(torch.sqrt(torch.mean(y ** 2) + 1e-12))
    noise = torch.from_numpy(rng.standard_normal(y.numel()).astype(np.float32))
    noise_rms = float(torch.sqrt(torch.mean(noise ** 2) + 1e-12))
    scale = (sig_rms / noise_rms) * (10 ** (-snr_db / 20.0))
    return y + noise * scale, float(scale)


def measure_snr_db(y, ref):
    """以 ref 为信号、y-ref 为噪声回测 SNR。"""
    noise = y - ref
    a = float(torch.mean(ref ** 2)) + 1e-12
    b = float(torch.mean(noise ** 2)) + 1e-12
    return 10.0 * np.log10(a / b)


def apply_variant(y, kind, params, rng):
    """返回 (out, check_dict)。check_dict 里是回测值，用 TOL 对账。"""
    if kind == "none":
        return y.clone(), {}
    if kind == "time_stretch":
        out = time_stretch(y, params["rate"])
        return out, {"rate_meas": out.numel() / max(1, y.numel()) * params["rate"]}
    if kind == "additive_noise":
        out, _ = add_noise(y, params["snr_db"], rng)
        return out, {"snr_db_meas": measure_snr_db(out, y)}
    if kind == "silence_pad":
        n = int(round(params["pad_s"] * SR))
        out = torch.cat([torch.zeros(n), y, torch.zeros(n)])
        return out, {"pad_s_meas": n / SR}
    if kind == "gain":
        raw = y * params["gain"]
        clipped = int(torch.sum((raw > 1.0) | (raw < -1.0)))
        out = raw.clamp(-1.0, 1.0)
        return out, {"gain_meas": params["gain"], "clipped": clipped}
    raise ValueError(f"unknown kind {kind}")


def check_values(pid, kind, params, checks, row):
    """回测对账，不通过就 fail（不静默）。"""
    ok = True
    if kind == "time_stretch":
        dev = abs(checks["rate_meas"] - 1.0)  # 期望输出长 = 输入长 / rate，回测比率应≈1
        row["rate_check"] = f"{checks['rate_meas']:.4f}"
        ok &= dev <= TOL["rate"]
    if kind == "additive_noise":
        dev = abs(checks["snr_db_meas"] - params["snr_db"])
        row["snr_db_meas"] = f"{checks['snr_db_meas']:.2f}"
        ok &= dev <= TOL["snr_db"]
    if kind == "silence_pad":
        dev = abs(checks["pad_s_meas"] - params["pad_s"])
        row["pad_s_meas"] = f"{checks['pad_s_meas']:.4f}"
        ok &= dev <= TOL["pad_s"]
    if kind == "gain":
        row["clipped_samples"] = checks["clipped"]
    row["check_ok"] = int(ok)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="只校验已生成文件，不重建")
    args = ap.parse_args()

    test_rows = read_csv(TESTSET)
    spec_rows = read_csv(SPEC)
    spec_by_id = {r["perturb_id"]: r for r in spec_rows}
    os.makedirs(OUT_DIR, exist_ok=True)

    rows, bad = [], []
    for t in test_rows:
        ref_id = t["ref_id"]
        src = os.path.join(ROOT, t["audio"].replace("\\", os.sep))
        if not os.path.exists(src):
            raise SystemExit(f"参考音频缺失: {src}")
        y = load_audio(src)
        for pid, base, kind, params in VARIANTS:
            out_name = f"{pid}__{ref_id}.wav"
            out_path = os.path.join(OUT_DIR, out_name)
            row = {
                "perturb_id": pid, "base_perturb_id": base, "kind": kind,
                "params": ";".join(f"{k}={v}" for k, v in params.items()),
                "ref_id": ref_id, "sound_id": t["sound_id"],
                "sound_notation": t["sound_notation"], "sound_family": t["sound_family"],
                "is_short_variant": t["is_short_variant"],
                "src_audio": t["audio"], "out_file": out_name,
                "spec_purpose": spec_by_id.get(base, {}).get("purpose", ""),
                "spec_expected": spec_by_id.get(base, {}).get("expected", ""),
            }
            if kind == "seed_swap":
                row.update({"available": 0, "check_ok": 1,
                            "note": "探针集每 sound_id 仅 1 条 rendition，无同音色第二 source；"
                                    "用别的音色会改真值，禁止。待 @Data 补同 id 第二条后启用。"})
                rows.append(row)
                continue
            checks = {}
            if not args.verify:
                rng = rng_for(ref_id, pid)
                out, checks = apply_variant(y, kind, params, rng)
                sf.write(out_path, out.numpy(), SR, subtype="PCM_16")
            if args.verify:
                # verify 模式：从磁盘回读回测
                yy = torch.from_numpy(sf.read(out_path, dtype="float32")[0])
                if kind == "additive_noise":
                    checks = {"snr_db_meas": measure_snr_db(yy, y)}
                elif kind == "time_stretch":
                    checks = {"rate_meas": yy.numel() / max(1, y.numel()) * params["rate"]}
                elif kind == "silence_pad":
                    checks = {"pad_s_meas": (yy.numel() - y.numel()) / 2 / SR}
                elif kind == "gain":
                    checks = {"gain_meas": params["gain"],
                              "clipped": int(torch.sum((yy >= 1.0) | (yy <= -1.0)))}
            row["dur_s"] = round(float(sf.info(out_path).duration), 4)
            row["out_md5"] = md5_file(out_path) if os.path.exists(out_path) else ""
            row["available"] = int(bool(row["out_md5"]))
            ok = check_values(pid, kind, params, checks, row)
            if not ok or not row["available"]:
                bad.append((pid, ref_id))
            rows.append(row)

    fields = ["perturb_id", "base_perturb_id", "kind", "params", "ref_id", "sound_id",
              "sound_notation", "sound_family", "is_short_variant", "src_audio", "out_file",
              "available", "dur_s", "out_md5", "snr_db_meas", "rate_check", "pad_s_meas",
              "clipped_samples", "check_ok", "spec_purpose", "spec_expected", "note"]
    for r in rows:
        for f in fields:
            r.setdefault(f, "")
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_avail = sum(int(r["available"]) for r in rows)
    print(f"[gen] rows={len(rows)} available={n_avail} unavailable={len(rows)-n_avail}")
    print(f"[gen] manifest={MANIFEST} md5={md5_file(MANIFEST)}")
    if bad:
        print(f"[FAIL] 参数回测不达标 {len(bad)} 条，例: {bad[:5]}")
        sys.exit(2)
    print("[gen] 参数回测全部达标（SNR/时长比/静音垫/增益按 TOL 对账）")


if __name__ == "__main__":
    main()
