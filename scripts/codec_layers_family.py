"""
镲类/Hi-hats 的 16 层 vs 32 层 A/B（§7.3c 遗留：历史抽样只覆盖 Clicks/Drag/Liprolls）。

为什么单列：§8 Gate A 要求「镲类不明显劣化」。codec_rate_ab.py 已实测 16 层在
hihats 上 mel 中位 0.826–0.845 > 0.8，越过 Gate A 的 mel<0.8 线。本脚本回答
「这是 16 层的带宽/层数上限，还是 EnCodec 24kbps 本身的上限」：
同一批序列、同一 mel 实现，只改 RVQ 层数 16 → 32。

做法：
  1. 取所有含 hihats / cymbal 击打的可用序列（54 条）
  2. 用 EnCodec 24kHz/24kbps 重新编码，取全 32 层；断言前 16 层与已交付 npz 完全一致
     （顺带交叉验证 @Data 的 token 管线是可复现的）
  3. 分别用前 16 层、前 32 层解码
  4. 在逐击段上算 log-mel 距离 + 高频能量占比，按 family 分组
  5. 同序列内的其他 family 击打作为对照（避免「镲类的改善来自序列不同」）

用法：
  python scripts/codec_layers_family.py --self-test
  python scripts/codec_layers_family.py
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec_rate_ab import (HF_CUTOFF, HOP, MEL_SPEC, SR, frame_rms,  # noqa: E402
                           hf_ratio, mel_dist)

BASE = Path("/gemini/code/lightfour-train")
TARGET_FAMS = ("hihats", "cymbal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(BASE / "data/sequence_samples.csv"))
    ap.add_argument("--hits", default=str(BASE / "data/sequence_hits.csv"))
    ap.add_argument("--codes-dir", default=str(BASE / "data/seq_tokens/codes"))
    ap.add_argument("--vocals-root", default=str(BASE / "data/bbox_raw/demucs_srv/htdemucs"))
    ap.add_argument("--out-dir", default=str(BASE / "outputs/codec_layers_family"))
    ap.add_argument("--model", default=str(BASE / "models/encodec_24khz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import librosa
    import torch
    from codec_tokens import load_model

    torch.set_num_threads(args.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, dev)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = {r["sequence_id"]: r for r in csv.DictReader(open(args.samples, encoding="utf-8"))}
    hits = defaultdict(list)
    for r in csv.DictReader(open(args.hits, encoding="utf-8")):
        hits[r["sequence_id"]].append(r)
    seq_ids = [sid for sid in samples if any(h["sound_family"] in TARGET_FAMS
                                            for h in hits.get(sid, []))]
    seq_ids.sort()
    if args.limit:
        seq_ids = seq_ids[: args.limit]
    print(f"目标序列 {len(seq_ids)}", flush=True)

    cache, rows, mismatch = {}, [], 0
    for i, sid in enumerate(seq_ids, 1):
        s = samples[sid]
        src = s["source"]
        if src not in cache:
            p = Path(args.vocals_root) / src / "vocals.wav"
            cache[src] = librosa.load(str(p), sr=SR, mono=True)[0]
        y = cache[src][int(round(float(s["start"]) * SR)):int(round(float(s["end"]) * SR))]
        inp = proc(raw_audio=y.astype("float32"), sampling_rate=SR, return_tensors="pt")
        inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.no_grad():
            codes = model.encode(inp["input_values"], inp["padding_mask"],
                                 bandwidth=24.0).audio_codes
        q = int(codes.shape[2])
        assert q >= 32, q
        stored = np.load(Path(args.codes_dir) / f"{sid}.npz")["codes"]
        got16 = codes[:, :, :16, :].cpu().numpy()
        if got16.shape != stored.shape or not np.array_equal(got16, stored):
            mismatch += 1
        dec = {}
        for L in (16, 32):
            with torch.no_grad():
                o = model.decode(codes[:, :, :L, :].contiguous(), [None], None)
            dec[L] = o.audio_values[0].mean(dim=0).cpu().numpy().astype("float32")
        n = min(len(y), len(dec[16]), len(dec[32]))
        for h in hits.get(sid, []):
            a = int(round(float(h["offset"]) * SR))
            b = min(n, int(round((float(h["offset"]) + float(h["dur"])) * SR)))
            if b - a < MEL_SPEC["n_fft"]:
                continue
            ys = y[a:b]
            rec = {"sequence_id": sid, "hit_index": h["hit_index"], "hit_id": h["hit_id"],
                   "split": s["split"], "sound_family": h["sound_family"],
                   "dur_ms": round(float(h["dur"]) * 1000, 1),
                   "is_target": int(h["sound_family"] in TARGET_FAMS),
                   "hf_true": round(hf_ratio(ys), 5)}
            for L in (16, 32):
                rec[f"mel_L{L}"] = round(mel_dist(ys, dec[L][a:b]), 5)
                rec[f"hf_L{L}"] = round(hf_ratio(dec[L][a:b]), 5)
            rows.append(rec)
        if i % 10 == 0:
            print(f"[{i}/{len(seq_ids)}] hits={len(rows)}", flush=True)

    def agg(sel):
        m16 = np.array([r["mel_L16"] for r in sel])
        m32 = np.array([r["mel_L32"] for r in sel])
        d = (m32 - m16) / np.maximum(m16, 1e-9)
        h16 = np.array([r["hf_L16"] for r in sel])
        h32 = np.array([r["hf_L32"] for r in sel])
        ht = np.array([r["hf_true"] for r in sel])
        return {"n": len(sel),
                "mel16_median": round(float(np.median(m16)), 4),
                "mel32_median": round(float(np.median(m32)), 4),
                "delta_pct_median": round(float(np.median(d)) * 100, 1),
                "share_32_better": round(float((m32 < m16).mean()), 4),
                "hf_true": round(float(np.nanmean(ht)), 4),
                "hf16": round(float(np.nanmean(h16)), 4),
                "hf32": round(float(np.nanmean(h32)), 4)}

    groups = {"target_hihats_cymbal": [r for r in rows if r["is_target"] == 1],
              "control_other_fams": [r for r in rows if r["is_target"] == 0]}
    for f in TARGET_FAMS:
        groups[f"family:{f}"] = [r for r in rows if r["sound_family"] == f]
    table = [{"group": g, **agg(v)} for g, v in groups.items() if v]

    summary = {"experiment": "镲类/Hi-hats 16 层 vs 32 层（同批、同 mel 实现）",
               "codec": "encodec_24khz / 24kbps / codebook 1024",
               "n_sequences": len(seq_ids), "n_hits": len(rows),
               "mel_spec": MEL_SPEC, "hf_cutoff_hz": HF_CUTOFF,
               "codes16_reencode_mismatch": mismatch,
               "table": table,
               "gate_a_note": "§8 Gate A：mel 中位 <0.8、镲类不明显劣化；16 层 hihats 0.826 已越线",
               "generated_by": "scripts/codec_layers_family.py"}
    with open(out / "per_hit.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    md = ["# 镲类/Hi-hats：16 层 vs 32 层", "",
          f"- 序列 {len(seq_ids)} / 击打段 {len(rows)}；前 16 层重编码与已交付 npz 不一致 **{mismatch}** 条",
          f"- mel: {MEL_SPEC}；hf >= {HF_CUTOFF}Hz", "",
          "| group | n | mel16 | mel32 | 32层相对变化 | 32更好占比 | hf_true | hf16 | hf32 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        md.append(f"| {t['group']} | {t['n']} | {t['mel16_median']} | {t['mel32_median']} | "
                  f"{t['delta_pct_median']}% | {t['share_32_better']:.2f} | {t['hf_true']} | "
                  f"{t['hf16']} | {t['hf32']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:1200])


if __name__ == "__main__":
    main()
