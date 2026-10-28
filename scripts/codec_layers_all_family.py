"""
L16 vs L32 全 family 分层重建表（287 条序列 / 5472 条击打 / 16 family）。

回答的问题只有一个：EnCodec 24kHz 的 24kbps 原始码流是 32 层，取前 16 层（等效 12kbps）
与取满 32 层（24kbps），逐击打重建的 log-mel 距离差多少，且这个差在**每个 family** 上
是否稳定（不是只看 hihats/cymbal）。

口径全部继承已冻结实现，本脚本不新造指标：
  - mel：scripts/mel_metric.py（min 帧口径 + 帧差断言，唯一实现）
  - hf ：codec_rate_ab.hf_ratio（>=2000Hz 能量占比）
  - 输入：data/sequence_samples.csv 全量 287 条（按 token manifest，不按 audio_available 过滤）
  - 真值：data/bbox_raw/demucs_srv/htdemucs/<source>/vocals.wav
  - 可复现：重编码的前 16 层必须与已交付 data/seq_tokens/codes/<sid>.npz 逐元素一致
  - CI：按 source 聚类 bootstrap（同一 source 的击打不独立），2000 次，seed 20261027

边界（写死在 summary 里，防止下游误读）：
  - 16 vs 32 **不是控制变量**：码率同时从 12kbps 翻到 24kbps。本表只能称「码率加倍增益」，
    不得写成「16 层容量不足」或「层数改善」。
  - 本表不重置裁决㉛的镲类三层口径，也不改「codec 锁 16 层 / 1024 码本 / 12kbps」的结论。
  - teeth n=3：只报数，不给 CI、不进任何结论。

用法：
  python scripts/codec_layers_all_family.py --self-test
  python scripts/codec_layers_all_family.py --out-dir outputs/codec_layers_all_family
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec_rate_ab import HF_CUTOFF, MEL_SPEC, SR, hf_ratio, mel_dist  # noqa: E402

BASE = Path("/gemini/code/lightfour-train")
LAYERS = (16, 32)
N_BOOT = 2000
SEED = 20261027
REPORT_ONLY_MIN_HITS = 10      # 少于 10 条击打 → 只报数
REPORT_ONLY_MIN_SOURCES = 2    # 少于 2 个 source → 聚类 CI 无意义


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def codes_digest(codes_dir: Path, seq_ids) -> str:
    """参与计算的 npz 清单（name:md5）再取 sha256，避免整目录遍历的歧义。"""
    h = hashlib.sha256()
    for sid in sorted(seq_ids):
        p = codes_dir / f"{sid}.npz"
        if p.exists():
            h.update(f"{sid}:{md5_file(p)}\n".encode("utf-8"))
    return h.hexdigest()


def load_inputs(samples_path: Path, hits_path: Path):
    samples = {r["sequence_id"]: r for r in csv.DictReader(open(samples_path, encoding="utf-8"))}
    hits = defaultdict(list)
    for r in csv.DictReader(open(hits_path, encoding="utf-8")):
        hits[r["sequence_id"]].append(r)
    for v in hits.values():
        v.sort(key=lambda r: int(r["hit_index"]))
    return samples, hits


def cluster_ci(sel, n_boot=N_BOOT, seed=SEED):
    """mel 相对变化（L32 vs L16）中位数的 95% CI，**按 source 重采样**。"""
    if not sel:
        return None
    by_src = defaultdict(list)
    for r in sel:
        by_src[r["source"]].append(float(r["delta_pct"]))
    srcs = sorted(by_src)
    arrs = [np.array(by_src[s], dtype=float) for s in srcs]
    point = float(np.median(np.concatenate(arrs)))
    out = {"n_hits": len(sel), "n_sources": len(srcs), "point_median_pct": round(point, 1)}
    if len(srcs) < REPORT_ONLY_MIN_SOURCES:
        out.update({"n_boot": 0, "seed": seed, "ci95_median_pct": None, "report_only": True,
                    "note": f"source={len(srcs)}<2，聚类 bootstrap 无意义，只报点估计"})
        return out
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(srcs), len(srcs))
        meds[b] = np.median(np.concatenate([arrs[i] for i in pick]))
    out.update({"n_boot": n_boot, "seed": seed, "report_only": False,
                "ci95_median_pct": [round(float(np.percentile(meds, 2.5)), 1),
                                    round(float(np.percentile(meds, 97.5)), 1)]})
    return out


def agg(sel, n_boot=N_BOOT, seed=SEED):
    m16 = np.array([r["mel_L16"] for r in sel], dtype=float)
    m32 = np.array([r["mel_L32"] for r in sel], dtype=float)
    d = (m32 - m16) / np.maximum(m16, 1e-9) * 100.0
    ci = cluster_ci(sel, n_boot=n_boot, seed=seed)
    report_only = len(sel) < REPORT_ONLY_MIN_HITS or (ci or {}).get("report_only", False)
    return {
        "n_hits": len(sel),
        "n_sequences": len({r["sequence_id"] for r in sel}),
        "n_sources": len({r["source"] for r in sel}),
        "mel16_median": round(float(np.median(m16)), 4),
        "mel32_median": round(float(np.median(m32)), 4),
        "delta_pct_median": round(float(np.median(d)), 1),
        "delta_pct_mean": round(float(np.mean(d)), 1),
        "ci95_delta_pct_median": (ci or {}).get("ci95_median_pct"),
        "n_boot": (ci or {}).get("n_boot", 0),
        "share_32_better": round(float((m32 < m16).mean()), 4),
        "report_only": bool(report_only),
        "report_only_reason": (None if not report_only else
                               (f"n_hits={len(sel)}<{REPORT_ONLY_MIN_HITS}" if len(sel) < REPORT_ONLY_MIN_HITS
                                else (ci or {}).get("note"))),
        "hf_true": round(float(np.nanmean([r["hf_true"] for r in sel])), 4),
        "hf_L16": round(float(np.nanmean([r["hf_L16"] for r in sel])), 4),
        "hf_L32": round(float(np.nanmean([r["hf_L32"] for r in sel])), 4),
    }


def self_test(samples_path: Path, hits_path: Path, codes_dir: Path) -> int:
    from mel_metric import MEL_SPEC as MS
    ok = True
    print(f"mel_metric MEL_SPEC = {MS}")
    if MS != MEL_SPEC:
        print("FAIL: MEL_SPEC 与 codec_rate_ab 不一致")
        ok = False
    samples, hits = load_inputs(samples_path, hits_path)
    fams = sorted({h["sound_family"] for v in hits.values() for h in v})
    n_hit = sum(len(v) for v in hits.values())
    print(f"samples={len(samples)} hits={n_hit} families={len(fams)}")
    print(f"families: {fams}")
    missing_codes = [sid for sid in samples if not (codes_dir / f"{sid}.npz").exists()]
    print(f"codes missing: {len(missing_codes)}")
    if missing_codes:
        ok = False
    if not (len(samples) == 287 and n_hit == 5472):
        print("WARN: 行数与 287/5472 不一致（口径可能变了）")
    print("SELF_TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(BASE / "data/sequence_samples.csv"))
    ap.add_argument("--hits", default=str(BASE / "data/sequence_hits.csv"))
    ap.add_argument("--codes-dir", default=str(BASE / "data/seq_tokens/codes"))
    ap.add_argument("--vocals-root", default=str(BASE / "data/bbox_raw/demucs_srv/htdemucs"))
    ap.add_argument("--out-dir", default=str(BASE / "outputs/codec_layers_all_family"))
    ap.add_argument("--model", default=str(BASE / "models/encodec_24khz"))
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条序列（冒烟用）")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    samples_path, hits_path = Path(args.samples), Path(args.hits)
    codes_dir = Path(args.codes_dir)
    if args.self_test:
        return self_test(samples_path, hits_path, codes_dir)

    import librosa
    import torch
    from codec_tokens import load_model

    torch.set_num_threads(args.threads)
    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    model, proc = load_model(args.model, dev)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples, hits = load_inputs(samples_path, hits_path)
    seq_ids = sorted(samples)
    if args.limit:
        seq_ids = seq_ids[: args.limit]
    print(f"序列 {len(seq_ids)} / 击打 {sum(len(hits[s]) for s in seq_ids)} / 设备 {dev}", flush=True)

    cache, rows, failures = {}, [], []
    mismatch_seqs = []
    t0 = time.time()
    for i, sid in enumerate(seq_ids, 1):
        s = samples[sid]
        src = s["source"]
        try:
            if src not in cache:
                p = Path(args.vocals_root) / src / "vocals.wav"
                cache[src] = librosa.load(str(p), sr=SR, mono=True)[0]
            y = cache[src][int(round(float(s["start"]) * SR)):int(round(float(s["end"]) * SR))]
            inp = proc(raw_audio=y.astype("float32"), sampling_rate=SR, return_tensors="pt")
            inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.no_grad():
                codes = model.encode(inp["input_values"], inp["padding_mask"],
                                     bandwidth=24.0).audio_codes
            assert int(codes.shape[2]) >= 32, f"{sid} n_q={codes.shape[2]}"
            got16 = codes[:, :, :16, :].cpu().numpy()
            npz = codes_dir / f"{sid}.npz"
            if npz.exists():
                stored = np.load(npz)["codes"]
                if got16.shape != stored.shape or not np.array_equal(got16, stored):
                    mismatch_seqs.append(sid)
            dec = {}
            for L in LAYERS:
                with torch.no_grad():
                    o = model.decode(codes[:, :, :L, :].contiguous(), [None], None)
                dec[L] = o.audio_values[0].mean(dim=0).cpu().numpy().astype("float32")
            n = min(len(y), len(dec[16]), len(dec[32]))
            for h in hits.get(sid, []):
                a = int(round(float(h["offset"]) * SR))
                b = min(n, int(round((float(h["offset"]) + float(h["dur"])) * SR)))
                if a < 0 or b - a < MEL_SPEC["n_fft"]:
                    continue
                ys = y[a:b]
                m = {L: mel_dist(ys, dec[L][a:b]) for L in LAYERS}
                if any(v is None for v in m.values()):
                    continue
                rec = {"sequence_id": sid, "hit_index": h["hit_index"], "hit_id": h["hit_id"],
                       "source": src, "split": s["split"], "sound_family": h["sound_family"],
                       "sound_notation": h["sound_notation"],
                       "dur_ms": round(float(h["dur"]) * 1000, 1),
                       "mel_L16": round(m[16], 6), "mel_L32": round(m[32], 6),
                       "delta_pct": round((m[32] - m[16]) / max(m[16], 1e-9) * 100, 2),
                       "hf_true": round(float(hf_ratio(ys)), 5),
                       "hf_L16": round(float(hf_ratio(dec[16][a:b])), 5),
                       "hf_L32": round(float(hf_ratio(dec[32][a:b])), 5)}
                rows.append(rec)
        except Exception as e:  # 一条序列失败不拖垮整批，但要记账
            failures.append({"sequence_id": sid, "source": src, "error": f"{type(e).__name__}: {e}"})
        if i % 20 == 0:
            print(f"[{i}/{len(seq_ids)}] hits={len(rows)} fail={len(failures)} "
                  f"{time.time() - t0:.0f}s", flush=True)

    if not rows:
        print("FAIL: 0 条击打产出，终止")
        return 2

    fams = sorted({r["sound_family"] for r in rows},
                  key=lambda f: -sum(1 for r in rows if r["sound_family"] == f))
    table = [{"group": "ALL", **agg(rows, args.n_boot)}]
    for f in fams:
        table.append({"group": f"family:{f}", **agg([r for r in rows if r["sound_family"] == f],
                                                    args.n_boot)})

    summary = {
        "experiment": "全 family L16 vs L32 逐击打重建（EnCodec 24kHz / 24kbps 原始 32 层，取前 16 层）",
        "codec": "encodec_24khz / 24kbps / codebook 1024 / 75Hz",
        "bitrate_note": "L32=24kbps，L16=12kbps（等效）。16 vs 32 **非控制变量**，只可称「码率加倍增益」",
        "frame_policy": "min（mel_metric 默认，带帧差断言）",
        "n_sequences": len({r["sequence_id"] for r in rows}),
        "n_hits": len(rows),
        "n_families": len(fams),
        "mel_spec": MEL_SPEC,
        "hf_cutoff_hz": HF_CUTOFF,
        "codes16_reencode_mismatch": {"n": len(mismatch_seqs), "sequences": mismatch_seqs[:20]},
        "failures": failures,
        "cluster_bootstrap": {"method": "按 source 重采样，统计量 = (mel_L32-mel_L16)/mel_L16 的中位数",
                              "n_boot": args.n_boot, "seed": SEED},
        "report_only_rule": f"n_hits<{REPORT_ONLY_MIN_HITS} 或 n_sources<{REPORT_ONLY_MIN_SOURCES} → 只报数，不给 CI",
        "table": table,
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "librosa": librosa.__version__, "numpy": np.__version__,
                "device": dev, "cuda_available": bool(torch.cuda.is_available()),
                "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)},
        "fingerprints": {"code_sha256": sha256_file(Path(__file__).resolve()),
                         "samples_md5": md5_file(samples_path),
                         "hits_md5": md5_file(hits_path),
                         "codes_digest_sha256": codes_digest(codes_dir, {r["sequence_id"] for r in rows})},
        "runtime_sec": round(time.time() - t0, 1),
        "caveat": "mel 是重建保真度代理，不是可闻性判定；本表不得单独用于宣布 codec 过关",
        "generated_by": "scripts/codec_layers_all_family.py",
    }

    fields = list(rows[0].keys())
    with open(out / "per_hit_layers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    md = ["# 全 family L16 vs L32 逐击打重建", "",
          f"- 序列 **{summary['n_sequences']}** / 击打 **{summary['n_hits']}** / family **{summary['n_families']}**",
          f"- codec：{summary['codec']}；{summary['bitrate_note']}",
          f"- mel：`scripts/mel_metric.py`（frame_policy={summary['frame_policy']}）；hf >= {HF_CUTOFF}Hz",
          f"- 前 16 层重编码与已交付 npz 不一致：**{len(mismatch_seqs)}** 条序列；失败序列 {len(failures)} 条",
          f"- CI：按 source 聚类 bootstrap（{args.n_boot} 次，seed {SEED}）；"
          f"{summary['report_only_rule']}",
          "",
          "| group | n_hits | n_seq | n_src | mel16 | mel32 | Δ中位，负=更好 (95% CI, source 聚类) | 32更好占比 | 状态 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for t in table:
        ci = t["ci95_delta_pct_median"]
        ci_s = "只报数" if t["report_only"] else (f"[{ci[0]}, {ci[1]}]" if ci else "—")
        md.append(f"| {t['group']} | {t['n_hits']} | {t['n_sequences']} | {t['n_sources']} | "
                  f"{t['mel16_median']} | {t['mel32_median']} | {t['delta_pct_median']}% {ci_s} | "
                  f"{t['share_32_better']:.3f} | {'report_only' if t['report_only'] else 'ok'} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_sequences", "n_hits", "n_families",
                                              "codes16_reencode_mismatch", "runtime_sec")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
