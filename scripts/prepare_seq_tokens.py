"""
为 3–8s 序列样本生成 EnCodec token（Talker 序列监督的数据侧）。

背景（2026-10-27 @Talker 裁决）：
  - 时间轴口径统一到 **75Hz**（EnCodec 24kHz / hop 320 原生帧率）。
    13Hz 是 Qwen3-Omni Talker 本体的码本速率，不是本数据集的 token 帧率；
    13Hz 下 60–200ms 的一次击打只剩 0.8–2.6 帧，序列监督无法成立。
  - 逐击单独编码再拼接（Talker 的 stitch_hit_codes 方案）会**破坏时间轴**：
    EnCodec 是有感受野的编解码器，单独编码一个 200ms 片段得到的是
    「该片段自成一段」的码流，拼起来不等于整段音频的码流，
    帧位置也不再对应真实时间。因此这里**整段编码**，逐击只提供帧区间标注。

输入（远端持久盘 / 已拉取的清单）：
  data/sequence_samples.csv     序列级 manifest（source / start / end / dur / label）
  data/sequence_hits.csv        击打级明细（offset / dur / 记法）
  data/bbox_raw/demucs_srv/htdemucs/<source>/vocals.wav   分离后人声（权威音频源）
  data/real_bbox/segments_batch3/<source>/<hit_id>.wav    已有切片（可选，仅做交叉校验）

输出：
  data/seq_tokens/codes/<sequence_id>.npz   codes (1,1,16,T)，T = ceil(n_samples/320)
  data/seq_tokens/sequence_manifest.jsonl  每条序列一行（T / dur / 击打帧区间 / 校验值）
  data/seq_tokens/failures.jsonl            失败清单
  data/seq_tokens/hit_frames.csv            逐击 -> 帧区间映射（Talker 对齐用）

口径铁律：
  - 帧率 75Hz；T 由实际采样点决定 T = ceil(len(y_24k) / 320)，与 round(dur*75) 允许 ±1 帧差。
  - 只取前 16 层（RVQ 由粗到细），与 talker_sequence.py 的 layers=16 一致。
  - 音频只在服务器上读取，不落本地、不入 git；产物是 token（可逆的离散码），仍不入 git。
  - 绝不跨 source 拼接；每条序列只来自一个 source 的 vocals.wav。
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

SR_SEG = 22050      # 切片管线采样率（segment_beatbox_onsets.py）
SR_CODEC = 24000    # EnCodec 采样率
HOP = 320           # 24kHz / 75Hz
LAYERS = 16
CODEBOOK = 1024


def split_of(source):
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def load_manifest(samples_csv, hits_csv):
    samples = list(csv.DictReader(open(samples_csv, encoding="utf-8")))
    hits = defaultdict(list)
    for r in csv.DictReader(open(hits_csv, encoding="utf-8")):
        hits[r["sequence_id"]].append(r)
    for v in hits.values():
        v.sort(key=lambda x: float(x["offset"]))
    return samples, hits


def self_test():
    """纯 numpy 校验帧率换算，不依赖音频/模型。"""
    for dur, exp in ((3.0, 225), (4.0, 300), (7.9003, 593), (8.0, 600)):
        n = int(round(dur * SR_CODEC))
        t = math.ceil(n / HOP)
        assert abs(t - exp) <= 1, (dur, t, exp)
        assert abs(t - round(dur * 75)) <= 1, (dur, t)
    # 帧区间必须落在 [0, T)
    t = 593
    for off, hdur in ((0.0, 0.3175), (7.0, 0.9)):
        s = int(round(off * 75))
        e = min(t, int(round((off + hdur) * 75)))
        assert 0 <= s < e <= t, (s, e, t)
    print("self-test OK: 75Hz 帧换算 + 帧区间越界检查通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/gemini/code/lightfour-train/data/sequence_samples.csv")
    ap.add_argument("--hits", default="/gemini/code/lightfour-train/data/sequence_hits.csv")
    ap.add_argument("--vocals-root",
                    default="/gemini/code/lightfour-train/data/bbox_raw/demucs_srv/htdemucs")
    ap.add_argument("--out-dir", default="/gemini/code/lightfour-train/data/seq_tokens")
    ap.add_argument("--model", default="/gemini/code/lightfour-train/models/encodec_24khz")
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import librosa
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codec_tokens import load_model

    torch.set_num_threads(args.threads)
    samples, hits = load_manifest(args.samples, args.hits)
    if args.limit:
        samples = samples[: args.limit]

    out_dir = Path(args.out_dir)
    code_dir = out_dir / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)

    # 每条序列的 vocals.wav 只读一次（缓存整段 24kHz 单声道）
    wav_cache = {}
    manifest, failures = [], []
    hit_frames = []

    def get_wav(source):
        if source not in wav_cache:
            p = Path(args.vocals_root) / source / "vocals.wav"
            if not p.exists():
                wav_cache[source] = None
            else:
                y, _ = librosa.load(str(p), sr=SR_CODEC, mono=True)
                wav_cache[source] = y
        return wav_cache[source]

    for i, r in enumerate(samples, 1):
        sid = r["sequence_id"]
        try:
            y = get_wav(r["source"])
            if y is None:
                failures.append({"sequence_id": sid, "source": r["source"],
                                 "error": "vocals.wav missing"})
                continue
            # 序列窗口在 vocals.wav 时间轴上的绝对位置
            # （segment_beatbox_onsets.py 的 start/end 就是 vocals.wav 的时间轴）
            t0 = float(r["start"])
            t1 = float(r["end"])
            s0 = int(round(t0 * SR_CODEC))
            s1 = int(round(t1 * SR_CODEC))
            seg = y[s0:s1]
            if len(seg) < int(0.05 * SR_CODEC):
                failures.append({"sequence_id": sid, "source": r["source"],
                                 "error": f"window too short: {len(seg)/SR_CODEC:.3f}s"})
                continue
            inp = proc(raw_audio=seg.astype("float32"), sampling_rate=SR_CODEC,
                       return_tensors="pt")
            inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.no_grad():
                codes = model.encode(inp["input_values"], inp["padding_mask"]).audio_codes
            assert codes.shape[0] == 1 and codes.shape[1] == 1, codes.shape
            q = int(codes.shape[2])
            assert q >= args.layers, f"quantizers {q} < layers {args.layers}"
            sliced = codes[:, :, : args.layers, :].cpu().numpy().astype(np.int64)
            T = int(sliced.shape[3])
            dur = len(seg) / SR_CODEC
            # 帧数校验：T 必须等于 ceil(n/320)，且与 round(dur*75) 差 <= 1
            assert T == math.ceil(len(seg) / HOP), (T, len(seg))
            assert abs(T - round(dur * 75)) <= 1, (T, dur)
            assert int(sliced.min()) >= 0 and int(sliced.max()) < CODEBOOK, \
                (int(sliced.min()), int(sliced.max()))
            np.savez_compressed(code_dir / f"{sid}.npz", codes=sliced)

            # 逐击 -> 帧区间（offset 相对序列起点）
            hf = []
            for h in hits.get(sid, []):
                off = float(h["offset"])
                hdur = float(h["dur"])
                fs = int(round(off * 75))
                fe = min(T, int(round((off + hdur) * 75)))
                fe = max(fe, min(T, fs + 1))
                hf.append((h, fs, fe))
                hit_frames.append({
                    "sequence_id": sid, "hit_index": h["hit_index"], "hit_id": h["hit_id"],
                    "source": r["source"], "split": r["split"],
                    "offset_sec": off, "dur_sec": hdur,
                    "frame_start": fs, "frame_end": fe,
                    "sound_id": h["sound_id"], "sound_notation": h["sound_notation"],
                    "sound_family": h["sound_family"], "label_source": h["label_source"],
                })

            manifest.append({
                "sequence_id": sid, "source": r["source"], "split": r["split"],
                "start": t0, "end": t1, "dur": round(dur, 4),
                "codes": str(code_dir / f"{sid}.npz"), "codes_shape": list(sliced.shape),
                "T": T, "frame_rate_hz": 75, "layers": args.layers,
                "n_hits": len(hf), "n_hit_frames": sum(fe - fs for _, fs, fe in hf),
                "hit_frame_coverage": round(sum(fe - fs for _, fs, fe in hf) / max(T, 1), 4),
                "raw_quantizers": q,
                "text": r["sound_notations"].replace("|", " | "),
                "label_source": r["label_source"],
                "codec_model": "encodec_24khz", "bandwidth": 24.0,
                "codes_md5": hashlib.md5(
                    (code_dir / f"{sid}.npz").read_bytes()).hexdigest(),
            })
        except Exception as exc:  # noqa: BLE001
            failures.append({"sequence_id": sid, "source": r["source"],
                             "error": f"{type(exc).__name__}: {exc}",
                             "tb": traceback.format_exc()[-400:]})
        if i % 20 == 0:
            print(f"[{i}/{len(samples)}] ok={len(manifest)} fail={len(failures)}", flush=True)

    with open(out_dir / "sequence_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "failures.jsonl", "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if hit_frames:
        with open(out_dir / "hit_frames.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hit_frames[0].keys()))
            w.writeheader()
            w.writerows(hit_frames)

    per_split = defaultdict(int)
    for r in manifest:
        per_split[r["split"]] += 1
    Ts = [r["T"] for r in manifest]
    print(json.dumps({
        "sequences_ok": len(manifest), "failures": len(failures),
        "per_split": dict(per_split),
        "T_min": min(Ts) if Ts else 0,
        "T_median": sorted(Ts)[len(Ts) // 2] if Ts else 0,
        "T_max": max(Ts) if Ts else 0,
        "hits_total": len(hit_frames),
        "frame_rate_hz": 75,
    }, ensure_ascii=False, indent=1))
    if failures:
        print(f"FAIL: {len(failures)} 条失败，详见 failures.jsonl", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
