"""重渲 hihats 专用试听包（12 击 × 4 臂），不触碰 live 产物。

背景：11:41 `codec_rate_ab.py` 重跑把 `outputs/codec_rate_ab/listen/` 覆盖成
「cymbal 4 / hihats 4 / click 4」混合包，原「12 条全 hihats」包已不存在。
本脚本从 live R4 `per_hit.csv` 的 247 条 hihats 里**重抽 12 条**（每 source 至多 1 条），
渲染 4 臂（true / 75 / 13c / 13p）到独立目录 `outputs/listen_hihats_v2/`。

选样直接复用 `codec_rate_ab._spread_by_source`，不另写一套逻辑：
  每 source 取 mel13p_vs_75 中位代表 → 按该值排序 → 均匀取 12（最差/中位/最好都覆盖）
  → 12 个不同 source，避免单 source 包场。

写盘用 FLOAT（与 codec_rate_ab listen 同 subtype），否则 PCM_16 量化会把
重算 mel 抖到 1e-2 量级、对不上 per_hit.csv。

用法（远端，不落脚本文件）：
  base64 编码本文件后 `python -c "import base64;exec(base64.b64decode('...'))"`
产物：wav/*.wav、index.json、composition.json、README.md
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/gemini/code/lightfour-train")
# LF_OUT 只用于「同参数换目录重跑」的确定性对拍，正式产物仍写 outputs/listen_hihats_v2
OUT = Path(os.environ.get("LF_OUT", str(BASE / "outputs/listen_hihats_v2")))
PER_HIT = BASE / "outputs/codec_rate_ab/per_hit.csv"
SAMPLES = BASE / "data/sequence_samples.csv"
HITS = BASE / "data/sequence_hits.csv"
CODES = BASE / "data/seq_tokens/codes"
VOCALS = BASE / "data/bbox_raw/demucs_srv/htdemucs"
MODEL = BASE / "models/encodec_24khz"
N_PICK = 12

sys.path.insert(0, str(BASE / "scripts"))
import codec_rate_ab as cra          # noqa: E402  复用 rate_view / frame_rms / _spread_by_source
from codec_tokens import load_model  # noqa: E402

import librosa                       # noqa: E402
import soundfile as sf               # noqa: E402
import torch                         # noqa: E402

SR, HOP = cra.SR, cra.HOP


def sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = [r for r in csv.DictReader(open(PER_HIT, encoding="utf-8"))
            if r["sound_family"] == "hihats"]
    assert len(rows) == 247, len(rows)
    picked = cra._spread_by_source(rows, N_PICK)
    picked_keys = {(r["sequence_id"], r["hit_index"]) for r in picked}
    # 备选池：仍按退化排序，渲染失败时按序补位
    rest = sorted([r for r in rows if (r["sequence_id"], r["hit_index"]) not in picked_keys],
                  key=lambda r: float(r["mel13p_vs_75"]))
    queue = picked + rest

    samples = {r["sequence_id"]: r for r in csv.DictReader(open(SAMPLES, encoding="utf-8"))}
    hits_by_seq = {}
    for r in csv.DictReader(open(HITS, encoding="utf-8")):
        hits_by_seq.setdefault(r["sequence_id"], []).append(r)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    model, _ = load_model(str(MODEL), dev)

    wav_cache = {}
    wav_dir = OUT / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for stale in wav_dir.glob("*.wav"):
        stale.unlink()

    entries, skipped = [], []
    used_seq = set()
    for r in queue:
        if len(entries) >= N_PICK:
            break
        sid, hi = r["sequence_id"], r["hit_index"]
        s = samples.get(sid)
        if s is None:
            skipped.append({"hit_id": r["hit_id"], "reason": "sequence not in samples"})
            continue
        src = s["source"]
        if src not in wav_cache:
            p = VOCALS / src / "vocals.wav"
            wav_cache[src] = librosa.load(str(p), sr=SR, mono=True)[0] if p.exists() else None
        y_all = wav_cache[src]
        if y_all is None:
            skipped.append({"hit_id": r["hit_id"], "reason": "vocals missing"})
            continue
        npz = CODES / f"{sid}.npz"
        if not npz.exists():
            skipped.append({"hit_id": r["hit_id"], "reason": "codes missing"})
            continue

        y = y_all[int(round(float(s["start"]) * SR)):int(round(float(s["end"]) * SR))]
        codes = np.load(npz)["codes"]
        assert codes.ndim == 4 and codes.shape[:3] == (1, 1, 16), codes.shape
        T = int(codes.shape[3])
        rms = cra.frame_rms(y, T)
        views = {
            "true": None,
            "75": codes,
            "13c": cra.rate_view(codes, T, 13.0, "center"),
            "13p": cra.rate_view(codes, T, 13.0, "peak", rms=rms),
        }
        auds = {}
        for k, v in views.items():
            if v is None:
                auds[k] = y.astype("float32")
            else:
                with torch.no_grad():
                    o = model.decode(torch.from_numpy(v).to(dev), [None], None)
                auds[k] = o.audio_values[0].mean(dim=0).cpu().numpy().astype("float32")
        wav_len = [len(a) for a in auds.values()]
        n = min(wav_len)
        trunc = 1.0 - n / max(wav_len)
        if trunc > 0.05:
            skipped.append({"hit_id": r["hit_id"], "reason": f"trunc {trunc:.2%}"})
            continue

        hit = next((h for h in hits_by_seq.get(sid, []) if h["hit_index"] == hi), None)
        if hit is None:
            skipped.append({"hit_id": r["hit_id"], "reason": "hit row missing"})
            continue
        a = int(round(float(hit["offset"]) * SR))
        b = min(n, int(round((float(hit["offset"]) + float(hit["dur"])) * SR)))
        if b - a < 1024:
            skipped.append({"hit_id": r["hit_id"], "reason": "segment < n_fft"})
            continue

        files = {}
        for k, au in auds.items():
            seg = au[a:b]
            fp = wav_dir / f"{hit['hit_id']}_h{hi}_{k}.wav"
            sf.write(str(fp), seg, SR, subtype="FLOAT")
            files[k] = {"file": f"wav/{fp.name}", "sha1": sha1_bytes(fp.read_bytes()),
                        "n_samples": int(len(seg)),
                        "dur_ms": round(1000.0 * len(seg) / SR, 2)}
        entries.append({
            "hit_id": hit["hit_id"], "hit_index": hi, "sequence_id": sid,
            "source": src, "split": s.get("split", ""),
            "sound_id": r["sound_id"], "sound_notation": r["sound_notation"],
            "sound_family": r["sound_family"], "dur_ms": r["dur_ms"],
            "is_click_60_200": r["is_click_60_200"],
            "mel75_vs_true": r["mel75_vs_true"],
            "mel13c_vs_75": r["mel13c_vs_75"], "mel13p_vs_75": r["mel13p_vs_75"],
            "nmel75_vs_true": r["nmel75_vs_true"],
            "hf_true": r["hf_true"], "hf75": r["hf75"],
            "hf13c": r["hf13c"], "hf13p": r["hf13p"],
            "wav_trunc_ratio_seq": r["wav_trunc_ratio"],
            "files": files,
        })
        used_seq.add(sid)

    assert len(entries) == N_PICK, (len(entries), skipped[:5])
    assert len({e["source"] for e in entries}) == N_PICK, "source 去重失败"

    composition = {
        "listen_n": len(entries),
        "pool": "hihats",
        "pool_size_total": len(rows),
        "selection": "codec_rate_ab._spread_by_source(hihats_rows, 12)：每 source 取 "
                     "mel13p_vs_75 中位代表后均匀取 12 → 12 个不同 source",
        "source_of_truth": "outputs/codec_rate_ab/per_hit.csv (live R4, 31 col)",
        "per_hit_md5": md5_file(PER_HIT),
        "arms": ["true", "75", "13c", "13p"],
        "subtype": "FLOAT",
        "n_wav": len(list(wav_dir.glob('*.wav'))),
        "n_sources": len({e["source"] for e in entries}),
        "skipped": skipped,
        "warning": "本包全部为 hihats，**只能判 hihats**；不得表述为三池/全 family 未见退化",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "composition.json").write_text(json.dumps(composition, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps({"n": len(entries), "wav": composition["n_wav"],
                      "sources": composition["n_sources"], "out": str(OUT)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
