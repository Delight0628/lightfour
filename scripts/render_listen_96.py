"""正式 96 wav 试听包（12/6/6 配额），不触碰 live 产物与冻结生成器。

背景：`codec_rate_ab.py:166` 注释写「--listen-n 24 → 12/6/6、96 wav」，
但实际 `LISTEN_POOL_WEIGHT` 三池各 1/3 → n=24 只能出 8/8/8（72 wav）。
改权重会改 `codec_rate_ab.py` 字节 → `sha256 de837b26` 断 → R4 三件套（`235158c7 /
aab304ec / 0cd9441d`）要重冻一次。

因此本脚本走旁路：**不改冻结生成器一个字节**，只读它已冻结的 `per_hit.csv`（R4，31 列），
按 PI ㉝ 定的 12/6/6 显式配额重选 24 条，直接把 4 臂渲染到独立目录。
选样逻辑全部 import 自 `codec_rate_ab`（`build_pools` / `_spread` / `_spread_by_source`），
不另写第二套排序，避免「第二份实现」。

配额来源：PI ㉝（cymbal 12 / hihats 6 / click_60_200ms 6），
hihats 走「池内 247 段重抽 6 条」而非旧 12 条重渲（旧包 PCM_16 字节不可复现）。

产物：
  wav/            96 个（24 击 × 4 臂 true/75/13c/13p，FLOAT）
  dup/            24 个 true 的字节副本（隐藏对照，**不计入 96**）
  index.json      逐条 5 个文件的 sha1
  composition.json 配额/池规模/来源数/逐条 picks/dup 清单

用法（远端，不落脚本文件）：
  base64 编码本文件后 `python -c "import base64;exec(base64.b64decode('...'))"`
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/gemini/code/lightfour-train")
OUT = Path(os.environ.get("LF_OUT", str(BASE / "outputs/listen_96_v0")))
PER_HIT = BASE / "outputs/codec_rate_ab/per_hit.csv"
SAMPLES = BASE / "data/sequence_samples.csv"
HITS = BASE / "data/sequence_hits.csv"
CODES = BASE / "data/seq_tokens/codes"
VOCALS = BASE / "data/bbox_raw/demucs_srv/htdemucs"
MODEL = BASE / "models/encodec_24khz"

# PI ㉝：12/6/6，共 24 击 × 4 臂 = 96 wav
QUOTA = {"cymbal": 12, "hihats": 6, "click_60_200ms": 6}

sys.path.insert(0, str(BASE / "scripts"))
import codec_rate_ab as cra          # noqa: E402  复用 rate_view / frame_rms / 选样
from codec_tokens import load_model  # noqa: E402

import librosa                       # noqa: E402
import soundfile as sf               # noqa: E402
import torch                         # noqa: E402

SR = cra.SR


def sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(PER_HIT, encoding="utf-8")))
    pools = cra.build_pools(rows)
    assert {k: len(v) for k, v in pools.items()} == {
        "cymbal": 38, "hihats": 247, "click_60_200ms": 2891}, {k: len(v) for k, v in pools.items()}

    picked, per_pool = [], {}
    for name, k in QUOTA.items():
        take = (cra._spread_by_source(pools[name], k) if name in ("cymbal", "hihats")
                else cra._spread(pools[name], k))
        assert len(take) == k, (name, len(take), k)
        picked += [(name, r) for r in take]
        per_pool[name] = take
    assert len(picked) == 24, len(picked)
    used = {(r["sequence_id"], r["hit_index"]) for _, r in picked}
    assert len(used) == 24, "24 条有重复击打"

    samples = {r["sequence_id"]: r for r in csv.DictReader(open(SAMPLES, encoding="utf-8"))}
    hits_by_seq = {}
    for r in csv.DictReader(open(HITS, encoding="utf-8")):
        hits_by_seq.setdefault(r["sequence_id"], []).append(r)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    model, _ = load_model(str(MODEL), dev)

    wav_dir = OUT / "wav"
    dup_dir = OUT / "dup"
    for d in (wav_dir, dup_dir):
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("*.wav"):
            stale.unlink()

    wav_cache = {}
    entries, skipped = [], []
    for pool, r in picked:
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
        lens = [len(a) for a in auds.values()]
        n = min(lens)
        trunc = 1.0 - n / max(lens)
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
        # dup 隐藏对照：true 的字节副本，单独目录，不计入 96
        dp = dup_dir / f"{hit['hit_id']}_h{hi}_dup.wav"
        dp.write_bytes((wav_dir / f"{hit['hit_id']}_h{hi}_true.wav").read_bytes())
        assert sha1_bytes(dp.read_bytes()) == files["true"]["sha1"]

        entries.append({
            "pool": pool, "hit_id": hit["hit_id"], "hit_index": hi, "sequence_id": sid,
            "source": src, "split": s.get("split", ""),
            "sound_id": r["sound_id"], "sound_notation": r["sound_notation"],
            "sound_family": r["sound_family"], "dur_ms": r["dur_ms"],
            "is_click_60_200": r["is_click_60_200"],
            "mel75_vs_true": r["mel75_vs_true"],
            "mel13c_vs_75": r["mel13c_vs_75"], "mel13p_vs_75": r["mel13p_vs_75"],
            "nmel75_vs_true": r["nmel75_vs_true"],
            "hf_true": r["hf_true"], "hf75": r["hf75"],
            "hf13c": r["hf13c"], "hf13p": r["hf13p"],
            "files": files,
            "dup": {"file": f"dup/{dp.name}", "sha1": files["true"]["sha1"]},
        })

    assert len(entries) == 24, (len(entries), skipped[:5])
    assert len(list(wav_dir.glob("*.wav"))) == 96, len(list(wav_dir.glob("*.wav")))
    assert len(list(dup_dir.glob("*.wav"))) == 24

    counts = {k: sum(1 for e in entries if e["pool"] == k) for k in QUOTA}
    composition = {
        "listen_n": 24,
        "quota_plan": QUOTA,
        "counts": counts,
        **counts,
        "n_wav": len(list(wav_dir.glob("*.wav"))),
        "n_dup": len(list(dup_dir.glob("*.wav"))),
        "arms": ["true", "75", "13c", "13p"],
        "dup_note": "dup/ = true 的字节副本，隐藏对照，不计入 96；判读率不达标整卡作废、不补样硬凑",
        "subtype": "FLOAT",
        "selection": "codec_rate_ab.build_pools → cymbal/hihats 用 _spread_by_source（每 source 至多 1 条），"
                     "click_60_200ms 用 _spread；族内按 mel13p_vs_75 排序均匀取（最差/中位/最好都覆盖）",
        "source_of_truth": "outputs/codec_rate_ab/per_hit.csv (live R4, 31 col)",
        "per_hit_md5": md5_file(PER_HIT),
        "generator": "codec_rate_ab.py 未修改（sha256 de837b26…），配额走旁路，不改冻结件",
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "pool_n_sources": {k: len({r["source"] for r in v}) for k, v in pools.items()},
        "n_sources_picked": {k: len({e["source"] for e in entries if e["pool"] == k}) for k in QUOTA},
        "warning": "click_60_200ms 是时长桶（跨 12 family，非 clicks 家族）；本包只供人耳试听，"
                   "不构成任何 family 已验收的证明",
        "skipped": skipped,
    }
    (OUT / "index.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    (OUT / "composition.json").write_text(json.dumps(composition, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps({"n": len(entries), "wav": composition["n_wav"], "dup": composition["n_dup"],
                      "counts": counts, "n_sources": composition["n_sources_picked"],
                      "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
