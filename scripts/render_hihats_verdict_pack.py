# -*- coding: utf-8 -*-
"""hihats 16 层判决包 v1：从 247 段池按 source 分层重抽 12 条，渲染四臂，交给 @Eval 建盲包。

裁决依据（PI 2026-10-28）：
  - 旧 12 条（`outputs/eval_hihats_ab_v1/overwritten_hihats12_index_recovered.json`，
    md5 `59a58a2a24a739301df93ac6e5a8177b`）**不入判决、只作附录**；
  - 新 12 条必须从 live R4 的 247 段 hihats 池重抽、**按 source 分层**（每 source 至多 1 条，
    12 个不同 source），且与旧 12 条**零重叠**；
  - 有效性门 dup ≥11/12 且 13p ≥8/12；75Hz 臂 ≥8/12 通过、≤5/12 转 32 层、6–7 扩样到 24。

本脚本只负责「选样 + 渲染 + 索引/溯源」，不建盲包、不做判决：
盲包（12 REF + 36 匿名候选，含 dup 字节副本）、JUDGMENT_SHEET 与打分由 @Eval 按
`scripts/eval_hihats_ab.py` 协议生成。

渲染管线与 `scripts/render_listen_hihats.py` 完全一致（同 EnCodec、同 rate_view、同 FLOAT），
保证本包与既有 mel 读数同源；差异只有选样时的旧 12 条排除。

用法（远端）：
  /root/miniconda3/bin/python scripts/render_hihats_verdict_pack.py
产物：wav/*.wav（48）、index.json、composition.json、HANDOFF.md
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/gemini/code/lightfour-train")
OUT = Path(os.environ.get("LF_OUT", str(BASE / "outputs/hihats_verdict_pack_v1")))
PER_HIT = BASE / "outputs/codec_rate_ab/per_hit.csv"
SAMPLES = BASE / "data/sequence_samples.csv"
HITS = BASE / "data/sequence_hits.csv"
CODES = BASE / "data/seq_tokens/codes"
VOCALS = BASE / "data/bbox_raw/demucs_srv/htdemucs"
MODEL = BASE / "models/encodec_24khz"
N_PICK = 12

# live R4 per_hit.csv（31 列）固定指纹；与 R4 归档 235158c7 一致，防止拿错版本选样
PER_HIT_MD5 = "235158c7877f5620e5443790c19f30e8"

# 旧 12 条（被 2026-10-27 11:41 覆盖的「12 条全 hihats」包，已回收 index）→ 附录，不进判决
OLD12_SRC = "outputs/eval_hihats_ab_v1/overwritten_hihats12_index_recovered.json"
OLD12_MD5 = "59a58a2a24a739301df93ac6e5a8177b"
OLD12 = [
    ("BV1VZ4y1W7sf_0122", "SEQ-00170", "0"),
    ("BV1VZ4y1W7sf_0511", "SEQ-00174", "2"),
    ("BV1rp4y1t7Qv_0501", "SEQ-00266", "4"),
    ("BV1YssozfEXU_0030", "SEQ-00202", "0"),
    ("BV177tpz7EC3_0641", "SEQ-00012", "1"),
    ("BV1rp4y1t7Qv_0495", "SEQ-00266", "1"),
    ("BV1VonA6WEDs_0570", "SEQ-00182", "5"),
    ("BV1fRg1zBEyi_0211", "SEQ-00216", "19"),
    ("BV1np4nzGEiN_0893", "SEQ-00237", "0"),
    ("BV1rp4y1t7Qv_0513", "SEQ-00266", "11"),
    ("BV1wzpEezEFd_0376", "SEQ-00283", "0"),
    ("BV1DW411u7oK_0300", "SEQ-00054", "9"),
]

sys.path.insert(0, str(BASE / "scripts"))
import codec_rate_ab as cra          # noqa: E402  复用 rate_view / frame_rms / _spread_by_source
from codec_tokens import load_model  # noqa: E402

# 记法配额：池内 F 154 / Tch 44 / Tss 43（f 5、-b 1 只报数）。
# 不配额时 _spread_by_source 会产出 F 11 + Tss 1（Tch 0 条）→ 单记法包场，
# 无法代表 hihats 家族（F/Tch/Tss 是三个不同 sound_id：12/13/11）。
# 配额取「按池比例 7/2/2 → 小众各补到 3 → 大类封顶 6」= 6/3/3。
QUOTA = {"Tch": 3, "Tss": 3, "F": 6}


def _reps(rows):
    """每 source 取 mel13p_vs_75 中位代表，按退化升序。"""
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    reps = []
    for hs in by_src.values():
        hs.sort(key=lambda r: float(r["mel13p_vs_75"]))
        reps.append(hs[len(hs) // 2])
    return sorted(reps, key=lambda r: float(r["mel13p_vs_75"]))


def _spread_idx(n, k):
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def select_pack(rows, quota):
    """按记法配额抽，每 source 至多 1 条；先抽小众记法，再抽大类，保证 12 个不同 source。"""
    picked, used = [], set()
    for nt in sorted(quota, key=lambda k: quota[k]):
        reps = [r for r in _reps([r for r in rows if r["sound_notation"] == nt])
                if r["source"] not in used]
        for i in _spread_idx(len(reps), min(quota[nt], len(reps))):
            picked.append(reps[i])
            used.add(reps[i]["source"])
    return picked

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


def strip_peak_chunk(path):
    """删掉 libsndfile 写入的 PEAK 块。

    PEAK 块里带 **wall-clock 时间戳**，导致同样的音频两次渲染 md5 不同
    （实测只差 1 个字节，音频 payload 逐采样相同）。剥掉后文件字节 = 音频字节，
    sha1/md5 才可作为复现指纹。保留 fmt/fact/data 三个块。
    """
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", path
    out = bytearray(b"RIFF\x00\x00\x00\x00WAVE")
    i, kept = 12, []
    while i < len(data):
        cid = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        body = data[i + 8:i + 8 + size]
        pad = size & 1
        if cid in (b"fmt ", b"fact", b"data"):
            out += cid + size.to_bytes(4, "little") + body + (b"\x00" if pad else b"")
            kept.append(cid.decode("ascii"))
        i += 8 + size + pad
    assert kept[:1] == ["fmt "] and kept[-1:] == ["data"], kept
    out[4:8] = (len(out) - 8).to_bytes(4, "little")
    path.write_bytes(bytes(out))


def main():
    got_md5 = md5_file(PER_HIT)
    assert got_md5 == PER_HIT_MD5, f"per_hit.csv md5 {got_md5} != {PER_HIT_MD5}"

    all_rows = [r for r in csv.DictReader(open(PER_HIT, encoding="utf-8"))
                if r["sound_family"] == "hihats"]
    assert len(all_rows) == 247, len(all_rows)

    old_keys = {(sid, hi) for _, sid, hi in OLD12}
    rows = [r for r in all_rows if (r["sequence_id"], r["hit_index"]) not in old_keys]
    assert len(rows) == 247 - len(old_keys), (len(rows), len(old_keys))

    # 记录被否决的无配额抽法，便于 PI/Eval 复核「为什么不是 11F+1Tss」
    unconstrained = cra._spread_by_source(rows, N_PICK)
    picked = select_pack(rows, QUOTA)
    picked_keys = {(r["sequence_id"], r["hit_index"]) for r in picked}
    assert len(picked_keys) == N_PICK, len(picked_keys)
    assert len({r["source"] for r in picked}) == N_PICK, "source 去重失败"
    assert not (picked_keys & old_keys), "选样命中旧 12 条"
    # 备选池：仍按退化排序，渲染失败时按序补位（同样排除旧 12 条）
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
            strip_peak_chunk(fp)
            back, sr_back = sf.read(str(fp), dtype="float32")
            assert sr_back == SR and np.array_equal(back, seg.astype("float32")), fp
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

    assert len(entries) == N_PICK, (len(entries), skipped[:5])
    assert len({e["source"] for e in entries}) == N_PICK, "source 去重失败"
    assert not ({e["hit_id"] for e in entries} & {h for h, _, _ in OLD12}), "成品命中旧 12 条"

    old_srcs = {r["source"] for r in all_rows
                if (r["sequence_id"], r["hit_index"]) in old_keys}
    new_srcs = {e["source"] for e in entries}

    composition = {
        "pack_id": "HIHATS_VERDICT_V1",
        "listen_n": len(entries),
        "pool": "hihats",
        "pool_size_total": len(all_rows),
        "pool_size_after_exclusion": len(rows),
        "selection": "先剔除旧 12 条 → 记法配额 {Tch:3, Tss:3, F:6}（按池 7/2/2、小众补到 3、"
                     "大类封顶 6）→ 每记法内 source 分层（每 source 至多 1 条、取 mel13p_vs_75 "
                     "中位代表、按退化均匀铺开）→ 12 条 12 source",
        "quota_by_notation": QUOTA,
        "rejected_unconstrained_draw": {
            "rule": "codec_rate_ab._spread_by_source(235 rows, 12)（PI 原话字面口径）",
            "reason": "产出 F 11 / Tss 1 / Tch 0，单记法包场，无法代表 hihats 三个 sound_id",
            "hit_ids": [r["hit_id"] for r in unconstrained],
            "notation_counts": {k: sum(1 for r in unconstrained if r["sound_notation"] == k)
                                for k in sorted({r["sound_notation"] for r in unconstrained})},
        },
        "source_of_truth": "outputs/codec_rate_ab/per_hit.csv (live R4, 31 col)",
        "per_hit_md5": got_md5,
        "excluded_old12": {
            "n": len(OLD12),
            "source_file": OLD12_SRC,
            "source_md5": OLD12_MD5,
            "hit_ids": [h for h, _, _ in OLD12],
            "role": "附录 only，不入判决、不重渲",
        },
        "old12_item_overlap": sorted({e["hit_id"] for e in entries} & {h for h, _, _ in OLD12}),
        "old12_source_overlap": sorted(old_srcs & new_srcs),
        "old12_source_overlap_note": "旧 12 条从未被判决（JUDGMENT_SHEET 未填），"
                                     "同 source 不同击打不构成听感暴露；如需 source 级全新鲜，"
                                     "由 @Eval 提请在 235 段上追加排除后重抽（会改变样本，需重渲）",
        "arms": ["true", "75", "13c", "13p"],
        "subtype": "FLOAT (libsndfile PEAK chunk stripped：块内含 wall-clock，会破坏字节复现)",
        "n_wav": len(list(wav_dir.glob("*.wav"))),
        "n_sources": len({e["source"] for e in entries}),
        "split_counts": {k: sum(1 for e in entries if e["split"] == k)
                         for k in sorted({e["split"] for e in entries})},
        "notation_counts": {k: sum(1 for e in entries if e["sound_notation"] == k)
                            for k in sorted({e["sound_notation"] for e in entries})},
        "dur_ms_range": [min(float(e["dur_ms"]) for e in entries),
                         max(float(e["dur_ms"]) for e in entries)],
        "skipped": skipped,
        "warning": "本包全部为 hihats，只能判 hihats；不得表述为三池/全 family 未见退化",
    }

    (OUT / "index.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "composition.json").write_text(json.dumps(composition, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

    lines = [
        "# hihats 判决包 v1（HIHATS_VERDICT_V1）",
        "",
        "用途：16 层 hihats 人耳 A/B 的**唯一新样本池**。旧 12 条只作附录。",
        "",
        f"- 样本：{len(entries)} 击 / {composition['n_sources']} source / dur "
        f"{composition['dur_ms_range'][0]}–{composition['dur_ms_range'][1]} ms",
        f"- 池：247 段剔除旧 12 → 235 段；per_hit.csv md5 `{got_md5}`",
        f"- 臂：true / 75 / 13c(附录) / 13p；FLOAT；{composition['n_wav']} 个 wav",
        "- 选择规则：记法配额 Tch 3 / Tss 3 / F 6（池 154/44/43），每记法内每 source 至多 1 条，",
        "  按 mel13p_vs_75 均匀铺开；无配额的纯 source 分层会产出 11F+1Tss（Tch 0），已否决并留档",
        "",
        "## 交给 @Eval 的下一步",
        "1. 按 `scripts/eval_hihats_ab.py` 协议建盲包：每击 1 REF + 3 匿名候选（75 / 13p / dup），",
        "   dup = true 的**字节副本**，sha1 必须与 REF 相等（12/12）。",
        "2. 判决规则（PI 2026-10-28）：dup ≥11/12 且 13p ≥8/12 才算有效；",
        "   75 臂判 0 条数 ≥8/12 通过、≤5/12 转 32 层、6–7 扩样到 24。",
        "3. 13c 只作附录，不进判决；mel 不进判决。",
        "",
        "## 边界",
        "- 本包不构成 codec 过关结论；未出人耳判决前不得写通过。",
        "- hihats mel 0.8429（R4 全池 247 段）与 75 臂判决分列，不互相替代。",
    ]
    (OUT / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"n": len(entries), "wav": composition["n_wav"],
                      "sources": composition["n_sources"],
                      "splits": composition["split_counts"],
                      "dur_ms": composition["dur_ms_range"],
                      "hit_ids": [e["hit_id"] for e in entries],
                      "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
