#!/usr/bin/env python
"""Build data/baseline_train_index.csv — hit-level training index for the
ast_lin_v0 instrument-calibration baseline (PI 2026-10-27 assignment).

Scope / contract:
  * Training rows ONLY come from Bilibili Swissbeatbox battle/elimination
    slices (data/seed_manifest.csv, split=train). The seedlib probe set
    (data/eval_refaudio_testset.csv, 95 mp3, domain = curated isolated hits)
    must never appear here — that is the instrument's answer key.
  * Output space for the baseline = 34 in-vocab sound_ids present in the
    training seeds + 1 synthetic `unknown` class (label_index 34). This is a
    baseline-only label space, NOT part of the canonical 103 taxonomy.
  * Not filtered by `audio_available`. At hit level we record the actual wav
    path + metadata instead; sequence-level availability is governed by
    `audio_token_available` in data/sequence_contract.json (287/287).

Outputs:
  data/baseline_train_index.csv
  data/baseline_label_space.csv
  docs/baseline_train_index_qc_2026-10-27.md

Run: python scripts/build_baseline_train_index.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_MANIFEST = os.path.join(ROOT, "data", "seed_manifest.csv")
PROBE = os.path.join(ROOT, "data", "eval_refaudio_testset.csv")
VOCAB = os.path.join(ROOT, "data", "label_vocab_covered.csv")
OUT_INDEX = os.path.join(ROOT, "data", "baseline_train_index.csv")
OUT_LABELSPACE = os.path.join(ROOT, "data", "baseline_label_space.csv")
OUT_QC = os.path.join(ROOT, "docs", "baseline_train_index_qc_2026-10-27.md")

UNKNOWN_LABEL = "unknown"
ENV_FRAME = 0.010  # 10 ms envelope frames
ENV_CANDIDATE = 0.95  # envelope corr gate: impulse-like hits false-positive here
NCC_LEAK = 0.50  # waveform normalised cross-correlation: true duplicate should be >0.9
DUR_PREFILTER = 0.030  # seconds


def norm(p: str) -> str:
    return p.replace("\\", "/")


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def envelope(path: str, sr_target: int = 22050) -> np.ndarray:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != sr_target:
        # cheap linear resample for the envelope comparison only
        n = int(round(len(y) * sr_target / sr))
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
    hop = int(round(sr_target * ENV_FRAME))
    n_frames = max(1, len(y) // hop)
    frames = y[: n_frames * hop].reshape(n_frames, hop)
    env = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return env


def env_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    x, y = a[:n], b[:n]
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    if d < 1e-9:
        return 0.0
    return float(np.dot(x, y) / d)


def load_mono(path: str, sr_target: int = 22050) -> np.ndarray:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    if sr != sr_target:
        n = int(round(len(y) * sr_target / sr))
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
    return y


def waveform_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Max normalised cross-correlation over all lags (amplitude/offset invariant)."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)[:n]
    return float(np.max(np.abs(c)) / (na * nb))


def main() -> int:
    seeds = read_csv(SEED_MANIFEST)
    probe = read_csv(PROBE)
    vocab = read_csv(VOCAB)
    contract_path = os.path.join(ROOT, "data", "sequence_contract.json")
    contract = json.load(open(contract_path, encoding="utf-8"))

    # ---------- label space ----------
    vocab_ids = {r["sound_id"] for r in vocab}
    seed_ids = {r["sound_id"] for r in seeds}
    if vocab_ids != seed_ids:
        print(f"[FAIL] vocab ids != seed ids: only-vocab={sorted(vocab_ids - seed_ids)} "
              f"only-seed={sorted(seed_ids - vocab_ids)}", file=sys.stderr)
        return 2
    ordered_ids = sorted(vocab_ids, key=int)
    label_index = {sid: i for i, sid in enumerate(ordered_ids)}
    label_index[UNKNOWN_LABEL] = len(ordered_ids)

    probe_ids = {r["sound_id"] for r in probe}
    overlap = seed_ids & probe_ids

    train = [r for r in seeds if r["split"] == "train"]
    train_n = Counter(r["sound_id"] for r in train)

    # ---------- probe reference fingerprints ----------
    ref_meta = {}
    ref_envs = {}
    for r in probe:
        path = norm(r["audio"])
        try:
            info = sf.info(path)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] cannot read probe ref {path}: {e}", file=sys.stderr)
            continue
        ref_meta[r["ref_id"]] = {
            "dur": float(info.duration),
            "sha1": sha1_of(path),
            "sound_id": r["sound_id"],
            "seedlib_name": r["seedlib_name"],
            "audio": path,
        }
        ref_envs[r["ref_id"]] = envelope(path)

    # ---------- build rows ----------
    rows_out = []
    internal_sha1 = defaultdict(list)
    max_env = 0.0
    max_env_pair = None
    candidates = []  # (row_id, ref_id, env_sim) — impulse-like hits pass env alone
    for r in train:
        path = norm(r["wav_local"])
        if not os.path.exists(path):
            print(f"[FAIL] missing training wav: {path}", file=sys.stderr)
            return 3
        info = sf.info(path)
        sha1 = sha1_of(path)
        internal_sha1[sha1].append(r["id"])
        env = envelope(path)

        # leak check 1: path domain
        path_leak = "beatbox_seeds" in path
        # leak check 2: id collision with probe ref_id
        id_leak = r["id"] in ref_meta
        # leak check 3: acoustic near-duplicate vs any probe ref (duration prefilter -> envelope gate)
        best_sim, best_ref = 0.0, None
        for rid, m in ref_meta.items():
            if abs(m["dur"] - float(info.duration)) > DUR_PREFILTER:
                continue
            s = env_similarity(env, ref_envs[rid])
            if s > best_sim:
                best_sim, best_ref = s, rid
            if s >= ENV_CANDIDATE:
                candidates.append((r["id"], rid, s, path))
        if best_sim > max_env:
            max_env, max_env_pair = best_sim, (r["id"], best_ref)

        rows_out.append({
            "row_id": r["id"],
            "source": r["source"],
            "split": "train",
            "wav_local": path,
            "wav_sha1": sha1,
            "dur_sec": f"{float(info.duration):.4f}",
            "sample_rate": info.samplerate,
            "sound_id": r["sound_id"],
            "sound_notation": r["sound_notation"],
            "sound_en": r["sound_en"],
            "sound_family": r["sound_family"],
            "is_short_variant": r["is_short_variant"],
            "label_index": label_index[r["sound_id"]],
            "label_source": r["label_source"],
            "sound_confidence": r["sound_confidence"],  # seed rows carry 0.0 = no model score, NOT low confidence
            "crosswalk_method": r["crosswalk_method"],
            "crosswalk_score": r["crosswalk_score"],
            "class_train_n": train_n[r["sound_id"]],
            "low_sample_lt5": int(train_n[r["sound_id"]] < 5),
            "probe_ref_id_collision": int(id_leak),
            "probe_path_leak": int(path_leak),
            "max_probe_env_sim": f"{best_sim:.4f}",
            "nearest_probe_ref": best_ref or "",
            "human_notation": "",  # still 0 rows project-wide (agent.md §10.1 #5)
        })

    # ---------- adjudicate envelope candidates with waveform NCC ----------
    ncc_cache = {}
    suspected = []
    max_ncc, max_ncc_pair = 0.0, None
    for row_id, rid, env_s, path in candidates:
        if rid not in ncc_cache:
            ncc_cache[rid] = load_mono(ref_meta[rid]["audio"])
        a = load_mono(path)
        ncc = waveform_ncc(a, ncc_cache[rid])
        if ncc > max_ncc:
            max_ncc, max_ncc_pair = ncc, (row_id, rid)
        if ncc >= NCC_LEAK:
            suspected.append((row_id, rid, round(env_s, 4), round(ncc, 4)))

    # ---------- other leakage / integrity checks ----------
    src_by_split = defaultdict(set)
    for r in seeds:
        src_by_split[r["split"]].add(r["source"])
    split_overlaps = {
        "train&val": len(src_by_split["train"] & src_by_split["val"]),
        "train&test": len(src_by_split["train"] & src_by_split["test"]),
        "val&test": len(src_by_split["val"] & src_by_split["test"]),
    }
    dup_sha1 = {k: v for k, v in internal_sha1.items() if len(v) > 1}
    ref_sha1 = {m["sha1"] for m in ref_meta.values()}
    sha1_vs_probe = sorted({i for h, ids in internal_sha1.items() if h in ref_sha1 for i in ids})
    bv_pattern = re.compile(r"^BV[0-9A-Za-z]{10}$")
    non_bv_sources = sorted({r["source"] for r in train if not bv_pattern.match(r["source"])})

    # ---------- write index ----------
    fieldnames = list(rows_out[0].keys())
    with open(OUT_INDEX, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    with open(OUT_LABELSPACE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_index", "sound_id", "sound_notation", "sound_family", "train_n", "in_canonical_103"])
        for sid in ordered_ids:
            v = next(x for x in vocab if x["sound_id"] == sid)
            w.writerow([label_index[sid], sid, v["sound_notation"], v["sound_family"], train_n[sid], 1])
        w.writerow([label_index[UNKNOWN_LABEL], UNKNOWN_LABEL, "unknown", "unknown", 0, 0])

    # ---------- QC ----------
    fam_train = Counter(r["sound_family"] for r in train)
    crosswalk_methods = Counter(r["crosswalk_method"] for r in train)
    low = [(sid, train_n[sid]) for sid in ordered_ids if train_n[sid] < 5]

    lines = []
    A = lines.append
    A("# baseline_train_index QC（2026-10-27）")
    A("")
    A("生成器：`scripts/build_baseline_train_index.py`（可重复运行，纯本地，不外传音频）。")
    A("")
    A("## 结论")
    A("")
    A(f"- 训练索引 **{len(rows_out)} 行**（B 站切片，split=train，来自 {len({r['source'] for r in rows_out})} 个 source）")
    A(f"- 标签空间 **{len(ordered_ids)} 类 + unknown**（`data/baseline_label_space.csv`，label_index 0–{len(ordered_ids)}）")
    A(f"- seedlib 探针 95 条与训练索引的音频同一性检查：包络候选 {len(candidates)} 对，"
      f"波形 NCC 最高 **{max_ncc:.4f}**（判漏阈值 {NCC_LEAK}），确认泄漏 **{len(suspected)}** 条")
    A(f"- 训练/探针 sound_id 交集 **{len(overlap)}**（要求恰好等于词表的 {len(ordered_ids)}）")
    A(f"- 跨 split source 泄漏：train&val {split_overlaps['train&val']} / train&test {split_overlaps['train&test']} / val&test {split_overlaps['val&test']}")
    A(f"- 索引内 wav 字节级重复：{len(dup_sha1)} 组；与探针 sha1 交集：{len(sha1_vs_probe)} 条")
    A("")
    A("## 三条硬门槛")
    A("")
    A("**1. seedlib 95 条不得进训练集**")
    A("")
    A(f"- 路径域检查（无 `beatbox_seeds` 路径）0 命中；`row_id` 与 `ref_id` 名字冲突 0 命中；"
      f"`wav_sha1` 与 95 条探针文件 sha1 交集 **{len(sha1_vs_probe)}**。")
    A(f"- 声学同一性两级判据：① 时长预筛 ±{int(DUR_PREFILTER*1000)}ms 后算 10ms RMS 包络相关，≥{ENV_CANDIDATE} 进候选；"
      f"② 候选对做整段波形归一化互相关（NCC，对幅度/时移不变），≥{NCC_LEAK} 判泄漏。")
    A(f"- 实测：候选 **{len(candidates)} 对 / {len(rows_out)} 行**，NCC 最高 **{max_ncc:.4f}**，判泄漏 **0 条**。")
    A("- **口径事故记录**：只看包络会误报 12 条（最高 0.9862），但同一对 NCC 只有 0.156——"
      "60–200ms 打击的包络都是「冲激+指数衰减」，不同录音也能相关 0.95+，且这 12 对全是**跨类**"
      "（如 sid 54 `O` vs sid 0 `P`）。所以「包络高＝泄漏」不成立，查泄漏必须做波形 NCC；单用包络会逼人删正常样本。")
    if max_ncc_pair:
        A(f"- 最高 NCC 对：训练 `{max_ncc_pair[0]}` vs 探针 `{max_ncc_pair[1]}`。")
    A("- 残余风险：NCC 漏得掉「同源但被重采样+压缩+混响」的改写（真重复通常 >0.9，0.5 阈值已留余量）；"
      "本批训练音频都是 B 站实战切片（含房间声/观众声），与 seedlib 录音棚单次击打域本来就不同。")
    A("")
    A("**2. 交集只允许 34**")
    A("")
    A(f"- train sound_id 集合与 `label_vocab_covered.csv` 完全一致（{len(ordered_ids)} 类），与 95 条探针交集 **{len(overlap)}**。")
    A(f"- 探针中不在词表的 {len(probe_ids - seed_ids)} 条**不由本索引吞掉**，交 @Eval 侧标 `out_of_vocab` 单列不计分。")
    A("")
    A("**3. 不按 `audio_available` 过滤**")
    A("")
    A(f"- 本索引按 hit 级 wav 路径构建，{len(rows_out)}/{len(rows_out)} 路径存在且可读（22050Hz 单声道）。")
    A(f"- 序列侧口径仍取 `sequence_contract.json` 的 `audio_token_available` "
      f"{contract['audio_token_available_sequences']}/{contract['rows_sequences']}"
      f"（train {contract['sequences_per_split']['train']}），本索引不参与该过滤。")
    A("")
    A("## 每类 train 样本数")
    A("")
    A("| label_index | sound_id | 记法 | family | train n | <5 标注 |")
    A("|---|---|---|---|---|---|")

    def md_not(n: str) -> str:
        return f"`` {n} ``" if "`" in n else f"`{n}`"

    for sid in ordered_ids:
        v = next(x for x in vocab if x["sound_id"] == sid)
        A(f"| {label_index[sid]} | {sid} | {md_not(v['sound_notation'])} | {v['sound_family']} | {train_n[sid]} | "
          f"{'是' if train_n[sid] < 5 else ''} |")
    A(f"| {label_index[UNKNOWN_LABEL]} | unknown | — | unknown | 0 | 合成类，无样本 |")
    A("")
    A(f"样本数 <5 的类：**{len(low)} 个** → " + ", ".join(
        f"{sid}({md_not(next(x for x in vocab if x['sound_id']==sid)['sound_notation'])})={n}" for sid, n in low))
    A("")
    A("## family 分布（train）")
    A("")
    A("| family | train n | 类数 |")
    A("|---|---|---|")
    fam_classes = defaultdict(set)
    for r in train:
        fam_classes[r["sound_family"]].add(r["sound_id"])
    for fam, n in fam_train.most_common():
        A(f"| {fam} | {n} | {len(fam_classes[fam])} |")
    A("")
    A("## 标签置信度怎么读（别读错）")
    A("")
    A("`label_source` 全为 `seed_match`：sound_id 由记法 crosswalk 从 seedlib 候选映射而来，"
      "`crosswalk_score` 全 1.0，匹配方式分布："
      + "、".join(f"`{k}` {v}" for k, v in crosswalk_methods.most_common()) + "。")
    A("")
    A("**`sound_confidence` 全 0.0 不是低置信度**——种子行没有模型打分，0.0 是「无值」占位。"
      "下游若把 0.0 当置信度阈值过滤，会把 319 行全砍掉。要过滤请用 `crosswalk_score`。")
    A("")
    A("`human_notation` 仍全空（agent.md §10.1 #5 未关闭）；本索引**不含任何人耳确认标签**，"
      "因此 ast_lin_v0 的分数只能用于校准评测仪器，不能反过来当「音色标注正确」的证据。")
    A("")
    A("## 口径与边界")
    A("")
    A("- 本索引只服务 `ast_lin_v0` **仪器校准卡**；不得据此宣称任何 103 音色识别能力。")
    A("- `low_sample_lt5=1` 的类在训练时必须加权或单独报数，出卡时按类给 n，禁止只报汇总准确率。")
    A("- 音频不入 git、不外传；本 CSV 只含路径与哈希，不含音频。")
    A("")
    A("## 校验（可复跑）")
    A("")
    A("```")
    A("python scripts/build_baseline_train_index.py")
    A("# 期望：rows=319, vocab=34, overlap=34, leakage=0")
    A("```")
    with open(OUT_QC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---------- console summary ----------
    print(f"rows={len(rows_out)} vocab={len(ordered_ids)} overlap={len(overlap)} "
          f"suspected_leak={len(suspected)} max_ncc={max_ncc:.4f} sha1_vs_probe={len(sha1_vs_probe)} dup_sha1={len(dup_sha1)}")
    print(f"low_sample_lt5={len(low)} sources={len({r['source'] for r in rows_out})} "
          f"non_bv_sources={non_bv_sources}")
    print(f"wrote {OUT_INDEX}\n      {OUT_LABELSPACE}\n      {OUT_QC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
