# -*- coding: utf-8 -*-
"""Eval 交付：hihats 12 条「true vs 75 vs 13p」盲听包 + 逐条感知指标（16 层判决）。

任务来源：PI 裁决⑨–⑭ 条目「对 12 条 hihats 试听包做 true vs 75 vs 13p，这是 16 层生死的唯一判决」。

口径（写卡时同样生效）：
  - mel 只准出自 scripts/mel_metric.py（@Eval 已签收，md5 240f6f18），frame_policy=min（默认）。
  - 指标一律从 **磁盘 wav（PCM_16）** 重算，与 Delight 实际听到的是同一信号；
    与 Codec 内存 float 路径的存储值差 ≤0.0585 属 PCM_16 量化，不是实现差异。
  - 75 臂 = EnCodec 24kHz/24kbps 码流取前 16 层（raw 32 层，等效 ≈12kbps，Critic M3 口径），
    16 vs 32 不构成控制变量，本卡只判「16 层在 hihats 上够不够」。
  - 13c 不进本判决（appendix only），本卡只做 true / 75 / 13p 三臂。
  - frame_diff_max=0 是 n=min(四臂) 在先的结构性恒等，不得当对齐/截断已证。
  - R1 对齐臂唯一命名 `13p_align`（结论值 150.4% = ±1 桶粒度、±3 帧级别未测；对齐前 155.4%，同一臂前后一对）；
    试听 wav 臂名沿用 `13p`（仅作样本标识）；其余 `13p_*` 名禁止入卡，更细粒度只准 `13p_subalign` 且须同表并注不可比。
  - 中间带单列 `n_midband`（judgment=1「可辨·轻微」），卡头标「未判定」，不进通过/不通过汇总、不进 §0 或结论段。
  - hihats 判决出口只有「扩样复判 / 记 v0 已知短板」，32 层评估出口已删除、永不触发（PI ㉝）；
    判决样本 = 正式 96 wav 包（counts 12/6/6）hihats 池 6 条、现包 247 段重抽，不走旧 12 条重渲；
    n=6 结论限「hihats 未见退化（低置信）」，与「mel 0.8429 越 §8 0.8 线」两行并列在卡头；
    扩样 12 条同池同四臂、只准一次（PI 62②）：≥10/12 → 只准写「hihats 未见退化（低置信）」；
    7–9/12 → 结案「证据不足以判定」+ 记 v0 已知短板、不再扩样；≤6/12 → 记「hihats 退化，n=12」。
    分族只报数（PI 62③）：cymbal / click 不设通过线、不标置信档；cymbal ≥10/12 判 0 只进「观察记录」；
    总判决 = hihats 门 + dup 门，其余族进失败案例清单。

盲听协议（per item：1 个 REF + 3 个匿名候选，候选含 75 / 13p / dup-true 隐藏对照）：
  judgment: 0 = 与 REF 听不出差别 / 1 = 可辨·轻微 / 2 = 可辨·明显
  正对照 13p 应判 ≥1（n=12 维持 ≥8/12；n=6 阶段只报数，PI 61/62 未给正对照数字，Eval 不自创）；
  负对照 dup（REF 字节副本）**满额零容错**（PI 61①：当轮全部隐藏 dup 必须全判 0，n=6 → 6/6、
  n=12 → 12/12、正式包 24 击 → 24/24；差 1 条整卡作废、不补样不扩样）。

用法：
  python scripts/eval_hihats_ab.py --build      # 建包 + 指标 + 自检
  python scripts/eval_hihats_ab.py --verify     # 独立二次校验（不复用 build 的中间结果）
  python scripts/eval_hihats_ab.py --score      # Delight 填完 JUDGMENT_SHEET.csv 后出判决卡
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mel_metric  # noqa: E402  唯一 mel 实现（评测红线）

BASE = HERE.parent
SRC_PACK = BASE / "outputs/codec_rate_ab/listen"
OUT = BASE / "outputs/eval_hihats_ab_v1"
BLIND = OUT / "listen_blind"

CARD_ID = "CODEC_LISTEN_HIHATS_12"
SEED = 20261027
SR = mel_metric.SR
HOP = mel_metric.HOP
HF_HZ = 6000.0            # 高频能量占比的分界（hihats 主能量区）
JUDGMENT_SCALE = {"0": "与 REF 听不出差别", "1": "可辨·轻微", "2": "可辨·明显"}
LISTEN_ARMS = ["75", "13p"]          # 进盲听
APPENDIX_ARMS = ["75", "13c", "13p"]  # 进指标表（13c 仅附录）


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
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


def read_wav(p: Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    assert sr == SR, (p, sr)
    return np.asarray(y, dtype=np.float32).ravel()


def rms_envelope(y: np.ndarray) -> np.ndarray:
    n = len(y) // HOP
    return np.array([float(np.sqrt(np.mean(y[i * HOP:(i + 1) * HOP] ** 2)))
                     for i in range(n)], dtype=np.float64)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)
    if n < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def onset_ms(y: np.ndarray) -> float:
    """能量起振：首帧 RMS > 20% 峰值 RMS 的时刻。短击打只 ~10 帧，只用于逐条比较。"""
    env = rms_envelope(y)
    if env.size == 0 or env.max() <= 0:
        return float("nan")
    idx = int(np.argmax(env > 0.2 * env.max()))
    return 1000.0 * idx * HOP / SR


def hf_share(y: np.ndarray) -> float:
    """全信号周期谱中 >6kHz 的能量占比。"""
    if len(y) < 8:
        return float("nan")
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / SR)
    tot = float(spec.sum())
    return float(spec[freqs >= HF_HZ].sum() / tot) if tot > 0 else float("nan")


def flux_series(y: np.ndarray) -> np.ndarray:
    m = mel_metric.logmel(y)
    d = np.diff(m, axis=1)
    return np.maximum(d, 0.0).sum(axis=0)


def arm_metrics(ref: np.ndarray, arm: np.ndarray) -> dict:
    cmp_ = mel_metric.compare(ref, arm, "min")
    e_ref, e_arm = rms_envelope(ref), rms_envelope(arm)
    return {
        "mel_l1": cmp_["value"],
        "mel_frame_diff": cmp_["frame_diff"],
        "mel_truncated": cmp_["truncated"],
        "env_corr": pearson(e_ref, e_arm),
        "flux_corr": pearson(flux_series(ref), flux_series(arm)),
        "onset_ref_ms": round(onset_ms(ref), 2),
        "onset_arm_ms": round(onset_ms(arm), 2),
        "onset_delta_ms": (None if np.isnan(onset_ms(ref)) or np.isnan(onset_ms(arm))
                           else round(onset_ms(arm) - onset_ms(ref), 2)),
        "hf_ref": round(hf_share(ref), 5),
        "hf_arm": round(hf_share(arm), 5),
        "hf_delta": (None if np.isnan(hf_share(ref)) or np.isnan(hf_share(arm))
                     else round(hf_share(arm) - hf_share(ref), 5)),
    }


def med(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.median(v)), 5) if v else None


def mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.mean(v)), 5) if v else None


# ------------------------------------------------------------------ build
def build() -> int:
    import soundfile as sf

    idx = json.loads((SRC_PACK / "index.json").read_text(encoding="utf-8"))
    assert len(idx) == 12, f"预期 12 条，实际 {len(idx)}"
    comp = json.loads((SRC_PACK / "composition.json").read_text(encoding="utf-8"))
    pool_of = {p["hit_id"]: p["pool"] for p in comp["picks"]}
    fam_of = {p["hit_id"]: p["sound_family"] for p in comp["picks"]}

    if BLIND.exists():
        shutil.rmtree(BLIND)
    BLIND.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)

    metric_rows, manifest, key, sheet = [], [], {}, []
    src_file_count = 0
    for i, r in enumerate(idx):
        base = f"{r['hit_id']}_h{r['hit_index']}"
        srcs = {a: SRC_PACK / f"{base}_{a}.wav" for a in ("true", "75", "13c", "13p")}
        for p in srcs.values():
            assert p.exists(), p
            sf.info(str(p))  # wav 头可读
            src_file_count += 1

        y_true = read_wav(srcs["true"])
        item = {
            "item": i, "hit_id": r["hit_id"], "hit_index": r["hit_index"],
            "sequence_id": r["sequence_id"], "source": r["source"], "split": r["split"],
            "sound_id": r["sound_id"], "sound_notation": r["sound_notation"],
            "sound_family": fam_of.get(r["hit_id"], r["sound_family"]),
            "pool": pool_of.get(r["hit_id"], "?"), "dur_ms": r["dur_ms"],
            "ref_file": "", "cand_files": {},
        }
        for a in APPENDIX_ARMS:
            m = arm_metrics(y_true, read_wav(srcs[a]))
            metric_rows.append({"item": i, "hit_id": r["hit_id"], "source": r["source"],
                                "sound_id": r["sound_id"], "sound_notation": r["sound_notation"],
                                "sound_family": fam_of.get(r["hit_id"], r["sound_family"]),
                                "pool": pool_of.get(r["hit_id"], "?"),
                                "dur_ms": r["dur_ms"], "arm": a, **m})

        # 盲听文件：REF + 3 匿名候选（75 / 13p / dup-true 隐藏对照），候选顺序按 item 固定种子打乱
        h8 = hashlib.sha1(r["hit_id"].encode("utf-8")).hexdigest()[:8]
        ref_name = f"{i:03d}_{h8}_ref.wav"
        shutil.copy2(srcs["true"], BLIND / ref_name)
        item["ref_file"] = ref_name

        candidates = {"75": srcs["75"], "13p": srcs["13p"], "dup": srcs["true"]}
        slots = ["c1", "c2", "c3"]
        order = list(candidates.keys())
        random.Random(SEED + i).shuffle(order)
        slot_map = dict(zip(slots, order))
        key[str(i)] = {"hit_id": r["hit_id"], "slots": slot_map}
        for slot in slots:
            arm = slot_map[slot]
            name = f"{i:03d}_{h8}_{slot}.wav"
            shutil.copy2(candidates[arm], BLIND / name)
            item["cand_files"][slot] = name
            sheet.append({"item": i, "slot": slot, "arm_hidden": arm,
                          "ref_file": ref_name, "cand_file": name,
                          "judgment": "", "note": ""})
        manifest.append(item)

    # ---- 自检门（build 内 fail-fast）
    names = [p.name for p in BLIND.glob("*.wav")]
    assert len(names) == 12 * 4, len(names)
    assert all(n.isascii() for n in names), "文件名必须纯 ASCII"
    assert len(set(names)) == len(names), "文件名重复"
    for it in manifest:
        for n in [it["ref_file"], *it["cand_files"].values()]:
            assert (BLIND / n).exists(), n
    for row in sheet:
        src_arm = "true" if row["arm_hidden"] == "dup" else row["arm_hidden"]
        src = SRC_PACK / f"{idx[row['item']]['hit_id']}_h{idx[row['item']]['hit_index']}_{src_arm}.wav"
        assert sha1_file(BLIND / row["cand_file"]) == sha1_file(src), row
        row["sha1"] = sha1_file(BLIND / row["cand_file"])
    assert len(sheet) == 36

    # ---- 落盘
    (OUT / "metrics_per_item.csv").write_text(
        "".join([",".join(metric_rows[0].keys()) + "\n"] +
                [",".join("" if v is None else str(v) for v in row.values()) + "\n"
                 for row in metric_rows]), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    (BLIND / "BLIND_KEY.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    with open(BLIND / "JUDGMENT_SHEET.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "slot", "ref_file", "cand_file",
                                          "judgment", "note"])
        w.writeheader()
        for row in sheet:
            w.writerow({k: row[k] for k in w.fieldnames})

    html_rows = []
    for it in manifest:
        cand_html = "".join(
            f'<div class="c"><b>{s}</b><br><audio controls src="{html.escape(it["cand_files"][s])}"></audio></div>'
            for s in ("c1", "c2", "c3"))
        html_rows.append(
            f'<div class="item"><h3>item {it["item"]:02d}  '
            f'{html.escape(str(it["sound_notation"]))} / {html.escape(str(it["dur_ms"]))}ms</h3>'
            f'<div class="c"><b>REF（原始）</b><br><audio controls src="{html.escape(it["ref_file"])}"></audio></div>'
            f'{cand_html}</div>')
    (BLIND / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>" + CARD_ID + " blind</title>"
        "<style>body{font-family:sans-serif}.item{border:1px solid #ccc;margin:12px;padding:8px}"
        ".c{display:inline-block;margin-right:16px;text-align:center}audio{width:220px}</style>"
        "</head><body><h2>hihats 12 条盲听：REF vs 3 匿名候选</h2>"
        "<p>每条候选判 0=与 REF 听不出差别 / 1=可辨·轻微 / 2=可辨·明显，填 JUDGMENT_SHEET.csv 的 judgment 列。"
        "听完再看 BLIND_KEY.json。</p>" + "".join(html_rows) + "</body></html>",
        encoding="utf-8")

    summary = summarize(metric_rows, manifest)
    summary["card_id"] = CARD_ID
    summary["source_index_md5"] = md5_file(SRC_PACK / "index.json")
    summary["source_composition_md5"] = md5_file(SRC_PACK / "composition.json")
    summary["source_pack"] = str(SRC_PACK)
    summary["n_src_wav"] = src_file_count
    summary["seed"] = SEED
    summary["mel_impl"] = {"file": "scripts/mel_metric.py", "md5": md5_file(HERE / "mel_metric.py"),
                           "frame_policy": "min"}
    summary["wav_subtype_source"] = (
        sf.info(str(BLIND / manifest[0]["ref_file"])).subtype
        + "（指标与试听同源 = 同一 wav 字节）")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (OUT / "EVAL_CARD.md").write_text(render_card(summary, manifest), encoding="utf-8")

    verify = {
        "gate": "build-self-check", "pass": True,
        "n_items": 12, "n_wav_on_disk": len(names), "n_sheet_rows": len(sheet),
        "join_sheet_to_files": f"{len(sheet)}/{len(sheet)}",
        "sha1_copy_equals_source": "36/36",
        "filenames_ascii": True, "wav_headers_readable": True,
        "names_unique": True,
    }
    (OUT / "VERIFY.json").write_text(json.dumps(verify, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("BUILD OK ->", OUT)
    return 0


def summarize(metric_rows: list[dict], manifest: list[dict]) -> dict:
    out = {"n_items": len(manifest),
           "sources": sorted({m["source"] for m in manifest}),
           "n_sources": len({m["source"] for m in manifest}),
           "sound_ids": sorted({str(m["sound_id"]) for m in manifest}),
           "dur_ms_range": [min(float(m["dur_ms"]) for m in manifest),
                            max(float(m["dur_ms"]) for m in manifest)],
           "splits": sorted({m["split"] for m in manifest}),
           "by_arm": {}}
    for a in APPENDIX_ARMS:
        rows = [r for r in metric_rows if r["arm"] == a]
        out["by_arm"][a] = {
            "n": len(rows),
            "mel_l1_mean": mean([r["mel_l1"] for r in rows]),
            "mel_l1_median": med([r["mel_l1"] for r in rows]),
            "mel_l1_min": min((r["mel_l1"] for r in rows if r["mel_l1"] is not None), default=None),
            "mel_l1_max": max((r["mel_l1"] for r in rows if r["mel_l1"] is not None), default=None),
            "env_corr_median": med([r["env_corr"] for r in rows]),
            "flux_corr_median": med([r["flux_corr"] for r in rows]),
            "onset_delta_ms_median": med([r["onset_delta_ms"] for r in rows]),
            "onset_delta_ms_max_abs": max((abs(r["onset_delta_ms"]) for r in rows
                                           if r["onset_delta_ms"] is not None), default=None),
            "hf_delta_median": med([r["hf_delta"] for r in rows]),
            "mel_frame_diff_max": max(r["mel_frame_diff"] for r in rows),
        }
    out["by_pool"] = {}
    for pool in sorted({r.get("pool", "?") for r in metric_rows}):
        out["by_pool"][pool] = {}
        for a in APPENDIX_ARMS:
            rows = [r for r in metric_rows if r["arm"] == a and r.get("pool") == pool]
            out["by_pool"][pool][a] = {
                "n": len(rows),
                "mel_l1_median": med([r["mel_l1"] for r in rows]),
                "mel_l1_mean": mean([r["mel_l1"] for r in rows]),
                "env_corr_median": med([r["env_corr"] for r in rows]),
                "flux_corr_median": med([r["flux_corr"] for r in rows]),
                "onset_delta_ms_median": med([r["onset_delta_ms"] for r in rows]),
                "hf_delta_median": med([r["hf_delta"] for r in rows]),
            }
    return out


def render_card(summary: dict, manifest: list[dict]) -> str:
    b = summary["by_arm"]
    lines = [
        f"# {CARD_ID} — 试听包 true vs 75 vs 13p（16 层判决用）",
        "",
        "## 结论占位",
        "本卡只交「指标 + 盲听包 + 判决规则」；**人耳判决未出之前不得写通过/不通过**。",
        "唯一判决数据 = Delight 填完 `listen_blind/JUDGMENT_SHEET.csv` 后由 `--score` 出的 SCORE_CARD。",
        "**JUDGMENT_SHEET 维持关闭**：正式 96 wav 包（--listen-n 24、counts 12/6/6、FLOAT、含 dup 隐藏对照、"
        "`composition.json` 核过）到位前不得开；本 v1 混合包只留档，判决以正式包为准。",
        "",
        "## 口径",
        "- mel 唯一实现 `scripts/mel_metric.py`（md5 " + summary["mel_impl"]["md5"] + "），frame_policy=min。",
        "- 指标从磁盘 wav（subtype 见 summary.json）重算 = 试听同一信号（同字节）；旧 PCM_16 包与内存 float 路径差 ≤0.0585 属量化差、非实现差，旧包数值不与本卡并表。",
        "- 75 臂 = EnCodec 24kHz/24kbps 码流取前 16 层（raw 32 层，等效 ≈12kbps）；16 vs 32 非控制变量。",
        "- 13c 不进本判决（仅附录）；`frame_diff_max=0` 是结构性恒等，不当对齐/截断已证。",
        "- R1 对齐臂唯一命名 `13p_align`（结论值 150.4%；对齐前 155.4%，同一臂前后一对，不构成两个臂）；"
        "试听 wav 臂名沿用 `13p`（文件名同源，仅作样本标识）；其余 `13p_*` 名一律禁止入卡，命中即打回；"
        "更细对齐粒度只准新实验名 `13p_subalign`，且必须与 `150.4%` 同表并注「不同对齐粒度、不可比」。",
        "- `150.4%` 为 ±1 桶粒度、`±3 帧` 级别未测；只写「信息丢失为主」，不写「时域损失已排除」"
        "（77.6% 击打 shift=0，±80ms 窗分辨不了 ±38ms 抖动）。",
        "- 中间带单列 `n_midband`（同时报 n 与 CI；CI 跨判定线 = 证据不足以判定，不得取三档任一档）；"
        "卡头标「未判定」，不进任何通过/不通过汇总、不进 §0 或结论段。",
        "",
        "## 样本",
        f"- n={summary['n_items']} 条（cymbal 4 / hihats 4 / click_60_200ms 4，见 composition.json），"
        f"{summary['n_sources']} 个 source，dur {summary['dur_ms_range'][0]}–{summary['dur_ms_range'][1]}ms，"
        f"split {summary['splits']}。",
        "- **源包变更事故**：PI 点名的「12 条全 hihats 试听包」（PCM_16）在 2026-10-27 11:41 被 "
        "`codec_rate_ab` 重跑覆盖为本混合包（FLOAT32）；旧包本地已不存在，其 index.json 已由 "
        "@Eval 从会话记录回收存档（`overwritten_hihats12_index_recovered.json`），可照单重渲染。",
        "- 覆盖缺口：hihats 仅 4 条 <5 → **只报数不判定**（红线）；hihats 全池 247 段 / 43 source。"
        "「16 层 hihats 生死」不能由 n=4 承担；判决样本已定（PI 2026-10-27 改判）＝正式 96 wav 包"
        "（--listen-n 24、counts 12/6/6、FLOAT、含 dup）hihats 池 6 条，从现包 247 段重抽；"
        "旧 12 条 hit_id 追认作废、不重渲（PCM_16 量化差会回灌 A/B）。",
        "- **置信度：低**（n=12 混合、每池 n=4）。",
        "",
        "## 逐臂指标（vs true，n=12，中位/均值）",
        "",
        "| arm | mel L1 中位 | mel L1 均值 | mel min–max | 包络相关中位 | 谱通量相关中位 | onset Δms 中位 | onset Δmax\\|\\| | HF 占比 Δ中位 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in APPENDIX_ARMS:
        d = b[a]
        tag = "（附录）" if a == "13c" else ""
        lines.append(
            f"| {a}{tag} | {d['mel_l1_median']} | {d['mel_l1_mean']} | "
            f"{d['mel_l1_min']}–{d['mel_l1_max']} | {d['env_corr_median']} | {d['flux_corr_median']} | "
            f"{d['onset_delta_ms_median']} | {d['onset_delta_ms_max_abs']} | {d['hf_delta_median']} |")
    lines += ["", "## 分池指标（vs true，每池 n=4，只报数不判定）", "",
              "| pool | 75 mel 中位 | 75 env | 75 flux | 13p mel 中位 | 13p env | 13p flux |",
              "|---|---|---|---|---|---|---|"]
    for pool, d in summary.get("by_pool", {}).items():
        a75, a13 = d.get("75", {}), d.get("13p", {})
        lines.append(f"| {pool} | {a75.get('mel_l1_median')} | {a75.get('env_corr_median')} | "
                     f"{a75.get('flux_corr_median')} | {a13.get('mel_l1_median')} | "
                     f"{a13.get('env_corr_median')} | {a13.get('flux_corr_median')} |")
    lines += [
        "",
        "## 判决规则（PI ㉝ 已确认并收窄：32 层出口删除）",
        "- 每条候选判 `0=听不出 / 1=可辨·轻微 / 2=可辨·明显`（36 条候选 = 12×[75, 13p, dup]）；"
        "`judgment=1` 一律单列 `n_midband`（未判定，不进通过/不通过汇总）。",
        "- **有效性门（PI 61/62 改判）**：负对照 dup（`true` 字节副本）**满额零容错**——当轮全部隐藏 dup 必须全判 0"
        "（n=6 → 6/6、n=12 → 12/12、正式包 24 击 → 24/24），差 1 条整卡作废、不补样不扩样；dup 判 1/2 → "
        "当轮全部听感结论作废，卡面只写「仪器失效」。dup 只是仪器自检，**不得**当任何臂的统计证据"
        "（禁写「通过 dup 检验故 75 臂可接受」）。正对照 13p 判 ≥1：n=12 维持 ≥8/12；n=6 阶段只报数不设门"
        "（PI 61/62 未给 n=6 正对照数字，Eval 不自创）。",
        "- **旧「16 层判决（本混合包 12 条）」阈值（≥8/12 / ≤5/12 / 6–7/12）作废**（PI 62③：分族只报数、"
        "总判决 = hihats 门 + dup 门）；本 v1 混合包只留档、永不判决。",
        "- **hihats 单 family（本包 n=4）只报数不判定**；hihats 判决 n=6（样本 = 正式 96 wav 包现包重抽 6 条，"
        "不走旧 12 条重渲）。规则（只看 75 臂判 0 条数）：≥5/6 → 结论限「hihats 未见退化（低置信）」；"
        "3–4/6 → 中间带（n_midband 单列、未判定）扩样 12 复判；≤2/6 → 扩样 12 条复判。"
        "**扩样 12 条同池同四臂、只准一次（PI 62②）**：≥10/12 → 只准写「hihats 未见退化（低置信）」；"
        "7–9/12 → 结案「证据不足以判定」+ 记 v0 已知短板，**不再扩样**；≤6/12 → 记「hihats 退化，n=12」。"
        "（32 层出口已删除、永不触发）。结论必须与「mel 0.8429 越 §8 0.8 线」两行并列在卡头，只写其一打回；"
        "人耳通过不解除越线登记；**禁止任何单行结论把 hihats 单独判为通过（含「已通过」「已稳」类表述，命中即打回）**。",
        "- **分族只报数（PI 62③）**：cymbal / click 只报「判 0/1/2 计数 + 样本量 + 三档明细」，不设通过线、"
        "不标置信档；`cymbal ≥10/12 判 0` 只进「观察记录」栏。总判决 = hihats 门 + dup 门，其余族进失败案例清单。",
        "- n=12 混合、每池 n=4，标低置信度；任何「通过」不得把 hihats 写成已稳类表述。",
        "",
        "## 边界",
        "- 本包 wav 为 FLOAT32（11:41 重跑产物），指标与试听同源；旧 PCM_16 包的 mel 数值与本卡不可并表（跨 codec 版本）。",
        "- P5（同 sound_id 第二 rendition）缺项与本卡无关；本卡不做跨 source 泛化声明。",
        "- 生成侧 mel/模型误差分账不在本卡；本卡只判 codec 16 层重建。",
        "",
        "## 产物",
        "- `listen_blind/`：12 ref + 36 匿名候选 + `index.html` + `JUDGMENT_SHEET.csv` + `BLIND_KEY.json`。",
        "- `metrics_per_item.csv`（36 行逐条指标）、`summary.json`、`manifest.json`、`VERIFY.json`。",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ verify（独立二次校验，不复用 build 中间结果）
def verify() -> int:
    import soundfile as sf

    fails = []
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    key = json.loads((BLIND / "BLIND_KEY.json").read_text(encoding="utf-8"))
    idx = json.loads((SRC_PACK / "index.json").read_text(encoding="utf-8"))

    referenced = set()
    for it in manifest:
        referenced.add(it["ref_file"])
        referenced.update(it["cand_files"].values())
    on_disk = {p.name for p in BLIND.glob("*.wav")}
    if referenced != on_disk:
        fails.append(f"文件集合双向不一致: missing={referenced-on_disk} orphan={on_disk-referenced}")

    sha_ok = 0
    with open(BLIND / "JUDGMENT_SHEET.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 36:
        fails.append(f"sheet 行数 {len(rows)} != 36")
    for row in rows:
        i = row["item"]
        arm = key[i]["slots"][row["slot"]]
        src_arm = "true" if arm == "dup" else arm
        src = SRC_PACK / f"{idx[int(i)]['hit_id']}_h{idx[int(i)]['hit_index']}_{src_arm}.wav"
        if not (BLIND / row["cand_file"]).exists():
            fails.append(f"缺候选 {row['cand_file']}")
            continue
        if sha1_file(BLIND / row["cand_file"]) == sha1_file(src):
            sha_ok += 1
        else:
            fails.append(f"sha1 不等 {row['cand_file']}")
    ref_sha_ok = 0
    for it in manifest:
        src = SRC_PACK / f"{it['hit_id']}_h{it['hit_index']}_true.wav"
        if sha1_file(BLIND / it["ref_file"]) == sha1_file(src):
            ref_sha_ok += 1
        else:
            fails.append(f"ref sha1 不等 {it['ref_file']}")
        try:
            sf.info(str(BLIND / it["ref_file"]))
        except Exception as e:  # noqa: BLE001
            fails.append(f"wav 头不可读 {it['ref_file']}: {e}")

    html_txt = (BLIND / "index.html").read_text(encoding="utf-8")
    for n in on_disk:
        if n not in html_txt:
            fails.append(f"index.html 未引用 {n}")
    if not all(n.isascii() for n in on_disk):
        fails.append("存在非 ASCII 文件名")

    res = {"gate": "independent-verify", "pass": not fails,
           "sha1_cand_vs_source": f"{sha_ok}/36", "sha1_ref_vs_source": f"{ref_sha_ok}/12",
           "file_set_bidirectional": referenced == on_disk,
           "sheet_join": f"{len(rows)}/36", "failures": fails}
    (OUT / "VERIFY_INDEPENDENT.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if not fails else 1


# ------------------------------------------------------------------ score
def score() -> int:
    key = json.loads((BLIND / "BLIND_KEY.json").read_text(encoding="utf-8"))
    with open(BLIND / "JUDGMENT_SHEET.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filled = [r for r in rows if str(r["judgment"]).strip() in JUDGMENT_SCALE]
    if len(filled) != len(rows):
        print(f"未填完：{len(filled)}/{len(rows)}，先让 Delight 填完 JUDGMENT_SHEET.csv")
        return 2

    try:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        fam_of = {int(it["item"]): str(it.get("sound_family", "?")) for it in manifest}
    except Exception:  # noqa: BLE001
        fam_of = {}
    per_arm = {"75": [], "13p": [], "dup": []}
    fam_arm = {}
    for r in rows:
        arm = key[r["item"]]["slots"][r["slot"]]
        j = int(str(r["judgment"]).strip())
        per_arm[arm].append(j)
        fam_arm.setdefault((fam_of.get(int(r["item"]), "?"), arm), []).append(j)
    cnt = {a: {"0": v.count(0), "1": v.count(1), "2": v.count(2)} for a, v in per_arm.items()}
    # 中间带 = judgment=1「可辨·轻微」，单列 n_midband（未判定，不进通过/不通过汇总、不进 §0 或结论段）
    n_midband = {a: cnt[a]["1"] for a in ("75", "13p", "dup")}
    # ---- 有效性门（PI 61①）：dup 满额零容错，只作仪器自检、不作统计证据
    dup_n = len(per_arm["dup"])
    dup_ok = cnt["dup"]["0"] == dup_n
    n_items = len(per_arm["75"])
    pos_hit = cnt["13p"]["1"] + cnt["13p"]["2"]
    pos_ok = (pos_hit >= 8) if n_items == 12 else None  # 仅 n=12 有 PI 正对照数字；其余 n 只报数

    # ---- 分族只报数（PI 62③）：不设通过线；cymbal ≥10/12 判 0 只进「观察记录」
    family_report, fam_n75 = {}, {}
    for (fam, arm), js in sorted(fam_arm.items()):
        family_report.setdefault(fam, {})[arm] = {
            "n": len(js), "0": js.count(0), "1": js.count(1), "2": js.count(2)}
        if arm == "75":
            fam_n75[fam] = js
    observation = [f"{fam}: 75 臂判 0 {js.count(0)}/{len(js)}（观察记录，不构成通过线）"
                   for fam, js in sorted(fam_n75.items())
                   if fam != "hihats" and len(js) >= 10 and js.count(0) >= 10]

    # ---- hihats 门（PI ㉝ + 62②）：只看 hihats 75 臂判 0 条数
    h75 = fam_n75.get("hihats", [])
    n_h, h75_0 = len(h75), h75.count(0)
    if not dup_ok:
        verdict = (f"仪器失效（dup 满额零容错失败：{cnt['dup']['0']}/{dup_n} 判 0）→ "
                   "当轮全部听感结论作废；不补样、不扩样、不追问")
    elif pos_ok is False:
        verdict = "INVALID（n=12 正对照 13p 未达 ≥8/12 判 ≥1 → 本次判决无效，不许出结论）"
    elif n_h == 6:
        if h75_0 >= 5:
            verdict = ("hihats 未见退化（低置信）（n=6）；必须与「mel 0.8429 越 §8 0.8 线」两行并列在卡头，"
                       "人耳通过不解除越线登记；禁写「已通过/已稳」")
        elif h75_0 >= 3:
            verdict = "MIDBAND（hihats 3–4/6 判 0：n_midband 单列、未判定）→ 扩样 12 复判一次（同池同四臂）"
        else:
            verdict = "hihats 判 0 ≤2/6 → 扩样 12 条复判一次（同池同四臂）"
    elif n_h == 12:
        if h75_0 >= 10:
            verdict = ("hihats 未见退化（低置信）（n=12 扩样）；不解除「mel 0.8429 越 §8 0.8 线」登记；"
                       "禁写「已通过/已稳」")
        elif h75_0 >= 7:
            verdict = "证据不足以判定（hihats 7–9/12 判 0）+ 记 v0 已知短板；不再扩样（禁无限抽样爬坡）"
        else:
            verdict = "hihats 退化，n=12（判 0 ≤6/12）；记 v0 已知短板"
    else:
        verdict = (f"hihats n={n_h} 不属于判决集（6 或 12）→ 只报数不判定"
                   "（n<5 红线；其它 n 需 PI 给口径）")
    out = {"card_id": CARD_ID, "counts": cnt,
           "n_midband": n_midband,
           "n_midband_note": "judgment=1「可辨·轻微」条数；未判定，不进通过/不通过汇总、不进 §0 或结论段",
           "neg_control_dup_ok": dup_ok,
           "neg_control_dup_quota": f"{cnt['dup']['0']}/{dup_n}（满额零容错，PI 61①；仅仪器自检、不作统计证据）",
           "pos_control_13p_ok": pos_ok,
           "pos_control_13p_note": ("≥8/12 判 ≥1（n=12，PI 61/62 未改）" if n_items == 12
                                    else "n≠12 只报数不设门（PI 61/62 未给数字，Eval 不自创）"),
           "hihats_n": n_h, "hihats_75_zero": h75_0,
           "n75_indistinguishable_all": per_arm["75"].count(0),
           "family_report_only": family_report,
           "family_note": "分族只报数不判定（PI 62③）：不设通过线、不标置信档；总判决 = hihats 门 + dup 门",
           "observation": observation,
           "verdict": verdict,
           "arm_naming": "R1 对齐臂唯一名 13p_align（150.4%，±1 桶粒度、±3 帧级别未测；对齐前 155.4% 同一臂）；"
                         "试听 wav 臂名 13p 仅作样本标识；其余 13p_* 名禁入卡",
           "confidence": "低（判决样本 n=6、扩样上限 n=12 且只准一次；n_midband 单列不进汇总）"}
    (OUT / "SCORE_CARD.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.build:
        return build()
    if args.verify:
        return verify()
    if args.score:
        return score()
    ap.error("需要 --build / --verify / --score")
    return 2


if __name__ == "__main__":
    sys.exit(main())
