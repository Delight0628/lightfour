"""
构建 3–8s 序列样本（§10.1 #4）：Talker 的实际阻塞项。

单次击打样本只能训「音色→单个 token 序列」，不能训「节奏/序列」。
本脚本把同一 source(BV) 内时间上连续的击打片段拼成 3–8s 的序列窗口，
每击带 canonical 103 标签，供 Talker 做 (1,1,16,T) 的 T>1 监督。

输入（本地可复算，零依赖）:
  data/real_bbox/segments_ast_filtered*.jsonl   切片清单（start/end/dur/source）
  data/real_bbox/hybrid_labels_all_highconf_l2.jsonl  高置信伪标签（seed_notation/seed_dist）
  data/beatbox_seed_crosswalk.csv               种子库记法 -> canonical 103

输出:
  data/sequence_samples.csv        序列级 manifest（source 级 split，3–8s，含击打数/时长/标签序列）
  data/sequence_hits.csv           击打级明细（每个 hit 一行，供 Talker 造 (1,1,16,T) 对齐）
  data/sequence_family_coverage.csv 序列级 family 覆盖
  data/sequence_contract.json      契约：窗口参数/泄漏断言/标签口径

口径铁律:
  - 序列窗口必须落在**单一 source** 内，绝不跨 source 拼接（否则 split 隔离失效）。
  - split 沿用 md5(source)%100（与 seed_manifest / eval_splits 同一函数）。
  - 标签一律 canonical（sound_id / sound_notation / sound_en / sound_family），
    绝不把种子库记法或已弃用的 4 类标签当训练文本。
  - label_source 只能是 seed_match / model / rule_hat / human / unknown，
    伪标签一律不写 sound_confidence>0 的 human 值。
  - 音频不落本地、不入 git；本脚本只产出元数据。
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

MIN_HITS = 12          # 每条序列最少击打数（§10.1 #4 门槛：>=12 击/条）
MIN_SEC = 3.0          # 序列最短时长（秒）
MAX_SEC = 8.0          # 序列最长时长（秒）
MAX_GAP = 0.8          # 相邻击打最大静默间隔（秒），超过则断窗
MIN_SEED_DIST = 2.5    # nearest-seed 距离上限（越小越可信）
MAX_FAMILY_MIX = 6     # 单条序列内允许的 family 种类上限（避免噪声串味）
CODEC_RATE_HZ = 75.0   # EnCodec 24kHz / hop 320 的原生帧率（token 时间轴口径）


def split_of(source):
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_windows(hits):
    """把同一 source 内按时间排序的击打切成 3–8s 窗口。

    hits: 已按 start 排序的 list
    返回 list of (i, j) 半开区间
    """
    wins = []
    n = len(hits)
    i = 0
    while i < n:
        j = i
        # 扩展窗口：时长 <= MAX_SEC，且相邻击打间隔 <= MAX_GAP
        while j + 1 < n:
            nxt = hits[j + 1]["start"] - hits[j]["end"]
            span = hits[j + 1]["end"] - hits[i]["start"]
            if nxt > MAX_GAP or span > MAX_SEC:
                break
            j += 1
        span = hits[j]["end"] - hits[i]["start"]
        count = j - i + 1
        if count >= MIN_HITS and MIN_SEC <= span <= MAX_SEC:
            wins.append((i, j))
            i = j + 1          # 不重叠：下一窗从本窗末尾之后开始
        else:
            i = j + 1 if j > i else i + 1
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="*", default=[
        "data/real_bbox/segments_ast_filtered.jsonl",
        "data/real_bbox/segments_ast_filtered_78v.jsonl",
        "data/real_bbox/segments_ast_filtered_new_battles.jsonl",
        "data/real_bbox/segments_batch3_ast.jsonl",
    ])
    ap.add_argument("--labels", default="data/real_bbox/hybrid_labels_all_highconf_l2.jsonl")
    ap.add_argument("--crosswalk", default="data/beatbox_seed_crosswalk.csv")
    ap.add_argument("--taxonomy", default="data/beatbox_taxonomy.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--audio-root", default="data/real_bbox/segments_batch3",
                    help="远端持久盘切片根目录（仅记录路径，不复制音频）")
    ap.add_argument("--persisted-sources", default="data/persisted_audio_sources.txt",
                    help="远端持久盘已有 wav 的 source 清单（每行 'SOURCE COUNT'），"
                         "用于标注 audio_available；缺省则全部标 0")
    ap.add_argument("--vocals-sources", default="data/vocals_available_sources.txt",
                    help="远端 demucs 输出 vocals.wav 的 source 清单（每行 'SOURCE COUNT'）。"
                         "vocals.wav 是序列 token 编码的真实来源（切片 WAV 只是它的裁切），"
                         "因此 audio_available 以它为准")
    ap.add_argument("--contract", default="data/sequence_contract.json")
    args = ap.parse_args()

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    sounds = taxonomy["sounds"]
    tax_by_id = {i: s for i, s in enumerate(sounds)}

    cw = {r["seedlib_notation"]: r for r in
          csv.DictReader(open(args.crosswalk, encoding="utf-8"))}

    # 远端持久盘已保留 wav 的 source（2026-10-27 实测：87 个 source / 63114 wav）
    persisted = set()
    if args.persisted_sources and Path(args.persisted_sources).exists():
        for line in Path(args.persisted_sources).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 1 and p[0]:
                persisted.add(p[0])
    print(f"[in] persisted audio sources: {len(persisted)}")

    # 远端 demucs 输出 vocals.wav 的 source（2026-10-27 实测：380 个 source）。
    # 序列 token 就是在 vocals.wav 上整段编码的，切片 WAV 只是它的裁切，
    # 所以「能否真跑」必须以 vocals 为准，不能用切片是否持久化来代替。
    vocals = set()
    if args.vocals_sources and Path(args.vocals_sources).exists():
        for line in Path(args.vocals_sources).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 1 and p[0]:
                vocals.add(p[0])
    print(f"[in] vocals (demucs) sources: {len(vocals)}")

    # 切片清单（并集，id 去重）
    segs = {}
    for p in args.segments:
        for r in load_jsonl(p):
            segs[r["id"]] = r
    print(f"[in] segments union: {len(segs)}")

    # 高置信伪标签
    labels = load_jsonl(args.labels)
    print(f"[in] labels: {len(labels)}")

    # join：只保留同时有切片清单与标签的
    joined = [r for r in labels if r["id"] in segs]
    print(f"[in] joined: {len(joined)}")

    # 按 source 分组、按 start 排序
    by_source = defaultdict(list)
    for r in joined:
        by_source[r["source"]].append(r)
    for s in by_source:
        by_source[s].sort(key=lambda x: (x["start"], x["id"]))

    seq_rows, hit_rows = [], []
    seq_id = 0
    dropped = Counter()

    for source in sorted(by_source):
        hits = by_source[source]
        for (i, j) in build_windows(hits):
            win = hits[i:j + 1]
            # 标签：必须全部能映射到 canonical
            tagged = []
            for h in win:
                c = cw.get(h.get("seed_notation") or "")
                if not c or not str(c["canonical_sound_id"]).strip():
                    tagged = None
                    dropped["no_crosswalk"] += 1
                    break
                sid = int(c["canonical_sound_id"])
                t = tax_by_id[sid]
                tagged.append((h, sid, t, c))
            if tagged is None:
                continue
            if max(h.get("seed_dist") or 99 for h, _, _, _ in tagged) > MIN_SEED_DIST:
                dropped["seed_dist"] += 1
                continue
            fams = {t["family"] for _, _, t, _ in tagged}
            if len(fams) > MAX_FAMILY_MIX:
                dropped["family_mix"] += 1
                continue

            seq_id += 1
            start = tagged[0][0]["start"]
            end = tagged[-1][0]["end"]
            dur = round(end - start, 4)
            sid_seq = "|".join(str(sid) for _, sid, _, _ in tagged)
            not_seq = "|".join(t["notation"] for _, _, t, _ in tagged)
            en_seq = "|".join(t["en"] for _, _, t, _ in tagged)
            fam_seq = "|".join(t["family"] for _, _, t, _ in tagged)
            confs = [float(h.get("confidence") or 0.0) for h, _, _, _ in tagged]
            labelsrc = Counter(h.get("label_source") or "unknown" for h, _, _, _ in tagged)
            ls = "|".join(f"{k}:{v}" for k, v in sorted(labelsrc.items()))

            seq_rows.append({
                "sequence_id": f"SEQ-{seq_id:05d}",
                "source": source,
                "split": split_of(source),
                "start": round(start, 4),
                "end": round(end, 4),
                "dur": dur,
                "n_hits": len(tagged),
                "hits_per_sec": round(len(tagged) / dur, 3) if dur else 0.0,
                "sound_ids": sid_seq,
                "sound_notations": not_seq,
                "sound_ens": en_seq,
                "sound_families": fam_seq,
                "n_families": len(fams),
                "n_short_variant_hits": sum(1 for _, _, t, _ in tagged
                                            if t.get("is_short_variant")),
                "label_source": ls,
                "mean_confidence": round(sum(confs) / len(confs), 4),
                "min_confidence": round(min(confs), 4),
                "first_hit_id": tagged[0][0]["id"],
                "last_hit_id": tagged[-1][0]["id"],
                "audio_root": args.audio_root,
                "audio_available": int(source in persisted),
                "audio_token_available": int(source in vocals),
                "audio_truth": ("vocals" if source in vocals
                                else ("segments" if source in persisted else "none")),
                "note": "序列级监督；标签为伪标签(canonical 记法)，非人工真值",
            })

            t0 = tagged[0][0]["start"]
            for k, (h, sid, t, c) in enumerate(tagged):
                hit_rows.append({
                    "sequence_id": f"SEQ-{seq_id:05d}",
                    "hit_index": k,
                    "source": source,
                    "split": split_of(source),
                    "hit_id": h["id"],
                    "offset": round(h["start"] - t0, 4),
                    "dur": round(h["dur"], 4),
                    "wav": h.get("wav", ""),
                    "sound_id": sid,
                    "sound_notation": t["notation"],
                    "sound_en": t["en"],
                    "sound_cn": t["cn"],
                    "sound_family": t["family"],
                    "mechanism": t.get("mechanism", ""),
                    "airflow": t.get("airflow", ""),
                    "is_short_variant": int(bool(t.get("is_short_variant"))),
                    "label_source": h.get("label_source") or "unknown",
                    "confidence": round(float(h.get("confidence") or 0.0), 4),
                    "seed_dist": round(float(h.get("seed_dist") or 0.0), 4),
                    "nearest_seed": h.get("nearest_seed", ""),
                    "crosswalk_method": c.get("match_method", ""),
                    "pseudo_label_4class": h.get("pseudo_label", ""),
                    "audio_available": int(source in persisted),
                    "audio_token_available": int(source in vocals),
                    "audio_truth": ("vocals" if source in vocals
                                    else ("segments" if source in persisted else "none")),
                })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "sequence_samples.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        w.writeheader()
        w.writerows(seq_rows)

    with open(out / "sequence_hits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hit_rows[0].keys()))
        w.writeheader()
        w.writerows(hit_rows)

    # family 覆盖
    fam_seqs = defaultdict(set)
    fam_hits = Counter()
    for r in seq_rows:
        for f in set(r["sound_families"].split("|")):
            fam_seqs[f].add(r["sequence_id"])
    for r in hit_rows:
        fam_hits[r["sound_family"]] += 1
    fam_total = Counter(s["family"] for s in sounds)
    cov = []
    for f in sorted(fam_total):
        cov.append({
            "sound_family": f,
            "taxonomy_sounds": fam_total[f],
            "sequences": len(fam_seqs.get(f, ())),
            "hits": fam_hits.get(f, 0),
            "covered_sounds": len({r["sound_id"] for r in hit_rows
                                   if r["sound_family"] == f}),
        })
    with open(out / "sequence_family_coverage.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0].keys()))
        w.writeheader()
        w.writerows(cov)

    # 契约 + 泄漏断言
    by_split_src = defaultdict(set)
    for r in seq_rows:
        by_split_src[r["split"]].add(r["source"])
    keys = sorted(by_split_src)
    leaks = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            leaks += sorted(by_split_src[keys[i]] & by_split_src[keys[j]])
    hit_leaks = []
    seq_by_id = {r["sequence_id"]: r for r in seq_rows}
    seq_src = defaultdict(set)
    for r in hit_rows:
        seq_src[r["sequence_id"]].add(r["source"])
    for sid, ss in seq_src.items():
        if len(ss) > 1:
            hit_leaks.append(sid)
    contract = {
        "generator": "scripts/build_sequence_samples.py",
        "window": {"min_sec": MIN_SEC, "max_sec": MAX_SEC, "min_hits": MIN_HITS,
                   "max_gap_sec": MAX_GAP, "max_family_mix": MAX_FAMILY_MIX,
                   "min_seed_dist": MIN_SEED_DIST, "overlap": False},
        "label_space": "canonical 103 (data/beatbox_taxonomy.json)",
        "label_source_values": sorted({r["label_source"] for r in hit_rows}),
        "pseudo_label_4class_note": "pseudo_label 字段仅存档，已弃用的 kd/sd/hhc/hho 不得作为训练文本",
        "rows_sequences": len(seq_rows),
        "rows_hits": len(hit_rows),
        "sources": len({r["source"] for r in seq_rows}),
        "sequences_per_split": dict(Counter(r["split"] for r in seq_rows)),
        "hits_per_split": dict(Counter(r["split"] for r in hit_rows)),
        "cross_split_source_leakage": len(leaks),
        "leaked_sources": leaks,
        "sequences_spanning_multiple_sources": len(hit_leaks),
        "dropped_windows": dict(dropped),
        "distinct_sound_ids": len({r["sound_id"] for r in hit_rows}),
        "families_covered": len({r["sound_family"] for r in hit_rows}),
        "total_families": len({s["family"] for s in sounds}),
        "human_confirmed_hits": sum(1 for r in hit_rows if r["label_source"] == "human"),
        "audio_available_sequences": sum(1 for r in seq_rows if r["audio_available"]),
        "audio_available_hits": sum(1 for r in hit_rows if r["audio_available"]),
        "audio_available_note": "audio_available = 本地切片(segments_batch3)是否存在，"
                               "PI 2026-10-27 已钉死该语义，不是训练可用性；"
                               "训练侧一律按 token manifest 取数",
        "audio_token_available_sequences": sum(1 for r in seq_rows if r["audio_token_available"]),
        "audio_token_available_hits": sum(1 for r in hit_rows if r["audio_token_available"]),
        "audio_truth_counts": dict(Counter(r["audio_truth"] for r in seq_rows)),
        "persisted_sources": len(persisted),
        "vocals_sources": len(vocals),
        "audio_note": "音频留在远端持久盘，本地不复制、不入 git。audio_token_available 以远端 "
                      "demucs 输出 vocals.wav 为准（data/vocals_available_sources.txt，380 个 "
                      "source 实测）：序列 token 就是在 vocals.wav 上整段编码的，切片 WAV 只是"
                      "它的裁切，因此切片未持久化不代表不可用；audio_truth 标注真实来源"
                      "（vocals / segments / none）",
        "talker_handoff": {
            "join_key": "data/sequence_hits.csv 的 sequence_id + hit_index",
            "frame_rate_hz": CODEC_RATE_HZ,
            "per_hit": f"每击按 offset 拼进 3–8s 时间轴，token 段长 = dur * {CODEC_RATE_HZ:.0f}Hz"
                       "（EnCodec 24kHz / hop 320 原生帧率）",
            "text_label": "整条序列的 canonical 记法序列，用 '|' 连接（见 sound_notations 列）",
            "shape": f"(1,1,16,T)，T = ceil(dur * {CODEC_RATE_HZ:.0f})"
                     "（= EnCodec ceil(n_samples_24k / 320)）",
            "frame_rate_note": "13Hz 是 Qwen3-Omni Talker 本体码本速率，不是本数据集 token 帧率；"
                               "60–200ms 的一次击打在 13Hz 下只剩 0.8–2.6 帧，"
                               "序列监督必须按 75Hz 做（Qwen3-TTS 12.5Hz 已判死同理）",
            "max_frames": 600,
            "warning": "全部为伪标签（model/rule_hat），仅可用于训练节奏/序列结构，"
                       "不得作为 103 音色识别准确率证据",
        },
        "token_handoff": {
            "generator": "scripts/prepare_seq_tokens.py（训练服务器上跑，音频不出服务器）",
            "verifier": "scripts/verify_seq_tokens.py",
            "loader": "scripts/seq_token_dataset.py",
            "remote_codes_dir": "data/seq_tokens/codes/<sequence_id>.npz",
            "remote_manifest": "data/seq_tokens/sequence_manifest.jsonl",
            "remote_hit_frames": "data/seq_tokens/hit_frames.csv",
            "codes_shape": f"(1,1,16,T)，T = ceil(n_samples_24k/320) = round(dur * {CODEC_RATE_HZ:.0f})",
            "codebook": 1024,
            "encoding": "EnCodec 24kHz / 24kbps，取前 16 层 RVQ（由粗到细）",
            "join_key": "sequence_id + hit_index（hit_frames.csv 的 frame_start/frame_end 给帧区间）",
            "text_label": "sequence_manifest.jsonl 的 text 字段：整条序列的 canonical 记法序列",
            "encoding_granularity": "整段编码（不是逐击编码再拼接）：EnCodec 有感受野，"
                                    "单独编码 200ms 片段的码流与同段音频处于长音频中的码流不同；"
                                    "整段编码后每帧都是真实码，帧区间只用于记法->帧对齐",
            "known_artifact": "切片起点是 onset-0.02s（segment_beatbox_onsets.py 的 pre-roll），"
                              "因此相邻击打的帧区间有约 20ms（1.5 帧）重叠，"
                              "hit_frame_coverage 可能略大于 1；这是切分设计而非泄漏",
        },
        "gate": "全部标签为伪标签（model/rule_hat），human 覆盖 >=15 family 前不得据此宣称识别准确率",
    }
    Path(args.contract).write_text(json.dumps(contract, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(json.dumps(contract, ensure_ascii=False, indent=1))
    print(f"-> {out}/sequence_samples.csv  {out}/sequence_hits.csv  "
          f"{out}/sequence_family_coverage.csv  {args.contract}")


if __name__ == "__main__":
    main()
