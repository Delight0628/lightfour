"""
序列样本校验脚本（§10.1 #4 出口门槛）。

用法：
  python scripts/verify_sequence_samples.py
  python scripts/verify_seq_tokens.py     # 远端 token 形状/md5/帧区间校验（单独脚本）

校验项（全部对应 PI 2026-10-26 裁决的验收门槛）：
  1. 序列数 >= 200
  2. 每条序列：3–8s、击打数 >= 12、单一 source
  3. 跨 split source 泄漏 == 0
  4. 每击标签必须能映射到 canonical 103 体系（sound_id 在 0..102）
  5. 标签里不得出现已弃用的 4 类标签（kd/sd/hhc/hho）
  6. label_source ∈ {seed_match, model, rule_hat, human, unknown}
  7. Talker 形状对齐：按 EnCodec 原生帧率 75Hz（24kHz / hop 320），每条序列的
     token 帧数 T = ceil(dur*75)，且 T >= 225（3s*75），单条不超过 max_frames；
     给出 (N,16,T) 形状预估
  8. split 与 md5(source) 口径一致（同一 source 不跨 split）
  8b. source_split_*.txt 覆盖完整：每个序列 source 都在且只在一个 split 文件里，
      文件归属与 md5(source)%100 一致，三个文件互不重叠
  8c. 音频可用性以远端 vocals.wav 为准（token 在 vocals 上整段编码）：
      audio_token_available 全为 1；audio_available 只表示本地切片是否存在（PI 已钉死）
  9. 产物可复现：重跑 build_sequence_samples.py 后 md5 一致（--repro）

不读取音频，不依赖 librosa/transformers，纯 csv/json。
"""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 时间轴口径：EnCodec 24kHz / hop 320 = 75Hz 原生帧率。
# 13Hz 是 Qwen3-Omni Talker 本体的码本速率，不是本数据集的 token 帧率：
# 13Hz 下 60–200ms 的一次击打只剩 0.8–2.6 帧（75Hz 下是 5–15 帧），
# 序列监督必须按 75Hz 做。详见 data/sequence_contract.json 的 talker_handoff。
CODEC_RATE_HZ = 75.0     # EnCodec 24kHz / hop 320（原生帧率）
LAYERS = 16
BANNED_4CLASS = {"kd", "sd", "hhc", "hho"}
VALID_LABEL_SOURCE = {"seed_match", "model", "rule_hat", "human", "unknown"}


def split_of(source):
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def md5_file(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/sequence_samples.csv")
    ap.add_argument("--hits", default="data/sequence_hits.csv")
    ap.add_argument("--contract", default="data/sequence_contract.json")
    ap.add_argument("--taxonomy", default="data/beatbox_taxonomy.json")
    ap.add_argument("--split-dir", default="data",
                    help="source_split_{train,val,test}.txt 所在目录（覆盖完整性校验）")
    ap.add_argument("--vocals-sources", default="data/vocals_available_sources.txt",
                    help="远端 demucs vocals.wav 清单（音频可用性口径）")
    ap.add_argument("--min-sequences", type=int, default=200)
    ap.add_argument("--repro", action="store_true",
                    help="重跑 build_sequence_samples.py 并比对 md5（可复现性）")
    ap.add_argument("--max-frames", type=int, default=600,
                    help="Talker collator 的 max_frames 上限（train_omni_talker.py 默认）")
    args = ap.parse_args()

    fails, warns = [], []

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    n_tax = len(taxonomy["sounds"])
    samples = list(csv.DictReader(open(args.samples, encoding="utf-8")))
    hits = list(csv.DictReader(open(args.hits, encoding="utf-8")))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))

    # 1. 序列数
    if len(samples) < args.min_sequences:
        fails.append(f"序列数 {len(samples)} < 门槛 {args.min_sequences}")
    else:
        print(f"[ok] 序列数 {len(samples)} >= {args.min_sequences}；击打数 {len(hits)}")

    # 2. 窗口约束
    bad_window = []
    for r in samples:
        dur = float(r["dur"])
        nh = int(r["n_hits"])
        if not (3.0 <= dur <= 8.0) or nh < 12:
            bad_window.append((r["sequence_id"], dur, nh))
    if bad_window:
        fails.append(f"{len(bad_window)} 条序列违反 3–8s / >=12 击: {bad_window[:5]}")
    else:
        print(f"[ok] 全部 {len(samples)} 条序列满足 3–8s 且 >=12 击")

    # 3. 跨 split source 泄漏
    by_split_src = defaultdict(set)
    for r in samples:
        by_split_src[r["split"]].add(r["source"])
    keys = sorted(by_split_src)
    leaks = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            leaks += sorted(by_split_src[keys[i]] & by_split_src[keys[j]])
    if leaks:
        fails.append(f"跨 split source 泄漏 {len(leaks)}: {leaks[:10]}")
    else:
        print(f"[ok] 跨 split source 泄漏 0（{len(keys)} 个 split）")

    # 4. 击打标签映射
    bad_sid = [r for r in hits if not (0 <= int(r["sound_id"]) < n_tax)]
    if bad_sid:
        fails.append(f"{len(bad_sid)} 条击打的 sound_id 越界: {bad_sid[:3]}")
    else:
        print(f"[ok] 全部 {len(hits)} 条击打 sound_id 在 0..{n_tax - 1}")

    # 5. 4 类残留
    banned = []
    for r in hits:
        pl = (r.get("pseudo_label_4class") or "").strip().lower()
        if pl in BANNED_4CLASS:
            banned.append((r["sequence_id"], r["hit_index"], pl))
    if banned:
        warns.append(f"{len(banned)} 条击打的存档字段含已弃用 4 类标签（仅存档，"
                     f"不得作为训练文本）: {banned[:3]}")
    else:
        print("[ok] 无 4 类标签残留")

    # 6. label_source 取值
    ls = {r["label_source"] for r in hits}
    illegal = ls - VALID_LABEL_SOURCE
    if illegal:
        fails.append(f"非法 label_source: {sorted(illegal)}")
    else:
        print(f"[ok] label_source 取值合法: {sorted(ls)}")

    # 7. Talker 形状对齐 (N,16,T)
    frames = []
    for r in samples:
        # T = ceil(dur*75)：EnCodec 帧数是 ceil(n_samples_24k/320)，
        # 用 round 会有一半序列差 1 帧（实测 round 148/287、ceil 286/287 与远端 npz 一致；
        # 差的 1 条是 dur 只保留 4 位小数导致，容差 ±1 帧）
        t = int(math.ceil(float(r["dur"]) * CODEC_RATE_HZ))
        frames.append(t)
    too_long = sum(1 for t in frames if t > args.max_frames)
    if too_long:
        fails.append(f"{too_long} 条序列 token 帧数 > max_frames({args.max_frames})，"
                     f"训练时会被截断（最长 {max(frames)} 帧 = {max(frames)/CODEC_RATE_HZ:.1f}s）")
    else:
        print(f"[ok] 全部序列 T <= max_frames({args.max_frames})，不会被截断")
    short = sum(1 for t in frames if t < 225)   # 3s * 75Hz
    if short:
        fails.append(f"{short} 条序列 T < 225（3s 最小帧数）")
    print(f"[ok] Talker 形状: 每条 (1,1,{LAYERS},T)，T=dur*{CODEC_RATE_HZ:.0f}Hz "
          f"-> min {min(frames)} / median {sorted(frames)[len(frames)//2]} / max {max(frames)}；"
          f"batch 后 (N,{LAYERS},T_max)")

    # 7b. 契约与脚本的帧率口径必须一致（防止契约再写回 13Hz）
    handoff = contract.get("talker_handoff") or {}
    cr = handoff.get("frame_rate_hz")
    if cr is None:
        fails.append("sequence_contract.json 的 talker_handoff 缺 frame_rate_hz 字段")
    elif float(cr) != CODEC_RATE_HZ:
        fails.append(f"契约 frame_rate_hz={cr} 与校验脚本 CODEC_RATE_HZ={CODEC_RATE_HZ} 不一致")
    else:
        print(f"[ok] 契约 frame_rate_hz={cr} 与校验脚本一致")

    # 8. split 口径一致性
    mism = [r for r in samples if split_of(r["source"]) != r["split"]]
    if mism:
        fails.append(f"{len(mism)} 条序列 split 与 md5(source) 口径不符: {mism[:3]}")
    else:
        print("[ok] 全部序列 split 与 md5(source)%100 口径一致")

    # 8b. source_split_*.txt 覆盖：每个序列 source 必须落在且只落在一个 split 文件里，
    #     且该文件的 split 与 md5(source) 一致。
    #     （2026-10-27 @Eval 复核发现：校验文件不完整会让下游 join 误判 source 缺失）
    split_files = {}
    for sp in ("train", "val", "test"):
        p = Path(args.split_dir) / f"source_split_{sp}.txt"
        split_files[sp] = {l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                           if l.strip()} if p.exists() else set()
    missing_split_file = sorted({r["source"] for r in samples
                                 if not any(r["source"] in s for s in split_files.values())})
    if missing_split_file:
        fails.append(f"{len(missing_split_file)} 个序列 source 不在任何 source_split_*.txt 里: "
                     f"{missing_split_file[:10]}")
    else:
        print(f"[ok] 全部 {len({r['source'] for r in samples})} 个序列 source 都在 "
              f"source_split_*.txt 中")
    wrong_file = []
    for sp, srcs in split_files.items():
        for s in srcs:
            if split_of(s) != sp:
                wrong_file.append((sp, s))
    if wrong_file:
        fails.append(f"{len(wrong_file)} 个 source 在 split 文件里的归属与 md5 不符: "
                     f"{wrong_file[:10]}")
    else:
        n_all = sum(len(s) for s in split_files.values())
        print(f"[ok] source_split_*.txt 共 {n_all} 个 source，归属与 md5(source)%100 全部一致")
    overlap = []
    sps = sorted(split_files)
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            overlap += sorted(split_files[sps[i]] & split_files[sps[j]])
    if overlap:
        fails.append(f"source_split 文件互相重叠 {len(overlap)}: {overlap[:10]}")
    else:
        print("[ok] source_split_*.txt 三个文件互不重叠")

    # 8c. 音频可用性口径：token 是在远端 demucs vocals.wav 上整段编码的，
    #     所以「能否真跑」必须以 vocals 为准。此前 audio_available 只看切片是否持久化，
    #     把 135 条实际可用的序列误标成不可用（2026-10-27 PI 据此认为只有 152 条能跑）。
    vocals_path = Path(args.vocals_sources)
    vocals = set()
    if vocals_path.exists():
        vocals = {l.split()[0] for l in vocals_path.read_text(encoding="utf-8").splitlines()
                  if l.strip()}
    else:
        fails.append(f"缺 {vocals_path}（vocals.wav 清单），无法判定音频可用性")
    seq_sources = {r["source"] for r in samples}
    no_vocals = sorted(seq_sources - vocals)
    if vocals and no_vocals:
        fails.append(f"{len(no_vocals)} 个序列 source 远端没有 vocals.wav: {no_vocals[:10]}")
    elif vocals:
        print(f"[ok] 全部 {len(seq_sources)} 个序列 source 远端都有 vocals.wav（共 {len(vocals)} 个）")
    not_avail = [r["sequence_id"] for r in samples if r.get("audio_token_available") != "1"]
    if not_avail:
        fails.append(f"{len(not_avail)} 条序列 audio_token_available != 1（有 vocals 却不可编码）: "
                     f"{not_avail[:10]}")
    else:
        truth = Counter(r.get("audio_truth") for r in samples)
        n_seg = sum(1 for r in samples if r.get("audio_available") == "1")
        print(f"[ok] 全部 {len(samples)} 条序列 audio_token_available=1，"
              f"audio_truth={dict(truth)}（audio_available=1 仅 {n_seg} 条，"
              f"该字段=本地切片是否存在，非训练可用性）")

    # 序列与击打的 source 一致性
    seq_src = {r["sequence_id"]: r["source"] for r in samples}
    hit_seq_src = defaultdict(set)
    for r in hits:
        hit_seq_src[r["sequence_id"]].add(r["source"])
    span = [k for k, v in hit_seq_src.items() if len(v) > 1]
    if span:
        fails.append(f"{len(span)} 条序列的击打跨 source: {span[:5]}")
    else:
        print("[ok] 无序列跨 source 拼接")
    orphan = [r["sequence_id"] for r in hits if r["sequence_id"] not in seq_src]
    if orphan:
        fails.append(f"{len(orphan)} 条击打找不到所属序列")
    else:
        print("[ok] 击打与序列一一归属")

    # 9. 可复现性
    if args.repro:
        before = {p: md5_file(p) for p in (args.samples, args.hits, args.contract)}
        r = subprocess.run([sys.executable, "scripts/build_sequence_samples.py"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(f"重跑 build_sequence_samples.py 失败: {r.stderr[-300:]}")
        else:
            after = {p: md5_file(p) for p in (args.samples, args.hits, args.contract)}
            diff = [p for p in before if before[p] != after[p]]
            if diff:
                fails.append(f"重跑后 md5 不一致: {diff}")
            else:
                print(f"[ok] 重跑 build_sequence_samples.py 后 3 个产物 md5 一致（可复现）")

    # 契约一致性
    if contract.get("cross_split_source_leakage") != 0:
        fails.append("sequence_contract.json 记录的泄漏不为 0")
    if contract.get("rows_sequences") != len(samples):
        fails.append(f"contract rows_sequences={contract.get('rows_sequences')} "
                     f"!= 实际 {len(samples)}")

    print()
    if warns:
        print(f"WARN ({len(warns)}):")
        for w in warns:
            print("  ~", w)
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails:
            print("  !", f)
        return 1
    print("VERIFY OK: 全部门槛通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
