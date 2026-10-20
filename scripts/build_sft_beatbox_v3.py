# beatbox SFT 数据 v3：滑动窗口 1-3 秒 + 节奏序列答案（防模板记忆，逼模型听音频）
# 变更点（相对 v1 单音切片）：
#   1. 每条源 wav 生成 3 个滑动窗口（1.5/2.0/2.5s，随机起点）
#   2. 答案列出窗口内所有音符（最多 2 个），不同窗口答案不同 → 无法用模板套
#   3. 保留单音片段用于子类评测（eval）
# 服务器端运行：python3 build_sft_beatbox_v3.py
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path("/gemini/code/lightfour-train")
SRC = BASE / "data" / "beatbox_raw" / "AVP-LVT_Dataset" / "AVP_Dataset" / "Personal"
OUT_DIR = BASE / "data" / "sft"
WIN_DIR = OUT_DIR / "beatbox_windows"
CLIP_DIR = OUT_DIR / "beatbox_clips"   # v1 单音片段（评测用）
SFT_PATH = OUT_DIR / "beatbox_sft_v3.jsonl"

TEST_PARTICIPANTS = {"Participant_9", "Participant_11", "Participant_19", "Participant_24"}
SEED = 42
WINDOW_SIZES = [1.5, 2.0, 2.5]
MAX_NOTES = 2
MIN_GAP = 0.06

LABEL_MAP = {
    "kd": ("底鼓", "Kick", "低沉有力的闷响"),
    "kick": ("底鼓", "Kick", "低沉有力的闷响"),
    "sd": ("军鼓", "Snare", "清脆的爆破声"),
    "snare": ("军鼓", "Snare", "清脆的爆破声"),
    "hhc": ("闭镲", "Closed Hi-Hat", "短促的高频颗粒声"),
    "hho": ("开镲", "Open Hi-Hat", "带余韵的金属声"),
    "hh": ("踩镲", "Hi-Hat", "高频沙沙声"),
}

Q_SEQ = [
    "请按顺序听出这段口技（beatbox）里出现了哪些鼓音色？",
    "这段 beatbox 节奏里依次模仿了哪几种鼓？按出现顺序列出。",
    "听这段口技音频，说出里面出现的鼓组音色顺序。",
    "这段 beatbox 里有几个音？分别对应哪种鼓？按顺序回答。",
]

Q_MAIN = [
    "这段口技（beatbox）音频主要模仿的是什么打击乐音色？",
    "听这段音频，说说人声模仿的是哪种鼓机音色？",
    "这段 beatbox 音频的主音色是什么？",
]


def parse_csv(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.reader(f):
            if len(r) < 2:
                continue
            try:
                onset = float(r[0])
            except ValueError:
                continue
            label = r[1].strip()
            syll = r[2].strip() if len(r) > 2 else ""
            rows.append((onset, label, syll))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    for part in sorted(SRC.glob("Participant_*")):
        if not part.is_dir():
            continue
        for wav in sorted(part.glob("*.wav")):
            csv_path = wav.with_suffix(".csv")
            if csv_path.exists():
                pairs.append((part.name, wav, csv_path))
    print(f"找到 {len(pairs)} 对 wav+csv")

    samples = []
    stats = {"win": 0, "clip": 0, "skip_short": 0, "skip_label": 0}

    for part_name, wav_path, csv_path in pairs:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr
        rows = parse_csv(csv_path)
        if not rows:
            continue
        stem = wav_path.stem

        # 标注音符（带规范标签）
        notes = []
        for onset, label_raw, syll in rows:
            info = LABEL_MAP.get(label_raw.strip().lower())
            if info is None:
                stats["skip_label"] += 1
                continue
            notes.append((onset, label_raw.strip().lower(), info, syll))
        if not notes:
            continue

        # 单音片段（评测用，同 v1 逻辑）
        for i, (onset, lraw, info, syll) in enumerate(notes):
            cn, en, _ = info
            end = notes[i + 1][0] if i + 1 < len(notes) else duration
            clip_end = min(end, onset + 1.5)
            if clip_end - onset < MIN_GAP:
                continue
            s0, s1 = int(onset * sr), int(clip_end * sr)
            clip = audio[s0:min(s1, len(audio))]
            if len(clip) < int(MIN_GAP * sr):
                continue
            clip_name = f"{stem}_{i:03d}_{lraw}.wav"
            clip_path = CLIP_DIR / clip_name
            if not clip_path.exists():
                sf.write(clip_path, clip, sr)
            stats["clip"] += 1

            samples.append({
                "task": "beatbox_clip_single",
                "split": "test" if part_name in TEST_PARTICIPANTS else "train",
                "participant": part_name,
                "messages": [
                    {"role": "user",
                     "content": f"<audio>{random.choice(Q_MAIN)}",
                     "audios": [str(clip_path).replace("\\", "/")]},
                    {"role": "assistant",
                     "content": f"这段音频模仿的是鼓组的**{cn}（{en}）**音色，{info[2]}。"},
                ],
            })

        # 滑动窗口（训练主力）
        for w_i, wsize in enumerate(WINDOW_SIZES):
            if duration <= wsize:
                starts = [0.0]
            else:
                max_start = duration - wsize
                starts = [max_start * (w_i + 1) / (len(WINDOW_SIZES) + 1)]
                starts[0] += random.uniform(-0.3, 0.3)
                starts[0] = max(0.0, min(starts[0], max_start))

            for w_j, wstart in enumerate(starts):
                wend = wstart + wsize
                win_notes = [(onset, lraw, info, syll)
                             for (onset, lraw, info, syll) in notes
                             if wstart <= onset < wend][:MAX_NOTES]
                if not win_notes:
                    stats["skip_short"] += 1
                    continue

                s0, s1 = int(wstart * sr), int(wend * sr)
                seg = audio[s0:min(s1, len(audio))]
                if len(seg) < int(wsize * sr * 0.8):
                    continue

                wname = f"{stem}_w{w_i}{w_j}_{int(wstart*1000)}ms.wav"
                wpath = WIN_DIR / wname
                if not wpath.exists():
                    sf.write(wpath, seg, sr)
                stats["win"] += 1

                names = [info[0] for (_, _, info, _) in win_notes]
                seq_str = "、".join(f"**{info[0]}（{info[1]}）**" for (_, _, info, _) in win_notes)
                if len(win_notes) == 1:
                    ans = f"这段音频里只有 1 个音：{seq_str}，{win_notes[0][2][2]}。"
                else:
                    ans = f"这段音频里有 {len(win_notes)} 个音，按顺序依次是：{seq_str}。节奏由这几个基础鼓音色串联而成。"

                samples.append({
                    "task": "beatbox_window_seq",
                    "split": "test" if part_name in TEST_PARTICIPANTS else "train",
                    "participant": part_name,
                    "window_sec": wsize,
                    "notes": names,
                    "messages": [
                        {"role": "user",
                         "content": f"<audio>{random.choice(Q_SEQ)}",
                         "audios": [str(wpath).replace("\\", "/")]},
                        {"role": "assistant", "content": ans},
                    ],
                })

    random.shuffle(samples)
    with open(SFT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_train = sum(1 for s in samples if s["split"] == "train")
    n_test = len(samples) - n_train
    print(f"窗口音频: {WIN_DIR} ({len(list(WIN_DIR.glob('*.wav')))} 个)")
    print(f"单音片段: {CLIP_DIR} ({len(list(CLIP_DIR.glob('*.wav')))} 个)")
    print(f"SFT v3 样本: {SFT_PATH} 共 {len(samples)} 条 (train {n_train} / test {n_test})")
    print(f"统计: {stats}")


if __name__ == "__main__":
    main()
