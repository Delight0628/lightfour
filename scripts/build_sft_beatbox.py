# 将 AVP-LVT beatbox 数据（wav + CSV 标注）转成 SFT 训练样本（ms-swift 多模态 jsonl）
# 输入: data/beatbox/AVP-LVT_Dataset/AVP_Dataset/Personal/Participant_*/{wav,csv}
# 输出: data/sft/beatbox_clips/*.wav 切片 + data/sft/beatbox_sft.jsonl
# 划分: 按参与者留出 4 人作 test（对齐论文评测集约定: AVP 9/11/19/24）
import csv
import json
import os
import random
import re
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "beatbox" / "AVP-LVT_Dataset" / "AVP_Dataset" / "Personal"
OUT_DIR = ROOT / "data" / "sft"
CLIP_DIR = OUT_DIR / "beatbox_clips"
SFT_PATH = OUT_DIR / "beatbox_sft.jsonl"

TEST_PARTICIPANTS = {"Participant_9", "Participant_11", "Participant_19", "Participant_24"}

CLIP_MIN, CLIP_MAX = 0.06, 1.5   # 单音切片时长范围（秒）
SEED = 42

# 鼓类标签 → 中文/英文规范名（AVP 规范 kd/sd/hhc/hho，LVT 旧命名 Kick/Snare/HH）
LABEL_MAP = {
    "kd": ("底鼓", "Kick"),
    "kick": ("底鼓", "Kick"),
    "sd": ("军鼓", "Snare"),
    "snare": ("军鼓", "Snare"),
    "hhc": ("闭镲", "Closed Hi-Hat"),
    "hho": ("开镲", "Open Hi-Hat"),
    "hh": ("踩镲", "Hi-Hat"),
}

Q_CLASSIFY = [
    "这段口技（beatbox）音频模仿的是什么打击乐音色？",
    "听这段音频，说说人声模仿的是哪种鼓机音色？",
    "请判断这段节奏口技发出的是哪种打击乐声音。",
    "这段音频是 beatbox 的哪个基础音色？",
    "听音频回答：这个口技音模仿了鼓组里的哪一件乐器？",
]

A_CLASSIFY = {
    "底鼓": "这段音频模仿的是鼓组的**底鼓（Kick）**音色，低沉有力、瞬态强，是节奏的强拍基础。",
    "军鼓": "这段音频模仿的是鼓组的**军鼓（Snare）**音色，带有明显的爆破感和响弦质感。",
    "闭镲": "这段音频模仿的是鼓组的**闭镲（Closed Hi-Hat）**音色，短促清脆的高频颗粒声。",
    "开镲": "这段音频模仿的是鼓组的**开镲（Open Hi-Hat）**音色，带有延续的金属泛音。",
    "踩镲": "这段音频模仿的是鼓组的**踩镲（Hi-Hat）**音色。",
}


def parse_csv(csv_path: Path):
    """返回 [(onset, label_raw, syllable, ipa)]，按 onset 排序"""
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
            ipa = r[3].strip() if len(r) > 3 else ""
            rows.append((onset, label, syll, ipa))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    random.seed(SEED)
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
    skipped = {"short": 0, "long_trunc": 0, "bad_label": 0, "no_audio": 0}

    for part_name, wav_path, csv_path in pairs:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr
        rows = parse_csv(csv_path)
        stem = wav_path.stem

        for i, (onset, label_raw, syll, ipa) in enumerate(rows):
            info = LABEL_MAP.get(label_raw.strip().lower())
            if info is None:
                skipped["bad_label"] += 1
                continue
            cn, en = info

            end = rows[i + 1][0] if i + 1 < len(rows) else duration
            # 单音切片：到下一 onset，上限 CLIP_MAX，下限 CLIP_MIN
            clip_end = min(end, onset + CLIP_MAX)
            if clip_end - onset < CLIP_MIN:
                skipped["short"] += 1
                continue
            if end - onset > CLIP_MAX:
                skipped["long_trunc"] += 1

            s0, s1 = int(onset * sr), int(clip_end * sr)
            s1 = min(s1, len(audio))
            clip = audio[s0:s1]
            if len(clip) < int(CLIP_MIN * sr):
                skipped["short"] += 1
                continue

            clip_name = f"{stem}_{i:03d}_{label_raw.strip().lower()}.wav"
            clip_path = CLIP_DIR / clip_name
            if not clip_path.exists():
                sf.write(clip_path, clip, sr)

            # 音色描述任务的补充信息
            extra = []
            if syll:
                extra.append(f"音节近似为 {syll}")
            if ipa:
                extra.append(f"IPA 音素标注为 /{ipa}/")
            extra_txt = "，" + "，".join(extra) if extra else ""

            if random.random() < 0.75:
                q = random.choice(Q_CLASSIFY)
                a = A_CLASSIFY[cn] + f"发音上{extra_txt}。" if extra else A_CLASSIFY[cn]
            else:
                q = f"请描述这段 beatbox 音频的音色特征并判断它对应哪种鼓。"
                a = (f"这是节奏口技（beatbox）中模仿**{cn}（{en}）**的单音片段"
                     f"{extra_txt}。这类人声打击乐常见于 B-Box 表演中模仿鼓机节奏。")

            samples.append({
                "task": "beatbox_vocal_percussion",
                "split": "test" if part_name in TEST_PARTICIPANTS else "train",
                "participant": part_name,
                "source_wav": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                "messages": [
                    {"role": "user", "content": f"<audio>{q}", "audios": [str(clip_path.relative_to(ROOT)).replace("\\", "/")]},
                    {"role": "assistant", "content": a},
                ],
            })

    out = OUT_DIR / "beatbox_sft.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_train = sum(1 for s in samples if s["split"] == "train")
    n_test = len(samples) - n_train
    print(f"切片音频: {CLIP_DIR} ({len(list(CLIP_DIR.glob('*.wav')))} 个)")
    print(f"SFT 样本: {out} 共 {len(samples)} 条 (train {n_train} / test {n_test})")
    print(f"跳过: {skipped}")


if __name__ == "__main__":
    main()
