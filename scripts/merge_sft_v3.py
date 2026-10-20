# 合并 v3 训练集/评测集：beatbox v3（长窗口+单音）+ fma v1（30s mp3，本身即完整片段）
# 划分口径与 v2 一致：train = beatbox train + fma train + fma val；eval = beatbox test + fma test
import json
from pathlib import Path

BASE = Path("/gemini/code/lightfour-train")
SFT = BASE / "data" / "sft"


def load(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save(p, items):
    with open(p, "w", encoding="utf-8") as f:
        for s in items:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


bb = load(SFT / "beatbox_sft_v3.jsonl")
# FMA 沿用 v1 的 30s mp3 样本（mp3 本身就是完整片段，无需重写 wav）
fm = load(SFT / "fma_sft.jsonl")

train = [s for s in bb if s["split"] == "train"] + [s for s in fm if s["split"] in ("train", "val")]
eval_ = [s for s in bb if s["split"] == "test"] + [s for s in fm if s["split"] == "test"]

import random
random.seed(42)
random.shuffle(train)
random.shuffle(eval_)

save(SFT / "train_v3.jsonl", train)
save(SFT / "eval_v3.jsonl", eval_)

n_bb_w = sum(1 for s in train if s["task"] == "beatbox_window_seq")
n_bb_c = sum(1 for s in train if s["task"] == "beatbox_clip_single")
n_fm_s = sum(1 for s in train if s["task"] == "fma_electronic_subgenre")
print(f"train_v3.jsonl: {len(train)} 条 (beatbox窗口 {n_bb_w} / beatbox单音 {n_bb_c} / fma顶层 {len(train)-n_bb_w-n_bb_c-n_fm_s} / fma子类 {n_fm_s})")
print(f"eval_v3.jsonl:  {len(eval_)} 条")

# 确认音频路径存在性抽查
miss = 0
for s in (train + eval_)[:200]:
    a = s["messages"][0]["audios"][0]
    if not Path(a).exists():
        miss += 1
        if miss <= 3:
            print("MISSING:", a)
print(f"前200条路径检查: 缺失 {miss} 个")
