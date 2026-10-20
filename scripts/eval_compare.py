# 正式评测：v3 best (checkpoint-4500) vs v2 best (checkpoint-1000)
# 任务：1) 100 条抽样  2) test 全集
# 指标：beatbox 单音分类准确率 / FMA 顶层分类准确率 / FMA 子类准确率
# 服务器端运行：python3 eval_compare.py [mode]  mode=sample|full|all
import json
import re
import sys
from pathlib import Path

BASE = Path("/gemini/code/lightfour-train")
SFT = BASE / "data" / "sft"
V3_CKPT = BASE / "checkpoints/omni3b-edm-v3/v0-20261015-181231/checkpoint-4500"
V2_CKPT = BASE / "checkpoints/omni3b-edm-v2/v0-20261015-110843/checkpoint-1000"
OUT = BASE / "logs"

# 标签关键词 → 规范类别（用于从自由文本预测中抽取标签）
BEATBOX_KEYS = {
    "底鼓": "Kick", "Kick": "Kick", "kick": "Kick",
    "军鼓": "Snare", "Snare": "Snare", "snare": "Snare",
    "闭镲": "ClosedHiHat", "Closed Hi-Hat": "ClosedHiHat", "Closed Hi-Hat": "ClosedHiHat",
    "开镲": "OpenHiHat", "Open Hi-Hat": "OpenHiHat",
    "踩镲": "HiHat", "Hi-Hat": "HiHat",
}
FMA_TOP_KEYS = {
    "电子音乐": "Electronic", "Electronic": "Electronic",
    "嘻哈": "Hip-Hop", "Hip-Hop": "Hip-Hop",
    "摇滚": "Rock", "Rock": "Rock",
    "流行": "Pop", "Pop": "Pop",
    "民谣": "Folk", "Folk": "Folk",
    "实验音乐": "Experimental", "Experimental": "Experimental",
    "器乐": "Instrumental", "Instrumental": "Instrumental",
    "世界音乐": "International", "International": "International",
}


def extract_label(pred: str, keymap: dict):
    """从预测文本中抽取第一个匹配的类别标签"""
    # 优先找 **XX** 强调格式
    for m in re.finditer(r"\*\*(.+?)\*\*", pred):
        inner = m.group(1)
        for k, v in keymap.items():
            if k in inner:
                return v
    for k, v in keymap.items():
        if k in pred:
            return v
    return None


def gt_label(sample, keymap):
    """从样本元数据或答案中抽取真实标签"""
    if sample.get("task") == "beatbox_clip_single":
        for k, v in keymap.items():
            if k in sample["messages"][1]["content"]:
                return v
    elif sample.get("task") == "fma_top_genre":
        for k, v in keymap.items():
            if k in sample["messages"][1]["content"]:
                return v
    elif sample.get("task") == "fma_electronic_subgenre":
        return sample["messages"][1]["content"]  # 子类直接比对全文关键词
    return None


def load_eval_samples(mode: str):
    """加载评测样本"""
    samples = []
    # beatbox test（单音）
    with open(SFT / "beatbox_sft_v3.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["split"] == "test" and s["task"] == "beatbox_clip_single":
                samples.append(s)
    # fma test
    with open(SFT / "fma_sft.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["split"] == "test":
                samples.append(s)
    print(f"test 全集: {len(samples)} 条 (beatbox {sum(1 for s in samples if s['task']=='beatbox_clip_single')}, "
          f"fma_top {sum(1 for s in samples if s['task']=='fma_top_genre')}, "
          f"fma_sub {sum(1 for s in samples if s['task']=='fma_electronic_subgenre')})")

    if mode == "sample":
        import random
        random.seed(42)
        samples = random.sample(samples, min(100, len(samples)))
        print(f"抽样: {len(samples)} 条")
    return samples


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    print(f"评测模式: {mode}")

    import torch
    from transformers import BitsAndBytesConfig
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

    samples = load_eval_samples(mode)
    qconf = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    req_cfg = RequestConfig(max_tokens=150, temperature=0.1)

    results = {}
    for name, ckpt in [("v3", V3_CKPT), ("v2", V2_CKPT)]:
        print(f"\n=== 加载 {name}: {ckpt} ===")
        engine = TransformersEngine(
            "/gemini/pretrain", adapters=[str(ckpt)], template_type="qwen2_5_omni",
            quantization_config=qconf, device_map="auto", torch_dtype=torch.bfloat16)

        stats = {"beatbox": [0, 0], "fma_top": [0, 0], "fma_sub": [0, 0]}
        preds_out = []
        for i, s in enumerate(samples):
            m = s["messages"][0]
            ir = InferRequest(messages=[{"role": m["role"], "content": m["content"]}], audios=m["audios"])
            out = engine.infer([ir], request_config=req_cfg)
            pred = out[0].choices[0].message.content
            task = s["task"]

            if task == "beatbox_clip_single":
                keymap = BEATBOX_KEYS
                gt = gt_label(s, keymap)
                pl = extract_label(pred, keymap)
                key = "beatbox"
            elif task == "fma_top_genre":
                keymap = FMA_TOP_KEYS
                gt = gt_label(s, keymap)
                pl = extract_label(pred, keymap)
                key = "fma_top"
            else:  # fma_electronic_subgenre
                gt_txt = s["messages"][1]["content"]
                gt = gt_txt
                pl = pred
                key = "fma_sub"

            correct = 0
            if key == "fma_sub":
                # 子类：看预测是否含真实子类关键词
                gt_m = re.search(r"\*\*(.+?)（(.+?)）\*\*", gt_txt)
                if gt_m:
                    correct = int(gt_m.group(1) in pred or gt_m.group(2) in pred)
            else:
                correct = int(pl is not None and gt is not None and pl == gt)

            stats[key][0] += correct
            stats[key][1] += 1
            preds_out.append({"task": task, "pred": pred[:200], "gt": gt, "correct": correct})
            if (i + 1) % 20 == 0:
                print(f"  {name}: {i+1}/{len(samples)}")

        acc = {k: (v[0] / v[1] * 100 if v[1] else 0) for k, v in stats.items()}
        print(f"{name} 准确率: beatbox={acc['beatbox']:.1f}% fma_top={acc['fma_top']:.1f}% fma_sub={acc['fma_sub']:.1f}%")
        results[name] = {"acc": acc, "stats": stats, "preds": preds_out}

    with open(OUT / f"eval_{mode}_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: logs/eval_{mode}_result.json")

    # 对比汇总
    print("\n=== v3 vs v2 对比 ===")
    for k in ["beatbox", "fma_top", "fma_sub"]:
        v3a = results["v3"]["acc"][k]
        v2a = results["v2"]["acc"][k]
        print(f"{k:10s}: v3={v3a:.1f}%  v2={v2a:.1f}%  Δ={v3a-v2a:+.1f}pp")


if __name__ == "__main__":
    main()
