"""
Omni v4 评测（服务器端）：音频敏感性测试 + test 准确率
【验收铁律】eval_loss 不可信，唯一可信指标 = ①音频敏感性 ②test 准确率 vs 随机基线

用法（服务器）：
  python eval_omni_v4.py --ckpt /gemini/code/lightfour-train/checkpoints/omni3b-edm-v4/v0-*/checkpoint-XXX \
      --data /gemini/code/lightfour-train/data/sft/test.jsonl --out /gemini/code/lightfour-train/logs/eval_v4.json
"""
import os, re, json, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
from collections import Counter, defaultdict
import numpy as np
import torch
import torchaudio
from swift.llm import infer_main, InferArguments
from swift.llm import SwiftInfer

CLASS_CN = {"底鼓": "kd", "军鼓": "sd", "闭镲": "hhc", "开镲": "hho"}
GENRES = ["电子音乐", "实验音乐", "民谣", "嘻哈", "器乐", "国际音乐", "流行", "摇滚"]


def label_of(ans):
    """从模型回答里抽类别（纯标签配方：回答就是类别名，或含类别的序列）"""
    hits = [cn for cn in CLASS_CN if cn in ans]
    if hits:
        return CLASS_CN[hits[0]]
    for g in GENRES:
        if g in ans:
            return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sens", type=int, default=20, help="敏感性测试：每类取 N 条互相比对")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.seed(42); random.shuffle(recs)
    recs = recs[: args.n]
    print(f"评测样本 {len(recs)} 条", flush=True)

    # ---------- 音频敏感性测试：同问题 + 不同音频 ----------
    groups = defaultdict(list)
    for r in recs:
        a = r["messages"][1]["content"]
        key = tuple(CLASS_CN[c] for c in a.replace("、", ",").split(",") if c in CLASS_CN) or (a,)
        groups[key].append(r)
    sens = []
    for key, rs in list(groups.items())[:6]:
        if len(rs) < 2: continue
        sens.append((key, rs[: args.sens]))

    infer = SwiftInfer(args=InferArguments(
        model="/gemini/pretrain", adapters=[args.ckpt], load_in_4bit=True,
        max_new_tokens=24, temperature=0.0, use_logits_to_generate=False,
    ))

    def run(records):
        reqs = []
        for r in records:
            reqs.append({"messages": r["messages"], "audios": r["messages"][0]["audios"]})
        outs = infer.infer(reqs)
        res = []
        for r, o in zip(records, outs):
            txt = o if isinstance(o, str) else str(o)
            res.append({"audio": r["messages"][0]["audios"][0], "gold": r["messages"][1]["content"],
                        "pred_text": txt[:120], "pred": label_of(txt)})
        return res

    all_res = run(recs)
    ok = sum(1 for r in all_res if r["pred"] and (r["pred"] in r["gold"] or r["gold"] in str(r["pred"])))
    acc = ok / max(len(all_res), 1)

    # 敏感性：同组不同音频，预测必须不同
    sens_report = []
    for key, rs in sens:
        outs = run(rs)
        preds = [o["pred"] for o in outs]
        distinct = len(set(p for p in preds if p))
        sens_report.append({"group": list(key), "n": len(rs), "distinct_preds": distinct,
                            "preds": preds[:12], "golds": [o["gold"] for o in outs[:12]]})
    # 纯文本对照（不给音频）
    txt_only = []
    for r in recs[:3]:
        rr = {"messages": [{"role": "user", "content": r["messages"][0]["content"].replace("<audio>", "")},
                            {"role": "assistant", "content": r["messages"][1]["content"]}]}
        txt_only.append(rr)
    noaudio = run([{"messages": x["messages"], "audios": []} for x in txt_only])

    print("\n" + "=" * 60, flush=True)
    print(f"test 准确率 = {acc:.4f}  ({ok}/{len(all_res)})", flush=True)
    print(f"预测分布: {Counter(r['pred'] for r in all_res)}", flush=True)
    print("\n音频敏感性（同问题不同音频，预测应随音频变化）:", flush=True)
    for s in sens_report:
        print(f"  组{s['group']}: {s['n']} 条 -> {s['distinct_preds']} 种预测 {s['preds'][:8]}", flush=True)
    print(f"\n无音频对照: {[o['pred_text'][:40] for o in noaudio]}", flush=True)
    passed = acc >= 0.70 and all(s["distinct_preds"] >= 2 for s in sens_report)
    print(f"\n>>> Gate: {'通过' if passed else '未通过'} (准确率>=0.70 且 敏感性通过)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"acc": acc, "n": len(all_res), "results": all_res,
                   "sensitivity": sens_report, "no_audio": noaudio,
                   "gate_passed": passed}, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
