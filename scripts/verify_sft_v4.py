"""校验 data/sft_v4 数据：抽样读音频、统计答案分布、检查路径存在"""
import os, json, random, collections, wave
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "sft_v4")

for name in ("train", "eval", "test"):
    fp = os.path.join(OUT, f"{name}.jsonl")
    if not os.path.exists(fp): print(f"{name}: 缺失"); continue
    recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    ans = collections.Counter(r["messages"][1]["content"] for r in recs)
    lens = [len(r["messages"][1]["content"]) for r in recs]
    miss = 0; bad = 0; durs = []
    random.seed(0)
    for r in random.sample(recs, min(60, len(recs))):
        p = r["messages"][0]["audios"][0]
        if not os.path.exists(p): miss += 1; continue
        try:
            with wave.open(p, "rb") as w: durs.append(w.getnframes() / w.getframerate())
        except Exception: bad += 1
    print(f"\n=== {name}: {len(recs)} 条 ===")
    print(f"  答案 top5: {ans.most_common(5)}")
    print(f"  答案长度: 均值 {sum(lens)/len(lens):.1f} 最大 {max(lens)}")
    print(f"  抽样 60 条: 文件缺失 {miss}  读取失败 {bad}  音频时长 min={min(durs):.2f}s max={max(durs):.2f}s" if durs else
          f"  抽样 60 条: 文件缺失 {miss}  读取失败 {bad}")
    # 音频路径是否绝对
    rel = sum(1 for r in recs if not r["messages"][0]["audios"][0].startswith("D:"))
    print(f"  绝对路径: {len(recs)-rel}/{len(recs)}")
