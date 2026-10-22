"""
用 AST 10 秒窗口分数过滤 onset 切片：去掉主持/采访/纯伴奏区域。

用法：
  python scripts/filter_segments_by_ast.py \
    --manifest /tmp/bbox_segments_all35/segments_manifest.jsonl \
    --windows data/bbox_raw/ast_windows \
    --out data/bbox_raw/segments_ast_filtered.jsonl \
    --beat 0.30
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat", type=float, default=0.30)
    ap.add_argument("--margin", type=float, default=0.8, help="beat_score >= neg_score * margin")
    args = ap.parse_args()

    win_root = Path(args.windows)
    win_map = {}
    for p in win_root.glob("*/windows.jsonl"):
        src = p.parent.name
        win_map[src] = list(read_jsonl(p))
    print("sources with windows", len(win_map))

    kept, dropped = [], 0
    per_src = defaultdict(lambda: [0, 0])
    for r in read_jsonl(args.manifest):
        src = r["source"]
        mid = (float(r["start"]) + float(r["end"])) / 2
        best = None
        for w in win_map.get(src, []):
            if w["start"] <= mid <= w["end"]:
                if best is None or w["beat_score"] > best["beat_score"]:
                    best = w
        if best is None:
            dropped += 1
            per_src[src][1] += 1
            continue
        ok = best["beat_score"] >= args.beat and best["beat_score"] >= best["neg_score"] * args.margin
        if ok:
            kept.append({**r, "ast_beat": best["beat_score"], "ast_neg": best["neg_score"],
                         "ast_top": best["top_beat"]})
            per_src[src][0] += 1
        else:
            dropped += 1
            per_src[src][1] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"input={len(kept)+dropped} kept={len(kept)} dropped={dropped} "
          f"keep_rate={len(kept)/max(len(kept)+dropped,1):.1%}")
    print("per_source kept/dropped:")
    for s, (k, d) in sorted(per_src.items(), key=lambda x: -x[1][0]):
        print(f"  {s}: {k}/{k+d}")


if __name__ == "__main__":
    main()
