"""family 缺口三因诊断（PI 裁决 139）。

对每个 family 给出：n_seeds / n_label_hits / n_windows(基线) / n_windows(min_hits=3)，
并给出判定：
  (a) 窗口参数    —— 标签里有击打，放宽 min_hits=3 后能出窗（不动数据，只动窗口）
  (b) 上游过滤    —— 标签里有击打，但整窗被 seed_dist / no_crosswalk / family_mix 丢掉
  (c) 种子不足    —— 标签语料里该 family 的击打数为 0，改窗口/过滤器一定无效

注意：`hybrid_labels_*.jsonl` 的 `sound_family` 是种子库口径（"Hi-hats"/"Snares"/None），
不是 canonical family id。本脚本一律用 crosswalk -> taxonomy 取 canonical family，
不用标签文件自带的 sound_family。

基线必须与已发布产物逐数对齐（287 窗 / 5,472 击），否则 ABORT —— 防止诊断脚本
自己跑出另一套口径。

用法：python scripts/diag_family_gaps.py [--out outputs/family_gap_diag.json]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sequence_samples as B  # noqa: E402

SEG_FILES = [
    "data/real_bbox/segments_ast_filtered.jsonl",
    "data/real_bbox/segments_ast_filtered_78v.jsonl",
    "data/real_bbox/segments_ast_filtered_new_battles.jsonl",
    "data/real_bbox/segments_batch3_ast.jsonl",
]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_all():
    segs = {}
    for p in SEG_FILES:
        for r in load_jsonl(p):
            segs[r["id"]] = r
    labels = [r for r in load_jsonl("data/real_bbox/hybrid_labels_all_highconf_l2.jsonl")
              if r["id"] in segs]
    sounds = json.loads(Path("data/beatbox_taxonomy.json").read_text(encoding="utf-8"))["sounds"]
    cw = {r["seedlib_notation"]: r for r in
          csv.DictReader(open("data/beatbox_seed_crosswalk.csv", encoding="utf-8"))}
    by_source = defaultdict(list)
    for r in labels:
        by_source[r["source"]].append(r)
    for s in by_source:
        by_source[s].sort(key=lambda x: (x["start"], x["id"]))
    return by_source, sounds, cw


def canon(h, sounds, cw):
    c = cw.get(h.get("seed_notation") or "")
    if not c or not str(c["canonical_sound_id"]).strip():
        return None, c
    return sounds[int(c["canonical_sound_id"])], c


def run(min_hits, by_source, sounds, cw):
    old = B.MIN_HITS
    B.MIN_HITS = min_hits
    kept, dropped = [], []
    for source in sorted(by_source):
        hits = by_source[source]
        for (i, j) in B.build_windows(hits):
            win = hits[i:j + 1]
            reason = None
            for h in win:
                t, _ = canon(h, sounds, cw)
                if t is None:
                    reason = "no_crosswalk:" + str(h.get("seed_notation"))
                    break
            if reason is None and max(h.get("seed_dist") or 99 for h in win) > B.MIN_SEED_DIST:
                reason = "seed_dist"
            if reason is None:
                fams = {canon(h, sounds, cw)[0]["family"] for h in win}
                if len(fams) > B.MAX_FAMILY_MIX:
                    reason = "family_mix"
            (kept if reason is None else dropped).append((win, reason))
    B.MIN_HITS = old
    return kept, dropped


def fam_of(h, sounds, cw):
    t, _ = canon(h, sounds, cw)
    return t["family"] if t else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/family_gap_diag.json")
    args = ap.parse_args()

    by_source, sounds, cw = load_all()
    base_kept, base_drop = run(12, by_source, sounds, cw)
    ctrl_kept, _ = run(3, by_source, sounds, cw)

    n_seq = len(base_kept)
    n_hit = sum(len(w) for w, _ in base_kept)
    print(f"[baseline] windows={n_seq} hits={n_hit} "
          f"drops={dict(Counter(r.split(':')[0] for _, r in base_drop))}")
    if (n_seq, n_hit) != (287, 5472):
        raise SystemExit(f"ABORT: 基线未复现 287/5472，实得 {n_seq}/{n_hit}")
    print(f"[control min_hits=3] windows={len(ctrl_kept)} "
          f"hits={sum(len(w) for w, _ in ctrl_kept)}")

    # 每个 family 的三层计数
    lab = Counter(fam_of(h, sounds, cw) for src in by_source.values() for h in src)
    inwin = Counter()
    for src in by_source.values():
        for (i, j) in B.build_windows(src):
            for h in src[i:j + 1]:
                inwin[fam_of(h, sounds, cw)] += 1
    win12, win3, drops = Counter(), Counter(), defaultdict(Counter)
    for win, _ in base_kept:
        for f in {fam_of(h, sounds, cw) for h in win}:
            win12[f] += 1
    for win, _ in ctrl_kept:
        for f in {fam_of(h, sounds, cw) for h in win}:
            win3[f] += 1
    for win, reason in base_drop:
        for f in {fam_of(h, sounds, cw) for h in win}:
            drops[f][reason.split(":")[0]] += 1

    tax = Counter(s["family"] for s in sounds)
    seeds = list(csv.DictReader(open("Data/seed_manifest.csv", encoding="utf-8-sig")))
    fseed = Counter(r["sound_family"] for r in seeds)

    rows = []
    print(f"\n{'family':16}{'tax':>4}{'seed':>5}{'lab':>6}{'inwin':>6}{'w12':>5}{'w3':>4}"
          f"{'judge':>7}  drops")
    for f in sorted(tax, key=lambda x: (win12.get(x, 0), x)):
        if win12.get(f, 0) > 0:
            judge = "ok"
        elif lab.get(f, 0) == 0:
            judge = "c"
        else:
            a_flag = win3.get(f, 0) > 0
            b_flag = bool(drops.get(f))
            judge = "a+b" if (a_flag and b_flag) else ("a" if a_flag else ("b" if b_flag else "?"))
        rows.append({
            "family": f, "taxonomy_sounds": tax[f], "n_seeds": fseed.get(f, 0),
            "n_label_hits": lab.get(f, 0), "n_label_hits_in_window": inwin.get(f, 0),
            "n_windows_minhits12": win12.get(f, 0), "n_windows_minhits3": win3.get(f, 0),
            "judge": judge, "drops": dict(drops.get(f, {})),
        })
        print(f"{f:16}{tax[f]:4}{fseed.get(f,0):5}{lab.get(f,0):6}{inwin.get(f,0):6}"
              f"{win12.get(f,0):5}{win3.get(f,0):4}{judge:>7}  {dict(drops.get(f, {}))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "baseline": {"min_hits": 12, "windows": n_seq, "hits": n_hit,
                     "drops": dict(Counter(r.split(":")[0] for _, r in base_drop))},
        "control": {"min_hits": 3, "windows": len(ctrl_kept),
                    "hits": sum(len(w) for w, _ in ctrl_kept)},
        "note": "family 由 crosswalk->taxonomy 取 canonical，不用 labels 自带 sound_family",
        "families": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out}")

    csv_path = out.with_suffix(".csv")
    cols = ["family", "taxonomy_sounds", "n_seeds", "n_label_hits", "n_label_hits_in_window",
            "n_windows_minhits12", "n_windows_minhits3", "judge", "drops"]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["drops"] = json.dumps(r["drops"], ensure_ascii=False)
            w.writerow(row)
    print(f"[out] {csv_path}")


if __name__ == "__main__":
    main()
