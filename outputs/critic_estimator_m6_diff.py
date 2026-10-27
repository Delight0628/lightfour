import csv, statistics as st
ROWS = list(csv.DictReader(open("outputs/codec_rate_ab/per_hit.csv", newline="", encoding="utf-8")))
f = lambda r, k: float(r[k])
def minarm(rs):  # 逐条 min-arm raw，对真值：min(mel13c_vs_true, mel13p_vs_true) / mel75_vs_true - 1
    return sorted(min(f(r, "mel13c_vs_true"), f(r, "mel13p_vs_true")) / f(r, "mel75_vs_true") - 1 for r in rs)
pools = [("click_60_200ms", [r for r in ROWS if r["is_click_60_200"] == "1"]),
         ("hihats",         [r for r in ROWS if r["sound_family"] == "hihats"]),
         ("cymbal",         [r for r in ROWS if r["sound_family"] == "cymbal"])]
for name, rs in pools:
    v = minarm(rs)
    print(f"{name:16s} n={len(v):5d} min={100*v[0]:7.1f} med={100*st.median(v):7.1f} max={100*v[-1]:7.1f}")
v = minarm(pools[2][1]); print("cymbal 三元组 =", round(100*v[0], 1), round(100*st.median(v), 1), round(100*v[-1], 1))
