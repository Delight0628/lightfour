import ast
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

root = Path(r"D:\butheisDelight\data\fma")
zpath = root / "fma_metadata.zip"
out = root

print("zip_size", zpath.stat().st_size)
with zipfile.ZipFile(zpath) as z:
    print("entries", z.namelist())
    z.extractall(out)

meta = out / "fma_metadata"
for p in sorted(meta.glob("*")):
    print("file", p.name, p.stat().st_size)

genres = pd.read_csv(meta / "genres.csv", index_col=0)
tracks = pd.read_csv(
    meta / "tracks.csv",
    index_col=0,
    header=[0, 1],
    low_memory=False,
)
for col in [("track", "genres"), ("track", "genres_all")]:
    if col in tracks.columns:
        tracks[col] = tracks[col].map(ast.literal_eval)

print("n_tracks", len(tracks))
print("n_genres", len(genres))

# Electronic subtree
electronic = genres[genres["top_level"] == 15].sort_values("#tracks", ascending=False)
print("\n=== Electronic subtree (top_level=15) ===")
print(electronic[["title", "parent", "#tracks"]].to_string())

# Top-level genre distribution
top = tracks[("track", "genre_top")].value_counts(dropna=False)
print("\n=== genre_top distribution ===")
print(top.head(30).to_string())

# Tracks whose genre_top is Electronic
n_elec = (tracks[("track", "genre_top")] == "Electronic").sum()
print("\nElectronic genre_top count:", n_elec)

# Multi-label: tracks that include any Electronic-subtree genre id
elec_ids = set(electronic.index.tolist())
counts = Counter()
has_elec = 0
for glist in tracks[("track", "genres_all")]:
    s = set(glist) & elec_ids
    if s:
        has_elec += 1
        for g in s:
            counts[genres.at[g, "title"]] += 1
print("tracks with any Electronic-subtree label:", has_elec)
print("\nElectronic subgenre track frequencies (multi-label):")
for title, c in counts.most_common():
    print(f"  {title}: {c}")

# Compare with Beatport taxonomy overlap
beatport = Path(r"D:\butheisDelight\data\beatport\genres_taxonomy.txt")
if beatport.exists():
    bp = [ln.strip() for ln in beatport.read_text(encoding="utf-8").splitlines() if ln.strip()]
    fma_titles = {t.lower() for t in genres.loc[elec_ids, "title"].tolist()}
    print("\nBeatport genres:", len(bp))
    rough = []
    for g in bp:
        keys = [w.strip().lower() for w in g.replace("/", " ").split() if len(w) > 2]
        hit = [f for f in fma_titles if any(k in f for k in keys)]
        if hit:
            rough.append((g, hit[:5]))
    print("Beatport genres with rough FMA Electronic name overlap:")
    for g, hit in rough:
        print(f"  {g} -> {hit}")

report = root / "electronic_stats.md"
with report.open("w", encoding="utf-8") as f:
    f.write("# FMA Electronic 子树统计\n\n")
    f.write(f"- tracks total: {len(tracks)}\n")
    f.write(f"- Electronic genre_top: {n_elec}\n")
    f.write(f"- any Electronic-subtree label: {has_elec}\n\n")
    f.write("## Electronic subtree\n\n")
    f.write(electronic[["title", "parent", "#tracks"]].to_string())
    f.write("\n\n## Multi-label frequencies\n\n")
    for title, c in counts.most_common():
        f.write(f"- {title}: {c}\n")
print("\nWrote", report)
