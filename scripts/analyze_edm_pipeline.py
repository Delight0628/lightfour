import ast
from collections import Counter
from pathlib import Path

import pandas as pd

root = Path(r"D:\butheisDelight\data\fma")
meta = root / "fma_metadata"
beatport_dir = Path(r"D:\butheisDelight\data\beatport")

genres = pd.read_csv(meta / "genres.csv", index_col=0)
tracks = pd.read_csv(meta / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
tracks[("track", "genres_all")] = tracks[("track", "genres_all")].map(ast.literal_eval)

elec = genres[genres["top_level"] == 15]
elec_ids = list(elec.index)
elec_titles = {gid: genres.at[gid, "title"] for gid in elec_ids}

counts = Counter()
has_elec = 0
for glist in tracks[("track", "genres_all")]:
    s = set(glist) & set(elec_ids)
    if s:
        has_elec += 1
        for g in s:
            counts[elec_titles[g]] += 1

# Beatport taxonomy
bp = [ln.strip() for ln in (beatport_dir / "genres_taxonomy.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

# Manual mapping Beatport -> closest FMA Electronic labels (and whether FMA has usable audio labels)
mapping = [
    ("House", "House", 1482, "直接对应"),
    ("Deep House", "House / Chill-out / Downtempo", 1482 + 823 + 2061, "FMA 无 Deep House，只能映射到宽泛 House 系"),
    ("Tech House", "House / Techno", 1482 + 2140, "无精确标签"),
    ("Progressive House", "House", 1482, "无精确标签"),
    ("Afro House", "(无)", 0, "FMA 无此标签"),
    ("Bass House", "(无)", 0, "FMA 无此标签"),
    ("Funky House", "(无)", 0, "FMA 无此标签"),
    ("Jackin House", "(无)", 0, "FMA 无此标签"),
    ("Organic House / Downtempo", "Downtempo", 2061, "Downtempo 可映射"),
    ("Techno (Peak Time / Driving)", "Techno", 2140, "FMA 只有粗 Techno"),
    ("Techno (Raw / Deep / Hypnotic)", "Techno / Minimal Electronic", 2140 + 1013, "无子类"),
    ("Hard Techno", "Techno / Breakcore - Hard", 2140 + 511, "无精确标签"),
    ("Melodic House & Techno", "House / Techno", 1482 + 2140, "无精确标签"),
    ("Minimal / Deep Tech", "Minimal Electronic", 1013, "部分对应"),
    ("Electro (Classic / Detroit / Modern)", "Electronic / Dance / Techno", 34413 + 1414 + 2140, "过粗"),
    ("Dubstep", "Dubstep", 1144, "直接对应，量偏少"),
    ("140 / Deep Dubstep / Grime", "Dubstep", 1144, "FMA 无 140/Grime 细分"),
    ("Drum & Bass", "Drum & Bass / Jungle", 500 + 258, "直接对应，量少"),
    ("Breaks / Breakbeat / UK Bass", "Bigbeat / Breakcore / Jungle", 189 + 511 + 258, "部分重叠"),
    ("UK Garage / Bassline", "(无)", 0, "FMA 无 UK Garage"),
    ("Trap / Wave", "(无)", 0, "FMA 无 Trap"),
    ("Bass / Club", "Dubstep / Drum & Bass", 1144 + 500, "无精确标签"),
    ("Trance (Main Floor)", "(无)", 0, "FMA Electronic 子树无 Trance"),
    ("Trance (Raw / Deep / Hypnotic)", "(无)", 0, "FMA 无此标签"),
    ("Psy-Trance", "(无)", 0, "FMA 无此标签"),
    ("Hard Dance / Hardcore", "Breakcore - Hard", 511, "不是 club hardstyle"),
    ("Mainstage", "(无)", 0, "FMA 无 Mainstage/Big Room"),
    ("Amapiano", "(无)", 0, "FMA 无此标签"),
    ("Nu Disco / Disco", "Dance / House", 1414 + 1482, "无精确 Nu Disco"),
    ("Indie Dance", "Electronic / Dance", 34413 + 1414, "过粗"),
    ("Dance / Electro Pop", "Dance / Electronic", 1414 + 34413, "过粗"),
    ("Electronica", "Electronic / IDM / Ambient Electronic", 34413 + 3472 + 5723, "FMA 偏 experimental electronica"),
]

report = root / "edm_pipeline_report.md"
with report.open("w", encoding="utf-8") as f:
    f.write("# FMA → Beatport 两阶段 EDM 分类：可行性与数据现状\n\n")
    f.write("## 方案结论\n\n")
    f.write("**部分可行，不能直接端到端落地。**\n\n")
    f.write("- Stage 1（FMA 粗分类 / 识别 Electronic）：**可行**。有 10.6 万曲元数据 + 层次流派，Electronic 子树约 3.4 万曲。\n")
    f.write("- Stage 2（Beatport 细分 trap/house/dubstep…）：**标签体系可行，监督训练数据不可行**。WhatBPM 只有榜单元数据，没有音频；FMA 也缺 Trap/Trance/UK Garage 等 club 细类。\n\n")
    f.write("## 数据集落盘位置\n\n")
    f.write("- `D:/butheisDelight/data/fma/fma_metadata/` — 已解压（tracks/genres/features 等）\n")
    f.write("- `D:/butheisDelight/data/fma/fma_small.zip` — 音频子集续传中（约 7.3GB，源站不稳定）\n")
    f.write("- `D:/butheisDelight/data/beatport/latest.json` — WhatBPM 最新榜单元数据\n")
    f.write("- `D:/butheisDelight/data/beatport/genres_taxonomy.txt` — 32 个 Beatport genre\n\n")
    f.write("## FMA Electronic 子树\n\n")
    f.write(f"- n_tracks total: {len(tracks)}\n")
    f.write(f"- genre_top=Electronic: {(tracks[('track','genre_top')]=='Electronic').sum()}\n")
    f.write(f"- 任一 Electronic 子树标签: {has_elec}\n\n")
    f.write("| genre_id | title | parent | #tracks |\n|---|---|---|---|\n")
    for gid in elec.sort_values("#tracks", ascending=False).index:
        f.write(f"| {gid} | {genres.at[gid,'title']} | {genres.at[gid,'parent']} | {int(genres.at[gid,'#tracks'])} |\n")
    f.write("\n## 多标签频次\n\n")
    for t, c in counts.most_common():
        f.write(f"- {t}: {c}\n")
    f.write("\n## Beatport 32 类 → FMA 覆盖\n\n")
    f.write("| Beatport genre | FMA 映射 | 可用曲数(约) | 说明 |\n|---|---|---:|---|\n")
    for bp_g, fma_g, n, note in mapping:
        f.write(f"| {bp_g} | {fma_g} | {n} | {note} |\n")
    covered = sum(1 for *_, n, _ in mapping if n > 0)
    f.write(f"\n粗略覆盖: {covered}/{len(mapping)} 个 Beatport 标签在 FMA 有弱映射；**Trap / Trance / Amapiano / UK Garage / Mainstage 等在 FMA 基本为 0**。\n\n")
    f.write("## 推荐落地路径\n\n")
    f.write("1. **Stage-1 可立刻做**：用 FMA 元数据+音频训 Electronic vs non-Electronic，或 16 类 top-genre。\n")
    f.write("2. **Stage-2 不能用 WhatBPM 直接训练**：先爬 Beatport charts 拿 track 元数据（BPM/genre/label），再从 SoundCloud/YouTube 采样音频自建集（研究用途，注意版权）。\n")
    f.write("3. **迁移**：Stage-1 特征/ backbone 在 FMA Electronic 上预训练，Stage-2 在 Beatport 自建集上 fine-tune。\n")
    f.write("4. **中间方案**：先只做 FMA 已有的 House/Techno/Dubstep/DnB 子类，验证 pipeline，再扩 club 类。\n")

print("Wrote", report)
print("has_elec", has_elec)
print("n_beatport", len(bp))
print("covered_weak", covered, "/", len(mapping))
print("top FMA electronic:")
for t, c in counts.most_common(10):
    print(f"  {t}: {c}")
