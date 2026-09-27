# FMA → Beatport 两阶段 EDM 分类：可行性与数据现状

## 方案结论

**部分可行，不能直接端到端落地。**

- Stage 1（FMA 粗分类 / 识别 Electronic）：**可行**。有 10.6 万曲元数据 + 层次流派，Electronic 子树约 3.4 万曲。
- Stage 2（Beatport 细分 trap/house/dubstep…）：**标签体系可行，监督训练数据不可行**。WhatBPM 只有榜单元数据，没有音频；FMA 也缺 Trap/Trance/UK Garage 等 club 细类。

## 数据集落盘位置

- `D:/butheisDelight/data/fma/fma_metadata/` — 已解压（tracks/genres/features 等）
- `D:/butheisDelight/data/fma/fma_small.zip` — 音频子集续传中（约 7.3GB，源站不稳定）
- `D:/butheisDelight/data/beatport/latest.json` — WhatBPM 最新榜单元数据
- `D:/butheisDelight/data/beatport/genres_taxonomy.txt` — 32 个 Beatport genre

## FMA Electronic 子树

- n_tracks total: 106574
- genre_top=Electronic: 9372
- 任一 Electronic 子树标签: 34413

| genre_id | title | parent | #tracks |
|---|---|---|---|
| 15 | Electronic | 0 | 34413 |
| 42 | Ambient Electronic | 15 | 5723 |
| 236 | IDM | 15 | 3472 |
| 183 | Glitch | 15 | 2809 |
| 297 | Chip Music | 15 | 2208 |
| 181 | Techno | 15 | 2140 |
| 495 | Downtempo | 15 | 2061 |
| 286 | Trip-Hop | 15 | 1751 |
| 182 | House | 15 | 1482 |
| 296 | Dance | 15 | 1414 |
| 240 | Chiptune | 297 | 1231 |
| 468 | Dubstep | 15 | 1144 |
| 184 | Minimal Electronic | 15 | 1013 |
| 400 | Chill-out | 182 | 823 |
| 185 | Breakcore - Hard | 15 | 511 |
| 337 | Drum & Bass | 15 | 500 |
| 695 | Jungle | 15 | 258 |
| 401 | Bigbeat | 181 | 189 |
| 491 | Skweee | 468 | 78 |

## 多标签频次

- Electronic: 34413
- Ambient Electronic: 5723
- IDM: 3472
- Glitch: 2809
- Chip Music: 2208
- Techno: 2140
- Downtempo: 2061
- Trip-Hop: 1751
- House: 1482
- Dance: 1414
- Chiptune: 1231
- Dubstep: 1144
- Minimal Electronic: 1013
- Chill-out: 823
- Breakcore - Hard: 511
- Drum & Bass: 500
- Jungle: 258
- Bigbeat: 189
- Skweee: 78

## Beatport 32 类 → FMA 覆盖

| Beatport genre | FMA 映射 | 可用曲数(约) | 说明 |
|---|---|---:|---|
| House | House | 1482 | 直接对应 |
| Deep House | House / Chill-out / Downtempo | 4366 | FMA 无 Deep House，只能映射到宽泛 House 系 |
| Tech House | House / Techno | 3622 | 无精确标签 |
| Progressive House | House | 1482 | 无精确标签 |
| Afro House | (无) | 0 | FMA 无此标签 |
| Bass House | (无) | 0 | FMA 无此标签 |
| Funky House | (无) | 0 | FMA 无此标签 |
| Jackin House | (无) | 0 | FMA 无此标签 |
| Organic House / Downtempo | Downtempo | 2061 | Downtempo 可映射 |
| Techno (Peak Time / Driving) | Techno | 2140 | FMA 只有粗 Techno |
| Techno (Raw / Deep / Hypnotic) | Techno / Minimal Electronic | 3153 | 无子类 |
| Hard Techno | Techno / Breakcore - Hard | 2651 | 无精确标签 |
| Melodic House & Techno | House / Techno | 3622 | 无精确标签 |
| Minimal / Deep Tech | Minimal Electronic | 1013 | 部分对应 |
| Electro (Classic / Detroit / Modern) | Electronic / Dance / Techno | 37967 | 过粗 |
| Dubstep | Dubstep | 1144 | 直接对应，量偏少 |
| 140 / Deep Dubstep / Grime | Dubstep | 1144 | FMA 无 140/Grime 细分 |
| Drum & Bass | Drum & Bass / Jungle | 758 | 直接对应，量少 |
| Breaks / Breakbeat / UK Bass | Bigbeat / Breakcore / Jungle | 958 | 部分重叠 |
| UK Garage / Bassline | (无) | 0 | FMA 无 UK Garage |
| Trap / Wave | (无) | 0 | FMA 无 Trap |
| Bass / Club | Dubstep / Drum & Bass | 1644 | 无精确标签 |
| Trance (Main Floor) | (无) | 0 | FMA Electronic 子树无 Trance |
| Trance (Raw / Deep / Hypnotic) | (无) | 0 | FMA 无此标签 |
| Psy-Trance | (无) | 0 | FMA 无此标签 |
| Hard Dance / Hardcore | Breakcore - Hard | 511 | 不是 club hardstyle |
| Mainstage | (无) | 0 | FMA 无 Mainstage/Big Room |
| Amapiano | (无) | 0 | FMA 无此标签 |
| Nu Disco / Disco | Dance / House | 2896 | 无精确 Nu Disco |
| Indie Dance | Electronic / Dance | 35827 | 过粗 |
| Dance / Electro Pop | Dance / Electronic | 35827 | 过粗 |
| Electronica | Electronic / IDM / Ambient Electronic | 43608 | FMA 偏 experimental electronica |

粗略覆盖: 21/32 个 Beatport 标签在 FMA 有弱映射；**Trap / Trance / Amapiano / UK Garage / Mainstage 等在 FMA 基本为 0**。

## 推荐落地路径

1. **Stage-1 可立刻做**：用 FMA 元数据+音频训 Electronic vs non-Electronic，或 16 类 top-genre。
2. **Stage-2 不能用 WhatBPM 直接训练**：先爬 Beatport charts 拿 track 元数据（BPM/genre/label），再从 SoundCloud/YouTube 采样音频自建集（研究用途，注意版权）。
3. **迁移**：Stage-1 特征/ backbone 在 FMA Electronic 上预训练，Stage-2 在 Beatport 自建集上 fine-tune。
4. **中间方案**：先只做 FMA 已有的 House/Techno/Dubstep/DnB 子类，验证 pipeline，再扩 club 类。
