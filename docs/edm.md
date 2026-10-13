# edm

**我：** 你去全网搜寻一下有没有edm种类分类（如细分成trap、house、dubstep等）音乐的机器学习训练数据集

**Xiaomi MiMo:** 搜完了。结论先说：**没有一个“正统、干净、现成”的 EDM 子流派多分类基准数据集**（类似 ImageNet 那种），但有几类可用资源，质量差很多。

## 现成可用的开源数据集

### 1. FMA（Free Music Archive）——最值得先看
- **规模**：106,574 首曲，161 个**层次化**流派，CC 授权可再分发，带音频
- **层级**：顶层 `Electronic` 约 3.4 万首，下面有 House、Techno、Ambient、IDM 等子类
- **局限**：偏 experimental / indie CC 音乐，不是商业 EDM（Beatport 上的主流 trap/dubstep/future bass 很少）；子流派粒度也偏“地下电子”，不是 club 文化那套分类
- **入口**：https://github.com/mdeff/fma

### 2. MTG-Jamendo——标签更细的电子子类
- **规模**：5.5 万+ 全曲，87 个 genre 标签（多标签）
- **电子相关标签**：`electronic`、`house`、`techno`、`electro`、`dance`、`trance`、`dub`、`downtempo`、`chillout`、`electrohouse`、`electronica` 等
- **局限**：同样是 CC/Jamendo 曲库，标签是上传者自标，噪声大；**没有 trap、dubstep、future bass、hardstyle** 这类 2010 后 club 子流派
- **入口**：https://github.com/MTG/mtg-jamendo-dataset

### 3. WhatBPM / Beatport 元数据——子流派分类最接近你的需求
- 基于 Beatport Top 100，每天更新的 JSON，**有官方 genre 字段**
- 当前子流派包括：
  - House 分支：Deep / Tech / Progressive / Afro / Bass / Funky / Jackin / Organic House
  - Techno 分支：Peak Time / Hard / Melodic / Raw-Deep-Hypnotic / Minimal
  - Bass：Dubstep、140 Deep Dubstep、Drum & Bass、Breaks、UK Garage、Bass Club、Trap/Wave
  - Trance：Main Floor / Raw-Deep-Hypnotic / Psy-Trance
  - 其他：Hard Dance、Indie Dance、Nu Disco、Amapiano、Mainstage、Electronica
- **局限**：只有元数据统计（BPM/key/genre/label），**没有音频文件**；需要自己抓 Beatport/YouTube 拼音频
- **入口**：https://github.com/sergree/whatbpm （`latest.json` + 历史 JSON）

### 4. 通用大库（可筛出电子，但不是专门 EDM 数据集）
| 数据集 | 特点 | 对 EDM 子流派的用处 |
|---|---|---|
| **Million Song Dataset + Last.fm tags** | 100 万曲元数据，Last.fm 流派标签 | 可筛 house/techno 等，但音频要自己找，标签很脏 |
| **AudioSet** | Google 200 万 clip 事件标签 | 有 electronic music 事件，但不是细分子流派分类任务 |
| **AcousticBrainz** | 250 万曲 high-level 特征（含 danceability） | 特征可用，流派标签来自社区，粒度不统一 |
| **GTZAN** | 经典基准 | **只有一个粗类 “Electronic”**，完全不能做子流派 |

## 实际可用性判断

| 你的需求 | 现有情况 |
|---|---|
| House / Techno / Trance 细分 | FMA + MTG-Jamendo 能覆盖一部分 |
| **Trap / Dubstep / Future Bass / Hardstyle** | **几乎没有干净的公开音频+标签对**；Beatport 元数据有标签但没音频 |
| 可直接训练的现成 benchmark | **没有** |

## 常见做法（如果你要做这个任务）

1. **Beatport 路线**：用 WhatBPM 的 genre 清单当分类本体，爬 Beatport charts / SoundCloud / YouTube 自建音频集（注意版权，通常只用于研究）
2. **FMA/MTG 路线**：先在 `Electronic` 子树上做迁移学习，再用自建小集 fine-tune
3. **半自动标注**：用 Spotify/Beatport API 的 genre + 众包复核

---

**一句话**：公开界最接近的是 **FMA / MTG-Jamendo**（有音频，但电子子类偏地下、缺 trap/dubstep）和 **WhatBPM/Beatport**（子流派最全，但只有元数据没音频）。真正“trap vs house vs dubstep”级别的现成 ML 训练集，目前基本要自己搭。

如果你要，我可以接着帮你：从 FMA 里筛出 Electronic 子树并列出各子类曲目量，或者写一个 Beatport 榜单元数据 → 分类标签表的抓取方案。

**我：** 咱们先用fma这个训练模型的基础歌曲分类分出edm再用你说的beatport链路细分edm的具体分类，你深度思考后回到这个方案可行吗？如果可行请你开始行动起来帮我下载对应的数据集

**Xiaomi MiMo:** **总体结论：方案部分可行，不能直接端到端落地。** Stage 1 可以马上做；Stage 2 现在只有标签本体，没有监督训练用的音频。

1. **FMA 用于基础歌曲分类（分出 EDM/Electronic）——可行，已下载并核对**
   - `fma_metadata` 已解压到 `D:\butheisDelight\data\fma\fma_metadata\`（106,574 曲，163 流派）。
   - Electronic 子树：`genre_top=Electronic` **9,372** 首；任一 Electronic 子树标签 **34,413** 首。
   - 子类可用量（偏地下/实验向，不是商业 club）：Ambient Electronic 5723、IDM 3472、Glitch 2809、Techno 2140、House 1482、Dubstep 1144、DnB 500 等。
   - **Trap / Trance / UK Garage / Amapiano / Mainstage 在 FMA Electronic 子树基本为 0**。
   - `fma_small.zip` 音频约 **3.66GB / 7.3GB**，UNIL 源站很慢且易断，已可断点续传；元数据已足够先把 Stage-1 的标签管线搭起来。

2. **Beatport 链路细分 EDM——taxonomy 可用，训练数据缺口在音频**
   - 已下载 WhatBPM `latest.json`（约 327KB）并抽出 **32 个 Beatport genre** 到 `genres_taxonomy.txt`（含 House/Techno 各子类、Dubstep、Trap/Wave、DnB、Trance、Amapiano 等）。
   - WhatBPM **只有榜单元数据（genre/BPM/key/label），没有音频**，无法直接训 Stage-2 分类器。
   - 与 FMA 粗对齐：32 类里约 **21 类有弱映射**（如 House/Techno/Dubstep/DnB），但 Trap、Trance、UK Garage、Amapiano、Mainstage 等 **FMA 侧不可用**；即便有映射的也是粗类，缺 club 细分。
   - 因此 Stage-2 必须：用 WhatBPM genre 当标签本体 → 爬 Beatport charts 元数据 → 再从 SoundCloud/YouTube 等补音频自建训练集（研究用途，注意版权）→ 用 Stage-1 在 FMA Electronic 上预训练的特征做 fine-tune。

**更稳妥的落地顺序**：先用 FMA 训 Stage-1（Electronic vs 其他，或 House/Techno/Dubstep/DnB 子类）验证管线；Beatport 只作目标标签体系；等自建音频集到位后再做 Stage-2 细分。完整对照与报告见 `data/fma/edm_pipeline_report.md`。

下一步可以：继续后台补全 `fma_small` 音频，或直接搭 Stage-1 训练脚本（从 `tracks.csv`/`genres.csv` 导出 Electronic 标签表）。