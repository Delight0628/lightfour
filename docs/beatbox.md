# beatbox

**我：** 你去全网搜寻一下有没有beatbox这种音乐表现形式的机器学习训练数据集

**Xiaomi MiMo:** 搜完了。结论先说：**没有像 ImageNet 那种大规模、标准化的公开 beatbox 数据集**，但有一组学术界公开的 **vocal percussion（人声打击乐）** 数据集，是目前最接近你需求的。

---

## 一、可直接下载的学术数据集（首选）

### 1. AVP（Amateur Vocal Percussion）— 最主流
- 论文：Delgado et al., *A New Dataset for Amateur Vocal Percussion Analysis*, Audio Mostly 2019
  - arXiv: [2009.11737](https://arxiv.org/abs/2009.11737) · DOI: `10.1145/3356590.3356844`
- 规模：**9780 条样本，28 名参与者**（业余、几乎无 beatbox 经验）
- 标签：kick / snare / closed hi-hat / open hi-hat，**onset 与乐器标签齐全**
- 任务方向：vocal percussion → 鼓机类别的分类、onset 检测

### 2. LVT（Live Vocalised Transcription）
- 来源：Ramires 硕士论文（波尔图大学，2017），是**最早公开的业余 vocal percussion 数据集**
- 规模：841 条 boxeme，20 名参与者
- 标签：kick / snare / closed hi-hat
- 论文：[arXiv:1811.02406](https://arxiv.org/abs/1811.02406)

### 3. AVP-LVT 合并增强版（带音素/音节标注）— **推荐直接用这个**
- 论文：Delgado et al., *Deep Embeddings for Robust User-Based Amateur Vocal Percussion Classification*, SMC 2022
  - [arXiv:2204.04646](https://arxiv.org/abs/2204.04646)
- 内容：AVP + LVT 合并，共 5714 条 boxeme，并**人工扩展了 IPA 音素、音节级标注**
- 下载：Zenodo `https://zenodo.org/record/5578744`
- 代码：[github.com/alejandrodl/vocal-percussion-transcription](https://github.com/alejandrodl/vocal-percussion-transcription)（Apache-2.0）

---

## 二、可当原料库的开放音频

| 来源 | 内容 | 备注 |
|------|------|------|
| **Freesound** | 约 **1129** 条 beatbox 相关 wav | 含多个 pack：Ipa Beatbox Kit (92)、Human Beatbox (59)、Vocal Percussion (73) 等；多数 CC0/CC-BY |
| **Vocal Imitation Set** (DCASE 2018) | 人声模仿各类音效（含打击） | Zenodo DOI: `10.5281/zenodo.1340763` |
| **jaCappella Corpus** | 日本阿卡贝拉，含 vocal percussion 声部 | [项目页](https://tomohikonakamura.github.io/jaCappella_corpus/)，更适合分离/合成 |

---

## 三、相关但**数据未公开**的工程项目

这些做过 beatbox→MIDI / 分类，但训练数据是自采私有数据：

- [Deepbeat-beatbox2midi](https://github.com/eupston/Deepbeat-beatbox2midi) — 称用「数千条」自采样本训练 CNN
- [Deepbox](https://github.com/eupston/Deepbox) — 实时 beatbox 分类插件，路径里是私有 `myBeatBoxDataset`
- [beat2d](https://github.com/jrobchin/beat2d) — beatbox→鼓型

HuggingFace / Kaggle 上**没有**现成的 beatbox 专用数据集。

---

## 四、学术脉络（做综述时有用）

| 年份 | 工作 | 方向 |
|------|------|------|
| 2004 | Kapur et al., Query-by-Beat-Boxing (ISMIR) | 早期检索 |
| 2005 | Sinyor et al., Beatbox classification using ACE | 分类 |
| 2010 | Stowell & Plumbley, JNMR | 实时 beatbox 分类 |
| 2018–2022 | Ramires / Delgado 系列 | LVT、AVP、深度嵌入 |
| 2021 | Evain et al., Biomed. Signal Process. Control | 用 ASR 工具箱做 beatbox 识别 |
| 2025–2026 | TRIA (ISMIR 2025)、DARC、Drum-to-VP 转换 | 把 beatbox/tapping 当节奏 prompt 生成鼓轨 |

---

## 实操建议

1. **做分类/转 MIDI**：直接下 **AVP-LVT（Zenodo 5578744）** + 上面的代码仓库，标签和音素层都齐。
2. **做生成/风格迁移**：Freesound 的 beatbox pack 当粗粒度语料，但没有细标注，需自己切片标 kick/snare/hh。
3. **做专业 beatboxer 级别（复杂音色、特效）**：公开数据基本没有；只能自己采，或从 Swissbeatbox 等社区视频提取（注意版权）。

核心空缺：**专业 beatboxer 的大规模、多音色（除了基础四件套）公开语料几乎没有**——现有数据都偏「业余模仿鼓机」这条线。如果你要做的是更接近真 B-Box 表演的模型，数据得自己建。