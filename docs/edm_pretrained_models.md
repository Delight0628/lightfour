# EDM 电子音乐分类预训练模型参考清单

> 检索时间：2026-10-18
> 检索范围：Hugging Face、GitHub、Kaggle、ArXiv、学术/开源社区（MTG-UPF、Essentia 等）
> 用途：为本项目 EDM 子流派分类（House / Techno / Trance / Dubstep / DnB 等，对齐 Beatport 类目）寻找可直接使用或低成本适配的预训练模型
> 关联文档：`agent.md` §6 演进路线（P2 兜底方案）、`data/fma/edm_pipeline_report.md`

---

## 结论速览

1. **存在直接可用的预训练模型**，其中 **MAEST（discogs-maest-\*）** 的 400 类 Discogs/Beatport 风格标签与我们的 Stage-2 目标类目几乎同源，为首选。
2. **mir-aidj/EDM-subgenre-classifier** 任务定义最对口（30 类 Beatport EDM 子流派），可作外部对标基线（60.6% @ 30 类）。
3. **重要验证点**：HouseX 论文明确指出"多模态大模型在此任务上表现不佳，专用模型才行"——与我们 v3 发现 Qwen-Omni 不听音频、套模板的结论互相印证。**分类任务走专用小模型/专用特征是正确路径，Omni 应专注语音交互层。**

---

## 一、可直接使用的预训练模型

### 1. MAEST（discogs-maest-\*）⭐ 首选推荐

| 项 | 内容 |
|---|---|
| 模型名称 | MAEST（Music Audio Efficient Spectrogram Transformer），`discogs-maest-10s/20s/30s-pw-73e-ts` 系列 |
| 发布来源 | MTG-UPF（庞培法布拉大学音乐技术组），Alonso et al., 2023；权重托管 HuggingFace（`mtg-maest`，含 Spaces/API）、Essentia 集成、GitHub 可下载 |
| 支持类别 | **400 个 Discogs/Beatport 风格标签**，原生区分 Hard Techno、Tech House、Minimal Techno、Trance、Deep House、Bass House 等电子细类 |
| 使用方式 | HuggingFace transformers 一行加载（`ASTForAudioClassification.from_pretrained`）；86M 参数；提供 10s/20s/30s 输入窗口版本 |
| 性能指标 | 论文中音乐流派/风格分类超越 MusiCNN、Discogs-EffNet；电子音乐表现"尤其突出"；有 DJ 生产级使用案例（cratemind、settag） |
| 适用场景 | **EDM 细分类主分类器**（P2 首选）；400 类输出需映射到 Beatport 32 类本体（映射表成本约半天） |
| 使用限制 | 输出为 400 类概率分布，需人工映射表；10s 窗口版本适合短片段，30s 版本适合整曲 |

### 2. mir-aidj/EDM-subgenre-classifier —— 任务定义最对口

| 项 | 内容 |
|---|---|
| 模型名称 | EDM-subgenre-classifier |
| 发布来源 | ArXiv 2110.08862（2021）；GitHub `mir-aidj/EDM-subgenre-classifier`（代码 + checkpoint 公开） |
| 支持类别 | **30 类 EDM 子流派**（Beatport 体系：house/techno/trance/dubstep/DnB 各子类） |
| 使用方式 | GitHub 提供代码与模型权重（Mel 频谱 + tempogram → CNN）；音频因版权不可再分发，权重可下 |
| 性能指标 | **30 类 top-1 60.6%**（Mel+tempogram）；Mel-only 基线 55.4% |
| 适用场景 | 直接对标实验基线（我们的分类准确率应与此对照） |
| 使用限制 | 2021 年 CNN 架构偏旧；60.6% 绝对值不高；需验证 checkpoint 在新版框架可加载 |

### 3. HouseX-v2 —— Mainstage 电子细分 benchmark + 模型

| 项 | 内容 |
|---|---|
| 模型名称 | HouseX-v2 baseline（含 HouseX 数据集，APSIPA 2022 版 + ArXiv 2409.06690 benchmark 版） |
| 发布来源 | GitHub `Gariscat/housex-v2`（代码 + 数据全开源） |
| 支持类别 | Future House / Bass House / Tech House / Deep House / Bigroom 等 mainstage 子类 |
| 使用方式 | 开源训练管线（CNN / ViT baseline）；数据集带人工标注 drop 段 |
| 性能指标 | 专用 baseline "高准确率"；论文明确：**多模态大模型（MLLM）在此任务上表现挣扎，专用模型才行** |
| 适用场景 | House 子类细分 + 现成 baseline 训练管线 |
| 使用限制 | 仅 House 系子类，不含 Techno/Trance 全谱 |

### 4. HF 社区 GTZAN 系（仅作 sanity 基线）

| 模型 | 来源 | 类别 | 说明 |
|---|---|---|---|
| `S3Sound/distilhubert-finetuned-gtzan` | HF | GTZAN 10 类 | 通管线冒烟验证用 |
| `Youghourta/distilhubert-finetuned-gtzan` | HF | GTZAN 10 类 | 同上 |
| `tomragus/ast-finetuned-audioset-finetuned-gtzan` | HF | GTZAN 10 类 | 同上 |
| `ihanif/distilhubert-music-gtzan-classification` | HF | GTZAN 10 类 | 同上 |

**限制**：GTZAN 中"Electronic"仅一个粗类，无子流派粒度，对我们目标基本无用，不建议投入训练。

---

## 二、半现成方案（需自行微调）与适配成本

| 方案 | 权重来源 | 适配方式 | 成本估计 | 预期收益 |
|---|---|---|---|---|
| **MAEST 特征 + 分类头**（86M） | 公开 | 冻结特征 + 线性头，FMA 9k Electronic 上训 | **低（约 1 天）** | ⭐ 高（特征自带电子细类区分度） |
| musicnn / MusiCNN | 公开（MTT 50 标签含 house/techno/trance/dnb/dubstep） | 冻结特征 + 分类头或全量微调 | 低-中 | 中 |
| Discogs-EffNet + 分类头 | 公开（Essentia） | 冻结特征 + 分类头 | 低 | 中 |
| CLAP 零样本 | 公开（LAION） | **无需训练**，文本 prompt 当标签 | 极低（约半天） | 粗分类可用；细类区分弱（"hard techno" vs "tech house" 文本嵌入过近） |
| MERT / MusicFM / MuQ-MuLan 等音乐基础模型 + 线性探针 | 公开 | kNN / 浅层分类头（有九模型对比论文可参考） | 中（2-3 天，需跑对比） | 中-高 |
| Dasheng（小米，详见《MiDashengLM 调研》） | 公开（Apache 2.0） | 特征 + 分类头 | 低 | 中-高（声音+音乐统一基座） |

---

## 三、配套数据集

| 数据集 | 规模 | 电子类标签 | 用途 |
|---|---|---|---|
| **FMA Electronic**（本地已有） | genre_top=Electronic 9,372 首 | House 1482 / Techno 2140 / Dubstep 1144 / DnB 500 等 | 主训练/评测集 |
| **MTG-Jamendo** | 55k 全曲，87 标签 | house/techno/trance/dubstep/electro/dance 等齐全 | 扩充训练集 |
| **HouseX** | 细粒度 House 子类 | Future/Bass/Tech/Deep House、Bigroom | House 子类专项 |

---

## 四、落地建议（对照 agent.md §6 演进路线）

1. **立即可做**：用 **MAEST 替代 CLAP/MERT 作为 P2 首选分类器**——在 FMA 电子子类上同时跑 zero-shot 与 "+分类头微调"两个版本，半天到一天可知效果，远快于训 Omni。
2. **基线对标**：将 mir-aidj 的 60.6%@30 类、HouseX baseline 作为我们分类准确率的**外部参照线**（P0 验收 beatbox ≥70%、FMA ≥55% 之外的行业坐标）。
3. **架构结论**：HouseX 论文"多模态大模型做不好专用音频分类"与我们 v3 的发现一致——**识别/分类交给专用模型（MAEST/Dasheng），Omni 专注语音交互对话层**，这也是 P4 语音交互集成的合理分工。
4. **唯一手工项**：制作 MAEST 400 类/Discogs 样式 → Beatport 32 类（`data/beatport/genres_taxonomy.txt`）的映射表。

---

## 参考链接

- MAEST 论文与预训练分类器综述笔记：https://perhalvorsen.com/media/notes/pretrained_classifiers.html
- MAEST 实战（400 Discogs styles）：https://blog.serendeep.tech/blog/your-music-has-no-genre
- mtg/maest 模型入口：https://cloudernative.com/ai-models/tag/music-understanding
- discogs-maest-30s-pw-73e-ts：https://model.aibase.com/models/details/1915693234606727169
- MTG/essentia 预训练模型库：https://deepwiki.com/MTG/essentia/6.2-pre-trained-models
- EDM Subgenre Classification（ArXiv 2110.08862）：https://arxiv.org/pdf/2110.08862
- mir-aidj/EDM-subgenre-classifier：https://github.com/mir-aidj/EDM-subgenre-classifier
- Benchmarking Sub-Genre Classification for Mainstage Dance Music（ArXiv 2409.06690）：https://arxiv.org/html/2409.06690v3
- HouseX-v2（GitHub）：https://github.com/Gariscat/housex-v2
- HouseX 数据集（APSIPA 2022）：https://www.apsipa.org/proceedings/2022/APSIPA%202022/TuPM1-2/1570833801.pdf
- distilhubert-finetuned-gtzan（HF）：https://radar-huggingface.686f6c61.dev/models/s3sound-distilhubert-finetuned-gtzan
