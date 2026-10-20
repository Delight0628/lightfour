# Beatbox 音频识别 — 公开预训练模型资源调研

> 调研日期：2026-10-18
> 目的：为 lightfour 项目的 beatbox 识别（P0-P1）与伪标注数据飞轮（P3+）筛选可复用/可二次开发的模型资源
> 结论先行：**没有任何"拿来即用"的 beatbox 专业识别预训练模型**。可复用资产分三类：①学术代码库（需复训）②邻域鼓机分类模型（可迁移）③通用音频编码器 + 微调（最务实主线）

---

## 1. 候选模型对比清单

### A 类：beatbox / 人声打击乐专用（最对口）

| # | 名称 | 来源 | 架构 | 训练数据 | 任务 | 协议 | 权重状态 | 评价 |
|---|---|---|---|---|---|---|---|---|
| A1 | **vocal-percussion-transcription**（Delgado et al.） | [GitHub](https://github.com/alejandrodl/vocal-percussion-transcription) + [论文 2204.04646](https://arxiv.org/pdf/2204.04646v1.pdf) | CNN 分类器 + 监督式 embedding（7 种监督策略） | **AVP-LVT（5,714 boxeme）**——正是我们在用的数据集 | boxeme 分类（音节级监督效果最好）、转写 | Apache-2.0 | **代码开源，预训练权重未确认公开**（仓库以训练/评测管线为主，需按论文配置复训） | ⭐⭐⭐⭐ **首选二次开发基座**：数据集与我们完全一致，音节级监督结论可直接指导 P0 标签设计 |
| A2 | Neural Beatbox（Nao Tokui, 2018） | [项目页](https://naotokui.net/works/neural-beatbox-2018/) / [CodePen](https://codepen.io/naotokui/pen/NBzJMW/) / [训练脚本 gist](https://gist.github.com/naotokui/a2b331dd206b13a70800e862cfe7da3c) | CNN（频谱图分类）+ DrumRNN 生成 | 在线鼓机音色集（**非 beatbox 人声**） | 7 类鼓音色分类（kick/snare/hihat×2/tom×3），浏览器实时 | 代码公开（gist），TFJS 模型托管 S3 | **有权重**（TFJS，S3 链接，2018 年老物） | ⭐⭐ 可跑通 demo 但太老、类目少、非人声训练；仅作管线参考 |
| A3 | beatbox2drums（MSU AI Club） | [项目页](https://www.msuaiclub.com/projects/5e0a171d-9ba3-470e-b60d-637532659094) | 双 CNN（onset 检测 + 分类） | 自建（规模不明） | beatbox→鼓型转换，宣称 99.39% 准确率 | 未明确 | **未见权重发布**（课程项目） | ⭐ 思路可参考（onset+分类双阶段），无可用资产 |
| A4 | Beatbot（alexcrist） | [Demo](https://alexcrist.github.io/beatbot/) | 传统信号处理 + 轻量模型混合 | 自建 | beatbox→鼓转写 | 未明确 | Demo 级，**未见完整权重** | ⭐ 工程参考（转写管线设计） |
| A5 | HMM-GMM beatbox 识别（JEP-TALN 2020） | [论文](https://aclanthology.org/2020.jeptalnrecital-jep.24/) | HMM-GMM（ASR 移植） | 80 个孤立音（2 人） | 单音识别 + Vocal Grammatics 转写 | 论文级 | 无实现资产 | ✗ 过时，排除 |
| A6 | Beatbox 传统 ML 系列（ACE 2005 / IEEE 2020 / JCSSP 2025） | 见 docs/beatbox.md §四 | SVM/MLP/KNN + MFCC | 自采 900 音等 | 子类分类 | 论文级 | 无实现资产 | ✗ 理论验证，排除 |

### B 类：邻域鼓音色分类模型（可迁移识别"人声模仿的鼓"）

| # | 名称 | 来源 | 架构 | 训练数据 | 协议 | 权重 | 评价 |
|---|---|---|---|---|---|---|---|
| B1 | **airasoul/wav2vec2 鼓分类器**（HF） | [HF: airasoul/wav2vec2-*](https://www.airasoul.com/audio-classification-training-drum-kit-classifier) | wav2vec2-base 微调，10 类鼓 | 鼓机音色数据集，3s@16kHz | HF 模型（需查证） | ✅ HF 直接下载 | ⭐⭐⭐ **迁移学习起点**：鼓类边界与 beatbox 类目高度重合；但鼓机真实音色 ≠ 人声模仿，domain gap 需微调 |
| B2 | drum-sample-retrieval-vocalisation（Delgado et al.） | [GitHub](https://github.com/alejandrodl/drum-sample-retrieval-vocalisation) + [论文 2204.04651](https://arxiv.org/pdf/2204.04651v1.pdf) | 条件表示学习 | 人声模仿 + 鼓样本 | 人声→鼓样本检索（embedding） | Apache-2.0，代码开源 | ⭐⭐⭐ 与 A1 同组作者；**输出"人声↔鼓音色"embedding**，正好是伪标注飞轮需要的对齐能力 |
| B3 | Beat This!（ISMIR 2024） | [PyPI: beat-this](https://pypi.org/project/beat-this/) | Transformer beat tracker | 大规模标注音乐 | 节拍/鼓点跟踪 | ✅ `load_model()` 即用 | ⭐⭐⭐ **预处理管线组件**：视频切片后的 onset/节拍网格，生成任务的"时值"信息来源 |

### C 类：通用音频编码器 + 微调（P2 兜底 / 特征底座）

| # | 名称 | 参数量 | 特点 | 协议 | 评价 |
|---|---|---|---|---|---|
| C1 | **Dasheng / MiDashengLM**（小米，2026-10 开源） | Base 86M / Huge 0.6B / LLM 7B | 语音+环境声+**音乐**统一理解；HEAR 18 任务领先 | Apache-2.0 | ⭐⭐⭐⭐ 前文已论证：音乐音色区分最对口 |
| C2 | CLAP（LAION） | ~600M | 音频-文本对比学习，zero-shot 分类 | Apache-2.0 | ⭐⭐⭐ 成熟生态，但音乐细粒度弱于 C1/C3 |
| C3 | MERT | 95M/330M | 音乐理解专用 SSL | CC-BY-NC | ⭐⭐⭐ 音乐强，但 NC 协议限制商用 |
| C4 | BEATs / OMAR-RQ | 90M/1.2B | 通用音频 SSL，AudioSet SOTA | MIT / 开源 | ⭐⭐ 通用事件识别，打击乐细分一般 |
| C5 | Vibrato（唱歌发声分析） | 269K | 极小 CNN，发声技术分类 | 开源 | ⭐ 邻域（发声器官视角），类目不匹配 |

---

## 2. 筛选结论：可直接复用 / 二次开发的成熟项

**排除项**（仅理论/无实现）：A3-A6 全部。

**优先级排序**（适配我们后续流程、可快速落地验证）：

| 优先级 | 选项 | 用途 | 落地动作 |
|---|---|---|---|
| **P-1** | A1 vocal-percussion-transcription | P0/P1 识别主基座 | 克隆仓库→按论文音节级监督配置→在 AVP-LVT 复训（数据现成）；若仓库含权重则直接评 |
| **P-2** | B3 Beat This! | 伪标注飞轮预处理 | `pip install beat-this` 即用，给视频切片打节拍网格 |
| **P-3** | B2 drum-sample-retrieval-vocalisation | 人声↔鼓音色 embedding | 伪标注阶段做"人声→标准鼓"对齐/置信度判断 |
| **P-4** | B1 wav2vec2 鼓分类器 | 迁移初始化 | HF 下载→在 AVP-LVT 上 fine-tune 对比 P-1 |
| **P-5** | C1 Dasheng-Base 86M | 特征底座（P2 路线） | 与 C2/C3 跑特征对比评测 |

**最重要的战略判断**：beatbox 识别这个领域**太小众**（学术产出集中在 AVP-LVT 一个数据集、两位作者），不存在"下载权重直接用"的捷径——**正因如此，我们自训的识别模型本身就是稀缺资产**，伪标注飞轮的先发优势成立。

## 3. 与项目演进的衔接

- **P0（当前）**：标签设计直接采纳 A1 论文结论——**音节级监督 > 鼓级监督**（稳健性更好），佐证我们 v3 序列答案方向正确
- **P3 数据飞轮**：B3（节拍网格）+ 自训识别模型（onset+分类）+ B2（对齐 embedding）= 伪标注管线三件套
- 生成分支远期：A1 的 Vocal Grammatics 转写体系可当"鼓谱"格式的参考

## 4. 信息来源

- Delgado et al. 2022, Deep Embeddings for Robust User-Based Amateur Vocal Percussion Classification (arXiv 2204.04646)
- Delgado et al. 2022, Deep Conditional Representation Learning for Drum Sample Retrieval by Vocalisation (arXiv 2204.04651)
- Delgado et al. 2020, A New Dataset for Amateur Vocal Percussion Analysis (arXiv 2009.11737)
- Nao Tokui, Neural Beatbox (2018)；NVIDIA AI Art Gallery 报道
- MSU AI Club beatbox2drums 项目页；Beatbot demo 页
- JEP-TALN 2020 HMM-GMM beatbox 论文；JCSSP 2025 / IEEE 2020 / ISMIR 2005 传统 ML 论文
- PyPI beat-this；airasoul wav2vec2 鼓分类教程；CLAP/MERT/BEATs/OMAR-RQ/Dasheng 各官方仓库
