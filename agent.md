# agent.md — lightfour 项目主线（伪标注数据飞轮）

> **每次对话开始时必须先读本文件**，确保对项目目标、当前主线、进度和下一步动作有统一认知。
> 有重大决策、路线变更、训练进度变化时，**同步更新本文件**。

最后更新：2026-10-19

---

## 1. 项目定位与总体目标

**一句话**：做一个**节奏口技（beatbox）识别 + 生成**的个人训练项目，最终能识别电子音乐细类并用语音/beatbox 交互。

核心能力诉求：

1. **beatbox 音色识别**：听人声打击乐，区分 Kick / Snare / 闭镲 / 开镲（4 类起步，扩到 8-15 类）。
2. **beatbox 鼓谱转写**：把一段 beatbox 转成 onset + 鼓类的"鼓谱"序列。
3. **beatbox 生成（远期）**：给定鼓谱/音素序列，生成 beatbox 音频。
4. **音乐识别（次要）**：电子音乐子类分类（House/Techno/…，FMA 已有数据）。

**技术边界（已确认，勿再走弯路）**：
- 识别/转写 = 监督分类任务，**专用音频分类器可解**（关键在数据，不在大模型）。
- 生成 = 独立的条件生成任务，需专门的生成模型，**不是大模型微调能解决的**。
- **beatbox 领域太小众，没有"下载权重直接用"的捷径**（见 `docs/beatbox_recognition_models.md`）——正因如此，**我们自训的识别模型本身就是稀缺资产**，这是数据飞轮的先发优势。

---

## 2. 核心结论：为什么从 Omni 转向「专用识别模型 + 数据飞轮」

### 2.1 Omni（Qwen2.5-Omni）路线已验证失败 ❌

v1/v2/v3 三轮 QLoRA 微调的教训（详见 `docs/微调lightfour.md`）：

- v3 确实解冻了 audio_tower（384 个 LoRA keys），eval_loss 降到 0.06859（比 v2 低 22.5%），**但推理实测模型不听音频**：
  - **音频敏感性测试**：同一问题 + 不同音频 → 输出几乎不变；**完全不给音频 → 输出还是一样**（铁证）。
  - test 准确率 ≈ 随机（beatbox 29.5% vs 4 类随机 25%）。
  - eval_loss 低是"背模板"假象（答案 99% 是公共模板 token）。
- **根因**：① 答案模板主导 loss，类别 token 被淹没；② beatbox 单音太短（0.06-1.5s），Qwen-Omni 音频塔（语音导向）区分打击乐音色先天不足。
- **教训（写入验收铁律）**：**eval_loss 不能作为验收指标**。唯一可信指标 = **音频敏感性测试** + **test 准确率 vs 随机基线**。

### 2.2 战略转向：用专用音频分类器做识别

不与 Omni 的弱音频塔缠斗。改用**为音频/音乐设计的编码器 + 轻量分类头**（`docs/beatbox_recognition_models.md` 的 P-1/P-4/P-5）：

| 优先级 | 方案 | 用途 |
|---|---|---|
| **P-1** | vocal-percussion-transcription（Delgado）复训 | 识别主基座（数据集与我们一致） |
| **P-4** | wav2vec2 鼓分类器（HF）微调 | 迁移初始化，快速验证 |
| **P-5** | **Dasheng-Base 86M / Huge 0.6B**（小米，音乐理解专用） | 特征底座，最对口 |
| B3 | Beat This!（beat tracker） | 预处理管线的 onset/节拍网格 |

**关键**：专用分类器直接在音频上做分类，**天然会"听"音频**（不存在 Omni 的模板背诵问题），且在小数据上更快收敛。

---

## 3. 主线演进路线：伪标注数据飞轮

> **核心思路**：小标注集训识别模型 → 识别模型当"自动标注员"标注海量无标注 beatbox 视频 → 滚动生成训练库。
> 交付节点以「交付物 + 通过条件（Gate）」定义，不设时间预估。

### Stage 1 — 识别模型（飞轮启动前提）⭐ 当前阶段

**目标**：训一个真正"听音频"的 beatbox 分类器。

| 动作 | 交付物 | Gate（通过条件） |
|---|---|---|
| 1. 专用分类器（log-mel CNN 起步 → wav2vec2/Dasheng） | `scripts/train_beatbox_classifier.py` + checkpoint | 在 AVP-LVT 上跑通 |
| 2. 标签设计采纳音节级监督（A1 论文结论：音节级 > 鼓级） | 类目表（kd/sd/hhc/hho → 中文+音节+IPA） | 标签口径统一 |
| 3. **验收评测**：test 准确率 + 音频敏感性测试 | 评测脚本 | **4 类准确率 ≥ 70%**（随机 25%）；敏感性测试通过（不同音频→不同输出） |
| 4. 音频加长：2-5s 窗口 | `build_beatbox_windows.py` | 窗口 ≥2s 占比 100% |

**验收铁律**：准确率 ≥70% 且音频敏感性通过，才能进入 Stage 2。识别不过关，飞轮是垃圾进垃圾出。

### Stage 2 — 伪标注数据飞轮（识别 → 大数据库）

**目标**：把识别模型当自动标注员，滚出大规模 beatbox 鼓谱数据库。

| 步骤 | 动作 | 交付物 | Gate |
|---|---|---|---|
| A | 依赖 Stage 1 识别模型过关 | recognizer checkpoint | ≥70% |
| B | 下载 + 预处理管线：yt-dlp 下视频 → Demucs 人声分离 → VAD 剔说话 → onset 切片 | `scripts/preprocess_video.py` | 10 个视频切出干净片段 |
| C | 伪标注 + 置信度过滤 + 人工抽检 | `scripts/pseudo_label.py` | 抽检错误率 ≤10% |
| D | 扩类目（4→8-15 类）+ 滚大数据库 | 50-100h 片段 + 鼓谱序列 | 每类 ≥500 片段 |
| E | （见 Stage 3） | | |

**四条对策（必做）**：
1. **类目覆盖**：AVP-LVT 只有 4 基础鼓，视频里有 lip roll/throat bass/scratch 等 → **置信度过滤**（低置信弃掉）+ 类目逐步扩到 8-15 类。
2. **预处理**：视频混欢呼/说话/伴奏 → **Demucs 分离 → VAD → onset 切片**，只留纯人声打击乐段。
3. **版权**：YouTube/Swissbeatbox 视频**有版权**，**只下载做研究、不进 git、不外传音频/模型**，发布只放标注和指标（`docs/beatbox.md` 结论）。
4. **标注质量**：伪标注必有错，直接训生成会错误复利 → 置信度阈值 + **人工抽检 3-5%** + 高置信高权重。

### Stage 3 — beatbox 生成（远期分支）

**目标**：条件生成「鼓谱/音素序列 → beatbox 音频」。

- **不做** Omni 的"beatbox 当语言"开放式生成（数据差 2 个量级、声码器保真不了、"音色=音素"不成立）。
- **做条件生成**：用 AVP-LVT 的音节/IPA 标注当"乐谱"，训练「IPA/音素序列 → beatbox 音频」（类似 TTS 反向），数据需求小得多，且与识别模型共享数据标注。
- 数据来源 = Stage 2 飞轮滚出的「音频片段 + 鼓谱序列」配对库。
- **Gate**：能按给定鼓谱生成可辨识的 beatbox 片段（人耳能听出对应鼓类）。

---

## 4. 资源、环境与风险

| 资源 | 现状 | 协调 |
|---|---|---|
| 本地算力 | Windows / AMD RX 7800 XT（ROCm 不可用）/ RAM 15.4GB / D 盘 ~130GB 空闲 | **专用分类器本地 CPU/小模型可训**（log-mel CNN / wav2vec2 小模型）；大模型上云 |
| 云端 GPU | 趋动云 B1.gpu.large 25GB vGPU（历史用过） | Dasheng/wav2vec2 扩展训练、Demucs 批处理上云 |
| 持久化盘 | `/gemini/code`（seaweedfs，用量计费） | 产物直写持久化盘（v1 教训：overlay 盘停机即丢） |
| 数据资产 | AVP-LVT 5,714 boxeme（4,873 带音频）+ FMA 8,000 首 + Beatport 32 类 | 本地 `data/` |
| 模型资产 | Qwen2.5-Omni-3B（基座 11.97GB）+ v2 checkpoint-1000（已下载校验） | 作对照基线/对话层，非识别主路径 |

**风险应对**：
1. **识别不过关**（≥70% 不达标）→ 切换 backbone（log-mel CNN → wav2vec2 → Dasheng），不在单一模型恋战。
2. **伪标注污染**（抽检错误率 >10%）→ 提高置信度阈值、回头修识别模型，不盲滚数据。
3. **版权风险** → 研究用途、不入 git、不外传。
4. **实例停机丢产物** → 产物直写持久化盘 + 定期下载本地。
5. **本地算力不足** → 小模型本地验证，大训练上云。

---

## 5. 资产清单

### 数据
- `data/beatbox/AVP-LVT_Dataset/` — beatbox 原始（141 wav + 180 CSV，onset+鼓类+音节+IPA）
- `data/sft/beatbox_clips/` — 4,872 单音切片（wav）
- `data/sft/beatbox_sft.jsonl` / `fma_sft.jsonl` — SFT 样本
- `data/fma/` — FMA 元数据 + 8,000 首 mp3
- `data/beatport/` — 32 类 taxonomy + 榜单元数据

### 模型
- `lightfour/Qwen2.5-Omni-3B/` — 基座（对照/对话层）
- `lightfour/checkpoint-1000/` — v2 LoRA adapter（失败基线，留作对照）

### 文档
- `docs/beatbox.md` / `docs/edm.md` — 领域数据集调研
- `docs/beatbox_recognition_models.md` — 识别模型资源调研（P-1~P-5 选型依据）
- `docs/微调lightfour.md` — Omni 三轮微调完整记录（失败教训）
- `docs/omni的音乐输出.md` / `docs/omni_run_guide.md` — Omni 语音/运行

---

## 6. 会话启动检查清单（Agent 必做）

1. **读本文件** — 恢复主线（数据飞轮）与当前阶段。
2. **验收铁律**：识别/评测只认 **test 准确率 + 音频敏感性测试**，**绝不看 eval_loss**（§2.1 教训）。
3. Stage 1 未过 Gate（≥70%）前，**不启动** Stage 2 飞轮滚数据。
4. 远程操作：产物直写 `/gemini/code`；实例可能重建，先确认通道。
5. 版权约定：爬取音频只做研究、不入 git、不外传。

---

## 7. 安全与规范

1. **禁止**提交 SSH/SFTP 口令、API Key、`恢复会话_*.md`、`docs/微调lightfour.md` 之外的敏感信息进 git。
2. 新脚本以 `D:\lightfour` 或相对路径为准，**不要写死**历史路径。
3. 大文件（zip/wav/safetensors）、原始音频数据保持 gitignore。
4. 爬取/采样音频做训练注意**版权与研究用途**声明。

---

## 8. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 首次建立 agent.md |
| 2026-10-13 | 核实趋动云环境、`/gemini/pretrain` 权重 |
| 2026-10-15 | v2 Omni QLoRA 失败（不听音频），定位 freeze_vit 根因 |
| 2026-10-16 | v3 修复 audio_tower 仍不听音频；音频敏感性测试证实模板背诵；eval_loss 判死刑 |
| 2026-10-18 | 识别模型资源调研（`beatbox_recognition_models.md`）：beatbox 太小众无现成权重，自训即稀缺资产 |
| 2026-10-19 | **战略转向**：Omni → 专用音频分类器 + 伪标注数据飞轮（Stage 1 识别 → Stage 2 飞轮 → Stage 3 生成）定为主线 |
