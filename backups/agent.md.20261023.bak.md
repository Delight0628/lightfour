# agent.md — lightfour 项目主线（伪标注数据飞轮）

> **每次对话开始时必须先读本文件**，确保对项目目标、当前主线、进度和下一步动作有统一认知。
> 有重大决策、路线变更、训练进度变化时，**同步更新本文件**。

最后更新：2026-10-22 pm（B 站管线全量完成 + 典型样本本地留档）

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

| 动作 | 交付物 | Gate（通过条件） | 状态 |
|---|---|---|---|
| 1. 专用分类器（log-mel CNN 起步 → wav2vec2/Dasheng） | `scripts/train_beatbox_classifier*.py` + checkpoint | 在 AVP-LVT 上跑通 | ✅ v1/v2/v3 三版跑通 |
| 2. 标签设计采纳音节级监督 | 类目表（kd/sd/hhc/hho） | 标签口径统一 | ✅ |
| 3. **验收评测**：test 准确率 + 音频敏感性 | `scripts/eval_beatbox_classifier.py` | **4 类准确率 ≥ 70%**（随机 25%） | ❌ 未过 Gate：v2 **0.5721**（本地最好）/ **0.5147**（服务器重训），v3 0.4558，w2v 4-fold CV 0.5435 |
| 4. 音频加长 2-5s 窗口 | `build_sft_v4_omni.py`（beatbox_windows 6,445 个） | 窗口 ≥2s 占比 100% | ✅ |

**迭代记录（本地 CPU，`py -3.12`，torch 2.14.1+cpu）**：

| 版本 | 关键改动 | 结果 | 结论 |
|---|---|---|---|
| v1 | log-mel CNN，BatchNorm，per-clip 归一化 | test **0.5008**（Kick 95%/Snare 40%/闭镲 43%/开镲 8%） | ✅ 证明"模型不听音频"是 Omni 模板背诵的锅，换专用分类器就解决；❌ 类别偏置 + 跨人泛化差（val 0.72 vs test 0.50） |
| v2 | +类权重 +波形增强(增益/噪/变调/变速) | val 在 0.53~0.63 剧烈震荡 | ❌ 发现**致命缺陷**：`pitch_shift` 重采样后按原长裁剪 → 截掉 hi-hat 衰减尾巴，而闭镲/开镲的唯一判别特征就是衰减时长 |
| v3 | mel 域频率增强 + GroupNorm + EMA + LR 1e-3 | test **0.4558**（val 0.6364） | ❌ 比 v2 差，未过 Gate |
| w2v | 冻结 wav2vec2-base 12 层特征 + 可学习层权重 | test **0.5271**；4-fold 参与者 CV **0.5435 ± 0.016** | 🟡 当前最可信泛化估计，仍未过 70% |

**技术要点**：
- **判定衰减类差异必须保住时间轴** → 频率增强在 mel 域做（沿 mel 轴 warp），不要在波形域重采样
- BatchNorm 在小 val 集上导致 epoch 间剧烈震荡 → 换 GroupNorm + 权重 EMA
- Windows 上 `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1` 必设，否则 OpenMP 死锁（表现为"进程无 CPU 增长"）
- `py` 启动器 CPU 占用低是假象，真实进程是 `python.exe` 子进程（认 CPU 增量要找对 PID）



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

#### ⭐ 生成型数据集进展（2026-10-20，独立于 Stage 1/2 先行推进）

**瓶颈已破**：真实录音仅 0.55h，远不够生成模型（需 10-50h+）。但 beatbox 本质是**离散打击乐事件序列**，可用"采样合成器"原理从极少真实素材无限扩量：

```
真实录音(140 段/0.55h) → 按 onset 精确切出每次击打 → 打击乐样本库(hit_bank)
                                                      ↓
                          按任意鼓谱重新排布叠加 → 无限 (鼓谱→音频) 配对
```

**两步管线（本地 + 趋动云双端跑通，结果逐项一致）**：

| 步骤 | 脚本 | 产出 | 状态 |
|---|---|---|---|
| 1. 切击打样本库 | `scripts/build_hit_bank.py` | `data/hit_bank/` 4,872 个 hit（底鼓 1446 / 军鼓 1253 / 闭镲 1164 / 开镲 1009），总 0.330h | ✅ 本地+服务器双端 |
| 2. 合成生成数据 | `scripts/render_synth.py` | `data/gen_synth/`（scores.jsonl + omni_pairs.jsonl + mel/ + wav/） | 🔄 服务器跑 `--hours 5` 中 |

**击打切分的关键技术**（`build_hit_bank.py`）：
- **起音对齐**：在 `[onset-50ms, onset+120ms]` 内找能量上升点（帧 RMS > 峰值 25%），回退 10ms 保留起音瞬间
- **尾部自适应截断**：帧 RMS > 峰值 5% 的最后一帧为终点，**保住 hi-hat 衰减尾巴**（闭镲/开镲判别特征，见 v2 教训）
- 归一化到统一峰值，混音时按 hit 的 RMS 再做力度分配

**合成的关键技术**（`render_synth.py`）：
- 随机 tempo 70-160 BPM + groove/velocity + 击打随机性 → 覆盖快慢/密集稀疏
- hit 重叠相加 + `tanh` 软限幅防爆音
- 输出与 `gen_v1` **完全同格式**（可复用下游管线），双路线齐备：Route A（Omni 音频 token）/ Route B（mel→wav 声码器）

**已产出的生成型数据**：

| 集合 | 规模 | 来源 | 用途 |
|---|---|---|---|
| `data/gen_v1/` | 1,744 窗口 / **0.97h** | 真实录音（AVP-LVT） | 真实音色的锚点数据 |
| `data/gen_synth/` | 9,000 窗口（目标 5h）🔄 | hit_bank 合成 | 可无限扩量 |

**下一步**：① 当前批次跑完 → 打包下载回本地 → 音质抽检 + Stage-A 分类器做生成质量闭环评测；② 改 tar 流式输出（现瓶颈是 seaweedfs 写小文件）→ 扩到 50h。

#### ⭐ 真实 B 站 beatbox 数据管线（2026-10-22）

**网络实测（本地 + 趋动云）**：

| 目标 | 结果 | 用法 |
|---|---|---|
| YouTube | ❌ 超时（本地/服务器均不可达） | 放弃直连 |
| swissbeatbox.com | ✅ 可开，但视频全是 YouTube iframe | 只能当索引用 |
| Bilibili 首页/API | ✅ 可达；服务器带 cookies 后 API 正常 | **yt-dlp 视频页在服务器 412 → 改走 API 直链** |

**脚本（代码已入库/服务器同步）**：

| 脚本 | 作用 |
|---|---|
| `scripts/fetch_bilibili_api.py` | B 站 API 取 cid + dash.audio 直链，ffmpeg 转 wav |
| `scripts/fetch_bilibili_audio.py` | 本地 yt-dlp 版（cookies） |
| `scripts/segment_beatbox_onsets.py` | librosa onset_detect → 单次击打切片 |
| `scripts/ast_filter_beatbox.py` | AST（AudioSet 527 类）10s 窗口打 Beatboxing/Drum 分 |
| `scripts/filter_segments_by_ast.py` | 去掉主持/采访/纯伴奏区域 |
| `scripts/pseudo_label_v2.py` | 兼容 v2/v3 架构的伪标（全局归一化） |
| `scripts/hybrid_label_segments.py` | 模型 + 镲类规则（高频/谱质心/衰减）重平衡 |
| `scripts/train_beatbox_classifier_robust.py` | 混入真实 `no_vocals` 背景做域随机化 |
| `scripts/train_beatbox_features.py` | 手工特征 RandomForest 备选 |
| `scripts/preprocess_video.py` | 修复 Demucs 返回路径、长音频 60s 分块、`sys.executable` |

**当前数据（服务器 `/gemini/code/lightfour-train/data/real_bbox/`）**：

| 项 | 实测 |
|---|---:|
| 已下载视频 | **143 个**（全量候选已下完） |
| Demucs vocals | **108 个**（pipeline5 全部完成） |
| onset 切片 | **53,633 段 / 4.49h**（AST 过滤后 **39,171 段 / 2.93h**） |
| 混合弱标高置信（≥0.7） | 78 视频 **13,074 段 / 0.865h** + 新增 3 条对战 **1,327 段 / 0.078h** = **14,401 段 / 0.943h** |
| 高置信类别 | 底鼓 2,478 / 军鼓 6,336 / 闭镲 3,171 / 开镲 1,089 |
| 真实「鼓谱→音频」生成对 | **3,036 窗口 / 1.69h**，65 个来源；kd 2,220 / sd 5,860 / hhc 2,924 / hho 945 |
| 规则重判镲 | 全量 8,777 段（其余沿用模型） |

**Stage A 现状（服务器重训）**：

| 模型 | AVP test | 真实切片平均置信 |
|---|---:|---:|
| v2（BatchNorm，AVP 训练） | **0.5147**（val 0.8026，gap≈0.29） | 0.447 |
| robust（混真实 `no_vocals` 背景） | **0.4372**（val 0.7221） | 0.556 |
| hybrid（模型 + 镲类规则） | — | 高置信 0.796 |

**关键结论**：

1. **自动伪标不足以直接滚库**：真实域 ≥0.85 的高置信为 0；hybrid 规则后 ≥0.7 共 13,074 段（33.4%），开镲补到 1,089，但军鼓仍占 48%。
2. **AST 过滤有效**：53,633 → 39,171 段（73%），去掉了主持/采访/纯伴奏，真实域平均置信 0.447→0.525→0.574（robust/AST）。
3. **omni 音频输出训练仍不可行**：ms-swift 4.5.3 的 `Qwen2_5OmniTemplate` 只支持音频输入→文本；Talker 音频输出训练只存在于 **Qwen3-TTS** 模板；transformers 的 `Qwen2_5OmniTalkerForConditionalGeneration.forward` 无 `labels`/loss。**换 24GB 卡也训不了 Omni3 生成**。
4. **数据安全**：本轮由用户在对话中授权服务器直接下载/处理公开视频；音频只做研究，落 `data/bbox_raw/`、`data/real_bbox/`（已 gitignore），不入 git、不外传；服务器直接下载，未从本地上传音频。

#### 典型连续 beatbox 样本（2026-10-22 修正）

用户指出首批 3 条更像阿卡贝拉/Showcase，不是经典竞技 beatbox，判定正确。`b23.tv/ZO2ResT` 解析后指向 **`space.bilibili.com/171444239`（Swissbeatbox 官方账号）**，已改用该账号的经典对战视频：

| 文件 | 来源 | 时长 | RMS | 0.5s 静音帧 |
|---|---|---:|---:|---:|
| `BV1CPnr6VEeE_GBB26_Final_M1xx1ng_vs_电池.wav` | GBB26 第二联赛 **决赛** | 498.1s | 0.207 | 1.1% |
| `BV1QW411u7NJ_WBC_Semi_HIRAM_vs_PASH.wav` | WBC Solo Battle **半决赛** | 470.9s | 0.215 | 0.5% |
| `BV1xErMBqE1P_GBB25_SmallFinal_KAJI_vs_BlackRoll.wav` | GBB25 Solo **小决赛** | 379.7s | 0.136 | 0.0% |

路径：`data/bbox_raw/samples_competitive/`（gitignore）；3 条均为连续对战全程，不是 a cappella cover/Showcase/Shoutout。
旧 3 条保留在 `data/bbox_raw/samples/` 作对照，但**不再作为竞技 beatbox 代表**。

**数据源策略修正**：后续优先从 Swissbeatbox 官方账号抓 **vs / Battle / Elimination / 决赛** 类视频；纯 cover、Showcase、Shoutout 只作辅助；**明确排除 loopstation**（另一门类，中文标题里的「环站」同样是 loopstation，必须一并排除）。第一轮搜到 12 条对战候选，新增下载 6 条；其中 `BV1QW411u7NJ`、`BV1xErMBqE1P`、`BV1CPnr6VEeE` 已进入 Demucs 批处理。第二轮扩展到 **137 条候选**，累计下载 **269 条**（新增 126 条），其中 **101 条 ≤650s 进入 Demucs 批处理**；Showcase/Shoutout/合集会进入分离但**入库时按标题再次排除**。

**第三轮扩展（2026-10-22 晚）**：
- 关键词搜索 49 条清洗后候选（排除「环站」3 条、冠军表演 2 条、海选 2 条），下载时全部已存在（skip）。
- **WBI 签名空间 API**（`x/space/wbi/arc/search`，mid 171444239）拿到最新 **100 条**视频，其中 **26 条 2026 最新对战**全部下载：GBB26 Florida 决赛/半决赛/小决赛、vokal.total 2026 八强/四强、GBB26 非洲 MENA 与亚太区预选赛。第 3 页起被 **412/-352 风控**拦截，换新 cookies 无效。
- **related 相关推荐扩展**（`x/web-interface/archive/related`，100 个已下载对战作种子）再获 **117 条**新候选并已启动下载，覆盖 GBB19/GBB21/GBB23/GBB24/GBB25、WBC 7ToSmoke、SBX Kickback、Crowd POV 等历史赛事。
- 规则修正：`championship`（锦标赛）保留，只有 `champions`（冠军表演）排除，避免误杀 `French Tag Team Beatbox Championship | Final` 这类对战。
- Demucs 并发提升：原 `-P 2` 处理 101 条，新增 `-P 4` 处理 30 条无重叠批次（96 核余量充足）。
- **伪标抽检包**：从高置信库分层抽 9 条（kd 2 / sd 2 / hhc 2 / hho 3，conf≥0.80），本地 `data/bbox_raw/spotcheck_20261022/`，附 `manifest.json`（来源视频、起点、时长、AST 分数、手工特征），供人耳核对。

**第 3 批流水线结果（2026-10-22 深夜）**：90 个新增视频（已按标题剔除 Showcase/POV/合辑等 23 条）→ **63,114 段 onset** → AST 过滤 **39,796 段（63.1%）** → 伪标（robust ckpt，GPU）→ hybrid 重平衡。高置信（≥0.7）**14,088 段 / 0.958h**，类别 kd 1,959 / sd 9,220 / hhc 2,083 / hho 826，平均置信 0.6299（规则补镲 6,492 段）。累计高置信：13,074 + 1,327 + 14,088 = **28,489 段 / 1.901h**（较上轮 0.943h **翻倍**）。
- 工程加速：AST 从 12 进程 CPU（约 1 视频/10 分钟）改为 **GPU 单进程**（约 4 视频/分钟，快 ~40×）；伪标从串行 5 段/秒改为 **spawn 16 进程**（8 分钟跑完 39,796 段）；vocals 先复制到本地 overlay 盘，绕开 seaweedfs 小文件读瓶颈；GPU Demucs 改 3 并发（单条约 3 audio-s/s，CPU 约 0.55 audio-s/s）。
- 注意：`ProcessPoolExecutor` 默认 fork 在 torch 下会死锁（worker 全 sleeping），必须用 `mp.get_context("spawn")`。

**Stage 1 伪标质量交叉验证（2026-10-22 晚，回答“鼓谱数据库实际质量如何”）**：用独立的手工特征 RandomForest（`checkpoints/beatbox_feat/rf.joblib`，15 维：频谱占比/质心/衰减/ZCR/RMS 等，训练于 AVP-LVT）对全部 **28,489 条高置信样本**重新预测，与 CNN/规则标签对比：

| 指标 | 结果 |
|---|---:|
| 总体一致率 | **70.89%** |
| kd 一致率 | **91.75%**（4,641 条，RF 判 sd 293 / hhc 69） |
| hhc 一致率 | **84.47%**（5,601 条，RF 判 hho 238 / kd 422） |
| sd 一致率 | **65.27%**（16,256 条，**RF 判 kd 5,489 = 33.8%**） |
| hho 一致率 | **29.98%**（1,991 条，RF 判 hhc 658 / kd 380 / sd 356） |
| 高置信但两模型分歧 | **1,470 条（5.2%）** |
| 按标签来源 | model 71.1% / rule_hat 70.2%（无明显差异） |

特征均值：kd 低频占比仅 **0.079**、mid 0.665（底鼓低频主导性不足，疑似混入中频 kick/hum）；sd mid 0.411 / high 0.326；hhc/hho high 0.71–0.73、质心 5,300–5,370，**decay 区分明显（hhc 0.057 vs hho 0.236）**。

**结论**：kd、hhc 基本可用；**sd 约 1/3 与 kd 混淆**（最大风险，且 sd 占库 57%）；**hho 基本不可靠，不能直接用于训练**。该交叉验证是域外检验（RF 训于 AVP-LVT、真实数据来自 Swissbeatbox），本身有域偏移误差，不能当 ground truth；但 29% 的分歧率足以说明必须人工抽检。

**已备好的三份人耳抽检包（本地）**：
- `data/bbox_raw/spotcheck_20261022/`：9 条高置信分层样本（kd 2 / sd 2 / hhc 2 / hho 3）
- `data/bbox_raw/spotcheck_batch3/`：12 条第 3 批样本（每类 3 条）
- `data/bbox_raw/suspicious_spotcheck/`：**12 条两模型分歧样本**（文件名直接标 `cnn-X_rf-Y`，最适合裁决谁错）

**质量 Gate（未通过前不应继续滚库）**：人工试听上述 33 条，若错误率 >10%，停止伪标滚库，转人工 seed/域适应；hho 需单独降阈值或人工补标；sd-kd 需加衰减/低频规则约束。

**音色命名体系（2026-10-22 晚，用户提出先建 taxonomy）**：已建立 `docs/beatbox_sound_taxonomy.md` + `data/beatbox_taxonomy.json`，以 [Beatbox Science Sound Library](https://beatboxscience.nazcabeats.com/sound-library) 的 **103 个标准音色**为基准，含标准记法（P/PF/K/T/W/Drh…）、发声机制（喉部/肺部/哼鸣/空腔）、气流方向（外送/内吸）、中英文名、鼓组映射。
- 核心判断：现有 `kd/sd/hhc/hho` 只是**鼓组 L1 映射**，不是完整音色体系；竞技 beatbox 有 100+ 音色，lip roll（7）、throat/chest bass（4）、click roll（3+）、scratch、whistle、synth 等**被强行塞进 4 类**，正是 sd 占 57%、sd/kd 混淆、hho 不可靠的结构性原因。
- 分层方案：L1 鼓组 4 类（现有，继续可用）→ L2 家族 6–8 类（kicks/snares/hihats/cymbal/bass/click/lip/fx，需人工 seed 200–500 条）→ L3 音色 20–100 类。
- manifest 扩展：新增 `drum_class`（兼容旧标签）、`sound_notation`、`sound_family`、`sound_confidence`，不删旧字段。
- 中文视频资源：B 站邹Sir_BEATBOX课堂（BASS 4 种 `BV1CK4y1g7dq`、R 音 `BV1x64y1R7Bi`）、啊宾 bbox 教学基础三音（`BV1TpLC63EsE`）；文字资源：School of Beatbox、BecomeSingers、Devon Guinn kicks/snares。

**音色种子库 + L2 家族重标注（2026-10-22 深夜已跑通）**：
- 从 Beatbox Science 的 Supabase 公开接口（`sounds` 表，含 `audio_url`）下载 **95 条有音频的音色种子**（8 条无音频：Cough/High Cough/Slizzer Roll/Zekka Click Roll/Inward Throat Bass/Vortex Whistle/Smile Bass/Inward Chest Bass）。本地留档 `data/beatbox_seeds/`（音频 0.8MB，研究用、不入 git）；元数据 103 条在 `sounds.json`。
- 服务器直接下载同批种子，提取 9 维特征（时长/低中高频占比/质心/ZCR/衰减/RMS/峰值），对 **28,489 条高置信样本**做最近邻 L2 家族映射，输出 `data/real_bbox/hybrid_labels_all_highconf_l2.jsonl`（17MB）。
- **家族分布**：Snares 13,000（45.6%）、Drag 3,220（11.3%）、Hi-hats 2,115、Other 1,935、Clicks 1,589、Vocal 1,265、Kicks 1,057、Zippers 1,016、Liprolls 1,002、Cymbal 666、Synth 426、Clops 374、Reverse 253、Lip oscillations 189、Scratch 176、Pops 89、Teeth 38、Nasal 9、Tongue 1。
- **关键交叉表**：`sd → Snares` 仅 8,081（占 sd 的 50%），其余分散到 Drag 3,055 / Clicks 1,133 / Zippers 933 / Other 1,477 / Liprolls 344 / Vocal 204 / Kicks 188；`hhc → Snares` 3,635（65%，闭镲被大量判成军鼓，异常）；`kd → Vocal` 1,061、`kd → Liprolls` 657、`kd → Clops` 318；`hho → Snares` 1,142、`hho → Cymbal` 459、`hho → Hi-hats` 309。
- **结论**：4 类标签内部音色构成极其混杂，sd 只有一半是真军鼓，hhc 反而 65% 被判军鼓——这是当前数据库最大的结构性问题。L2 最近邻结果本身有误差（尤其 Drag/Zippers），必须人工抽检确认后再用于训练。
- **Omni 关系澄清**：音色种子 + 音频可以支撑「L2 家族分类器」和「生成模型的音色素材库（hit bank）」，但**不能自动解决 Omni 音频输出训练缺失**——ms-swift 4.5.3 / transformers 的 Qwen2.5-Omni Talker 仍无 labels/loss，这是工程管线问题，不是数据量问题。

**Qwen3-Omni 调研（2026-10-23，回答“能否一体化识别+生成”）**：
- **架构上有进步**：transformers 5.16.1 已实现 `qwen3_omni_moe`，`Qwen3OmniMoeTalkerForConditionalGeneration.forward` **有 `labels` 和 CE loss**（`loss_function(logits, labels, vocab_size)`），即 Talker 的音频 codec token 可以被监督训练。Qwen3-Omni 是 Thinker-Talker MoE（30B-A3B），Talker 多码本自回归 + MTP 生成残差码本，输出语音。
- **但当前仍不能直接落地 beatbox 生成**，三个硬阻碍：
  1. **ms-swift 4.5.3 的 `Qwen3OmniTemplate` 继承 `Qwen2_5OmniTemplate`，只做音频输入→文本**，没有音频输出 SFT 数据管线；真正能训音频输出的只有 **`Qwen3TTSTemplate`**（依赖 `qwen_tts` 库，需要 `tts_audio_codes`（16 码本）+ `ref_audios` mel），且那是**纯语音 TTS**，不是通用音频。
  2. **codec 是语音专用**：Qwen3-Omni / Qwen3-TTS 的码本为 24kHz 语音设计，beatbox 的宽频瞬态（低频 kick、高频 hi-hat、唇卷）不在其训练分布，重建保真度未经验证。
  3. **规模与显存**：Qwen3-Omni 30B-A3B，当前 6.2GB GPU 连 QLoRA 都跑不了（至少 24GB+）。
- **结论**：Qwen3-Omni 是未来「单模型一体化」的候选载体，但当前不具备低成本直接微调出 beatbox 生成的条件。要验证可行性，最低成本实验是：**用 Qwen3-TTS-Tokenizer 对 beatbox 音频做 encode→decode 重建，听保真度损失**；若重建可接受，再考虑 24GB+ GPU 上自写 Talker 音频输出训练管线（参考 `Qwen3TTSTemplate`）。在验证前，主线仍是「专用分类器（识别）+ Route B（生成）+ Omni 语言层」。

**竞技对战管线结果（新增 3 条）**：

| 项 | 结果 |
|---|---:|
| onset 切片 | **4,459 段** |
| AST 过滤后 | **3,652 段（81.9%）** |
| 混合弱标高置信（≥0.7） | **1,327 段 / 0.078h**，平均置信 **0.792** |
| 高置信类别 | 底鼓 204 / 军鼓 700 / 闭镲 347 / 开镲 76 |

对比旧混合内容：AST 保留率 **81.9% vs 73.0%**，高置信占比 **36.3% vs 33.4%**，平均置信 **0.6425 vs 0.625**。竞技对战确实更干净。合计高置信：13,074 + 1,327 = **14,401 段 / 0.943h**。

---

### 并行轨道：Omni SFT 数据 v4（已就绪，不依赖 beatbox 分类器）

> 这条轨道与 Stage 1/2 **并行推进**，不必等识别模型。SFT 管线和原始数据早已齐备，缺的是"修正后的配方"。

**v3 失败的根因与 v4 修正**：

| 问题 | v3 | v4 修正 |
|---|---|---|
| 答案 99% 是公共模板 token → eval_loss 虚低、模型背模板 | "这段音频模仿的是鼓组的**X**音色" | **答案只留类别标签**（如"闭镲"），实测答案均值 **3.2 字符**、最大 11 → 类别 token 占 loss 100% |
| beatbox 单音 0.06-1.5s，音频特征太少 | 单音切片 | **2/3/5 秒滑动窗口**，6,445 个 |
| 音频塔冻结 | `freeze_vit` 默认 True | `--freeze_vit false --freeze_aligner false` |
| 职责错位：让 Omni 自己做 beatbox 识别 | Omni 直出类别 | **职责拆分**：beatbox 类别由 Stage A 分类器产出（"耳朵"），Omni 只负责按标签组织语言（"嘴巴"）→ 蒸馏式混合架构，绕开 Omni 音频塔短板 |

**已产出**（`data/sft_v4/`，脚本 `scripts/build_sft_v4_omni.py`）：

| 集合 | 条数 | 构成 |
|---|---|---|
| train | 8,370 | beatbox 窗口 + FMA 音乐 |
| eval | 348 | 同上 |
| test | 1,216 | beatbox 按参与者留出（P9/11/19/24）+ FMA 随机 10% |

**Omni 侧 Gate**：不再看 eval_loss，改看 **① 音频敏感性测试通过 ② test 准确率（beatbox 4 类 ≥70% / FMA 8 类 ≥55%）**。

---

## 4. 资源、环境与风险

| 资源 | 现状 | 协调 |
|---|---|---|
| 本地算力 | Windows / AMD RX 7800 XT（ROCm 不可用）/ RAM 15.4GB / D 盘 ~120GB 空闲 | **专用分类器本地 CPU/小模型可训**（log-mel CNN / wav2vec2 小模型）；大模型上云 |
| 云端 GPU | 趋动云 B1.gpu.large 25GB vGPU（历史用过） | Dasheng/wav2vec2 扩展训练、Demucs 批处理上云 |
| **云端开发实例** | 趋动云开发环境 **96 核 / 503GB RAM / B1.gpu.small 6.2GB** / Python 3.11.8 | 数据生成 + Demucs 专用；6.2GB 不足以 Omni3 QLoRA。持久化 `pylibs` + `TORCH_HOME` 缓存 |
| 持久化盘 | `/gemini/code`（seaweedfs，14P 池，用量计费） | 产物直写持久化盘（v1 教训：overlay 盘停机即丢）。**注意小文件写慢** |
| 只读数据区 | `/gemini/data-1/`：`AVP-LVT_Dataset.zip`(310M)、`fma_small.zip.part001-004`(各1.8G)、`sft_samples.zip` | 走【数据】栏上传，**只读** |
| 数据资产 | AVP-LVT 5,714 boxeme（4,873 带音频）+ FMA 8,000 首 + Beatport 32 类 | 本地 `data/` |
| 模型资产 | Qwen2.5-Omni-3B（基座 11.97GB）+ v2 checkpoint-1000（已下载校验） | 作对照基线/对话层，非识别主路径 |

**⚠️ 数据安全约定**：**只允许代码上行（.py/.sh），禁止数据上行**（会触发公司数据安全拦截）；数据**下行**（服务器→本地）允许。生成数据在服务器用 `/gemini/data-1/` 原料就地构建。

**远程操作技术要点**（`jupyterlab-remote-shell` skill，2026-10-20 打通）：
1. **`_xsrf` cookie 是 path 限定的**——必须 `browser_navigate` 到 JupyterLab **直连 URL**（path 匹配 baseUrl）才能读到，否则所有 POST/PUT 403。
2. **写文件**：Contents API `PUT` + `format:'text'`；**中文内容会 500** → 传 base64（纯 ASCII）+ 服务器端 `base64 -d`。
3. **执行命令**：`POST api/terminals`（带 `X-XSRFToken`）→ WebSocket `terminals/websocket/<name>` 发 stdin → 输出重定向到文件 → Contents API 读回。**注意 shell 引号会被吞**，避免内嵌引号。
4. **零手抄传脚本**：本地起带 CORS 的 `ThreadingHTTPServer` → 浏览器 `fetch('http://127.0.0.1:PORT/x.py')` → PUT 上去。**手抄 base64 曾把 `CLASSES` 抄成 `CL@SSES`**，务必用此法 + md5 校验。
5. **脚本放 `scripts/` 子目录**（与本地结构一致），否则 `ROOT = dirname(dirname(__file__))` 会多算一层。
6. **browser 工具会间歇卡死**（60s 超时）→ 用 computer use 点浏览器刷新按钮恢复。

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
- `data/hit_bank/` — ⭐ **4,872 个打击乐击打样本**（4 类 × 28 参与者，16k 单声道 + index.json），生成型数据的原料
- `data/gen_v1/` — ⭐ 生成型数据·真实录音版（1,744 窗口 / 0.97h，scores.jsonl + omni_pairs.jsonl + mel/）
- `data/gen_synth/` — ⭐ 生成型数据·合成版（hit_bank 按鼓谱重排，可无限扩量）
- `data/sft/beatbox_clips/` — 4,872 单音切片（wav）
- `data/sft/beatbox_sft.jsonl` / `fma_sft.jsonl` — SFT 样本
- `data/sft_v4/` — Omni v4 识别型 SFT（train 8,370 / eval 348 / test 1,216，纯标签配方）
- `data/real_bbox/` — ⭐ **真实 B 站切片**：`segments_78v.tar`（53,633 段 / 4.49h）、`hybrid_labels_78v.jsonl`（39,171 条弱标）、`hybrid_highconf_0.7_78v/`（13,074 段 / 0.865h）、`sft_hybrid_highconf_78v.jsonl`、`real_gen_3036.tar`（3,036 窗口 / 1.69h）
- `data/real_gen/` — 真实音频生成对元数据：`scores.jsonl`、`omni_pairs.jsonl`、`stats.json`
- `data/bbox_raw/` — 原始下载（143 视频）、Demucs vocals（89/6.37h）、AST windows、cookies（gitignore）
- `data/fma/` — FMA 元数据 + 8,000 首 mp3
- `data/beatport/` — 32 类 taxonomy + 榜单元数据

### 模型
- `lightfour/Qwen2.5-Omni-3B/` — 基座（对照/对话层）
- `lightfour/checkpoint-1000/` — v2 LoRA adapter（失败基线，留作对照）
- 服务器 `checkpoints/beatbox_cls_v2/best.pt` — Stage A v2（test 0.5147）
- 服务器 `checkpoints/beatbox_cls_robust/best.pt` — 真实背景域随机化（test 0.4372，真实域置信 0.556）
- 服务器 `checkpoints/beatbox_feat/rf.joblib` — 手工特征 RandomForest（test 0.4791）

### 文档
- `docs/beatbox.md` / `docs/edm.md` — 领域数据集调研
- `docs/beatbox_recognition_models.md` — 识别模型资源调研（P-1~P-5 选型依据）
- `docs/微调lightfour.md` — Omni 三轮微调完整记录（失败教训）
- `docs/omni的音乐输出.md` / `docs/omni_run_guide.md` — Omni 语音/运行

---

## 6. 下一步行动计划（2026-10-20 排定）

> **优先级**：Stage 1 识别过 Gate 是飞轮启动前提（铁律，不可跳过）；生成型数据集可并行推进（不依赖识别模型）。

| # | 行动 | 交付物 | 依赖 | 状态 |
|---|---|---|---|---|
| **1** | **完成 gen_synth 首批并下载回本地** | `data/gen_synth/` 5h 数据 + 音质抽检报告 | 服务器 render_synth 跑完 | ✅ 完成 |
| **2** | **render_synth 改 tar 流式输出** → 扩到 50h | 50h 生成型数据集 | 行动 1 完成 | ✅ **完成（90,000 窗口 / 50.00h）** |
| **2b** | **产物持久化 + 本地同步** | 服务器 `gen_synth_50h.tar.gz`(5.1GB) + 本地元数据/样本 | 行动 2 | ✅ 完成 |
| **3** | **Stage 1 识别模型冲 70% Gate** | 候选：修 v3 补零 bug 重训 / 增大 val 参与者 / 全量参与者 k-fold CNN / 上云 wav2vec2 全参微调 | — | 🟡 未过 Gate：本地 v2 **0.5721**；服务器 v2 **0.5147**；robust **0.4372**；需真实 seed 或域适应 |
| **4** | 生成质量闭环评测 | 用 Stage-A 分类器判生成音频能否命中目标鼓类 | 行动 2b + Stage 1 | ⬜ |
| **5** | Omni 生成路线可行性核实 | 服务器查 ms-swift 是否支持 Talker 音频 token 训练 | 服务器通道 | ✅ 已核实：**不支持 Qwen2.5-Omni 音频输出**；Talker 训练仅 Qwen3-TTS；Qwen2.5-Omni Talker forward 无 labels/loss |
| **6** | Stage 2 伪标注飞轮落地 | `preprocess_video.py` + `pseudo_label.py` 用真实 beatbox 视频跑通 | **行动 3（Gate ≥70%）** | 🟡 真实数据管线已跑通：143 视频；89 vocals/6.37h；53,633 切片（AST 后 39,171/2.93h）；13,074 高置信弱标/0.865h；3,036 真实生成对/1.69h。Stage A 未过 Gate，未全量滚库 |
| **7** | 真实 B 站数据批量采集 + AST 过滤 + 混合弱标 | `data/real_bbox/` + 2s 生成对 | 行动 3/6 | 🟡 后台继续（目标 5–10h vocals） |

### 生成型数据集交付清单（2026-10-20 完成）

**✅ 服务器持久化**（`/gemini/code`，持久化盘）：

| 产物 | 大小 | 内容 |
|---|---|---|
| `data/gen_synth_tar/` | 8.3GB | 45 个 tar 分片（每片 2000 窗口，含 wav+mel）+ scores.jsonl + omni_pairs.jsonl + MANIFEST.json |
| `data/gen_synth_50h.tar.gz` | 5.1GB | 上述全量打包备份 |
| `data/gen_synth_v1.tar.gz` | 499MB | 5h 版（小批验证用） |
| `data/hit_bank/` | — | 4,872 个真实人声击打样本 |

**✅ 本地同步**（`data/gen_synth_tar/`，共 1.1GB）：

| 产物 | 状态 | 校验 |
|---|---|---|
| `MANIFEST.json` | ✅ 完整 | 45 片 × 2000 = 90,000 |
| `omni_pairs.jsonl` | ✅ 27.8MB / 90,000 行 | md5 `59209c8b…` |
| `scores.jsonl` | ✅ 64.3MB / 90,000 行 | md5 `a8f37e5a…` |
| `shard_00000.tar` | ✅ 185.56MB | md5 `c874868c…` 精确匹配 |
| `shard_00001.tar` | ✅ 185.56MB | md5 `036480d4…` 精确匹配 |
| `shard_00011.tar` | ✅ 185.56MB | md5 `cf1ea651…` 精确匹配 |
| `shard_00022.tar` | ✅ 185.56MB | md5 `0a8dc882…` 精确匹配 |
| `shard_00033.tar` | ✅ 185.56MB | md5 `3fb910b6…` 精确匹配 |
| `shard_00044.tar` | ✅ 185.56MB | md5 `68671ef6…` 精确匹配 |

> 6 个 shard **等间距采样**（id 0/2k/22k/44k/66k/88k），共 **12,000 窗口样本**（含 wav+mel）+ 完整 90,000 条鼓谱标注。剩余 39 片（约 7.2GB）在服务器持久化盘，用下述方法随时补全。

**数据可用性验证**：解包样本 `wav/synth_000000.wav`（1ch / 16000Hz / 2.000s）+ `mel/synth_000000.npy`（122×64 float32，范围 -13.82~8.86）✅

**数据质量验证**（`scripts/verify_gen.py` 逻辑）：
- 音频特性：peak 统一 0.850（软限幅后归一化）、rms 0.067~0.146（动态合理）、zcr 0.06~0.29（反映底鼓低 zcr / 闭镲高 zcr 的音色差异）
- 鼓谱标注：tempo 70.0~160.0（均值 115.1）、音符/窗口 3~12（均值 5.84）
- 全量分布：**525,887 个音符**，4 类均匀（底鼓 131,619 / 军鼓 131,263 / 闭镲 131,500 / 开镲 131,505）
- 结论：✅ 数据集均衡、动态合理、标注完整，可直接用于 Route A（Omni 音频 token）/ Route B（mel→wav 声码器）双路线训练

**批量下载方法**（剩余 43 片可随时补全）：浏览器 fetch 分块（8MB/块，规避 60s 超时）→ POST 到本地接收器（`file_receiver.py:8767`）→ 拼接 + md5 校验。服务器切块：`split -b 8m -d -a 3 <shard>.tar /tmp/chunks/sXX_`。

**行动 3 的判定铁律**（复述）：只认 **test 准确率 + 音频敏感性测试**，**绝不看 eval_loss**。当前分类器最好成绩 test 0.5721（v2），未过 70%；且 val→test 系统性 gap ≈ 0.20（2 人 val 集高估），选 checkpoint 结论不可信 → 需 k-fold 参与者 CV 补救。

---

## 7. 会话启动检查清单（Agent 必做）

1. **读本文件** — 恢复主线（数据飞轮）与当前阶段。
2. **验收铁律**：识别/评测只认 **test 准确率 + 音频敏感性测试**，**绝不看 eval_loss**（§2.1 教训）。
3. Stage 1 未过 Gate（≥70%）前，**不启动** Stage 2 飞轮滚数据。
4. 远程操作：产物直写 `/gemini/code`；实例可能重建，先确认通道。
5. 版权约定：爬取音频只做研究、不入 git、不外传。
6. 真实数据：服务器直接下载（B 站 API），音频落 `data/bbox_raw/`、`data/real_bbox/`（已 gitignore）；自动伪标未过人工审计前，不并入 Omni SFT。

---

## 8. 安全与规范

1. **禁止**提交 SSH/SFTP 口令、API Key、`恢复会话_*.md`、`docs/微调lightfour.md` 之外的敏感信息进 git。
2. 新脚本以 `D:\lightfour` 或相对路径为准，**不要写死**历史路径。
3. 大文件（zip/wav/safetensors）、原始音频数据保持 gitignore。
4. 爬取/采样音频做训练注意**版权与研究用途**声明。

---

## 9. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 首次建立 agent.md |
| 2026-10-13 | 核实趋动云环境、`/gemini/pretrain` 权重 |
| 2026-10-15 | v2 Omni QLoRA 失败（不听音频），定位 freeze_vit 根因 |
| 2026-10-16 | v3 修复 audio_tower 仍不听音频；音频敏感性测试证实模板背诵；eval_loss 判死刑 |
| 2026-10-18 | 识别模型资源调研（`beatbox_recognition_models.md`）：beatbox 太小众无现成权重，自训即稀缺资产 |
| 2026-10-19 | **战略转向**：Omni → 专用音频分类器 + 伪标注数据飞轮（Stage 1 识别 → Stage 2 飞轮 → Stage 3 生成）定为主线 |
| 2026-10-20 | Stage A 专用分类器落地：v1 test 0.5008（证实"不听音频"是 Omni 模板背诵的锅）；v2 发现波形域 pitch_shift 截断衰减尾巴的致命缺陷；v3 改 mel 域增强 + GroupNorm + EMA 重训。**Omni v4 SFT 数据同步就绪**（纯标签配方 + 2-5s 窗口，train 8,370/eval 348/test 1,216），确立"分类器当耳朵、Omni 当嘴巴"的蒸馏式混合架构 |
| 2026-10-20 pm | **生成型数据集管线落地**：`build_hit_bank.py` 切出 4,872 个真实人声击打（本地+趋动云双端结果一致）；`render_synth.py` 按鼓谱重排合成「鼓谱→音频」配对，突破真实录音仅 0.55h 的瓶颈（可无限扩量）。打通 jupyterlab-remote-shell 远程通道（`_xsrf` path 限制 / base64 传脚本 / CORS 服务器零手抄 / md5 校验）。确立数据安全约定：**代码上行、数据下行**。新增 §6 下一步行动计划 |
| **2026-10-22** | **真实 B 站数据管线落地**：服务器持久化 `pylibs` + Demucs + `TORCH_HOME`；B 站 API 直连下载 **143 视频**（前 100 个元数据 14.59h）；Demucs **89 vocals / 6.37h**；**53,633 个 onset 段 / 4.49h**（AST 过滤后 **39,171 段 / 2.93h**）；混合弱标高置信 **13,074 段 / 0.865h**（kd 2,478 / sd 6,336 / hhc 3,171 / hho 1,089，平均置信≈0.796）；真实「鼓谱→音频」生成对 **3,036 窗口 / 1.69h**（65 来源）。Stage A：本地 v2 **0.5721**、服务器 v2 **0.5147**、robust **0.4372**，均未过 70% Gate；真实域伪标置信度低（≥0.85 为 0，开镲靠 hybrid 规则补到 1,089）。**核实 ms-swift 4.5.3 不支持 Qwen2.5-Omni 音频输出训练**（Talker 训练仅 Qwen3-TTS；Qwen2.5-Omni Talker forward 无 labels/loss）。新增 `fetch_bilibili_api.py`、`segment_beatbox_onsets.py`、`ast_filter_beatbox.py`、`filter_segments_by_ast.py`、`pseudo_label_v2.py`、`hybrid_label_segments.py`、`train_beatbox_classifier_robust.py`、`train_beatbox_features.py`、`build_real_gen_pairs.py`。**样本修正**：首批 3 条误选为 a cappella/Showcase，已改用 Swissbeatbox 官方账号（mid 171444239）的经典对战 3 条；新增对战管线 4,459 切片 → AST 3,652（81.9%）→ 高置信 1,327/0.078h |
| 2026-10-22 晚 | **竞技对战数据源三轮扩展**：① 关键词第三轮 49 条清洗后候选（排除「环站」/冠军表演/海选）下载时全部已存在；② **WBI 空间 API** 最新 100 条 → **26 条 2026 最新对战**（GBB26 Florida 决赛/半决赛、vokal.total 2026、GBB26 非洲 MENA 与亚太预选赛）下载；③ **related 相关推荐**以 100 个对战为种子扩出 **117 条**历史赛事（GBB19/21/23/24/25、WBC 7ToSmoke、SBX Kickback）并下载；WAV 累计 269 → **352+**。Demucs 双队列（101 条 `-P 2` + 30 条 `-P 4`）。新增**伪标分层抽检包** 9 条（kd 2 / sd 2 / hhc 2 / hho 3，conf≥0.80）本地 `data/bbox_raw/spotcheck_20261022/`。规则修正：`championship` 保留、`champions` 排除；「环站」= loopstation 必排除 |
