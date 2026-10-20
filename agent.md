# agent.md — lightfour 项目主线（伪标注数据飞轮）

> **每次对话开始时必须先读本文件**，确保对项目目标、当前主线、进度和下一步动作有统一认知。
> 有重大决策、路线变更、训练进度变化时，**同步更新本文件**。

最后更新：2026-10-20（生成型数据集管线落地）

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
| 3. **验收评测**：test 准确率 + 音频敏感性 | `scripts/eval_beatbox_classifier.py` | **4 类准确率 ≥ 70%**（随机 25%） | 🟡 v1=0.5008，v3 训练中 |
| 4. 音频加长 2-5s 窗口 | `build_sft_v4_omni.py`（beatbox_windows 6,445 个） | 窗口 ≥2s 占比 100% | ✅ |

**迭代记录（本地 CPU，`py -3.12`，torch 2.14.1+cpu）**：

| 版本 | 关键改动 | 结果 | 结论 |
|---|---|---|---|
| v1 | log-mel CNN，BatchNorm，per-clip 归一化 | test **0.5008**（Kick 95%/Snare 40%/闭镲 43%/开镲 8%） | ✅ 证明"模型不听音频"是 Omni 模板背诵的锅，换专用分类器就解决；❌ 类别偏置 + 跨人泛化差（val 0.72 vs test 0.50） |
| v2 | +类权重 +波形增强(增益/噪/变调/变速) | val 在 0.53~0.63 剧烈震荡 | ❌ 发现**致命缺陷**：`pitch_shift` 重采样后按原长裁剪 → 截掉 hi-hat 衰减尾巴，而闭镲/开镲的唯一判别特征就是衰减时长 |
| v3 | **mel 域频率增强（保时间轴）** + GroupNorm + EMA + LR 1e-3 | 运行中 | 目标 test ≥ 0.70 |

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
| **云端开发实例** | 趋动云「开发环境」**96 核 CPU / 无 GPU** / Python 3.11.8 | **数据生成专用**（CPU 密集，不占本机） |
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

## 6. 下一步行动计划（2026-10-20 排定）

> **优先级**：Stage 1 识别过 Gate 是飞轮启动前提（铁律，不可跳过）；生成型数据集可并行推进（不依赖识别模型）。

| # | 行动 | 交付物 | 依赖 | 状态 |
|---|---|---|---|---|
| **1** | **完成 gen_synth 首批并下载回本地** | `data/gen_synth/` 5h 数据 + 音质抽检报告 | 服务器 render_synth 跑完 | 🔄 约 50% |
| **2** | **render_synth 改 tar 流式输出** → 扩到 50h | 50h 生成型数据集 | 行动 1 完成 | ⬜ |
| **3** | **Stage 1 识别模型冲 70% Gate** | 候选：修 v3 补零 bug 重训 / 增大 val 参与者 / 全量参与者 k-fold CNN / 上云 wav2vec2 全参微调 | — | ⬜ 最高优先 |
| **4** | 生成质量闭环评测 | 用 Stage-A 分类器判生成音频能否命中目标鼓类 | 行动 1 + Stage 1 | ⬜ |
| **5** | Omni 生成路线可行性核实 | 服务器查 ms-swift 是否支持 Talker 音频 token 训练 | 服务器通道 | ⬜ |
| **6** | Stage 2 伪标注飞轮落地 | `preprocess_video.py` + `pseudo_label.py` 用真实 beatbox 视频跑通 | **行动 3（Gate ≥70%）** | ⬜ 阻塞中 |

**行动 3 的判定铁律**（复述）：只认 **test 准确率 + 音频敏感性测试**，**绝不看 eval_loss**。当前分类器最好成绩 test 0.5721（v2），未过 70%；且 val→test 系统性 gap ≈ 0.20（2 人 val 集高估），选 checkpoint 结论不可信 → 需 k-fold 参与者 CV 补救。

---

## 7. 会话启动检查清单（Agent 必做）

1. **读本文件** — 恢复主线（数据飞轮）与当前阶段。
2. **验收铁律**：识别/评测只认 **test 准确率 + 音频敏感性测试**，**绝不看 eval_loss**（§2.1 教训）。
3. Stage 1 未过 Gate（≥70%）前，**不启动** Stage 2 飞轮滚数据。
4. 远程操作：产物直写 `/gemini/code`；实例可能重建，先确认通道。
5. 版权约定：爬取音频只做研究、不入 git、不外传。

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
