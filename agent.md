# agent.md — 项目脉络与会话约定

> **每次对话开始时必须先读本文件**，确保对项目目标、数据、环境和下一步动作有统一认知。  
> 有重大决策、路径变更、训练进度变化时，**同步更新本文件**。

最后更新：2026-09-25

---

## 1. 项目是干什么的

**目标**：做一个**可实时语音交互的电子音乐识别 / 对话模型**。

用户期望它能：

1. **听歌识曲 / 音乐理解**：识别电子音乐，并区分**细分子类**（House、Techno、Dubstep、DnB、Trance 等，对齐 Beatport 类目）。
2. **语音对话**：能听懂语音输入，并用语音回复（端到端或 S2S）。
3. **体量可控**：小模型优先，本地 / 租卡可训可推。
4. （可选延伸）**电子音乐生成 / 风格化**（ACE-Step / MusicGen 等，非主路径）。

一句话：**语音交互 + 电子音乐细分类** 的个人训练项目，不是完整商业 App，优先 MVP。

---

## 2. 当前技术路线（已收敛）

| 模块 | 选型 | 状态 |
|---|---|---|
| 语音理解 + 对话底座 | **Qwen2.5-Omni-3B / 7B**（优先 3B 本地包，7B Int4 备选） | 权重已有本地 zip，待上传/解压部署 |
| 微调方式 | **QLoRA / LoRA SFT**（Thinker 侧，教它输出电子音乐子类标签） | 未开始 |
| 训练框架 | **ms-swift**（modelscope）或 transformers + peft | 依赖未装齐 |
| 音乐分类辅助 | CLAP / MERT（可作 Stage-1 特征或对照 baseline） | 仅调研，未落地 |
| 两阶段分类 | Stage-1：FMA 粗分类（Electronic）→ Stage-2：Beatport 32 类细类 | **数据结论已出**（见 §4） |
| 音乐生成（可选） | ACE-Step 3.5B LoRA 或 MusicGen 无条件微调 | 仅调研 |

**为什么是 Qwen2.5-Omni 而不是 Qwen3-Omni**：Qwen3-Omni 目前只有 30B-A3B，单卡训练/部署都重；2.5 有 3B/7B，显存与 QLoRA 可行性更匹配。  
**能力边界（技术报告结论）**：语音对话、音乐理解分类（需 SFT）可微调满足；**真正从零生成电子音乐 / Beatbox 不是微调能解决的**，需独立音乐生成模型。

---

## 3. 硬件与运行环境

| 环境 | 说明 |
|---|---|
| 本地 GPU | **AMD RX 7800 XT 16GB**（RDNA3）。Windows 下 ROCm 基本不可用，CUDA 栈（vLLM/flash-attn/GPTQ 训练）受限；本地更适合 **llama.cpp/Vulkan / GGUF 推理** 或轻量实验 |
| 训练 | **租 NVIDIA 卡**（A10/4090/A100）。QLoRA 微调 7B 约需 18–24GB；3B 更宽松 |
| 云端 SSH | virtaicloud（`ssh.virtaicloud.com:30022`，注意厂商文档 hostname）。详见会话恢复文档；**凭据勿写入 git / 勿外传** |
| 模型存储（云端） | **`/gemini/code/models/`**（可写、超大共享盘）。`/var/tmp/orion/comm` **只读不可用**（曾误判，已纠正） |
| 本地路径注意 | 历史脚本写死 `D:\butheisDelight\...`，当前工作区是 **`D:\lightfour`**。新脚本请用相对路径或本仓库路径 |

---

## 4. 数据与关键结论

### 4.1 FMA（Free Music Archive）

- 元数据：`data/fma/fma_metadata/`（tracks/genres/features 等，约 10.6 万曲）
- 音频子集：`data/fma/fma_small.zip`（约 7.3GB，源站不稳，可能需续传）
- Electronic 子树约 **3.4 万**曲标签；`genre_top=Electronic` 约 9.4k
- 报告：`data/fma/edm_pipeline_report.md`

### 4.2 Beatport

- `data/beatport/genres_taxonomy.txt` — 32 个 club 向 genre
- `data/beatport/latest.json` — WhatBPM 榜单元数据（**无音频**）

### 4.3 两阶段分类可行性（已写死结论）

> **部分可行，不能直接端到端落地。**

- **Stage 1（FMA 粗分类 / 是否 Electronic）**：可行。
- **Stage 2（Beatport 细类：Trap / Trance / UK Garage / Amapiano 等）**：**标签体系可行，监督数据不可行**。FMA 缺 club 细类；WhatBPM 无音频。
- **落地建议**：先做 FMA 已有 House/Techno/Dubstep/DnB 子类验证 pipeline → 再自建 Beatport 风格音频集（注意版权）→ Stage-1 预训练 + Stage-2 fine-tune。

### 4.4 模型权重

- 本地：`Qwen2.5-Omni-3B.zip`（约 19GB，在仓库根目录，**已在 .gitignore**）
- 云端：截至 2026-09-24 检查**尚未下载** Qwen 权重；需 ModelScope/HF 拉到 `/gemini/code/models/`

---

## 5. 目录结构（本仓库）

```
D:\lightfour\
├── agent.md                 # 本文件 —— 会话脉络，每次先读
├── check_now.py             # virtaicloud SSH 连通性检查（含凭据，勿提交/勿外传）
├── sftp_probe.py            # SFTP 网关握手探测
├── sftp_upload.py           # 上传本地 Qwen2.5-Omni-3B 到云端（含凭据）
├── Qwen2.5-Omni-3B.zip      # 模型权重包（大文件，gitignore）
├── 恢复会话_01a0a096_*.md    # 历史完整会话导出（含敏感信息，gitignore）
├── omni_voice_test/         # Qwen/Omni 语音输入→语音/音乐输出 API 实验
│   ├── test_omni_voice.py   # 非流式：TTS 语音请求 → Omni 音频回复
│   ├── stream_music.py      # 流式：语音请求 → 唱儿歌/旋律
│   ├── probe_voices.py      # 音色探测
│   └── *.wav / *.json       # 实验音频与响应样本
├── scripts/
│   ├── extract_fma_meta.py  # 解压 FMA 元数据并统计 Electronic 子树
│   └── analyze_edm_pipeline.py  # FMA↔Beatport 映射分析，生成 edm_pipeline_report.md
└── data/
    ├── fma/                 # FMA 元数据 + 音频 zip + 分析报告
    └── beatport/            # Beatport taxonomy + 榜单元数据
```

---

## 6. 已完成 / 进行中 / 待做

### 已完成

- [x] 开源方案调研（Moshi / Qwen-Omni / CLAP / MERT / ACE-Step / ms-swift 等）
- [x] 显存与硬件约束分析（16GB AMD vs 训练租卡）
- [x] Qwen2.5/3-Omni 技术报告能力边界（对话✓，分类需 SFT，音乐生成✗）
- [x] FMA 元数据拉取 + Electronic 统计 + Beatport 映射报告
- [x] Omni 语音输入→语音/简单旋律输出 API 通路实验（`omni_voice_test/`）
- [x] 云端 SSH/SFTP 环境摸底（可写路径、无权重、依赖缺口）

### 进行中

- [ ] 将 `Qwen2.5-Omni-3B` 上传云端并解压到 `/gemini/code/models/`
- [ ] 云端安装训练依赖（torch/transformers/peft/librosa/qwen-omni-utils 等）

### 待做（建议顺序）

1. **数据集**：FMA Electronic 子类 → 训练样本格式（音频 + 子类标签 + 对话指令）
2. **微调**：ms-swift / QLoRA 对 Qwen2.5-Omni 做「听音频 → 输出细类 + 自然语言解释」SFT
3. **评测**：细类准确率 + 语音对话连贯性
4. （可选）Stage-2 club 细类自建数据；音乐生成模块

---

## 7. 安全与规范

1. **禁止**把 SSH/SFTP 账号密码、API Key、`恢复会话_*.md` 提交进 git（`.gitignore` 已覆盖；根目录脚本仍含明文凭据，**勿复制到公开处**）。
2. 新脚本**不要写死** `D:\butheisDelight\`，以 `D:\lightfour` 或相对路径为准。
3. 大文件（zip/wav/safetensors）保持 gitignore。
4. 涉及爬取/采样音频做训练时注意**版权与研究用途**声明。

---

## 8. 会话启动检查清单（Agent 必做）

每轮新对话开始时：

1. **读 `agent.md`（本文件）** — 恢复项目目标与进度。
2. 若用户提到训练/数据/服务器新进展，**先更新 §6 进度与相关章节**，再动手。
3. 不确定细节时查：`data/fma/edm_pipeline_report.md`、`恢复会话_01a0a096_语音电子音乐模型.md`（仅本地）。
4. 涉及远程操作时，确认凭据来源与路径是否仍有效（云主机可能重建）。

---

## 9. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 首次建立 `agent.md`，汇总目标、路线、数据结论、目录与待办 |
| 2026-09-28 | GitHub 初始化：`https://github.com/Delight0628/lightfour`；历史中剔除含凭据的 `check_now.py` / `sftp_*.py`；补 README/LICENSE |
