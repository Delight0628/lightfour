# agent.md — lightfour 项目主线（Qwen3-Omni 一体化）

> **每次对话开始先读本文件。** 本项目唯一主线 = **All in Qwen3-Omni**：把 beatbox 当成一门语言，用单一模型完成「识别音色 + 输出 beatbox」。
> Route B 已拆分到 [docs/route_b.md](file:///e:/lightfour/docs/route_b.md)，仅作对照/降级方案，不在本文件展开。
> 最后更新：2026-10-23

---

## 1. 项目定位

**一句话**：做一个 **beatbox 语言模型**——听懂 beatbox 的音色与鼓谱，并按鼓谱生成 beatbox 音频。识别和生成在**同一个 Qwen3-Omni 模型**内完成。

最终能力：

1. 音频输入 → 输出音色序列（103 音色词表）与鼓谱。
2. 文本/鼓谱输入 → 输出 beatbox 音频。
3. 对话式交互：「来一段 GBB 风格的 beat + 开镲收尾」→ 生成音频。

---

## 2. 核心主张：beatbox 即语言

| 语言模型概念 | beatbox 对应 |
|---|---|
| 词表 / vocabulary | **103 个音色**（`P`、`PF`、`K`、`T`、`W`、`Drh`…） |
| token | 一次击打的音色 + 时长/力度 |
| 句子 / sequence | 鼓谱（音色序列 + 节奏） |
| 发音 / phoneme | 通用音频 codec 的一帧 token（EnCodec/DAC） |
| 文本输出 | 音色名/鼓谱（Thinker） |
| 语音输出 | 通用 codec token → 解码器 → beatbox 音频（Talker） |

**这意味着**：
- 不再区分「识别模型」「生成模型」——它们是同一个 Omni 的两种解码模式（文本 vs 音频 codec）。
- **4 类标签（kd/sd/hhc/hho）正式弃用**，仅作为历史数据字段保留，不进入新训练。
- 训练与对照的唯一映射体系 = **完整 103 音色 taxonomy**。

---

## 3. Omni 一体化架构（唯一架构）

```
beatbox 音频 ──► AuT 音频塔 ──► Thinker（Qwen3-Omni MoE）
                                   │
                                   ├─ 文本头：音色序列 / 鼓谱（103 音色 token）
                                   │
                                   └─► Talker（多码本自回归）
                                          │
                                          ├─ 通用音频 codec token（EnCodec/DAC）
                                          │
                                          └─► codec 解码器 ──► beatbox 音频
```

**两个方向共用同一套 token 空间**：
- 输入侧：音频 → 音色 token（103 词表）
- 输出侧：音色/鼓谱 → 通用 codec token → 音频

**关键工程点**：
- Qwen3-Omni Talker 的 `codec_head` 需要与选定通用 codec 的码本对齐（vocab / codebook size / 帧率）。
- 训练目标是多任务：文本头（音色/鼓谱 CE）+ Talker（codec token CE）。
- 识别与生成共享 Thinker 的音频理解能力；Talker 只负责「把鼓谱读出来」。

**已弃用的旧表述**（不要再写回）：
- ❌ 「识别层用专用分类器、生成层用 Route B、Omni 只做语言层」
- ❌ 「Omni 不做 beatbox 音频 I/O」

---

## 4. 音色词表（唯一映射体系：103 音色）

### 4.1 4 类标签已弃用

`kd/sd/hhc/hho` 是 AVP-LVT 鼓组标签，与真实 beatbox 音色严重不匹配（L2 重标注显示 sd 只有 50% 是真军鼓、hhc 65% 被判军鼓）。**新训练、新标注、新 manifest 一律使用 103 音色体系**；旧字段只用于追溯。

### 4.2 103 音色（Beatbox Science）

- 完整表：`docs/beatbox_sound_taxonomy.md`
- 程序读取：`data/beatbox_taxonomy.json`
- 种子音频：`data/beatbox_seeds/`（95 条 mp3，0.8MB，研究用、gitignore；元数据 `sounds.json` 已入库）
- 8 条无参考音频：Cough / High Cough / Slizzer Roll / Zekka Click Roll / Inward Throat Bass / Vortex Whistle / Smile Bass / Inward Chest Bass

**分类维度**（用于 token / 条件编码）：
- family：kicks / hihats / snares / cymbal / clicks / clops / vocal / lip_oscillation / liprolls / tongue / nasal / whistles / teeth / drag / zippers / scratch / reverse / synth / other
- mechanism：laryngeal / pulmonary / humming / hollow
- airflow：outward / inward

### 4.3 manifest 规范（新）

```json
{
  "id": "BV..._0123",
  "source": "BV...",
  "wav": "...",
  "sound_notation": "PF",
  "sound_en": "PF Snare",
  "sound_cn": "PF 军鼓",
  "sound_family": "snares",
  "sound_confidence": 0.0,
  "start": 12.34,
  "dur": 0.18,
  "label_source": "human|model|seed_match"
}
```

- 每条切片必须映射到 103 音色之一；无法确定时用 `unknown`，不强行归到 4 类。
- `sound_confidence` 在人工标注前为 0；模型伪标需给出置信度。

---

## 5. 数据资产

### 5.1 B 站真实数据管线（已跑通）

| 步骤 | 脚本 | 结果 |
|---|---|---|
| 下载 | `fetch_bilibili_api.py` | **468 个 WAV** |
| 人声分离 | Demucs `htdemucs`（`-j 4`） | **380 个 vocals**（88 个 >1200s 跳过） |
| onset 切片 | `segment_beatbox_onsets.py` | 第 3 批 90 视频 → **63,114 段** |
| AST 过滤 | `ast_filter_beatbox.py` + `filter_segments_by_ast.py` | **39,796 段（63.1%）** |
| 旧伪标 | `pseudo_label_v2.py`（4 类） | 仅历史参考，**不再作为新训练标签** |
| 高置信 | `hybrid_label_segments.py` | 28,489 段 / 1.901h（旧 4 类口径） |

**数据源策略**：
- 只抓 Swissbeatbox 官方账号（mid `171444239`）的 vs / Battle / Elimination / 决赛/半决赛/淘汰赛。
- 排除 loopstation（中文「环站」）、Showcase / Shoutout / 合辑 / Crowd POV / 海选 / 冠军表演。
- `championship` 保留，`champions` 排除。
- 版权：只做研究，音频不入 git、不外传。

### 5.2 音色种子与重标注

- 95 条种子音频 + 9 维特征（时长/低中高频/质心/ZCR/衰减/RMS/峰值）。
- 已做最近邻 L2 家族映射（`hybrid_labels_all_highconf_l2.jsonl`），作为**103 音色重标注的起点**，不是最终标签。
- 下一步：按 103 音色人工标注 200–500 条 seed（每家族 20–50 条），再用模型扩标全部数据。

### 5.3 旧 4 类质量结果（仅历史参考）

- CNN vs RF 一致率 70.89%；kd 91.75% / hhc 84.47% / sd 65.27% / hho 29.98%。
- 该结果证明「4 类不够」，是弃用 4 类、转向 103 音色的直接依据。
- 旧抽检包（33 条）保留在 `data/bbox_raw/spotcheck_*`，不再作为新标注标准。

---

## 6. Qwen3-Omni 技术现状

| 项 | 状态 |
|---|---|
| 模型 | Qwen3-Omni 30B-A3B（Thinker-Talker MoE） |
| 输入 | 音频（AuT）、图像、视频、文本 |
| 输出 | 文本 + 语音 codec token（Talker 多码本 + MTP 残差码本） |
| Talker 训练接口 | ✅ transformers 5.16.1 的 `Qwen3OmniMoeTalkerForConditionalGeneration.forward` **有 labels 和 CE loss** |
| ms-swift 支持 | ❌ `Qwen3OmniTemplate` 只做音频输入→文本；只有 `Qwen3TTSTemplate` 支持音频输出 SFT（纯语音 TTS） |
| 显存 | BF16 60GB+；4bit 推理 ~21–25GB；QLoRA ~24GB；推荐 48GB |

**结论**：Omni 一体化在模型接口上可行，缺的是**自写 Talker 训练管线 + 通用 codec 对齐**，不是概念问题。

**config 实测（2026-10-23，只下 config.json）**：
- `talker_config.num_code_groups = 16`（16 码本组）
- `position_id_per_seconds = 13`（**13Hz，约 77ms/帧**）
- `talker.text_config.vocab_size = 3072`；`code_predictor_config.vocab_size = 2048`
- `codec_bos_id = 2149`、`codec_eos_token_id = 2150`、`tts_bos/eos/pad` 齐全
- `thinker_config.audio_config`：128 mel、d_model 1280、32 层、output 2048
- transformers 5.16.1 **确有** `Qwen3OmniMoeTalkerForConditionalGeneration` 与 `Qwen3OmniMoeForConditionalGeneration`

**Talker smoke（tiny config，已通过）**：
- forward 签名关键参数：`inputs_embeds`、`attention_mask`、`labels`、`residual_codes`、`trailing_text_hidden`、`tts_pad_embed`、`generation_step`、`talker_input_ids`
- 第一层：`logits (2, 20, 3072)`、loss 8.03、梯度正常
- **16 层需要 `residual_codes`（MTP 残差码本），不是把 labels 堆成 16 层**；16 层 loss 待补

---

## 7. 通用 codec 选型（一体化关键）

### 7.1 已测结果（20 条真实切片）

| 指标 | Qwen3-TTS 12Hz（语音） | EnCodec 24kHz / 24kbps | EnCodec 48kHz / 24kbps |
|---|---:|---:|---:|
| 时长保持 | 差（+36ms，80ms 量化） | **完美（0 偏差）** | 完美 |
| mel 距离（中位） | 0.889 | **0.752** | 0.830 |
| Kicks | 0.87–1.04 | **0.65–0.80** | 0.73–0.74 |
| Clicks / Clops / Drag | 0.86–0.99 | **0.59–0.84** | 0.53–0.87 |
| Liprolls / Lip oscillations | 0.68–0.99 | **0.55–0.76** | 0.56–0.87 |
| Cymbal | 0.31–0.80 | 1.46–1.54 | 1.41–1.71 |
| Hi-hats | 0.54–0.93 | 0.87–1.18 | 1.09–1.33 |

**结论**：
- 12Hz 语音 codec 时间分辨率（80ms）低于 beatbox 击打（60–200ms）→ **不可用**。
- **EnCodec 24kHz / 24kbps 是当前最佳候选**：短击打明显最好、时长零偏差、中位 mel 0.752。
- EnCodec 48kHz 没有改善镲类（Cymbal 1.41–1.71、Hi-hats 1.09–1.33），反而略差；镲类高频损失是所有通用 codec 的共同短板。
- DAC 24kHz 因 `audiotools` 依赖在当前镜像不可用，暂未测；后续可用独立 venv 或官方 wheel 补测。
- **16 层维度对齐**：EnCodec 24kbps 输出 **32 层**，取前 16 层（≈12kbps）重建 mel 均值 0.902 / 中位 **0.786**，仅比 32 层的 0.752 差 4.5%——**码本层数对齐成立**。
- **但帧率尚未对齐**：EnCodec 是 75Hz，Omni 原生是 13Hz。是否改位置编码/降采样，见 §7.1b 决策。

### 7.1b codec_head 决策（租卡前必须拍板）

**实测澄清（2026-10-23，证据链闭合）**：`Qwen3-TTS-Tokenizer-12Hz` 的 config 是 **12.5Hz**（24000/1920）、encoder 32 层（`encoder_valid_num_quantizers=16`）、`codebook_size=2048`、decoder 16 层；与 Omni Talker 的 **13Hz / 16 组 / 2048 码本**完全对应。**它就是 Omni 音频输出的原生 codec 对**（encoder + decoder），不是"另一个模型"。因此 §7.1 的 20 条 beatbox 往返（mel 0.838、时长 +36ms、短击打失真）**就是原生 codec 的实测结果 → A' 基本不可行**。

**真正的矛盾不是帧率，是 codec 空间**：原生 codec 是语音数据训练的 2048 码本（12.5Hz），EnCodec 是通用音频训练的 1024 码本（75Hz）。即使把帧率/层数对齐，token 语义也不一样；换 codec = `codec_head` 从零重训，预训练先验丢失。

| 方案 | 做法 | 代价 | 数据需求 |
|---|---|---|---|
| **A' 原生 + LoRA** | 不改 codec；只用 LoRA 训 Talker | **已实测否**：12.5Hz/2048 往返 mel 0.838、+36ms，短击打失真 | — |
| **B 换 EnCodec 75Hz** | `codec_head` 1024 + `code_predictor` 1024 + 位置编码 13→75Hz + special token 重映射 | 丢预训练先验；序列 5.8×；需重训 codec head | 50h+，408 条不够 |
| **C 降采样** | 把 EnCodec 75Hz 压到 25/13Hz | **机制不成立**：离散 token 不能抽帧/平均；75→13 非整数比；瞬态帧丢失；抽帧失败不能归因于帧率 | 需学习式 adapter，成本高 |
| **E 混合** | Omni 只做“听懂 + 输出 103 音色文本”，Route B/EnCodec 出声 | 不是单模型，违背 All in Omni | 50h 合成 + 1.9h 真实已有 |

**决策顺序**：
1. ~~确认 Omni 原生 codec 是否带 encoder~~ ✅ 已完成：原生 codec 就是 Qwen3-TTS-Tokenizer-12Hz（12.5Hz / 16 有效层 / 2048 码本）。
2. 检查 `residual_codes`、`codec_head`、`code_predictor`、`embed_tokens` 的实际 shape 和 3072 词表构成（需要权重 index，或从 config 推断）。
3. ~~若原生往返可接受 → A'~~ ❌ 已实测否。
4. **在 B 和 E 之间拍板**：坚持单模型 + 有 50h+ → B（48GB、改 5 处、从零训 codec head）；要低风险 + 6.2GB 先出成果 → E。
5. C 只在愿意训练 25Hz adapter 时才做；**不要把抽帧重建当作 25Hz 可行性的证据**。

**未拍板前不写全量训练循环。**

### 7.2 选型后的 token 设计

- 帧率：优先 75Hz（13ms）或更高；每帧码本层数按带宽确定。
- token 空间：`[codec_layer_0, ..., codec_layer_N]`，Talker 的 `codec_head` vocab 与码本大小对齐。
- 音色 token（103 词表）走**文本头**，音频 token 走 **Talker 头**；两者在同一模型内联合训练。

---

## 8. 落地路线图（All in Omni）

| 阶段 | 动作 | 硬件 | Gate |
|---|---|---|---|
| **A. codec 选型** | EnCodec vs DAC vs Qwen3-TTS，20 条重建对比 | 6.2GB 可跑 | 时长偏差 <20ms、mel 中位 <0.8、镲类不明显劣化 |
| **B. token 工具** | `beatbox → codec token`、`token → wav`、103 音色词表编码 | CPU/6.2GB | 20 条可逆重建 |
| **C. 103 音色 seed** | 人工标注 200–500 条（每家族 20–50） | CPU | 家族覆盖 ≥15 类 |
| **D. Talker 改造** | `codec_head` 与 codec 码本对齐；写数据 collator 与 loss | 24GB+ | 能加载、前向、loss 下降 |
| **E. 小数据过拟合** | 1–2h 真实 beatbox 对，训「音色/鼓谱 → codec token」 | 24GB（QLoRA）/48GB | 训练集重建可辨识 |
| **F. 联合训练** | 音频输入 → 音色文本 + 音频输出 多任务 | 24–48GB | 同一模型既能标音色又能生成 |
| **G. 扩数据** | 28,489 高置信 + 380 vocals + 95 种子 | 24–48GB | 泛化到未见视频 |

**顺序原则**：A/B 在 6.2GB 完成 → C 人工 seed → 买卡 → D/E/F。

---

## 9. 资源与买卡

| 资源 | 现状 | 用途 |
|---|---|---|
| 本地 | Windows / AMD RX 7800 XT（ROCm 不可用）/ RAM 15.4GB | 代码、试听、小模型 |
| 云端开发实例 | 趋动云 **96 核 / 503GB RAM / B1.gpu.small 6.2GB** | 数据管线、codec 测试、小模型 |
| 持久化盘 | `/gemini/code`（seaweedfs，小文件写慢） | 产物直写持久化盘 |
| 服务器模型 | `models/Qwen3-TTS-Tokenizer-12Hz`、`models/encodec_24khz`、`checkpoints/beatbox_*` | codec/识别历史资产 |

**买卡建议**：
- **24GB（4090/A10/3090）**：4bit 推理 + QLoRA 最低门槛，Talker 训练紧，易 OOM。
- **48GB（A6000/A100 40G）**：推荐，多模块叠加更稳。
- 现在先不买：A/B/C 三步在 6.2GB 上完成后再买，避免 codec 选错。

---

## 10. 下一步行动

| # | 行动 | 状态 |
|---|---|---|
| 1 | codec 选型（12Hz / EnCodec 24k / 48k / 16 层） | ✅ 已测；**帧率决策待定（§7.1b）** |
| 2 | `beatbox→token / token→wav` 工具 | ✅ `codec_tokens.py`，往返 mel 中位 0.661 |
| 3 | 103 音色 seed 标注候选 | ✅ 408 条 / 19 家族 + 86 条试听包；`human_notation` 全空（待人工） |
| 4 | Talker 数据审计 | ✅ 408 条 `(1,1,16,T)`、raw_q=32、0 失败、split 319/43/46；manifest 已生成 |
| 5 | Talker 前向 smoke | ✅ 第一层 logits 3072 / loss 8.03 / 梯度正常；**16 层 `residual_codes` loss 待补** |
| 6 | codec_head 决策（原生 13Hz 往返 + head shape 审计） | ⬜ 下一步；先测 A'，C 降为备选 |
| 7 | 补 L2 映射脚本 / 产物 checksum 入库 | ⬜ 序列数据的前置 |
| 8 | 序列数据（vocals 滑窗 3–8s → 103 音色序列） | ⬜ 依赖 7 |
| 9 | 人工确认 200–500 条 CSV（≥15 family） | ⬜ 立即启动；86 条试听包优先 |
| 10 | 4bit + LoRA + gradient checkpointing 训练循环 | ⏸ codec_head 拍板后再写 |
| 11 | 租 48GB，`--bench-step 1` 看显存 | ⏸ 6–10 完成后再租 |

### 10.1 租卡前置阻塞项（P0，2026-10-23 评估）

代码核对结论：当前不具备租 24/48GB 卡直接开训的条件，必须先清掉：

1. **Talker 前向闭环未实现**：`train_omni_talker.py` 真实训练分支仍是骨架，Thinker hidden → Talker 的 `inputs_embeds/trailing_text_hidden` 拼接是 TODO；只监督第 1 层 codec，15 个残差层未接 loss。
2. **显存路径与预算不符**：30B BF16 权重约 60GB，单卡 48GB 加载即 OOM；代码未注入 LoRA、未 4bit 量化、未 gradient checkpointing，AdamW 全参 Talker 的优化器状态会再爆一次。
3. **数据形状未固化（原“切片写反”判断已撤回）**：dry-run 输出 `(4,16,15)` 与 transformers 5.16.1 的四维 `(B,C,Q,T)` 一致，`codes[:, :, :16]` 切的是 quantizer 维，未写反。但仍需在生成时加断言（`ndim==4`、`shape[2]==16`、`T≈dur×75`），并把 `raw_quantizers / codes_shape / T / dur` 写进 manifest，防止换 transformers 或换 bandwidth 后再次歧义。
4. **任务形态不匹配**：当前数据是「单次击打 + 单记法」，Talker 目标是「鼓谱序列 → 数秒音频」；缺序列样本、缺 train/val/test、缺按 BV 分组。
5. **标签未确认**：CSV 的 `human_notation` 全空，`candidate_notation` 来自 seed 最近邻；103 音色只覆盖 34 个记法。
6. **权重与依赖缺失**：`/gemini/pretrain` 当前是 Qwen2.5-Omni-3B（12GB），不是 Qwen3-Omni 30B；Dockerfile/env_check 仍是 Qwen2.5 口径。
7. **数据不在持久盘**：候选 wav 指向 `/tmp/batch3_segments`，实例重建即失效；`*.npz/*.npy` 不入 git 且无 manifest 入库。
8. **无生成侧评测**：`eval_omni_v4.py` 是旧 4 类文本分类评测，不能验收 Talker 音频。

租卡门槛：以上 8 条全部关闭，且 `--bench-step` 在目标卡上记录 peak < 40GB，再租卡长跑。

**Route B** 已拆分到 [docs/route_b.md](file:///e:/lightfour/docs/route_b.md)，仅作对照/降级；其数据、taxonomy、codec 选型与本文共享。

---

## 11. 会话启动检查清单

1. 读本文件，确认阶段（codec 选型 / token 工具 / seed / Talker / 训练）。
2. **All in Omni**：不再新增「专用分类器」「三层架构」表述。
3. **103 音色是唯一映射体系**；4 类只作历史字段。
4. 验收铁律：识别看 test 准确率 + 音频敏感性；生成看人耳可辨识 + codec 重建指标；**不看 eval_loss**。
5. 数据安全：代码上行、数据下行；音频不入 git、不外传。
6. 服务器可能重建：先确认 SSH / GPU / 缓存。

---

## 12. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 建立 agent.md |
| 2026-10-15~16 | Omni v2/v3 QLoRA 失败：不听音频、模板背诵；确立「eval_loss 不能验收」 |
| 2026-10-19 | 转向专用分类器 + 伪标注飞轮（后于 10-23 废弃） |
| 2026-10-20 | Stage A v1/v2/v3 + w2v；生成数据管线（hit_bank 4,872 / 合成 50h） |
| 2026-10-22 | B 站管线：468 视频 / 380 vocals / 28,489 高置信 / 1.901h；103 音色 taxonomy + 95 种子；L2 重标注 |
| 2026-10-23 | **All in Qwen3-Omni 决策**：移除三层架构、弃用 4 类、103 音色为唯一映射；Route B 拆分到 `docs/route_b.md`；Qwen3-TTS 12Hz 判死、EnCodec 24kbps 短击打可用、DAC 待测 |
| 2026-10-23 | 租卡前置评估：**不满足**。Talker 闭环未打通、显存路径不符、数据仅单击打且标签未确认、30B 权重缺失、无划分/评测；阻塞项见 §10.1 |
| 2026-10-23 | 修正 §10.1 P0-3：dry-run `(4,16,15)` 证实 Encodec 输出为四维 `(B,C,Q,T)`，`codes[:, :, :16]` 未写反；改为“形状未固化 + 需断言/manifest 审计” |
