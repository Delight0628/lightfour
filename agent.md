# agent.md — lightfour 项目主线（Qwen3-Omni 一体化）

> **每次对话开始先读本文件。** 本项目唯一主线 = **All in Qwen3-Omni**：把 beatbox 当成一门语言，用单一模型完成「识别音色 + 输出 beatbox」。
> Route B 已拆分到 [docs/route_b.md](file:///e:/lightfour/docs/route_b.md)，仅作对照/降级方案，不在本文件展开。
> 最后更新：2026-10-26

---

## 0. 当前状态快照（2026-10-26）

**一句话**：数据与接口侧的活儿基本干完了，卡在**「B（换 EnCodec 改 Omni）还是 E（Omni 只出音色文本、Route B 出声）」这一拍板**。

| 项 | 状态 |
|---|---|
| 数据管线（468 视频 / 380 vocals / 28,489 高置信 / 1.9h） | ✅ 可用；`data/` 下共 83,338 个 wav（含 vocals、切片、种子） |
| 103 音色 taxonomy + 408 条候选 + 86 条试听包 | ✅ 已产出；**人工标注 CSV 全空，是长周期卡点** |
| codec 选型 | ✅ EnCodec 24k/24kbps 最佳；原生 Qwen3-TTS 12.5Hz 判死（A' 否） |
| Talker 权重结构 | ✅ 已从源码读实（§7.1b）；三条旧估算被推翻 |
| 16 层训练闭环 | ✅ 可训练性已验证（§7.3b）；**框架不算 15 层 loss，必须自写** |
| 小模型验证 F（103 音色 → 16 层 token） | ✅ loss 6.93→3.75；低频可学、高频学不动（§7.3） |
| 试听包 | ✅ `data/bbox_raw/tiny_codec_gen/` 20 组 gen/gt 对照，**等人耳判断** |
| 全量训练循环 / 租 48GB | ⏸ **B vs E 未拍板，不动** |
| 多 Agent 协作组（PI/Data/Codec/Talker/Eval/Critic/Scout） | ✅ 已建，见 §13 |

**当前唯一阻塞**：`B vs E` 的取舍。判别实验是「把 TinyCodecLM 扩到 dim 256 / 6 层后，`CL*`、`W` 这批高频音色能不能追上」——追得上说明此前只是容量不足（B 可行），追不上说明是 codec 空间装不下高频（应转 E）。

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

**关键工程点**（2026-10-26 按源码实测修正）：
- `codec_head` **不需要改 shape**：它是 3072 类，其中 0–2047 是码字区、2048–3071 是 special 区。换 1024 码本的 EnCodec 时，码字占 0–1023、1024–2047 永久 mask、special 原样不动（详见 §7.1b）。
- 真正要改的是：`code_predictor` 的 **15 个 2048→1024 嵌入表**、`position_id_per_seconds: 13→75`，以及**自写 15 个残差层的 CE**（框架的 `forward()` 只算第 1 层）。
- 训练目标是多任务：文本头（音色/鼓谱 CE）+ Talker（第 1 层 3072 类 CE）+ predictor（15×1024 类 CE）。
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
| Talker 训练接口 | ⚠️ transformers 5.16.1 的 `forward` **只有第 1 层的 labels/CE**；15 个残差层必须自写（§7.1b / §7.3b） |
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

> **口径警告（2026-10-23 修 bug 后）**：上表由 `lf_codec_encodec24.py` 系列脚本产出，decode 时用的是 processor 的**采样点级** `padding_mask`，不会截断波形，**数字有效**。但该脚本目前**不在仓库**；在把 0.752 当作 codec 选型依据前，需用修复后的 `codec_tokens.py` 重跑同一批 20 条，对比新旧 mel，并把脚本/产物入库。
> 另外，`codec_tokens.py` 早期版本自造了一个 `(1,1,T)` 的**帧级** mask 传给 `decode`；transformers 会拿 `padding_mask.shape[-1]` 截断波形，导致 13 帧音频只剩 13 个采样点。它报出的"往返 mel 中位 0.661"因此**无效**，已修（见 §7.4），需重测后才可引用。

### 7.1b Talker 权重结构实测（transformers 5.16.1 源码，非估算）

这一节回答"换 codec 到底要改什么"。全部来自 `modeling_qwen3_omni_moe.py` 源码，不是推断。

| 组件 | 实际定义 | 含义 |
|---|---|---|
| `codec_head` | `nn.Linear(hidden, 3072, bias=False)`，`_tied_weights_keys = {"codec_head": "model.codec_embedding.weight"}` | 第 1 层 codec token 输出头，**与输入嵌入共享权重** |
| `model.codec_embedding` | `nn.Embedding(3072, hidden)` | 第 1 层 token 输入嵌入 |
| `code_predictor.codec_embedding` | `ModuleList([nn.Embedding(2048, hidden)] × 15)` | 第 2–16 层，**15 个独立嵌入表** |
| `residual_codes` | `cat((第1层 input_ids, predictor 生成的 15 层), dim=-1)` → 每帧 `(B,16)` | 16 层 token 的实际载体 |
| `position_id_per_seconds` | 仅用于 `get_rope_index` 的时间步长换算（Talker=13） | **不是序列长度上限** |

**三条推翻先前估算的结论**：

1. **3072 不是码本，是「2048 码字 + 1024 special」**。`codec_bos_id=2149`、`codec_eos=2150` 都落在 2048–3071 区间。换 1024 码本时**可以保持 3072 词表不变**：码字占 0–1023，1024–2047 永久 mask，2048–3071 继续放 special。`codec_head.weight.shape` 不用改，`codec_bos/eos` 不用重映射。
2. **`position_id_per_seconds: 13→75` 不需要重训位置编码**。它只参与 RoPE 时间步长换算；8s×75Hz=600 token 对 Qwen3 的 RoPE 是短序列。
3. **16 层 loss 在框架层面没有实现**。`forward()` 把 `residual_codes` 原样塞进 `hidden_states` 返回（源码注释 `# TODO: hack here to take residual codes out`），**不参与任何 loss**；`labels` 只喂第 1 层的 3072 类 CE。训练 15 个残差层必须**自写** `code_predictor` teacher forcing + 15 个 CE。

**B 方案真实改动清单（修正版）**：

| 处 | 改法 | 必须？ |
|---|---|---|
| `codec_head` / `codec_embedding` | **不改 shape**；重训前 1024 行语义，1024–2047 永久 mask | 必须（部分重训，非重建） |
| `code_predictor.codec_embedding` ×15 | `2048 → 1024` resize | 必须 |
| `position_id_per_seconds` | `13 → 75`（一行 config） | 必须 |
| `codec_bos/eos` 重映射 | **不需要**（仍在 3072 内） | ~~否~~ |
| 位置编码重训 | **不需要** | ~~否~~ |
| 16 层训练分支 | 自写 predictor teacher forcing + 15×CE | 必须（框架缺失） |

**数据需求不变**：408 条只够验证单音色过拟合；换 codec 后语义重对齐仍需 50h+。

### 7.1c 方案对比与拍板

**实测澄清（2026-10-23，证据链闭合）**：`Qwen3-TTS-Tokenizer-12Hz` 的 config 是 **12.5Hz**（24000/1920）、encoder 32 层（`encoder_valid_num_quantizers=16`）、`codebook_size=2048`、decoder 16 层；与 Omni Talker 的 **13Hz / 16 组 / 2048 码本**完全对应。**它就是 Omni 音频输出的原生 codec 对**（encoder + decoder）。因此 §7.1 的 20 条 beatbox 往返（mel 0.838、时长 +36ms、短击打失真）**就是原生 codec 的实测结果 → A' 不可行**。

**真正的矛盾不是帧率，是 codec 空间**：原生 codec 是语音数据训练的 2048 码本（12.5Hz），EnCodec 是通用音频训练的 1024 码本（75Hz）。即使帧率/层数对齐，token 语义也不一样。

| 方案 | 做法 | 代价 | 数据需求 |
|---|---|---|---|
| **A' 原生 + LoRA** | 不改 codec；只用 LoRA 训 Talker | **已实测否**：12.5Hz/2048 往返 mel 0.838、+36ms，短击打失真 | — |
| **B 换 EnCodec 75Hz** | 见 §7.1b 修正清单：head 不 resize、predictor 15 表 1024、位置步长 75、自写 16 层 loss | 丢部分预训练先验；序列 5.8×；需重训第 1 层前 1024 行语义 | 50h+，408 条不够 |
| **C 降采样** | 把 EnCodec 75Hz 压到 25/13Hz | **机制不成立**：离散 token 不能抽帧/平均；75→13 非整数比；瞬态帧丢失；抽帧失败不能归因于帧率 | 需学习式 adapter，成本高 |
| **E 混合** | Omni 只做「听懂 + 输出 103 音色文本」，Route B/EnCodec 出声 | 不是单模型，违背 All in Omni | 50h 合成 + 1.9h 真实已有 |
| **F 小模型先验（已选）** | 2.75M TinyCodecLM 验证「103 音色 → EnCodec 16 层 token」可学性 | 只验证单音色，不验证鼓谱序列 | 408 条 seed |

**决策顺序**：
1. ~~确认 Omni 原生 codec 是否带 encoder~~ ✅ 原生 codec = Qwen3-TTS-Tokenizer-12Hz。
2. ~~检查 head / predictor / residual_codes 结构~~ ✅ 见 §7.1b，三条估算被推翻。
3. ~~原生往返可接受 → A'~~ ❌ 已实测否。
4. **B vs E 暂缓拍板**，先用 F 拿到「75Hz/1024 码本能否承载 beatbox」的独立证据（§7.3）。
5. C 只在愿意训练 25Hz adapter 时才做。

### 7.3 小模型验证结果（F，2026-10-23）

`train_tiny_codec_lm.py`：2.75M 参数（dim 128 / 3 层 / 4 头），条件 = 记法 id，目标 = EnCodec 16 层 token。408 条、34 个记法、30 epoch。

- loss **6.931（=ln1024，均匀分布）→ 3.750**，仍在下降，未收敛。
- 该 loss 是**跨 16 层 × 1024 类的平均 CE**，3.75 相当于每层困惑度约 42，属于"能学到结构但远不够精确"。
- 生成方式：固定接受域一次性输出 `(16,T)` 并 argmax，**不是自回归**，本质是对同一记法做众数平均。

20 条 A/B（`gen` vs 训练集真值重建 `gt`）的多维评估：

| 指标 | 均值 | 中位 | 判读 |
|---|---:|---:|---|
| 时长比 | 1.13 | 1.087 | 生成略长（用了记法平均帧数） |
| onset 对齐误差 | 7.87ms | **4.0ms** | 起音位置学到了 |
| 包络相关 | 0.483 | 0.481 | 衰减形状只学到一半 |
| 衰减时长（gen） | 110.7ms | 100.0ms | 比真值 74.7ms 偏长、偏糊 |
| 高频能量占比 | 0.400 | 0.380 | 与真值 0.383 基本一致 |
| 谱通量相关 | 0.400 | 0.417 | 瞬态密度中等 |
| mel 距离 | 1.926 | 1.934 | **差**（真值重建是 0.75） |

**分层结论**（比汇总更有信息量）：

- **低频击打可学**：`<d`（mel 0.43 / env 0.998 / flux 0.995）、`<Drhb`（0.57 / 0.986 / 0.858）、`<K\``（0.80 / 0.977 / 0.906）、`<Peh`（env 0.987）。
- **高频/摩擦音学不动**：`CL*`（mel 3.11 / env 0.105 / flux 0.068）、`W`（3.80 / 0.072）、`D`（2.10 / -0.154）、`Chῤ`（env -0.04）。
- 与 §7.1 的 codec 短板完全重合：**镲类/clicks 是 EnCodec 最弱、也是小模型最难学的同一批音色**。

**必须同时说清的局限**：
1. 任务形态是「单记法 → 单次击打」，**不验证鼓谱序列生成**（后者需要滑窗 3–8s 序列数据）。
2. 一次性 argmax 生成 = 众数平均，天然抹瞬态；换成自回归 + 采样可能改善，但不改变"高频难学"的结论。
3. 2.75M 参数远小于 30B Omni，不能直接外参。

**试听包**：`data/bbox_raw/tiny_codec_gen/`（20 组 `*_gen.wav` + `*_gt.wav`，共 40 个）。

### 7.3b 16 层训练闭环 smoke（`talker_16layer_smoke.py`）

结论：**15 个残差层可以训练，但必须自己串，框架不提供。**

- `predictor` 初始 15 层平均 CE = **7.639**（≈ ln(2048) = 7.62，随机猜测水平），说明 shape 与语义对齐正确。
- 固定 batch 过拟合 40 步：**7.639 → 3.285**（降 4.35），41/42 参数获得梯度，`lm_head.0/14` 与 `codec_embedding.0/14` 都在其中。
- **第 1 层 Talker 在 tiny 配置下 loss = NaN**，hook 定位到 `model.layers.0.mlp.experts.act_fn`（fused MoE）。真实 30B 权重不会遇到，但 tiny smoke 必须绕开：`--part predictor` 用 `torch.randn * 0.02` 模拟 Talker hidden。
- 第 1 层的可反传性已由 `talker_forward_smoke.py` 单独证明（loss 8.05、32/83 参数有梯度）。

**tiny config 的两个硬约束**（踩过）：
1. `mrope_section` 的和**必须等于 `head_dim`**（真实是 [24,20,20]=64）。缩放 head_dim 时必须同比缩放，否则 RoPE 直接出 NaN。
2. tiny 配置下 fused MoE 会出 NaN；smoke 用 `num_experts=1` 或绕开 Talker。

### 7.3c 层数天花板 A/B：16 层 vs 32 层（`codec_layers_ab.py`）

对 §7.3 里最差的四类（Clicks / Liprolls / Lip oscillations / Drag）各取 3 条，同一批音频分别用 EnCodec 前 16 层与前 32 层解码：

| 指标 | 16 层（≈12kbps） | 32 层（24kbps） | 变化 |
|---|---:|---:|---|
| mel 距离（中位） | 0.768 | **0.658** | **改善 14.3%** |
| 单条改善比例 | — | 12/12 条全部更好 | — |
| 高频能量占比偏差 | 0.014 | 0.013 | 基本不动 |

**判定：16 层不是 codec 天花板。** 因此 §7.3 里小模型学不动高频，**不能全部归因于 codec 空间**，至少有一部分是编码端层数不足被提前截断了。

**但对 Omni 是个新问题**：Talker 固定 `num_code_groups=16`。要上 32 层必须把 `num_code_groups` 改成 32，`code_predictor` 的 15 个 embedding + 15 个 lm_head 扩成 31 个 —— 这比「换 1024 码本」的改动更大，且预训练先验几乎全废（15→31 个新表）。

**两个可选方向**：
- **16 组**：接受 14% 的编码损失，改动最小，但高频短板会被固化。
- **32 组**：编码质量最好，但要重建 predictor 的 31 个表，且序列再翻一倍（75Hz×32 组 token/帧）。

**注意**：这轮抽样只覆盖 Clicks/Drag/Liprolls，**没覆盖 Cymbal/Hi-hats**（按 confidence 排序时前四类先凑满）。镲类是 beatbox 最关键的短板，需补测。

### 7.4 已修 bug：decode 帧级 mask 截断波形

`EncodecModel.decode(audio_codes, audio_scales, padding_mask=None, ...)` 里有一句：

```python
if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
    audio_values = audio_values[..., : padding_mask.shape[-1]]
```

它把 `padding_mask.shape[-1]` 当作**采样点数**。`codec_tokens.py` 早期传了帧级 `(1,1,T)` mask，于是 13 帧音频被截成 13 个采样点（0.0005s）。症状：生成音频全部 0.001s；roundtrip 的 mel 因为只比较极短片段而虚低。
修复：`decode_codes` 改传 `None`（`audio_scales=[None]`）。验证：15 帧 → 4800 采样点 = 0.200s；14.6M 采样长音频往返 ratio 1.000。

### 7.5 选型后的 token 设计

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
| 1 | codec 选型（12Hz / EnCodec 24k / 48k / 16 层） | ✅ 已测；**帧率决策见 §7.1b/7.1c** |
| 2 | `beatbox→token / token→wav` 工具 | ⚠️ `codec_tokens.py` 往返 mel 0.661 **无效**（§7.4 截断 bug），修复后需重测 |
| 3 | 103 音色 seed 标注候选 | ✅ 408 条 / 19 家族 + 86 条试听包；`human_notation` 全空（待人工） |
| 4 | Talker 数据审计 | ✅ 408 条 `(1,1,16,T)`、raw_q=32、0 失败、split 319/43/46；manifest 已生成 |
| 5 | Talker 前向 smoke | ✅ 第一层 logits 3072 / loss 8.03 / 梯度正常 |
| 6 | Talker 权重结构实测 | ✅ `codec_head` tied、predictor 15×2048、`residual_codes` 不进 loss（§7.1b） |
| 7 | 小模型验证 F（103 音色 → 16 层 token） | ✅ loss 6.93→3.75；20 组 A/B 已出；低频可学、高频学不动（§7.3） |
| 8 | 自写 16 层 predictor 训练分支 | ✅ 可训练性已验证：CE 7.639→3.285 / 41-42 参数有梯度（§7.3b） |
| 9 | **重测 `codec_tokens.py` 往返指标** | ⬜ §7.4 修完 bug 后 0.661 作废，需用新口径重跑并回填 §7.1 |
| 10 | **扩容量判别实验（dim 256 / 6 层）** | ⬜ **当前最关键**：决定 B 与 E 谁上 |
| 11 | 补 L2 映射脚本 / 产物 checksum 入库 | ⬜ 序列数据的前置 |
| 12 | 序列数据（vocals 滑窗 3–8s → 103 音色序列） | ⬜ 依赖 11；**这是验证鼓谱生成的必要条件** |
| 13 | 人工确认 200–500 条 CSV（≥15 family） | ⬜ 立即启动；86 条试听包优先 |
| 14 | 4bit + LoRA + gradient checkpointing 训练循环 | ⏸ B vs E 拍板后再写 |
| 15 | 租 48GB，`--bench-step 1` 看显存 | ⏸ 10、11、13 完成后再租 |

### 10.1 租卡前置阻塞项（P0，2026-10-23 评估）

代码核对结论：当前不具备租 24/48GB 卡直接开训的条件，必须先清掉：

1. **Talker 训练脚本仍未闭环**：`train_omni_talker.py` 真实训练分支仍是骨架，Thinker hidden → Talker 的 `inputs_embeds/trailing_text_hidden` 拼接是 TODO，且仍只监督第 1 层 codec。**§7.1b 已确认：transformers 的 `forward()` 不算 15 个残差层的 loss**；`talker_16layer_smoke.py` 已自写 `code_predictor` teacher forcing + 15×CE 并验证可训练，但尚未接回训练脚本。
2. **显存路径与预算不符**：30B BF16 权重约 60GB，单卡 48GB 加载即 OOM；代码未注入 LoRA、未 4bit 量化、未 gradient checkpointing，AdamW 全参 Talker 的优化器状态会再爆一次。
3. ✅ **已关闭：数据形状审计通过**。408 条 `(1,1,16,T)`、`raw_q=32`、0 失败、manifest 已生成；4D 断言与 `--self-test` 已加。
4. **任务形态不匹配**：当前数据是「单次击打 + 单记法」，Talker 目标是「鼓谱序列 → 数秒音频」；缺序列样本。split 已按 `source` BV 的 md5 分组（319/43/46），泄漏风险已关闭。
5. **标签未确认**：CSV 的 `human_notation` 全空，`candidate_notation` 来自 seed 最近邻；103 音色只覆盖 34 个记法。
6. **权重与依赖缺失**：transformers 5.16.1 的 Qwen3-Omni 类已实测存在；但 `/gemini/pretrain` 仍是 Qwen2.5-Omni-3B（12GB），30B 权重未下载；Dockerfile/env_check 仍是 Qwen2.5 口径。
7. **数据不在持久盘**：候选 wav 指向 `/tmp/batch3_segments`，实例重建即失效；`*.npz/*.npy` 不入 git，manifest 已生成但源 wav 迁移待确认。
8. **无真实生成侧评测**：tiny A/B 评测已有（§7.3），但 `eval_omni_v4.py` 仍是旧 4 类文本分类评测；真实 Talker 的生成评测（decode → mel/瞬态/音色命中）尚缺。

租卡门槛：以上阻塞项全部关闭，且 `--bench-step` 在目标卡上记录 peak < 40GB，再租卡长跑。

**Route B** 已拆分到 [docs/route_b.md](file:///e:/lightfour/docs/route_b.md)，仅作对照/降级；其数据、taxonomy、codec 选型与本文共享。

---

## 11. 会话启动检查清单

1. **先看 §0 状态快照**，确认当前卡点（现在是 `B vs E` 未拍板、等扩容量判别实验）。
2. 读本文件，确认阶段（codec 选型 / token 工具 / seed / Talker / 训练）。
3. **All in Omni**：不再新增「专用分类器」「三层架构」表述。
4. **103 音色是唯一映射体系**；4 类只作历史字段。
5. 验收铁律：识别看 test 准确率 + 音频敏感性；生成看人耳可辨识 + codec 重建指标；**不看 eval_loss**。
6. 数据安全：代码上行、数据下行；音频不入 git、不外传。
7. 服务器可能重建：先确认 SSH / GPU / 缓存；SSH 的 `/tmp` 与 SFTP 根**不互通**，取文件走 base64。
8. 多 Agent 协作组见 §13；群聊讨论不能替代本文件的决策记录。

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
| 2026-10-23 | **§7.1b Talker 结构实测**：`codec_head` 与 `codec_embedding` tied、3072=2048 码字+1024 special、predictor 15×2048、`residual_codes` 不进 loss。推翻三条旧估算（head 不需 resize、bos/eos 不需重映射、位置编码不需重训） |
| 2026-10-23 | **§7.3 小模型验证 F**：TinyCodecLM 2.75M、loss 6.93→3.75；20 组 A/B 多维评估；低频击打可学、高频/摩擦音学不动 |
| 2026-10-23 | **§7.4 修 decode 截断 bug**：帧级 `padding_mask` 把波形截到 T 个采样点，导致生成音频 0.001s、roundtrip mel 虚低；改传 `None` 后 15 帧→0.200s |
| 2026-10-23 | 新增 `scripts/eval_gen_audio.py`：mel 之外加 onset 对齐 / 包络相关 / 衰减时长 / 高频能量 / 谱通量五维 |
| 2026-10-23 | **§7.3b 16 层闭环 smoke**：predictor CE 7.639→3.285、41/42 参数有梯度；第 1 层 tiny MoE 出 NaN 已定位并绕开；记录 `mrope_section` 必须等于 `head_dim` |
| 2026-10-26 | 建立 **多 Agent 协作组**：7 个 Ekko profile（`lightfour-pi/data/codec/talker/eval/critic/scout`）已加入群聊 room `mvp32tmla68so5`，角色与协作协议见 §13 |
| 2026-10-26 | **同步状态快照**：新增 §0（一眼看到当前卡点 B vs E）；修正 §3 关键工程点（`codec_head` 不 resize、真正要改的是 predictor 15 表 + 位置步长 + 自写 15 层 CE）；§6 Talker 训练接口标为 ⚠️（只有第 1 层）；行动表重排，新增 #9 重测往返指标、#10 扩容量判别实验；§11 加 §0 入口与 SFTP 不通提醒 |

---

## 13. 多 Agent 协作组（2026-10-26）

群聊 room：`mvp32tmla68so5`（Lightfour），工作区 `E:\lightfour`。
Profile 根目录：`C:\Users\Administrator\.hermes\profiles\`；每个 profile 含 `SOUL.md`（角色章程）、`config.yaml`（模型 xiaomi/mimo-v2.6-pro）、`.env` 与独立 skills/workspace。

| 群聊名 | profile | 职责 | 主要交接 |
|---|---|---|---|
| `@PI` | `lightfour-pi` | 首席科学家兼总控：拆任务、维护本文件、拍板、守 §10.1 门槛 | 向 Data/Codec/Talker/Eval/Critic/Scout 派工 |
| `@Data` | `lightfour-data` | 103 音色数据、B 站管线、标注、manifest、序列滑窗 | Codec / Talker / Eval |
| `@Codec` | `lightfour-codec` | codec 选型、token 化、16/32 层、75/13Hz、码本对齐 | Talker / Eval |
| `@Talker` | `lightfour-talker` | Qwen3-Omni 训练工程、16 层 loss、QLoRA、显存与过拟合 | Codec / Data / Eval |
| `@Eval` | `lightfour-eval` | 识别/生成评测、分 family 指标、听感 A/B | Codec / Talker / Critic |
| `@Critic` | `lightfour-critic` | 红队审稿：反例、混淆变量、口径与可复现性 | PI / 全体执行角色 |
| `@Scout` | `lightfour-scout` | GitHub/论文 SOTA、codec/音频 LLM、许可证与硬件门槛 | Codec / Talker / PI |

**协作规则**：
1. 只在被 @ 或任务确实需要时发言；先结论、再证据、再下一步。
2. Agent 间一次只交接一个对象，写清交付物与验收门槛；避免连续 @ 形成长链（默认 mention depth 4）。
3. 验收口径仍按 §11.4：识别看 test 准确率 + 音频敏感性，生成看人耳可辨识 + codec 重建指标，禁止用 `eval_loss`。
4. 新结论、被推翻的估算、失败实验继续写回本文件；群聊讨论不能替代 agent.md 的决策记录。
