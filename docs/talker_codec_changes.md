# Talker codec 接口改动清单（@Codec → @Talker）

> 2026-10-27 @Codec ｜ 全部结论来自 transformers 5.16.1 源码行号，不是估算。
> 前置事实：本项目的音频 token 是 **EnCodec 24kHz / 24kbps 码流（raw 32 层）取前 16 层，等效 ≈12kbps / 1024 码本 / 75Hz**，
> 形状 `(1,1,16,T)`，T = ceil(n_24k / 320)。
> **层数口径（2026-10-27 冻结，禁止再改）**：层数翻倍即码率翻倍，**16 层 vs 32 层不构成控制变量**，
> 任何「32 层更好」只能表述为「码率加倍带来的增益」，不得用于论证层数本身有效。
> v0/v1 目标层数一律 **16（15 残差 + 1 首层）**，与 Talker 原生 16 个 code group 对齐；
> 不重导 `data/seq_tokens/`（287 条码值/形状/md5 全部不变）。

## 0. 先说三条会直接改错的

1. **`position_id_per_seconds` 不要改成 75。** `get_rope_index` 里只有 **image / video** 两个分支读它
   （`modeling_qwen3_omni_moe.py` L299 取值、L376 图像、L393/L416 视频）；audio 与 codec token 走普通顺序位置。
   id 改成 75 对 codec 帧率**零收益**，只会把 Thinker 的视觉 mrope 时间轴缩放 5.77×。
   75Hz 的代价 100% 体现在**序列长度**（8s = 600 token），不是 config 里一个整数。
2. **`codec_head` 不要 resize。** 它是 `Linear(1024 → 3072, bias=False)`（L3021），且
   `_tied_weights_keys = {"codec_head": "model.codec_embedding.weight"}`（L3000），
   与 `nn.Embedding(3072, 1024)`（L2891）**共享同一块权重**。改任何一边的 shape 都会连带改另一边，
   破坏 tied 关系。正确做法是 shape 不动、**按行分区使用**（见 §1）。
3. **不要用逐击编码再拼接。** `talker_sequence.py` 的 `stitch_hit_codes` 已实测 `AssertionError`
   （`DEFAULT_PAD=2050 > CODEBOOK=1024`，触发 `codec_ce_16` 的 target 越界断言）；
   即使改小 pad，语义上也不成立——EnCodec 有感受野。用整段 codes + `seq_token_dataset.py`。

## 1. 改动清单

| # | 处 | 现状（源码/配置实证） | 改成 | 必须? |
|---|---|---|---|---|
| 1 | **码字映射** | — | EnCodec 码 0..1023 **直通** Talker 词表 0..1023（不做 offset） | 是 |
| 2 | `codec_embedding`（L2891）/ `codec_head`（L3021） | `Embedding(3072,1024)` + `Linear(1024→3072)`，tied | **shape 不动**。0..1023 = EnCodec 码字（这 1024 行语义需重训）；**1024..2047 永久 −inf**（旧 Qwen3-TTS 码字，本任务永不出现）；2048..3071 保留 special | 是 |
| 3 | `code_predictor.codec_embedding` ×15（L2542-2543） | `ModuleList([Embedding(2048,1024)] × 15)` | 只用 0..1023；**保留 2048 行 + 全程 mask ≥1024**（推荐：checkpoint 结构不变）。可选 resize 到 1024，只省 15×1024×1024 = **15.7M**＝30B 的 **0.05%**，不值得动结构 | 是 |
| 4 | `code_predictor.lm_head` ×15（L2631-2633） | `ModuleList([Linear(1024→2048,bias=False)] × 15)` | CE 只在 0..1023 上算；**targets 硬断言 `0 <= t < 1024`** | 是 |
| 5 | `codec_bos_id` / `codec_eos_id` | 2149 / 2150（落在 2048–3071 段） | **不重映射** | 否 |
| 6 | `position_id_per_seconds` | talker 13 / thinker 13 | **不动**（见 §0.1） | 否 |
| 7 | 序列长度与显存 | 13Hz 下 8s = 104 帧 | 75Hz 下 8s = **600 帧**，token 数 5.77×；显存预算必须按 T=600 重算（§10.1 #9(b)） | 是（工程后果） |
| 8 | 推理终止 | 采样范围 0..3071 | v0 **定长生成**：长度由序列 `dur` 给定，采样只允许 0..1023（§10.1 #10） | 是 |
| 9 | 训练 target | — | 16 层 target 来自**整段编码**（`data/seq_tokens/codes/*.npz`），第 1 层 + 15 个残差层都必须有 CE（框架 `forward()` 只算第 1 层，L3112 的 `logits = self.codec_head(hidden_states)` 单一出口） | 是 |

## 2. 为什么「按行分区」比 resize 安全

- `codec_head` 与 `codec_embedding` tied：resize 必须两边同步、且 `_tied_weights_keys` 的 name 映射仍然有效，
  一旦顺序错就会静默训成两个独立矩阵（loss 照样下降，推理时才崩）。
- 1024..2047 是**旧 codec 的码字行**，权重非零。不 mask 的话，CE 会在这些行上给出非零概率，
  采样时有概率吐出「旧 codec 的码」，而它对应不到 EnCodec 码本 → 解码端直接越界或语义错乱。
  这属于「训练指标正常、推理必炸」的那类错误。
- mask 的两个位置都要做：**loss 里**（target 已 <1024，但 softmax 分母仍包含 1024..2047，会稀释梯度）
  和 **采样里**（`logits[..., 1024:2048] = -inf`）。只做 target 断言不够。

## 3. token 数据集（本清单的输入，已交付）

| 项 | 值 |
|---|---|
| 形状 | `(1,1,16,T)`，T = ceil(n_24k/320) = round(dur×75) |
| T 范围 | min 232 / median 413 / max **600**（8s 上限，不截断） |
| 码值域 | `[0, 1023]`（实测，287×16×T 全量） |
| 序列 / 击打 | **287 条序列 / 5,472 击打 / 80 source**；split = train 217 / val 38 / test 32 |
| 逐击帧区间 | `data/seq_tokens/hit_frames.csv`，join key = `sequence_id + hit_index` |
| 完整性 | md5 全一致、逐位还原 79,773 帧 0 不匹配、跨 split source 泄漏 0 |
| 交叉验证 | 本仓 `codec_layers_family.py` 重编码 54 条，**前 16 层与 npz 逐位相同（0 不一致）** |

⚠️ `audio_available` 只表示**本地切片在不在**，不是训练可用性：287 条 token 全部在远端，训练按 token manifest 取数，
按 `audio_available=1` 过滤会把 287 砍到 152。

## 4. 还需要 @Talker 确认的两件事

1. `--bench-step` 用 **T=600** 重跑，给 peak VRAM（旧 `max_frames=200` 的数字低估约 3×）。
2. 第 3/4 项选「保留 2048 + mask」还是「resize 到 1024」——我建议前者，但这是训练侧结构决定，你拍。

## 5. 已冻结参数与已知短板（2026-10-27，随 agent.md §5.2f 生效）

| 参数 | 冻结值 | 依据 |
|---|---|---|
| 帧率 | **75 Hz**（13 Hz 已 KILL） | 287 条 / 5,472 段：单击段 gain_matched mel 退化中位 155.4%，13 更差占比 1.00 |
| RVQ 层数 / code groups | **16** | PI §5.2f 裁决④「按 75Hz/16 层推进」 |
| 码本 | **1024**（码值 0–1023） | EnCodec 24k 每组 1024；Talker head 3072 中 1024–2047 永久 −inf、2048–3071 留给 special |
| 序列代价 | 1200 tok/s | 16 × 75 |

**已知短板（登记，不算过关）**：hihats 在 16 层 mel 中位 **0.8429**，越过 §8 的 mel<0.8 线（n=247，同批同实现）；
cymbal 16 层 0.7480 在线内（n=38，方向性证据）。32 层能把 hihats 拉到 0.7816（−7.6%），但收益不是镲类专属
（同序列其他 family −9.7%），代价是 num_code_groups 翻倍 + predictor 15→31 套表 + 序列 token 翻倍。
**是否接受 0.843 不由 mel 决定**，等 12 条 hihats 试听包人耳 A/B 判决。

⚠️ 高频能量占比不随层数改善（hihats 真值 0.9039 / L16 0.8962 / L32 0.8982）→ 16 层的镲类误差是**瞬态/时频细节不足**，
不是「高频被削」。不要按「补高频」思路换 codec。

**同步状态**：`codec_tokens.py` / `codec_rate_ab.py` / `codec_layers_family.py` 三件套 local↔remote md5 一致，
Codec 侧不计入 diff 的 6 个不一致脚本。
