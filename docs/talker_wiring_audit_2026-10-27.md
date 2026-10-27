# Talker 接线审计 vs 官方实现（2026-10-27）

依据：`transformers==5.16.1`，`/root/miniconda3/lib/python3.11/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`
（远端实读，非第三方 config dump）。行号为该文件行号。

## 1. `accept_hidden_layer = 24` 的真实作用域

- L3985：`hidden_states[self.config.talker_config.accept_hidden_layer]`，`hidden_states` 是
  `thinker_result.hidden_states` 元组，**index 0 = 词嵌入输出**，所以 index 24 = 第 24 层之后
  （不是末层 48，也不是第 25 层）。
- L3819 / L3822：`hidden_projection(thinker_hidden)` 只用于**多模态位置**；
  文本位置走 `text_projection(thinker_embed)`。
- L3830：`assistant_hidden = text_projection(thinker_embed[...])` —— 助手段**完全不看 24 层 hidden**。

结论：text→beatbox 的 v0 路径上，24 层 hidden 根本不参与；把它换成末层 hidden **不会**造成
条件分布偏移（因为压根没用到）。真正要用 24 层 hidden 的场景只有两种：
(a) 音频输入位置（`hidden_projection`，输入宽 = `thinker_hidden_size = 2048`，不是 1024）；
(b) 若将来做 audio-prompted 生成。
`talker_config.thinker_hidden_size = 2048`（快照实测），这是 `text_projection` 的输入宽。

## 2. code_predictor 的真实结构（口径要改）

- L2631：`self.lm_head = nn.ModuleList([Linear(hidden, vocab) for _ in range(num_code_groups - 1)])`
  → **15 个 head**，不是 1 个 16 类 head。
- L2542：`self.codec_embedding = nn.ModuleList([... for _ in range(15)])`，实测 len=15
  → **每组一张输入嵌入表**，第 k 组用 `embs[k-1]`。
- L2659 / L2676：`generation_steps = inputs_embeds.shape[1]-2`，`logits = lm_head[generation_steps](hidden_states)`
  → 整合 forward **每次只施加 1 个 head**，这是「必须自写 15×CE」的代码证据。
- 术语纠正：`code_predictor_config.num_hidden_layers = 5` 是**共享的 5 层 predictor body**，
  16 是 code groups。所以正确说法是「15 个残差组，各有独立 head 与独立输入嵌入表，共享同一个
  5 层 transformer」——不是「15 个 predictor 层」。加载权重前必须按这个 layering 对齐。

## 3. 单次前向等价（本轮优化）

因果性使 `hidden[p]` 只见到 `res[:p-2]`，于是
`[h, emb_group0(c0), embs[0](c1), ..., embs[13](c14)]`（16 位）一次前向 → 第 `k+1` 位套
`lm_head[k]` 即等价于 15 次逐步前向。

- 验证：`/tmp/pe.py`（真实 `Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration`，
  随机权重，hidden=1024/2 层/vocab=32/B=2），A=15 次逐步，B=1 次+逐位切片：
  `max|Δlogits| = 2.09e-6`，`rel = 9.7e-7`，`EQUIV True`。
- 落地：`scripts/talker_sequence.py::codec_ce_16` 改为 1 次 forward + 15 个 head，
  token 级 CE 项数仍为 16，`--self-test` 本地/远端均 `terms=16 grads=44`。

## 4. 6-token 前导的归属（回答 Scout 的开放问题）

L3836-3861：`assistant_codec_hidden = [zeros(3), codec_emb([nothink, think_bos, think_eos, speaker, pad, bos])]`
与 `assistant_text_hidden = [proj[:3], tts_pad×4, tts_bos, proj[3]]` 相加，
prefill 共 9 位；`returns inputs_embeds, input_ids(全 tts_pad), trailing_text_hidden`。

→ 这 6 个 token **只是输入条件，不进 target**。target 从第 9 位（`labels[8]`）起才是 codec 帧。
本仓 `build_assistant_frames` 已与此逐位一致。`speaker` 位是嵌入表真实条目（2301/2302/2303），
传 0 等于把普通码字当说话人；已把 `--speaker-id` 默认改为 2301 并加断言（本轮修复）。

## 5. 已知偏离（未修，需 PI 裁决）

真实推理的**下一帧输入**（L3241-3248）是：
`sum([talker emb(group0 code)] + 15 个 predictor 中间 hidden + [最后一组 code 的嵌入]) + trailing_text_hidden`
即 17 项求和。本仓 v0 用 `codec_emb(code0) + trailing_text_hidden`（1 项），
是**可并行训练所必需的近似**（精确版必须按帧串行，T=600 不可行）。
影响：输入方向分布与推理不完全一致；不影响「16 项 CE 都真的被监督」。
建议 v0 保留近似并在契约里写明，v1 用 `T<=64` 的精确串行路径量一次 gap。

## 6. 显存实测（T=600，bf16，grad ckpt ON，随机初始化）

| 层数 | T=232 | T=413 | T=600 | 旧（15 次前向）T=600 |
|---|---|---|---|---|
| 2/20 | 1.95GB | 2.08GB | **2.47GB** | 3.75GB |
| 4/20 | 3.22GB | 3.24GB | **3.24GB** | 4.39GB |

产物：`outputs/talker_bench_L2_oneshot.json`、`outputs/talker_bench_L4_oneshot.json`（远端）。
T 方向已基本平坦（T=413→600 无增长），说明原斜率来自 15 次重复的 predictor 图，不是长序列本身。
2 层→4 层增量 0.77GB（T=600）→ 20 层外推 ≈ 1.7 + 7.7 ≈ 9.4GB（不含 LoRA/优化器状态，
**外推、非实测**）。注意 0.385GB/层 与「3.325B/20 层 = 0.333GB/层」仍不自洽，
下权重后必须先核参数账，再定 24GB 是否够。
