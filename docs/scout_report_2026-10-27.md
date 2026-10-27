# Scout 外部侦察报告 2026-10-27

> 范围：Qwen3-Omni 音频输出微调代码 / 通用 codec 与帧率证据 / beatbox 公开工作 / 许可证与硬件门槛。
> 证据来源分级：**[实测]** 本机直接拉取的 GitHub API / 论文 API 元数据；**[三方]** 第三方仓库中的 config dump 或推导；**[官方]** 权重仓或官方仓字段。本轮 **[官方]** 权重 config.json 未拉到（hf-mirror 间歇不可达），标注为待核。
> 边界：论文与第三方数字一律不当实测；不引入 loopstation 类数据源；不替 @PI 拍板。

## 0. 本轮网络条件 [实测]

- GitHub API 通（限速 60/h，剩 45）；`huggingface.co` 直连不通；`hf-mirror.com` 间歇（1 次通、3 次超时）。
- 结论：凡 HF 侧数字，标注来源与时效；权重体量与 Qwen 许可文本待网络稳定后补核。

## 1. 候选清单

| # | 候选 | 链接 | 版本/权重 | 许可证 | 硬件 | 与 Lightfour 的接口差异 | 复现风险 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| A | ms-swift（Qwen3-Omni 训练框架） | https://github.com/modelscope/ms-swift | main，pushed 2026-10-26；`examples/models/qwen3_omni/transformers.sh`、`zero3.sh` | Apache-2.0 | **2×48GB** LoRA SFT（rank8 all-linear、freeze_vit/aligner、max_length 4096）[实测：脚本头注释] | **只训文本输出侧（Thinker）**，不覆盖 Talker codec 头与 16 组 predictor；我们生成侧仍须自写 | 低（15.9k stars、持续维护、有 example） | **P0** |
| B | qwen3-omni-duplex（Talker 接线设计） | https://github.com/thecodacus/qwen3-omni-duplex | main，pushed 2026-08-28，3 stars | **NO-LICENSE**（默认保留所有权利）→ 只读设计，**不得复制代码** | 推理 12GB int4（RTX 3060 实测 p99.9 77ms）[三方] | 它做全双工流式（200ms 帧、静音帧语义）；我们做离线定长生成。但它给出的 **Thinker→Talker 接口事实**（见 §2）与我们 #1 高度相关 | 高（单作者、无 CI、无许可证） | **P1**（只取接口事实） |
| C | QwenLM/Qwen3-Omni 官方仓 | https://github.com/QwenLM/Qwen3-Omni | main，pushed 2026-04-23 | 代码 Apache-2.0；**权重 HF 标记 `license:other`（Qwen 自定义许可，非 Apache）**[实测：HF API tags] | 权重 30B-A3B 约 60GB BF16（§10.1 #6 未关闭） | **只有 18 个 cookbook + web_demo，无任何训练代码**[实测：仓顶层与 cookbooks 目录] → Talker 训练脚本无官方参考实现可抄 | 低（官方）；但训练路径为零 | **P0**（事实基线） |
| D | ishine/qwen3_omni_speechllm | https://github.com/ishine/qwen3_omni_speechllm | 单次提交 2026-09-29，0 stars | **NO-LICENSE** → 不得抄代码 | 未给显存数字 | 范围是 ASR（音频→文本）+ LoRA target 清单（`lora_targets_thinker_attn_audio_proj.txt`），**不覆盖音频输出** | 高（0 star、无测试、无许可） | **P2** |
| E | beatbox 公开工作 | GitHub / arXiv | 见 §3 | 多为 none | — | 无可用 baseline | — | **P2/P3** |

## 2. 接口事实（可直接交付 @Talker / @Codec / @PI）

以下字段来自 `thecodacus/qwen3-omni-duplex/docs/qwen3_omni-config.json`（**[三方]**，是权重 config 的副本）与 `docs/design.md`（**[三方]** 推导）。**权威值以远端 `config.snapshot.json` / 权重 config.json 为准**，请 @PI 复核同名字段。

1. **`talker.accept_hidden_layer = 24`** —— Talker 取 **Thinker 48 层中的第 24 层** hidden 做条件（design.md 原文："The Talker is conditioned on Thinker layer-24 hidden states per token position, and never sees layers 25–47"）。
   → **风险提示**：若 `train_omni_talker.py` 现在喂的是末层 hidden，与预训练条件分布不一致，属 §10.1 #1 的隐藏坑。接线前必须核我们取的是第 24 层还是末层。
2. **`position_id_per_seconds = 13`（thinker / talker 两处均为 13）**，`seconds_per_chunk = 2`。
   → **支持 PI 裁决②**（该项保持 13 不改 75）。与 @Codec 的 #9(a) 不冲突：#9(a) 争的是**codec token 帧率 75Hz vs 13Hz 视图**，config 的 13 是 RoPE 时间轴。
3. **codec special tokens 是一次性 6-token 前导，不是逐帧控制，也没有可预测的逐帧 eos**：
   `[codec_nothink_id=2155, codec_think_bos_id=2156, codec_think_eos_id=2157, speaker_id, codec_pad_id=2148, codec_bos_id=2149]`（design.md 引源码：固定 6-token 前导，模式+音色头）。
   → **直接支持 §10.1 #10**（v0 定长生成、eos 推 v1）。
   → **连带待定**：v0 的 CE 只切 0–1023 时，这 6 个前导 token 是"作为前缀条件输入、不进 target"还是直接丢弃，要在契约里定死（建议前者，否则丢失音色/模式条件）。
4. **`code_predictor_config`：hidden 1024、5 层、16 code groups（MTP）**；Thinker 48 层 / Talker.text 20 层 hidden 1024 / MoE 128e/6a。
   → 与我们「15 残差层 + 16 组」口径需对齐：**16 = code groups**，而"5 层"与"15 残差层"不是同一个维度。请 @Talker 用远端 config 复核 `code_predictor_config` 各字段，再定自写 predictor 的结构，避免权重形状对不上。
5. **code2wav**：`codebook_size=2048 / semantic_codebook_size=4096 / codebook_dim=512`；参考实现 `chunked_decode(talker_codes, chunk_size=300, left_context_size=25)`（300 codes @12.5Hz = 24s/次）。三方实测：`chunk_size=1` 时**每帧丢 28.9% 样本**（干净截断、保留部分 bit-accurate）。
   → 我们走 EnCodec 解码，**不踩这条**；仅当未来复用 code2wav 时须改 stateful decode。
6. **硬件旁证**：ms-swift 官方 example 标注 **2×48GB 可做 30B-A3B LoRA SFT**（rank8 all-linear、max_length 4096）。→ §10.1 #2/#15 的显存结论仍以 **T=600 的 `--bench-step`** 为准；"单卡 48GB 装 30B 全参 + 优化器"依旧不成立。

## 3. beatbox 侧公开工作 [实测]

- GitHub 检索 `beatbox generation neural` / `beatboxing dataset` / `beatbox recognition`：**无可复现的公开仓库**（命中均为 0-star 玩具：`bhardvaja/BeatBoxer`、`ONEmanLAW/Cypher`、`hkgdarren/beatboxing-gesture-recognition`）。
- arXiv 检索 `beatbox / beatboxing`（按提交日期倒序）：
  - **DARC: Drum accompaniment generation with fine-grained rhythm control**（2026-01-05）—— 鼓伴奏生成 + 细粒度节奏控制，可作"鼓谱→音频"的对照口径参考。
  - **The Rhythm In Anything: Audio-Prompted Drums Generation with Masked Language Modeling**（2025-09-19）—— audio-prompted 鼓生成，与"音色→音频"条件注入同构。
  - **A New Dataset for Amateur Vocal Percussion Analysis**（2020-09-24）—— 人声打击数据集，**许可证与内容未核**，是否可用于识别侧预训练待定（非 loopstation，但引入前须过许可证与边界）。
  - 其余命中与 beatbox 无关（心脏电生理 BeatBox 等）。
- **结论**：我们没有可借的 beatbox baseline，Swissbeatbox 管线是稀缺资产；上述 3 项只能作对照/评测参考，不得当作指标背书，数据引入前须核许可证。

## 4. 对当前阻塞项的直接输入

- **#1（Talker 训练脚本未闭环）**：确认**无官方训练代码、无可用第三方训练实现**（duplex 是推理/流式改造，ishine 是 ASR）。自写是唯一路径；但 §2-1（第 24 层）与 §2-4（predictor 5 层/16 组）是接线前必须先核的两个结构字段。
- **#2/#15（显存预算）**：2×48GB LoRA SFT 是外部旁证；我们自己的 T=600 bench 仍不可省。
- **#9(a)（75Hz vs 13Hz 判定）**：本轮未找到可直接引用的"击打瞬态 vs codec 帧率"公开对照指标，@Codec 的 30 分钟级实验仍是唯一判定依据（P0，不可用文献替代）。
- **#10（终止条件）**：6-token 前导事实使"v0 定长、eos 推 v1"有了外部佐证。

## 5. 下一轮计划

1. 网络稳定后补：Qwen3-Omni 权重分片/总容量、`Qwen LICENSE` 具体条款（HF `license:other`）。
2. 核 "Amateur Vocal Percussion" 数据集的许可证与内容（是否可作识别侧预训练、是否触碰 loopstation 边界）。
3. 跟踪 Moshi/Mimi 与 EnCodec 在瞬态重建上的公开指标，为 #9(a) 提供外部对照口径（不能替代实测）。
4. 复核 `code_predictor_config` 完整字段（等远端 config 可读时）。
