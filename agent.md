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

---

## 7. 通用 codec 选型（一体化关键）

### 7.1 已测结果（20 条真实切片）

| 指标 | Qwen3-TTS 12Hz（语音） | EnCodec 24kHz / 24kbps |
|---|---:|---:|
| 时长保持 | 差（+36ms，80ms 量化） | **完美（0 偏差）** |
| mel 距离（中位） | 0.889 | **0.752** |
| Kicks | 0.87–1.04 | **0.65–0.80** |
| Clicks / Clops / Drag | 0.86–0.99 | **0.59–0.84** |
| Liprolls / Lip oscillations | 0.68–0.99 | **0.55–0.76** |
| Cymbal | 0.31–0.80 | **1.46–1.54（变差）** |
| Hi-hats | 0.54–0.93 | 0.87–1.18 |

**结论**：
- 12Hz 语音 codec 时间分辨率（80ms）低于 beatbox 击打（60–200ms）→ **不可用**。
- EnCodec 24kbps 在短击打上明显更好、时长零偏差，但镲类高频损失大。
- **待测 DAC 24kHz**（重建质量通常优于 EnCodec），选定标准：时间分辨率 ≤20ms、mel 距离最低、镲类不明显劣化。

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
| 1 | DAC 24kHz 对照实验，选定 codec | 🔄 待跑 |
| 2 | 写 `beatbox→token / token→wav` 工具 | ⬜ |
| 3 | 103 音色 seed 标注（200–500 条） | ⬜ |
| 4 | 写 Talker 改造 + 训练管线骨架 | ⬜ |
| 5 | 买 24/48GB GPU | ⏸ 等 A/B/C 完成 |
| 6 | 小数据过拟合验证 | ⏸ 依赖 5 |
| 7 | 联合训练 + 扩数据 | ⏸ 依赖 6 |

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
