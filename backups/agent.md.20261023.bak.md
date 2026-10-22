# agent.md — lightfour 项目主线

> **每次对话开始先读本文件。** 这是项目唯一主线文档（没有 project.md）；重大决策、路线变更、数据/训练进度必须同步更新。
> 最后更新：2026-10-23（Qwen3-Omni 调研 + 音色 taxonomy/L2 重标注 + 第 3 批数据入库）

---

## 1. 项目定位与最终目标

**一句话**：做一个 beatbox 的「识别 → 鼓谱转写 → 生成」系统，最终能听懂音色、写出鼓谱，并按鼓谱生成 beatbox 音频。

四条能力：

1. **音色识别**：听人声打击乐，区分 kick / snare / 闭镲 / 开镲，并逐步扩到 8–15 个音色家族。
2. **鼓谱转写**：把连续 beatbox 转成 `(时间, 音色, 力度)` 序列。
3. **beatbox 生成**：给定鼓谱/符号序列，生成可辨识的 beatbox 音频。
4. **音乐识别（次要）**：电子音乐子类（House/Techno…），FMA 数据已就绪。

**已确认的技术边界**：
- 识别/转写是监督任务，**专用音频分类器**比通用大模型更合适（关键在数据质量，不在模型大小）。
- 生成是独立的条件生成任务，需要**专用生成模型**（mel→声码器 或 codec token→解码器）。
- beatbox 太小众，没有可直接下载的现成权重；**自训模型 + 数据集本身就是稀缺资产**。

---

## 2. 当前状态 TL;DR（2026-10-23）

| 维度 | 状态 |
|---|---|
| 真实数据 | B 站 **468 个视频**下载，Demucs **380 个 vocals**；高置信切片 **28,489 段 / 1.901h / 141 个来源** |
| 音色体系 | **103 音色 taxonomy** 已建（`docs/beatbox_sound_taxonomy.md` + `data/beatbox_taxonomy.json`）；**95 条种子音频**已下载 |
| L2 家族重标注 | 已完成（`hybrid_labels_all_highconf_l2.jsonl`），**sd 只有 50% 是真军鼓、hhc 65% 被判军鼓** |
| 识别模型 | 4 类最好 **0.5721**（本地 v2）/ 0.5147（服务器）/ 0.4558（v3）/ w2v CV 0.5435，**未过 70% Gate** |
| 生成模型 | Route B 条件 Flow Matching 原型跑通（3.96M，smoke loss 2.09→0.60）；合成数据 90k 窗口 / 50h；真实生成对 3,036 / 1.69h |
| Qwen3-Omni | Talker **有 labels/loss**，但 ms-swift 只打通 TTS 音频输出训练；codec 语音专用；30B-A3B 需 24GB+ GPU。**codec 重建实验进行中** |
| 质量 Gate | CNN vs RF 一致率 **70.89%**；33 条人工抽检包已备好，**尚未人耳确认** |

**一句话结论**：数据和音色体系已经够用，当前瓶颈是**标签质量**（4 类混杂）和**识别 Gate 未过**；生成路线 Route B 已有原型，Qwen3-Omni 一体化是待验证分支，不是当前主线。

---

## 3. 三层架构（识别 / 生成 / 语言层）

```
beatbox 音频
   │
   ├─[识别层] 专用分类器（L1 4类 → L2 8-10家族 → L3 音色级/CTC 序列）
   │        └→ 符号序列（taxonomy 记法：P / PF / T / W / Drh …）
   │
   ├─[生成层] Route B：符号序列 → piano-roll 条件 → Flow Matching → mel → 声码器 → 音频
   │
   └─[语言层] Omni（只做文本对话/描述/调度，不做 beatbox 音频 I/O）
```

**关键判断**：
- 用户提出的「beatbox 当语言、音色=token」在概念上成立，**Route B 的条件向量就是音色 token 编码**（每个家族一个通道 + tempo），只是不在 LLM 词表里。
- 识别与生成拆成两个专用模型，比强行用一个 Omni 更可靠、更便宜、更容易验证。
- Omni 的语言层可以描述「刚才听到的是 PF 军鼓 + T 镲」，但不负责生成音频。

---

## 4. 音色体系（taxonomy，必须遵守）

### 4.1 现有 4 类只是鼓组 L1

| 缩写 | 英文 | 中文 | 说明 |
|---|---|---|---|
| `kd` | kick drum | 底鼓 | 来自 AVP-LVT 鼓组标签 |
| `sd` | snare drum | 军鼓 | 同上 |
| `hhc` | hi-hat closed | 闭镲 | 高频 + 短衰减（<120ms） |
| `hho` | hi-hat open | 开镲 | 高频 + 长衰减（≥120ms） |

**注意**：文件名里的 `kd/sd/hhc/hho` 是鼓组缩写，不是 beatbox 术语；没有 `kb`/`sb`。

### 4.2 完整 103 音色（Beatbox Science）

- 标准记法：`P`（kick）、`PF`（PF snare）、`K`（inward K snare）、`T`（hi-hat）、`W`（lip roll）、`Drh`（click roll）、`O`（throat bass）等。
- 发声机制：Laryngeal / Pulmonary / Humming / Hollow；气流：outward / inward。
- 完整表见 `docs/beatbox_sound_taxonomy.md`；程序读取 `data/beatbox_taxonomy.json`。
- 种子音频：`data/beatbox_seeds/`（95 条 mp3，0.8MB，研究用、gitignore；元数据 `sounds.json` 已入库）。

### 4.3 分层方案

| 层级 | 类别 | 用途 | 状态 |
|---|---|---|---|
| L1 鼓组 | 4 类 | 兼容现有数据 | ✅ 现有 |
| L2 家族 | 8–10 类（kicks/snares/hihats/cymbal/bass/click/lip/fx） | 训练下一代分类器 | 🟡 种子 + 重标注已就绪 |
| L3 音色 | 20–100 类 | 精细转写 | ⬜ 长期 |

**manifest 字段扩展**（向后兼容）：
```json
{"drum_class": "sd", "sound_family": "Snares", "sound_notation": "PF", "sound_confidence": 0.0}
```

---

## 5. 数据资产现状

### 5.1 B 站真实数据管线（全部跑通）

| 步骤 | 脚本 | 结果 |
|---|---|---|
| 下载 | `fetch_bilibili_api.py` | **468 个 WAV**（468/468 下载完成） |
| 人声分离 | Demucs `htdemucs`（`-j 4` GPU 批处理） | **380 个 vocals**（其余 88 个 >1200s 跳过） |
| onset 切片 | `segment_beatbox_onsets.py` | 第 3 批 90 视频 → **63,114 段** |
| AST 过滤 | `ast_filter_beatbox.py` + `filter_segments_by_ast.py` | **39,796 段（63.1%）** |
| 伪标 | `pseudo_label_v2.py`（robust ckpt，GPU） | 39,796 段全标 |
| hybrid 规则 | `hybrid_label_segments.py` | 高置信 **14,088 段 / 0.958h** |

**累计高置信**：13,074（78v）+ 1,327（首批对战）+ 14,088（第 3 批）= **28,489 段 / 1.901h / 141 来源**。

### 5.2 数据源策略

- 只抓 Swissbeatbox 官方账号（mid `171444239`）的 **vs / Battle / Elimination / 决赛/半决赛/淘汰赛**。
- **排除 loopstation**（中文「环站」同样排除，KBB「环站版线上赛」是 loopstation）。
- 排除 Showcase / Shoutout / 合辑 / 合辑 / Crowd POV / 海选 / 冠军表演。
- `championship`（锦标赛）保留，`champions`（冠军表演）排除。
- 版权：只做研究，音频不入 git、不外传。

### 5.3 音色种子 + L2 重标注

- 种子：103 音色中 **95 条有参考音频**（8 条无音频：Cough/High Cough/Slizzer Roll/Zekka Click Roll/Inward Throat Bass/Vortex Whistle/Smile Bass/Inward Chest Bass）。
- 方法：种子 9 维特征（时长/低中高频/质心/ZCR/衰减/RMS/峰值）→ 对 28,489 条做最近邻 → `hybrid_labels_all_highconf_l2.jsonl`。
- **家族分布**：Snares 13,000（45.6%）、Drag 3,220、Hi-hats 2,115、Other 1,935、Clicks 1,589、Vocal 1,265、Kicks 1,057、Zippers 1,016、Liprolls 1,002、Cymbal 666、Synth 426、Clops 374、Reverse 253、Lip oscillations 189、Scratch 176、Pops 89、Teeth 38。
- **关键交叉**：`sd→Snares` 8,081（50%）、`sd→Drag` 3,055、`sd→Clicks` 1,133；`hhc→Snares` 3,635（65%，异常）；`kd→Vocal` 1,061、`kd→Liprolls` 657；`hho→Snares` 1,142、`hho→Cymbal` 459。

### 5.4 质量评估（Stage 1）

用独立 RF（15 维手工特征）对 28,489 条重预测：

| 标签 | 一致率 | 主要分歧 |
|---|---:|---|
| kd | **91.75%** | 293→sd、69→hhc |
| hhc | **84.47%** | 238→hho、422→kd |
| sd | **65.27%** | **5,489→kd（33.8%）** |
| hho | **29.98%** | 658→hhc、380→kd、356→sd |

- 总体一致率 **70.89%**；高置信但两模型分歧 **1,470 条（5.2%）**。
- 结论：**kd/hhc 可用；sd 约 1/3 与 kd 混淆；hho 基本不可靠**。
- RF 训练于 AVP-LVT，存在域偏移，不能当 ground truth；**必须人工抽检**。

**人工抽检包（本地，33 条）**：
- `data/bbox_raw/spotcheck_20261022/`（9 条，分层）
- `data/bbox_raw/spotcheck_batch3/`（12 条，每类 3 条）
- `data/bbox_raw/suspicious_spotcheck/`（12 条，`cnn-X_rf-Y` 分歧样本，最适合裁决）

**质量 Gate**：错误率 >10% → 停止伪标滚库，转人工 seed/域适应；hho 单独降阈值或补标；sd-kd 加衰减/低频规则。

---

## 6. 识别模型现状与 Gate

| 版本 | 关键改动 | 结果 |
|---|---|---|
| v1 | log-mel CNN + BatchNorm | test 0.5008（Kick 95% / Snare 40% / 闭镲 43% / 开镲 8%） |
| v2 | 类权重 + 波形增强 | 本地 **0.5721**（最好）；服务器重训 0.5147 |
| v3 | mel 域增强 + GroupNorm + EMA | 0.4558 |
| w2v | 冻结 wav2vec2-base + 层权重 | test 0.5271；4-fold 参与者 CV **0.5435 ± 0.016** |
| robust | 混真实 `no_vocals` 背景 | 0.4372；真实域平均置信 0.556 |

**Gate（未过）**：4 类 test ≥70% + 音频敏感性测试通过。**eval_loss 不作为验收指标**（v3 的教训：loss 低是背模板）。

**已知问题**：
- val→test 系统性 gap ≈ 0.20（2 人 val 集高估）。
- 4 类标签内部混杂（见 §5.3），继续优化 4 类收益有限。

**下一步**：用 95 条种子 + L2 标注训 **8–10 类家族分类器**，替换 4 类伪标器。

---

## 7. 生成路线（Route B）

### 7.1 数据

| 集合 | 规模 | 说明 |
|---|---|---|
| 合成 | **90,000 窗口 / 50.00h** | hit_bank 按鼓谱重排；服务器 `gen_synth_tar/` 8.3GB，45 tar 分片；本地 6 个等间距分片 + 全部标注 |
| 真实 | **3,036 窗口 / 1.69h** | 65 个来源；kd 2,220 / sd 5,860 / hhc 2,924 / hho 945 |
| 真实 hit bank | 4,872 个击打 | 4 类 × 28 参与者，0.330h |

合成数据类别均衡（每类 131k 音符），但 hit 复用 91–130×，**真实录音仍是瓶颈**。

### 7.2 模型

- `scripts/train_gen_routeb.py`：小型 2D U-Net 条件 Flow Matching，**3.96M 参数**。
- 条件：4 类 piano-roll + tempo，`(5,122)`；目标 log-mel `(1,64,122)`。
- Smoke：loss **2.09→0.60**（100 步 CPU，54s），checkpoint 15.9MB，能生成 4 条 mel。
- **扩类**：条件从 4 类扩到 8–10 类（L2 家族），用 95 条种子做音色素材扩 hit bank。

**Gate**：按给定鼓谱生成可辨识的 beatbox 片段（人耳/分类器能听出对应音色）。

---

## 8. Qwen3-Omni 一体化方案（调研与待验证分支）

### 8.1 为什么 Qwen2.5-Omni 不行（已验证）

1. **输入侧不听音频**：v3 解冻 audio_tower 后，换音频/不给音频输出都一样（音频敏感性测试失败），test 29.5% ≈ 随机 25%。
2. **输出侧无训练接口**：`Qwen2_5OmniTalkerForConditionalGeneration.forward` 无 `labels`/loss；ms-swift 4.5.3 的 `Qwen2_5OmniTemplate` 只支持音频输入→文本。
3. **根因**：答案模板主导 loss + 音频塔是语音导向，不适合打击乐瞬态。

**教训**：eval_loss 不能验收；唯一可信 = 音频敏感性 + test 准确率。

### 8.2 Qwen3-Omni 的进展与阻碍（2026-10-23 实测代码）

**进展**：
- transformers 5.16.1 已实现 `qwen3_omni_moe`。
- `Qwen3OmniMoeTalkerForConditionalGeneration.forward` **有 `labels` 和 CE loss**，codec token 可被监督训练。
- 架构：Thinker-Talker MoE（30B-A3B），Talker 多码本自回归 + MTP 残差码本。

**三个硬阻碍**：
1. **ms-swift 未打通 Qwen3-Omni 音频输出 SFT**：`Qwen3OmniTemplate` 继承 2.5 模板，只做输入→文本；只有 **`Qwen3TTSTemplate`** 支持音频输出训练（`tts_audio_codes` 16 码本 + `ref_audios` mel，依赖 `qwen_tts`），且是**纯语音 TTS**。
2. **codec 语音专用**：24kHz 多码本为语音设计，beatbox 宽频瞬态（kick/hi-hat/lip roll）重建保真度未验证。
3. **规模**：30B-A3B，6.2GB GPU 训不了，至少 24GB+。

### 8.3 codec 重建实验结果（2026-10-23 已完成）

**方法**：Qwen3-TTS-Tokenizer-12Hz，20 条真实切片（Snares/Kicks/Hi-hats/Cymbal/Liprolls/Lip oscillations/Vocal/Clicks/Clops/Drag 各 2 条），`encode → decode`，24kHz。

| 指标 | 结果 |
|---|---|
| 成功重建 | **20/20** |
| log-mel 距离 | 均值 **0.838**（中位 0.889，最好 0.315，最差 1.038） |
| 时长变化 | 平均 **+36ms**（12Hz = 80ms/帧量化：0.10s→0.16s、0.11s→0.16s、0.14s→0.16s） |
| 频谱质心 | 平均 **-58 Hz**（略变暗） |
| 相对最好 | Cymbal 1.02s（0.315）、Hi-hats 0.18s（0.537） |
| 相对最差 | Kicks 0.12s（1.038）、Vocal 0.17s（1.019）、Lip oscillations（0.97–0.99）、Clicks/Clops/Drag（0.86–0.99） |

**结论（重要）**：
- **12Hz 语音 codec 的时间分辨率（80ms/帧）低于 beatbox 击打时长（60–200ms）**，短瞬态被量化拉长、衰减细节丢失；低频 kick 与短 click 失真最大，长音镲类尚可。
- **Qwen3-Omni「直接用语音 codec 生成 beatbox」路线判死**——不是模型规模问题，是 codec 粒度不匹配。
- 若仍要走 Omni/LLM 一体化，需要换成**通用音频 codec**（EnCodec 50/75Hz、DAC、SoundStream）或 Route B 的 mel+声码器，并自写训练管线；Qwen3-Omni 现有 Talker 接口按语音 token 设计，改造成本高。
- 对比音频已下载本地：`data/bbox_raw/codec_test/orig/` 与 `recon/`（20 对），`report.json` 有逐条指标。

**下一步**：人耳试听 20 对；若确认瞬态损失不可接受，则关闭「Omni 一体化生成」分支，集中做 Route B；若想保留可能性，再做一次 EnCodec/DAC 对照实验。

### 8.4 若 codec 验证通过，落地条件

1. 24GB+ GPU（QLoRA 30B-A3B 或更小 Qwen3-Omni 变体）。
2. 自写 Qwen3-Omni Talker 音频输出 SFT 管线（参考 `Qwen3TTSTemplate`：audio codes + ref mel + loss）。
3. 音色 token 映射：taxonomy notation → codec token 序列；或直接用 Route B 的 piano-roll 条件。
4. 先做小规模（1–2h）过拟合验证，再扩数据。

---

## 9. 资源、环境与安全

| 资源 | 现状 | 用途 |
|---|---|---|
| 本地 | Windows / AMD RX 7800 XT（ROCm 不可用）/ RAM 15.4GB | 小模型 CPU 训练、试听、代码 |
| 云端开发实例 | 趋动云 **96 核 / 503GB RAM / B1.gpu.small 6.2GB** / Python 3.11.8 | 数据管线、Demucs、GPU AST、伪标、小模型推理 |
| 持久化盘 | `/gemini/code`（seaweedfs，小文件写慢） | 产物直写持久化盘 |
| 只读数据 | `/gemini/data-1/`：AVP-LVT、FMA、sft_samples | 就地构建 |
| 服务器模型 | `checkpoints/beatbox_cls_v2`、`beatbox_cls_robust`、`beatbox_feat/rf.joblib`、`models/Qwen3-TTS-Tokenizer-12Hz` | 识别/评测/codec |

**数据安全约定**：只允许**代码上行**（.py/.sh），**禁止数据上行**；数据下行允许。公开研究音频在服务器直接下载，不从本地上传。

**工程要点**：
- `ProcessPoolExecutor` 在 torch 下默认 fork 会死锁 → 必须 `mp.get_context("spawn")`。
- seaweedfs 小文件读慢 → 先复制到 `/tmp` overlay 盘再处理。
- Demucs 用 `-j 4` 单进程批处理（约 12.6 audio-s/s），不要开多个 GPU 进程争抢。
- AST 用 GPU 单进程（约 4 视频/分钟），不要 12 进程 CPU。
- Windows 上 `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1` 防 OpenMP 死锁。

---

## 10. 下一步行动计划

| # | 行动 | 交付物 | 状态 |
|---|---|---|---|
| 1 | **codec 重建实验**（Qwen3-TTS-Tokenizer → beatbox encode/decode） | `report.json` + 原始/重建 wav | 🔄 进行中 |
| 2 | **人工抽检 33 条**（分层 + 分歧样本） | 错误率报告 | ⬜ 等用户试听 |
| 3 | **L2 家族分类器**（8–10 类，95 种子 + 28k 标注） | `train_beatbox_family_clf.py` + checkpoint | ⬜ 待启动 |
| 4 | **L2 抽检验证**（每家族 3–5 条） | 家族准确率 | ⬜ 依赖 3 |
| 5 | **Route B 扩类**（4→8–10 类条件） | 新 checkpoint + 试听样本 | ⬜ 待启动 |
| 6 | **第 4 批伪标**（380 vocals 中剩余视频） | 新增高置信切片 | ⏸ 等质量 Gate |
| 7 | **Omni 语言层**（描述音色/对话） | SFT v4 修正版 | ⏸ 非当前优先 |
| 8 | **Omni 一体化**（若 codec 验证通过） | Talker 音频输出 SFT 管线 | ⏸ 待验证 + 大 GPU |

**顺序原则**：先验证 codec 和人工抽检 → 再训 L2 → 再扩 Route B → 最后考虑 Omni 一体化。

---

## 11. 会话启动检查清单

1. 读本文件，确认当前阶段（识别 Gate / L2 / 生成 / codec 验证）。
2. **验收铁律**：识别只认 test 准确率 + 音频敏感性，绝不看 eval_loss。
3. 数据质量未过人工审计前，不并入 Omni SFT、不继续大规模滚库。
4. 远程操作：代码上行、数据下行；产物写 `/gemini/code`。
5. 版权：音频只做研究、不入 git、不外传。
6. 服务器可能重建：先确认 SSH/Demucs/缓存可用。

---

## 12. 变更日志（精简）

| 日期 | 变更 |
|---|---|
| 2026-09-25 | 建立 agent.md |
| 2026-10-15~16 | Omni v2/v3 QLoRA 失败：不听音频、模板背诵；确立「eval_loss 不能验收」 |
| 2026-10-19 | 战略转向专用音频分类器 + 伪标注数据飞轮 |
| 2026-10-20 | Stage A v1/v2/v3 + w2v；生成数据管线（hit_bank 4,872 / 合成 50h） |
| 2026-10-22 | B 站管线全量：468 视频 / 380 vocals / 28,489 高置信 / 1.901h；质量交叉验证 70.89%；103 音色 taxonomy + 95 种子；L2 重标注 |
| 2026-10-23 | Qwen3-Omni 调研：Talker 有 labels/loss，但 ms-swift 仅 TTS 支持音频输出；codec 重建实验启动；agent.md 重构 |
