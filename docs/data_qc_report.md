# Data QC 报告（103 音色资产）

生成脚本：`scripts/build_seed_manifest.py`（可复算）

## 摘要

- manifest 行数: 408
- crosswalk 条目: 103
- 覆盖 canonical 103 音色: 34/103
- label_source 分布: {'seed_match': 408}
- split 行数: train=319, val=43, test=46

## 阻塞问题（BLOCKER）

1. human_notation 全空：任何识别准确率在 human_notation 覆盖 >=15 family 前不得写入 §0

## 事实记录

- taxonomy 103 条中 notation 唯一值 97/103，重复记法 6 组: {'P': 2, 'K': 2, 'Kch': 2, 'Df': 2, 'Rhm': 2, 'A': 2}
- label_source 分布: {'seed_match': 408}
- crosswalk 未命中 canonical 的种子库条目: 0/103
- crosswalk 覆盖 canonical 音色 103/103，重复映射 0 个，无种子库条目的 canonical 音色 0 个: []
- crosswalk 匹配方式分布: {'exact_notation_stripped_lt': 39, 'exact_notation': 52, 'notation_ambiguous+name': 12}
- 种子库条目带参考音频(mp3): 95/103
- 短变体条目 6 条（已归位到独立 canonical 音色，未并入基型）: id25(Inward K Rimshot (short)), id2(Fat Kick Short), id70(Inward Hollow Lip Roll (Short)), id19(K palate snare (short)), id65(Vibrator Lip Bass (Short)), id10(Hi-Hat T (short))
- 短变体集合与 canonical taxonomy 一致: [2, 10, 19, 25, 65, 70]（taxonomy 已带 is_short_variant 字段，下游 manifest/refaudio/sequence 均可 join）
- 跨 split source 泄漏: 0（同一 BV 的所有切片同 split）
- source_split_*.txt 覆盖完整: 80/80 个序列 source 全部在清单内（全集 212 个 source）
- 已弃用的 4 类标签（kd/sd/hhc/hho）现状：taxonomy.drum_class 仅存为上游元数据，交付物中不作为标签列；sequence_hits.csv 的 pseudo_label_4class 仅存档，校验脚本持续报 WARN，禁止作为训练文本
- 本地可听（wav_local 指向的文件真实存在）: 408/408
- 人工标注工作单: data\annotation_worklist.csv —— 86 条 / 18 个 family；每 family n: 1–5；达门槛需 ≥15 family 且每 family n≥5
- 标签空间错配：131/408 条 candidate_notation 与 canonical 记法不同（种子库记法带 '<' 前缀）
- 408 条候选覆盖 canonical 103 音色中的 34 个 / 18 个 family（共 19 个 family）
- 记法碰撞：6 组重复记法，25 条候选落在歧义记法上（['A', 'Df', 'K', 'Kch', 'P', 'Rhm']）
- => 以 notation 为唯一 CE 标签单元时，103 类准确率天花板 = 97/103 = 94.2%（6 个音色与他人共享记法，每组至多对 1 个）；必须以 (notation, en) 或 sound_id 作为标签单元
- 标签空间错配：131/408 条的 candidate_notation （种子库空间，带 '<' 前缀）≠ canonical 记法。Talker 训练文本标签必须取 canonical sound_notation
- split train: 319 条 / 53 个 source
- split val: 43 条 / 5 个 source
- split test: 46 条 / 9 个 source

## family 覆盖（canonical 103 口径）

| family | taxonomy 音色数 | 有种子 | 无种子 | 覆盖率 | 种子段数 | 观察到的记法 | human 已确认 |
|---|---|---|---|---|---|---|---|
| clicks | 8 | 2 | 6 | 25.0% | 25 | d|da | 0 |
| clops | 3 | 2 | 1 | 66.7% | 47 | CL*|L | 0 |
| cymbal | 2 | 1 | 1 | 50.0% | 25 | Pss | 0 |
| drag | 2 | 1 | 1 | 50.0% | 25 | A | 0 |
| hihats | 8 | 3 | 5 | 37.5% | 25 | F|Tch|Tss | 0 |
| kicks | 9 | 2 | 7 | 22.2% | 25 | D|Hb | 0 |
| lip_oscillation | 8 | 4 | 4 | 50.0% | 22 | Z|w°|w🗣️P0|w🗣️P8 | 0 |
| liprolls | 7 | 1 | 6 | 14.3% | 25 | W | 0 |
| nasal | 2 | 0 | 2 | 0.0% | 0 | - | 0 |
| other | 10 | 2 | 8 | 20.0% | 27 | Mew|🍾 | 0 |
| reverse | 2 | 1 | 1 | 50.0% | 15 | Chῤ | 0 |
| scratch | 2 | 2 | 0 | 100.0% | 20 | Juy|Shuy | 0 |
| snares | 24 | 3 | 21 | 12.5% | 25 | K`|M|Peh | 0 |
| synth | 2 | 2 | 0 | 100.0% | 25 | Ow|Ow🎵 | 0 |
| teeth | 2 | 1 | 1 | 50.0% | 11 | 🦷🗣️° | 0 |
| tongue | 3 | 1 | 2 | 33.3% | 1 | Drhb | 0 |
| vocal | 4 | 2 | 2 | 50.0% | 25 | O|Rr | 0 |
| whistles | 3 | 1 | 2 | 33.3% | 15 | Hw | 0 |
| zippers | 2 | 2 | 0 | 100.0% | 25 | Zip|Zip~ | 0 |

## 产物

- `data\beatbox_seed_crosswalk.csv`
- `data\seed_manifest.csv`
- `data\seed_family_coverage.csv`
- `data\sound_seed_annotation.csv`
- `data/seed_split_train.txt|val.txt|test.txt`（408 条种子涉及的 source）
- `data/source_split_train.txt|val.txt|test.txt`（全 source 宇宙，新切片按此归位）
- `data/annotation_worklist.csv`（人工标注工作单，由 scripts/annotate_ui.py 导出）

## 序列样本（§10.1 #4）

- 序列数: 287（3–8s，≥12 击/条），击打数: 5472
- source 数: 80；序列/击打 split 分布: {'val': 38, 'train': 217, 'test': 32} / {'val': 712, 'train': 4192, 'test': 568}
- 跨 split source 泄漏: 0；跨 source 拼接序列: 0
- 覆盖 canonical 音色: 49，family: 16/19
- 音频可用性（两个字段，语义不同）:
  - `audio_token_available`（**训练可用性**，以远端 demucs `vocals.wav` 为准）: **序列 287 / 击打 5472（100%）**，
    `audio_truth` 全部为 `vocals`，清单 `data/vocals_available_sources.txt`（380 个 source，80/80 序列 source 全覆盖）
  - `audio_available`（**本地切片是否存在**，PI 2026-10-27 钉死语义，非训练可用性）: 序列 152 / 击打 2943
- T 口径: `T = ceil(dur*75)`（= EnCodec `ceil(n_samples_24k/320)`），实测 min 232 / median 413 / max 600
- label_source 取值: ['model', 'rule_hat']（全部伪标签，human 确认 0）
- 门槛: 全部标签为伪标签（model/rule_hat），human 覆盖 >=15 family 前不得据此宣称识别准确率

- `data\sequence_samples.csv`
- `data/sequence_hits.csv`
- `data/sequence_family_coverage.csv`
- `data\sequence_contract.json`

## 序列 token（Talker 序列监督，2026-10-27）

生成：`scripts/prepare_seq_tokens.py`（训练服务器上跑，音频不出服务器）
校验：`scripts/verify_seq_tokens.py`　加载：`scripts/seq_token_dataset.py`

- 编码成功率: **287/287 序列，0 失败**（远端 `data/seq_tokens/`）
- 时间轴口径: **75Hz**（EnCodec 24kHz / hop 320 原生帧率），非 13Hz
- T 分布: min 232 / median 413 / max 600；全部 ≤ max_frames(600)，**不会被截断**
- codes 形状: `(1,1,16,T)`，码值域实测 [0,1023]（EnCodec 24kbps 码本 1024）
- 逐击帧区间: 5472/5472 覆盖，全部落在 [0,T)，与 hits.csv 逐条 offset 一致
- 跨 split source 泄漏: **0**；无序列跨 source 拼接；split 与 md5(source)%100 全部一致
- 解码往返 QC（6 条抽样）: mel 距离 mean 0.742 / median 0.746 / max 0.800（16 层，与已知 0.768 同量级）
- 解码时长漂移: +0.04% ~ +0.28%（无截断迹象）

### 关键口径修正

1. **13Hz → 75Hz**。13Hz 是 Qwen3-Omni Talker 本体的码本速率，不是本数据集的 token 帧率。
   13Hz 下 60–200ms 的一次击打只剩 0.8–2.6 帧（75Hz 下是 5–15 帧），
   序列监督在 13Hz 上不成立。`sequence_contract.json` 的 `talker_handoff.frame_rate_hz=75`，
   校验脚本新增交叉断言（契约帧率 ≠ 脚本帧率直接 FAIL），防止口径再被写回 13Hz。
2. **整段编码，不是逐击编码再拼接**。EnCodec 有感受野，单独编码一段 200ms 得到的码流
   与同一段音频处在长音频中的码流不同；逐击编码 + 按 offset 拼接既得不到真实时间轴的码，
   又要在空档塞码本外的 pad 码（`codec_ce_16` 断言 `target < 1024`，pad=2050 直接 AssertionError）。
   现在整段编码，每帧都是真实码，帧区间只用于「记法 → 帧」对齐。
3. **max_frames 200 → 600**。200 帧在 75Hz 下只有 2.67s，287/287 条序列会被静默截断。
4. **`round(dur*75)` → `ceil(dur*75)`**。EnCodec 帧数是 `ceil(n_samples/320)`，
   用 `round` 只有 **148/287** 与远端 npz 实际 T 一致，`ceil` 是 **286/287**
   （差 1 条 SEQ-00153 因 CSV 的 `dur` 只保留 4 位小数，容差 ±1 帧）。校验脚本已改 `ceil`。
5. **新增 `audio_token_available` / `audio_truth`，不动 `audio_available` 语义**。
   `audio_available`（本地切片是否存在）被 PI 钉死，语义不改；新增的 `audio_token_available`
   以远端 `vocals.wav` 为准，答案 **287/287**。此前用 `audio_available` 推断"只有 152 条能跑"
   是判据错位：切片只是 `vocals.wav` 的裁切，切片丢了不代表音频丢了。
   校验脚本新增 8c 组断言：有 vocals 却 `audio_token_available != 1` 直接 FAIL。

### 已知伪影（非缺陷）

- 相邻击打帧区间有约 **20ms（1.5 帧）重叠**（中位 0.020s，3270 对）：切片起点是
  `onset-0.02s`（`segment_beatbox_onsets.py` 的 pre-roll），是切分设计而非泄漏；
  因此 `hit_frame_coverage` 可能略大于 1（实测最大 1.127）。
- 全部序列标签为伪标签（model/rule_hat），**human 确认 0**，
  不得据此宣称 103 音色识别准确率。

### 远端产物

- `data/seq_tokens/codes/SEQ-xxxxx.npz`（287 个，2.9MB）
- `data/seq_tokens/hit_codes/<hit_id>.npz`（5,472 个，由整段切出）
- `data/seq_tokens/sequence_manifest.jsonl`（T / dur / text / split / codes_md5）
- `data/seq_tokens/hit_codes_manifest.csv`（5,472 行，hit_id → frame_start/frame_end）
- `data/seq_tokens/hit_frames.csv`（sequence_id + hit_index → frame_start/frame_end）
- `data/seq_tokens/failures.jsonl`（**空**，0 失败）

### 完整性校验

- 287/287 个 npz 的 md5 与 `sequence_manifest.jsonl` 的 `codes_md5` **全部一致**（0 缺失 / 0 不匹配）。
- 逐击 npz **全量逐位还原**校验：79,773 帧 / 287 条序列，**0 处不一致** —— 证明帧区间映射精确，不是近似。
- 码值域实测 [0,1023]，全部落在 EnCodec 24kbps 码本 1024 内。

### ⚠️ Talker 逐击拼接路径已实测崩溃（@Data 在远端复现）

| 路径 | codes 最大值 | `codec_ce_16` 结果 |
|---|---|---|
| A. `SequenceDataset` + `stitch_hit_codes` | **2050** | ❌ `AssertionError: codec target 越界，必须 < codebook` |
| B. 整段 codes（本轮交付） | 1023 | ✅ PASS loss=131.25 terms=16 frames=593 |
| C. 整段 batch（4 条） | 1023 | ✅ PASS loss=131.14 terms=16 frames=1650 |

根因：`talker_sequence.py` 的 `DEFAULT_PAD=2050` > `CODEBOOK=1024`，
而 `codec_ce_16` 断言 `codes.max() < codebook`。
**即使把 pad 改小，逐击拼接在语义上仍不成立**（EnCodec 有感受野，逐击码拼不回整段码流）。
→ @Talker 应改用 `scripts/seq_token_dataset.py`（整段 codes + `slot` 帧→击打序号 + `mask`）。

## 16 层冻结后的数据侧状态（PI §5.2f ④，2026-10-27）

**结论：不重导、不改契约。** `sequence_contract.json` 的 `talker_handoff.shape` 一直是 `(1,1,16,T)`，与 PI 锁定的 16 层一致；已交付的 287 条序列 / 5,472 逐击 npz 全部继续有效（md5 287/287 一致、79,773 帧逐位还原 0 不匹配）。

| 项 | 状态 | 说明 |
|---|---|---|
| 形状 | `(1,1,16,T)`，T=232/413/600 | 不变 |
| 帧率 | 75Hz，`T=ceil(dur×75)`≤600 | 不变 |
| 码值域 | 0–1023（1024 码本） | 不变；不因 head=3072 而扩 |
| 重导工作量 | **0** | 若 PI 改判 32 层才需重导（+32 层、md5 全量重跑） |

**码率口径更正（Critic 2026-10-27 提出，数据侧已核）**：EnCodec `bandwidth=24.0` 实测产出 **32 层** RVQ（远端 `sequence_manifest.jsonl` 的 `raw_quantizers=32` 已逐条记录），本数据集取前 16 层 ⇒ **等效约 12kbps，不是 24kbps**。因此「16 层 = 24kbps」的写法在数据侧一律作废；`sequence_contract.json` 已加 `raw_quantizers / effective_layers / effective_bitrate_kbps / layer_semantics_note` 四字段。已交付的 287 条码流**不受影响**（码值、形状、md5 均不变，只是标注口径），16 层 vs 32 层在 EnCodec 下**不构成控制变量**（层数翻倍即码率翻倍）。

**已知短板登记（codec 侧测量，非数据缺陷）**：Codec 实测 16 层 hihats mel 中位 0.8429（越 §8 Gate 的 0.8 线），cymbal 0.7480（线内）。数据侧计数与其全量口径一致，无需置信度打折：

**split 必须带层级**（序列级与 hit 级不可互换引用）：

| family | 层级 | n | sources | split(train/val/test) | 说明 |
|---|---|---:|---:|---|---|
| hihats | **序列级** | 96 | 43 | **76/11/9** | 序列计数口径；与「115 条含镲类序列」同层 |
| hihats | **hit 级** | 247 | 43 | **190/27/30** | Codec 表里的 190/27/30 是这一层 |
| cymbal | **序列级** | 30 | 20 | **20/6/4** | 同上 |
| cymbal | **hit 级** | 38 | 20 | **27/6/5** | Codec 表里的 27/6/5 是这一层 |
| 同序列对照 | hit 级 | 1,889 | — | — | 用于剥离「同序列其他 family」混淆 |

> 易错点：hihats 190 是**击打数**不是序列数；当序列数用会与 96 对不上（Codec 表已按此更正表头）。

| family | 数据侧判定 |
|---|---|
| hihats | 全量覆盖（247/247，非抽样），短板属 codec 非数据 |
| cymbal | 全量覆盖 38 击打，但 source 仅 20 → 只作方向性证据，中等置信 |

- 不把 hihats 从训练/评测集中剔除：mel 只能证明它最差，不能证明听感不可接受，判决在试听包人耳 A/B。
- family 计数以 `data/sequence_family_coverage.csv` 与 `data/sequence_hits.csv` 为准（本文不改 CSV，避免已交付产物 md5 漂移）。

## 脚本一致性认领（PI §5.2h ⑫，2026-10-27，3 个）

通道 `lf_remote.py test` 退出码 0 / `LF_ALIVE`；GPU 段仅表头 → 记「GPU 不可观测」。本轮不推不拉，仅本地按「脚本以远端为准」归一行尾。

| 脚本 | 远端 md5 | 本地 md5（归一行） | 内容差异 | 判定 |
|---|---|---|---|---|
| `build_hit_bank.py` | `fd379df4` | `fd379df4` | **0 行**（原差异 177/177 行全部来自 CRLF） | 以远端为准；本地已 CRLF→LF 归一，md5 现已相等 |
| `render_synth_tar.py` | `ec8c8234` | `ec8c8234` | **0 行**（同上，229/229 行均为行尾） | 同上 |
| `build_sound_seeds.py` | `1e83449e` | `4764ed69` | **本地前进**：新增 `split_of()`（source md5 80/10/10，与 `prepare_talker_data.py` 同口径）、`--crosswalk` 读 `beatbox_seed_crosswalk.csv` 写 `canonical_notation/en/family/sound_id`，输出 CSV 从 7 列扩到 18 列 | **建议以本地为准**，但 push 须同时带 crosswalk CSV；远端无该文件，单推脚本会把 canonical 字段全写成 `unknown` |

**本地前进的旁证**：已交付的 `Data/sound_seed_annotation.csv` 是 18 列（含 `canonical_*`、`sound_id`、`split`、`crosswalk_method`），只能由本地版产生；远端版输出为 7 列。故「以远端为准」在此脚本上等于回退 103 体系映射。

**前置缺口（记录，不在本轮处理）**：远端 `data/` 下无 `beatbox_seed_crosswalk.csv`，且远端 `sequence_hits.csv` 表头缺 `audio_token_available` / `audio_truth` 两列（23 列 vs 本地 25 列）→ 远端重跑 `build_sound_seeds.py` 或任何依赖这两列的脚本前，需先补 crosswalk 与全量 CSV。

## push 执行记录（PI 裁决⑮，2026-10-27，只放行 2 个文件）

| 步骤 | 动作 | 结果 |
|---|---|---|
| 0 | `lf_remote.py test` | 退出码 0 / `LF_ALIVE`；GPU 段仅表头 → 「GPU 不可观测」 |
| 1 | 远端备份 | `backups/2026-10-27_build_sound_seeds_1e83449e.py`，md5 `1e83449e` 与备份前一致 |
| 2 | push `scripts/build_sound_seeds.py` | 双端 md5 均 `4764ed69` |
| 3 | push `data/beatbox_seed_crosswalk.csv` | 双端 md5 均 `df15038d`，104 行（103 + 表头） |
| 4 | 重跑 `diff` | 不一致 **4 → 3**（剩 `eval_gen_audio` / `prepare_talker_data` / `train_tiny_codec_lm`），仅本地 67 / 仅远端 1703 不变 |
| 5 | 影子跑（只写 `/tmp/seed_check`，已删） | 脚本可执行、crosswalk 被读到（对照组：给假 crosswalk 会打 `[warn]`）；但 `families=0`，原因见下 |

**影子跑为什么 `families=0`（新发现的缺口，不是脚本 bug）**：`--labels` 默认读 `data/real_bbox/hybrid_labels_all_highconf_l2.jsonl`，其 `wav` 指向 `/tmp/bbox_segments_all78/...`，该临时目录已被清理（`os.path.exists` 全 False），且 `confidence` 首条 0.7187 < `--min-conf 0.8`、`sound_family` 为 None。**结论：seed 标注管线在远端当前不可复现**，要重出必须把 `--labels` 重新指向仍存在的 wav（`segments_batch3` + `demucs_srv/htdemucs` 重建），或先落一份持久化 labels。本轮未做，等 PI 决定。

**本轮未动**：远端 `data/sequence_hits.csv`（mtime 2026-10-27T01:32，md5 `3bbcc348`）、`data/sequence_samples.csv`（mtime 01:31，md5 `4e2e61b7`）、`data/sound_seed_annotation.csv`（mtime 2026-10-23，7 列旧版）。本地三个 CSV md5 未变。

### 23 列 vs 25 列：谁在读这两个字段（PI 要的下游影响表）

| 列 | 写入者 | 读取者 | 读的位置 | 缺列的后果 |
|---|---|---|---|---|
| `audio_token_available` | `build_sequence_samples.py`（唯一） | `verify_sequence_samples.py:232`（唯一） | 读 `sequence_samples.csv`，`r.get(...) != "1"` | **假 FAIL**：缺列 → 全部 287 条判为 `!=1`，报「287 条有 vocals 却不可编码」 |
| `audio_truth` | `build_sequence_samples.py`（唯一） | `verify_sequence_samples.py:237`（唯一，仅打印 `Counter`） | 读 `sequence_samples.csv`，`r.get(...)` | 无硬失败，只打印 `{None: 287}` |

**训练/评测主链路 0 依赖**（逐个核过，含远端 grep）：

| 消费方 | 实际读的列 | 是否受影响 |
|---|---|---|
| `codec_rate_ab.py` | `audio_available`（samples）+ hits 的 `offset/dur/notation` | 否 |
| `codec_layers_family.py` | hits 的 `offset/dur/family` | 否 |
| `prepare_seq_tokens.py` / `talker_sequence.py` | hits 的 `offset/dur/记法`、samples 的 `audio_available` | 否 |
| `prepare_talker_data.py` | 只读 npz，不读这两个 CSV | 否 |
| Eval 评测链（`eval_*`） | 不读这两个 CSV | 否 |
| `build_baseline_train_index.py` | 读 `sequence_contract.json` 的 `audio_token_available_*` 计数，不读 CSV 列 | 否 |

**远端 grep 实测**：`grep -rl 'audio_token_available\|audio_truth' scripts/` → **0 个文件**。即远端现有脚本无任何读取，**23 列对远端全部消费者充分**。
**唯一风险点**：若把本地 `verify_sequence_samples.py` 推上远端对 23 列 CSV 跑，会得到 287 条假 FAIL。要么不推，要么推前把第 232 行改成「缺列→跳过并 warn」。**是否升远端契约为 25 列，等 PI 单独裁决。**


## 2026-10-27 增补：序列音频落地位置复核（Data）

脚本：`scripts/qc_sequence_audio.py`（新增，只读输入 CSV，不改任何原列；listing 快照在 `tmp/`，不入库）
证据快照：`segments_batch3` 63,114 wav / `hybrid_highconf_0.7[_78v]` 13,074 wav / `seq_tokens/hit_codes` 5,472 npz（远端实列）

### 结论

1. **训练侧不受影响**：5,472/5,472 hit 有 `.npz` token，287/287 序列 `audio_token_available=1` 复核成立。
2. **音频实体位置比 `audio_available` 列记载的更全**：2,943 hit 在 `segments_batch3`（= 契约 `audio_available_hits` 原值）+ 2,224 hit 只存在于 `hybrid_highconf_0.7[_78v]` 平铺目录 = **5,167/5,472（94.4%）有音频**；305 hit（5.6%）三处皆无。两臂无交集（2,943+2,224+305=5,472）。
3. **序列级**：152（契约口径）→ **272（全位置口径）**，120 条由 0 翻 1、15 条确缺。**test 32/32、val 38/38 全有**，15 条缺音频全部在 train。
4. **305 缺口全部来自 3 个源**：`BV1QW411u7NJ` 213 / `BV1CPnr6VEeE` 58 / `BV1xErMBqE1P` 34；三源均不在 `Data/persisted_audio_sources.txt`（90 源）→ 属「源未持久化」的口径遗留，不是本轮丢失。族分布 snares 197 / drag 25 / clops 17 / clicks 13 / kicks 13 / vocal 10 / hihats 8，其余 ≤6。
5. **双端 CSV 口径**：远端 23 列 / 本地 25 列，共享 22（samples）、23（hits）列逐行内容 **0 差异**，差异仅 `audio_token_available` / `audio_truth` 两列 —— 与既有「23 vs 25 列」结论一致，本轮复核对双端同样成立。指纹：local samples `56433755` / hits `189cc063`，remote samples `4e2e61b7` / hits `3bbcc348`，`sequence_contract.json` 双端同为 `f4d105f4`。

### 影响面（按 PI 已钉死的语义划清）

- `audio_available` 的语义是「`segments_batch3` 是否存在」，**不是**训练可用性，本轮不改列、不改生成器。
- **不影响**：Talker token 训练取数、Eval 固定测试集（test/val 音频全在）。
- **影响**：生成侧/听感侧若按 `audio_available` 取 wav，会把 120 条误判为「无音频」；305 hit 只能做 token 侧监督，不能做音频重建或人耳 A/B。
- 交付物：`Data/sequence_audio_resolution.csv`（5,472 行，逐 hit：`segment_batch3_hit` / `flat_listing_hit` / `token_listing_hit` / `basis`）、`Data/sequence_availability_v2.csv`（287 行，逐序列：`n_hits_audio` / `missing_hit_ids`）。

### 标注门槛修正（旧结论作废）

旧文「人工标注工作单 86 条，每 family n: 1–5」不成立：实测 86 行 = **17 个 family × 5 + tongue × 1**，即 17/18 family 已达 n≥5。门槛「≥15 family 且每 family n≥5」**结构上已满足**，只差人工填写（`human_notation` 仍 0/86，`annotate_status` 86 行全空；slice_wav 86/86、ref_wav 86/86 本地存在，408/408 种子 wav 存在）。tongue 种子池仅 1 条，凑不到 5，按「标注不足」单独记账，不作为门槛阻塞项。

### 远端 AST 权重污染路径

`/gemini/code/lightfour-train/E:lightfour/.cache/hf` 本轮实测**不存在**（远端根目录无 `E:` 目录，`find -maxdepth 2 -name 'E:*'` 空）→ 该项可销账；是否曾清理由 PI 确认，本轮未做任何删除。
