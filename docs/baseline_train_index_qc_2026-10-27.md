# baseline_train_index QC（2026-10-27）

生成器：`scripts/build_baseline_train_index.py`（可重复运行，纯本地，不外传音频）。

## 结论

- 训练索引 **319 行**（B 站切片，split=train，来自 53 个 source）
- 标签空间 **34 类 + unknown**（`data/baseline_label_space.csv`，label_index 0–34）
- seedlib 探针 95 条与训练索引的音频同一性检查：包络候选 12 对，波形 NCC 最高 **0.1558**（判漏阈值 0.5），确认泄漏 **0** 条
- 训练/探针 sound_id 交集 **34**（要求恰好等于词表的 34）
- 跨 split source 泄漏：train&val 0 / train&test 0 / val&test 0
- 索引内 wav 字节级重复：0 组；与探针 sha1 交集：0 条

## 三条硬门槛

**1. seedlib 95 条不得进训练集**

- 路径域检查（无 `beatbox_seeds` 路径）0 命中；`row_id` 与 `ref_id` 名字冲突 0 命中；`wav_sha1` 与 95 条探针文件 sha1 交集 **0**。
- 声学同一性两级判据：① 时长预筛 ±30ms 后算 10ms RMS 包络相关，≥0.95 进候选；② 候选对做整段波形归一化互相关（NCC，对幅度/时移不变），≥0.5 判泄漏。
- 实测：候选 **12 对 / 319 行**，NCC 最高 **0.1558**，判泄漏 **0 条**。
- **口径事故记录**：只看包络会误报 12 条（最高 0.9862），但同一对 NCC 只有 0.156——60–200ms 打击的包络都是「冲激+指数衰减」，不同录音也能相关 0.95+，且这 12 对全是**跨类**（如 sid 54 `O` vs sid 0 `P`）。所以「包络高＝泄漏」不成立，查泄漏必须做波形 NCC；单用包络会逼人删正常样本。
- 最高 NCC 对：训练 `BV1ig4y1v7Fu_0067` vs 探针 `025_kick`。
- 残余风险：NCC 漏得掉「同源但被重采样+压缩+混响」的改写（真重复通常 >0.9，0.5 阈值已留余量）；本批训练音频都是 B 站实战切片（含房间声/观众声），与 seedlib 录音棚单次击打域本来就不同。

**2. 交集只允许 34**

- train sound_id 集合与 `label_vocab_covered.csv` 完全一致（34 类），与 95 条探针交集 **34**。
- 探针中不在词表的 61 条**不由本索引吞掉**，交 @Eval 侧标 `out_of_vocab` 单列不计分。

**3. 不按 `audio_available` 过滤**

- 本索引按 hit 级 wav 路径构建，319/319 路径存在且可读（22050Hz 单声道）。
- 序列侧口径仍取 `sequence_contract.json` 的 `audio_token_available` 287/287（train 217），本索引不参与该过滤。

## 每类 train 样本数

| label_index | sound_id | 记法 | family | train n | <5 标注 |
|---|---|---|---|---|---|
| 0 | 6 | `D` | kicks | 14 |  |
| 1 | 7 | `Hb` | kicks | 2 | 是 |
| 2 | 11 | `Tss` | hihats | 10 |  |
| 3 | 12 | `F` | hihats | 9 |  |
| 4 | 13 | `Tch` | hihats | 1 | 是 |
| 5 | 25 | `` K` `` | snares | 0 | 是 |
| 6 | 37 | `M` | snares | 21 |  |
| 7 | 38 | `Peh` | snares | 1 | 是 |
| 8 | 41 | `Pss` | cymbal | 22 |  |
| 9 | 43 | `d` | clicks | 1 | 是 |
| 10 | 44 | `da` | clicks | 18 |  |
| 11 | 51 | `L` | clops | 20 |  |
| 12 | 53 | `CL*` | clops | 16 |  |
| 13 | 54 | `O` | vocal | 17 |  |
| 14 | 57 | `Rr` | vocal | 3 | 是 |
| 15 | 59 | `w°` | lip_oscillation | 5 |  |
| 16 | 60 | `w🗣️P0` | lip_oscillation | 1 | 是 |
| 17 | 62 | `w🗣️P8` | lip_oscillation | 6 |  |
| 18 | 65 | `Z` | lip_oscillation | 4 | 是 |
| 19 | 66 | `W` | liprolls | 21 |  |
| 20 | 74 | `Drhb` | tongue | 1 | 是 |
| 21 | 79 | `Hw` | whistles | 10 |  |
| 22 | 82 | `🦷🗣️°` | teeth | 10 |  |
| 23 | 83 | `A` | drag | 6 |  |
| 24 | 84 | `A` | drag | 12 |  |
| 25 | 85 | `Zip` | zippers | 1 | 是 |
| 26 | 86 | `Zip~` | zippers | 18 |  |
| 27 | 87 | `Juy` | scratch | 9 |  |
| 28 | 88 | `Shuy` | scratch | 8 |  |
| 29 | 90 | `Chῤ` | reverse | 13 |  |
| 30 | 91 | `Ow` | synth | 3 | 是 |
| 31 | 92 | `Ow🎵` | synth | 17 |  |
| 32 | 94 | `🍾` | other | 16 |  |
| 33 | 101 | `Mew` | other | 3 | 是 |
| 34 | unknown | — | unknown | 0 | 合成类，无样本 |

样本数 <5 的类：**12 个** → 7(`Hb`)=2, 13(`Tch`)=1, 25(`` K` ``)=0, 38(`Peh`)=1, 43(`d`)=1, 57(`Rr`)=3, 60(`w🗣️P0`)=1, 65(`Z`)=4, 74(`Drhb`)=1, 85(`Zip`)=1, 91(`Ow`)=3, 101(`Mew`)=3

## family 分布（train）

| family | train n | 类数 |
|---|---|---|
| clops | 36 | 2 |
| cymbal | 22 | 1 |
| snares | 22 | 2 |
| liprolls | 21 | 1 |
| hihats | 20 | 3 |
| synth | 20 | 2 |
| vocal | 20 | 2 |
| clicks | 19 | 2 |
| other | 19 | 2 |
| zippers | 19 | 2 |
| drag | 18 | 2 |
| scratch | 17 | 2 |
| kicks | 16 | 2 |
| lip_oscillation | 16 | 4 |
| reverse | 13 | 1 |
| teeth | 10 | 1 |
| whistles | 10 | 1 |
| tongue | 1 | 1 |

## 标签置信度怎么读（别读错）

`label_source` 全为 `seed_match`：sound_id 由记法 crosswalk 从 seedlib 候选映射而来，`crosswalk_score` 全 1.0，匹配方式分布：`exact_notation` 209、`exact_notation_stripped_lt` 92、`notation_ambiguous+name` 18。

**`sound_confidence` 全 0.0 不是低置信度**——种子行没有模型打分，0.0 是「无值」占位。下游若把 0.0 当置信度阈值过滤，会把 319 行全砍掉。要过滤请用 `crosswalk_score`。

`human_notation` 仍全空（agent.md §10.1 #5 未关闭）；本索引**不含任何人耳确认标签**，因此 ast_lin_v0 的分数只能用于校准评测仪器，不能反过来当「音色标注正确」的证据。

## 口径与边界

- 本索引只服务 `ast_lin_v0` **仪器校准卡**；不得据此宣称任何 103 音色识别能力。
- `low_sample_lt5=1` 的类在训练时必须加权或单独报数，出卡时按类给 n，禁止只报汇总准确率。
- 音频不入 git、不外传；本 CSV 只含路径与哈希，不含音频。

## 校验（可复跑）

```
python scripts/build_baseline_train_index.py
# 期望：rows=319, vocab=34, overlap=34, leakage=0
```
