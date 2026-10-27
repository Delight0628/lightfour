# codec 帧率判定：75Hz 直通 vs 13Hz 视图（§10.1 #9(a)）

> 产出：2026-10-27 @Codec ｜ 脚本 `scripts/codec_rate_ab.py` ｜ 产物 `outputs/codec_rate_ab/`
> 同一批音频、同一个 codec、同一 mel 实现，**唯一变量 = 帧率视图**。
> 实验基数按 PI 裁决用**全量 287 条**序列（不是本地 `audio_available=1` 过滤后的 152 条）。

## 1. 结论

1. **13Hz 视图作废**。3100 条 60–200ms 单击段，mel 退化中位 **155.4%**（判定线 10%，差 15.5 倍）；3100/3100 全部更差。即使给 13Hz 最有利的口径（增益对齐 + 择优选帧）也救不回来。
2. **Hi-hats / Cymbal 单列（此前抽样未覆盖）**：16 层 EnCodec 自身重建 mel 中位 hihats **0.843**、cymbal **0.748**。**hihats 越过 §8 Gate A 的「mel 中位 <0.8」线 → codec 不能对镲类宣布过关**。32 层能把 hihats 拉回 0.771（见 `docs/codec_layers_family.md`），代价是 Talker 扩 31 套 predictor 表。
3. **`codec_tokens.py` 的 padding_mask 截断 bug 已闭环**：`decode_codes()` 增加了「输出长度必须 ≈ 帧数×320」的硬断言（再传帧级 mask 会直接抛错，而不是静默出 0.001s 音频）。40 条真实切片重测：解码时长 **40/40 与真值一致（<20ms）**，mel 中位 **0.715**。历史「0.661」作废；「0.752」是另一批音频 + 另一套 mel 实现，**不与之比较**。
4. **`position_id_per_seconds` 不是 codec 帧率参数**。transformers 5.16.1 的 `get_rope_index` 只在 **image / video** 两个分支读它（`modeling_qwen3_omni_moe.py` L299 取值、L376 图像、L393/L416 视频），audio 与 codec token 走普通顺序位置。所以改 75 影响的是**视觉** mrope，对音频零影响。75Hz 的代价 100% 是序列长度（8s = 600 token）。详见 `docs/talker_codec_changes.md`。

## 2. 方法

- 数据：**287 条** 3–8s 序列 / 80 source / **5472 个击打段**（60–200ms **3100**、hihats **247**、cymbal **38**）。
- 不重新编码：直接用已交付的 `(1,1,16,T)` EnCodec codes（75Hz），避免编码批次差。
- 三条臂：
  - `75`：整段 codes 直通解码（**codec 上限参照**）；
  - `13c`：每 76.9ms 取**中心帧**，hold-repeat 回 75Hz 再解码（确定性策略）；
  - `13p`：每 76.9ms 取**能量最大帧**，hold-repeat（oracle 选帧，对 13Hz 最有利）。
- hold-repeat 的理由：Talker 输出 13Hz token，codec 解码器仍按 75Hz 跑，一码必须管 77ms —— 这是真实推理时会发生的事，不是人为构造。
- 两套口径：`raw`（不归一化，电平差算误差）+ `gain_matched`（两臂都缩放到真值段 RMS，剥掉重复帧造成的电平增益）。**判定用 gain_matched + 退化更小的臂**，即对 13Hz 最宽松的方向。
- 指标：log-mel 距离（**唯一实现 `scripts/mel_metric.py`**，`n_fft=1024 / hop=256 / n_mels=128 / log(m+1e-6)`，`frame_policy=min` + 帧差断言）、高频能量占比（STFT `n_fft=512 / hop=128`，≥2kHz 能量/总能量）。
- 帧口径（2026-10-27 @Data 实测后改）：`min` 只在**过断言**后截断，四臂 mel 帧数差 > max(1, 2%) 直接 raise；`pad` 仅作敏感性附录 —— 一条臂短 1 帧时 pad 会把 mel 从 p50 0.0087 抬到 0.0245（2.2×），伪差比 L16→L32 的真实改善还大，不能进决策表。每行 `mel_frames`（true|75|13c|13p）+ `mel_frames_max_diff` + `wav_len`/`wav_trunc_ratio` 全部落 `per_hit.csv`。

## 3. 结果

n = 击打段数；mel 为中位；退化 = (e13 − e75)/e75。

| 组 | 臂 | 口径 | n | mel75 | mel13 | 退化中位 | 13 更差占比 |
|---|---|---|---:|---:|---:|---:|---:|
| 60–200ms 单击 | 13p | gain_matched | 3100 | 0.7436 | 1.9014 | **155.4%** | 1.00 |
| 60–200ms 单击 | 13c | gain_matched | 3100 | 0.7436 | 1.9545 | 162.1% | 1.00 |
| 60–200ms 单击 | 13c | raw | 3100 | 0.7438 | 2.1221 | 183.6% | 1.00 |
| **hihats** | 13p | gain_matched | 247 | 0.8544 | 1.9579 | **128.3%** | 1.00 |
| **hihats** | 13c | gain_matched | 247 | 0.8544 | 2.0360 | 145.5% | 1.00 |
| **hihats** | 13c | raw | 247 | 0.8429 | 2.1198 | 153.7% | 1.00 |
| **cymbal** | 13c | gain_matched | 38 | 0.7630 | 1.5612 | 101.6% | 1.00 |
| **cymbal** | 13p | gain_matched | 38 | 0.7630 | 1.6859 | 120.4% | 1.00 |
| 短变体 | 13p | gain_matched | 36 | 0.7818 | 2.0273 | 157.1% | 1.00 |
| 全体 | 13c | gain_matched | 5472 | 0.7393 | 1.8160 | 145.9% | 1.00 |

高频能量占比（真值 → 75Hz 直通 → 13Hz）：

| 组 | hf_true | hf_75 | hf_13c | hf_13p |
|---|---:|---:|---:|---:|
| hihats | 0.9039 | 0.8962 | **0.8087**（−9.8pt） | 0.9097 |
| cymbal | 0.8036 | 0.7891 | 0.7986 | 0.7962 |
| 60–200ms 单击 | 0.6217 | 0.6132 | 0.5851 | 0.6366 |

**关于 hihats 高频**：`13c`（中心帧）掉 9.8 个百分点，`13p`（能量帧）不掉反升。说明高频损失**依赖选帧策略**，不作为结构性结论；两臂的 mel 退化都在 128% 以上，与选帧无关。

分 family 的 mel75（codec 自身上限，gain_matched）：snares 0.756、drag 0.718、clicks 0.721、clops 0.690、zippers 0.718、kicks 0.666、**hihats 0.854**、**cymbal 0.763**。

## 4. 时间分辨率硬约束（与解码无关的算术）

13Hz = 76.9ms/格。同一序列内相邻 onset 间隔（5185 对）：

- 间隔 **< 1 格（76.9ms）占 2.9%**，**< 2 格占 30.6%**
- p05 = 87.0ms，p25 = 137.9ms，中位 = 225.0ms

即：把 token 流放到 13Hz，**近三分之一相邻击打落进 2 格以内**，其中 2.9% 直接合并成一格。这条与「用哪个 codec、模型多强」无关，任何 13Hz tokenizer 都受同一约束。

## 5. 边界（这条结论不覆盖什么）

1. **本实验测的是「EnCodec 75Hz 的 token 流压到 13Hz」**，不是「原生训练的 13Hz codec」。原生 13Hz codec 有自己的解码器，每个 token 生成 77ms 波形，理论上能把一次 100ms 击打渲染得比 hold-repeat 好。本实验**不证明**原生 13Hz codec 不可用，只证明**降采样路线（§7.1c 方案 C）不可用**。
2. 但原生 13Hz 路线同样受第 4 节约束，且 Qwen3-TTS 12.5Hz 已在 60–200ms 击打上实测失败。**结论按「Go 75Hz」执行**，不是「13Hz 物理不可能」。
3. mel 实现已抽成唯一实现 `scripts/mel_metric.py`（`--self-test` + `--reproduce-listen`）。**注意：本页表格里的绝对值出自抽签前的同函数版本，尚未用新脚本整批重跑**，跨脚本比较仍无效；`--reproduce-listen` 在旧试听包上 max|重算−存储| = 0.0585，全部来自 PCM_16 量化（新包已改 FLOAT 写出），不是实现差异。整批重跑后本页数字才可升级为「唯一实现口径」。
4. cymbal n=38（38 击打段 / 28 序列），仍偏小，只作方向性证据，不单独下「通过/不通过」。
5. 16 层 vs 32 层的镲类 A/B 见 `docs/codec_layers_family.md`。

## 6. 复现

```bash
# 自检（无模型依赖）：13Hz 视图帧数/覆盖率/hold-repeat/无损基线
python scripts/codec_rate_ab.py --self-test
# mel 唯一实现自检：冻结口径 / 帧差断言 / pad 抬升方向 / max 帧差
python scripts/mel_metric.py --self-test

# 离线复算试听包选样（不碰音频、不写文件，只打 JSON）——改配额后先跑这个
python scripts/codec_rate_ab.py --plan-listen outputs/codec_rate_ab/per_hit.csv --listen-n 24
# 试听包三臂 mel 复现（对 index.json 逐条核对）
python scripts/mel_metric.py --reproduce-listen outputs/codec_rate_ab/listen

# 全量（服务器，GPU，~12 分钟）
cd /gemini/code/lightfour-train
/root/miniconda3/bin/python scripts/codec_rate_ab.py

# 冒烟（2 条序列，~30 秒）
/root/miniconda3/bin/python scripts/codec_rate_ab.py --limit 2 --listen-n 0 --out-dir /tmp/rate_ab_smoke
```

- 逐击明细：`outputs/codec_rate_ab/per_hit.csv`（5472 行，含两臂两口径全部明细，Eval 可自行重新聚合）
- 试听包：`outputs/codec_rate_ab/listen/`，默认 `--listen-n 24` = **cymbal 12（按 source 去重，每 source ≤1）/ hihats 6 / click_60_200ms 6**，每条 4 臂 `true/75/13c/13p`，wav 以 FLOAT 写出（音频不入 git）。
  **`click_60_200ms` 是时长桶（跨 12 个 family，snares 占 1829/2891），不是 clicks 家族**——旧键名 `click` 已被读成「clicks 家族证据」，改名的原因就在这。实际构成、配额、池规模、来源数与 12+12 条逐条明细（`hit_id / sound_family / sound_notation / sound_id / source / mel13p_vs_75`）都在 `listen/composition.json`。本地现存 `listen/` 仍是旧包（12 条全 hihats、PCM_16），重跑后才会更新。
- 可复现性（**指旧版脚本 `31e1b4e0`，2026-10-27 之前的产物**）：全量跑了两遍，三个产物 md5 完全相同 —— `per_hit.csv 6e0ef1acbc67c738a71e134d3cc965db`、`summary.json 1b6ce1e13edecdf5104184d2da32c0cd`、`summary.md b31f409ed796319e41705766668a76ca`。指标链路无随机性（GPU 解码 + 定值 mel，两次独立进程逐位一致）。**当前脚本已改（mel 走 `mel_metric.py`、新增 `mel_frames_max_diff`/`wav_len` 列、试听包配额与 composition 结构），这三个 md5 不再适用**，重跑后须重新记录并做列级回归等价（按 `(sequence_id,hit_index)` join 5472 行，旧列逐值相等）。旧版本已归档 `backups/2026-10-27_codec_rate_ab_31e1b4e0.py`。
- 另：先在 152 条子集上跑过一版，`raw` 口径方向和量级完全一致（单击 191.9% → 全量 183.6%），扩样未改变结论。
