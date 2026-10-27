# 探针就绪卡 · 识别侧 seedlib-consistency probe（Eval，2026-10-27）

> 本卡只报**资产与口径就绪度**。真正的能力卡要等 103 类预测文件（preds.csv），
> 当前没有任何可用识别器，故**不出能力结论、不写 agent.md §0**。

## 1. 通道与资产

- `lf_remote.py test` 退出码 0（LF_ALIVE），GPU 段空（无训练进程）。
- 测试集 `data/eval_refaudio_testset.csv`：95 行 / 95 唯一 sound_id / 19 family，
  md5=`e668ce45e311cc2873847860240d52a5`；**95/95 参考音频本地可读**（mp3 全部存在）。
- 口径：评分单元 sound_id（0-based 0..102）；随机基线 1/95=1.05%；通过线 top-1 ≥5×随机
  且 family 非零命中 ≥15/19；n<5 family 只报数；本卡任何数字**不得写入 §0 识别准确率**。

## 2. G5 口径已锁定并落进脚本（不是聊天记录口径）

`scripts/eval_score_probe.py` 新增 G5 段，出卡自动包含：

| 项 | 内容 |
|---|---|
| 三栏分层 | 基型组(n=89) / 短变体组(n=6) / 全体(n=95)，各报 top-1 + Wilson 区间 + notation pair-aware |
| 短变体逐条 | 6 行：真值/预测/结果（命中 / 互混→基型 / 其他错），n=1/id 只报数 |
| 成对混淆 | **5 对**逐对列出（不折算均值）：短→基、基→短、双命中、双混、其他错 |
| 无基型孤儿 | id 65 `Z` Vibrator Lip Bass (Short)：taxonomy/crosswalk **均无基型**，无法成对，单列报数 |
| 判定边界 | G5 **不参与**通过/不通过；短变体组只以「命中 x/6」给方向性证据 |

锁定的 5 对（`SHORT_PAIRS`，带自检断言：短变体标 1 / 基型标 0 / 两者都在测试集 / 记法不同）：

| # | 短变体 | 基型 |
|---|---|---|
| 1 | 25 `K'`（short） | 23 `K'` Inward K Rimshot |
| 2 | 2 `Pw` Fat Kick Short | 3 `Pww` Fat Kick Long（种子库无中性 Fat Kick，按短/长配对）|
| 3 | 10 `T'` Hi-Hat T (short) | 9 `T` Hi-hat T |
| 4 | 19 `k'` K palate snare (short) | 18 `k` K palate snare |
| 5 | 70 `Ww*` Inward Hollow Lip Roll (Short) | 71 `Ww*~` ... (Longer) |

**边界确认**：`is_short_variant` 只进分层统计，不进标签空间（仍 103 类）。

验证：`--g5-unit-test` 用手工构造的互混预测断言分层计数 / 6 行逐条 / 5 对成对结果 / Z 孤儿，
**PASS**；`--self-test` 全链路出卡含 G5 段，PASS（合成预测，不代表任何模型能力）。

## 3. 扰动音频包已落地（敏感性一票否决的前提）

`scripts/eval_make_perturbations.py` → `data/eval_perturb_audio/` + `data/eval_perturb_manifest.csv`

- 95 refs × 12 变体 = 1140 行；**1045 个 wav 已生成**，全部参数回测达标
  （SNR / 时长比 / 静音垫 / 增益按 TOL 对账）。
- **确定性**：连续两次生成 manifest md5 全同 = `85334ae7463c`，1045 个 wav md5 全同。
- 磁盘回读校验（`--verify`）：音频 md5 全同；仅 `snr_db_meas`（max Δ0.48dB，PCM_16 量化）
  与 `clipped_samples` 有量化级差异，均在 TOL 内。
- P1/P2/P4/P7 展开成 a/b 子变体；scorer 已支持 `P1a→P1` 的 kind 归并。

**P5 seed_swap 不可用（置信度事实）**：探针集每 sound_id 仅 1 条 rendition，
没有“同音色另一 source 的同类片段”。用别的音色冒充 swap 会改真值 = 伪造测试集，故不生成，
manifest 记 `available=0`。**需要 @Data 提供同 id 第二条 rendition 才能补 P5**，
在此之前敏感性结论必须注明「P5 缺项」。

## 4. 生成侧指标链路已校准

`scripts/eval_gen_audio.py` 自检（合成三对：identical / 高频损失 / 10ms 延迟）：

- 零误差锚点：identical → mel 0.0 / env 1.0 / flux 1.0 / onset 0.0 ✓
- 高频损失被 hf_ratio（0.823→0.327）与 mel（1.256）双重抓到 ✓
- **修掉 onset 指标系统性低估**：旧实现按帧（hop=64=2.67ms）取整且偏 1 帧，
  真值 5.33/10/20/40ms → 测得 2.67/5.33/16/37.33ms。改为样本级 + 阈值交点线性插值后，
  校准误差 **worst 0.76ms**（0/1/2.67/5.33/10/20/40ms 全点位）。
- librosa 本地已装好（首次安装遇 pypi 超时重试后成功）；**远端训练机无 librosa**，
  生成侧评分若要在远端跑，需先装或改用本地跑（音频本地在 `data/beatbox_seeds/audio`）。

## 5. 阻塞项（能力卡开跑前必须解决）

1. **无识别器**：本地 `checkpoints/` 只有 4 类旧模型（beatbox_cls*，已弃用），
   远端 `checkpoints/` 也只有 beatbox_cls_robust / v2 / beatbox_feat / gen_routeb_smoke / omni3b-edm。
   103 类预测（preds.csv，列 `ref_id,pred_sound_id[,pred_sound_id_top3]`）目前**无人产出**。
   → 交给 @Talker（Thinker 文本头）或以伪标 hit 训练的基线；拿到 preds 我 1 小时内出能力卡。
2. **P5 缺项**（见 §3），敏感性结论带保留。
3. **试听包**：出卡时附试听包路径；识别侧试听包 = 探针音频 + 扰动样本 + 失败案例 wav，
   等真预测到位后由我打包。

## 6. 本次改动清单（同步待 @PI 拍板）

- 改：`scripts/eval_score_probe.py`（G5 段 + 配对自检 + `--g5-unit-test` + P 子变体 kind 归并）
- 改：`scripts/eval_gen_audio.py`（onset 指标修正 + 校准）
- 新增：`scripts/eval_make_perturbations.py`
- 新增：`data/eval_perturb_manifest.csv`（可入库）；`data/eval_perturb_audio/*.wav` 不入 git（*.wav 已 ignore）
