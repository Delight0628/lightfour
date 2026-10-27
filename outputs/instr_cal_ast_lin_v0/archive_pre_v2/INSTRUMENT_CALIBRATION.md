# INSTRUMENT_CALIBRATION · 仪器校准卡（ast_lin_v0）

> **卡头自检**：本卡标题即 `INSTRUMENT_CALIBRATION`。它是**评测链路的仪器校准卡**，不是 103 音色识别能力卡，**不得出现「103 音色识别能力」措辞**，**任何数字不得写入 agent.md §0 识别准确率**。

## A0 元数据

| 字段 | 值 |
|---|---|
| 实验 ID / 日期 | ast_lin_v0_instr_cal_2026-10-27 / 2026-10-27 |
| 预测源 | `ast_lin_v0`（冻结 AST + 线性探针，**非 Omni**） |
| 骨干 | `MIT/ast-finetuned-audioset-10-10-0.4593` rev `f826b80d`（冻结，可训练参数 0） |
| 探针头 | StandardScaler（只 fit 训练集）+ LogisticRegression(lbfgs/L2/C=1.0/无超参搜索) |
| 训练索引 | `data/baseline_train_index.csv` md5=`8186ab5d44d47cf621e584bce9486b87`  319 行 |
| 标签空间 | `data/baseline_label_space.csv` md5=`8a5788fb3ae0f3855b8e66a2793a191a`  34 类 + unknown |
| 测试集 | `data/eval_refaudio_testset.csv` md5=`e668ce45e311cc2873847860240d52a5`  95 行 |
| 预测文件 | `E:\lightfour\outputs\instr_cal_ast_lin_v0\preds.csv` md5=`efc821bfe15b2a936d641cd4263e9c78`  95 行 |
| 扰动预测 | `E:\lightfour\outputs\instr_cal_ast_lin_v0\perturb_preds.csv` md5=`cb0f76d86cebecc37df90920d77713d9`  1045 行 |
| 试听包路径 | `E:\lightfour\outputs\instr_cal_ast_lin_v0\listen` |
| 样本数 / 音色数 | in-vocab **34** 条 / 34 个 sound_id；out_of_vocab 61 条（不计分） |

## A1 主指标（**只报 in-vocab n=34**，sound_id 为评分单元）

| 指标 | 命中/样本 | 值 | 95% Wilson CI | 随机基线 | 5× 线 |
|---|---|---|---|---|---|
| sound_id top-1 | 5/34 | 0.1471 | [0.0645, 0.3013] | 34-way 0.0294 | 0.1471 |
| sound_id top-3 | 12/34 | 0.3529 | [0.2149, 0.5209] | — | — |
| notation raw（仅参考） | 5/34 | 0.1471 | [0.0645, 0.3013] | — | — |

- 随机基线口径（PI 锁定）：34-way = 1/34 = **2.94%**；5× 线 = **14.71%**；95-way 1.05% 不适用于本卡。
- ⚠️ **5× 线在本卡退化**：5×(1/34) = 5/34 与「答对 5 条」在数值上恒等，所以 top-1 恰好 5/34 时必然「压线」。该线在 34-way 下**没有判别力**，不能当作「通过/不通过」依据；有意义的读数是 top-3、family 覆盖与扰动敏感性。
- 实际可学标签数 = **33** / 35 个 label_index（`label_index 5 = sid 25` 与 `label_index 34 = unknown` 训练样本 0，模型永远输出不了这两格）。
- 剔除「不可学」sid 25 后的参考值：top-1 5/33（**仅参考，主指标仍是上面的 n=34**）。

## A1b out_of_vocab 单列（61 条，**不计分**）

> 这 61 条的真值 sound_id 根本不在 ast_lin_v0 的 34 类标签空间里，模型**结构上不可能答对**。下面只做诊断统计，**一律不进主指标、不判通过/不通过**。

- 被迫在 34 类里选一个：100%（模型没有 unknown 拒识能力，`unknown` 格训练样本 0）
- 真值恰好落进 34 类的条数：0/61（理论恒为 0，此处为 0 说明映射一致）
- out_of_vocab 分 family：{'clicks': 6, 'clops': 1, 'cymbal': 1, 'hihats': 5, 'kicks': 7, 'lip_oscillation': 3, 'liprolls': 6, 'nasal': 2, 'other': 7, 'reverse': 1, 'snares': 19, 'teeth': 1, 'tongue': 1, 'whistles': 1}

<details><summary>out_of_vocab 61 条逐条（点开）</summary>

| ref_id | 真值 sound_id | 真值记法 | family | 预测 sound_id | 预测记法 | prob |
|---|---|---|---|---|---|---|
| 025_kick | 0 | P | kicks | 51 | L | 0.517 |
| 033_inward_kick | 1 | P | kicks | 51 | L | 0.637 |
| 013_fat_kick_short | 2 | Pw | kicks | 54 | O | 0.389 |
| 006_fat_kick_long | 3 | Pww | kicks | 54 | O | 0.904 |
| 034_kick_w_chest_bass | 4 | Phrm | kicks | 51 | L | 0.291 |
| 003_slap_kick | 5 | Lp👅 | kicks | 51 | L | 0.434 |
| 061_throat_kick_with_nasal_resonance | 8 | Hm | kicks | 65 | Z | 0.566 |
| 024_hi_hat_t | 9 | T | hihats | 53 | CL* | 0.645 |
| 083_hi_hat_t_short_for_fast_patterns | 10 | T' | hihats | 53 | CL* | 0.683 |
| 066_hi_hat_t_humming | 14 | t | hihats | 12 | F | 0.204 |
| 021_hi_hat_f_humming | 15 | f | hihats | 90 | Chῤ | 0.827 |
| 028_kiss_hi_hat | 16 | -b | hihats | 6 | D | 0.926 |
| 039_pf_snare | 17 | PF | snares | 86 | Zip~ | 0.908 |
| 040_k_palate_snare | 18 | k | snares | 6 | D | 0.669 |
| 064_k_palate_snare_short_for_fast_patterns | 19 | k' | snares | 6 | D | 0.705 |
| 044_inward_k_snare | 20 | K | snares | 84 | A | 0.455 |
| 011_inward_k_snare_closing_release | 21 | Kb | snares | 53 | CL* | 0.454 |
| 042_double_inward_k_snare | 22 | K² | snares | 86 | Zip~ | 0.781 |
| 027_inward_k_rimshot | 23 | K' | snares | 86 | Zip~ | 0.775 |
| 001_double_inward_k_rimshot | 24 | K'² | snares | 86 | Zip~ | 0.968 |
| 041_bongo_drum | 26 | Km | snares | 54 | O | 0.432 |
| 094_inward_k_tonal_rimshot | 27 | Koo | snares | 84 | A | 0.546 |
| 060_inward_kch_snare | 28 | Kch | snares | 90 | Chῤ | 0.413 |
| 014_sub_bass_snare | 29 | Kw | snares | 51 | L | 0.570 |
| 101_inward_k_snare_click_roll | 30 | Krr | snares | 86 | Zip~ | 0.404 |
| 031_spit_snare | 31 | s | snares | 86 | Zip~ | 0.408 |
| 008_double_spit_snare | 32 | s² | snares | 37 | M | 0.575 |
| 022_outward_k_snare | 33 | K | snares | 86 | Zip~ | 0.408 |
| 026_outward_kch_snare_vocalized | 34 | Kch | snares | 41 | Pss | 0.953 |
| 067_dsh_snare | 35 | Dsh | snares | 86 | Zip~ | 0.941 |
| 051_duf_snare | 36 | Df | snares | 54 | O | 0.924 |
| 077_whistling_crash | 42 | Pss° | cymbal | 92 | Ow🎵 | 0.282 |
| 012_click_do_phonem | 45 | do | clicks | 6 | D | 0.363 |
| 002_hollow_click | 46 | -d* | clicks | 53 | CL* | 0.453 |
| 080_uvula_click | 47 | G* | clicks | 53 | CL* | 0.334 |
| 071_uvula_click_ga_phonem | 48 | G*a | clicks | 53 | CL* | 0.783 |
| 092_uvula_click_go_phonem | 49 | G*o | clicks | 53 | CL* | 0.607 |
| 043_tongue_clap | 50 | cl | clicks | 90 | Chῤ | 0.440 |
| 073_hollow_clop | 52 | L* | clops | 54 | O | 0.877 |
| 046_lip_oscillation_classic | 58 | w | lip_oscillation | 54 | O | 0.945 |
| 097_vocalized_lip_oscillation_perfect_fifth | 61 | w🗣️5 | lip_oscillation | 54 | O | 0.974 |
| 079_fart_bass | 63 | Fb~ | lip_oscillation | 74 | Drhb | 0.230 |
| 038_hollow_lip_roll | 67 | W* | liprolls | 54 | O | 0.966 |
| 029_inward_lip_roll | 68 | Ww | liprolls | 54 | O | 0.974 |
| 095_inward_sub_lip_roll | 69 | Ww~ | liprolls | 79 | Hw | 0.402 |
| 049_inward_hollow_lip_roll_short | 70 | Ww* | liprolls | 54 | O | 0.943 |
| 070_inward_hollow_lip_roll_longer | 71 | Ww*~ | liprolls | 54 | O | 0.882 |
| 050_inward_lip_roll_with_tongue | 72 | Wwd | liprolls | 54 | O | 0.975 |
| 081_inward_click_roll | 73 | Drh | tongue | 87 | Juy | 0.294 |
| 096_nasal_bass | 76 | Rhm | nasal | 54 | O | 0.971 |
| 020_snore_bass | 77 | Rhm | nasal | 54 | O | 0.903 |
| 078_inward_click_roll_whistle | 78 | Drh🦗 | whistles | 41 | Pss | 0.727 |
| 082_tooth_whistle | 81 | 🦷° | teeth | 12 | F | 0.495 |
| 030_reverse_kick | 89 | ῤ | reverse | 51 | L | 0.305 |
| 058_dry_kick | 93 | D* | other | 6 | D | 0.824 |
| 048_sucker_punch | 95 | HD | other | 51 | L | 0.279 |
| 074_inward_hd | 96 | Hd | other | 51 | L | 0.956 |
| 075_inward_df | 97 | Df | other | 54 | O | 0.180 |
| 047_inward_dp | 98 | Dp | other | 51 | L | 0.504 |
| 045_inward_fd | 99 | <Fd | other | 53 | CL* | 0.352 |
| 055_tight_tongue_roll | 100 | Dr' | other | 54 | O | 0.574 |

</details>

## A2 分 family（in-vocab n=34；n<5 只报数不判通过）

| family | n | correct | acc | 备注 |
|---|---|---|---|---|
| clicks | 2 | 0 | 0.0000 | n<5，只报数 |
| clops | 2 | 0 | 0.0000 | n<5，只报数 |
| cymbal | 1 | 0 | 0.0000 | n<5，只报数 |
| drag | 2 | 1 | 0.5000 | n<5，只报数 |
| hihats | 3 | 1 | 0.3333 | n<5，只报数 |
| kicks | 2 | 1 | 0.5000 | n<5，只报数 |
| lip_oscillation | 4 | 0 | 0.0000 | n<5，只报数 |
| liprolls | 1 | 0 | 0.0000 | n<5，只报数 |
| other | 2 | 0 | 0.0000 | n<5，只报数 |
| reverse | 1 | 0 | 0.0000 | n<5，只报数 |
| scratch | 2 | 0 | 0.0000 | n<5，只报数 |
| snares | 3 | 1 | 0.3333 | n<5，只报数 |
| synth | 2 | 0 | 0.0000 | n<5，只报数 |
| teeth | 1 | 0 | 0.0000 | n<5，只报数 |
| tongue | 1 | 0 | 0.0000 | n<5，只报数 |
| vocal | 2 | 1 | 0.5000 | n<5，只报数 |
| whistles | 1 | 0 | 0.0000 | n<5，只报数 |
| zippers | 2 | 0 | 0.0000 | n<5，只报数 |

family 非零命中: **5/18**

- ⚠️ in-vocab 34 条分到 18 个 family，其中 **18/18 个 n<5**（即全部）；按红线这些 family **一律只报数、不判通过**。标准探针卡的「family 非零命中 ≥15/19」通过线**不适用于本卡**（本卡不是 19-family 全量探针）。

## A3 混淆（in-vocab，按 sound_id）

| 真值 sound_id | 预测 sound_id | 次数 |
|---|---|---|
| 25 Inward K Rimshot (short) | 53 Double Hollow Clop | 1 |
| 7 Throat Kick | 54 Throat Bass | 1 |
| 86 Inward Zipper Roll | 92 Synth (high pitched) | 1 |
| 53 Double Hollow Clop | 54 Throat Bass | 1 |
| 12 Hi-hat F | 51 Clop | 1 |
| 94 Pop sound | 44 Click (da phonem) | 1 |
| 66 Lip Roll | 54 Throat Bass | 1 |
| 44 Click (da phonem) | 53 Double Hollow Clop | 1 |
| 83 Outward Drag | 41 Crash | 1 |
| 91 Synth | 54 Throat Bass | 1 |
| 92 Synth (high pitched) | 54 Throat Bass | 1 |
| 13 Tch Hi-hat | 53 Double Hollow Clop | 1 |
| 62 Vocalized Lip Oscillation (Perfect Octave) | 54 Throat Bass | 1 |
| 51 Clop | 54 Throat Bass | 1 |
| 60 Vocalized Lip Oscillation (Unisson) | 79 Hollow Whistle | 1 |
| 79 Hollow Whistle | 66 Lip Roll | 1 |
| 87 Zede Scratch | 53 Double Hollow Clop | 1 |
| 90 Reverse Snare | 41 Crash | 1 |
| 82 Inward Vocalized Tooth Whistle | 86 Inward Zipper Roll | 1 |
| 65 Vibrator Lip Bass (Short) | 91 Synth | 1 |
| 101 Mew Squeak | 86 Inward Zipper Roll | 1 |
| 74 Inward Click Roll (Closing Release) | 51 Clop | 1 |
| 38 Peh Snare | 83 Outward Drag | 1 |
| 57 Uvula Bass | 54 Throat Bass | 1 |
| 43 Click | 53 Double Hollow Clop | 1 |
| 85 Zipper | 86 Inward Zipper Roll | 1 |
| 59 Whistling Lip Oscillation | 41 Crash | 1 |
| 41 Crash | 11 Hi-Hat (Opened) | 1 |
| 88 Shuy Air Scratch | 37 Msh Snare | 1 |

<details><summary>family 块状混淆矩阵（行为真值 family，列为预测 family）</summary>

| true\pred | clicks | clops | cymbal | drag | hihats | kicks | lip_oscillation | liprolls | other | reverse | scratch | snares | synth | teeth | tongue | vocal | whistles | zippers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clicks | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clops | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| cymbal | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| drag | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hihats | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kicks | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| lip_oscillation | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| liprolls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| other | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| reverse | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scratch | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| snares | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| synth | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| teeth | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| tongue | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vocal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| whistles | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zippers | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |

</details>

## A4 G5 短变体 / 基型分层（三栏 + 5 对混淆，只报数，不参与通过判定）

> G5 口径锁定 v1（Eval）：`is_short_variant` 只进分层统计，不进标签空间。本卡标签空间仅 34 类，锁定的 5 对里只有 **25↔23** 的短变体侧落在空间内（基型 23 是 out_of_vocab），其余 4 对双侧都不在空间内，只能按 out_of_vocab 报数；id 65 `Z` 无基型孤儿。

| 组 | n | top-1 | 95% Wilson CI |
|---|---|---|---|
| 基型组 | 32 | 5/32 = 0.1562 | [0.0686, 0.3175] |
| 短变体组 | 2 | 0/2 = 0.0000 | [0.0000, 0.6576] |
| 全体 | 34 | 5/34 = 0.1471 | [0.0645, 0.3013] |

### 5 对成对混淆（逐对列出，不折算均值）

| # | 短变体 | 基型 | 短在空间 | 基在空间 | 短样本→预测 | 基样本→预测 | 结果 |
|---|---|---|---|---|---|---|---|
| 1 | 25 K` | 23 K' | ✓ | ✗ | 53 | 86 | 短变体在标签空间内，基型 out_of_vocab → 无法判「互混→基型」 |
| 2 | 2 Pw | 3 Pww | ✗ | ✗ | 54 | 54 | 双侧 out_of_vocab，不计 |
| 3 | 10 T' | 9 T | ✗ | ✗ | 53 | 53 | 双侧 out_of_vocab，不计 |
| 4 | 19 k' | 18 k | ✗ | ✗ | 6 | 6 | 双侧 out_of_vocab，不计 |
| 5 | 70 Ww* | 71 Ww*~ | ✗ | ✗ | 54 | 54 | 双侧 out_of_vocab，不计 |

- 无基型孤儿：id 65 —— Z Vibrator Lip Bass (Short)：taxonomy/crosswalk 均无 Vibrator Lip Bass 基型，无法成对。本卡内预测 = 91，单列报数，不计混淆对。
- 置信度：短变体组 n=2（sid 25 / sid 65），Wilson 区间仅具形式意义；G5 **不参与**通过/不通过判定。

## A5 音频敏感性（扰动 P0–P7，一票否决项；只在 in-vocab n=34 上算）

| perturb_id | kind | n | acc | 与 P0 预测翻转率 |
|---|---|---|---|---|
| P0 | none | 34 | 0.1765 | 0.0000 |
| P1a | time_stretch | 34 | 0.2059 | 0.4118 |
| P1b | time_stretch | 34 | 0.2059 | 0.3235 |
| P2a | time_stretch | 34 | 0.2647 | 0.5000 |
| P2b | time_stretch | 34 | 0.2353 | 0.4118 |
| P3 | additive_noise | 34 | 0.2059 | 0.5588 |
| P4a | additive_noise | 34 | 0.1765 | 0.7353 |
| P4b | additive_noise | 34 | 0.0882 | 0.9118 |
| P6 | silence_pad | 34 | 0.1176 | 0.6176 |
| P7a | gain | 34 | 0.2353 | 0.2059 |
| P7b | gain | 34 | 0.2059 | 0.2647 |

- 全部扰动（含 P0）预测完全不变的样本占比: **0.0588**（阈值 ≥0.99 → 一票否决）
- 敏感性判定: **敏感（预测随音频变化）**
- **P0 与原始 mp3 预测一致率**：33/34（P0 是扰动管线从 mp3 重编码出的 wav，一致率低说明扰动管线本身引入了偏移，敏感性结论须按此打折）
- **P5 seed_swap 缺项**：探针集每 sound_id 仅 1 条 rendition，无同音色另一 source 的同类片段，故 P5 `available=0` 未生成。敏感性结论永久带「P5 缺项」保留；补齐需 @Data 提供同 sound_id 第二条 rendition。

## A6 失败案例清单（28/34，已剔除「不可学」sid 25）

| ref_id | 真值 | 预测 | family | 真值记法 | 预测记法 | prob |
|---|---|---|---|---|---|---|
| 088_uvula_bass | 57 Uvula Bass | 54 Throat Bass | vocal | Rr | O | 0.995 |
| 056_vocalized_lip_oscillation_perfect_octave | 62 Vocalized Lip Oscillation (Perfect Octave) | 54 Throat Bass | lip_oscillation | w🗣️P8 | O | 0.987 |
| 089_click | 43 Click | 53 Double Hollow Clop | clicks | d | CL* | 0.981 |
| 072_inward_vocalized_tooth_whistle | 82 Inward Vocalized Tooth Whistle | 86 Inward Zipper Roll | teeth | 🦷🗣️° | Zip~ | 0.883 |
| 103_shuy_air_scratch | 88 Shuy Air Scratch | 37 Msh Snare | scratch | Shuy | M | 0.860 |
| 032_click_da_phonem | 44 Click (da phonem) | 53 Double Hollow Clop | clicks | da | CL* | 0.818 |
| 085_inward_click_roll_closing_release | 74 Inward Click Roll (Closing Release) | 51 Clop | tongue | Drhb | L | 0.758 |
| 023_lip_roll | 66 Lip Roll | 54 Throat Bass | liprolls | W | O | 0.735 |
| 084_mew_squeak | 101 Mew Squeak | 86 Inward Zipper Roll | other | Mew | Zip~ | 0.722 |
| 100_whistling_lip_oscillation | 59 Whistling Lip Oscillation | 41 Crash | lip_oscillation | w° | Pss | 0.663 |
| 037_synth | 91 Synth | 54 Throat Bass | synth | Ow | O | 0.651 |
| 068_reverse_snare | 90 Reverse Snare | 41 Crash | reverse | Chῤ | Pss | 0.595 |
| 007_inward_zipper_roll | 86 Inward Zipper Roll | 92 Synth (high pitched) | zippers | Zip~ | Ow🎵 | 0.592 |
| 054_tch_hi_hat | 13 Tch Hi-hat | 53 Double Hollow Clop | hihats | Tch | CL* | 0.585 |
| 052_synth_high_pitched | 92 Synth (high pitched) | 54 Throat Bass | synth | Ow🎵 | O | 0.583 |
| 063_hollow_whistle | 79 Hollow Whistle | 66 Lip Roll | whistles | Hw | W | 0.567 |
| 035_outward_drag | 83 Outward Drag | 41 Crash | drag | A | Pss | 0.458 |
| 005_throat_kick | 7 Throat Kick | 54 Throat Bass | kicks | Hb | O | 0.443 |
| 099_zipper | 85 Zipper | 86 Inward Zipper Roll | zippers | Zip | Zip~ | 0.428 |
| 102_crash | 41 Crash | 11 Hi-Hat (Opened) | cymbal | Pss | Tss | 0.425 |
| 009_double_hollow_clop | 53 Double Hollow Clop | 54 Throat Bass | clops | CL* | O | 0.405 |
| 065_zede_scratch | 87 Zede Scratch | 53 Double Hollow Clop | scratch | Juy | CL* | 0.400 |
| 019_pop_sound | 94 Pop sound | 44 Click (da phonem) | other | 🍾 | da | 0.387 |
| 057_clop | 51 Clop | 54 Throat Bass | clops | L | O | 0.375 |
| 015_hi_hat_f | 12 Hi-hat F | 51 Clop | hihats | F | L | 0.348 |
| 076_vibrator_lip_bass_short | 65 Vibrator Lip Bass (Short) | 91 Synth | lip_oscillation | Z | Ow | 0.287 |
| 086_peh_snare | 38 Peh Snare | 83 Outward Drag | snares | Peh | A | 0.264 |
| 059_vocalized_lip_oscillation_unisson | 60 Vocalized Lip Oscillation (Unisson) | 79 Hollow Whistle | lip_oscillation | w🗣️P0 | Hw | 0.234 |

- **该类不可学**：ref `004_inward_k_rimshot_shorter_for_fast_patterns` 真值 sid **25 `K``**（K`（Inward K Rimshot 短变体）），训练样本 **0**（`label_index 5` 无任何训练行）→ 模型结构上不可能答对，**不计入失败案例**。预测 = 53（仅供诊断）。

## A6b 试听包

- 路径：`E:\lightfour\outputs\instr_cal_ast_lin_v0\listen`，条目 **497**
- 结构：`probe_in_vocab/` 34 条（文件名前缀=真值 sound_id）；`probe_out_of_vocab/` 61 条（不计分）；`perturb_in_vocab/` 374 条（P5 缺项）；`failures/` 28 条（文件名=`truth<真值>_pred<预测>_<ref_id>`，**不含 sid 25**）
- `MANIFEST.csv` 含每条的 kind/ref_id/文件名/字节数/md5；`README.md` 说明口径。
- 音频不入 git、不外传。

## A7 结论

### 本卡验收（评测链路是否可用）

- ✅ 卡头为 INSTRUMENT_CALIBRATION
- ✅ 主指标只报 in-vocab n=34 + Wilson
- ✅ out_of_vocab 61 条单列且不计分
- ✅ sid 25 标注「该类不可学」且不计入失败案例
- ✅ G5 三栏 + 5 对混淆已出
- ✅ 扰动敏感性已出
- ✅ 失败清单已出
- ✅ 试听包已出

**评测链路结论：通过**——卡片字段齐全，preds/测试集/标签空间 md5 自证可复现，链路可交付给 @Talker 产出真实预测。

### 仪器读数（不是能力结论）

- ast_lin_v0 在 in-vocab n=34 上 top-1 = **5/34 = 0.1471**（95% Wilson [0.0645, 0.3013]）；top-3 = **12/34 = 0.3529**（[0.2149, 0.5209]）。
- 5× 随机基线 = 0.1471，本读数压线；但如 A1 所述，该线在 34-way 下与「答对 5 条」恒等，**不具判别力，不作为通过条件**。
- 该读数的含义仅是「**冻结 AST + 线性探针在这套评测链路上的地板值**」；它既不是 103 音色识别能力，也不是 Thinker 文本头的能力上限。
- **不得**把本卡任何数字写入 agent.md §0 识别准确率；§0 只能由 `human_notation` ≥15 family 后的人工标注集出具。

### 置信度与已知混淆变量

- **域差异（最大混淆变量）**：训练 = B 站实战切片 22050Hz wav（70–390ms，Demucs 人声 + onset 切片），评测 = seedlib 参考音频 mp3（64–776ms，录音棚/单条示范）。两者录音链路、混响、底噪完全不同，绝对准确率偏低**不能**直接解读成「模型不听音频」或「评测工具坏了」。
- **时长混淆**：短切片被 pad 到 1024 帧（10s）输入 AST，pad 占比 92–99%；训练/评测两侧时长分布不同会让探针部分依赖时长线索。
- **样本倾斜**：33 个可学类里 12 类 n<5（含 1 类 n=0）；319 行只覆盖 33/103 音色、53 个 source，且全部 `split=train`（无独立训练内验证集，本卡也不需要）。
- **标签噪声**：训练标签来自 `beatbox_seed_crosswalk.csv` 记法匹配（exact_notation 209 / stripped_lt 92 / ambiguous+name 18），`sound_confidence=0.0` 是「无模型打分」占位值，不是低置信度。
- **P5 缺项**：敏感性结论带保留（见 A5）。

## C 通用红线自检

1. ✅ 未只报汇总均值——A2 分 family + 样本数在。
2. ✅ 未跨 codec 版本比较（本卡纯文本/标签层面，不涉 codec 数字）。
3. ✅ 未用训练集重建冒充泛化——测试集是 `eval_refaudio_testset.csv` 固定版 + md5。
4. ✅ n<5 分组只报数、不下结论。
5. ✅ 探针分未写入 §0。
