# 识别侧三臂敏感性表（ast_lin_v0，INSTRUMENT_CALIBRATION 附件）

> 仪器校准材料：说明评测链路读音频，**不是识别能力结论**；任何数字不得写入 agent.md §0。

> 口径：in-vocab n=**34**（sound_id top-1）；`prob_top1` 未校准；shuffle = 音频级 400 采样块置乱（seed 20261027），与门③标签置换无关。


## 三臂主表

| 臂 | n | top-1 命中 | top-1 | 95% Wilson CI | 与 normal 预测翻转数 |
|---|---:|---:|---:|---|---:|
| normal | 34 | 5/34 | 0.1471 | [0.0645, 0.3013] | — |
| silence | 34 | 0/34 | 0.0000 | [0.0000, 0.1015] | 31 |
| shuffle | 34 | 9/34 | 0.2647 | [0.1460, 0.4312] | 25 |

（normal top-3 参考：12/34，非门判据。）


## 门槛判定

- **silence 门**：top-1 0/34 = 0.0000 ≤ 0.05 → **PASS**（门槛：103-way 随机 0.97% 的 5 倍余量；n=34 下命中 ≤1）
- **shuffle 门**：配对 bootstrap（normal-shuffle，B=10000，seed=20261027）差值均值 -0.1176，95% CI **[-0.2654, 0.0294]**，P(mean≤0)=0.9535；McNemar 精确（discordant b=2, c=6）p=0.289062 → 下界 ≤ 0 → **FAIL：不能写「shuffle 显著低于 normal」**

## 解释性补充臂：shuffle_feed（事后定义，不进门槛）

- 读法②「shuffle」= 跨样本错配音频（固定置换 seed=20261027）：top-1 1/34 = 0.0294，Wilson [0.0052, 0.1492]；预测跟随供体音频 5/34。
- 与 normal 配对差值均值 0.1176，95% CI [0.0000, 0.2647]；McNemar（b=5, c=1）p=0.218750 → 未过「显著低于 normal」。
- 25ms 注册臂的块数分布：min=3 / median=11 / max=33（短击打块数少，置乱强度有限，是该臂未过线的可能解释；不作为豁免理由）。

## 分 family（n 小，只报数不外推）

| truth family | n | normal 命中 | silence 命中 | shuffle 命中 |
|---|---:|---:|---:|---:|
| clicks | 6 | 0 | 0 | 1 |
| clops | 6 | 0 | 0 | 0 |
| cymbal | 3 | 0 | 0 | 1 |
| drag | 6 | 1 | 0 | 0 |
| hihats | 9 | 1 | 0 | 0 |
| kicks | 6 | 1 | 0 | 1 |
| lip_oscillation | 12 | 0 | 0 | 1 |
| liprolls | 3 | 0 | 0 | 0 |
| other | 6 | 0 | 0 | 0 |
| reverse | 3 | 0 | 0 | 0 |
| scratch | 6 | 0 | 0 | 1 |
| snares | 9 | 1 | 0 | 1 |
| synth | 6 | 0 | 0 | 1 |
| teeth | 3 | 0 | 0 | 0 |
| tongue | 3 | 0 | 0 | 0 |
| vocal | 6 | 1 | 0 | 1 |
| whistles | 3 | 0 | 0 | 0 |
| zippers | 6 | 0 | 0 | 1 |

## 失败/异常清单

- `normal_hit_but_shuffle_hit`（3）：018_tongue_kick, 062_msh_snare, 087_throat_bass
- `normal_miss_but_shuffle_hit`（6）：007_inward_zipper_roll, 032_click_da_phonem, 052_synth_high_pitched, 056_vocalized_lip_oscillation_perfect_octave, 065_zede_scratch, 102_crash
- `silence_pred_equals_normal_pred`（3）：005_throat_kick, 009_double_hollow_clop, 052_synth_high_pitched
- `silence_hits`（0）：—

## 自证

- normal 特征 vs 冻结 `probe__*.npy`：exact 34/34，max_abs_diff=0.000e+00；
- normal 预测 vs 冻结 `preds.csv`：**34/34 逐条一致**（不一致即 ABORT，本表不产出）；
- 输入指纹：preds.md5=`efc821bfe15b2a936d641cd4263e9c78` / testset.md5=`e668ce45e311cc2873847860240d52a5` / label_space.md5=`8a5788fb3ae0f3855b8e66a2793a191a` / model.md5=`f28bbb325c195b2d83e198a20b26e3a7`。

## 限幅

- n=34，family 级 n 多为个位数，**只报数不外推**；
- 本表只证明「链路读音频、对静音/置乱有反应」，不证明 103 音色识别能力；
- shuffle 臂未过线时不得写任何方向结论，只写「未过线」。

