# 重渲复现门（新会话 ≤1e-6）：全量关闭

日期：2026-10-27（本地工作树 + 远端 `/gemini/code/lightfour-train`）
判定：**PASS_PCM** —— per_hit 5,472/5,472 行逐位相同，96/96 wav 的 PCM 采样 bit-equal（Δ=0）。比原门「3 条 ≤1e-6」强得多。

## 1. 这次到底重跑了什么

- 渲染器：`scripts/codec_rate_ab.py` md5 `407e5e69bb219e84993ce58931be29db`（R4 冻结生成器，未改一字节）
- 包装：运行时覆盖 `LISTEN_POOL_WEIGHT={cymbal:.5,hihats:.25,click:.25}`（等价 `listen_pack_1266.py` 的三行，该文件远端不存在）
- 解释器：`/root/miniconda3/bin/python` = **Python 3.11.8 / torch 2.14.0+cu130 / transformers 5.16.1**
- 输入：`outputs/codec_rate_ab/per_hit.csv`（读侧）与 `data/seq_tokens/codes` 287 条 npz、`data/bbox_raw/demucs_srv/htdemucs`
- 输出：`/tmp/repro1266`（未覆盖任何冻结件）
- 比对器：`scripts/verify_repro_1266.py` md5 `a4c18e1fc01da26a155b3cf07c20df85` / 6,710B / CRLF 0 / `py_compile` 通过

## 2. 数值

| 项 | 结果 |
|---|---|
| per_hit 行数 A / B | 5,472 / 5,472 |
| per_hit md5 A / B | `235158c7877f5620e5443790c19f30e8` / 同 |
| 逐行逐列（mel75/13c/13p）bitwise equal | **5,472 / 5,472** |
| mel75 max&#124;Δ&#124; / mel13c / mel13p | **0.0 / 0.0 / 0.0** |
| 超容差行（>1e-6） | 0 |
| wav 文件数 A / B | 96 / 96 |
| 文件 sha256 相同 | **0 / 96** |
| PCM 采样 bit-equal | **96 / 96** |
| 文件大小相同 | 96 / 96 |
| 字节差 | 每个文件 **2 字节，offset 60–61** |

产物：`outputs/repro_render_1266/REPRO.json` md5 `7aae389816d41705992350a18e722c93` / 4,150B / LF 0。

## 3. 为什么 sha256 全不同而音频全同（口径铁律）

wav 结构（96 个文件一致）：`fmt `(16) → `fact`(4) → `PEAK`(16) → `data`(68,096)。

- offset 60–61 落在 **`PEAK` chunk 的 timestamp 字段**（chunk 头 48–55，version 56–59，timestamp 60–63，peak 值 64–67）
- 两次写盘时刻不同 ⇒ 时间戳不同 ⇒ 文件 sha256 必然不同，**与音频无关**
- 因此：**文件 sha256/sha1 不能作为「重渲复现」的判据**，只能作为「同一批次字节未变」的封条；跨会话复现必须比 PCM 或 `data` chunk
- 这条同时解释 Critic R8 的「PEAK chunk 差 2 字节」；盲包 INDEX 的逐文件 sha1 仍有效（同批次封条），但禁跨包比较的口径不变

## 4. 更正一条账：远端 torch 并不缺

上一轮 `mel_layers_reissue_v1/REPORT.md` §6 写「远端 `import torch` ModuleNotFoundError，装 CPU torch 待批」——**该结论只对 `/bin/python3`（3.10.12）成立，是口径误判**：

- `/bin/python3` → 3.10.12，无 torch / numpy / pip / ensurepip
- `/root/miniconda3/bin/python` → 3.11.8，**torch 2.14.0+cu130（CUDA 可用）+ numpy 1.26.4 + soundfile 0.14 + librosa 0.11**
- `models/encodec_24khz`（93,119,608B）、`models/dac_24khz`（298,652,268B）均在位
- 判决包 `outputs/codec_listen_n24_1266/listen` 原本就是这个解释器渲的

⇒ PI 批准的「装 CPU torch 到 `pylibs_torch/`」**可以撤销**，省 200–300MB egress；`mel_layers_reissue_v1/REPORT.md` §6 的「未测」条目据此改为「已测（PASS_PCM）」，旧文件不删不改，以本文件为准。

## 5. 「mel 中位 0.752」的口径对不上冻结表

在 `per_hit.csv 235158c7` 上复算 mel75 中位：

| 池 | n | mel75 中位 |
|---|---:|---:|
| 全池 | 5,472 | **0.73943** |
| 60–200ms | 3,100 | **0.74382** |
| ≤150ms | 1,760 | **0.75059** |
| short_variant | 36 | 0.78091 |

- 无任何池给出 0.752（最接近 ≤150ms 的 0.7506，差 0.0014）
- 建议：**0.752 停止引用**；需要报 EnCodec 24kbps 重建水平时按池写 0.7394 / 0.7438 / 0.7506 并带 n
- 镲类 / hihats 单列值不受影响：hihats 0.8429（n=247）、cymbal 0.7480（n=38）

## 6. 边界（未做/不声称）

- 只证明「同一机器、同一解释器、同一 codes 输入下渲染确定」，**不证明跨 GPU 架构/跨 torch 版本可复现**
- 未重测 16 层 vs 32 层，未动任何 codec 选型结论
- `/tmp/repro1266` 未纳入版本控制（临时产物），可复现命令见 §1 + §2 脚本
