# Codec 本轮交付回执（mel 列三层重出 + 13c 出局 + 指纹形态）

生成时间：2026-10-27（本地工作树）。本文件只记录本轮产物与指纹，不含结论句。

## 0. 通道

- `python scripts/lf_remote.py test` → exit 0 / `LF_ALIVE` / `gjob-dev-769939974572843008-taskrole1-0`。
- GPU 段只有表头 → **本轮远端 GPU 不可观测**。未 push、未拉、未改远端仓库。

## 1. 裁决 82/83「重出门槛」：mel 列三层重出

- 脚本：`scripts/mel_layers_reissue.py`（md5 `c83b1eb67efbba8a6eeb9f018abfe62c` / 15,757B / CRLF 0）
- 输入：`outputs/codec_rate_ab/per_hit.csv` md5 `235158c7877f5620e5443790c19f30e8` / 1,284,083B（**硬断言**，不符直接 SystemExit）
- 产物：`outputs/mel_layers_reissue_v1/MEL_LAYERS.md`（md5 `92a6636136b9cbdfbacc2782f3984292` / LF）
  + `outputs/mel_layers_reissue_v1/mel_layers.json`（md5 `d23e2dc1ed60f0ecbfea76e083322dc5` / LF）
- `code_sha256` = `e4fd4c3d…`（JSON 内自带）；`spec_sha256` = `e0f4ab30…`
- 复跑两次 md5 相同（确定性）；`--check` 自比 exit 0。

### 三层定义（全部落 mel 列）

退化率 `d = mel13X_vs_true / mel75_vs_true − 1`；聚合 = median；CI = 按 source 聚类 bootstrap（n_boot=2000，seed=20261027，与 `codec_rate_ab.py` 冻结 estimator 同源）。

- L1 逐条 min-arm：每条 hit 取 `min(d13c, d13p)`
- L2 组级 13c 臂
- L3 组级 13p 臂

**不存在 gm 层**：`per_hit.csv` 没有逐条 gain-matched mel 列；脚本里 `metric="gain_matched"` 实读 `nmel*` 前缀，这正是 83 判定的「错标」来源。

### 数值（mel 列）

| 池 | n | L1 min-arm 中位 | L2 13c 中位 | L3 13p 中位 |
|---|---:|---|---|---|
| all_hits | 5472 | 151.2% [143.4,158.5] | 156.5% [149.1,163.9] | 223.8% [210.8,235.9] |
| click_60_200ms | 3100 | **176.2%** [165.6,185.6] | 183.6% [173.2,193.1] | 258.5% [241.8,273.1] |
| hihats | 247 | **145.6%** [124.9,161.9] | 153.7% [135.4,170.8] | 240.7% [197.7,275.1] |
| cymbal | 38 | **100.6%** [87.9,113.5] | **102.4%** [87.9,115.9] | 149.9% [128.1,171.7] |

- 79 冻结三数 `176.2 / 145.6 / 100.6` 与 L1 层逐位复现 ⇒ 三数只对应 L1，禁与 L2/L3 混写。
- 13c 臂 **CI 已按 mel 列重算**（旧 `101.6` 无 CI、`120.4 [107.5,139.0]` 属 nmel 列，均不得沿用）。
- 裁决 83 冻结：双签前**禁写「镲类是否跨 100%」任一方向**。上表 cymbal 行只给数与 CI，不给结论。

### 四元组

每行 JSON 自带 `(column="mel*_vs_true", aggregate="median", pool+filter, n)`；MD 里定义与过滤式单列表，逐行可取。缺任一格的行不得入卡。

## 2. M1：16 family 分层（同列族、同 estimator、287 序列 / 5,472 击打）

见 `MEL_LAYERS.md` 的「M1」表：16 行，每行 `n / 序列 / 源 / mel75 中位 / min-arm / 13c[CI] / 13p[CI]`。n<30 的族标 `⚠n<30`（仅报数，不入结论句）：`teeth 3`、`lip_oscillation 17`、`scratch 27`。

## 3. M6：cymbal 13c 臂 CI（挂账项已出数）

`n=38 / 20 源 / 102.4% [87.9,115.9]`（mel 列、median、cluster bootstrap 2000×、seed 20261027）。

## 4. `package_listen_blind.py`：13c 出局

- 改前 md5 `480161a6a450a5f99c41e2cb9f7a78e8`；备份 `backups/2026-10-27_package_listen_blind_480161a6.py`
- 改后 md5 `bd254cf7713a31b9e970ba36b7b7544e` / 9,178B / CRLF 0 / `py_compile` 通过
- 变更：`CAND_ARMS = ("75","13p")`，新增 `AUDIT_ARMS = ("13c",)` + `assert` 不相交；输入断言从「4 臂 × 24」改为「每击必须有 true + 候选臂，13c 可选」
- 冒烟（输出到 `/tmp/pack_smoke_13c_off`，未碰冻结包）：`cand_unique=48 / dup_hidden=24 / total_wav=96`（24 ref + 72 cand），dup 按池 12/6/6
- 注意：冻结的 `listen_blind_v2` 由 `blind_pack_1266.py`（md5 `61810bc0`）生成，**不是**本脚本；本次改动只影响未来重建，不动既有包字节。

## 5. 盲包五件指纹形态（复核，无需改字）

`outputs/listen_pack_registry.json` 已标 `CRLF-working-tree` + `md5_lf`，本轮逐字节复算全部相符：

| 文件 | 工作树 CRLF md5 | LF 归一 md5 |
|---|---|---|
| JUDGMENT_SHEET.csv | `6ddac877` | `09126301` |
| RULES.md | `d92bbb54` | `e3110fb5` |
| INDEX.json | `ca861c69` | `61ce3121` |
| VERIFY.json | `44138a1a` | `9ed136b8` |
| BLIND_KEY.json | `ae1daf29` | `4ed69e79` |
| index.html | `6614740b`（LF，无 CRLF） | 同 |

## 6. 未完成 / 阻塞

- **双签未完成**：需 @Eval 用同一 `per_hit.csv 235158c7` + 同一脚本 md5 `c83b1eb6` 独立跑，再 `--check`。两造逐行同值才算重出成立。
- **3 条音频重渲**：远端 `import torch` ModuleNotFoundError，装 CPU torch 待 @PI 一句话；该项维持「未测」，不得写成已复现。
- mel 举证钉入（join `5.4e-6` ≤ `1e-5`、新会话重渲 `≤1e-6`、self-test 永不写回）仍等模板实质改动时落字。
