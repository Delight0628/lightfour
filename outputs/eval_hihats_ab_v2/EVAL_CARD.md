# HIHATS_VERDICT_AB_V2 — hihats 16 层人耳 A/B 判决仪器（评测卡）

## 卡头（两行并列铁律）
- **登记行（不因人耳结果改变）**：hihats mel **0.8429** > §8 0.8 Gate 越线（登记值 = R4 hihats 全池 247 段中位，
  provisional，人耳通过不解除登记）。
- **人耳结果行**：`未判（Delight 未填 JUDGMENT_SHEET）`。

## 状态
盲包 V2 已建成并通过双重校验 + 盲性硬门，**判决未出前不得写通过/不通过**；唯一判决数据 = Delight 填完
`listen_blind/JUDGMENT_SHEET.csv` 后由 `scripts/score_hihats_verdict_12.py` 出的 SCORE_CARD（落
`outputs/eval_hihats_ab_v2/SCORE_CARD.md`，不写进试听目录）。

## 盲性修复记录（2026-10-28，PI 点名的洞已修）
- **修复前的洞（v1）**：`listen_blind/INDEX.json` 48 条记录逐条带答案字段、`BLIND_KEY.json` 与试听文件同目录，
  「答案键不随包发放」名不副实。
- **修复后的目录边界**：
  - `listen_blind/` = 判决人唯一拿到的目录：48 wav + `INDEX_PUBLIC.json`（脱敏索引，字段只有
    blind_id/item/role/bytes/sha1，凭证=逐文件 sha1）+ `JUDGMENT_SHEET.csv` + `RULES.md` + `index.html`
    + 两套 VERIFY；
  - `ADMIN/` = 评测侧保管、不随包发放：`BLIND_KEY.json`（答案键）+ `INDEX_ADMIN.json`（含臂别/击打映射的完整索引）；
  - `archive_pre_v2/` = 修复前原件逐字节留档（含 `wav_sha1_manifest.json`）。
- **验收读数（一手实测）**：`listen_blind/` 文本文件 `'arm'` 子串命中 **0**、`"arm"` 命中 **0**、
  48 个 wav 二进制内 `arm` 字节命中 **0**；无 `BLIND_KEY.json`、无完整 `INDEX.json`；
  `VERIFY_INDEPENDENT.json` **48/48** sha1 重算一致、`failures=[]`、PASS；wav 与 archive 清单逐文件比对
  **48/48 一字节不动**；判分脚本盲性守卫实测（临时副本塞回答案键）**exit 2 INVALID**，合成填表端到端出三门判定
  （临时副本已删）。
- **修复脚本（可复现）**：`scripts/build_hihats_ab_v2_public.py`（脱敏索引 + 构建自检，盲性扫描不动点、
  pass 门控 exit）、`scripts/verify_hihats_ab_v2_public.py`（独立复核，不复用 build 中间结果）。
- **残留风险（不在包内、须知悉）**：源判决包 `outputs/hihats_verdict_pack_v1/` 的文件名带臂别后缀，
  与试听 wav 做字节/哈希比对即可解盲——这是环境级问题，包内已无任何答案键或答案字段；
  RULES.md 已写明「判完前请勿打开该目录」。如需绝对盲性，由 @PI 决定是否封存源包。
- **留档提示**：`archive_pre_v2/RULES.md` 为 v1 原件（逐字节保留），其「禁用字样」段以反例形式引用了旧口径
  误读数与旧臂名；留档不改写，是否接受该留档形式请 @PI 拍板。

## 样本与代表性
- 12 击 / 12 source / 48 wav（true、75、13c、13p 各 12），FLOAT；dur 85.3–230.4 ms；split train 11 / test 1。
- **记法配额 F 6 / Tch 3 / Tss 3**（sound_id 12/13/11），每记法内 source 分层、每 source 至多 1 条。
  偏离 PI 字面口径（纯 source 分层实测会产出 F 11 / Tss 1 / Tch 0，单记法包场）已由 **PI 裁决 188 追认**
  （2026-10-28）；被否决的 12 条 hit_id 存档于判决包 `composition.json`（`rejected_unconstrained_draw`）。
- **source 级重叠 2 条**（BV177tpz7EC3、BV1VonA6WEDs）：与旧 12 条为**不同击打**，旧 12 条从未被判决
  （JUDGMENT_SHEET 未填），无听感暴露，接受；如需 source 级全新鲜，须在 235 段上追加排除后重渲。
- **置信度：低**（n=12；Tch n=3 / Tss n=3 只报数，不单独下结论）。本包全部为 hihats，只判 hihats，
  不得外推三池或全 family。

## 判决规则（PI 2026-10-28，本包唯一适用规则集）
1. 有效性门：dup 判 0 **≥11/12** 且 13p 判 ≥1 **≥8/12**，两门同时过才可出 75 臂判决；任一不过 = 仪器失效。
2. 75 臂（判 0 条数）：**≥8/12 通过 / ≤5/12 触发 32 层评估 / 6–7/12 灰区扩样到 24（只准一次）**。
3. dup 仅仪器自检，不得当任何臂的统计证据；judgment=1 单列 `n_midband`（未判定），不进汇总。
4. **13c 不入判决**；**mel 不进判决**；判定只认人耳 A/B。

## 判决池描述值（不进判决、不与登记值并排）
包内 12 条 `mel75_vs_true` 中位 **0.8257**（F 0.8822 / Tch 0.7904 / Tss 0.8182，逐条见判决包 `index.json`）。
登记值 0.8429（247 段全池）与本描述值样本口径不同，只准进各自栏位。

## 交付物
- `listen_blind/`（判决人唯一目录）：12 REF + 36 匿名候选（75/13p/dup，槽位 4/4/4 平衡）、
  `INDEX_PUBLIC.json`（脱敏索引）、`JUDGMENT_SHEET.csv`（36 行）、`RULES.md`、`index.html`（试听页）、
  `VERIFY.json`（构建自检）、`VERIFY_INDEPENDENT.json`（独立复核）。
- `ADMIN/`（不随包发放）：`BLIND_KEY.json`、`INDEX_ADMIN.json`、`ADMIN_NOTE.md`。
- `archive_pre_v2/`：修复前 8 件原件 + `wav_sha1_manifest.json` + `ARCHIVE_NOTE.md`（含逐件 md5）。
- 评分入口：`python scripts/score_hihats_verdict_12.py`（三门硬编码 PI 2026-10-28 口径 + 盲性硬门）。
- 试听入口：`outputs/eval_hihats_ab_v2/listen_blind/index.html`（本地文件，非公网页面）。

## 边界
- 人耳判决未出前不得写「hihats 通过/已稳」；判决通过也不解除 0.8429 越线登记。
- 本包不构成 codec 过关结论；16 vs 32 层非控制变量，32 层评估若触发须单独立项。
- 卡内禁用字样按 agent.md ㉛ / §14.14；R1 唯一臂名 `13p_align`（150.4%）。
