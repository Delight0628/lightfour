# PACK_ROLE — listen_blind_midband（v1，审计留痕包）

- **状态**：`audit_only_no13p`（PI 裁决 115）。**judge_allowed = false**，不作任何判决用途、不进任何门/三档统计。
- **定性**：v1 未含 13p 臂，未达裁决 93 锁定格式（REF + 75/13p + dup）→ 属**格式缺陷**（裁决 116），不是可裁项。
- **usage**：`listen_midband_only` 已撤销（registry `listen_pack_registry.json` 3a118ff8 已落账）；本包只作审计留痕与复现对照。
- **处置约束**：不删、不 mv、不改任何既有字节；本文件是本目录唯一新增件（裁决 115 处置口径）。
- **判决入口**：唯一入口 = `outputs/codec_listen_n24_1266/listen_blind_v2/`（72 行）。扩样判决用
  `listen_blind_midband_v2/`（裁决 116 改写形状：12 item × REF+75/13p+dup = 48 wav / sheet 36 行）。
- **本包六件指纹**（md5(LF)，CRLF 全 0，与裁决 115 报值一致）：
  - `JUDGMENT_SHEET.csv` `264f5241`
  - `RULES.md` `fcb68fb4`
  - `INDEX.json` `882584f4`
  - `VERIFY.json` `b63bae36`
  - `index.html` `ecdabe5f`
  - `BLIND_KEY.json` `63937d7e`（答案键，不随包发）
- **卡面禁引**：本包任何数字不得进卡面数字栏、不得进结论句；如需留痕只准写「midband v1（audit_only_no13p，未判）」。
- **结构留档**：24 组 A/B（12 真实臂 cand=75 + 12 dup），48 wav，`dup==ref` 12/12、真实臂 12/12 有差、
  本池构成 F 9 / Tch 2 / Tss 1、跨 split 混采（train 10 / val 1 / test 1）、12 uniq source。
- **已知结构风险（裁决 119）**：同 hit 双现（v1 item_13/14 = `BV1VonA6WEDs_1354` 相邻），听者可凭记忆认出 dup
  → dup 判 0 只是必要不充分；v2 已按 126 改形（每 hit 恰 1 item）消除相邻同 hit。
