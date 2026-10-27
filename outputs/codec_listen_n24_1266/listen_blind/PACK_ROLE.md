# PACK_ROLE: audit_only（禁判决）

- **本目录** = `outputs/codec_listen_n24_1266/listen_blind/`
- **登记日期** 2026-10-27 ｜ 登记人 Codec ｜ 台账 `outputs/listen_pack_registry.json` → `codec_listen_n24_1266/listen_blind`

## 为什么是审计包

每击落 5 个文件 = REF + 75 + **13c** + 13p + dup，即盲位 **4 候选**；锁定格式是 **3 候选**（REF + 75 / 13p / dup）。13c 不进判决（appendix only），所以本包结构上不可能给出与规则卡一致的判决。

## 硬约束

1. 禁止用作 Delight 判读输入；禁止投喂 `JUDGMENT_SHEET`。
2. 禁止与 `listen_blind_v2/` 并表、并图、比中位。
3. 禁止作为扩样池（扩样只用 `outputs/listen_hihats_v2/`，且只准一次）。
4. 只登记、不删、不 mv、不改名、不改内容。

## 内容指纹（变更即台账失效）

| 文件 | 形态 | md5 |
|---|---|---|
| `manifest.csv` | LF | `f630581e37a7a51642c1ee04e4c6db86` |
| `pack_summary.json` | LF | `5a67c1870e1e093c47818378e7dd41ee` |

规模：120 wav = 24 ref + 72 cand(kind=cand) + 24 dup；分池 cymbal 60 / hihats 30 / click 30。

## 判决入口

`outputs/codec_listen_n24_1266/listen_blind_v2/`（96 wav = 24 击 × REF + 75/13p/dup，dup 与 REF 逐字节相等 24/24）。
