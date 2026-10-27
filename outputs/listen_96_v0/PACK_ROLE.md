# PACK_ROLE: historical —— 其 hihats 判决池数字「错池、禁引」

- **本目录** = `outputs/listen_96_v0/`
- **登记日期** 2026-10-27 ｜ 登记人 Codec ｜ 台账 `outputs/listen_pack_registry.json` → `listen_96_v0`

## 结论

本包的 hihats 判决池（6 条）与 **1266 判决池重叠 = 0**，两套池互不替代。凡引用本包 hihats 中位数者，说的都不是 PI 60 锁定的判决池。

**错池数字（禁引，禁入卡、禁入结论句）**

| hit_id | mel75_vs_true |
|---|---|
| `BV11m421W7Xc_1005` | 0.72136 |
| `BV1EeJR6TEzm_0164` | 0.91871 |
| `BV1Ut411Z7rw_0293` | 0.76949 |
| `BV1QW411u7NJ_0494` | 0.90306 |
| `BV1VZ4y1W7sf_0170` | 0.81590 |
| `BV1YssozfEXU_0030` | 0.99040 |

聚合 = median，**中位 0.85948，4/6 > 0.8**（n=6）。这组数只作历史留档。

## 正确读数在哪

- 判决池 hihats 6 条 = `codec_listen_n24_1266/listen/index.json` 的 hihats 族：描述中位 **0.80383**，4/6 > 0.8（本包描述值，非登记值）。
- 登记值仍是 R4 全池 247 中位 **0.8429**（铁律字面量，卡头越线行用这个）。

## 硬约束

1. 本目录 `judge_allowed = false`，不参与判决、不参与扩样。
2. 其 `index.json` 含逐文件 sha1，但 `files{}` 只覆盖本包四臂；**不得搬到 1266 或 v2 盲包**。
3. 其余内容（wav、composition、index）只登记、不删、不 mv、不改。
