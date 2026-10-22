# Beatbox 音色命名体系（Sound Taxonomy）

> 目的：给 `manifest.json` / 伪标 / 分类器提供**统一的音色命名标准**，避免"把 20 多种音色硬塞进 4 个鼓类"。
> 主要来源：[Beatbox Science Sound Library](https://beatboxscience.nazcabeats.com/sound-library)（103 个音色，标准化记法 + 发声机制）；辅助：School of Beatbox、BecomeSingers、Devon Guinn、B 站邹Sir/啊宾教学。
> 最后更新：2026-10-22

---

## 1. 当前样本文件命名规则（先回答"这三份文件怎么看"）

```
{标签}_{BV号}_{切片序号}_{置信度}.wav
```

| 部分 | 示例 | 含义 |
|---|---|---|
| 标签 | `kd` / `sd` / `hhc` / `hho` | 4 类鼓组标签（见下表），**不是 beatbox 术语**，是鼓组缩写 |
| BV号 | `BV1ii2SYsE3k` | B 站视频 ID，可在 `https://www.bilibili.com/video/BV1ii2SYsE3k/` 找到原视频 |
| 切片序号 | `1238` | 该视频内第 1,238 个 onset 切片（按时间顺序编号） |
| 置信度 | `0.95` | 伪标模型/规则给出的置信度 |

**没有 `kb` / `sb`**——你看到的应该是 `kd`（底鼓）和 `sd`（军鼓），小写字母 `d` 在截图里容易看成 `b`。

### 三份抽检包的区别

| 目录 | 内容 | 用途 |
|---|---|---|
| `spotcheck_20261022` | 9 条，第一轮高置信分层抽样（kd 2 / sd 2 / hhc 2 / hho 3） | 看整体标注质量 |
| `spotcheck_batch3` | 12 条，第 3 批每类 3 条 | 看新批次标注质量 |
| `suspicious_spotcheck` | 12 条，**CNN 与 RF 判断不一致**的高置信样本 | 文件名 `cnn-X_rf-Y_...`：`X` 是 CNN/规则判的类，`Y` 是随机森林判的类。听完可以直接裁决谁对 |

`manifest.json` 每条记录包含：`file`、`label`、`confidence`、`source_video`（BV 号）、`start`（视频内起点秒）、`dur`（切片时长）、`ast_beat`（AST beatbox 分数）、`feat`（低频/中频/高频占比、谱质心、衰减时长）。

---

## 2. 当前 4 类是什么（鼓组映射，不是完整音色体系）

| 缩写 | 英文 | 中文 | 声学特征 |
|---|---|---|---|
| `kd` | kick drum | 底鼓 | 低频主导、衰减中等 |
| `sd` | snare drum | 军鼓 | 中高频噪声、衰减短 |
| `hhc` | hi-hat closed | 闭镲 | 高频主导、衰减很短（<120ms） |
| `hho` | hi-hat open | 开镲 | 高频主导、衰减较长（≥120ms） |

这 4 类来自 AVP-LVT 数据集，是**鼓组视角**。竞技 beatbox 实际有 **100+ 种音色**，强行四分类正是 sd/kd 混淆、hho 不可靠的根源之一。

---

## 3. 完整音色体系（103 音色，按家族分组）

记法说明：`P`/`PF`/`K`/`T`/`W` 等是 beatbox 社区标准记法；机制列 `Laryngeal`（喉部）、`Pulmonary`（肺部气流）、`Humming`（哼鸣）、`Hollow`（空腔）；气流列 `outward`（外送）/ `inward`（内吸）。

### 3.1 Kicks 底鼓（8）→ `kd`

| 记法 | 英文名 | 中文 | 机制/气流 |
|---|---|---|---|
| P | Kick | 底鼓 | Laryngeal / outward |
| P | Inward Kick | 内吸底鼓 | Pulmonary / inward |
| Pw | Fat Kick Short | 厚底鼓（短） | Pulmonary / outward |
| Pww | Fat Kick Long | 厚底鼓（长） | Pulmonary / outward |
| Phrm | Kick w. Chest Bass | 胸腔贝斯底鼓 | Pulmonary / outward |
| Lp👅 | Slap Kick | 拍击底鼓 | Pulmonary / outward |
| D | Tongue Kick | 舌击底鼓 | Laryngeal / outward |
| Hb | Throat Kick | 喉音底鼓 | Laryngeal / outward |
| Hm | Throat Kick (Nasal) | 鼻音喉底鼓 | Pulmonary / outward |

### 3.2 Hi-hats 镲（8）→ `hhc` / `hho`

| 记法 | 英文名 | 中文 | 鼓类映射 |
|---|---|---|---|
| T | Hi-hat T | T 镲 | `hhc` |
| T' | Hi-Hat T (short) | 短 T 镲 | `hhc` |
| Tss | Hi-Hat (Opened) | 开镲 | `hho` |
| F | Hi-hat F | F 镲 | `hhc` |
| Tch | Tch Hi-hat | Tch 镲 | `hhc` |
| t | Hi-Hat T (Humming) | 哼鸣 T 镲 | `hhc` |
| f | Hi-Hat F (Humming) | 哼鸣 F 镲 | `hhc` |
| -b | Kiss hi-hat | 亲吻镲 | `hhc` |

### 3.3 Snares 军鼓（24）→ `sd`

| 记法 | 英文名 | 中文 |
|---|---|---|
| PF | PF Snare | PF 军鼓 |
| k | K palate snare | 腭音 K 军鼓 |
| k' | K palate snare (short) | 短腭音 K 军鼓 |
| K | Inward K Snare | 内吸 K 军鼓 |
| Kb | Inward K (closing release) | 内吸 K 军鼓（闭合释放） |
| K² | Double Inward K Snare | 双内吸 K 军鼓 |
| K' | Inward K Rimshot | 内吸 K 边击 |
| K'² | Double Inward K Rimshot | 双内吸 K 边击 |
| K` | Inward K Rimshot (short) | 短内吸 K 边击 |
| Km | Bongo Drum | 邦戈鼓 |
| Koo | Inward K Tonal Rimshot | 内吸 K 音调边击 |
| Kch | Inward Kch Snare | 内吸 Kch 军鼓 |
| Kw | Sub Bass Snare | 超低音军鼓 |
| Krr | Inward K Snare Click Roll | 内吸 K 军鼓 click roll |
| s | Spit Snare | 吐音军鼓 |
| s² | Double Spit Snare | 双吐音军鼓 |
| K | Outward K Snare | 外送 K 军鼓 |
| Kch | Outward Kch Snare (Vocalized) | 外送 Kch 军鼓（带声） |
| Dsh | Dsh Snare | Dsh 军鼓 |
| Df | Duf Snare | Duf 军鼓 |
| M | Msh Snare | Msh 军鼓 |
| Peh | Peh Snare | Peh 军鼓 |
| Eh | Cough Snare | 咳嗽军鼓 |
| Eh^ | High Cough Snare | 高音咳嗽军鼓 |

### 3.4 Cymbal 镲片（2）→ `hho` / 其他

| 记法 | 英文名 | 中文 | 映射 |
|---|---|---|---|
| Pss | Crash | 碎音镲 | `hho`（长衰减） |
| Pss° | Whistling Crash | 哨音碎镲 | 未覆盖 |

### 3.5 Clicks / Clops 点击类（11）→ 未覆盖

| 记法 | 英文名 | 中文 |
|---|---|---|
| d | Click | 点击 |
| da | Click (da) | da 点击 |
| do | Click (do) | do 点击 |
| -d* | Hollow Click | 空腔点击 |
| G* | Uvula Click | 小舌点击 |
| G*a / G*o | Uvula Click (ga/go) | 小舌点击（ga/go） |
| cl | Tongue Clap | 舌拍 |
| L | Clop | 马蹄音 |
| L* | Hollow Clop | 空腔马蹄音 |
| CL* | Double Hollow Clop | 双空腔马蹄音 |

### 3.6 Vocal / Bass 低音（5）→ 未覆盖

| 记法 | 英文名 | 中文 |
|---|---|---|
| O | Throat Bass | 喉音贝斯 |
| 👿🗣️ | Inward Throat Bass | 内吸喉音贝斯 |
| 👿🫁 | Inward Chest Bass | 内吸胸腔贝斯 |
| Rr | Uvula Bass | 小舌贝斯 |

### 3.7 Lip Oscillations / Lip Rolls 唇部（16）→ 未覆盖

| 记法 | 英文名 | 中文 |
|---|---|---|
| w | Lip Oscillation (classic) | 唇颤音（经典） |
| w° | Whistling Lip Oscillation | 哨音唇颤 |
| w🗣️P0 / 5 / P8 | Vocalized Lip Oscillation | 带声唇颤（同度/五度/八度） |
| Fb~ | Fart Bass | 放屁贝斯 |
| w☺️ | Smile Bass | 微笑贝斯 |
| Z | Vibrator Lip Bass (Short) | 振动唇贝斯 |
| W | Lip Roll | 唇卷 |
| W* | Hollow Lip Roll | 空腔唇卷 |
| Ww | Inward Lip Roll | 内吸唇卷 |
| Ww~ | Inward Sub Lip Roll | 内吸超低唇卷 |
| Ww* / Ww*~ | Inward Hollow Lip Roll | 内吸空腔唇卷（短/长） |
| Wwd | Inward Lip Roll with Tongue | 带舌内吸唇卷 |

### 3.8 Tongue / Nasal / Whistles / Teeth / Drag / Zippers / Scratch / Reverse / Synth / Other（20+）→ 未覆盖

| 记法 | 英文名 | 中文 |
|---|---|---|
| Drh / Drhb / Drch | Inward Click Roll（含 Zekka Click Roll） | 内吸 click roll |
| Rhm | Nasal Bass / Snore Bass | 鼻音贝斯 / 鼾声贝斯 |
| Drh🦗 / Hw / 🦷° / 🦷🗣️° / 🌀° | Click Roll Whistle / Hollow Whistle / Tooth Whistle / Vocalized Tooth Whistle / Vortex Whistle | 各类哨音 |
| A（out/in） | Outward / Inward Drag | 拖拽音 |
| Zip / Zip~ | Zipper / Inward Zipper Roll | 拉链音 |
| Juy / Shuy | Zede Scratch / Shuy Air Scratch | 刮擦音 |
| ῤ / Chῤ | Reverse Kick / Reverse Snare | 反向底鼓/军鼓 |
| Ow / Ow🎵 | Synth / Synth (high) | 合成器音 |
| D* | Dry Kick | 干底鼓 |
| 🍾 | Pop | 瓶塞音 |
| HD / Hd / Df / Dp / <Fd | Sucker Punch / Inward Hd / Inward Df / Inward Dp / Inward Fd | 特殊内吸音 |
| Dr' | Tight Tongue Roll | 紧舌卷 |
| Mew | Mew Squeak | 喵叫 |
| SR~ | Slizzer Roll | 滑舌卷 |

> 完整结构化版本见 `data/beatbox_taxonomy.json`（供程序读取）。

---

## 4. 这个体系怎么用（改造 manifest 与分类器）

### 4.1 manifest 字段扩展

现有：
```json
{"id": "...", "source": "BV...", "wav": "...", "pseudo_label": "sd", "confidence": 0.95, "feat": {...}}
```

建议新增（向后兼容，不删旧字段）：
```json
{
  "drum_class": "sd",          // 现有 4 类，保留
  "sound_notation": "PF",      // 标准记法，可为空
  "sound_family": "snare",     // kicks / snares / hihats / cymbal / clicks / bass / liproll / fx
  "sound_confidence": 0.0      // 音色级置信度，人工标注前为 0
}
```

### 4.2 分类器分层方案

| 层级 | 类别数 | 内容 | 状态 |
|---|---|---|---|
| L1 鼓组 | 4 | kd / sd / hhc / hho | 现有，继续可用 |
| L2 家族 | 6–8 | kicks / snares / hihats / cymbal / bass / click / lip / fx | 需人工 seed |
| L3 音色 | 20–100 | 上表 103 音色 | 长期目标，需专家标注 |

**关键结论**：现有 4 类不必推翻，但要明确它只是 L1。被强行塞进 `sd` 的 lips roll / click roll / bass 类，是当前 sd 占 57%、且 1/3 被判成 kd 的主要原因。

### 4.3 下一步落地顺序

1. 人工从现有 28,489 条里按新 taxonomy 标 200–500 条 seed（每家族 20–50 条）。
2. 用 seed 训练 L2 家族分类器（6–8 类），替换"4 类硬分"。
3. L2 稳定后，再对每个家族内部细分 L3 音色。
4. manifest 先补 `sound_family`，`sound_notation` 留给人工标注。

---

## 5. 学习与参考资源

**权威音色库 / 文字列表**
- [Beatbox Science Sound Library](https://beatboxscience.nazcabeats.com/sound-library) —— 103 音色，标准记法、机制、气流、音频
- [School of Beatbox – Essential Beatbox Sounds](https://www.schoolofbeatbox.com/essential-beatbox-sounds/)
- [BecomeSingers – Beatbox Sounds List](https://becomesingers.com/techniques/beatbox-sounds-list)
- [Devon Guinn – Kicks](http://www.devonguinn.com/kicks) / [Snares](http://www.devonguinn.com/snares)

**视频教程（B 站）**
- 邹Sir_BEATBOX课堂：BASS 低音 4 种发音（[BV1CK4y1g7dq](https://m.bilibili.com/video/BV1CK4y1g7dq)）、R 音教学（[BV1x64y1R7Bi](https://www.bilibili.com/video/BV1x64y1R7Bi/)）
- 啊宾的 bbox 教学第 1 期：基础三音（大鼓、镲片、小鼓）（[BV1TpLC63EsE](https://www.bilibili.com/video/BV1TpLC63EsE/)）

**视频教程（YouTube 搜索关键词）**
- `beatbox sounds tutorial all sounds`
- `Swissbeatbox tutorial basics`
- `human beatbox sound library`
