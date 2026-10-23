# route_b.md — Route B 条件生成（Omni 的对照/降级方案）

> **定位**：Route B 不再是主线。主线是 [agent.md](file:///e:/lightfour/agent.md) 的 **All in Qwen3-Omni**。
> 本文保留 Route B（小型条件 Flow Matching）作为：① Omni 的对照基线；② 训练/数据管线出问题时的降级方案；③ 合成数据生成器。
> 与 Omni **共享**的数据源、103 音色 taxonomy、预处理管线、codec 选型结果，均在本文列出并引用 agent.md。
> 最后更新：2026-10-23

---

## 1. 与 Omni 的关系

| 维度 | Omni（主线） | Route B（本文） |
|---|---|---|
| 模型 | Qwen3-Omni 30B-A3B（Thinker-Talker） | 3.96M 条件 Flow Matching（2D U-Net） |
| 输入 | 音频 / 文本 / 鼓谱 | 鼓谱 piano-roll + tempo |
| 输出 | 文本 + 通用 codec token → 音频 | log-mel → 声码器 → 音频 |
| 训练目标 | 多任务（文本 CE + codec CE） | Flow Matching（mel 回归） |
| 硬件 | 24–48GB | CPU / 6.2GB 可跑 |
| 角色 | 最终一体化 | 对照、降级、数据生成 |

**共享内容（与 agent.md 一致）**：
- 数据源：B 站 Swissbeatbox 竞技对战，排除 loopstation/Showcase/POV。
- 预处理：Demucs → onset → AST → （新）103 音色标注。
- **103 音色 taxonomy**：Route B 的条件空间也按 103 音色编码，不再用 4 类。
- codec 选型：EnCodec/DAC 的结论同时影响 Omni Talker 和 Route B 的声码器选择。
- 评估集：同一批 20 条切片用于 codec 重建与生成质量对比。

---

## 2. Route B 数据

| 集合 | 规模 | 来源/说明 |
|---|---|---|
| 合成 | **90,000 窗口 / 50.00h** | hit_bank 按鼓谱重排；服务器 `gen_synth_tar/` 8.3GB，45 tar 分片（每片 2000 窗口） |
| 合成元数据 | 90,000 行 | `scores.jsonl`（鼓谱）+ `omni_pairs.jsonl`（配对） |
| 真实生成对 | **3,036 窗口 / 1.69h** | 65 个来源；旧 4 类口径（待按 103 音色重标） |
| 真实 hit bank | 4,872 个击打 | 4 类 × 28 参与者，0.330h；待扩到 103 音色种子 |

**已知问题**：合成数据 hit 复用 91–130×，真实录音仍是瓶颈；旧 4 类标签需按 103 音色重标后才能与新体系对齐。

---

## 3. Route B 模型与训练

- 脚本：`scripts/prepare_gen_routeb.py`、`scripts/train_gen_routeb.py`
- 架构：小型 2D U-Net 条件 Flow Matching，**3.96M 参数**
- 条件：原为 4 类 piano-roll + tempo，`(5,122)`；**新版应按 103 音色编码**（可先用 family 分组降维）
- 目标：log-mel `(1,64,122)`（64 mel × 122 帧，约 2s）
- Smoke：loss **2.09→0.60**（100 步 CPU，54s），checkpoint 15.9MB，能生成 4 条 mel
- Gate：按给定鼓谱生成可辨识 beatbox（人耳 + 分类器双验证）

**与 Omni 的迁移关系**：
- Route B 训好后可作为 Omni Talker 的**初始化/对照**（同为「条件 → 音频表示」）。
- Route B 的 mel 目标可换成通用 codec token，直接复用同一条件编码。
- Route B 的合成数据可作为 Omni 的预训练/增广数据（需标注 103 音色）。

---

## 4. 训练与评估流程

1. `prepare_gen_routeb.py`：把 tar 分片转成 `mel/*.npy`、`wav/*.bin`、`index.jsonl`、`train/val.jsonl`。
2. `train_gen_routeb.py --root ... --epochs ... --batch-size ...`：Flow Matching 训练。
3. 生成：从验证集采样鼓谱 → 条件 → 采样 mel → 声码器 → wav。
4. 评估：
   - 客观：mel 距离、时长、频谱质心。
   - 主观：人耳能否听出目标音色（103 音色词表）。
   - 交叉：用识别模型（待建 103 音色分类器）判生成音频的音色命中率。

---

## 5. Route B 下一步（不阻塞 Omni 主线）

| # | 行动 | 状态 |
|---|---|---|
| 1 | 条件空间从 4 类扩到 103 音色（先按 family 分组） | ⬜ |
| 2 | 用 95 条音色种子扩 hit bank（103 音色） | ⬜ |
| 3 | 合成数据按新条件重新渲染 | ⬜ |
| 4 | 用 EnCodec/DAC 作为声码器替代 mel→wav，与 Omni codec 对齐 | ⬜ |
| 5 | 小规模训练 + 人耳评估 | ⬜ |

---

## 6. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-10-20 | Route B 数据管线：hit_bank 4,872 / 合成 50h / 真实生成对 3,036；smoke 训练跑通 |
| 2026-10-23 | 从 agent.md 拆分独立；定位改为 Omni 的对照/降级方案；条件空间待扩到 103 音色 |
