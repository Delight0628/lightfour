# lightfour

> 可实时语音交互的**电子音乐识别 / 细分类**训练项目  
> Voice-interactive EDM recognition & subgenre classification

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![Qwen2.5--Omni](https://img.shields.io/badge/Qwen2.5--Omni-3B%20%7C%207B-purple)](https://github.com/QwenLM/Qwen2.5-Omni)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 这是什么

`lightfour` 目标是训练一个**小体量、可语音对话**的模型，能够：

1. **听歌识曲 / 音乐理解** — 识别电子音乐，并区分细分子类（House / Techno / Dubstep / DnB …，对齐 Beatport 类目）
2. **语音交互** — 听懂语音输入，用语音自然回复
3. **可落地** — QLoRA / LoRA 微调，租卡可训，本地可推

主路径是 **Qwen2.5-Omni（3B/7B）+ QLoRA SFT**，音乐分类数据走 **FMA → Beatport 两阶段**。

## 架构速览

```text
语音 / 音频输入
      │
      ▼
┌─────────────────┐     QLoRA / SFT      ┌──────────────────────┐
│  Qwen2.5-Omni   │ ───────────────────► │  听音频 → 细类标签    │
│  (Thinker/Talker)│                      │  + 自然语言解释 / 对话 │
└─────────────────┘                      └──────────────────────┘
      │
      ├─ Stage 1: FMA Electronic 粗分类
      └─ Stage 2: Beatport 32 类细类（需自建监督数据）
```

更完整的目标、数据结论、硬件与待办见 [agent.md](agent.md)。

## 目录结构

| 路径 | 说明 |
|---|---|
| `agent.md` | **项目脉络与会话约定**（每次对话先读） |
| `Dockerfile` | 云端训练/推理环境（ms-swift + PyTorch + Qwen-Omni 依赖） |
| `docs/omni_run_guide.md` | Qwen2.5-Omni 服务器运行指南 |
| `scripts/omni_*.py` | 模型加载 / 推理 / 环境检查 |
| `scripts/analyze_edm_pipeline.py` | FMA ↔ Beatport 可行性分析 |
| `scripts/extract_fma_meta.py` | FMA 元数据统计 |
| `omni_voice_test/` | 语音输入 → 语音/旋律输出 API 实验 |
| `data/` | FMA / Beatport 数据引用与报告（大文件不入库） |

## 快速开始

### 云端（推荐训练）

```bash
# 环境检查
/root/miniconda3/bin/python scripts/omni_env_check.py

# 模型加载测试（约 5–6 分钟，12GB 权重）
/root/miniconda3/bin/python scripts/omni_load_test.py

# 推理：文本 → 文本 + 语音
/root/miniconda3/bin/python scripts/omni_infer.py --text "你好，请介绍一下你自己。"
```

QLoRA 微调（ms-swift 4.x）示例见 `Dockerfile` 底部注释。

### 数据

- FMA 元数据：`data/fma/fma_metadata/`（约 10.6 万曲，Electronic 子树 ~3.4 万）
- 结论报告：[`data/fma/edm_pipeline_report.md`](data/fma/edm_pipeline_report.md)
  - Stage-1（Electronic 粗分类）**可行**
  - Stage-2（Beatport club 细类）**标签可行、监督数据需自建**

## 路线图

- [x] 开源方案调研与能力边界（对话 ✓ / 分类需 SFT / 音乐生成 ✗）
- [x] FMA 元数据 + Beatport 映射分析
- [x] Omni 语音 API 通路实验
- [x] Docker 运行环境与推理脚本
- [ ] 电子音乐子类 SFT 数据集
- [ ] Qwen2.5-Omni QLoRA 微调与评测
- [ ] （可选）Stage-2 club 细类 / 音乐生成模块

## 安全说明

- **不要**提交 SSH/SFTP 口令、API Token、模型权重、原始音频大包
- 服务器连通 / 上传脚本请放在本地并读环境变量，勿写进仓库
- 详见 [agent.md](agent.md) §7

## License

[MIT](LICENSE)
