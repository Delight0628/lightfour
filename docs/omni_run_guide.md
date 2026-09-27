# Qwen2.5-Omni 模型运行指南

## 服务器环境（B1.large）

| 资源 | 规格 | 备注 |
|---|---|---|
| GPU | 1× B1.gpu.large 24GB | vGPU |
| CPU | 12 核 | |
| 内存 | 32 GB | |
| 临时存储 | 50 GB | |

## 模型位置

```
/gemini/pretrain/          # Qwen2.5-Omni-3B 完整权重（12GB）
├── model-0000{1,2,3}-of-00003.safetensors
├── config.json
├── tokenizer.json / vocab.json / merges.txt
├── preprocessor_config.json
└── ...
```

## 已装依赖

- Python 3.11.8（`/root/miniconda3/bin/python`）
- PyTorch 2.14.0+cu130（CUDA 可用）
- transformers 5.17.0
- torchvision 0.29.0+cu130
- accelerate / safetensors / tokenizers
- librosa / soundfile / qwen-omni-utils / qwen-vl-utils
- ffmpeg

## 快速开始

```bash
# 1. 环境检查
/root/miniconda3/bin/python scripts/omni_env_check.py

# 2. 模型加载测试（约 5-6 分钟）
/root/miniconda3/bin/python scripts/omni_load_test.py

# 3. 推理（文本 → 文本+语音）
/root/miniconda3/bin/python scripts/omni_infer.py --text "你好，请介绍一下你自己。"

# 4. 后台运行（推荐，SSH 断开不丢进度）
nohup /root/miniconda3/bin/python scripts/omni_infer.py > /tmp/omni_infer.log 2>&1 &
tail -f /tmp/omni_infer.log
```

## 已验证结果

- 模型加载：335s，GPU 占用 12.0GB / 25.0GB
- 文本生成：正常
- 语音合成：正常（`/tmp/reply.wav`，5.3s 音频）

## 与 MAPPO 训练并行

MAPPO 训练（CPU 模式）不占 GPU，Omni 可独享 24GB 显存。
但 RAM 是瓶颈（MAPPO ~12GB + Omni ~12-16GB = 24-28GB / 32GB），注意：
- MAPPO worker 数建议控制在 5 以内
- Omni 推理时 `max_new_tokens` 不超过 256

## 常见问题

| 问题 | 解决 |
|---|---|
| 加载慢（5-6 分钟） | 正常，12GB 从共享存储读入；二次加载走 cache 会快很多 |
| token ID 警告 | transformers 版本差异，可忽略 |
| OOM | 降低 `max_new_tokens`，或设置 `enable_talker=False` 关闭语音输出 |
| torchvision 缺失 | `pip install torchvision` |
