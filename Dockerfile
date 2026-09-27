# ============================================================
# Qwen2.5-Omni 电子音乐识别微调 — 运行环境 Dockerfile
#
# 对应 agent.md §2/§6：QLoRA/LoRA 微调 Qwen2.5-Omni-3B/7B
# 训练框架：ms-swift（modelscope）+ transformers + peft
#
# 云平台使用说明：
#   1. 基础镜像在平台 UI 选择，本文件不写 FROM
#      推荐：CUDA 12.1 / Python 3.11 / Ubuntu 22.04 / CuDNN 8
#   2. 平台不允许 FROM / EXPOSE / CMD / ENTRYPOINT
# ============================================================

# ---------- 环境变量 ----------
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Qwen2.5-Omni 模型路径（云端共享盘）
ENV MODEL_PATH=/gemini/pretrain
ENV MODEL_OUTPUT_DIR=/gemini/code/models

# TensorFlow / PyTorch 显存按需增长
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- 系统依赖 ----------
# 重写 apt 源为阿里云（云平台内部镜像不稳定）
RUN echo "deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        locales \
        unzip \
        zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 zh_CN.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ---------- Python 核心 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "numpy>=1.21.0,<2.0" \
        "pandas>=1.5.0" \
        "scipy>=1.10.0" \
        "scikit-learn>=1.3.0" \
        "protobuf>=3.20,<4.0"

# ---------- PyTorch + CUDA ----------
# 大包（~554MB），加大超时防下载中断
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
        torch torchvision torchaudio

# ---------- transformers 全家桶 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "transformers>=4.52.3" \
        "accelerate>=1.0.0" \
        "safetensors>=0.4.0" \
        "tokenizers>=0.20.0" \
        "sentencepiece>=0.1.99"

# ---------- QLoRA / LoRA 训练 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "peft>=0.13.0" \
        "bitsandbytes>=0.45.0" \
        "datasets>=3.0.0" \
        "trl>=0.12.0"

# ---------- ms-swift 训练框架（agent.md §2 首选） ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "modelscope>=1.20.0" \
        "ms-swift>=2.0.0"

# ---------- 音频处理 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "librosa>=0.10.0" \
        "soundfile>=0.12.0" \
        "av>=12.0.0" \
        "resampy>=0.4.0"

# ---------- Qwen2.5-Omni 专用 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "qwen-omni-utils>=0.0.3" \
        "qwen-vl-utils>=0.0.8"

# ---------- 训练监控 / 工具 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "tensorboard>=2.10.0" \
        "pyyaml>=6.0" \
        "psutil" \
        "tqdm" \
        "rich"

# ---------- 数据分析 / 可视化 ----------
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        "matplotlib>=3.6.0" \
        "seaborn>=0.12.0" \
        "plotly>=5.0.0"

# ---------- 开发工具 ----------
RUN pip install --no-cache-dir \
        "pytest>=7.0.0" \
        "black>=22.0.0"

# ---------- 验证安装 ----------
RUN python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'transformers {transformers.__version__}')" && \
    python -c "import peft; print(f'peft {peft.__version__}')" && \
    python -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')" && \
    python -c "import librosa; print(f'librosa {librosa.__version__}')" && \
    python -c "import modelscope; print(f'modelscope {modelscope.__version__}')" && \
    python -c "import qwen_omni_utils; print('qwen-omni-utils OK')"

# ---------- 工作目录 ----------
WORKDIR /workspace/lightfour

# 拷贝项目代码
COPY . /workspace/lightfour

# 确保目录可写
RUN mkdir -p data output logs checkpoints

# ---------- 启动说明（平台不允许 CMD） ----------
# QLoRA 微调（ms-swift）：
#   swift sft \
#     --model ${MODEL_PATH} \
#     --train_type lora \
#     --dataset data/train.jsonl \
#     --lora_rank 8 \
#     --num_train_epochs 3 \
#     --output_dir output/qwen-omni-edm-lora
#
# QLoRA 微调（transformers + peft）：
#   python scripts/train_lora.py --model ${MODEL_PATH} --data data/train.jsonl
#
# 推理测试：
#   python -c "from transformers import Qwen2_5OmniForConditionalGeneration; print('OK')"
#
# 资源提示（agent.md §3）：
#   3B 模型 QLoRA ≈ 12-16GB 显存（单张 3090 够）
#   7B 模型 QLoRA ≈ 18-24GB 显存（需 A100/4090）
#   MAPPO 项目同时跑时注意 RAM 24-28GB / 32GB
