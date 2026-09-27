#!/usr/bin/env python3
"""Qwen2.5-Omni 模型加载验证脚本

验证模型能否完整加载到 GPU（不执行推理，只测加载）。
在服务器上运行：
    /root/miniconda3/bin/python scripts/omni_load_test.py
"""
import time

MODEL_PATH = "/gemini/pretrain"


def main():
    import torch

    print(f"GPU before: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")
    print(f"Loading from {MODEL_PATH} ...")

    t0 = time.time()
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    t1 = time.time()

    used = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Loaded in {t1 - t0:.0f}s")
    print(f"GPU after: {used:.1f}GB / {total:.1f}GB (free: {total - used:.1f}GB)")
    print("=== MODEL + PROCESSOR READY ===")


if __name__ == "__main__":
    main()
