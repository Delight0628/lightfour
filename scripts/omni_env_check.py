#!/usr/bin/env python3
"""Qwen2.5-Omni 环境检查脚本

在服务器上运行，验证所有依赖是否就绪：
    /root/miniconda3/bin/python scripts/omni_env_check.py
"""
import sys
import subprocess

def check_import(module_name, import_name=None):
    import_name = import_name or module_name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'OK')
        print(f"  ✅ {module_name}: {ver}")
        return True
    except ImportError:
        print(f"  ❌ {module_name}: MISSING")
        return False

def main():
    print("=" * 50)
    print("Qwen2.5-Omni 运行环境检查")
    print("=" * 50)

    ok = True

    # Python 版本
    print(f"\nPython: {sys.version}")

    # 核心依赖
    print("\n--- 核心依赖 ---")
    ok &= check_import("torch")
    ok &= check_import("transformers")
    ok &= check_import("torchvision")
    ok &= check_import("accelerate")
    ok &= check_import("safetensors")

    # 音频依赖
    print("\n--- 音频依赖 ---")
    ok &= check_import("librosa")
    ok &= check_import("soundfile")
    ok &= check_import("qwen-omni-utils", "qwen_omni_utils")

    # GPU
    print("\n--- GPU ---")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  ✅ GPU {i}: {p.name}, {p.total_memory / 1e9:.1f}GB")
        else:
            print("  ❌ CUDA 不可用")
            ok = False
    except Exception as e:
        print(f"  ❌ GPU 检查失败: {e}")
        ok = False

    # 模型文件
    print("\n--- 模型文件 ---")
    import os
    model_path = "/gemini/pretrain"
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        st = [f for f in files if f.endswith(".safetensors")]
        print(f"  ✅ 模型目录: {model_path}")
        print(f"  ✅ safetensors 分片: {len(st)} 个")
        total = sum(os.path.getsize(os.path.join(model_path, f)) for f in st)
        print(f"  ✅ 权重总大小: {total / 1e9:.1f}GB")
    else:
        print(f"  ❌ 模型目录不存在: {model_path}")
        ok = False

    # ffmpeg
    print("\n--- 系统工具 ---")
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f"  ✅ ffmpeg: {r.stdout.splitlines()[0][:50]}")
        else:
            print("  ❌ ffmpeg: 异常")
            ok = False
    except FileNotFoundError:
        print("  ❌ ffmpeg: 未安装 (apt-get install -y ffmpeg)")
        ok = False

    print("\n" + "=" * 50)
    if ok:
        print("✅ 全部检查通过，可以运行模型！")
    else:
        print("❌ 有依赖缺失，请先安装")
    print("=" * 50)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
