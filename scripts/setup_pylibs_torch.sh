#!/usr/bin/env sh
# 重建 /root/pylibs_torch —— CPU torch 隔离环境（一键、可复现、不碰系统 site-packages）。
#
# 为什么需要：容器重启会清空 /root。2026-10-28 10:22 重启后
#   /root/pylibs_torch 与 /root/pylibs_hf034 **都已消失**（agent.md 裁决 163 只批准了安装，
#   没有脚本化 = 容器重建即静默坏）。本脚本是那次安装的可重建版本。
#
# 约束（裁决 163）：
#   1. 只装进独立目录（默认 /root/pylibs_torch），`PYTHONPATH` 前置；
#   2. **禁改系统 site-packages**：全程 `pip --target`，不加 `--user`、不写 miniconda；
#   3. 用 `--ignore-installed` 让目标目录自足（否则 pip 认为系统里已满足，目录会缺包，
#      换到干净容器就 import 失败）；
#   4. 装完自证：`torch.__version__ == 2.8.0+cpu` 且 `torch.cuda.is_available() is False`，
#      并写出 outputs/env_fingerprint_torch.json（实测值，不是声称值）。
#
# 用法：
#   sh scripts/setup_pylibs_torch.sh                 # 重建 + 自证 + 写指纹
#   sh scripts/setup_pylibs_torch.sh --verify-only   # 只自证 + 写指纹（不装）
#   TARGET=/root/pylibs_torch sh scripts/setup_pylibs_torch.sh
set -eu

TARGET=${TARGET:-/root/pylibs_torch}
PY=${PY:-/root/miniconda3/bin/python}
ROOT=${ROOT:-/gemini/code/lightfour-train}
OUT=${OUT:-$ROOT/outputs/env_fingerprint_torch.json}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
VERIFY_ONLY=0

for a in "$@"; do
  case "$a" in
    --verify-only) VERIFY_ONLY=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

echo "[1/3] target=$TARGET py=$PY"

if [ "$VERIFY_ONLY" -eq 0 ]; then
  # 顺序很关键（2026-10-28 实测）：torch 的**纯 python 依赖**（sympy/networkx/jinja2/
  # filelock/fsspec/typing_extensions）必须走 tuna。若跟着 `--index-url download.pytorch.org`
  # 一起解析，sympy 6.3MB 会卡在同一台 R2 节点上（本轮实测 >7 分钟零字节、反复重试）。
  echo "[2/4] install torch deps (index: $PIP_INDEX)"
  "$PY" -m pip install --no-cache-dir --ignore-installed \
      --index-url "$PIP_INDEX" --target "$TARGET" \
      "sympy" "networkx" "jinja2" "filelock" "fsspec" "typing_extensions"

  echo "[3/4] install torch==2.8.0+cpu --no-deps (index: $TORCH_INDEX)"
  if [ -n "${TORCH_WHEEL:-}" ] && [ -f "$TORCH_WHEEL" ]; then
    # 断点续装：download.pytorch.org 的 R2 节点会长时间零字节（2026-10-28 实测两次）。
    # 已下完整轮子时直接装本地文件，fingerprint 里会如实写 torch_install_source。
    echo "       using local wheel: $TORCH_WHEEL"
    "$PY" -m pip install --no-cache-dir --ignore-installed --no-deps --no-index \
        --target "$TARGET" "$TORCH_WHEEL"
  else
    "$PY" -m pip install --no-cache-dir --ignore-installed --no-deps \
        --index-url "$TORCH_INDEX" --target "$TARGET" "torch==2.8.0+cpu"
  fi

  echo "[3/4] install transformers==4.57.1 librosa soundfile (index: $PIP_INDEX)"
  "$PY" -m pip install --no-cache-dir --ignore-installed \
      --index-url "$PIP_INDEX" --target "$TARGET" \
      "transformers==4.57.1" "librosa" "soundfile"
else
  echo "[2/4] --verify-only: skip install"
fi

echo "[4/4] verify + fingerprint -> $OUT"
mkdir -p "$(dirname "$OUT")"
PYTHONPATH="$TARGET" TARGET="$TARGET" OUT="$OUT" ROOT="$ROOT" \
TORCH_INDEX="$TORCH_INDEX" PIP_INDEX="$PIP_INDEX" TORCH_WHEEL="${TORCH_WHEEL:-}" "$PY" - <<'PYEOF'
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

target = Path(os.environ["TARGET"]).resolve()
out = Path(os.environ["OUT"])
root = Path(os.environ["ROOT"])

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

torch_path = Path(torch.__file__).resolve()
assert str(torch_path).startswith(str(target)), \
    f"torch 不是从 {target} 导入的: {torch_path}"
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.__version__.endswith("+cpu"), torch.__version__
assert torch.cuda.is_available() is False, "CPU 轮子却报告 CUDA 可用，拒绝签收"

py_files = [p for p in target.rglob("*") if p.is_file()]
size = sum(p.stat().st_size for p in py_files)
script = root / "scripts/setup_pylibs_torch.sh"
sha = hashlib.sha256(script.read_bytes()).hexdigest() if script.exists() else None

fp = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "kind": "pylibs_torch (CPU torch, isolated --target dir)",
    "exe": sys.executable,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "target_path": str(target),
    "target_size_bytes": size,
    "target_n_files": len(py_files),
    "torch_version": torch.__version__,
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "torch_device_count": torch.cuda.device_count(),
    "torch_file": str(torch_path),
    "transformers_version": transformers.__version__,
    "numpy_version": np.__version__,
    "librosa_version": librosa.__version__,
    "soundfile_version": soundfile.__version__,
    "torch_install_source": (
        f"local wheel {os.environ['TORCH_WHEEL']}" if os.environ.get("TORCH_WHEEL")
        else f"index {os.environ['TORCH_INDEX']}"),
    "install_cmd": (
        f"{sys.executable} -m pip install --no-cache-dir --ignore-installed "
        f"--index-url {os.environ['PIP_INDEX']} --target {target} "
        f"sympy networkx jinja2 filelock fsspec typing_extensions && "
        f"{sys.executable} -m pip install --no-cache-dir --ignore-installed --no-deps "
        f"--index-url {os.environ['TORCH_INDEX']} --target {target} torch==2.8.0+cpu && "
        f"{sys.executable} -m pip install --no-cache-dir --ignore-installed "
        f"--index-url {os.environ['PIP_INDEX']} --target {target} "
        f"transformers==4.57.1 librosa soundfile"
    ),
    "script_sha256": sha,
    "import_guard": "assert torch.__file__ startswith target; assert +cpu; assert cuda False",
}
out.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(fp, ensure_ascii=False, indent=1))
print("ENV_OK", torch.__version__, "cuda=", torch.cuda.is_available())
PYEOF
