#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分块跑 ast_lin_v0 的特征抽取（逐条落盘，断点续跑）。

单独一个薄启动器的原因：本机只有 ~1 GiB 空闲物理内存，一次性跑完 1000+ 条会在
torch CPU forward 中随机 ACCESS_VIOLATION。分块 + 逐条缓存可以把每次进程的存活
时间压到最短，崩了重跑即续。

用法:
  python scripts/featurize_chunk.py --set perturb --max-items 40
  python scripts/featurize_chunk.py --set train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_ast_lin_v0 import stage_featurize, DEFAULT_AST_DIR, DEFAULT_CACHE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True,
                    choices=["train", "probe", "perturb"])
    ap.add_argument("--max-items", type=int, default=48)
    ap.add_argument("--ast-model-dir", default=str(DEFAULT_AST_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    return stage_featurize(args.set_name, Path(args.cache_dir), Path(args.ast_model_dir),
                           args.batch_size, args.threads, max_items=args.max_items)


if __name__ == "__main__":
    sys.exit(main())
