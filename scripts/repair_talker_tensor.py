#!/usr/bin/env python3
"""单张量重取修复：只重下坏掉的那一个张量，不整包重抽、不再 truncate-then-fill。

背景（PI ㊱/㊲，2026-10-27）：`talker_subset.safetensors` 里
`talker.model.layers.0.mlp.experts.0.down_proj.weight` 的 absmax = 1.836e+37、std = inf，
其余 8037 张量体检正常。整包重抽 = 6.6GB@1.1MB/s ≈ 100 分钟且会重蹈「先占位再填」的写法，
所以按 header `data_offsets` 定点重取这一块（1024×384 bf16 = 786,432 B）。

安全顺序（agent.md 新规则：尺寸/mtime/文件名都不是判据，先验内容 → 再切换 → 后删除）：
  1. Range 下载到内存，先做 absmax ≤ 1e2 / 全有限 / 0 < std < 1e1 体检；
  2. 再与「同层同角色其它专家 down_proj 的 absmax 中位数」比，偏离 >10× 直接拒写；
  3. 全过才把旧字节备份到 `outputs/corrupt_backup/` 并 `os.pwrite` 原地写回；
  4. 写回后从磁盘重读该区间复检，不通过即报错（备份可一键回滚）。

用法（远端）：
  /root/miniconda3/bin/python repair_talker_tensor.py \
      --key talker.model.layers.0.mlp.experts.0.down_proj.weight --apply
  不加 --apply 为 dry-run（只下载+体检，不落盘）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import struct
import sys
import time
import urllib.request
from pathlib import Path

ABSMAX_LIMIT = 1e2
STD_MAX = 1e1
ROLE_DEVIATION = 10.0


def endpoints() -> list[str]:
    raw = os.environ.get(
        "LF_ENDPOINTS",
        "https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/master"
        "|https://hf-mirror.com/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/main",
    )
    return [e.strip().rstrip("/") for e in raw.split("|") if e.strip()]


def read_subset_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n


def find_in_shards(key: str, header_dir: Path) -> tuple[str, int, int]:
    for p in sorted(header_dir.glob("model-*.safetensors.json")):
        h = json.loads(p.read_text(encoding="utf-8"))
        if key in h:
            s, e = h[key]["data_offsets"]
            return p.name[: -len(".json")], s, e - s
    raise SystemExit(f"分片 header 里找不到 {key}")


def fetch(shard: str, start: int, length: int, tries: int = 5) -> bytes:
    last = None
    for attempt in range(1, tries + 1):
        for ep in endpoints():
            req = urllib.request.Request(
                f"{ep}/{shard}", headers={"Range": f"bytes={start}-{start + length - 1}"}
            )
            try:
                buf = bytearray()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    while len(buf) < length:
                        chunk = resp.read(min(1 << 20, length - len(buf)))
                        if not chunk:
                            break
                        buf += chunk
                if len(buf) == length:
                    return bytes(buf)
                last = f"短读 {len(buf)}/{length} @{ep.split('/')[2]}"
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc} @{ep.split('/')[2]}"
        time.sleep(1.5 * attempt)
    raise SystemExit(f"下载失败 {shard}[{start},{start + length}): {last}")


def health(buf: bytes, dtype: str, shape) -> dict:
    import torch

    if dtype != "BF16":
        raise SystemExit(f"只支持 BF16，收到 {dtype}")
    t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(shape).float()
    finite = bool(t.isfinite().all())
    absmax = float(t.abs().max())
    std = float(t.std(unbiased=False))
    return {
        "n_bytes": len(buf),
        "shape": list(shape),
        "all_finite": finite,
        "absmax": absmax,
        "std": std,
        "mean": float(t.mean()),
        "sha1": hashlib.sha1(buf).hexdigest(),
    }


def role_median(subset_path: Path, layer: int, role: str, exclude: str) -> float:
    """同层同角色其它专家的 absmax 中位数（只读需要的张量，量小）。"""
    import torch
    from safetensors import safe_open

    pref = f"talker.model.layers.{layer}.mlp.experts."
    vals = []
    with safe_open(str(subset_path), framework="pt") as f:
        for k in f.keys():
            if not (k.startswith(pref) and k.endswith(f".{role}_proj.weight")) or k == exclude:
                continue
            vals.append(float(f.get_tensor(k).float().abs().max()))
    return statistics.median(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/qwen3_omni_talker_subset/talker_subset.safetensors")
    ap.add_argument("--key", default="talker.model.layers.0.mlp.experts.0.down_proj.weight")
    ap.add_argument("--header-dir", default="outputs/hf_shard_headers")
    ap.add_argument("--backup-dir", default="outputs/corrupt_backup")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    raise SystemExit(
        "DISABLED (PI 裁决 70, 2026-10-27)：写回分支已废止。\n"
        "  单张量写回会产出「1 个正确 + 8,037 个错位」且可能骗过 (v) 体检；\n"
        "  平移重写/原地写同样禁止（源目标区间重叠，且生成端与读取端不同实现）。\n"
        "  正确路径：修正版 extract_talker_subset 用 safetensors 库读权威分片，\n"
        "  落 models/talker_subset_v2_lib/，旧文件改名 .corrupt。"
    )

    wpath = Path(args.weights)
    hdr, base = read_subset_header(wpath)
    if args.key not in hdr:
        raise SystemExit(f"子集里没有 {args.key}")
    ds, de = hdr[args.key]["data_offsets"]
    dtype, shape = hdr[args.key]["dtype"], hdr[args.key]["shape"]
    nbytes = de - ds
    dst = base + ds
    print(f"[subset] {args.key} dtype={dtype} shape={shape} bytes={nbytes} dst={dst}", flush=True)

    shard, sstart, slen = find_in_shards(args.key, Path(args.header_dir))
    if slen != nbytes:
        raise SystemExit(f"长度不一致：分片 {slen} vs 子集 {nbytes}")
    print(f"[source] {shard}[{sstart},{sstart + slen})", flush=True)

    with open(wpath, "rb") as f:
        f.seek(dst)
        old = f.read(nbytes)
    old_h = health(old, dtype, shape)
    print(f"[old] absmax={old_h['absmax']:.4g} std={old_h['std']:.4g} finite={old_h['all_finite']}", flush=True)

    buf = fetch(shard, sstart, slen)
    new_h = health(buf, dtype, shape)
    print(f"[new] absmax={new_h['absmax']:.4g} std={new_h['std']:.4g} finite={new_h['all_finite']} "
          f"sha1={new_h['sha1'][:12]}", flush=True)

    ok = (new_h["all_finite"] and new_h["absmax"] <= ABSMAX_LIMIT
          and 0 < new_h["std"] < STD_MAX and new_h["sha1"] != old_h["sha1"])
    if not ok:
        print("[abort] 新字节未过体检，不写盘", flush=True)
        return 4

    import re

    m = re.match(r"^talker\.model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$",
                 args.key)
    if m:
        med = role_median(wpath, int(m.group(1)), f"{m.group(2)}_proj", exclude=args.key)
        ratio = new_h["absmax"] / med if med else float("inf")
        print(f"[role] {args.key.split('.')[2]} {m.group(2)} 同角色 absmax 中位={med:.4g} "
              f"本次/中位={ratio:.3f}", flush=True)
        if med and not (1 / ROLE_DEVIATION <= ratio <= ROLE_DEVIATION):
            print("[abort] 偏离同角色中位 >10×，拒绝写入", flush=True)
            return 5
    else:
        print("[role] 非专家张量，跳过同角色比对（只做绝对体检）", flush=True)

    if not args.apply:
        print("[dry-run] 体检通过，未写盘（加 --apply 落盘）", flush=True)
        return 0

    bk = Path(args.backup_dir)
    bk.mkdir(parents=True, exist_ok=True)
    bpath = bk / (args.key.replace("/", "_") + ".corrupt.bin")
    bpath.write_bytes(old)
    print(f"[backup] {bpath} ({len(old)}B) sha1={old_h['sha1'][:12]}", flush=True)

    fd = os.open(wpath, os.O_WRONLY)
    try:
        written = 0
        while written < len(buf):
            written += os.pwrite(fd, buf[written:], dst + written)
    finally:
        os.close(fd)

    with open(wpath, "rb") as f:
        f.seek(dst)
        back = health(f.read(nbytes), dtype, shape)
    print(f"[verify] absmax={back['absmax']:.4g} std={back['std']:.4g} sha1={back['sha1'][:12]}", flush=True)
    if back["sha1"] != new_h["sha1"]:
        raise SystemExit("写回后校验不一致（备份在 outputs/corrupt_backup/）")
    print("[done] 单张量修复落地，下一步跑 gate v4 全量体检", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
