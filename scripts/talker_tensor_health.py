#!/usr/bin/env python3
"""Talker 子集逐张量体检（gate v4 的 (v)+(vi) 两项）。

为什么需要：v3 的 `moe_equiv_max_abs_diff=0.0` 与 `nondegenerate 240/240` 都是无效断言——
快慢两条路径读的是同一个坏张量，坏值也是非零。PI ㊲ 实测
`layers.0.mlp.experts.0.down_proj` absmax=1.836e+37 / std=inf，v3 仍判 PASS。

本脚本不训练、不实例化模型、不依赖 GPU，只做两件事：
  (v)  8038 张量全量体检：absmax ≤ 1e2、0 < std < 1e1、全部 isfinite → `n_corrupt` + `corrupt_list`
  (vi) 同层同角色（layer, role）专家张量 absmax 的 min/median/max，任一偏离中位 >10× → `role_outliers`
verdict 由 (v)(vi) 共同决定，禁止「某项 FAIL 但 verdict=PASS」。

内存口径（v2 教训：8.86GB RSS + 7.27GB page cache 顶爆 16GiB cgroup，oom_kill=3）：
  - 只走 `safe_open` 的 slice 读取，按行切块，单块目标 2^22 个元素；
  - 每 256 张量 `posix_fadvise(DONTNEED)` + `gc.collect()`，把 page cache 还给 cgroup；
  - 不整包 mmap、不 `get_tensor` 整张（最大 embed_tokens 623MB）。

用法（远端）：
  /root/miniconda3/bin/python talker_tensor_health.py --weights models/qwen3_omni_talker_subset/talker_subset.safetensors
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

EXPERT_KEY_RE = re.compile(
    r"^talker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)
ABSMAX_LIMIT = 1e2          # PI ㊲ (v)
STD_MAX = 1e1               # PI ㊲ (v)
STD_MIN = 0.0               # 严格大于 0
ROLE_OUTLIER_RATIO = 10.0   # PI ㊲ (vi)
CHUNK_ELEMS = 1 << 22       # 单块 ~4M 元素


def peak_rss_gb() -> float | None:
    try:
        txt = Path("/proc/self/status").read_text()
        return round(int(txt.split("VmHWM:")[1].split()[0]) / 1024 / 1024, 3)
    except Exception:
        return None


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scan_tensor(f, key: str) -> dict:
    """按行流式读取，返回 absmax / std / finite / numel，不保留数据。"""
    sl = f.get_slice(key)
    shape = list(sl.get_shape())
    n = 1
    for d in shape:
        n *= int(d)
    rows = int(shape[0]) if shape else 1
    per_row = max(1, n // max(1, rows))
    step = max(1, CHUNK_ELEMS // per_row)

    absmax = 0.0
    s = 0.0
    ss = 0.0
    finite = True
    seen = 0
    for i in range(0, rows, step):
        part = sl[i:min(i + step, rows)]
        x = part.to("cpu", non_blocking=False).float()
        flat = x.reshape(-1)
        if not bool(flat.isfinite().all()):
            finite = False
            flat = flat[flat.isfinite()]
            if flat.numel() == 0:
                continue
        absmax = max(absmax, float(flat.abs().max()))
        s += float(flat.sum(dtype=torch.float64))
        ss += float((flat.double() ** 2).sum())
        seen += int(flat.numel())
        del part, x, flat
    mean = s / seen if seen else 0.0
    var = max(0.0, (ss / seen if seen else 0.0) - mean * mean)
    return {
        "key": key,
        "shape": shape,
        "numel": n,
        "absmax": absmax,
        "std": var ** 0.5,
        "all_finite": finite,
        "n_finite": seen,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/qwen3_omni_talker_subset/talker_subset.safetensors")
    ap.add_argument("--out", default="outputs/talker_tensor_health.json")
    ap.add_argument("--flush-every", type=int, default=256)
    args = ap.parse_args()

    from safetensors import safe_open

    wpath = Path(args.weights)
    payload: dict = {
        "code_sha256": code_sha256(),
        "weights": str(wpath),
        "bytes_total": wpath.stat().st_size,
        "thresholds": {
            "absmax_le": ABSMAX_LIMIT,
            "std_between": [STD_MIN, STD_MAX],
            "role_outlier_ratio": ROLE_OUTLIER_RATIO,
        },
        "scope": "all tensors, streaming slice read",
    }

    corrupt: list[dict] = []
    absmax_by_role: dict[str, list[float]] = {}
    n_tensors = 0
    total_elems = 0
    worst = []

    with safe_open(str(wpath), framework="pt") as f, open(wpath, "rb") as raw:
        keys = list(f.keys())
        print(f"[health] tensors={len(keys)}", flush=True)
        for i, key in enumerate(keys):
            st = scan_tensor(f, key)
            n_tensors += 1
            total_elems += st["numel"]
            bad = []
            if not st["all_finite"]:
                bad.append("nonfinite")
            if not (st["absmax"] <= ABSMAX_LIMIT):
                bad.append("absmax")
            if not (STD_MIN < st["std"] < STD_MAX):
                bad.append("std")
            if bad:
                corrupt.append({**st, "reasons": bad})
                print(f"[corrupt] {key} absmax={st['absmax']:.4g} std={st['std']:.4g} {bad}", flush=True)
            m = EXPERT_KEY_RE.match(key)
            if m:
                absmax_by_role.setdefault(f"layers.{m.group(1)}.{m.group(3)}", []).append(st["absmax"])
                worst.append((st["absmax"], key))
            if (i + 1) % args.flush_every == 0:
                try:
                    os.posix_fadvise(raw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
                gc.collect()
                print(f"[health] {i + 1}/{len(keys)} corrupt={len(corrupt)} "
                      f"rss={peak_rss_gb()}GB", flush=True)

    role_rows = []
    role_outliers = []
    for role, vals in sorted(absmax_by_role.items()):
        med = statistics.median(vals)
        lo, hi = min(vals), max(vals)
        row = {"role": role, "n": len(vals), "absmax_min": lo, "absmax_median": med, "absmax_max": hi,
               "max_over_median": hi / med if med else float("inf"),
               "min_over_median": lo / med if med else float("inf")}
        row["outlier"] = bool(med > 0 and (row["max_over_median"] > ROLE_OUTLIER_RATIO
                                          or row["min_over_median"] < 1 / ROLE_OUTLIER_RATIO))
        role_rows.append(row)
        if row["outlier"]:
            role_outliers.append(role)

    payload["n_tensors"] = n_tensors
    payload["n_elems"] = total_elems
    payload["n_corrupt"] = len(corrupt)
    payload["corrupt_list"] = corrupt
    payload["role_absmax"] = role_rows
    payload["role_outliers"] = role_outliers
    payload["v_pass"] = len(corrupt) == 0
    payload["vi_pass"] = len(role_outliers) == 0
    payload["verdict"] = "PASS" if (payload["v_pass"] and payload["vi_pass"]) else "FAIL"
    payload["peak_rss_gb"] = peak_rss_gb()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[health] verdict={payload['verdict']} n_corrupt={len(corrupt)} "
          f"role_outliers={len(role_outliers)} peak_rss={payload['peak_rss_gb']}GB -> {out}", flush=True)
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    sys.exit(main())
