#!/usr/bin/env python3
"""E-D1a 替代举证：hf-mirror 与 ModelScope 两源 manifest 逐片对账。

背景：远端到 huggingface.co 的 TCP 建连被 DNS 污染（解析到 2a03:2880::/32），
无法直取上游 API。本脚本用 ModelScope（另一个独立 CDN / 不同运营方）的
文件清单作为**旁证源**，逐片比 sha256 与 size。注意：按裁决 112，
ModelScope 出处仍是 provisional，只有在 HF API 可达后做 3/3 全等才算追溯生效。

用法：python3 ed1_meta_compare.py --dir /gemini/code/lightfour-train/outputs/download_meta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARDS = [f"model-{i:05d}-of-00015.safetensors" for i in range(1, 16)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/gemini/code/lightfour-train/outputs/download_meta")
    args = ap.parse_args()
    d = Path(args.dir)

    mt = json.loads((d / "mirror_tree.json").read_text())
    mirror = {}
    for s in mt.get("siblings", []):
        lfs = s.get("lfs") or {}
        mirror[s["rfilename"]] = {"bytes": int(lfs.get("size") or s.get("size") or 0),
                                  "sha256": (lfs.get("sha256") or "")}

    ms_raw = json.loads((d / "ms_files.json").read_text())
    files = ((ms_raw.get("Data") or {}).get("Files")) or []
    ms = {}
    for f in files:
        name = f.get("Path") or f.get("Name")
        ms[name] = {"bytes": int(f.get("Size") or 0),
                    "sha256": (f.get("Sha256") or f.get("SHA256") or "").lower()}

    rows = []
    n_eq = 0
    for sh in SHARDS:
        a, b = mirror.get(sh, {}), ms.get(sh, {})
        eq = bool(a.get("sha256")) and a.get("sha256") == b.get("sha256") and a.get("bytes") == b.get("bytes")
        n_eq += eq
        rows.append({"shard": sh, "mirror_bytes": a.get("bytes"), "ms_bytes": b.get("bytes"),
                     "mirror_sha256": a.get("sha256"), "ms_sha256": b.get("sha256"), "EQ": eq})

    out = {
        "mirror_model_sha": mt.get("sha"),
        "mirror_last_modified": mt.get("lastModified"),
        "n_shards_compared": len(SHARDS),
        "n_eq": n_eq,
        "upstream_reachable": False,
        "upstream_note": "huggingface.co TCP connect timeout; DNS -> 2a03:2880:f10f:83:face:b00c:0:25de (polluted)",
        "rows": rows,
    }
    (d / "META_CROSSCHECK.json").write_text(json.dumps(out, indent=1))
    print(f"{'shard':14s} {'mirror_B':>12s} {'ms_B':>12s} {'EQ':>3s}")
    for r in rows:
        print(f"{r['shard'][6:11]:14s} {r['mirror_bytes']!s:>12s} {r['ms_bytes']!s:>12s} "
              f"{'Y' if r['EQ'] else 'N':>3s}")
    print(f"EQ {n_eq}/{len(SHARDS)}  mirror_sha={out['mirror_model_sha']}")
    return 0 if n_eq == len(SHARDS) else 1


if __name__ == "__main__":
    sys.exit(main())
