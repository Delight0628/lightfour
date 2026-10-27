#!/usr/bin/env python3
"""E-D2：清理已落地分片的 .incomplete 残留（裁决 133）。

规则（硬编码，不接受命令行放宽）：
  * 只有当同名沙片的落盘文件 **已通过 sha256 与镜像 manifest 逐字节相等** 时，
    该沙片名下的 .incomplete 才可删；
  * 01/02/13/15 永久保护，任何情况下不动一个字节；
  * 默认 dry-run，只有 --apply 才真删；
  * 删前落 INCOMPLETE_LEDGER.json（含全部条目与判定依据）。

用法：
  python3 incomplete_ledger_d2.py --root /gemini/code/lightfour-train [--apply]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SHARD_RE = re.compile(r"^model-(\d{5})-of-00015\.safetensors$")
PROTECTED = {"model-00001-of-00015.safetensors", "model-00002-of-00015.safetensors",
             "model-00013-of-00015.safetensors", "model-00015-of-00015.safetensors"}
MD5ISH = re.compile(r"\.([0-9a-f]{64})(?:\.[0-9a-f]{8})?\.incomplete$")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gemini/code/lightfour-train")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    model_dir = root / "models/Qwen3-Omni-30B-A3B-Instruct"
    meta = root / "outputs/download_meta"
    meta.mkdir(parents=True, exist_ok=True)

    tree = json.loads((meta / "mirror_tree.json").read_text())
    api = {}
    for s in tree.get("siblings", []):
        lfs = s.get("lfs") or {}
        if s["rfilename"].endswith(".safetensors") and lfs.get("sha256"):
            api[s["rfilename"]] = {"bytes": int(lfs["size"]), "sha256": lfs["sha256"]}
    by_sha = {v["sha256"]: k for k, v in api.items()}

    # E-D1b 已实算过的沙片直接读台账，避免二次读盘与下载抢 IO
    verified = {}
    tsv = meta / "SHARD_SHA256.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2]:
                verified[parts[0]] = {"bytes": int(parts[1]), "sha256": parts[2]}

    entries = []
    for p in sorted((model_dir / ".cache/huggingface/download").glob("*.incomplete")):
        m = MD5ISH.search(p.name)
        blob_sha = m.group(1) if m else None
        target = by_sha.get(blob_sha) if blob_sha else None
        target_path = model_dir / target if target else None
        rec = {
            "path": str(p.relative_to(model_dir)),
            "bytes": p.stat().st_size,
            "blob_sha256": blob_sha,
            "target_shard": target,
            "target_exists": bool(target_path and target_path.exists()),
            "target_bytes": target_path.stat().st_size if (target_path and target_path.exists()) else None,
            "api_bytes": api[target]["bytes"] if target else None,
            "sha_verified": False,
            "action": "KEEP",
            "reason": "",
        }
        if target is None:
            rec["reason"] = "unmapped_blob_sha"
        elif target in PROTECTED:
            rec["reason"] = "protected_shard"
        elif not rec["target_exists"]:
            rec["reason"] = "target_not_landed"
        elif rec["target_bytes"] != rec["api_bytes"]:
            rec["reason"] = "target_size_mismatch"
        else:
            if target in verified:
                rec["sha_verified"] = (verified[target]["bytes"] == rec["target_bytes"]
                                       and verified[target]["sha256"] == api[target]["sha256"])
                rec["sha_source"] = "SHARD_SHA256.tsv(E-D1b)"
            else:
                rec["sha_verified"] = sha256_file(target_path) == api[target]["sha256"]
                rec["sha_source"] = "computed_in_run"
            if rec["sha_verified"]:
                rec["action"] = "DELETE"
                rec["reason"] = "target_sha256_eq_api"
            else:
                rec["reason"] = "target_sha256_mismatch"
        entries.append(rec)

    dele = [e for e in entries if e["action"] == "DELETE"]
    ledger = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": "delete only .incomplete whose target shard sha256 == mirror manifest; 01/02/13/15 untouched",
        "apply": bool(args.apply),
        "n_total": len(entries),
        "n_delete": len(dele),
        "bytes_total": sum(e["bytes"] for e in entries),
        "bytes_delete": sum(e["bytes"] for e in dele),
        "entries": entries,
    }

    if args.apply:
        for e in dele:
            (model_dir / e["path"]).unlink()
        ledger["deleted"] = [e["path"] for e in dele]
        ledger["after_n_total"] = len(list((model_dir / ".cache/huggingface/download").glob("*.incomplete")))
        ledger["after_bytes_total"] = sum(
            p.stat().st_size for p in (model_dir / ".cache/huggingface/download").glob("*.incomplete"))

    out = meta / "INCOMPLETE_LEDGER.json"
    out.write_text(json.dumps(ledger, indent=1, ensure_ascii=False))
    print(json.dumps({k: ledger[k] for k in ledger if k != "entries"}, indent=1, ensure_ascii=False))
    for e in entries:
        print(f'{e["action"]:6s} {e["bytes"]:>12d} {e["target_shard"] or "-":42s} {e["reason"]}')
    print("ledger:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
