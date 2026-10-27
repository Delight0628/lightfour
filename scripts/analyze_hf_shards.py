#!/usr/bin/env python3
"""分析 Qwen3-Omni-30B 的 HF 分片布局：每个 shard 里 talker / thinker / code2wav 各占多少字节。

用途：PI §5.2f 裁决③ —— 若 talker.* 集中在少数 shard，就只下那几个分片；
若与 thinker.* 混装，则报"分片不可拆"，改为下全量或用 safetensors 子集抽取。

做法：不走 HF 下载，直接对 shard 做 HTTP Range 请求拿 safetensors header，
     header 里每个 tensor 的 data_offsets 差就是字节数（精确，不是估算）。

输出：JSON + 文本表，写到 outputs/hf_shard_layout.json。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = os.environ.get("LF_REPO", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
INDEX = os.environ.get(
    "LF_INDEX",
    "/root/.cache/hf/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/"
    "26291f793822fb6be9555850f06dfe95f2d7e695/model.safetensors.index.json",
)
OUT = Path("outputs/hf_shard_layout.json")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


def fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """带重试的 Range 请求。镜像对大 header 会慢，读到一半超时要能续。"""
    import time
    import urllib.request

    buf = bytearray()
    for attempt in range(1, retries + 1):
        want = end - (start + len(buf)) + 1
        if want <= 0:
            break
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start + len(buf)}-{end}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                while want > 0:
                    chunk = resp.read(min(want, 1 << 20))
                    if not chunk:
                        break
                    buf += chunk
                    want -= len(chunk)
            if want <= 0:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"    retry {attempt}/{retries} at +{len(buf)}: {exc}", flush=True)
            time.sleep(2 * attempt)
    if len(buf) != end - start + 1:
        raise RuntimeError(f"range 读取不完整 {len(buf)}/{end - start + 1}")
    return bytes(buf)


def shard_header(name: str) -> dict:
    """读一个 safetensors shard 的 header（只取前 8 字节 + header 本体），带磁盘缓存。"""
    cache = Path("outputs/hf_shard_headers") / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"{ENDPOINT}/{REPO}/resolve/main/{name}"
    n = int.from_bytes(fetch_range(url, 0, 7), "little")
    if n <= 0 or n > 200_000_000:
        raise RuntimeError(f"{name}: 非法 header 长度 {n}")
    raw = fetch_range(url, 8, 8 + n - 1)
    header = json.loads(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(header), encoding="utf-8")
    return header


def group_of(key: str) -> str:
    top = key.split(".")[0]
    if top == "talker":
        return "talker"
    if top == "thinker":
        return "thinker"
    return top


def main() -> int:
    idx = json.load(open(INDEX))
    wm: dict[str, str] = idx["weight_map"]
    shards = sorted(set(wm.values()))
    print(f"repo={REPO} tensors={len(wm)} shards={len(shards)} endpoint={ENDPOINT}", flush=True)

    by_shard: dict[str, dict[str, int]] = {}
    by_shard_count: dict[str, dict[str, int]] = {}
    total_by_group: dict[str, int] = defaultdict(int)
    talker_keys: dict[str, list[str]] = defaultdict(list)

    for i, s in enumerate(shards, 1):
        header = shard_header(s)
        header.pop("__metadata__", None)
        per = defaultdict(int)
        cnt = defaultdict(int)
        for key, info in header.items():
            size = info["data_offsets"][1] - info["data_offsets"][0]
            g = group_of(key)
            per[g] += size
            cnt[g] += 1
            total_by_group[g] += size
            if g == "talker":
                sub = ".".join(key.split(".")[:2])
                talker_keys[sub].append(key)
        by_shard[s] = dict(per)
        by_shard_count[s] = dict(cnt)
        line = " ".join(f"{g}={human(v)}" for g, v in sorted(per.items(), key=lambda kv: -kv[1]))
        print(f"[{i:2d}/{len(shards)}] {s} total={human(sum(per.values()))} | {line}", flush=True)

    print("\n=== 全仓分组字节 ===")
    for g, v in sorted(total_by_group.items(), key=lambda kv: -kv[1]):
        print(f"{g:10s} {human(v)}")

    pure_talker = [s for s, per in by_shard.items() if set(per) == {"talker"}]
    mixed = [s for s, per in by_shard.items() if "talker" in per and len(per) > 1]
    no_talker = [s for s, per in by_shard.items() if "talker" not in per]
    talker_only_bytes = sum(by_shard[s].get("talker", 0) for s in pure_talker)

    print("\n=== 分片可拆性 ===")
    print(f"纯 talker 分片 {len(pure_talker)} 个，合计 talker 字节 {human(talker_only_bytes)}")
    print(f"混装分片 {len(mixed)} 个；不含 talker 的分片 {len(no_talker)} 个")
    splittable = len(mixed) == 0
    print(f"结论: {'分片可拆（只下纯 talker 分片）' if splittable else '分片不可拆（talker 与 thinker 混装）'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo": REPO,
        "endpoint": ENDPOINT,
        "index": INDEX,
        "num_tensors": len(wm),
        "num_shards": len(shards),
        "total_by_group_bytes": dict(total_by_group),
        "per_shard": by_shard,
        "per_shard_count": by_shard_count,
        "pure_talker_shards": pure_talker,
        "mixed_shards": mixed,
        "no_talker_shards": no_talker,
        "splittable": splittable,
        "talker_subgroups": {k: len(v) for k, v in talker_keys.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
