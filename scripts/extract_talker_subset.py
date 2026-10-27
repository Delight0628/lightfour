#!/usr/bin/env python3
"""并行版 Talker 子集抽取：把每个张量的字节区间切成块，多线程 Range 下载后 pwrite 到定长文件。

为什么要有 v2（实测数据，2026-10-27）：
  - 本机 egress 上限约 1.1–1.2MB/s（hf-mirror 单连接 216KB/s，8 并发≈370KB/s；
    ModelScope 单连接 373KB/s，8 并发≈955KB/s）。**单连接顺序下载 6.77GB = 11 小时**，不可接受。
  - 因此必须多连接打满这条 10Mbit 管子；两个源（hf-mirror / ModelScope）都可 Range，
    可互为 fallback。
  - 写盘用 os.pwrite（按绝对偏移写），多线程不需要 seek 锁。

布局与 v1 一致：header + 定长 data 段，8 字节对齐；产物可被 safetensors.safe_open 直接读。

用法:
  python scripts/extract_talker_subset.py --threads 12 --include-thinker-embed
  python scripts/extract_talker_subset.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
HEADER_DIR = Path("outputs/hf_shard_headers")
DEFAULT_OUT = Path("models/qwen3_omni_talker_subset/talker_subset.safetensors")
EMBED_KEY = "thinker.model.embed_tokens.weight"
CHUNK = 32 << 20  # 单任务 32MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


def endpoints() -> list[str]:
    raw = os.environ.get(
        "LF_ENDPOINTS",
        "https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/master"
        "|https://hf-mirror.com/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/main",
    )
    return [e.strip().rstrip("/") for e in raw.split("|") if e.strip()]


def load_headers() -> dict[str, dict]:
    out = {}
    for p in sorted(HEADER_DIR.glob("model-*.safetensors.json")):
        h = json.loads(p.read_text(encoding="utf-8"))
        h.pop("__metadata__", None)
        out[p.name[: -len(".json")]] = h
    if not out:
        raise SystemExit(f"没有 header 缓存，先跑 scripts/analyze_hf_shards.py（{HEADER_DIR}）")
    return out


def select(include_embed: bool) -> dict[str, list]:
    headers = load_headers()
    plan: dict[str, list] = {}
    for shard, h in headers.items():
        items = []
        for key, info in h.items():
            if key.startswith("talker.") or (include_embed and key == EMBED_KEY):
                s, e = info["data_offsets"]
                items.append((key, s, e - s, info["dtype"], info["shape"]))
        if items:
            items.sort(key=lambda x: x[1])
            plan[shard] = items
    return plan


def layout_of(plan: dict[str, list]) -> tuple[dict, int, dict]:
    layout, meta, off = {}, {}, 0
    for shard in sorted(plan):
        for key, _s, nbytes, dt, shape in plan[shard]:
            off += (-off) % 8
            layout[key] = (off, nbytes)
            meta[key] = (dt, shape)
            off += nbytes
    return layout, off, meta


class Downloader:
    def __init__(self, fd: int, base: int, plan, layout, threads: int):
        self.fd = fd
        self.base = base
        self.plan = plan
        self.layout = layout
        self.threads = threads
        self.eps = endpoints()
        self.done = 0
        self.total = sum(i[2] for v in plan.values() for i in v)
        self.lock = threading.Lock()
        self.t0 = time.time()

    def _fetch_one(self, shard: str, start: int, length: int) -> bytes:
        last = None
        for attempt in range(1, 6):
            for ep in self.eps:
                url = f"{ep}/{shard}"
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{start + length - 1}"}
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
        raise RuntimeError(f"{shard} [{start},{start + length}) 失败: {last}")

    def run(self) -> None:
        tasks = []
        for shard in sorted(self.plan):
            for key, s, n, _dt, _sh in self.plan[shard]:
                for off in range(0, n, CHUNK):
                    tasks.append((shard, s + off, min(CHUNK, n - off), key, off))
        print(f"任务数 {len(tasks)}（chunk={human(CHUNK)}），线程 {self.threads}", flush=True)
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            futs = {pool.submit(self._fetch_one, t[0], t[1], t[2]): t for t in tasks}
            last = 0.0
            for fut in as_completed(futs):
                shard, start, length, key, key_off = futs[fut]
                data = fut.result()
                dst = self.base + self.layout[key][0] + key_off
                written = 0
                while written < len(data):
                    written += os.pwrite(self.fd, data[written:], dst + written)
                with self.lock:
                    self.done += len(data)
                now = time.time()
                if now - last > 20 or self.done == self.total:
                    el = now - self.t0
                    rate = self.done / max(el, 1e-6)
                    eta = (self.total - self.done) / max(rate, 1)
                    print(
                        f"  {human(self.done)}/{human(self.total)} ({100 * self.done / self.total:.1f}%) "
                        f"{rate / 1e3:.0f}KB/s eta {eta / 60:.0f}min",
                        flush=True,
                    )
                    last = now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--include-thinker-embed", action="store_true")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = select(args.include_thinker_embed)
    total_tensors = sum(len(v) for v in plan.values())
    total_bytes = sum(i[2] for v in plan.values() for i in v)
    print(f"选中 {total_tensors} 张量 / {human(total_bytes)} / 分片 {sorted(plan)}", flush=True)
    for shard in sorted(plan):
        print(f"  {shard}: {len(plan[shard])} tensors / {human(sum(i[2] for i in plan[shard]))}", flush=True)
    print(f"endpoints: {endpoints()}", flush=True)
    if args.dry_run:
        return 0

    layout, data_len, meta = layout_of(plan)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        k: {"dtype": meta[k][0], "shape": meta[k][1],
            "data_offsets": [layout[k][0], layout[k][0] + layout[k][1]]}
        for k in layout
    }
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hbytes += b" " * ((-len(hbytes)) % 8)
    base = 8 + len(hbytes)
    print(f"header {human(len(hbytes))} / data {human(data_len)} / 文件 {human(base + data_len)}", flush=True)

    tmp = out.with_suffix(".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.flush()
        f.truncate(base + data_len)
    fd = os.open(tmp, os.O_WRONLY)
    t0 = time.time()
    try:
        Downloader(fd, base, plan, layout, args.threads).run()
    finally:
        os.close(fd)

    from safetensors import safe_open

    with safe_open(str(tmp), framework="pt") as fh:
        keys = set(fh.keys())
        bad = [
            k for k in layout
            if list(fh.get_slice(k).get_shape()) != list(meta[k][1])
            or fh.get_slice(k).get_dtype() != meta[k][0]
        ]
    missing, extra = set(layout) - keys, keys - set(layout)
    md5 = hashlib.md5()
    with open(tmp, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            md5.update(blk)
    ok = not missing and not extra and not bad
    if ok:
        os.replace(tmp, out)
    print(f"\n{out} size={human(out.stat().st_size) if ok else 'N/A'} md5={md5.hexdigest()}", flush=True)
    print(f"keys={len(keys)} missing={len(missing)} extra={len(extra)} bad_shape={len(bad)} ok={ok}", flush=True)
    payload = {
        "repo": REPO, "endpoints": endpoints(), "path": str(out),
        "size_bytes": (out if ok else tmp).stat().st_size, "md5": md5.hexdigest(),
        "num_tensors": len(keys), "missing": sorted(missing)[:20], "extra": sorted(extra)[:20],
        "bad_shape": bad[:20], "shards_used": sorted(plan), "download_bytes": total_bytes,
        "include_thinker_embed": args.include_thinker_embed,
        "elapsed_s": round(time.time() - t0, 1), "threads": args.threads, "ok": ok,
    }
    Path("outputs/talker_subset_extract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
