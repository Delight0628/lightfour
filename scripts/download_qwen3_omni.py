"""下载 Qwen3-Omni-30B-A3B-Instruct 到持久化盘。

用途：
  - 服务器上 Qwen3-Omni 30B 权重尚未部署，本脚本负责从 HuggingFace 镜像拉取；
  - 默认写入 `/gemini/code/lightfour-train/models/`（持久化盘，实例重建不丢）；
  - 支持断点续传（snapshot_download 会跳过已完成的文件）。

用法：
  python scripts/download_qwen3_omni.py --dry-run      # 只列文件与总大小
  python scripts/download_qwen3_omni.py                # 实际下载（可后台跑）
  python scripts/download_qwen3_omni.py --repo Qwen/Qwen3-Omni-30B-A3B-Thinking
  python scripts/download_qwen3_omni.py --priority model-00013-of-00015.safetensors,model-00014-of-00015.safetensors,model-00015-of-00015.safetensors

2026-10-27 修复（18:31 下载进程死亡事故）：
  - 强制 HF_HUB_ENABLE_HF_TRANSFER=0：hf_transfer 会直接吃 hf-mirror 回传的
    cas-bridge 预签名 URL，预签名 3600s 过期后重试仍复用旧 URL → 403 AccessDenied
    直接把 snapshot_download 整个打挂（日志 logs/qwen3_omni_download_20261027.log）。
  - 强制 HF_HUB_DISABLE_XET=1（Xet 在本机实测无法真正绕开：mirror 回传的就是 CAS URL）。
  - 新增 --priority 顺序下载 + 单文件重试（每次重试重新取 metadata，拿到新签名 URL）。
  - workers 默认降到 1：实测 mirror HEAD 会 429，并发越高越容易被限流。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# 必须在 import huggingface_hub 之前设置环境变量
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/gemini/code/lightfour-train/.cache/hf")
# 这两项必须强制覆盖，不能用 setdefault：父 shell 里的旧值会原样继承
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

DEFAULT_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_DEST = "/gemini/code/lightfour-train/models/Qwen3-Omni-30B-A3B-Instruct"


def human(n: int) -> str:
    """把字节数格式化成人类可读单位。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--dry-run", action="store_true", help="只列出文件与总大小，不下载")
    ap.add_argument("--workers", type=int, default=1, help="并发下载线程数（mirror 会 429，默认 1）")
    ap.add_argument("--priority", default="", help="逗号分隔的优先文件列表，按给定顺序逐个下载")
    ap.add_argument("--retries", type=int, default=6, help="每个优先文件的单文件重试次数")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    total = sum(size for _, size in files)
    print(f"repo={args.repo} files={len(files)} total={human(total)}")
    for name, size in sorted(files, key=lambda x: -x[1])[:5]:
        print(f"  {human(size):>8}  {name}")

    if args.dry_run:
        return 0

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    status_path = dest / "DEPLOY_STATUS.json"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    prio = [x.strip() for x in args.priority.split(",") if x.strip()]
    prev = None
    if status_path.exists():
        try:
            prev = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            prev = None
    restarts = list((prev or {}).get("restarts", []))
    restarts.append({"at": started, "priority": prio, "workers": args.workers,
                     "hf_transfer": 0, "xet_disabled": 1})
    status_path.write_text(
        json.dumps(
            {"repo": args.repo, "state": "downloading", "started": started,
             "expected_bytes": total, "files_expected": len(files),
             "priority": prio, "workers": args.workers,
             "hf_transfer": 0, "xet_disabled": 1, "restarts": restarts},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"RESTART#{len(restarts)} {started} workers={args.workers} priority={prio}", flush=True)
    print(f"开始下载 -> {dest}（断点续传，HF_ENDPOINT={os.environ['HF_ENDPOINT']}）", flush=True)

    # 第一段：按裁决 103 的顺序逐个抓关键分片（单文件重试，每次重试重新取 metadata 拿新签名 URL）
    for name in prio:
        done = False
        for att in range(1, args.retries + 1):
            try:
                t0 = time.time()
                path = hf_hub_download(repo_id=args.repo, filename=name, local_dir=str(dest))
                size = Path(path).stat().st_size
                print(f"[PRIORITY-OK] {name} {human(size)} {time.time() - t0:.0f}s", flush=True)
                done = True
                break
            except Exception as exc:  # noqa: BLE001 - 网络异常全部重试
                wait = min(30 * 2 ** (att - 1), 600)
                print(f"[PRIORITY-RETRY] {name} att={att}/{args.retries} "
                      f"{type(exc).__name__}: {str(exc)[:160]} wait={wait}s", flush=True)
                time.sleep(wait)
        if not done:
            print(f"[PRIORITY-GIVEUP] {name}", flush=True)

    # 第二段：其余文件（已完成的自动跳过，进行中的 .incomplete 自动续传）
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        max_workers=args.workers,
    )
    # 下载完成后核对文件总大小，防止镜像返回截断文件
    got = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    ok = got >= total * 0.99
    status_path.write_text(
        json.dumps(
            {"repo": args.repo, "state": "done" if ok else "size_mismatch",
             "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
             "expected_bytes": total, "actual_bytes": got,
             "files_expected": len(files), "dest": str(dest)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"完成：{dest} 实际 {human(got)} / 期望 {human(total)}", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
