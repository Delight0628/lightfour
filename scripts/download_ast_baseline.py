#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备 ast_lin_v0 的 AST 冻结骨干：MIT/ast-finetuned-audioset-10-10-0.4593（rev f826b80d）。

两种取权重方式，落盘位置相同，最终都可用 --ast-model-dir 指向的本地目录加载：

  A. --from-remote  从训练服务器（virtaicloud-delight）已有的 HF 缓存里拉 3 个文件。
     本地网络当前到 huggingface.co / hf-mirror.com 都不通（2026-10-27 实测超时），
     但训练服务器上有完整 snapshot，故用这条路径；走的是 lf_remote.py 的 SFTP 通道。
  B. 默认           直接从 HF hub 下载（需要能访问 HF 或镜像），写进 <repo>/.cache/hf。

产出目录（本地模型目录，非 HF cache 布局，from_pretrained 可直接吃）：
  .cache/ast_baseline/config.json
  .cache/ast_baseline/preprocessor_config.json
  .cache/ast_baseline/model.safetensors
  .cache/ast_baseline/PROVENANCE.json     （模型 id / revision / 来源 / md5）

用法:
  python scripts/download_ast_baseline.py --check-only
  python scripts/download_ast_baseline.py --from-remote
  python scripts/download_ast_baseline.py                  # 需要能连 HF

校验：--check-only 会真加载一次 ASTFeatureExtractor + ASTForAudioClassification。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
REMOTE_HUB = ("/gemini/code/lightfour-train/.cache/hf/hub/"
              "models--MIT--ast-finetuned-audioset-10-10-0.4593")
FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(out_dir: Path) -> int:
    if not out_dir.is_dir():
        print("CHECK-ONLY FAIL: 目录不存在", out_dir)
        return 2
    missing = [f for f in FILES if not (out_dir / f).is_file()]
    if missing:
        print("CHECK-ONLY FAIL: 缺文件", missing)
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(str(out_dir))
    md = ASTForAudioClassification.from_pretrained(str(out_dir))
    print("CHECK-ONLY OK")
    print("  id2label n =", len(md.config.id2label))
    print("  hidden_size =", md.config.hidden_size,
          "num_hidden_layers =", getattr(md.config, "num_hidden_layers", "?"))
    print("  fbank max_length =", fe.max_length, "sampling_rate =", fe.sampling_rate)
    del fe, md
    return 0


def from_remote(out_dir: Path) -> int:
    spec = importlib_spec_lf_remote()
    m = spec
    creds = m.load_credentials(m.Path(str(m.DEFAULT_ENV_FILE)))
    client = m.connect(creds)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        transport = client.get_transport()
        sftp = __import__("paramiko").SFTPClient.from_transport(transport)
        try:
            for name in FILES:
                remote = f"{REMOTE_HUB}/snapshots/{REVISION}/{name}"
                local = out_dir / name
                print(f"pulling {remote} -> {local}", flush=True)
                sftp.get(remote, str(local))
                print(f"  {local.stat().st_size} bytes  md5={md5_file(local)}", flush=True)
        finally:
            sftp.close()
    finally:
        client.close()
    (out_dir / "PROVENANCE.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source": "remote_hf_cache",
        "remote": f"{REMOTE_HUB}/snapshots/{REVISION}",
        "files": {f: {"bytes": (out_dir / f).stat().st_size,
                      "md5": md5_file(out_dir / f)} for f in FILES},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FROM-REMOTE OK ->", out_dir)
    return 0


def importlib_spec_lf_remote():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lf_remote", str(ROOT / "scripts" / "lf_remote.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def from_hub(out_dir: Path) -> int:
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "hf")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL_ID, revision=REVISION))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(snap / name, out_dir / name)
        print(f"  {name}: {(out_dir / name).stat().st_size} bytes  md5={md5_file(out_dir / name)}")
    (out_dir / "PROVENANCE.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": REVISION,
        "source": "hf_hub", "snapshot": str(snap),
        "files": {f: {"bytes": (out_dir / f).stat().st_size,
                      "md5": md5_file(out_dir / f)} for f in FILES},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FROM-HUB OK ->", out_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / ".cache" / "ast_baseline"))
    ap.add_argument("--from-remote", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.check_only:
        return check(out_dir)
    rc = from_remote(out_dir) if args.from_remote else from_hub(out_dir)
    if rc == 0:
        rc = check(out_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
