"""
租卡前的前置检查（只读 config，不下载 60GB 权重）：
  1. transformers 是否真的存在 Qwen3OmniMoeTalkerForConditionalGeneration
  2. Qwen3-Omni config 的 talker_config：层数、码本大小、hidden、codec 帧率
  3. 保存 config 到持久盘，供后续 smoke / 训练使用

用法：
  python scripts/check_qwen3_omni_config.py --repo Qwen/Qwen3-Omni-30B-A3B-Instruct
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/gemini/code/lightfour-train/.cache/hf")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE = Path("/gemini/code/lightfour-train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--out", default=str(BASE / "models/qwen3_omni_config"))
    args = ap.parse_args()

    import transformers
    print("transformers", transformers.__version__, flush=True)

    try:
        from transformers import Qwen3OmniMoeTalkerForConditionalGeneration
        print("Talker class: OK", Qwen3OmniMoeTalkerForConditionalGeneration, flush=True)
    except Exception as exc:
        print("Talker class: MISSING", type(exc).__name__, exc, flush=True)

    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        print("Omni class: OK", flush=True)
    except Exception as exc:
        print("Omni class: MISSING", type(exc).__name__, exc, flush=True)

    from huggingface_hub import hf_hub_download
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(args.repo, "config.json", local_dir=str(out))
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    (out / "config.snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    print("top-level keys:", list(cfg.keys()), flush=True)
    talker = cfg.get("talker_config") or {}
    for k in ("num_hidden_layers", "hidden_size", "vocab_size", "num_codebooks",
              "codebook_size", "codec_chunk_frames", "codec_left_context_frames"):
        if k in talker:
            print(f"talker.{k} =", talker[k], flush=True)
    print("talker keys:", list(talker.keys()), flush=True)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
