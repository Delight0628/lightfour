#!/usr/bin/env python3
"""校验 Talker 子集：真实权重能否 0 missing / 0 unexpected 装进 transformers 的 Talker 模块。

同时落盘四件 PI 裁决③/① 需要的事实：
  1. state_dict 键集合 vs 子集键集合（missing / unexpected）
  2. codec_head 是否真的 tied 到 model.codec_embedding.weight（加载后 data_ptr 相同）
  3. code_predictor.lm_head 的个数（PI 口径 15 个残差层，这里数一遍）
  4. 参数结构：总参数 / 融合专家参数 / nn.Linear 覆盖量（判 bnb 4bit 能不能压）

用法:
  python scripts/verify_talker_subset.py --dry-keys   # 只比对键，不建模型（省内存）
  python scripts/verify_talker_subset.py             # 建模型 + 真装权重
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

INDEX = os.environ.get(
    "LF_INDEX",
    "/root/.cache/hf/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/"
    "26291f793822fb6be9555850f06dfe95f2d7e695/model.safetensors.index.json",
)
CFG = os.environ.get("LF_OMNI_CFG", "models/qwen3_omni_config/config.json")
SUB = os.environ.get("LF_TALKER_SUB", "models/qwen3_omni_talker_subset/talker_subset.safetensors")
OUT = Path("outputs/talker_subset_verify.json")
CACHE_KEYS = Path("outputs/talker_state_dict_keys.json")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


def build_talker(cfg):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as M

    cls = getattr(M, "Qwen3OmniMoeTalkerForConditionalGeneration", None)
    if cls is None:
        raise SystemExit("找不到 Qwen3OmniMoeTalkerForConditionalGeneration")
    print(f"talker class: {cls.__name__}")
    return cls(cfg.talker_config)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-keys", action="store_true")
    ap.add_argument("--from-index", action="store_true",
                    help="不读 safetensors，直接用 model.safetensors.index.json 的 weight_map 取键（下载未完成时用）")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CFG)
    print(f"subset: {SUB} exists={Path(SUB).exists()}")
    if not args.from_index and not Path(SUB).exists():
        print("子集还不存在（下载未完成）")
        return 4

    if args.from_index:
        idx = json.loads(Path(INDEX).read_text(encoding="utf-8"))
        all_keys = list(idx["weight_map"])
        talker_keys = {k[len("talker.") :]: k for k in all_keys if k.startswith("talker.")}
        other = [k for k in all_keys if not k.startswith("talker.")]
        print(f"[index 模式] {INDEX}: 全仓 {len(all_keys)} 键")
    else:
        with safe_open(SUB, framework="pt") as f:
            sub_keys = list(f.keys())
        talker_keys = {k[len("talker.") :]: k for k in sub_keys if k.startswith("talker.")}
        other = [k for k in sub_keys if not k.startswith("talker.")]
        print(f"[safetensors 模式] {SUB}")
    print(f"张量 {len(talker_keys) + len(other)}：talker.* {len(talker_keys)}，其它 {len(other)}")
    by_group: dict[str, int] = {}
    for k in talker_keys:
        g = k.split(".")[0]
        by_group[g] = by_group.get(g, 0) + 1
    print(f"talker 子组: {by_group}")

    if CACHE_KEYS.exists():
        model_keys = set(json.loads(CACHE_KEYS.read_text(encoding="utf-8")))
        print(f"复用缓存的 state_dict 键 {len(model_keys)}")
        model = None
    else:
        model = build_talker(cfg)
        sd = model.state_dict()
        model_keys = set(sd.keys())
        CACHE_KEYS.parent.mkdir(parents=True, exist_ok=True)
        CACHE_KEYS.write_text(json.dumps(sorted(model_keys), indent=0), encoding="utf-8")
        print(f"state_dict 键 {len(model_keys)}，已缓存 {CACHE_KEYS}")

    missing = sorted(model_keys - set(talker_keys))
    unexpected = sorted(set(talker_keys) - model_keys)
    print(f"\nmissing(model 有/权重无)={len(missing)}  unexpected(权重有/model 无)={len(unexpected)}")
    for k in missing[:20]:
        print(f"  MISSING {k}")
    for k in unexpected[:20]:
        print(f"  UNEXPECTED {k}")

    payload = {
        "subset": SUB, "from_index": args.from_index,
        "subset_tensors": len(talker_keys) + len(other),
        "talker_tensors": len(talker_keys),
        "other_tensors": other,
        "talker_subgroups": by_group,
        "model_keys": len(model_keys),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_head": missing[:50],
        "unexpected_head": unexpected[:50],
    }

    if args.dry_keys or missing or unexpected:
        OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {OUT}")
        return 0 if not (missing or unexpected) else 3

    # 真装权重
    if model is None:
        model = build_talker(cfg)
    sd = model.state_dict()
    with safe_open(SUB, framework="pt") as f:
        for name in list(sd.keys()):
            sd[name].copy_(f.get_tensor(talker_keys[name]))
    res = model.load_state_dict(sd, strict=True)
    print(f"load_state_dict(strict=True): {res}")

    # codec_head tied 验证
    ch = model.codec_head.weight
    ce = model.model.codec_embedding.weight
    tied_ptr = ch.data_ptr() == ce.data_ptr()
    print(f"codec_head tied to codec_embedding: data_ptr_equal={tied_ptr} equal={torch.equal(ch, ce)}")
    print(f"codec_head shape={tuple(ch.shape)} codec_embedding shape={tuple(ce.shape)}")

    # code_predictor lm_head 个数
    heads = getattr(model.code_predictor, "lm_head", None)
    n_heads = len(heads) if heads is not None else None
    print(f"code_predictor.lm_head 个数={n_heads}，num_code_groups={cfg.talker_config.num_code_groups}")
    print(f"lm_head[0] shape={tuple(heads[0].weight.shape)}")

    # 参数结构（对齐 PI 裁决① 的 4bit 判断）
    total = sum(p.numel() for p in model.parameters())
    experts = 0
    lin = 0
    n_lin = 0
    for name, p in model.named_parameters():
        if "experts" in name:
            experts += p.numel()
        if p.dim() == 2 and "experts" not in name:
            lin += p.numel()
            n_lin += 1
    print(f"总参数 {total / 1e9:.3f}B；融合专家 {experts / 1e9:.3f}B；非专家 2D(≈nn.Linear) {lin / 1e9:.3f}B/{n_lin} 个")

    payload.update(
        {
            "dtype": str(model.codec_head.weight.dtype),
            "codec_head_tied": tied_ptr,
            "codec_head_shape": list(ch.shape),
            "codec_embedding_shape": list(ce.shape),
            "code_predictor_lm_heads": n_heads,
            "total_params_b": round(total / 1e9, 4),
            "expert_params_b": round(experts / 1e9, 4),
            "non_expert_2d_params_b": round(lin / 1e9, 4),
            "non_expert_2d_count": n_lin,
            "verdict": "PASS",
        }
    )
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT} verdict=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
