#!/usr/bin/env python3
"""把 Talker 子集权重装进 transformers 的 Talker 模块（含专家融合）。

为什么需要这个文件（实测，2026-10-27）：
  checkpoint 里专家是**逐专家 2D**：`talker.model.layers.N.mlp.experts.{0..127}.{gate,up,down}_proj.weight`；
  模型里专家是**融合 3D**：`...experts.gate_up_proj` [E,2I,H] + `...experts.down_proj` [E,H,I]。
  直接 `load_state_dict` 的结果实测：**40 missing / 7680 unexpected**，且**非专家键 0 差异**
  （40 = 20 层 × 2 融合参数；7680 = 20 × 128 × 3）。
  transformers 的自动融合只挂在 model_type `qwen3_omni_moe` / `qwen3_omni_moe_thinker`（→ qwen2_moe 转换表）上，
  talker 单体的 model_type `qwen3_omni_moe_talker` **不在表里**，所以走 talker-only 加载必须自己融合。

融合规则（与 qwen2_moe 转换表一致，gate 在前）：
  gate_up_proj[e] = cat([gate_proj[e], up_proj[e]], dim=0)   # forward 里 (gate, up) = chunk(2, -1)
  down_proj[e]    = down_proj[e]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)


def load_talker_weights(model, path: str | Path, verbose: bool = True) -> dict:
    """就地加载；返回 {missing, unexpected, merged_experts_layers, tensors_read}。"""
    import torch
    from safetensors import safe_open

    path = str(path)
    sd = model.state_dict()
    with safe_open(path, framework="pt") as f:
        keys = [
            k[len("talker.") :] if k.startswith("talker.") else k
            for k in f.keys()
        ]
        raw_to_key = {
            (k[len("talker.") :] if k.startswith("talker.") else k): k for k in f.keys()
        }

    # 分桶
    layers: dict[str, dict] = {}
    plain: list[str] = []
    unexpected: list[str] = []
    for k in keys:
        m = EXPERT_RE.match(k)
        if m:
            layer, idx, kind = f"model.layers.{m.group(1)}", int(m.group(2)), m.group(3)
            layers.setdefault(layer, {}).setdefault(idx, {})[kind] = k
        elif k in sd:
            plain.append(k)
        else:
            unexpected.append(k)

    n_read = 0
    with safe_open(path, framework="pt") as f:
        for k in plain:
            sd[k].copy_(f.get_tensor(raw_to_key[k]))
            n_read += 1

        merged_layers = 0
        for layer, experts in sorted(layers.items(), key=lambda kv: int(kv[0].split(".")[2])):
            idxs = sorted(experts)
            if idxs != list(range(len(idxs))):
                raise RuntimeError(f"{layer}: 专家编号不连续 {idxs[:5]}...")
            gates, ups, downs = [], [], []
            for i in idxs:
                e = experts[i]
                gates.append(f.get_tensor(raw_to_key[e["gate"]]))
                ups.append(f.get_tensor(raw_to_key[e["up"]]))
                downs.append(f.get_tensor(raw_to_key[e["down"]]))
            gu = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gates, ups)], dim=0)
            dp = torch.stack(downs, 0)
            k_gu, k_dp = f"{layer}.mlp.experts.gate_up_proj", f"{layer}.mlp.experts.down_proj"
            if k_gu not in sd or k_dp not in sd:
                raise RuntimeError(f"{layer}: 模型里没有融合专家参数")
            if tuple(sd[k_gu].shape) != tuple(gu.shape):
                raise RuntimeError(f"{k_gu}: {tuple(gu.shape)} != {tuple(sd[k_gu].shape)}")
            if tuple(sd[k_dp].shape) != tuple(dp.shape):
                raise RuntimeError(f"{k_dp}: {tuple(dp.shape)} != {tuple(sd[k_dp].shape)}")
            sd[k_gu].copy_(gu)
            sd[k_dp].copy_(dp)
            n_read += len(idxs) * 3
            merged_layers += 1
            del gates, ups, downs, gu, dp

    missing = sorted(set(sd) - set(plain) - {f"{l}.mlp.experts.gate_up_proj" for l in layers} - {f"{l}.mlp.experts.down_proj" for l in layers})
    res = {
        "missing": missing,
        "unexpected": sorted(unexpected),
        "merged_experts_layers": merged_layers,
        "tensors_read": n_read,
    }
    if verbose:
        print(
            f"[load_talker_weights] tensors_read={n_read} merged_layers={merged_layers} "
            f"missing={len(missing)} unexpected={len(res['unexpected'])}"
        )
        for k in missing[:10]:
            print(f"  MISSING {k}")
        for k in res["unexpected"][:10]:
            print(f"  UNEXPECTED {k}")
    return res


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/qwen3_omni_config/config.json")
    ap.add_argument("--weights", default="models/qwen3_omni_talker_subset/talker_subset.safetensors")
    ap.add_argument("--out", default="outputs/talker_load_gate.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(args.config)
    model = Qwen3OmniMoeTalkerForConditionalGeneration(cfg.talker_config).eval()
    res = load_talker_weights(model, args.weights)

    # 融合正确性抽检：gate_up_proj[0] 前半 == expert0.gate，后半 == expert0.up
    from safetensors import safe_open

    with safe_open(args.weights, framework="pt") as f:
        gu0 = model.model.layers[0].mlp.experts.gate_up_proj[0]
        g = f.get_tensor("talker.model.layers.0.mlp.experts.0.gate_proj.weight")
        u = f.get_tensor("talker.model.layers.0.mlp.experts.0.up_proj.weight")
        d = f.get_tensor("talker.model.layers.0.mlp.experts.0.down_proj.weight")
    half = g.shape[0]
    checks = {
        "gate_half_equal": torch.equal(gu0[:half], g),
        "up_half_equal": torch.equal(gu0[half:], u),
        "down_equal": torch.equal(model.model.layers[0].mlp.experts.down_proj[0], d),
    }
    heads = getattr(model.code_predictor, "lm_head", None)
    tie = model.codec_head.weight.data_ptr() == model.model.codec_embedding.weight.data_ptr()
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for n, p in model.named_parameters() if "experts" in n)
    two_d = sum(p.numel() for n, p in model.named_parameters() if p.dim() == 2 and "experts" not in n)
    n_2d = sum(1 for n, p in model.named_parameters() if p.dim() == 2 and "experts" not in n)
    print(
        f"融合抽检 {checks}\ncode_predictor.lm_head 个数={len(heads) if heads is not None else None} "
        f"codec_head_tied={tie}\n总参数 {total / 1e9:.3f}B 专家 {experts / 1e9:.3f}B "
        f"非专家2D {two_d / 1e9:.3f}B/{n_2d} 个"
    )
    gate_ok = not res["missing"] and not res["unexpected"] and all(checks.values())
    payload = {
        **res,
        "checks": checks,
        "code_predictor_lm_heads": len(heads) if heads is not None else None,
        "codec_head_tied": tie,
        "total_params_b": round(total / 1e9, 4),
        "expert_params_b": round(experts / 1e9, 4),
        "non_expert_2d_params_b": round(two_d / 1e9, 4),
        "non_expert_2d_count": n_2d,
        "verdict": "PASS" if gate_ok else "FAIL",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} verdict={payload['verdict']}")
    return 0 if gate_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
