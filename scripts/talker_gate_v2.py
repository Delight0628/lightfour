#!/usr/bin/env python3
"""Talker 权重加载门禁 v2（PI 裁决⑯三条门槛 + 非退化前置断言）。

为什么要有 v1 的教训（2026-10-27 实测）：
  `talker_subset.safetensors` 在抽取未完成时就已经是 **header + 全零占位** 的
  7.27GB 文件（脚本先 write(header) 再 truncate 到全尺寸，再按 chunk 填数据）。
  v1 门禁直接读它，得到 0 missing / 0 unexpected、moe_equiv=0.0、split 8/8 ——
  **全零权重下这些指标必然成立**，等价于零对零。因此 v2 先把「权重非退化」
  作为前置断言，不通过就直接判 INVALID，不算任何后续指标。

三条门槛（PI 裁决⑯）：
  ① 融合后 strict=True → 0 missing / 0 unexpected
  ② 数值等价：融合 3D 算一层 MoE 前向 vs 逐专家 2D 显式循环，max_abs_diff ≤ 1e-3(bf16)
  ③ gate/up 拼接方向逐位抽检，≥8/128 且含首末专家（0 与 127）
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

EXPERT_KEY_RE = re.compile(r"^talker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
LORA_SKIP = ("codec_head",)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_code() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def check_nondegenerate(path: str, n_sample: int = 64, seed: int = 0) -> dict:
    """按字节偏移分层抽样，确认数据区不是全零/半截。"""
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        spans = []
        for k in keys:
            s = f.get_slice(k)
            spans.append((k, s.get_shape(), str(s.get_dtype())))
        # 按 key 顺序近似等价于按字节顺序；分层取 n_sample 个
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(keys)), min(n_sample, len(keys))))
        bad = []
        worst = None
        for i in idxs:
            k = keys[i]
            t = f.get_tensor(k).reshape(-1)
            head = t[: min(t.numel(), 8192)]
            nz = int(torch.count_nonzero(head).item())
            if nz == 0:
                bad.append({"key": k, "order": i, "nonzero_in_head": 0})
            elif worst is None:
                worst = {"key": k, "order": i, "nonzero_in_head": nz}
    n_total = len(keys)
    tail_covered = max(idxs) >= int(n_total * 0.9) if idxs else False
    return {
        "n_tensors": n_total,
        "n_sampled": len(idxs),
        "sampled_max_order": max(idxs) if idxs else None,
        "tail_90pct_covered": tail_covered,
        "n_zero_tensors": len(bad),
        "zero_examples": bad[:5],
        "first_nonzero_example": worst,
        "pass": len(bad) == 0 and tail_covered,
    }


def expert_keys(path: str) -> dict[int, dict[int, dict[str, str]]]:
    from safetensors import safe_open

    out: dict[int, dict[int, dict[str, str]]] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            m = EXPERT_KEY_RE.match(k)
            if not m:
                continue
            layer, idx, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            out.setdefault(layer, {}).setdefault(idx, {})[kind] = k
    return out


def split_direction_check(model, path: str, layers: dict, n_check: int = 8) -> dict:
    """③ gate/up 拼接方向逐位抽检：gate_up_proj[e] 前半 == gate[e]，后半 == up[e]。"""
    import torch
    from safetensors import safe_open

    layer = max(layers)  # 用最后一层（下沉到文件尾部，最不易被半截污染）
    idxs = sorted(layers[layer])
    want = n_check
    picks = sorted(set([idxs[0], idxs[-1]] + random.Random(0).sample(idxs, min(want, len(idxs)) - 2)))
    picks = picks[:max(n_check, 2)]
    results = []
    with safe_open(path, framework="pt") as f:
        gu_all = model.model.layers[layer].mlp.experts.gate_up_proj
        dp_all = model.model.layers[layer].mlp.experts.down_proj
        for e in picks:
            g = f.get_tensor(layers[layer][e]["gate"])
            u = f.get_tensor(layers[layer][e]["up"])
            d = f.get_tensor(layers[layer][e]["down"])
            gu = gu_all[e]
            half = g.shape[0]
            results.append({
                "expert": e,
                "gate_half_equal": bool(torch.equal(gu[:half], g)),
                "up_half_equal": bool(torch.equal(gu[half:], u)),
                "down_equal": bool(torch.equal(dp_all[e], d)),
            })
    ok = all(r["gate_half_equal"] and r["up_half_equal"] and r["down_equal"] for r in results)
    return {
        "layer": layer,
        "checked": len(picks),
        "experts": picks,
        "includes_first_last": picks[0] == idxs[0] and picks[-1] == idxs[-1],
        "results": results,
        "pass": ok and len(picks) >= 8 and picks[0] == idxs[0] and picks[-1] == idxs[-1],
    }


def moe_equiv_check(model, path: str, layers: dict, n_experts: int = 8, seed: int = 0) -> dict:
    """② 融合 3D 前向 vs 逐专家 2D 显式循环，逐专家 max_abs_diff。"""
    import torch
    from safetensors import safe_open

    torch.manual_seed(seed)
    layer = max(layers)
    H = model.config.text_config.hidden_size
    I = model.config.text_config.moe_intermediate_size
    x = torch.randn(4, H, dtype=torch.bfloat16)
    idxs = sorted(layers[layer])
    picks = sorted(set([idxs[0], idxs[-1]] + random.Random(1).sample(idxs, n_experts - 2)))[:n_experts]
    per_expert = []
    with safe_open(path, framework="pt") as f:
        gu_all = model.model.layers[layer].mlp.experts.gate_up_proj
        dp_all = model.model.layers[layer].mlp.experts.down_proj
        for e in picks:
            g = f.get_tensor(layers[layer][e]["gate"]).to(torch.bfloat16)
            u = f.get_tensor(layers[layer][e]["up"]).to(torch.bfloat16)
            d = f.get_tensor(layers[layer][e]["down"]).to(torch.bfloat16)
            # 参照实现：逐专家 2D 显式循环
            ref = torch.nn.functional.silu(x @ g.T) * (x @ u.T)
            ref = ref @ d.T
            # 被测实现：模型里的融合 3D 切片，与 MoE forward 内部同样的切分方式
            gu = gu_all[e].to(torch.bfloat16)
            gate_w, up_w = gu[:I], gu[I:]
            got = torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)
            got = got @ dp_all[e].to(torch.bfloat16).T
            per_expert.append({
                "expert": e,
                "max_abs_diff": float((got.float() - ref.float()).abs().max().item()),
                "ref_abs_max": float(ref.float().abs().max().item()),
            })
    worst = max(p["max_abs_diff"] for p in per_expert) if per_expert else None
    tol = 1e-3  # bf16
    return {
        "layer": layer,
        "dtype": "bfloat16",
        "tolerance": tol,
        "experts": picks,
        "per_expert": per_expert,
        "max_abs_diff": worst,
        "pass": worst is not None and worst <= tol,
        "note": "ref_abs_max 用于证明不是零对零（非退化对照）",
    }


def lora_plan(model, rank: int = 16, alpha: int = 32, skip: tuple[str, ...] = LORA_SKIP) -> dict:
    """v0 = bf16 + LoRA + gradient checkpointing：统计 LoRA 目标与可训练占比。"""
    import torch

    targets = []
    lora_params = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        targets.append(name)
        lora_params += rank * (mod.in_features + mod.out_features)
    total = sum(p.numel() for p in model.parameters())
    return {
        "rank": rank,
        "alpha": alpha,
        "n_targets": len(targets),
        "targets": targets,
        "lora_params": lora_params,
        "total_params": total,
        "trainable_ratio": round(lora_params / total, 8),
        "note": "3D 融合专家参数不是 nn.Linear，LoRA 不覆盖；专家全冻结时只能说 head+LoRA 可学",
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/qwen3_omni_config/config.json")
    ap.add_argument("--weights", default="models/qwen3_omni_talker_subset/talker_subset.safetensors")
    ap.add_argument("--out", default="outputs/talker_load_gate.json")
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--n-equiv", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from talker_loader import load_talker_weights

    payload: dict = {
        "weights": args.weights,
        "weights_bytes": Path(args.weights).stat().st_size,
        "weights_sha256": sha256_file(args.weights),
        "code_sha256": sha256_code(),
        "dtype": "bfloat16",
    }

    # 前置：非退化断言
    nd = check_nondegenerate(args.weights)
    payload["nondegenerate"] = nd
    if not nd["pass"]:
        payload["verdict"] = "INVALID_ZERO_WEIGHTS"
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[gate] 非退化断言失败 verdict=INVALID_ZERO_WEIGHTS -> {args.out}")
        return 4

    cfg = AutoConfig.from_pretrained(args.config)
    model = Qwen3OmniMoeTalkerForConditionalGeneration(cfg.talker_config).eval()
    payload["codec_head_tied_declared"] = getattr(
        Qwen3OmniMoeTalkerForConditionalGeneration, "_tied_weights_keys", None
    )
    payload["codec_head_tied_before"] = bool(
        model.codec_head.weight.data_ptr() == model.model.codec_embedding.weight.data_ptr()
    )
    model.tie_weights()
    payload["codec_head_tied_after"] = bool(
        model.codec_head.weight.data_ptr() == model.model.codec_embedding.weight.data_ptr()
    )
    payload["tie_word_embeddings_in_cfg"] = getattr(cfg.talker_config, "tie_word_embeddings", None)

    res = load_talker_weights(model, args.weights)
    payload["missing"] = res["missing"]
    payload["unexpected"] = res["unexpected"]
    payload["tensors_loaded"] = res["tensors_read"]
    payload["layers_fused"] = res["merged_experts_layers"]
    payload["gate1_pass"] = not res["missing"] and not res["unexpected"]

    layers = expert_keys(args.weights)
    sd3 = split_direction_check(model, args.weights, layers, n_check=args.n_split)
    payload["gate3_split"] = sd3
    eq = moe_equiv_check(model, args.weights, layers, n_experts=args.n_equiv)
    payload["gate2_equiv"] = eq

    payload["lora"] = lora_plan(model, rank=args.lora_rank)
    payload["total_params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 4)
    payload["expert_params_b"] = round(
        sum(p.numel() for n, p in model.named_parameters() if "experts" in n) / 1e9, 4
    )
    payload["gate_pass"] = bool(payload["gate1_pass"] and sd3["pass"] and eq["pass"])
    payload["verdict"] = "PASS" if payload["gate_pass"] else "FAIL"
    payload["peak_rss_gb"] = round(
        int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) / 1024 / 1024, 3
    ) if Path("/proc/self/status").exists() else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gate] verdict={payload['verdict']} gate1={payload['gate1_pass']} "
          f"gate2={eq['pass']}({eq['max_abs_diff']}) gate3={sd3['pass']}({sd3['checked']}) -> {args.out}")
    return 0 if payload["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
