#!/usr/bin/env python3
"""Talker 权重加载门禁 v3（低内存版，替代 v2）。

为什么有 v3（2026-10-27 实测 OOM 记录，不许再踩）：
  v2 完整实例化 20 层 talker（bf16 6.6GB）+ 读 7.27GB 权重算 sha256 + 逐层融合，
  在 16GiB cgroup 里被杀：`memory.oom_kill = 3`（v1 `memory.limit_in_bytes=17179869184`）。
  关键点不是模型太大，而是 **page cache 也计入 cgroup**：8.8GB RSS + 7.27GB 文件页
  ≈ 16.1GB，正好越线。所以「上一次能跑过」不能推「这次也不会 OOM」。

v3 的分工（把"全量"和"数值"拆开，各用最省的方式）：
  ① key/shape 层：meta device 实例化全量 20 层（**0 显存**），与 checkpoint 的
     key + shape 集合做全量比对 → missing / unexpected / shape_mismatch 覆盖 8038 张量。
     专家键按融合规则折算成 [E,2I,H] / [E,H,I] 再比。
  ② 数值层：只真实实例化 **1 层**（config 裁到 1），加载第 0 层 128 专家做
     慢路径（逐专家 2D）vs 快路径（融合 3D 切片）逐专家对拍，并做 gate/up 拼接方向抽检。
     等价性是**逐层独立**的性质，融合代码是同一个循环，20 层走同一条路径。

显式声明不是"全量数值验证"：JSON 里写 `numeric_scope = "layers[0] only (1/20)"`，
禁止把 v3 的结果表述成「20 层逐层数值对拍通过」。

用法（远端）：
  /root/miniconda3/bin/python scripts/talker_gate_v3.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

EXPERT_KEY_RE = re.compile(
    r"^talker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)
# 子集里额外带的 thinker 词嵌入：给 TextEmbedder 用，不属于 talker 模型本身
ALLOW_EXTRA = {"thinker.model.embed_tokens.weight"}
LORA_SKIP = ("codec_head",)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
        try:  # 关键：读完把 page cache 还给 cgroup，否则 7.27GB 缓存会把 16GiB 撑爆
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
    return h.hexdigest()


def sha256_code() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def peak_rss_gb() -> float | None:
    try:
        txt = Path("/proc/self/status").read_text()
        return round(int(txt.split("VmHWM:")[1].split()[0]) / 1024 / 1024, 3)
    except Exception:
        return None


def key_plan(ckpt_path: str, full_sd: dict, verbose=True) -> dict:
    """① 全量 key/shape 比对（只读 safetensors header，不读数据）。"""
    from safetensors import safe_open

    plain, experts, unexpected, shape_bad = [], {}, [], []
    with safe_open(ckpt_path, framework="pt") as f:
        for raw in f.keys():
            shape = list(f.get_slice(raw).get_shape())
            m = EXPERT_KEY_RE.match(raw)
            if m:
                layer, idx, kind = int(m.group(1)), int(m.group(2)), m.group(3)
                experts.setdefault(layer, {}).setdefault(idx, {})[kind] = shape
            else:
                k = raw[len("talker."):] if raw.startswith("talker.") else raw
                if k in full_sd:
                    plain.append(k)
                    if list(full_sd[k]) != shape:
                        shape_bad.append({"key": k, "ckpt": shape, "model": list(full_sd[k])})
                elif k not in ALLOW_EXTRA:
                    unexpected.append(k)

    fused = {}
    for layer, idxs in experts.items():
        e0 = idxs[min(idxs)]
        I = e0["gate"][0]
        H = e0["gate"][1]
        fused[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = [len(idxs), 2 * I, H]
        fused[f"model.layers.{layer}.mlp.experts.down_proj"] = [len(idxs), H, I]
        for i, kinds in idxs.items():
            for kind in ("gate", "up", "down"):
                if kinds.get(kind) is None:
                    unexpected.append(f"talker.model.layers.{layer}.mlp.experts.{i}.{kind}_proj.weight (缺)")
    for k, shp in fused.items():
        if k not in full_sd:
            unexpected.append(k + " (融合目标在模型里不存在)")
        elif list(full_sd[k]) != shp:
            shape_bad.append({"key": k, "ckpt": shp, "model": list(full_sd[k])})

    covered = set(plain) | set(fused)
    missing = sorted(set(full_sd) - covered)
    if verbose:
        print(f"[gate1] plain={len(plain)} expert_layers={len(experts)} "
              f"missing={len(missing)} unexpected={len(unexpected)} shape_bad={len(shape_bad)}",
              flush=True)
    return {
        "n_plain": len(plain),
        "n_expert_layers": len(experts),
        "n_expert_tensors": sum(len(v) * 3 for v in experts.values()),
        "missing": missing,
        "unexpected": sorted(unexpected),
        "shape_mismatch": shape_bad,
        "allow_extra": sorted(ALLOW_EXTRA),
        "pass": not missing and not unexpected and not shape_bad,
    }


def lora_plan_meta(rank=16, alpha=32) -> dict:
    """在 meta 全量模型上统计 LoRA 目标（不加载权重）。"""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as TalkerModel,
    )

    cfg = AutoConfig.from_pretrained("models/qwen3_omni_config/config.json")
    with torch.device("meta"):
        m = TalkerModel(cfg.talker_config)
    targets, lora_params, total = [], 0, 0
    for name, mod in m.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(s in name for s in LORA_SKIP):
            continue
        targets.append(name)
        lora_params += rank * (mod.in_features + mod.out_features)
    total = sum(p.numel() for p in m.parameters())
    return {
        "rank": rank,
        "alpha": alpha,
        "n_targets": len(targets),
        "lora_targets": targets,
        "lora_params": lora_params,
        "total_params": total,
        "trainable_params_ratio": round(lora_params / total, 8),
        "note": "3D 融合专家不是 nn.Linear，LoRA 不覆盖；专家冻结时只能说 head+LoRA 可学",
    }


def numeric_layer0(ckpt_path: str, n_split=8, n_equiv=8) -> dict:
    """② 只实例化 1 层，加载 layer 0 的 128 专家做数值对拍。"""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as TalkerModel,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    cfg = AutoConfig.from_pretrained("models/qwen3_omni_config/config.json")
    tcfg = cfg.talker_config
    tcfg.text_config.num_hidden_layers = 1
    model = TalkerModel(tcfg).to(torch.bfloat16).eval()
    H = model.config.text_config.hidden_size
    I = model.config.text_config.moe_intermediate_size

    exp = model.model.layers[0].mlp.experts
    with safe_open(ckpt_path, framework="pt") as f:
        idxs = []
        pref = "talker.model.layers.0.mlp.experts."
        for k in f.keys():
            if k.startswith(pref) and k.endswith(".down_proj.weight"):
                idxs.append(int(k[len(pref):].split(".")[0]))
        idxs.sort()
        assert idxs == list(range(len(idxs))), f"专家编号不连续: {idxs[:5]}"
        gates, ups, downs = [], [], []
        for i in idxs:
            gates.append(f.get_tensor(f"{pref}{i}.gate_proj.weight"))
            ups.append(f.get_tensor(f"{pref}{i}.up_proj.weight"))
            downs.append(f.get_tensor(f"{pref}{i}.down_proj.weight"))
    gu = torch.stack([torch.cat([g, u], 0) for g, u in zip(gates, ups)], 0)
    dp = torch.stack(downs, 0)
    assert tuple(gu.shape) == (len(idxs), 2 * I, H), (gu.shape, len(idxs), 2 * I, H)
    assert tuple(dp.shape) == (len(idxs), H, I), (dp.shape, len(idxs), H, I)
    with torch.no_grad():
        exp.gate_up_proj.copy_(gu)
        exp.down_proj.copy_(dp)
    del gates, ups, downs, gu, dp

    # ③ gate/up 拼接方向抽检：≥8 个，含首末专家
    picks = sorted(set([idxs[0], idxs[-1]] + random.Random(0).sample(idxs, n_split - 2)))
    split_rows = []
    with safe_open(ckpt_path, framework="pt") as f2:
        for e in picks:
            half = I
            g = f2.get_tensor(f"{pref}{e}.gate_proj.weight")
            u = f2.get_tensor(f"{pref}{e}.up_proj.weight")
            d = f2.get_tensor(f"{pref}{e}.down_proj.weight")
            got = exp.gate_up_proj[e]
            split_rows.append({
                "expert": e,
                "gate_half_equal": bool(torch.equal(got[:half], g)),
                "up_half_equal": bool(torch.equal(got[half:], u)),
                "down_equal": bool(torch.equal(exp.down_proj[e], d)),
            })
    split_ok = all(r["gate_half_equal"] and r["up_half_equal"] and r["down_equal"] for r in split_rows)

    # ② 慢路径（逐专家 2D）vs 快路径（融合 3D 切片）
    torch.manual_seed(0)
    x = torch.randn(4, H, dtype=torch.bfloat16)
    eq_picks = sorted(set([idxs[0], idxs[-1]] + random.Random(1).sample(idxs, n_equiv - 2)))
    per = []
    with safe_open(ckpt_path, framework="pt") as f3:
        for e in eq_picks:
            g = f3.get_tensor(f"{pref}{e}.gate_proj.weight").to(torch.bfloat16)
            u = f3.get_tensor(f"{pref}{e}.up_proj.weight").to(torch.bfloat16)
            d = f3.get_tensor(f"{pref}{e}.down_proj.weight").to(torch.bfloat16)
            ref = (torch.nn.functional.silu(x @ g.T) * (x @ u.T)) @ d.T
            slice_gu = exp.gate_up_proj[e].to(torch.bfloat16)
            gw, uw = slice_gu[:I], slice_gu[I:]
            got = (torch.nn.functional.silu(x @ gw.T) * (x @ uw.T)) @ exp.down_proj[e].to(torch.bfloat16).T
            per.append({
                "expert": e,
                "max_abs_diff": float((got.float() - ref.float()).abs().max().item()),
                "ref_abs_max": float(ref.float().abs().max().item()),
            })
    worst = max(p["max_abs_diff"] for p in per)
    return {
        "numeric_scope": "layers[0] only (1/20)",
        "dtype": "bfloat16",
        "tolerance": 1e-3,
        "n_experts_loaded": len(idxs),
        "fused_split_check": {
            "checked": len(picks),
            "experts": picks,
            "includes_first_last": picks[0] == idxs[0] and picks[-1] == idxs[-1],
            "rows": split_rows,
            "pass": split_ok and len(picks) >= 8,
        },
        "moe_equiv": {
            "experts": eq_picks,
            "per_expert": per,
            "max_abs_diff": worst,
            "nondegenerate_ref": min(p["ref_abs_max"] for p in per) > 1e-2,
            "pass": worst <= 1e-3,
        },
    }


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
        Qwen3OmniMoeTalkerForConditionalGeneration as TalkerModel,
    )

    wpath = Path(args.weights)
    payload: dict = {
        "code_sha256": sha256_code(),
        "weights": args.weights,
        "bytes_total": wpath.stat().st_size,
        "dtype": "bfloat16",
        "gate_version": "v3-lowmem",
    }
    print(f"[gate] weights={payload['bytes_total']}B", flush=True)

    # 非退化前置（抽样 240，读少量页）
    with_nz = None
    from safetensors import safe_open
    with safe_open(str(wpath), framework="pt") as f:
        keys = list(f.keys())
        idxs_s = sorted(random.Random(0).sample(range(len(keys)), 240))
        nz = 0
        for i in idxs_s:
            t = f.get_tensor(keys[i]).reshape(-1)[:4096]
            nz += int(torch.count_nonzero(t).item() > 0)
    with_nz = {"sampled": 240, "n_nonzero": nz, "pass": nz == 240}
    payload["nondegenerate"] = with_nz
    print(f"[gate] nondegenerate {nz}/240", flush=True)
    if not with_nz["pass"]:
        payload["verdict"] = "INVALID_ZERO_WEIGHTS"
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[gate] INVALID_ZERO_WEIGHTS", flush=True)
        return 4

    payload["weights_sha256"] = sha256_file(wpath)
    print(f"[gate] sha256={payload['weights_sha256'][:16]}…", flush=True)

    # ① 全量 key/shape（meta，0 显存）
    cfg = AutoConfig.from_pretrained(args.config)
    with torch.device("meta"):
        full = TalkerModel(cfg.talker_config)
    full_sd = {k: list(v.shape) for k, v in full.state_dict().items()}
    del full
    payload["gate1_keys"] = key_plan(str(wpath), full_sd)

    # LoRA 目标（meta 全量）
    payload["lora"] = lora_plan_meta()
    print(f"[gate] lora targets={payload['lora']['n_targets']} "
          f"ratio={payload['lora']['trainable_params_ratio']}", flush=True)

    # ②③ 数值（1 层）
    num = numeric_layer0(str(wpath))
    payload["gate2_moe_equiv"] = num["moe_equiv"]
    payload["gate3_split"] = num["fused_split_check"]
    payload["numeric_scope"] = num["numeric_scope"]
    payload["dtype"] = num["dtype"]
    payload["n_experts_loaded"] = num["n_experts_loaded"]
    print(f"[gate] moe_equiv={num['moe_equiv']['max_abs_diff']} split={num['fused_split_check']['pass']}",
          flush=True)

    # 派生于 PI 验收卡口径的顶层别名
    payload["missing"] = payload["gate1_keys"]["missing"]
    payload["unexpected"] = payload["gate1_keys"]["unexpected"]
    payload["fused_split_check"] = payload["gate3_split"]
    payload["moe_equiv_max_abs_diff"] = payload["gate2_moe_equiv"]["max_abs_diff"]
    payload["lora_targets"] = payload["lora"]["lora_targets"]
    payload["trainable_params_ratio"] = payload["lora"]["trainable_params_ratio"]

    payload["gate_pass"] = bool(
        payload["gate1_keys"]["pass"]
        and payload["gate3_split"]["pass"]
        and payload["gate2_moe_equiv"]["pass"]
        and payload["gate2_moe_equiv"]["nondegenerate_ref"]
    )
    payload["verdict"] = "PASS" if payload["gate_pass"] else "FAIL"
    payload["peak_rss_gb"] = peak_rss_gb()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gate] verdict={payload['verdict']} peak_rss={payload['peak_rss_gb']}GB -> {out}", flush=True)
    return 0 if payload["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
