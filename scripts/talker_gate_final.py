#!/usr/bin/env python3
"""Talker 权重加载门禁 · 合并版（v2 + v3 并列 + 第四门 forward 对拍）。

背景（PI 裁决 186 / #1 硬阻塞）：
  v2（全量 20 层融合）与 v3（meta 全量 key/shape + 1 层数值）是两份半成品，
  必须合成**一份** `outputs/talker_load_gate.json`：
    - v2 / v3 并列保留原始读数，不覆盖、不改写；
    - `allow_extra=["thinker.model.embed_tokens.weight"]` 与 missing/unexpected 分开计数；
    - 顶层 verdict 取四门里最严的一份。

第四门为什么必须单独做（v2/v3 都没做）：
  v2/v3 的"数值等价"只比到 **张量级**（2D 逐专家 vs 3D 切片，同一个矩阵乘）。
  真正的 forward 还包含：router（softmax/top-k/renorm）→ 派发（index_add 顺序）
  → shared expert → sigmoid 门控 → 相加。任何一环错都不会在张量级暴露。
  本门用 **checkpoint 里的原始 2D 权重 + 原始 router/shared 权重**，手写一条
  与模块内部逐位同序的慢路径，和 `layer.mlp(x)`（融合 3D 快路径）对拍。

口径声明：
  - 只做 **layer 0 / 1 层模型**，结果不外推到 20 层（`numeric_scope` 字段明写）；
  - 快慢两路用**同一份 checkpoint 权重**，差异只可能来自融合/派发实现；
  - 覆盖度必须 128/128 专家，否则加 token 重跑，达不到就判 FAIL。

用法（远端）：
  /root/miniconda3/bin/python scripts/talker_gate_final.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ALLOW_EXTRA = ["thinker.model.embed_tokens.weight"]
CONFIG = "models/qwen3_omni_config/config.json"
WEIGHTS = "models/talker_subset_v2_shift/talker_subset.safetensors"
V2_JSON = "outputs/archive_pre_merge_gate/talker_load_gate_v2_raw.json"
V3_JSON = "outputs/talker_load_gate_v3.json"
OUT = "outputs/talker_load_gate.json"
TRAIN_SCRIPT = "scripts/train_omni_talker.py"
SMOKE_JSON = "checkpoints/tiny_retie_smoke/summary.json"

TOL_FORWARD = 1e-2  # bf16 单层 MoE 输出 O(1)；逐位同序时实测应为 0.0


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
        try:  # 7.27GB 文件页会计入 cgroup，读完立刻还给内核
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


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.exists() else BASE / p


def _read_json_opt(p: str) -> dict | None:
    q = _resolve(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.exists() else None


def _sha256_opt(p: str) -> str | None:
    q = _resolve(p)
    return sha256_file(q) if q.exists() else None


def build_one_layer(config: str):
    """只实例化 1 层（config 裁到 1），bf16 直接构建，避免 fp32→bf16 的 2 倍峰值。"""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as TalkerModel,
    )

    cfg = AutoConfig.from_pretrained(config)
    cfg.talker_config.text_config.num_hidden_layers = 1
    lt = getattr(cfg.talker_config.text_config, "layer_types", None)
    if lt:
        cfg.talker_config.text_config.layer_types = lt[:1]
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = TalkerModel(cfg.talker_config)
    finally:
        torch.set_default_dtype(old)
    return model.to(torch.bfloat16).eval()


def load_layer0_mlp(model, weights: str) -> dict:
    """把 layer 0 的 router / shared / 128 专家全部从 checkpoint 装进模型（融合 3D）。"""
    import torch
    from safetensors import safe_open

    pref = "talker.model.layers.0.mlp."
    blk = model.model.layers[0].mlp
    H = model.config.text_config.hidden_size
    I = model.config.text_config.moe_intermediate_size

    with safe_open(weights, framework="pt") as f:
        idxs = sorted(
            int(k[len(pref + "experts."):].split(".")[0])
            for k in f.keys()
            if k.startswith(pref + "experts.") and k.endswith(".down_proj.weight")
        )
        assert idxs == list(range(len(idxs))), f"专家编号不连续: {idxs[:5]}"
        router = f.get_tensor(pref + "gate.weight").to(torch.bfloat16)
        wsg = f.get_tensor(pref + "shared_expert.gate_proj.weight").to(torch.bfloat16)
        wsu = f.get_tensor(pref + "shared_expert.up_proj.weight").to(torch.bfloat16)
        wsd = f.get_tensor(pref + "shared_expert.down_proj.weight").to(torch.bfloat16)
        wgate = f.get_tensor(pref + "shared_expert_gate.weight").to(torch.bfloat16)

        gates, ups, downs = [], [], []
        for i in idxs:
            gates.append(f.get_tensor(f"{pref}experts.{i}.gate_proj.weight").to(torch.bfloat16))
            ups.append(f.get_tensor(f"{pref}experts.{i}.up_proj.weight").to(torch.bfloat16))
            downs.append(f.get_tensor(f"{pref}experts.{i}.down_proj.weight").to(torch.bfloat16))

    gu = torch.stack([torch.cat([g, u], 0) for g, u in zip(gates, ups)], 0)
    dp = torch.stack(downs, 0)
    assert tuple(gu.shape) == (len(idxs), 2 * I, H), (gu.shape, len(idxs), 2 * I, H)
    assert tuple(dp.shape) == (len(idxs), H, I), (dp.shape, len(idxs), H, I)

    with torch.no_grad():
        blk.gate.weight.copy_(router)
        blk.shared_expert.gate_proj.weight.copy_(wsg)
        blk.shared_expert.up_proj.weight.copy_(wsu)
        blk.shared_expert.down_proj.weight.copy_(wsd)
        blk.shared_expert_gate.weight.copy_(wgate)
        blk.experts.gate_up_proj.copy_(gu)
        blk.experts.down_proj.copy_(dp)

    # 融合结果与 checkpoint 逐位核对（不是 shape 核对）
    bitwise = {
        "gate_up_equal": bool(torch.equal(gu, blk.experts.gate_up_proj)),
        "down_equal": bool(torch.equal(dp, blk.experts.down_proj)),
    }
    del gates, ups, downs, gu, dp
    return {
        "layer": 0,
        "n_experts": len(idxs),
        "hidden": H,
        "intermediate": I,
        "dtype": "bfloat16",
        "fused_bitwise_equal_ckpt": bitwise,
    }


def _slow_path(x, topv, topi, ckpt_weights, model, pref, E, H):
    """慢路径：与模块内部逐位同序，但权重来自 checkpoint 的 2D 张量。"""
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    with safe_open(ckpt_weights, framework="pt") as f:
        wsg = f.get_tensor(pref + "shared_expert.gate_proj.weight").to(torch.bfloat16)
        wsu = f.get_tensor(pref + "shared_expert.up_proj.weight").to(torch.bfloat16)
        wsd = f.get_tensor(pref + "shared_expert.down_proj.weight").to(torch.bfloat16)
        wgate = f.get_tensor(pref + "shared_expert_gate.weight").to(torch.bfloat16)

    final = torch.zeros_like(x)
    hit = sorted(set(topi.reshape(-1).tolist()))
    per_expert = []
    with safe_open(ckpt_weights, framework="pt") as f:
        for e in hit:
            # 模块内部顺序：one_hot(topi).permute(2,1,0)[e] → where → (top_k_pos, token_idx)
            mask_e = (topi == e).T
            pos, tok = mask_e.nonzero(as_tuple=True)
            xs = x[tok]
            g = f.get_tensor(f"{pref}experts.{e}.gate_proj.weight").to(torch.bfloat16)
            u = f.get_tensor(f"{pref}experts.{e}.up_proj.weight").to(torch.bfloat16)
            d = f.get_tensor(f"{pref}experts.{e}.down_proj.weight").to(torch.bfloat16)
            gate, up = F.linear(xs, torch.cat([g, u], 0)).chunk(2, dim=-1)
            h = F.silu(gate) * up
            h = F.linear(h, d)
            h = h * topv[tok, pos, None]
            final.index_add_(0, tok, h.to(final.dtype))
            per_expert.append({"expert": e, "n_tokens": int(tok.numel()),
                               "out_abs_max": round(float(h.float().abs().max()), 5)})
            del g, u, d, gate, up, h, xs

    shared = F.linear(F.silu(F.linear(x, wsg)) * F.linear(x, wsu), wsd)
    shared = torch.sigmoid(F.linear(x, wgate)) * shared
    return final + shared, hit, per_expert


def gate4_forward(model, weights: str, base_tokens=4096, max_tokens=16384, seed=0):
    """第四门：模块 forward（融合 3D）vs 手写慢路径（checkpoint 2D），最多覆盖 128 专家。"""
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    blk = model.model.layers[0].mlp
    cfg = model.config.text_config
    H = cfg.hidden_size
    E = blk.experts.num_experts
    k = blk.gate.top_k
    pref = "talker.model.layers.0.mlp."

    with safe_open(weights, framework="pt") as f:
        router_w = f.get_tensor(pref + "gate.weight").to(torch.bfloat16)

    torch.manual_seed(seed)
    attempts = []
    n_tok, result = base_tokens, None
    while n_tok <= max_tokens:
        x = (torch.randn(n_tok, H, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        with torch.no_grad():
            y_fast = blk(x.view(1, n_tok, H))
            if isinstance(y_fast, tuple):
                y_fast = y_fast[0]
            y_fast = y_fast.view(n_tok, H)

            # router 复算（与模块 router 逐位对拍）
            logits = F.linear(x, router_w)
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            topv, topi = torch.topk(probs, k, dim=-1)
            if blk.gate.norm_topk_prob:
                topv = topv / topv.sum(dim=-1, keepdim=True)
            topv = topv.to(logits.dtype)
            _, rw_gate, se_gate = blk.gate(x)
            router_match = bool(torch.equal(topi, se_gate)) and bool(torch.equal(topv, rw_gate))

            y_ref, hit, per_expert = _slow_path(x, topv, topi, weights, model, pref, E, H)

        diff = (y_fast.float() - y_ref.float()).abs()
        rec = {
            "n_tokens": n_tok,
            "n_experts_hit": len(hit),
            "experts_missing": sorted(set(range(E)) - set(hit)),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "ref_abs_max": float(y_ref.float().abs().max().item()),
            "bitwise_equal": bool(torch.equal(y_fast, y_ref)),
            "router_topk_scores_match": router_match,
        }
        attempts.append(rec)
        result = (rec, per_expert)
        if len(hit) == E and router_match:
            break
        del x, y_fast, y_ref, diff
        n_tok *= 2

    rec, per_expert = result
    # 每专家 dispatch 计数（证明不是少数专家撑起来的）
    counts = {str(e): 0 for e in range(E)}
    for p in per_expert:
        counts[str(p["expert"])] = p["n_tokens"]
    nondeg = rec["ref_abs_max"] > 1e-2
    passed = (
        rec["max_abs_diff"] <= TOL_FORWARD
        and rec["n_experts_hit"] == E
        and rec["router_topk_scores_match"]
        and nondeg
    )
    return {
        "gate": "gate4_forward",
        "numeric_scope": "layers[0] only (1/1 layer instantiated; talker has 20)",
        "dtype": "bfloat16",
        "top_k": int(k),
        "n_experts": int(E),
        "tolerance": TOL_FORWARD,
        "attempts": attempts,
        "final": rec,
        "expert_dispatch_counts": counts,
        "nondegenerate_ref": nondeg,
        "pass": bool(passed),
        "note": "慢路径权重取自 checkpoint 2D；快路径为 layer.mlp 融合 3D；同序 index_add",
    }


def codec_head_tie_block(v2: dict) -> dict:
    """codec_head ↔ codec_embedding：声明态 / 训练脚本实现 / 实跑证据。"""
    smoke = _read_json_opt(SMOKE_JSON)
    ev = (smoke or {}).get("codec_head_tie")
    ok = bool(ev and ev.get("data_ptr_equal") and ev.get("n_calls", 0) >= 2)
    return {
        "declared_in_model": v2.get("codec_head_tied_declared"),
        "tied_before_model_tie_weights": v2.get("codec_head_tied_before"),
        "tied_after_model_tie_weights": v2.get("codec_head_tied_after"),
        "tie_word_embeddings_in_cfg": v2.get("tie_word_embeddings_in_cfg"),
        "contract": "transformers tie_weights 未接通 → train_omni_talker.py 每步显式 re-tie + data_ptr 断言（失败 exit 9）",
        "train_script": TRAIN_SCRIPT,
        "train_script_sha256": _sha256_opt(TRAIN_SCRIPT),
        "retie_evidence": ev,
        "evidence_source": SMOKE_JSON,
        "evidence_scope": "tiny 2 层/1 专家 random-init 2 步（只证明 re-tie 代码路径与断言，不代表 30B）",
        "status": "PASS_WITH_RETIE_AND_ASSERT" if ok else "OPEN_UNTIL_RETIE_EVIDENCE",
    }


def merge(v2: dict, v3: dict, gate4: dict) -> dict:
    """v2 / v3 并列 + allow_extra 单列 + 四门取最严。"""
    allow = set(ALLOW_EXTRA)

    # ---- gate1
    v2_unexpected = list(v2.get("unexpected", []))
    v2_allow = sorted(k for k in v2_unexpected if k in allow)
    v2_true = sorted(k for k in v2_unexpected if k not in allow)
    g1_v2 = {
        "missing": len(v2.get("missing", [])),
        "unexpected_raw": len(v2_unexpected),
        "unexpected_allowlisted": v2_allow,
        "unexpected_true": v2_true,
        "n_unexpected_allowlisted": len(v2_allow),
        "n_unexpected_true": len(v2_true),
        "pass": not v2.get("missing") and not v2_true,
        "scope": "20 层全量实例化 + 真实融合",
    }
    g1_v3 = {
        "missing": len(v3["gate1_keys"]["missing"]),
        "unexpected": len(v3["gate1_keys"]["unexpected"]),
        "unexpected_allowlisted": sorted(
            k for k in v3["gate1_keys"]["unexpected"] if k in allow
        ),
        "shape_mismatch": len(v3["gate1_keys"]["shape_mismatch"]),
        "n_plain": v3["gate1_keys"]["n_plain"],
        "n_expert_layers": v3["gate1_keys"]["n_expert_layers"],
        "n_expert_tensors": v3["gate1_keys"]["n_expert_tensors"],
        "allow_extra": v3["gate1_keys"].get("allow_extra", []),
        "pass": v3["gate1_keys"]["pass"],
        "scope": "meta 全量 key/shape（0 显存）+ 1 层真实数值",
    }

    # ---- gate2
    g2_v2 = {
        "max_abs_diff": v2["gate2_equiv"]["max_abs_diff"],
        "tolerance": v2["gate2_equiv"]["tolerance"],
        "layer": v2["gate2_equiv"]["layer"],
        "n_experts": len(v2["gate2_equiv"]["experts"]),
        "dtype": v2["gate2_equiv"]["dtype"],
        "pass": v2["gate2_equiv"]["pass"],
        "scope": "20 层融合权重里最末层的 8 专家，张量级",
    }
    g2_v3 = {
        "max_abs_diff": v3["gate2_moe_equiv"]["max_abs_diff"],
        "tolerance": 1e-3,
        "layer": 0,
        "n_experts": len(v3["gate2_moe_equiv"]["experts"]),
        "dtype": v3["dtype"],
        "nondegenerate_ref": v3["gate2_moe_equiv"]["nondegenerate_ref"],
        "pass": v3["gate2_moe_equiv"]["pass"],
        "scope": "layers[0] 的 8 专家，张量级",
    }

    # ---- gate3
    g3_v2 = {
        "layer": v2["gate3_split"]["layer"],
        "checked": v2["gate3_split"]["checked"],
        "includes_first_last": v2["gate3_split"]["includes_first_last"],
        "pass": v2["gate3_split"]["pass"],
    }
    g3_v3 = {
        "layer": 0,
        "checked": v3["gate3_split"]["checked"],
        "includes_first_last": v3["gate3_split"]["includes_first_last"],
        "experts": v3["gate3_split"]["experts"],
        "pass": v3["gate3_split"]["pass"],
    }

    gates = {
        "gate1_keys_shape": {"v2": g1_v2, "v3": g1_v3, "pass": g1_v2["pass"] and g1_v3["pass"]},
        "gate2_numeric_tensor": {"v2": g2_v2, "v3": g2_v3,
                                 "pass": g2_v2["pass"] and g2_v3["pass"]},
        "gate3_split_direction": {"v2": g3_v2, "v3": g3_v3,
                                  "pass": g3_v2["pass"] and g3_v3["pass"]},
        "gate4_forward": gate4,
    }
    verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"

    return {
        "schema": "talker_load_gate/merged/v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": {"merge_script": sha256_code(),
                        "v2": v2.get("code_sha256"), "v3": v3.get("code_sha256")},
        "weights": v3["weights"],
        "weights_bytes": v3["bytes_total"],
        "weights_sha256": v3["weights_sha256"],
        "allow_extra": ALLOW_EXTRA,
        "counts": {
            "missing": g1_v3["missing"],
            "unexpected_raw_v2": g1_v2["unexpected_raw"],
            "unexpected_allowlisted": len(v2_allow),
            "unexpected_true": len(v2_true),
            "shape_mismatch": g1_v3["shape_mismatch"],
            "note": "allow_extra 单列计数，不计入 unexpected_true",
        },
        "gates": gates,
        "verdict": verdict,
        "verdict_rule": "四门全过才 PASS；v2/v3 任一门 FAIL 即 FAIL（取最严）",
        "variant_verdicts": {
            "v2_raw": v2.get("verdict"),
            "v2_raw_reason": "唯一 FAIL 项 = 1 个 allow_extra 张量；按裁决 186 单列后为 PASS",
            "v3_raw": v3.get("verdict"),
            "merged": verdict,
        },
        "numeric_scope": {
            "gate2": "v3=layers[0]/8 专家；v2=layer 19/8 专家（张量级）",
            "gate4": "layers[0]/128 专家（forward 级，含 router+dispatch+shared）",
            "not_claimed": "不声称 20 层逐层 forward 对拍",
        },
        "codec_head_tie": codec_head_tie_block(v2),
        "lora": {
            "n_targets": v3["lora"]["n_targets"],
            "lora_targets": v3["lora"]["lora_targets"],
            "lora_params": v3["lora"]["lora_params"],
            "total_params": v3["lora"]["total_params"],
            "trainable_params_ratio": v3["lora"]["trainable_params_ratio"],
            "rank": v3["lora"]["rank"],
            "alpha": v3["lora"]["alpha"],
        },
        "peak_rss_gb": peak_rss_gb(),
        "sources": {
            "v2": V2_JSON,
            "v3": V3_JSON,
            "v2_verdict": v2.get("verdict"),
            "v3_verdict": v3.get("verdict"),
        },
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(BASE / CONFIG) if not Path(CONFIG).exists() else CONFIG)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--v2", default=V2_JSON)
    ap.add_argument("--v3", default=V3_JSON)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-gate4", action="store_true")
    ap.add_argument("--base-tokens", type=int, default=4096)
    args = ap.parse_args()

    cfg_path = args.config
    if not Path(cfg_path).exists() and Path(BASE / CONFIG).exists():
        cfg_path = str(BASE / CONFIG)
    wpath = args.weights if Path(args.weights).exists() else str(BASE / WEIGHTS)
    v2p = args.v2 if Path(args.v2).exists() else str(BASE / V2_JSON)
    v3p = args.v3 if Path(args.v3).exists() else str(BASE / V3_JSON)
    outp = args.out if Path(args.out).parent.exists() else str(BASE / OUT)

    for p, name in ((cfg_path, "config"), (wpath, "weights"), (v2p, "v2"), (v3p, "v3")):
        if not Path(p).exists():
            raise SystemExit(f"[fatal] 缺 {name}: {p}")

    v2 = json.loads(Path(v2p).read_text(encoding="utf-8"))
    v3 = json.loads(Path(v3p).read_text(encoding="utf-8"))
    print(f"[merge] v2={v2.get('verdict')} v3={v3.get('verdict')}", flush=True)

    if args.skip_gate4:
        gate4 = {"pass": False, "skipped": True}
    else:
        t0 = time.time()
        model = build_one_layer(cfg_path)
        meta = load_layer0_mlp(model, wpath)
        print(f"[gate4] layer0 loaded experts={meta['n_experts']} "
              f"fused_bitwise={meta['fused_bitwise_equal_ckpt']}", flush=True)
        gate4 = gate4_forward(model, wpath, base_tokens=args.base_tokens)
        gate4["fused_bitwise_equal_ckpt"] = meta["fused_bitwise_equal_ckpt"]
        gate4["n_experts_loaded"] = meta["n_experts"]
        gate4["sec"] = round(time.time() - t0, 1)
        print(f"[gate4] tokens={gate4['final']['n_tokens']} "
              f"experts={gate4['final']['n_experts_hit']}/{gate4['n_experts']} "
              f"max_abs_diff={gate4['final']['max_abs_diff']} pass={gate4['pass']}", flush=True)
        del model

    payload = merge(v2, v3, gate4)
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[merge] verdict={payload['verdict']} -> {outp}", flush=True)
    return 0 if payload["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
