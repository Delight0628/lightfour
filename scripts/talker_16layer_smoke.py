"""
16 层 codec 训练闭环 smoke：验证 Talker 第 1 层 + code_predictor 15 层能否一起反传。

背景（transformers 5.16.1 源码）：
  - Talker.forward 只算第 1 层：`codec_head(hidden)` → (B,T,3072)，labels 为 (B,T)。
  - 第 2–16 层由 `code_predictor` 预测，它内部是 15 个独立 Embedding + 15 个独立 lm_head。
  - `forward()` 的 `residual_codes` 只被原样回传，**不参与 loss**（源码注释 TODO）。
  所以要训练全 16 层，必须自己在 predictor 上做 teacher forcing。

teacher forcing 对齐 generate 的调用方式：
  step k: 输入 = [talker_hidden, emb(c1), emb[0](c2), ..., emb[k-1](c_{k+1})]
          目标 = c_{k+2}
  prefill 时 predictor 内部 generation_steps = seq_len - 2，正好选中 lm_head[k]。

用法：
  python scripts/talker_16layer_smoke.py
  python scripts/talker_16layer_smoke.py --overfit-steps 30
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path("/gemini/code/lightfour-train")
CFG_SNAPSHOT = BASE / "models/qwen3_omni_config/config.snapshot.json"


def build_models(args):
    from transformers import (Qwen3OmniMoeTalkerConfig,
                              Qwen3OmniMoeTalkerForConditionalGeneration)

    full = json.loads(CFG_SNAPSHOT.read_text(encoding="utf-8"))["talker_config"]
    head_dim = args.hidden // 2
    t = full["text_config"]
    t.update(num_hidden_layers=2, hidden_size=args.hidden,
             intermediate_size=args.hidden * 2,
             shared_expert_intermediate_size=args.hidden,
             moe_intermediate_size=max(args.hidden // 2, 1),
             num_experts=1, num_experts_per_tok=1,
             num_attention_heads=2, num_key_value_heads=1,
             head_dim=head_dim, max_position_embeddings=64)
    cp = full.get("code_predictor_config", {})
    cp.update(num_hidden_layers=1, hidden_size=args.hidden,
              intermediate_size=args.hidden * 2,
              num_attention_heads=2, num_key_value_heads=1,
              head_dim=head_dim)
    if t.get("layer_types"):
        t["layer_types"] = t["layer_types"][:2]
    if cp.get("layer_types"):
        cp["layer_types"] = cp["layer_types"][:1]
    if cp.get("num_code_groups"):
        cp["num_code_groups"] = args.groups

    # mrope_section 的和必须等于 head_dim，否则 RoPE 会产生 NaN
    rs = t.get("rope_scaling") or {}
    sec = rs.get("mrope_section")
    if sec and sum(sec) != head_dim:
        scaled = [max(1, int(round(x * head_dim / sum(sec)))) for x in sec]
        scaled[-1] += head_dim - sum(scaled)
        rs["mrope_section"] = scaled
        t["rope_scaling"] = rs

    cfg = Qwen3OmniMoeTalkerConfig(**full)
    cfg.vocab_size = t.get("vocab_size", 3072)
    model = Qwen3OmniMoeTalkerForConditionalGeneration(cfg)
    return model


def print_params(model, tag):
    groups = {"codec_head": [], "codec_embedding": [], "code_predictor": [], "other": []}
    for n, p in model.named_parameters():
        if "code_predictor" in n:
            groups["code_predictor"].append((n, p.numel()))
        elif n.startswith("codec_head"):
            groups["codec_head"].append((n, p.numel()))
        elif "codec_embedding" in n:
            groups["codec_embedding"].append((n, p.numel()))
        else:
            groups["other"].append((n, p.numel()))
    print(f"--- {tag} 参数分组 ---")
    for g, items in groups.items():
        total = sum(x[1] for x in items)
        print(f"{g:<18} {len(items):>4} 个 / {total/1e6:>8.3f}M")
        for n, c in items[:4]:
            print(f"    {n:<62} {c/1e6:>8.3f}M")
        if len(items) > 4:
            print(f"    ... 另 {len(items)-4} 个")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--overfit-steps", type=int, default=40,
                    help="固定 batch 过拟合步数，验证 loss 能明显下降")
    ap.add_argument("--part", choices=["predictor", "talker", "both"], default="predictor",
                    help="talker=第一层 codec_head；predictor=15 个残差层；both=串起来")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build_models(args)
    print_params(model, "Talker")

    groups = args.groups
    B, T = args.batch, args.seq
    print(f"vocab={model.config.text_config.vocab_size} groups={groups} "
          f"codebook={args.codebook} "
          f"predictor_vocab={model.code_predictor.config.vocab_size} "
          f"num_lm_heads={len(model.code_predictor.lm_head)} "
          f"num_embeds={len(model.code_predictor.model.get_input_embeddings())}", flush=True)

    layer1 = torch.randint(0, args.codebook, (B, T))
    res = torch.randint(0, min(args.codebook, model.code_predictor.config.vocab_size),
                        (B, T, groups - 1))
    ce = nn.CrossEntropyLoss()

    # ---- Part A：Talker 第一层（MoE 路径，可能数值不稳） ----
    attn = torch.ones(B, T, dtype=torch.long)
    l1_loss, l1_ok, hidden_out = None, False, None
    try:
        emb = model.get_input_embeddings()(layer1).detach().clone()
        out = model(inputs_embeds=emb, attention_mask=attn, labels=layer1,
                    output_hidden_states=True)
        l1_loss = out.loss
        l1_ok = l1_loss is not None and torch.isfinite(l1_loss)
        if out.hidden_states and out.hidden_states[0] is not None:
            hidden_out = out.hidden_states[0][-1].detach()
        print(f"Talker 第一层 labels{tuple(layer1.shape)} -> logits{tuple(out.logits.shape)} "
              f"loss={None if l1_loss is None else float(l1_loss):.4f} "
              f"finite={l1_ok}", flush=True)
    except Exception as exc:
        print("Talker 第一层失败:", type(exc).__name__, exc, flush=True)

    # ---- Part B：predictor 15 层（纯 dense，不经过 MoE） ----
    # 模拟 Talker 输出；generate 里 predictor 每步只看 (hidden 最后一个 token, 第1层 token)
    hidden = (hidden_out if (args.part == "both" and hidden_out is not None)
              else torch.randn(B, T, args.hidden) * 0.02)
    h = hidden.reshape(B * T, 1, -1)
    # l1_emb 来自 Talker 的 codec_embedding（不优化），必须 detach，否则多步循环会重复反向同一张图
    l1_emb = model.get_input_embeddings()(layer1.reshape(B * T)).unsqueeze(1).detach()
    res_flat = res.reshape(B * T, groups - 1)
    embeds = model.code_predictor.model.get_input_embeddings()
    pout = model.code_predictor(inputs_embeds=torch.cat([h, l1_emb], dim=1))
    print(f"predictor 第2层 logits{tuple(pout.logits.shape)} "
          f"target{tuple(res_flat[:, 0].shape)}", flush=True)

    @torch.no_grad()
    def predictor_ce():
        tot = 0.0
        for k in range(groups - 1):
            parts = [h, l1_emb]
            for i in range(k):
                parts.append(embeds[i](res_flat[:, i]).unsqueeze(1))
            o = model.code_predictor(inputs_embeds=torch.cat(parts, dim=1))
            tot += float(ce(o.logits[:, -1], res_flat[:, k]))
        return tot / (groups - 1)

    base = predictor_ce()
    print(f"predictor 初始 15 层平均 CE = {base:.4f}（随机猜测应 ≈ ln(2048) = 7.62）",
          flush=True)

    opt = torch.optim.AdamW(model.code_predictor.parameters(), lr=1e-3)
    trace = []
    for step in range(args.overfit_steps):
        loss = 0.0
        for k in range(groups - 1):
            parts = [h, l1_emb]
            for i in range(k):
                parts.append(embeds[i](res_flat[:, i]).unsqueeze(1))
            o = model.code_predictor(inputs_embeds=torch.cat(parts, dim=1))
            loss = loss + ce(o.logits[:, -1], res_flat[:, k])
        loss = loss / (groups - 1)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(model.code_predictor.parameters()), 1.0)
        opt.step()
        trace.append(float(loss))
        if step % 8 == 0 or step == args.overfit_steps - 1:
            print(f"step {step:>3} predictor_loss {float(loss):.4f} gradnorm {float(gn):.2f}",
                  flush=True)

    named = [(n, p) for n, p in model.code_predictor.named_parameters()
             if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"\npredictor 有梯度参数 {len(named)}/{sum(1 for _ in model.code_predictor.parameters())}")
    for key in ("lm_head.0", "lm_head.14", "model.codec_embedding.0",
                "model.codec_embedding.14", "model.layers.0.self_attn.q_proj"):
        hit = [n for n, _ in named if key in n]
        print(f"  {key:<36} {'OK' if hit else 'MISSING'}")

    first, last = trace[0], trace[-1]
    drop = first - last
    print(f"\n第一层 {(None if l1_loss is None else round(float(l1_loss), 3))} | "
          f"predictor {first:.4f} -> {last:.4f}（降 {drop:.3f}）")
    print("PREDICTOR", "OK（15 层 CE 可反传且明显下降）" if drop > 0.3 else "WARN（下降不足）")


if __name__ == "__main__":
    main()
