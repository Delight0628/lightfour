"""
Talker 前向 smoke（不加载 60GB 权重）：用 tiny config 验证 16 层 codec logits 与 loss。

用法：
  python scripts/talker_forward_smoke.py --layers 16 --codebook 1024

判定：
  - logits 形状包含 (B, T, layers, codebook) 或 (B, layers, T, codebook)
  - loss 为有限值，且能 backward 到梯度
  - 打印 forward 的真实签名，供后续训练脚本对齐
"""
import argparse
import inspect

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seq", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    import transformers
    print("transformers", transformers.__version__, flush=True)
    from transformers import Qwen3OmniMoeTalkerConfig, Qwen3OmniMoeTalkerForConditionalGeneration
    print("forward signature:", inspect.signature(
        Qwen3OmniMoeTalkerForConditionalGeneration.forward), flush=True)

    import json
    from pathlib import Path

    cfg_path = Path("/gemini/code/lightfour-train/models/qwen3_omni_config/config.snapshot.json")
    if not cfg_path.exists():
        raise SystemExit("先跑 check_qwen3_omni_config.py 生成 config.snapshot.json")
    full = json.loads(cfg_path.read_text(encoding="utf-8"))["talker_config"]
    t = full["text_config"]
    t.update(num_hidden_layers=2, hidden_size=args.hidden, intermediate_size=args.hidden * 2,
             shared_expert_intermediate_size=args.hidden, moe_intermediate_size=args.hidden // 2,
             num_experts=2, num_experts_per_tok=1, num_attention_heads=2,
             num_key_value_heads=1, head_dim=args.hidden // 2, max_position_embeddings=64)
    cp = full.get("code_predictor_config", {})
    cp.update(num_hidden_layers=1, hidden_size=args.hidden, intermediate_size=args.hidden * 2,
              num_attention_heads=2, num_key_value_heads=1, head_dim=args.hidden // 2)
    if t.get("layer_types"):
        t["layer_types"] = t["layer_types"][:2]
    if cp.get("layer_types"):
        cp["layer_types"] = cp["layer_types"][:1]
    cfg = Qwen3OmniMoeTalkerConfig(**full)
    cfg.vocab_size = t.get("vocab_size", args.codebook)
    model = Qwen3OmniMoeTalkerForConditionalGeneration(cfg)
    print("model built", flush=True)
    inputs_embeds = torch.randn(args.batch, args.seq, args.hidden)
    attention_mask = torch.ones(args.batch, args.seq, dtype=torch.long)
    labels = torch.randint(0, args.codebook, (args.batch, args.seq))
    try:
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        print("logits", None if out.logits is None else tuple(out.logits.shape), flush=True)
        print("loss", float(out.loss) if out.loss is not None else None, flush=True)
        out.loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        print("params with grad", sum(grads), "/", len(grads), flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("SMOKE FAILED:", type(exc).__name__, exc, flush=True)
        raise SystemExit(1)

    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
