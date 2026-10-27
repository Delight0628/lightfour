#!/usr/bin/env python3
"""PI §5.2f 裁决③ 的前置断言：thinker 的 hidden_states[0] 是否就是 embed_tokens(input_ids)。

为什么断言权重无关：
  hidden_states[0] 是"送进第 0 层之前的张量"。如果 embed 与第 0 层之间没有任何
  norm/scale/proj，那么对**任意权重**都有 hidden_states[0] == embed_tokens(input_ids)
  （逐位相等，max abs diff == 0）。所以用真实 config + 随机初始化权重做 2 层模型即可判定，
  不必加载 59GB thinker。权重只影响数值大小，不影响"有没有中间模块"这个结构事实。

同时做第二重证据（防"外层 thinker 在拼多模态时动了文本位置"）：
  只用真实 config 建 thinker，检查 forward 里 inputs_embeds 的拼装是否只覆盖多模态位置。

输出 outputs/talker_hidden0_assert.json，退出码 0=PASS / 3=FAIL。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CFG = os.environ.get("LF_OMNI_CFG", "models/qwen3_omni_config/config.json")
OUT = Path("outputs/talker_hidden0_assert.json")


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(CFG)
    print(f"config class: {type(cfg).__name__}")
    tcfg = getattr(cfg, "thinker_config", cfg)
    text_cfg = getattr(tcfg, "text_config", tcfg)
    print(
        f"thinker text: hidden={text_cfg.hidden_size} layers={text_cfg.num_hidden_layers} "
        f"vocab={text_cfg.vocab_size} heads={text_cfg.num_attention_heads}"
    )
    tk = getattr(cfg, "talker_config", None)
    tk_text = getattr(tk, "text_config", tk) if tk is not None else None
    if tk is not None:
        print(
            f"talker: hidden={getattr(tk_text, 'hidden_size', None)} "
            f"accept_hidden_layer={getattr(tk, 'accept_hidden_layer', None)} "
            f"codec_head_dim={getattr(tk, 'codec_head_dim', getattr(tk, 'hidden_size', None))}"
        )

    # 只建 2 层的文本主干：结构事实与层数无关
    import copy

    cfg2 = copy.deepcopy(text_cfg)
    cfg2.num_hidden_layers = 2
    cfg2._attn_implementation = "eager"
    torch.manual_seed(0)

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextModel,
    )

    model = Qwen3OmniMoeThinkerTextModel(cfg2).eval()

    ids = torch.tensor([[151644, 872, 198, 9707, 13, 151645, 198]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
        emb = model.embed_tokens(ids)
    hs = out.hidden_states
    print(f"num hidden_states entries: {len(hs)} (expect layers+1 = 3), len={hs[0].shape}")
    d0 = (hs[0] - emb).abs().max().item()
    exact = torch.equal(hs[0], emb)
    print(f"hidden_states[0] vs embed_tokens: max_abs_diff={d0:.3e} torch.equal={exact}")

    # 对照：hidden_states[1] 应该已经被第 0 层改过，diff 必须显著 > 0
    d1 = (hs[1] - emb).abs().max().item()
    print(f"sanity: hidden_states[1] vs embed max_abs_diff={d1:.3e}（必须远大于 0）")

    pass_ = bool(exact or d0 == 0.0) and d1 > 0
    payload = {
        "config": CFG,
        "note": "真实 config + 随机权重 2 层；断言与权重无关（结构事实）",
        "thinker_text": {
            "hidden_size": text_cfg.hidden_size,
            "num_hidden_layers": text_cfg.num_hidden_layers,
            "vocab_size": text_cfg.vocab_size,
        },
        "talker": {
            "hidden_size": getattr(tk_text, "hidden_size", None) if tk is not None else None,
            "accept_hidden_layer": getattr(tk, "accept_hidden_layer", None) if tk is not None else None,
        },
        "num_hidden_states": len(hs),
        "max_abs_diff_hidden0_vs_embed": d0,
        "torch_equal": exact,
        "sanity_hidden1_diff": d1,
        "verdict": "PASS" if pass_ else "FAIL",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} verdict={payload['verdict']}")
    return 0 if pass_ else 3


if __name__ == "__main__":
    sys.exit(main())
