#!/usr/bin/env python3
"""M4：Talker 可训练参数账 + 先验可用性表（进 §10.1 #1 验收卡口径）。

要回答的问题不是"参数量多少"，而是**哪些参数在 v0 里还能算预训练先验**。
因为 v0 换了 codec 口径：checkpoint 里的 15 个 lm_head / 15 张嵌入表 / codec_head
都是对着 **Qwen3-TTS 2048 码本** 训好的，而 v0 的监督目标是 **EnCodec 1024 码本**：
  - 表的"形状"在（2048 / 3072 类都在），所以加载不会 missing；
  - 表的"语义"不在（第 k 行对应哪个音色/哪层残差，两套码本毫无关系），
    所以不能把 `loss 从 12 降到 3` 说成"预训练先验起了作用"。

这条是 #1 验收卡必填项：必须逐项标「有先验 / 名义有先验但语义失配 / 从零」，
禁止只报「16 项 loss 全降」。

用 meta device 实例化，0 显存，可与权重抽取并行跑。

用法（远端）：
  /root/miniconda3/bin/python scripts/talker_param_table.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# v0 监督口径
V0_CODEBOOK = 1024      # EnCodec 24kbps codebook_size
V0_GROUPS = 16          # 1 首层 + 15 残差层
PRETRAIN_CODEBOOK = 2048  # Qwen3-TTS / Talker 原生码本
CODEC_HEAD_VOCAB = 3072   # 2048 码字 + 1024 special


def sha256_code() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/qwen3_omni_config/config.json")
    ap.add_argument("--out", default="outputs/talker_param_table.json")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-skip", default="codec_head")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as TalkerModel,
    )

    cfg = AutoConfig.from_pretrained(args.config)
    with torch.device("meta"):
        model = TalkerModel(cfg.talker_config)

    # ---------------------------------------------------------- 分组
    groups: dict[str, dict] = {}

    def add(g: str, n: int, name: str) -> None:
        d = groups.setdefault(g, {"params": 0, "tensors": 0, "examples": []})
        d["params"] += n
        d["tensors"] += 1
        if len(d["examples"]) < 3:
            d["examples"].append(name)

    per_head, per_emb_tab = [], []
    for name, p in model.named_parameters():
        n = p.numel()
        if "code_predictor.model.codec_embedding" in name:
            add("cp_residual_embedding_tables", n, name)
            if name.endswith(".weight") and "codec_embedding." in name:
                per_emb_tab.append({"name": name, "shape": list(p.shape)})
        elif "code_predictor.lm_head" in name:
            add("cp_residual_lm_heads", n, name)
            if name.endswith(".weight"):
                per_head.append({"name": name, "shape": list(p.shape)})
        elif "code_predictor" in name:
            add("cp_body", n, name)
        elif name == "codec_head.weight":
            add("codec_head", n, name)
        elif name == "model.codec_embedding.weight":
            add("codec_embedding", n, name)
        elif "text_projection" in name:
            add("text_projection", n, name)
        elif "embed_tokens" in name:
            add("text_embed_tokens", n, name)
        elif ".mlp.experts." in name:
            add("moe_experts_3d", n, name)
        elif ".layers." in name:
            add("layers_other", n, name)
        else:
            add("other", n, name)

    total = sum(v["params"] for v in groups.values())
    for v in groups.values():
        v["share_pct"] = round(v["params"] / total * 100, 4)
        v["params_b"] = round(v["params"] / 1e9, 6)

    # ---------------------------------------------------------- 先验可用性
    prior = {
        "codec_head": {
            "v0_state": "名义有先验，语义失配",
            "why": f"预训练时 codec_head 是 {CODEC_HEAD_VOCAB} 类（{PRETRAIN_CODEBOOK} 码字 + 1024 special），"
                   f"tied 到 Qwen3-TTS 2048 码本；v0 只监督前 {V0_CODEBOOK} 类且目标是 EnCodec 码本，"
                   "第 k 行对应的音色语义与预训练无关",
            "must_report": "第 1 层 CE 必须与 EnCodec 码本分布对照，不能与预训练 loss 比",
        },
        "cp_residual_lm_heads": {
            "v0_state": "名义有先验，语义失配（15 个全部）",
            "why": "15 个 lm_head 各自 2048 类，预训练对齐 Qwen3-TTS 残差码本；"
                   "v0 只切 0–1023 且对应 EnCodec 第 2–16 层，行号到音色的映射整体重排",
            "must_report": "15 个 CE 必须逐个报，不能只报均值；且要报「CE/上限 log(1024)=6.9307」",
        },
        "cp_residual_embedding_tables": {
            "v0_state": "名义有先验，语义失配（15 个全部）",
            "why": "同上：15 张 2048×1024 嵌入表按 Qwen3-TTS 码本索引，EnCodec 码值同号不同义",
            "must_report": "teacher forcing 输入是**真值**码嵌入，所以输入分布会被先验表扭曲成"
                           "「看起来 loss 降得快」的假象，必须与随机初始化同结构对照",
        },
        "cp_body": {
            "v0_state": "有先验（5 层 self-attn+MLP，与码本无关）",
            "why": "只做残差层的自回归建模，不依赖码本语义，可视为可用先验",
            "must_report": "单独报这部分的冻结/训练策略",
        },
        "moe_experts_3d": {
            "v0_state": "有先验（但 v0 冻结）",
            "why": "30B 主干专家，v0 只做 LoRA，专家不训练",
            "must_report": "冻结比例与 LoRA 可训练占比",
        },
        "text_projection": {
            "v0_state": "有先验",
            "why": "2048→1024 的两层 MLP + SiLU， thinker 词嵌入→talker 空间的映射，与码本无关",
            "must_report": "v0 是否解冻（决定文本条件能不能适配 beatbox 域）",
        },
    }

    # ---------------------------------------------------------- LoRA 目标
    skip = tuple(s for s in args.lora_skip.split(",") if s)
    targets, lora_params = [], 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        targets.append(name)
        lora_params += args.lora_rank * (mod.in_features + mod.out_features)

    # 30 张"换码本"表：v0 真正要重学的核心
    retarget = groups["cp_residual_lm_heads"]["params"] + groups["cp_residual_embedding_tables"]["params"] \
        + groups["codec_head"]["params"]

    payload = {
        "code_sha256": sha256_code(),
        "config": args.config,
        "total_params": total,
        "total_params_b": round(total / 1e9, 6),
        "groups": groups,
        "v0_protocol": {
            "codebook": V0_CODEBOOK,
            "groups": V0_GROUPS,
            "ce_upper_bound_log1024": round(__import__("math").log(V0_CODEBOOK), 6),
            "note": "PI 裁决⑪：loss 与采样双侧 mask 1024–2047 及 2048+，报数必须给 CE/上限",
        },
        "prior_availability": prior,
        "retarget_params": retarget,
        "retarget_params_b": round(retarget / 1e9, 6),
        "retarget_share_pct": round(retarget / total * 100, 4),
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "skip_contains": list(skip),
            "n_targets": len(targets),
            "lora_params": lora_params,
            "trainable_ratio": round(lora_params / total, 8),
            "note": "3D 融合专家不是 nn.Linear，LoRA 不覆盖；专家冻结时只能说「head + LoRA 可学」",
        },
        "detail": {
            "n_residual_lm_heads": len(per_head),
            "n_residual_embedding_tables": len(per_emb_tab),
            "lm_head_shapes": sorted({tuple(t["shape"]) for t in per_head}),
            "embedding_shapes": sorted({tuple(t["shape"]) for t in per_emb_tab}),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[M4] total={payload['total_params_b']}B retarget={payload['retarget_params_b']}B "
          f"({payload['retarget_share_pct']}%) heads={len(per_head)} tables={len(per_emb_tab)} "
          f"lora={lora_params} ({payload['lora']['trainable_ratio']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
