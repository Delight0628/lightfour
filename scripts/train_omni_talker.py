"""
Qwen3-Omni Talker SFT：真实训练分支（v0，定长生成口径）。

本文件替换 2026-10-23 的骨架版。三件事：
  1. 忠实复刻 transformers 5.16.1 推理期的 Talker 输入拼装（`_get_talker_assistant_parts`
     与 `_update_model_kwargs_for_generation` 的 per-step 规则），因此训练时的序列
     和推理时逐帧输入是同一条路径，不是另写一套。
  2. 16 层 codec 监督：第 1 层走 codec_head（CE 只切 0–1023，PI 裁决③），
     第 2–16 层由 code_predictor 的 15 个 lm_head 做 teacher forcing，15 个独立 CE。
     **模型自带 forward 只算第 1 层，且 loss 覆盖全 3072 类**，所以这里不用它的 loss。
  3. `--bench-step`：按真实架构、真实 T（最长 600 帧 = 8s@75Hz）测 fwd+bwd peak VRAM。

为什么文本侧不需要跑 30B Thinker：
  上游 `_get_talker_assistant_parts` 对**文本**段用的是 `text_projection(thinker_embed)`，
  其中 `thinker_embed = thinker_result.hidden_states[0]`，即 Thinker 的**输入词嵌入**，
  不是深层 hidden（深层 hidden 只给多模态段用，`accept_hidden_layer`）。所以纯文本
  提示词的 v0 SFT 只需要 thinker 的词嵌入矩阵 + text_projection，不需要加载 30B 主干。
  这是省显存的关键，不是取巧：改口径就等于偏离推理路径。

用法（远端 /gemini/code/lightfour-train）：
  # 逻辑自测（纯 torch，CPU，tiny dense 绕开 tiny-MoE NaN）
  python scripts/train_omni_talker.py --self-test

  # 显存 bench：真实架构 + 随机初始化（权重值不影响显存，只影响 loss）
  python scripts/train_omni_talker.py --bench-step 1 --bench-frames 232,413,600 --quant 4bit

  # 真实训练（需 30B 权重已下载）
  python scripts/train_omni_talker.py --model /gemini/pretrain/Qwen3-Omni-30B-A3B-Instruct \
      --quant 4bit --lora-r 16 --grad-ckpt --grad-accum 8 --max-steps 200

  # 定长生成自检（v0：长度由 dur 给定，采样只允许 0–1023，无 eos）
  python scripts/train_omni_talker.py --gen-check --gen-frames 64
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path("/gemini/code/lightfour-train")
SNAPSHOT = BASE / "models/qwen3_omni_config/config.snapshot.json"
FRAME_RATE = 75
LAYERS = 16
CODEBOOK = 1024
# prefill 长度：assistant_text_hidden(9) / assistant_codec_hidden(9)
PREFILL_LEN = 9


# ------------------------------------------------------------------ 配置/建模
def load_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_talker_config(snap, tiny=None):
    """tiny=(层数, 专家数)：只砍深度/专家数，**不动宽度**。

    踩过的坑：把 hidden 一起缩到 64/128 时，第 2 层必出 NaN（head_dim 缩到 32/64 后
    RoPE/RMSNorm 的尺度出了问题，与专家数无关，seed 相关）。宽度保持生产值
    (1024 / 16 heads / head_dim 128) 后稳定可跑，代价只是 2 层 ~18M 参数，CPU 也能跑。
    """
    from transformers import Qwen3OmniMoeTalkerConfig
    full = json.loads(json.dumps(snap["talker_config"]))
    if tiny:
        layers, experts = tiny
        t = full["text_config"]
        t["num_hidden_layers"] = layers
        t["num_experts"] = experts
        t["num_experts_per_tok"] = 1
        if t.get("layer_types"):
            t["layer_types"] = t["layer_types"][:layers]
    return Qwen3OmniMoeTalkerConfig(**full)


def build_talker(cfg, quant="4bit", device="cuda", seed=0):
    """按配置随机初始化 Talker（真实架构），可选 4bit 量化。"""
    from transformers import Qwen3OmniMoeTalkerForConditionalGeneration

    torch.manual_seed(seed)
    # 3.4B Talker 用 fp32 在 CPU 上要 13.6GB、在卡上要 13GB；统一 bf16 构建
    torch.set_default_dtype(torch.bfloat16)
    model = Qwen3OmniMoeTalkerForConditionalGeneration(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[build] talker params {n/1e9:.3f}B dtype={next(model.parameters()).dtype}", flush=True)

    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        # codec_head 与 codec_embedding tied、lm_head 是输出头：这三类保持 bf16
        replaced = _replace_linear_4bit(model, qcfg)
        print(f"[build] 4bit 替换 Linear {replaced} 个（codec_head / lm_head 保持 bf16）",
              flush=True)
    if quant == "4bit" and device.startswith("cuda"):
        # 3.4B 整体 .to(cuda) 会先把 bf16 权重全搬上卡（6.8GB > 5.8GB）直接 OOM。
        # 逐模块量化：每个 Linear 上卡→量化→释放原权重，峰值只有一个模块。
        import bitsandbytes as bnb
        n_q = 0
        for m in model.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                m.to(device)
                n_q += 1
                if n_q % 200 == 0:
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(f"[build] 逐模块量化 {n_q} 个 Linear 完成，"
              f"allocated {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    model = model.to(device)
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    if device.startswith("cuda"):
        print(f"[build] after load: allocated {torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved {torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    return model


def _replace_linear_4bit(module, qcfg, prefix=""):
    import bitsandbytes as bnb
    skip = ("codec_head", "lm_head")
    n = 0
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and not full.endswith(skip) and "lm_head" not in full:
            l4 = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                compute_dtype=qcfg.bnb_4bit_compute_dtype,
                compress_statistics=qcfg.bnb_4bit_use_double_quant,
                quant_type=qcfg.bnb_4bit_quant_type)
            l4.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False,
                quant_type=qcfg.bnb_4bit_quant_type,
                compress_statistics=qcfg.bnb_4bit_use_double_quant)
            if child.bias is not None:
                l4.bias = child.bias
            setattr(module, name, l4)
            n += 1
        else:
            n += _replace_linear_4bit(child, qcfg, full)
    return n


def load_talker_weights(model, model_dir):
    """加载 talker 权重（**只此一份实现**，与 `talker_gate_v3.py` 共用 `scripts/talker_loader.py`）。

    strict 语义（§10.1 #1 硬门槛）：missing/unexpected 只要有一项不在白名单 → `SystemExit`。
    旧实现在本文件里另写了一份朴素 `load_state_dict(strict=False)`，在「checkpoint 逐专家 2D
    → 模型融合 3D」上会给出 40 missing + 7680 unexpected，且只 print 不 raise = 假绿灯
    （练的是随机初始化的专家，不炸也不 NaN）。白名单与 `outputs/talker_load_gate.json`
    的 `gate1_keys.allow_extra` 必须保持一致。
    """
    from talker_loader import load_talker_weights as _load

    allow_missing = set()
    allow_unexpected = {"thinker.model.embed_tokens.weight"}
    res = _load(model, model_dir)
    missing = [k for k in res["missing"] if k not in allow_missing]
    unexpected = [k for k in res["unexpected"] if k not in allow_unexpected]
    print(f"[weights] tensors_read={res['tensors_read']} "
          f"merged_experts_layers={res['merged_experts_layers']} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        print(f"[weights] missing(前5)={missing[:5]} unexpected(前5)={unexpected[:5]}", flush=True)
        raise SystemExit(
            f"[fatal] talker 权重不完整：missing={len(missing)} unexpected={len(unexpected)}；"
            f"不做静默加载，先过 outputs/talker_load_gate.json 三道门"
        )
    return model


# ------------------------------------------------------------------ 输入拼装
class TextEmbedder(nn.Module):
    """thinker 词嵌入。有 30B 权重时用真实矩阵，bench 时用同宽随机矩阵。"""

    def __init__(self, vocab, dim, real=None):
        super().__init__()
        self.emb = real if real is not None else nn.Embedding(vocab, dim)

    def forward(self, ids):
        return self.emb(ids)


def build_assistant_frames(talker, text_ids, text_emb, tts_ids, cfg, speaker_id,
                           code1, T, device, dtype):
    """复刻 `_get_talker_assistant_parts` + 逐帧 `codec_embedding(prev)+trailing_text_hidden`。

    text_ids: (1,L) 助手段 token（至少 5 个，前 4 个进 prefill，其余进 trailing）
    tts_ids:  (1,3) **顺序固定为 [bos, eos, pad]**（下方 L201 按 [:,0]=bos, [:,1]=eos,
              [:,2]=pad 解包）。顺序传错不会报错、不会 NaN，只会把 pad 位喂成 eos 向量，
              属于静默错训 —— `scripts/talker_special_pos.py` 用负对照专测这一条。
              且三个 id 只存在于**顶层** Qwen3OmniMoeConfig，talker_config 没有 tts_*_token_id。
    code1:    (1,T) 第 1 层目标码，teacher forcing 用
    返回 inputs_embeds (1, 8+T, H)、labels (1, 8+T)，前 8 位 -100。
    """
    assert tts_ids.shape == (1, 3), tts_ids.shape
    H = talker.config.text_config.hidden_size
    proj = talker.text_projection(text_emb(text_ids))            # (1,L,H)
    if proj.shape[1] < 5:
        pad = torch.zeros(1, 5 - proj.shape[1], proj.shape[2], device=device, dtype=dtype)
        proj = torch.cat([proj, pad], dim=1)
    tts = talker.text_projection(text_emb(tts_ids))              # (1,3,H) bos/eos/pad
    tts_bos, tts_eos, tts_pad = tts[:, 0:1], tts[:, 1:2], tts[:, 2:3]

    a_text = torch.cat([proj[:, :3], tts_pad.expand(-1, 4, -1), tts_bos, proj[:, 3:4]],
                       dim=1)                                    # (1,9,H)
    special = torch.tensor([[cfg.codec_nothink_id, cfg.codec_think_bos_id,
                             cfg.codec_think_eos_id, speaker_id,
                             cfg.codec_pad_id, cfg.codec_bos_id]],
                           device=device, dtype=torch.long)
    codec_emb = talker.get_input_embeddings()
    a_codec = torch.cat([torch.zeros(1, 3, H, device=device, dtype=dtype),
                         codec_emb(special)], dim=1)             # (1,9,H)
    prefill = a_text + a_codec

    trailing = torch.cat([proj[:, 4:], tts_eos], dim=1)          # (1, L-3, H)
    n_trail = trailing.shape[1]

    # 位置 9+k 的输入 = codec_embedding(code1[k]) + trailing[k]（越界用 tts_pad）
    idx = torch.arange(max(T - 1, 0), device=device)
    frame_tok = codec_emb(code1[:, :max(T - 1, 0)])               # (1,T-1,H)
    if n_trail >= max(T - 1, 0):
        trail = trailing[:, idx]
    else:
        trail = torch.cat([trailing,
                           tts_pad.expand(-1, max(T - 1, 0) - n_trail, -1)], dim=1)
    frames = frame_tok + trail if T > 1 else frame_tok
    embeds = torch.cat([prefill, frames], dim=1)                 # (1,8+T,H)

    labels = torch.full((1, embeds.shape[1]), -100, dtype=torch.long, device=device)
    labels[:, PREFILL_LEN - 1: PREFILL_LEN - 1 + T] = code1
    assert embeds.shape[1] == 8 + T, (embeds.shape, T)
    return embeds.to(dtype), labels, trailing


# ------------------------------------------------------------------ 前向入口
def talker_forward(model, embeds):
    """统一前向：显式给 position_ids，绕开 get_rope_index 对 talker_input_ids 的依赖。

    transformers 5.16.1 的 Talker.forward 在 position_ids=None 时要求 attention_mask 与
    talker_input_ids（3D mrope 索引）。我们做的是纯文本定长序列，时间轴就是等距帧，
    所以直接给 (3,1,N) 的 arange，等价于 mrope 的文本分支，也避免 mrope_section 配错导致 NaN。
    """
    N = embeds.shape[1]
    pos = torch.arange(N, device=embeds.device).view(1, 1, N).expand(3, 1, N)
    return model(inputs_embeds=embeds, position_ids=pos, output_hidden_states=True)


# ------------------------------------------------------------------ 16 层 CE
def talker_ce_16(out, codes, mask, model, codebook=CODEBOOK):
    """第 1 层（codec_head，切 0–1023）+ 15 个残差层（code_predictor teacher forcing）。"""
    from talker_sequence import codec_ce_16
    logits = out.logits
    hs = out.hidden_states
    # Talker 的 hidden_states 是 (all_layer_hidden, residual_codes) 的嵌套 tuple（上游 hack）
    if isinstance(hs, tuple) and isinstance(hs[0], (tuple, list)):
        hidden = hs[0][-1]
    elif isinstance(hs, (tuple, list)):
        hidden = hs[-1]
    else:
        hidden = hs
    h_frame = hidden[:, PREFILL_LEN - 1: PREFILL_LEN - 1 + codes.shape[2]]
    l1 = logits[:, PREFILL_LEN - 1: PREFILL_LEN - 1 + codes.shape[2], :codebook]
    return codec_ce_16(h_frame, codes, mask,
                       codec_head=model.codec_head,
                       input_embedding=model.get_input_embeddings(),
                       predictor=model.code_predictor,
                       layer1_logits=l1, codebook=codebook)


# ------------------------------------------------------------------ 数据
def make_dataset(args):
    from seq_token_dataset import SeqTokenDataset, collate as seq_collate
    ds = SeqTokenDataset(args.manifest, args.codes_dir, args.hit_frames,
                         split=args.split, max_frames=args.max_frames)
    return ds, seq_collate


def batch_codes(b, device):
    return b["codes"].to(device), b["mask"].to(device)


# ------------------------------------------------------------------ 自测
def model_dtype(model):
    """取模型真实的浮点 dtype；bnb 4bit 下 parameters() 首位可能是 uint8/int8，不能直接用。"""
    for p in model.parameters():
        if p.dtype.is_floating_point:
            return p.dtype
    return torch.bfloat16


def self_test(args):
    """tiny dense 端到端：拼装长度、标签对齐、16 项 CE、15 个 lm_head 全有梯度。"""
    snap = load_snapshot(args.snapshot)
    cfg = make_talker_config(snap, tiny=(2, 1))
    model = build_talker(cfg, quant="bf16", device="cpu")
    model.train()
    cfg_t = model.config
    TH = model.config.thinker_hidden_size   # text_projection 的输入宽（=2048），不是 talker hidden
    T = 11
    text_ids = torch.randint(0, 100, (1, 9))
    text_emb = TextEmbedder(128, TH)
    tts_ids = torch.tensor([[cfg_t.audio_start_token_id if False else 1, 2, 3]])
    tts_ids = torch.tensor([[1, 2, 3]])
    code1 = torch.randint(0, CODEBOOK, (1, T))
    codes = torch.randint(0, CODEBOOK, (1, LAYERS, T))
    mask = torch.ones(1, T, dtype=torch.bool)
    mask[0, T - 2:] = False

    embeds, labels, trailing = build_assistant_frames(
        model, text_ids, text_emb, tts_ids, cfg_t, 2301, code1, T, "cpu", model_dtype(model))
    print(f"[self-test] embeds {tuple(embeds.shape)} labels {tuple(labels.shape)} "
          f"trailing {tuple(trailing.shape)}")
    assert embeds.shape[1] == 8 + T
    assert labels[0, 0] == -100 and labels[0, PREFILL_LEN - 1] == code1[0, 0]
    assert labels[0, PREFILL_LEN - 1 + T - 1] == code1[0, T - 1]

    out = talker_forward(model, embeds)
    assert out.logits.shape[-1] == cfg_t.text_config.vocab_size, out.logits.shape
    res = talker_ce_16(out, codes, mask, model)
    print(f"[self-test] terms={res['num_terms']} frames={res['num_frames']} "
          f"(应为 {1 + LAYERS - 1} / {int(mask.sum())}) loss={res['loss']:.3f}")
    assert res["num_terms"] == LAYERS
    assert res["num_frames"] == int(mask.sum()) == T - 2
    res["loss"].backward()
    heads = [k for k in range(LAYERS - 1)
             if model.code_predictor.lm_head[k].weight.grad is not None
             and model.code_predictor.lm_head[k].weight.grad.abs().sum() > 0]
    assert len(heads) == LAYERS - 1, f"只有 {len(heads)} 个 lm_head 有梯度"
    assert model.get_input_embeddings().weight.grad is not None
    print(f"[self-test] 15 个 lm_head 全部有梯度；codec_embedding 有梯度 ✓")
    print("[self-test] OK")
    return 0


# ------------------------------------------------------------------ bench
def bench(args):
    snap = load_snapshot(args.snapshot)
    cfg = make_talker_config(snap, tiny=(args.bench_layers, 128) if args.bench_layers else None)
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("[fatal] 无 CUDA")
    p = torch.cuda.get_device_properties(0)
    n_layers = cfg.text_config.num_hidden_layers
    print(f"[bench] GPU {p.name} {p.total_memory/1024**3:.1f}GB quant={args.quant} "
          f"layers={n_layers}/{snap['talker_config']['text_config']['num_hidden_layers']} "
          f"experts={cfg.text_config.num_experts}", flush=True)

    model = build_talker(cfg, quant=args.quant, device=device)
    model.train()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        print("[bench] gradient checkpointing ON")

    H = model.config.text_config.hidden_size
    text_ids = torch.randint(0, 1000, (1, args.prompt_len), device=device)
    text_emb = TextEmbedder(2048, 2048).to(device)  # thinker 词嵌入同宽随机矩阵
    tts_ids = torch.tensor([[1, 2, 3]], device=device)

    rows = []
    for T in [int(x) for x in args.bench_frames.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        code1 = torch.randint(0, CODEBOOK, (1, T), device=device)
        codes = torch.randint(0, CODEBOOK, (1, LAYERS, T), device=device)
        mask = torch.ones(1, T, dtype=torch.bool, device=device)
        try:
            t0 = time.time()
            embeds, labels, _ = build_assistant_frames(
                model, text_ids, text_emb, tts_ids, model.config, 7, code1, T,
                device, torch.bfloat16)
            out = talker_forward(model, embeds)
            res = talker_ce_16(out, codes, mask, model)
            res["loss"].backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak_a = torch.cuda.max_memory_allocated() / 1e9
            peak_r = torch.cuda.max_memory_reserved() / 1e9
            rows.append({"T": T, "seq_len": int(embeds.shape[1]), "ok": True,
                         "peak_alloc_GB": round(peak_a, 2), "peak_reserved_GB": round(peak_r, 2),
                         "base_GB": round(base/1e9, 2), "sec_step": round(dt, 2),
                         "frames_per_s": round(T / dt, 1),
                         "loss": round(float(res["loss"]), 3)})
            print(f"[bench] T={T:>3} seq={embeds.shape[1]:>3} peak={peak_a:.2f}GB "
                  f"(reserved {peak_r:.2f}) {dt:.2f}s loss={float(res['loss']):.3f}", flush=True)
            model.zero_grad(set_to_none=True)
            del embeds, out, res, code1, codes, mask
        except torch.cuda.OutOfMemoryError as e:
            rows.append({"T": T, "ok": False, "error": "OOM", "detail": str(e)[:200]})
            print(f"[bench] T={T} OOM", flush=True)
            torch.cuda.empty_cache()

    out_path = Path(args.bench_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"gpu": p.name, "total_GB": round(p.total_memory/1024**3, 1),
                                    "quant": args.quant, "grad_ckpt": args.grad_ckpt,
                                    "layers": n_layers,
                                    "full_layers": snap["talker_config"]["text_config"]["num_hidden_layers"],
                                    "experts": cfg.text_config.num_experts,
                                    "rows": rows}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"[bench] wrote {out_path}")
    return 0


# ------------------------------------------------------------------ 定长生成
@torch.no_grad()
def generate_fixed(model, text_ids, text_emb, tts_ids, cfg, speaker_id, n_frames,
                   device, temperature=1.0):
    """v0 定长生成：跑满 n_frames 步就停，采样只在 0–1023，无 eos（PI 裁决③）。"""
    H = model.config.text_config.hidden_size
    embeds, _, trailing = build_assistant_frames(
        model, text_ids, text_emb, tts_ids, cfg, speaker_id,
        torch.zeros(1, 2, dtype=torch.long, device=device), 2, device, torch.bfloat16)
    prefix = embeds[:, :PREFILL_LEN]                      # (1,9,H)
    cur = prefix
    out_codes = []
    for k in range(n_frames):
        o = talker_forward(model, cur)
        l1 = o.logits[:, -1, :CODEBOOK].float()
        if temperature <= 0:
            tok = l1.argmax(-1)
        else:
            tok = torch.multinomial(torch.softmax(l1 / temperature, -1), 1).squeeze(-1)
        out_codes.append(tok)
        step = torch.cat([model.get_input_embeddings()(tok).unsqueeze(1),
                          trailing[:, k:k+1] if k < trailing.shape[1]
                          else trailing[:, -1:]], dim=0).sum(0, keepdim=True)
        cur = torch.cat([cur, step], dim=1)               # 无 KV cache：O(T^2)，仅用于自检
    return torch.stack(out_codes, dim=1)                   # (1,n_frames)


def gen_check(args):
    snap = load_snapshot(args.snapshot)
    cfg = make_talker_config(snap, tiny=(2, 1))
    model = build_talker(cfg, quant="bf16", device="cpu").eval()
    TH = model.config.thinker_hidden_size
    text_ids = torch.randint(0, 100, (1, 9))
    text_emb = TextEmbedder(128, TH)
    tts_ids = torch.tensor([[1, 2, 3]])
    codes = generate_fixed(model, text_ids, text_emb, tts_ids, model.config, 7,
                           args.gen_frames, "cpu", temperature=0.0)
    assert codes.shape == (1, args.gen_frames), codes.shape
    assert int(codes.min()) >= 0 and int(codes.max()) < CODEBOOK, "越界采样"
    print(f"[gen-check] 定长 {codes.shape} 码域 [{int(codes.min())},{int(codes.max())}] "
          f"无 eos、按 dur 定长 OK")
    return 0


# ------------------------------------------------------------------ 训练
def train(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snap = load_snapshot(args.snapshot)
    cfg = make_talker_config(snap, tiny=(args.tiny_layers, 1) if args.tiny_layers else None)
    if not args.bench_step and not args.self_test:
        # speaker 位是 codec 嵌入表的真实条目，传 0 会喂一个普通码字进去当下一个 token
        _spk = set(cfg.speaker_id.values())
        assert args.speaker_id in _spk, f"--speaker-id 必须是 {sorted(_spk)} 之一，当前 {args.speaker_id}"
    if args.tiny_layers:
        print(f"[warn] TINY {args.tiny_layers} 层/1 专家：只验证训练闭环与 loss 下降，不代表 30B 结果",
              flush=True)
    model = build_talker(cfg, quant=args.quant, device=device)

    if not args.random_init and not args.tiny_layers:
        model = load_talker_weights(model, args.model)
    else:
        print("[warn] RANDOM_INIT：loss 数字无意义，只用于跑通/测显存", flush=True)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    if args.lora_r > 0:
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    tok = AutoTokenizer.from_pretrained(args.model) if Path(args.model).exists() else None
    if tok is None:
        print("[warn] 无 tokenizer：用随机 text ids（跑通/测显存可以，真实训练不行）",
              flush=True)
    think_emb = None
    if Path(args.model).exists():
        try:
            from transformers import AutoModelForCausalLM  # noqa: F401
            from safetensors import safe_open
            d = Path(args.model)
            idx = d / "model.safetensors.index.json"
            wm = json.loads(idx.read_text())["weight_map"]
            key = "thinker.model.embed_tokens.weight"
            sh = d / wm[key]
            with safe_open(str(sh), framework="pt") as f:
                think_emb = nn.Embedding.from_pretrained(f.get_tensor(key), freeze=True)
            print(f"[weights] thinker 词嵌入 {tuple(think_emb.weight.shape)} 已载入（仅此一张量）",
                  flush=True)
        except Exception as exc:
            print(f"[warn] 载入 thinker 词嵌入失败：{type(exc).__name__} {exc}", flush=True)

    H = model.config.text_config.hidden_size
    t_dim = 2048 if think_emb is None else think_emb.weight.shape[1]
    text_emb = TextEmbedder(4096, t_dim, real=think_emb).to(device)
    # tts 三件套：**顶层** config 才有 tts_*_token_id（talker_config 实测全 ABSENT）。
    # 旧实现 `hasattr(model.config,"tts_bos_token_id")` 恒 False → 退化成词表第 1/2/3 行
    # （普通词向量）当 tts_bos/tts_pad/tts_eos：不炸 loss、不 NaN，只让 prefill 的 4 个
    # pad 位与 trailing 末尾的 eos 位学错语义 = 静默错训。
    # 顺序契约：build_assistant_frames 内部 tts[:,0]=bos, [:,1]=eos, [:,2]=pad。
    _root_cfg_path = Path(args.model) / "config.json"
    if _root_cfg_path.exists():
        _root_cfg = json.loads(_root_cfg_path.read_text(encoding="utf-8"))
        _tts = {k: _root_cfg.get(f"tts_{k}_token_id") for k in ("pad", "bos", "eos")}
        if any(v is None for v in _tts.values()):
            raise SystemExit(f"[fatal] 顶层 config 缺 tts_pad/bos/eos_token_id: {_root_cfg_path}")
        tts_ids = torch.tensor([[_tts["bos"], _tts["eos"], _tts["pad"]]], device=device)
        print(f"[special] tts ids (bos/eos/pad) = {[_tts['bos'], _tts['eos'], _tts['pad']]}",
              flush=True)
    else:
        # random-init / 无权重路径：text 词表只有 4096，真实 tts id（151671+）会越界
        tts_ids = torch.tensor([[1, 2, 3]], device=device)
        print("[warn] 无顶层 config：tts ids 用占位 [1,2,3]，禁止用于真实训练", flush=True)

    ds, _ = make_dataset(args)
    print(f"[data] {len(ds)} 条 (split={args.split}, max_frames={args.max_frames})", flush=True)
    if len(ds) == 0:
        raise SystemExit("[fatal] 数据集为空")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    hist, step, t_start = [], 0, time.time()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    order = list(range(len(ds)))
    for epoch in range(args.epochs):
        if args.shuffle:
            torch.randperm(len(order)).tolist()
            order = torch.randperm(len(ds)).tolist()
        for i in order:
            b = ds[i]
            codes = b["codes"].unsqueeze(0).to(device)        # (1,16,T)
            mask = b["mask"].unsqueeze(0).to(device)
            T = codes.shape[2]
            if tok is not None:
                ids = tok(b["text"], return_tensors="pt", add_special_tokens=False,
                          truncation=True, max_length=256)["input_ids"].to(device)
                if ids.shape[1] < 5:
                    ids = torch.cat([ids, ids[:, -1:].repeat(1, 5 - ids.shape[1])], dim=1)
            else:
                ids = torch.randint(0, 1000, (1, 9), device=device)

            embeds, labels, _ = build_assistant_frames(
                model, ids, text_emb, tts_ids, model.config, args.speaker_id,
                codes[:, 0, :], T, device, model_dtype(model))
            out = talker_forward(model, embeds)
            res = talker_ce_16(out, codes, mask, model)
            loss = res["loss"] / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t_start
                rec = {"step": step, "epoch": epoch, "T": T, "loss": round(float(res["loss"]), 4),
                       "layer1": round(res["layer1"], 4),
                       "residual": round(res["residual_mean"], 4),
                       "terms": res["num_terms"], "frames": res["num_frames"],
                       "lr": opt.param_groups[0]["lr"], "sec": round(el, 1),
                       "frames_per_s": round(step * T / el, 1)}
                hist.append(rec)
                print(f"[train] step {step} T={T} loss {rec['loss']:.4f} "
                      f"(L1 {rec['layer1']:.4f} / res {rec['residual']:.4f}) "
                      f"terms={rec['terms']} {rec['frames_per_s']}f/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loss_trace.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    model.save_pretrained(out_dir / "adapter")
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    summary = {"steps": step, "peak_alloc_GB": round(peak, 2),
               "first_loss": hist[0]["loss"] if hist else None,
               "last_loss": hist[-1]["loss"] if hist else None,
               "lora_r": args.lora_r, "quant": args.quant, "grad_ckpt": args.grad_ckpt,
               "frame_rate": FRAME_RATE, "layers": LAYERS}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[done] {summary}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--model", default=str(BASE / ".." / "pretrain/Qwen3-Omni-30B-A3B-Instruct"))
    ap.add_argument("--out", default=str(BASE / "checkpoints/omni_talker_v0"))
    ap.add_argument("--manifest", default=str(BASE / "data/seq_tokens/sequence_manifest.jsonl"))
    ap.add_argument("--codes-dir", default=str(BASE / "data/seq_tokens/codes"))
    ap.add_argument("--hit-frames", default=str(BASE / "data/seq_tokens/hit_frames.csv"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--quant", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1, help="v0 固定 1（序列长度 232–600 不等）")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--speaker-id", type=int, default=2301,
                    help="必须是 cfg.speaker_id 里的真实 id（2301/2302/2303）；0 是码字不是说话人")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--tiny-layers", type=int, default=0,
                    help="N>0 时用 N 层/1 专家的 tiny 模型跑训练闭环（不加载权重）")
    ap.add_argument("--bench-step", type=int, default=0)
    ap.add_argument("--bench-frames", default="232,413,600")
    ap.add_argument("--bench-out", default=str(BASE / "outputs/talker_bench.json"))
    ap.add_argument("--bench-layers", type=int, default=0,
                    help="只跑前 N 层（真实宽度/专家数不变），用于在显存不足的卡上量 activation 斜率")
    ap.add_argument("--prompt-len", type=int, default=40)
    ap.add_argument("--gen-check", action="store_true")
    ap.add_argument("--gen-frames", type=int, default=64)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test(args))
    if args.gen_check:
        sys.exit(gen_check(args))
    if args.bench_step:
        sys.exit(bench(args))
    sys.exit(train(args))


if __name__ == "__main__":
    main()
