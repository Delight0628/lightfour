#!/usr/bin/env python3
"""M5：special 序列脚本化 + position_ids 三维 mrope 逐位对拍。

为什么必须落脚本（PI 裁决：从「缺失证据」降为「已闭合、未脚本化」还不够）：
  2026-10-27 那次 special 对拍（prefill 9 vs 9、trailing 6 vs 6、label 起点 8、
  监督帧 5/5、max_abs_diff=0.0）是 inline 跑的，没有代码、没有产物、没有可复跑入口。
  一旦有人改了 `build_assistant_frames` 或换了 transformers 版本，没有任何东西会响。

为什么 position_ids 比 special 序列更危险（Talker 自报）：
  `train_omni_talker.talker_forward` 用 `arange(N)` 展成 (3,1,N) 喂给 mrope，
  绕开了官方 `get_rope_index`。这条路径错了不会炸 loss、不会 NaN，只会让**生成**
  的码流在长序列上慢慢错位 —— 属于最贵的静默故障。所以这里必须逐位对拍，
  而不是"看起来等价"。

对拍对象（两边都来自 transformers 5.16.1 真实函数，不是复述）：
  A. `Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts`
     vs `train_omni_talker.build_assistant_frames`
  B. `Qwen3OmniMoeTalkerForConditionalGeneration.get_rope_index`
     vs `arange(N).view(1,1,N).expand(3,1,N)`

内存策略：结构对拍与 hidden_size / 层数无关，所以把 talker 裁到 1 层再实例化
（~1GB），避免和正在跑的 7.27GB 权重抽取 / gate 抢 16GiB cgroup。

用法（远端）：
  /root/miniconda3/bin/python scripts/talker_special_pos.py
  /root/miniconda3/bin/python scripts/talker_special_pos.py --frames 64,232,600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path("/gemini/code/lightfour-train")
sys.path.insert(0, str(Path(__file__).resolve().parent))

SPEAKER_ID = 2301


def sha256_code() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/qwen3_omni_config/config.json")
    ap.add_argument("--out", default="outputs/talker_special_pos.json")
    ap.add_argument("--frames", default="64,232,600", help="要测的 codec 帧数 T")
    ap.add_argument("--truncate-layers", type=int, default=1,
                    help="把 talker 裁到几层再实例化（只影响内存，不影响结构对拍）")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration as OmniModel,
        Qwen3OmniMoeTalkerForConditionalGeneration as TalkerModel,
    )
    from train_omni_talker import PREFILL_LEN, build_assistant_frames

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    cfg = AutoConfig.from_pretrained(args.config if Path(args.config).exists() else BASE / args.config)

    # ---- 裁剪：只保留 1 层，token id / projection 形状全不动
    tcfg = cfg.talker_config
    orig_layers = tcfg.text_config.num_hidden_layers
    if args.truncate_layers and orig_layers > args.truncate_layers:
        tcfg.text_config.num_hidden_layers = args.truncate_layers

    talker = TalkerModel(tcfg).to(dtype).eval()

    H = talker.config.text_config.hidden_size
    thinker_hidden = talker.config.thinker_hidden_size
    # talker.get_input_embeddings() 是 codec 侧（vocab 3072），不是文本侧。
    # text_projection 的输入是 thinker 的**输入词嵌入**（vocab=text vocab, hidden=thinker_hidden）。
    # 结构对拍与数值无关，这里用同形状随机 embedding 代替 thinker 权重。
    _tc = getattr(cfg, "thinker_config", None)
    text_vocab = getattr(getattr(_tc, "text_config", None), "vocab_size", None) \
        or cfg.get_text_config().vocab_size
    text_emb = torch.nn.Embedding(text_vocab, thinker_hidden).to(dtype).eval()

    payload: dict = {
        "code_sha256": sha256_code(),
        "dtype": args.dtype,
        "transformers": __import__("transformers").__version__,
        "talker_hidden": H,
        "thinker_hidden": thinker_hidden,
        "layers_instantiated": tcfg.text_config.num_hidden_layers,
        "layers_in_config": orig_layers,
        "text_projection_type": type(talker.text_projection).__name__,
        "text_projection_shapes": [list(p.shape) for p in talker.text_projection.parameters()],
        "position_id_per_seconds": getattr(tcfg, "position_id_per_seconds", None),
        "mrope_section": getattr(tcfg.text_config, "rope_scaling", {}).get("mrope_section"),
        "head_dim": tcfg.text_config.head_dim,
        "special_ids": {
            "codec_nothink": tcfg.codec_nothink_id,
            "codec_think_bos": tcfg.codec_think_bos_id,
            "codec_think_eos": tcfg.codec_think_eos_id,
            "speaker": SPEAKER_ID,
            "codec_pad": tcfg.codec_pad_id,
            "codec_bos": tcfg.codec_bos_id,
        },
    }

    # ------------------------------------------------ A. special 序列结构对拍
    n_text = 9  # 助手段 token 数（im_start..segment_end），>=5 即可
    text_ids = torch.tensor(
        [[cfg.im_start_token_id, cfg.assistant_token_id, 1000, cfg.im_end_token_id, 2000][:n_text - 4]
         + [2001, 2002, 2003]], dtype=torch.long
    )[:, :n_text]
    thinker_embed = text_emb(text_ids).to(dtype)  # (1,n_text,thinker_hidden)
    assert thinker_embed.shape[-1] == thinker_hidden, thinker_embed.shape

    # tts 三件套：pad / bos / eos 是**顶层** config 的字段，talker_config 里没有
    # （实测 talker_config.tts_*_token_id 全 ABSENT，只有 Qwen3OmniMoeConfig 有）。
    _pad_id, _bos_id, _eos_id = int(cfg.tts_pad_token_id), int(cfg.tts_bos_token_id), int(cfg.tts_eos_token_id)

    def _proj_one(tok_id):
        return talker.text_projection(text_emb(torch.tensor([[int(tok_id)]])).to(dtype))

    with torch.no_grad():
        tts_pad_embed = _proj_one(_pad_id)
        tts_bos_embed = _proj_one(_bos_id)
        tts_eos_embed = _proj_one(_eos_id)

    import types
    stub = types.SimpleNamespace(talker=talker, config=cfg)

    def _run_local(tts_ids_row, T_probe=3):
        tts_ids = torch.tensor([tts_ids_row], dtype=torch.long)
        code1 = torch.randint(0, 1024, (1, T_probe), dtype=torch.long)
        with torch.no_grad():
            return build_assistant_frames(
                talker, text_ids, text_emb, tts_ids, talker.config, SPEAKER_ID,
                code1, T_probe, torch.device("cpu"), dtype,
            )

    with torch.no_grad():
        off_embeds, off_ids, off_trailing = OmniModel._get_talker_assistant_parts(
            stub,
            im_start_index=0,
            segment_end_index=n_text,
            speaker_id=SPEAKER_ID,
            thinker_embed=thinker_embed,
            tts_pad_embed=tts_pad_embed,
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
        )

    T_probe = 3
    # 内部契约（build_assistant_frames L201）：tts[:,0]=bos, [:,1]=eos, [:,2]=pad
    loc_embeds, loc_labels, loc_trailing = _run_local([_bos_id, _eos_id, _pad_id], T_probe)
    bad_embeds, _, bad_trailing = _run_local([_pad_id, _bos_id, _eos_id], T_probe)

    def _diff(a, b):
        n = min(a.shape[1], b.shape[1])
        return float((a[:, :n].float() - b[:, :n].float()).abs().max().item())

    # 官方 prefill 只是 (1,9)，本地 embeds = prefill + 前 T-1 帧，比对前 9 位
    prefill_diff = _diff(off_embeds, loc_embeds)
    trail_diff = _diff(off_trailing, loc_trailing)
    # 负对照：tts 三件套按 [pad,bos,eos] 传（最容易犯的错）必须被抓出来
    bad_prefill_diff = _diff(off_embeds, bad_embeds)
    bad_trail_diff = _diff(off_trailing, bad_trailing)

    payload["special"] = {
        "official_prefill_len": int(off_embeds.shape[1]),
        "local_prefill_len": int(min(off_embeds.shape[1], loc_embeds.shape[1])),
        "official_trailing_len": int(off_trailing.shape[1]),
        "local_trailing_len": int(loc_trailing.shape[1]),
        "official_input_ids_unique": sorted(set(off_ids.flatten().tolist())),
        "expected_tts_pad_token_id": int(cfg.tts_pad_token_id),
        "tts_ids_contract": "[bos, eos, pad]（build_assistant_frames L201: tts[:,0]=bos, [:,1]=eos, [:,2]=pad）",
        "tts_ids_used_ok": [_bos_id, _eos_id, _pad_id],
        "tts_ids_used_negative_control": [_pad_id, _bos_id, _eos_id],
        "prefill_max_abs_diff": prefill_diff,
        "trailing_max_abs_diff": trail_diff,
        "negative_control_prefill_max_abs_diff": bad_prefill_diff,
        "negative_control_trailing_max_abs_diff": bad_trail_diff,
        "negative_control_detected": bool(bad_prefill_diff > 1e-3 or bad_trail_diff > 1e-3),
        "label_first_supervised_index": int((loc_labels[0] != -100).nonzero()[0].item()),
        "label_n_supervised": int((loc_labels[0] != -100).sum().item()),
        "local_seq_len": int(loc_embeds.shape[1]),
        "expected_seq_len_prefill_plus_T": PREFILL_LEN - 1 + T_probe,
        "pass": bool(
            off_embeds.shape[1] == 9
            and loc_embeds.shape[1] == PREFILL_LEN - 1 + T_probe
            and prefill_diff == 0.0
            and trail_diff == 0.0
            and int((loc_labels[0] != -100).nonzero()[0].item()) == PREFILL_LEN - 1
            and int((loc_labels[0] != -100).sum().item()) == T_probe
        ),
    }

    # ------------------------------------------------ B. position_ids 对拍
    # 官方推理真实喂给 get_rope_index 的 input_ids 就是全 tts_pad（见 A 的 unique）
    frames = [int(x) for x in args.frames.split(",")]
    pos_cases = []
    for T in frames:
        N = PREFILL_LEN - 1 + T
        ids = torch.full((1, N), int(cfg.tts_pad_token_id), dtype=torch.long)
        am = torch.ones(1, N, dtype=torch.long)
        with torch.no_grad():
            off_pos, delta = talker.get_rope_index(input_ids=ids, attention_mask=am)
        local_pos = torch.arange(N).view(1, 1, N).expand(3, 1, N)
        d = (off_pos.long() - local_pos.long()).abs()
        pos_cases.append({
            "T": T,
            "N": N,
            "official_shape": list(off_pos.shape),
            "delta_sec_shape": list(delta.shape),
            "max_abs_diff": int(d.max().item()) if d.numel() else None,
            "n_mismatch": int((d != 0).sum().item()),
            "axis0_head": off_pos[0, 0, :3].tolist(),
            "axis0_tail": off_pos[0, 0, -3:].tolist(),
            "all_three_axes_equal": bool(torch.equal(off_pos[0], off_pos[1]) and torch.equal(off_pos[0], off_pos[2])),
            "pass": bool(off_pos.shape == local_pos.shape and int(d.max().item()) == 0),
        })
    payload["position_ids"] = {
        "cases": pos_cases,
        "pass": all(c["pass"] for c in pos_cases),
        "note": "input_ids 全为 tts_pad_token_id，与 _get_talker_assistant_parts 返回的 input_ids 一致",
    }

    # ------------------------------------------------ C. mrope 配置自洽
    sec = payload["mrope_section"] or []
    payload["mrope_consistency"] = {
        "mrope_section": sec,
        "sum": sum(sec),
        "head_dim": payload["head_dim"],
        "half_head_dim": payload["head_dim"] / 2 if payload["head_dim"] else None,
        "sum_equals_half_head_dim": (sum(sec) == payload["head_dim"] / 2) if sec and payload["head_dim"] else None,
        "note": "RoPE 的 mrope_section 按 head_dim/2 分配三轴 inv_freq；和等于 head_dim/2 才自洽",
    }

    payload["verdict"] = (
        "PASS"
        if payload["special"]["pass"]
        and payload["special"]["negative_control_detected"]
        and payload["position_ids"]["pass"]
        else "FAIL"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[M5] verdict={payload['verdict']} "
          f"special(prefix {payload['special']['prefill_max_abs_diff']}, trail {payload['special']['trailing_max_abs_diff']}) "
          f"pos({[c['max_abs_diff'] for c in pos_cases]}) -> {out}")
    return 0 if payload["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
