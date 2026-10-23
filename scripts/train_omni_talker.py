"""
Qwen3-Omni Talker 音频输出 SFT 训练骨架。

当前状态（2026-10-23）：
  - 6.2GB GPU 无法加载 30B-A3B；本脚本提供 --dry-run 验证数据管线。
  - 真实训练需 24–48GB GPU + Qwen3-Omni 权重。
  - ms-swift 4.5.3 没有 Qwen3-Omni 音频输出模板，因此这里直接对接 transformers 的
    `Qwen3OmniMoeTalkerForConditionalGeneration`（forward 有 labels/loss）。

用法：
  # 数据管线自检（不加载模型）
  python scripts/train_omni_talker.py --dry-run --data data/omni_talker/talker_train.jsonl

  # 真实训练（24GB+）
  python scripts/train_omni_talker.py \
    --model /gemini/pretrain/Qwen3-Omni-30B-A3B-Instruct \
    --data data/omni_talker/talker_train.jsonl \
    --out checkpoints/omni_talker_lora --epochs 3 --batch-size 1 --grad-accum 8

设计：
  - Thinker 冻结；只训练 Talker（LoRA）。
  - 每个样本：`text`（103 音色记法）→ Thinker 文本 hidden → Talker 生成 16 层 codec token。
  - labels = EnCodec codes (T, 16)，首层走 codec_head CE，残差层走 code_predictor（MTP）。
  - codec vocab 对齐：EnCodec codebook_size=1024；若模型 codec_head 词表不是 1024，
    需要替换 `codec_head` 与 `code_predictor` 的输出层。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TalkerDataset(Dataset):
    def __init__(self, path, max_frames=200):
        self.rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        self.max_frames = max_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        z = np.load(r["codes"])
        codes = z["codes"]  # (1, 1, layers, T)
        codes = torch.from_numpy(codes.astype("long")).squeeze(0).squeeze(0)  # (layers, T)
        codes = codes[:, : self.max_frames]
        return {"id": r["id"], "text": r["text"], "codes": codes,
                "dur": r["dur"], "family": r.get("family")}


def collate(batch):
    texts = [b["text"] for b in batch]
    max_t = max(b["codes"].shape[1] for b in batch)
    layers = batch[0]["codes"].shape[0]
    codes = torch.zeros((len(batch), layers, max_t), dtype=torch.long)
    mask = torch.zeros((len(batch), max_t), dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["codes"].shape[1]
        codes[i, :, :t] = b["codes"]
        mask[i, :t] = True
    return {"texts": texts, "codes": codes, "mask": mask, "ids": [b["id"] for b in batch]}


def inspect_codec_head(model_path):
    """检查 Talker codec_head 的 vocab 与 EnCodec 码本是否对齐。"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    talker = getattr(cfg, "talker_config", None)
    if talker is None:
        print("[warn] config 没有 talker_config")
        return
    vocab = getattr(talker, "vocab_size", None) or getattr(talker.text_config, "vocab_size", None)
    print("[info] talker vocab_size =", vocab)
    print("[expect] EnCodec codebook_size = 1024；若不一致需要替换 codec_head/MTP 输出层")


def dry_run(args):
    ds = TalkerDataset(args.data)
    print("dataset", len(ds), "sample", json.dumps(
        {"id": ds[0]["id"], "text": ds[0]["text"], "codes": list(ds[0]["codes"].shape),
         "dur": ds[0]["dur"]}, ensure_ascii=False), flush=True)
    batch = collate([ds[i] for i in range(min(4, len(ds)))])
    print("batch texts", batch["texts"], flush=True)
    print("codes", tuple(batch["codes"].shape), "mask", tuple(batch["mask"].shape), flush=True)
    print("codec vocab max", int(batch["codes"].max()), "min", int(batch["codes"].min()),
          flush=True)
    if batch["codes"].max() >= 1024:
        print("[warn] codes 超出 1024，需要检查 codec 层", flush=True)
    print("dry-run OK", flush=True)


def train(args):
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, "free GB",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
          if device == "cuda" else "n/a", flush=True)
    inspect_codec_head(args.model)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    # 冻结 Thinker，只训 Talker（LoRA 由 peft 注入）
    for p in model.thinker.parameters():
        p.requires_grad = False
    for p in model.talker.parameters():
        p.requires_grad = True

    ds = TalkerDataset(args.data)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.talker.parameters() if p.requires_grad],
                            lr=args.lr)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            codes = batch["codes"].to(device)  # (B, 16, T)
            texts = batch["texts"]
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
            # Talker 目标：第一层 + 残差层
            labels = codes[:, 0, :]  # (B, T) 第一层码本
            # 说明：Thinker 的文本 hidden 通过 model.thinker 获取；
            # Talker forward 需要 inputs_embeds/trailing_text_hidden，完整拼接见 TODO。
            out = model.talker(input_ids=inputs["input_ids"], labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item() * args.grad_accum:.4f}",
                      flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.talker.state_dict(), out_dir / "talker_sft.pt")
    print("saved", out_dir / "talker_sft.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/gemini/pretrain/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--data", default="/gemini/code/lightfour-train/data/omni_talker/talker_train.jsonl")
    ap.add_argument("--out", default="/gemini/code/lightfour-train/checkpoints/omni_talker_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
