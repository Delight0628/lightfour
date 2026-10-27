"""
小型条件 token 预测器：验证「103 音色 → EnCodec 16 层 token」在 6.2GB 上可学、可生成。

这不是最终模型，只是 B 路线（Omni 换 codec_head）的可行性前置实验：
  - 若小模型能把训练集 loss 从 ln(1024)≈6.93 明显压低，并生成可辨识音频，
    说明 75Hz/16 层/1024 的 token 空间是可学的，值得投入 48GB 改 Omni。
  - 若学不动或生成不可辨识，说明问题在 codec 空间本身，应转 E 或换 codec。

用法：
  python scripts/train_tiny_codec_lm.py train --epochs 30 --out /tmp/tiny_codec_lm.pt
  python scripts/train_tiny_codec_lm.py generate --ckpt /tmp/tiny_codec_lm.pt --out-dir /tmp/tiny_codec_gen
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path("/gemini/code/lightfour-train")


class TinyCodecLM(nn.Module):
    def __init__(self, vocab, codebook=1024, layers=16, dim=128, nhead=4, nlayers=3, max_t=256):
        super().__init__()
        self.layers = layers
        self.codebook = codebook
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_t, dim)
        block = nn.TransformerEncoderLayer(dim, nhead, dim * 4, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(block, nlayers)
        self.head = nn.Linear(dim, layers * codebook)

    def forward(self, cond_ids, t_len):
        b = cond_ids.shape[0]
        pos = torch.arange(t_len, device=cond_ids.device)
        h = self.tok(cond_ids)[:, None, :] + self.pos(pos)[None, :, :]
        h = self.encoder(h)
        return self.head(h).view(b, t_len, self.layers, self.codebook)


def load_rows(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    notations = sorted({r["text"] for r in rows})
    ids = {n: i for i, n in enumerate(notations)}
    data = []
    for r in rows:
        z = np.load(r["codes"])["codes"]  # (1,1,16,T)
        codes = z.squeeze(0).squeeze(0).transpose(1, 0)  # (T,16)
        data.append({"cond": ids[r["text"]], "codes": codes, "notation": r["text"],
                     "dur": r["dur"]})
    return data, notations


def collate(batch):
    max_t = max(b["codes"].shape[0] for b in batch)
    layers = batch[0]["codes"].shape[1]
    codes = torch.zeros((len(batch), max_t, layers), dtype=torch.long)
    mask = torch.zeros((len(batch), max_t), dtype=torch.bool)
    cond = torch.tensor([b["cond"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["codes"].shape[0]
        codes[i, :t] = torch.from_numpy(b["codes"].astype("long"))
        mask[i, :t] = True
    return cond, codes, mask


def train(args):
    data, notations = load_rows(BASE / "data/omni_talker/talker_train.jsonl")
    print("rows", len(data), "notations", len(notations), flush=True)
    model = TinyCodecLM(len(notations), layers=args.layers, dim=args.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=collate)
    base = math.log(1024)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  base_loss {base:.3f}",
          flush=True)
    for epoch in range(1, args.epochs + 1):
        total, n = 0.0, 0
        for cond, codes, mask in loader:
            logits = model(cond, codes.shape[1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), codes.reshape(-1), reduction="none")
            loss = loss.view(codes.shape[0], codes.shape[1], codes.shape[2])
            loss = (loss * mask[..., None]).sum() / (mask.sum() * codes.shape[2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * mask.sum().item()
            n += mask.sum().item()
        print(f"epoch {epoch} loss {total / max(n, 1):.4f}", flush=True)
    torch.save({"state": model.state_dict(), "notations": notations,
                "config": {"layers": args.layers, "dim": args.dim}}, args.out)
    print("saved", args.out, flush=True)


def generate(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codec_tokens import load_model, decode_codes, SR
    import soundfile as sf

    ckpt = torch.load(args.ckpt, map_location="cpu")
    notations = ckpt["notations"]
    model = TinyCodecLM(len(notations), layers=ckpt["config"]["layers"],
                        dim=ckpt["config"]["dim"])
    model.load_state_dict(ckpt["state"])
    model.eval()

    # 按每个记法在训练集里的平均帧数生成（单次击打只有 0.07–1.0s，固定 30 帧会拉长）
    frames_of = {}
    gt_codes = {}
    if args.match_train:
        rows = [json.loads(l) for l in
                (BASE / "data/omni_talker/talker_train.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        bucket = {}
        for r in rows:
            bucket.setdefault(r["text"], []).append(r)
        for name, rs in bucket.items():
            ts = [np.load(r["codes"])["codes"].shape[-1] for r in rs]
            frames_of[name] = max(1, int(round(float(np.mean(ts)))))
            gt_codes[name] = np.load(rs[0]["codes"])["codes"]  # (1,1,16,T) 真值参照

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_model, _ = load_model(str(BASE / "models/encodec_24khz"), device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for i, name in enumerate(notations[: args.limit]):
            t = frames_of.get(name, args.frames) if args.match_train else args.frames
            cond = torch.tensor([i], dtype=torch.long)
            logits = model(cond, t)
            tokens = logits.argmax(dim=-1)[0].transpose(0, 1).cpu().numpy()  # (16,T)
            codes = tokens[None, None, :, :]  # (1,1,16,T)
            w = decode_codes(codes, enc_model, device)
            stem = f"{i:03d}_{name.replace('/', '_').replace(chr(92), '_')}"
            sf.write(str(out / f"{stem}_gen.wav"), w, SR)

            rec = {"idx": i, "notation": name, "frames": int(t),
                   "dur": round(len(w) / SR, 3)}
            if name in gt_codes:
                g = gt_codes[name]
                # 真值帧数与平均帧数可能不同，用真值自身长度解码做参照
                wg = decode_codes(g, enc_model, device)
                sf.write(str(out / f"{stem}_gt.wav"), wg, SR)
                rec["gt_frames"] = int(g.shape[-1])
                rec["gt_dur"] = round(len(wg) / SR, 3)
            manifest.append(rec)
            print(stem, "frames", t, "dur", rec["dur"], flush=True)
    with open(out / "gen_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("done ->", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--layers", type=int, default=16)
    t.add_argument("--out", default="/tmp/tiny_codec_lm.pt")
    t.set_defaults(func=train)

    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default="/tmp/tiny_codec_lm.pt")
    g.add_argument("--out-dir", default="/tmp/tiny_codec_gen")
    g.add_argument("--frames", type=int, default=30)
    g.add_argument("--limit", type=int, default=20)
    g.add_argument("--match-train", action="store_true",
                   help="按各记法在训练集的平均帧数生成，并同时输出真值重建音频")
    g.set_defaults(func=generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
