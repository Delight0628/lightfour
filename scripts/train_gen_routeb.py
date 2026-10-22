"""
Route B — 条件 Flow Matching 生成 beatbox log-mel。

条件：鼓谱（4 类 piano-roll）+ tempo
目标：2 秒窗口的 log-mel（64 x 122，16kHz / hop 256）
模型：小型 2D U-Net（约 3-4M 参数），CPU 可跑 smoke，24GB GPU 可全量

训练配方（Conditional Flow Matching）：
  x1 = 真实 mel（标准化后）
  x0 ~ N(0,1)
  xt = (1-t) * x0 + t * x1
  目标速度 v = x1 - x0
  loss = MSE(model(xt, cond, t), v)
推理：从噪声出发，Euler 积分 20-50 步得到 mel；mel 再交给声码器/声码器替代方案。

数据依赖（由 prepare_gen_routeb.py 产出）：
  data/gen_synth_routeb/
    mel/shard_XXXXX.npy    (2000, 122, 64) float32
    train.jsonl / val.jsonl
    routeb_stats.json      mel_mean / mel_std

运行（CPU smoke，100 步）：
  /root/miniconda3/bin/python scripts/train_gen_routeb.py \
    --root /gemini/code/lightfour-train/data/gen_synth_routeb \
    --out /gemini/code/lightfour-train/checkpoints/gen_routeb_smoke \
    --smoke-steps 100 --batch-size 8 --device cpu --num-workers 0

全量（24GB GPU）：
  /root/miniconda3/bin/python scripts/train_gen_routeb.py \
    --root /gemini/code/lightfour-train/data/gen_synth_routeb \
    --out /gemini/code/lightfour-train/checkpoints/gen_routeb \
    --epochs 30 --batch-size 64 --device cuda --num-workers 4
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FRAMES = 122
N_MELS = 64
WIN_SEC = 2.0
CLASSES = ["kd", "sd", "hhc", "hho"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_cond(rec):
    """(5, 122)：前 4 通道为鼓类 piano-roll，第 5 通道为归一化 tempo。"""
    cond = np.zeros((5, FRAMES), dtype=np.float32)
    for n in rec.get("score", []):
        f = int(round(float(n["onset"]) / WIN_SEC * (FRAMES - 1)))
        f = min(max(f, 0), FRAMES - 1)
        ci = CLASSES.index(n["cls"])
        cond[ci, f] = 1.0
        if f + 1 < FRAMES:
            cond[ci, f + 1] = max(cond[ci, f + 1], 0.5)
    tempo = min(max((float(rec.get("tempo_bpm", 115.0)) - 70.0) / 90.0, 0.0), 1.5)
    cond[4, :] = tempo
    return cond


class RouteBDataset(Dataset):
    def __init__(self, index_path, root, max_samples=0, seed=42):
        self.root = Path(root)
        self.items = list(read_jsonl(index_path))
        if max_samples and max_samples < len(self.items):
            random.Random(seed).shuffle(self.items)
            self.items = self.items[:max_samples]
        self._mm = {}

    def __len__(self):
        return len(self.items)

    def _mel(self, rec):
        key = rec["shard"]
        if key not in self._mm:
            self._mm[key] = np.load(self.root / rec["mel"], mmap_mode="r")
        # npy 里是 (帧 122, mel 64)，模型输入要 (mel 64, 帧 122)
        return np.ascontiguousarray(self._mm[key][rec["row"]].T, dtype=np.float32)

    def __getitem__(self, i):
        rec = self.items[i]
        mel = self._mel(rec)
        cond = build_cond(rec)
        return torch.from_numpy(mel).unsqueeze(0), torch.from_numpy(cond)


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CondUNet(nn.Module):
    def __init__(self, base=32, emb=128):
        super().__init__()
        self.emb_dim = emb
        self.time_mlp = nn.Sequential(
            nn.Linear(emb, emb * 4), nn.SiLU(), nn.Linear(emb * 4, emb * 4)
        )
        self.inp = nn.Conv2d(6, base, 3, padding=1)
        self.e1 = ResBlock(base, base, emb * 4)
        self.d1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.e2 = ResBlock(base * 2, base * 2, emb * 4)
        self.d2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.e3 = ResBlock(base * 4, base * 4, emb * 4)
        self.d3 = nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1)
        self.mid = ResBlock(base * 8, base * 8, emb * 4)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1)
        self.r3 = ResBlock(base * 8, base * 4, emb * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.r2 = ResBlock(base * 4, base * 2, emb * 4)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.r1 = ResBlock(base * 2, base, emb * 4)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    @staticmethod
    def _match(h, skip):
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        return h

    def forward(self, x, cond, t):
        emb = self.time_mlp(time_embedding(t, self.emb_dim))
        cond = cond.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        h = self.inp(torch.cat([x, cond], dim=1))
        e1 = self.e1(h, emb)
        e2 = self.e2(self.d1(e1), emb)
        e3 = self.e3(self.d2(e2), emb)
        m = self.mid(self.d3(e3), emb)
        h = self.r3(torch.cat([self._match(self.u3(m), e3), e3], 1), emb)
        h = self.r2(torch.cat([self._match(self.u2(h), e2), e2], 1), emb)
        h = self.r1(torch.cat([self._match(self.u1(h), e1), e1], 1), emb)
        return self.out(h)


def fm_loss(model, mel, cond, mean, std, device):
    x1 = (mel - mean) / std
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    tb = t[:, None, None, None]
    xt = (1.0 - tb) * x0 + tb * x1
    v = x1 - x0
    pred = model(xt, cond, t)
    return F.mse_loss(pred, v)


@torch.no_grad()
def sample(model, cond, steps=30, device="cpu"):
    model.eval()
    x = torch.randn(cond.shape[0], 1, N_MELS, FRAMES, device=device)
    for i in range(steps):
        t = torch.full((cond.shape[0],), (i + 0.5) / steps, device=device)
        v = model(x, cond, t)
        x = x + v / steps
    model.train()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gemini/code/lightfour-train/data/gen_synth_routeb")
    ap.add_argument("--index", default="")
    ap.add_argument("--val-index", default="")
    ap.add_argument("--stats", default="")
    ap.add_argument("--out", default="/gemini/code/lightfour-train/checkpoints/gen_routeb")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--smoke-steps", type=int, default=0, help=">0 时只跑指定步数后退出")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--val-samples", type=int, default=0)
    ap.add_argument("--sample-n", type=int, default=4, help="smoke 结束生成几条样例 mel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root)
    index_path = Path(args.index) if args.index else root / "train.jsonl"
    val_path = Path(args.val_index) if args.val_index else root / "val.jsonl"
    stats_path = Path(args.stats) if args.stats else root / "routeb_stats.json"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = json.load(open(stats_path, encoding="utf-8")) if stats_path.exists() else {}
    mean = float(stats.get("mel_mean", -4.0))
    std = float(stats.get("mel_std", 3.0)) or 3.0
    print(f"[stats] mean={mean:.4f} std={std:.4f}", flush=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] {device}", flush=True)

    train_ds = RouteBDataset(index_path, root, args.max_train_samples, args.seed)
    val_ds = RouteBDataset(val_path, root, args.val_samples or 0, args.seed) if val_path.exists() else None
    print(f"[data] train={len(train_ds)} val={len(val_ds) if val_ds else 0}", flush=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    ) if val_ds else None

    model = CondUNet(args.base_channels, args.emb_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.smoke_steps or max(1, args.epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    step = 0
    t0 = time.time()
    running = 0.0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for mel, cond in train_dl:
            mel = mel.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            loss = fm_loss(model, mel, cond, mean, std, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()

            if step % args.log_every == 0 or step == 1:
                avg = running / args.log_every if step > 1 else loss.item()
                running = 0.0
                lr = sched.get_last_lr()[0]
                print(
                    f"step {step} loss {avg:.5f} lr {lr:.2e} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

            if args.smoke_steps and step >= args.smoke_steps:
                break

        if args.smoke_steps and step >= args.smoke_steps:
            break

        # 验证
        if val_dl is not None:
            model.eval()
            vloss = 0.0
            nb = 0
            with torch.no_grad():
                for mel, cond in val_dl:
                    mel = mel.to(device)
                    cond = cond.to(device)
                    vloss += fm_loss(model, mel, cond, mean, std, device).item()
                    nb += 1
            vloss /= max(nb, 1)
            print(f"[epoch {epoch}] val_loss {vloss:.5f}", flush=True)
            if vloss < best_val:
                best_val = vloss
                torch.save(
                    {"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args)},
                    out / "best.pt",
                )
        torch.save(
            {"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args)},
            out / "last.pt",
        )

    # smoke 结束：生成几条样例 mel，验证推理路径
    if args.sample_n > 0 and val_ds is not None and len(val_ds) > 0:
        n = min(args.sample_n, len(val_ds))
        mels, conds = [], []
        for i in range(n):
            mel, cond = val_ds[i]
            mels.append(mel)
            conds.append(cond)
        cond = torch.stack(conds).to(device)
        gen = sample(model, cond, steps=30, device=device)
        gen = gen * std + mean
        np.save(out / "smoke_samples.npy", gen.detach().cpu().numpy())
        print(f"[sample] saved {n} x {tuple(gen.shape[1:])} -> {out/'smoke_samples.npy'}", flush=True)
        print(
            f"[sample] range {float(gen.min()):.2f} ~ {float(gen.max()):.2f}",
            flush=True,
        )

    torch.save(
        {"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args)},
        out / "smoke.pt" if args.smoke_steps else out / "last.pt",
    )
    print(f"[done] steps={step} out={out}", flush=True)


if __name__ == "__main__":
    main()
