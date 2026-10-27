"""
序列 token 数据集加载器（Talker 直接可用）。

数据形态（由 scripts/prepare_seq_tokens.py 在训练服务器上生成）：
  data/seq_tokens/codes/<sequence_id>.npz   codes (1,1,16,T)，T = 75Hz 帧数
  data/seq_tokens/sequence_manifest.jsonl   每条序列一行（T / dur / text / split / md5）
  data/seq_tokens/hit_frames.csv            逐击 -> 帧区间映射

为什么是「整段编码」而不是「逐击编码再拼接」：
  EnCodec 是有感受野的编解码器，单独编码一段 200ms 片段得到的码流，
  与同一段音频处在长音频中的码流不同；逐击编码再按 offset 拼接，
  既得不到真实时间轴上的码，又会在空档塞入码本外的 pad 码
  （codec_ce_16 断言 target < 1024，pad=2050 会直接触发 AssertionError）。
  整段编码后每帧都是真实码，帧区间只用于「记法 -> 帧」的对齐标注。

输出给 Talker 的 batch：
  codes (N,16,T_max)   int64，padding 用 pad_id（必须是码本内的合法值）
  mask  (N,T_max)      bool，True = 有效帧（整段编码下全部为 True）
  spans                每条序列的击打帧区间，供记法序列对齐
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

FRAME_RATE = 75
LAYERS = 16
CODEBOOK = 1024


class SeqTokenDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, codes_dir, hit_frames_csv=None, split=None,
                 max_frames=600, pad_id=1023):
        self.rows = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if split and r["split"] != split:
                    continue
                if int(r["T"]) > max_frames:
                    continue
                self.rows.append(r)
        self.codes_dir = Path(codes_dir)
        self.pad_id = int(pad_id)
        self.max_frames = int(max_frames)
        self.spans = {}
        if hit_frames_csv and Path(hit_frames_csv).exists():
            with open(hit_frames_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.spans.setdefault(r["sequence_id"], []).append(
                        (int(r["hit_index"]), int(r["frame_start"]), int(r["frame_end"]),
                         r["sound_notation"], r["sound_family"], r["label_source"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        c = np.load(self.codes_dir / f"{r['sequence_id']}.npz")["codes"]
        assert c.ndim == 4 and c.shape[:3] == (1, 1, LAYERS), (r["sequence_id"], c.shape)
        arr = c[0, 0].astype(np.int64)                      # (16, T)
        assert int(arr.min()) >= 0 and int(arr.max()) < CODEBOOK, \
            f"{r['sequence_id']} 码值越界"
        T = arr.shape[1]
        assert abs(T - round(float(r["dur"]) * FRAME_RATE)) <= 1, (r["sequence_id"], T)
        if T > self.max_frames:
            arr = arr[:, : self.max_frames]
            T = self.max_frames
        # frame -> 击打序号（-1 = 击打之间的空档帧）。相邻击打帧区间有 ~20ms
        # 重叠（切片起点是 onset-0.02s 的 pre-roll），重叠帧归属后一击。
        slot = np.full(T, -1, dtype=np.int64)
        for hi, fs, fe, *_ in self.spans.get(r["sequence_id"], []):
            fs2 = min(max(fs, 0), T)
            fe2 = min(max(fe, 0), T)
            if fe2 > fs2:
                slot[fs2:fe2] = hi
        return {
            "id": r["sequence_id"],
            "text": r["text"],
            "codes": torch.from_numpy(arr),                # (16, T)
            "mask": torch.ones(T, dtype=torch.bool),        # 整段编码，无补位帧
            "slot": torch.from_numpy(slot),                 # (T,) 击打序号，-1=空档
            "dur": float(r["dur"]),
            "n_hits": int(r.get("n_hits", 0)),
            "split": r["split"],
            "spans": self.spans.get(r["sequence_id"], []),
        }


def collate(batch, pad_id=1023):
    layers = LAYERS
    T = max(b["codes"].shape[1] for b in batch)
    codes = torch.full((len(batch), layers, T), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.bool)
    slot = torch.full((len(batch), T), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["codes"].shape[1]
        codes[i, :, :t] = b["codes"]
        mask[i, :t] = b["mask"]
        slot[i, :t] = b["slot"]
    return {
        "ids": [b["id"] for b in batch],
        "texts": [b["text"] for b in batch],
        "codes": codes,          # (N,16,T_max)
        "mask": mask,            # (N,T_max)
        "slot": slot,            # (N,T_max) 帧 -> 击打序号，-1 = 空档
        "spans": [b["spans"] for b in batch],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/gemini/code/lightfour-train/data/seq_tokens/sequence_manifest.jsonl")
    ap.add_argument("--codes-dir", default="/gemini/code/lightfour-train/data/seq_tokens/codes")
    ap.add_argument("--hit-frames", default="/gemini/code/lightfour-train/data/seq_tokens/hit_frames.csv")
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--pad-id", type=int, default=1023,
                    help="批内 padding 码（必须在 [0,1024) 内；"
                         "注意这与 config.codec_pad_id 是两回事，前者是 padding，"
                         "后者是控制 token")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        # 纯合成数据自测：形状、padding、mask、码值域
        ds = SeqTokenDataset.__new__(SeqTokenDataset)
        ds.rows = [{"sequence_id": "S1", "text": "a | b", "T": 300, "dur": 4.0,
                    "split": "train", "n_hits": 2},
                   {"sequence_id": "S2", "text": "c", "T": 240, "dur": 3.2,
                    "split": "train", "n_hits": 1}]
        ds.codes_dir = None
        ds.pad_id = 1023
        ds.max_frames = 600
        ds.spans = {}
        arr = np.random.randint(0, 1024, (LAYERS, 300))
        items = [{"id": "S1", "text": "a | b", "codes": torch.from_numpy(arr),
                  "mask": torch.ones(300, dtype=torch.bool),
                  "slot": torch.zeros(300, dtype=torch.long), "dur": 4.0, "n_hits": 2,
                  "split": "train", "spans": []},
                 {"id": "S2", "text": "c",
                  "codes": torch.from_numpy(arr[:, :240]), "mask": torch.ones(240, dtype=torch.bool),
                  "slot": torch.zeros(240, dtype=torch.long),
                  "dur": 3.2, "n_hits": 1, "split": "train", "spans": []}]
        b = collate(items, pad_id=1023)
        assert b["codes"].shape == (2, LAYERS, 300), b["codes"].shape
        assert b["mask"].shape == (2, 300)
        assert b["slot"].shape == (2, 300)
        assert bool(b["mask"][1, 240:].sum() == 0), "padding 帧必须是 mask=False"
        assert int(b["slot"][1, 240:].max() == -1), "padding 帧 slot 必须是 -1"
        assert int(b["codes"].max()) < CODEBOOK
        assert int(b["codes"][1, :, 240:].unique().tolist() == [1023]), "padding 码不符"
        print(f"self-test OK: codes {tuple(b['codes'].shape)} mask {tuple(b['mask'].shape)} "
              f"slot {tuple(b['slot'].shape)} pad=1023 有效帧 {int(b['mask'].sum())}")
        return

    ds = SeqTokenDataset(args.manifest, args.codes_dir, args.hit_frames,
                         split=args.split, max_frames=args.max_frames,
                         pad_id=args.pad_id)
    print(f"dataset: {len(ds)} 条 (split={args.split})")
    b = collate([ds[i] for i in range(min(4, len(ds)))], pad_id=args.pad_id)
    print(f"batch: codes {tuple(b['codes'].shape)} mask {tuple(b['mask'].shape)} "
          f"ids={b['ids']}")
    print(f"text[0]: {b['texts'][0][:80]}")


if __name__ == "__main__":
    main()
