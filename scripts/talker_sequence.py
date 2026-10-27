"""序列级 Talker 数据 + 16 层 codec CE（可脱离 30B 权重单测）。

解决两件事：
  1. @Data 交付的 3–8s 序列（sequence_samples.csv + sequence_hits.csv）怎么变成
     Talker 能吃的 (layers, T) codec 目标 —— 每个击打按 offset 拼到 75Hz 时间轴上，
     空档填 codec_pad_id，并返回 frame_mask（哪些帧是真击打，哪些是补位）。
  2. transformers 5.16.1 的 Talker.forward 只算第 1 层 CE；这里实现第 1 层 +
     code_predictor 15 个残差层的 teacher forcing，15 个 lm_head 各出一次 CE。

帧率：EnCodec 24kHz / hop 320 = 75Hz。Talker 本体是 13Hz，但 13Hz 下 60–200ms
的击打只剩 0.8–2.6 帧（Qwen3-TTS 12.5Hz 已被判死的原因），所以序列监督按 75Hz
原生帧率做，不改用 13Hz 时间轴。代价是 3–8s → 225–600 帧（=Codec 报的 5.77×）。

用法：
  python scripts/talker_sequence.py --self-test        # 纯 torch，本地可跑
  python scripts/talker_sequence.py --stats            # 统计序列→帧数分布
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_RATE = 75          # EnCodec 24kHz / hop 320
LAYERS = 16              # Talker codec groups
CODEBOOK = 1024          # EnCodec 24kbps codebook_size
DEFAULT_PAD = 2050       # 占位；必须用 config.codec_pad_id 覆盖后再开训

BASE = Path("/gemini/code/lightfour-train")


# ---------------------------------------------------------------- 数据拼装
def stitch_hit_codes(hit_codes, offsets, dur, pad_id=DEFAULT_PAD,
                     frame_rate=FRAME_RATE, layers=LAYERS):
    """把单次击打的 codec codes 按 offset 拼进一条时间轴。

    hit_codes: list of (layers, Ti) int
    offsets:   每击在序列内的起始秒
    returns:   codes (layers, T) int64, frame_mask (T,) bool, conflicts int
    """
    T = max(1, int(round(dur * frame_rate)))
    codes = np.full((layers, T), int(pad_id), dtype=np.int64)
    frame_mask = np.zeros(T, dtype=bool)
    conflicts = 0
    for c, off in zip(hit_codes, offsets):
        assert c.ndim == 2 and c.shape[0] == layers, c.shape
        s = int(round(float(off) * frame_rate))
        if s >= T:
            continue
        e = min(s + c.shape[1], T)
        w = e - s
        if w <= 0:
            continue
        if frame_mask[s:e].any():
            conflicts += 1
        codes[:, s:e] = c[:, :w]
        frame_mask[s:e] = True
    return codes, frame_mask, conflicts


def load_codes(path, layers=LAYERS):
    """读 npz 里的 codes，兼容 (1,1,L,T) / (1,L,T) / (L,T) / (T,L)。"""
    a = np.load(path)["codes"]
    if a.ndim == 4:
        a = a[0, 0]
    elif a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"bad codes shape {a.shape} in {path}")
    if a.shape[0] != layers and a.shape[1] == layers:
        a = a.T
    if a.shape[0] != layers:
        raise ValueError(f"expect {layers} layers, got {a.shape} in {path}")
    return a.astype(np.int64)


class SequenceDataset(torch.utils.data.Dataset):
    """一条样本 = 一条 3–8s 序列（多击打），text 为整条记法序列。"""

    def __init__(self, samples_csv, hits_csv, codes_dir, split=None,
                 layers=LAYERS, frame_rate=FRAME_RATE, max_frames=600):
        self.layers = layers
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        self.codes_dir = Path(codes_dir)
        self.hits = {}
        with open(hits_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.hits.setdefault(r["sequence_id"], []).append(r)
        self.rows = []
        with open(samples_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if split and r["split"] != split:
                    continue
                if r.get("audio_available", "1") != "1":
                    continue
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        hs = sorted(self.hits[r["sequence_id"]], key=lambda x: float(x["offset"]))
        codes, offsets, missing = [], [], 0
        for h in hs:
            p = self.codes_dir / f"{h['hit_id']}.npz"
            if not p.exists():
                missing += 1
                continue
            codes.append(load_codes(p, self.layers))
            offsets.append(float(h["offset"]))
        if not codes:
            raise FileNotFoundError(f"no hit codes for {r['sequence_id']}")
        arr, mask, conflicts = stitch_hit_codes(
            codes, offsets, float(r["dur"]), frame_rate=self.frame_rate,
            layers=self.layers)
        if arr.shape[1] > self.max_frames:
            arr = arr[:, : self.max_frames]
            mask = mask[: self.max_frames]
        return {
            "id": r["sequence_id"],
            "text": r["sound_notations"].replace("|", " | "),
            "codes": torch.from_numpy(arr),
            "mask": torch.from_numpy(mask),
            "dur": float(r["dur"]), "n_hits": len(codes),
            "missing_hits": missing, "conflicts": conflicts,
            "split": r["split"],
        }


def collate_sequences(batch, pad_id=DEFAULT_PAD):
    layers = batch[0]["codes"].shape[0]
    T = max(b["codes"].shape[1] for b in batch)
    codes = torch.full((len(batch), layers, T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["codes"].shape[1]
        codes[i, :, :t] = b["codes"]
        mask[i, :t] = b["mask"]
    return {"ids": [b["id"] for b in batch], "texts": [b["text"] for b in batch],
            "codes": codes, "mask": mask}


# ---------------------------------------------------------------- 16 层 CE
def codec_ce_16(hidden, codes, mask, *, codec_head, input_embedding,
                predictor, layer1_logits=None, codebook=CODEBOOK,
                shift=0, weights=None):
    """第 1 层 codec_head CE + 15 个残差层 teacher forcing CE。

    hidden:   (B,T,H) Talker 输出（Thinker hidden 拼进来的那一路）
    codes:    (B,L,T) L=16，codes[:,0] 是第一层
    mask:     (B,T) bool，False = 补位（不参与 loss）
    predictor: 具备 forward(inputs_embeds=...) -> .logits 的模块；
               约定 generation_steps = seq_len-2 选中 lm_head[k]（与 generate 一致）
    shift:    0 = hidden[t] 预测 codes[t]（与 16layer smoke 同口径）；
              1 = 预测 codes[t+1]。两种口径不能混用，改变前必须与模型 forward 对齐。
    """
    B, T, H = hidden.shape
    L = codes.shape[1]
    assert L == 1 + len(predictor.lm_head), (L, len(predictor.lm_head))
    valid = mask.bool()
    if shift:
        valid = torch.zeros_like(valid)
        valid[:, :-shift] = mask[:, shift:].bool()
    assert int(codes[:, :, :].max()) < codebook, "codec target 越界，必须 < codebook"

    if layer1_logits is None:
        layer1_logits = codec_head(hidden)                      # (B,T,V)
    # PI 裁决③：CE 只切 0..codebook-1。codec_head 是 3072 类（2048 码字 + 1024 special），
    # 全 3072 类做 CE 会把 codec_bos/eos 等 special 当负样本持续压低；v0 定长生成不用 eos。
    layer1_logits = layer1_logits[..., :codebook]
    ce = F.cross_entropy(layer1_logits.reshape(-1, codebook),
                         codes[:, 0].reshape(-1), reduction="none").reshape(B, T)

    h = hidden.reshape(B * T, 1, H)
    l1 = input_embedding(codes[:, 0].reshape(B * T)).unsqueeze(1)
    res = codes[:, 1:].permute(0, 2, 1).reshape(B * T, L - 1)   # (B*T, 15)
    embs = predictor.model.get_input_embeddings()
    # 单次前向覆盖全部 15 个残差组：位置 p 只因果见到 res[:p-2]，
    # 因此 hidden[:, k+1] 正好是「给定 res[:k] 预测 res[k]」的那一步，
    # 逐位套 lm_head[k] 即等价于 15 次逐步前向（已在 5.16.1 真实类上验过，
    # max|Δ| 2.1e-6）。省掉 14 次重复 5 层前向。
    parts = [h, l1]
    for i in range(L - 2):
        parts.append(embs[i](res[:, i]).unsqueeze(1))
    hid = predictor.model(inputs_embeds=torch.cat(parts, dim=1)).last_hidden_state
    per_layer = [ce]
    for k in range(L - 1):
        # code_predictor 的 lm_head 输出 2048 类，同样只切 0..codebook-1
        logits = predictor.lm_head[k](hid[:, k + 1])
        per_layer.append(F.cross_entropy(logits[:, :codebook].float(), res[:, k],
                                         reduction="none").reshape(B, T))

    flat_mask = valid.reshape(-1)
    w = torch.ones(L, device=hidden.device) if weights is None else weights
    terms = [(p.reshape(-1)[flat_mask].mean(), w[i]) for i, p in enumerate(per_layer)]
    total = sum(t * wi for t, wi in terms)
    return {
        "loss": total,
        "layer1": float(terms[0][0]),
        "residual_mean": float(sum(float(t) for t, _ in terms[1:]) / (L - 1)),
        "num_terms": len(terms), "num_frames": int(flat_mask.sum()),
        "n_valid": int(flat_mask.sum()),
    }


# ---------------------------------------------------------------- 自测
class _FakePredictor(nn.Module):
    """只用形状/梯度语义复刻 code_predictor：15 个 lm_head + 15 张按组输入的嵌入表。

    model 用 cumsum 代替 5 层注意力，保证因果性（位置 p 只见 p 及之前）且 h 能拿到梯度。
    """

    class _Body(nn.Module):
        def __init__(self, hidden, vocab, n):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(vocab, hidden) for _ in range(n)])

        def get_input_embeddings(self):
            return self.embs

        def forward(self, inputs_embeds):
            return SimpleNamespace(last_hidden_state=torch.cumsum(inputs_embeds, dim=1))

    def __init__(self, hidden, vocab, layers=LAYERS):
        super().__init__()
        self.lm_head = nn.ModuleList([nn.Linear(hidden, vocab) for _ in range(layers - 1)])
        self.model = _FakePredictor._Body(hidden, vocab, layers - 1)

    @property
    def _embs(self):
        return self.model.embs

    def forward(self, inputs_embeds):
        raise AssertionError("codec_ce_16 现在只调用 predictor.model / lm_head，不应走整合 forward")


def self_test():
    torch.manual_seed(0)
    B, T, H, V = 2, 9, 16, CODEBOOK
    hidden = torch.randn(B, T, H, requires_grad=True)
    codes = torch.randint(0, V, (B, LAYERS, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 6:] = False          # 右侧补位，必须被排除

    head = nn.Linear(H, 3072)     # 3072 = 2048 codec + 1024 special
    emb = nn.Embedding(3072, H)
    emb.weight = head.weight      # tied，与真实模型一致
    pred = _FakePredictor(H, 3072)

    out = codec_ce_16(hidden, codes, mask, codec_head=head, input_embedding=emb,
                      predictor=pred)
    assert out["num_terms"] == 16, out
    assert out["num_frames"] == B * T - 3, out          # 3 个补位帧被剔除
    assert torch.isfinite(out["loss"]), out

    out["loss"].backward()
    grads = {n: p.grad for n, p in pred.named_parameters() if p.grad is not None}
    # 15 个 lm_head 全部要有梯度；输入侧只有 emb[0..13]，emb[14] 是最后一层目标、
    # teacher forcing 里永远不作为输入（missing 是设计，不是漏训）
    for k in range(LAYERS - 1):
        assert pred.lm_head[k].weight.grad.abs().sum() > 0, f"lm_head[{k}] 无梯度"
    for k in range(LAYERS - 2):
        assert pred._embs[k].weight.grad.abs().sum() > 0, f"emb[{k}] 无梯度"
    assert pred._embs[LAYERS - 2].weight.grad is None, "emb[14] 不应有梯度"
    assert len(grads) == 3 * (LAYERS - 1) - 1, len(grads)
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    # 第 15 个 lm_head 必须真的有梯度（否则就是只训了前几层）
    assert pred.lm_head[LAYERS - 2].weight.grad.abs().sum() > 0

    # 只监督第 1 层时 loss 不同 → 证明 15 层真的进了总 loss
    # 第 1 层单独 CE 与总 loss 必须不同，否则说明残差层没进 loss
    l1_only = float(F.cross_entropy(head(hidden)[..., :CODEBOOK][mask].reshape(-1, CODEBOOK),
                                    codes[:, 0][mask].reshape(-1)))
    assert abs(l1_only - out["layer1"]) < 1e-5, (l1_only, out["layer1"])
    assert abs(float(out["loss"]) - l1_only) > 1e-3, "总 loss 等于第 1 层，残差层未生效"
    print(f"self-test OK: terms=16 frames={out['num_frames']} "
          f"loss={float(out['loss']):.3f} layer1={out['layer1']:.3f} "
          f"residual={out['residual_mean']:.3f} grads={len(grads)}")

    # 拼装：两击 + 空档
    c1 = np.random.randint(0, CODEBOOK, (LAYERS, 24))
    c2 = np.random.randint(0, CODEBOOK, (LAYERS, 30))
    arr, fm, conf = stitch_hit_codes([c1, c2], [0.1, 1.0], 2.0)
    assert arr.shape == (LAYERS, 150), arr.shape
    assert fm.sum() == 54 and conf == 0, (fm.sum(), conf)
    # c1 占 8..31（offset .1s×75），c2 占 75..104；空档 32..74 与尾部必须仍是 pad
    assert (arr[:, :8] == DEFAULT_PAD).all() and (arr[:, 32:75] == DEFAULT_PAD).all()
    assert (arr[:, 105:] == DEFAULT_PAD).all()
    print(f"stitch OK: {arr.shape} hits={int(fm.sum())} frames conflicts={conf}")


def stats(args):
    hits, samples = {}, {}
    with open(args.hits, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hits.setdefault(r["sequence_id"], []).append(r)
    with open(args.samples, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            samples[r["sequence_id"]] = r
    n = len(samples)
    have = sum(1 for v in samples.values() if v.get("audio_available") == "1")
    frames = [int(round(float(r["dur"]) * FRAME_RATE)) for r in samples.values()]
    over = sum(1 for x in frames if x > args.max_frames)
    per = [len(hits.get(s, [])) for s in samples]
    print(json.dumps({
        "sequences": n, "audio_available": have, "hits_rows": sum(per),
        "frames_min": min(frames), "frames_med": sorted(frames)[len(frames) // 2],
        "frames_max": max(frames), "over_max_frames": over,
        "max_frames": args.max_frames, "frame_rate": FRAME_RATE,
        "note": "13Hz 口径下 3–8s 只有 39–104 帧，但 60–200ms 击打会被压成 <3 帧",
    }, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(BASE / "data/sequence_samples.csv"))
    ap.add_argument("--hits", default=str(BASE / "data/sequence_hits.csv"))
    ap.add_argument("--codes-dir", default=str(BASE / "data/omni_talker/codes"))
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.stats:
        stats(args)
    else:
        ds = SequenceDataset(args.samples, args.hits, args.codes_dir,
                             max_frames=args.max_frames)
        print("dataset", len(ds), " rows; first id", ds.rows[0]["sequence_id"])


if __name__ == "__main__":
    main()
