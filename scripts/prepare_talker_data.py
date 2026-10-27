"""
为 Qwen3-Omni Talker 准备训练数据：
  103 音色文本序列 + EnCodec 16 层 codec token。

用法：
  python scripts/prepare_talker_data.py --self-test
  python scripts/prepare_talker_data.py \
    --candidates data/real_bbox/sound_seed_candidates.jsonl \
    --out-dir data/omni_talker --layers 16

输出：
  data/omni_talker/codes/<id>.npz       codes shape (1, 1, 16, T)
  data/omni_talker/talker_train.jsonl   训练样本（含 split）
  data/omni_talker/talker_manifest.jsonl 全量审计（形状/层数/帧数/来源）
  data/omni_talker/failures.jsonl       失败清单

形状约定（transformers 5.16.1 EnCodec）：
  audio_codes = (B, C, Q, T)，Q=量化层数，T=帧数（24kHz / 320 hop = 75Hz）
  - 切片必须用 codes[:, :, :layers, :]（切 Q），不能用 codes[:, :, :layers]（会切 T）
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# codec_tokens 依赖 librosa/transformers，--self-test 不需要它们，
# 因此延迟到真正要编码时才导入，保证 self-test 在纯 numpy 环境可跑。

BASE = Path("/gemini/code/lightfour-train")
HOP = 320  # 24kHz / 75Hz


def audit_shape(codes, y, layers):
    """校验 (B, C, Q, T) 形状与层数/帧数。"""
    assert codes.ndim == 4, f"expect 4D (B,C,Q,T), got {codes.shape}"
    assert codes.shape[0] == 1 and codes.shape[1] == 1, f"expect single item, got {codes.shape}"
    assert codes.shape[2] == layers, f"expect {layers} layers, got {codes.shape[2]}"
    expected_t = int(round(len(y) / HOP))
    assert abs(codes.shape[3] - expected_t) <= 2, (codes.shape, expected_t)
    assert int(codes.min()) >= 0 and int(codes.max()) < 1024, \
        f"codec vocab out of range: {codes.min()}..{codes.max()}"


def split_of(source):
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def self_test():
    """不依赖服务器/模型：用假数组验证切片与断言。"""
    fake = np.random.randint(0, 1024, size=(1, 1, 32, 75), dtype=np.int64)
    y = np.zeros(75 * HOP, dtype=np.float32)
    sliced = fake[:, :, :16, :]
    audit_shape(sliced, y, 16)
    assert sliced.shape == (1, 1, 16, 75), sliced.shape
    print("self-test OK:", sliced.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(BASE / "data/real_bbox/sound_seed_candidates.jsonl"))
    ap.add_argument("--out-dir", default=str(BASE / "data/omni_talker"))
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--model", default=str(BASE / "models/encodec_24khz"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    out_dir = Path(args.out_dir)
    code_dir = out_dir / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)

    from codec_tokens import load_model, encode_wav, SR, BANDWIDTH  # noqa: E402

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)

    rows = [json.loads(l) for l in Path(args.candidates).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    train, manifest, failures = [], [], []
    for i, r in enumerate(rows, 1):
        wav = r["wav"]
        if not Path(wav).exists():
            failures.append({"id": r.get("id"), "wav": wav, "error": "missing wav"})
            continue
        try:
            codes, y = encode_wav(wav, model, proc, device)
            raw_q = int(codes.shape[2])
            sliced = codes[:, :, : args.layers, :]
            audit_shape(sliced, y, args.layers)
            np.savez_compressed(code_dir / f"{r['id']}.npz", codes=sliced)
            # 文本标签必须取 canonical 103 体系的记法（sound_notation），
            # 不能用 candidate_notation：后者是种子库记法空间（带 '<' 前缀），
            # 与 103 音色词表不是同一空间，直接当 CE 标签会训出错的词表。
            text = r.get("sound_notation") or r.get("human_notation") or "unknown"
            rec = {
                "id": r["id"], "text": text, "codes": str(code_dir / f"{r['id']}.npz"),
                "dur": round(len(y) / SR, 3), "family": r.get("sound_family"),
                "notation": r.get("sound_notation"),
                "sound_id": r.get("sound_id"),
                "label_source": r.get("label_source") or "seed_match",
                "sound_confidence": r.get("sound_confidence", 0.0),
                "source": r.get("source"),
                "split": split_of(r.get("source") or r.get("id")),
            }
            train.append(rec)
            manifest.append({
                **{k: rec[k] for k in ("id", "source", "split", "family", "notation",
                                        "sound_id", "label_source", "sound_confidence", "dur")},
                "wav": wav, "raw_quantizers": raw_q, "layers": args.layers,
                "codes_shape": list(sliced.shape), "T": int(sliced.shape[3]),
                "codec_model": "encodec_24khz", "bandwidth": BANDWIDTH,
            })
        except Exception as exc:
            failures.append({"id": r.get("id"), "wav": wav, "error": f"{type(exc).__name__}: {exc}"})
        if i % 50 == 0:
            print(f"{i}/{len(rows)} ok={len(train)} fail={len(failures)}", flush=True)

    for name, data in [("talker_train.jsonl", train), ("talker_manifest.jsonl", manifest)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "failures.jsonl", "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rate = len(failures) / max(len(rows), 1)
    split_counts = {s: sum(1 for r in train if r["split"] == s) for s in ("train", "val", "test")}
    print(f"wrote ok={len(train)} fail={len(failures)} rate={rate:.1%} splits={split_counts}",
          flush=True)
    if rate > 0.05:
        print("FAIL: error rate > 5%", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
