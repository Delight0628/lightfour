"""
为 Qwen3-Omni Talker 准备训练数据：
  103 音色文本序列 + EnCodec 16 层 codec token。

用法：
  python scripts/prepare_talker_data.py \
    --candidates data/real_bbox/sound_seed_candidates.jsonl \
    --out-dir data/omni_talker --layers 16

输出：
  data/omni_talker/codes/<id>.npz   codes shape (1, 1, 16, T)
  data/omni_talker/talker_train.jsonl
    {"id", "text", "codes", "dur", "family", "notation"}

说明：
  - text 用 taxonomy 记法（如 "PF" / "P T PF T"），Talker 学「读鼓谱」。
  - EnCodec 前 16 层 ≈ 12kbps，与 Qwen3-Omni 的 16 码本对齐。
  - 只做研究用途。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec_tokens import load_model, encode_wav, SR  # noqa: E402

BASE = Path("/gemini/code/lightfour-train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(BASE / "data/real_bbox/sound_seed_candidates.jsonl"))
    ap.add_argument("--out-dir", default=str(BASE / "data/omni_talker"))
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--model", default=str(BASE / "models/encodec_24khz"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    code_dir = out_dir / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)

    rows = [json.loads(l) for l in Path(args.candidates).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    out = []
    for i, r in enumerate(rows, 1):
        wav = r["wav"]
        if not Path(wav).exists():
            continue
        try:
            codes, y = encode_wav(wav, model, proc, device)
            codes = codes[:, :, : args.layers]
            np.savez_compressed(code_dir / f"{r['id']}.npz", codes=codes)
            text = r.get("candidate_notation") or r.get("family") or "unknown"
            out.append({
                "id": r["id"], "text": text, "codes": str(code_dir / f"{r['id']}.npz"),
                "dur": round(len(y) / SR, 3), "family": r.get("family"),
                "notation": r.get("candidate_notation"),
            })
        except Exception as exc:
            print(f"[FAIL] {r['id']}: {exc}", flush=True)
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    with open(out_dir / "talker_train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} -> {out_dir / 'talker_train.jsonl'}", flush=True)
    if out:
        print("sample:", json.dumps(out[0], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
