"""
用 AST（AudioSet 527 类）给长音频打 10 秒窗口标签，过滤非 beatbox 区域。

用途：
  Demucs vocals 里混有主持说话、采访、伴奏；直接 onset 切片会把语音当鼓。
  AST 的 "Beatboxing / Drum / Percussion" 标签可以把这些区域筛掉。

输出：
  <out>/<source>/windows.jsonl
  {start, end, beatbox, drum, speech, music, beat_score, neg_score}

运行：
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=/gemini/code/lightfour-train/.cache/hf \
  /root/miniconda3/bin/python scripts/ast_filter_beatbox.py \
    --input data/bbox_raw/demucs_srv --out data/bbox_raw/ast_windows --workers 1
"""
import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

WIN = 10.0
HOP = 5.0
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/bbox_raw/demucs_srv")
    ap.add_argument("--out", default="data/bbox_raw/ast_windows")
    ap.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("**/vocals.wav"))
    if args.limit:
        files = files[: args.limit]
    print(f"files={len(files)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = ASTFeatureExtractor.from_pretrained(args.model)
    model = ASTForAudioClassification.from_pretrained(args.model).to(device).eval()
    id2 = model.config.id2label
    name2id = {v.lower(): k for k, v in id2.items()}

    def ids_for(names):
        return [name2id[n.lower()] for n in names if n.lower() in name2id]

    pos_ids = ids_for(["Beatboxing", "Drum", "Drum kit", "Percussion", "Snare drum",
                       "Bass drum", "Hi-hat", "Cymbal", "Drum machine"])
    neg_ids = ids_for(["Speech", "Male speech, man speaking", "Female speech, woman speaking",
                       "Singing", "Music", "Conversation", "Narration"])
    print("pos", [id2[i] for i in pos_ids])
    print("neg", [id2[i] for i in neg_ids])

    for fi, wav_path in enumerate(files, 1):
        src = wav_path.parent.name
        dst = out / src
        dst.mkdir(parents=True, exist_ok=True)
        y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
        rows = []
        n = int(len(y) / SR)
        for start in np.arange(0, max(0.0, n - WIN + 1e-6), HOP):
            seg = y[int(start * SR):int((start + WIN) * SR)]
            if len(seg) < SR:
                seg = np.pad(seg, (0, SR - len(seg)))
            inp = fe(seg, sampling_rate=SR, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                logits = model(**inp).logits[0]
            p = torch.sigmoid(logits).cpu().numpy()
            beat = float(p[pos_ids].max()) if pos_ids else 0.0
            neg = float(p[neg_ids].max()) if neg_ids else 0.0
            rows.append({
                "source": src, "start": round(float(start), 2),
                "end": round(float(start + WIN), 2),
                "beat_score": round(beat, 4), "neg_score": round(neg, 4),
                "top_beat": id2[pos_ids[int(np.argmax(p[pos_ids]))]] if pos_ids else "",
            })
        with open(dst / "windows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        keep = sum(1 for r in rows if r["beat_score"] >= 0.3 and r["beat_score"] >= r["neg_score"])
        print(f"[{fi}/{len(files)}] {src}: windows={len(rows)} keep≈{keep}", flush=True)


if __name__ == "__main__":
    main()
