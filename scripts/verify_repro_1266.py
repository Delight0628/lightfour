"""1266 试听包「新会话重渲」比对器（mel 举证第二道门：≤1e-6）。

用途：
  在**新进程/新会话**里用同一渲染器（codec_rate_ab.py 407e5e69）重跑一次 1266 包，
  把新产物与冻结产物逐项比对，区分三件事：
    1. per_hit 逐行数值差（mel75 / mel13c / mel13p）—— 判「指标是否确定性」
    2. wav 逐文件 sha256 —— 判「波形是否字节复现」（比 1e-6 更强）
    3. 若字节不同，再用冻结 mel_metric 240f6f18 算 mel 差 —— 判「差是否落在容差内」

口径约束：
  - 只做比对，不重写任何冻结件；输出只写 --out（默认 outputs/repro_render_1266/）
  - join 键 = hit_id（同一 per_hit 表内唯一）
  - mel 差为 |a-b| 绝对值，单位与 mel_metric.compare 相同（log-mel 距离）

用法：
  python scripts/verify_repro_1266.py \
      --per-hit-a outputs/codec_rate_ab/per_hit.csv \
      --per-hit-b /tmp/repro1266/per_hit.csv \
      --wav-a outputs/codec_listen_n24_1266/listen \
      --wav-b /tmp/repro1266/listen \
      --out outputs/repro_render_1266
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mel_metric import (  # noqa: E402
    MEL_SPEC,
    compare as mel_compare,
)

BASE = Path(__file__).resolve().parent.parent
MEL_COLS = ("mel75_vs_true", "mel13c_vs_true", "mel13p_vs_true")
TOL = 1e-6


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_per_hit(p: Path) -> dict[str, dict]:
    rows = {}
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["hit_id"]] = r
    return rows


def load_wav(p: Path):
    import soundfile as sf

    y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    assert sr == MEL_SPEC["sr"], (p, sr)
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-hit-a", required=True)
    ap.add_argument("--per-hit-b", required=True)
    ap.add_argument("--wav-a", required=True)
    ap.add_argument("--wav-b", required=True)
    ap.add_argument("--out", default=str(BASE / "outputs/repro_render_1266"))
    args = ap.parse_args()

    a = load_per_hit(Path(args.per_hit_a))
    b = load_per_hit(Path(args.per_hit_b))
    common = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    col_stats = {}
    exact_rows = 0
    for col in MEL_COLS:
        diffs = [abs(float(b[k][col]) - float(a[k][col])) for k in common]
        col_stats[col] = {
            "max_abs_delta": max(diffs) if diffs else None,
            "median_abs_delta": float(np.median(diffs)) if diffs else None,
            "n_over_tol": int(sum(d > TOL for d in diffs)),
        }
    exact_rows = sum(
        1 for k in common if all(float(a[k][c]) == float(b[k][c]) for c in MEL_COLS)
    )

    wa, wb = Path(args.wav_a), Path(args.wav_b)
    files_a = {p.name: p for p in sorted(wa.glob("*.wav"))}
    files_b = {p.name: p for p in sorted(wb.glob("*.wav"))}
    names = sorted(set(files_a) & set(files_b))
    bytes_equal, byte_diff = [], []
    mel_pairs, mel_frames = [], []
    pcm_equal = 0
    header_diff_stats = []
    for n in names:
        sa, sb = sha256_file(files_a[n]), sha256_file(files_b[n])
        if sa == sb:
            bytes_equal.append(n)
        else:
            ba, bb = files_a[n].read_bytes(), files_b[n].read_bytes()
            dpos = [i for i, (x, y) in enumerate(zip(ba, bb)) if x != y]
            header_diff_stats.append({
                "file": n,
                "n_diff_bytes": len(dpos),
                "first_diff_offset": dpos[0] if dpos else None,
                "last_diff_offset": dpos[-1] if dpos else None,
                "len_a": len(ba),
                "len_b": len(bb),
            })
            byte_diff.append({"file": n, "sha256_a": sa, "sha256_b": sb})
            ya, yb = load_wav(files_a[n]), load_wav(files_b[n])
            if ya.shape == yb.shape and np.array_equal(ya, yb):
                pcm_equal += 1
            m = min(len(ya), len(yb))
            r = mel_compare(ya[:m], yb[:m])
            mel_pairs.append(float(r["value"]))
            mel_frames.append({"file": n, "n_frames": r["n_frames"], "frame_diff": r["frame_diff"]})

    result = {
        "per_hit_a": str(args.per_hit_a),
        "per_hit_a_md5": hashlib.md5(Path(args.per_hit_a).read_bytes()).hexdigest(),
        "per_hit_b_md5": hashlib.md5(Path(args.per_hit_b).read_bytes()).hexdigest(),
        "n_rows_a": len(a),
        "n_rows_b": len(b),
        "n_common": len(common),
        "only_a": only_a[:20],
        "only_b": only_b[:20],
        "per_col": col_stats,
        "rows_bitwise_equal": exact_rows,
        "wav": {
            "dir_a": str(wa),
            "dir_b": str(wb),
            "n_a": len(files_a),
            "n_b": len(files_b),
            "n_common": len(names),
            "bytes_equal": len(bytes_equal),
            "bytes_diff": len(byte_diff),
            "pcm_bit_equal_on_byte_diff": pcm_equal,
            "header_diff_stats": header_diff_stats[:5],
            "diff_examples": byte_diff[:5],
            "mel_delta_on_diff": {
                "max": max(mel_pairs) if mel_pairs else None,
                "median": float(np.median(mel_pairs)) if mel_pairs else None,
                "n": len(mel_pairs),
                "frames": mel_frames[:5],
            },
        },
        "tol": TOL,
        "verdict": None,
    }
    ok_rows = all(v["n_over_tol"] == 0 for v in col_stats.values())
    ok_pcm = result["wav"]["pcm_bit_equal_on_byte_diff"] == result["wav"]["bytes_diff"]
    ok_bytes = result["wav"]["bytes_diff"] == 0
    ok_cover = not only_a and not only_b
    if not (ok_rows and ok_pcm and ok_cover):
        result["verdict"] = "FAIL"
    elif ok_bytes:
        result["verdict"] = "PASS_BYTES"
    else:
        # 文件 sha256 不同但 PCM bit-equal：属 PEAK chunk 写入时刻差异，音频本身零差
        result["verdict"] = "PASS_PCM"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPRO.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
