"""codec_rate_ab 六条验收门槛的机器化校验（R2 重跑验收包）。

只读 CSV/JSON/文件，不解码音频、不碰模型 —— 谁都能在本地跑一遍复核 @Codec 的结论。
六条门槛与 @PI 2026-10-27 裁决一一对应：

  G1 指纹     summary.json.code_sha256 == sha256(脚本文件)；input_md5 三项齐全
  G2 回归等价 新旧 per_hit 按 (sequence_id, hit_index) join 必须 5472/5472，
              旧版本已有的每一列逐值相等（float 按 round 5 比，字符串按原样比）
  G3 帧级     mel_frames（true|75|13c|13p 四臂）逐行 max-min <= 1
  G4 波形级   wav_trunc_ratio 最大值 <= 5%（截断前四臂采样数写进 wav_len 列）
  G5 试听包   composition.json 计数 cymbal=4 / hihats=4 / click_60_200ms=4，listen/*.wav = 48
  G6 归因     summary.md 里有 R1 click gain_matched 退化中位（一个数）+ 四条聚类 CI

任一条 FAIL → 本轮产物作废，回退 archive_pre_r2/。

用法：
  python scripts/codec_rate_ab_verify.py \
      --new outputs/codec_rate_ab --archive outputs/codec_rate_ab/archive_pre_r2
  # 远端跑完直接把两个 dir 都用 lf_remote.py pull 下来复核
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

KEY = ("sequence_id", "hit_index")
EXPECT_ROWS = 5472
EXPECT_LISTEN = {"cymbal": 4, "hihats": 4, "click_60_200ms": 4}
EXPECT_WAV = 48
TRUNC_LIMIT = 0.05


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num_eq(a: str, b: str, ndigits: int = 5) -> tuple[bool, float]:
    """逐值比较：能转 float 的按 round(ndigits) 比，转不动的按字符串比。"""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return (str(a) == str(b)), 0.0
    if math.isnan(fa) and math.isnan(fb):
        return True, 0.0
    d = abs(round(fa, ndigits) - round(fb, ndigits))
    return d < 1e-9, d


def gate1(new_dir: Path, script: Path) -> list[str]:
    s = json.loads((new_dir / "summary.json").read_text(encoding="utf-8"))["summary"]
    msgs, ok = [], True
    want = _sha256(script)
    got = s.get("code_sha256")
    msgs.append(f"  G1a code_sha256  summary={got}")
    msgs.append(f"      {'PASS' if got == want else 'FAIL'}  == sha256({script.name})={want}")
    ok &= got == want
    im = s.get("input_md5") or {}
    need = ("sequence_samples", "sequence_hits", "codes_npz")
    miss = [k for k in need if not im.get(k)]
    msgs.append(f"  G1b input_md5 三项 {'PASS' if not miss else 'FAIL 缺 ' + str(miss)}")
    for k in need:
        msgs.append(f"      {k}: {im.get(k)}")
    ok &= not miss
    msgs.append(f"  G1 → {'PASS' if ok else 'FAIL'}")
    return msgs


def gate2(new_dir: Path, arch_dir: Path) -> list[str]:
    new = _rows(new_dir / "per_hit.csv")
    old = _rows(arch_dir / "per_hit.csv")
    msgs = []
    new_map = {(r["sequence_id"], r["hit_index"]): r for r in new}
    old_map = {(r["sequence_id"], r["hit_index"]): r for r in old}
    common = set(new_map) & set(old_map)
    msgs.append(f"  G2a join  {len(common)}/{len(old)}（旧 {len(old)} 新 {len(new)}，"
                f"缺 {len(set(old_map) - common)} 多 {len(set(new_map) - common)}）")
    ok = len(common) == len(old) == len(new) == EXPECT_ROWS

    old_cols = [c for c in old[0].keys()]
    new_cols = set(new[0].keys())
    missing_cols = [c for c in old_cols if c not in new_cols]
    if missing_cols:
        msgs.append(f"  G2b FAIL 新表缺旧列: {missing_cols}")
        ok = False

    bad: dict[str, list] = {}
    for k in common:
        for c in old_cols:
            if c in KEY or c not in new_cols:
                continue
            eq, d = _num_eq(new_map[k][c], old_map[k][c])
            if not eq:
                bad.setdefault(c, []).append((k, old_map[k][c], new_map[k][c], d))
    if bad:
        for c, v in bad.items():
            msgs.append(f"  G2b FAIL 列 {c}: {len(v)} 条不等，例 {v[0]}")
        ok = False
    else:
        msgs.append(f"  G2b 旧列逐值相等 PASS（{len(old_cols)} 列 × {len(common)} 行，round 5）")
    new_only = [c for c in new[0].keys() if c not in old_cols]
    msgs.append(f"  G2c 新增列: {new_only}")
    msgs.append(f"  G2 → {'PASS' if ok else 'FAIL'}")
    return msgs


def gate3(new_dir: Path) -> list[str]:
    rows = _rows(new_dir / "per_hit.csv")
    if "mel_frames" not in rows[0]:
        return ["  G3 FAIL 无 mel_frames 列", "  G3 → FAIL"]
    diffs = [max(map(int, r["mel_frames"].split("|"))) - min(map(int, r["mel_frames"].split("|")))
             for r in rows]
    mx, n_gt1 = max(diffs), sum(1 for d in diffs if d > 1)
    ok = mx <= 1
    return [f"  G3 mel_frames max-min: max={mx}, >1 的行 {n_gt1}/{len(diffs)}",
            f"  G3 → {'PASS' if ok else 'FAIL'}（门槛 ≤1）"]


def gate4(new_dir: Path) -> list[str]:
    rows = _rows(new_dir / "per_hit.csv")
    if "wav_trunc_ratio" not in rows[0]:
        return ["  G4 FAIL 无 wav_trunc_ratio 列（wav_len 审计缺失）", "  G4 → FAIL"]
    r = [float(x["wav_trunc_ratio"]) for x in rows]
    top = max(range(len(r)), key=lambda i: r[i])
    ok = max(r) <= TRUNC_LIMIT
    return [f"  G4 wav 截断比: max={max(r):.6f}（{rows[top]['sequence_id']} h"
            f"{rows[top]['hit_index']} wav_len={rows[top]['wav_len']}），"
            f"p50={sorted(r)[len(r)//2]:.6f}, >5% 的行 {sum(1 for x in r if x > TRUNC_LIMIT)}",
            f"  G4 → {'PASS' if ok else 'FAIL'}（门槛 ≤5%）"]


def gate5(new_dir: Path) -> list[str]:
    lp = new_dir / "listen"
    msgs, ok = [], True
    comp_p = lp / "composition.json"
    if not comp_p.exists():
        return ["  G5 FAIL 无 listen/composition.json", "  G5 → FAIL"]
    comp = json.loads(comp_p.read_text(encoding="utf-8"))
    counts = comp.get("counts") or {k: comp.get(k) for k in EXPECT_LISTEN}
    for k, v in EXPECT_LISTEN.items():
        got = counts.get(k)
        good = got == v
        ok &= good
        msgs.append(f"  G5a {k}: {got} {'PASS' if good else 'FAIL'}（期望 {v}）")
    wavs = sorted(lp.glob("*.wav"))
    ok &= len(wavs) == EXPECT_WAV
    msgs.append(f"  G5b listen/*.wav = {len(wavs)} "
                f"{'PASS' if len(wavs) == EXPECT_WAV else 'FAIL'}（期望 {EXPECT_WAV}）")
    msgs.append(f"  G5 → {'PASS' if ok else 'FAIL'}")
    return msgs


def gate6(new_dir: Path) -> list[str]:
    md = (new_dir / "summary.md").read_text(encoding="utf-8")
    msgs, ok = [], True
    has_r1 = "R1" in md and "click_60_200ms gain_matched 退化中位" in md
    has_shift = "1920 采样" in md
    ci_rows = [ln for ln in md.splitlines()
               if ln.startswith("| ") and ln.count("|") >= 6 and "%" in ln and "group" not in ln]
    msgs.append(f"  G6a R1 click 单一数 {'PASS' if has_r1 else 'FAIL'}")
    msgs.append(f"  G6b R1 位移量写成 80.0ms/1920 采样 {'PASS' if has_shift else 'FAIL'}")
    msgs.append(f"  G6c 聚类 CI 行数 = {len(ci_rows)} {'PASS' if len(ci_rows) >= 4 else 'FAIL'}（期望 ≥4）")
    for ln in ci_rows[:6]:
        msgs.append(f"      {ln}")
    ok &= has_r1 and has_shift and len(ci_rows) >= 4
    msgs.append(f"  G6 → {'PASS' if ok else 'FAIL'}")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="outputs/codec_rate_ab")
    ap.add_argument("--archive", default="outputs/codec_rate_ab/archive_pre_r2")
    ap.add_argument("--script", default="scripts/codec_rate_ab.py")
    args = ap.parse_args()

    new_dir, arch_dir, script = Path(args.new), Path(args.archive), Path(args.script)
    for p in (new_dir / "per_hit.csv", new_dir / "summary.json", new_dir / "summary.md"):
        if not p.exists():
            print(f"缺 {p}")
            return 2

    blocks = [("G1 指纹", gate1(new_dir, script)),
              ("G2 回归等价", gate2(new_dir, arch_dir)),
              ("G3 帧级", gate3(new_dir)),
              ("G4 波形级", gate4(new_dir)),
              ("G5 试听包", gate5(new_dir)),
              ("G6 归因", gate6(new_dir))]
    for title, msgs in blocks:
        print(f"\n== {title} ==")
        for m in msgs:
            print(m)

    fails = [t for t, ms in blocks if any("FAIL" in m for m in ms)]
    print(f"\n结论: {'全部 PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
