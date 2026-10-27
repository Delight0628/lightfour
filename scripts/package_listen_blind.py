"""把 codec_rate_ab 的试听包封装成盲听包（24 REF + 72 候选，其中 24 条为隐藏 dup）。

设计约束（来自 Eval 12-6-6 盲听协议 + PI ㉝）：
  * REF  = 原始波形（`_true` 臂），24 条；
  * 候选 = 24 条 × 2 臂（75 / 13p）= 48 条，**匿名**；13c 出局（见 `AUDIT_ARMS`），
    只在审计时使用，不进候选、不进 dup；
  * 隐藏 dup = 每击重复投喂 1 条候选的字节副本（另给新 id），共 24 条
    （cymbal 12 / hihats 6 / click_60_200ms 6），用于「同一段听两次是否给同一分」的一致性门；
  * 因此 `cand/` = 72 个文件 = 48 唯一 + 24 dup，判决人无法区分；
  * 答案键只写 `key/blind_key.json`，**不得随包发给判决人**。

本脚本只读 `listen/` 下已渲染好的 wav，不重新渲染、不触碰 R4 三件套。
用法：
  python scripts/package_listen_blind.py --listen-dir outputs/codec_listen_n24/listen \
      --out-dir outputs/codec_listen_n24/listen_blind [--seed 20261027]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

ARMS = ("true", "75", "13c", "13p")
# 13c 出局：物理文件存在、可作审计臂，但**不进候选**（PI 裁决 60–62 与 Eval 12/6/6 锁定）。
# 候选 = 24 击 × 2 臂（75 / 13p）= 48；dup 只从候选臂里取副本，避免 dup 指向包内不存在的文件。
AUDIT_ARMS = ("13c",)
CAND_ARMS = ("75", "13p")
assert set(CAND_ARMS).isdisjoint(AUDIT_ARMS)
SR_EXPECT = 24000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_name(stem: str) -> tuple[str, str, str]:
    """`BV1Pw411F7hx_1165_h13_13p` -> (hit_id, hit_index, arm)。"""
    for arm in ARMS:
        if stem.endswith("_" + arm):
            head = stem[: -(len(arm) + 1)]
            hit_id, _, idx = head.rpartition("_h")
            return hit_id, idx, arm
    raise SystemExit(f"无法解析文件名（缺臂后缀）: {stem}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()

    listen = Path(args.listen_dir)
    out = Path(args.out_dir)
    index_p = listen / "index.json"
    comp_p = listen / "composition.json"
    for p in (index_p, comp_p):
        if not p.exists():
            raise SystemExit(f"缺文件: {p}")

    index = json.loads(index_p.read_text(encoding="utf-8"))
    comp = json.loads(comp_p.read_text(encoding="utf-8"))
    pool_of = {(r["hit_id"], str(r["hit_index"])): r["pool"] for r in index}
    if len(pool_of) != len(index):
        raise SystemExit(f"index.json 里 (hit_id,hit_index) 不唯一: {len(index)}")

    files: dict[tuple[str, str, str], Path] = {}
    for w in sorted(listen.glob("*.wav")):
        key = parse_name(w.stem)
        files[key] = w
    # 硬断言：每击必须有 true + 全部候选臂（13c 为审计臂，存在与否都不进组装）
    required_arms = ("true",) + CAND_ARMS
    missing = [f"{h}_{i}_{a}" for (h, i) in pool_of for a in required_arms if (h, i, a) not in files]
    if missing:
        raise SystemExit(f"缺必需 wav（true + 候选臂 {CAND_ARMS}）: {missing[:5]} 共 {len(missing)}")

    # 每个 wav 必须是 24kHz FLOAT（PCM_16 会把复现抖到 1e-2 并重新引入量化差）
    bad = []
    for k, w in files.items():
        info = sf.info(str(w))
        if info.samplerate != SR_EXPECT or "FLOAT" not in info.subtype.upper():
            bad.append((str(w), info.samplerate, info.subtype))
    if bad:
        raise SystemExit(f"格式不合规（要求 {SR_EXPECT}Hz/FLOAT）: {bad[:5]} 共 {len(bad)} 条")

    # ------------------------------------------------ 组包
    refs, cands, dups = [], [], []
    for (hit_id, idx), pool in pool_of.items():
        refs.append({"hit_id": hit_id, "hit_index": idx, "pool": pool, "arm": "true"})
        for arm in CAND_ARMS:
            cands.append({"hit_id": hit_id, "hit_index": idx, "pool": pool, "arm": arm})
        # dup 臂按 (hit_id,index) 稳定散开，避免「三个臂恒定重复 75 臂」被听出门道
        h = int(hashlib.sha256(f"{hit_id}|{idx}".encode()).hexdigest(), 16)
        dups.append({"hit_id": hit_id, "hit_index": idx, "pool": pool,
                     "arm": CAND_ARMS[h % len(CAND_ARMS)], "is_dup": True})

    rng = random.Random(args.seed)
    rng.shuffle(refs)
    mixed = cands + dups
    rng.shuffle(mixed)

    for sub in ("ref", "cand", "key"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*/*.wav")):
        stale.unlink()

    key: dict[str, dict] = {}
    rows = []
    for i, r in enumerate(refs, 1):
        bid = f"L4-R{i:03d}"
        src = files[(r["hit_id"], r["hit_index"], "true")]
        dst = out / "ref" / f"{bid}.wav"
        shutil.copyfile(src, dst)
        key[bid] = {**r, "kind": "ref", "sha256": sha256(dst), "bytes": dst.stat().st_size}
        rows.append({"blind_id": bid, "folder": "ref", "kind": "ref",
                     "pool": r["pool"], "bytes": dst.stat().st_size})

    for i, c in enumerate(mixed, 1):
        bid = f"L4-C{i:03d}"
        src = files[(c["hit_id"], c["hit_index"], c["arm"])]
        dst = out / "cand" / f"{bid}.wav"
        shutil.copyfile(src, dst)
        sha = sha256(dst)
        kind = "dup" if c.get("is_dup") else "cand"
        rec = {**c, "kind": kind, "sha256": sha, "bytes": dst.stat().st_size,
               "src_wav": src.name}
        key[bid] = rec
        rows.append({"blind_id": bid, "folder": "cand", "kind": kind,
                     "pool": c["pool"], "bytes": dst.stat().st_size})

    # ------------------------------------------------ 自检
    by_sha = defaultdict(list)
    for bid, rec in key.items():
        if rec["kind"] != "ref":
            by_sha[rec["sha256"]].append(bid)
    dup_pairs = {bid: rec for bid, rec in key.items() if rec["kind"] == "dup"}
    for bid, rec in dup_pairs.items():
        twins = [b for b in by_sha[rec["sha256"]] if b != bid and key[b]["kind"] == "cand"]
        if len(twins) != 1:
            raise SystemExit(f"dup {bid} 找不到唯一孪生候选（找到 {twins}）")
        rec["dup_of"] = twins[0]
    # 除 dup↔cand 这 24 对外，不允许出现任何其它同字节文件
    for sha, bids in by_sha.items():
        kinds = sorted(key[b]["kind"] for b in bids)
        if kinds not in (["cand"], ["cand", "dup"]):
            raise SystemExit(f"出现意外重复组 {bids} kinds={kinds}")

    # 每击恰好 1 条 dup ⇒ dup 分池必须与 composition 的计数一致
    pool_dup = Counter(rec["pool"] for rec in dup_pairs.values())
    if comp["counts"].get("fill", 0) != 0:
        raise SystemExit(f"composition fill={comp['counts']['fill']}（应 0）")
    for pool, n in comp["counts"].items():
        if pool == "fill" or n == 0:
            continue
        if pool_dup.get(pool) != n:
            raise SystemExit(f"dup 分池 {pool}={pool_dup.get(pool)} != composition {n}")
    if set(pool_dup) != {p for p, n in comp["counts"].items() if p != "fill" and n}:
        raise SystemExit(f"dup 池集合与 composition 不一致: {sorted(pool_dup)}")
    # 协议级硬门：listen_n=24 时配额必须是 12/6/6（Eval 12-6-6 盲听协议 + PI ㉝）
    if comp["listen_n"] == 24:
        proto = {"cymbal": 12, "hihats": 6, "click_60_200ms": 6}
        if {k: comp["counts"].get(k) for k in proto} != proto:
            raise SystemExit(f"listen_n=24 但配额 {comp['counts']} != 协议 {proto}")

    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "folder", "kind", "pool", "bytes"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["blind_id"]))

    (out / "key" / "blind_key.json").write_text(json.dumps({
        "seed": args.seed,
        "note": "答案键：不得随包发给判决人。kind=dup 的 dup_of 指向其孪生候选 blind_id。",
        "listen_composition_md5": hashlib.md5(comp_p.read_bytes()).hexdigest(),
        "counts": {"ref": len(refs), "cand_unique": len(cands), "cand_total": len(mixed),
                   "dup": len(dups)},
        "dup_by_pool": dict(pool_dup),
        "entries": key,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "ref": len(refs), "cand_files": len(mixed),
        "cand_unique": len(cands), "dup_hidden": len(dups),
        "total_wav": len(refs) + len(mixed),
        "dup_by_pool": dict(pool_dup),
        "listen_composition_counts": comp["counts"],
    }
    (out / "pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
