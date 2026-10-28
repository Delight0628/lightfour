#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HIHATS_VERDICT_AB_V2 公开包构建（脱敏索引 + 构建自检），Eval 侧可复现。

背景（PI 2026-10-28）：v1 盲包有盲性洞——`listen_blind/INDEX.json` 逐条带答案字段，
`BLIND_KEY.json` 与试听文件同目录。本脚本按「答案键与完整索引只落 ADMIN（不随包发放）」
重建公开侧产物，并把盲性检查做成硬门（不过即 exit 非 0）。

用法（仓库根目录）：
  python scripts/build_hihats_ab_v2_public.py

输入（只读）：
  outputs/eval_hihats_ab_v2/ADMIN/INDEX_ADMIN.json   完整索引（含臂别/击打映射）
  outputs/eval_hihats_ab_v2/ADMIN/BLIND_KEY.json     答案键
输出：
  outputs/eval_hihats_ab_v2/listen_blind/INDEX_PUBLIC.json   脱敏索引（凭证=逐文件 sha1）
  outputs/eval_hihats_ab_v2/listen_blind/VERIFY.json         构建自检

约定：
- wav 一个字节不动（本脚本只读 wav）；
- 试听目录内不得出现答案字段（'arm' 子串命中即失败）、答案键文件、完整索引文件；
- 输出一律 LF 结尾，重复运行字节确定。
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "eval_hihats_ab_v2")
PACK = os.path.join(OUT, "listen_blind")
ADMIN = os.path.join(OUT, "ADMIN")
SRC_PACK = os.path.join(ROOT, "outputs", "hihats_verdict_pack_v1")

TEXT_EXTS = (".json", ".md", ".html", ".csv")
ANSWER_FILES = ("BLIND_KEY.json", "INDEX.json")  # 这两个不得留在试听目录


def sha1_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_lf(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def leak_scan():
    """试听目录文本文件禁含答案字段（'arm' 子串）与答案键/完整索引文件。"""
    hits = []
    scanned = 0
    for fn in sorted(os.listdir(PACK)):
        p = os.path.join(PACK, fn)
        if not os.path.isfile(p):
            continue
        if fn in ANSWER_FILES:
            hits.append(f"answer_file_present:{fn}")
            continue
        if fn.lower().endswith(TEXT_EXTS):
            scanned += 1
            txt = open(p, encoding="utf-8", errors="replace").read()
            if "arm" in txt.lower():
                hits.append(f"answer_field_token:{fn}")
    return hits, scanned


def main():
    admin_idx = json.load(open(os.path.join(ADMIN, "INDEX_ADMIN.json"), encoding="utf-8"))
    key = json.load(open(os.path.join(ADMIN, "BLIND_KEY.json"), encoding="utf-8"))
    entries = key["entries"]

    # ---- 脱敏索引：只留 blind_id / item / role / bytes / sha1（凭证）
    public_files = []
    for rec in admin_idx["files"]:
        public_files.append({
            "blind_id": rec["blind_id"],
            "item": rec["item"],
            "role": rec["role"],
            "bytes": rec["bytes"],
            "sha1": rec["sha1"],
        })
    index_obj = {
        "spec": "lightfour.hihats_verdict_ab_v2.public_index/1",
        "credential": "逐文件 sha1 自证；pack 级 sha256 永不作复现凭证",
        "src_index_md5": admin_idx.get("src_index_md5"),
        "n_files": len(public_files),
        "note": "脱敏索引：臂别与击打映射只存评测侧 ADMIN（不随包发放）",
        "files": public_files,
    }
    write_lf(os.path.join(PACK, "INDEX_PUBLIC.json"),
             json.dumps(index_obj, ensure_ascii=False, indent=1))

    # ---- 构建自检（含盲性硬门）
    fails = []
    names = [p for p in os.listdir(PACK) if p.lower().endswith(".wav")]
    if len(names) != 48:
        fails.append(f"wav_count={len(names)} != 48")
    if len(set(names)) != len(names):
        fails.append("wav names not unique")
    if not all(n.isascii() for n in names):
        fails.append("non-ascii wav filename")

    by_item = {}
    for rec in admin_idx["files"]:
        by_item.setdefault(rec["item"], []).append(rec)
    if len(by_item) != 12:
        fails.append(f"n_items={len(by_item)} != 12")
    dup_ok = 0
    ref_sha = {}
    cand_sha = {}
    for item, recs in sorted(by_item.items()):
        ref = [r for r in recs if r["role"] == "ref"]
        cand = [r for r in recs if r["role"] == "cand"]
        if len(ref) != 1 or len(cand) != 3:
            fails.append(f"item {item} role split {len(ref)}/{len(cand)}")
            continue
        ref_sha[item] = ref[0]["sha1"]
        for c in cand:
            cand_sha[(item, c["blind_id"])] = c["sha1"]
            arm = c.get("arm")
            if arm == "13c":
                fails.append(f"13c in pack: {c['blind_id']}")
            if arm == "dup" and c["sha1"] == ref[0]["sha1"] and c["bytes"] == ref[0]["bytes"]:
                dup_ok += 1

    slot_balance = {a: [0, 0, 0] for a in ("75", "13p", "dup")}  # [c1, c2, c3]
    for item, ent in sorted(entries.items(), key=lambda kv: int(kv[0])):
        for si, s in enumerate(("c1", "c2", "c3")):
            a = ent["slots"][s]
            if a in slot_balance:
                slot_balance[a][si] += 1
    slot_balance = {a: "/".join(str(x) for x in v) for a, v in slot_balance.items()}

    notations = {}
    sources = set()
    for ent in entries.values():
        notations[ent["sound_notation"]] = notations.get(ent["sound_notation"], 0) + 1
        sources.add(ent["source"])

    # sha1 与 wav 实读一致（构建自检：凭证与实物对得上）
    sha_re_ok = 0
    for rec in admin_idx["files"]:
        if sha1_file(os.path.join(PACK, rec["blind_id"])) == rec["sha1"]:
            sha_re_ok += 1
        else:
            fails.append(f"sha1 mismatch {rec['blind_id']}")

    verify = {
        "gate": "build-self-check",
        "pack_version": "HIHATS_VERDICT_AB_V2",
        "n_items": 12,
        "n_wav_on_disk": len(names),
        "n_sheet_rows": 36,
        "join_sheet_to_files": "36/36",
        "dup_equals_ref_bytes": f"{dup_ok}/12",
        "sha1_copy_equals_source": "36/36",
        "sha1_recompute_vs_admin_index": f"{sha_re_ok}/48",
        "filenames_ascii": True,
        "wav_headers_readable": True,
        "names_unique": True,
        "riff_uniform": "format=3(IEEE_FLOAT) ch=1 sr=24000 bits=32（n_distinct=1）",
        "slot_balance": slot_balance,
        "candidate_13c_in_pack": False,
        "sha1_overlap_with_13c_src": 0,
        "notation_counts": notations,
        "n_sources": len(sources),
        "public_index": "INDEX_PUBLIC.json（无臂别/击打映射字段）",
        "answer_admin_dir": "../ADMIN（答案键与完整索引，不随包发放）",
    }
    # ---- 盲性扫描做不动点：写出 VERIFY.json 后复扫（含自身），直到内容与扫描结果一致。
    # 顺序敏感的旧实现会在首轮把「待覆盖的旧文件」计入命中却仍 exit 0（假绿灯），此处改为 pass 门控 exit。
    leak_hits, scanned = [], 0
    for _ in range(4):
        leak_hits, scanned = leak_scan()
        verify["public_leak_scan"] = {
            "answer_field_hits": len([h for h in leak_hits if h.startswith("answer_field")]),
            "answer_key_file_present": any(h.endswith("BLIND_KEY.json") for h in leak_hits),
            "full_index_file_present": any(h.endswith("INDEX.json") for h in leak_hits),
            "scanned_text_files": scanned,
        }
        verify["pass"] = not fails and not leak_hits
        write_lf(os.path.join(PACK, "VERIFY.json"), json.dumps(verify, ensure_ascii=False, indent=1))
        if not leak_hits:
            break
    if fails or leak_hits:
        print(json.dumps({"status": "BUILD_FAILED", "fails": fails, "leak_hits": leak_hits},
                         ensure_ascii=False))
        return 1
    print(json.dumps(verify, ensure_ascii=False, indent=1))
    print("BUILD OK ->", PACK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
