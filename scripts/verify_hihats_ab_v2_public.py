#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HIHATS_VERDICT_AB_V2 独立复核（不复用 build 的任何中间结果），Eval 侧可复现。

PI 2026-10-28 验收口径：
  `listen_blind/` 内 grep "arm" = 0、无 BLIND_KEY.json、VERIFY_INDEPENDENT 仍 48/48 PASS、wav 一字节不动。
本脚本从磁盘重新计算（sha1 / 文件集 / 表连接 / 源包比对 / 盲性扫描），只落结论到
outputs/eval_hihats_ab_v2/listen_blind/VERIFY_INDEPENDENT.json。

用法（仓库根目录）：
  python scripts/verify_hihats_ab_v2_public.py

输入（只读）：listen_blind/、ADMIN/{INDEX_ADMIN.json,BLIND_KEY.json}、outputs/hihats_verdict_pack_v1/
输出：listen_blind/VERIFY_INDEPENDENT.json（failures 非空则 exit 1）
"""
import csv
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
ANSWER_FILES = ("BLIND_KEY.json", "INDEX.json")


def sha1_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_lf(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def main():
    fails = []
    idx = json.load(open(os.path.join(ADMIN, "INDEX_ADMIN.json"), encoding="utf-8"))
    key = json.load(open(os.path.join(ADMIN, "BLIND_KEY.json"), encoding="utf-8"))
    entries = key["entries"]
    idx_by_name = {r["blind_id"]: r for r in idx["files"]}

    on_disk = {p for p in os.listdir(PACK) if p.lower().endswith(".wav")}
    referenced = set(idx_by_name)
    if referenced != on_disk:
        fails.append(f"file_set_bidirectional missing={sorted(referenced-on_disk)} orphan={sorted(on_disk-referenced)}")

    # 1) 逐文件 sha1 重算 vs ADMIN 完整索引
    sha_ok = 0
    for name in sorted(referenced & on_disk):
        if sha1_file(os.path.join(PACK, name)) == idx_by_name[name]["sha1"]:
            sha_ok += 1
        else:
            fails.append(f"sha1 recompute mismatch {name}")

    # 2) dup 与其 REF 字节全等
    dup_ok = 0
    for item, ent in entries.items():
        ref_sha = sha1_file(os.path.join(PACK, ent["ref_file"]))
        for slot, arm in ent["slots"].items():
            if arm != "dup":
                continue
            cand = f"{int(item):03d}_{ent['ref_file'].split('_')[1]}_{slot}.wav"
            if sha1_file(os.path.join(PACK, cand)) == ref_sha:
                dup_ok += 1
            else:
                fails.append(f"dup != ref {cand}")
    if dup_ok != 12:
        fails.append(f"dup_equals_ref_bytes {dup_ok}/12")

    # 3) 候选逐一回源包（true/75/13p），13c 不入包
    cand_ok = 0
    overlap_13c = 0
    src_sha = {}
    for fn in os.listdir(os.path.join(SRC_PACK, "wav")):
        if fn.lower().endswith(".wav"):
            src_sha[fn] = sha1_file(os.path.join(SRC_PACK, "wav", fn))
    for item, ent in entries.items():
        for slot, arm in ent["slots"].items():
            cand = f"{int(item):03d}_{ent['ref_file'].split('_')[1]}_{slot}.wav"
            src_arm = "true" if arm == "dup" else arm
            src_name = os.path.basename(ent["src_files"][src_arm])
            if sha1_file(os.path.join(PACK, cand)) == src_sha.get(src_name):
                cand_ok += 1
            else:
                fails.append(f"candidate != source {cand} <- {src_name}")
            if sha1_file(os.path.join(PACK, cand)) in {
                    v for k, v in src_sha.items() if k.endswith("_13c.wav")}:
                overlap_13c += 1
    if cand_ok != 36:
        fails.append(f"candidate_sha1_vs_source {cand_ok}/36")

    # 4) 表连接
    with open(os.path.join(PACK, "JUDGMENT_SHEET.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 36:
        fails.append(f"sheet_rows={len(rows)} != 36")
    join_ok = 0
    for r in rows:
        ent = entries[str(int(r["item"]))]
        expect = f"{int(r['item']):03d}_{ent['ref_file'].split('_')[1]}_{r['slot']}.wav"
        if r["cand_file"] == expect and os.path.exists(os.path.join(PACK, expect)):
            join_ok += 1
        else:
            fails.append(f"sheet join mismatch {r}")
    filled = [r for r in rows if str(r["judgment"]).strip() != ""]

    html_txt = open(os.path.join(PACK, "index.html"), encoding="utf-8").read()
    missing_html = [n for n in sorted(on_disk) if n not in html_txt]
    if missing_html:
        fails.append(f"index.html missing refs {missing_html}")

    # 5) 盲性扫描（PI 2026-10-28 硬门）
    leak_hits = []
    scanned = 0
    for fn in sorted(os.listdir(PACK)):
        p = os.path.join(PACK, fn)
        if not os.path.isfile(p):
            continue
        if fn in ANSWER_FILES:
            leak_hits.append(f"answer_file_present:{fn}")
            continue
        if fn.lower().endswith(TEXT_EXTS):
            scanned += 1
            if "arm" in open(p, encoding="utf-8", errors="replace").read().lower():
                leak_hits.append(f"answer_field_token:{fn}")
    if leak_hits:
        fails.append(f"leak_scan {leak_hits}")

    notations = {}
    sources = set()
    hits = set()
    for ent in entries.values():
        notations[ent["sound_notation"]] = notations.get(ent["sound_notation"], 0) + 1
        sources.add(ent["source"])
        hits.add(ent["hit_id"])

    res = {
        "gate": "independent-verify",
        "pass": not fails,
        "pack_version": "HIHATS_VERDICT_AB_V2",
        "n_wav": len(on_disk),
        "sha1_recompute_vs_index": f"{sha_ok}/48",
        "index_ref": "../ADMIN/INDEX_ADMIN.json（完整索引，不随包发放）",
        "dup_equals_ref_bytes": f"{dup_ok}/12",
        "file_set_bidirectional": referenced == on_disk,
        "sheet_join": f"{join_ok}/36",
        "sheet_rows": len(rows),
        "sheet_filled_rows": len(filled),
        "sheet_status": "未判（Delight 未填）" if not filled else f"已填 {len(filled)}/36",
        "candidate_sha1_in_src_true75_13p": f"{cand_ok}/36",
        "sha1_overlap_with_13c": overlap_13c,
        "distinct_hits": len(hits),
        "distinct_sources": len(sources),
        "notation_counts": notations,
        "public_leak_scan": {
            "answer_field_hits": len([h for h in leak_hits if h.startswith("answer_field")]),
            "answer_key_file_present": any(h.endswith("BLIND_KEY.json") for h in leak_hits),
            "full_index_file_present": any(h.endswith("INDEX.json") for h in leak_hits),
            "scanned_text_files": scanned,
        },
        "failures": fails,
    }
    write_lf(os.path.join(PACK, "VERIFY_INDEPENDENT.json"),
             json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
