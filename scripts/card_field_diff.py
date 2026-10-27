#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""card_field_diff.py — 卡面逐字段对账件（PI 裁决 170）。

把两张评测卡（markdown）解析成「字段路径 → 值」后逐字段比对，产出机器可读 JSON：
每条差异含 field / old / new / class / subclass。class 只有四种：
  * A9_NEW_SECTION              —— A9 新增段（裁决 165 批准的挂卡）
  * VERSION_STRING              —— 版本号 v2 -> v2.1（归一化后逐字相同）
  * A3_CELLS_AND_NOTES_AUTH_169 —— 裁决 169 授权：A3 family 块状矩阵 other/teeth/zippers 行三格 +
                                    对账注记 + v2 自述句（「全部实测数字未动」）删改
  * UNAUTHORIZED                —— 以上都不是：任何数字/文字变化，按 165② 修订版 = 停手报 PI

规则：
  * 无时间戳、无绝对路径（路径按 ROOT 相对 POSIX 存），同一对文件 + 同一脚本 → 逐字节相同 JSON。
  * 出现任何 UNAUTHORIZED → exit 3（非 0）；全授权 → exit 0。
  * 用法：python scripts/card_field_diff.py --old <旧卡.md> --new <新卡.md> --out <diff.json>
    （--check：不写文件，只比对生成 JSON 与盘上 --out 是否逐字节相同）

以后所有卡面自述（「全部实测数字未动」「只改文案没动数字」类句子）必须挂本脚本的产物。
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL_VERSION = "1.0"
AUTHORIZED_CLASSES = ["A9_NEW_SECTION", "VERSION_STRING", "A3_CELLS_AND_NOTES_AUTH_169"]
A3_AUTH_ROWS = {"other", "teeth", "zippers"}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def parse_card(path: Path) -> dict[str, str]:
    """markdown 卡 → {字段路径: 值}。路径确定性、与运行次序无关。
    章节标题在路径里按版本归一化（v2/v2.1 → vX），标题原文另立 heading 字段，
    避免「只改版本号」让全卡字段错位。"""
    fields: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    heading_stack: list[str] = []
    line_idx: dict[str, int] = {}
    table_idx: dict[str, int] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            level = len(m.group(1))
            raw_title = m.group(2).strip()
            title = norm_ver(raw_title)
            # 标题字段放独立命名空间（与父链解耦）：插段/挪段不会让既有标题键漂移
            fields[f"HEADINGS :: heading[{level}:{title}]"] = raw_title
            del heading_stack[level - 1:]
            heading_stack.append(title)
            i += 1
            continue
        if ln.strip().startswith("|") and i + 1 < len(lines) and is_sep(lines[i + 1]):
            hdr = cells(ln)
            sec = " > ".join(heading_stack) if heading_stack else "HEADER"
            slug = "→".join(hdr)
            tkey = f"{sec} :: table[{slug}]"
            table_idx[tkey] = table_idx.get(tkey, 0) + 1
            tid = f"{tkey}#{table_idx[tkey]}"
            row_seen: dict[str, int] = {}
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|") and not is_sep(lines[j]):
                vals = cells(lines[j])
                rk = vals[0] if vals else ""
                row_seen[rk] = row_seen.get(rk, 0) + 1
                if row_seen[rk] > 1:
                    rk = f"{rk}#{row_seen[rk]}"
                for ci, col in enumerate(hdr):
                    v = vals[ci] if ci < len(vals) else ""
                    fields[f"{tid} :: row[{rk}] :: col[{col}]"] = v
                j += 1
            i = j
            continue
        if ln.strip():
            sec = " > ".join(heading_stack) if heading_stack else "HEADER"
            line_idx[sec] = line_idx.get(sec, 0) + 1
            fields[f"{sec} :: line[{line_idx[sec]}]"] = ln
        i += 1
    return fields


def norm_ver(s: str) -> str:
    return re.sub(r"[vV]2(?:\.1)?", "vX", s)


def classify(field: str, old: str | None, new: str | None) -> tuple[str, str]:
    sec = field.split(" :: ")[0]
    if any(seg.startswith("A9") for seg in sec.split(" > ")):
        return "A9_NEW_SECTION", ("recon_note" if "对账注记" in (old or "") + (new or "") else "a9_body")
    if old is not None and new is not None and norm_ver(old) == norm_ver(new):
        return "VERSION_STRING", "v2_to_v21"
    if ":: row[" in field:
        row = field.split(" :: row[", 1)[1].split("]", 1)[0]
        table = field.split(" :: table[", 1)[1].split("]", 1)[0] if " :: table[" in field else ""
        if row in A3_AUTH_ROWS and ("true\\pred" in table or "true\\\\pred" in table or "true" in table.split("→")[0]):
            return "A3_CELLS_AND_NOTES_AUTH_169", "a3_family_matrix_cell"
        if row in A3_AUTH_ROWS:
            return "A3_CELLS_AND_NOTES_AUTH_169", "a3_row_cell"
    if (new or "").startswith("A9") and old is None:
        return "A9_NEW_SECTION", "a9_heading_added"
    if "全部实测数字未动" in (old or "") or "全部实测数字未动" in (new or ""):
        return "A3_CELLS_AND_NOTES_AUTH_169", "selfclaim_correction_169_1"
    if "对账注记" in (old or "") or "对账注记" in (new or ""):
        return "A3_CELLS_AND_NOTES_AUTH_169", "recon_note"
    return "UNAUTHORIZED", "none"


def build(old_path: Path, new_path: Path) -> dict:
    old_f = parse_card(old_path)
    new_f = parse_card(new_path)
    diffs = []
    for field in sorted(set(old_f) | set(new_f)):
        ov, nv = old_f.get(field), new_f.get(field)
        if ov == nv:
            continue
        cls, sub = classify(field, ov, nv)
        diffs.append({"field": field, "old": ov, "new": nv, "class": cls, "subclass": sub})
    counts: dict[str, int] = {c: 0 for c in AUTHORIZED_CLASSES + ["UNAUTHORIZED"]}
    for d in diffs:
        counts[d["class"]] += 1
    old_txt = old_path.read_text(encoding="utf-8").split("\n")
    new_txt = new_path.read_text(encoding="utf-8").split("\n")
    sm = difflib.SequenceMatcher(a=old_txt, b=new_txt, autojunk=False)
    removed = added = replaced_pairs = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1
        elif tag == "replace":
            removed += i2 - i1
            added += j2 - j1
            replaced_pairs += min(i2 - i1, j2 - j1)
    verdict = "AUTHORIZED_ONLY" if counts["UNAUTHORIZED"] == 0 else "UNAUTHORIZED_PRESENT"
    return {
        "tool": "card_field_diff.py",
        "tool_version": TOOL_VERSION,
        "tool_md5": md5_file(Path(__file__)),
        "old": {"path": rel_posix(old_path), "md5": md5_file(old_path)},
        "new": {"path": rel_posix(new_path), "md5": md5_file(new_path)},
        "authorized_classes": AUTHORIZED_CLASSES,
        "a3_auth_rows": sorted(A3_AUTH_ROWS),
        "counts": counts,
        "line_level": {"removed": removed, "added": added, "changed_pairs": replaced_pairs,
                       "changed_plus_removed_plus_added": removed + added},
        "diffs": diffs,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="不写文件；比对生成 JSON 与盘上 --out 逐字节相同")
    args = ap.parse_args()

    rep = build(Path(args.old), Path(args.new))
    blob = json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=False) + "\n"
    out = Path(args.out)
    if args.check:
        on_disk = out.read_text(encoding="utf-8") if out.is_file() else ""
        ok = on_disk == blob
        print("CHECK", "PASS" if ok else "FAIL", "| verdict:", rep["verdict"],
              "| counts:", rep["counts"], "| diff rows:", len(rep["diffs"]))
        return 0 if ok and rep["verdict"] == "AUTHORIZED_ONLY" else 3
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob, encoding="utf-8")
    print("out:", out)
    print("out md5:", md5_file(out))
    print("verdict:", rep["verdict"], "| counts:", rep["counts"], "| diff rows:", len(rep["diffs"]))
    return 0 if rep["verdict"] == "AUTHORIZED_ONLY" else 3


if __name__ == "__main__":
    sys.exit(main())
