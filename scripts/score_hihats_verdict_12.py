#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HIHATS_VERDICT_AB_V2 scorer. Gates hard-coded per PI 2026-10-28:
dup judged-0 >=11/12 AND 13p judged>=1 >=8/12 => session valid;
75 arm judged-0: >=8/12 PASS, <=5/12 -> 32-layer eval, 6-7/12 -> expand to 24.
13c not in judgment; mel not in judgment.

盲性硬门（PI 2026-10-28）：答案键只从 ../ADMIN/BLIND_KEY.json 读取；
试听目录内出现答案键 / 完整索引 / 答案字段（'arm' 子串）即 INVALID（exit 2）。
SCORE_CARD.md 写到 outputs/eval_hihats_ab_v2/，不写进试听目录。
"""
import os, sys, json, csv, argparse

TEXT_EXTS = (".json", ".md", ".html", ".csv")
ANSWER_FILES = ("BLIND_KEY.json", "INDEX.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.path.join("outputs", "eval_hihats_ab_v2", "listen_blind"))
    ap.add_argument("--key", default=None,
                    help="答案键路径；默认 <pack>/../ADMIN/BLIND_KEY.json（不随包发放）")
    ap.add_argument("--write-card", action="store_true")
    a = ap.parse_args()

    # ---- 盲性硬门：试听目录必须干净
    for fn in ANSWER_FILES:
        if os.path.exists(os.path.join(a.pack, fn)):
            print(json.dumps({"status": "INVALID", "reason": f"answer file in pack: {fn}"},
                             ensure_ascii=False))
            return 2
    for fn in sorted(os.listdir(a.pack)):
        p = os.path.join(a.pack, fn)
        if os.path.isfile(p) and fn.lower().endswith(TEXT_EXTS):
            if "arm" in open(p, encoding="utf-8", errors="replace").read().lower():
                print(json.dumps({"status": "INVALID", "reason": f"answer field token in pack: {fn}"},
                                 ensure_ascii=False))
                return 2

    key_path = a.key or os.path.join(os.path.dirname(os.path.abspath(a.pack)), "ADMIN", "BLIND_KEY.json")
    key = json.load(open(key_path, encoding="utf-8"))
    entries = key["entries"]
    rows = list(csv.DictReader(open(os.path.join(a.pack, "JUDGMENT_SHEET.csv"), encoding="utf-8")))
    if len(rows) != 36:
        print(json.dumps({"status": "INVALID", "reason": f"sheet_rows={len(rows)} != 36"}))
        return 1
    arm_of = {}
    for r in rows:
        ent = entries[str(int(r["item"]))]
        tok = os.path.basename(ent["ref_file"]).split("_")[1]
        expect = f"{int(r['item']):03d}_{tok}_{r['slot']}.wav"
        if r["cand_file"] != expect:
            print(json.dumps({"status": "INVALID", "reason": f"join mismatch {r['cand_file']} != {expect}"}))
            return 1
        arm_of[(int(r["item"]), r["slot"])] = ent["slots"][r["slot"]]
    filled = [r for r in rows if str(r["judgment"]).strip() != ""]
    if len(filled) < 36:
        out = {"status": "未判（Delight 未填 JUDGMENT_SHEET）", "filled_rows": len(filled), "n_rows": 36}
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    def jud(r):
        v = str(r["judgment"]).strip()
        if v not in ("0", "1", "2"):
            print(json.dumps({"status": "INVALID", "reason": f"bad judgment {v!r}"}))
            sys.exit(1)
        return int(v)

    dup0 = sum(1 for r in rows if arm_of[(int(r["item"]), r["slot"])] == "dup" and jud(r) == 0)
    p13det = sum(1 for r in rows if arm_of[(int(r["item"]), r["slot"])] == "13p" and jud(r) >= 1)
    n75_0 = sum(1 for r in rows if arm_of[(int(r["item"]), r["slot"])] == "75" and jud(r) == 0)
    n_midband = sum(1 for r in rows if jud(r) == 1)
    gates = {"dup_gate": f"{dup0}/12 (>=11 required)", "p13_gate": f"{p13det}/12 (>=8 required)"}
    valid = dup0 >= 11 and p13det >= 8
    if not valid:
        verdict = "仪器失效（有效性门未过，本次判决作废，不补样不复判）"
    elif n75_0 >= 8:
        verdict = "75 臂通过（判 0 >=8/12）；注意：不解除 hihats mel 0.8429 越线登记"
    elif n75_0 <= 5:
        verdict = "75 臂不通过（判 0 <=5/12）-> 触发 32 层评估"
    else:
        verdict = "灰区（判 0 6-7/12）-> 扩样到 24（同池同臂，只准一次）"
    out = {"status": "SCORED", "session_valid": valid, **gates,
           "arm75_judged0": f"{n75_0}/12", "verdict": verdict,
           "n_midband": {"value": n_midband, "status": "未判定"}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.write_card:
        card = ("# SCORE_CARD — HIHATS_VERDICT_AB_V2\n\n"
                "## 卡头（两行并列铁律）\n"
                "- 登记行（不因人耳结果改变）：hihats mel **0.8429** > 0.8 Gate 越线（provisional）\n"
                "- 人耳结果行：%s\n\n"
                "- 有效性门：%s；%s\n- 75 臂判 0：%s\n- n_midband：%s（未判定，不进结论段）\n"
                "- 答案键位置：评测侧 ADMIN/BLIND_KEY.json（不随包发放）；判分脚本 "
                "scripts/score_hihats_verdict_12.py（盲性硬门 exit 2）\n") % (
                verdict, gates["dup_gate"], gates["p13_gate"], out["arm75_judged0"], n_midband)
        out_dir = os.path.dirname(os.path.abspath(a.pack))
        open(os.path.join(out_dir, "SCORE_CARD.md"), "w", encoding="utf-8", newline="\n").write(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
