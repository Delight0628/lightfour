#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blind_pack_1266.py — 正式 96 wav 包（`outputs/codec_listen_n24_1266/listen/`）盲包构建 / 核验 / 评分。

来源与口径（不自创，逐条对应裁决号）：
- 判决包锁定 12/6/6（PI 60；8/8/8 降级审计物，永不判决）。
- 盲包格式 = 正式包 24 击 x (REF + 75 / 13p / dup)（PI ㉝ + 盲包锁定格式；13c 不进判决、appendix only）。
- dup = 1266 包内 REF(true) 的逐字节副本，只换文件名；建包逐条 assert sha1(dup)==sha1(ref) 24/24。
  禁止从 listen_96_v0 / 8/8/8 探针包搬 dup；禁止重新渲染 REF 充当 dup。
- dup 负对照满额零容错（PI 61①）：n=6→6/6、n=12→12/12、正式包 24 击→24/24；差 1 条整卡作废、
  不补样不扩样不复判，卡面只写「仪器失效」；dup 只作仪器自检，不得当任何臂的统计证据。
- hihats 门（PI ㉝ + 62②）：n=6：>=5/6 → 只准「hihats 未见退化（低置信）」；3-4/6 → MIDBAND 扩样 12 一次；
  <=2/6 → 扩样 12。扩样 12（只准一次）：>=10/12 / 7-9/12 / <=6/12 → 限幅结论 / 证据不足以判定+v0 短板 /
  hihats 退化 n=12。32 层出口已删除、永不触发。
- 分族不设通过线（PI 62③ + 74/75）：cymbal / click 只填「报数区（不参与 verdict）」判 0/1/2 三元组 + 样本量 +
  三档明细；cymbal >=10/12 判 0 只进观察记录。总 verdict = hihats 门 + dup 门。
- 判决人 = Delight；0/1/2 三档；judgment=1 一律单列 n_midband（未判定，不进汇总、不进 §0、不进结论段）。
- 卡头两行铁律：人耳结果行与「hihats mel 0.8429 > 0.8 越线登记」行必须并列；人耳通过不解除登记。
- 凭证 = 逐文件 sha1（INDEX.json）；pack 级 sha256 永不作复现凭证。
- 判决包「听感样本，跨 split 混采，非泛化证据」（PI 裁决 78）。

评分门逻辑沿用 scripts/eval_hihats_ab.py 的 --score（61/62 六分支冒烟过），按 pool 分族；
本脚本只换数据源（正式 96 包）与分族字段（pool），不改任何阈值数字。

用法：
  python scripts/blind_pack_1266.py --build    # 建盲包（先自动跑源包核验）
  python scripts/blind_pack_1266.py --verify   # 只核验源包与盲包
  python scripts/blind_pack_1266.py --score    # Delight 填完 JUDGMENT_SHEET.csv 后出 SCORE_CARD
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "outputs" / "codec_listen_n24_1266" / "listen"
BLIND = ROOT / "outputs" / "codec_listen_n24_1266" / "listen_blind_v2"

CARD_ID = "CODEC_LISTEN_24_V2"
SEED = 20261027
JUDGMENT_SCALE = {"0": "与 REF 听不出差别", "1": "可辨·轻微", "2": "可辨·明显"}
ARMS = ("75", "13p", "dup")  # dup = REF(true) 字节副本；13c 不进判决
GATE_NOTE_TAIL = ("必须与「hihats mel 0.8429 > 0.8 越线登记」两行并列在卡头，"
                  "人耳通过不解除越线登记；禁写「已通过/已稳」")


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def riff_fmt(p: Path) -> tuple:
    """(fmt_tag, channels, sample_rate, bits_per_sample)，手工走 chunk，不依赖 soundfile。"""
    d = p.read_bytes()
    assert d[:4] == b"RIFF" and d[8:12] == b"WAVE", p
    i = 12
    while i + 8 <= len(d):
        cid = d[i:i + 4]
        sz = struct.unpack("<I", d[i + 4:i + 8])[0]
        if cid == b"fmt " and i + 24 <= len(d):
            fmt, ch, sr, _, _block, bits = struct.unpack("<HHIIHH", d[i + 8:i + 24])
            return fmt, ch, sr, bits
        i += 8 + sz + (sz & 1)
    raise AssertionError(f"no fmt chunk: {p}")


def load_src():
    idx = json.loads((SRC / "index.json").read_text(encoding="utf-8"))
    comp = json.loads((SRC / "composition.json").read_text(encoding="utf-8"))
    assert len(idx) == 24, f"预期 24 击，实际 {len(idx)}"
    assert comp["counts"] == {"cymbal": 12, "hihats": 6, "click_60_200ms": 6, "fill": 0}, comp["counts"]
    pool_of = {p["hit_id"]: p["pool"] for p in comp["picks"]}
    assert len(comp["picks"]) == 24
    return idx, comp, pool_of


def verify_src() -> dict:
    idx, comp, pool_of = load_src()
    n, riff_set, bad = 0, set(), []
    for e in idx:
        base = f"{e['hit_id']}_h{e['hit_index']}"
        for arm in ("true", "75", "13c", "13p"):
            p = SRC / f"{base}_{arm}.wav"
            if not p.exists():
                bad.append(str(p))
                continue
            n += 1
            riff_set.add(riff_fmt(p))
    assert n == 96 and not bad, (n, bad[:3])
    assert riff_set == {(3, 1, 24000, 32)}, f"格式不统一：{riff_set}"
    dup_on_src = [p.name for p in SRC.glob("*dup*.wav")]
    return {
        "n_items": 24, "n_src_wav": n,
        "riff_fmt_ch_sr_bits": "3/1/24000/32（IEEE_FLOAT）",
        "counts": comp["counts"],
        "src_index_md5": md5_file(SRC / "index.json"),
        "src_composition_md5": md5_file(SRC / "composition.json"),
        "per_hit_csv_md5": md5_file(ROOT / "outputs" / "codec_rate_ab" / "per_hit.csv"),
        "dup_files_in_src": dup_on_src,
        "split_by_pool": {
            pool_of[e["hit_id"]]: {} for e in idx
        },
    }


def build() -> int:
    idx, comp, pool_of = load_src()
    src_check = verify_src()

    if BLIND.exists():
        shutil.rmtree(BLIND)
    BLIND.mkdir(parents=True)

    key, sheet, index_rows = {}, [], []
    src_to_blind = {}
    for i, r in enumerate(idx):
        base = f"{r['hit_id']}_h{r['hit_index']}"
        srcs = {a: SRC / f"{base}_{a}.wav" for a in ("true", "75", "13p")}
        for p in srcs.values():
            assert p.exists(), p
        h8 = hashlib.sha1(r["hit_id"].encode("utf-8")).hexdigest()[:8]
        ref_name = f"{i:03d}_{h8}_ref.wav"
        shutil.copy2(srcs["true"], BLIND / ref_name)

        candidates = {"75": srcs["75"], "13p": srcs["13p"], "dup": srcs["true"]}
        slots = ["c1", "c2", "c3"]
        order = list(candidates.keys())
        random.Random(SEED + i).shuffle(order)
        slot_map = dict(zip(slots, order))
        key[str(i)] = {
            "hit_id": r["hit_id"], "hit_index": r["hit_index"], "pool": pool_of[r["hit_id"]],
            "sound_family": r.get("sound_family"), "sound_notation": r.get("sound_notation"),
            "split": r.get("split"), "dur_ms": r.get("dur_ms"), "slots": slot_map,
        }
        index_rows.append({"blind_id": ref_name, "item": i, "role": "ref", "arm": "true",
                           "hit_id": r["hit_id"], "sha1": sha1_file(BLIND / ref_name),
                           "bytes": (BLIND / ref_name).stat().st_size})
        src_to_blind[ref_name] = (srcs["true"], i, "true")

        for slot in slots:
            arm = slot_map[slot]
            name = f"{i:03d}_{h8}_{slot}.wav"
            shutil.copy2(candidates[arm], BLIND / name)
            sheet.append({"item": i, "slot": slot, "arm_hidden": arm,
                          "ref_file": ref_name, "cand_file": name,
                          "judgment": "", "note": ""})
            index_rows.append({"blind_id": name, "item": i, "role": "cand", "arm": arm,
                               "hit_id": r["hit_id"], "sha1": sha1_file(BLIND / name),
                               "bytes": (BLIND / name).stat().st_size})
            src_to_blind[name] = (candidates[arm], i, arm)

    # ---- 自检门（build 内 fail-fast）
    names = [p.name for p in BLIND.glob("*.wav")]
    assert len(names) == 24 * 4, len(names)                    # 24 ref + 24*3 cand
    assert all(n.isascii() for n in names), "文件名必须纯 ASCII"
    assert len(set(names)) == len(names), "文件名重复"
    for row in sheet:
        src, _, arm = src_to_blind[row["cand_file"]]
        assert sha1_file(BLIND / row["cand_file"]) == sha1_file(src), row
    assert len(sheet) == 72, len(sheet)
    # dup 满额字节副本断言：24/24
    dup_rows = [r for r in sheet if r["arm_hidden"] == "dup"]
    assert len(dup_rows) == 24
    for row in dup_rows:
        assert sha1_file(BLIND / row["cand_file"]) == sha1_file(BLIND / row["ref_file"]), row

    # ---- 落盘
    (BLIND / "BLIND_KEY.json").write_text(
        json.dumps({"seed": SEED,
                    "note": "答案键：不得随包发给判决人。kind=dup 的候选是其 REF 的字节副本。",
                    "src_index_md5": src_check["src_index_md5"],
                    "src_composition_md5": src_check["src_composition_md5"],
                    "counts": {"items": 24, "cand_per_item": 3, "dup": 24,
                               "arms_hidden": list(ARMS)},
                    "entries": key}, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(BLIND / "JUDGMENT_SHEET.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "slot", "ref_file", "cand_file", "judgment", "note"])
        w.writeheader()
        for row in sheet:
            w.writerow({k: row[k] for k in w.fieldnames})

    (BLIND / "INDEX.json").write_text(
        json.dumps({"credential": "逐文件 sha1 自证；pack 级 sha256 永不作复现凭证",
                    "src_index_md5": src_check["src_index_md5"],
                    "n_files": len(index_rows), "files": index_rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    html_rows = []
    for i in range(24):
        e = idx[i]
        h8 = hashlib.sha1(e["hit_id"].encode("utf-8")).hexdigest()[:8]
        ref_name = f"{i:03d}_{h8}_ref.wav"
        cand_html = "".join(
            f'<div class="c"><b>{s}</b><br><audio controls src="{html.escape(f"{i:03d}_{h8}_{s}.wav")}"></audio></div>'
            for s in ("c1", "c2", "c3"))
        html_rows.append(
            f'<div class="item"><h3>item {i:02d}  {html.escape(str(e.get("sound_notation")))}</h3>'
            f'<div class="c"><b>REF（原始）</b><br><audio controls src="{html.escape(ref_name)}"></audio></div>'
            f'{cand_html}</div>')
    (BLIND / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>" + CARD_ID + " blind</title>"
        "<style>body{font-family:sans-serif}.item{border:1px solid #ccc;margin:12px;padding:8px}"
        ".c{display:inline-block;margin-right:16px;text-align:center}audio{width:220px}</style>"
        "</head><body><h2>正式 96 包 · 24 击盲听：REF vs 3 匿名候选（75 / 13p / dup 隐藏对照）</h2>"
        "<p>每条候选判 0=与 REF 听不出差别 / 1=可辨·轻微 / 2=可辨·明显，填 JUDGMENT_SHEET.csv 的 judgment 列。"
        "听完再看 BLIND_KEY.json。判决人 = Delight。</p>" + "".join(html_rows) + "</body></html>",
        encoding="utf-8")

    (BLIND / "RULES.md").write_text(render_rules(src_check), encoding="utf-8")

    verify = {
        "gate": "build-self-check", "pass": True,
        "n_items": 24, "n_wav_on_disk": len(names), "n_sheet_rows": len(sheet),
        "join_sheet_to_files": f"{len(sheet)}/{len(sheet)}",
        "sha1_copy_equals_source": f"{len(sheet)}/{len(sheet)}",
        "dup_equals_ref_bytes": "24/24",
        "filenames_ascii": True, "wav_headers_readable": True, "names_unique": True,
        "riff_uniform": "3/1/24000/32（IEEE_FLOAT）",
        "src_check": src_check,
        "script_md5": md5_file(Path(__file__)),
    }
    (BLIND / "VERIFY.json").write_text(json.dumps(verify, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verify, ensure_ascii=False, indent=1))
    print("BUILD OK ->", BLIND)
    return 0


def render_rules(src_check: dict) -> str:
    return f"""# 判读规则卡（{CARD_ID}，正式 96 包 24 击 × REF + 75/13p/dup）

> 判决人 = **Delight**。每条候选判 `0=与 REF 听不出差别 / 1=可辨·轻微 / 2=可辨·明显`，
> 填 `JUDGMENT_SHEET.csv` 的 `judgment` 列。听完再开 `BLIND_KEY.json`（答案键不随包发放）。
> 判决包 = `outputs/codec_listen_n24_1266/listen/`（composition `baee2cdf` / index `ad7c430f`）。
> **听感样本，跨 split 混采（cymbal 12 = test 3 / val 3 / train 6；hihats 6 = train 4 / test 1 / val 1；click 6 = train 6），非泛化证据**（PI 裁决 78）。

## 门（按序执行，缺一不可）
1. **dup 负对照满额零容错（PI 61①）**：24 条隐藏 dup 必须 **24/24 全判 0**；差 1 条整卡作废，
   不补样、不扩样、不复判，卡面只写「仪器失效」。dup 仅仪器自检，**不得**当任何臂的统计证据
   （禁写「通过 dup 检验故 75 臂可接受」）。机械换算：n=6 → 6/6、n=12 → 12/12、24 击 → 24/24。
2. **hihats 门（PI ㉝ + 62②，只看 hihats 池 75 臂判 0 条数，n=6）**：
   - `>=5/6` → 只准写「hihats 未见退化（低置信）」；
   - `3–4/6` → 中间带（`n_midband` 单列、未判定）**扩样 12 复判一次**（`listen_hihats_v2` 同池同四臂，只准一次）；
   - `<=2/6` → 扩样 12 条复判一次。
   扩样 12（同池同四臂、只准一次）：`>=10/12` → 只准写「hihats 未见退化（低置信）」；
   `7–9/12` → 结案「证据不足以判定」+ 记 v0 已知短板，不再扩样；`<=6/12` → 记「hihats 退化，n=12」。
   **32 层出口已删除、永不触发。**
3. **总 verdict = hihats 门 + dup 门**（PI 62③）。其余族进失败案例清单。
4. **报数区（不参与 verdict）**：cymbal n=12 / click_60_200ms n=6，**不设通过线、不标置信档**，
   只填 `判0 n / 判1 n / 判2 n` + 样本量 + 三档明细；`cymbal >=10/12 判 0` 只进「观察记录」栏。
   栏位禁空栏：未判写「未判（Delight 未填）」，不许出现空格子。
5. **13p 正对照**：仅 n=12 维持 `>=8/12 判 >=1`；n=6 / n=24 阶段只报数、不设门（PI 未给数字，Eval 不自创）。
6. **中间带**：`judgment=1` 一律单列 `n_midband`，同时报 n 与 95% CI；CI 跨判定线 = 「证据不足以判定」，
   不得取三档任一档；不进通过/不通过汇总、不进 §0、不进结论段。
7. **卡头两行并列铁律**：人耳结果行与「hihats mel **0.8429** > 0.8 Gate 越线登记（provisional）」行必须同时在卡头；
   只写其一打回；**人耳通过不解除越线登记**。禁任何单行结论把 hihats 单独判为通过（含「已通过」「已稳」）。

## 禁用字样（grep 0 命中为硬门槛）
按 agent.md ㉛ / §14.14 / §14.24 清单执行（含口径误判的跨重建增量误读数、臂比较形容词、历史误名臂名、
「组级取优」类表述），**具体字面量不在此复述**，以 agent.md 为准；卡产物 grep 0 命中为硬门槛。
臂命名唯一：R1 对齐臂 = `13p_align`（150.4%；对齐前 155.4%，同一臂前后一对）；试听 wav 文件名 `13p` 仅作样本标识，
**文件名 `13p` ≠ 独立臂**。

## 凭证
逐文件 sha1 见 `INDEX.json`；`dup` 与其 `REF` 字节全等 24/24（`VERIFY.json`）；pack 级 sha256 永不作复现凭证。
源包锚定：`index.json` md5 `{src_check["src_index_md5"][:8]}` / `composition.json` md5 `{src_check["src_composition_md5"][:8]}` /
`per_hit.csv` md5 `{src_check["per_hit_csv_md5"][:8]}`（R4 冻结件，join 0 mismatch）。
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (round((c - r) / d, 4), round((c + r) / d, 4))


def score() -> int:
    keydoc = json.loads((BLIND / "BLIND_KEY.json").read_text(encoding="utf-8"))
    key = keydoc["entries"]
    with open(BLIND / "JUDGMENT_SHEET.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filled = [r for r in rows if str(r["judgment"]).strip() in JUDGMENT_SCALE]
    if len(filled) != len(rows):
        print(f"未填完：{len(filled)}/{len(rows)}，先让 Delight 填完 JUDGMENT_SHEET.csv")
        return 2

    per_arm = {a: [] for a in ARMS}
    pool_arm = {}
    for r in rows:
        ent = key[r["item"]]
        arm = ent["slots"][r["slot"]]
        j = int(str(r["judgment"]).strip())
        per_arm[arm].append(j)
        pool_arm.setdefault((ent["pool"], arm), []).append(j)
    cnt = {a: {"0": v.count(0), "1": v.count(1), "2": v.count(2)} for a, v in per_arm.items()}
    n_midband = {a: cnt[a]["1"] for a in ARMS}

    # ---- 门 1：dup 满额零容错（PI 61①）
    dup_n = len(per_arm["dup"])
    dup_ok = cnt["dup"]["0"] == dup_n

    # ---- 门 2：hihats（只看 hihats 池 75 臂判 0）
    h75 = pool_arm.get(("hihats", "75"), [])
    n_h, h75_0 = len(h75), h75.count(0)
    pos_ok = None  # 仅 n=12 有 PI 正对照数字；本包 n=24 只报数
    if not dup_ok:
        verdict = (f"仪器失效（dup 满额零容错失败：{cnt['dup']['0']}/{dup_n} 判 0）→ "
                   "当轮全部听感结论作废；不补样、不扩样、不复判")
    elif n_h == 6:
        if h75_0 >= 5:
            verdict = f"hihats 未见退化（低置信）（n=6）；{GATE_NOTE_TAIL}"
        elif h75_0 >= 3:
            verdict = "MIDBAND（hihats 3–4/6 判 0：n_midband 单列、未判定）→ 扩样 12 复判一次（同池同四臂，只准一次）"
        else:
            verdict = "hihats 判 0 ≤2/6 → 扩样 12 条复判一次（同池同四臂，只准一次）"
    elif n_h == 12:
        if h75_0 >= 10:
            verdict = f"hihats 未见退化（低置信）（n=12 扩样）；{GATE_NOTE_TAIL}"
        elif h75_0 >= 7:
            verdict = "证据不足以判定（hihats 7–9/12 判 0）+ 记 v0 已知短板；不再扩样（禁无限抽样爬坡）"
        else:
            verdict = "hihats 退化，n=12（判 0 ≤6/12）；记 v0 已知短板"
    else:
        verdict = f"hihats n={n_h} 不属于判决集（6 或 12）→ 只报数不判定"

    # ---- 报数区（不参与 verdict）：分族三元组，禁空栏
    report_only, observation = {}, []
    for pool in sorted({p for (p, _a) in pool_arm}):
        report_only[pool] = {}
        for a in ARMS:
            js = pool_arm.get((pool, a), [])
            if not js:
                report_only[pool][a] = {"n": 0, "判0": "未测", "判1": "未测", "判2": "未测"}
                continue
            report_only[pool][a] = {"n": len(js), "判0": js.count(0), "判1": js.count(1), "判2": js.count(2),
                                    "三档明细": f"判0 {js.count(0)} / 判1 {js.count(1)} / 判2 {js.count(2)}（n={len(js)}）",
                                    "判0率_Wilson95CI": wilson(js.count(0), len(js))}
        js75 = pool_arm.get((pool, "75"), [])
        if pool == "cymbal" and len(js75) == 12 and js75.count(0) >= 10:
            observation.append(f"cymbal: 75 臂判 0 {js75.count(0)}/12（观察记录，不构成通过线，PI 62③）")

    out = {
        "card_id": CARD_ID,
        "judger": "Delight",
        "counts": cnt, "n_midband": n_midband,
        "n_midband_note": "judgment=1「可辨·轻微」条数；未判定，不进通过/不通过汇总、不进 §0 或结论段；同时报 n 与 Wilson 95% CI，CI 跨判定线 = 证据不足以判定",
        "neg_control_dup_ok": dup_ok,
        "neg_control_dup_quota": f"{cnt['dup']['0']}/{dup_n}（满额零容错，PI 61①；仅仪器自检、不作统计证据）",
        "pos_control_13p_note": "本包 n=24 只报数不设门（仅 n=12 维持 ≥8/12；PI 未给其余 n 的数字，Eval 不自创）",
        "hihats_n": n_h, "hihats_75_zero": h75_0,
        "n75_indistinguishable_all": per_arm["75"].count(0),
        "report_only_by_pool": report_only,
        "report_only_note": "报数区（不参与 verdict）：分族不设通过线、不标置信档（PI 62③ + 74/75）；栏位禁空栏",
        "observation": observation,
        "verdict": verdict,
        "card_head_rule": "卡头两行并列铁律：本 verdict 行 + 「hihats mel 0.8429 > 0.8 Gate 越线登记（provisional）」行必须同时在；人耳通过不解除登记",
        "cross_split_note": "听感样本，跨 split 混采（PI 裁决 78），非泛化证据",
        "arm_naming": "R1 对齐臂唯一名 13p_align（150.4%；对齐前 155.4% 同一臂）；试听 wav 臂名 13p 仅作样本标识；文件名 13p ≠ 独立臂",
        "confidence": "低（判决样本 hihats n=6、扩样上限 n=12 且只准一次；n_midband 单列不进汇总）",
    }
    (BLIND / "SCORE_CARD.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# {CARD_ID} SCORE_CARD", "", f"- verdict: **{out['verdict']}**",
             f"- dup 负对照：{out['neg_control_dup_quota']}", f"- hihats：75 臂判 0 {h75_0}/{n_h}",
             f"- n_midband：{json.dumps(n_midband, ensure_ascii=False)}", "",
             "## 报数区（不参与 verdict）"]
    for pool, arms in report_only.items():
        for a, d in arms.items():
            lines.append(f"- {pool} / {a}: {json.dumps(d, ensure_ascii=False)}")
    lines += ["", "## 观察记录", *(observation or ["（空）"]), "",
              "> 卡头两行并列铁律 + 跨 split 混采非泛化证据 + 禁用字样清单见 RULES.md。"]
    (BLIND / "SCORE_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.build:
        return build()
    if args.verify:
        print(json.dumps({"src": verify_src(),
                          "blind_files": len(list(BLIND.glob("*.wav"))) if BLIND.exists() else 0},
                         ensure_ascii=False, indent=1))
        return 0
    if args.score:
        return score()
    ap.error("需要 --build / --verify / --score")
    return 2


if __name__ == "__main__":
    sys.exit(main())
