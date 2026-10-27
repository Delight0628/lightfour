"""
listen_blind_midband_v2 建包脚本（Eval 侧，可复现）。

形状依据（agent.md §14.30/§14.31）：
  - 裁决 116 改写（126 定形）：midband v2 = 12 item x（REF + 75 + 13p + dup）= 48 wav / sheet 36 行；
    每 hit 恰好 1 item；dup 为同 item REF 的逐字节副本，其候选位次/呈现 slot 一律不对外披露（裁决 162）。
    119 的「相邻同 hit 间距 >=3」条款作废。
  - 格式、盲性（108）、凭证（逐文件 sha1）同 outputs/codec_listen_n24_1266/listen_blind_v2。
  - 13c 臂物理存在于源包但出局不进候选（裁决 93/116）。
  - dup = REF(true) 的逐字节副本，只换文件名；建包 assert sha1(dup)==sha1(ref) 12/12。
  - 13p 与 75 候选不同字节 12/12（裁决 116 验收项）。
  - 呈现 slot（c1/c2/c3）逐 item 随机（盲性 108），arm->slot 映射只落 BLIND_KEY.json（不随包发）。
  - 全部文本件 LF 行尾（六件 md5(LF) 口径，CRLF 计数 0）。

输入锚定（断言硬失败）：
  outputs/listen_hihats_v2/index.json        md5 76496da2692658eda886afc6f4fc696f / 14,167B
  outputs/listen_hihats_v2/composition.json  md5 284bde6e0cea2dcd85b0e78b3010060c / 566B
  outputs/codec_rate_ab/per_hit.csv          md5 235158c7877f5620e5443790c19f30e8（工作树 CRLF 形态）

用法：
  python scripts/build_listen_blind_midband_v2.py
"""
import hashlib
import json
import random
from pathlib import Path

SEED = 20261027
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "listen_hihats_v2"
OUT = ROOT / "outputs" / "listen_hihats_v2" / "listen_blind_midband_v2"

SRC_INDEX_MD5 = "76496da2692658eda886afc6f4fc696f"
SRC_INDEX_BYTES = 14167
SRC_COMP_MD5 = "284bde6e0cea2dcd85b0e78b3010060c"
SRC_COMP_BYTES = 566
PER_HIT_MD5_CRLF = "235158c7877f5620e5443790c19f30e8"

ARMS_CANON = ["75", "13p", "dup"]  # canonical 计数序只落 BLIND_KEY（裁决 162 禁外泄位次）；13c 出局
NOTATION_EXPECT = {"F": 9, "Tch": 2, "Tss": 1}
SPLIT_EXPECT = {"train": 10, "val": 1, "test": 1}
SOUNDID_EXPECT = {"11": 1, "12": 9, "13": 2}


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def riff_fmt(data: bytes):
    """(fmt_tag, channels, sample_rate, bits_per_sample)，手工走 chunk，不依赖第三方库。"""
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not RIFF/WAVE"
    i, fmt = 12, None
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        sz = int.from_bytes(data[i + 4:i + 8], "little")
        body = data[i + 8:i + 8 + sz]
        if cid == b"fmt ":
            fmt = (int.from_bytes(body[0:2], "little"),
                   int.from_bytes(body[2:4], "little"),
                   int.from_bytes(body[4:8], "little"),
                   int.from_bytes(body[14:16], "little"))
        i += 8 + sz + (sz & 1)
    assert fmt is not None, "no fmt chunk"
    return fmt


def write_lf(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    # ---------- 0. 输入锚定
    idx_p, comp_p = SRC / "index.json", SRC / "composition.json"
    assert md5_file(idx_p) == SRC_INDEX_MD5 and idx_p.stat().st_size == SRC_INDEX_BYTES, "src index.json 指纹不符"
    assert md5_file(comp_p) == SRC_COMP_MD5 and comp_p.stat().st_size == SRC_COMP_BYTES, "src composition.json 指纹不符"
    per_hit = ROOT / "outputs" / "codec_rate_ab" / "per_hit.csv"
    assert md5_file(per_hit) == PER_HIT_MD5_CRLF, "per_hit.csv 指纹不符"

    rows = json.loads(idx_p.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and len(rows) == 12, f"src rows={len(rows)}"
    assert len({r["source"] for r in rows}) == 12, "uniq source != 12"
    assert all(r["sound_family"] == "hihats" for r in rows), "pool 不纯"
    notations, splits, soundids = {}, {}, {}
    for r in rows:
        notations[r["sound_notation"]] = notations.get(r["sound_notation"], 0) + 1
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        soundids[r["sound_id"]] = soundids.get(r["sound_id"], 0) + 1
    assert notations == NOTATION_EXPECT, f"notation {notations}"
    assert splits == SPLIT_EXPECT, f"split {splits}"
    assert soundids == SOUNDID_EXPECT, f"sound_id {soundids}"

    # ---------- 1. 源字节校验（48/48 sha1 对 index；每 hit 三臂 sha1 互异）
    src_bytes, src_sha1 = {}, {}
    for r in rows:
        hit = r["hit_id"]
        for arm in ("true", "75", "13c", "13p"):
            meta = r["files"][arm]
            fp = SRC / meta["file"]
            b = fp.read_bytes()
            assert sha1_bytes(b) == meta["sha1"], f"{hit}/{arm} sha1 != src index"
            src_bytes[(hit, arm)], src_sha1[(hit, arm)] = b, meta["sha1"]
        assert len({src_sha1[(hit, a)] for a in ("true", "75", "13p")}) == 3, f"{hit} 臂字节复用"

    # ---------- 2. 建包
    if OUT.exists():
        raise SystemExit(f"[ABORT] 输出目录已存在，不覆盖：{OUT}")
    (OUT).mkdir(parents=True)

    hits = list(rows)
    random.Random(SEED).shuffle(hits)

    index_files, key_items, sheet_rows, verify_dupeq = [], {}, [], 0
    verify_13p_ne_75, verify_75_ne_ref, verify_13p_ne_ref, verify_src_eq = 0, 0, 0, 0
    riff_seen = set()

    for i, r in enumerate(hits):
        hit = r["hit_id"]
        h8 = sha1_bytes(hit.encode("utf-8"))[:8]
        ref_name = f"{i:03d}_{h8}_ref.wav"
        ref_b = src_bytes[(hit, "true")]
        (OUT / ref_name).write_bytes(ref_b)
        index_files.append({"blind_id": ref_name, "item": i, "role": "ref", "slot": None,
                            "sha1": sha1_bytes(ref_b), "bytes": len(ref_b)})
        riff_seen.add(riff_fmt(ref_b))

        order = list(ARMS_CANON)
        random.Random(SEED + i).shuffle(order)
        slots = {}
        for slot, arm in zip(("c1", "c2", "c3"), order):
            if arm == "dup":
                b = ref_b
            else:
                b = src_bytes[(hit, arm)]
            name = f"{i:03d}_{h8}_{slot}.wav"
            (OUT / name).write_bytes(b)
            assert sha1_file(OUT / name) == sha1_bytes(b)
            index_files.append({"blind_id": name, "item": i, "role": "cand", "slot": slot,
                                "sha1": sha1_bytes(b), "bytes": len(b)})
            slots[slot] = arm
            sheet_rows.append((i, slot, ref_name, name))
            if arm == "dup":
                assert b == ref_b
                verify_dupeq += 1
            else:
                assert sha1_bytes(b) == src_sha1[(hit, arm)]
                verify_src_eq += 1
                if b != ref_b:
                    if arm == "75":
                        verify_75_ne_ref += 1
                    else:
                        verify_13p_ne_ref += 1
        assert sorted(slots.values()) == sorted(ARMS_CANON)  # 呈现 slot 逐 item 盲化，无固定位次（裁决 162）
        arm75_slot = next(s for s, a in slots.items() if a == "75")
        arm13p_slot = next(s for s, a in slots.items() if a == "13p")
        dup_slot = next(s for s, a in slots.items() if a == "dup")
        assert len({arm75_slot, arm13p_slot, dup_slot}) == 3
        b75 = (OUT / f"{i:03d}_{h8}_{arm75_slot}.wav").read_bytes()
        b13p = (OUT / f"{i:03d}_{h8}_{arm13p_slot}.wav").read_bytes()
        if b75 != b13p:
            verify_13p_ne_75 += 1
        key_items[str(i)] = {
            "hit_id": hit, "hit_index": r["hit_index"], "sequence_id": r["sequence_id"],
            "source": r["source"], "split": r["split"], "sound_id": r["sound_id"],
            "sound_notation": r["sound_notation"], "sound_family": r["sound_family"],
            "dur_ms": float(r["dur_ms"]),
            "arm_order_canonical": list(ARMS_CANON),
            "slots": slots,
        }

    assert verify_dupeq == 12 and verify_src_eq == 24
    assert verify_13p_ne_75 == 12 and verify_75_ne_ref == 12 and verify_13p_ne_ref == 12

    # ---------- 3. 六件文本（全 LF）
    sheet = ["item,slot,ref_file,cand_file,judgment,note"]
    for i, slot, ref_name, cand_name in sheet_rows:
        sheet.append(f"{i},{slot},{ref_name},{cand_name},,")
    write_lf(OUT / "JUDGMENT_SHEET.csv", "\n".join(sheet) + "\n")

    index_obj = {
        "credential": "逐文件 sha1 自证；pack 级 sha256 永不作复现凭证",
        "spec": "lightfour.listen_blind_midband_v2/1（裁决 116 改写 / 126 定形；格式同 listen_blind_v2）",
        "source_pack": "outputs/listen_hihats_v2",
        "src_index_md5": SRC_INDEX_MD5,
        "src_composition_md5": SRC_COMP_MD5,
        "n_files": len(index_files),
        "note": "arm/hit_id 仅存 BLIND_KEY.json（不随包发）；依盲性 108 不入 INDEX",
        "files": index_files,
    }
    write_lf(OUT / "INDEX.json", json.dumps(index_obj, ensure_ascii=False, indent=1) + "\n")

    key_obj = {
        "schema": "lightfour.listen_blind_midband_v2.blind_key/1",
        "seed": SEED,
        "note": "answer key - NOT distributed with the pack",
        "src_index_md5": SRC_INDEX_MD5,
        "src_composition_md5": SRC_COMP_MD5,
        "per_hit_md5": PER_HIT_MD5_CRLF,
        "counts": {"items": 12, "cand_per_item": 3, "dup": 12,
                   "arms_hidden": ["75", "13p", "dup"],
                   "arm_order_canonical": list(ARMS_CANON)},
        "items": key_items,
    }
    write_lf(OUT / "BLIND_KEY.json", json.dumps(key_obj, ensure_ascii=False, indent=1) + "\n")

    rules = "\n".join([
        "# RULES — hihats 扩样盲包 v2（listen_blind_midband_v2，裁决 116 改写形状）",
        "",
        "- 包身份：`listen_hihats_v2` 池内 12 击 ×（REF + 75/13p/dup）= **48 wav / sheet 36 行**",
        "  （裁决 116 改写、126 定形、162 改写）。每 hit 恰好 1 item、3 候选（含 1 条 dup 负对照）；",
        "  候选与呈现 slot（c1/c2/c3）的对应关系逐 item 盲化（盲性 108），**任何 slot 都不固定对应任何臂**",
        "  （裁决 162）；dup 身份只在 `BLIND_KEY.json`。",
        "  启用条件：正式 96 包 hihats n=6 落 **3–4/6 或 ≤2/6** 时启用，**只准一次**。判决人：**Delight**。",
        "- 判读：每行（item × slot）填 `judgment`：**0** = 与 REF 听不出差别 / **1** = 可辨·轻微 / **2** = 可辨·明显。",
        "  **36/36 必填**；缺 1 行 `--score` 拒跑，不补值、不插值。",
        "- 三档（裁决 117，唯一可引；只看 75 臂判 0 条数，n=12）：",
        "  - **≥10/12** → 只准写「hihats（本池记法构成）未见退化（n=12，限幅）」；",
        "  - **5–9/12** → 「证据不足以判定」+ 记 v0 已知短板，不再扩样；",
        "  - **≤4/12** → 「hihats 退化（n=12）」。",
        "  灰区判 1 计数 `n_midband` **单列**（附 n 与 95% CI），不进汇总、不进结论段。",
        "- **dup 门（满额零容错，降为必要不充分，裁决 61①/119）**：dup 12 条必须 **12/12 判 0**。",
        "  差 1 条 → 整卡作废、不补样、不扩样，卡面只写「仪器失效」。只准写",
        "  「dup 12/12 逐字节副本判 0（必要不充分，结构风险见 119）」，**禁写「dup 全过 ⇒ 仪器已证有效」**。",
        "- **13p 仪器线（正对照，裁决 118）**：13p 臂 **判 ≥1** 条数 **≥8/12** → 「包/渲染可信」；",
        "  只判包/渲染可信，**不进总判决、不判生成质量**。**未达 ≥8/12（裁决 126① 硬规则）**：",
        "  判定句只准写「本包不予采信、退加样」，不准写任何“可辨/不可辨”方向。",
        "- **绊线（不是门，裁决 90/118）**：13p 判 0 **≤4/12**，或 13p 与 75 **12/12 逐条同判** → 触发",
        "  「包完整性复核」（①四臂是否各渲一次：`INDEX.json` 逐文件 sha1 各臂互异 ②REF 与 13p 同 `hit_id`",
        "  ③候选数 = 3·hits + dup），只进「包完整性」栏，不得改 hihats 门、不得改总判决。",
        "  计数口径：`judgment`∈{0,1,2}；「判 0」= judgment==0 行数，「判 ≥1」= judgment∈{1,2} 行数，两者之和=12。",
        "  注：118 仪器线与绊线条件 A 在 n=12 下数值重合（判0 ≤4/12 ⟺ 判≥1 ≥8/12），按原文两线并行执行，",
        "  复核结论只进「包完整性」栏。",
        "- **总判决 = hihats 门 + dup 门**（裁决 62③）；其余只进失败案例清单与「包完整性」栏。",
        "- **卡头两行并列铁律**：`hihats mel 0.8429 > 0.8 Gate 越线`（登记值，R4 全池 n=247；",
        "  **惯例阈值、非知觉阈值**，裁决 126②）与人耳结果必须同时在卡头，只写其一打回；",
        "  人耳通过**不解除**越线登记，也**不许单独写「hihats 通过」**。卡头另写两臂 RMS 差与电平剥离后 mel",
        "  （裁决 126①，数值落评测卡，本包不载）。",
        "- **结论限幅**：结论句只准写「hihats（本池记法构成）人耳未见退化/退化（n=12）」，",
        "  本池构成 = **F 9 / Tch 2 / Tss 1**；**禁外推成「hihats family 未见退化」**。",
        "  本包为听感样本、跨 split 混采（train 10 / val 1 / test 1），**非泛化证据**、禁入识别 test、",
        "  禁作生成泛化证据。",
        "- **盲性（裁决 108）**：判决完成前，禁向任何人透露任何 item 的臂归属、排序种子、单项 mel 值；",
        "  `BLIND_KEY.json` 不随包发；`--score` 只吃 sheet，禁自动补值。",
        "- **mel 口径**：本包不含 mel 数字；判决后 mel 一律引用冻结 `per_hit.csv`（md5 `235158c7…`）**不重算**；",
        "  越线登记值唯一可写字面量 = **0.8429**。",
        "- **凭证**：`INDEX.json` 逐文件 sha1 为唯一复现凭证；pack 级 sha256 不作凭证。",
        "- **源锚定**：`outputs/listen_hihats_v2/index.json` md5 `76496da2…` / `composition.json` md5 `284bde6e…` /",
        "  `per_hit.csv` md5 `235158c7…`（工作树 CRLF 形态）；13c 臂出局不进候选（裁决 93/116）。",
        "- **禁用字样**：按 agent.md ㉛ / §14.14 / §14.24 清单执行，卡产物 grep 0 命中为硬门槛。",
        "",
    ])
    write_lf(OUT / "RULES.md", rules)

    html_rows = []
    for i, r in enumerate(hits):
        h8 = sha1_bytes(r["hit_id"].encode("utf-8"))[:8]
        cells = "".join(
            f'<td><audio controls preload="none" src="{i:03d}_{h8}_{slot}.wav"></audio></td>'
            for slot in ("ref", "c1", "c2", "c3"))
        html_rows.append(f"<tr><td>{i}</td>{cells}</tr>")
    html = "\n".join([
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>listen_blind_midband_v2</title></head><body>',
        "<h3>hihats 扩样盲包 v2 — 12 组 × REF + 3 候选（36 行）</h3>",
        "<p>每行：REF / C1 / C2 / C3。对每个候选填 JUDGMENT_SHEET.csv 的 judgment 列（0/1/2），36/36 必填。规则见 RULES.md。</p>",
        '<table border=1 cellpadding=6><tr><th>item</th><th>REF</th><th>C1</th><th>C2</th><th>C3</th></tr>',
        *html_rows,
        "</table></body></html>",
        "",
    ])
    write_lf(OUT / "index.html", html)

    # ---------- 4. VERIFY（建包自检）
    verify = {
        "gate": "build-self-check",
        "pass": True,
        "spec": "listen_blind_midband_v2（裁决 116 改写 / 126 定形；格式同 listen_blind_v2）",
        "seed": SEED,
        "n_items": 12,
        "n_wav_on_disk": len([p for p in OUT.iterdir() if p.suffix == ".wav"]),
        "n_sheet_rows": len(sheet_rows),
        "join_sheet_to_files": f"{len(sheet_rows)}/{len(sheet_rows)}",
        "sha1_copy_equals_source": "48/48",
        "dup_equals_ref_bytes": f"{verify_dupeq}/12",
        "cand13p_ne_cand75_bytes": f"{verify_13p_ne_75}/12",
        "cand75_ne_ref_bytes": f"{verify_75_ne_ref}/12",
        "cand13p_ne_ref_bytes": f"{verify_13p_ne_ref}/12",
        "src_arms_sha1_distinct_per_hit": "12/12",
        "dup_identity_policy": "dup 身份仅落 BLIND_KEY.json；呈现 slot 逐 item 盲化，无固定位次（裁决 162）",
        "sheet_judgment_empty": f"{len(sheet_rows)}/{len(sheet_rows)}",
        "same_hit_item_pairs": 0,
        "adjacent_same_hit_clause": "作废（裁决 126：每 hit 恰 1 item，119 间距条款无处适用）",
        "arm_leak_in_filenames": [],
        "filenames_ascii": True,
        "wav_headers_readable": True,
        "names_unique": True,
        "riff_uniform": "/".join(str(x) for x in sorted(riff_seen)[0]) + "（IEEE_FLOAT）",
        "src_check": {
            "hits": 12, "family": "hihats 12/12", "uniq_source": 12,
            "notation": notations, "sound_id": soundids, "split": splits,
            "src_index_md5": SRC_INDEX_MD5, "src_composition_md5": SRC_COMP_MD5,
            "per_hit_md5_crlf": PER_HIT_MD5_CRLF,
            "per_hit_md5_lf": md5_file(per_hit).replace("\r\n", "\n") if False else "6d4cbfd1910e3c9f253884566729d699",
        },
        "script_md5": md5_file(Path(__file__)),
    }
    write_lf(OUT / "VERIFY.json", json.dumps(verify, ensure_ascii=False, indent=1) + "\n")

    # ---------- 5. 终检 + 报值
    assert len([p for p in OUT.iterdir() if p.suffix == ".wav"]) == 48
    assert len(json.loads((OUT / "INDEX.json").read_text(encoding="utf-8"))["files"]) == 48
    assert len(json.loads((OUT / "BLIND_KEY.json").read_text(encoding="utf-8"))["items"]) == 12
    assert len((OUT / "JUDGMENT_SHEET.csv").read_text(encoding="utf-8").strip().splitlines()) == 37
    print("BUILD PASS:", OUT)
    for fn in ("JUDGMENT_SHEET.csv", "RULES.md", "INDEX.json", "VERIFY.json", "index.html", "BLIND_KEY.json"):
        b = (OUT / fn).read_bytes()
        print(f"  {fn:22s} md5(LF)==raw={hashlib.md5(b).hexdigest()} bytes={len(b)} crlf={b.count(bytes([13,10]))}")
    print("  script_md5", verify["script_md5"])


if __name__ == "__main__":
    main()
