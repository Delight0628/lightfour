"""
构建 103 音色资产：crosswalk + seed manifest + source 分组 split + family 覆盖统计 + QC。

用法（本地可跑，零依赖）:
  python scripts/build_seed_manifest.py \
      --taxonomy data/beatbox_taxonomy.json \
      --seedlib  data/beatbox_seeds/sounds.json \
      --csv      data/sound_seed_annotation.csv \
      --listen-pack data/bbox_raw/sound_seed_pack \
      --out-dir  data

可选（通道恢复后，合并远端 hybrid 标签里的 confidence/seed_dist/dur/start）:
  python scripts/build_seed_manifest.py --labels /gemini/code/lightfour-train/data/real_bbox/hybrid_labels_all_highconf_l2.jsonl ...

产物:
  data/beatbox_seed_crosswalk.csv   种子库记法 -> canonical 103 记法/家族 映射（可复算）
  data/seed_manifest.csv             408 条种子 canonical manifest（含 sound_notation/family/confidence/label_source/split）
  data/seed_split_train.txt|val.txt|test.txt   按 source(BV) 的 md5 分组切分
  data/seed_family_coverage.csv      family 覆盖统计（103 音色体系口径）
  data/sound_seed_annotation.csv     人耳标注 CSV（原 7 列保留 + 追加 canonical 列 + 标注状态）

关键口径（不得改动）:
  - 标签空间以 data/beatbox_taxonomy.json 的 103 音色列为准（sound_id 唯一，notation 不唯一）。
  - 种子库（schoolofbeatbox/beatboxscience）记法带 '<' 前缀，与 canonical 记法不是同一空间，
    必须经 crosswalk 归一，禁止把 '<W' 之类直接当训练标签。
  - label_source ∈ {human, seed_match, model, unknown}；sound_confidence 仅 human/model 非零。
  - split 按 source(BV) 的 md5 分组，禁止跨 source 泄漏；同一 BV 的所有切片必须同 split。
"""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- 归一化规则

_FAMILY_ALIAS = {
    "hi-hats": "hihats",
    "lip oscillations": "lip_oscillation",
    "clicks": "clicks",
    "clops": "clops",
    "kicks": "kicks",
    "snares": "snares",
    "vocal": "vocal",
    "tongue": "tongue",
    "nasal": "nasal",
    "whistles": "whistles",
    "teeth": "teeth",
    "drag": "drag",
    "zippers": "zippers",
    "scratch": "scratch",
    "reverse": "reverse",
    "synth": "synth",
    "other": "other",
    "pops": "other",      # 种子库 'Pops' 在 103 体系里归 other（Pop sound）
}


def norm_name(s):
    s = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def strip_paren(s):
    """'Inward K Rimshot (shorter - for fast patterns)' -> 'Inward K Rimshot'"""
    return re.sub(r"\s*\([^)]*\)\s*", " ", str(s)).strip()


def split_of(source):
    """按 source(BV) 的 md5 分组切分。与 prepare_talker_data.py / agent.md §10.1-4 同口径。"""
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


# ---------------------------------------------------------------- crosswalk

def build_crosswalk(taxonomy, seedlib):
    """种子库条目 -> canonical 103 音色条目。

    匹配顺序（2026-10-26 修订，见 docs/data_qc_report.md）：
      1. 记法精确匹配：seedlib 记法原样命中 canonical 记法 -> exact_notation
      2. 记法精确匹配：seedlib 记法去掉 '<' 前缀后命中 -> exact_notation_stripped_lt
         （种子库用 '<' 表达 inward；canonical 里 inward 也可能保留 '<'，故先试原样再试去前缀）
      3. 记法歧义：同一 canonical 记法对应多个音色（P/K/Kch/Df/Rhm/A 六组）时，
         只在该候选集合内用名称消歧 -> notation_ambiguous+name / +stripped_name
      4. 兜底：精确名 -> 去括号名 -> 同家族模糊匹配(>=0.62) -> unknown

    为何改为记法优先：原实现用「去括号名」匹配，把种子库的
    '(short - for fast patterns)' 短变体错并到基型（id 9/18/23），
    导致 canonical id 10/19/25 三个音色永远无种子覆盖（不可训练/不可评测）。
    改为记法优先后，103 条种子库条目与 103 个 canonical 音色构成完美双射
    （无重复、无孤儿），这是名称匹配无法给出的结构性证据。

    返回 (rows, tax_by_id, sd_by_not)
    """
    sounds = {i: s for i, s in enumerate(taxonomy["sounds"])}
    tax_by_name, tax_by_stripped = {}, {}
    for i, s in sounds.items():
        tax_by_name.setdefault(norm_name(s["en"]), i)
        tax_by_stripped.setdefault(norm_name(strip_paren(s["en"])), i)

    tax_by_notation = defaultdict(list)
    for i, s in sounds.items():
        tax_by_notation[s["notation"]].append(i)

    rows = []
    for j, e in enumerate(seedlib):
        cname = norm_name(e["name"])
        cname_stripped = norm_name(strip_paren(e["name"]))
        seed_not = str(e.get("notation") or "")
        tax_id, method, score = None, "none", 0.0

        # 1/2. 记法优先
        cands = list(tax_by_notation.get(seed_not, []))
        how = "exact_notation"
        if not cands and seed_not.startswith("<"):
            cands = list(tax_by_notation.get(seed_not[1:], []))
            how = "exact_notation_stripped_lt"
        if len(cands) == 1:
            tax_id, method, score = cands[0], how, 1.0
        elif len(cands) > 1:
            # 3. 歧义记法 -> 限定在候选集合内用名称消歧
            if cname in tax_by_name and tax_by_name[cname] in cands:
                tax_id, method, score = tax_by_name[cname], "notation_ambiguous+name", 1.0
            elif cname_stripped in tax_by_stripped and tax_by_stripped[cname_stripped] in cands:
                tax_id, method, score = tax_by_stripped[cname_stripped], "notation_ambiguous+stripped_name", 0.95

        if tax_id is None:
            # 4. 兜底
            if cname in tax_by_name:
                tax_id, method, score = tax_by_name[cname], "exact_name", 1.0
            elif cname_stripped in tax_by_stripped:
                tax_id, method, score = tax_by_stripped[cname_stripped], "stripped_paren", 0.95
            else:
                fam = _FAMILY_ALIAS.get(str(e.get("subcategory") or "").lower())
                cand = [i for i, s in sounds.items() if fam and s["family"] == fam] or list(sounds)
                best, best_r = None, 0.0
                for i in cand:
                    r = _ratio(cname, norm_name(sounds[i]["en"]))
                    if r > best_r:
                        best, best_r = i, r
                if best is not None and best_r >= 0.62:
                    tax_id, method, score = best, "fuzzy_family", round(best_r, 4)
        t = sounds[tax_id] if tax_id is not None else None
        is_short, variant_note = _short_variant_info(
            e.get("name"), t["en"] if t else "", method,
            bool(t.get("is_short_variant")) if t else False)
        rows.append({
            "seedlib_index": j,
            "seedlib_id": e.get("id"),
            "seedlib_name": e.get("name"),
            "seedlib_notation": e.get("notation"),
            "seedlib_subcategory": e.get("subcategory"),
            "canonical_sound_id": tax_id if tax_id is not None else "",
            "canonical_notation": t["notation"] if t else "unknown",
            "canonical_en": t["en"] if t else "unknown",
            "canonical_cn": t["cn"] if t else "unknown",
            "canonical_family": t["family"] if t else "unknown",
            "canonical_mechanism": t.get("mechanism", "") if t else "",
            "canonical_airflow": t.get("airflow", "") if t else "",
            "is_short_variant": is_short,
            "variant_note": variant_note,
            "match_method": method,
            "match_score": score,
            "notation_ambiguous": int(bool(t) and _notation_is_ambiguous(taxonomy, t["notation"])),
            "has_reference_audio": int(_seedlib_has_audio(e)),
        })
    return rows, sounds


def _ratio(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _notation_is_ambiguous(taxonomy, notation):
    c = Counter(s["notation"] for s in taxonomy["sounds"])
    return c[notation] > 1


def _seedlib_has_audio(e):
    """种子库条目是否带参考音频。

    参考音频文件名形如 '004_inward_k_rimshot_shorter_for_fast_patterns.mp3'：
    三位序号前缀 + 名称 slug。原实现只按 slug 查找，导致 has_reference_audio
    全为 0（2026-10-26 修正：按序号前缀匹配）。
    """
    name = e.get("name") or ""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    audio_dir = Path("data/beatbox_seeds/audio")
    if not audio_dir.exists():
        return 0
    for f in audio_dir.glob("*.mp3"):
        if re.sub(r"^\d+_", "", f.stem) == slug:
            return 1
    return 0


_SHORT_MARK = re.compile(r"short", re.I)


def _short_variant_info(seed_name, canon_en, match_method, canon_is_short=False):
    """短变体标记 + 来源说明。

    种子库里有一批 '... (short - for fast patterns)' 短变体条目，其 canonical
    对应关系必须落到 103 体系里自己的短变体音色（如 id 10/19/25），而不是基型，
    否则短变体永远不可单独训练/评测（PI 2026-10-26 裁决）。

    判据以 **canonical taxonomy 的 is_short_variant 字段为准**（canonical 体系里
    6 条短变体是独立音色：id 2/10/19/25/65/70），种子库名称里的 'short' 只是旁证。
    名称命中但 canonical 未标记 -> 记 1 并写「需人工复核」。

    返回 (is_short_variant, variant_note)。
    """
    seed_name = str(seed_name or "")
    canon_en = str(canon_en or "")
    seed_short = bool(_SHORT_MARK.search(seed_name))
    if canon_is_short:
        note = (f"种子库 '{seed_name}' -> canonical '{canon_en}'"
                f"（短变体，保留独立音色 id，不并入基型）")
        return 1, note
    if not seed_short:
        return 0, ""
    note = (f"种子库 '{seed_name}' 含 short 标记但 canonical '{canon_en}' 非短变体；"
            f"匹配方式 {match_method}，需人工复核")
    return 1, note


# ---------------------------------------------------------------- manifest

def build_manifest(csv_rows, crosswalk_by_not, taxonomy, labels_by_id, listen_pack):
    """CSV 408 行 -> canonical manifest 行。"""
    tax_by_id = {i: s for i, s in enumerate(taxonomy["sounds"])}
    pack_ids = set()
    if listen_pack and Path(listen_pack).exists():
        for f in Path(listen_pack).glob("*.wav"):
            pack_ids.add(re.sub(r"^[^_]+_", "", f.stem))

    out = []
    for r in csv_rows:
        seed_not = (r.get("candidate_notation") or "").strip()
        cw = crosswalk_by_not.get(seed_not)
        if cw is None:
            tax_id = None
            match_method, match_score = "no_seedlib_entry", 0.0
        else:
            tax_id = cw["canonical_sound_id"]
            match_method, match_score = cw["match_method"], cw["match_score"]
        t = tax_by_id.get(tax_id) if tax_id is not None else None

        wav = (r.get("wav") or "").strip()
        source = _source_of(wav) or _source_of(r.get("id", ""))
        extra = labels_by_id.get(r.get("id"), {}) if r.get("id") else {}

        human = (r.get("human_notation") or "").strip()
        status = (r.get("annotate_status") or "").strip()
        human_sid = None
        if status == "unknown" and not human:
            # 人耳判「不确定」：标签整体置 unknown，不保留伪标签的 sound_id
            sound_notation, sound_en, sound_cn, family = "unknown", "unknown", "unknown", "unknown"
            conf, label_source = 0.0, "unknown"
            tax_id = None
        elif human and status in ("confirmed", "unknown", "rejected"):
            # 人工确认：标签取 canonical (notation, en) 对应的 sound_id
            sound_notation = human
            hit = [s for s in taxonomy["sounds"] if s["notation"] == human]
            sound_en = hit[0]["en"] if len(hit) == 1 else "|".join(s["en"] for s in hit)
            sound_cn = hit[0]["cn"] if len(hit) == 1 else "|".join(s["cn"] for s in hit)
            family = _family_of_notation(taxonomy, human)
            conf = 1.0 if status == "confirmed" else 0.0
            label_source = "human" if status == "confirmed" else "unknown"
            # 人耳确认的记法若唯一对应一个 canonical 音色，sound_id 必须跟着标签走，
            # 否则 manifest 会出现「sound_id 属于 A、记法却是 B」的错配
            # （2026-10-27 复核时发现；is_short_variant 也必须按最终标签判定）。
            if len(hit) == 1:
                human_sid = taxonomy["sounds"].index(hit[0])
            if status == "unknown":
                sound_notation, sound_en, sound_cn = "unknown", "unknown", "unknown"
                family = "unknown"
                human_sid = None
                tax_id = None      # 不确定的样本不保留伪标签的 sound_id
        elif t is not None:
            sound_notation, sound_en, sound_cn = t["notation"], t["en"], t["cn"]
            family = t["family"]
            conf, label_source = 0.0, "seed_match"
        else:
            sound_notation, sound_en, sound_cn, family = "unknown", "unknown", "unknown", "unknown"
            conf, label_source = 0.0, "unknown"

        final_sid = human_sid if human_sid is not None else tax_id
        final_t = tax_by_id.get(final_sid) if final_sid is not None else None

        out.append({
            "id": r.get("id"),
            "source": source,
            "split": split_of(source) if source else "unknown",
            "wav": wav,
            "wav_local": _local_wav(wav, pack_ids, listen_pack),
            "in_listen_pack": int(r.get("id") in pack_ids),
            "sound_id": final_sid if final_sid is not None else "",
            "sound_notation": sound_notation,
            "sound_en": sound_en,
            "sound_cn": sound_cn,
            "sound_family": family,
            "is_short_variant": int(final_t is not None
                                    and bool(final_t.get("is_short_variant"))),
            "sound_confidence": conf,
            "label_source": label_source,
            "candidate_notation_seedlib": seed_not,
            "candidate_family_l2": (r.get("family") or "").strip(),
            "nearest_seed": (r.get("nearest_seed") or "").strip(),
            "crosswalk_method": match_method,
            "crosswalk_score": match_score,
            "seed_dist": extra.get("seed_dist", ""),
            "model_confidence": extra.get("confidence", ""),
            "dur": extra.get("dur", ""),
            "start": extra.get("start", ""),
            "annotate_status": status or ("annotated" if human else "pending"),
            "annotator": (r.get("annotator") or "").strip(),
            "note": (r.get("note") or "").strip(),
        })
    return out


def _source_of(path):
    m = re.search(r"(BV[0-9A-Za-z]+)", str(path))
    return m.group(1) if m else ""


# ---------------------------------------------------------------- self-test

def self_test(taxonomy_path="data/beatbox_taxonomy.json",
              seedlib_path="data/beatbox_seeds/sounds.json"):
    """不依赖音频/服务器：断言 crosswalk 的双射性与短变体归位。

    断言（PI 2026-10-26 裁决的门槛）：
      1. crosswalk 行数 == taxonomy 音色数 == seedlib 条目数
      2. canonical_sound_id 唯一值 == 103/103（无重复、无孤儿）
      3. 三条 'short - for fast patterns' 短变体必须归位到 10/19/25
      4. 短变体行必须带 is_short_variant=1 与非空 variant_note
      4b. canonical 体系全部 6 条短变体（2/10/19/25/65/70）都被标记且与 crosswalk 一致
      5. split_of 是 source 的纯函数（同一 source 恒等同一 split）
    """
    taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    seedlib = json.loads(Path(seedlib_path).read_text(encoding="utf-8"))
    crosswalk, _ = build_crosswalk(taxonomy, seedlib)

    n_tax = len(taxonomy["sounds"])
    assert len(crosswalk) == len(seedlib) == n_tax, \
        f"行数不一致 crosswalk={len(crosswalk)} seedlib={len(seedlib)} taxonomy={n_tax}"

    ids = [int(c["canonical_sound_id"]) for c in crosswalk
           if str(c["canonical_sound_id"]).strip() != ""]
    assert len(ids) == n_tax, f"有 {n_tax - len(ids)} 条未命中 canonical"
    assert len(set(ids)) == n_tax, \
        f"canonical_sound_id 非双射：唯一 {len(set(ids))}/{n_tax}，重复 {sorted(set(i for i in ids if ids.count(i) > 1))}"
    assert set(ids) == set(range(n_tax)), "canonical_sound_id 集合 != 0..n-1"

    # 3/4. 短变体归位
    expected_short = {
        "Inward K Rimshot (shorter - for fast patterns)": 25,
        "K palate snare (short - for fast patterns)": 19,
        "Hi-Hat T (short - for fast patterns)": 10,
    }
    by_name = {c["seedlib_name"]: c for c in crosswalk}
    for name, want_id in expected_short.items():
        c = by_name.get(name)
        assert c is not None, f"crosswalk 缺少种子库条目: {name}"
        assert int(c["canonical_sound_id"]) == want_id, \
            f"{name} 应归位 canonical id {want_id}，实为 {c['canonical_sound_id']}"
        assert int(c["is_short_variant"]) == 1, f"{name} 的 is_short_variant 应为 1"
        assert c["variant_note"], f"{name} 缺少 variant_note"

    # 短变体不得与基型共享 canonical id
    short_ids = {int(c["canonical_sound_id"]) for c in crosswalk
                 if int(c["is_short_variant"]) == 1}
    assert len(short_ids) == len([c for c in crosswalk if int(c["is_short_variant"]) == 1]), \
        "短变体之间存在 canonical id 冲突"

    # 4b. canonical 体系里全部 6 条短变体都必须被标记（含 3 条非 fast-pattern 的：
    #     Fat Kick Short / Inward Hollow Lip Roll (Short) / Vibrator Lip Bass (Short)）。
    #     判据以 taxonomy 的 is_short_variant 字段为准，不只认 '(short - for fast patterns)'。
    tax_short = {i for i, s in enumerate(taxonomy["sounds"]) if s.get("is_short_variant")}
    assert tax_short == {2, 10, 19, 25, 65, 70}, \
        f"taxonomy 的短变体集合应为 {{2,10,19,25,65,70}}，实为 {sorted(tax_short)}"
    cw_short = {int(c["canonical_sound_id"]) for c in crosswalk
                if int(c["is_short_variant"]) == 1}
    assert cw_short == tax_short, \
        f"crosswalk 短变体集合 {sorted(cw_short)} != taxonomy {sorted(tax_short)}"
    for cid in sorted(tax_short):
        row = [c for c in crosswalk if int(c["canonical_sound_id"]) == cid]
        assert len(row) == 1, f"canonical id {cid} 的短变体行数应为 1，实为 {len(row)}"
        assert row[0]["variant_note"], f"canonical id {cid} 的短变体缺 variant_note"

    # 5. split 纯函数
    for src in ("BV11L4AeCEYo", "BV1x411c7mD", "BV17x411w7KC"):
        assert split_of(src) == split_of(src)
    assert split_of("BV11L4AeCEYo") in ("train", "val", "test")

    print(f"self-test OK: crosswalk {len(crosswalk)} 行，canonical {len(set(ids))}/{n_tax} 双射；"
          f"短变体 {len(short_ids)} 个已归位（含 {sorted(short_ids & {10, 19, 25})}）；"
          f"split_of 纯函数已验证")
    return True


def _local_wav(wav, pack_ids, listen_pack):
    """本地试听包路径。

    优先级：
      1. sound_seed_pack（每 family 5 条试听包，文件名 '{Family}_{id}.wav'）
      2. seed_candidates（408 条候选全集，从远端持久盘拉回的扁平目录 '{id}.wav'）

    注意：pack 文件名带 Family_ 前缀，而 manifest 的 wav 列是 '{id}.wav'，
    必须按两种命名各查一次，否则 wav_local 会指向不存在的文件
    （2026-10-27 实测 86/86 失效）。
    """
    stem = os.path.basename(wav).replace(".wav", "")
    if not listen_pack:
        return ""
    pack = Path(listen_pack)
    p = pack / f"{stem}.wav"
    if p.exists():
        return str(p)
    for f in pack.glob(f"*_{stem}.wav"):
        return str(f)
    for f in pack.glob("*.wav"):
        if re.sub(r"^[^_]+_", "", f.stem) == stem:
            return str(f)
    # 候选全集目录
    for cand_root in (Path("data/bbox_raw/seed_candidates/cand408"),
                      Path("data/bbox_raw/seed_candidates")):
        c = cand_root / f"{stem}.wav"
        if c.exists():
            return str(c)
    return ""


def _family_of_notation(taxonomy, notation):
    fams = {s["family"] for s in taxonomy["sounds"] if s["notation"] == notation}
    return sorted(fams)[0] if len(fams) == 1 else ("ambiguous:" + "|".join(sorted(fams)) if fams else "unknown")


# ---------------------------------------------------------------- 统计 & QC

def family_coverage(manifest, taxonomy):
    """103 音色体系口径的 family 覆盖统计。"""
    tax_by_id = {i: s for i, s in enumerate(taxonomy["sounds"])}
    by_sound = defaultdict(list)
    for m in manifest:
        if m["sound_id"] != "":
            by_sound[m["sound_id"]].append(m)
    fam_total = Counter(s["family"] for s in taxonomy["sounds"])
    fam_hit = Counter()
    for sid, ms in by_sound.items():
        fam_hit[tax_by_id[sid]["family"]] += len(ms)
    fam_not = defaultdict(set)
    for sid in by_sound:
        fam_not[tax_by_id[sid]["family"]].add(tax_by_id[sid]["notation"])

    rows = []
    for fam in sorted(fam_total):
        hit_sounds = sorted(n for n in fam_not.get(fam, ()))
        rows.append({
            "sound_family": fam,
            "taxonomy_sounds": fam_total[fam],
            "sounds_with_seed": len(hit_sounds),
            "sounds_without_seed": fam_total[fam] - len(hit_sounds),
            "coverage_pct": round(len(hit_sounds) / fam_total[fam] * 100, 1),
            "seed_segments": fam_hit.get(fam, 0),
            "notations_observed": "|".join(hit_sounds),
            "human_confirmed": sum(1 for m in manifest
                                   if m["sound_family"] == fam and m["label_source"] == "human"),
        })
    return rows


def qc_report(manifest, crosswalk, taxonomy, coverage, split_files, out=Path("data")):
    issues = []
    notes = []

    amb = Counter(s["notation"] for s in taxonomy["sounds"])
    amb = {k: v for k, v in amb.items() if v > 1}
    notes.append(f"taxonomy 103 条中 notation 唯一值 {len(amb) and 103 - sum(v - 1 for v in amb.values())}"
                 f"/103，重复记法 {len(amb)} 组: {amb}")

    human = sum(1 for m in manifest if m["label_source"] == "human")
    notes.append(f"label_source 分布: {dict(Counter(m['label_source'] for m in manifest))}")

    # crosswalk 未命中（注意：canonical_sound_id 可能为整数 0，不能用 not 判断空）
    unmatched = [c for c in crosswalk
                 if str(c["canonical_sound_id"]).strip() == ""
                 and c["match_method"] != "exact_name"]
    notes.append(f"crosswalk 未命中 canonical 的种子库条目: {len(unmatched)}/{len(crosswalk)}")

    # crosswalk 覆盖与双射（2026-10-26 新增）
    cw_ids = [int(c["canonical_sound_id"]) for c in crosswalk
              if str(c["canonical_sound_id"]).strip() != ""]
    cw_distinct = len(set(cw_ids))
    cw_dup = {k: v for k, v in Counter(cw_ids).items() if v > 1}
    orphan_tax = sorted(set(range(len(taxonomy["sounds"]))) - set(cw_ids))
    notes.append(f"crosswalk 覆盖 canonical 音色 {cw_distinct}/{len(taxonomy['sounds'])}，"
                 f"重复映射 {len(cw_dup)} 个，"
                 f"无种子库条目的 canonical 音色 {len(orphan_tax)} 个: {orphan_tax}")
    if cw_dup:
        issues.append(f"crosswalk 存在重复映射 {cw_dup}：同一 canonical 音色被多条种子库条目指向，"
                      "会让短变体等音色永远不可单独训练/评测")
    if orphan_tax:
        notes.append(f"无种子库条目的 canonical 音色（须由人工标注补齐）: "
                     + ", ".join(f"{i}({taxonomy['sounds'][i]['en']})" for i in orphan_tax))
    notes.append("crosswalk 匹配方式分布: " + str(dict(Counter(c["match_method"] for c in crosswalk))))
    n_audio = sum(1 for c in crosswalk if c["has_reference_audio"])
    notes.append(f"种子库条目带参考音频(mp3): {n_audio}/{len(crosswalk)}")

    # 短变体归位（PI 2026-10-26 裁决）
    short_rows = [c for c in crosswalk if int(c.get("is_short_variant") or 0) == 1]
    if short_rows:
        notes.append(f"短变体条目 {len(short_rows)} 条（已归位到独立 canonical 音色，未并入基型）: "
                     + ", ".join(f"id{c['canonical_sound_id']}({c['canonical_en']})"
                                 for c in short_rows))
        # 全部 6 条短变体都必须归位，不只认 3 条 '(short - for fast patterns)'
        tax_short = {i for i, s in enumerate(taxonomy["sounds"])
                     if s.get("is_short_variant")}
        cw_short = {int(c["canonical_sound_id"]) for c in short_rows}
        if cw_short != tax_short:
            issues.append(f"短变体集合 crosswalk{sorted(cw_short)} != taxonomy{sorted(tax_short)}")
        else:
            notes.append(f"短变体集合与 canonical taxonomy 一致: {sorted(tax_short)}"
                         f"（taxonomy 已带 is_short_variant 字段，下游 manifest/refaudio/sequence 均可 join）")
        for want in sorted(tax_short):
            hit = [c for c in short_rows if int(c["canonical_sound_id"]) == want]
            if not hit:
                issues.append(f"canonical id {want} 的短变体未归位（该音色将不可单独训练/评测）")

    # source 泄漏
    by_split = defaultdict(set)
    for m in manifest:
        by_split[m["split"]].add(m["source"])
    leaks = []
    keys = list(by_split)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            leaks += sorted(by_split[keys[i]] & by_split[keys[j]])
    if leaks:
        issues.append(f"跨 split source 泄漏 {len(leaks)} 个: {leaks[:10]}")
    else:
        notes.append("跨 split source 泄漏: 0（同一 BV 的所有切片同 split）")

    # 孤儿 split
    unknown_split = [m["id"] for m in manifest if m["split"] == "unknown"]
    if unknown_split:
        issues.append(f"{len(unknown_split)} 条无法判定 source，无法归 split")

    # source_split_*.txt 覆盖完整性：序列样本用到的 source 必须全部在清单里，
    # 否则下游 join 会误判「source 不存在」（2026-10-27 @Eval 复核发现 3 个缺口）。
    sp_files = {}
    for sp in ("train", "val", "test"):
        p = out / f"source_split_{sp}.txt"
        sp_files[sp] = {l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                        if l.strip()} if p.exists() else set()
    all_sp = set().union(*sp_files.values()) if sp_files else set()
    seq_csv = out / "sequence_samples.csv"
    if seq_csv.exists():
        seq_src = {r["source"] for r in
                   csv.DictReader(open(seq_csv, encoding="utf-8-sig"))}
        miss = sorted(seq_src - all_sp)
        if miss:
            issues.append(f"{len(miss)} 个序列 source 不在 source_split_*.txt 里: {miss[:10]}")
        else:
            notes.append(f"source_split_*.txt 覆盖完整: {len(seq_src)}/{len(seq_src)} "
                         f"个序列 source 全部在清单内（全集 {len(all_sp)} 个 source）")
    over = []
    sk = sorted(sp_files)
    for i in range(len(sk)):
        for j in range(i + 1, len(sk)):
            over += sorted(sp_files[sk[i]] & sp_files[sk[j]])
    if over:
        issues.append(f"source_split 文件互相重叠 {len(over)}: {over[:10]}")

    # 弃用 4 类标签（kd/sd/hhc/hho）：只能作为上游元数据存在，不得进入训练/评测标签
    notes.append("已弃用的 4 类标签（kd/sd/hhc/hho）现状：taxonomy.drum_class 仅存为上游元数据，"
                 "交付物中不作为标签列；sequence_hits.csv 的 pseudo_label_4class 仅存档，"
                 "校验脚本持续报 WARN，禁止作为训练文本")

    # 本地可听：pack + 候选全集目录
    n_audible = sum(1 for m in manifest if m["wav_local"])
    notes.append(f"本地可听（wav_local 指向的文件真实存在）: {n_audible}/{len(manifest)}")
    stale = [m["id"] for m in manifest if m["wav_local"] and not os.path.exists(m["wav_local"])]
    if stale:
        issues.append(f"{len(stale)} 条 manifest 的 wav_local 指向不存在的文件: {stale[:5]}")

    # 人工标注工作单
    wl_path = out / "annotation_worklist.csv"
    if wl_path.exists():
        wl = list(csv.DictReader(open(wl_path, encoding="utf-8-sig")))
        wl_fams = Counter(r["sound_family"] for r in wl)
        notes.append(f"人工标注工作单: {wl_path} —— {len(wl)} 条 / {len(wl_fams)} 个 family；"
                     f"每 family n: {min(wl_fams.values())}–{max(wl_fams.values())}；"
                     f"达门槛需 ≥15 family 且每 family n≥5")
        miss_audio = [r["id"] for r in wl
                      if not r.get("slice_wav") or not os.path.exists(r["slice_wav"])]
        if miss_audio:
            issues.append(f"工作单中 {len(miss_audio)} 条切片本地不可读: {miss_audio[:5]}")
        miss_ref = [r["id"] for r in wl
                    if not r.get("ref_wav") or not os.path.exists(r["ref_wav"])]
        if miss_ref:
            issues.append(f"工作单中 {len(miss_ref)} 条缺参考音: {miss_ref[:5]}")

    # 空 human_notation
    if human == 0:
        issues.append("human_notation 全空：任何识别准确率在 human_notation 覆盖 >=15 family 前不得写入 §0")

    # 4 类标签残留
    legacy = [m["id"] for m in manifest
              if (m.get("candidate_family_l2") or "") in ("kd", "sd", "hhc", "hho")]
    if legacy:
        issues.append(f"manifest 内出现已弃用 4 类标签 {len(legacy)} 条")

    # 种子库记法误用作标签
    mism = sum(1 for m in manifest if m["candidate_notation_seedlib"]
               and m["candidate_notation_seedlib"] != m["sound_notation"])
    if mism:
        notes.append(f"标签空间错配：{mism}/{len(manifest)} 条 candidate_notation 与 canonical "
                     "记法不同（种子库记法带 '<' 前缀）")

    # canonical 覆盖
    hit = len({m["sound_id"] for m in manifest if m["sound_id"] != ""})
    fams_hit = {m["sound_family"] for m in manifest if m["sound_family"] != "unknown"}
    notes.append(f"408 条候选覆盖 canonical 103 音色中的 {hit} 个 / {len(fams_hit)} 个 family"
                 f"（共 {len({s['family'] for s in taxonomy['sounds']})} 个 family）")

    # 记法碰撞：103 音色只有 97 个唯一记法，纯记法 CE 的天花板
    if amb:
        amb_rows = sum(1 for m in manifest if m["sound_notation"] in amb)
        notes.append(f"记法碰撞：{len(amb)} 组重复记法，"
                     f"{amb_rows} 条候选落在歧义记法上（{sorted(amb)}）")
        # 以 notation 为标签单元时，每组碰撞只有 1 个能判对 -> 准确率天花板
        # 91 个独占记法全对 + 每组碰撞最多对 1 个
        ceiling = 103 - sum(v - 1 for v in amb.values())
        notes.append(f"=> 以 notation 为唯一 CE 标签单元时，103 类准确率天花板 = "
                     f"{ceiling}/103 = {ceiling / 103 * 100:.1f}%"
                     f"（{sum(v - 1 for v in amb.values())} 个音色与他人共享记法，"
                     "每组至多对 1 个）；必须以 (notation, en) 或 sound_id 作为标签单元")

    # 标签空间错配：种子库记法 vs canonical 记法
    seedlib_space = sum(1 for m in manifest if m["candidate_notation_seedlib"]
                        and m["candidate_notation_seedlib"] != m["sound_notation"])
    notes.append(f"标签空间错配：{seedlib_space}/{len(manifest)} 条的 candidate_notation "
                 "（种子库空间，带 '<' 前缀）≠ canonical 记法。"
                 "Talker 训练文本标签必须取 canonical sound_notation")

    # split 分布
    for name, rows in split_files.items():
        notes.append(f"split {name}: {len(rows)} 条 / {len({r['source'] for r in rows})} 个 source")

    return issues, notes


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="data/beatbox_taxonomy.json")
    ap.add_argument("--seedlib", default="data/beatbox_seeds/sounds.json")
    ap.add_argument("--csv", default="data/sound_seed_annotation.csv")
    ap.add_argument("--labels", default="", help="远端 hybrid_labels_all_highconf_l2.jsonl（可选）")
    ap.add_argument("--human-csv", default="data/sound_seed_annotation_human.csv",
                    help="人工标注结果 CSV（由 annotate_ui.py 导出）；存在则合并 human_notation")
    ap.add_argument("--listen-pack", default="data/bbox_raw/sound_seed_pack")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--report", default="docs/data_qc_report.md")
    ap.add_argument("--self-test", action="store_true",
                    help="只跑断言（双射/短变体归位/split 纯函数），不写任何文件")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.taxonomy, args.seedlib)
        return

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    seedlib = json.loads(Path(args.seedlib).read_text(encoding="utf-8"))
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))

    # 合并人工标注（若存在）：以 human_notation / annotate_status / annotator 覆盖
    human_path = Path(args.human_csv) if args.human_csv else None
    if human_path and human_path.exists():
        human_rows = {r["id"]: r for r in
                      csv.DictReader(open(human_path, encoding="utf-8-sig"))}
        merged = 0
        for r in csv_rows:
            h = human_rows.get(r.get("id"))
            if not h:
                continue
            hnot = (h.get("human_notation") or "").strip()
            hstat = (h.get("annotate_status") or "").strip()
            # 「不确定」也是有效的人工判定：annotate_status=unknown 且记法留空时，
            # 必须把状态合并进来，否则这条样本会退回伪标签、永远不算人工确认过
            # （2026-10-27 复核发现：unknown 判定被静默丢弃）。
            if hnot or hstat in ("unknown", "confirmed", "rejected"):
                if hnot:
                    r["human_notation"] = hnot
                r["annotate_status"] = hstat or ("confirmed" if hnot else r.get("annotate_status", ""))
                r["annotator"] = h.get("annotator") or ""
                if (h.get("note") or "").strip():
                    r["note"] = h["note"]
                merged += 1
        print(f"合并人工标注 {merged} 条 <- {human_path}", flush=True)

    labels_by_id = {}
    if args.labels and Path(args.labels).exists():
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id"):
                labels_by_id[r["id"]] = r

    crosswalk, _tax_by_id = build_crosswalk(taxonomy, seedlib)
    crosswalk_by_not = {c["seedlib_notation"]: c for c in crosswalk}

    manifest = build_manifest(csv_rows, crosswalk_by_not, taxonomy, labels_by_id, args.listen_pack)
    coverage = family_coverage(manifest, taxonomy)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cw_path = out / "beatbox_seed_crosswalk.csv"
    with open(cw_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crosswalk[0].keys()))
        w.writeheader()
        w.writerows(crosswalk)

    mf_path = out / "seed_manifest.csv"
    with open(mf_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    cov_path = out / "seed_family_coverage.csv"
    with open(cov_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0].keys()))
        w.writeheader()
        w.writerows(coverage)

    split_files = {}
    for sp in ("train", "val", "test"):
        rows = [m for m in manifest if m["split"] == sp]
        split_files[sp] = rows
        (out / f"seed_split_{sp}.txt").write_text(
            "\n".join(sorted({m["source"] for m in rows if m["source"]})) + "\n", encoding="utf-8")

    # 全 source 宇宙的 split 清单：split_of 是 source 的纯函数（md5），
    # 因此任何新切片只要按同一 BV 算出同一个 split 就自动归位，
    # 不会因为「当时没在清单里」而漂移。新增 source 需并入对应文件。
    universe = set()
    url_file = Path("data/bbox_raw/bilibili_urls.txt")
    if url_file.exists():
        for line in url_file.read_text(encoding="utf-8").splitlines():
            m = re.search(r"(BV[0-9A-Za-z]+)", line)
            if m:
                universe.add(m.group(1))
    meta_file = Path("data/bbox_raw/downloads/metadata.jsonl")
    if meta_file.exists():
        for line in meta_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                universe.add(json.loads(line)["id"])
    universe |= {m["source"] for m in manifest if m["source"]}
    # 实战切片清单里的 source 也必须进宇宙：它们的 BV 不一定出现在 url/metadata 里
    # （例如 new_battles 批次的 3 个 BV），漏掉会让下游 join 误判「source 不在任何
    # split 文件里」（2026-10-27 @Eval 复核发现）。
    for seg in sorted(Path("data/real_bbox").glob("segments_*.jsonl")):
        for line in seg.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                s = json.loads(line).get("source")
            except json.JSONDecodeError:
                continue
            if s:
                universe.add(s)
    for sp in ("train", "val", "test"):
        (out / f"source_split_{sp}.txt").write_text(
            "\n".join(sorted(s for s in universe if split_of(s) == sp)) + "\n", encoding="utf-8")

    # 人耳标注 CSV：保留原 7 列，追加 canonical 线索与标注状态
    ann_path = out / "sound_seed_annotation.csv"
    src_by_id = {m["id"]: m for m in manifest}
    with open(ann_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "family", "candidate_notation", "nearest_seed", "wav",
                    "human_notation", "note", "source", "split",
                    "canonical_notation", "canonical_en", "canonical_family",
                    "sound_id", "crosswalk_method", "label_source", "sound_confidence",
                    "annotate_status", "annotator"])
        for r in csv_rows:
            m = src_by_id.get(r.get("id"), {})
            w.writerow([
                r.get("id"), r.get("family"), r.get("candidate_notation"), r.get("nearest_seed"),
                r.get("wav"), (r.get("human_notation") or ""), (r.get("note") or ""),
                m.get("source", ""), m.get("split", ""),
                m.get("sound_notation", ""), m.get("sound_en", ""), m.get("sound_family", ""),
                m.get("sound_id", ""), m.get("crosswalk_method", ""),
                m.get("label_source", ""), m.get("sound_confidence", 0.0),
                m.get("annotate_status", "pending"), (r.get("annotator") or ""),
            ])

    issues, notes = qc_report(manifest, crosswalk, taxonomy, coverage, split_files, out)

    md = ["# Data QC 报告（103 音色资产）", "",
          f"生成脚本：`scripts/build_seed_manifest.py`（可复算）", "",
          "## 摘要", ""]
    md += [f"- manifest 行数: {len(manifest)}", f"- crosswalk 条目: {len(crosswalk)}",
           f"- 覆盖 canonical 103 音色: {len({m['sound_id'] for m in manifest if m['sound_id'] != ''})}/103",
           f"- label_source 分布: {dict(Counter(m['label_source'] for m in manifest))}",
           f"- split 行数: " + ", ".join(f"{k}={len(v)}" for k, v in split_files.items()), ""]
    md += ["## 阻塞问题（BLOCKER）", ""]
    md += [f"{i + 1}. {x}" for i, x in enumerate(issues)] or ["无"]
    md += ["", "## 事实记录", ""]
    md += [f"- {x}" for x in notes]
    md += ["", "## family 覆盖（canonical 103 口径）", "",
           "| family | taxonomy 音色数 | 有种子 | 无种子 | 覆盖率 | 种子段数 | 观察到的记法 | human 已确认 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in coverage:
        md.append(f"| {r['sound_family']} | {r['taxonomy_sounds']} | {r['sounds_with_seed']} | "
                  f"{r['sounds_without_seed']} | {r['coverage_pct']}% | {r['seed_segments']} | "
                  f"{r['notations_observed'] or '-'} | {r['human_confirmed']} |")
    md += ["", "## 产物", "",
           f"- `{cw_path}`", f"- `{mf_path}`", f"- `{cov_path}`", f"- `{ann_path}`",
           f"- `{out}/seed_split_train.txt|val.txt|test.txt`（408 条种子涉及的 source）",
           f"- `{out}/source_split_train.txt|val.txt|test.txt`（全 source 宇宙，新切片按此归位）",
           f"- `{out}/annotation_worklist.csv`（人工标注工作单，由 scripts/annotate_ui.py 导出）", ""]

    # 序列样本（由 scripts/build_sequence_samples.py 生成，若存在则附摘要）
    seq_csv = out / "sequence_samples.csv"
    seq_contract = out / "sequence_contract.json"
    if seq_csv.exists() and seq_contract.exists():
        sc = json.loads(seq_contract.read_text(encoding="utf-8"))
        md += ["## 序列样本（§10.1 #4）", "",
               f"- 序列数: {sc['rows_sequences']}（3–8s，≥12 击/条），击打数: {sc['rows_hits']}",
               f"- source 数: {sc['sources']}；序列/击打 split 分布: "
               f"{sc['sequences_per_split']} / {sc['hits_per_split']}",
               f"- 跨 split source 泄漏: {sc['cross_split_source_leakage']}；"
               f"跨 source 拼接序列: {sc['sequences_spanning_multiple_sources']}",
               f"- 覆盖 canonical 音色: {sc['distinct_sound_ids']}，"
               f"family: {sc['families_covered']}/{sc['total_families']}",
               f"- 音频可用（远端持久盘）: 序列 {sc['audio_available_sequences']} / "
               f"击打 {sc['audio_available_hits']}",
               f"- label_source 取值: {sc['label_source_values']}（全部伪标签，"
               f"human 确认 {sc['human_confirmed_hits']}）",
               f"- 门槛: {sc['gate']}", "",
               f"- `{seq_csv}`", f"- `{out}/sequence_hits.csv`",
               f"- `{out}/sequence_family_coverage.csv`", f"- `{seq_contract}`", ""]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(md), encoding="utf-8")

    print(f"crosswalk={cw_path} manifest={mf_path} coverage={cov_path} annotation={ann_path}")
    print(f"splits: " + ", ".join(f"{k}={len(v)}" for k, v in split_files.items()))
    print(f"BLOCKERS: {len(issues)}")
    for x in issues:
        print("  !", x)
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
