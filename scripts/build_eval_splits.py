"""
固定评测集与切分契约（供 @Eval 识别侧基线使用）。

产出（全部本地可复算，零依赖）:
  data/label_vocab_covered.csv       canonical 103 体系下、当前有种子覆盖的 34 个音色的标签词表
  data/eval_testset.csv              固定测试集：source 级隔离的 test 段（46 条 / 9 个 source）
  data/eval_folds_leave_notation.csv 留一记法（leave-one-notation-out）折叠定义
  data/eval_perturbations.csv        音频扰动协议（时间拉伸 / 加噪 / 换 seed 上下文）
  data/split_contract.json           split 契约（函数定义 + md5 规则 + 泄漏断言结果）

口径铁律:
  - split 一律按 source(BV) 的 md5 分组；同一 BV 的所有切片必须落在同一 split。
  - 标签用 canonical 103 体系（sound_id / sound_notation / sound_en / sound_family），
    禁止使用种子库原始记法（带 '<' 前缀，与 canonical 不是同一空间）。
  - 测试集在人工标注完成前只能做「管道自检」，不得作为识别准确率证据。
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def split_of(source):
    h = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/seed_manifest.csv")
    ap.add_argument("--taxonomy", default="data/beatbox_taxonomy.json")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    sounds = {i: s for i, s in enumerate(taxonomy["sounds"])}
    manifest = list(csv.DictReader(open(args.manifest, encoding="utf-8")))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. 标签词表：canonical 103 体系下有种子覆盖的音色
    by_sound = defaultdict(list)
    for m in manifest:
        if m["sound_id"] != "":
            by_sound[int(m["sound_id"])].append(m)

    vocab = []
    for sid in sorted(by_sound):
        s = sounds[sid]
        ms = by_sound[sid]
        vocab.append({
            "sound_id": sid,
            "sound_notation": s["notation"],
            "sound_en": s["en"],
            "sound_cn": s["cn"],
            "sound_family": s["family"],
            "mechanism": s.get("mechanism", ""),
            "airflow": s.get("airflow", ""),
            "seed_count": len(ms),
            "is_short_variant": int(bool(s.get("is_short_variant"))),
            "train": sum(1 for m in ms if m["split"] == "train"),
            "val": sum(1 for m in ms if m["split"] == "val"),
            "test": sum(1 for m in ms if m["split"] == "test"),
            "human_confirmed": sum(1 for m in ms if m["label_source"] == "human"),
            "notation_ambiguous": int(sum(1 for x in taxonomy["sounds"]
                                          if x["notation"] == s["notation"]) > 1),
        })
    with open(out / "label_vocab_covered.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vocab[0].keys()))
        w.writeheader()
        w.writerows(vocab)

    # ---- 2. 固定测试集（source 级隔离）
    test = [m for m in manifest if m["split"] == "test"]
    test.sort(key=lambda m: (m["sound_family"], m["id"]))
    with open(out / "eval_testset.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test[0].keys()))
        w.writeheader()
        w.writerows(test)

    # ---- 3. 留一记法折叠：以 canonical sound_id 为留出单位
    folds = []
    for sid in sorted(by_sound):
        s = sounds[sid]
        held = by_sound[sid]
        rest = [m for m in manifest if m["sound_id"] != sid]
        folds.append({
            "fold_id": f"LOO-{sid:03d}",
            "held_out_sound_id": sid,
            "held_out_notation": s["notation"],
            "held_out_en": s["en"],
            "held_out_family": s["family"],
            "held_out_count": len(held),
            "held_out_splits": "|".join(sorted({m["split"] for m in held})),
            "train_size": len(rest),
            "test_splits": "test",
            "note": "留出该音色全部片段；其余音色仍参与训练；跨 source 泄漏由 split 契约保证",
        })
    with open(out / "eval_folds_leave_notation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(folds[0].keys()))
        w.writeheader()
        w.writerows(folds)

    # ---- 4. 扰动协议
    perturbations = [
        {"perturb_id": "P0", "kind": "none", "params": "", "purpose": "基线", "expected": "准确率最高基准"},
        {"perturb_id": "P1", "kind": "time_stretch", "params": "rate=0.9,1.1",
         "purpose": "音频敏感性：时长不变时模型应仍正确", "expected": "准确率下降应 <10 个百分点"},
        {"perturb_id": "P2", "kind": "time_stretch", "params": "rate=0.8,1.25",
         "purpose": "音频敏感性：强扰动", "expected": "准确率显著下降但不应塌到随机"},
        {"perturb_id": "P3", "kind": "additive_noise", "params": "snr=20dB",
         "purpose": "音频敏感性：轻噪", "expected": "轻微下降"},
        {"perturb_id": "P4", "kind": "additive_noise", "params": "snr=10dB,0dB",
         "purpose": "音频敏感性：重噪（0dB 必测）", "expected": "0dB 时准确率应大幅下降"},
        {"perturb_id": "P5", "kind": "seed_swap", "params": "同 family 另一 source 的同类片段",
         "purpose": "跨 source 泛化：换上下文不应改变音色判定", "expected": "准确率不应显著下降"},
        {"perturb_id": "P6", "kind": "silence_pad", "params": "前后各加 0.5s 静音",
         "purpose": "切分鲁棒性", "expected": "准确率基本不变"},
        {"perturb_id": "P7", "kind": "gain", "params": "x0.5, x2.0",
         "purpose": "音量不变性", "expected": "准确率基本不变"},
    ]
    with open(out / "eval_perturbations.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perturbations[0].keys()))
        w.writeheader()
        w.writerows(perturbations)

    # ---- 5. split 契约 + 泄漏断言
    by_split_src = defaultdict(set)
    for m in manifest:
        by_split_src[m["split"]].add(m["source"])
    leaks = []
    keys = sorted(by_split_src)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            leaks += sorted(by_split_src[keys[i]] & by_split_src[keys[j]])

    dup_within = {m["id"]: m["split"] for m in manifest}
    contract = {
        "split_function": "md5(str(source).encode()).hexdigest()[:8] as int % 100 -> "
                          "<80 train, <90 val, else test",
        "group_key": "source (bilibili BV id) —— 同一 BV 的所有切片必须同 split",
        "generated_by": "scripts/build_eval_splits.py",
        "manifest": args.manifest,
        "rows": len(manifest),
        "rows_per_split": dict(Counter(m["split"] for m in manifest)),
        "sources_per_split": {k: len(v) for k, v in sorted(by_split_src.items())},
        "cross_split_source_leakage": len(leaks),
        "leaked_sources": leaks,
        "unique_ids": len(dup_within),
        "duplicate_ids": len(manifest) - len(dup_within),
        "label_space": "canonical 103 (data/beatbox_taxonomy.json)",
        "covered_sounds": len(by_sound),
        "total_sounds": len(sounds),
        "covered_families": len({sounds[sid]["family"] for sid in by_sound}),
        "total_families": len({s["family"] for s in taxonomy["sounds"]}),
        "human_confirmed_rows": sum(1 for m in manifest if m["label_source"] == "human"),
        "gate": "human_notation 覆盖 >=15 family 前，识别准确率不得写入 agent.md §0",
    }
    (out / "split_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 6. 参考音频测试集：95 条种子库 mp3 -> canonical 103 标签
    # 这是当前唯一「零伪标」的固定测试集：标签来自种子库官方命名的 crosswalk，
    # 不依赖任何模型伪标，可直接用于检验识别模型是否在听音频。
    # 局限：录音棚/教学环境，与 B 站实战切片域不同；只能作 sanity check + 上限探针。
    ref_dir = Path("data/beatbox_seeds/audio")
    if ref_dir.exists():
        seedlib = json.loads(Path("data/beatbox_seeds/sounds.json").read_text(encoding="utf-8"))
        cw = {c["seedlib_notation"]: c for c in csv.DictReader(
            open(out / "beatbox_seed_crosswalk.csv", encoding="utf-8"))}
        ref = []
        for f in sorted(ref_dir.glob("*.mp3")):
            m = re.match(r"^(\d+)_", f.stem)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if idx >= len(seedlib):
                continue
            e = seedlib[idx]
            c = cw.get(e["notation"])
            if not c or not c["canonical_sound_id"]:
                continue
            ref.append({
                "ref_id": f.stem,
                "audio": str(f),
                "sound_id": int(c["canonical_sound_id"]),
                "sound_notation": c["canonical_notation"],
                "sound_en": c["canonical_en"],
                "sound_cn": c["canonical_cn"],
                "sound_family": c["canonical_family"],
                "mechanism": c["canonical_mechanism"],
                "airflow": c["canonical_airflow"],
                "seedlib_name": e["name"],
                "seedlib_notation": e["notation"],
                "is_short_variant": int(bool(sounds[int(c["canonical_sound_id"])]
                                             .get("is_short_variant"))),
                "label_source": "seedlib_reference",
                "sound_confidence": 1.0,
                "split": "test",
                "note": "零伪标参考集；域不同于 B 站实战切片",
            })
        with open(out / "eval_refaudio_testset.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ref[0].keys()))
            w.writeheader()
            w.writerows(ref)
        contract["refaudio_testset"] = {
            "rows": len(ref),
            "sounds": len({r["sound_id"] for r in ref}),
            "families": len({r["sound_family"] for r in ref}),
            "label_source": "seedlib_reference（零伪标）",
        }
        (out / "split_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"refaudio testset: {len(ref)} rows / "
              f"{len({r['sound_id'] for r in ref})} sounds / "
              f"{len({r['sound_family'] for r in ref})} families")

    print(json.dumps(contract, ensure_ascii=False, indent=1))
    print(f"-> {out}/label_vocab_covered.csv  eval_testset.csv  "
          f"eval_folds_leave_notation.csv  eval_perturbations.csv  split_contract.json")


if __name__ == "__main__":
    main()
