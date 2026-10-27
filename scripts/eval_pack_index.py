"""
重建可追溯试听包：ASCII 安全文件名 + manifest 映射 + join 自检。

解决的问题：
  旧试听包把 notation 直接拼进文件名，'<'、'*' 被吞、emoji/变音符乱码，
  导致 eval_report.json 的 name 无法 join 到磁盘 wav，人耳 A/B 不可追溯。

做法：
  1. 以三位序号 (idx) 为稳定主键，把 report 行、gen_manifest 行、磁盘 wav 对齐；
  2. 新文件名 = f"{idx3}_{sha1(notation)[:8]}_{gen|gt}.wav"，纯 ASCII；
  3. 产出 manifest.json（notation/原始名/新文件名/校验和全映射）与
     eval_report_indexed.json（指标行自带 id/gen_file/gt_file）；
  4. 自检 report->文件名 join 命中率、gen/gt 缺失、孤儿文件、wav 头可读，
     任何一项不达标即 exit code 1。

用法：
  python scripts/eval_pack_index.py --src data/bbox_raw/tiny_codec_gen \
      --out data/bbox_raw/tiny_codec_gen/ab_pack_v2
"""
import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import wave
from pathlib import Path

IDX_RE = re.compile(r"^(\d{3})[_-]")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_info(p: Path) -> dict:
    """校验 wav 头可读并返回时长；损坏文件记 error 而不是抛异常。"""
    try:
        with wave.open(str(p), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            return {
                "ok": True,
                "sr": sr,
                "channels": w.getnchannels(),
                "frames": n,
                "dur_s": round(n / sr, 4) if sr else None,
            }
    except Exception as e:  # noqa: BLE001 - 自检要把错误写进报告
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def index_of(name: str):
    m = IDX_RE.match(name)
    return m.group(1) if m else None


def audio_tag(name):
    if not name:
        return '<span style="color:#f85149">MISSING</span>'
    return f'<audio controls preload="none" src="{html.escape(name)}"></audio>'


def build_html(rows, out_html: Path, title: str) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>body{font:14px/1.6 system-ui,Microsoft YaHei,sans-serif;background:#0f1115;"
        "color:#e6e8eb;padding:24px 28px 60px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #272b33;padding:8px 10px;vertical-align:middle}"
        "th{background:#181b21;text-align:left}code{color:#79c0ff}"
        "tr.fail td{background:#2b1618}.tag{font-size:12px;color:#8b929e}</style>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        '<p class="tag">gen/gt 成对盲听；notation 见 manifest.json。文件名纯 ASCII，'
        "report→文件名 join 以 manifest.json 为准。</p>",
        "<table><tr><th>id</th><th>notation</th><th>gen</th><th>gt</th>"
        "<th>mel</th><th>env</th><th>onset(ms)</th></tr>",
    ]
    for r in rows:
        cls = ' class="fail"' if r.get("mel_dist") is not None and r["mel_dist"] > 2.0 else ""
        parts.append(
            f"<tr{cls}><td><code>{html.escape(r['id'])}</code></td>"
            f"<td>{html.escape(r['notation'])}</td>"
            f'<td>{audio_tag(r.get("gen_file"))}</td>'
            f'<td>{audio_tag(r.get("gt_file"))}</td>'
            f"<td>{r.get('mel_dist')}</td><td>{r.get('env_corr')}</td>"
            f"<td>{r.get('onset_err_ms')}</td></tr>"
        )
    parts += ["</table></body></html>"]
    out_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="旧试听包目录（含 *_gen.wav/_gt.wav/eval_report.json）")
    ap.add_argument("--out", required=True, help="新试听包输出目录")
    ap.add_argument("--title", default="Lightfour 生成试听包（可追溯版）")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = json.loads((src / "eval_report.json").read_text(encoding="utf-8"))
    mani_src = json.loads((src / "gen_manifest.json").read_text(encoding="utf-8"))
    notation_by_idx = {str(r["idx"]).zfill(3): r["notation"] for r in mani_src}

    # 1) 以 idx 为主键对齐磁盘 wav
    wav_by_idx = {}
    orphans = []
    for p in sorted(src.glob("*.wav")):
        idx = index_of(p.name)
        if idx is None:
            orphans.append(p.name)
            continue
        # 磁盘名可能是乱码（UTF-8 被按 GBK 解读），'_' 分隔符可能已并入乱码字符，
        # 因此用结尾的 gen|gt 判类型，不依赖精确的 "_gen"/"_gt" 后缀。
        m2 = re.search(r"(gen|gt)\.wav$", p.name)
        kind = m2.group(1) if m2 else None
        if kind is None:
            orphans.append(p.name)
            continue
        wav_by_idx.setdefault(idx, {})[kind] = p

    report_by_idx = {}
    rows_out, manifest = [], []
    missing_report_key, dup_report_key = [], []
    problems = []

    for rec in report:
        name = rec.get("name", "")
        idx = index_of(name)
        if idx is None:
            missing_report_key.append(name)
            continue
        if idx in report_by_idx:
            dup_report_key.append(name)
            continue
        report_by_idx[idx] = rec

        pair = wav_by_idx.get(idx, {})
        src_gen, src_gt = pair.get("gen"), pair.get("gt")
        notation = notation_by_idx.get(idx)
        if notation is None:
            problems.append(f"idx {idx}: gen_manifest 无对应 notation")
            notation = name
        else:
            # report.name 应等于 f"{idx}_{notation}"，核对防止错位
            if name not in (f"{idx}_{notation}", f"{idx}__{notation}"):
                problems.append(
                    f"idx {idx}: report.name={name!r} 与 manifest notation={notation!r} 不一致"
                )

        h8 = sha1_text(notation)[:8]
        gen_file = f"{idx}_{h8}_gen.wav"
        gt_file = f"{idx}_{h8}_gt.wav"

        entry = {
            "id": f"{idx}_{h8}",
            "index": idx,
            "notation": notation,
            "notation_sha1_8": h8,
            "report_name": name,
            "source_gen": src_gen.name if src_gen else None,
            "source_gt": src_gt.name if src_gt else None,
            "gen_file": gen_file if src_gen else None,
            "gt_file": gt_file if src_gt else None,
        }
        for kind, src_p, dst_name in (("gen", src_gen, gen_file), ("gt", src_gt, gt_file)):
            if src_p is None:
                entry[f"{kind}_error"] = "源文件缺失"
                continue
            shutil.copy2(src_p, out / dst_name)
            entry[f"{kind}_sha1"] = sha1_file(src_p)
            entry[f"{kind}_wav"] = wav_info(out / dst_name)
        manifest.append(entry)

        row = dict(rec)
        row["id"] = entry["id"]
        row["notation"] = notation
        row["gen_file"] = entry["gen_file"]
        row["gt_file"] = entry["gt_file"]
        rows_out.append(row)

    # 2) 自检
    orphan_out = [
        p.name
        for p in sorted(out.glob("*.wav"))
        if p.name not in {m["gen_file"] for m in manifest if m["gen_file"]} | {m["gt_file"] for m in manifest if m["gt_file"]}
    ]
    missing_gen = [m["id"] for m in manifest if not m.get("gen_file") or not (out / m["gen_file"]).exists()]
    missing_gt = [m["id"] for m in manifest if not m.get("gt_file") or not (out / m["gt_file"]).exists()]
    wav_bad = [
        f"{m['id']}:{k}" for m in manifest for k in ("gen", "gt")
        if m.get(f"{k}_wav") and not m[f"{k}_wav"].get("ok")
    ]
    joined = [m for m in manifest if m.get("gen_file") and m.get("gt_file")
              and (out / m["gen_file"]).exists() and (out / m["gt_file"]).exists()]
    hit = len(joined) / len(report) if report else 0.0

    verify = {
        "src": str(src),
        "out": str(out),
        "n_report_rows": len(report),
        "n_manifest_rows": len(manifest),
        "n_pairs_joined": len(joined),
        "join_hit_rate": round(hit, 4),
        "report_name_unparseable": missing_report_key,
        "report_name_duplicate": dup_report_key,
        "missing_gen": missing_gen,
        "missing_gt": missing_gt,
        "orphan_files_in_src": orphans,
        "orphan_files_in_out": orphan_out,
        "wav_header_errors": wav_bad,
        "consistency_problems": problems,
    }
    verify["pass"] = (
        hit >= 1.0
        and not missing_gen and not missing_gt and not wav_bad
        and not orphan_out and not missing_report_key and not dup_report_key
        and not problems
    )

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "eval_report_indexed.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "VERIFY.json").write_text(
        json.dumps(verify, ensure_ascii=False, indent=1), encoding="utf-8")
    build_html(rows_out, out / "index.html", args.title)

    print(f"report rows      : {verify['n_report_rows']}")
    print(f"pairs joined     : {verify['n_pairs_joined']}")
    print(f"join hit rate    : {verify['join_hit_rate']:.1%}")
    print(f"missing gen/gt   : {len(missing_gen)}/{len(missing_gt)}")
    print(f"wav header errors: {len(wav_bad)}")
    print(f"orphans src/out  : {len(orphans)}/{len(orphan_out)}")
    print(f"-> {out / 'manifest.json'}")
    print(f"-> {out / 'eval_report_indexed.json'}")
    print(f"-> {out / 'index.html'}")
    print("PASS" if verify["pass"] else "FAIL")
    return 0 if verify["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
