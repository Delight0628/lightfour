"""Resolve per-hit audio availability for the 3-8s sequence dataset.

Problem this fixes
------------------
`Data/sequence_hits.csv` / `Data/sequence_samples.csv` set `audio_available`
by checking only `data/real_bbox/segments_batch3/<source>/<hit_id>.wav`.
A large part of the beatbox slices is stored flat instead, in
`data/real_bbox/hybrid_highconf_0.7[_78v]/<hit_id>.wav`
(see `docs/data_qc_report.md`), so the flag under-reports availability.

This script resolves every hit against the three real locations plus the token
sidecar and writes a corrected table. It never modifies the input CSVs.

Evidence inputs
---------------
Listings must be produced on the training server (audio is not in git), e.g.

    cd /gemini/code/lightfour-train/data/real_bbox
    ls hybrid_highconf_0.7_78v/*.wav | sed 's#.*/##' > /tmp/flat.txt
    ls /gemini/code/lightfour-train/data/seq_tokens/hit_codes/*.npz \
        | sed 's#.*/##' > /tmp/npz.txt

then pulled locally and passed via --flat-listing / --token-listing.

Usage
-----
    python scripts/qc_sequence_audio.py \
        --flat-listing tmp/_flat_listing.txt \
        --token-listing tmp/_npz_listing.txt

Outputs (default --out-dir Data):
    sequence_audio_resolution.csv   one row per hit
    sequence_availability_v2.csv    one row per sequence
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_listing(path: str, suffix: str) -> set[str]:
    names = set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for line in fh:
            name = line.strip()
            if not name:
                continue
            if not name.endswith(suffix):
                continue
            names.add(name[: -len(suffix)])
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default="Data/sequence_samples.csv")
    ap.add_argument("--hits", default="Data/sequence_hits.csv")
    ap.add_argument("--segment-root", default="data/real_bbox/segments_batch3")
    ap.add_argument("--segment-listing", required=True,
                    help="one .wav basename per line, from segments_batch3/*/")
    ap.add_argument("--flat-listing", required=True,
                    help="one .wav basename per line, from hybrid_highconf_0.7*")
    ap.add_argument("--token-listing", required=True,
                    help="one .npz basename per line, from data/seq_tokens/hit_codes")
    ap.add_argument("--out-dir", default="Data")
    args = ap.parse_args()

    samples = read_csv(args.samples)
    hits = read_csv(args.hits)
    segs = read_listing(args.segment_listing, ".wav")
    flat = read_listing(args.flat_listing, ".wav")
    toks = read_listing(args.token_listing, ".npz")

    seg_root = args.segment_root.rstrip("/\\")
    rows = []
    for h in hits:
        hid = h["hit_id"]
        seg_path = f"{seg_root}/{h['source']}/{hid}.wav"
        seg_found = os.path.exists(seg_path) or hid in segs
        flat_hit = hid in flat
        token_hit = hid in toks
        if seg_found:
            basis, path = "segment_batch3", seg_path
        elif flat_hit:
            basis, path = "remote_flat", f"data/real_bbox/hybrid_highconf_0.7*/{hid}.wav"
        else:
            basis, path = "missing", ""
        rows.append({
            "hit_id": hid,
            "source": h["source"],
            "split": h["split"],
            "sound_family": h["sound_family"],
            "segment_batch3_hit": int(seg_found),
            "flat_listing_hit": int(flat_hit),
            "token_listing_hit": int(token_hit),
            "audio_available_v2": int(bool(path)),
            "basis": basis,
            "audio_path": path,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, "sequence_audio_resolution.csv")
    with open(res_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_seq: dict[str, list[dict]] = defaultdict(list)
    seq_for_hit = {h["hit_id"]: h["sequence_id"] for h in hits}
    for r in rows:
        by_seq[seq_for_hit[r["hit_id"]]].append(r)

    seq_rows = []
    for s in samples:
        rs = by_seq.get(s["sequence_id"], [])
        n_audio = sum(x["audio_available_v2"] for x in rs)
        miss = [x["hit_id"] for x in rs if not x["audio_available_v2"]]
        seq_rows.append({
            "sequence_id": s["sequence_id"],
            "source": s["source"],
            "split": s["split"],
            "n_hits": len(rs),
            "n_hits_audio": n_audio,
            "n_hits_token": sum(x["token_listing_hit"] for x in rs),
            "audio_available_v2": int(n_audio == len(rs) and len(rs) > 0),
            "audio_available_v1": int(s["audio_available"]),
            "missing_hit_ids": "|".join(miss),
        })

    seq_path = os.path.join(args.out_dir, "sequence_availability_v2.csv")
    with open(seq_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq_rows[0].keys()))
        w.writeheader()
        w.writerows(seq_rows)

    v1_zero = [s for s in seq_rows if s["audio_available_v1"] == 0]
    flips = [s for s in v1_zero if s["audio_available_v2"] == 1]
    still = [s for s in v1_zero if s["audio_available_v2"] == 0]
    print(f"hits            : {len(rows)}")
    print(f"  segments_batch3: {sum(r['segment_batch3_hit'] for r in rows)}")
    print(f"  flat listing  : {sum(r['flat_listing_hit'] for r in rows)}")
    print(f"  token listing : {sum(r['token_listing_hit'] for r in rows)}")
    print(f"  no audio      : {sum(1 for r in rows if not r['audio_available_v2'])}")
    print(f"sequences       : {len(seq_rows)}")
    print(f"  v1 available  : {sum(s['audio_available_v1'] for s in seq_rows)}")
    print(f"  v2 available  : {sum(s['audio_available_v2'] for s in seq_rows)}")
    print(f"  v1=0 -> v2=1  : {len(flips)} (flag was wrong)")
    print(f"  v1=0 -> v2=0  : {len(still)} (genuinely missing)")
    print("  v2 available by split:",
          dict(Counter(s["split"] for s in seq_rows if s["audio_available_v2"])))
    print("  still-missing sources:",
          dict(Counter(s["source"] for s in still).most_common()))
    print("  still-missing by split:",
          dict(Counter(s["split"] for s in still)))
    print("  still-missing by family:",
          dict(Counter(r["sound_family"] for r in rows
                       if not r["audio_available_v2"]).most_common()))
    print(f"wrote {res_path}")
    print(f"wrote {seq_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
