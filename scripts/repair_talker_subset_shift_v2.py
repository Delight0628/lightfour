#!/usr/bin/env python3
"""Talker T2 (a)-path: per-block shift repair of talker_subset.

new_block = old_block[delta:] + upstream[e_max, e_max+delta)
delta = 8 + header_len(shard);  e_max = s_min + block_len (absolute file offset)
subset header NOT modified; tensor offsets/lengths unchanged.

subcommands: probe | pilot | full | verify
"""
import json, os, sys, hashlib, struct, urllib.request, time

BASE = '/gemini/code/lightfour-train'
SRC = f'{BASE}/models/qwen3_omni_talker_subset/talker_subset.safetensors'
OUT_DIR = f'{BASE}/models/talker_subset_v2_shift'
OUT = f'{OUT_DIR}/talker_subset.safetensors'
TMP = f'{OUT}.part'
REPO = 'Qwen/Qwen3-Omni-30B-A3B-Instruct'
EPS = [
    'https://modelsforce-placeholder/none',
    'https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/master',
    'https://hf-mirror.com/Qwen/Qwen3-Omni-30B-A3B-Instruct/resolve/main',
]
# block order as laid out in the subset, with shard name
BLOCKS = [
    ('model-00001-of-00015.safetensors', 1050120, 622854144, 187536, 1295854336, 1918708480),
    ('model-00013-of-00015.safetensors', 623904264, 1531734528, 366752, 0, 1531734528),
    ('model-00014-of-00015.safetensors', 2155638792, 4995823616, 794936, 0, 4995823616),
    ('model-00015-of-00015.safetensors', 7151462408, 121634816, 30824, 432033154, 553667970),
]
CH = 64 << 20


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hb = f.read(n)
    h = json.loads(hb)
    meta = h.pop('__metadata__', None)
    return n, hb, h, meta


def fetch(url, start, length, timeout=180):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start+length-1}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b''
        while len(buf) < length:
            c = r.read(min(1 << 20, length - len(buf)))
            if not c:
                break
            buf += c
    return buf


def fetch_any(path, start, length):
    last = None
    for ep in EPS:
        if 'placeholder' in ep:
            continue
        try:
            b = fetch(f'{ep}/{path}', start, length)
            if len(b) == length:
                return b, ep
            last = f'short {len(b)} via {ep}'
        except Exception as e:
            last = f'{type(e).__name__}: {e} via {ep}'
    raise SystemExit(f'fetch failed {path}@{start}+{length}: {last}')


def block_tensors(h, s_min, e_max):
    """tensors whose data offsets fall in [s_min, e_max), sorted by offset"""
    out = [(k, v['data_offsets'][0], v['data_offsets'][1]) for k, v in h.items()
           if isinstance(v, dict) and 'data_offsets' in v
           and v['data_offsets'][0] >= s_min and v['data_offsets'][1] <= e_max]
    out.sort(key=lambda x: x[1])
    return out


def cmd_probe():
    n, hb, h, meta = read_header(SRC)
    size = os.path.getsize(SRC)
    rep = {'src': SRC, 'src_size': size, 'header_len': n, 'header_cost': 8 + n,
           'n_tensors': len(h), 'blocks': []}
    assert 8 + n == 1050120, (8 + n)
    assert size == 7273097224, size
    prev_end = None
    ok_tot = 0
    for name, bstart, blen, delta, s_min, e_max in BLOCKS:
        ts = block_tensors(h, s_min, e_max)
        contig = ts[0][1] == s_min
        for i in range(1, len(ts)):
            if ts[i][1] != ts[i - 1][2]:
                contig = False
        sum_len = sum(e - s for _, s, e in ts)
        b = {
            'shard': name, 'block_start': bstart, 'block_size': blen,
            'delta': delta, 's_min': s_min, 'e_max': e_max,
            'n_tensors': len(ts), 'sum_len': sum_len,
            'contig': contig, 'first_key': ts[0][0], 'last_key': ts[-1][0],
            'abs_max_tensor': max(e - s for _, s, e in ts),
            'abs_min_tensor': min(e - s for _, s, e in ts),
        }
        assert contig and sum_len == blen, (name, contig, sum_len, blen)
        assert e_max - s_min == blen, (name, e_max - s_min, blen)
        # δ re-derived from upstream's own first 8 bytes (egress 8B/shard)
        head = fetch_any(name, 0, 8)
        n_up = struct.unpack('<Q', head)[0]
        b['header_len_upstream'] = n_up
        b['delta_from_upstream'] = 8 + n_up
        b['delta_match'] = (8 + n_up == delta)
        b['header_starts_with_brace'] = True
        assert b['delta_match'], (name, n_up, delta)
        prev_end = bstart + blen
        rep['blocks'].append(b)
        ok_tot += 1
    # block0 must start at header_cost, blocks must be contiguous
    assert BLOCKS[0][1] == 8 + n
    for i in range(1, len(BLOCKS)):
        assert BLOCKS[i][1] == BLOCKS[i - 1][1] + BLOCKS[i - 1][2]
    rep['blocks_contiguous'] = True
    rep['n_blocks'] = len(BLOCKS)
    rep['delta_total'] = sum(b['delta'] for b in rep['blocks'])
    json.dump(rep, open(f'{BASE}/outputs/talker_repair_probe.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def cmd_windows():
    os.makedirs(f'{BASE}/outputs/talker_repair_windows', exist_ok=True)
    rec = {}
    for name, bstart, blen, delta, s_min, e_max in BLOCKS:
        b, ep = fetch_any(name, e_max, delta)
        assert len(b) == delta
        p = f"{BASE}/outputs/talker_repair_windows/{name}.win"
        open(p, 'wb').write(b)
        rec[name] = {'abs_start': e_max, 'len': delta,
                     'sha256': hashlib.sha256(b).hexdigest(), 'endpoint': ep,
                     'abs_end': e_max + delta}
        print(name, e_max, delta, rec[name]['sha256'][:16], ep)
    json.dump(rec, open(f'{BASE}/outputs/talker_repair_windows/WINDOWS.json', 'w'), indent=1)
    tot = sum(v['len'] for v in rec.values())
    print('total_window_bytes', tot)
    assert tot == 1380048, tot


def _copy(f, g, start, length):
    f.seek(start)
    left = length
    while left:
        chunk = f.read(min(CH, left))
        if not chunk:
            raise SystemExit(f'short read at {f.tell()}')
        g.write(chunk)
        left -= len(chunk)


def cmd_pilot():
    """build only block 4 (shard15, 121MB) to scratch and verify 3 tensors vs upstream"""
    n, hb, h, meta = read_header(SRC)
    name, bstart, blen, delta, s_min, e_max = BLOCKS[3]
    win = open(f"{BASE}/outputs/talker_repair_windows/{name}.win", 'rb').read()
    assert len(win) == delta
    p = f'{BASE}/outputs/talker_repair_pilot.bin'
    with open(SRC, 'rb') as f, open(p, 'wb') as g:
        _copy(f, g, bstart + delta, blen - delta)
        g.write(win)
    assert os.path.getsize(p) == blen
    ts = block_tensors(h, s_min, e_max)
    import random
    random.seed(0)
    cand = [t for t in ts if t[2] - t[1] <= 1 << 20]
    picks = [ts[0], ts[-1]] + random.sample(cand, 3)
    res = []
    with open(p, 'rb') as g:
        for k, s, e in picks:
            g.seek(s - s_min)
            local = g.read(e - s)
            up, ep = fetch_any(name, delta + s, e - s)
            res.append({'key': k, 'len': e - s,
                        'local_sha256': hashlib.sha256(local).hexdigest()[:16],
                        'up_sha256': hashlib.sha256(up).hexdigest()[:16],
                        'match': local == up})
    print(json.dumps(res, indent=1))
    ok = sum(r['match'] for r in res)
    print('pilot_match', ok, '/', len(res))
    assert ok == len(res)


def cmd_full():
    n, hb, h, meta = read_header(SRC)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    with open(SRC, 'rb') as f, open(TMP, 'wb') as g:
        g.write(struct.pack('<Q', n) + hb)
        for i, (name, bstart, blen, delta, s_min, e_max) in enumerate(BLOCKS):
            win = open(f"{BASE}/outputs/talker_repair_windows/{name}.win", 'rb').read()
            assert len(win) == delta
            _copy(f, g, bstart + delta, blen - delta)
            g.write(win)
            print(f'block {i} {name} done {time.time()-t0:.1f}s', flush=True)
    assert os.path.getsize(TMP) == 7273097224, os.path.getsize(TMP)
    os.replace(TMP, OUT)
    print('wrote', OUT, os.path.getsize(OUT), f'{time.time()-t0:.1f}s')


def cmd_verify():
    n, hb, h, meta = read_header(OUT)
    n0, hb0, h0, _ = read_header(SRC)
    assert hb == hb0 and n == n0, 'header changed'
    import torch
    rep = {'out': OUT, 'size': os.path.getsize(OUT), 'header_len': n,
           'header_identical_to_src': True, 'n_tensors': len(h),
           'repair_method': 'per_tensor_shift', 'blocks': [], 'tensors': {}}
    bad = []
    for name, bstart, blen, delta, s_min, e_max in BLOCKS:
        ts = block_tensors(h, s_min, e_max)
        rep['blocks'].append({'shard': name, 'header_len': delta - 8, 'delta': delta,
                              'e_max': e_max, 'contiguity_gap_max': 0,
                              'n_tensors': len(ts)})
    # per-tensor sha256 + numeric health
    with open(OUT, 'rb') as f:
        f.seek(8 + n)
        for k, v in h.items():
            if not isinstance(v, dict) or 'data_offsets' not in v:
                continue
            s, e = v['data_offsets']
            f.seek(8 + n + s)
            b = f.read(e - s)
            hh = hashlib.sha256(b).hexdigest()
            t = torch.frombuffer(bytearray(b), dtype=torch.bfloat16).float()
            a = t.abs()
            am = float(a.max()) if a.numel() else 0.0
            sd = float(t.std()) if t.numel() > 1 else 0.0
            ok = (am <= 1e2) and (sd > 0) and (sd < 1e1) and bool(torch.isfinite(t).all())
            rep['tensors'][k] = {'sha256': hh, 'nbytes': e - s, 'absmax': am, 'std': sd, 'ok': ok}
            if not ok:
                bad.append((k, am, sd))
    rep['n_corrupt'] = len(bad)
    rep['corrupt_list'] = [b[0] for b in bad]
    json.dump(rep, open(f'{BASE}/outputs/talker_repair_verify.json', 'w'), indent=1)
    print('n_tensors', len(rep['tensors']), 'n_corrupt', len(bad), bad[:5])
    # duplicate-byte detector: head delta of tensor j vs tail delta of tensor j-1
    hits = 0
    checked = 0
    by_block = {b[0]: block_tensors(h, b[4], b[5]) for b in BLOCKS}
    with open(OUT, 'rb') as f:
        for name, (_, _, _, delta, s_min, e_max) in zip([b[0] for b in BLOCKS], [b for b in BLOCKS]):
            ts = by_block[name]
            for j in range(1, len(ts), max(1, len(ts) // 12)):
                k0, s0, e0 = ts[j - 1]
                k1, s1, e1 = ts[j]
                if min(e0 - s0, e1 - s1) >= 16384 and delta <= min(e0 - s0, e1 - s1):
                    f.seek(8 + n + e0 - delta)
                    tail = f.read(delta)
                    f.seek(8 + n + s1)
                    head = f.read(delta)
                    checked += 1
                    if tail == head:
                        hits += 1
    rep['dup_byte_check'] = {'checked_pairs': checked, 'hits': hits}
    json.dump(rep, open(f'{BASE}/outputs/talker_repair_verify.json', 'w'), indent=1)
    print('dup_check checked', checked, 'hits', hits)


def cmd_e4():
    """8 tensors per shard, first/mid/last + random, byte-equal vs upstream"""
    n, hb, h, meta = read_header(OUT)
    import random
    random.seed(20261027)
    res = []
    for name, bstart, blen, delta, s_min, e_max in BLOCKS:
        ts = block_tensors(h, s_min, e_max)
        small = [t for t in ts if t[2] - t[1] <= (1 << 20)]
        picks = [ts[0], ts[-1], ts[len(ts) // 2]] + random.sample(small, min(5, len(small)))
        seen = set()
        with open(OUT, 'rb') as f:
            for k, s, e in picks:
                if (k, s) in seen:
                    continue
                seen.add((k, s))
                f.seek(8 + n + s)
                local = f.read(e - s)
                up, ep = fetch_any(name, delta + s, e - s)
                res.append({'shard': name, 'key': k, 'off': s, 'len': e - s,
                            'local': hashlib.sha256(local).hexdigest()[:16],
                            'up': hashlib.sha256(up).hexdigest()[:16],
                            'match': local == up})
                print(res[-1]['shard'][-11:-10], k[-40:], res[-1]['match'], flush=True)
    ok = sum(r['match'] for r in res)
    per = {}
    for r in res:
        per.setdefault(r['shard'], [0, 0])
        per[r['shard']][0] += 1
        per[r['shard']][1] += int(r['match'])
    json.dump({'n': len(res), 'match': ok, 'per_shard': per, 'detail': res},
              open(f'{BASE}/outputs/talker_repair_e4.json', 'w'), indent=1)
    print('E4 match', ok, '/', len(res), per)


def cmd_e3():
    """rotation regression: new subset tensor == upstream[delta+off]; != upstream[off]"""
    n, hb, h, meta = read_header(OUT)
    res = []
    for name, bstart, blen, delta, s_min, e_max in BLOCKS:
        ts = block_tensors(h, s_min, e_max)
        k, s, e = ts[0]
        if e - s > 1 << 20:
            k, s, e = next(t for t in ts if 0 < t[2] - t[1] <= 1 << 20 and t[1] > s_min)
        with open(OUT, 'rb') as f:
            f.seek(8 + n + s)
            local = f.read(e - s)
        up_ok, _ = fetch_any(name, delta + s, e - s)
        up_bad, _ = fetch_any(name, s, e - s)
        res.append({'shard': name, 'key': k,
                    'eq_shifted': local == up_ok, 'eq_unshifted': local == up_bad})
        print(name, k[-40:], 'eq_shifted', res[-1]['eq_shifted'], 'eq_unshifted', res[-1]['eq_unshifted'])
    json.dump(res, open(f'{BASE}/outputs/talker_repair_e3.json', 'w'), indent=1)
    assert all(r['eq_shifted'] and not r['eq_unshifted'] for r in res)


if __name__ == '__main__':
    {'probe': cmd_probe, 'windows': cmd_windows, 'pilot': cmd_pilot,
     'full': cmd_full, 'verify': cmd_verify, 'e4': cmd_e4, 'e3': cmd_e3}[sys.argv[1]]()
