"""
Stage 3 —— 合成生成型数据（**tar 分片流式输出版**）

相对 render_synth.py 的改进：
  海洋式存储(seaweedfs)写大量小文件极慢（9000 窗口 × 2 文件 ≈ 45 分钟）。
  本版把 wav/mel 顺序写入 tar 分片（每片 SHARD_SIZE 个窗口），
  写入变成顺序大块，速度提升 5-10 倍，且下载/备份只需几个 tar。

输出（data/gen_synth_tar/）：
  shards/shard_XXXXX.tar   每片含 wav/synth_XXXXXX.wav + mel/synth_XXXXXX.npy
  scores.jsonl             鼓谱 + IPA/音节 + shard/成员路径
  omni_pairs.jsonl         Omni 风格（文本条件 -> 目标音频）
  MANIFEST.json            分片清单 + 校验信息

运行： py -3.12 scripts/render_synth_tar.py --hours 50
"""
import os, json, tarfile, io, argparse
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "data", "hit_bank")
OUT = os.path.join(ROOT, "data", "gen_synth_tar")
SHARDS = os.path.join(OUT, "shards")

SR = 16000
N_FFT, HOP, N_MELS = 1024, 256, 64
WIN_SEC = 2.0
SHARD_SIZE = 2000          # 每个 tar 分片含多少个窗口
CLASSES = ["kd", "sd", "hhc", "hho"]
CLS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
CLS_SYLL = {"kd": "p", "sd": "ts", "hhc": "tʃ", "hho": "tʃ"}
CLS_IPA = {"kd": "p", "sd": "ts", "hhc": "x", "hho": "æ"}


def log_mel(x, n_mels=N_MELS, n_fft=N_FFT, hop=HOP, sr=SR):
    n = len(x)
    if n < n_fft:
        x = np.pad(x, (0, n_fft - n))
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    mel_min, mel_max = 2595 * np.log10(1 + 40 / 700), 2595 * np.log10(1 + (sr / 2) / 700)
    m = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = 700 * (10 ** (m / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, len(f) - 1)
        c = min(c, r)
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    mel = S @ fb.T
    return np.log(mel + 1e-6).astype(np.float32)


def load_bank():
    with open(os.path.join(BANK, "index.json"), encoding="utf-8") as f:
        idx = json.load(f)
    by_cls = {c: [] for c in CLASSES}
    for h in idx:
        p = os.path.join(BANK, h["cls"], h["hit_id"] + ".wav")
        if not os.path.exists(p):
            continue
        w, sr = sf.read(p, dtype="float32")
        if sr != SR:
            n = int(round(len(w) * SR / sr))
            w = np.interp(np.linspace(0, 1, n, endpoint=False),
                          np.linspace(0, 1, len(w), endpoint=False), w).astype("float32")
        rms = float(np.sqrt(np.mean(w ** 2) + 1e-12))
        by_cls[h["cls"]].append((w / rms * 0.10, h))
    for c in CLASSES:
        print(f"  {CLS_CN[c]}: {len(by_cls[c])} hits", flush=True)
    return by_cls


def render_window(rng, by_cls, n_notes, tempo_bpm, groove, velocity):
    n_samp = int(WIN_SEC * SR)
    buf = np.zeros(n_samp + SR, dtype=np.float32)
    beat = 60.0 / tempo_bpm
    t = rng.uniform(0.0, 0.15)
    notes = []
    for i in range(n_notes):
        cls = CLASSES[rng.integers(0, 4)]
        hits = by_cls[cls]
        if not hits:
            continue
        w, h = hits[rng.integers(0, len(hits))]
        v = velocity * (1.25 if cls in ("kd", "sd") else 0.8)
        if i % 2 == 1:
            v *= groove
        v *= rng.uniform(0.85, 1.15)
        seg = w * v
        s = int(t * SR)
        e = min(len(buf), s + len(seg))
        if s >= len(buf):
            break
        buf[s:e] += seg[:e - s]
        notes.append({"onset": round(t, 3), "cls": cls,
                      "code": cls, "syll": CLS_SYLL[cls], "ipa": CLS_IPA[cls]})
        step = beat * rng.choice([0.5, 0.5, 1.0, 0.25, 0.75],
                                 p=[0.45, 0.20, 0.15, 0.10, 0.10])
        t += step
        if t > WIN_SEC:
            break
    buf = buf[:n_samp]
    buf = np.tanh(buf * 1.6) / 1.6
    peak = np.max(np.abs(buf)) + 1e-8
    return buf / peak * 0.85, notes


def wav_bytes(seg):
    """float32 -> 16bit PCM WAV bytes（写入 tar 用）"""
    buf = io.BytesIO()
    sf.write(buf, seg, SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def add_to_tar(tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(SHARDS, exist_ok=True)

    print("载入 hit bank ...", flush=True)
    by_cls = load_bank()
    n_win = int(args.hours * 3600 / WIN_SEC)
    n_shards = (n_win + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"目标 {n_win} 窗口 -> {n_shards} 个 tar 分片（每片 {SHARD_SIZE}）", flush=True)

    scores, pairs, manifest = [], [], []
    cnt_cls = {c: 0 for c in CLASSES}
    n_notes_total = 0
    si = 0
    tar = None

    for i in range(n_win):
        if i % SHARD_SIZE == 0:
            if tar is not None:
                tar.close()
            shard_name = f"shard_{si:05d}.tar"
            tar = tarfile.open(os.path.join(SHARDS, shard_name), "w")
            manifest.append({"shard": shard_name, "start_id": i, "count": 0})
            si += 1

        tempo = float(rng.uniform(70, 160))
        groove = float(rng.uniform(0.65, 1.0))
        velocity = float(rng.uniform(0.75, 1.2))
        n_notes = int(rng.integers(3, 13))
        wav, notes = render_window(rng, by_cls, n_notes, tempo, groove, velocity)
        if not notes:
            continue

        sid = f"synth_{i:06d}"
        add_to_tar(tar, f"wav/{sid}.wav", wav_bytes(wav))
        add_to_tar(tar, f"mel/{sid}.npy", npy_bytes(log_mel(wav)))
        manifest[-1]["count"] += 1

        rec = {"id": sid,
               "shard": manifest[-1]["shard"],
               "wav_member": f"wav/{sid}.wav",
               "mel_member": f"mel/{sid}.npy",
               "window": [0.0, round(WIN_SEC, 3)],
               "tempo_bpm": round(tempo, 1), "n_notes": len(notes),
               "score": notes,
               "drum_text": "、".join(CLS_CN[n["cls"]] for n in notes),
               "ipa_text": " ".join(n["ipa"] for n in notes),
               "syll_text": " ".join(n["syll"] for n in notes)}
        scores.append(rec)
        for n in notes:
            cnt_cls[n["cls"]] += 1
        n_notes_total += len(notes)

        cond = rec["ipa_text"] or rec["drum_text"]
        pairs.append({"id": sid, "shard": rec["shard"],
                      "wav_member": rec["wav_member"],
                      "messages": [
                          {"role": "user",
                           "content": f"用beatbox演绎这段鼓谱：{rec['drum_text']}（{cond}）"},
                          {"role": "assistant", "content": rec["drum_text"]}]})

        if (i + 1) % 5000 == 0:
            print(f"  已合成 {i+1}/{n_win} 窗口", flush=True)

    if tar is not None:
        tar.close()

    with open(os.path.join(OUT, "scores.jsonl"), "w", encoding="utf-8") as f:
        for r in scores:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "omni_pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({"total_windows": len(scores), "shard_size": SHARD_SIZE,
                   "n_shards": len(manifest), "shards": manifest}, f, ensure_ascii=False, indent=2)

    tot = len(scores) * WIN_SEC
    print(f"\n合成窗口 {len(scores)} 个, 总时长 {tot/3600:.2f} 小时, {len(manifest)} 个 tar 分片")
    print(f"鼓类分布: {cnt_cls}")
    print(f"窗口内音符数: 均值 {n_notes_total/max(len(scores),1):.2f}")
    print(f"输出: {OUT}/scores.jsonl, omni_pairs.jsonl, MANIFEST.json, shards/ ({len(manifest)} tar)")


if __name__ == "__main__":
    main()
