"""
Stage 3 —— 用【打击乐样本库】合成无限量的 (鼓谱 -> 音频) 生成型训练数据

背景：AVP-LVT 真实录音仅 0.55h，远不够训生成模型(10-50h+)。
      build_hit_bank.py 已把真实人声击打切成 4872 个 hit（4 类 x 28 参与者）。
      beatbox 本质是离散打击乐事件序列 —— 用采样合成器原理：
        按鼓谱把真实 hit 重新排布、叠加 → 任意长度的新音频
      => 0.55h 真实素材 -> 无限 (鼓谱 -> 音频) 配对，且音色是真实人声

输出（与 gen_v1 完全同格式，可直接复用下游管线）：
  data/gen_synth/scores.jsonl      鼓谱 + IPA/音节 + mel 路径 + 音频路径
  data/gen_synth/mel/*.npy         log-mel（路线 B：mel->wav 声码器目标）
  data/gen_synth/wav/*.wav         合成音频（路线 A：Omni 原生音频 token 目标）
  data/gen_synth/omni_pairs.jsonl  Omni 风格 jsonl（文本条件 -> 目标音频）

运行： py -3.12 scripts/render_synth.py --hours 50
"""
import os, json, argparse
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "data", "hit_bank")
OUT = os.path.join(ROOT, "data", "gen_synth")

SR = 16000
N_FFT, HOP, N_MELS = 1024, 256, 64
WIN_SEC = 2.0
CLASSES = ["kd", "sd", "hhc", "hho"]
CLS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
# AVP-LVT 论文风格的音节/音素映射（与 gen_v1 保持一致）
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
    """按类别载入全部 hit，并归一化到统一 RMS（混音时按响度动态分配）"""
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
        w = w / rms * 0.10        # 统一到 RMS 0.10，混音时再调力度
        by_cls[h["cls"]].append((w, h))
    for c in CLASSES:
        print(f"  {CLS_CN[c]}: {len(by_cls[c])} hits", flush=True)
    return by_cls


def render_window(rng, by_cls, n_notes, tempo_bpm, groove, velocity):
    """按合成的鼓谱渲染一个窗口，返回 (音频, 音符序列)"""
    n_samp = int(WIN_SEC * SR)
    buf = np.zeros(n_samp + SR, dtype=np.float32)   # 多留 1s 容衰减尾
    beat = 60.0 / tempo_bpm
    t = rng.uniform(0.0, 0.15)                     # 随机起始偏移
    notes = []
    for i in range(n_notes):
        cls = CLASSES[rng.integers(0, 4)]
        hits = by_cls[cls]
        if not hits:
            continue
        w, h = hits[rng.integers(0, len(hits))]
        # 力度：底鼓/军鼓重，镲轻；groove 让偶数拍略弱（人味）
        v = velocity * (1.25 if cls in ("kd", "sd") else 0.8)
        if i % 2 == 1:
            v *= groove
        v *= rng.uniform(0.85, 1.15)               # 击打随机性
        seg = w * v
        s = int(t * SR)
        e = min(len(buf), s + len(seg))
        if s >= len(buf):
            break
        buf[s:e] += seg[:e - s]
        notes.append({"onset": round(t, 3), "cls": cls,
                      "code": cls, "syll": CLS_SYLL[cls], "ipa": CLS_IPA[cls]})
        # 下一次击打时间
        step = beat * rng.choice([0.5, 0.5, 1.0, 0.25, 0.75],
                                 p=[0.45, 0.20, 0.15, 0.10, 0.10])
        t += step
        if t > WIN_SEC:
            break
    buf = buf[:n_samp]
    # 软限幅（避免叠加爆音）
    buf = np.tanh(buf * 1.6) / 1.6
    peak = np.max(np.abs(buf)) + 1e-8
    buf = buf / peak * 0.85
    return buf, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=30.0, help="合成总时长（小时）")
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(OUT, "mel"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "wav"), exist_ok=True)

    print("载入 hit bank ...", flush=True)
    by_cls = load_bank()
    n_win = int(args.hours * 3600 / WIN_SEC)

    scores, pairs = [], []
    for i in range(n_win):
        # 节奏参数随机化 → 覆盖快/慢/密集/稀疏
        tempo = float(rng.uniform(70, 160))
        groove = float(rng.uniform(0.65, 1.0))
        velocity = float(rng.uniform(0.75, 1.2))
        n_notes = int(rng.integers(3, 13))
        wav, notes = render_window(rng, by_cls, n_notes, tempo, groove, velocity)
        if not notes:
            continue
        sid = f"synth_{i:06d}"
        wp = os.path.join(OUT, "wav", sid + ".wav")
        sf.write(wp, wav, SR, subtype="PCM_16")
        mel = log_mel(wav)
        np.save(os.path.join(OUT, "mel", sid + ".npy"), mel)

        rec = {"id": sid, "wav": wp.replace("\\", "/"),
               "window": [0.0, round(WIN_SEC, 3)],
               "tempo_bpm": round(tempo, 1), "n_notes": len(notes),
               "score": notes,
               "drum_text": "、".join(CLS_CN[n["cls"]] for n in notes),
               "ipa_text": " ".join(n["ipa"] for n in notes),
               "syll_text": " ".join(n["syll"] for n in notes),
               "mel": f"mel/{sid}.npy"}
        scores.append(rec)
        cond = rec["ipa_text"] or rec["drum_text"]
        pairs.append({"messages": [
            {"role": "user", "content": f"用beatbox演绎这段鼓谱：{rec['drum_text']}（{cond}）",
             "audios": [rec["wav"]]},
            {"role": "assistant", "content": rec["drum_text"], "audios": [rec["wav"]]}]})

        if (i + 1) % 2000 == 0:
            print(f"  已合成 {i+1}/{n_win} 窗口", flush=True)

    with open(os.path.join(OUT, "scores.jsonl"), "w", encoding="utf-8") as f:
        for r in scores:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "omni_pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    cnt = Counter(n["cls"] for r in scores for n in r["score"])
    tot = sum(r["window"][1] - r["window"][0] for r in scores)
    print(f"\n合成窗口 {len(scores)} 个, 总时长 {tot/3600:.2f} 小时")
    print(f"鼓类分布: {{k: v for k, v in cnt.items()}}" .replace("{k: v for k, v in cnt.items()}", str(dict(cnt))))
    print(f"窗口内音符数: 均值 {sum(len(r['score']) for r in scores)/max(len(scores),1):.2f}")
    print(f"输出: {OUT}/scores.jsonl, omni_pairs.jsonl, mel/ ({len(os.listdir(os.path.join(OUT,'mel')))} npy), wav/ ({len(os.listdir(os.path.join(OUT,'wav')))} wav)")


if __name__ == "__main__":
    main()
