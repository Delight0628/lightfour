"""
Stage 3 前置 — beatbox【生成型】数据集构建（格式就绪版）
回答"Omni 生成 beatbox 的数据准备好了吗"：
  - 识别型数据：已就绪（data/sft_v4/，音频→标签）
  - 生成型数据：AVP-LVT 仅 0.55h / 140 段，【不足以训生成模型】，
    但可以先把【条件-目标配对格式】和【训练管线】搭好，等 Stage 2 飞轮灌数据直接用

两条技术路线都产出配对：
  路线 A（Omni 原生·音频 token）：cond = IPA/音节序列（文本） -> target = beatbox 音频
  路线 B（mel→wav 声码器）：cond = 鼓谱序列(onset+鼓类) -> target = log-mel 频谱 -> 再合成波形
本脚本产出：
  data/gen_v1/scores.jsonl     每条：id, wav, 鼓谱序列, IPA序列, 音节序列, 时长
  data/gen_v1/mel/*.npy        路线 B 的 log-mel 目标（16k/64mel/hop256）
  data/gen_v1/omni_pairs.jsonl 路线 A 的 Omni 风格 jsonl（prompt=IPA文本, audios=目标音频）
运行： py -3.12 scripts/build_gen_dataset.py
"""
import os, json, csv, wave, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "beatbox", "AVP-LVT_Dataset", "AVP_Dataset", "Personal")
OUT = os.path.join(ROOT, "data", "gen_v1")
MEL_DIR = os.path.join(OUT, "mel")
os.makedirs(MEL_DIR, exist_ok=True)

SR = 16000
N_FFT, HOP, N_MELS = 1024, 256, 64
WIN_SEC = 2.0          # 生成窗口
STRIDE_SEC = 1.0       # 滑窗步长
CLS_CN = {"Kick": "底鼓", "Snare": "军鼓", "HHclosed": "闭镲", "HHopened": "开镲",
          "HH": "镲", "kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
CLS_IPA = {"kd": "p", "sd": "ts", "hhc": "t", "hho": "t"}   # 音节级音素映射（AVP 论文风格）


def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    import array
    a = array.array("h"); a.frombytes(raw)
    if ch > 1:
        a = array.array("h", [int(sum(a[i:i + ch]) / ch) for i in range(0, len(a), ch)])
    if sr != SR:
        a = array.array("h", [int(v) for v in np.interp(
            np.linspace(0, 1, int(len(a) * SR / sr), endpoint=False),
            np.linspace(0, 1, len(a), endpoint=False), np.array(a, dtype=np.float32))])
    return a, SR


def log_mel(x, n_mels=N_MELS, n_fft=N_FFT, hop=HOP, sr=SR):
    """轻量 log-mel（numpy 实现，避免 librosa 依赖）"""
    n = len(x)
    if n < n_fft: x = np.pad(x, (0, n_fft - n))
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
        if c == l: c = l + 1
        if r == c: r = c + 1
        r = min(r, len(f) - 1); c = min(c, r)
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    mel = S @ fb.T
    return np.log(mel + 1e-6).astype(np.float32)


def main():
    random.seed(42)
    scores, pairs = [], []
    n_wav = n_win = 0
    for part in sorted(os.listdir(BASE)):
        pdir = os.path.join(BASE, part)
        if not os.path.isdir(pdir): continue
        for csvf in sorted(os.listdir(pdir)):
            if not csvf.endswith(".csv"): continue
            stem = csvf[:-4]
            wavp = os.path.join(pdir, stem + ".wav")
            if not os.path.exists(wavp): continue
            ann = []
            with open(os.path.join(pdir, csvf), newline="", encoding="utf-8-sig") as fh:
                for r in csv.reader(fh):
                    if len(r) < 2: continue
                    try: on = float(r[0])
                    except ValueError: continue
                    lab = r[1].strip()
                    cn = CLS_CN.get(lab) or CLS_CN.get(lab.lower())
                    if not cn: continue
                    ann.append({"onset": round(on, 4), "cls": cn,
                                "code": lab.lower() if lab.lower() in CLS_IPA else lab,
                                "syll": r[2].strip() if len(r) > 2 else "",
                                "ipa": r[3].strip() if len(r) > 3 else ""})
            if not ann: continue
            samples, sr = read_wav(wavp)
            arr = np.array(samples, dtype=np.float32)
            dur = len(arr) / sr
            n_wav += 1
            W, S = int(WIN_SEC * sr), int(STRIDE_SEC * sr)
            for wi, s0 in enumerate(range(0, max(len(arr) - W, 0) + 1, S)):
                e0 = s0 + W
                seg = arr[s0:e0]
                if len(seg) < W: seg = np.pad(seg, (0, W - len(seg)))
                # 该窗口内落下的标注
                t0, t1 = s0 / sr, e0 / sr
                win = [a for a in ann if t0 - 0.05 <= a["onset"] < t1 - 0.05]
                if not win: continue
                sid = f"{part}_{stem}_{wi:03d}"
                mel = log_mel(seg)
                np.save(os.path.join(MEL_DIR, sid + ".npy"), mel)
                seq = [{"onset": round(a["onset"] - t0, 3), "cls": a["cls"], "code": a["code"],
                        "syll": a["syll"], "ipa": a["ipa"]} for a in win]
                rec = {"id": sid, "wav": wavp.replace("\\", "/"),
                       "window": [round(t0, 3), round(t1, 3)],
                       "score": seq,
                       "drum_text": "、".join(a["cls"] for a in win),
                       "ipa_text": " ".join(a["ipa"] for a in win if a["ipa"]),
                       "syll_text": " ".join(a["syll"] for a in win if a["syll"]),
                       "mel": f"mel/{sid}.npy"}
                scores.append(rec); n_win += 1
                # 路线 A：Omni 风格（文本条件 -> 目标音频）
                cond = rec["ipa_text"] or rec["drum_text"]
                pairs.append({"messages": [
                    {"role": "user", "content": f"用beatbox演绎这段鼓谱：{rec['drum_text']}（{cond}）",
                     "audios": [rec["wav"]]},
                    {"role": "assistant", "content": rec["drum_text"], "audios": [rec["wav"]]}]})

    with open(os.path.join(OUT, "scores.jsonl"), "w", encoding="utf-8") as f:
        for r in scores: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "omni_pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in pairs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    cnt = Counter(x["cls"] for r in scores for x in r["score"])
    tot_s = sum(r["window"][1] - r["window"][0] for r in scores)
    print(f"录音 {n_wav} 段 -> 生成窗口 {n_win} 个, 总时长 {tot_s/3600:.2f} 小时")
    print(f"鼓类分布: {dict(cnt)}")
    print(f"窗口内音符数: 均值 {sum(len(r['score']) for r in scores)/max(len(scores),1):.2f}")
    print(f"输出: {OUT}/scores.jsonl, {OUT}/omni_pairs.jsonl, {OUT}/mel/ ({len(os.listdir(MEL_DIR))} 个 npy)")
    print("\n【数据量评估】0.55h 远低于生成模型所需(10-50h+)。"
          "\n  格式与管线已就绪，等 Stage 2 伪标注飞轮灌入 50-100h 视频 beatbox 后直接扩量。")


if __name__ == "__main__":
    main()
