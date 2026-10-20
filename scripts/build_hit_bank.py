"""
Stage 3 关键解锁 —— beatbox【打击乐样本库】(hit bank)

问题：生成型数据只有 AVP-LVT 的 0.55h / 140 段，远不够训生成模型(10-50h+)。
      等 Stage 2 视频飞轮灌数据周期太长。

解法：beatbox 本质是【离散打击乐事件序列】——每个音（底鼓/军鼓/闭镲/开镲）
      都是一次独立的瞬态击打。这与"采样合成器"(sampler) 的工作方式完全一致：
        1) 从已标注的真实录音里，按 onset 精确切出每一次击打（hit）
        2) 建库：4 类 × 28 参与者 × ~几十次 = 数千个真实人声击打样本
        3) 用库里的真实击打，按任意【鼓谱】重新排布合成新音频
      => 0.55h 的真实素材 -> 无限的 (鼓谱 -> 音频) 配对训练数据
      => 且合成的音频是【真实人声音色】，不是合成器假声

本脚本只做第 1、2 步（切分 + 建库），合成在 render_synth.py。

输出：
  data/hit_bank/{kd,sd,hhc,hho}/*.wav   16k 单声道，每个是一次击打（含自然衰减尾）
  data/hit_bank/index.json              元数据：类别/参与者/IPA/音节/时长/能量
运行： py -3.12 scripts/build_hit_bank.py
"""
import os, csv, json
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "beatbox", "AVP-LVT_Dataset", "AVP_Dataset", "Personal")
OUT = os.path.join(ROOT, "data", "hit_bank")

SR = 16000
PRE_ROLL = 0.030      # 起音前留一点，避免削掉起音瞬间
MAX_TAIL = 1.20       # 最大截取长度
KEEP_REL = 0.05       # 尾部保留阈值：帧RMS > 峰值帧RMS * 0.05
FRAME = 160           # 10ms 帧 @16k

CLASSES = ["kd", "sd", "hhc", "hho"]
CLS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}


def read_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        n = int(round(len(wav) * SR / sr))
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    return wav


def frame_rms(x):
    n = len(x) // FRAME
    if n == 0:
        return np.array([float(np.sqrt(np.mean(x ** 2) + 1e-12))])
    f = x[:n * FRAME].reshape(n, FRAME)
    return np.sqrt((f ** 2).mean(axis=1) + 1e-12)


def extract_hit(wav, onset, next_onset):
    """按 onset 精确对齐并切出一次击打：起音前 30ms，尾部按能量自适应截断"""
    # 1) 起音对齐：在 [onset-50ms, onset+120ms] 内找能量上升点
    a = max(0, int((onset - 0.050) * SR))
    b = min(len(wav), int((onset + 0.120) * SR))
    seg = wav[a:b]
    if len(seg) < FRAME:
        return None
    fr = frame_rms(seg)
    thr = fr.max() * 0.25
    idx = np.argmax(fr > thr)
    hit_start = a + int(idx * FRAME)
    # 往回退一点，保留起音瞬间
    hit_start = max(0, hit_start - int(0.010 * SR))

    # 2) 尾部截断
    max_end = min(len(wav), hit_start + int(MAX_TAIL * SR))
    if next_onset is not None:
        # 下一次击打开始即本次结束（开镲的尾音可与下次重叠，混音时自然相加）
        max_end = min(max_end, int(next_onset * SR))
    seg = wav[hit_start:max_end]
    if len(seg) < FRAME * 3:
        return None
    fr = frame_rms(seg)
    peak = fr.max()
    keep = np.where(fr > max(peak * KEEP_REL, 1e-4))[0]
    if len(keep) == 0:
        return None
    end = hit_start + int((keep[-1] + 1) * FRAME)
    seg = wav[hit_start:end]
    if len(seg) < FRAME * 3:
        return None
    return seg, (hit_start - int(onset * SR)) / SR


def main():
    for c in CLASSES:
        os.makedirs(os.path.join(OUT, c), exist_ok=True)
    index = []
    stats = {"wav": 0, "notes": 0, "skip": 0, "cls_skip": 0}

    for part in sorted(os.listdir(BASE)):
        pdir = os.path.join(BASE, part)
        if not os.path.isdir(pdir):
            continue
        spk = part.replace("Participant_", "P")
        for csvf in sorted(os.listdir(pdir)):
            if not csvf.endswith(".csv"):
                continue
            stem = csvf[:-4]
            wavp = os.path.join(pdir, stem + ".wav")
            if not os.path.exists(wavp):
                continue
            rows = []
            with open(os.path.join(pdir, csvf), newline="", encoding="utf-8-sig") as fh:
                for r in csv.reader(fh):
                    if len(r) < 2:
                        continue
                    try:
                        on = float(r[0])
                    except ValueError:
                        continue
                    cls = r[1].strip().lower()
                    if cls not in CLASSES:
                        stats["cls_skip"] += 1
                        continue
                    rows.append((on, cls,
                                 r[2].strip() if len(r) > 2 else "",
                                 r[3].strip() if len(r) > 3 else ""))
            if not rows:
                continue
            rows.sort(key=lambda x: x[0])
            wav = read_wav(wavp)
            stats["wav"] += 1
            for i, (on, cls, syll, ipa) in enumerate(rows):
                nxt = rows[i + 1][0] if i + 1 < len(rows) else None
                res = extract_hit(wav, on, nxt)
                if res is None:
                    stats["skip"] += 1
                    continue
                seg, align_err = res
                peak = float(np.max(np.abs(seg)))
                if peak < 1e-4:
                    stats["skip"] += 1
                    continue
                # 归一化到统一峰值，混音时按 hit 的 RMS 再做动态分配
                seg = seg / peak * 0.9
                hit_id = f"{stem}_{i:03d}_{cls}"
                hp = os.path.join(OUT, cls, hit_id + ".wav")
                sf.write(hp, seg, SR, subtype="PCM_16")
                index.append({
                    "hit_id": hit_id, "cls": cls, "cls_cn": CLS_CN[cls],
                    "participant": spk, "source": f"{part}/{stem}.wav",
                    "onset_in_source": round(on, 4),
                    "align_err_ms": round(align_err * 1000, 2),
                    "dur": round(len(seg) / SR, 4),
                    "rms": float(np.sqrt(np.mean(seg ** 2))),
                    "syll": syll, "ipa": ipa,
                })
                stats["notes"] += 1

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    from collections import Counter
    cnt = Counter(x["cls"] for x in index)
    durs = {}
    for c in CLASSES:
        d = [x["dur"] for x in index if x["cls"] == c]
        durs[CLS_CN[c]] = (round(min(d), 3), round(float(np.mean(d)), 3), round(max(d), 3)) if d else None
    print(f"源 wav: {stats['wav']}  切出 hit: {stats['notes']}  跳过: {stats['skip']}  非4类标注: {stats['cls_skip']}")
    print(f"每类数量: {dict(cnt)}")
    print(f"每类时长(min/mean/max): {durs}")
    print(f"总击打时长: {sum(x['dur'] for x in index)/3600:.3f} 小时")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
