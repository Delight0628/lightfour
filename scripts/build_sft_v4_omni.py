"""
Omni SFT 数据 v4 — 修正配方（去掉模板主导 loss）
v3 失败原因：答案 99% 是公共模板 token，eval_loss 虚低、模型背模板不听音频。
v4 修正：
  1. 答案只保留【类别标签】（≤4 token），不写解释模板 → 类别 token 占 loss 100%
  2. beatbox 音频加长到 2-5s 窗口（v1 只有 0.06-1.5s，特征太少）
  3. 混合两种任务：分类标签（主导）+ 可选解释（少量，10%）
  4. 音乐任务(EDM)保留（Omni 音频塔在音乐上相对可用），beatbox 走"蒸馏式"：由
     Stage A 分类器产出标签，Omni 只需学会按标签组织语言（见 agent.md §3 Stage3）
输出：data/sft_v4/{train,eval,test}.jsonl
运行： py -3.12 scripts/build_sft_v4_omni.py
"""
import os, json, csv, random, wave, array
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAT_DIR = os.path.join(ROOT, "data", "beatbox", "AVP-LVT_Dataset", "AVP_Dataset", "Personal")
CLIP_DIR = os.path.join(ROOT, "data", "sft", "beatbox_clips")
FMA_DIR = os.path.join(ROOT, "data", "fma", "fma_small")
OUT = os.path.join(ROOT, "data", "sft_v4")
os.makedirs(OUT, exist_ok=True)

BEAT_CN = {"Kick": "底鼓", "Snare": "军鼓", "HHclosed": "闭镲", "HHopened": "开镲",
           "HH": "镲", "kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
FMA_TOP = {"Electronic": "电子音乐", "Experimental": "实验音乐", "Folk": "民谣", "Hip-Hop": "嘻哈",
           "Instrumental": "器乐", "International": "国际音乐", "Pop": "流行", "Rock": "摇滚"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}

def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes(); sr = w.getframerate()
        raw = w.readframes(n)
        import array
        a = array.array("h"); a.frombytes(raw)
        return a, sr, w.getnchannels()

def write_wav(path, samples, sr):
    import array
    a = array.array("h", [max(-32768, min(32767, int(s))) for s in samples])
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(a.tobytes())

def make_rec(audio_path, question, answer):
    return {"messages": [{"role": "user", "content": f"<audio>{question}", "audios": [audio_path]},
                         {"role": "assistant", "content": answer}]}

# ---------- 1. beatbox：长窗口 + 纯标签答案 ----------
def build_beatbox():
    rows = []
    for part in sorted(os.listdir(BEAT_DIR)):
        pdir = os.path.join(BEAT_DIR, part)
        if not os.path.isdir(pdir): continue
        for csvf in sorted(os.listdir(pdir)):
            if not csvf.endswith(".csv"): continue
            stem = csvf[:-4]
            wavf = os.path.join(pdir, stem + ".wav")
            if not os.path.exists(wavf): continue
            with open(os.path.join(pdir, csvf), newline="", encoding="utf-8-sig") as f:
                for r in csv.reader(f):
                    if len(r) < 2: continue
                    try: onset = float(r[0])
                    except ValueError: continue
                    label = r[1].strip()
                    cn = BEAT_CN.get(label) or BEAT_CN.get(label.lower())
                    if not cn: continue
                    rows.append((part, stem, wavf, onset, cn, label))
    print(f"beatbox 标注行: {len(rows)}")
    return rows

def build_windows(rows, out_dir, win_list=(2.0, 3.0, 5.0), min_gap=0.12):
    os.makedirs(out_dir, exist_ok=True)
    byfile = defaultdict(list)
    for part, stem, wavf, onset, cn, label in rows: byfile[(part, stem, wavf)].append((onset, cn, label))
    out = []
    for (part, stem, wavf), items in byfile.items():
        items.sort()
        samples, sr, ch = read_wav(wavf)
        if ch > 1:
            samples = [sum(samples[i:i+ch]) / ch for i in range(0, len(samples), ch)]
        for wsec in win_list:
            W = int(sr * wsec)
            for i in range(0, len(items) - 1, 2):     # 步进取窗口
                seq = items[i:i+4]
                if not seq: continue
                s0 = int(seq[0][0] * sr)
                seg = samples[s0:s0 + W]
                if len(seg) < W * 0.6: continue
                if len(seg) < W: seg = seg + array.array("h", [0] * (W - len(seg)))
                seg_id = f"{part}_{stem}_w{wsec:.0f}_{i:03d}"
                wp = os.path.join(out_dir, seg_id + ".wav")
                if not os.path.exists(wp):
                    write_wav(wp, seg, sr)
                labels = [c for _, c, _ in seq]
                # 纯标签答案（去模板）：唯一类别集合 + 序列
                answer = "、".join(dict.fromkeys(labels))
                out.append({"path": wp, "answer": answer, "part": part,
                            "pnum": "P" + "".join(ch for ch in part if ch.isdigit()),
                            "question": "这段 beatbox 模仿的是什么鼓？只回答鼓类名称。"})
    return out

# ---------- 2. FMA 音乐分类（保留，纯标签） ----------
def build_fma():
    import csv as c2
    meta = os.path.join(ROOT, "data", "fma", "fma_metadata")
    trp = os.path.join(meta, "tracks.csv")
    if not os.path.exists(trp): return []
    rows = []
    with open(trp, newline="", encoding="utf-8") as f:
        rd = list(c2.reader(f))
    hdr = rd[1]
    try: gi = hdr.index("genre_top")
    except ValueError: return []
    for r in rd[2:]:
        if len(r) <= gi: continue
        gt = r[gi].strip()
        if gt not in FMA_TOP: continue
        tid = r[0].strip()
        p = os.path.join(FMA_DIR, tid[:3], tid + ".mp3")
        if os.path.exists(p): rows.append((p, FMA_TOP[gt]))
    return rows

def main():
    random.seed(42)
    print("=== beatbox 长窗口 ===", flush=True)
    rows = build_beatbox()
    win_dir = os.path.join(OUT, "beatbox_windows")
    beats = build_windows(rows, win_dir)
    print(f"beatbox 窗口: {len(beats)}", flush=True)

    print("=== FMA 音乐 ===", flush=True)
    fmas = build_fma()
    print(f"FMA: {len(fmas)}", flush=True)

    # 合并 + 划分（beatbox 按参与者留出：test=P9/11/19/24, val=P20/21；FMA 随机划分）
    VAL_PARTS = {"P20", "P21"}
    data = []
    for b in beats:
        if b["pnum"] in TEST_PARTS: sp = "test"
        elif b["pnum"] in VAL_PARTS: sp = "val"
        else: sp = "train"
        data.append((sp, make_rec(b["path"].replace("\\", "/"), b["question"], b["answer"])))
    fma_recs = [make_rec(p.replace("\\", "/"), "这段音乐属于哪个流派？只回答流派名称。", lab)
                for p, lab in fmas]
    random.shuffle(fma_recs)
    n_fma_te = int(len(fma_recs) * 0.10); n_fma_va = int(len(fma_recs) * 0.10)
    data += [("test", r) for r in fma_recs[:n_fma_te]]
    data += [("val", r) for r in fma_recs[n_fma_te:n_fma_te + n_fma_va]]
    data += [("train", r) for r in fma_recs[n_fma_te + n_fma_va:]]

    by = defaultdict(list)
    for split, r in data: by[split].append(r)
    train, val, test = by["train"], by["val"], by["test"]
    random.shuffle(train); random.shuffle(val); random.shuffle(test)

    for name, obj in (("train", train), ("eval", val), ("test", test)):
        fp = os.path.join(OUT, f"{name}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for r in obj: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(obj)} -> {fp}", flush=True)

    # 答案长度统计（验证"类别 token 主导"）
    lens = [len(r["messages"][1]["content"]) for r in train]
    print(f"答案字符长度: 均值 {sum(lens)/len(lens):.1f}  最大 {max(lens)}", flush=True)
    print(f"答案样例: {train[0]['messages'][1]['content']!r} | {train[5]['messages'][1]['content']!r}", flush=True)

if __name__ == "__main__":
    main()
