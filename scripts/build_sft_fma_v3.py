# FMA SFT 数据 v3：30 秒整轨 mp3 → 重写为 30s 16kHz wav（消除 mp3 解码截断风险）
# 任务同 v1：(1) 顶层 8 大类 (2) Electronic 子类细分
# 服务器端运行：python3 build_sft_fma_v3.py
# 仅重写音频 + 生成新 jsonl，标签逻辑复用 v1 的 tracks.csv 解析
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path("/gemini/code/lightfour-train")
AUDIO_DIR = BASE / "data" / "fma_small"          # 000-155/*.mp3
META_DIR = BASE / "data" / "beatbox_raw"          # fma_metadata 解压到这
TRACKS_CSV = META_DIR / "fma_metadata" / "tracks.csv"
GENRES_CSV = META_DIR / "fma_metadata" / "genres.csv"
OUT_DIR = BASE / "data" / "sft"
WAV_DIR = OUT_DIR / "fma_wav30"                   # 重写后的 wav
SFT_PATH = OUT_DIR / "fma_sft_v3.jsonl"

SEED = 42
TEST_RATIO, VAL_RATIO = 0.1, 0.1
TARGET_SR = 16000
TARGET_SEC = 30.0

TOP_GENRES = {
    "Electronic": ("电子音乐", "Electronic", "以合成器、鼓机与采样为核心制作的舞曲/电子音乐"),
    "Hip-Hop": ("嘻哈", "Hip-Hop", "以节奏说唱、搓盘与节拍制作为核心"),
    "Rock": ("摇滚", "Rock", "以吉他、贝斯、鼓三大件驱动的摇滚乐"),
    "Pop": ("流行", "Pop", "旋律通俗、面向大众的流行音乐"),
    "Folk": ("民谣", "Folk", "以原声乐器与传统曲式为特点的民谣"),
    "Experimental": ("实验音乐", "Experimental", "打破常规曲式与音色的实验/先锋音乐"),
    "Instrumental": ("器乐", "Instrumental", "无人声主导的器乐作品"),
    "International": ("世界音乐", "International", "各地域民族风格的世界音乐"),
}

SUB_GENRES = {
    "Ambient Electronic": ("氛围电子", "Ambient Electronic"),
    "IDM": ("智能舞曲", "IDM"),
    "Glitch": ("故障电子", "Glitch"),
    "Techno": ("泰克诺", "Techno"),
    "House": ("浩室", "House"),
    "Dubstep": ("回响贝斯", "Dubstep"),
    "Drum & Bass": ("鼓打贝斯", "Drum & Bass"),
    "Electro": ("电音放克", "Electro"),
    "Chiptune": ("芯片音乐", "Chiptune"),
    "Synthpop": ("合成器流行", "Synthpop"),
    "Breakbeat": ("碎拍", "Breakbeat"),
    "Industrial": ("工业电子", "Industrial"),
    "Noise": ("噪音", "Noise"),
    "Trip-Hop": ("神游舞曲", "Trip-Hop"),
    "Electronica": ("电子乐", "Electronica"),
    "Ambient": ("氛围音乐", "Ambient"),
    "Post-Rock": ("后摇", "Post-Rock"),
    "Shoegaze": ("自赏", "Shoegaze"),
    "Garage": ("车库", "Garage"),
    "Chill-out": ("驰放", "Chill-out"),
    "Dance": ("舞曲", "Dance"),
    "Downtempo": ("缓拍", "Downtempo"),
    "Jungle": ("丛林舞曲", "Jungle"),
    "Minimal Electronic": ("极简电子", "Minimal Electronic"),
    "Breakcore - Hard": ("硬核碎拍", "Breakcore"),
}

Q_TOP = [
    "这段音乐属于哪种大的音乐流派？",
    "请听这段音乐，判断它属于电子、摇滚、嘻哈还是其他大类？",
    "给这段音频分一个顶层流派类别。",
    "听音频，说说这首歌属于哪个大类流派？",
]

Q_SUB = [
    "这是一段电子音乐，请细分它属于哪种电子子流派？",
    "听这段电子乐，它是 House、Techno 还是其他子类？",
    "请判断这段电子音乐的具体子流派（如浩室、泰克诺、回响贝斯等）。",
    "细分这段音频的电子音乐子类。",
]


def load_genre_titles():
    titles = {}
    with open(GENRES_CSV, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            try:
                titles[int(r["genre_id"])] = r["title"].strip()
            except (ValueError, KeyError):
                continue
    return titles


def load_tracks():
    titles = load_genre_titles()
    out = {}
    with open(TRACKS_CSV, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    top_level, header = rows[0], rows[1]
    idx_genre_top = idx_genres = None
    for i, (t, h) in enumerate(zip(top_level, header)):
        if t == "track" and h == "genre_top":
            idx_genre_top = i
        if t == "track" and h == "genres":
            idx_genres = i
    if idx_genre_top is None or idx_genres is None:
        raise RuntimeError(f"columns not found: genre_top={idx_genre_top}, genres={idx_genres}")
    for r in rows[3:]:
        if len(r) <= max(idx_genre_top, idx_genres):
            continue
        try:
            tid = int(float(r[0]))
        except ValueError:
            continue
        gtop = r[idx_genre_top].strip()
        raw = r[idx_genres].strip().strip("[]")
        genre_title = ""
        candidates = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                try:
                    t = titles.get(int(part))
                    if t:
                        candidates.append(t)
                except ValueError:
                    pass
        for t in candidates:
            if t != gtop:
                genre_title = t
                break
        if not genre_title and candidates:
            genre_title = candidates[0]
        out[tid] = (gtop, genre_title)
    return out


def mp3_path(tid: int) -> Path:
    sub = f"{tid:06d}"[:3]
    return AUDIO_DIR / sub / f"{tid:06d}.mp3"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks()
    print(f"tracks.csv 解析出 {len(tracks)} 首")

    items = []
    for tid, (gtop, genre) in tracks.items():
        if mp3_path(tid).exists():
            items.append((tid, gtop, genre))
    print(f"音频存在: {len(items)} 首")

    random.shuffle(items)
    n = len(items)
    n_test, n_val = int(n * TEST_RATIO), int(n * VAL_RATIO)

    def split_of(i):
        if i < n_test:
            return "test"
        if i < n_test + n_val:
            return "val"
        return "train"

    samples = []
    stats = {"top": 0, "sub": 0, "rewritten": 0, "skip": 0}

    for i, (tid, gtop, genre) in enumerate(items):
        sp = split_of(i)
        src = mp3_path(tid)
        dst = WAV_DIR / f"{tid:06d}.wav"

        # 重写 30s 16kHz wav（只做一次，存在即复用）
        if not dst.exists():
            try:
                a, sr = sf.read(src, dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                if sr != TARGET_SR:
                    idx = np.linspace(0, len(a) - 1, int(len(a) * TARGET_SR / sr))
                    a = np.interp(idx, np.arange(len(a)), a).astype("float32")
                need = int(TARGET_SEC * TARGET_SR)
                if len(a) < need:
                    a = np.pad(a, (0, need - len(a)))
                else:
                    a = a[:need]
                sf.write(dst, a, TARGET_SR)
                stats["rewritten"] += 1
            except Exception as e:
                print(f"  skip {tid}: {e}")
                stats["skip"] += 1
                continue
        rel = str(dst).replace("\\", "/")

        if gtop in TOP_GENRES:
            cn, en, desc = TOP_GENRES[gtop]
            samples.append({
                "task": "fma_top_genre",
                "split": sp,
                "track_id": tid,
                "messages": [
                    {"role": "user", "content": f"<audio>{random.choice(Q_TOP)}", "audios": [rel]},
                    {"role": "assistant", "content": f"这首曲子属于**{cn}（{en}）**流派，{desc}。"},
                ],
            })
            stats["top"] += 1

        if gtop == "Electronic" and genre in SUB_GENRES:
            scn, sen = SUB_GENRES[genre]
            samples.append({
                "task": "fma_electronic_subgenre",
                "split": sp,
                "track_id": tid,
                "messages": [
                    {"role": "user", "content": f"<audio>{random.choice(Q_SUB)}", "audios": [rel]},
                    {"role": "assistant", "content": f"这段电子音乐的子流派是**{scn}（{sen}）**。"},
                ],
            })
            stats["sub"] += 1

    random.shuffle(samples)
    with open(SFT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_train = sum(1 for s in samples if s["split"] == "train")
    n_val = sum(1 for s in samples if s["split"] == "val")
    n_test = len(samples) - n_train - n_val
    print(f"重写 wav: {WAV_DIR} ({stats['rewritten']} 个新写)")
    print(f"SFT v3 样本: {SFT_PATH} 共 {len(samples)} 条 (train {n_train} / val {n_val} / test {n_test})")
    print(f"统计: {stats}")


if __name__ == "__main__":
    main()
