# 将 FMA-small 音频 + tracks.csv 流派标签转成 SFT 训练样本（ms-swift 多模态 jsonl）
# 两个任务: (1) 粗分类 8 大类 (genre_top)  (2) Electronic 子类细分 (House/Techno/Dubstep/DnB 等)
# 输入: data/fma/fma_small/000-155/*.mp3 + data/fma/fma_metadata/tracks.csv
# 输出: data/sft/fma_sft.jsonl
# 划分: 固定 seed 80/10/10 → train/test/val（分类任务按 track 分层近似: 全局随机）
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "fma" / "fma_small"
TRACKS_CSV = ROOT / "data" / "fma" / "fma_metadata" / "tracks.csv"
GENRES_CSV = ROOT / "data" / "fma" / "fma_metadata" / "genres.csv"
OUT_PATH = ROOT / "data" / "sft" / "fma_sft.jsonl"

SEED = 42
TEST_RATIO, VAL_RATIO = 0.1, 0.1

# 8 大顶层流派 → 中英名与描述
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

# Electronic 子类（tracks.csv 的 genre 字段）→ 中英名
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
    "Chip Music": ("芯片音乐", "Chip Music"),
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
    """genres.csv: genre_id,#tracks,parent,title,top_level → {id: title}"""
    titles = {}
    with open(GENRES_CSV, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            try:
                titles[int(r["genre_id"])] = r["title"].strip()
            except (ValueError, KeyError):
                continue
    return titles


def load_tracks():
    """解析 tracks.csv。返回 {track_id: (genre_top, genre_title)}
    tracks.csv 表头: rows[0]=分组, rows[1]=字段名; genres 列是 id 列表如 [21]，取首个 id 映射标题"""
    titles = load_genre_titles()
    out = {}
    with open(TRACKS_CSV, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    top_level = rows[0]
    header = rows[1]
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
        # 优先取与 genre_top 不同的具体子类（排除被折成泛标签的父类 id）
        for t in candidates:
            if t != gtop:
                genre_title = t
                break
        if not genre_title and candidates:
            genre_title = candidates[0]
        out[tid] = (gtop, genre_title)
    return out


def audio_rel_path(tid: int):
    sub = f"{tid:06d}"[:3]
    return f"data/fma/fma_small/{sub}/{tid:06d}.mp3"


def main():
    random.seed(SEED)
    tracks = load_tracks()
    print(f"tracks.csv 解析出 {len(tracks)} 首")

    # 只保留音频存在的曲目
    items = []
    for tid, (gtop, genre) in tracks.items():
        rel = audio_rel_path(tid)
        if (ROOT / rel).exists():
            items.append((tid, gtop, genre, rel))
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
    stats = {"top": 0, "sub": 0, "skipped_genre": 0}

    for i, (tid, gtop, genre, rel) in enumerate(items):
        sp = split_of(i)

        # 任务 1: 顶层分类
        if gtop in TOP_GENRES:
            cn, en, desc = TOP_GENRES[gtop]
            q = random.choice(Q_TOP)
            a = f"这首曲子属于**{cn}（{en}）**流派，{desc}。"
            samples.append({
                "task": "fma_top_genre",
                "split": sp,
                "track_id": tid,
                "messages": [
                    {"role": "user", "content": f"<audio>{q}", "audios": [rel]},
                    {"role": "assistant", "content": a},
                ],
            })
            stats["top"] += 1

        # 任务 2: Electronic 子类细分
        if gtop == "Electronic" and genre in SUB_GENRES:
            scn, sen = SUB_GENRES[genre]
            q = random.choice(Q_SUB)
            a = f"这段电子音乐的子流派是**{scn}（{sen}）**。"
            samples.append({
                "task": "fma_electronic_subgenre",
                "split": sp,
                "track_id": tid,
                "messages": [
                    {"role": "user", "content": f"<audio>{q}", "audios": [rel]},
                    {"role": "assistant", "content": a},
                ],
            })
            stats["sub"] += 1
        elif gtop == "Electronic":
            stats["skipped_genre"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter((s["task"], s["split"]) for s in samples)
    print(f"SFT 样本: {OUT_PATH} 共 {len(samples)} 条")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")
    print(f"任务统计: {stats}")


if __name__ == "__main__":
    main()
