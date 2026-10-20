import tarfile
import wave
import numpy as np
import io
import json

print("=== 生成音频特性验证（shard_00000.tar 前 5 个）===")
tf = tarfile.open(r"d:\lightfour\data\gen_synth_tar\shard_00000.tar")
names = [n for n in tf.getnames() if n.startswith("wav/")][:5]
for nm in names:
    f = tf.extractfile(nm)
    data = f.read()
    w = wave.open(io.BytesIO(data))
    frames = w.readframes(w.getnframes())
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    w.close()
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(arr)))) / 2)
    print(f"{nm}: dur={len(arr)/16000:.2f}s rms={rms:.3f} peak={peak:.3f} zcr={zcr:.3f}")
tf.close()

print("\n=== 鼓谱标注验证（前 5 条）===")
sc = [json.loads(l) for l in open(r"d:\lightfour\data\gen_synth_tar\scores.jsonl", encoding="utf-8")][:5]
for s in sc:
    clss = [n["cls"] for n in s["score"]]
    print(f"{s['id']}: tempo={s['tempo_bpm']} n_notes={s['n_notes']} classes={clss}")

print("\n=== 全量统计 ===")
allsc = [json.loads(l) for l in open(r"d:\lightfour\data\gen_synth_tar\scores.jsonl", encoding="utf-8")]
from collections import Counter
cnt = Counter(n["cls"] for r in allsc for n in r["score"])
notes = [len(r["score"]) for r in allsc]
tempos = [r["tempo_bpm"] for r in allsc]
print(f"总窗口: {len(allsc)}  总音符: {sum(notes)}")
print(f"鼓类分布: {dict(cnt)}")
print(f"音符/窗口: min={min(notes)} mean={np.mean(notes):.2f} max={max(notes)}")
print(f"tempo: min={min(tempos):.1f} mean={np.mean(tempos):.1f} max={max(tempos):.1f}")
