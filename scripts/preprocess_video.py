"""
Stage B — beatbox 视频/音频预处理管线
视频/音频 → (可选 Demucs 人声分离) → VAD 剔说话/静音 → onset 切片 → 干净片段

用途：为伪标注飞轮准备干净 beatbox 片段。
用法：
  python preprocess_video.py --input <文件/目录> --out data/bbox_raw --mode full --keep_wav
  python preprocess_video.py --input xxx.mp3 --out out_dir --mode light   # 仅 onset 切片
依赖：本地轻量（soundfile+numpy）；full 模式需云端 demucs/yt-dlp
版权：输入音频仅研究用途，不入 git、不外传。
"""
import os, sys, argparse, json, subprocess
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def download_audio(url, out_dir):
    raw = os.path.join(out_dir, "raw"); ensure_dir(raw)
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", os.path.join(raw, "%(id)s.%(ext)s"), url]
    try: subprocess.run(cmd, check=True)
    except Exception as e: print(f"[warn] yt-dlp 失败（需云端/网络）: {e}")

def _run_demucs(in_path, out_root):
    """对单个文件跑 Demucs，返回 vocals.wav；失败抛异常。"""
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", out_root, in_path]
    subprocess.run(cmd, check=True)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    vocals = os.path.join(out_root, "htdemucs", stem, "vocals.wav")
    if os.path.exists(vocals):
        return vocals
    for root, _dirs, files in os.walk(out_root):
        if "vocals.wav" in files:
            return os.path.join(root, "vocals.wav")
    raise FileNotFoundError("Demucs 未找到 vocals.wav")


def separate_vocals(in_path, out_dir, chunk_sec=60):
    """返回分离后的 vocals.wav 路径；长音频按 chunk_sec 切块跑，失败回退原音频。"""
    import soundfile as sf
    out = os.path.join(out_dir, "demucs"); ensure_dir(out)
    wav, sr = sf.read(in_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / sr
    if dur <= chunk_sec:
        try:
            return _run_demucs(in_path, out)
        except Exception as e:
            print(f"[warn] Demucs 失败，用原音频: {e}")
            return in_path

    # 长音频：切块跑，规避 Windows 上长音频 Demucs 的内存/崩溃问题
    print(f"[demucs] 长音频 {dur:.1f}s -> 按 {chunk_sec}s 切块")
    chunk_dir = os.path.join(out, "chunks"); ensure_dir(chunk_dir)
    parts, out_sr = [], None
    for i, start in enumerate(range(0, len(wav), int(chunk_sec * sr))):
        seg = wav[start:start + int(chunk_sec * sr)]
        cp = os.path.join(chunk_dir, f"chunk_{i:03d}.wav")
        sf.write(cp, seg, sr)
        try:
            vp = _run_demucs(cp, out)
            v, vsr = sf.read(vp, dtype="float32")
            if v.ndim > 1:
                v = v.mean(axis=1)
            parts.append(v)
            out_sr = vsr
            print(f"  chunk {i:03d}: {len(seg)/sr:.1f}s -> vocals {len(v)/vsr:.1f}s")
        except Exception as e:
            print(f"[warn] Demucs 在 chunk {i:03d} 失败，整体回退原音频: {e}")
            return in_path
    full = np.concatenate(parts)
    fp = os.path.join(out, "vocals_full.wav")
    sf.write(fp, full, out_sr or sr)
    return fp

def load_audio(path, sr=22050):
    import soundfile as sf
    wav, orig = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if orig != sr:
        n = int(len(wav)*sr/orig)
        wav = np.interp(np.linspace(0,1,num=n,endpoint=False),
                        np.linspace(0,1,num=len(wav),endpoint=False), wav).astype("float32")
    return wav, sr

def energy_vad_zero(wav, sr, frame=1024, hop=512, thresh_db=-40.0):
    n=len(wav)
    for i in range(0, max(n-frame,0), hop):
        seg=wav[i:i+frame]; rms=np.sqrt((seg**2).mean()+1e-10); db=20*np.log10(rms+1e-10)
        if db<=thresh_db: wav[i:i+frame]=0.0
    return wav

def onset_slice(wav, sr, min_gap_ms=30, min_dur_ms=60, top_db=30.0):
    frame, hop = 1024, 256; n=len(wav)
    if n<frame: return []
    env=np.array([np.sqrt((wav[i:i+frame]**2).mean()+1e-10) for i in range(0,n-frame,hop)])
    logenv=20*np.log10(env+1e-10); thresh=logenv.max()-top_db; active=logenv>thresh
    min_gap=int(min_gap_ms/1000*sr/hop); min_dur=int(min_dur_ms/1000*sr/hop)
    segs=[]; i=0
    while i<len(active):
        if active[i]:
            j=i; gap=0
            while j<len(active) and gap<=min_gap:
                gap = 0 if active[j] else gap+1; j+=1
            if j-i>=min_dur: segs.append((i*hop, min(n, j*hop+frame)))
            i=j
        else: i+=1
    return segs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/bbox_raw")
    ap.add_argument("--mode", default="light", choices=["light","full"])
    ap.add_argument("--keep_wav", action="store_true")
    ap.add_argument("--demucs_chunk", type=float, default=60.0, help="Demucs 长音频切块秒数")
    args=ap.parse_args()
    ensure_dir(args.out); seg_dir=os.path.join(args.out,"segments"); ensure_dir(seg_dir)

    inputs=[]
    if os.path.isdir(args.input):
        inputs=[os.path.join(args.input,f) for f in os.listdir(args.input)
                if f.lower().endswith((".mp3",".wav",".m4a",".mp4",".webm",".flac"))]
    else: inputs.append(args.input)

    manifest=[]
    for path in inputs:
        print(f"\n=== {os.path.basename(path)} ===")
        src = separate_vocals(path, args.out, args.demucs_chunk) if args.mode=="full" else path
        wav, sr = load_audio(src)
        print(f"  时长 {len(wav)/sr:.1f}s")
        if args.mode=="full": wav=energy_vad_zero(wav, sr)
        segs=onset_slice(wav, sr)
        print(f"  onset 切片: {len(segs)}")
        for idx,(s,e) in enumerate(segs):
            seg_id=f"{os.path.splitext(os.path.basename(path))[0]}_{idx:04d}"
            entry={"id":seg_id,"source":os.path.basename(path),
                   "start":round(s/sr,3),"end":round(e/sr,3),"dur":round((e-s)/sr,3)}
            if args.keep_wav:
                import soundfile as sf
                wp=os.path.join(seg_dir,seg_id+".wav"); sf.write(wp, wav[s:e], sr); entry["wav"]=wp
            manifest.append(entry)
    mf=os.path.join(args.out,"segments_manifest.jsonl")
    with open(mf,"w",encoding="utf-8") as f:
        for e in manifest: f.write(json.dumps(e,ensure_ascii=False)+"\n")
    print(f"\n完成：{len(manifest)} 片段 -> {mf}")
    print("下一步：python pseudo_label.py --manifest ... --ckpt checkpoints/beatbox_cls/best.pt")

if __name__=="__main__":
    main()
