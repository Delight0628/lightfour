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
import os, argparse, json, subprocess
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def download_audio(url, out_dir):
    raw = os.path.join(out_dir, "raw"); ensure_dir(raw)
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", os.path.join(raw, "%(id)s.%(ext)s"), url]
    try: subprocess.run(cmd, check=True)
    except Exception as e: print(f"[warn] yt-dlp 失败（需云端/网络）: {e}")

def separate_vocals(in_path, out_dir):
    out = os.path.join(out_dir, "vocals"); ensure_dir(out)
    cmd = ["python", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", out, in_path]
    try:
        subprocess.run(cmd, check=True); return out
    except Exception as e:
        print(f"[warn] Demucs 不可用（需云端 GPU），用原音频: {e}"); return in_path

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
        src = separate_vocals(path, args.out) if args.mode=="full" else path
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
