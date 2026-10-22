"""
B 站 beatbox 音频采集（研究用途，音频不入 git、不外传）。

背景：
  YouTube 在本机/服务器均不可达；Swissbeatbox 官网只嵌 YouTube，没有直链 mp3/mp4。
  Swissbeatbox 在 B 站有官方账号（mid=171444239），视频可用 yt-dlp 取音频。

用法：
  # 先用浏览器登录 B 站，导出 Netscape 格式 cookies 到本地（不要提交进仓库）
  python scripts/fetch_bilibili_audio.py \
    --url https://www.bilibili.com/video/BV1PXV36CEUp/ \
    --out data/bbox_raw/downloads \
    --cookies %TEMP%/bili_local_cookies.txt

  # 批量：urls.txt 每行一个 URL
  python scripts/fetch_bilibili_audio.py --url-file urls.txt --out data/bbox_raw/downloads --cookies ...

依赖：
  pip install yt-dlp imageio-ffmpeg   # 或系统已装 ffmpeg
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yt_dlp

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def load_urls(args):
    urls = list(args.url or [])
    if args.url_file:
        for line in Path(args.url_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        raise SystemExit("需要 --url 或 --url-file")
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", help="可重复")
    ap.add_argument("--url-file", default="")
    ap.add_argument("--out", default="data/bbox_raw/downloads")
    ap.add_argument("--cookies", default=os.environ.get("BILI_COOKIES", ""),
                    help="Netscape 格式 cookies 文件；B 站 412 时必需")
    ap.add_argument("--audio-format", default="wav", choices=["wav", "m4a", "mp3"])
    ap.add_argument("--keep-original", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    urls = load_urls(args)

    ffmpeg = find_ffmpeg()
    opts = {
        "outtmpl": str(out / "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": True,
        "http_headers": {"User-Agent": UA, "Referer": "https://www.bilibili.com/"},
        "quiet": False,
        "no_warnings": False,
        "keepvideo": args.keep_original,
    }
    if args.cookies and Path(args.cookies).exists():
        opts["cookiefile"] = args.cookies
    else:
        print("[warn] 未提供有效 cookies，B 站可能返回 412", flush=True)
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    if args.audio_format != "m4a":
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": args.audio_format,
            "preferredquality": "0",
        }]

    meta_path = out / "metadata.jsonl"
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            print(f"\n=== {url} ===", flush=True)
            try:
                info = ydl.extract_info(url, download=False)
                rec = {
                    "url": url,
                    "id": info.get("id"),
                    "title": info.get("title"),
                    "duration": info.get("duration"),
                    "uploader": info.get("uploader"),
                    "webpage_url": info.get("webpage_url"),
                }
                print(f"{rec['id']} | {rec['duration']}s | {rec['uploader']} | {rec['title']}", flush=True)
                ydl.download([url])
                with open(meta_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"[FAIL] {url}: {exc}", flush=True)

    print(f"\n输出目录: {out}")
    for p in sorted(out.glob("*")):
        if p.is_file():
            print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
