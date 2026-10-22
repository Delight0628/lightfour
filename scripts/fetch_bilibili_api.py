"""
B 站 API 直链音频下载器（服务器/本地通用，绕过 yt-dlp 视频页 412）。

用法：
  python scripts/fetch_bilibili_api.py \
    --url-file data/bbox_raw/bilibili_urls.txt \
    --out data/bbox_raw/downloads \
    --cookies data/bbox_raw/bili_cookies.txt \
    --limit 50

说明：
  - 先用 curl 访问 https://www.bilibili.com/ 生成 cookies（buvid3 等）；
  - view API 取 cid，playurl API 取 dash.audio 最高码率；
  - 直接下载 m4a，再用 ffmpeg 转 wav；
  - 只做研究用途，音频不入 git、不外传。
"""
import argparse
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def cookie_header(path):
    parts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 7:
            parts.append(f"{f[5]}={f[6]}")
    return "; ".join(parts)


def get_json(url, bv, cookie):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Referer": f"https://www.bilibili.com/video/{bv}/",
        "Cookie": cookie,
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def find_bv(url):
    m = re.search(r"(BV[0-9A-Za-z]+)", url)
    if not m:
        raise ValueError(f"无法从 URL 解析 BV 号: {url}")
    return m.group(1)


def download(url, out, cookie, fmt, ffmpeg):
    bv = find_bv(url)
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / f"{bv}.wav"
    m4a_path = out / f"{bv}.m4a"
    if wav_path.exists() and wav_path.stat().st_size > 10000:
        print(f"[skip] {bv} 已存在")
        return {"bvid": bv, "url": url, "wav": str(wav_path), "skipped": True}

    view = get_json(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}", bv, cookie)
    if view.get("code") != 0:
        raise RuntimeError(f"view API {view.get('code')}: {view.get('message')}")
    data = view["data"]
    cid = data["cid"]
    title = data.get("title", "")
    duration = data.get("duration", 0)
    owner = data.get("owner", {}).get("name", "")

    play = get_json(
        f"https://api.bilibili.com/x/player/playurl?bvid={bv}&cid={cid}&fnval=16&fourk=0",
        bv, cookie,
    )
    if play.get("code") != 0:
        raise RuntimeError(f"playurl API {play.get('code')}: {play.get('message')}")
    audios = (play.get("data", {}).get("dash", {}) or {}).get("audio", [])
    if not audios:
        raise RuntimeError("playurl 没有 dash.audio")
    audios.sort(key=lambda x: x.get("bandwidth", 0), reverse=True)
    best = audios[0]

    req = urllib.request.Request(best["baseUrl"], headers={
        "User-Agent": UA,
        "Referer": f"https://www.bilibili.com/video/{bv}/",
        "Cookie": cookie,
    })
    with urllib.request.urlopen(req, timeout=60) as r, open(m4a_path, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)

    if fmt == "wav":
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，无法转 wav")
        subprocess.run([ffmpeg, "-y", "-i", str(m4a_path), str(wav_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        m4a_path.unlink(missing_ok=True)
    else:
        wav_path = m4a_path

    print(f"[ok] {bv} | {duration}s | {owner} | {title[:50]}")
    return {
        "bvid": bv, "url": url, "title": title, "duration": duration,
        "uploader": owner, "wav": str(wav_path), "bandwidth": best.get("bandwidth"),
        "skipped": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-file", required=True)
    ap.add_argument("--out", default="data/bbox_raw/downloads")
    ap.add_argument("--cookies", required=True)
    ap.add_argument("--format", default="wav", choices=["wav", "m4a"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cookie = cookie_header(args.cookies)
    ffmpeg = shutil.which("ffmpeg")
    urls = [l.strip() for l in Path(args.url_file).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    if args.limit:
        urls = urls[: args.limit]
    print(f"URLs={len(urls)} ffmpeg={ffmpeg}")

    meta_path = out / "metadata.jsonl"
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}", flush=True)
        try:
            rec = download(url, out, cookie, args.format, ffmpeg)
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[FAIL] {exc}", flush=True)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
