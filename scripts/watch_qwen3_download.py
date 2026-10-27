"""本地定时看板：实时查看服务器上 Qwen3-Omni 的下载进度。

原理：
  1. 复用 `scripts/lf_remote.py` 的 SSH 凭据与连接；
  2. 每隔 N 秒查询一次服务器（目录大小 / 进程 / 状态文件 / 日志尾部）；
  3. 生成一个自包含 HTML 看板（`outputs/qwen3_download_dashboard.html`），
     页面用 meta refresh 自动刷新，数据由本脚本持续覆盖写入；
  4. 速度取最近几次采样的移动平均，避免 du 抖动。

用法：
  python scripts/watch_qwen3_download.py --once        # 只刷新一次
  python scripts/watch_qwen3_download.py --once --open # 刷新一次并打开浏览器
  python scripts/watch_qwen3_download.py --interval 30 # 常驻，每 30 秒刷新
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lf_remote import DEFAULT_ENV_FILE, connect, load_credentials, run  # noqa: E402

MODEL_DIR = "/gemini/code/lightfour-train/models/Qwen3-Omni-30B-A3B-Instruct"
STATUS_FILE = f"{MODEL_DIR}/DEPLOY_STATUS.json"
LOG_FILE = "/gemini/code/lightfour-train/logs/qwen3_omni_download_20261027.log"
STATE_FILE = REPO_ROOT / ".tmp" / "qwen3_watch_state.json"
OUT_HTML = REPO_ROOT / "outputs" / "qwen3_download_dashboard.html"

REMOTE_CMD = f"""
echo TS=$(date +%s)
echo SERVER_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo SIZE=$(du -sb {MODEL_DIR} 2>/dev/null | cut -f1)
echo CACHE=$(du -sb {MODEL_DIR}/.cache 2>/dev/null | cut -f1)
echo PID=$(pgrep -f download_qwen3_omni.py | head -1)
echo STATUS_BEGIN
cat {STATUS_FILE} 2>/dev/null
echo
echo STATUS_END
echo LOG_BEGIN
tail -c 600 {LOG_FILE} 2>/dev/null | tr '\\n' ' '
echo
echo LOG_END
echo DF=$(df -h /gemini/code 2>/dev/null | tail -1)
"""


def human_bytes(n: float) -> str:
    """字节数转人类可读。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def human_eta(seconds: float) -> str:
    """剩余秒数转 h/m。"""
    if seconds <= 0 or seconds == float("inf"):
        return "—"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def parse_kv(out: str) -> dict[str, str]:
    """解析 `KEY=value` 行；STATUS/LOG 走区间标记。"""
    data: dict[str, str] = {}
    status_lines: list[str] = []
    log_lines: list[str] = []
    section = ""
    for raw in out.splitlines():
        line = raw.rstrip()
        if line in ("STATUS_BEGIN", "LOG_BEGIN"):
            section = line[:-6].lower()
            continue
        if line in ("STATUS_END", "LOG_END"):
            section = ""
            continue
        if section == "status":
            status_lines.append(line)
        elif section == "log":
            log_lines.append(line)
        elif "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    data["STATUS_JSON"] = "\n".join(status_lines).strip()
    data["LOG_TAIL"] = " ".join(log_lines).strip()
    return data


def collect(creds: dict[str, str]) -> dict:
    """查询一次服务器并组装指标。"""
    client = connect(creds)
    try:
        code, out, err = run(client, REMOTE_CMD, timeout=180)
    finally:
        client.close()
    if code != 0 and not out:
        raise RuntimeError(f"远程查询失败 exit={code} {err[:200]}")

    kv = parse_kv(out)
    status = {}
    try:
        status = json.loads(kv.get("STATUS_JSON") or "{}")
    except json.JSONDecodeError:
        pass

    total = int(status.get("expected_bytes") or 70_530_595_574)
    size = int(kv.get("SIZE") or 0)
    cache = int(kv.get("CACHE") or 0)
    return {
        "ts": int(kv.get("TS") or time.time()),
        "server_time": kv.get("SERVER_TIME", "—"),
        "size": size,
        "cache": cache,
        "total": total,
        "pid": kv.get("PID", ""),
        "state": status.get("state", "unknown"),
        "started": status.get("started", "—"),
        "log_tail": kv.get("LOG_TAIL", ""),
        "df": kv.get("DF", ""),
    }


def update_state(sample: dict) -> dict:
    """保存最近采样，返回带速度/ETA 的指标。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {"samples": []}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text("utf-8"))
        except json.JSONDecodeError:
            pass
    samples = [s for s in state.get("samples", []) if isinstance(s, dict)]
    samples.append({"ts": sample["ts"], "size": sample["size"]})
    samples = samples[-12:]  # 最多保留最近 12 次
    STATE_FILE.write_text(json.dumps({"samples": samples}, ensure_ascii=False), "utf-8")

    speed = 0.0
    valid = [s for s in samples if s["size"] > 0]
    if len(valid) >= 2:
        dt = valid[-1]["ts"] - valid[0]["ts"]
        db = valid[-1]["size"] - valid[0]["size"]
        if dt >= 5 and db > 0:
            speed = db / dt
    sample["speed"] = speed
    remain = max(sample["total"] - sample["size"], 0)
    sample["remain"] = remain
    sample["eta"] = (remain / speed) if speed > 0 else float("inf")
    sample["pct"] = min(sample["size"] / sample["total"] * 100, 100) if sample["total"] else 0
    return sample


def render(sample: dict, interval: int) -> None:
    """写自包含 HTML 看板。"""
    pct = sample["pct"]
    bar_color = "#22c55e" if pct >= 90 else "#3b82f6" if pct >= 50 else "#f59e0b"
    running = bool(sample["pid"])
    status_text = f"运行中（PID {sample['pid']}）" if running else "未检测到下载进程"
    status_class = "ok" if running else "warn"
    speed_text = f"{human_bytes(sample['speed'])}/s" if sample["speed"] else "采样中…"
    eta_text = human_eta(sample["eta"])
    log_tail = html.escape(sample["log_tail"][-500:] or "（暂无日志）")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{interval}">
<title>Qwen3-Omni 下载进度</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle at 20% 0%, #1e293b 0%, #0b1120 55%, #060a13 100%);
    color: #e2e8f0; font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
  }}
  .card {{
    width: min(860px, 92vw); padding: 34px 38px; border-radius: 22px;
    background: rgba(15, 23, 42, .82); border: 1px solid rgba(148, 163, 184, .18);
    box-shadow: 0 24px 70px rgba(0, 0, 0, .55); backdrop-filter: blur(10px);
  }}
  h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 650; letter-spacing: .3px; }}
  .sub {{ color: #94a3b8; font-size: 13px; margin-bottom: 26px; }}
  .big {{ font-size: 42px; font-weight: 700; letter-spacing: -1px; }}
  .big span {{ font-size: 18px; color: #94a3b8; font-weight: 500; margin-left: 8px; }}
  .bar {{ height: 16px; border-radius: 99px; background: #1e293b; overflow: hidden; margin: 18px 0 10px; }}
  .bar > i {{
    display: block; height: 100%; width: {pct:.2f}%; border-radius: 99px;
    background: linear-gradient(90deg, {bar_color}, #a855f7);
    box-shadow: 0 0 18px {bar_color}; transition: width .6s ease;
  }}
  .meta {{ display: flex; justify-content: space-between; color: #94a3b8; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 24px; }}
  .cell {{ background: rgba(30, 41, 59, .65); border: 1px solid rgba(148, 163, 184, .12);
           border-radius: 14px; padding: 14px 16px; }}
  .cell b {{ display: block; font-size: 17px; margin-top: 6px; font-weight: 600; }}
  .label {{ color: #94a3b8; font-size: 12px; letter-spacing: .4px; }}
  .ok {{ color: #4ade80; }}
  .warn {{ color: #fbbf24; }}
  pre {{ margin: 18px 0 0; padding: 14px 16px; border-radius: 12px; background: #0b1120;
         border: 1px solid rgba(148, 163, 184, .12); color: #94a3b8; font-size: 12px;
         white-space: pre-wrap; word-break: break-all; max-height: 130px; overflow: auto; }}
  .foot {{ margin-top: 16px; color: #64748b; font-size: 12px; display: flex;
           justify-content: space-between; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Qwen3-Omni-30B-A3B-Instruct 下载进度</h1>
    <div class="sub">{html.escape(MODEL_DIR)} · 持久化盘 /gemini/code · 每 {interval}s 刷新</div>

    <div class="big">{pct:.2f}%<span>{human_bytes(sample['size'])} / {human_bytes(sample['total'])}</span></div>
    <div class="bar"><i></i></div>
    <div class="meta">
      <span>剩余 {human_bytes(sample['remain'])}</span>
      <span>速度 {speed_text}</span>
      <span>预计剩余 {eta_text}</span>
    </div>

    <div class="grid">
      <div class="cell"><div class="label">下载进程</div><b class="{status_class}">{html.escape(status_text)}</b></div>
      <div class="cell"><div class="label">状态文件</div><b>{html.escape(sample['state'])}</b></div>
      <div class="cell"><div class="label">暂存 .cache</div><b>{human_bytes(sample['cache'])}</b></div>
      <div class="cell"><div class="label">开始时间</div><b style="font-size:14px">{html.escape(sample['started'])}</b></div>
    </div>

    <pre>{log_tail}</pre>

    <div class="foot">
      <span>服务器时间：{html.escape(sample['server_time'])}</span>
      <span>本页刷新：{time.strftime('%H:%M:%S')}</span>
    </div>
    <div class="foot"><span>{html.escape(sample['df'])}</span><span>进度含暂存 partial，为近似值</span></div>
  </div>
</body>
</html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, "utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {pct:.2f}%  {human_bytes(sample['size'])} / "
          f"{human_bytes(sample['total'])}  速度 {speed_text}  ETA {eta_text}  -> {OUT_HTML}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30, help="刷新间隔（秒）")
    ap.add_argument("--once", action="store_true", help="只刷新一次")
    ap.add_argument("--open", action="store_true", help="生成后打开浏览器")
    args = ap.parse_args()

    creds = load_credentials(DEFAULT_ENV_FILE)
    first = True
    while True:
        try:
            sample = update_state(collect(creds))
            render(sample, args.interval)
            if first and args.open:
                webbrowser.open(OUT_HTML.resolve().as_uri())
                first = False
        except Exception as exc:  # noqa: BLE001
            print(f"[{time.strftime('%H:%M:%S')}] 采样失败：{type(exc).__name__}: {exc}")
        if args.once:
            return 0
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    raise SystemExit(main())
