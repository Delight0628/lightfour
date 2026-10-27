@echo off
chcp 65001 >nul
rem 双击本文件即可启动 Qwen3-Omni 下载进度看板：
rem   1) 打开本地 HTML 看板（页面每 30 秒自动刷新）
rem   2) 在当前窗口每 30 秒通过 SSH 拉一次服务器进度
rem 关闭本窗口即停止刷新。
cd /d "%~dp0.."
start "" "%~dp0..\outputs\qwen3_download_dashboard.html"
python "%~dp0watch_qwen3_download.py" --interval 30
pause
