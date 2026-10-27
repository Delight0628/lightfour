"""Lightfour 远程服务器工具（趋动云 SSH 网关）。

用途：
  1. PI  —— 核对本地代码与服务器代码差异（`diff`），保持一致；
  2. Talker —— 训练监工（`train-status` / `tail-log` / `exec`）；
  3. 其他角色 —— 按需读取服务器产物清单（`ls` / `pull`）。

凭据来源（按优先级）：
  1) 环境变量 LIGHTFOUR_SSH_* ；
  2) 仓库根目录 `secrets.lightfour-server.env`（已在 .gitignore 中，禁止入库/外传）。

常见用法：
  python scripts/lf_remote.py test
  python scripts/lf_remote.py exec "nvidia-smi; df -h /gemini/code | tail -1"
  python scripts/lf_remote.py ls /gemini/code/lightfour-train
  python scripts/lf_remote.py diff --local . --remote /gemini/code/lightfour-train --ext .py,.sh,.md,.json
  python scripts/lf_remote.py train-status
  python scripts/lf_remote.py tail-log --file /gemini/code/lightfour-train/logs/train.log --lines 60
  python scripts/lf_remote.py pull --remote /gemini/code/lightfour-train/logs/train.log --local .tmp/train.log

已知网关行为（2026-10-26 实测）：
  - TCP 可通，banner 为 `SSH-2.0-Go`（不是 OpenSSH）；
  - 复合用户名认证通过后，若渠道未在趋动云控制台开启/已失效，服务端返回
    `必须是唯一的资产 <asset>`（exec）或 `未发现匹配的资产 <asset>`（交互 shell）。
    此时需要在控制台重新开启 SSH 渠道并更新 Host 配置。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import posixpath
import re
import sys
from pathlib import Path

import paramiko

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / "secrets.lightfour-server.env"
ASSET_ERROR_HINTS = ("必须是唯一的资产", "未发现匹配的资产")


def load_credentials(env_file: Path) -> dict[str, str]:
    creds = {
        "hostname": os.environ.get("LIGHTFOUR_SSH_HOSTNAME", ""),
        "port": os.environ.get("LIGHTFOUR_SSH_PORT", ""),
        "user": os.environ.get("LIGHTFOUR_SSH_USER", ""),
        "password": os.environ.get("LIGHTFOUR_SSH_PASSWORD", ""),
        "remote_root": os.environ.get("LIGHTFOUR_REMOTE_ROOT", ""),
    }
    if env_file.exists():
        for raw in env_file.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            mapping = {
                "LIGHTFOUR_SSH_HOSTNAME": "hostname",
                "LIGHTFOUR_SSH_PORT": "port",
                "LIGHTFOUR_SSH_USER": "user",
                "LIGHTFOUR_SSH_PASSWORD": "password",
                "LIGHTFOUR_REMOTE_ROOT": "remote_root",
            }
            if key in mapping and not creds[mapping[key]]:
                creds[mapping[key]] = value.strip()
    missing = [k for k, v in creds.items() if not v and k != "remote_root"]
    if missing:
        raise SystemExit(f"缺少凭据字段: {missing}（检查 {env_file} 或 LIGHTFOUR_SSH_* 环境变量）")
    return creds


def connect(creds: dict[str, str]) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        creds["hostname"],
        port=int(creds["port"]),
        username=creds["user"],
        password=creds["password"],
        timeout=25,
        banner_timeout=30,
        auth_timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def classify_gateway_output(text: str) -> str | None:
    for hint in ASSET_ERROR_HINTS:
        if hint in text:
            return (
                f"网关拒绝：{text.strip()}\n"
                "→ 该 SSH 资产当前不可用：请在趋动云控制台重新开启 SSH 渠道（或确认实例是否重建），"
                "并把新的 Host 配置更新到 secrets.lightfour-server.env 与 ~/.ssh/config。"
            )
    return None


def run(client: paramiko.SSHClient, command: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def run_raw(client: paramiko.SSHClient, command: str, timeout: int = 300) -> tuple[int, bytes, str]:
    """按字节读取，用于 pull 二进制/文本产物，避免 UTF-8 解码损坏。"""
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read()
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def cmd_test(creds: dict[str, str]) -> int:
    client = connect(creds)
    try:
        code, out, err = run(client, "echo LF_ALIVE; whoami; hostname; pwd; uname -r")
        hint = classify_gateway_output(out + err)
        if hint:
            print(hint)
            return 3
        print(f"exit={code}\n{out}{err}")
        code, out, _ = run(client, f"ls -la {creds.get('remote_root') or '/gemini/code'} 2>&1 | head -40")
        print(f"=== {creds.get('remote_root') or '/gemini/code'} ===\n{out}")
        code, out, _ = run(client, "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 | head -5")
        print(f"=== GPU ===\n{out}")
        return 0
    finally:
        client.close()


def cmd_exec(creds: dict[str, str], command: str, timeout: int) -> int:
    client = connect(creds)
    try:
        code, out, err = run(client, command, timeout=timeout)
        hint = classify_gateway_output(out + err)
        if hint:
            print(hint)
            return 3
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return code
    finally:
        client.close()


def cmd_ls(creds: dict[str, str], path: str) -> int:
    return cmd_exec(creds, f"ls -la {_safe_remote_path(path)}", 60)


def cmd_train_status(creds: dict[str, str], root: str) -> int:
    command = (
        f"echo '=== 进程 ==='; ps -ef | grep -E 'train|swift|accelerate|torchrun' | grep -v grep | head -20; "
        f"echo '=== GPU ==='; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv; "
        f"echo '=== 日志尾部 ==='; ls -t {root}/logs/*.log 2>/dev/null | head -3; "
        f"for f in $(ls -t {root}/logs/*.log 2>/dev/null | head -1); do echo \"--- $f\"; tail -n 30 \"$f\"; done; "
        f"echo '=== checkpoint ==='; ls -lt {root}/checkpoints 2>/dev/null | head -10"
    )
    return cmd_exec(creds, command, 120)


def cmd_tail_log(creds: dict[str, str], file: str, lines: int) -> int:
    return cmd_exec(creds, f"tail -n {lines} {file}", 60)


def cmd_pull(creds: dict[str, str], remote: str, local: str) -> int:
    remote = _safe_remote_path(remote)
    client = connect(creds)
    try:
        channels = None
        try:
            transport = client.get_transport()
            sftp = paramiko.SFTPClient.from_transport(transport)
            channels = sftp
        except Exception as exc:  # noqa: BLE001
            print(f"SFTP 不可用（{type(exc).__name__}: {exc}），回退到 cat 通道")
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        if channels is not None:
            channels.get(remote, str(target))
            channels.close()
        else:
            code, out, err = run_raw(client, f"cat {remote}")
            hint = classify_gateway_output(out.decode("utf-8", "replace") + err)
            if hint:
                print(hint)
                return 3
            if code != 0:
                print(f"拉取失败 exit={code} {err}")
                return code
            target.write_bytes(out)
        size = target.stat().st_size
        print(f"已拉取 {remote} -> {target} ({size} bytes)")
        return 0
    finally:
        client.close()


def _safe_remote_path(path: str) -> str:
    """还原 Git Bash 对单个路径参数的改写。

    MSYS2 会把看起来像绝对路径的**独立参数**改写成 Windows 路径：
        /gemini/code/x  ->  C:/Program Files/Git/gemini/code/x
    远端路径都是 POSIX 绝对路径，必须在传参前还原，
    否则远端会收到一个荒谬的 "C:/Program Files/Git/..." 路径。
    （嵌在带空格的整条命令里的路径不会被改写，所以 exec 不受影响。）
    """
    if not path:
        return path
    # /gemini/code/x -> C:/Program Files/Git/gemini/code/x
    m = re.match(r"^[A-Za-z]:[/\\](.*?)[/\\]Git([/\\].*)$", path)
    if m:
        return "/" + m.group(2).lstrip("/\\")
    # /tmp/x -> C:/Users/<u>/AppData/Local/Temp/x
    m = re.match(
        r"^[A-Za-z]:[/\\]Users[/\\][^/\\]+[/\\]AppData[/\\]Local[/\\]Temp[/\\](.*)$",
        path, re.IGNORECASE)
    if m:
        return "/tmp/" + m.group(1).replace("\\", "/")
    # 任何残留的盘符形态都直接拒绝：远端是 POSIX，写进去只会造出 C: 怪目录
    if re.match(r"^[A-Za-z]:[/\\]", path):
        raise SystemExit(
            f"拒绝上传：远端路径仍是 Windows 形态 {path!r}；"
            "请改用 POSIX 绝对路径（如 /tmp/x 或 /gemini/code/lightfour-train/x）")
    return path


def cmd_push(creds: dict[str, str], local: str, remote: str) -> int:
    """上传单个文件到服务器。

    优先 SFTP；不可用时回退 base64 分块通道。分块是因为网关对单条命令长度有限制，
    超过上限会静默截断，导致远端文件损坏。
    注意：SFTP 失败后必须换一条新连接——同一连接上后续 exec_command 会直接失败。
    """
    src = Path(local)
    if not src.is_file():
        print(f"本地文件不存在: {src}")
        return 2
    remote = _safe_remote_path(remote)
    data = src.read_bytes()

    try:
        client = connect(creds)
    except Exception as exc:  # noqa: BLE001
        print(f"连接失败: {type(exc).__name__}: {exc}")
        return 3
    try:
        rdir = posixpath.dirname(remote)
        if rdir:
            run(client, f"mkdir -p {rdir}", 60)
        try:
            transport = client.get_transport()
            sftp = paramiko.SFTPClient.from_transport(transport)
            with sftp.file(remote, "wb") as fh:
                fh.write(data)
            sftp.close()
            sftp_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"SFTP 不可用（{type(exc).__name__}: {exc}），回退到 base64 分块通道")
            sftp_ok = False
    finally:
        client.close()

    if not sftp_ok:
        client = connect(creds)
        try:
            b64 = base64.b64encode(data).decode("ascii")
            tmp = remote + ".lfb64.txt"
            run(client, f"rm -f {tmp}", 60)
            chunk = 3000
            for i in range(0, len(b64), chunk):
                piece = b64[i:i + chunk]
                code, out, err = run(client, f"printf %s '{piece}' >> {tmp}", 120)
                if code != 0:
                    print(f"base64 分块上传失败 exit={code} out={out[:120]!r} err={err[:200]!r} "
                          f"cmd=printf {len(piece)}B >> {tmp}")
                    return code
            code, out, err = run(client, f"base64 -d {tmp} > {remote} && rm -f {tmp}", 120)
            if code != 0:
                print(f"base64 解码失败 exit={code} {err}")
                return code
        finally:
            client.close()

    # 回读校验 md5，避免"以为传上去了"
    client = connect(creds)
    try:
        code, out, err = run(client, f"md5sum {remote}", 60)
        hint = classify_gateway_output(out + err)
        if hint:
            print(hint)
            return 3
        remote_md5 = out.strip().split()[0] if out.strip() else ""
        local_md5 = hashlib.md5(data).hexdigest()
        if remote_md5 != local_md5:
            print(f"md5 不一致: 本地 {local_md5} vs 远端 {remote_md5}")
            return 4
        print(f"已上传 {src} -> {remote} ({len(data)} bytes, md5={local_md5} 一致)")
        return 0
    finally:
        client.close()


CODE_EXTS_DEFAULT = ".py,.sh,.md,.json,.yaml,.yml,.toml,.ini,.cfg"
CODE_EXCLUDES_DEFAULT = ".git,data,checkpoints,logs,backups,lightfour,omni_voice_test,.tmp,__pycache__,.venv,venv,node_modules"


def local_hashes(local_root: Path, exts: set[str], excludes: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(local_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(local_root).as_posix()
        if any(part in excludes for part in Path(rel).parts):
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        result[rel] = digest
    return result


def cmd_diff(creds: dict[str, str], local: str, remote: str, exts: str, excludes: str) -> int:
    remote = _safe_remote_path(remote)
    ext_set = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
               for e in exts.split(",") if e.strip()}
    exclude_set = {e.strip().strip("/") for e in excludes.split(",") if e.strip()}
    local_root = Path(local).resolve()
    if not local_root.is_dir():
        raise SystemExit(f"本地目录不存在: {local_root}")

    names = " -o ".join(f"-name '*{e}'" for e in sorted(ext_set))
    remote_excludes = " ".join(f"-not -path './{e}/*'" for e in sorted(exclude_set))
    remote_cmd = (
        f"cd {remote} 2>/dev/null || exit 9; "
        f"find . -type f \\( {names} \\) {remote_excludes} -print0 | "
        f"xargs -0 md5sum 2>/dev/null"
    )
    client = connect(creds)
    try:
        code, out, err = run(client, remote_cmd, timeout=300)
        hint = classify_gateway_output(out + err)
        if hint:
            print(hint)
            return 3
        if code == 9:
            print(f"远端目录不存在: {remote}")
            return 9
    finally:
        client.close()

    remote_map: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        digest, path = parts
        rel = posixpath.normpath(path).lstrip("./")
        remote_map[rel] = digest

    local_map = local_hashes(local_root, ext_set, exclude_set)

    only_local = sorted(set(local_map) - set(remote_map))
    only_remote = sorted(set(remote_map) - set(local_map))
    changed = sorted(rel for rel in (set(local_map) & set(remote_map)) if local_map[rel] != remote_map[rel])

    print(f"本地 {local_root} 文件数={len(local_map)} | 远端 {remote} 文件数={len(remote_map)}")
    print(f"仅本地有 ({len(only_local)}):")
    for rel in only_local[:60]:
        print(f"  + {rel}")
    if len(only_local) > 60:
        print(f"  … 其余 {len(only_local) - 60} 条未打印（未打印 ≠ 不存在，需定点核对路径）")
    print(f"仅远端有 ({len(only_remote)}):")
    for rel in only_remote[:60]:
        print(f"  - {rel}")
    if len(only_remote) > 60:
        print(f"  … 其余 {len(only_remote) - 60} 条未打印（未打印 ≠ 不存在，需定点核对路径）")
    print(f"内容不一致 ({len(changed)}):")
    for rel in changed[:60]:
        print(f"  ~ {rel}")
    if len(changed) > 60:
        print(f"  … 其余 {len(changed) - 60} 条未打印（未打印 ≠ 不存在，需定点核对路径）")
    if len(local_map) and len(remote_map) and not (only_local or only_remote or changed):
        print("结论：一致 ✅")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightfour 远程服务器工具")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="凭据文件路径")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("test", help="连通性 + 环境探测")

    p_exec = sub.add_parser("exec", help="在服务器执行命令")
    p_exec.add_argument("command")
    p_exec.add_argument("--timeout", type=int, default=300)

    p_ls = sub.add_parser("ls", help="列目录")
    p_ls.add_argument("path")

    p_status = sub.add_parser("train-status", help="训练监工快照")
    p_status.add_argument("--root", default="")

    p_tail = sub.add_parser("tail-log", help="查看日志尾部")
    p_tail.add_argument("--file", required=True)
    p_tail.add_argument("--lines", type=int, default=50)

    p_pull = sub.add_parser("pull", help="从服务器拉取文件")
    p_pull.add_argument("--remote", required=True)
    p_pull.add_argument("--local", required=True)

    p_push = sub.add_parser("push", help="上传单个文件到服务器")
    p_push.add_argument("--local", required=True)
    p_push.add_argument("--remote", required=True)

    p_diff = sub.add_parser("diff", help="本地/服务器代码差异")
    p_diff.add_argument("--local", default=".")
    p_diff.add_argument("--remote", required=True)
    p_diff.add_argument("--ext", default=CODE_EXTS_DEFAULT)
    p_diff.add_argument("--exclude", default=CODE_EXCLUDES_DEFAULT,
                        help="远端/本地都跳过的目录名，逗号分隔")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    creds = load_credentials(Path(args.env_file))
    if args.action == "test":
        return cmd_test(creds)
    if args.action == "exec":
        return cmd_exec(creds, args.command, args.timeout)
    if args.action == "ls":
        return cmd_ls(creds, args.path)
    if args.action == "train-status":
        return cmd_train_status(creds, args.root or creds.get("remote_root") or "/gemini/code/lightfour-train")
    if args.action == "tail-log":
        return cmd_tail_log(creds, args.file, args.lines)
    if args.action == "pull":
        return cmd_pull(creds, args.remote, args.local)
    if args.action == "push":
        return cmd_push(creds, args.local, args.remote)
    if args.action == "diff":
        return cmd_diff(creds, args.local, args.remote, args.ext, args.exclude)
    raise SystemExit(f"未知动作: {args.action}")


if __name__ == "__main__":
    sys.exit(main())
