"""
远程分片下载工具 —— 拼接分块 + MD5 校验

配合 jupyterlab-remote-shell 通道使用（规避 browser_evaluate 60s 超时）：

  1) 服务器切块（终端 WebSocket 执行）：
       split -b 8m -d -a 3 /gemini/code/lightfour-train/data/gen_synth_tar/shards/shard_XXXXX.tar /tmp/chunks/sXX_
  2) 浏览器分块下载（每次 8 块，约 50s 不超时）：
       fetch(base + 'api/contents/tmp/chunks/sXX_NNN?content=1&format=base64')
         -> atob -> POST http://127.0.0.1:8767/recv  (header: X-Filename=sXX_NNN)
       本地接收器： py -3.12 .upload_code_only/file_receiver.py
  3) 本地拼接 + 校验（本脚本）：
       py -3.12 scripts/fetch_remote_shard.py --prefix s22 --count 24 --out shard_00022.tar --md5 0a8dc882656159debb2fec5be738fd61

说明：md5 取值可用服务器端 `md5sum <shard>.tar` 获取。
"""
import os
import glob
import hashlib
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="分块文件名前缀，如 s22")
    ap.add_argument("--count", type=int, required=True, help="分块数量")
    ap.add_argument("--out", required=True, help="输出 tar 路径")
    ap.add_argument("--md5", default=None, help="期望的 md5（小写），用于校验")
    ap.add_argument("--dir", default=r"d:\lightfour\data\gen_synth_tar", help="分块所在目录")
    args = ap.parse_args()

    parts = []
    for i in range(args.count):
        p = os.path.join(args.dir, f"{args.prefix}_{i:03d}")
        if not os.path.exists(p):
            raise SystemExit(f"缺少分块: {p}")
        parts.append(p)

    if os.path.exists(args.out):
        os.remove(args.out)
    h = hashlib.md5()
    total = 0
    with open(args.out, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    h.update(b)
                    total += len(b)

    got = h.hexdigest()
    print(f"拼接完成: {args.out}  {total/1024/1024:.2f}MB  md5={got}")
    if args.md5:
        ok = got.lower() == args.md5.lower()
        print(f"校验: {'✅ 通过' if ok else '❌ 不匹配 期望 ' + args.md5}")
        if not ok:
            raise SystemExit(1)
    # 清理分块
    for p in parts:
        os.remove(p)
    print(f"已清理 {len(parts)} 个分块")


if __name__ == "__main__":
    main()
