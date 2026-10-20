# 把 fma_small.zip 拆成 <2GB 分卷，同时算 SHA256 供云端合并校验
# 输出: data/upload/fma_small.zip.partNNN + data/upload/fma_small.zip.sha256
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "fma" / "fma_small.zip"
OUT_DIR = ROOT / "data" / "upload"
CHUNK_SIZE = 1_920_000_000  # 每卷 1.92GB，4 卷，均小于 2G
BUF = 8 * 1024 * 1024


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = SRC.stat().st_size
    print(f"源文件: {SRC} ({total:,} 字节)")
    sha = hashlib.sha256()
    part_idx = 0
    written = 0

    with open(SRC, "rb") as f:
        while written < total:
            part_idx += 1
            part_path = OUT_DIR / f"{SRC.name}.part{part_idx:03d}"
            target = min(CHUNK_SIZE, total - written)
            part_sha = hashlib.sha256()
            with open(part_path, "wb") as out:
                left = target
                while left > 0:
                    n = min(BUF, left)
                    buf = f.read(n)
                    if not buf:
                        raise RuntimeError("unexpected EOF")
                    out.write(buf)
                    sha.update(buf)
                    part_sha.update(buf)
                    left -= len(buf)
                    written += len(buf)
            print(f"分卷 {part_path.name}: {target:,} 字节  sha256={part_sha.hexdigest()[:16]}...")

    full_sha = sha.hexdigest()
    (OUT_DIR / f"{SRC.name}.sha256").write_text(f"{full_sha}  {SRC.name}\n", encoding="utf-8")
    print(f"完整 SHA256: {full_sha}")
    print(f"分卷数: {part_idx}，均写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
