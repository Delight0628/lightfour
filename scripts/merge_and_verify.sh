#!/bin/bash
# 在云服务器上合并 fma_small.zip 分卷并校验完整性
# 用法: bash merge_and_verify.sh [分卷所在目录]
# 依赖: sha256sum, cat (Linux 标配)
set -e

DIR="${1:-.}"
cd "$DIR"

echo "=== 分卷列表 ==="
ls -l fma_small.zip.part* fma_small.zip.sha256 2>/dev/null || { echo "找不到分卷文件"; exit 1; }

echo "=== 合并分卷 ==="
cat fma_small.zip.part001 fma_small.zip.part002 fma_small.zip.part003 fma_small.zip.part004 > fma_small.zip

echo "=== SHA256 校验 ==="
if sha256sum -c fma_small.zip.sha256; then
  echo "=== 校验通过，合并成功 ==="
  ls -l fma_small.zip
  echo "=== zip 自检 ==="
  python3 -c "import zipfile; z=zipfile.ZipFile('fma_small.zip'); bad=z.testzip(); print('zip OK, entries:', len(z.namelist())) if bad is None else print('BAD ENTRY:', bad)"
else
  echo "!!! 校验失败，请检查分卷是否完整上传 !!!"
  exit 1
fi
