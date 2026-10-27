"""以 12/6/6 配额渲染试听包（Eval 12-6-6 盲听协议 + PI ㉝ 的正式包）。

为什么需要这个包装器：
  `codec_rate_ab.py`（md5 407e5e69，R4 生成器，已入库 7c88438）里
  `LISTEN_POOL_WEIGHT` 写死成三池等分 `1/3`，于是 `--listen-n 24` 得到的是 **8/8/8**，
  不是协议要求的 **12/6/6**；文件内注释「--listen-n 24 → 12/6/6」与代码矛盾。
  `outputs/codec_rate_ab/listen_plan_preview.json`（11:26，quota 12/6/6、
  `_per_family_quota=12`）正是由 **未入库的 0.5/0.25/0.25 变体**跑出来的，
  git 历史里从未出现过该权重（`git log -p` 只有 `1/3`）。

  为了不篡改已冻结的生成器字节（改常量会让 md5 变、R4 指纹失效），
  本包装器只在**运行时**覆盖权重常量，生成器本体一个字节不改，其余路径（选样/渲染/
  命名/FLOAT 写出）全部走冻结实现。

用法（在训练服务器上，资产齐全）：
  /root/miniconda3/bin/python scripts/listen_pack_1266.py \
      --out-dir outputs/codec_listen_n24_1266 --listen-n 24 --threads 8

产出：24 击 × 4 臂 = 96 wav + listen/composition.json + listen/index.json，
      以及同目录的 per_hit.csv（应与 R4 冻结版 235158c7 逐字节相同）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec_rate_ab as cra  # noqa: E402

# 协议权重：镲类加量（12/6/6），其余两池各 6
cra.LISTEN_POOL_WEIGHT = {"cymbal": 0.5, "hihats": 0.25, "click_60_200ms": 0.25}

if __name__ == "__main__":
    if "--listen-n" not in sys.argv:
        sys.argv += ["--listen-n", "24"]
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]]
    raise SystemExit(cra.main())
