"""
Omni v4 上云打包器
做三件事：
  1) 把 data/sft_v4/*.jsonl 里的本地绝对路径改写成服务器路径
  2) 打包音频（beatbox_windows wav + 引用的 FMA mp3）成 <2G 分卷
  3) 生成服务器端 deploy + train + eval 脚本
服务器端基准目录：/gemini/code/lightfour-train/
运行： py -3.12 scripts/package_omni_v4.py
"""
import os, json, shutil, zipfile, hashlib
from pathlib import Path

ROOT = Path(r"D:\lightfour")
SFT = ROOT / "data" / "sft_v4"
PKG = ROOT / "data" / "upload_v4"
PKG.mkdir(parents=True, exist_ok=True)
SERVER_BASE = "/gemini/code/lightfour-train/data"

# ---------- 1. 路径改写 ----------
def rewrite():
    mapping = [
        (str(ROOT / "data" / "sft_v4" / "beatbox_windows").replace("\\", "/"),
         f"{SERVER_BASE}/beatbox_windows"),
        (str(ROOT / "data" / "fma" / "fma_small").replace("\\", "/"),
         f"{SERVER_BASE}/fma_small"),
    ]
    out_dir = PKG / "sft"
    out_dir.mkdir(exist_ok=True)
    stats = {}
    for name in ("train", "eval", "test"):
        src = SFT / f"{name}.jsonl"
        recs = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        n = 0
        for r in recs:
            aud = r["messages"][0]["audios"][0]
            new = aud
            for lo, hi in mapping:
                if aud.startswith(lo):
                    new = hi + aud[len(lo):]
                    break
            r["messages"][0]["audios"][0] = new
            n += 1
        dst = out_dir / f"{name}.jsonl"
        with dst.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = n
        print(f"{name}: {n} 条 -> {dst}")
    return stats

# ---------- 2. 音频打包 ----------
def collect_audio():
    need = set()
    for name in ("train", "eval", "test"):
        for l in (PKG / "sft" / f"{name}.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            need.add(r["messages"][0]["audios"][0])
    local = []
    for srv in need:
        rel = srv.replace(f"{SERVER_BASE}/", "")
        local.append(ROOT / "data" / "sft_v4" / rel if rel.startswith("beatbox_windows")
                     else ROOT / "data" / "fma" / rel)     # rel 已是 fma_small/...
    missing = [p for p in local if not p.exists()]
    print(f"引用音频 {len(local)} 个，缺失 {len(missing)}")
    if missing[:3]: print("  缺失样例:", missing[:3])
    return [p for p in local if p.exists()]

def make_zips(files, tag):
    """打成 <2G 的 zip 分卷（wav/mp3 已压缩，用 STORED 省 CPU）"""
    part = PKG / f"{tag}.zip"
    maxsize = 1_900_000_000
    zips, cur, cur_size = [], zipfile.ZipFile(part, "w", zipfile.ZIP_STORED), 0
    for i, f in enumerate(files):
        rel = f.relative_to(ROOT).as_posix()
        zips.append(cur)
        cur.write(f, arcname=rel)
        cur_size += f.stat().st_size
        if cur_size >= maxsize:
            cur.close()
            nxt = PKG / f"{tag}.zip.{len(zips)}"
            cur = zipfile.ZipFile(nxt, "w", zipfile.ZIP_STORED)
            cur_size = 0
            print(f"  {tag}: 写满一卷 -> {nxt.name}")
    cur.close()
    # 命名成 .part001 形式
    outs = sorted(PKG.glob(f"{tag}.zip*"))
    for i, o in enumerate(outs, 1):
        o.rename(PKG / f"{tag}.part{i:03d}.zip")
    # sha256
    h = hashlib.sha256()
    for o in sorted(PKG.glob(f"{tag}.part*.zip")):
        with o.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    (PKG / f"{tag}.sha256").write_text(h.hexdigest() + f"  {tag}.merged.zip\n", encoding="utf-8")
    return len(outs)

# ---------- 3. 服务器端脚本 ----------
DEPLOY = r'''#!/bin/bash
# 服务器端：合并分卷 -> 解压 -> 校验 -> 改写 jsonl 路径
set -e
BASE=/gemini/code/lightfour-train
UP=/gemini/data-1            # 只读上传区
WORK=$BASE/data
mkdir -p $WORK && cd $WORK

echo "== 合并分卷 =="
cat $UP/omni_audio.part*.zip > $WORK/omni_audio.merged.zip
echo "$WORK/omni_audio.merged.zip  $(cat $UP/omni_audio.sha256 | awk '{print $1}')" | sha256sum -c -
echo "== 解压 =="
unzip -oq $WORK/omni_audio.merged.zip -d $BASE/
rm -f $WORK/omni_audio.merged.zip
echo "== 拷 jsonl =="
mkdir -p $WORK/sft && cp $UP/sft/*.jsonl $WORK/sft/
echo "== 抽检音频可达性 =="
head -200 $WORK/sft/train.jsonl | python3 -c "
import sys, json, os
miss=0; tot=0
for l in sys.stdin:
    p=json.loads(l)['messages'][0]['audios'][0]; tot+=1
    if not os.path.exists(p): miss+=1
print(f'抽样 {tot} 条, 缺失 {miss}')
"
echo "== 持久化目录用量 =="
du -sh $BASE
'''

TRAIN = r'''#!/bin/bash
# Omni v4 QLoRA 训练（v3 获胜配置 + v4 数据）
BASE=/gemini/code/lightfour-train
export OMP_NUM_THREADS=4
nohup swift sft \
  --model /gemini/pretrain \
  --tuner_type lora --tuner_backend peft \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit false --freeze_aligner false \
  --quant_method bnb --quant_bits 4 \
  --dataset $BASE/data/sft/train.jsonl \
  --val_dataset $BASE/data/sft/eval.jsonl \
  --max_length 8192 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --bf16 true --gradient_checkpointing true \
  --lazy_tokenize true --dataset_num_proc 4 --dataloader_num_workers 2 \
  --save_strategy steps --save_steps 300 --save_total_limit 10 \
  --eval_strategy steps --eval_steps 300 \
  --logging_steps 5 --report_to tensorboard \
  --output_dir $BASE/checkpoints/omni3b-edm-v4 \
  --seed 42 > $BASE/logs/train_v4.log 2>&1 &
echo "训练已启动 PID=$!"
'''

if __name__ == "__main__":
    stats = rewrite()
    files = collect_audio()
    print(f"打包 {len(files)} 个音频文件")
    n = make_zips(files, "omni_audio")
    (PKG / "deploy.sh").write_text(DEPLOY, encoding="utf-8")
    (PKG / "train_omni_v4.sh").write_text(TRAIN, encoding="utf-8")
    print("\n=== 上传清单 ===")
    for f in sorted(PKG.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(PKG).as_posix():45s} {f.stat().st_size/1e6:9.1f} MB")
    print(f"\n分卷数: {n}  总体积: {sum(f.stat().st_size for f in PKG.rglob('*') if f.is_file())/1e9:.2f} GB")
    print("上传到 /gemini/data-1/ 后，服务器执行: bash deploy.sh && bash train_omni_v4.sh")
