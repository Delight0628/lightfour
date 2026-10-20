"""
Stage C — 伪标注脚本（识别模型当自动标注员）
输入：preprocess_video.py 产出的片段（manifest + wav）+ 训练好的分类器
输出：带置信度的伪标注鼓谱 jsonl（供 Stage D 扩库 / Stage E 生成用）

四条对策落地：
  1. 置信度过滤：低于阈值的片段弃掉（应对未知音色强行归类）
  2. 抽检格式：导出待人工抽检样本（--audit 比例）
  3. 按置信度加权：confidence 字段供训练时加权
  4. 版权：只写标注结果，不复制音频

用法：
  python pseudo_label.py --manifest data/bbox_raw/segments_manifest.jsonl \
      --ckpt checkpoints/beatbox_cls/best.pt --conf 0.85 --audit 0.05 --out data/bbox_pseudo
"""
import os, json, argparse, random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd":"底鼓(Kick)","sd":"军鼓(Snare)","hhc":"闭镲(Closed Hi-Hat)","hho":"开镲(Open Hi-Hat)"}
SR=22050; N_MELS=64; N_FFT=1024; HOP=256
MAX_FRAMES=int(1.5*SR/HOP)+1
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,
    n_mels=N_MELS,f_min=40,f_max=SR//2)

class MelCNN(nn.Module):
    def __init__(self,n=4):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)))
        self.classifier=nn.Sequential(nn.Dropout(0.3),nn.Linear(128,n))
    def forward(self,x):
        if x.dim()==3: x=x.unsqueeze(1)
        return self.classifier(self.features(x).flatten(1))

def load_logmel(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim>1: wav=wav.mean(axis=1)
    if sr!=SR:
        n=int(len(wav)*SR/sr)
        wav=np.interp(np.linspace(0,1,num=n,endpoint=False),
                      np.linspace(0,1,num=len(wav),endpoint=False), wav).astype("float32")
    t=torch.from_numpy(wav)
    if t.numel()<16: t=F.pad(t,(0,16-t.numel()))
    lm=torch.log(mel_tf(t)+1e-6)
    fr=lm.shape[-1]
    lm = F.pad(lm,(0,MAX_FRAMES-fr)) if fr<MAX_FRAMES else lm[:,:MAX_FRAMES]
    return (lm-lm.mean())/(lm.std()+1e-6)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conf", type=float, default=0.85, help="置信度阈值（低于弃掉）")
    ap.add_argument("--audit", type=float, default=0.05, help="人工抽检比例")
    ap.add_argument("--out", default="data/bbox_pseudo")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=MelCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"分类器已加载 ({device})  置信度阈值={args.conf}")

    entries=[json.loads(l) for l in open(args.manifest,encoding="utf-8") if l.strip()]
    print(f"待标注片段: {len(entries)}")

    kept=[]; dropped=0
    for e in entries:
        wp=e.get("wav")
        if not wp or not os.path.exists(wp): continue
        x=load_logmel(wp).unsqueeze(0).to(device)
        with torch.no_grad():
            prob=F.softmax(model(x),1).squeeze()
        conf, idx = prob.max().item(), prob.argmax().item()
        label=CLASSES[idx]
        rec={**e, "pseudo_label":label, "label_cn":CLASS_CN[label],
             "confidence":round(conf,4), "probs":{CLASSES[i]:round(prob[i].item(),4) for i in range(4)}}
        if conf>=args.conf: kept.append(rec)
        else: dropped+=1

    print(f"\n通过置信度过滤: {len(kept)}  弃掉: {dropped}  保留率 {len(kept)/max(len(entries),1):.1%}")

    # 抽检样本（人工核对）
    random.shuffle(kept)
    audit_n=max(1,int(len(kept)*args.audit))
    audit=kept[:audit_n]; train_pool=kept  # 全部保留，audit 只是标记待人工核对

    # 保存
    def dump(obj, fn):
        with open(os.path.join(args.out,fn),"w",encoding="utf-8") as f:
            for r in obj: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    dump(kept, "pseudo_labels.jsonl")
    dump(audit, "audit_sample.jsonl")

    # 统计
    from collections import Counter
    dist=Counter(r["pseudo_label"] for r in kept)
    print(f"伪标注分布: { {CLASS_CN[k]:v for k,v in dist.items()} }")
    print(f"平均置信度: {np.mean([r['confidence'] for r in kept]):.4f}")
    print(f"\n输出: {args.out}/pseudo_labels.jsonl ({len(kept)} 条)")
    print(f"待人工抽检: {args.out}/audit_sample.jsonl ({audit_n} 条, 建议错误率<=10%才继续滚库)")
    print("下一步（Stage D）: 扩类目 + 汇总成生成用鼓谱序列")

if __name__=="__main__":
    main()
