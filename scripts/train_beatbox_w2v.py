"""
Stage A v4 — wav2vec2 预训练特征 + 可学习层权重分类头（本地 CPU 可跑）
动机：log-mel CNN 在跨人泛化上卡在 test~0.57（军鼓/开镲混淆）；
      wav2vec2 预训练表征对嗓音差异更鲁棒（Delgado 论文的 embedding 思路）。
做法：冻结 wav2vec2 base（只前向）→ 缓存 12 层 hidden 的 mean+std 池化
      → 训练一个带【可学习层权重】的轻量分类头（秒级迭代）
运行： py -3.12 -u scripts/train_beatbox_w2v.py --extract
"""
import os, json, argparse, random, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(ROOT, "data", "sft", "beatbox_clips")
FEAT_DIR = os.path.join(ROOT, "data", "feat_w2v")
OUT_DIR = os.path.join(ROOT, "checkpoints", "beatbox_w2v")
os.makedirs(FEAT_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

W2V_SR = 16000
PAD_SEC = 1.5
CLASSES = ["kd", "sd", "hhc", "hho"]
CLASS_CN = {"kd": "底鼓", "sd": "军鼓", "hhc": "闭镲", "hho": "开镲"}
TEST_PARTS = {"P9", "P11", "P19", "P24"}
VAL_PARTS = {"P20", "P21"}
N_LAYERS = 12
HID = 768

random.seed(42); np.random.seed(42); torch.manual_seed(42)

def collect():
    splits = defaultdict(list)
    for fn in sorted(os.listdir(CLIP_DIR)):
        if not fn.endswith(".wav"): continue
        p = fn.split("_")[0]; c = fn[:-4].split("_")[-1]
        if c not in CLASSES: continue
        sp = "test" if p in TEST_PARTS else ("val" if p in VAL_PARTS else "train")
        splits[sp].append((os.path.join(CLIP_DIR, fn), c, p))
    return splits

def load_16k(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != W2V_SR:
        n = int(len(wav) * W2V_SR / sr)
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    n_pad = int(PAD_SEC * W2V_SR)
    if len(wav) > n_pad: wav = wav[:n_pad]
    else: wav = np.pad(wav, (0, n_pad - len(wav)))
    w = torch.from_numpy(np.ascontiguousarray(wav))
    w = (w - w.mean()) / (w.std() + 1e-6)     # wav2vec2 期望零均值单位方差
    return w

@torch.no_grad()
def extract(splits, batch=16):
    model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
    model.eval()
    feats = {}
    t0 = time.time()
    for sp, items in splits.items():
        feats_buf = np.zeros((len(items), N_LAYERS, HID * 2), dtype="float16")
        for s in range(0, len(items), batch):
            chunk = items[s:s + batch]
            wavs = torch.stack([load_16k(p) for p, _, _ in chunk])
            out = model.extract_features(wavs, num_layers=N_LAYERS)
            layers = out[0] if isinstance(out, tuple) else out      # tuple(list[12], None)
            for li, h in enumerate(layers):
                m = h.mean(dim=1); sd = h.std(dim=1)
                outf = torch.cat([m, sd], dim=1).to(torch.float16).numpy()
                feats_buf[s:s + len(chunk), li] = outf
            if (s // batch) % 20 == 0:
                done = s + len(chunk)
                print(f"  {sp} {done}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
        feats[sp] = feats_buf
        print(f"{sp} 特征完成: {feats_buf.shape}", flush=True)
    np.savez_compressed(os.path.join(FEAT_DIR, "w2v_feats.npz"),
                        **{k: v for k, v in feats.items()})
    meta = {sp: [(c, p) for _, c, p in items] for sp, items in splits.items()}
    with open(os.path.join(FEAT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return feats, meta

class Head(nn.Module):
    def __init__(self, n_layers=N_LAYERS, hid=HID, n=4):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(hid * 2)
        self.mlp = nn.Sequential(nn.Linear(hid * 2, 256), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, n))
    def forward(self, x):            # x: (B, L, hid*2)
        w = torch.softmax(self.lw, 0).view(1, -1, 1)
        z = (x * w).sum(1)
        return self.mlp(self.norm(z))

def run_head(feats, meta, epochs=40):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ys = {sp: [CLASSES.index(c) for c, _ in meta[sp]] for sp in meta}
    cnt = Counter(ys["train"])
    cw = torch.tensor([len(ys["train"]) / (len(CLASSES) * cnt[CLASSES.index(c)]) for c in CLASSES],
                      dtype=torch.float32, device=dev)
    X = {sp: torch.from_numpy(feats[sp]).float().to(dev) for sp in feats}
    Y = {sp: torch.tensor(ys[sp], device=dev) for sp in ys}
    model = Head().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    best, best_ep, best_state = 0.0, 0, None
    n = X["train"].shape[0]; bs = 128
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = X["train"][idx], Y["train"][idx]
            lam, pm = 0.8, torch.randperm(xb.size(0), device=dev)
            mix = torch.rand(1).item() < 0.5
            if mix: xb = lam * xb + (1 - lam) * xb[pm]
            opt.zero_grad(); out = model(xb)
            loss = (lam * crit(out, yb) + (1 - lam) * crit(out, yb[pm])) if mix else crit(out, yb)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X["val"]).argmax(1) == Y["val"]).float().mean().item()
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:02d} val {acc:.4f} best {best:.4f}(ep{best_ep}) layerw "
                  f"{torch.softmax(model.lw,0).detach().cpu().numpy().round(2).tolist()}", flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(X["test"]).argmax(1)
        cm = np.zeros((4, 4), dtype=int)
        for t, p in zip(Y["test"].tolist(), pred.tolist()): cm[t, p] += 1
        valpred = model(X["val"]).argmax(1)
        vacc = (valpred == Y["val"]).float().mean().item()
    tacc = np.trace(cm) / max(cm.sum(), 1)
    print("\n" + "=" * 60, flush=True)
    print(f"最佳 ep{best_ep}  val={vacc:.4f}  test={tacc:.4f}  (随机 0.25, v1=0.5008, v2=0.5721)", flush=True)
    for i, c in enumerate(CLASSES):
        print(f"  {CLASS_CN[c]:4s} {cm[i,i]/max(cm[i].sum(),1):.4f} ({cm[i,i]}/{cm[i].sum()})", flush=True)
    print("混淆矩阵:", flush=True); print(cm, flush=True)
    torch.save(best_state, os.path.join(OUT_DIR, "head.pt"))
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"val_acc": vacc, "test_acc": float(tacc), "best_epoch": best_ep,
                   "per_class_acc": {CLASS_CN[c]: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
                   "confusion_matrix": cm.tolist(), "class_order": CLASSES}, f, ensure_ascii=False, indent=2)
    print(f"-> {OUT_DIR}", flush=True)

def run_cv(feats, meta, n_folds=4, epochs=40, seed=0):
    """k-fold 参与者交叉验证：解决 2 人 val 集系统性高估 0.20 的问题"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 汇总全部样本 + 参与者
    Xs, Ys, Ps = [], [], []
    for sp in feats:
        Xs.append(torch.from_numpy(feats[sp]).float())
        Ys += [CLASSES.index(c) for c, _ in meta[sp]]
        Ps += [p for _, p in meta[sp]]
    X = torch.cat(Xs).to(dev)
    Y = torch.tensor(Ys, device=dev)
    parts = sorted(set(Ps))
    rng = random.Random(seed); rng.shuffle(parts)
    folds = [parts[i::n_folds] for i in range(n_folds)]
    accs, per_cls_acc = [], np.zeros(4)
    for fi, held in enumerate(folds):
        hset = set(held)
        tr_idx = [i for i, p in enumerate(Ps) if p not in hset]
        te_idx = [i for i, p in enumerate(Ps) if p in hset]
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xte, Yte = X[te_idx], Y[te_idx]
        cnt = Counter(Ytr.tolist())
        cw = torch.tensor([len(Ytr) / (len(CLASSES) * max(cnt.get(i, 1), 1)) for i in range(4)],
                          dtype=torch.float32, device=dev)
        torch.manual_seed(100 + fi)
        model = Head().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
        n = Xtr.shape[0]; bs = 128
        for ep in range(epochs):
            model.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]
                xb, yb = Xtr[idx], Ytr[idx]
                lam, pm = 0.8, torch.randperm(xb.size(0), device=dev)
                mix = torch.rand(1).item() < 0.5
                if mix: xb = lam * xb + (1 - lam) * xb[pm]
                opt.zero_grad(); out = model(xb)
                loss = (lam * crit(out, yb) + (1 - lam) * crit(out, yb[pm])) if mix else crit(out, yb)
                loss.backward(); opt.step()
            sch.step()
        model.eval()
        with torch.no_grad():
            pr = model(Xte).argmax(1)
            acc = (pr == Yte).float().mean().item()
            cm = np.zeros((4, 4), dtype=int)
            for t, p in zip(Yte.tolist(), pr.tolist()): cm[t, p] += 1
        accs.append(acc)
        for i in range(4):
            if cm[i].sum(): per_cls_acc[i] += cm[i, i] / cm[i].sum() / n_folds
        print(f"  fold{fi} 留出 {len(held)} 人  acc={acc:.4f}  "
              f"percls[{' '.join(f'{CLASS_CN[CLASSES[i]]}:{cm[i,i]/max(cm[i].sum(),1):.2f}' for i in range(4))}]", flush=True)
    print(f"\n=== {n_folds}-fold 参与者 CV ===", flush=True)
    print(f"mean={np.mean(accs):.4f}  std={np.std(accs):.4f}  folds={[round(a,4) for a in accs]}", flush=True)
    print(f"平均每类: " + " ".join(f"{CLASS_CN[CLASSES[i]]}:{per_cls_acc[i]:.4f}" for i in range(4)), flush=True)
    return float(np.mean(accs)), float(np.std(accs)), per_cls_acc.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="先做特征提取")
    args = ap.parse_args()
    splits = collect()
    for k, v in splits.items(): print(f"{k}: {len(v)}", flush=True)
    fpath = os.path.join(FEAT_DIR, "w2v_feats.npz")
    if args.extract or not os.path.exists(fpath):
        feats, meta = extract(splits)
    else:
        z = np.load(fpath); feats = {k: z[k] for k in z.files}
        meta = json.load(open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8"))
        print("使用已缓存特征", flush=True)
    run_head(feats, meta)
    print("\n\n########## k-fold 参与者交叉验证（可靠评测） ##########", flush=True)
    cv_mean, cv_std, cv_pc = run_cv(feats, meta, n_folds=4)
    with open(os.path.join(OUT_DIR, "report_cv.json"), "w", encoding="utf-8") as f:
        json.dump({"cv_mean": cv_mean, "cv_std": cv_std, "per_class_avg": cv_pc,
                   "class_order": CLASSES}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
