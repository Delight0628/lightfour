#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ast_lin_v0：冻结 AST + 线性探针（Lightfour 首个识别预测源，仪器校准用）。

设计（Eval 锁定 v0，2026-10-27）：
  * 骨干  : MIT/ast-finetuned-audioset-10-10-0.4593（rev f826b80d），**完全冻结**，
            取 ASTModel 的 pooler_output（768 维）——即官方分类头 ASTMLPHead 吃的那一个向量。
  * 输入  : 任意时长音频 → 重采样 16 kHz 单声道 → ASTFeatureExtractor(max_length=1024)
            定长 1024 帧 fbank。**不能改短**：该 checkpoint 的 position_embeddings
            是 2+12×101=1214 固定长度，ASTEmbeddings 不做插值，输入帧数必须是 1024。
  * 探针  : StandardScaler（只在训练集上 fit）+ sklearn LogisticRegression
            （lbfgs / L2 / C=1.0 / max_iter=5000 / 无 class_weight）。
            **不做任何超参搜索**，避免偷看测试集。
  * 训练集: data/baseline_train_index.csv（319 行，label_index 0..34，其中
            label_index 5 = sid 25 `K\`` 与 label_index 34 = unknown 训练样本 0，
            故实际可学 33 类）。
  * 测试集: data/eval_refaudio_testset.csv（95 条 seedlib 零伪标参考音频），
            其中 in-vocab 34 条 / out_of_vocab 61 条，由 eval_instrument_calibration.py 分账。

红线：
  * 该模型不是 103 音色识别器，任何数字**不得写入 agent.md §0 识别准确率**。
  * 不得用训练集重建冒充泛化；本脚本只在 seedlib 参考音频上出 preds.csv。

用法:
  python scripts/train_ast_lin_v0.py --stage featurize --set train
  python scripts/train_ast_lin_v0.py --stage featurize --set probe
  python scripts/train_ast_lin_v0.py --stage featurize --set perturb
  python scripts/train_ast_lin_v0.py --stage train
  python scripts/train_ast_lin_v0.py --stage predict
  python scripts/train_ast_lin_v0.py --stage all

产物（--out-dir 默认 outputs/instr_cal_ast_lin_v0）：
  preds.csv / perturb_preds.csv / model_ast_lin_v0.npz / run_meta.json
特征缓存：.cache/ast_baseline/emb/<set>__<id>.npy  （逐条落盘，断点可续）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AST_DIR = ROOT / ".cache" / "ast_baseline"
DEFAULT_CACHE = ROOT / ".cache" / "ast_baseline" / "emb"
DEFAULT_OUT = ROOT / "outputs" / "instr_cal_ast_lin_v0"
SR = 16000

PRED_SOURCE = "ast_lin_v0"


# ---------------------------------------------------------------- utilities
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_audio_16k(path: Path):
    """soundfile 读取（wav/mp3 都支持）→ 单声道 → soxr 重采样 16 kHz。"""
    import numpy as np
    import soundfile as sf
    last = None
    for _attempt in range(4):
        try:
            y, sr = sf.read(str(path), always_2d=True, dtype="float32")
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    else:
        raise RuntimeError(f"读取失败 {path}: {type(last).__name__}: {last}")
    y = y.mean(axis=1)
    if sr != SR:
        import soxr  # 不用 librosa：它会拖进 numba/llvmlite/scipy，本机内存吃不消
        y = soxr.resample(y, sr, SR, quality="HQ")
    return y.astype("float32")


# ---------------------------------------------------------------- set tables
def set_items(name: str):
    """返回 [{key, id, path, label_index|None, meta...}]，顺序固定（可复现）。"""
    if name == "train":
        rows = read_csv(ROOT / "data" / "baseline_train_index.csv")
        out = []
        for r in rows:
            out.append({
                "key": f"train__{r['row_id']}",
                "id": r["row_id"],
                "path": ROOT / r["wav_local"].replace("\\", "/"),
                "label_index": int(r["label_index"]),
                "sound_id": int(r["sound_id"]),
                "sound_family": r["sound_family"],
                "sound_notation": r["sound_notation"],
                "source": r["source"],
            })
        return sorted(out, key=lambda x: x["id"])
    if name == "probe":
        rows = read_csv(ROOT / "data" / "eval_refaudio_testset.csv")
        out = []
        for r in rows:
            out.append({
                "key": f"probe__{r['ref_id']}",
                "id": r["ref_id"],
                "path": ROOT / r["audio"].replace("\\", "/"),
                "label_index": None,
                "sound_id": int(r["sound_id"]),
                "sound_family": r["sound_family"],
                "sound_notation": r["sound_notation"],
                "is_short_variant": r["is_short_variant"],
            })
        return sorted(out, key=lambda x: x["id"])
    if name == "perturb":
        rows = read_csv(ROOT / "data" / "eval_perturb_manifest.csv")
        out = []
        for r in rows:
            if str(r.get("available", "1")) != "1":
                continue
            out.append({
                "key": f"perturb__{r['perturb_id']}__{r['ref_id']}",
                "id": f"{r['perturb_id']}__{r['ref_id']}",
                "path": ROOT / "data" / "eval_perturb_audio" / r["out_file"],
                "label_index": None,
                "sound_id": int(r["sound_id"]),
                "sound_family": r["sound_family"],
                "sound_notation": r["sound_notation"],
                "is_short_variant": r["is_short_variant"],
                "perturb_id": r["perturb_id"],
                "base_perturb_id": r["base_perturb_id"],
                "kind": r["kind"],
                "ref_id": r["ref_id"],
            })
        return sorted(out, key=lambda x: x["id"])
    raise SystemExit(f"未知 set: {name}")


# ---------------------------------------------------------------- featurize
def stage_featurize(set_name: str, cache: Path, ast_dir: Path, batch_size: int, threads: int, max_items=None):
    import numpy as np
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(threads)

    items = set_items(set_name)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (cache / f"{it['key']}.npy").is_file()]
    if max_items:
        todo = todo[:max_items]
    print(f"[{set_name}] total={len(items)} todo={len(todo)} cache={cache}", flush=True)
    if not todo:
        print(f"[{set_name}] 全部命中缓存，跳过", flush=True)
        return 0

    fe = ASTFeatureExtractor.from_pretrained(str(ast_dir))
    md = None
    for attempt in range(6):
        try:
            md = ASTForAudioClassification.from_pretrained(str(ast_dir)).eval()
            break
        except OSError as exc:  # 本机提交内存不足时 safe_open 会抛 os error 1455
            print(f"[{set_name}] model load retry {attempt + 1}/6: {exc}", flush=True)
            time.sleep(3.0)
    if md is None:
        raise SystemExit(f"[{set_name}] model 加载 6 次均失败（页面文件/提交内存不足）")
    print(f"[{set_name}] model loaded, params={sum(p.numel() for p in md.parameters())/1e6:.2f}M", flush=True)

    t0 = time.time()
    done = 0
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        waves = []
        keep = []
        for it in chunk:
            try:
                waves.append(load_audio_16k(it["path"]))
                keep.append(it)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! 音频读取失败 {it['id']}: {type(exc).__name__}: {exc}", flush=True)
        if not keep:
            continue
        batch = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = md.audio_spectrogram_transformer(
                input_values=batch["input_values"]).pooler_output
        emb = emb.to(torch.float32).cpu().numpy()
        for it, vec in zip(keep, emb):
            tmp = cache / f"{it['key']}.__tmp.npy"
            np.save(tmp, vec.astype(np.float32))
            os.replace(tmp, cache / f"{it['key']}.npy")
        done += len(keep)
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(todo) - done) / max(rate, 1e-6)
        print(f"[{set_name}] {done}/{len(todo)}  {rate:.2f} it/s  elapsed={time.time()-t0:.0f}s eta={eta:.0f}s", flush=True)
    print(f"[{set_name}] featurize done in {time.time()-t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------- train
def load_matrix(set_name: str, cache: Path):
    import numpy as np
    items = set_items(set_name)
    X, meta = [], []
    for it in items:
        p = cache / f"{it['key']}.npy"
        if not p.is_file():
            raise SystemExit(f"缺特征 {p}，先跑 --stage featurize --set {set_name}")
        try:
            vec = np.load(p)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"特征损坏 {p}: {exc}；删掉该文件后重跑 featurize")
        if vec.shape != (768,):
            raise SystemExit(f"特征形状异常 {p}: {vec.shape}")
        X.append(vec)
        meta.append(it)
    return np.asarray(X, dtype=np.float32), meta


def stage_train(cache: Path, out_dir: Path, label_space_path: Path):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, meta = load_matrix("train", cache)
    y = np.asarray([m["label_index"] for m in meta], dtype=np.int64)
    print(f"train X={X.shape}  classes={sorted(set(y.tolist()))}  n_classes={len(set(y.tolist()))}", flush=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=0)
    clf.fit(Xs, y)

    train_acc = float((clf.predict(Xs) == y).mean())
    print(f"train_acc={train_acc:.4f}  （仅 sanity，不构成任何验收）", flush=True)
    print(f"可学 label_index={sorted(clf.classes_.tolist())}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "model_ast_lin_v0.npz",
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        coef=clf.coef_.astype(np.float64),
        intercept=clf.intercept_.astype(np.float64),
        classes=clf.classes_.astype(np.int64),
    )
    print("saved", out_dir / "model_ast_lin_v0.npz", flush=True)

    # 落盘可训练参数口径（PI §5.2f 要求：每次训练落盘 lora_targets 与可训练参数占比）
    n_feat = X.shape[1] * len(clf.classes_) + len(clf.classes_)
    meta_out = {
        "probe": "ast_lin_v0",
        "backbone": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "backbone_revision": "f826b80d28226b62986cc218e5cec390b1096902",
        "backbone_trainable": False,
        "backbone_trainable_params": 0,
        "head": "StandardScaler + LogisticRegression(lbfgs, L2, C=1.0, max_iter=5000)",
        "head_trainable_params": int(n_feat),
        "trainable_params_ratio": float(n_feat / (86.594063e6 + n_feat)),
        "lora_targets": [],
        "lora_note": "ast_lin_v0 无 LoRA；骨干全冻结，只训练线性探针。",
        "feature": "ASTModel.pooler_output (768-d)",
        "n_train_rows": int(X.shape[0]),
        "n_train_classes": int(len(clf.classes_)),
        "learnable_label_index": sorted(int(c) for c in clf.classes_),
        "train_acc_sanity": train_acc,
        "standard_scaler": {"mean_zero": True, "std_one": True, "fit_on": "train only"},
        "hyperparams": {"C": 1.0, "solver": "lbfgs", "max_iter": 5000,
                        "class_weight": None, "random_state": 0,
                        "note": "无超参搜索，避免偷看测试集"},
    }
    (out_dir / "model_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- predict
def stage_predict(cache: Path, out_dir: Path, label_space_path: Path):
    import numpy as np

    label_rows = read_csv(label_space_path)
    li2 = {int(r["label_index"]): r for r in label_rows}
    z = np.load(out_dir / "model_ast_lin_v0.npz")
    mean, scale = z["scaler_mean"], z["scaler_scale"]
    coef, intercept, classes = z["coef"], z["intercept"], z["classes"]
    classes = classes.astype(np.int64)

    def predict_one(vec):
        x = (vec.astype(np.float64) - mean) / scale
        logits = x @ coef.T + intercept
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        order = np.argsort(-p)[:3]
        return [(int(classes[i]), float(p[i])) for i in order]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- probe → preds.csv
    X, meta = load_matrix("probe", cache)
    rows = []
    for vec, m in zip(X, meta):
        top = predict_one(vec)
        li0, p0 = top[0]
        sid = int(li2[li0]["sound_id"]) if li0 in li2 else -1
        rows.append({
            "ref_id": m["id"],
            "pred_sound_id": sid,
            "top3": "|".join(str(int(li2[li]["sound_id"])) for li, _ in top),
            "pred_sound_id_top3": "|".join(str(int(li2[li]["sound_id"])) for li, _ in top),
            "pred_label_index": li0,
            "pred_notation": li2[li0]["sound_notation"] if li0 in li2 else "",
            "pred_en": li2[li0].get("sound_en", "") if li0 in li2 else "",
            "pred_family": li2[li0].get("sound_family", "") if li0 in li2 else "",
            "pred_source": PRED_SOURCE,
            "prob_top1": f"{p0:.6f}",
            "prob_top3": "|".join(f"{p:.6f}" for _, p in top),
            "top3_label_index": "|".join(str(li) for li, _ in top),
            "truth_sound_id": m["sound_id"],
            "truth_family": m["sound_family"],
            "is_short_variant": m.get("is_short_variant", ""),
        })
    pred_path = out_dir / "preds.csv"
    write_rows(pred_path, rows)
    print("saved", pred_path, "rows", len(rows), flush=True)

    # ---- perturb → perturb_preds.csv
    Xp, mp = load_matrix("perturb", cache)
    prows = []
    for vec, m in zip(Xp, mp):
        top = predict_one(vec)
        li0, p0 = top[0]
        sid = int(li2[li0]["sound_id"]) if li0 in li2 else -1
        prows.append({
            "perturb_id": m["perturb_id"],
            "base_perturb_id": m["base_perturb_id"],
            "kind": m["kind"],
            "ref_id": m["ref_id"],
            "pred_sound_id": sid,
            "top3": "|".join(str(int(li2[li]["sound_id"])) for li, _ in top),
            "pred_label_index": li0,
            "pred_notation": li2[li0]["sound_notation"] if li0 in li2 else "",
            "pred_source": PRED_SOURCE,
            "prob_top1": f"{p0:.6f}",
            "truth_sound_id": m["sound_id"],
        })
    ppath = out_dir / "perturb_preds.csv"
    write_rows(ppath, prows)
    print("saved", ppath, "rows", len(prows), flush=True)

    meta_out = {
        "preds_csv": {"path": str(pred_path.relative_to(ROOT)),
                      "md5": md5_file(pred_path), "rows": len(rows)},
        "perturb_preds_csv": {"path": str(ppath.relative_to(ROOT)),
                              "md5": md5_file(ppath), "rows": len(prows)},
        "testset": {"path": "data/eval_refaudio_testset.csv",
                    "md5": md5_file(ROOT / "data" / "eval_refaudio_testset.csv"),
                    "rows": 95},
        "train_index": {"path": "data/baseline_train_index.csv",
                        "md5": md5_file(ROOT / "data" / "baseline_train_index.csv"),
                        "rows": 319},
        "label_space": {"path": "data/baseline_label_space.csv",
                        "md5": md5_file(label_space_path), "rows": 35},
        "versions": {
            "python": platform.python_version(),
            "numpy": _ver("numpy"), "sklearn": _ver("sklearn"),
            "torch": _ver("torch"), "transformers": _ver("transformers"),
            "librosa": _ver("librosa"), "soundfile": _ver("soundfile"),
        },
    }
    (out_dir / "predict_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _ver(name: str) -> str:
    try:
        return __import__(name).__version__
    except Exception:  # noqa: BLE001
        return "?"


def write_rows(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["featurize", "train", "predict", "all"])
    ap.add_argument("--set", dest="set_name", default="all",
                    choices=["train", "probe", "perturb", "all"])
    ap.add_argument("--ast-model-dir", default=str(DEFAULT_AST_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--label-space", default=str(ROOT / "data" / "baseline_label_space.csv"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    cache, out_dir = Path(args.cache_dir), Path(args.out_dir)
    ast_dir, label_space = Path(args.ast_model_dir), Path(args.label_space)

    if args.stage in ("featurize", "all"):
        names = ["train", "probe", "perturb"] if args.set_name == "all" else [args.set_name]
        for n in names:
            stage_featurize(n, cache, ast_dir, args.batch_size, args.threads)
    if args.stage in ("train", "all"):
        stage_train(cache, out_dir, label_space)
    if args.stage in ("predict", "all"):
        stage_predict(cache, out_dir, label_space)
    return 0


if __name__ == "__main__":
    sys.exit(main())
