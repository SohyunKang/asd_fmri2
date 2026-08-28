import os
import json
import datetime

import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import FCDataset
from model import HiTraNet
from utils import set_seed, normalize_fc, accuracy_from_logits, AverageMeter
from utils import load_roi_names_from_txt
from visualization.viz_umap import extract_seed_step_graphrepr, save_umap_seed_trajectories


import matplotlib.pyplot as plt
import pandas as pd

import csv

def train_one_epoch(model, loader, optimizer, device, class_weights, fc_norm: str) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, batch in enumerate(loader):

        fc = batch["fc"].to(device)
        y = batch["label"].to(device)

        # A = normalize_fc(fc, method=fc_norm)

        logits, _aux = model(fc)

        # cls_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        # loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.05)

        cls_loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.05)
        alpha = _aux["seed_weights"].to(device)  # (M,)
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum()
        # loss = cls_loss - 0.0001 * entropy
        loss = cls_loss
        # print("CE:", cls_loss.item(), "REG:", entropy.item(), "lam*reg:", (0.01*entropy).item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad = model.periphery_seedness.u_per.grad

        with torch.no_grad():

            seed_indices = _aux["seed_indices"].detach().long()

            selected = torch.zeros_like(
                grad,
                dtype=torch.bool
            )
            selected[seed_indices] = True

            # selected seed의 gradient만 약화
            grad[selected] *= 0.01

        if batch_idx == 0:

            u = model.periphery_seedness.u_per.detach()
            g = model.periphery_seedness.u_per.grad.detach()

            top_idx = torch.topk(
                u,
                k=15,
                largest=True
            ).indices

            print("\n[SEED RANK DEBUG]")

            for rank, idx in enumerate(top_idx, start=1):

                i = idx.item()

                print(
                    f"{rank:2d}. ROI={i:3d} | "
                    f"score={u[i].item():+.6f} | "
                    f"grad={g[i].item():+.6e}"
                )


        optimizer.step()

        acc = accuracy_from_logits(logits, y)
        loss_meter.update(loss.item(), n=fc.size(0))
        acc_meter.update(acc, n=fc.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

def node_variance(H):
    # H: (B, N, d)
    H_centered = H - H.mean(dim=1, keepdim=True)
    var = (H_centered ** 2).sum(dim=-1).mean()
    return var.item()

@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    fc_norm: str,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_y = []
    all_p1 = []
    all_pred = []
    all_sub_id = []

    all_g = []
    all_p = []
    all_H = []

    for batch_idx, batch in enumerate(loader):
        fc = batch["fc"].to(device)
        y = batch["label"].to(device)
        subjects = batch["sub_id"]

        # A = normalize_fc(fc, method=fc_norm)

        logits, aux = model(
            fc,
            debug=(batch_idx == 0),
        )

        if batch_idx == 0:

            u = model.periphery_seedness.u_per.detach()
            g = model.periphery_seedness.u_per.grad.detach()

            top_idx = torch.topk(
                u,
                k=15,
                largest=True
            ).indices

            print("\n[SEED RANK DEBUG]")

            for rank, idx in enumerate(top_idx, start=1):

                i = idx.item()

                print(
                    f"{rank:2d}. ROI={i:3d} | "
                    f"score={u[i].item():+.6f} | "
                    f"grad={g[i].item():+.6e}"
                )
        
        if len(all_y) == 0:

            g_last = aux["g_seed_list"][-1]   # (B,M,d)

            per_seed_std = (
                g_last
                .std(dim=0)
                .mean(dim=1)
            )

            alpha = aux["seed_weights"].to(g_last.device)

            g_alpha = (
                g_last
                * alpha.view(1, -1, 1)
            ).sum(dim=1)

            print(
                "[EVAL DEBUG] per-seed g std: "
                f"mean={per_seed_std.mean().item():.6f}, "
                f"max={per_seed_std.max().item():.6f}"
            )

            print(
                "[EVAL DEBUG] alpha g std:",
                g_alpha.std(dim=0).mean().item()
            )


            if "g_bar_list" in aux:
                g_attn = aux["g_bar_list"][-1]

                print(
                    "[EVAL DEBUG] attention g std:",
                    g_attn.std(dim=0).mean().item()
                )

        loss = F.cross_entropy(logits, y)

        acc = accuracy_from_logits(logits, y)
        loss_meter.update(loss.item(), n=fc.size(0))
        acc_meter.update(acc, n=fc.size(0))

        probs = F.softmax(logits, dim=1)
        p1 = probs[:, 1]
        pred = torch.argmax(logits, dim=1)

        all_y.append(y.detach().cpu().numpy())
        all_p1.append(p1.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_sub_id.extend(subjects)

        # -----------------------------
        # g_seed trajectory
        # -----------------------------
        g_list = aux["g_seed_list"]            # list[K] of (B,M,d)
        g_stack = torch.stack(g_list, dim=0)   # (K,B,M,d)
        g_stack = g_stack.permute(1,0,2,3)     # (B,K,M,d)

        all_g.append(g_stack.cpu())

        # -----------------------------
        # p trajectory
        # -----------------------------
        p_list = aux["p_list"]                 # list[K] of (B,M,N)
        p_stack = torch.stack(p_list, dim=0)   # (K,B,M,N)
        p_stack = p_stack.permute(1,0,2,3)     # (B,K,M,N)

        all_p.append(p_stack.cpu())

        # -----------------------------
        # H trajectory
        # -----------------------------
        H_list = aux["H_list"]                 # list[K] of (B,M,N,d)
        H_stack = torch.stack(H_list, dim=0)   # (K,B,M,N,d)
        H_stack = H_stack.permute(1,0,2,3,4)   # (B,K,M,N,d)

        all_H.append(H_stack.cpu())

    y_true = np.concatenate(all_y, axis=0)
    y_score = np.concatenate(all_p1, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    # ============================================================
    # Validation probability distribution check
    # ============================================================

    mask0 = (y_true == 0)
    mask1 = (y_true == 1)

    mean0 = y_score[mask0].mean() if mask0.sum() > 0 else float("nan")
    std0  = y_score[mask0].std()  if mask0.sum() > 0 else float("nan")

    mean1 = y_score[mask1].mean() if mask1.sum() > 0 else float("nan")
    std1  = y_score[mask1].std()  if mask1.sum() > 0 else float("nan")

    pred_pos_ratio = (y_score >= 0.5).mean()

    print(
        f"[VAL PROB] "
        f"y=0: {mean0:.4f} ± {std0:.4f} | "
        f"y=1: {mean1:.4f} ± {std1:.4f} | "
        f"min={y_score.min():.4f} | "
        f"max={y_score.max():.4f} | "
        f"pred_pos_ratio@0.5={pred_pos_ratio:.4f}"
    )

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    # ---------- Youden index threshold ----------
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)

    best_threshold = thresholds[best_idx]

    y_pred_youden = (y_score >= best_threshold).astype(int)

    tn_y = float(np.sum((y_true == 0) & (y_pred_youden == 0)))
    fp_y = float(np.sum((y_true == 0) & (y_pred_youden == 1)))
    fn_y = float(np.sum((y_true == 1) & (y_pred_youden == 0)))
    tp_y = float(np.sum((y_true == 1) & (y_pred_youden == 1)))

    sens_y = tp_y / (tp_y + fn_y) if (tp_y + fn_y) > 0 else float("nan")
    spec_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else float("nan")
    acc_y = (tp_y + tn_y) / (tp_y + tn_y + fp_y + fn_y)

    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "auc": auc,
        "sensitivity": float(sens),
        "specificity": float(spec),

        "youden_threshold": float(best_threshold),
        "youden_acc": float(acc_y),
        "youden_sensitivity": float(sens_y),
        "youden_specificity": float(spec_y),

        # -------- analysis tensors --------
        "g": torch.cat(all_g, dim=0),  # (Nsubjects, K, M, d)
        "p": torch.cat(all_p, dim=0),  # (Nsubjects, K, M, N)
        "H": torch.cat(all_H, dim=0),  # (Nsubjects, K, M, N, d)

        "y": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "subject": all_sub_id
    }


def build_model(args, device):
    return HiTraNet(
        n_rois=args.n_rois,
        d_model=args.d_model,
        n_hops=args.n_hops,
        top_m=args.top_m,
        mlp_hidden=args.mlp_hidden,
        n_classes=args.n_classes,
        depth=args.depth,
        dropout=args.dropout,
        seed_std=args.seed_std,
        act=args.act,
        use_gated_residual=True,
    ).to(device)

def plot_site_distribution(train_df, val_df, site_col, fold, save_dir):

    train_counts = train_df[site_col].value_counts()
    val_counts = val_df[site_col].value_counts()

    # 공통 site 정렬
    all_sites = sorted(set(train_counts.index).union(set(val_counts.index)))

    train_counts = train_counts.reindex(all_sites, fill_value=0)
    val_counts = val_counts.reindex(all_sites, fill_value=0)

    # 🔥 percentage로 변환
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100

    x = np.arange(len(all_sites))
    width = 0.4

    plt.figure(figsize=(10, 4))

    bars1 = plt.bar(x - width/2, train_pct, width, label="Train")
    bars2 = plt.bar(x + width/2, val_pct, width, label="Validation")

    plt.xticks(x, all_sites, rotation=45)
    plt.ylabel("Percentage (%)")
    plt.title(f"Fold {fold} - Site Distribution (%)")
    plt.legend()

    # 🔥 숫자 표시까지 (이게 핵심)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f"{height:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=7
                )

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"fold{fold}_site_dist_pct.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"📊 Saved: {save_path}")


def fc_stats(dataset, indices):
    means, stds, abs_means = [], [], []

    for i in indices:
        fc = dataset[i]["fc"].numpy()

        iu = np.triu_indices_from(fc, k=1)
        vals = fc[iu]

        means.append(vals.mean())
        stds.append(vals.std())
        abs_means.append(np.abs(vals).mean())

    return {
        "mean": np.mean(means),
        "std": np.mean(stds),
        "abs_mean": np.mean(abs_means),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/abide_subjects_fc_with_demo_260202.xlsx")
    parser.add_argument("--aal_txt", type=str, default="data/ROI_MNI_V4.txt")

    parser.add_argument("--n_rois", type=int, default=116)
    parser.add_argument("--n_classes", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_hops", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--top_m", type=int, default=116)
    parser.add_argument("--seed_std", type=float, default=0.5)

    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--act", type=str, default="gelu", choices=["gelu", "relu"])

    parser.add_argument("--fc_norm", type=str, default="row", choices=["row", "sym", "none"])

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="Run only a specific fold (1-based index)")

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--worker_id", type=int, default=0)
    
    parser.add_argument("--vis", type=bool, default=False)
    parser.add_argument("--save_model", type=bool, default=True)
    
    args = parser.parse_args()

    set_seed(args.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # for log save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_dir = f"./experiments/w{args.worker_id}_hitranet_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    print("Experiment dir:", exp_dir)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    log_path = os.path.join(exp_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "fold",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_auc",
            "val_sens",
            "val_spec",
            "youden_acc",
            "youden_sens",
            "youden_spec",
            "youden_thr"
        ])

    # dataset
    dataset = FCDataset(
        xlsx_path=args.csv,
        n_rois=args.n_rois,
        label_col="DX_GROUP",
    )

    # strat labels
    labels = np.array([int(dataset[i]["label"].item()) for i in range(len(dataset))], dtype=np.int64)

    # -------------------------
    # 🔥 dataset → DataFrame (분포 체크용)
    # -------------------------
    import pandas as pd

    rows = []

    for i in range(len(dataset)):
        item = dataset[i]

        row = {
            "label": int(item["label"]),
            "sub_id": item["sub_id"],
        }

        # optional fields
        for key in ["AGE_AT_SCAN", "SEX", "func_mean_fd", "SITE_ID"]:
            if key in item:
                row[key] = item[key]

        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n📊 Dataset columns:", df.columns.tolist())

    aal_roi_names = load_roi_names_from_txt(args.aal_txt)
    assert len(aal_roi_names) == args.n_rois, f"ROI names ({len(aal_roi_names)}) != n_rois ({args.n_rois})"

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    fold_best_aucs: List[float] = []
    fold_best_accs: List[float] = []
    fold_best_sens: List[float] = []
    fold_best_specs: List[float] = []


    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        # 특정 fold만 실행
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n========== Fold {fold}/{args.n_splits} ==========")
        set_seed(args.random_seed)

        # ===== class weight 계산 (train set 기준) =====
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels)

        # 0으로 나누는 것 방지
        class_counts = np.maximum(class_counts, 1)

        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)

        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        print("Train class distribution:", class_counts)
        print("Class weights:", class_weights)

        train_set = Subset(dataset, train_idx.tolist())
        val_set = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        print("\n--- Data distribution check ---")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # -------------------------
        # class
        # -------------------------
        print("Train class:", train_df["label"].value_counts().to_dict())
        print("Val   class:", val_df["label"].value_counts().to_dict())

        # -------------------------
        # demographics + motion
        # -------------------------
        for col in ["AGE_AT_SCAN", "SEX", "func_mean_fd"]:
            if col in train_df.columns:

                print(f"\n[{col}]")

                tr_mean, tr_std = train_df[col].mean(), train_df[col].std()
                va_mean, va_std = val_df[col].mean(), val_df[col].std()

                print(f"Train mean/std: {tr_mean:.4f} / {tr_std:.4f}")
                print(f"Val   mean/std: {va_mean:.4f} / {va_std:.4f}")

                print(f"Diff (val-train): {va_mean - tr_mean:.4f}")
         
        print("Train site:", train_df['SITE_ID'].value_counts().to_dict())
        print("Val   site :", val_df['SITE_ID'].value_counts().to_dict())
        
        plot_site_distribution(train_df, val_df, "SITE_ID", fold, exp_dir)

        train_fc = fc_stats(dataset, train_idx)
        val_fc = fc_stats(dataset, val_idx)

        print("Train FC:", train_fc)
        print("Val   FC:", val_fc)

        print("Diff (val-train):",
            {k: val_fc[k] - train_fc[k] for k in train_fc})

        model = build_model(args, device)

        initial_u_per = (
            model.periphery_seedness.u_per
            .detach()
            .clone()
        )

        # Save Model Architecture
        if fold == 1:
            with open(os.path.join(exp_dir, "model_arch.txt"), "w") as f:
                f.write(str(model))

        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters()},
            {"params": model.head.parameters()},
            {"params": model.periphery_seedness.parameters(), "lr": args.seed_lr, "weight_decay": 0.0}
            ], lr=args.lr, weight_decay=args.weight_decay)
       
        best_val_auc = -1.0
        best_metrics = {
            "acc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "youden_threshold": 0.0,
            "youden_acc": 0.0,
            "youden_sensitivity": 0.0,
            "youden_specificity": 0.0,
        }
        best_path = os.path.join(exp_dir, f"fold{fold}_best_model.pt")
        
        import time

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.current_epoch = epoch
            tr = train_one_epoch(model, train_loader, optimizer, device, class_weights, fc_norm=args.fc_norm)
            va = eval_one_epoch(
                model,
                val_loader,
                device,
                fc_norm=args.fc_norm,
            )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    fold,
                    epoch,
                    tr["loss"],
                    tr["acc"],
                    va["loss"],
                    va["acc"],
                    va["auc"],
                    va["sensitivity"],
                    va["specificity"],
                    va["youden_acc"],
                    va["youden_sensitivity"],
                    va["youden_specificity"],
                    va["youden_threshold"],
                ])

            auc_str = f"{va['auc']:.4f}" if np.isfinite(va["auc"]) else "NA"

            print(
                f"[Fold {fold} | Epoch {epoch:03d}/{args.epochs}] "
                f"train loss={tr['loss']:.4f}, acc={tr['acc']:.4f} | "
                f"val loss={va['loss']:.4f}, acc={va['acc']:.4f}, auc={auc_str}, "
                f"sens={va['sensitivity']:.4f}, spec={va['specificity']:.4f} | "
                f"you_acc={va['youden_acc']:.4f}, "
                f"you_sens={va['youden_sensitivity']:.4f}, "
                f"you_spec={va['youden_specificity']:.4f}, "
                f"thr={va['youden_threshold']:.4f}, "
                f"time={time.time()-start:.2f} sec"
            )

            # if args.vis and (epoch % 10 == 0 or epoch == args.epochs):
            #     steps = list(range(1, args.n_hops + 1))

            #     fc_norm_fn = lambda fc: normalize_fc(fc, method=args.fc_norm)

            #     train_pack = extract_seed_step_graphrepr(
            #         model, train_loader, device, fc_norm_fn,
            #         split_name="train", steps=steps, seed_indices=seed_indices,
            #         seed_weights=train_seed_weight,
            #     )
            #     val_pack = extract_seed_step_graphrepr(
            #         model, val_loader, device, fc_norm_fn,
            #         split_name="val", steps=steps, seed_indices=seed_indices,
            #         seed_weights=train_seed_weight,
            #     )
            #     umap_dir = os.path.join(exp_dir, "umap")
            #     os.makedirs(umap_dir, exist_ok=True)

            #     save_umap_seed_trajectories(
            #         train_pack,
            #         val_pack,
            #         epoch=epoch,
            #         save_dir=umap_dir,
            #         fold=fold,
            #         roi_names=aal_roi_names
            #     )              


            if va["auc"] > best_val_auc:
                tr_eval = eval_one_epoch(
                    model,
                    train_loader,
                    device,
                    fc_norm=args.fc_norm,
                )
                best_val_auc = va["auc"]
                best_metrics["epoch"] = epoch
                best_metrics["acc"] = va["acc"]
                best_metrics["sensitivity"] = va["sensitivity"]
                best_metrics["specificity"] = va["specificity"]

                best_metrics["youden_threshold"] = va["youden_threshold"]
                best_metrics["youden_acc"] = va["youden_acc"]
                best_metrics["youden_sensitivity"] = va["youden_sensitivity"]
                best_metrics["youden_specificity"] = va["youden_specificity"]

                
                if args.save_model:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "args": vars(args),
                            "fold": fold,
                            "best_metrics": best_metrics,
                        },
                        best_path,
                    )
                    
                aux_path = os.path.join(exp_dir, f"fold{fold}_best_aux.pt")

                seed_idx, seed_w = model.topm_seeds()
                seed_idx_cpu = seed_idx.detach().cpu()
                seed_w_cpu = seed_w.detach().cpu()

                torch.save({
                        "epoch": epoch,

                        # -----------------------
                        # validation
                        # -----------------------
                        "val_subject": va["subject"],
                        "val_y": va["y"],
                        "val_g": va["g"],
                        "val_p": va["p"],
                        "val_H": va["H"],
                        "val_pred": va["y_pred"],
                        "val_score": va["y_score"],

                        # -----------------------
                        # train
                        # -----------------------
                        "train_subject": tr_eval["subject"],
                        "train_y": tr_eval["y"],
                        "train_g": tr_eval["g"],
                        "train_p": tr_eval["p"],
                        "train_H": tr_eval["H"],

                        # -----------------------
                        # seeds
                        # -----------------------
                        "seed_indices": seed_idx.detach().cpu(),
                        "seed_weights": seed_w.detach().cpu(),

                    }, aux_path)

                print(
                f" * [saved: {best_path}] "
                f"(best val auc={best_val_auc:.4f}) "
                )
            
                order = torch.argsort(
                    seed_w_cpu,
                    descending=True
                )

                sorted_seed_idx = seed_idx_cpu[order]
                sorted_seed_w = seed_w_cpu[order]

                fold_result = {
                    "fold": fold,
                    "auc": best_val_auc,
                    "acc": best_metrics["acc"],
                    "sensitivity": best_metrics["sensitivity"],
                    "specificity": best_metrics["specificity"],

                    "youden_threshold": best_metrics["youden_threshold"],
                    "youden_acc": best_metrics["youden_acc"],
                    "youden_sensitivity": best_metrics["youden_sensitivity"],
                    "youden_specificity": best_metrics["youden_specificity"],

                    "epoch": best_metrics["epoch"],

                    "seed_indices": sorted_seed_idx.tolist(),
                    "seed_weights": sorted_seed_w.tolist(),
                }

                with open(os.path.join(exp_dir, f"fold{fold}_metrics.json"), "w") as f:
                    json.dump(fold_result, f, indent=4)

        fold_best_aucs.append(best_val_auc)
        fold_best_accs.append(best_metrics["acc"])
        fold_best_sens.append(best_metrics["sensitivity"])
        fold_best_specs.append(best_metrics["specificity"])
        
        print(
            f"Fold {fold} best metrics → "
            f"AUC={best_val_auc:.4f}, "
            f"ACC={best_metrics['acc']:.4f}, "
            f"SENS={best_metrics['sensitivity']:.4f}, "
            f"SPEC={best_metrics['specificity']:.4f}, "
            f"YOU_ACC={best_metrics['youden_acc']:.4f}, "
            f"YOU_SENS={best_metrics['youden_sensitivity']:.4f}, "
            f"YOU_SPEC={best_metrics['youden_specificity']:.4f}, "
            f"THR={best_metrics['youden_threshold']:.4f}, "
            f"EPOCH={best_metrics['epoch']}"
        )

    if args.fold is None:
        print("\n========== 5-Fold CV Result ==========")
        print("Best val auc per fold:", [f"{a:.4f}" for a in fold_best_aucs])
        print("Best val acc per fold:", [f"{a:.4f}" for a in fold_best_accs])
        print("Best val sens per fold:", [f"{a:.4f}" for a in fold_best_sens])
        print("Best val spec per fold:", [f"{a:.4f}" for a in fold_best_specs])

        print(f"AUC  Mean ± Std: {np.mean(fold_best_aucs):.4f} ± {np.std(fold_best_aucs):.4f}")
        print(f"ACC  Mean ± Std: {np.mean(fold_best_accs):.4f} ± {np.std(fold_best_accs):.4f}")
        print(f"SENS Mean ± Std: {np.mean(fold_best_sens):.4f} ± {np.std(fold_best_sens):.4f}")
        print(f"SPEC Mean ± Std: {np.mean(fold_best_specs):.4f} ± {np.std(fold_best_specs):.4f}")

        # =========================
        # Fold별 결과
        # =========================
        fold_results = []

        for i, (auc, acc, sens, spec) in enumerate(zip(
            fold_best_aucs,
            fold_best_accs,
            fold_best_sens,
            fold_best_specs
        )):
            fold_results.append({
                "fold": i + 1,
                "auc": float(auc),
                "acc": float(acc),
                "sens": float(sens),
                "spec": float(spec),
            })

        # =========================
        # 전체 CV 결과
        # =========================
        cv_results = {
            "folds": fold_results,

            "mean": {
                "auc": float(np.mean(fold_best_aucs)),
                "acc": float(np.mean(fold_best_accs)),
                "sens": float(np.mean(fold_best_sens)),
                "spec": float(np.mean(fold_best_specs)),
            },

            "std": {
                "auc": float(np.std(fold_best_aucs)),
                "acc": float(np.std(fold_best_accs)),
                "sens": float(np.std(fold_best_sens)),
                "spec": float(np.std(fold_best_specs)),
            }
        }

        with open(os.path.join(exp_dir, "cv_results.json"), "w") as f:
            json.dump(cv_results, f, indent=4)

        print(f"Saved CV results: {os.path.join(exp_dir, 'cv_results.json')}")


if __name__ == "__main__":
    main()
