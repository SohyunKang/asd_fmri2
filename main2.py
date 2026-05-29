import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold  # pip install scikit-learn

from dataset import FCDataset
from model2 import HiTraNet
from utils import set_seed, normalize_fc, accuracy_from_logits, AverageMeter
from utils import normalize_fc
from utils import load_roi_names_from_txt

from viz_umap import extract_seed_step_embeddings, save_umap_seed_trajectories
from viz_arrival import (
    extract_arrival_outputs,
    save_arrival_weight_map,
    save_arrival_embedding_umap,
)

from sklearn.metrics import roc_auc_score


def train_one_epoch(model, loader, optimizer, device, fc_norm: str) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in loader:
        fc = batch["fc"].to(device)          # (B, N, N)
        y = batch["label"].to(device)        # (B,)

        A = normalize_fc(fc, method=fc_norm)

        logits, _aux = model(A)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits, y)
        loss_meter.update(loss.item(), n=fc.size(0))
        acc_meter.update(acc, n=fc.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

@torch.no_grad()
def eval_one_epoch(model, loader, device, fc_norm: str) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    all_y = []
    all_p1 = []   # prob of class 1
    all_pred = [] # predicted class (argmax)

    for batch in loader:
        fc = batch["fc"].to(device)
        y = batch["label"].to(device)

        A = normalize_fc(fc, method=fc_norm)

        logits, _aux = model(A)
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

    y_true = np.concatenate(all_y, axis=0)
    y_score = np.concatenate(all_p1, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    # AUROC: val fold에 한 클래스만 있으면 계산 불가
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    # Confusion matrix components (binary: 0/1)
    # tn: true 0 predicted 0
    # fp: true 0 predicted 1
    # fn: true 1 predicted 0
    # tp: true 1 predicted 1
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))

    # Sensitivity (TPR) = TP / (TP + FN)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    # Specificity (TNR) = TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "auc": auc,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,  # 원하면 로그/디버깅용
    }



def build_model(args, device):
    return HiTraNet(
        n_rois=args.n_rois,
        d_model=args.d_model,
        n_hops=args.n_hops,
        mlp_hidden=args.mlp_hidden,
        n_classes=args.n_classes,
        dropout=args.dropout,
        pool=args.pool,
        use_residual=True,
        norm_type="layer",
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV")
    parser.add_argument("--aal_txt", type=str, required=True, help="AAL ROI names txt (2nd column)")

    parser.add_argument("--n_rois", type=int, required=True, help="Number of ROIs (N)")
    parser.add_argument("--n_classes", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_hops", type=int, default=5)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "sum"])

    parser.add_argument("--fc_norm", type=str, default="row", choices=["row", "sym", "none"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FCDataset(
        xlsx_path=args.csv,
        n_rois=args.n_rois,
        label_col="DX_GROUP",
    )

    # Extract labels for stratification
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i]["label"].item()))
    labels = np.array(labels)

    aal_roi_names = load_roi_names_from_txt(args.aal_txt)
    assert len(aal_roi_names) == args.n_rois, f"ROI names ({len(aal_roi_names)}) != n_rois ({args.n_rois})"


    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_best_aucs: List[float] = []
    fold_best_accs = []
    fold_best_sens = []
    fold_best_specs = []


    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n========== Fold {fold}/{args.n_splits} ==========")

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

        model = build_model(args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_auc = -1.0
        best_val_metrics = {
            "acc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }
        best_path = f"hitranet_fold{fold}_best.pt"

        fc_norm_fn = lambda fc: normalize_fc(fc, method="row")

        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, fc_norm=args.fc_norm)
            va = eval_one_epoch(model, val_loader, device, fc_norm=args.fc_norm)

            auc_str = f"{va['auc']:.4f}" if np.isfinite(va["auc"]) else "NA"
            print(
                f"[Fold {fold} | Epoch {epoch:03d}/{args.epochs}] "
                f"train loss={tr['loss']:.4f}, acc={tr['acc']:.4f} | "
                f"val loss={va['loss']:.4f}, acc={va['acc']:.4f}, auc={auc_str}"
            )

            
            if epoch % 50 == 0 or epoch == args.epochs:
                fc_norm_fn = lambda fc: normalize_fc(fc, method="row")
                steps = list(range(1, args.n_hops + 1))

                # arrival outputs pack
                train_arr = extract_arrival_outputs(
                    model, train_loader, device, fc_norm_fn,
                    split_id=0,
                    arrival_w_key="arrival", 
                    arrival_g_key="g",    
                )
                val_arr = extract_arrival_outputs(
                    model, val_loader, device, fc_norm_fn,
                    split_id=1,
                    arrival_w_key="arrival",
                    arrival_g_key="g",
                )

                save_arrival_weight_map(
                    train_arr, val_arr,
                    epoch=epoch,
                    save_dir="arrival_logs",
                    fold=fold,
                    roi_names=aal_roi_names,
                    topk=20,
                )

                save_arrival_embedding_umap(
                    train_arr, val_arr,
                    epoch=epoch,
                    save_dir="arrival_logs",
                    fold=fold,
                )

                 # 예: seedness top-10 seeds만 그림 (추천)
                with torch.no_grad():
                    s = model.periphery_seedness().detach().cpu().numpy()
                seed_indices = np.argsort(-s)[:10]  # top 10 seeds

                # epoch loop 끝에서
                train_seed_pack = extract_seed_step_embeddings(
                    model, train_loader, device, fc_norm_fn,
                    split_name="train", steps=steps, seed_indices=seed_indices
                )
                val_seed_pack = extract_seed_step_embeddings(
                    model, val_loader, device, fc_norm_fn,
                    split_name="val", steps=steps, seed_indices=seed_indices
                )

                # aal_roi_names: List[str] (len == N)
                save_umap_seed_trajectories(
                    train_seed_pack,
                    val_seed_pack,
                    epoch=epoch,
                    save_dir="umap_logs",
                    fold=fold,
                    roi_names=aal_roi_names,   # ✅ 여기
                    max_subject_lines=200,
                    draw_centroids=True,
                    annotate_steps=True,
                    arrow_steps=True,
                )

            if va["auc"] > best_val_auc:
                best_val_auc = va["auc"]
                best_val_metrics["acc"] = va.get("acc", 0.0)
                best_val_metrics["sensitivity"] = va.get("sensitivity", 0.0)
                best_val_metrics["specificity"] = va.get("specificity", 0.0)

                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "fold": fold,
                        "best_metrics": best_val_metrics,
                    },
                    best_path,
                )

                print(
                    f"  saved: {best_path} "
                    f"(best val auc={best_val_auc:.4f}, "
                    f"acc={best_val_metrics['acc']:.4f}, "
                    f"sens={best_val_metrics['sensitivity']:.4f}, "
                    f"spec={best_val_metrics['specificity']:.4f})"
                )


        fold_best_aucs.append(best_val_auc)
        fold_best_accs.append(best_val_metrics["acc"])
        fold_best_sens.append(best_val_metrics["sensitivity"])
        fold_best_specs.append(best_val_metrics["specificity"])

        print(
            f"Fold {fold} best metrics → "
            f"AUC={best_val_auc:.4f}, "
            f"ACC={best_val_metrics['acc']:.4f}, "
            f"SENS={best_val_metrics['sensitivity']:.4f}, "
            f"SPEC={best_val_metrics['specificity']:.4f}"
        )

    mean_auc = float(np.mean(fold_best_aucs))
    std_auc = float(np.std(fold_best_aucs))
    print("\n========== 5-Fold CV Result ==========")

    print("Best val auc per fold:", [f"{a:.4f}" for a in fold_best_aucs])
    print("Best val acc per fold:", [f"{a:.4f}" for a in fold_best_accs])
    print("Best val sens per fold:", [f"{a:.4f}" for a in fold_best_sens])
    print("Best val spec per fold:", [f"{a:.4f}" for a in fold_best_specs])

    print(f"AUC  Mean ± Std: {np.mean(fold_best_aucs):.4f} ± {np.std(fold_best_aucs):.4f}")
    print(f"ACC  Mean ± Std: {np.mean(fold_best_accs):.4f} ± {np.std(fold_best_accs):.4f}")
    print(f"SENS Mean ± Std: {np.mean(fold_best_sens):.4f} ± {np.std(fold_best_sens):.4f}")
    print(f"SPEC Mean ± Std: {np.mean(fold_best_specs):.4f} ± {np.std(fold_best_specs):.4f}")



if __name__ == "__main__":
    main()
