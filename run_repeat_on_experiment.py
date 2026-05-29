import os
import json
import argparse
import torch
import numpy as np

import sys
import select

from sklearn.model_selection import StratifiedKFold

from dataset import FCDataset
from model import HiTraNet
from utils import set_seed

from main import train_one_epoch, eval_one_epoch, build_model

def check_keypress():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr != []

def run_repeat_on_experiment(exp_dir, target_fold, n_repeat, device, epochs):

    # -----------------------------
    # args.json 로드
    # -----------------------------
    args_path = os.path.join(exp_dir, "args.json")

    with open(args_path, "r") as f:
        args_dict = json.load(f)

    args = argparse.Namespace(**args_dict)

    print("Loaded args:", args)

    # -----------------------------
    # dataset
    # -----------------------------
    dataset = FCDataset(
        xlsx_path=args.csv,
        n_rois=args.n_rois,
        label_col="DX_GROUP",
    )

    labels = np.array([int(dataset[i]["label"].item()) for i in range(len(dataset))])

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # -----------------------------
    # fold loop
    # -----------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):

        if fold != target_fold:
            continue

        print(f"\n🔥 Running repeat on Fold {fold}")

        metrics_path = os.path.join(exp_dir, f"fold{fold}_metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                prev_metrics = json.load(f)

            best_fold_auc = prev_metrics["auc"]
            best_fold_metrics = prev_metrics

            print(f"📌 Loaded previous best AUC: {best_fold_auc:.4f}")

        else:
            best_fold_auc = -1
            best_fold_metrics = None
            print("⚠️ No previous metrics found, starting fresh")

        # -----------------------------
        # repeat loop
        # -----------------------------
        for r in range(n_repeat):

            print(f"\n--- Repeat {r+1}/{n_repeat} ---")

            set_seed(args.seed + r)

            model = build_model(args, device).to(device)

            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters()},
                {"params": model.head.parameters()},
                {"params": model.periphery_seedness.parameters(), "lr": args.seed_lr, "weight_decay": 0.0}
            ], lr=args.lr, weight_decay=args.weight_decay)

            train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
            val_set = torch.utils.data.Subset(dataset, val_idx.tolist())

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
            )

            best_metrics = None

            # -----------------------------
            # epoch loop
            # -----------------------------
            for epoch in range(1, epochs + 1):

                # 🔥 키 누르면 skip
                if check_keypress():
                    key = sys.stdin.readline().strip()
                    if key == "s":
                        print("⏭️ Skip to next repeat!")
                        break

                tr = train_one_epoch(model, train_loader, optimizer, device, None, args.fc_norm)
                va = eval_one_epoch(model, val_loader, device, args.fc_norm)

                auc_str = f"{va['auc']:.4f}" if np.isfinite(va["auc"]) else "NA"

                print(
                    f"[Fold {fold} | Repeat {r} | Epoch {epoch:03d}/{epochs}] "
                    f"train loss={tr['loss']:.4f}, acc={tr['acc']:.4f} | "
                    f"val loss={va['loss']:.4f}, acc={va['acc']:.4f}, auc={auc_str}, "
                    f"sens={va['sensitivity']:.4f}, spec={va['specificity']:.4f} | "
                    f"you_acc={va['youden_acc']:.4f}, "
                    f"you_sens={va['youden_sensitivity']:.4f}, "
                    f"you_spec={va['youden_specificity']:.4f}, "
                    f"thr={va['youden_threshold']:.4f}"
                )

                if va["auc"] > best_fold_auc:
                    best_fold_auc = va["auc"]
                    best_metrics = va.copy()

                    best_fold_metrics = best_metrics

                    print(f"🔥 NEW BEST: AUC={best_fold_auc:.4f}")

                    # 🔥 model 저장
                    torch.save(
                        {"model": model.state_dict(), "fold": fold, "repeat": r},
                        os.path.join(exp_dir, f"fold{fold}_best_model.pt")
                    )

                    # 🔥🔥🔥 aux 다시 저장 (핵심 추가)
                    print("Saving best aux...")

                    val_eval = eval_one_epoch(model, val_loader, device, args.fc_norm)
                    train_eval = eval_one_epoch(model, train_loader, device, args.fc_norm)

                    seed_idx, seed_w = model.topm_seeds()

                    aux_path = os.path.join(exp_dir, f"fold{fold}_best_aux.pt")

                    torch.save({
                        "repeat": r,

                        # validation
                        "val_subject": val_eval["subject"],
                        "val_y": val_eval["y"],
                        "val_pred": val_eval["y_pred"],  # 아래에서 채움
                        "val_g": val_eval["g"],
                        "val_p": val_eval["p"],
                        "val_H": val_eval["H"],

                        # train
                        "train_subject": train_eval["subject"],
                        "train_y": train_eval["y"],
                        "train_g": train_eval["g"],
                        "train_p": train_eval["p"],
                        "train_H": train_eval["H"],

                        # seeds
                        "seed_indices": seed_idx.cpu(),
                        "seed_weights": seed_w.cpu(),

                    }, aux_path)

                    print(f"✅ saved best aux: {aux_path}")

                    # -----------------------------
                    # fold metrics 저장
                    # -----------------------------
                    fold_result = {
                        "fold": fold,
                        "auc": float(best_fold_auc),
                        "acc": float(best_fold_metrics["acc"]),
                        "sensitivity": float(best_fold_metrics["sensitivity"]),
                        "specificity": float(best_fold_metrics["specificity"]),
                        "youden_threshold": float(best_fold_metrics["youden_threshold"]),
                        "youden_acc": float(best_fold_metrics["youden_acc"]),
                        "youden_sensitivity": float(best_fold_metrics["youden_sensitivity"]),
                        "youden_specificity": float(best_fold_metrics["youden_specificity"]),
                        "epoch": epoch,
                    }

                    with open(os.path.join(exp_dir, f"fold{fold}_metrics.json"), "w") as f:
                        json.dump(fold_result, f, indent=4)

                    print("✅ Fold metrics updated")

                    # -----------------------------
                    # cv_results 재계산
                    # -----------------------------
                    fold_metrics_files = sorted([
                        f for f in os.listdir(exp_dir) if f.startswith("fold") and f.endswith("_metrics.json")
                    ])

                    aucs, accs, sens, specs = [], [], [], []

                    for fpath in fold_metrics_files:
                        with open(os.path.join(exp_dir, fpath), "r") as f:
                            d = json.load(f)

                        aucs.append(d["auc"])
                        accs.append(d["acc"])
                        sens.append(d["sensitivity"])
                        specs.append(d["specificity"])

                    cv_results = {
                        "auc_mean": float(np.mean(aucs)),
                        "auc_std": float(np.std(aucs)),
                        "acc_mean": float(np.mean(accs)),
                        "acc_std": float(np.std(accs)),
                        "sens_mean": float(np.mean(sens)),
                        "sens_std": float(np.std(sens)),
                        "spec_mean": float(np.mean(specs)),
                        "spec_std": float(np.std(specs)),
                    }

                    with open(os.path.join(exp_dir, "cv_results.json"), "w") as f:
                        json.dump(cv_results, f, indent=4)

                    print("✅ CV results updated")

        break
# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n_repeat", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    epochs = 350

    run_repeat_on_experiment(
        exp_dir=args.exp_dir,
        target_fold=args.fold,
        n_repeat=args.n_repeat,
        device=device,
        epochs=epochs,
    )


#    CUDA_VISIBLE_DEVICES=3 python repeat_fold.py --exp_dir /data5/sohyunkang/experiments/w2_hitranet_20260411_005252_215818 --fold 4 --n_repeat