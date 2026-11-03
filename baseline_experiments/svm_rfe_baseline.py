import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import csv
import json
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
)

from dataset import FCDataset
import datetime

# =========================================================
# CONFIG
# =========================================================
xlsx_path = "data/abide_subjects_fc_with_demo_260202.xlsx"
label_col = "DX_GROUP"
n_rois = 116
n_splits = 5
seed = 40

# -------- RFE --------
n_features_to_select = 1000
rfe_step = 0.1

# -------- model --------
hidden_dim1 = 500
hidden_dim2 = 100
n_classes = 2
dropout = 0.2

# -------- sparsity --------
rho = 0.2          # target activation
beta = 2         # KL divergence weight

# -------- pretraining --------
pretrain_epochs = 100
pretrain_batch_size = 128
pretrain_lr = 1e-3
pretrain_weight_decay = 1e-4

# -------- finetuning --------
finetune_epochs = 300
finetune_batch_size = 128
finetune_lr = 1e-4
finetune_weight_decay = 1e-4
patience = 50
inner_val_size = 0.15

# -------- loss weights --------
recon_weight = 0.1

# -------- others --------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
save_dir = f"./baseline_experiments/sae_rfe_results/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)


# =========================================================
# MODEL
# =========================================================
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


def kl_divergence_sparse(z: torch.Tensor, rho: float = 0.05, beta: float = 1e-3):
    """
    z: (B, H), expected nonnegative activations after ReLU
    """
    rho_hat = torch.mean(z, dim=0)
    rho_tensor = torch.full_like(rho_hat, rho)

    rho_hat = torch.clamp(rho_hat, min=1e-8, max=1 - 1e-8)

    kl = rho_tensor * torch.log(rho_tensor / rho_hat) + \
         (1 - rho_tensor) * torch.log((1 - rho_tensor) / (1 - rho_hat))

    # Mean prevents the penalty scale from growing with hidden dimensionality.
    return beta * torch.mean(kl)


class StackedSAEClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 256,
        hidden_dim2: int = 64,
        n_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.ae1 = SparseAutoencoder(input_dim, hidden_dim1, dropout=dropout)
        self.ae2 = SparseAutoencoder(hidden_dim1, hidden_dim2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim2, n_classes)

    def forward(self, x):
        z1, x1 = self.ae1(x)
        z2, x2 = self.ae2(z1)
        logits = self.classifier(self.dropout(z2))
        return logits, (z1, z2, x1, x2)


# =========================================================
# DATA HELPERS
# =========================================================
def fc_to_vector(fc: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(fc, k=1)
    return fc[iu]


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# =========================================================
# PRETRAIN
# =========================================================
def pretrain_ae_epoch(autoencoder, loader, optimizer, device, rho, beta):
    autoencoder.train()
    total_loss = total_recon = total_kl = total_n = 0.0

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        z, reconstruction = autoencoder(xb)
        recon = F.mse_loss(reconstruction, xb)
        kl = kl_divergence_sparse(z, rho=rho, beta=beta)
        loss = recon + kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_recon += recon.item() * bs
        total_kl += kl.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "recon": total_recon / total_n,
        "kl": total_kl / total_n,
    }


@torch.no_grad()
def encode_array(autoencoder, X, batch_size, device):
    autoencoder.eval()
    loader = make_loader(
        X,
        np.zeros(len(X), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
    )
    encoded = []
    for xb, _ in loader:
        z, _ = autoencoder(xb.to(device, non_blocking=True))
        encoded.append(z.cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float32)


# =========================================================
# FINETUNE
# =========================================================
def finetune_epoch(model, loader, optimizer, device, rho, beta, recon_weight):
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits, (z1, z2, x1, x2) = model(xb)

        cls_loss = F.cross_entropy(logits, yb)
        recon_loss = F.mse_loss(x1, xb) + F.mse_loss(x2, z1)
        kl_loss = kl_divergence_sparse(z1, rho=rho, beta=beta) + \
                  kl_divergence_sparse(z2, rho=rho, beta=beta)

        loss = cls_loss + recon_weight * recon_loss + kl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl_loss.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "cls": total_cls / total_n,
        "recon": total_recon / total_n,
        "kl": total_kl / total_n,
    }


@torch.no_grad()
def evaluate(model, loader, device, rho, beta, recon_weight):
    model.eval()

    all_probs = []
    all_preds = []
    all_y = []
    total_loss, total_n = 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        yb_device = yb.to(device, non_blocking=True)
        logits, (z1, z2, x1, x2) = model(xb)
        cls_loss = F.cross_entropy(logits, yb_device)
        recon_loss = F.mse_loss(x1, xb) + F.mse_loss(x2, z1.detach())
        kl_loss = kl_divergence_sparse(z1, rho=rho, beta=beta) + \
                  kl_divergence_sparse(z2, rho=rho, beta=beta)
        loss = cls_loss + recon_weight * recon_loss + kl_loss
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
        probs = torch.softmax(logits, dim=1)[:, 0]  # ASD probability (ASD=0)
        preds = torch.argmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_y.append(yb.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    acc = accuracy_score(all_y, all_preds)
    sensitivity = recall_score(all_y, all_preds, pos_label=0, zero_division=0)
    specificity = recall_score(all_y, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_y, all_preds, pos_label=0, zero_division=0)
    y_asd = (all_y == 0).astype(np.int64)

    try:
        auc = roc_auc_score(y_asd, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / total_n,
        "acc": float(acc),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "y_true": all_y,
        "y_prob": all_probs,
        "y_pred": all_preds,
    }


# =========================================================
# LOAD DATASET
# =========================================================
dataset = FCDataset(
    xlsx_path=xlsx_path,
    n_rois=n_rois,
    label_col=label_col,
)

labels = np.array([dataset[i]["label"].item() for i in range(len(dataset))], dtype=np.int64)

X = []
for i in range(len(dataset)):
    fc = dataset[i]["fc"].numpy()
    X.append(fc_to_vector(fc))
X = np.asarray(X, dtype=np.float32)

print("X shape:", X.shape)
print("labels shape:", labels.shape)


# =========================================================
# CV
# =========================================================
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

fold_summaries = []
history_path = os.path.join(save_dir, "training_history.csv")
with open(history_path, "w", newline="") as f:
    csv.writer(f).writerow([
        "fold", "epoch", "train_loss", "val_loss", "val_auc",
        "val_accuracy", "val_sensitivity", "val_specificity", "val_f1",
        "is_best", "stale_epochs",
    ])

for fold, (outer_train_idx, test_idx) in enumerate(skf.split(X, labels), start=1):
    set_seed(seed + fold)
    print(f"\n{'=' * 20} Fold {fold}/{n_splits} {'=' * 20}")

    train_idx, val_idx = train_test_split(
        outer_train_idx, test_size=inner_val_size,
        stratify=labels[outer_train_idx], random_state=seed + fold
    )
    X_train_raw, X_val_raw, X_test_raw = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    print("Train shape:", X_train_raw.shape, y_train.shape)
    print("Val shape:  ", X_val_raw.shape, y_val.shape)

    # -----------------------------------------------------
    # STANDARDIZATION + RFE ON FOUR TRAINING FOLDS ONLY
    # -----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print("Running RFE...")
    rfe = RFE(
        estimator=SVC(kernel="linear"),
        n_features_to_select=n_features_to_select,
        step=rfe_step,
    )
    rfe.fit(X_train_scaled, y_train)

    X_train_sel = rfe.transform(X_train_scaled).astype(np.float32)
    X_val_sel = rfe.transform(X_val_scaled).astype(np.float32)
    X_test_sel = rfe.transform(X_test_scaled).astype(np.float32)

    selected_mask = rfe.support_
    selected_indices = np.where(selected_mask)[0]

    print("Selected feature shape:", X_train_sel.shape)

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------
    ae1_loader = make_loader(
        X_train_sel, y_train,
        batch_size=pretrain_batch_size,
        shuffle=True,
    )

    finetune_train_loader = make_loader(
        X_train_sel, y_train,
        batch_size=finetune_batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val_sel, y_val,
        batch_size=finetune_batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        X_test_sel, y_test, batch_size=finetune_batch_size, shuffle=False,
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------
    model = StackedSAEClassifier(
        input_dim=X_train_sel.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    # -----------------------------------------------------
    # STAGE 1: SEQUENTIAL LAYER-WISE PRETRAINING
    # -----------------------------------------------------
    print("Starting AE1 pretraining...")
    ae1_optimizer = torch.optim.Adam(
        model.ae1.parameters(),
        lr=pretrain_lr,
        weight_decay=pretrain_weight_decay,
    )

    for epoch in range(1, pretrain_epochs + 1):
        tr_pre = pretrain_ae_epoch(
            autoencoder=model.ae1,
            loader=ae1_loader,
            optimizer=ae1_optimizer,
            device=device,
            rho=rho,
            beta=beta,
        )

        print(
            f"[Fold {fold} | AE1 {epoch:03d}/{pretrain_epochs}] "
            f"loss={tr_pre['loss']:.4f} "
            f"(recon={tr_pre['recon']:.4f}, kl={tr_pre['kl']:.4f})"
        )

    z1_fit = encode_array(model.ae1, X_train_sel, pretrain_batch_size, device)
    ae2_loader = make_loader(
        z1_fit, y_train, batch_size=pretrain_batch_size, shuffle=True
    )
    print("Starting AE2 pretraining...")
    ae2_optimizer = torch.optim.Adam(
        model.ae2.parameters(),
        lr=pretrain_lr,
        weight_decay=pretrain_weight_decay,
    )
    for epoch in range(1, pretrain_epochs + 1):
        tr_pre = pretrain_ae_epoch(
            autoencoder=model.ae2,
            loader=ae2_loader,
            optimizer=ae2_optimizer,
            device=device,
            rho=rho,
            beta=beta,
        )
        print(
            f"[Fold {fold} | AE2 {epoch:03d}/{pretrain_epochs}] "
            f"loss={tr_pre['loss']:.4f} "
            f"(recon={tr_pre['recon']:.4f}, kl={tr_pre['kl']:.4f})"
        )

    # -----------------------------------------------------
    # STAGE 2: FULL FINETUNE
    # -----------------------------------------------------
    print("Starting full finetuning...")
    finetune_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=finetune_weight_decay,
    )

    best_auc = -np.inf
    best_state = None
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, finetune_epochs + 1):
        tr_ft = finetune_epoch(
            model=model,
            loader=finetune_train_loader,
            optimizer=finetune_optimizer,
            device=device,
            rho=rho,
            beta=beta,
            recon_weight=recon_weight,
        )

        va = evaluate(model, val_loader, device, rho, beta, recon_weight)
        is_best = np.isfinite(va["auc"]) and va["auc"] > best_auc
        if is_best:
            best_auc = va["auc"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([
                fold, epoch, tr_ft["loss"], va["loss"], va["auc"], va["acc"],
                va["sensitivity"], va["specificity"], va["f1"],
                int(is_best), stale_epochs,
            ])
        print(
            f"[Fold {fold} | Finetune {epoch:03d}/{finetune_epochs}] "
            f"train_loss={tr_ft['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_auc={va['auc']:.4f} val_acc={va['acc']:.4f} "
            f"val_sens={va['sensitivity']:.4f} val_spec={va['specificity']:.4f} "
            f"val_f1={va['f1']:.4f} best={is_best} "
            f"stale={stale_epochs}/{patience}"
        )
        if stale_epochs >= patience:
            print(f"[Fold {fold}] early stop; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint was found for fold {fold}.")
    model.load_state_dict(best_state)
    heldout_metrics = evaluate(model, test_loader, device, rho, beta, recon_weight)

    print(
        f"Held-out fold {fold}: AUC={heldout_metrics['auc']:.4f}, "
        f"ACC={heldout_metrics['acc']:.4f}, SENS={heldout_metrics['sensitivity']:.4f}, "
        f"SPEC={heldout_metrics['specificity']:.4f}, F1={heldout_metrics['f1']:.4f}, "
        f"best_epoch={best_epoch}"
    )

    fold_summary = {
        "fold": fold,
        "best_epoch": int(best_epoch),
        "best_auc": float(best_auc),
        "auc": heldout_metrics["auc"],
        "accuracy": heldout_metrics["acc"],
        "sensitivity": heldout_metrics["sensitivity"],
        "specificity": heldout_metrics["specificity"],
        "f1": heldout_metrics["f1"],
        "n_selected_features": int(len(selected_indices)),
        "selected_feature_indices": selected_indices.tolist(),
    }
    fold_summaries.append(fold_summary)

    # save fold artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold,
            "best_epoch": best_epoch,
            "best_auc": best_auc,
            "heldout_metrics": heldout_metrics,
            "outer_train_indices": outer_train_idx,
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "test_indices": test_idx,
            "selected_feature_indices": selected_indices,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "config": {
                "n_features_to_select": n_features_to_select,
                "rho": rho,
                "beta": beta,
                "pretrain_epochs": pretrain_epochs,
                "finetune_epochs": finetune_epochs,
                "pretrain_batch_size": pretrain_batch_size,
                "finetune_batch_size": finetune_batch_size,
                "pretrain_lr": pretrain_lr,
                "finetune_lr": finetune_lr,
                "pretrain_weight_decay": pretrain_weight_decay,
                "finetune_weight_decay": finetune_weight_decay,
                "patience": patience,
                "inner_val_size": inner_val_size,
            },
        },
        os.path.join(save_dir, f"fold{fold}_best.pt"),
    )

    with open(os.path.join(save_dir, f"fold{fold}_summary.json"), "w") as f:
        json.dump(fold_summary, f, indent=2)


# =========================================================
# FINAL RESULT
# =========================================================
print("\n========== CV RESULT ==========")
cv_result = {"fold_summaries": fold_summaries}
for metric in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
    values = [summary[metric] for summary in fold_summaries]
    cv_result[f"{metric}_per_fold"] = values
    cv_result[f"{metric}_mean"] = float(np.mean(values))
    cv_result[f"{metric}_std"] = float(np.std(values))
    print(
        f"{metric.upper()}: {[f'{value:.4f}' for value in values]} | "
        f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    )

with open(os.path.join(save_dir, "cv_results.json"), "w") as f:
    json.dump(cv_result, f, indent=2)
