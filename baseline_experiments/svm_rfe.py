import os
import json
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import FCDataset


# =========================================================
# CONFIG
# =========================================================
xlsx_path = "data/abide_subjects_fc_with_demo_260202.xlsx"
label_col = "DX_GROUP"
n_rois = 116
n_splits = 5
seed = 42

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
pretrain_epochs = 50
pretrain_batch_size = 128
pretrain_lr = 1e-3
pretrain_weight_decay = 1e-4

# -------- finetuning --------
finetune_epochs = 50
finetune_batch_size = 128
finetune_lr = 1e-4
finetune_weight_decay = 1e-4

# -------- loss weights --------
recon_weight = 0.1

# -------- others --------
num_workers = 4
save_dir = "./sae_rfe_results"
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
            nn.ReLU(),
            nn.Dropout(dropout),
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

    return beta * torch.sum(kl)


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
        self.classifier = nn.Linear(hidden_dim2, n_classes)

    def forward(self, x):
        z1, x1 = self.ae1(x)
        z2, x2 = self.ae2(z1)
        logits = self.classifier(z2)
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
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# =========================================================
# PRETRAIN
# =========================================================
def pretrain_epoch(model, loader, optimizer, device, rho, beta):
    model.train()

    total_loss = 0.0
    total_recon1 = 0.0
    total_recon2 = 0.0
    total_kl1 = 0.0
    total_kl2 = 0.0
    total_n = 0

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)

        # AE1
        z1, x1 = model.ae1(xb)

        # AE2 (layer-wise target = encoded representation of AE1)
        z2, x2 = model.ae2(z1.detach())

        recon1 = F.mse_loss(x1, xb)
        recon2 = F.mse_loss(x2, z1.detach())
        kl1 = kl_divergence_sparse(z1, rho=rho, beta=beta)
        kl2 = kl_divergence_sparse(z2, rho=rho, beta=beta)

        loss = recon1 + recon2 + kl1 + kl2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_recon1 += recon1.item() * bs
        total_recon2 += recon2.item() * bs
        total_kl1 += kl1.item() * bs
        total_kl2 += kl2.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "recon1": total_recon1 / total_n,
        "recon2": total_recon2 / total_n,
        "kl1": total_kl1 / total_n,
        "kl2": total_kl2 / total_n,
    }


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
def evaluate(model, loader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        logits, _ = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_y.append(yb.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    acc = accuracy_score(all_y, all_preds)

    try:
        auc = roc_auc_score(all_y, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "acc": float(acc),
        "auc": float(auc),
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

all_fold_aucs = []
all_fold_accs = []
fold_summaries = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels), start=1):
    print(f"\n{'=' * 20} Fold {fold}/{n_splits} {'=' * 20}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val   shape:", X_val.shape, y_val.shape)

    # -----------------------------------------------------
    # RFE on TRAIN ONLY
    # -----------------------------------------------------
    print("Running RFE...")
    rfe = RFE(
        estimator=SVC(kernel="linear"),
        n_features_to_select=n_features_to_select,
        step=rfe_step,
    )
    rfe.fit(X_train, y_train)

    X_train_sel = rfe.transform(X_train).astype(np.float32)
    X_val_sel = rfe.transform(X_val).astype(np.float32)

    selected_mask = rfe.support_
    selected_indices = np.where(selected_mask)[0]

    print("Selected feature shape:", X_train_sel.shape)

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------
    pretrain_loader = make_loader(
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
    # STAGE 1: PRETRAIN
    # -----------------------------------------------------
    print("Starting pretraining...")
    pretrain_optimizer = torch.optim.Adam(
        list(model.ae1.parameters()) + list(model.ae2.parameters()),
        lr=pretrain_lr,
        weight_decay=pretrain_weight_decay,
    )

    for epoch in range(1, pretrain_epochs + 1):
        tr_pre = pretrain_epoch(
            model=model,
            loader=pretrain_loader,
            optimizer=pretrain_optimizer,
            device=device,
            rho=rho,
            beta=beta,
        )

        print(
            f"[Fold {fold} | Pretrain {epoch:03d}/{pretrain_epochs}] "
            f"loss={tr_pre['loss']:.4f} "
            f"(recon1={tr_pre['recon1']:.4f}, recon2={tr_pre['recon2']:.4f}, "
            f"kl1={tr_pre['kl1']:.4f}, kl2={tr_pre['kl2']:.4f})"
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
    best_acc = 0.0
    best_state = None
    best_epoch = -1

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

        va = evaluate(model, val_loader, device)

        print(
            f"[Fold {fold} | Finetune {epoch:03d}/{finetune_epochs}] "
            f"train_loss={tr_ft['loss']:.4f} "
            f"(cls={tr_ft['cls']:.4f}, recon={tr_ft['recon']:.4f}, kl={tr_ft['kl']:.4f}) | "
            f"val_auc={va['auc']:.4f}, val_acc={va['acc']:.4f}"
        )

        if np.isfinite(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            best_acc = va["acc"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best fold {fold}: AUC={best_auc:.4f}, ACC={best_acc:.4f}, epoch={best_epoch}")

    all_fold_aucs.append(best_auc)
    all_fold_accs.append(best_acc)

    fold_summary = {
        "fold": fold,
        "best_auc": float(best_auc),
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "n_selected_features": int(len(selected_indices)),
        "selected_feature_indices": selected_indices.tolist(),
    }
    fold_summaries.append(fold_summary)

    # save fold artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold,
            "best_auc": best_auc,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "selected_feature_indices": selected_indices,
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
print("AUC:", [f"{x:.4f}" for x in all_fold_aucs])
print("ACC:", [f"{x:.4f}" for x in all_fold_accs])
print(f"AUC Mean ± Std: {np.mean(all_fold_aucs):.4f} ± {np.std(all_fold_aucs):.4f}")
print(f"ACC Mean ± Std: {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}")

cv_result = {
    "auc_per_fold": [float(x) for x in all_fold_aucs],
    "acc_per_fold": [float(x) for x in all_fold_accs],
    "auc_mean": float(np.mean(all_fold_aucs)),
    "auc_std": float(np.std(all_fold_aucs)),
    "acc_mean": float(np.mean(all_fold_accs)),
    "acc_std": float(np.std(all_fold_accs)),
    "fold_summaries": fold_summaries,
}

with open(os.path.join(save_dir, "cv_results.json"), "w") as f:
    json.dump(cv_result, f, indent=2)