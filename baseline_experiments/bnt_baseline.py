import json
import csv
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import FCDataset
import datetime

xlsx_path = "data/abide_subjects_fc_with_demo_260202.xlsx"
label_col = "DX_GROUP"
n_rois, n_classes, n_splits, seed = 116, 2, 5, 40
d_model, n_heads, n_layers, n_clusters = 64, 4, 2, 8
ff_dim, dropout, epochs, batch_size = 128, 0.3, 100, 16
patience, inner_val_size = 30, 0.15
lr, weight_decay, num_workers = 1e-4, 1e-4, 4
orthogonal_reg_weight = 1e-3
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
save_dir = Path(f"./baseline_experiments/bnt_results/{timestamp}/")
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = dict(model="BNT_FC", n_rois=n_rois, d_model=d_model,
              n_heads=n_heads, n_layers=n_layers, n_clusters=n_clusters,
              ff_dim=ff_dim, dropout=dropout, epochs=epochs,
              batch_size=batch_size, lr=lr, weight_decay=weight_decay,
              orthogonal_reg_weight=orthogonal_reg_weight,
              n_splits=n_splits, seed=seed, positive_class="ASD",
              patience=patience, inner_val_size=inner_val_size)


def set_seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClusterReadout(nn.Module):
    """Learned cluster queries summarize transformer ROI embeddings."""
    def __init__(self):
        super().__init__()
        self.cluster_tokens = nn.Parameter(torch.randn(n_clusters, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        q = self.cluster_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        pooled, weights = self.attn(q, x, x, need_weights=True,
                                    average_attn_weights=True)
        return self.norm(pooled + q), weights

    def orthogonal_loss(self):
        q = F.normalize(self.cluster_tokens, dim=-1)
        gram = q @ q.t()
        eye = torch.eye(n_clusters, device=q.device)
        return (gram - eye).pow(2).mean()


class BrainNetworkTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(n_rois, d_model)
        self.roi_embedding = nn.Parameter(torch.randn(1, n_rois, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             norm=nn.LayerNorm(d_model))
        self.readout = ClusterReadout()
        self.classifier = nn.Sequential(
            nn.Linear(n_clusters * d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, n_classes)
        )

    def forward(self, fc):
        x = self.input_proj(fc) + self.roi_embedding
        x = self.encoder(x)
        clusters, assignments = self.readout(x)
        logits = self.classifier(clusters.flatten(1))
        return logits, assignments

    def regularization(self):
        return self.readout.orthogonal_loss()


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def train_epoch(model, loader, optimizer):
    model.train(); total = 0.0; n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb) + \
               orthogonal_reg_weight * model.regularization()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += loss.item() * xb.size(0); n += xb.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, probs, preds = [], [], []; total_loss = 0.0; total_n = 0
    for xb, yb in loader:
        logits, _ = model(xb.to(device)); p = torch.softmax(logits, 1)[:, 0]
        total_loss += F.cross_entropy(logits, yb.to(device)).item() * xb.size(0)
        total_n += xb.size(0)
        ys.append(yb.numpy()); probs.append(p.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(probs); pred = np.concatenate(preds)
    y_asd = (y == 0).astype(np.int64)
    return {"loss": total_loss / total_n,
            "acc": float(accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y_asd, p)),
            "sensitivity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
            "specificity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y, pred, pos_label=0, zero_division=0))}


dataset = FCDataset(xlsx_path=xlsx_path, n_rois=n_rois, label_col=label_col)
X = np.stack([dataset[i]["fc"].numpy() for i in range(len(dataset))]).astype(np.float32)
labels = np.array([dataset[i]["label"].item() for i in range(len(dataset))], dtype=np.int64)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
history_path = os.path.join(save_dir, "training_history.csv")
with open(history_path, "w", newline="") as f:
    csv.writer(f).writerow(["fold", "epoch", "train_loss", "val_loss", "val_auc",
        "val_accuracy", "val_sensitivity", "val_specificity", "val_f1",
        "is_best", "stale_epochs"])
fold_summaries = []
for fold, (outer_train_idx, test_idx) in enumerate(skf.split(X, labels), 1):
    set_seed(seed + fold)
    train_idx, val_idx = train_test_split(
        outer_train_idx, test_size=inner_val_size,
        stratify=labels[outer_train_idx], random_state=seed + fold
    )
    train_loader = make_loader(X[train_idx], labels[train_idx], True)
    val_loader = make_loader(X[val_idx], labels[val_idx], False)
    test_loader = make_loader(X[test_idx], labels[test_idx], False)
    model = BrainNetworkTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc, best_acc, best_epoch, best_state = -np.inf, 0.0, -1, None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer); va = evaluate(model, val_loader)
        is_best = va["auc"] > best_auc
        if is_best:
            best_auc, best_acc, best_epoch = va["auc"], va["acc"], epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([fold, epoch, loss, va["loss"], va["auc"],
                va["acc"], va["sensitivity"], va["specificity"], va["f1"],
                int(is_best), stale_epochs])
        print(f"[Fold {fold} | {epoch:03d}/{epochs}] train_loss={loss:.4f} "
              f"val_loss={va['loss']:.4f} val_auc={va['auc']:.4f} "
              f"val_acc={va['acc']:.4f} val_sens={va['sensitivity']:.4f} "
              f"val_spec={va['specificity']:.4f} val_f1={va['f1']:.4f} "
              f"best={is_best} stale={stale_epochs}/{patience}")
        if stale_epochs >= patience:
            print(f"[Fold {fold}] early stop; best epoch={best_epoch}")
            break
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    summary = {"fold": fold, "best_auc": test_metrics["auc"],
               "best_acc": test_metrics["acc"],
               "best_sensitivity": test_metrics["sensitivity"],
               "best_specificity": test_metrics["specificity"],
               "best_f1": test_metrics["f1"], "best_epoch": best_epoch,
               "selection_val_auc": best_auc}
    fold_summaries.append(summary)
    torch.save({"model_state_dict": best_state, "config": config,
                "outer_train_indices": outer_train_idx, "train_indices": train_idx,
                "validation_indices": val_idx, "test_indices": test_idx, **summary},
               os.path.join(save_dir, f"fold{fold}_best.pt"))
    with open(os.path.join(save_dir, f"fold{fold}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

aucs = [x["best_auc"] for x in fold_summaries]; accs = [x["best_acc"] for x in fold_summaries]
sens = [x["best_sensitivity"] for x in fold_summaries]; spec = [x["best_specificity"] for x in fold_summaries]
f1s = [x["best_f1"] for x in fold_summaries]
result = {"auc_per_fold": aucs, "acc_per_fold": accs,
          "sensitivity_per_fold": sens, "specificity_per_fold": spec, "f1_per_fold": f1s,
          "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
          "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
          "sensitivity_mean": float(np.mean(sens)), "sensitivity_std": float(np.std(sens)),
          "specificity_mean": float(np.mean(spec)), "specificity_std": float(np.std(spec)),
          "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
          "fold_summaries": fold_summaries}
with open(os.path.join(save_dir, "cv_results.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
