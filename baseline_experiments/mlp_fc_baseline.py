"""MLP baseline for upper-triangular functional-connectivity features.

"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import FCDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP baseline for FC data")
    parser.add_argument("--xlsx-path", default="data/abide_subjects_fc_with_demo_260202.xlsx")
    parser.add_argument("--n-rois", type=int, default=116)
    parser.add_argument("--label-col", default="DX_GROUP")
    parser.add_argument("--output-dir", default="baseline_experiments/mlp_results")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 128])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--inner-val-size", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fc_vectors(dataset: FCDataset) -> Tuple[np.ndarray, np.ndarray]:
    triu = np.triu_indices(dataset.n_rois, k=1)
    features, labels = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        features.append(sample["fc"].numpy()[triu])
        labels.append(int(sample["label"].item()))

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(f"Expected mapped labels {{0, 1}}, found {np.unique(y)}")
    return x, y


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probs = torch.softmax(logits, dim=1)
        labels.append(y_batch.numpy())
        probabilities.append(probs[:, 0].cpu().numpy())  # ASD probability
    return np.concatenate(labels), np.concatenate(probabilities)


@torch.no_grad()
def evaluate_with_loss(model, loader, criterion, device):
    model.eval()
    labels, probabilities = [], []
    total_loss, total_n = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch, y_device = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_device)
        total_loss += loss.item() * x_batch.size(0)
        total_n += x_batch.size(0)
        labels.append(y_batch.numpy())
        probabilities.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities), total_loss / total_n


def asd_auc(y: np.ndarray, p_asd: np.ndarray) -> float:
    return float(roc_auc_score((y == 0).astype(np.int64), p_asd))


def classification_metrics(
    y: np.ndarray, p_asd: np.ndarray, threshold: float
) -> Dict[str, float]:
    # p_asd >= threshold predicts ASD=0; otherwise TD=1.
    pred = np.where(p_asd >= threshold, 0, 1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    asd_correct, asd_as_td = cm[0, 0], cm[0, 1]
    td_as_asd, td_correct = cm[1, 0], cm[1, 1]

    sensitivity = asd_correct / max(asd_correct + asd_as_td, 1)
    specificity = td_correct / max(td_correct + td_as_asd, 1)
    return {
        "auc": asd_auc(y, p_asd),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "threshold": float(threshold),
        "n_pred_asd": int((pred == 0).sum()),
        "n_pred_td": int((pred == 1).sum()),
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_fold(
    fold: int,
    x: np.ndarray,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)

    train_idx, val_idx = train_test_split(
        outer_train_idx,
        test_size=args.inner_val_size,
        stratify=y[outer_train_idx],
        random_state=fold_seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
    x_val = scaler.transform(x[val_idx]).astype(np.float32)
    x_test = scaler.transform(x[test_idx]).astype(np.float32)

    train_loader = make_loader(
        x_train, y[train_idx], args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(
        x_val, y[val_idx], args.batch_size, False, args.num_workers
    )
    test_loader = make_loader(
        x_test, y[test_idx], args.batch_size, False, args.num_workers
    )
    model = MLP(x.shape[1], args.hidden_dims, args.dropout).to(device)

    counts = np.bincount(y[train_idx], minlength=2).astype(np.float32)
    class_weights = len(train_idx) / (2.0 * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x_batch.size(0)
            train_n += x_batch.size(0)

        train_loss = train_loss_sum / train_n
        val_y, val_p_asd, val_loss = evaluate_with_loss(
            model, val_loader, criterion, device
        )
        val_auc = asd_auc(val_y, val_p_asd)
        val_metrics = classification_metrics(val_y, val_p_asd, 0.5)
        is_best = val_auc > best_auc + 1e-6
        if is_best:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        with (output_dir / "training_history.csv").open("a", newline="") as f:
            csv.writer(f).writerow([
                fold, epoch, train_loss, val_loss, val_auc,
                val_metrics["accuracy"], val_metrics["sensitivity"],
                val_metrics["specificity"], val_metrics["f1"],
                int(is_best), stale_epochs,
            ])
        print(
            f"[Fold {fold} | {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} best={is_best} "
            f"stale={stale_epochs}/{args.patience}"
        )
        if stale_epochs >= args.patience:
            print(f"[Fold {fold}] early stop; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold}: no valid checkpoint was produced")
    model.load_state_dict(best_state)

    threshold = 0.5
    test_y, test_p_asd = predict(model, test_loader, device)
    metrics = classification_metrics(test_y, test_p_asd, threshold)
    metrics.update(
        {
            "fold": fold,
            "best_epoch": best_epoch,
            "selection_val_auc": float(best_auc),
            "test_p_asd_mean": float(test_p_asd.mean()),
            "test_p_asd_std": float(test_p_asd.std()),
        }
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": x.shape[1],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "threshold": threshold,
            "outer_train_idx": outer_train_idx,
            "train_idx": train_idx,
            "validation_idx": val_idx,
            "test_idx": test_idx,
            "metrics": metrics,
        },
        output_dir / f"fold{fold}_best.pt",
    )

    print(
        f"Fold {fold}: AUC={metrics['auc']:.4f}, "
        f"Acc={metrics['accuracy']:.4f}, Sen(ASD)={metrics['sensitivity']:.4f}, "
        f"Spe(TD)={metrics['specificity']:.4f}, F1(ASD)={metrics['f1']:.4f}, "
        f"thr={threshold:.4f}, pred ASD/TD="
        f"{metrics['n_pred_asd']}/{metrics['n_pred_td']}"
    )
    return metrics


def save_results(results: list[Dict[str, float]], output_dir: Path) -> None:
    fieldnames = list(results[0].keys())
    with (output_dir / "fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in results])),
            "std": float(np.std([row[name] for row in results])),
        }
        for name in metric_names
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nFive-fold summary (mean +/- population std)")
    for name in metric_names:
        print(f"{name:>11}: {summary[name]['mean']:.4f} +/- {summary[name]['std']:.4f}")

import datetime

def main() -> None:
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(f"{args.output_dir}/{timestamp}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    dataset = FCDataset(
        xlsx_path=args.xlsx_path,
        n_rois=args.n_rois,
        label_col=args.label_col,
    )
    x, y = load_fc_vectors(dataset)
    print(f"Features: {x.shape}; labels ASD(0)/TD(1): {(y == 0).sum()}/{(y == 1).sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    with (output_dir / "training_history.csv").open("w", newline="") as f:
        csv.writer(f).writerow([
            "fold", "epoch", "train_loss", "val_loss", "val_auc",
            "val_accuracy", "val_sensitivity", "val_specificity", "val_f1",
            "is_best", "stale_epochs",
        ])

    results = []
    for fold, (outer_train_idx, test_idx) in enumerate(cv.split(x, y), 1):
        results.append(
            train_one_fold(
                fold,
                x,
                y,
                outer_train_idx,
                test_idx,
                args,
                device,
                output_dir,
            )
        )
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
