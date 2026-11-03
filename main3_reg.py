"""HiTraNet regression for predicting ADOS_TOTAL from resting-state FC.

This script keeps the original HiTraNet architecture and replaces the binary
classification task with a leakage-safe regression protocol:

1. retain ASD participants with a valid ADOS_TOTAL value;
2. create site-aware 5-fold splits balanced by ADOS quantile bins;
3. standardize ADOS using the training partition of each fold only;
4. select checkpoints using validation MAE;
5. evaluate the selected checkpoint once on the held-out test fold.

The existing project modules ``dataset.py``, ``model.py``, and ``utils.py`` are
expected to be available in the same environment as the original training
script.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import FCDataset
from model import HiTraNet
from utils import AverageMeter, set_seed


def _scalar(value):
    if torch.is_tensor(value):
        return value.item()
    return value


def _id_key(value) -> str:
    """Normalize ABIDE subject identifiers for spreadsheet/dataset matching."""
    value = _scalar(value)
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-5:] if digits else text.lower()


def _category_key(value) -> str:
    """Normalize numeric or string category values such as 1 and 1.0."""
    value = _scalar(value)
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    except ValueError:
        return text.lower()


def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    return pd.read_csv(path)


class ADOSRegressionDataset(Dataset):
    """Attach a valid ADOS_TOTAL target to each FC sample.

    By default, only ASD participants are retained because ADOS_TOTAL is not a
    meaningful zero-valued target for typically developing controls. Matching
    is performed by subject ID, not by row position, whenever SUB_ID exists.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        table_path: str,
        ados_col: str,
        subject_col: str,
        diagnosis_col: str,
        site_col: str,
        asd_only: bool,
        asd_value: str,
    ):
        self.base = base_dataset
        table = _load_table(table_path).copy()

        required = [ados_col]
        if asd_only:
            required.append(diagnosis_col)
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise KeyError(
                f"Missing required columns {missing}; available columns are "
                f"{table.columns.tolist()}"
            )

        table[ados_col] = pd.to_numeric(table[ados_col], errors="coerce")

        row_by_id = None
        if subject_col in table.columns:
            row_by_id = {
                _id_key(row[subject_col]): row
                for _, row in table.iterrows()
                if _id_key(row[subject_col])
            }

        expected_asd = _category_key(asd_value)
        base_indices: List[int] = []
        targets: List[float] = []
        subject_ids: List[str] = []
        sites: List[str] = []
        skipped_missing = 0
        skipped_non_asd = 0

        for base_index in range(len(base_dataset)):
            item = base_dataset[base_index]
            subject = item.get("sub_id", base_index)
            subject_key = _id_key(subject)

            if row_by_id is not None:
                row = row_by_id.get(subject_key)
                if row is None:
                    skipped_missing += 1
                    continue
            else:
                if base_index >= len(table):
                    skipped_missing += 1
                    continue
                row = table.iloc[base_index]

            target = row[ados_col]
            if pd.isna(target) or not np.isfinite(float(target)) or float(target) < 0:
                skipped_missing += 1
                continue

            if asd_only and _category_key(row[diagnosis_col]) != expected_asd:
                skipped_non_asd += 1
                continue

            site = row[site_col] if site_col in table.columns else item.get(site_col, "UNKNOWN")
            if pd.isna(site) or str(site).strip() in {"", "nan", "None"}:
                site = "UNKNOWN"

            base_indices.append(base_index)
            targets.append(float(target))
            subject_ids.append(str(subject))
            sites.append(str(site))

        if not base_indices:
            raise ValueError(
                f"No usable subjects remained for {ados_col}. Check subject ID "
                f"matching, diagnosis coding, and missing values."
            )

        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.subject_ids = np.asarray(subject_ids, dtype=object)
        self.sites = np.asarray(sites, dtype=str)

        print(
            f"ADOS dataset: N={len(self.base_indices)}, "
            f"missing/unmatched skipped={skipped_missing}, "
            f"non-ASD skipped={skipped_non_asd}"
        )

    def __len__(self):
        return len(self.base_indices)

    def __getitem__(self, index):
        item = dict(self.base[int(self.base_indices[index])])
        target = float(self.targets[index])
        item["label"] = torch.tensor(target, dtype=torch.float32)
        item["ados_total"] = torch.tensor(target, dtype=torch.float32)
        item["sub_id"] = str(self.subject_ids[index])
        item["SITE_ID"] = str(self.sites[index])
        return item


def _largest_remainder_sizes(n_samples: int, proportions: Sequence[float]) -> np.ndarray:
    proportions = np.asarray(proportions, dtype=float)
    if proportions.ndim != 1 or len(proportions) < 2:
        raise ValueError("proportions must contain at least two values")
    if np.any(proportions <= 0):
        raise ValueError("all proportions must be positive")

    proportions = proportions / proportions.sum()
    raw = proportions * n_samples
    sizes = np.floor(raw).astype(int)
    remainder = int(n_samples - sizes.sum())
    if remainder:
        order = np.argsort(-(raw - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    return sizes


def make_ados_strata(targets: np.ndarray, n_bins: int) -> np.ndarray:
    """Create approximately equal-frequency bins for regression stratification."""
    targets = np.asarray(targets, dtype=float)
    if len(targets) < 2 or np.nanstd(targets) == 0:
        raise ValueError("ADOS_TOTAL must contain at least two distinct values")

    n_bins = max(2, min(int(n_bins), len(targets)))
    # Ranking with method='first' keeps bins usable when many subjects have the
    # same integer ADOS score while preserving the overall score ordering.
    ranked = pd.Series(targets).rank(method="first")
    strata = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    strata = np.asarray(strata, dtype=np.int64)
    if len(np.unique(strata)) < 2:
        raise ValueError("Unable to construct at least two ADOS quantile bins")
    return strata


def _make_balance_targets(strata: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Encode ADOS-bin, site, and their joint category as multilabel targets."""
    tokens_per_subject = []
    all_tokens = set()

    for stratum, site in zip(strata, sites):
        tokens = (
            f"ados_bin={int(stratum)}",
            f"site={site}",
            f"ados_bin={int(stratum)}|site={site}",
        )
        tokens_per_subject.append(tokens)
        all_tokens.update(tokens)

    token_to_col = {token: col for col, token in enumerate(sorted(all_tokens))}
    encoded = np.zeros((len(strata), len(token_to_col)), dtype=np.int8)
    for row, tokens in enumerate(tokens_per_subject):
        for token in tokens:
            encoded[row, token_to_col[token]] = 1
    return encoded


def iterative_site_ados_partition(
    indices: np.ndarray,
    strata: np.ndarray,
    sites: np.ndarray,
    proportions: Sequence[float],
    random_state: int,
) -> List[np.ndarray]:
    """Partition indices while approximately balancing ADOS bins and sites."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("cannot split an empty index array")

    rng = np.random.default_rng(random_state)
    local_targets = _make_balance_targets(strata[indices], sites[indices])
    target_sizes = _largest_remainder_sizes(len(indices), proportions)
    weights = target_sizes / target_sizes.sum()
    desired = np.outer(weights, local_targets.sum(axis=0)).astype(float)
    capacity = target_sizes.copy()

    assignment = np.full(len(indices), -1, dtype=np.int64)
    unassigned = np.ones(len(indices), dtype=bool)

    while np.any(unassigned):
        remaining_counts = local_targets[unassigned].sum(axis=0)
        positive_cols = np.flatnonzero(remaining_counts > 0)

        if len(positive_cols):
            rare_count = remaining_counts[positive_cols].min()
            rare_cols = positive_cols[remaining_counts[positive_cols] == rare_count]
            selected_col = int(rng.choice(rare_cols))
            candidates = np.flatnonzero(
                unassigned & (local_targets[:, selected_col] == 1)
            )
        else:
            selected_col = None
            candidates = np.flatnonzero(unassigned)

        rng.shuffle(candidates)
        for sample_index in candidates:
            if not unassigned[sample_index]:
                continue
            eligible = np.flatnonzero(capacity > 0)
            if not len(eligible):
                raise RuntimeError("split capacity was exhausted unexpectedly")

            sample_cols = np.flatnonzero(local_targets[sample_index] == 1)
            if selected_col is not None:
                need = desired[eligible, selected_col]
                best = eligible[np.isclose(need, need.max())]
            else:
                best = eligible

            if len(best) > 1:
                joint_need = desired[np.ix_(best, sample_cols)].sum(axis=1)
                best = best[np.isclose(joint_need, joint_need.max())]
            if len(best) > 1:
                best = best[capacity[best] == capacity[best].max()]

            chosen = int(rng.choice(best))
            assignment[sample_index] = chosen
            unassigned[sample_index] = False
            capacity[chosen] -= 1
            desired[chosen] -= local_targets[sample_index]

    parts = [
        np.sort(indices[assignment == part_index])
        for part_index in range(len(target_sizes))
    ]
    if [len(part) for part in parts] != target_sizes.tolist():
        raise RuntimeError("split produced unexpected partition sizes")
    return parts


def build_cv_splits(
    targets: np.ndarray,
    sites: np.ndarray,
    n_splits: int,
    inner_val_size: float,
    random_seed: int,
    n_ados_bins: int,
):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0 < inner_val_size < 1:
        raise ValueError("inner_val_size must be between 0 and 1")
    if len(targets) < n_splits:
        raise ValueError("number of subjects must be at least n_splits")

    strata = make_ados_strata(targets, n_ados_bins)
    all_indices = np.arange(len(targets), dtype=np.int64)
    test_parts = iterative_site_ados_partition(
        all_indices,
        strata,
        sites,
        np.ones(n_splits, dtype=float),
        random_seed,
    )

    records = []
    for fold, test_idx in enumerate(test_parts, start=1):
        outer_train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        train_idx, val_idx = iterative_site_ados_partition(
            outer_train_idx,
            strata,
            sites,
            [1.0 - inner_val_size, inner_val_size],
            random_seed + fold,
        )

        split_sets = [set(train_idx), set(val_idx), set(test_idx)]
        if any(
            split_sets[i] & split_sets[j]
            for i in range(3)
            for j in range(i + 1, 3)
        ):
            raise RuntimeError(f"Fold {fold}: split overlap detected")
        if set.union(*split_sets) != set(all_indices):
            raise RuntimeError(f"Fold {fold}: split does not cover the dataset")
        if np.std(targets[train_idx]) == 0:
            raise RuntimeError(f"Fold {fold}: training ADOS targets are constant")

        records.append(
            {
                "fold": fold,
                "outer_train_idx": outer_train_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
        )
    return records, strata


def split_summary(
    indices: np.ndarray,
    targets: np.ndarray,
    sites: np.ndarray,
    strata: np.ndarray,
) -> Dict:
    indices = np.asarray(indices, dtype=np.int64)
    values = targets[indices]
    site_counts = pd.Series(sites[indices]).value_counts().sort_index()
    bin_counts = pd.Series(strata[indices]).value_counts().sort_index()
    return {
        "n": int(len(indices)),
        "target_mean": float(np.mean(values)),
        "target_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "target_median": float(np.median(values)),
        "target_min": float(np.min(values)),
        "target_max": float(np.max(values)),
        "site_counts": {str(k): int(v) for k, v in site_counts.items()},
        "ados_bin_counts": {str(k): int(v) for k, v in bin_counts.items()},
    }


def build_model(args, device):
    # n_classes=1 changes only the prediction head; the hierarchical encoder,
    # seed selection, and trajectory construction remain unchanged.
    return HiTraNet(
        n_rois=args.n_rois,
        d_model=args.d_model,
        n_hops=args.n_hops,
        top_m=args.top_m,
        mlp_hidden=args.mlp_hidden,
        n_classes=1,
        depth=args.depth,
        dropout=args.dropout,
        seed_std=args.seed_std,
        act=args.act,
        use_gated_residual=True,
    ).to(device)


def regression_loss(prediction, target, loss_name: str):
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "mae":
        return F.l1_loss(prediction, target)
    return F.smooth_l1_loss(prediction, target)


def _safe_correlations(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pearson_result = pearsonr(y_true, y_pred)
    spearman_result = spearmanr(y_true, y_pred)
    pearson_value = getattr(pearson_result, "statistic", pearson_result[0])
    pearson_p = getattr(pearson_result, "pvalue", pearson_result[1])
    spearman_value = getattr(spearman_result, "statistic", spearman_result[0])
    spearman_p = getattr(spearman_result, "pvalue", spearman_result[1])
    return (
        float(pearson_value),
        float(pearson_p),
        float(spearman_value),
        float(spearman_p),
    )


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson_r, pearson_p, spearman_rho, spearman_p = _safe_correlations(
        y_true, y_pred
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    target_mean: float,
    target_std: float,
    loss_name: str,
):
    model.train()
    loss_meter = AverageMeter()
    true_values = []
    predictions = []

    for batch in loader:
        fc = batch["fc"].to(device)
        target = batch["label"].to(device).float().reshape(-1)
        scaled_target = (target - target_mean) / target_std

        output, _ = model(fc)
        scaled_prediction = output.reshape(-1)
        loss = regression_loss(scaled_prediction, scaled_target, loss_name)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        raw_prediction = scaled_prediction.detach() * target_std + target_mean
        loss_meter.update(loss.item(), n=fc.size(0))
        true_values.append(target.detach().cpu().numpy())
        predictions.append(raw_prediction.cpu().numpy())

    y_true = np.concatenate(true_values)
    y_pred = np.concatenate(predictions)
    metrics = regression_metrics(y_true, y_pred)
    return {"loss": loss_meter.avg, **metrics}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    target_mean: float,
    target_std: float,
    loss_name: str,
    collect_analysis: bool = False,
):
    model.eval()
    loss_meter = AverageMeter()
    true_values = []
    predictions = []
    subjects = []
    all_g, all_p, all_h = [], [], []

    for batch in loader:
        fc = batch["fc"].to(device)
        target = batch["label"].to(device).float().reshape(-1)
        scaled_target = (target - target_mean) / target_std

        output, aux = model(fc)
        scaled_prediction = output.reshape(-1)
        loss = regression_loss(scaled_prediction, scaled_target, loss_name)
        raw_prediction = scaled_prediction * target_std + target_mean

        loss_meter.update(loss.item(), n=fc.size(0))
        true_values.append(target.cpu().numpy())
        predictions.append(raw_prediction.cpu().numpy())
        subjects.extend([str(subject) for subject in batch["sub_id"]])

        if collect_analysis:
            all_g.append(
                torch.stack(aux["g_seed_list"], dim=0)
                .permute(1, 0, 2, 3)
                .cpu()
            )
            all_p.append(
                torch.stack(aux["p_list"], dim=0)
                .permute(1, 0, 2, 3)
                .cpu()
            )
            all_h.append(
                torch.stack(aux["H_list"], dim=0)
                .permute(1, 0, 2, 3, 4)
                .cpu()
            )

    y_true = np.concatenate(true_values)
    y_pred = np.concatenate(predictions)
    metrics = regression_metrics(y_true, y_pred)
    return {
        "loss": loss_meter.avg,
        **metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "subject": subjects,
        "g": torch.cat(all_g, dim=0) if collect_analysis else None,
        "p": torch.cat(all_p, dim=0) if collect_analysis else None,
        "H": torch.cat(all_h, dim=0) if collect_analysis else None,
    }


def make_loader(dataset, indices, args, shuffle: bool):
    indices = np.asarray(indices, dtype=np.int64)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict ADOS_TOTAL from AAL-116 FC using HiTraNet"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/abide_subjects_fc_with_demo_260202.xlsx",
    )
    parser.add_argument("--ados_col", type=str, default="ADOS_TOTAL")
    parser.add_argument("--subject_col", type=str, default="SUB_ID")
    parser.add_argument("--diagnosis_col", type=str, default="DX_GROUP")
    parser.add_argument("--site_col", type=str, default="SITE_ID")
    parser.add_argument(
        "--asd_value",
        type=str,
        default="1",
        help="Raw spreadsheet value denoting ASD; ABIDE convention is 1",
    )
    parser.add_argument(
        "--asd_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain only ASD participants for ADOS regression",
    )

    parser.add_argument("--n_rois", type=int, default=116)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_hops", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--top_m", type=int, default=5)
    parser.add_argument("--seed_std", type=float, default=0.5)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--act", choices=["gelu", "relu"], default="gelu")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["smooth_l1", "mse", "mae"], default="smooth_l1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--inner_val_size", type=float, default=0.15)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_ados_bins", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--use_scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument(
        "--save_root",
        type=str,
        default="/data5/sohyunkang/experiments2",
    )
    parser.add_argument(
        "--save_model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save hop-wise g/p/H tensors for the selected checkpoint",
    )
    return parser.parse_args()


def _fold_metric_summary(fold_results: Iterable[Dict], metric_names: Sequence[str]):
    fold_results = list(fold_results)
    return {
        "folds": fold_results,
        "mean": {
            name: float(np.nanmean([record[name] for record in fold_results]))
            for name in metric_names
        },
        "std": {
            name: (
                float(np.nanstd([record[name] for record in fold_results], ddof=1))
                if len(fold_results) > 1
                else 0.0
            )
            for name in metric_names
        },
    }


def main():
    args = parse_args()
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_dir = Path(args.save_root) / f"w{args.worker_id}_hitranet_ados_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print("Experiment dir:", exp_dir)

    base_dataset = FCDataset(
        xlsx_path=args.csv,
        n_rois=args.n_rois,
        label_col=args.diagnosis_col,
    )
    dataset = ADOSRegressionDataset(
        base_dataset=base_dataset,
        table_path=args.csv,
        ados_col=args.ados_col,
        subject_col=args.subject_col,
        diagnosis_col=args.diagnosis_col,
        site_col=args.site_col,
        asd_only=args.asd_only,
        asd_value=args.asd_value,
    )

    targets = dataset.targets.astype(np.float64)
    sites = dataset.sites.astype(str)
    print(
        f"ADOS_TOTAL: mean={targets.mean():.4f}, std={targets.std(ddof=1):.4f}, "
        f"range=({targets.min():.4f}, {targets.max():.4f})"
    )

    splits, strata = build_cv_splits(
        targets=targets,
        sites=sites,
        n_splits=args.n_splits,
        inner_val_size=args.inner_val_size,
        random_seed=args.random_seed,
        n_ados_bins=args.n_ados_bins,
    )

    manifest = {
        "task": "ADOS_TOTAL regression",
        "strategy": "iterative_multilabel_ados_quantile_site_and_joint",
        "description": (
            "Outer test folds and inner validation sets approximately preserve "
            "ADOS quantile, site, and ADOS-quantile-by-site distributions."
        ),
        "random_seed": int(args.random_seed),
        "n_splits": int(args.n_splits),
        "inner_val_size": float(args.inner_val_size),
        "n_ados_bins": int(len(np.unique(strata))),
        "folds": [],
    }

    for record in splits:
        manifest["folds"].append(
            {
                "fold": int(record["fold"]),
                "outer_train_indices": record["outer_train_idx"].tolist(),
                "train_indices": record["train_idx"].tolist(),
                "validation_indices": record["val_idx"].tolist(),
                "test_indices": record["test_idx"].tolist(),
                "train_distribution": split_summary(
                    record["train_idx"], targets, sites, strata
                ),
                "validation_distribution": split_summary(
                    record["val_idx"], targets, sites, strata
                ),
                "test_distribution": split_summary(
                    record["test_idx"], targets, sites, strata
                ),
            }
        )

    with open(exp_dir / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    with open(exp_dir / "site_ados_aware_splits.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    history_path = exp_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "fold", "epoch", "lr", "train_loss", "train_mae",
                "val_loss", "val_mae", "val_rmse", "val_r2",
                "val_pearson_r", "val_spearman_rho", "is_best",
                "stale_epochs",
            ]
        )

    metric_names = [
        "mae", "rmse", "r2", "pearson_r", "pearson_p",
        "spearman_rho", "spearman_p",
    ]
    fold_results = []

    for record in splits:
        fold = int(record["fold"])
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n========== Fold {fold}/{args.n_splits} ==========")
        set_seed(args.random_seed + fold)
        train_idx = record["train_idx"]
        val_idx = record["val_idx"]
        test_idx = record["test_idx"]

        target_mean = float(np.mean(targets[train_idx]))
        target_std = float(np.std(targets[train_idx], ddof=0))
        if not np.isfinite(target_std) or target_std < 1e-8:
            raise RuntimeError(f"Fold {fold}: invalid training target std={target_std}")

        print(
            f"Target scaler from TRAIN only: mean={target_mean:.4f}, "
            f"std={target_std:.4f}"
        )
        print("Train:", split_summary(train_idx, targets, sites, strata))
        print("Val  :", split_summary(val_idx, targets, sites, strata))
        print("Test :", split_summary(test_idx, targets, sites, strata))

        train_loader = make_loader(dataset, train_idx, args, shuffle=True)
        val_loader = make_loader(dataset, val_idx, args, shuffle=False)
        test_loader = make_loader(dataset, test_idx, args, shuffle=False)

        model = build_model(args, device)
        if fold == 1:
            with open(exp_dir / "model_arch.txt", "w", encoding="utf-8") as file:
                file.write(str(model))

        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters()},
                {"params": model.head.parameters()},
                {
                    "params": model.periphery_seedness.parameters(),
                    "lr": args.seed_lr,
                    "weight_decay": 0.0,
                },
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = None
        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                min_lr=args.min_lr,
            )

        best_val_mae = math.inf
        best_epoch = 0
        best_state = None
        best_val_metrics = None
        stale_epochs = 0
        best_path = exp_dir / f"fold{fold}_best_model.pt"

        for epoch in range(1, args.epochs + 1):
            model.current_epoch = epoch
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                target_mean,
                target_std,
                args.loss,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                target_mean,
                target_std,
                args.loss,
            )

            if scheduler is not None and np.isfinite(val_metrics["mae"]):
                scheduler.step(val_metrics["mae"])

            eligible = epoch >= args.min_epochs
            improved = (
                eligible
                and np.isfinite(val_metrics["mae"])
                and val_metrics["mae"] < best_val_mae - args.min_delta
            )

            if improved:
                best_val_mae = float(val_metrics["mae"])
                best_epoch = epoch
                best_val_metrics = {
                    key: float(val_metrics[key]) for key in ["loss", *metric_names]
                }
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
                if args.save_model:
                    torch.save(
                        {
                            "model": best_state,
                            "args": vars(args),
                            "fold": fold,
                            "best_epoch": best_epoch,
                            "selection_metric": "validation_mae",
                            "selection_value": best_val_mae,
                            "target_mean": target_mean,
                            "target_std": target_std,
                            "outer_train_indices": record["outer_train_idx"],
                            "train_indices": train_idx,
                            "validation_indices": val_idx,
                            "test_indices": test_idx,
                        },
                        best_path,
                    )
            elif eligible:
                stale_epochs += 1

            current_lr = float(optimizer.param_groups[0]["lr"])
            with open(history_path, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    [
                        fold, epoch, current_lr,
                        train_metrics["loss"], train_metrics["mae"],
                        val_metrics["loss"], val_metrics["mae"],
                        val_metrics["rmse"], val_metrics["r2"],
                        val_metrics["pearson_r"], val_metrics["spearman_rho"],
                        int(improved), stale_epochs,
                    ]
                )

            print(
                f"[Fold {fold} | Epoch {epoch:03d}/{args.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_MAE={train_metrics['mae']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_MAE={val_metrics['mae']:.4f} "
                f"val_RMSE={val_metrics['rmse']:.4f} "
                f"val_R2={val_metrics['r2']:.4f} "
                f"val_r={val_metrics['pearson_r']:.4f} | "
                f"lr={current_lr:.2e} best={improved} "
                f"stale={stale_epochs}/{args.patience}"
            )

            if eligible and stale_epochs >= args.patience:
                print(
                    f"[Fold {fold}] Early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}, best val MAE={best_val_mae:.4f}"
                )
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold}: no valid checkpoint was produced")

        model.load_state_dict(best_state)
        val_final = evaluate(
            model,
            val_loader,
            device,
            target_mean,
            target_std,
            args.loss,
            collect_analysis=False,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            target_mean,
            target_std,
            args.loss,
            collect_analysis=args.save_aux,
        )

        seed_idx, seed_weight = model.topm_seeds()
        order = torch.argsort(seed_weight.detach().cpu(), descending=True)
        sorted_seed_idx = seed_idx.detach().cpu()[order]
        sorted_seed_weight = seed_weight.detach().cpu()[order]

        prediction_frame = pd.DataFrame(
            {
                "sub_id": test_metrics["subject"],
                "ados_true": test_metrics["y_true"],
                "ados_pred": test_metrics["y_pred"],
                "residual": test_metrics["y_true"] - test_metrics["y_pred"],
            }
        )
        prediction_frame.to_csv(
            exp_dir / f"fold{fold}_test_predictions.csv", index=False
        )

        if args.save_aux:
            torch.save(
                {
                    "best_epoch": best_epoch,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "test_subject": test_metrics["subject"],
                    "test_y": test_metrics["y_true"],
                    "test_pred": test_metrics["y_pred"],
                    "test_g": test_metrics["g"],
                    "test_p": test_metrics["p"],
                    "test_H": test_metrics["H"],
                    "seed_indices": sorted_seed_idx,
                    "seed_weights": sorted_seed_weight,
                    "test_indices": test_idx,
                },
                exp_dir / f"fold{fold}_best_aux.pt",
            )

        fold_result = {
            "fold": fold,
            "best_epoch": int(best_epoch),
            "selection_metric": "validation_mae",
            "selection_value": float(best_val_mae),
            "target_mean": target_mean,
            "target_std": target_std,
            "validation_metrics": {
                key: float(val_final[key]) for key in metric_names
            },
            **{key: float(test_metrics[key]) for key in metric_names},
            "seed_indices": sorted_seed_idx.tolist(),
            "seed_weights": sorted_seed_weight.tolist(),
        }
        fold_results.append(fold_result)

        with open(
            exp_dir / f"fold{fold}_metrics.json", "w", encoding="utf-8"
        ) as file:
            json.dump(fold_result, file, indent=2)

        if args.save_model:
            torch.save(
                {
                    "model": best_state,
                    "args": vars(args),
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "selection_metric": "validation_mae",
                    "selection_value": best_val_mae,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "test_metrics": {
                        key: float(test_metrics[key]) for key in metric_names
                    },
                    "outer_train_indices": record["outer_train_idx"],
                    "train_indices": train_idx,
                    "validation_indices": val_idx,
                    "test_indices": test_idx,
                },
                best_path,
            )

        print(
            f"Fold {fold} TEST -> "
            f"MAE={test_metrics['mae']:.4f}, "
            f"RMSE={test_metrics['rmse']:.4f}, "
            f"R2={test_metrics['r2']:.4f}, "
            f"Pearson r={test_metrics['pearson_r']:.4f}, "
            f"Spearman rho={test_metrics['spearman_rho']:.4f}, "
            f"EPOCH={best_epoch}"
        )

    if not fold_results:
        raise RuntimeError("No fold was executed; check --fold")

    summary = _fold_metric_summary(fold_results, metric_names)
    summary.update(
        {
            "task": "ADOS_TOTAL regression",
            "selection": "minimum validation MAE",
            "test_policy": "test evaluated once after checkpoint selection",
            "split_strategy": manifest["strategy"],
            "split_manifest": "site_ados_aware_splits.json",
        }
    )
    with open(exp_dir / "cv_results.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n========== ADOS Regression CV Result ==========")
    for metric in metric_names:
        values = [record[metric] for record in fold_results]
        print(f"{metric}: {values}")
        print(
            f"  mean +/- std = {summary['mean'][metric]:.4f} +/- "
            f"{summary['std'][metric]:.4f}"
        )
    print("Saved:", exp_dir / "cv_results.json")


if __name__ == "__main__":
    main()
