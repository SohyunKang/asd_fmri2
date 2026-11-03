import os
import json
import datetime

import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import f1_score, roc_auc_score

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
        cls_loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.05)
        # cls_loss = F.cross_entropy(logits, y)

        alpha = _aux["seed_weights"].to(device)  # (M,)
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum()
        # loss = cls_loss - 0.0001 * entropy
        loss = cls_loss
        # print("CE:", cls_loss.item(), "REG:", entropy.item(), "lam*reg:", (0.01*entropy).item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # grad = model.periphery_seedness.u_per.grad

        # with torch.no_grad():

        #     seed_indices = _aux["seed_indices"].detach().long()

        #     selected = torch.zeros_like(
        #         grad,
        #         dtype=torch.bool
        #     )
        #     selected[seed_indices] = True

        #     # selected seed의 gradient만 약화
        #     grad[selected] *= 0.01

        # if batch_idx == 0:

        #     u = model.periphery_seedness.u_per.detach()
        #     g = model.periphery_seedness.u_per.grad.detach()

        #     top_idx = torch.topk(
        #         u,
        #         k=15,
        #         largest=True
        #     ).indices

        #     print("\n[SEED RANK DEBUG]")

        #     for rank, idx in enumerate(top_idx, start=1):

        #         i = idx.item()

        #         print(
        #             f"{rank:2d}. ROI={i:3d} | "
        #             f"score={u[i].item():+.6f} | "
        #             f"grad={g[i].item():+.6e}"
        #         )


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
    split_name: str = "VAL",
    collect_analysis: bool = False,
    verbose: bool = True,
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

        logits, aux = model(fc, debug=(verbose and batch_idx == 0))

        # if verbose and batch_idx == 0:

        #     u = model.periphery_seedness.u_per.detach()
        #     grad = model.periphery_seedness.u_per.grad

        #     top_idx = torch.topk(
        #         u,
        #         k=15,
        #         largest=True
        #     ).indices

        #     print("\n[SEED RANK DEBUG]")

        #     for rank, idx in enumerate(top_idx, start=1):

        #         i = idx.item()

        #         print(
        #             f"{rank:2d}. ROI={i:3d} | "
        #             f"score={u[i].item():+.6f} | "
        #             f"grad={(grad[i].item() if grad is not None else float('nan')):+.6e}"
        #         )
        
        if verbose and len(all_y) == 0:

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

        # ASD = label 0
        p_asd = probs[:, 0]
        pred = torch.argmax(logits, dim=1)

        all_y.append(y.detach().cpu().numpy())
        all_p1.append(p_asd.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_sub_id.extend(subjects)
    
        if collect_analysis:
            g_stack = torch.stack(aux["g_seed_list"], dim=0).permute(1, 0, 2, 3)
            p_stack = torch.stack(aux["p_list"], dim=0).permute(1, 0, 2, 3)
            H_stack = torch.stack(aux["H_list"], dim=0).permute(1, 0, 2, 3, 4)
            all_g.append(g_stack.cpu())
            all_p.append(p_stack.cpu())
            all_H.append(H_stack.cpu())

    y_true = np.concatenate(all_y, axis=0)
    y_score = np.concatenate(all_p1, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    # ASD(label=0)를 positive class=1로 변환
    y_true_asd = (y_true == 0).astype(int)

    try:
        auc = float(
            roc_auc_score(
                y_true_asd,
                y_score
            )
        )
    except ValueError:
        auc = float("nan")

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

    if verbose:
        print(
            f"[{split_name} P(ASD)] "
            f"ASD: {mean0:.4f} ± {std0:.4f} | "
            f"TD: {mean1:.4f} ± {std1:.4f} | "
            f"min={y_score.min():.4f} | "
            f"max={y_score.max():.4f} | "
            f"pred_ASD_ratio@0.5={pred_pos_ratio:.4f}"
        )

   # ASD = 0을 positive class로 정의
    tp = float(np.sum((y_true == 0) & (y_pred == 0)))  # ASD -> ASD
    fn = float(np.sum((y_true == 0) & (y_pred == 1)))  # ASD -> TD
    tn = float(np.sum((y_true == 1) & (y_pred == 1)))  # TD  -> TD
    fp = float(np.sum((y_true == 1) & (y_pred == 0)))  # TD  -> ASD

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "auc": auc,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),

        # -------- analysis tensors --------
        "g": torch.cat(all_g, dim=0) if collect_analysis else None,
        "p": torch.cat(all_p, dim=0) if collect_analysis else None,
        "H": torch.cat(all_H, dim=0) if collect_analysis else None,

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


def _largest_remainder_sizes(n_samples, proportions):
    """Convert split proportions into integer sizes whose sum is n_samples."""
    proportions = np.asarray(proportions, dtype=float)

    if proportions.ndim != 1 or len(proportions) < 2:
        raise ValueError("proportions must contain at least two split ratios")
    if np.any(proportions <= 0):
        raise ValueError("all split proportions must be positive")

    proportions = proportions / proportions.sum()
    raw_sizes = proportions * n_samples
    sizes = np.floor(raw_sizes).astype(int)

    remainder = int(n_samples - sizes.sum())
    if remainder > 0:
        order = np.argsort(-(raw_sizes - sizes), kind="stable")
        sizes[order[:remainder]] += 1

    return sizes


def _make_site_aware_targets(labels, sites):
    """
    Build multilabel targets for iterative stratification.

    Each subject is represented by three categorical targets:
      1) diagnosis
      2) acquisition site
      3) diagnosis x acquisition-site combination

    Balancing the joint target is useful for ABIDE, while the separate diagnosis
    and site targets keep the marginal distributions stable when a joint group
    contains fewer subjects than the number of folds.
    """
    labels = np.asarray(labels)
    sites = np.asarray(sites, dtype=str)

    tokens_per_subject = []
    all_tokens = set()

    for label, site in zip(labels, sites):
        tokens = (
            f"diagnosis={label}",
            f"site={site}",
            f"diagnosis={label}|site={site}",
        )
        tokens_per_subject.append(tokens)
        all_tokens.update(tokens)

    token_to_col = {
        token: col
        for col, token in enumerate(sorted(all_tokens))
    }

    targets = np.zeros(
        (len(labels), len(token_to_col)),
        dtype=np.int8,
    )

    for row, tokens in enumerate(tokens_per_subject):
        for token in tokens:
            targets[row, token_to_col[token]] = 1

    return targets


def iterative_site_aware_partition(
    indices,
    labels,
    sites,
    proportions,
    random_state,
):
    """
    Partition indices while approximately preserving diagnosis, site, and
    diagnosis-by-site distributions.

    Rare diagnosis-by-site groups are handled directly instead of being passed
    to sklearn's composite-label stratifier, which can fail when a group has
    only one subject.
    """
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    sites = np.asarray(sites, dtype=str)

    if len(indices) == 0:
        raise ValueError("cannot split an empty index array")

    rng = np.random.default_rng(random_state)
    local_targets = _make_site_aware_targets(
        labels[indices],
        sites[indices],
    )

    target_sizes = _largest_remainder_sizes(
        len(indices),
        proportions,
    )
    n_parts = len(target_sizes)

    # Exact integer split sizes are used for capacity, while label targets can
    # remain fractional because perfect balancing is not always possible.
    split_weights = target_sizes / target_sizes.sum()
    desired_label_counts = np.outer(
        split_weights,
        local_targets.sum(axis=0),
    ).astype(float)
    remaining_capacity = target_sizes.copy()

    assignment = np.full(len(indices), -1, dtype=np.int64)
    unassigned = np.ones(len(indices), dtype=bool)

    while np.any(unassigned):
        remaining_label_counts = local_targets[unassigned].sum(axis=0)
        positive_cols = np.flatnonzero(remaining_label_counts > 0)

        if len(positive_cols) == 0:
            candidates = np.flatnonzero(unassigned)
        else:
            # Distribute the rarest remaining category first. This is the core
            # idea of iterative multilabel stratification.
            rare_count = remaining_label_counts[positive_cols].min()
            rare_cols = positive_cols[
                remaining_label_counts[positive_cols] == rare_count
            ]
            selected_col = int(rng.choice(rare_cols))
            candidates = np.flatnonzero(
                unassigned & (local_targets[:, selected_col] == 1)
            )

        rng.shuffle(candidates)

        for sample_idx in candidates:
            if not unassigned[sample_idx]:
                continue

            eligible = np.flatnonzero(remaining_capacity > 0)
            if len(eligible) == 0:
                raise RuntimeError("site-aware split exhausted all capacities")

            sample_cols = np.flatnonzero(local_targets[sample_idx] == 1)

            if len(positive_cols) > 0:
                primary_need = desired_label_counts[eligible, selected_col]
                best = eligible[np.isclose(primary_need, primary_need.max())]
            else:
                best = eligible

            if len(best) > 1:
                # Tie-break using all targets carried by the current subject.
                joint_need = desired_label_counts[
                    np.ix_(best, sample_cols)
                ].sum(axis=1)
                best = best[np.isclose(joint_need, joint_need.max())]

            if len(best) > 1:
                max_capacity = remaining_capacity[best].max()
                best = best[remaining_capacity[best] == max_capacity]

            chosen_part = int(rng.choice(best))
            assignment[sample_idx] = chosen_part
            unassigned[sample_idx] = False
            remaining_capacity[chosen_part] -= 1
            desired_label_counts[chosen_part] -= local_targets[sample_idx]

    parts = [
        np.sort(indices[assignment == part_idx])
        for part_idx in range(n_parts)
    ]

    if [len(part) for part in parts] != target_sizes.tolist():
        raise RuntimeError("site-aware split produced unexpected split sizes")

    return parts


def build_site_aware_cv_splits(
    labels,
    sites,
    n_splits,
    inner_val_size,
    random_seed,
):
    """Create reproducible outer-test and inner-validation site-aware splits."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not 0.0 < inner_val_size < 1.0:
        raise ValueError("inner_val_size must be between 0 and 1")

    all_indices = np.arange(len(labels), dtype=np.int64)
    outer_test_parts = iterative_site_aware_partition(
        indices=all_indices,
        labels=labels,
        sites=sites,
        proportions=np.ones(n_splits, dtype=float),
        random_state=random_seed,
    )

    split_records = []

    for fold, test_idx in enumerate(outer_test_parts, start=1):
        outer_train_idx = np.setdiff1d(
            all_indices,
            test_idx,
            assume_unique=True,
        )

        train_idx, val_idx = iterative_site_aware_partition(
            indices=outer_train_idx,
            labels=labels,
            sites=sites,
            proportions=[1.0 - inner_val_size, inner_val_size],
            random_state=random_seed + fold,
        )

        # Leakage and class-presence checks.
        split_sets = [set(train_idx), set(val_idx), set(test_idx)]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError(f"Fold {fold}: train/validation/test overlap detected")
        if set.union(*split_sets) != set(all_indices):
            raise RuntimeError(f"Fold {fold}: split does not cover the full dataset")

        for split_name, split_idx in [
            ("train", train_idx),
            ("validation", val_idx),
            ("test", test_idx),
        ]:
            if len(np.unique(labels[split_idx])) < 2:
                raise RuntimeError(
                    f"Fold {fold}: {split_name} does not contain both classes"
                )

        split_records.append({
            "fold": fold,
            "outer_train_idx": outer_train_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        })

    return split_records


def split_distribution_record(indices, labels, sites):
    """JSON-serializable distribution summary for split auditing."""
    indices = np.asarray(indices, dtype=np.int64)
    part = pd.DataFrame({
        "label": labels[indices],
        "site": sites[indices],
    })

    site_counts = part["site"].value_counts().sort_index()
    label_counts = part["label"].value_counts().sort_index()
    joint_counts = part.groupby(["site", "label"]).size()

    return {
        "n": int(len(indices)),
        "label_counts": {
            str(key): int(value)
            for key, value in label_counts.items()
        },
        "site_counts": {
            str(key): int(value)
            for key, value in site_counts.items()
        },
        "site_label_counts": {
            f"{site}|{label}": int(value)
            for (site, label), value in joint_counts.items()
        },
    }


def plot_site_distribution(train_df, val_df, test_df, site_col, fold, save_dir):

    train_counts = train_df[site_col].value_counts()
    val_counts = val_df[site_col].value_counts()
    test_counts = test_df[site_col].value_counts()

    # 공통 site 정렬
    all_sites = sorted(
        set(train_counts.index)
        .union(set(val_counts.index))
        .union(set(test_counts.index))
    )

    train_counts = train_counts.reindex(all_sites, fill_value=0)
    val_counts = val_counts.reindex(all_sites, fill_value=0)
    test_counts = test_counts.reindex(all_sites, fill_value=0)

    # 🔥 percentage로 변환
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100

    x = np.arange(len(all_sites))
    width = 0.26

    plt.figure(figsize=(10, 4))

    bars1 = plt.bar(x - width, train_pct, width, label="Train")
    bars2 = plt.bar(x, val_pct, width, label="Validation")
    bars3 = plt.bar(x + width, test_pct, width, label="Test")

    plt.xticks(x, all_sites, rotation=45)
    plt.ylabel("Percentage (%)")
    plt.title(f"Fold {fold} - Site Distribution (%)")
    plt.legend()

    # 🔥 숫자 표시까지 (이게 핵심)
    for bars in [bars1, bars2, bars3]:
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

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_hops", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--top_m", type=int, default=5)
    parser.add_argument("--seed_std", type=float, default=0.5)

    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--act", type=str, default="gelu", choices=["gelu", "relu"])

    parser.add_argument("--fc_norm", type=str, default="row", choices=["row", "sym", "none"])

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--inner_val_size", type=float, default=0.15)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--fold", type=int, default=None, help="Run only a specific fold (1-based index)")

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--worker_id", type=int, default=0)
    
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()

    set_seed(args.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # for log save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_dir = f"/data5/sohyunkang/experiments2/w{args.worker_id}_hitranet_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    print("Experiment dir:", exp_dir)

    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    log_path = os.path.join(exp_dir, "training_history.csv")

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
            "val_f1",
            "test_loss",
            "test_acc",
            "test_auc",
            "test_sens",
            "test_spec",
            "test_f1",
            "is_best",
            "stale_epochs",
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

    if "SITE_ID" not in df.columns:
        raise KeyError(
            "SITE_ID is required for site-aware splitting, but the dataset "
            "did not return it. Check FCDataset and the input spreadsheet."
        )

    # Missing/empty site names are kept as an explicit category so that every
    # subject participates in the same deterministic split procedure.
    sites = (
        df["SITE_ID"]
        .fillna("UNKNOWN")
        .astype(str)
        .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
        .to_numpy()
    )
    df["SITE_ID"] = sites

    site_aware_splits = build_site_aware_cv_splits(
        labels=labels,
        sites=sites,
        n_splits=args.n_splits,
        inner_val_size=args.inner_val_size,
        random_seed=args.random_seed,
    )

    # Persist exact indices and distributions. Reuse this manifest for every
    # baseline to make model comparisons independent of split differences.
    split_manifest = {
        "strategy": "iterative_multilabel_diagnosis_site_and_joint",
        "description": (
            "Outer test folds and inner validation sets approximately preserve "
            "diagnosis, site, and diagnosis-by-site distributions."
        ),
        "random_seed": int(args.random_seed),
        "n_splits": int(args.n_splits),
        "inner_val_size": float(args.inner_val_size),
        "folds": [],
    }

    for split_record in site_aware_splits:
        split_manifest["folds"].append({
            "fold": int(split_record["fold"]),
            "outer_train_indices": split_record["outer_train_idx"].tolist(),
            "train_indices": split_record["train_idx"].tolist(),
            "validation_indices": split_record["val_idx"].tolist(),
            "test_indices": split_record["test_idx"].tolist(),
            "train_distribution": split_distribution_record(
                split_record["train_idx"], labels, sites
            ),
            "validation_distribution": split_distribution_record(
                split_record["val_idx"], labels, sites
            ),
            "test_distribution": split_distribution_record(
                split_record["test_idx"], labels, sites
            ),
        })

    split_manifest_path = os.path.join(exp_dir, "site_aware_splits.json")
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=4)

    print("Site-aware split strategy:", split_manifest["strategy"])
    print("Saved split manifest:", split_manifest_path)

    aal_roi_names = load_roi_names_from_txt(args.aal_txt)
    assert len(aal_roi_names) == args.n_rois, f"ROI names ({len(aal_roi_names)}) != n_rois ({args.n_rois})"

    fold_best_aucs: List[float] = []
    fold_best_accs: List[float] = []
    fold_best_sens: List[float] = []
    fold_best_specs: List[float] = []
    fold_best_f1s: List[float] = []

    for split_record in site_aware_splits:
        fold = split_record["fold"]
        outer_train_idx = split_record["outer_train_idx"]
        train_idx = split_record["train_idx"]
        val_idx = split_record["val_idx"]
        test_idx = split_record["test_idx"]

        # 특정 fold만 실행
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n========== Fold {fold}/{args.n_splits} ==========")
        fold_seed = args.random_seed
        set_seed(fold_seed)

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
        test_set = Subset(dataset, test_idx.tolist())

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
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        print("\n--- Data distribution check ---")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

        # -------------------------
        # class
        # -------------------------
        print("Train class:", train_df["label"].value_counts().to_dict())
        print("Val   class:", val_df["label"].value_counts().to_dict())
        print("Test  class:", test_df["label"].value_counts().to_dict())

        # -------------------------
        # demographics + motion
        # -------------------------
        for col in ["AGE_AT_SCAN", "SEX", "func_mean_fd"]:
            if col in train_df.columns:

                print(f"\n[{col}]")

                tr_mean, tr_std = train_df[col].mean(), train_df[col].std()
                va_mean, va_std = val_df[col].mean(), val_df[col].std()
                te_mean, te_std = test_df[col].mean(), test_df[col].std()

                print(f"Train mean/std: {tr_mean:.4f} / {tr_std:.4f}")
                print(f"Val   mean/std: {va_mean:.4f} / {va_std:.4f}")
                print(f"Test  mean/std: {te_mean:.4f} / {te_std:.4f}")

                print(f"Diff (val-train): {va_mean - tr_mean:.4f}")
                print(f"Diff (test-train): {te_mean - tr_mean:.4f}")
         
        print("Train site:", train_df['SITE_ID'].value_counts().to_dict())
        print("Val   site :", val_df['SITE_ID'].value_counts().to_dict())
        print("Test  site :", test_df['SITE_ID'].value_counts().to_dict())
        
        plot_site_distribution(
            train_df,
            val_df,
            test_df,
            "SITE_ID",
            fold,
            exp_dir,
        )

        train_fc = fc_stats(dataset, train_idx)
        val_fc = fc_stats(dataset, val_idx)
        test_fc = fc_stats(dataset, test_idx)

        print("Train FC:", train_fc)
        print("Val   FC:", val_fc)
        print("Test  FC:", test_fc)

        print("FC diff (val-train):",
            {k: val_fc[k] - train_fc[k] for k in train_fc})
        print("FC diff (test-train):",
            {k: test_fc[k] - train_fc[k] for k in train_fc})

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
            "f1": 0.0,
        }
        best_path = os.path.join(exp_dir, f"fold{fold}_best_model.pt")
        best_state = None
        stale_epochs = 0
        
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
            # Test metrics are monitored every epoch, but they are never used
            # for checkpoint selection or early stopping.
            te = eval_one_epoch(
                model,
                test_loader,
                device,
                fc_norm=args.fc_norm,
                split_name="TEST",
                collect_analysis=False,
                verbose=False,
            )

            auc_str = f"{va['auc']:.4f}" if np.isfinite(va["auc"]) else "NA"
            test_auc_str = f"{te['auc']:.4f}" if np.isfinite(te["auc"]) else "NA"

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


            is_best = np.isfinite(va["auc"]) and va["auc"] > best_val_auc and epoch >= 5
            if is_best:
                best_val_auc = va["auc"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
                best_metrics["epoch"] = epoch
                best_metrics["acc"] = va["acc"]
                best_metrics["sensitivity"] = va["sensitivity"]
                best_metrics["specificity"] = va["specificity"]
                best_metrics["f1"] = va["f1"]

                
                if args.save_model:
                    torch.save(
                        {
                            "model": best_state,
                            "args": vars(args),
                            "fold": fold,
                            "split_strategy": split_manifest["strategy"],
                            "best_metrics": best_metrics,
                            "outer_train_indices": outer_train_idx,
                            "train_indices": train_idx,
                            "validation_indices": val_idx,
                            "test_indices": test_idx,
                        },
                        best_path,
                    )
                    
                seed_idx, seed_w = model.topm_seeds()
                seed_idx_cpu = seed_idx.detach().cpu()
                seed_w_cpu = seed_w.detach().cpu()

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

                    "f1": best_metrics["f1"],
                    "threshold": 0.5,

                    "epoch": best_metrics["epoch"],

                    "seed_indices": sorted_seed_idx.tolist(),
                    "seed_weights": sorted_seed_w.tolist(),
                }

                with open(os.path.join(exp_dir, f"fold{fold}_metrics.json"), "w") as f:
                    json.dump(fold_result, f, indent=4)
            else:
                stale_epochs += 1

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    fold, epoch, tr["loss"], tr["acc"],
                    va["loss"], va["acc"], va["auc"],
                    va["sensitivity"], va["specificity"], va["f1"],
                    te["loss"], te["acc"], te["auc"],
                    te["sensitivity"], te["specificity"], te["f1"],
                    int(is_best), stale_epochs,
                ])
            print(
                f"[Fold {fold} | Epoch {epoch:03d}/{args.epochs}] "
                f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
                f"val_loss={va['loss']:.4f} val_auc={auc_str} "
                f"val_acc={va['acc']:.4f} val_sens={va['sensitivity']:.4f} "
                f"val_spec={va['specificity']:.4f} val_f1={va['f1']:.4f} "
                f"| test_loss={te['loss']:.4f} test_auc={test_auc_str} "
                f"test_acc={te['acc']:.4f} test_sens={te['sensitivity']:.4f} "
                f"test_spec={te['specificity']:.4f} test_f1={te['f1']:.4f} "
                f"best={is_best} stale={stale_epochs}/{args.patience} "
                f"time={time.time()-start:.2f}s"
            )

            if stale_epochs >= args.patience:
                print(
                    f"[Fold {fold}] Early stopping at epoch {epoch}; "
                    f"best epoch={best_metrics['epoch']}, "
                    f"best val AUC={best_val_auc:.4f}"
                )
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold}: no valid checkpoint was produced")

        model.load_state_dict(best_state)
        train_metrics = eval_one_epoch(
            model, train_loader, device, fc_norm=args.fc_norm,
            split_name="TRAIN", collect_analysis=True,
        )
        val_metrics = eval_one_epoch(
            model, val_loader, device, fc_norm=args.fc_norm,
            split_name="VAL", collect_analysis=True,
        )
        test_metrics = eval_one_epoch(
            model, test_loader, device, fc_norm=args.fc_norm,
            split_name="TEST", collect_analysis=True, verbose=True,
        )

        seed_idx, seed_w = model.topm_seeds()
        order = torch.argsort(seed_w.detach().cpu(), descending=True)
        sorted_seed_idx = seed_idx.detach().cpu()[order]
        sorted_seed_w = seed_w.detach().cpu()[order]

        test_aux_path = os.path.join(exp_dir, f"fold{fold}_best_aux.pt")
        torch.save(
            {
                "best_epoch": best_metrics["epoch"],
                "selection_val_auc": best_val_auc,
                "train_subject": train_metrics["subject"],
                "train_y": train_metrics["y"],
                "train_g": train_metrics["g"],
                "train_p": train_metrics["p"],
                "train_H": train_metrics["H"],
                "val_subject": val_metrics["subject"],
                "val_y": val_metrics["y"],
                "val_g": val_metrics["g"],
                "val_p": val_metrics["p"],
                "val_H": val_metrics["H"],
                "val_pred": val_metrics["y_pred"],
                "val_score": val_metrics["y_score"],
                "test_subject": test_metrics["subject"],
                "test_y": test_metrics["y"],
                "test_g": test_metrics["g"],
                "test_p": test_metrics["p"],
                "test_H": test_metrics["H"],
                "test_pred": test_metrics["y_pred"],
                "test_score": test_metrics["y_score"],
                "seed_indices": sorted_seed_idx,
                "seed_weights": sorted_seed_w,
                "test_indices": test_idx,
            },
            test_aux_path,
        )

        fold_result = {
            "fold": fold,
            "split_strategy": split_manifest["strategy"],
            "best_epoch": int(best_metrics["epoch"]),
            "selection_val_auc": float(best_val_auc),
            "auc": float(test_metrics["auc"]),
            "acc": float(test_metrics["acc"]),
            "sensitivity": float(test_metrics["sensitivity"]),
            "specificity": float(test_metrics["specificity"]),
            "f1": float(test_metrics["f1"]),
            "threshold": 0.5,
            "seed_indices": sorted_seed_idx.tolist(),
            "seed_weights": sorted_seed_w.tolist(),
        }
        if args.save_model:
            torch.save(
                {
                    "model": best_state,
                    "args": vars(args),
                    "fold": fold,
                    "split_strategy": split_manifest["strategy"],
                    "best_epoch": int(best_metrics["epoch"]),
                    "selection_val_auc": float(best_val_auc),
                    "test_metrics": {
                        key: fold_result[key]
                        for key in [
                            "auc", "acc", "sensitivity", "specificity", "f1", "threshold"
                        ]
                    },
                    "outer_train_indices": outer_train_idx,
                    "train_indices": train_idx,
                    "validation_indices": val_idx,
                    "test_indices": test_idx,
                },
                best_path,
            )
        with open(os.path.join(exp_dir, f"fold{fold}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=4)

        fold_best_aucs.append(test_metrics["auc"])
        fold_best_accs.append(test_metrics["acc"])
        fold_best_sens.append(test_metrics["sensitivity"])
        fold_best_specs.append(test_metrics["specificity"])
        fold_best_f1s.append(test_metrics["f1"])
        
        print(
            f"Fold {fold} TEST → "
            f"AUC={test_metrics['auc']:.4f}, "
            f"ACC={test_metrics['acc']:.4f}, "
            f"SENS={test_metrics['sensitivity']:.4f}, "
            f"SPEC={test_metrics['specificity']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, THR=0.5, "
            f"EPOCH={best_metrics['epoch']}"
        )

    if args.fold is None:
        print("\n========== 5-Fold CV Result ==========")
        print("Test AUC per fold:", [f"{a:.4f}" for a in fold_best_aucs])
        print("Test ACC per fold:", [f"{a:.4f}" for a in fold_best_accs])
        print("Test SENS per fold:", [f"{a:.4f}" for a in fold_best_sens])
        print("Test SPEC per fold:", [f"{a:.4f}" for a in fold_best_specs])
        print("Test F1 per fold:", [f"{a:.4f}" for a in fold_best_f1s])

        print(f"AUC  Mean ± Std: {np.mean(fold_best_aucs):.4f} ± {np.std(fold_best_aucs):.4f}")
        print(f"ACC  Mean ± Std: {np.mean(fold_best_accs):.4f} ± {np.std(fold_best_accs):.4f}")
        print(f"SENS Mean ± Std: {np.mean(fold_best_sens):.4f} ± {np.std(fold_best_sens):.4f}")
        print(f"SPEC Mean ± Std: {np.mean(fold_best_specs):.4f} ± {np.std(fold_best_specs):.4f}")
        print(f"F1   Mean ± Std: {np.mean(fold_best_f1s):.4f} ± {np.std(fold_best_f1s):.4f}")

        # =========================
        # Fold별 결과
        # =========================
        fold_results = []

        for i, (auc, acc, sens, spec, f1) in enumerate(zip(
            fold_best_aucs,
            fold_best_accs,
            fold_best_sens,
            fold_best_specs,
            fold_best_f1s,
        )):
            fold_results.append({
                "fold": i + 1,
                "auc": float(auc),
                "acc": float(acc),
                "sens": float(sens),
                "spec": float(spec),
                "f1": float(f1),
            })

        # =========================
        # 전체 CV 결과
        # =========================
        cv_results = {
            "split_strategy": split_manifest["strategy"],
            "split_manifest": "site_aware_splits.json",
            "folds": fold_results,

            "mean": {
                "auc": float(np.mean(fold_best_aucs)),
                "acc": float(np.mean(fold_best_accs)),
                "sens": float(np.mean(fold_best_sens)),
                "spec": float(np.mean(fold_best_specs)),
                "f1": float(np.mean(fold_best_f1s)),
            },

            "std": {
                "auc": float(np.std(fold_best_aucs)),
                "acc": float(np.std(fold_best_accs)),
                "sens": float(np.std(fold_best_sens)),
                "spec": float(np.std(fold_best_specs)),
                "f1": float(np.std(fold_best_f1s)),
            }
        }

        with open(os.path.join(exp_dir, "cv_results.json"), "w") as f:
            json.dump(cv_results, f, indent=4)

        print(f"Saved CV results: {os.path.join(exp_dir, 'cv_results.json')}")


if __name__ == "__main__":
    main()
