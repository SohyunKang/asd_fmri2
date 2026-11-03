"""Single-file EAG-RS baseline adapted to AAL-116 FC matrices.

Based on the official implementation of Jung et al., IEEE TMI 2023:
https://github.com/ku-milab/EAG-RS (commit b3da8f1).

"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_args():
    p = argparse.ArgumentParser(description="EAG-RS AAL-116 single-file adaptation")
    p.add_argument("--xlsx-path", default="data/abide_subjects_fc_with_demo_260202.xlsx")
    p.add_argument("--label-col", default="DX_GROUP")
    p.add_argument("--save-dir", default="./baseline_experiments/eag_rs_results")
    p.add_argument("--n-rois", type=int, default=116)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--inner-val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--relation-epochs", type=int, default=100)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--classifier-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--remove-roi", type=float, default=0.10)
    p.add_argument("--hidden1-ratio", type=float, default=1.5)
    p.add_argument("--latent-ratio", type=float, default=0.3)
    p.add_argument("--classifier-hidden", type=int, default=10)
    p.add_argument("--gumbel-tau", type=float, default=0.1)
    p.add_argument("--gate-ratio-weight", type=float, default=1.0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(x, y, idx, batch, shuffle, workers):
    ds = TensorDataset(torch.as_tensor(x[idx], dtype=torch.float32),
                       torch.as_tensor(y[idx], dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


class RelationEncoder(nn.Module):
    """Two-level denoising relation learner following official EAG-RS sizes."""
    def __init__(self, n_edges, hidden1_ratio, latent_ratio):
        super().__init__()
        h1 = int(math.ceil(n_edges * hidden1_ratio))
        z = int(math.ceil(n_edges * latent_ratio))
        self.edge_encoder = nn.Sequential(nn.Linear(n_edges, h1), nn.SELU())
        self.edge_decoder = nn.Sequential(nn.Linear(h1, n_edges), nn.Tanh())
        self.relation_encoder = nn.Sequential(nn.Linear(h1, z), nn.Tanh())
        self.relation_decoder = nn.Sequential(nn.Linear(z, h1), nn.Tanh())
        self.n_edges, self.hidden1, self.latent_dim = n_edges, h1, z

    def level1(self, edge):
        h1 = self.edge_encoder(edge)
        return self.edge_decoder(h1), h1

    def level2(self, edge):
        h1 = self.edge_encoder(edge)
        z = self.relation_encoder(h1)
        h1_hat = self.relation_decoder(z)
        edge_hat = self.edge_decoder(h1_hat)
        return edge_hat, h1_hat, h1, z


class Probe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 2)
    def forward(self, z):
        return self.fc(z)


class ROISelector(nn.Module):
    """Subject-adaptive relevance-guided Gumbel ROI selector."""
    def __init__(self, n_rois):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_rois, n_rois), nn.ReLU(), nn.Linear(n_rois, n_rois)
        )

    def forward(self, roi_sum, roi_count, tau, training):
        logits = self.net(torch.cat((roi_sum, roi_count), dim=1))
        if training:
            keep = F.gumbel_softmax(
                torch.stack((torch.zeros_like(logits), logits), dim=-1),
                tau=tau, hard=False, dim=-1,
            )[..., 1]
        else:
            keep = torch.sigmoid(logits)
        return keep, logits


class FinalClassifier(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.5), nn.Linear(hidden, 2),
        )
    def forward(self, z):
        return self.net(z)


def random_roi_mask(fc, remove_fraction):
    batch, n, _ = fc.shape
    keep = torch.ones(batch, n, device=fc.device, dtype=fc.dtype)
    count = max(1, int(round(n * remove_fraction)))
    for i in range(batch):
        selected = torch.randperm(n, device=fc.device)[:count]
        keep[i, selected] = 0
    return fc * keep[:, :, None] * keep[:, None, :]


def apply_roi_keep(fc, keep):
    return fc * keep[:, :, None] * keep[:, None, :]


def edge_vector(fc, triu):
    return fc[:, triu[0], triu[1]]


def relevance_to_roi(relevance, triu, n_rois):
    """Convert per-edge relevance into the two ROI statistics used by EAG-RS."""
    batch = relevance.size(0)
    matrix = relevance.new_zeros(batch, n_rois, n_rois)
    matrix[:, triu[0], triu[1]] = relevance.abs()
    matrix = matrix + matrix.transpose(1, 2)
    threshold = matrix.mean(dim=2, keepdim=True)
    high = (matrix >= threshold).to(matrix.dtype)
    roi_sum = (matrix * high).sum(dim=2)
    roi_count = high.sum(dim=2)
    roi_sum = roi_sum / roi_sum.amax(dim=1, keepdim=True).clamp_min(1e-8)
    roi_count = roi_count / float(n_rois)
    return roi_sum, roi_count, matrix


def basic_metrics(y, p_asd):
    pred = np.where(p_asd >= 0.5, 0, 1)
    return {
        "auc": float(roc_auc_score((y == 0).astype(int), p_asd)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "n_pred_asd": int((pred == 0).sum()),
        "n_pred_td": int((pred == 1).sum()),
        "p_asd_mean": float(p_asd.mean()),
    }


def train_reconstruction(model, loader, optimizer, device, triu, args, level):
    model.train()
    total = count = 0
    for fc, _ in loader:
        fc = fc.to(device)
        masked = random_roi_mask(fc, args.remove_roi)
        source, target = edge_vector(masked, triu), edge_vector(fc, triu)
        if level == 1:
            predicted, _ = model.level1(source)
            loss = F.mse_loss(predicted, target)
        else:
            predicted, h1_hat, h1, _ = model.level2(source)
            loss = 0.5 * (F.mse_loss(predicted, target) + F.mse_loss(h1_hat, h1.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * fc.size(0); count += fc.size(0)
    return total / count


@torch.no_grad()
def eval_reconstruction(model, loader, device, triu, args, level):
    model.eval(); total = count = 0
    for fc, _ in loader:
        fc = fc.to(device)
        source = edge_vector(fc, triu); target = source
        if level == 1:
            predicted, _ = model.level1(source); loss = F.mse_loss(predicted, target)
        else:
            predicted, h1_hat, h1, _ = model.level2(source)
            loss = 0.5 * (F.mse_loss(predicted, target) + F.mse_loss(h1_hat, h1))
        total += loss.item() * fc.size(0); count += fc.size(0)
    return total / count


def train_probe(model, probe, loader, optimizer, device, triu):
    model.eval(); probe.train(); total = count = 0
    for fc, y in loader:
        fc, y = fc.to(device), y.to(device)
        with torch.no_grad(): _, _, _, z = model.level2(edge_vector(fc, triu))
        loss = F.cross_entropy(probe(z), y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += loss.item() * fc.size(0); count += fc.size(0)
    return total / count


@torch.no_grad()
def eval_probe(model, probe, loader, device, triu):
    model.eval(); probe.eval(); yy, pp = [], []
    for fc, y in loader:
        fc = fc.to(device); _, _, _, z = model.level2(edge_vector(fc, triu))
        yy.append(y.numpy()); pp.append(probe(z).softmax(1)[:, 0].cpu().numpy())
    y, p = np.concatenate(yy), np.concatenate(pp)
    return basic_metrics(y, p)


def compute_relevance(model, probe, fc, triu):
    """Class-conditional input-times-gradient relevance for each FC edge."""
    edge = edge_vector(fc, triu).detach().requires_grad_(True)
    _, _, _, z = model.level2(edge)
    logits = probe(z)
    predicted = logits.argmax(1)
    score = logits.gather(1, predicted[:, None]).sum()
    gradient = torch.autograd.grad(score, edge, create_graph=False)[0]
    return (edge * gradient).detach()


def final_forward(model, probe, selector, classifier, fc, triu, args, training):
    relevance = compute_relevance(model, probe, fc, triu)
    roi_sum, roi_count, relevance_matrix = relevance_to_roi(relevance, triu, args.n_rois)
    keep, gate_logits = selector(roi_sum, roi_count, args.gumbel_tau, training)
    masked_fc = apply_roi_keep(fc, keep)
    _, _, _, z = model.level2(edge_vector(masked_fc, triu))
    return classifier(z), keep, gate_logits, relevance_matrix


def train_final(model, probe, selector, classifier, loader, optimizer, device, triu, args):
    model.train(); selector.train(); classifier.train(); probe.eval()
    total = count = 0
    target_keep = 1.0 - args.remove_roi
    for fc, y in loader:
        fc, y = fc.to(device), y.to(device)
        logits, keep, _, _ = final_forward(model, probe, selector, classifier, fc, triu, args, True)
        ratio_loss = (keep.mean() - target_keep).pow(2)
        loss = F.cross_entropy(logits, y) + args.gate_ratio_weight * ratio_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += loss.item() * fc.size(0); count += fc.size(0)
    return total / count


def evaluate_final(model, probe, selector, classifier, loader, device, triu, args, collect=False):
    model.eval(); probe.eval(); selector.eval(); classifier.eval()
    yy, pp, keeps, relevance_all = [], [], [], []
    loss_total = count = 0
    # relevance requires gradients, so evaluation intentionally does not use no_grad.
    for fc, y in loader:
        fc, y_device = fc.to(device), y.to(device)
        logits, keep, _, relevance = final_forward(
            model, probe, selector, classifier, fc, triu, args, False
        )
        loss = F.cross_entropy(logits, y_device)
        loss_total += loss.item() * fc.size(0); count += fc.size(0)
        yy.append(y.numpy()); pp.append(logits.softmax(1)[:, 0].detach().cpu().numpy())
        if collect:
            keeps.append(keep.detach().cpu()); relevance_all.append(relevance.detach().cpu())
    y, p = np.concatenate(yy), np.concatenate(pp)
    result = basic_metrics(y, p); result["loss"] = loss_total / count
    if collect:
        result.update(y_true=y, probability_asd=p, roi_keep=torch.cat(keeps),
                      edge_relevance=torch.cat(relevance_all))
    return result


def fit_min_loss_stage(name, fold, epochs, model, train_loader, val_loader,
                       optimizer, device, triu, args, level, history_path):
    best_loss, best_epoch, best_state = np.inf, -1, None
    for epoch in range(1, epochs + 1):
        tr = train_reconstruction(model, train_loader, optimizer, device, triu, args, level)
        va = eval_reconstruction(model, val_loader, device, triu, args, level)
        is_best = va < best_loss
        if is_best: best_loss, best_epoch, best_state = va, epoch, copy.deepcopy(model.state_dict())
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow([fold, name, epoch, tr, va, "", "", int(is_best)])
        print(f"[Fold {fold} | {name} {epoch:03d}/{epochs}] train={tr:.4f} val={va:.4f} best={is_best}")
    model.load_state_dict(best_state)
    return best_epoch, best_loss


def train_fold(fold, x, y, outer_train, test_idx, args, device, save_dir, triu):
    set_seed(args.seed + fold)
    train_idx, val_idx = train_test_split(
        outer_train, test_size=args.inner_val_size, stratify=y[outer_train],
        random_state=args.seed + fold,
    )
    tr = make_loader(x, y, train_idx, args.batch_size, True, args.num_workers)
    va = make_loader(x, y, val_idx, args.batch_size, False, args.num_workers)
    te = make_loader(x, y, test_idx, args.batch_size, False, args.num_workers)
    n_edges = args.n_rois * (args.n_rois - 1) // 2
    model = RelationEncoder(n_edges, args.hidden1_ratio, args.latent_ratio).to(device)
    history_path = save_dir / "training_history.csv"

    # Stage 1
    for p in model.relation_encoder.parameters(): p.requires_grad = False
    for p in model.relation_decoder.parameters(): p.requires_grad = False
    opt1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    s1_epoch, s1_loss = fit_min_loss_stage(
        "relation_level1", fold, args.pretrain_epochs, model, tr, va, opt1,
        device, triu, args, 1, history_path,
    )

    # Stage 2
    for p in model.parameters(): p.requires_grad = True
    for p in model.edge_encoder.parameters(): p.requires_grad = False
    for p in model.edge_decoder.parameters(): p.requires_grad = False
    opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    s2_epoch, s2_loss = fit_min_loss_stage(
        "relation_level2", fold, args.relation_epochs, model, tr, va, opt2,
        device, triu, args, 2, history_path,
    )

    # Relevance probe
    for p in model.parameters(): p.requires_grad = False
    probe = Probe(model.latent_dim).to(device)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    probe_best_auc, probe_state = -np.inf, None
    for epoch in range(1, args.probe_epochs + 1):
        train_loss = train_probe(model, probe, tr, probe_opt, device, triu)
        vm = eval_probe(model, probe, va, device, triu)
        is_best = np.isfinite(vm["auc"]) and vm["auc"] > probe_best_auc
        if is_best: probe_best_auc, probe_state = vm["auc"], copy.deepcopy(probe.state_dict())
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow([fold, "relevance_probe", epoch, train_loss, "", vm["auc"], vm["accuracy"], int(is_best)])
        print(f"[Fold {fold} | Probe {epoch:03d}/{args.probe_epochs}] loss={train_loss:.4f} "
              f"val_auc={vm['auc']:.4f} val_acc={vm['accuracy']:.4f} best={is_best}")
    probe.load_state_dict(probe_state)

    # Stage 3
    selector = ROISelector(args.n_rois).to(device)
    classifier = FinalClassifier(model.latent_dim, args.classifier_hidden).to(device)
    for p in model.parameters(): p.requires_grad = True
    for p in probe.parameters(): p.requires_grad = False
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.gate_lr},
        {"params": selector.parameters(), "lr": args.gate_lr},
        {"params": classifier.parameters(), "lr": args.lr},
    ], weight_decay=args.weight_decay)
    best_auc, best_epoch, best_pack, stale = -np.inf, -1, None, 0
    for epoch in range(1, args.classifier_epochs + 1):
        train_loss = train_final(model, probe, selector, classifier, tr, optimizer,
                                 device, triu, args)
        vm = evaluate_final(model, probe, selector, classifier, va, device, triu, args)
        is_best = np.isfinite(vm["auc"]) and vm["auc"] > best_auc
        if is_best:
            best_auc, best_epoch, stale = vm["auc"], epoch, 0
            best_pack = (copy.deepcopy(model.state_dict()), copy.deepcopy(selector.state_dict()),
                         copy.deepcopy(classifier.state_dict()), copy.deepcopy(vm))
        else: stale += 1
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow([fold, "final_classifier", epoch, train_loss,
                                    vm["loss"], vm["auc"], vm["accuracy"], int(is_best)])
        print(f"[Fold {fold} | Final {epoch:03d}/{args.classifier_epochs}] train={train_loss:.4f} "
              f"val_loss={vm['loss']:.4f} auc={vm['auc']:.4f} acc={vm['accuracy']:.4f} "
              f"sen={vm['sensitivity']:.4f} spe={vm['specificity']:.4f} "
              f"f1={vm['f1']:.4f} best={is_best} stale={stale}/{args.patience}")
        if stale >= args.patience: break
    if best_pack is None: raise RuntimeError(f"Fold {fold}: no final checkpoint")
    model.load_state_dict(best_pack[0]); selector.load_state_dict(best_pack[1]); classifier.load_state_dict(best_pack[2])
    test = evaluate_final(model, probe, selector, classifier, te, device, triu, args, True)
    summary = {
        "fold": fold, "stage1_best_epoch": s1_epoch, "stage1_val_loss": s1_loss,
        "stage2_best_epoch": s2_epoch, "stage2_val_loss": s2_loss,
        "best_epoch": best_epoch, "selection_val_auc": best_auc,
        "selection_rule": "maximum_inner_validation_auc",
        **{k: test[k] for k in ("auc", "accuracy", "sensitivity", "specificity", "f1",
                                "n_pred_asd", "n_pred_td", "p_asd_mean")},
    }
    torch.save({
        "relation_model_state_dict": best_pack[0], "probe_state_dict": probe.state_dict(),
        "roi_selector_state_dict": best_pack[1], "classifier_state_dict": best_pack[2],
        "train_indices": train_idx, "validation_indices": val_idx, "test_indices": test_idx,
        "test_y_true": test["y_true"], "test_probability_asd": test["probability_asd"],
        "test_roi_keep": test["roi_keep"], "test_edge_relevance": test["edge_relevance"],
        "summary": summary, "args": vars(args),
        "official_commit": "b3da8f13dc6bbbe68b66c62e4c8cc719b254969a",
    }, save_dir / f"fold{fold}_best_and_explanations.pt")
    with (save_dir / f"fold{fold}_summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(f"[Fold {fold} TEST] AUC={test['auc']:.4f} Acc={test['accuracy']:.4f} "
          f"Sen={test['sensitivity']:.4f} Spe={test['specificity']:.4f} F1={test['f1']:.4f}")
    return summary

import datetime

def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = Path(f"./baseline_experiments/eag_rs_results/{timestamp}/")
    os.makedirs(save_dir, exist_ok=True)

    with (save_dir / "training_history.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["fold", "stage", "epoch", "train_loss", "val_loss",
                                "val_auc", "val_accuracy", "is_best"])
    dataset = FCDataset(xlsx_path=args.xlsx_path, n_rois=args.n_rois,
                        label_col=args.label_col)
    x = np.stack([dataset[i]["fc"].numpy() for i in range(len(dataset))]).astype(np.float32)
    y = np.asarray([int(dataset[i]["label"].item()) for i in range(len(dataset))])
    if set(np.unique(y)) != {0, 1}: raise ValueError(f"Expected ASD=0, TD=1; got {np.unique(y)}")
    triu = torch.triu_indices(args.n_rois, args.n_rois, offset=1, device=device)
    cv = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.split_seed)
    summaries: List[Dict] = []
    for fold, (outer_train, test_idx) in enumerate(cv.split(x, y), 1):
        summaries.append(train_fold(fold, x, y, outer_train, test_idx, args,
                                    device, save_dir, triu))
    result = {"fold_summaries": summaries}
    for metric in ("auc", "accuracy", "sensitivity", "specificity", "f1"):
        values = [float(s[metric]) for s in summaries]
        result[f"{metric}_per_fold"] = values
        result[f"{metric}_mean"] = float(np.mean(values)); result[f"{metric}_std"] = float(np.std(values))
    with (save_dir / "cv_results.json").open("w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
