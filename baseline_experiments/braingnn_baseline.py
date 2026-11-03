"""BrainGNN official-architecture baseline for AAL-116 FC.

Adapted from the authors' preliminary PyTorch implementation:
https://github.com/xxlya/BrainGNN_Pytorch

"""

import argparse
import csv
import json
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
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
    softmax,
    sort_edge_index,
)
from torch_sparse import spspmm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import FCDataset


EPS = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official-architecture BrainGNN on AAL-116 FC"
    )
    parser.add_argument(
        "--xlsx-path",
        default="data/abide_subjects_fc_with_demo_260202.xlsx",
    )
    parser.add_argument("--label-col", default="DX_GROUP")
    parser.add_argument("--save-dir", default="./baseline_experiments/braingnn_official_results")
    parser.add_argument("--n-rois", type=int, default=116)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--inner-val-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--communities", type=int, default=8)
    parser.add_argument("--lamb0", type=float, default=1.0)
    parser.add_argument("--lamb1", type=float, default=0.0)
    parser.add_argument("--lamb2", type=float, default=0.0)
    parser.add_argument("--lamb3", type=float, default=0.1)
    parser.add_argument("--lamb4", type=float, default=0.1)
    parser.add_argument("--lamb5", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyNNConv(MessagePassing):
    """Modern-PyG equivalent of the authors' MyNNConv (Ra-GConv)."""

    def __init__(self, in_channels, out_channels, edge_network, bias=True):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_network = edge_network
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            bound = 1.0 / np.sqrt(self.in_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index, edge_weight, pseudo):
        edge_weight = edge_weight.view(-1)
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0)
        )
        weight = self.edge_network(pseudo).view(
            -1, self.in_channels, self.out_channels
        )
        x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)
        return self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            size=None,
        )

    def message(self, x_j, edge_weight, index, ptr, size_i):
        alpha = softmax(edge_weight, index, ptr, size_i)
        return alpha.view(-1, 1) * x_j

    def update(self, inputs):
        if self.bias is not None:
            inputs = inputs + self.bias
        return inputs


def pooling_weight(pool):
    """Support both the old and current TopKPooling parameter locations."""
    if hasattr(pool, "weight"):
        return pool.weight
    if hasattr(pool, "select") and hasattr(pool.select, "weight"):
        return pool.select.weight
    raise AttributeError("Cannot locate the TopKPooling scoring weight.")


class BrainGNN(nn.Module):
    """Authors' two-layer BrainGNN architecture with R adapted to 116."""

    def __init__(self, indim=116, ratio=0.5, nclass=2, k=8, n_rois=116):
        super().__init__()
        self.indim = indim
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 512
        self.k = k
        self.n_rois = n_rois

        self.n1 = nn.Sequential(
            nn.Linear(n_rois, k, bias=False),
            nn.ReLU(),
            nn.Linear(k, self.dim1 * indim),
        )
        self.conv1 = MyNNConv(indim, self.dim1, self.n1)
        self.pool1 = TopKPooling(
            self.dim1, ratio=ratio, multiplier=1.0, nonlinearity=torch.sigmoid
        )

        self.n2 = nn.Sequential(
            nn.Linear(n_rois, k, bias=False),
            nn.ReLU(),
            nn.Linear(k, self.dim2 * self.dim1),
        )
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2)
        self.pool2 = TopKPooling(
            self.dim2, ratio=ratio, multiplier=1.0, nonlinearity=torch.sigmoid
        )

        self.fc1 = nn.Linear((self.dim1 + self.dim2) * 2, self.dim2)
        self.bn1 = nn.BatchNorm1d(self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.dim3)
        self.bn2 = nn.BatchNorm1d(self.dim3)
        self.fc3 = nn.Linear(self.dim3, nclass)

    def forward(self, x, edge_index, batch, edge_attr, pos):
        n_graphs = int(batch.max().item()) + 1

        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(
            x, edge_index, edge_attr, batch=batch
        )
        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_index, edge_attr = self.augment_adj(
            edge_index, edge_attr.view(-1), x.size(0)
        )
        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, _, score2 = self.pool2(
            x, edge_index, edge_attr, batch=batch
        )
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        graph_x = torch.cat([x1, x2], dim=1)
        graph_x = self.bn1(F.relu(self.fc1(graph_x)))
        graph_x = F.dropout(graph_x, p=0.5, training=self.training)
        graph_x = self.bn2(F.relu(self.fc2(graph_x)))
        graph_x = F.dropout(graph_x, p=0.5, training=self.training)
        log_prob = F.log_softmax(self.fc3(graph_x), dim=-1)

        s1 = torch.sigmoid(score1).view(n_graphs, -1)
        s2 = torch.sigmoid(score2).view(n_graphs, -1)
        return log_prob, pooling_weight(self.pool1), pooling_weight(self.pool2), s1, s2

    @staticmethod
    def augment_adj(edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        edge_index, edge_weight = sort_edge_index(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        edge_index, edge_weight = spspmm(
            edge_index,
            edge_weight,
            edge_index,
            edge_weight,
            num_nodes,
            num_nodes,
            num_nodes,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


def fc_to_graph(fc, label):
    """Convert one FCDataset sample to the graph representation used by BrainGNN."""
    fc = fc.float()
    n_rois = fc.size(0)
    mask = ~torch.eye(n_rois, dtype=torch.bool)
    row, col = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = fc.abs()[row, col].unsqueeze(1)
    pos = torch.eye(n_rois, dtype=torch.float32)
    return Data(
        x=fc,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.as_tensor(label, dtype=torch.long).view(1),
    )


def build_graphs(fc_dataset):
    graphs = []
    labels = []
    for idx in range(len(fc_dataset)):
        sample = fc_dataset[idx]
        label = int(sample["label"].item())
        graphs.append(fc_to_graph(sample["fc"], label))
        labels.append(label)
    labels = np.asarray(labels, dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("FCDataset must provide ASD=0 and TD=1 labels.")
    return graphs, labels


def topk_loss(scores, ratio):
    ratio = min(ratio, 1.0 - ratio)
    scores = scores.sort(dim=1).values
    k = max(1, int(scores.size(1) * ratio))
    return -torch.log(scores[:, -k:] + EPS).mean() - torch.log(
        1.0 - scores[:, :k] + EPS
    ).mean()


def consistency_loss(scores):
    if scores.size(0) < 2:
        return scores.new_tensor(0.0)
    scores = torch.sigmoid(scores)
    n = scores.size(0)
    weights = torch.ones((n, n), device=scores.device)
    degree = torch.eye(n, device=scores.device) * weights.sum(dim=1)
    laplacian = degree - weights
    return torch.trace(scores.t() @ laplacian @ scores) / (n * n)


def total_loss(output, y, w1, w2, s1, s2, args):
    loss_class = F.nll_loss(output, y)
    loss_unit1 = (torch.norm(w1, p=2) - 1.0).pow(2)
    loss_unit2 = (torch.norm(w2, p=2) - 1.0).pow(2)
    loss_topk1 = topk_loss(s1, args.ratio)
    loss_topk2 = topk_loss(s2, args.ratio)
    loss_consistency = output.new_tensor(0.0)
    for class_id in range(2):
        loss_consistency = loss_consistency + consistency_loss(s1[y == class_id])
    return (
        args.lamb0 * loss_class
        + args.lamb1 * loss_unit1
        + args.lamb2 * loss_unit2
        + args.lamb3 * loss_topk1
        + args.lamb4 * loss_topk2
        + args.lamb5 * loss_consistency
    )


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss_value = 0.0
    n_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        output, w1, w2, s1, s2 = model(
            data.x, data.edge_index, data.batch, data.edge_attr, data.pos
        )
        loss = total_loss(output, data.y.view(-1), w1, w2, s1, s2, args)
        loss.backward()
        optimizer.step()
        total_loss_value += loss.item() * data.num_graphs
        n_graphs += data.num_graphs
    return total_loss_value / n_graphs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob_asd, y_pred = [], [], []
    total_loss, total_n = 0.0, 0
    for data in loader:
        data = data.to(device)
        output, _, _, _, _ = model(
            data.x, data.edge_index, data.batch, data.edge_attr, data.pos
        )
        probability = output.exp()
        total_loss += F.nll_loss(output, data.y.view(-1)).item() * data.num_graphs
        total_n += data.num_graphs
        y_true.append(data.y.view(-1).cpu().numpy())
        y_prob_asd.append(probability[:, 0].cpu().numpy())
        y_pred.append(output.argmax(dim=1).cpu().numpy())

    y = np.concatenate(y_true)
    p_asd = np.concatenate(y_prob_asd)
    pred = np.concatenate(y_pred)
    y_asd = (y == 0).astype(np.int64)
    return {
        "loss": total_loss / total_n,
        "auc": float(roc_auc_score(y_asd, p_asd)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(
            recall_score(y, pred, pos_label=0, zero_division=0)
        ),
        "specificity": float(
            recall_score(y, pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
    }


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    fc_dataset = FCDataset(
        xlsx_path=args.xlsx_path,
        n_rois=args.n_rois,
        label_col=args.label_col,
    )
    graphs, labels = build_graphs(fc_dataset)
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    fold_summaries = []

    config = vars(args).copy()
    config.update(
        model="BrainGNN_official_architecture",
        input_node_features="signed_Pearson_FC_rows",
        input_edge_weights="absolute_Pearson_FC",
        class_order={"ASD": 0, "TD": 1},
        checkpoint_metric="validation_AUC",
    )
    with open(os.path.join(args.save_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2)
    history_path = os.path.join(args.save_dir, "training_history.csv")
    with open(history_path, "w", newline="") as file:
        csv.writer(file).writerow([
            "fold", "epoch", "train_loss", "val_loss", "val_auc",
            "val_accuracy", "val_sensitivity", "val_specificity", "val_f1",
            "is_best", "stale_epochs",
        ])

    for fold, (outer_train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        set_seed(args.seed + fold)
        train_idx, val_idx = train_test_split(
            outer_train_idx, test_size=args.inner_val_size,
            stratify=labels[outer_train_idx], random_state=args.seed + fold
        )
        train_loader = DataLoader(
            [graphs[i] for i in train_idx],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            [graphs[i] for i in val_idx],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            [graphs[i] for i in test_idx], batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )
        model = BrainGNN(
            indim=args.n_rois,
            ratio=args.ratio,
            nclass=2,
            k=args.communities,
            n_rois=args.n_rois,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

        best_auc = -np.inf
        best_epoch = -1
        best_state = None
        stale_epochs = 0
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, device, args)
            metrics = evaluate(model, val_loader, device)
            is_best = metrics["auc"] > best_auc
            if is_best:
                best_auc = metrics["auc"]
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            with open(history_path, "a", newline="") as file:
                csv.writer(file).writerow([
                    fold, epoch, loss, metrics["loss"], metrics["auc"],
                    metrics["accuracy"], metrics["sensitivity"],
                    metrics["specificity"], metrics["f1"],
                    int(is_best), stale_epochs,
                ])
            print(
                f"[Fold {fold} | {epoch:03d}/{args.epochs}] "
                f"train_loss={loss:.4f} val_loss={metrics['loss']:.4f} "
                f"val_auc={metrics['auc']:.4f} val_acc={metrics['accuracy']:.4f} "
                f"val_sens={metrics['sensitivity']:.4f} "
                f"val_spec={metrics['specificity']:.4f} "
                f"val_f1={metrics['f1']:.4f} best={is_best} "
                f"stale={stale_epochs}/{args.patience}"
            )
            scheduler.step()
            if stale_epochs >= args.patience:
                print(f"[Fold {fold}] early stop; best epoch={best_epoch}")
                break

        model.load_state_dict(best_state)
        best_metrics = evaluate(model, test_loader, device)
        summary = {
            "fold": fold,
            "best_epoch": best_epoch,
            "selection_val_auc": best_auc,
            **best_metrics,
        }
        fold_summaries.append(summary)
        torch.save(
            {
                "model_state_dict": best_state,
                "outer_train_indices": outer_train_idx,
                "train_indices": train_idx,
                "validation_indices": val_idx,
                "test_indices": test_idx,
                "config": config,
                **summary,
            },
            os.path.join(args.save_dir, f"fold{fold}_best.pt"),
        )
        with open(
            os.path.join(args.save_dir, f"fold{fold}_summary.json"), "w"
        ) as file:
            json.dump(summary, file, indent=2)

    result = {"fold_summaries": fold_summaries}
    for metric in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
        values = [fold[metric] for fold in fold_summaries]
        mean, std = mean_std(values)
        result[f"{metric}_per_fold"] = values
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std

    with open(os.path.join(args.save_dir, "cv_results.json"), "w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
