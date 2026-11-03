"""Official-architecture Learnable Counter-Condition baseline for AAL-116.

This single-file adaptation consolidates the authors' public implementation:
https://github.com/es-kang/learnable-counter-condition-FC
commit: 9eed4c1b01366eaa07d222de3017dd0605d138ac

Paper:
E. Kang, D.-W. Heo, J. Lee, and H.-I. Suk, "A Learnable
Counter-Condition Analysis Framework for Functional Connectivity-Based
Neurological Disorder Diagnosis," IEEE TMI, 2024.

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
from typing import Dict, Iterable, List, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official-architecture LCC on AAL-116")
    p.add_argument("--xlsx-path", default="data/abide_subjects_fc_with_demo_260202.xlsx")
    p.add_argument("--label-col", default="DX_GROUP")
    p.add_argument("--n-rois", type=int, default=116)
    p.add_argument("--save-dir", default="./baseline_experiments/counter_condition_official_results")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="CV split seed; keep identical across all baseline models",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--inner-val-size", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    # Official model settings; heads/d_ff are the AAL-116 adaptations.
    p.add_argument("--gumbel-tau", type=float, default=1.0)
    p.add_argument("--gumbel-soft", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-stacks", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=58)
    p.add_argument("--num-proto-asd", type=int, default=1)
    p.add_argument("--num-proto-td", type=int, default=1)

    # Official optimization settings.
    p.add_argument("--lr-mask", type=float, default=5e-4)
    p.add_argument("--lr-transformer", type=float, default=5e-4)
    p.add_argument("--lr-prototype", type=float, default=1e-4)
    p.add_argument("--lr-decoder", type=float, default=5e-4)
    p.add_argument("--lr-decoder-counter", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--h-cls", type=float, default=1.0)
    p.add_argument("--h-rec", type=float, default=1.0)
    p.add_argument("--h-scale", type=float, default=0.5)
    p.add_argument("--h-roi", type=float, default=1.0)
    p.add_argument("--h-cls-counter", type=float, default=0.1)
    p.add_argument("--h-rec-counter", type=float, default=1.0)
    p.add_argument("--counter-start-acc", type=float, default=0.6)
    p.add_argument("--balance-gap", type=float, default=0.1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


class GumbelSigmoid:
    def __init__(self, temperature: float, eps: float = 1e-20):
        if temperature == 0:
            raise ValueError("Gumbel temperature must be non-zero")
        self.temperature = temperature
        self.eps = eps

    def __call__(self, logits: torch.Tensor, soft: bool) -> torch.Tensor:
        u1 = torch.rand_like(logits)
        u2 = torch.rand_like(logits)
        noise = -torch.log(
            torch.log(u2 + self.eps) / torch.log(u1 + self.eps) + self.eps
        )
        y = torch.sigmoid((logits + noise) / self.temperature)
        if soft:
            return y
        hard = (y > 0.5).to(y.dtype)
        return hard - y.detach() + y


class AdaptiveMask(nn.Module):
    """Official subject-adaptive connection mask."""

    def __init__(self, n_rois: int, tau: float, soft: bool):
        super().__init__()
        self.n_rois = n_rois
        self.soft = soft
        self.tau = tau
        self.gumbel = GumbelSigmoid(tau)
        self.register_buffer("triu", torch.triu_indices(n_rois, n_rois, 1))
        n_edges = n_rois * (n_rois - 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(n_edges, n_edges // 2),
            nn.ReLU(),
            nn.Linear(n_edges // 2, n_edges),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.parameters():
            if weight.ndim > 1:
                nn.init.xavier_normal_(weight)

    def set_require_grad(self, enabled: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = enabled

    def forward(self, fc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = fc[:, self.triu[0], self.triu[1]]
        mask_logits = self.mlp(flat)
        if self.training:
            mask_flat = self.gumbel(mask_logits, self.soft)
        else:
            # nn.Module.eval() does not automatically disable custom Gumbel
            # sampling.  Use its noise-free expectation for reproducible
            # validation/test metrics and counter-condition export.
            mask_flat = torch.sigmoid(mask_logits / self.tau)
            if not self.soft:
                mask_flat = (mask_flat >= 0.5).to(mask_flat.dtype)
        upper = torch.zeros_like(fc)
        upper[:, self.triu[0], self.triu[1]] = mask_flat
        mask = upper + upper.transpose(1, 2)
        return mask * fc, mask_flat


class Norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )


class TransformerMLP(nn.Module):
    """Equivalent to timm.models.vision_transformer.Mlp used officially."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class TalkingHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"n_rois={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attention = self.attn_dropout(attention)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_dropout(self.proj(output))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, d_ff: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = Norm(dim)
        self.attention = TalkingHeadAttention(dim, num_heads, dropout)
        self.norm2 = Norm(dim)
        self.mlp = TransformerMLP(dim, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class NonLinearEmbedding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    table = torch.zeros(length, dim)
    position = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float64)
        * (-math.log(10000.0) / dim)
    )
    table[:, 0::2] = torch.sin(position * div).to(table.dtype)
    table[:, 1::2] = torch.cos(position * div[: table[:, 1::2].shape[1]]).to(table.dtype)
    return table


class FCTransformer(nn.Module):
    def __init__(self, n_rois: int, num_heads: int, d_ff: int, stacks: int):
        super().__init__()
        self.n_rois = n_rois
        self.intra_net = NonLinearEmbedding(n_rois)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_rois, num_heads, d_ff) for _ in range(stacks)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_rois))
        self.register_buffer("position", sinusoidal_encoding(n_rois + 1, n_rois))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.parameters():
            if weight.ndim > 1:
                nn.init.xavier_normal_(weight)

    def set_require_grad(self, enabled: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = enabled

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        x = self.intra_net(fc)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x * math.sqrt(self.n_rois) + self.position.to(x.dtype)[None]
        for block in self.blocks:
            x = block(x)
        return x


class ProtoClassifier(nn.Module):
    """Official prototype classifier, reordered for ASD=0 and TD=1."""

    def __init__(
        self, n_rois: int, num_asd: int, num_td: int, scale: float
    ):
        super().__init__()
        self.asd_prototypes = nn.Parameter(torch.randn(n_rois, num_asd))
        self.td_prototypes = nn.Parameter(torch.randn(n_rois, num_td))
        self.scale = scale
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.asd_prototypes)
        nn.init.xavier_normal_(self.td_prototypes)

    def set_require_grad(self, enabled: bool) -> None:
        self.asd_prototypes.requires_grad = enabled
        self.td_prototypes.requires_grad = enabled

    def forward(
        self, cls_token: torch.Tensor, avg_roi_token: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = cls_token.shape[0]
        asd = self.asd_prototypes.unsqueeze(0).expand(batch, -1, -1)
        td = self.td_prototypes.unsqueeze(0).expand(batch, -1, -1)
        asd_sim = self.cosine(cls_token.transpose(2, 1), asd)
        td_sim = self.cosine(cls_token.transpose(2, 1), td)
        logits = self.scale * torch.cat([asd_sim, td_sim], dim=1)
        if avg_roi_token is None:
            return logits, None
        asd_roi = self.cosine(avg_roi_token.transpose(2, 1), asd)
        td_roi = self.cosine(avg_roi_token.transpose(2, 1), td)
        roi_reg = torch.cat([asd_roi, td_roi], dim=1)
        return logits, roi_reg

    def target_vectors(self, labels: torch.Tensor) -> torch.Tensor:
        if self.asd_prototypes.shape[1] != 1 or self.td_prototypes.shape[1] != 1:
            raise ValueError("Counter FC export currently requires one prototype per class")
        asd = self.asd_prototypes[:, 0]
        td = self.td_prototypes[:, 0]
        return torch.stack([asd if int(label) == 0 else td for label in labels])


class CLSDecoder(nn.Module):
    def __init__(self, n_rois: int, dropout: float = 0.3):
        super().__init__()
        n_edges = n_rois * (n_rois - 1) // 2
        self.decoder = nn.Sequential(
            nn.Linear(n_rois, n_edges // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_edges // 2, n_edges),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.parameters():
            if weight.ndim > 1:
                nn.init.xavier_normal_(weight)

    def set_require_grad(self, enabled: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class LCCModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.n_rois = args.n_rois
        self.mask = AdaptiveMask(args.n_rois, args.gumbel_tau, args.gumbel_soft)
        self.transformer = FCTransformer(
            args.n_rois, args.num_heads, args.d_ff, args.num_stacks
        )
        self.prototype = ProtoClassifier(
            args.n_rois, args.num_proto_asd, args.num_proto_td, args.h_scale
        )
        self.decoder = CLSDecoder(args.n_rois)
        self.register_buffer("triu", torch.triu_indices(args.n_rois, args.n_rois, 1))

    def encode(self, fc: torch.Tensor) -> Dict[str, torch.Tensor]:
        masked_fc, mask_flat = self.mask(fc)
        tokens = self.transformer(masked_fc)
        cls = tokens[:, 0]
        roi = tokens[:, 1:]
        logits, roi_reg = self.prototype(
            cls.unsqueeze(1), roi.mean(dim=1).unsqueeze(1)
        )
        return {
            "masked_fc": masked_fc,
            "mask_flat": mask_flat,
            "tokens": tokens,
            "cls": cls,
            "roi": roi,
            "logits": logits,
            "roi_reg": roi_reg,
        }

    def decode_fc(self, representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = self.decoder(representation)
        fc = torch.zeros(
            representation.shape[0],
            self.n_rois,
            self.n_rois,
            dtype=representation.dtype,
            device=representation.device,
        )
        fc[:, self.triu[0], self.triu[1]] = edges
        fc = fc + fc.transpose(1, 2)
        return edges, fc

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        return self.encode(fc)["logits"]


def set_component_grad(model: LCCModel, enabled: bool) -> None:
    model.mask.set_require_grad(enabled)
    model.transformer.set_require_grad(enabled)
    model.prototype.set_require_grad(enabled)
    model.decoder.set_require_grad(enabled)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    dtype: torch.dtype,
    workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x[indices], dtype=dtype),
        torch.tensor(y[indices], dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) % batch_size == 1,
    )


def standard_train_epoch(
    model: LCCModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    set_component_grad(model, True)
    model.train()
    total_loss = 0.0
    total_n = 0
    for fc, labels in loader:
        fc, labels = fc.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model.encode(fc)
        reconstructed_edges, _ = model.decode_fc(out["roi"].mean(1) + out["cls"])
        target_edges = out["masked_fc"][:, model.triu[0], model.triu[1]]
        loss_cls = args.h_cls * F.cross_entropy(out["logits"], labels)
        loss_rec = args.h_rec * F.l1_loss(reconstructed_edges, target_edges)
        loss_roi = args.h_roi * out["roi_reg"].abs().mean()
        loss = loss_cls + loss_rec + loss_roi
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * fc.shape[0]
        total_n += fc.shape[0]
    return total_loss / max(total_n, 1)


def counter_train_epoch(
    model: LCCModel,
    loader: DataLoader,
    decoder_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Official step-2 idea with corrected class-balancing indices."""
    model.eval()
    set_component_grad(model, False)
    model.decoder.set_require_grad(True)
    model.decoder.train()
    total_loss = 0.0
    updates = 0

    for fc, labels in loader:
        fc, labels = fc.to(device), labels.to(device)
        with torch.no_grad():
            out = model.encode(fc)
            prediction = out["logits"].argmax(dim=1)
            correct = torch.where(prediction == labels)[0]
            asd = correct[labels[correct] == 0]
            td = correct[labels[correct] == 1]
            count = min(asd.numel(), td.numel())
            if count == 0:
                continue
            asd = asd[torch.randperm(asd.numel(), device=device)[:count]]
            td = td[torch.randperm(td.numel(), device=device)[:count]]
            selected = torch.cat([asd, td])
            opposite = torch.cat([td, asd])

            cls_selected = out["cls"][selected]
            cls_opposite = out["cls"][opposite]
            roi_selected = out["roi"][selected].mean(1)
            origin_norm = cls_selected.norm(dim=1, keepdim=True)
            opposite_norm = cls_opposite.norm(dim=1, keepdim=True).clamp_min(1e-8)
            counter_rep = roi_selected + cls_opposite / opposite_norm * origin_norm
            counter_labels = labels[opposite]
            original_rep = roi_selected + cls_selected
            target_masked_edges = out["masked_fc"][selected][
                :, model.triu[0], model.triu[1]
            ]

        decoder_optimizer.zero_grad(set_to_none=True)
        counter_edges, counter_fc = model.decode_fc(counter_rep)
        original_edges, _ = model.decode_fc(original_rep)
        counter_tokens = model.transformer(counter_fc)
        counter_logits, _ = model.prototype(counter_tokens[:, :1], None)
        loss_cls = args.h_cls_counter * F.cross_entropy(
            counter_logits, counter_labels
        )
        loss_rec = args.h_rec_counter * F.l1_loss(
            original_edges, target_masked_edges
        )
        loss = loss_cls + loss_rec
        loss.backward()
        decoder_optimizer.step()
        total_loss += float(loss.detach())
        updates += 1

    set_component_grad(model, True)
    return total_loss / max(updates, 1)


@torch.no_grad()
def evaluate(
    model: LCCModel, loader: DataLoader, args: argparse.Namespace, device: torch.device
) -> Dict[str, float]:
    model.eval()
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    for fc, labels in loader:
        fc, labels_device = fc.to(device), labels.to(device)
        out = model.encode(fc)
        reconstructed, _ = model.decode_fc(out["roi"].mean(1) + out["cls"])
        target = out["masked_fc"][:, model.triu[0], model.triu[1]]
        loss = (
            args.h_cls * F.cross_entropy(out["logits"], labels_device)
            + args.h_rec * F.l1_loss(reconstructed, target)
            + args.h_roi * out["roi_reg"].abs().mean()
        )
        probability_asd = out["logits"].softmax(dim=1)[:, 0]
        prediction = out["logits"].argmax(dim=1)
        total_loss += float(loss) * fc.shape[0]
        total_n += fc.shape[0]
        labels_all.append(labels.numpy())
        probs_all.append(probability_asd.cpu().numpy())
        preds_all.append(prediction.cpu().numpy())

    y = np.concatenate(labels_all)
    p_asd = np.concatenate(probs_all)
    pred = np.concatenate(preds_all)
    y_asd = (y == 0).astype(np.int64)
    return {
        "loss": total_loss / max(total_n, 1),
        "auc": float(roc_auc_score(y_asd, p_asd)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "p_asd_mean": float(p_asd.mean()),
        "p_asd_min": float(p_asd.min()),
        "p_asd_max": float(p_asd.max()),
        "n_pred_asd": int((pred == 0).sum()),
        "n_pred_td": int((pred == 1).sum()),
    }


@torch.no_grad()
def export_counter_conditions(
    model: LCCModel, loader: DataLoader, device: torch.device
) -> Dict[str, torch.Tensor]:
    model.eval()
    saved: Dict[str, List[torch.Tensor]] = {
        "original_fc": [],
        "masked_fc": [],
        "counter_fc": [],
        "delta_fc": [],
        "label": [],
        "prediction_before": [],
        "prediction_after": [],
        "probability_asd_before": [],
        "probability_asd_after": [],
    }
    for fc, labels in loader:
        fc = fc.to(device)
        labels_device = labels.to(device)
        out = model.encode(fc)
        before = out["logits"].argmax(dim=1)
        target = 1 - before
        target_proto = model.prototype.target_vectors(target)
        target_proto = target_proto / target_proto.norm(dim=1, keepdim=True).clamp_min(1e-8)
        target_proto = target_proto * out["cls"].norm(dim=1, keepdim=True)
        counter_rep = out["roi"].mean(1) + target_proto
        _, counter_fc = model.decode_fc(counter_rep)
        counter_out = model.encode(counter_fc)
        after = counter_out["logits"].argmax(dim=1)

        values = {
            "original_fc": fc,
            "masked_fc": out["masked_fc"],
            "counter_fc": counter_fc,
            "delta_fc": counter_fc - out["masked_fc"],
            "label": labels_device,
            "prediction_before": before,
            "prediction_after": after,
            "probability_asd_before": out["logits"].softmax(1)[:, 0],
            "probability_asd_after": counter_out["logits"].softmax(1)[:, 0],
        }
        for key, value in values.items():
            saved[key].append(value.detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in saved.items()}


def build_optimizers(
    model: LCCModel, args: argparse.Namespace
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizer = torch.optim.AdamW(
        [
            {"params": model.mask.parameters(), "lr": args.lr_mask},
            {"params": model.transformer.intra_net.parameters(), "lr": args.lr_transformer},
            {"params": model.transformer.blocks.parameters(), "lr": args.lr_transformer},
            {"params": model.transformer.cls_token, "lr": args.lr_prototype, "weight_decay": 0},
            {"params": model.prototype.parameters(), "lr": args.lr_prototype, "weight_decay": 0},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder},
        ],
        lr=args.lr_transformer,
        weight_decay=args.weight_decay,
    )
    decoder_optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=args.lr_decoder_counter,
        weight_decay=args.weight_decay,
    )
    return optimizer, decoder_optimizer


def train_fold(
    fold: int,
    x: np.ndarray,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    save_dir: Path,
) -> Dict[str, float]:
    fold_seed = args.seed + fold
    set_seed(fold_seed)
    train_idx, val_idx = train_test_split(
        outer_train_idx,
        test_size=args.inner_val_size,
        stratify=y[outer_train_idx],
        random_state=fold_seed,
    )
    train_loader = make_loader(
        x, y, train_idx, args.batch_size, True, dtype, args.num_workers
    )
    val_loader = make_loader(
        x, y, val_idx, args.batch_size, False, dtype, args.num_workers
    )
    test_loader = make_loader(
        x, y, test_idx, args.batch_size, False, dtype, args.num_workers
    )
    model = LCCModel(args).to(device=device, dtype=dtype)
    optimizer, decoder_optimizer = build_optimizers(model, args)
    # The official checkpoint rule is validation-loss based.  We retain two
    # checkpoints while completing all epochs:
    #   1) lowest validation loss among balanced, even-epoch candidates;
    #   2) lowest validation loss overall, used only when (1) does not exist.
    best_balanced_loss = np.inf
    best_balanced_state = None
    best_balanced_epoch = -1
    best_balanced_val = None
    best_fallback_loss = np.inf
    best_fallback_state = None
    best_fallback_epoch = -1
    best_fallback_val = None
    previous_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if previous_val_acc > args.counter_start_acc and epoch % 2 == 0:
            train_loss = counter_train_epoch(
                model, train_loader, decoder_optimizer, args, device
            )
            stage = "counter"
        else:
            train_loss = standard_train_epoch(model, train_loader, optimizer, args, device)
            stage = "standard"

        val = evaluate(model, val_loader, args, device)
        previous_val_acc = val["accuracy"]
        finite_loss = np.isfinite(val["loss"])
        balanced_candidate = (
            finite_loss
            and epoch % 2 == 0
            and abs(val["sensitivity"] - val["specificity"]) <= args.balance_gap
        )

        is_best_fallback = finite_loss and val["loss"] < best_fallback_loss
        if is_best_fallback:
            best_fallback_loss = val["loss"]
            best_fallback_state = copy.deepcopy(model.state_dict())
            best_fallback_epoch = epoch
            best_fallback_val = copy.deepcopy(val)

        is_best_balanced = (
            balanced_candidate and val["loss"] < best_balanced_loss
        )
        if is_best_balanced:
            best_balanced_loss = val["loss"]
            best_balanced_state = copy.deepcopy(model.state_dict())
            best_balanced_epoch = epoch
            best_balanced_val = copy.deepcopy(val)

        with (save_dir / "training_history.csv").open("a", newline="") as file:
            csv.writer(file).writerow([
                fold, epoch, train_loss, val["loss"], val["auc"],
                val["accuracy"], val["sensitivity"], val["specificity"],
                val["f1"], stage, int(balanced_candidate),
                int(is_best_balanced), int(is_best_fallback),
            ])

        print(
            f"[Fold {fold} | {epoch:03d}/{args.epochs} | {stage}] "
            f"train={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"auc={val['auc']:.4f} acc={val['accuracy']:.4f} "
            f"sen={val['sensitivity']:.4f} spe={val['specificity']:.4f} "
            f"f1={val['f1']:.4f} pASD={val['p_asd_mean']:.3f} "
            f"pred={val['n_pred_asd']}/{val['n_pred_td']} "
            f"balanced={balanced_candidate} "
            f"best_balanced={is_best_balanced} "
            f"best_fallback={is_best_fallback}"
        )

    if best_balanced_state is not None:
        best_state = best_balanced_state
        best_epoch = best_balanced_epoch
        best_val = best_balanced_val
        selection_rule = "balanced_even_epoch_min_val_loss"
    else:
        best_state = best_fallback_state
        best_epoch = best_fallback_epoch
        best_val = best_fallback_val
        selection_rule = "fallback_min_val_loss"

    if best_state is None or best_val is None:
        raise RuntimeError(f"Fold {fold}: no checkpoint was saved")

    model.load_state_dict(best_state)
    print(
        f"[Fold {fold} SELECTED] rule={selection_rule} epoch={best_epoch} "
        f"val_loss={best_val['loss']:.4f} "
        f"val_auc={best_val['auc']:.4f} "
        f"val_acc={best_val['accuracy']:.4f} "
        f"val_sen={best_val['sensitivity']:.4f} "
        f"val_spe={best_val['specificity']:.4f}"
    )
    final_metrics = evaluate(model, test_loader, args, device)
    explanations = export_counter_conditions(model, test_loader, device)
    summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "selection_rule": selection_rule,
        "balance_gap": args.balance_gap,
        "selection_validation_metrics": best_val,
        **final_metrics,
    }
    torch.save(
        {
            "model_state_dict": best_state,
            "outer_train_indices": outer_train_idx,
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "test_indices": test_idx,
            "summary": summary,
            "counter_conditions": explanations,
            "args": vars(args),
            "official_commit": "9eed4c1b01366eaa07d222de3017dd0605d138ac",
        },
        save_dir / f"fold{fold}_best_and_counter_conditions.pt",
    )
    with (save_dir / f"fold{fold}_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(
        f"[Fold {fold} FINAL] AUC={final_metrics['auc']:.4f} "
        f"Acc={final_metrics['accuracy']:.4f} "
        f"Sen(ASD)={final_metrics['sensitivity']:.4f} "
        f"Spe(TD)={final_metrics['specificity']:.4f} "
        f"F1(ASD)={final_metrics['f1']:.4f} pred ASD/TD="
        f"{final_metrics['n_pred_asd']}/{final_metrics['n_pred_td']}"
    )
    return summary


def summarize(folds: List[Dict[str, float]], save_dir: Path) -> Dict[str, object]:
    metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    result: Dict[str, object] = {"fold_summaries": folds}
    for metric in metrics:
        values = [float(fold[metric]) for fold in folds]
        result[f"{metric}_per_fold"] = values
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values))
    with (save_dir / "cv_results.json").open("w") as file:
        json.dump(result, file, indent=2)
    print("\nFive-fold held-out results (mean +/- population std)")
    for metric in metrics:
        print(
            f"{metric:>11}: {result[f'{metric}_mean']:.4f} +/- "
            f"{result[f'{metric}_std']:.4f}"
        )
    return result


def main() -> None:
    args = parse_args()
    if args.n_rois % args.num_heads != 0:
        raise ValueError("--num-heads must divide --n-rois exactly")
    set_seed(args.seed)
    dtype = model_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "training_history.csv").open("w", newline="") as file:
        csv.writer(file).writerow([
            "fold", "epoch", "train_loss", "val_loss", "val_auc",
            "val_accuracy", "val_sensitivity", "val_specificity", "val_f1",
            "stage", "balanced_candidate", "is_best_balanced_so_far",
            "is_best_fallback_so_far",
        ])

    dataset = FCDataset(
        xlsx_path=args.xlsx_path,
        n_rois=args.n_rois,
        label_col=args.label_col,
    )
    x = np.stack([dataset[i]["fc"].numpy() for i in range(len(dataset))])
    y = np.asarray(
        [int(dataset[i]["label"].item()) for i in range(len(dataset))],
        dtype=np.int64,
    )
    expected = {0, 1}
    if set(np.unique(y).tolist()) != expected:
        raise ValueError(f"Expected mapped labels {expected}, found {np.unique(y)}")
    x = x.astype(np.float64 if dtype == torch.float64 else np.float32)
    print(
        f"Data={x.shape}, ASD(0)={(y == 0).sum()}, TD(1)={(y == 1).sum()}, "
        f"device={device}, dtype={args.dtype}"
    )

    cv = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.split_seed
    )
    fold_results = []
    for fold, (outer_train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        fold_results.append(
            train_fold(
                fold,
                x,
                y,
                outer_train_idx,
                test_idx,
                args,
                device,
                dtype,
                save_dir,
            )
        )
    summarize(fold_results, save_dir)


if __name__ == "__main__":
    main()
