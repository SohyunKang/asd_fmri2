from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from typing import List


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic can slow down; enable if you want exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: (B, C)
    y: (B,)
    """
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def normalize_fc(
    fc: torch.Tensor,
    method: str = "row",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize FC to use as propagation operator A.

    fc: (B, N, N) or (N, N)
    method:
        - "row": row-stochastic (like random walk): A[i,:] sums to 1
        - "sym": symmetric normalization: D^{-1/2} A D^{-1/2} (requires nonnegative)
        - "none": no normalization
    """
    if method == "none":
        return fc

    if fc.dim() == 2:
        fc_ = fc.unsqueeze(0)
    else:
        fc_ = fc

    if method == "row":
        row_sum = fc_.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
        out = fc_ / row_sum
        
    elif method == "sym":
        deg = fc_.abs().sum(dim=-1).clamp_min(eps)  # (B, N)
        d_inv_sqrt = torch.pow(deg, -0.5)           # (B, N)
        D = torch.diag_embed(d_inv_sqrt)            # (B, N, N)
        out = D @ fc_ @ D
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return out.squeeze(0) if fc.dim() == 2 else out


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def load_roi_names_from_txt(txt_path: str) -> List[str]:
    """
    Load ROI names from the 2nd column of a txt file.
    Assumes whitespace/tab-separated lines like:
      1  Precentral_L
      2  Precentral_R
    """
    roi_names: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            roi_names.append(parts[1])
    return roi_names
