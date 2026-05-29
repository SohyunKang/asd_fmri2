import os
from typing import List, Optional, Dict

import numpy as np
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def extract_arrival_outputs(
    model,
    loader,
    device,
    fc_norm_fn,
    split_id: int,                 # 0=train, 1=val
    arrival_w_key: str = "arrival",
    arrival_g_key: str = "g_arr",
):
    """
    Returns dict with:
      w: (S, N)   arrival node weights
      g: (S, d)   arrival graph embeddings
      y: (S,)
      split: (S,)
    """
    model.eval()
    ws, gs, ys = [], [], []

    for batch in loader:
        fc = batch["fc"].to(device)
        y = batch["label"].to(device)

        A = fc_norm_fn(fc)
        _logits, aux = model(A)

        if arrival_w_key not in aux:
            raise KeyError(f"aux['{arrival_w_key}'] not found. Available keys: {list(aux.keys())}")
        if arrival_g_key not in aux:
            raise KeyError(f"aux['{arrival_g_key}'] not found. Available keys: {list(aux.keys())}")

        w = aux[arrival_w_key]   # (B,N)
        g = aux[arrival_g_key]   # (B,d)

        ws.append(w.detach().cpu().numpy())
        gs.append(g.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    w_all = np.concatenate(ws, axis=0)
    g_all = np.concatenate(gs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    split_all = np.full_like(y_all, split_id, dtype=np.int64)

    return {"w": w_all, "g": g_all, "y": y_all, "split": split_all}


def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("'", "")
            .replace('"', "")
    )

from matplotlib.patches import Patch
def save_arrival_weight_map(
    train_pack,
    val_pack,
    epoch,
    save_dir,
    fold=None,
    roi_names=None,
    topk=20,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(save_dir, exist_ok=True)

    # ✅ robust key resolution
    if "arrival" in val_pack:
        a = val_pack["arrival"]
    elif "w" in val_pack:
        a = val_pack["w"]
    elif "a" in val_pack:
        a = val_pack["a"]
    else:
        raise KeyError(f"val_pack has no arrival weights. keys={list(val_pack.keys())}")

    if "y" not in val_pack:
        raise KeyError(f"val_pack has no labels 'y'. keys={list(val_pack.keys())}")

    y = val_pack["y"]

    labs = sorted(np.unique(y).tolist())
    if len(labs) < 2:
        return

    mean = {int(lab): a[y == lab].mean(axis=0) for lab in labs}

    if 0 in mean and 1 in mean:
        lab0, lab1 = 0, 1
        diff = mean[1] - mean[0]
        diff_name = "TD - ASD"
        label_name = {0: "ASD", 1: "TD"}
    else:
        lab0, lab1 = labs[0], labs[-1]
        diff = mean[lab1] - mean[lab0]
        diff_name = f"label{lab1} - label{lab0}"
        label_name = {lab0: f"label{lab0}", lab1: f"label{lab1}"}

    def _names(idxs):
        out = []
        for i in idxs:
            if roi_names is not None and 0 <= int(i) < len(roi_names):
                out.append(roi_names[int(i)])
            else:
                out.append(f"ROI{int(i)+1}")
        return out

    idx_diff = np.argsort(-np.abs(diff))[:topk]
    idx_l0 = np.argsort(-mean[lab0])[:topk]
    idx_l1 = np.argsort(-mean[lab1])[:topk]

    names_diff = _names(idx_diff)
    names_l0 = _names(idx_l0)
    names_l1 = _names(idx_l1)

    vals_diff = diff[idx_diff]
    vals_l0 = mean[lab0][idx_l0]
    vals_l1 = mean[lab1][idx_l1]

    def _safe_name(name):
        return (
            name.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("'", "")
                .replace('"', "")
        )

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 13.0))

    # Panel 1: diff
    ax = axes[0]
    x = np.arange(topk)
    colors_diff = ["#d62728" if v > 0 else "#1f77b4" for v in vals_diff]
    ax.bar(x, vals_diff, color=colors_diff)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_safe_name(n) for n in names_diff], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(diff_name)
    ax.set_title(f"(1) Top{topk} by |diff| (val)")
    ax.legend(
        handles=[
            Patch(facecolor="#d62728", label=f"{label_name[lab1]} > {label_name[lab0]}"),
            Patch(facecolor="#1f77b4", label=f"{label_name[lab1]} < {label_name[lab0]}"),
        ],
        loc="upper right"
    )

    # Panel 2: label0
    ax = axes[1]
    ax.bar(x, vals_l0, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([_safe_name(n) for n in names_l0], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean arrival weight")
    ax.set_title(f"(2) Top{topk} ROIs (val) - {label_name[lab0]}")
    ax.legend(handles=[Patch(facecolor="#1f77b4", label=label_name[lab0])], loc="upper right")

    # Panel 3: label1
    ax = axes[2]
    ax.bar(x, vals_l1, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([_safe_name(n) for n in names_l1], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean arrival weight")
    ax.set_title(f"(3) Top{topk} ROIs (val) - {label_name[lab1]}")
    ax.legend(handles=[Patch(facecolor="#d62728", label=label_name[lab1])], loc="upper right")

    fig.suptitle(f"Arrival hub analysis | Epoch {epoch}", y=0.995)
    fig.tight_layout()

    prefix = f"fold{fold}_" if fold is not None else ""
    fig.savefig(os.path.join(save_dir, f"{prefix}epoch{epoch:03d}_arrival_weight_tripanel_top{topk}.png"), dpi=300)
    plt.close(fig)


def save_arrival_embedding_umap(
    pack_train: Dict[str, np.ndarray],
    pack_val: Dict[str, np.ndarray],
    epoch: int,
    save_dir: str,
    fold: Optional[int] = None,
    train_alpha: float = 0.20,
    val_alpha: float = 0.70,
):
    """
    B2) Arrival embedding scatter (UMAP):
      - Fit one UMAP on train+val arrival embeddings g_arr
      - Plot train/val markers with different alpha, colored by label
    """
    os.makedirs(save_dir, exist_ok=True)

    colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}

    g = np.concatenate([pack_train["g"], pack_val["g"]], axis=0)
    y = np.concatenate([pack_train["y"], pack_val["y"]], axis=0)
    split = np.concatenate([pack_train["split"], pack_val["split"]], axis=0)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    Z = reducer.fit_transform(g)

    labels = sorted(np.unique(y).tolist())

    plt.figure(figsize=(7.2, 6.2))

    # train and val
    for split_id, alpha, marker in [(0, train_alpha, "o"), (1, val_alpha, "^")]:
        msp = (split == split_id)
        for lab in labels:
            m = msp & (y == lab)
            if m.sum() == 0:
                continue
            plt.scatter(
                Z[m, 0], Z[m, 1],
                s=10,
                alpha=alpha,
                marker=marker,
                c=colors.get(int(lab), "#7f7f7f"),
                linewidths=0,
            )

    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.title(f"Arrival embedding (g_arr) UMAP | Epoch {epoch}")
    plt.tight_layout()

    prefix = f"fold{fold}_" if fold is not None else ""
    plt.savefig(os.path.join(save_dir, f"{prefix}epoch{epoch:03d}_arrival_garr_umap.png"), dpi=300)
    plt.close()
