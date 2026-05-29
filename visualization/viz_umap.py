import os
from typing import Dict, List, Optional

import numpy as np
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _default_colors():
    return {
        0: "#1f77b4",
        1: "#d62728",
        2: "#2ca02c",
        3: "#9467bd",
    }


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


def _legend_handles_overlay(labels, colors):
    handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="train"),
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=7, label="val"),
        Line2D([0], [0], color="black", lw=2, label="trajectory"),
    ]
    for lab in labels:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color=colors.get(int(lab), "#7f7f7f"),
                linestyle="None",
                markersize=6,
                label=f"label {int(lab)}",
            )
        )
    return handles


def _legend_handles_single(labels, colors):
    handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="points"),
        Line2D([0], [0], color="black", lw=2, label="trajectory"),
    ]
    for lab in labels:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color=colors.get(int(lab), "#7f7f7f"),
                linestyle="None",
                markersize=6,
                label=f"label {int(lab)}",
            )
        )
    return handles


def _plot_umap_trajectory(
    X: np.ndarray,
    y: np.ndarray,
    step: np.ndarray,
    split: Optional[np.ndarray] = None,
    title: str = "",
    save_path: str = "",
    colors: Optional[dict] = None,
    arrow_steps: bool = True,
    annotate_steps: bool = True,
    arrow_size: int = 10,
    text_offset: float = 0.015,
):
    if colors is None:
        colors = _default_colors()

    n_samples = X.shape[0]
    if n_samples < 3:
        return

    n_neighbors = min(30, max(2, n_samples - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    Z = reducer.fit_transform(X)

    unique_labels = sorted(np.unique(y).tolist())
    steps_sorted = sorted(np.unique(step).tolist())

    plt.figure(figsize=(7.8, 6.8))

    # scatter
    if split is None:
        for lab in unique_labels:
            m = (y == lab)
            if m.sum() == 0:
                continue
            plt.scatter(
                Z[m, 0],
                Z[m, 1],
                s=10,
                alpha=0.35,
                c=colors.get(int(lab), "#7f7f7f"),
                marker="o",
                linewidths=0,
                zorder=1,
            )
    else:
        for lab in unique_labels:
            m_train = (y == lab) & (split == 0)
            m_val = (y == lab) & (split == 1)

            if m_train.sum() > 0:
                plt.scatter(
                    Z[m_train, 0],
                    Z[m_train, 1],
                    s=10,
                    alpha=0.20,
                    c=colors.get(int(lab), "#7f7f7f"),
                    marker="o",
                    linewidths=0,
                    zorder=1,
                )

            if m_val.sum() > 0:
                plt.scatter(
                    Z[m_val, 0],
                    Z[m_val, 1],
                    s=16,
                    alpha=0.85,
                    c=colors.get(int(lab), "#7f7f7f"),
                    marker="^",
                    linewidths=0,
                    zorder=2,
                )

    # centroid trajectory
    for lab in unique_labels:
        centroids = []
        ks = []

        for k in steps_sorted:
            m = (y == lab) & (step == k)
            if m.sum() == 0:
                continue
            centroids.append(Z[m].mean(axis=0))
            ks.append(k)

        if len(centroids) < 2:
            continue

        centroids = np.stack(centroids, axis=0)

        plt.plot(
            centroids[:, 0],
            centroids[:, 1],
            lw=3,
            alpha=0.95,
            c=colors.get(int(lab), "#7f7f7f"),
            zorder=5,
        )

        if arrow_steps:
            for i in range(len(centroids) - 1):
                plt.annotate(
                    "",
                    xy=centroids[i + 1],
                    xytext=centroids[i],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=2,
                        color=colors.get(int(lab), "#7f7f7f"),
                        mutation_scale=arrow_size,
                    ),
                    zorder=6,
                )

        if annotate_steps:
            for k, (x_pt, y_pt) in zip(ks, centroids):
                plt.text(
                    x_pt + text_offset,
                    y_pt + text_offset,
                    str(int(k)),
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.6,
                        edgecolor="none",
                    ),
                    zorder=7,
                )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    if split is None:
        plt.legend(
            handles=_legend_handles_single(unique_labels, colors),
            frameon=False,
            fontsize=8,
            loc="best",
        )
    else:
        plt.legend(
            handles=_legend_handles_overlay(unique_labels, colors),
            frameon=False,
            fontsize=8,
            loc="best",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_umap_seed_trajectories(
    seed_train: Dict[int, Dict[str, np.ndarray]],
    seed_val: Dict[int, Dict[str, np.ndarray]],
    epoch: int,
    save_dir: str,
    fold: Optional[int] = None,
    roi_names: Optional[List[str]] = None,
    draw_split_figures: bool = True,
    **kwargs,  # main.py의 옛 인자들이 들어와도 무시
):
    os.makedirs(save_dir, exist_ok=True)
    colors = _default_colors()
    seeds = list(seed_train.keys())

    if len(seeds) == 0:
        return

    def _roi_name(seed: int) -> str:
        if roi_names is not None and 0 <= seed < len(roi_names):
            return roi_names[seed]
        return f"ROI{seed+1}"

    prefix = f"fold{fold}_" if fold is not None else ""

    # seed-specific figures
    for rank, seed in enumerate(seeds):
        if seed not in seed_train or seed not in seed_val:
            continue

        tr = seed_train[seed]
        va = seed_val[seed]

        roi_name = _roi_name(seed)
        roi_safe = _safe_name(roi_name)

        # train only
        if draw_split_figures:
            train_path = os.path.join(
                save_dir,
                f"{prefix}epoch{epoch:03d}_train_rank{rank+1:02d}_seed{seed:03d}_{roi_safe}.png",
            )
            _plot_umap_trajectory(
                X=tr["X"],
                y=tr["y"],
                step=tr["step"],
                split=None,
                title=f"{roi_name} train | Epoch {epoch}",
                save_path=train_path,
                colors=colors,
            )

            # val only
            val_path = os.path.join(
                save_dir,
                f"{prefix}epoch{epoch:03d}_val_rank{rank+1:02d}_seed{seed:03d}_{roi_safe}.png",
            )
            _plot_umap_trajectory(
                X=va["X"],
                y=va["y"],
                step=va["step"],
                split=None,
                title=f"{roi_name} val | Epoch {epoch}",
                save_path=val_path,
                colors=colors,
            )

        # overlay
        X = np.concatenate([tr["X"], va["X"]], axis=0)
        y = np.concatenate([tr["y"], va["y"]], axis=0)
        step = np.concatenate([tr["step"], va["step"]], axis=0)
        split = np.concatenate([tr["split"], va["split"]], axis=0)

        overlay_path = os.path.join(
            save_dir,
            f"{prefix}epoch{epoch:03d}_overlay_rank{rank+1:02d}_seed{seed:03d}_{roi_safe}.png",
        )
        _plot_umap_trajectory(
            X=X,
            y=y,
            step=step,
            split=split,
            title=f"{roi_name} overlay | Epoch {epoch}",
            save_path=overlay_path,
            colors=colors,
        )

    # top-K seed mean figure
    base_seed = seeds[0]
    if base_seed in seed_train and base_seed in seed_val:
        X_train_mean = np.mean([seed_train[s]["X"] for s in seeds if s in seed_train], axis=0)
        X_val_mean = np.mean([seed_val[s]["X"] for s in seeds if s in seed_val], axis=0)

        y_train = seed_train[base_seed]["y"]
        y_val = seed_val[base_seed]["y"]
        step_train = seed_train[base_seed]["step"]
        step_val = seed_val[base_seed]["step"]
        split_train = seed_train[base_seed]["split"]
        split_val = seed_val[base_seed]["split"]

        X = np.concatenate([X_train_mean, X_val_mean], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        step = np.concatenate([step_train, step_val], axis=0)
        split = np.concatenate([split_train, split_val], axis=0)

        mean_path = os.path.join(
            save_dir,
            f"{prefix}epoch{epoch:03d}_seedmean.png",
        )
        _plot_umap_trajectory(
            X=X,
            y=y,
            step=step,
            split=split,
            title=f"Top-{len(seeds)} seed mean trajectory | Epoch {epoch}",
            save_path=mean_path,
            colors=colors,
        )


@torch.no_grad()
def extract_seed_step_graphrepr(
    model,
    loader,
    device,
    fc_norm_fn,
    split_name,
    steps,
    seed_indices,
):
    model.eval()

    split_id = 0 if split_name == "train" else 1

    out = {
        int(seed): {"X": [], "y": [], "split": [], "sid": [], "step": []}
        for seed in seed_indices
    }

    sid_counter = 0

    for batch in loader:
        fc = batch["fc"].to(device)
        y = batch["label"].to(device)

        A = fc_norm_fn(fc)
        _, aux = model(A)

        g_seed_list = aux["g_seed_list"]
        seeds_used = np.asarray(aux["seed_indices"])
        pos = {int(s): i for i, s in enumerate(seeds_used.tolist())}

        B = fc.size(0)
        y_np = y.detach().cpu().numpy()
        sids = np.arange(sid_counter, sid_counter + B)
        sid_counter += B

        for k in steps:
            Gk = g_seed_list[k - 1].detach().cpu().numpy()

            for seed in seed_indices:
                seed = int(seed)
                if seed not in pos:
                    continue

                i = pos[seed]
                out[seed]["X"].append(Gk[:, i, :])
                out[seed]["y"].append(y_np)
                out[seed]["split"].append(np.full(B, split_id))
                out[seed]["sid"].append(sids)
                out[seed]["step"].append(np.full(B, k))

    packed = {}
    for seed in seed_indices:
        seed = int(seed)
        if len(out[seed]["X"]) == 0:
            continue

        packed[seed] = {
            "X": np.concatenate(out[seed]["X"], axis=0),
            "y": np.concatenate(out[seed]["y"], axis=0),
            "split": np.concatenate(out[seed]["split"], axis=0),
            "sid": np.concatenate(out[seed]["sid"], axis=0),
            "step": np.concatenate(out[seed]["step"], axis=0),
        }

    return packed