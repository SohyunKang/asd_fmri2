import torch
import numpy as np
import umap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import os
import glob


# -------------------------
# config: three-level hierarchy
# Sensory -> Intermediate association/control -> DMN
# -------------------------
net_order = [
    "Sensory",
    "Intermediate",
    "DMN",
]

net_colors = {
    "Sensory": "#1f77b4",
    "Intermediate": "#9467bd",
    "DMN": "#d62728",
}

net_labels = {
    "Sensory": "Sensory",
    "Intermediate": "Intermediate association/control",
    "DMN": "DMN",
}

network_3group = {
    "Sensory": "Sensory",
    "DANnSalience": "Intermediate",
    "Frontoparietal": "Intermediate",
    "DMN": "DMN",
}

EXCLUDE_NETS = {"Limbic", "Other"}


def to_hierarchy_group(net):
    return network_3group.get(net, "Other")


# -------------------------
# utils
# -------------------------
def load_roi_network_map(txt_path):
    mapping = {}

    with open(txt_path) as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            mapping[parts[0]] = parts[1]

    return mapping


def parse_roi_names(lines):
    return [
        line.strip().split()[0]
        for line in lines
        if line.strip()
    ]


def build_legend():

    class_colors = {
        0: "#1f77b4",
        1: "#d62728"
    }

    legend_elements = []

    for net in net_order:

        if net in EXCLUDE_NETS:
            continue

        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=net_labels[net],
                markerfacecolor=net_colors[net],
                markersize=8
            )
        )

    legend_elements.append(
        Line2D(
            [0], [0],
            linestyle="None",
            label=""
        )
    )

    for cls, c in class_colors.items():

        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=f"Class {cls}",
                markerfacecolor=c,
                markersize=6
            )
        )

    legend_elements.append(
        Line2D(
            [0], [0],
            color="black",
            lw=2,
            marker=">",
            label="Trajectory"
        )
    )

    return legend_elements


def build_network_indices(
    roi_names,
    roi_map
):

    net_to_idx = {}

    for i, name in enumerate(roi_names):

        net = roi_map.get(
            name,
            "Other"
        )

        net_to_idx.setdefault(
            net,
            []
        ).append(i)

    return net_to_idx
def bilateral_average_roi(roi_vecs, roi_names, roi_map):
    """
    L/R homologous ROIs are averaged for visualization only.

    Parameters
    ----------
    roi_vecs : (N, d)
        ROI latent representations
    roi_names : list[str]
        e.g. Precentral_L, Precentral_R
    roi_map : dict
        ROI -> functional system

    Returns
    -------
    bilateral_vecs : (N_bilateral, d)
    bilateral_names : list[str]
    bilateral_networks : list[str]
    """

    groups = {}

    for i, name in enumerate(roi_names):

        # remove hemisphere suffix
        if name.endswith("_L") or name.endswith("_R"):
            base = name[:-2]
        else:
            base = name

        if base not in groups:
            groups[base] = []

        groups[base].append(i)

    bilateral_vecs = []
    bilateral_names = []
    bilateral_networks = []

    for base, indices in groups.items():

        # L/R mean
        vec = roi_vecs[indices].mean(axis=0)

        bilateral_vecs.append(vec)
        bilateral_names.append(base)

        # functional system
        nets = [
            roi_map.get(roi_names[i], "Other")
            for i in indices
        ]

        # normally L/R should have same system
        # if not, use the first one
        bilateral_networks.append(to_hierarchy_group(nets[0]))

    return (
        np.stack(bilateral_vecs),
        bilateral_names,
        bilateral_networks
    )


def get_bilateral_seed_groups(seed_indices, roi_names):
    """
    Group selected seeds by homologous anatomical ROI name.
    Example: Cuneus_L + Cuneus_R -> Cuneus

    Returns
    -------
    groups : dict
        {bilateral_name: [seed_axis_positions]}
    """
    groups = {}

    for seed_pos, roi_idx in enumerate(seed_indices):
        name = roi_names[int(roi_idx)]

        if name.endswith("_L") or name.endswith("_R"):
            base = name[:-2]
        else:
            base = name

        groups.setdefault(base, []).append(seed_pos)

    return groups


def alpha_weighted_bilateral_trajectory(
    g,
    seed_positions,
    seed_weights
):
    """
    Alpha-weighted trajectory for homologous selected seeds.

    Parameters
    ----------
    g : torch.Tensor
        (S, K, M, d)
    seed_positions : list[int]
        Positions on the selected-seed axis.
    seed_weights : torch.Tensor
        Global alpha values for the M selected seeds.

    Returns
    -------
    g_group : torch.Tensor
        (S, K, d)
    group_alpha : float
        Sum of alpha values belonging to the bilateral group.
    local_alpha : torch.Tensor
        Alpha normalized within this bilateral group.
    """
    pos = torch.as_tensor(
        seed_positions,
        dtype=torch.long,
        device=g.device
    )

    alpha = torch.as_tensor(
        seed_weights,
        dtype=g.dtype,
        device=g.device
    )[pos]

    group_alpha = float(alpha.sum().detach().cpu())

    # Normalize within the L/R homologous group.
    local_alpha = alpha / (alpha.sum() + 1e-8)

    g_group = (
        g[:, :, pos, :]
        * local_alpha.view(1, 1, -1, 1)
    ).sum(dim=2)

    return (
        g_group,
        group_alpha,
        local_alpha.detach().cpu()
    )


# -------------------------
# plotting core
# -------------------------
def draw_plot(
    Z_roi,
    Zg,
    roi_names_filt,
    roi_networks_filt,   # ★ roi_map 대신 이것
    y,
    K,
    out_path,
    title
):

    class_colors = {
        0: "#1f77b4",
        1: "#d62728"
    }

    y = np.asarray(y)

    S = len(y)

    step = np.tile(
        np.arange(K),
        S
    )

    y_rep = np.repeat(
        y,
        K
    )

    plt.figure(
        figsize=(8, 8)
    )

    ax = plt.gca()

    # =========================================================
    # Functional-system indices
    # =========================================================
    net_to_idx = {}

    for i, net in enumerate(roi_networks_filt):
        net_to_idx.setdefault(net, []).append(i)


    # =========================================================
    # Functional-system ellipses
    # =========================================================
    for net, idx in net_to_idx.items():

        if net in EXCLUDE_NETS:
            continue

        pts2d = Z_roi[idx]

        if len(pts2d) < 2:
            continue

        center = pts2d.mean(axis=0)

        cov = np.cov(
            pts2d.T
        )

        vals, vecs = np.linalg.eigh(
            cov
        )

        vals = np.clip(
            vals,
            1e-6,
            None
        )

        order = vals.argsort()[::-1]

        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(
            np.arctan2(
                *vecs[:, 0][::-1]
            )
        )

        width, height = (
            2 * np.sqrt(vals)
        )

        ell = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,

            # ★ functional system color
            facecolor=net_colors[net],
            edgecolor=net_colors[net],

            alpha=0.20,
            linewidth=1.0,
            zorder=1
        )

        ax.add_patch(ell)


    # =========================================================
    # ROI points
    # =========================================================
    for i, (name, net) in enumerate(
        zip(
            roi_names_filt,
            roi_networks_filt
        )
    ):

        x = Z_roi[i, 0]
        y_ = Z_roi[i, 1]

        # ROI point
        ax.scatter(
            x,
            y_,
            s=25,
            color=net_colors.get(
                net,
                "#7f7f7f"
            ),
            alpha=0.85,
            zorder=3
        )

        # ROI label
        ax.text(
            x,
            y_,
            name,
            fontsize=5,
            color=net_colors.get(
                net,
                "#7f7f7f"
            ),
            alpha=0.85,
            zorder=4
        )

    # =========================================================
    # Network centroids in PCA space
    # =========================================================
    for net in ["Sensory", "Intermediate", "DMN"]:

        idx = [
            i
            for i, n in enumerate(roi_networks_filt)
            if n == net
        ]

        if len(idx) == 0:
            continue

        centroid = Z_roi[idx].mean(axis=0)

        # centroid marker
        ax.scatter(
            centroid[0],
            centroid[1],
            s=180,
            marker="X",
            color=net_colors[net],
            edgecolor="black",
            linewidth=1.2,
            zorder=10
        )

        # centroid label
        ax.text(
            centroid[0],
            centroid[1],
            "  " + net_labels[net],
            fontsize=9,
            fontweight="bold",
            color=net_colors[net],
            va="center",
            zorder=11
        )

    # =========================================================
    # Class trajectories
    # =========================================================
    for lab in np.unique(y):

        centroids = []

        for k in range(K):

            m = (
                (y_rep == lab)
                &
                (step == k)
            )

            if m.sum() == 0:
                continue

            centroids.append(
                Zg[m].mean(axis=0)
            )

        if len(centroids) < 2:
            continue

        centroids = np.stack(
            centroids
        )

        # trajectory line
        ax.plot(
            centroids[:, 0],
            centroids[:, 1],
            color=class_colors[int(lab)],
            lw=2.2,
            zorder=6
        )

        # arrows
        for i in range(
            len(centroids) - 1
        ):

            ax.annotate(
                "",
                xy=centroids[i + 1],
                xytext=centroids[i],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=class_colors[int(lab)],
                    lw=1.5
                ),
                zorder=7
            )


    # =========================================================
    # Legend
    # =========================================================
    ax.legend(
        handles=build_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False
    )

    ax.set_title(
        title
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        out_path
    )
    
# -------------------------
# core processing
# -------------------------
def process_aux(
    aux_path,
    roi_names,
    roi_map
):

    print(
        f"\nProcessing: {aux_path}"
    )

    aux = torch.load(
        aux_path,
        map_location="cpu",
        weights_only=False
    )

    if "val_g" not in aux:

        print(
            "No validation data, skipping."
        )

        return

    # ========================================================
    # 1. seed weights 먼저 가져오기
    # ========================================================
    seed_weights = aux["seed_weights"].detach().cpu().float()

    # 혹시 normalize 안 되어 있을 경우
    seed_weights = seed_weights / seed_weights.sum()

    print("seed_weights:", seed_weights)
    print("sum:", seed_weights.sum().item())

    # -------------------------
    # ROI embedding
    # -------------------------
    H_train = aux["train_H"]
    H_val = aux["val_H"]

    H = torch.cat(
        [
            H_train,
            H_val
        ],
        dim=0
    )

    # ========================================================
    # 3. H도 동일한 seed weights로 aggregation
    # ========================================================
    w_H = seed_weights.view(
        1, 1, -1, 1, 1
    )

    H_weighted = (
        H * w_H
    ).sum(dim=2)
    # (S, K, N, d)

    # subject mean
    H_weighted = H_weighted.mean(dim=0)
    # (K, N, d)

    # hop mean -> fixed ROI reference
    roi_vecs = H_weighted.mean(dim=0)
    # (N, d)

    # ============================================================
    # Bilateral ROI averaging
    # visualization only
    # ============================================================

    (
        roi_vecs_bilateral,
        roi_names_bilateral,
        roi_networks_bilateral
    ) = bilateral_average_roi(
        roi_vecs,
        roi_names,
        roi_map
    )

    print(
        "Original ROI:",
        roi_vecs.shape
    )

    print(
        "Bilateral ROI:",
        roi_vecs_bilateral.shape
    )

    roi_vecs = roi_vecs.numpy()


    # -------------------------
    # selected seeds
    # -------------------------
    seed_indices = (
        aux["seed_indices"]
        .cpu()
        .numpy()
    )

    seed_weights = (
        aux["seed_weights"]
        .cpu()
        .float()
    )

    # safety normalization
    seed_weights = (
        seed_weights
        /
        seed_weights.sum()
    )

    seed_names = [
        roi_names[i]
        for i in seed_indices
    ]

    print(
        "\nSelected seeds:"
    )

    for i, (
        idx,
        name,
        w
    ) in enumerate(
        zip(
            seed_indices,
            seed_names,
            seed_weights
        )
    ):

        print(
            f"{i+1:2d}. "
            f"ROI {idx:3d} | "
            f"{name:25s} | "
            f"alpha={w.item():.4f}"
        )


    # ============================================================
    # Filter bilateral ROIs by functional system
    # ============================================================

    filtered_idx = [
        i
        for i, net in enumerate(roi_networks_bilateral)
        if net not in EXCLUDE_NETS
    ]

    roi_vecs_filt = roi_vecs_bilateral[
        filtered_idx
    ]

    roi_names_filt = [
        roi_names_bilateral[i]
        for i in filtered_idx
    ]

    roi_networks_filt = [
        roi_networks_bilateral[i]
        for i in filtered_idx
    ]


    # ========================================================
    # helper:
    # weighted Top-M seed trajectory
    # ========================================================
    def get_Z_seedmean(
        g,
        y,
        seed_weights
    ):
        """
        g:
            (S, K, M, d)

        seed_weights:
            (M,)

        weighted trajectory:
            g_bar^(k)
            = sum_s alpha_s * g_s^(k)
        """

        w = torch.as_tensor(
            seed_weights,
            dtype=g.dtype,
            device=g.device
        )

        w = w / w.sum()

        # -------------------------
        # weighted aggregation
        # across selected seeds
        # -------------------------
        g_mean = (
            g
            *
            w.view(
                1,
                1,
                -1,
                1
            )
        ).sum(
            dim=2
        )

        # g_mean:
        # (S,K,d)

        S, K, d = (
            g_mean.shape
        )

        g_flat = (
            g_mean
            .reshape(
                S * K,
                d
            )
            .cpu()
            .numpy()
        )

        # X = np.concatenate(
        #     [
        #         roi_vecs_filt,
        #         g_flat
        #     ],
        #     axis=0
        # )

        # # Z = umap.UMAP(
        # #     n_neighbors=min(
        # #         30,
        # #         X.shape[0] - 1
        # #     ),
        # #     min_dist=0.1,
        # #     metric="cosine",
        # #     random_state=42,
        # #     n_jobs=1
        # # ).fit_transform(
        # #     X
        # # )

        # from sklearn.decomposition import PCA

        # pca = PCA(n_components=2)
        # Z = pca.fit_transform(X)

        # Z_roi = Z[
        #     :len(roi_vecs_filt)
        # ]

        # Zg = Z[
        #     len(roi_vecs_filt):
        # ]

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)

        # ROI geometry로 PCA 축 정의
        Z_roi = pca.fit_transform(
            roi_vecs_filt
        )

        # 같은 PCA space로 trajectory projection
        Zg = pca.transform(
            g_flat
        )

        print(
            "Explained variance:",
            pca.explained_variance_ratio_,
            "sum:",
            pca.explained_variance_ratio_.sum()
        )

        return (
            Z_roi,
            Zg,
            K
        )


    # ========================================================
    # helper:
    # already alpha-weighted bilateral trajectory
    # ========================================================
    def get_Z_group(g_group):
        """
        g_group: (S, K, d)
        """

        S, K, d = g_group.shape

        g_flat = (
            g_group
            .reshape(S * K, d)
            .detach()
            .cpu()
            .numpy()
        )

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)

        # ROI reference defines the PCA basis
        Z_roi = pca.fit_transform(
            roi_vecs_filt
        )

        # project alpha-weighted trajectory to the same PCA space
        Zg = pca.transform(
            g_flat
        )

        print(
            "Explained variance:",
            pca.explained_variance_ratio_,
            "sum:",
            pca.explained_variance_ratio_.sum()
        )

        return Z_roi, Zg, K


    # ========================================================
    # helper:
    # single seed trajectory
    # ========================================================
    def get_Z_seed(
        g,
        y,
        seed
    ):

        g_seed = g[
            :,
            :,
            seed,
            :
        ]

        S, K, d = (
            g_seed.shape
        )

        g_flat = (
            g_seed
            .reshape(
                S * K,
                d
            )
            .cpu()
            .numpy()
        )

        # X = np.concatenate(
        #     [
        #         roi_vecs_filt,
        #         g_flat
        #     ],
        #     axis=0
        # )

        # # Z = umap.UMAP(
        # #     n_neighbors=min(
        # #         30,
        # #         X.shape[0] - 1
        # #     ),
        # #     min_dist=0.1,
        # #     metric="cosine",
        # #     random_state=42,
        # #     n_jobs=1
        # # ).fit_transform(
        # #     X
        # # )

        # from sklearn.decomposition import PCA

        # pca = PCA(n_components=2)
        # Z = pca.fit_transform(X)

        # Z_roi = Z[
        #     :len(roi_vecs_filt)
        # ]

        # Zg = Z[
        #     len(roi_vecs_filt):
        # ]

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)

        # ROI geometry로 PCA 축 정의
        Z_roi = pca.fit_transform(
            roi_vecs_filt
        )

        # 같은 PCA space로 trajectory projection
        Zg = pca.transform(
            g_flat
        )

        print(
            "Explained variance:",
            pca.explained_variance_ratio_,
            "sum:",
            pca.explained_variance_ratio_.sum()
        )

        return (
            Z_roi,
            Zg,
            K
        )


    base, _ = os.path.splitext(
        aux_path
    )


    # ========================================================
    # 1. weighted seed-mean trajectory
    # ========================================================

    # -------------------------
    # Train
    # -------------------------
    Z_roi, Zg, K = get_Z_seedmean(
        aux["train_g"],
        aux["train_y"],
        seed_weights
    )

    draw_plot(
        Z_roi,
        Zg,
        roi_names_filt,
        roi_networks_filt,
        aux["train_y"],
        K,
        base + "_train_weightedSeedMean.png",
        "Train (weighted Top-M seed trajectory)"
    )


    # -------------------------
    # Validation
    # -------------------------
    Z_roi, Zg, K = get_Z_seedmean(
        aux["val_g"],
        aux["val_y"],
        seed_weights
    )

    draw_plot(
        Z_roi,
        Zg,
        roi_names_filt,
        roi_networks_filt,
        aux["val_y"],
        K,
        base + "_val_weightedSeedMean.png",
        "Val (weighted Top-M seed trajectory)"
    )


    # -------------------------
    # Train + Val
    # -------------------------
    g_cat = torch.cat(
        [
            aux["train_g"],
            aux["val_g"]
        ],
        dim=0
    )

    y_cat = np.concatenate(
        [
            aux["train_y"],
            aux["val_y"]
        ]
    )

    # ============================================================
    # Trajectory-to-network distance
    # Train + Val / ORIGINAL latent space
    #
    # g_cat: (S, K, M, d)
    # roi_vecs_filt: bilateral-averaged + filtered ROI references
    # roi_networks_filt: corresponding hierarchy labels
    # ============================================================

    # same alpha used by the model
    w_dist = seed_weights.detach().cpu().float()
    w_dist = w_dist / w_dist.sum()

    g_cat_cpu = g_cat.detach().cpu()

    # (S, K, M, d) -> (S, K, d)
    g_weighted_dist = (
        g_cat_cpu
        * w_dist.view(1, 1, -1, 1)
    ).sum(dim=2).numpy()

    y_dist = np.asarray(y_cat)

    # network centroids in the ORIGINAL d-dimensional latent space
    latent_net_centroids = {}

    for net in [
        "Sensory",
        "Intermediate",
        "DMN"
    ]:
        idx = [
            i
            for i, n in enumerate(roi_networks_filt)
            if n == net
        ]

        if len(idx) == 0:
            continue

        latent_net_centroids[net] = (
            roi_vecs_filt[idx].mean(axis=0)
        )

    print("\n==============================================")
    print("Trajectory-to-network distance")
    print("Train + Val / original latent space")
    print("==============================================")

    for lab in np.unique(y_dist):

        class_mask = (y_dist == lab)
        g_class = g_weighted_dist[class_mask]

        print(f"\nClass {int(lab)}")

        for k in range(g_class.shape[1]):

            traj_centroid = g_class[:, k, :].mean(axis=0)

            values = []

            for net in [
                "Sensory",
                "Intermediate",
                "DMN"
            ]:
                if net not in latent_net_centroids:
                    continue

                net_centroid = latent_net_centroids[net]

                # Euclidean distance
                dist = np.linalg.norm(
                    traj_centroid - net_centroid
                )

                values.append(
                    f"{net}={dist:.4f}"
                )

            print(
                f"Hop {k}: "
                + " | ".join(values)
            )

    # ============================================================
    # PCA visualization
    # ============================================================
    Z_roi, Zg, K = get_Z_seedmean(
        g_cat,
        y_cat,
        seed_weights
    )

    draw_plot(
        Z_roi,
        Zg,
        roi_names_filt,
        roi_networks_filt,
        y_cat,
        K,
        base + "_trainval_weightedSeedMean.png",
        "Train+Val (weighted Top-M seed trajectory)"
    )


    # ========================================================
    # 2. Bilateral alpha-weighted seed trajectories
    #
    # Example:
    # Cuneus_L + Cuneus_R
    # -> (alpha_L * g_L + alpha_R * g_R) / (alpha_L + alpha_R)
    #
    # If only one side is selected, its trajectory is unchanged.
    # ========================================================

    bilateral_groups = get_bilateral_seed_groups(
        seed_indices,
        roi_names
    )

    print("\n==============================================")
    print("Bilateral alpha-weighted seed trajectories")
    print("==============================================")

    for group_name, seed_positions in bilateral_groups.items():

        g_group, group_alpha, local_alpha = (
            alpha_weighted_bilateral_trajectory(
                g_cat,
                seed_positions,
                seed_weights
            )
        )

        member_names = [
            roi_names[int(seed_indices[pos])]
            for pos in seed_positions
        ]

        member_str = ", ".join(
            [
                f"{name}: {local_alpha[i].item():.4f}"
                for i, name in enumerate(member_names)
            ]
        )

        print(
            f"{group_name:25s} | "
            f"group alpha={group_alpha:.4f} | "
            f"{member_str}"
        )

        Z_roi, Zg, K = get_Z_group(
            g_group
        )

        safe_name = (
            group_name
            .replace("/", "_")
            .replace(" ", "_")
        )

        draw_plot(
            Z_roi,
            Zg,
            roi_names_filt,
            roi_networks_filt,
            y_cat,
            K,
            base
            + f"_bilateralAlphaWeighted_{safe_name}_trainval.png",
            f"Train+Val | {group_name} "
            f"(alpha-weighted bilateral trajectory)"
        )




# -------------------------
# run
# -------------------------
def run_all(
    root_dir,
    roi_names,
    roi_map
):

    exp_dirs = sorted(
        glob.glob(
            os.path.join(
                root_dir,
                "*"
            )
        )
    )

    for exp in exp_dirs:

        if not os.path.isdir(exp):
            continue

        # 원하는 experiment만 돌리려면 사용
        if "w0_hitranet_20260825_181242_144161" not in exp:
            continue

        aux_files = sorted(
            glob.glob(
                os.path.join(
                    exp,
                    "fold*_best_aux.pt"
                )
            )
        )

        for aux_path in aux_files:

            try:

                process_aux(
                    aux_path,
                    roi_names,
                    roi_map
                )

            except Exception as e:

                print(
                    "Error:",
                    e
                )


# -------------------------
# main
# -------------------------
if __name__ == "__main__":

    ROOT_DIR = "./experiments"

    txt_path = (
        "aal_network_labels2.txt"
    )

    roi_map = (
        load_roi_network_map(
            txt_path
        )
    )

    with open(txt_path) as f:

        roi_names = (
            parse_roi_names(
                f.readlines()
            )
        )

    run_all(
        ROOT_DIR,
        roi_names,
        roi_map
    )