import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import os
import glob


# -------------------------
# config
# -------------------------
net_order = [
    "Sensory",
    "DANnSalience",
    "Frontoparietal",
    "DMN",
    "Limbic",
    "Other"
]

net_colors = {
    "Sensory": "#1f77b4",
    "DANnSalience": "#9467bd",
    "Frontoparietal": "#2ca02c",
    "DMN": "#d62728",
    "Limbic": "#17becf",
    "Other": "#7f7f7f"
}

EXCLUDE_NETS = {"Limbic", "Other"}


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
    return [line.strip().split()[0] for line in lines if line.strip()]


def build_legend():
    class_colors = {0: "#1f77b4", 1: "#d62728"}

    legend_elements = []

    for net in net_order:
        if net in EXCLUDE_NETS:
            continue

        legend_elements.append(
            Line2D([0],[0], marker='o', color='w',
                   label=net,
                   markerfacecolor=net_colors[net],
                   markersize=8)
        )

    legend_elements.append(Line2D([0],[0], linestyle='None', label=''))

    for cls, c in class_colors.items():
        legend_elements.append(
            Line2D([0],[0], marker='o', color='w',
                   label=f"Class {cls}",
                   markerfacecolor=c,
                   markersize=6)
        )

    legend_elements.append(
        Line2D([0],[0], color='black', lw=2, marker='>', label='Trajectory')
    )

    return legend_elements


def build_network_indices(roi_names, roi_map):
    net_to_idx = {}
    for i, name in enumerate(roi_names):
        net = roi_map.get(name, "Other")
        net_to_idx.setdefault(net, []).append(i)
    return net_to_idx


# -------------------------
# plotting core
# -------------------------
def draw_plot(Z_roi, Zg, roi_names_filt, roi_map, y, K, out_path, title):

    class_colors = {0: "#1f77b4", 1: "#d62728"}

    S = len(y)
    step = np.tile(np.arange(K), S)
    y_rep = np.repeat(y, K)

    plt.figure(figsize=(8, 8))

    net_to_idx = build_network_indices(roi_names_filt, roi_map)

    # ellipse
    for net, idx in net_to_idx.items():
        if net in EXCLUDE_NETS:
            continue

        pts2d = Z_roi[idx]
        center = pts2d.mean(axis=0)
        cov = np.cov(pts2d.T)

        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-6, None)

        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 2*np.sqrt(vals)

        ell = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=net_colors[net],
            edgecolor=net_colors[net],
            alpha=0.2
        )
        plt.gca().add_patch(ell)

    # -------------------------
    # seed 이름 추출 (title 기반)
    # -------------------------
    seed_name = None
    if "|" in title:
        seed_name = title.split("|")[-1].strip().replace(")", "")

    # ROI points + text
    for i, name in enumerate(roi_names_filt):
        net = roi_map.get(name, "Other")

        x, y_ = Z_roi[i,0], Z_roi[i,1]

        is_seed = (seed_name is not None) and (name == seed_name)

        if is_seed:
            plt.scatter(
                x, y_,
                s=100,
                color="gold",
                edgecolors="black",
                marker="*",
                zorder=10
            )
        else:
            plt.scatter(
                x, y_,
                s=20,
                color=net_colors.get(net, "gray"),
                alpha=0.8,
                zorder=3
            )

        # 🔥 텍스트 추가
        plt.text(
            x, y_,
            name,
            fontsize=5,
            color=net_colors.get(net, "gray"),
            alpha=0.8,
            zorder=4
        )

    # trajectory
    for lab in np.unique(y):
        centroids = []

        for k in range(K):
            m = (y_rep == lab) & (step == k)
            if m.sum() == 0:
                continue
            centroids.append(Zg[m].mean(axis=0))

        if len(centroids) < 2:
            continue

        centroids = np.stack(centroids)

        plt.plot(centroids[:,0], centroids[:,1],
                 color=class_colors[int(lab)], lw=2)

        for i in range(len(centroids)-1):
            plt.annotate(
                "",
                xy=centroids[i+1],
                xytext=centroids[i],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=class_colors[int(lab)]
                )
            )

    plt.legend(handles=build_legend(),
               loc="upper center",
               bbox_to_anchor=(0.5, -0.08),
               ncol=4,
               frameon=False)

    plt.title(title)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


# -------------------------
# 핵심 처리
# -------------------------
def process_aux(aux_path, roi_names, roi_map):

    print(f"\n📂 Processing: {aux_path}")

    aux = torch.load(aux_path, map_location="cpu", weights_only=False)

    if "val_g" not in aux:
        print("⚠️ No validation data, skipping val plots")
        return

    # -------------------------
    # ROI embedding
    # -------------------------
    H_train = aux["train_H"]
    H_val = aux["val_H"]
    H = torch.tensor(np.concatenate([H_train, H_val], axis=0))
    H = H.mean(dim=0).mean(dim=1)
    roi_vecs = H.mean(dim=0).numpy()

    seed_indices = aux["seed_indices"].numpy()
    seed_names = [roi_names[i] for i in seed_indices]

    filtered_idx = [
        i for i, name in enumerate(roi_names)
        if (roi_map.get(name, "Other") not in EXCLUDE_NETS)
        or (name in seed_names)   # 🔥 추가
    ]

    roi_vecs_filt = roi_vecs[filtered_idx]
    roi_names_filt = [roi_names[i] for i in filtered_idx]

    # -------------------------
    # helper (seed mean)
    # -------------------------
    def get_Z_seedmean(g, y):
        g_mean = g.mean(dim=2)
        S, K, d = g_mean.shape

        g_flat = g_mean.reshape(S*K, d).cpu().numpy()
        X = np.concatenate([roi_vecs_filt, g_flat], axis=0)

        Z = umap.UMAP(
            n_neighbors=min(30, X.shape[0]-1),
            min_dist=0.1,
            metric="cosine",
            random_state=42
        ).fit_transform(X)

        Z_roi = Z[:len(roi_vecs_filt)]
        Zg = Z[len(roi_vecs_filt):]

        return Z_roi, Zg, K

    # -------------------------
    # helper (single seed)
    # -------------------------
    def get_Z_seed(g, y, seed):
        g_seed = g[:,:,seed,:]
        S, K, d = g_seed.shape

        g_flat = g_seed.reshape(S*K, d).cpu().numpy()
        X = np.concatenate([roi_vecs_filt, g_flat], axis=0)

        Z = umap.UMAP(
            n_neighbors=min(30, X.shape[0]-1),
            min_dist=0.1,
            metric="cosine",
            random_state=42
        ).fit_transform(X)

        Z_roi = Z[:len(roi_vecs_filt)]
        Zg = Z[len(roi_vecs_filt):]

        return Z_roi, Zg, K

    base, _ = os.path.splitext(aux_path)

    # -------------------------
    # 🔥 1. seed mean
    # -------------------------

    # Train
    Z_roi, Zg, K = get_Z_seedmean(aux["train_g"], aux["train_y"])
    draw_plot(
        Z_roi, Zg, roi_names_filt, roi_map,
        aux["train_y"], K,
        base + "_train.png",
        f"Train (seed-mean)"
    )

    # Val
    Z_roi, Zg, K = get_Z_seedmean(aux["val_g"], aux["val_y"])
    draw_plot(
        Z_roi, Zg, roi_names_filt, roi_map,
        aux["val_y"], K,
        base + "_val.png",
        f"Val (seed-mean)"
    )

    # Train+Val
    g_cat = torch.cat([aux["train_g"], aux["val_g"]], dim=0)
    y_cat = np.concatenate([aux["train_y"], aux["val_y"]])

    Z_roi, Zg, K = get_Z_seedmean(g_cat, y_cat)
    draw_plot(
        Z_roi, Zg, roi_names_filt, roi_map,
        y_cat, K,
        base + "_trainval.png",
        f"Train+Val (seed-mean)"
    )

    # -------------------------
    # 🔥 2. seed별 (train+val만)
    # -------------------------
    _, _, M, _ = aux["train_g"].shape

    for seed in range(M):

        roi_idx = seed_indices[seed]
        seed_name = roi_names[roi_idx]

        print(f"  → Seed {seed}: {seed_name}")

        g_cat = torch.cat([aux["train_g"], aux["val_g"]], dim=0)
        y_cat = np.concatenate([aux["train_y"], aux["val_y"]])

        Z_roi, Zg, K = get_Z_seed(g_cat, y_cat, seed)

        safe_name = seed_name.replace("/", "_")

        draw_plot(
            Z_roi, Zg, roi_names_filt, roi_map,
            y_cat, K,
            base + f"_seed{seed}_{safe_name}_trainval.png",
            f"Train+Val ({os.path.basename(aux_path)} | {seed_name})"
        )

# -------------------------
# run
# -------------------------
def run_all(root_dir, roi_names, roi_map):

    exp_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))

    for exp in exp_dirs:
        if not os.path.isdir(exp):
            continue
        if "w0_hitranet_20260331" not in exp:
            continue

        aux_files = glob.glob(os.path.join(exp, "fold*_best_aux.pt"))

        for aux_path in aux_files:
            try:
                process_aux(aux_path, roi_names, roi_map)
            except Exception as e:
                print("❌ Error:", e)


# -------------------------
# main
# -------------------------
if __name__ == "__main__":

    ROOT_DIR = "/data5/sohyunkang/experiments"
    txt_path = "aal_network_labels2.txt"

    roi_map = load_roi_network_map(txt_path)

    with open(txt_path) as f:
        roi_names = parse_roi_names(f.readlines())

    run_all(ROOT_DIR, roi_names, roi_map)