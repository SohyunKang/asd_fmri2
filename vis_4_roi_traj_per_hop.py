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
            roi, net = line.strip().split()
            mapping[roi] = net
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


# -------------------------
# ROI hop 풀기
# -------------------------
def build_roi_per_hop(H, roi_names, filtered_idx):
    # H: (K, N, d)

    K, N, d = H.shape

    vecs = []
    names = []
    hops = []

    for k in range(K):
        for i in filtered_idx:
            vecs.append(H[k, i].numpy())
            names.append(f"{roi_names[i]}_h{k}")
            hops.append(k)

    return np.stack(vecs), names, np.array(hops)


# -------------------------
# plotting
# -------------------------
def draw_plot(Z_roi, Zg, roi_names_hop, roi_hops, roi_map, y, K, out_path, title):

    class_colors = {0: "#1f77b4", 1: "#d62728"}

    S = len(y)
    step = np.tile(np.arange(K), S)
    y_rep = np.repeat(y, K)

    plt.figure(figsize=(8, 8))

    # -------------------------
    # ROI scatter (hop 색)
    # -------------------------
    plt.scatter(
        Z_roi[:,0],
        Z_roi[:,1],
        c=roi_hops,
        cmap="viridis",
        s=20,
        zorder=3
    )

    # 텍스트
    for i, name in enumerate(roi_names_hop):
        x, y_ = Z_roi[i]
        plt.text(x, y_, name, fontsize=4, alpha=0.8)

    plt.colorbar(label="Hop")

    # -------------------------
    # trajectory
    # -------------------------
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

        plt.plot(
            centroids[:,0],
            centroids[:,1],
            color=class_colors[int(lab)],
            lw=2
        )

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

    # -------------------------
    # legend
    # -------------------------
    plt.legend(
        handles=build_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False
    )

    plt.title(title)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


# -------------------------
# main processing
# -------------------------
def process_aux(aux_path, roi_names, roi_map):

    print(f"\n📂 Processing: {aux_path}")

    aux = torch.load(aux_path, map_location="cpu")

    # -------------------------
    # ROI embedding
    # -------------------------
    H = aux["train_H"]   # (S,K,Seed,N,d)

    # 🔥 subject + seed 평균 → hop 유지
    H = H.mean(dim=0).mean(dim=1)   # (K,N,d)

    # ROI filter
    filtered_idx = [
        i for i, name in enumerate(roi_names)
        if roi_map.get(name, "Other") not in EXCLUDE_NETS
    ]

    roi_vecs, roi_names_hop, roi_hops = build_roi_per_hop(
        H, roi_names, filtered_idx
    )

    # -------------------------
    # g (trajectory)
    # -------------------------
    g = aux["train_g"]   # (S,K,Seed,d)

    g_mean = g.mean(dim=2)   # seed mean
    S, K, d = g_mean.shape

    g_flat = g_mean.reshape(S*K, d).numpy()

    # -------------------------
    # UMAP
    # -------------------------
    X = np.concatenate([roi_vecs, g_flat], axis=0)

    Z = umap.UMAP(
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42
    ).fit_transform(X)

    Z_roi = Z[:len(roi_vecs)]
    Zg = Z[len(roi_vecs):]

    base, _ = os.path.splitext(aux_path)

    draw_plot(
        Z_roi,
        Zg,
        roi_names_hop,
        roi_hops,
        roi_map,
        aux["train_y"],
        K,
        base + "_roiHop.png",
        "ROI per-hop + trajectory"
    )


# -------------------------
# run all
# -------------------------
def run_all(root_dir, roi_names, roi_map):

    exp_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))

    for exp in exp_dirs:
        # entire folders
        # if not os.path.isdir(exp):
        #     continue
        # specific folder
        # if "w0_hitranet_20260326" not in exp:
        #     continue
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

    ROOT_DIR = "./experiments"
    txt_path = "aal_network_labels2.txt"

    roi_map = load_roi_network_map(txt_path)

    with open(txt_path) as f:
        roi_names = parse_roi_names(f.readlines())

    run_all(ROOT_DIR, roi_names, roi_map)