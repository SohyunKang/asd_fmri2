import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


# -------------------------
# config
# -------------------------
net_order = [
    "Somatomotor",
    "Visual",
    "Salience",
    "Frontoparietal",
    "DMN",
    "Limbic",
    "Other"
]

net_colors = {
    "Somatomotor": "#1f77b4",
    "Visual": "#17becf",
    "Salience": "#ff7f0e",
    "Frontoparietal": "#2ca02c",
    "DMN": "#d62728",
    "Limbic": "#9467bd",
    "Other": "#7f7f7f"
}

# 🔥 UMAP에서 제외할 네트워크
EXCLUDE_NETS = {"Limbic", "Salience", "Other"}


# -------------------------
# ROI → Network
# -------------------------
def load_roi_network_map(txt_path):
    mapping = {}
    with open(txt_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            roi = parts[0]
            net = parts[1]
            mapping[roi] = net
    return mapping


def parse_roi_names(lines):
    return [line.strip().split()[0] for line in lines if line.strip()]


# -------------------------
# legend
# -------------------------
def build_legend():
    class_colors = {0:"#1f77b4", 1:"#d62728"}

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
        Line2D([0],[0], color='black', lw=2, label='Trajectory')
    )

    return legend_elements


# -------------------------
# network points
# -------------------------
def build_network_points(roi_vecs, roi_names, roi_map):
    net_dict = {}
    for i, name in enumerate(roi_names):
        net = roi_map.get(name, "Other")
        net_dict.setdefault(net, []).append(roi_vecs[i])
    return {k: np.stack(v) for k,v in net_dict.items()}


# -------------------------
# plot
# -------------------------
def plot_per_seed(aux, roi_names, roi_map, prefix):

    H = aux["train_H"]
    g = aux["train_g"]
    y = np.asarray(aux["train_y"])
    seed_indices = aux["seed_indices"].numpy()

    # ROI embedding
    H = H.mean(dim=0).mean(dim=1)
    roi_vecs = H.mean(dim=0).numpy()

    S, K, M, d = g.shape
    print(g.shape)
    class_colors = {0:"#1f77b4", 1:"#d62728"}

    for seed in range(M):

        seed_roi_name = roi_names[seed_indices[seed]]

        g_seed = g[:,:,seed,:]
        g_flat = g_seed.reshape(S*K, d).cpu().numpy()

        step = np.tile(np.arange(1, K+1), S)
        y_rep = np.repeat(y, K)

        # -------------------------
        # 🔥 ROI 필터링 (UMAP 전에)
        # -------------------------
        filtered_idx = [
            i for i, name in enumerate(roi_names)
            if roi_map.get(name, "Other") not in EXCLUDE_NETS
        ]

        roi_vecs_filt = roi_vecs[filtered_idx]
        roi_names_filt = [roi_names[i] for i in filtered_idx]

        # -------------------------
        # UMAP
        # -------------------------
        X = np.concatenate([roi_vecs_filt, g_flat], axis=0)

        reducer = umap.UMAP(
            n_neighbors=min(30, X.shape[0]-1),
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )

        Z = reducer.fit_transform(X)

        Z_roi = Z[:len(roi_vecs_filt)]
        Zg = Z[len(roi_vecs_filt):]

        # -------------------------
        # plot
        # -------------------------
        plt.figure(figsize=(8,8))

        net_points = build_network_points(roi_vecs_filt, roi_names_filt, roi_map)

        # ellipse
        for net, pts in net_points.items():

            if net in EXCLUDE_NETS:
                continue

            if pts.shape[0] < 3:
                continue

            idx = [i for i,n in enumerate(roi_names_filt) if roi_map.get(n) == net]
            pts2d = Z_roi[idx]

            center = pts2d.mean(axis=0)
            cov = np.cov(pts2d.T)

            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            width, height = 2 * np.sqrt(vals)

            ell = Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                facecolor=net_colors[net],
                alpha=0.2,
                edgecolor=net_colors[net],
                lw=2
            )

            plt.gca().add_patch(ell)

        # network center
        for net, pts in net_points.items():

            if net in EXCLUDE_NETS:
                continue

            idx = [i for i,n in enumerate(roi_names_filt) if roi_map.get(n) == net]
            pts2d = Z_roi[idx]
            center = pts2d.mean(axis=0)

            plt.scatter(
                center[0], center[1],
                s=120,
                color=net_colors[net],
                edgecolors="black",
                zorder=10
            )

        # ROI scatter
        for i, name in enumerate(roi_names_filt):
            net = roi_map.get(name, "Other")

            plt.scatter(
                Z_roi[i,0],
                Z_roi[i,1],
                s=20,
                color=net_colors.get(net, "gray"),
                alpha=0.8
            )

        # ROI labels
        for i, name in enumerate(roi_names_filt):
            net = roi_map.get(name, "Other")

            plt.text(
                Z_roi[i,0],
                Z_roi[i,1],
                name,
                fontsize=5,
                color=net_colors.get(net, "gray"),
                alpha=0.8
            )

        # trajectory
        for lab in np.unique(y):

            centroids = []

            for k in range(1, K+1):
                m = (y_rep == lab) & (step == k)
                if m.sum()==0:
                    continue
                centroids.append(Zg[m].mean(axis=0))

            if len(centroids) < 2:
                continue

            centroids = np.stack(centroids)

            plt.plot(
                centroids[:,0], centroids[:,1],
                lw=2,
                color=class_colors[int(lab)]
            )

            plt.annotate(
                "",
                xy=centroids[-1],
                xytext=centroids[-2],
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=2,
                    color=class_colors[int(lab)]
                )
            )

        plt.legend(
            handles=build_legend(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=4,
            frameon=False,
            fontsize=8
        )

        plt.title(f"Seed {seed} ({seed_roi_name})")

        save_path = f"{prefix}_seed{seed}_{seed_roi_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", save_path)


# -------------------------
# 실행
# -------------------------
aux = torch.load(
    "experiments/w0_hitranet_20260319_113654_357915/fold3_best_aux.pt",
    map_location="cpu"
)

txt_path = "aal_network_labels2.txt"

roi_map = load_roi_network_map(txt_path)

with open(txt_path) as f:
    roi_names = parse_roi_names(f.readlines())

plot_per_seed(aux, roi_names, roi_map, "traj")