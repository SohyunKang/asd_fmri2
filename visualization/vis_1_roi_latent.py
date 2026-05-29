import numpy as np
import umap
import torch

def build_roi_umap_from_aux(aux):

    H = aux["H"]  # list of (B,M,N,d)

    roi_vecs = []

    for X in H:

        X = X.cpu().numpy()

        # batch mean
        X = X.mean(axis=0)   # (M,N,d)
        # seed mean
        X = X.mean(axis=0)   # (N,d)

        roi_vecs.append(X)

    X_umap = np.array(roi_vecs).mean(axis=0)   # (K,N,d)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, X_umap.shape[0] - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )

    reducer.fit(X_umap)
    print(X_umap.shape)

    return reducer, X_umap

import matplotlib.pyplot as plt


def plot_roi_latent_space(roi_vecs, reducer, roi_names, save_path):
    N = 116
    Z = reducer.transform(roi_vecs)

    colors = plt.cm.tab20(np.linspace(0,1,N))

    plt.figure(figsize=(9,9))

   
    for i in range(N):

        plt.scatter(
            Z[i,0],
            Z[i,1],
            color=colors[i],
            s=50,
            alpha=0.7
        )

    for i,name in enumerate(roi_names):

        plt.text(
            Z[i,0],
            Z[i,1],
            name,
            fontsize=6
        )

    plt.title("ROI latent manifold")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


aux = torch.load("experiments/w0_hitranet_20260312_140927_978430/fold1_best_aux.pt", map_location="cpu")

with open("aal_network_labels.txt") as f:
    roi_names = [line.strip() for line in f]

reducer, roi_vecs = build_roi_umap_from_aux(aux)

plot_roi_latent_space(
    roi_vecs,
    reducer,
    roi_names,
    "roi_latent_space.png"
)