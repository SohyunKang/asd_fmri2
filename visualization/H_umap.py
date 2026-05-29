import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import os


def save_umap_all_hops(aux, roi_names, save_path, n_hops):

    H = aux["H"]

    # hop별로 모으기
    hop_tensors = [[] for _ in range(n_hops)]

    for i, X in enumerate(H):

        hop_id = i % n_hops
        hop_tensors[hop_id].append(X)

    # batch concat
    hop_tensors = [
        torch.cat(hop_list, dim=0) for hop_list in hop_tensors
    ]

    K = n_hops
    N = len(roi_names)

    all_X = []

    for X in hop_tensors:

        # (B,M,N,d)
        X = X.cpu().numpy()

        # batch mean
        X = X.mean(axis=0)

        # seed mean
        X = X.mean(axis=0)

        all_X.append(X)

    all_X = np.stack(all_X)   # (K,N,d)

    X_umap = all_X.reshape(K*N, -1)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    embedding = reducer.fit_transform(X_umap)
    embedding = embedding.reshape(K, N, 2)

    colors = plt.cm.tab20(np.linspace(0,1,N))
    markers = ["o","s","^","D","v","P","X"]

    plt.figure(figsize=(10,10))

    for k in range(K):
        for i in range(N):

            plt.scatter(
                embedding[k,i,0],
                embedding[k,i,1],
                color=colors[i],
                marker=markers[k % len(markers)],
                s=60
            )

    plt.title("ROI Hierarchical Trajectory (All Hops)")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


aux = torch.load(
    "experiments/w0_hitranet_20260312_140927_978430/fold1_best_aux.pt",
    map_location="cpu"
)

with open("aal_network_labels.txt") as f:
    roi_names = [line.strip() for line in f]

save_umap_all_hops(
    aux,
    roi_names,
    "roi_umap_all_hops.png",
    n_hops=5
)