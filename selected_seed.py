import os
import glob
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from utils import load_roi_names_from_txt


# ============================================================
# SETTINGS
# ============================================================

EXP_DIR = "./experiments/w0_hitranet_20260825_181242_144161"

AAL_TXT = "./data/ROI_MNI_V4.txt"

# ROI-name / functional-network mapping
NETWORK_TXT = "./aal_network_labels2.txt"

TOP_K = 10
TEMPERATURE = 0.3

SAVE_NAME = "seed_ranking_5fold_mean_std.png"


# ============================================================
# Three-level cortical hierarchy
# Sensory -> Intermediate association/control -> DMN
# ============================================================

net_order = [
    "Sensory",
    "Intermediate",
    "DMN",
    "Other",
]

net_colors = {
    "Sensory": "#1f77b4",
    "Intermediate": "#9467bd",
    "DMN": "#d62728",
    "Other": "#7f7f7f",
}

net_labels = {
    "Sensory": "Sensory",
    "Intermediate": "Intermediate association/control",
    "DMN": "DMN",
    "Other": "Other",
}

# Original functional-system label -> hierarchy group
network_3group = {
    "Sensory": "Sensory",
    "DANnSalience": "Intermediate",
    "Frontoparietal": "Intermediate",
    "DMN": "DMN",
    "Limbic": "Other",
    "Other": "Other",
}

# Keep all groups in the seed plots, including Other.
EXCLUDE_NETS = set()


def to_hierarchy_group(net):
    """Collapse the original functional-system label to the hierarchy group."""
    return network_3group.get(net, "Other")


# ============================================================
# Utility
# ============================================================

def load_roi_network_map(txt_path):

    mapping = {}

    with open(txt_path) as f:

        for line in f:

            if not line.strip():
                continue

            roi, net = line.strip().split()

            mapping[roi] = net

    return mapping


def get_fold_number(path):

    name = os.path.basename(path)

    m = re.search(r"fold(\d+)_", name)

    return int(m.group(1)) if m else 999


# ============================================================
# 1. ROI names + functional network
# ============================================================

roi_names = load_roi_names_from_txt(AAL_TXT)

n_rois = len(roi_names)

roi_map = load_roi_network_map(NETWORK_TXT)

print(f"N ROIs: {n_rois}")

print("\nNetwork mapping example:")

for roi in roi_names[:5]:
    print(
        f"{roi:25s} -> "
        f"{roi_map.get(roi, 'Other')}"
    )


# ============================================================
# 2. Find fold best checkpoints
# ============================================================

model_paths = glob.glob(
    os.path.join(
        EXP_DIR,
        "fold*_best_model.pt"
    )
)

model_paths = sorted(
    model_paths,
    key=get_fold_number
)

if len(model_paths) == 0:

    raise FileNotFoundError(
        f"No fold*_best_model.pt found in:\n{EXP_DIR}"
    )


print(f"\nFound {len(model_paths)} checkpoints:")

for p in model_paths:
    print(" ", os.path.basename(p))


# ============================================================
# 3. Load raw u_per from each fold
# ============================================================

fold_seedness = []
fold_numbers = []

for path in model_paths:

    ckpt = torch.load(
        path,
        map_location="cpu"
    )

    state_dict = ckpt["model"]

    key = "periphery_seedness.u_per"

    if key not in state_dict:

        raise KeyError(
            f"{key} not found in {path}\n"
            f"Available keys:\n{list(state_dict.keys())}"
        )

    u_per = (
        state_dict[key]
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    if len(u_per) != n_rois:

        raise ValueError(
            f"{path}: "
            f"u_per length={len(u_per)}, "
            f"ROI names={n_rois}"
        )

    fold_seedness.append(u_per)

    fold_numbers.append(
        get_fold_number(path)
    )


fold_seedness = np.stack(
    fold_seedness,
    axis=0
)

# (n_folds, 116)
print(
    "\nfold_seedness shape:",
    fold_seedness.shape
)


# ============================================================
# 4. Mean RAW seedness across folds
#    -> common Top-K selection
# ============================================================

mean_raw = fold_seedness.mean(axis=0)
std_raw = fold_seedness.std(axis=0)

top_idx = np.argsort(
    mean_raw
)[::-1][:TOP_K]


print("\n==============================================")
print("Top-K based on mean RAW u_per across folds")
print("==============================================")


for rank, idx in enumerate(
    top_idx,
    start=1
):

    roi = roi_names[idx]

    network = to_hierarchy_group(roi_map.get(
        roi,
        "Other"
    ))

    print(
        f"{rank:2d}. "
        f"ROI {idx:3d} | "
        f"{roi:25s} | "
        f"{network:15s} | "
        f"raw={mean_raw[idx]:.6f} "
        f"± {std_raw[idx]:.6f}"
    )

# ============================================================
# 4-1. Fold-wise Top-K seed stability
# ============================================================

n_folds = fold_seedness.shape[0]

# ------------------------------------------------------------
# Top-K seeds independently selected in each fold
# ------------------------------------------------------------
fold_top_idx = []

for fold_i in range(n_folds):

    idx = np.argsort(
        fold_seedness[fold_i]
    )[::-1][:TOP_K]

    fold_top_idx.append(idx)

fold_top_idx = np.stack(
    fold_top_idx,
    axis=0
)
# shape: (n_folds, TOP_K)


print("\n==============================================")
print("Top-K seeds in each fold")
print("==============================================")

for f in range(n_folds):

    print(f"\nFold {fold_numbers[f]}")

    for rank, idx in enumerate(
        fold_top_idx[f],
        start=1
    ):

        roi = roi_names[idx]
        net = to_hierarchy_group(roi_map.get(roi, "Other"))

        print(
            f"{rank:2d}. "
            f"ROI {idx:3d} | "
            f"{roi:25s} | "
            f"{net}"
        )


# ============================================================
# 4-2. Pairwise Jaccard similarity
# ============================================================

jaccard = np.zeros(
    (n_folds, n_folds),
    dtype=float
)

overlap_count = np.zeros(
    (n_folds, n_folds),
    dtype=int
)


for i in range(n_folds):

    set_i = set(
        fold_top_idx[i].tolist()
    )

    for j in range(n_folds):

        set_j = set(
            fold_top_idx[j].tolist()
        )

        intersection = len(
            set_i & set_j
        )

        union = len(
            set_i | set_j
        )

        overlap_count[i, j] = intersection

        jaccard[i, j] = (
            intersection / union
            if union > 0
            else 0.0
        )


print("\n==============================================")
print("Pairwise Top-K overlap")
print("==============================================")

print(overlap_count)


print("\n==============================================")
print("Pairwise Jaccard similarity")
print("==============================================")

print(
    np.round(
        jaccard,
        3
    )
)


# ------------------------------------------------------------
# Mean pairwise similarity
# diagonal 제외
# ------------------------------------------------------------
upper_idx = np.triu_indices(
    n_folds,
    k=1
)

mean_jaccard = jaccard[
    upper_idx
].mean()

std_jaccard = jaccard[
    upper_idx
].std()

mean_overlap = overlap_count[
    upper_idx
].mean()

std_overlap = overlap_count[
    upper_idx
].std()


print(
    "\nMean pairwise Jaccard: "
    f"{mean_jaccard:.3f} ± {std_jaccard:.3f}"
)

print(
    "Mean Top-K overlap: "
    f"{mean_overlap:.2f} ± {std_overlap:.2f} "
    f"out of {TOP_K}"
)


# ============================================================
# 4-3. Jaccard heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.4, 4.8)
)

im = ax.imshow(
    jaccard,
    vmin=0,
    vmax=1,
    cmap="Blues"
)

ax.set_xticks(
    np.arange(n_folds)
)

ax.set_yticks(
    np.arange(n_folds)
)

ax.set_xticklabels(
    [
        f"Fold {f}"
        for f in fold_numbers
    ]
)

ax.set_yticklabels(
    [
        f"Fold {f}"
        for f in fold_numbers
    ]
)


# cell value
for i in range(n_folds):

    for j in range(n_folds):

        value = jaccard[i, j]

        text_color = (
            "white"
            if value >= 0.55
            else "black"
        )

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=10
        )


cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.set_label(
    "Jaccard similarity",
    fontsize=10
)

ax.set_xlabel("Fold")
ax.set_ylabel("Fold")

ax.set_title(
    f"Top-{TOP_K} seed stability",
    fontsize=12,
    fontweight="bold"
)

plt.tight_layout()


jaccard_path = os.path.join(
    EXP_DIR,
    f"seed_top{TOP_K}_jaccard_heatmap.png"
)

plt.savefig(
    jaccard_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved:",
    jaccard_path
)


# ============================================================
# 4-4. Seed selection frequency across folds
# ============================================================

selection_count = np.zeros(
    n_rois,
    dtype=int
)


for fold_i in range(n_folds):

    selection_count[
        fold_top_idx[fold_i]
    ] += 1


selection_frequency = (
    selection_count / n_folds
)


# Top-K에 한 번이라도 들어온 ROI만
selected_any = np.where(
    selection_count > 0
)[0]


# ------------------------------------------------------------
# sort:
# 1) number of folds selected
# 2) mean raw seed relevance
# ------------------------------------------------------------
selected_order = sorted(
    selected_any,
    key=lambda idx: (
        selection_count[idx],
        mean_raw[idx]
    ),
    reverse=True
)


print("\n==============================================")
print("Cross-fold seed selection frequency")
print("==============================================")

for rank, idx in enumerate(
    selected_order,
    start=1
):

    roi = roi_names[idx]

    net = to_hierarchy_group(roi_map.get(
        roi,
        "Other"
    ))

    print(
        f"{rank:2d}. "
        f"ROI {idx:3d} | "
        f"{roi:25s} | "
        f"{net:15s} | "
        f"{selection_count[idx]}/{n_folds} "
        f"({selection_frequency[idx] * 100:.0f}%)"
    )


# ============================================================
# 4-5. Selection frequency plot
# ============================================================

freq_idx = np.array(
    selected_order
)

freq_count = selection_count[
    freq_idx
]

freq_names = [
    roi_names[idx]
    for idx in freq_idx
]

freq_networks = [
    to_hierarchy_group(roi_map.get(
        roi_names[idx],
        "Other"
    ))
    for idx in freq_idx
]

freq_colors = [
    net_colors.get(
        net,
        net_colors["Other"]
    )
    for net in freq_networks
]


y_freq = np.arange(
    len(freq_idx)
)


fig, ax = plt.subplots(
    figsize=(
        7.2,
        max(
            5.0,
            len(freq_idx) * 0.35
        )
    )
)


ax.barh(
    y_freq,
    freq_count,
    height=0.62,
    color=freq_colors
)


ax.set_yticks(
    y_freq
)

ax.set_yticklabels(
    freq_names,
    fontsize=9
)

ax.invert_yaxis()


ax.set_xlim(
    0,
    n_folds + 0.5
)

ax.set_xticks(
    np.arange(
        0,
        n_folds + 1
    )
)


# ------------------------------------------------------------
# frequency text
# ------------------------------------------------------------
for yi, count in zip(
    y_freq,
    freq_count
):

    ax.text(
        count + 0.08,
        yi,
        f"{count}/{n_folds}",
        va="center",
        fontsize=9
    )


# ------------------------------------------------------------
# functional-system legend
# ------------------------------------------------------------
used_freq_nets = set(
    freq_networks
)

freq_legend_handles = []

for net in net_order:

    if net not in used_freq_nets:
        continue

    freq_legend_handles.append(
        Patch(
            facecolor=net_colors[net],
            edgecolor="none",
            label=net_labels[net]
        )
    )


ax.legend(
    handles=freq_legend_handles,
    title="Functional system",
    frameon=False,
    fontsize=8,
    title_fontsize=8,
    loc="lower right"
)


ax.set_xlabel(
    "Number of folds selected"
)

ax.set_title(
    f"Top-{TOP_K} seed selection frequency",
    loc="left",
    fontsize=12,
    fontweight="bold"
)


ax.xaxis.grid(
    True,
    linewidth=0.7,
    alpha=0.25
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.tick_params(
    axis="y",
    length=0
)

plt.tight_layout()


frequency_path = os.path.join(
    EXP_DIR,
    f"seed_top{TOP_K}_selection_frequency.png"
)

plt.savefig(
    frequency_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    frequency_path
)

# ============================================================
# 5. Same Top-K:
#    normalized alpha for each fold
#
# alpha = softmax(u_per / T)
# ============================================================

fold_top_weights = []


for fold_i in range(
    fold_seedness.shape[0]
):

    raw_top = torch.tensor(
        fold_seedness[
            fold_i,
            top_idx
        ],
        dtype=torch.float32
    )

    alpha = torch.softmax(
        raw_top / TEMPERATURE,
        dim=0
    )

    fold_top_weights.append(
        alpha.numpy()
    )


fold_top_weights = np.stack(
    fold_top_weights,
    axis=0
)

# (n_folds, TOP_K)
print(
    "\nfold_top_weights shape:",
    fold_top_weights.shape
)


# ============================================================
# 6. Mean ± SD normalized weights
# ============================================================

top_mean = fold_top_weights.mean(axis=0)
top_std = fold_top_weights.std(axis=0)


# ------------------------------------------------------------
# final display order
# ------------------------------------------------------------

display_order = np.argsort(
    top_mean
)[::-1]

top_idx = top_idx[display_order]

top_mean = top_mean[display_order]

top_std = top_std[display_order]

fold_top_weights = fold_top_weights[
    :,
    display_order
]


# ============================================================
# 7. Print final ranking
# ============================================================

print("\n==============================================")
print("Final seed ranking")
print("Normalized weight: mean ± SD across folds")
print("==============================================")


for rank, (idx, mean_w, std_w) in enumerate(
    zip(
        top_idx,
        top_mean,
        top_std
    ),
    start=1
):

    roi = roi_names[idx]

    network = to_hierarchy_group(roi_map.get(
        roi,
        "Other"
    ))

    print(
        f"{rank:2d}. "
        f"ROI {idx:3d} | "
        f"{roi:25s} | "
        f"{network:15s} | "
        f"{mean_w:.4f} ± {std_w:.4f}"
    )


# ============================================================
# 8. Fold-wise values
# ============================================================

print("\n==============================================")
print("Fold-wise normalized weights")
print("==============================================")


for rank, idx in enumerate(top_idx):

    vals = fold_top_weights[
        :,
        rank
    ]

    vals_str = "  ".join(
        [
            f"F{fold_numbers[i]}={vals[i]:.4f}"
            for i in range(len(vals))
        ]
    )

    roi = roi_names[idx]

    network = to_hierarchy_group(roi_map.get(
        roi,
        "Other"
    ))

    print(
        f"{rank+1:2d}. "
        f"{roi:25s} | "
        f"{network:15s} | "
        f"{vals_str}"
    )


# ============================================================
# 9. Plot
# ============================================================

top_names = [
    roi_names[idx]
    for idx in top_idx
]

top_networks = [
    to_hierarchy_group(roi_map.get(name, "Other"))
    for name in top_names
]


# ------------------------------------------------------------
# network -> bar color
# ------------------------------------------------------------

bar_colors = [
    net_colors.get(
        network,
        net_colors["Other"]
    )
    for network in top_networks
]


labels = [
    f"{rank}. {name}"
    for rank, name in enumerate(
        top_names,
        start=1
    )
]


y = np.arange(TOP_K)


fig, ax = plt.subplots(
    figsize=(7.2, 7.0)
)


bars = ax.barh(
    y,
    top_mean,
    xerr=top_std,
    height=0.62,
    capsize=3,

    color=bar_colors,

    error_kw={
        "elinewidth": 1.0,
        "capthick": 1.0,
        "ecolor": "black"
    }
)


# ============================================================
# Y labels
# ============================================================

ax.set_yticks(y)

ax.set_yticklabels(
    labels,
    fontsize=11
)

ax.invert_yaxis()


# ============================================================
# Value text
# ============================================================

offset = max(top_mean) * 0.015


for yi, mean_w, std_w in zip(
    y,
    top_mean,
    top_std
):

    ax.text(
        mean_w + std_w + offset,
        yi,
        f"{mean_w:.3f}",
        va="center",
        ha="left",
        fontsize=10
    )


# ============================================================
# Functional system legend
# ============================================================

# 실제 Top-K에 존재하는 network만 legend에 표시
used_networks = set(top_networks)


legend_handles = []

for net in net_order:

    if net not in used_networks:
        continue

    legend_handles.append(

        Patch(
            facecolor=net_colors[net],
            edgecolor="none",
            label=net_labels[net]
        )

    )


ax.legend(
    handles=legend_handles,
    title="Functional system",
    frameon=False,
    fontsize=9,
    title_fontsize=9,
    loc="lower right"
)


# ============================================================
# Style
# ============================================================

ax.set_xlabel(
    r"Normalized seed weight $\alpha_s$",
    fontsize=12
)


ax.set_title(
    "Top-M seed ranking",
    loc="left",
    fontsize=14,
    fontweight="bold",
    pad=15
)


# legend / value text 공간 확보
xmax = np.max(
    top_mean + top_std
) * 1.30

ax.set_xlim(
    0,
    xmax
)


ax.xaxis.grid(
    True,
    linewidth=0.7,
    alpha=0.25
)

ax.set_axisbelow(True)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


ax.tick_params(
    axis="y",
    length=0
)


plt.tight_layout()


# ============================================================
# 10. Save
# ============================================================

save_path = os.path.join(
    EXP_DIR,
    SAVE_NAME
)


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "\nSaved:",
    save_path
)