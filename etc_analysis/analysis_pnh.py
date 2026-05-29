import torch
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -------------------------
# load outputs
# -------------------------

data = torch.load("/home/sohyunkang/asd_fmri2/experiments/w0_hitranet_20260310_172609_092015/fold1_best_aux.pt")

p = data["p"].numpy()     # (S,K,M,N)

K = p.shape[1]

p_mean = p.mean(axis=0)
p_mean = p_mean.mean(axis=1)

# scale
vmax = np.percentile(p_mean,99)
norm = Normalize(vmin=0, vmax=vmax)

# -------------------------
# load atlas
# -------------------------

atlas_img = nib.load("aal.nii")
atlas = atlas_img.get_fdata()

# -------------------------
# network labels
# -------------------------

network_labels = []

for line in open("aal_network_labels.txt"):
    roi, net = line.strip().split()
    network_labels.append(net)

network_labels = np.array(network_labels)

sensory_mask = network_labels=="sensory"
assoc_mask   = network_labels=="association"
dmn_mask     = network_labels=="dmn"

# -------------------------
# ROI -> volume
# -------------------------

def roi_to_volume(weights):

    vol = np.zeros_like(atlas)

    for i,w in enumerate(weights):

        vol[atlas==(i+1)] = w

    return vol

# -------------------------
# stronger colormaps
# -------------------------

blue_cmap = LinearSegmentedColormap.from_list(
    "sensory",
    ["white","#2166ac"]
)

green_cmap = LinearSegmentedColormap.from_list(
    "assoc",
    ["white","#1b7837"]
)

red_cmap = LinearSegmentedColormap.from_list(
    "dmn",
    ["white","#b2182b"]
)

# -------------------------
# save dir
# -------------------------

save_dir = "brain_figures"
os.makedirs(save_dir,exist_ok=True)

# -------------------------
# plotting
# -------------------------

for k in range(K):

    p_sens = np.zeros_like(p_mean[k])
    p_assoc = np.zeros_like(p_mean[k])
    p_dmn = np.zeros_like(p_mean[k])

    p_sens[sensory_mask] = p_mean[k][sensory_mask]
    p_assoc[assoc_mask]  = p_mean[k][assoc_mask]
    p_dmn[dmn_mask]      = p_mean[k][dmn_mask]

    img_sens = nib.Nifti1Image(roi_to_volume(p_sens), atlas_img.affine)
    img_assoc = nib.Nifti1Image(roi_to_volume(p_assoc), atlas_img.affine)
    img_dmn = nib.Nifti1Image(roi_to_volume(p_dmn), atlas_img.affine)

    display = plotting.plot_glass_brain(
        None,
        title=f"Hop {k+1}",
        colorbar=False
    )

    display.add_overlay(
        img_sens,
        cmap=blue_cmap,
        vmin=0,
        vmax=vmax,
    )

    display.add_overlay(
        img_assoc,
        cmap=green_cmap,
        vmin=0,
        vmax=vmax,
    )

    display.add_overlay(
        img_dmn,
        cmap=red_cmap,
        vmin=0,
        vmax=vmax,
    )

    # save temporary brain image
    tmp_path = os.path.join(save_dir,f"hop{k+1}_tmp.png")
    display.savefig(tmp_path)
    display.close()

    # -------------------------
    # add colorbars
    # -------------------------

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes([0.02,0.05,0.75,0.9])

    img = plt.imread(tmp_path)
    ax.imshow(img)
    ax.axis("off")

    # colorbars

    sm_blue = ScalarMappable(norm=norm, cmap=blue_cmap)
    sm_green = ScalarMappable(norm=norm, cmap=green_cmap)
    sm_red = ScalarMappable(norm=norm, cmap=red_cmap)

  
    cax3 = fig.add_axes([0.82,0.11,0.02,0.2])
    cb3 = plt.colorbar(sm_red, cax=cax3)
    cb3.set_label("DMN p")

    cax2 = fig.add_axes([0.82,0.38,0.02,0.2])
    cb2 = plt.colorbar(sm_green, cax=cax2)
    cb2.set_label("Association p")

    cax1 = fig.add_axes([0.82,0.65,0.02,0.2])
    cb1 = plt.colorbar(sm_blue, cax=cax1)
    cb1.set_label("Sensory p")

    out_path = os.path.join(save_dir,f"hop{k+1}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()