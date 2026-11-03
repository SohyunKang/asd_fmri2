#!/usr/bin/env python3
"""Build the HiTraNet trajectory animation from fold*_best_aux.pt.

This consumes the tensors already saved by the user's training code, so an
existing experiment does not need to be retrained.

Expected saved shapes
---------------------
*_g : [subject, hop, seed, latent_dim]
*_p : [subject, hop, seed, ROI]
*_y : [subject] where ASD=0 and TD=1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_roi_names(aux, aal_txt: Path | None, n_rois: int):
    if "roi_names" in aux and len(aux["roi_names"]) == n_rois:
        return [str(x) for x in aux["roi_names"]]

    if aal_txt is None:
        raise ValueError(
            "The saved aux file has no roi_names. Pass --aal_txt data/ROI_MNI_V4.txt."
        )

    try:
        from utils import load_roi_names_from_txt

        names = load_roi_names_from_txt(str(aal_txt))
    except Exception:
        # Conservative fallback for one-name-per-line files.
        names = []
        for line in aal_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if parts and parts[0].isdigit() and len(parts) > 1:
                parts = parts[1:]
            names.append(parts[-1])

    if len(names) != n_rois:
        raise ValueError(f"ROI names ({len(names)}) != saved influence ROIs ({n_rois})")
    return [str(x) for x in names]


def infer_macro_network(roi_name: str) -> str:
    name = str(roi_name).lower()
    groups = [
        (
            "Sensory",
            (
                "occipital", "calcarine", "cuneus", "lingual", "fusiform",
                "precentral", "postcentral", "paracentral", "rolandic",
                "heschl", "supp_motor",
            ),
        ),
        (
            "DAN/Salience",
            (
                "insula", "cingulum_ant", "cingulum_mid", "parietal_sup",
                "frontal_sup", "supramarginal",
            ),
        ),
        ("Frontoparietal", ("frontal_mid", "frontal_inf", "parietal_inf")),
        (
            "DMN",
            (
                "precuneus", "cingulum_post", "angular", "frontal_med_orb",
                "frontal_sup_medial", "rectus", "hippocampus", "temporal_mid",
            ),
        ),
    ]
    for network, tokens in groups:
        if any(token in name for token in tokens):
            return network
    return "Other"


def schematic_position(network: str, rank: int):
    centers = {
        "Sensory": (0.30, 0.43),
        "DAN/Salience": (0.50, 0.54),
        "Frontoparietal": (0.68, 0.53),
        "DMN": (0.53, 0.42),
        "Other": (0.48, 0.46),
    }
    offsets = [(-0.025, 0.020), (0.025, 0.000), (0.000, -0.025), (0.040, 0.025)]
    cx, cy = centers.get(network, centers["Other"])
    ox, oy = offsets[rank % len(offsets)]
    return cx + ox, cy + oy


def select_saved_splits(aux, split: str):
    names = ["train", "val", "test"] if split == "all" else [split]
    g_parts, p_parts, y_parts, subjects, split_names = [], [], [], [], []
    for name in names:
        required = [f"{name}_g", f"{name}_p", f"{name}_y", f"{name}_subject"]
        missing = [key for key in required if key not in aux]
        if missing:
            raise KeyError(f"Saved aux is missing: {missing}")
        g_parts.append(torch.as_tensor(aux[f"{name}_g"]).float())
        p_parts.append(torch.as_tensor(aux[f"{name}_p"]).float())
        y = np.asarray(aux[f"{name}_y"], dtype=np.int64)
        ids = [str(x) for x in aux[f"{name}_subject"]]
        if len(y) != len(ids):
            raise ValueError(f"{name}: label and subject counts differ")
        y_parts.append(y)
        subjects.extend(ids)
        split_names.extend([name] * len(ids))

    return (
        torch.cat(g_parts, dim=0),
        torch.cat(p_parts, dim=0),
        np.concatenate(y_parts),
        np.asarray(subjects, dtype=str),
        np.asarray(split_names, dtype=str),
    )


def aggregate_seeds(aux, g, p, method: str):
    n_seeds = g.shape[2]
    if method == "mean":
        weights = torch.ones(n_seeds, dtype=torch.float32) / n_seeds
    else:
        if "seed_weights_model_order" in aux:
            raw_weights = aux["seed_weights_model_order"]
        elif "seed_weights" in aux:
            raw_weights = aux["seed_weights"]
            print(
                "Warning: this older aux file stores only reported seed weights. "
                "If their order differs from the g/p seed axis, use --seed_aggregation mean."
            )
        else:
            raise KeyError("No seed weights found; use --seed_aggregation mean")
        weights = torch.as_tensor(raw_weights, dtype=torch.float32).flatten()
        if len(weights) != n_seeds:
            raise ValueError(f"Seed weights ({len(weights)}) != g/p seed dimension ({n_seeds})")
        weights = weights.clamp_min(0)
        weights = weights / weights.sum().clamp_min(1e-12)

    latent = torch.einsum("shmd,m->shd", g, weights).numpy()
    influence = torch.einsum("shmn,m->shn", p, weights).numpy()
    return latent, influence, weights.numpy()


def export_inputs(aux_path: Path, output_dir: Path, split: str, aggregation: str, aal_txt):
    aux = load_torch_file(aux_path)
    g, p, raw_labels, subject_ids, split_names = select_saved_splits(aux, split)

    if g.ndim != 4 or p.ndim != 4:
        raise ValueError(f"Expected g/p rank 4; received g={g.shape}, p={p.shape}")
    if g.shape[:3] != p.shape[:3]:
        raise ValueError(f"g/p subject-hop-seed dimensions differ: {g.shape} vs {p.shape}")

    latent, influence, used_weights = aggregate_seeds(aux, g, p, aggregation)
    roi_names = load_roi_names(aux, aal_txt, influence.shape[-1])

    # Source labels are ASD=0, TD=1; animation uses TD=0, ASD=1.
    animation_labels = (raw_labels == 0).astype(np.int64)
    n_hops = latent.shape[1]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = aux_path.stem.replace("_best_aux", "")
    npz_path = output_dir / f"{stem}_trajectory_{split}.npz"
    csv_path = output_dir / f"{stem}_hop_region_importance_{split}.csv"

    seed_indices = aux.get("seed_indices_model_order", aux.get("seed_indices", []))
    np.savez_compressed(
        npz_path,
        latent=latent,
        influence=influence,
        labels=animation_labels,
        raw_labels=raw_labels,
        subject_ids=subject_ids,
        split=split_names,
        hop_numbers=np.arange(1, n_hops + 1, dtype=np.int64),
        roi_names=np.asarray(roi_names, dtype=str),
        seed_indices=np.asarray(seed_indices, dtype=np.int64),
        seed_weights=used_weights,
    )

    asd = raw_labels == 0
    td = raw_labels == 1
    overall_imp = influence.mean(axis=0)
    asd_imp = influence[asd].mean(axis=0) if asd.any() else np.full_like(overall_imp, np.nan)
    td_imp = influence[td].mean(axis=0) if td.any() else np.full_like(overall_imp, np.nan)

    rows = []
    for hop in range(n_hops):
        order = np.argsort(-overall_imp[hop], kind="stable")
        denom = max(float(np.nanmax(overall_imp[hop])), 1e-12)
        for rank, roi_idx in enumerate(order, start=1):
            region = roi_names[roi_idx]
            network = infer_macro_network(region)
            bx, by = schematic_position(network, rank - 1)
            rows.append({
                "hop": hop,
                "model_hop": hop + 1,
                "rank": rank,
                "roi_index": int(roi_idx),
                "region": region,
                "network": network,
                "importance": float(overall_imp[hop, roi_idx] / denom),
                "importance_raw": float(overall_imp[hop, roi_idx]),
                "importance_asd": float(asd_imp[hop, roi_idx]),
                "importance_td": float(td_imp[hop, roi_idx]),
                "asd_minus_td": float(asd_imp[hop, roi_idx] - td_imp[hop, roi_idx]),
                "brain_x": bx,
                "brain_y": by,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved trajectory NPZ: {npz_path}")
    print(f"Saved region CSV   : {csv_path}")
    return npz_path, csv_path, stem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux", type=Path, required=True, help="fold*_best_aux.pt")
    parser.add_argument("--aal_txt", type=Path, default=Path("data/ROI_MNI_V4.txt"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seed_aggregation", choices=["mean", "weighted"], default="mean")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--animation_script",
        type=Path,
        default=Path(__file__).with_name("hitranet_latent_trajectory_animation.py"),
    )
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make_gif", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.aux.parent
    npz_path, csv_path, stem = export_inputs(
        aux_path=args.aux,
        output_dir=output_dir,
        split=args.split,
        aggregation=args.seed_aggregation,
        aal_txt=args.aal_txt,
    )

    if args.render:
        mp4_path = output_dir / f"{stem}_trajectory_{args.split}.mp4"
        png_path = output_dir / f"{stem}_trajectory_{args.split}.png"
        cmd = [
            sys.executable,
            str(args.animation_script),
            "--input", str(npz_path),
            "--regions_csv", str(csv_path),
            "--output", str(mp4_path),
            "--preview", str(png_path),
        ]
        if not args.make_gif:
            cmd.append("--skip_gif")
        subprocess.run(cmd, check=True)
        print(f"Saved presentation video: {mp4_path}")
        print(f"Saved presentation still: {png_path}")


if __name__ == "__main__":
    main()
