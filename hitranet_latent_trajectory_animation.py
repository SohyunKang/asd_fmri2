#!/usr/bin/env python3
"""Create a presentation-ready HiTraNet latent-trajectory animation.

The script accepts either empirical latent representations or a synthetic
concept demo.  Empirical input can be:

1) NPZ with
   - ``latent``: [subjects, hops, dimensions] (projected globally to 2-D), or
   - ``coords``: [subjects, hops, 2]
   - ``labels``: [subjects], TD=0 and ASD=1
   - optional ``subject_ids``: [subjects]

2) CSV with columns: subject_id, group, hop, x, y

Without --input, a clearly labelled conceptual demonstration is generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch


BG = "#F7F9FC"
INK = "#172033"
MUTED = "#697386"
GRID = "#DFE5EE"
TD = "#2574D8"
ASD = "#E9683B"
NETWORK_COLORS = {
    "Sensory": "#2F83C5",
    "DAN/Salience": "#8664B5",
    "Frontoparietal": "#39A64A",
    "DMN": "#D83A35",
}


DEFAULT_HOP_REGIONS = {
    0: [
        ("Occipital_Sup_L", "Sensory", 1.00, (0.27, 0.48)),
        ("Calcarine_L", "Sensory", 0.82, (0.30, 0.43)),
        ("Lingual_R", "Sensory", 0.68, (0.34, 0.39)),
    ],
    1: [
        ("Postcentral_R", "Sensory", 1.00, (0.48, 0.56)),
        ("Precentral_L", "Sensory", 0.86, (0.52, 0.61)),
        ("Supp_Motor_Area_L", "DAN/Salience", 0.67, (0.58, 0.63)),
    ],
    2: [
        ("Parietal_Inf_L", "DAN/Salience", 1.00, (0.44, 0.50)),
        ("Precuneus_R", "DMN", 0.84, (0.40, 0.47)),
        ("Cingulum_Mid_L", "DAN/Salience", 0.69, (0.52, 0.47)),
    ],
    3: [
        ("Frontal_Mid_L", "Frontoparietal", 1.00, (0.68, 0.54)),
        ("Frontal_Inf_Tri_R", "Frontoparietal", 0.81, (0.75, 0.46)),
        ("Insula_L", "DAN/Salience", 0.65, (0.61, 0.44)),
    ],
    4: [
        ("Angular_L", "DMN", 1.00, (0.40, 0.50)),
        ("Temporal_Mid_R", "DMN", 0.83, (0.52, 0.34)),
        ("Cingulum_Ant_L", "DMN", 0.66, (0.65, 0.48)),
    ],
    5: [
        ("Frontal_Med_Orb_L", "DMN", 1.00, (0.72, 0.42)),
        ("Precuneus_L", "DMN", 0.85, (0.42, 0.48)),
        ("Cingulum_Post_R", "DMN", 0.71, (0.47, 0.46)),
    ],
}


def ease(t: float) -> float:
    """Smooth cubic easing from 0 to 1."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def pca_2d(latent: np.ndarray) -> np.ndarray:
    """Fit one PCA over every subject/hop so axes stay fixed over time."""
    n, h, d = latent.shape
    flat = latent.reshape(n * h, d).astype(float)
    flat -= flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    projected = flat @ vt[:2].T
    return projected.reshape(n, h, 2)


def load_input(path: Path):
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "coords" in data:
            coords = np.asarray(data["coords"], dtype=float)
        elif "latent" in data:
            latent = np.asarray(data["latent"], dtype=float)
            if latent.ndim != 3:
                raise ValueError("NPZ latent must have shape [subjects, hops, dimensions].")
            coords = pca_2d(latent)
        else:
            raise KeyError("NPZ must contain either 'coords' or 'latent'.")
        labels = np.asarray(data["labels"]).astype(int)
        ids = (
            np.asarray(data["subject_ids"]).astype(str)
            if "subject_ids" in data
            else np.asarray([f"S-{i + 1:03d}" for i in range(len(labels))])
        )
        hop_numbers = (
            np.asarray(data["hop_numbers"]).astype(int)
            if "hop_numbers" in data
            else np.arange(coords.shape[1], dtype=int)
        )
        return coords, labels, ids, False, hop_numbers

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        required = {"subject_id", "group", "hop", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"CSV missing columns: {sorted(missing)}")
        df = df.sort_values(["subject_id", "hop"])
        ids = df["subject_id"].drop_duplicates().astype(str).to_numpy()
        hops = sorted(df["hop"].unique())
        coords = np.empty((len(ids), len(hops), 2), dtype=float)
        labels = np.empty(len(ids), dtype=int)
        for i, sid in enumerate(ids):
            part = df[df["subject_id"].astype(str) == sid].set_index("hop").loc[hops]
            coords[i] = part[["x", "y"]].to_numpy(dtype=float)
            raw_group = str(part["group"].iloc[0]).strip().upper()
            labels[i] = 1 if raw_group in {"1", "ASD", "PATIENT"} else 0
        return coords, labels, ids, False, np.asarray(hops)

    raise ValueError("--input must be an .npz or .csv file.")


def make_concept_data(seed: int = 17):
    """Synthetic geometry shaped after the supplied seed-averaged figure."""
    rng = np.random.default_rng(seed)
    n_td, n_asd, hops = 30, 34, 6
    labels = np.r_[np.zeros(n_td, dtype=int), np.ones(n_asd, dtype=int)]
    ids = np.asarray(
        [f"TD-{i + 1:02d}" for i in range(n_td)]
        + [f"ASD-{i + 1:02d}" for i in range(n_asd)]
    )

    # The two zig-zagging class paths mirror the geometry of figure 3(a),
    # while subject-specific offsets make within-group heterogeneity visible.
    td_centers = np.array([
        [20.0, 9.0],
        [14.8, 5.5],
        [9.6, -6.6],
        [-5.3, 1.9],
        [-5.9, 3.8],
        [5.3, 18.8],
    ])
    asd_centers = np.array([
        [20.0, 8.0],
        [18.0, 3.5],
        [14.2, 13.8],
        [7.0, -6.8],
        [-5.7, 3.6],
        [5.2, 18.5],
    ])

    coords = np.zeros((n_td + n_asd, hops, 2), dtype=float)
    for i in range(n_td + n_asd):
        is_asd = labels[i] == 1
        centers = asd_centers if is_asd else td_centers
        base = rng.normal(0, [0.22, 0.18])
        tendency = rng.normal(0, [0.10, 0.10])
        severity = rng.normal(0.0, 0.48) if is_asd else rng.normal(0.0, 0.16)
        for h in range(hops):
            progress = h / (hops - 1)
            shared_offset = base * (0.85 - 0.25 * progress) + tendency * progress
            heterogeneity = np.array([
                0.16 * severity * progress,
                -0.85 * severity * progress ** 1.45,
            ]) if is_asd else np.array([0.02 * severity, 0.16 * severity * progress])
            noise_scale = (0.18 + 0.045 * h) * (1.35 if is_asd else 1.0)
            coords[i, h] = centers[h] + shared_offset + heterogeneity + rng.normal(0, noise_scale, 2)

    return coords, labels, ids, True, np.arange(hops, dtype=int)


def covariance_ellipse(points, color, alpha=0.10):
    if len(points) < 3:
        return None
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # About 80% visual envelope rather than a formal confidence interval.
    width, height = 3.2 * np.sqrt(np.maximum(vals, 1e-8))
    return Ellipse(
        xy=points.mean(axis=0), width=width, height=height, angle=angle,
        facecolor=color, edgecolor=color, lw=1.4, alpha=alpha,
    )


def current_state(coords, frame, hold_start, transition, hold_end):
    hops = coords.shape[1]
    if frame < hold_start:
        return 0, 0, 0.0, coords[:, 0]
    remaining = frame - hold_start
    block = transition + hold_end
    stage = min(remaining // block, hops - 2)
    within = remaining - stage * block
    if within >= transition:
        frac = 1.0
    else:
        frac = ease(within / max(transition - 1, 1))
    left = int(stage)
    right = min(left + 1, hops - 1)
    now = coords[:, left] * (1.0 - frac) + coords[:, right] * frac
    return right, left, frac, now


def load_region_table(path: Path | None, n_hops: int):
    """Return {hop: [(region, network, importance, (brain_x, brain_y)), ...]}."""
    if path is None:
        result = {}
        for hop in range(n_hops):
            result[hop] = DEFAULT_HOP_REGIONS.get(hop, DEFAULT_HOP_REGIONS[max(DEFAULT_HOP_REGIONS)])
        return result

    df = pd.read_csv(path)
    required = {"hop", "region", "network", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Region CSV missing columns: {sorted(missing)}")
    has_xy = {"brain_x", "brain_y"}.issubset(df.columns)
    fallback_pos = [(0.30, 0.45), (0.50, 0.58), (0.68, 0.45), (0.45, 0.36)]
    result = {}
    for hop in range(n_hops):
        part = df[df["hop"] == hop].sort_values("importance", ascending=False).head(4)
        rows = []
        for j, row in part.reset_index(drop=True).iterrows():
            position = (
                (float(row["brain_x"]), float(row["brain_y"]))
                if has_xy and pd.notna(row["brain_x"]) and pd.notna(row["brain_y"])
                else fallback_pos[j % len(fallback_pos)]
            )
            rows.append((str(row["region"]), str(row["network"]), float(row["importance"]), position))
        result[hop] = rows if rows else DEFAULT_HOP_REGIONS.get(hop, DEFAULT_HOP_REGIONS[5])
    return result


def add_network_background(ax):
    """Overlapping network fields styled after the supplied latent-space plot."""
    fields = [
        ((6.2, 3.6), 18.5, 15.4, "Sensory"),
        ((9.4, 8.8), 17.2, 14.0, "DAN/Salience"),
        ((12.1, 12.8), 14.0, 12.5, "Frontoparietal"),
        ((14.4, 12.2), 13.8, 14.0, "DMN"),
    ]
    for center, width, height, network in fields:
        ax.add_patch(Ellipse(
            center, width, height,
            facecolor=NETWORK_COLORS[network], edgecolor=NETWORK_COLORS[network],
            lw=1.1, alpha=0.095, zorder=0,
        ))


def draw_brain_region_panel(panel, regions, hop_label):
    """Draw a compact schematic brain and ranked regions for the active hop."""
    panel.text(0.09, 0.47, "DOMINANT REGIONS", fontsize=8.2, color=MUTED, fontweight="bold")

    # Schematic lateral brain; coordinates are normalized within this panel.
    brain = Ellipse(
        (0.51, 0.335), 0.72, 0.22, angle=4,
        transform=panel.transAxes, facecolor="white", edgecolor="#B9C5D5",
        lw=1.2, zorder=1,
    )
    panel.add_patch(brain)
    panel.plot([0.79, 0.86, 0.83], [0.31, 0.27, 0.24], color="#B9C5D5", lw=1.1,
               transform=panel.transAxes, zorder=1)
    panel.text(0.10, 0.395, f"Hop {hop_label} influence map", fontsize=8.5, color=MUTED)

    max_imp = max([r[2] for r in regions], default=1.0)
    for name, network, importance, (bx, by) in regions:
        color = NETWORK_COLORS.get(network, MUTED)
        # User-supplied brain_x/y can be either panel-normalized or MNI-like.
        px = bx if 0 <= bx <= 1 else 0.51 + np.clip(bx / 100.0, -0.28, 0.28)
        py = by if 0 <= by <= 1 else 0.335 + np.clip(by / 150.0, -0.08, 0.08)
        # Default positions were authored in the latent panel scale; remap to brain area.
        if 0.20 <= py <= 0.70:
            py = 0.255 + (py - 0.30) * 0.28
        panel.add_patch(Circle(
            (px, py), 0.018 + 0.012 * importance / max_imp,
            transform=panel.transAxes, fc=color, ec="white", lw=0.8,
            alpha=0.95, zorder=4,
        ))

    y0 = 0.175
    for rank, (name, network, importance, _) in enumerate(regions[:3], start=1):
        color = NETWORK_COLORS.get(network, MUTED)
        y = y0 - (rank - 1) * 0.052
        panel.text(0.09, y, f"{rank}", fontsize=8.4, color=MUTED, va="center")
        panel.text(0.15, y, name, fontsize=8.7, color=INK, va="center")
        panel.add_patch(FancyBboxPatch(
            (0.62, y - 0.010), 0.26 * importance / max_imp, 0.020,
            transform=panel.transAxes, boxstyle="round,pad=0.002,rounding_size=0.008",
            fc=color, ec="none", alpha=0.88,
        ))


def create_animation(
    coords, labels, ids, output, gif_output, preview_output,
    conceptual, region_table, hop_numbers, skip_gif=False, preview_only=False,
):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })

    fps = 30
    hold_start, transition, hold_end = 36, 48, 22
    total_frames = hold_start + (coords.shape[1] - 1) * (transition + hold_end) + 34

    xmin, ymin = coords.min(axis=(0, 1))
    xmax, ymax = coords.max(axis=(0, 1))
    padx = max((xmax - xmin) * 0.15, 0.5)
    pady = max((ymax - ymin) * 0.22, 0.5)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.85, 1.35], wspace=0.050)
    ax = fig.add_subplot(gs[0, 0], facecolor=BG)
    panel = fig.add_subplot(gs[0, 1], facecolor="#EEF2F7")

    stage_titles = [
        "Seed-conditioned origin",
        "Local sensory propagation",
        "Sensorimotor integration",
        "Intermediate transition",
        "Association recruitment",
        "Transmodal convergence",
    ]
    stage_copy = [
        "Seed-conditioned representations\nenter the latent hierarchy.",
        "The trajectory first emphasizes\nvisual and sensory organization.",
        "Sensorimotor systems contribute\nto the evolving representation.",
        "ASD and TD paths diverge across\nintermediate hierarchical levels.",
        "Distributed association regions\nare progressively recruited.",
        "A shared macro-flow remains, while\nsubject-specific variation persists.",
    ]

    def draw(frame):
        ax.clear()
        panel.clear()
        ax.set_facecolor(BG)
        panel.set_facecolor("#EEF2F7")
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.0)
        panel.set_autoscale_on(False)

        right, left, frac, now = current_state(
            coords, frame, hold_start, transition, hold_end
        )
        phase = right
        hop_label = int(hop_numbers[phase])

        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.grid(True, color=GRID, linewidth=0.75, alpha=0.72)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Latent dimension 1", fontsize=10, labelpad=10)
        ax.set_ylabel("Latent dimension 2", fontsize=10, labelpad=10)
        ax.tick_params(labelsize=8, length=0)
        ax.set_title(
            "Subject-specific hierarchical trajectories in latent space",
            loc="left", fontsize=16, color=INK, pad=24,
        )
        if conceptual:
            add_network_background(ax)
        ax.text(
            0.0, 1.01,
            "Thin paths: individuals   |   Thick paths: group tendency",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom",
        )

        for group, color, name in [(0, TD, "TD"), (1, ASD, "ASD")]:
            mask = labels == group
            group_points = now[mask]
            if not conceptual:
                ell = covariance_ellipse(group_points, color)
                if ell is not None:
                    ax.add_patch(ell)

            # Completed subject paths plus the current partial segment.
            for idx in np.where(mask)[0]:
                if left == 0 and frac == 0:
                    path = coords[idx, :1]
                else:
                    path = np.vstack([coords[idx, : left + 1], now[idx]])
                ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.16, lw=0.85)

            ax.scatter(
                group_points[:, 0], group_points[:, 1],
                s=28, c=color, edgecolor="white", linewidth=0.55,
                alpha=0.82, label=name, zorder=4,
            )

            group_all = coords[mask].mean(axis=0)
            group_now = group_points.mean(axis=0)
            centroid_path = np.vstack([group_all[: left + 1], group_now])
            ax.plot(
                centroid_path[:, 0], centroid_path[:, 1], color=color,
                lw=4.2, alpha=0.96, solid_capstyle="round", zorder=5,
            )
            ax.scatter(
                group_now[0], group_now[1], s=128, color=color,
                edgecolor="white", linewidth=2.0, zorder=6,
            )

        # Highlight three patient-specific endpoints in the final phase.
        if phase >= coords.shape[1] - 1 and frac > 0.55:
            asd_indices = np.where(labels == 1)[0]
            final_y = coords[asd_indices, -1, 1]
            picks = asd_indices[[np.argmin(final_y), np.argsort(final_y)[len(final_y)//2], np.argmax(final_y)]]
            fade = np.clip((frac - 0.55) / 0.35, 0, 1)
            for idx in picks:
                x, y = now[idx]
                ax.annotate(
                    ids[idx], xy=(x, y), xytext=(7, 7), textcoords="offset points",
                    fontsize=7.5, color=ASD, alpha=fade,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=ASD, alpha=0.88),
                    zorder=8,
                )

        legend = ax.legend(
            loc="lower right", frameon=True, fancybox=True, framealpha=0.95,
            facecolor="white", edgecolor=GRID, fontsize=9,
        )
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)

        # Hop badge.
        badge = FancyBboxPatch(
            (0.825, 0.925), 0.15, 0.065,
            transform=ax.transAxes, boxstyle="round,pad=0.012,rounding_size=0.025",
            fc=INK, ec="none", zorder=10,
        )
        ax.add_patch(badge)
        ax.text(
            0.9, 0.957, f"HOP {hop_label}", transform=ax.transAxes,
            ha="center", va="center", color="white", fontsize=9.5,
            fontweight="bold", zorder=11,
        )

        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.text(0.10, 0.90, f"0{phase + 1}", fontsize=13, color=MUTED, fontweight="bold")
        panel.text(
            0.10, 0.80, stage_titles[min(phase, len(stage_titles) - 1)],
            fontsize=15, color=INK, fontweight="bold", va="top", wrap=True,
        )
        panel.plot([0.10, 0.90], [0.735, 0.735], color="#CBD4E1", lw=1.1)
        panel.text(
            0.10, 0.68, stage_copy[min(phase, len(stage_copy) - 1)],
            fontsize=10.5, color=MUTED, va="top", linespacing=1.55,
        )

        draw_brain_region_panel(panel, region_table.get(phase, []), hop_label)

        if conceptual:
            fig.text(
                0.013, 0.012,
                "Conceptual demonstration — replace with empirical embeddings before publication.",
                fontsize=7.5, color=MUTED,
            )
        else:
            fig.text(
                0.013, 0.012,
                "Projection fitted jointly across all subjects and propagation hops.",
                fontsize=7.5, color=MUTED,
            )

    # A late frame makes a useful static preview.
    draw(total_frames - 12)
    fig.savefig(preview_output, dpi=180, facecolor=BG)

    if preview_only:
        plt.close(fig)
        return

    anim = FuncAnimation(fig, draw, frames=total_frames, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    anim.save(output, writer=writer, dpi=150)

    # Smaller, shareable preview GIF.
    if not skip_gif:
        gif_anim = FuncAnimation(
            fig, draw, frames=range(0, total_frames, 2), interval=2000 / fps, blit=False
        )
        gif_anim.save(gif_output, writer=PillowWriter(fps=fps // 2), dpi=85)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Optional .npz or .csv latent data")
    parser.add_argument("--output", type=Path, default=Path("hitranet_latent_trajectory.mp4"))
    parser.add_argument("--gif", type=Path, default=Path("hitranet_latent_trajectory_preview.gif"))
    parser.add_argument("--preview", type=Path, default=Path("hitranet_latent_trajectory_preview.png"))
    parser.add_argument(
        "--regions_csv", type=Path, default=None,
        help="Optional columns: hop,region,network,importance[,brain_x,brain_y]",
    )
    parser.add_argument("--skip_gif", action="store_true", help="Render MP4 and PNG only")
    parser.add_argument("--preview_only", action="store_true", help="Render only the static PNG")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input is None:
        coords, labels, ids, conceptual, hop_numbers = make_concept_data(args.seed)
    else:
        coords, labels, ids, conceptual, hop_numbers = load_input(args.input)

    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("Coordinates must have shape [subjects, hops, 2].")
    if coords.shape[0] != len(labels):
        raise ValueError("Number of coordinate subjects and labels must match.")
    if len(hop_numbers) != coords.shape[1]:
        raise ValueError("hop_numbers length must match the trajectory hop dimension.")
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or infinite values.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    region_table = load_region_table(args.regions_csv, coords.shape[1])
    create_animation(
        coords, labels, ids, args.output, args.gif, args.preview,
        conceptual, region_table, hop_numbers, args.skip_gif, args.preview_only,
    )
    if not args.preview_only:
        print(f"Saved MP4: {args.output.resolve()}")
    if not args.skip_gif and not args.preview_only:
        print(f"Saved GIF : {args.gif.resolve()}")
    print(f"Saved PNG : {args.preview.resolve()}")


if __name__ == "__main__":
    main()
