import os
import glob
import json
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

N_ROIS = 116
TOPKS = (5, 10, 20, 30)


# ============================================================
# BASIC JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD FOLD METRICS
# ============================================================

def load_fold_metrics(exp_dir):
    """
    Load:
        fold1_metrics.json
        fold2_metrics.json
        ...
        fold5_metrics.json

    Each JSON must contain:
        fold
        seed_indices
        seed_weights
    """

    fold_files = glob.glob(
        os.path.join(
            exp_dir,
            "fold*_metrics.json"
        )
    )

    if len(fold_files) == 0:
        return []

    results = []

    for path in fold_files:

        try:
            data = load_json(path)

        except Exception as e:

            print(
                f"[SKIP] Cannot read "
                f"{path}: {e}"
            )

            continue

        if not isinstance(data, dict):

            print(
                f"[SKIP] Not dict: {path}"
            )

            continue

        if "fold" not in data:

            print(
                f"[SKIP] 'fold' missing: "
                f"{path}"
            )

            continue

        if "seed_indices" not in data:

            print(
                f"[SKIP] 'seed_indices' missing: "
                f"{path}"
            )

            continue

        if "seed_weights" not in data:

            print(
                f"[SKIP] 'seed_weights' missing: "
                f"{path}"
            )

            continue

        if (
            len(data["seed_indices"])
            !=
            len(data["seed_weights"])
        ):

            print(
                f"[SKIP] seed length mismatch: "
                f"{path}"
            )

            continue

        results.append(data)

    results = sorted(
        results,
        key=lambda x: int(x["fold"])
    )

    return results


# ============================================================
# DISCOVER EXPERIMENTS
# ============================================================

def discover_experiments(root_dir):
    """
    root_dir:

        experiments/seed_stability/

    Each subdirectory represents one random-seed experiment.

    Random seed is read from:
        args.json["seed"]

    Fold results are read from:
        fold1_metrics.json
        ...
        fold5_metrics.json
    """

    if not os.path.isdir(root_dir):

        raise FileNotFoundError(
            f"Directory does not exist:\n"
            f"{root_dir}"
        )

    experiments = []

    print("\n")
    print("=" * 80)
    print("DISCOVER EXPERIMENTS")
    print("=" * 80)

    for name in sorted(
        os.listdir(root_dir)
    ):

        exp_dir = os.path.join(
            root_dir,
            name
        )

        if not os.path.isdir(exp_dir):
            continue

        # ----------------------------------------------------
        # args.json
        # ----------------------------------------------------

        args_path = os.path.join(
            exp_dir,
            "args.json"
        )

        if not os.path.exists(args_path):

            print(
                f"[SKIP] {name}: "
                f"args.json not found"
            )

            continue

        try:
            args_data = load_json(
                args_path
            )

        except Exception as e:

            print(
                f"[SKIP] {name}: "
                f"cannot read args.json ({e})"
            )

            continue

        # ----------------------------------------------------
        # Random seed
        # ----------------------------------------------------

        random_seed = args_data.get(
            "seed",
            None
        )

        if random_seed is None:

            print(
                f"[SKIP] {name}: "
                f"'seed' not found in args.json"
            )

            continue

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_results = load_fold_metrics(
            exp_dir
        )

        if len(fold_results) == 0:

            print(
                f"[SKIP] {name}: "
                f"no valid fold*_metrics.json"
            )

            continue

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        folds = {
            int(r["fold"]): r
            for r in fold_results
        }

        experiments.append({
            "name": name,
            "path": exp_dir,
            "random_seed": int(random_seed),
            "folds": folds
        })

        print(
            f"[LOAD] "
            f"{name} | "
            f"random_seed={random_seed} | "
            f"folds={len(folds)}"
        )

    print("-" * 80)

    print(
        f"Total experiments: "
        f"{len(experiments)}"
    )

    print("=" * 80)

    return experiments


# ============================================================
# RANK VECTOR
# ============================================================

def ranking_to_rank_vector(
    seed_indices,
    n_rois=N_ROIS
):
    """
    seed_indices is ordered from highest importance
    to lowest importance.

    Example:

        [76, 74, 95, 77, ...]

    becomes

        rank[76] = 1
        rank[74] = 2
        rank[95] = 3
        rank[77] = 4
    """

    rank_vector = np.full(
        n_rois,
        np.nan,
        dtype=float
    )

    for rank, roi in enumerate(
        seed_indices,
        start=1
    ):

        roi = int(roi)

        if 0 <= roi < n_rois:

            rank_vector[roi] = rank

    return rank_vector


# ============================================================
# WEIGHT VECTOR
# ============================================================

def ranking_to_weight_vector(
    seed_indices,
    seed_weights,
    n_rois=N_ROIS
):
    """
    Restore weights according to ROI index.
    """

    weight_vector = np.full(
        n_rois,
        np.nan,
        dtype=float
    )

    for roi, weight in zip(
        seed_indices,
        seed_weights
    ):

        roi = int(roi)

        if 0 <= roi < n_rois:

            weight_vector[roi] = float(
                weight
            )

    return weight_vector


# ============================================================
# JACCARD
# ============================================================

def jaccard(set_a, set_b):

    union = (
        set_a | set_b
    )

    if len(union) == 0:
        return np.nan

    return (
        len(set_a & set_b)
        /
        len(union)
    )


# ============================================================
# BUILD 5-FOLD CONSENSUS
# ============================================================

def build_5cv_consensus(
    experiments,
    n_rois=N_ROIS,
    topks=TOPKS
):
    """
    For EACH random-seed experiment:

        Fold 1
        Fold 2
        Fold 3
        Fold 4
        Fold 5
            |
            v
        5CV consensus

    Main consensus criterion:
        mean ROI rank across 5 folds

    Returns:
        fold_roi_df
        experiment_roi_df
        consensus_rankings
    """

    fold_rows = []
    experiment_rows = []

    consensus_rankings = {}

    for exp in experiments:

        exp_name = exp[
            "name"
        ]

        random_seed = exp[
            "random_seed"
        ]

        folds = exp[
            "folds"
        ]

        fold_ids = sorted(
            folds.keys()
        )

        print("\n")
        print("=" * 80)

        print(
            f"5CV CONSENSUS | "
            f"{exp_name} | "
            f"seed={random_seed}"
        )

        print("=" * 80)

        rank_matrix = []
        weight_matrix = []

        # ====================================================
        # Fold level
        # ====================================================

        for fold_id in fold_ids:

            result = folds[
                fold_id
            ]

            seed_indices = result[
                "seed_indices"
            ]

            seed_weights = result[
                "seed_weights"
            ]

            rank_vec = (
                ranking_to_rank_vector(
                    seed_indices,
                    n_rois
                )
            )

            weight_vec = (
                ranking_to_weight_vector(
                    seed_indices,
                    seed_weights,
                    n_rois
                )
            )

            rank_matrix.append(
                rank_vec
            )

            weight_matrix.append(
                weight_vec
            )

            # ------------------------------------------------
            # Save individual ROI information
            # ------------------------------------------------

            for roi in range(n_rois):

                fold_rows.append({

                    "experiment":
                        exp_name,

                    "random_seed":
                        random_seed,

                    "fold":
                        fold_id,

                    "roi":
                        roi,

                    "rank":
                        rank_vec[roi],

                    "weight":
                        weight_vec[roi],

                    "auc":
                        result.get(
                            "auc",
                            np.nan
                        )
                })

        rank_matrix = np.asarray(
            rank_matrix,
            dtype=float
        )

        weight_matrix = np.asarray(
            weight_matrix,
            dtype=float
        )

        # ====================================================
        # Statistics across 5 folds
        # ====================================================

        mean_rank = np.nanmean(
            rank_matrix,
            axis=0
        )

        median_rank = np.nanmedian(
            rank_matrix,
            axis=0
        )

        rank_std = np.nanstd(
            rank_matrix,
            axis=0,
            ddof=0
        )

        best_rank = np.nanmin(
            rank_matrix,
            axis=0
        )

        worst_rank = np.nanmax(
            rank_matrix,
            axis=0
        )

        mean_weight = np.nanmean(
            weight_matrix,
            axis=0
        )

        weight_std = np.nanstd(
            weight_matrix,
            axis=0,
            ddof=0
        )

        # ====================================================
        # Top-K frequency
        # ====================================================

        topk_counts = {}

        for k in topks:

            topk_counts[k] = np.sum(
                rank_matrix <= k,
                axis=0
            )

        # ====================================================
        # Save experiment-level ROI stats
        # ====================================================

        for roi in range(n_rois):

            row = {

                "experiment":
                    exp_name,

                "random_seed":
                    random_seed,

                "roi":
                    roi,

                "mean_rank_5cv":
                    mean_rank[roi],

                "median_rank_5cv":
                    median_rank[roi],

                "rank_std_5cv":
                    rank_std[roi],

                "best_rank_5cv":
                    best_rank[roi],

                "worst_rank_5cv":
                    worst_rank[roi],

                "mean_weight_5cv":
                    mean_weight[roi],

                "weight_std_5cv":
                    weight_std[roi],
            }

            for k in topks:

                row[
                    f"top{k}_fold_count"
                ] = int(
                    topk_counts[k][roi]
                )

                row[
                    f"top{k}_fold_ratio"
                ] = (
                    topk_counts[k][roi]
                    /
                    len(fold_ids)
                )

            experiment_rows.append(
                row
            )

        # ====================================================
        # Consensus ranking
        #
        # 1st: lower mean rank
        # 2nd: lower rank SD
        # ====================================================

        consensus_order = np.lexsort(
            (
                rank_std,
                mean_rank
            )
        )

        consensus_rankings[
            exp_name
        ] = {

            "random_seed":
                random_seed,

            "ranking":
                consensus_order.tolist(),

            "mean_rank":
                mean_rank,

            "rank_std":
                rank_std,

            "mean_weight":
                mean_weight
        }

        # ====================================================
        # Print Top-20
        # ====================================================

        print("\nTop-20 consensus ROIs")

        print(
            "-" * 80
        )

        for consensus_rank, roi in enumerate(
            consensus_order[:20],
            start=1
        ):

            print(
                f"{consensus_rank:2d}. "
                f"ROI {roi:3d} | "
                f"mean_rank={mean_rank[roi]:6.2f} | "
                f"SD={rank_std[roi]:6.2f} | "
                f"Top5={topk_counts[5][roi]}/{len(fold_ids)} | "
                f"Top10={topk_counts[10][roi]}/{len(fold_ids)} | "
                f"Top20={topk_counts[20][roi]}/{len(fold_ids)} | "
                f"weight={mean_weight[roi]:.6f}"
            )

    fold_roi_df = pd.DataFrame(
        fold_rows
    )

    experiment_roi_df = pd.DataFrame(
        experiment_rows
    )

    return (
        fold_roi_df,
        experiment_roi_df,
        consensus_rankings
    )


# ============================================================
# COMPARE 5CV CONSENSUS ACROSS RANDOM SEEDS
# ============================================================

def compare_5cv_consensus(
    consensus_rankings,
    n_rois=N_ROIS,
    topks=TOPKS
):
    """
    MAIN stability analysis.

    Example:

        random seed 42
            5CV consensus

                VS

        random seed 123
            5CV consensus
    """

    rows = []

    experiment_names = sorted(
        consensus_rankings.keys()
    )

    for exp_a, exp_b in itertools.combinations(
        experiment_names,
        2
    ):

        info_a = consensus_rankings[
            exp_a
        ]

        info_b = consensus_rankings[
            exp_b
        ]

        ranking_a = info_a[
            "ranking"
        ]

        ranking_b = info_b[
            "ranking"
        ]

        rank_a = (
            ranking_to_rank_vector(
                ranking_a,
                n_rois
            )
        )

        rank_b = (
            ranking_to_rank_vector(
                ranking_b,
                n_rois
            )
        )

        # ----------------------------------------------------
        # Full 116 ROI Spearman
        # ----------------------------------------------------

        rho, pvalue = spearmanr(
            rank_a,
            rank_b
        )

        row = {

            "experiment_a":
                exp_a,

            "experiment_b":
                exp_b,

            "random_seed_a":
                info_a["random_seed"],

            "random_seed_b":
                info_b["random_seed"],

            "spearman_rho":
                rho,

            "spearman_p":
                pvalue
        }

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        for k in topks:

            set_a = set(
                ranking_a[:k]
            )

            set_b = set(
                ranking_b[:k]
            )

            common = (
                set_a & set_b
            )

            union = (
                set_a | set_b
            )

            row[
                f"top{k}_overlap"
            ] = len(
                common
            )

            row[
                f"top{k}_overlap_ratio"
            ] = (
                len(common) / k
            )

            row[
                f"top{k}_jaccard"
            ] = jaccard(
                set_a,
                set_b
            )

            row[
                f"top{k}_common_rois"
            ] = ",".join(
                str(x)
                for x in sorted(
                    common
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAME FOLD ACROSS RANDOM SEEDS
# ============================================================

def compare_same_fold_across_random_seeds(
    experiments,
    n_rois=N_ROIS,
    topks=TOPKS
):
    """
    Secondary analysis.

    Compare:

        seed42 Fold1 vs seed123 Fold1
        seed42 Fold1 vs seed777 Fold1

        seed42 Fold2 vs seed123 Fold2
        ...

    IMPORTANT:
    This is meaningful as random-seed stability only if
    the CV split itself is identical between experiments.
    """

    rows = []

    for exp_a, exp_b in itertools.combinations(
        experiments,
        2
    ):

        folds_a = exp_a[
            "folds"
        ]

        folds_b = exp_b[
            "folds"
        ]

        common_folds = sorted(
            set(folds_a.keys())
            &
            set(folds_b.keys())
        )

        for fold in common_folds:

            result_a = folds_a[
                fold
            ]

            result_b = folds_b[
                fold
            ]

            ranking_a = result_a[
                "seed_indices"
            ]

            ranking_b = result_b[
                "seed_indices"
            ]

            rank_a = (
                ranking_to_rank_vector(
                    ranking_a,
                    n_rois
                )
            )

            rank_b = (
                ranking_to_rank_vector(
                    ranking_b,
                    n_rois
                )
            )

            rho, pvalue = spearmanr(
                rank_a,
                rank_b
            )

            row = {

                "fold":
                    fold,

                "experiment_a":
                    exp_a["name"],

                "experiment_b":
                    exp_b["name"],

                "random_seed_a":
                    exp_a["random_seed"],

                "random_seed_b":
                    exp_b["random_seed"],

                "spearman_rho":
                    rho,

                "spearman_p":
                    pvalue
            }

            for k in topks:

                set_a = set(
                    ranking_a[:k]
                )

                set_b = set(
                    ranking_b[:k]
                )

                common = (
                    set_a & set_b
                )

                row[
                    f"top{k}_overlap"
                ] = len(
                    common
                )

                row[
                    f"top{k}_overlap_ratio"
                ] = (
                    len(common) / k
                )

                row[
                    f"top{k}_jaccard"
                ] = jaccard(
                    set_a,
                    set_b
                )

                row[
                    f"top{k}_common_rois"
                ] = ",".join(
                    str(x)
                    for x in sorted(
                        common
                    )
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GLOBAL CONSENSUS
# ============================================================

def build_global_consensus(
    experiment_roi_df,
    topks=TOPKS
):
    """
    Combine all random-seed 5CV results.

    Example:

        random seed 42  -> 5CV
        random seed 123 -> 5CV
        random seed 777 -> 5CV

                 |
                 v

          Global consensus
    """

    rows = []

    n_experiments = (
        experiment_roi_df[
            "experiment"
        ].nunique()
    )

    # Total number of models:
    # random seeds x folds

    total_models = 0

    for _, g in experiment_roi_df.groupby(
        "experiment"
    ):

        # fold ratio denominator can vary,
        # but normally = 5
        first_roi = g.iloc[0]

        # infer number folds from ratios/counts
        # use Top5 as reference
        ratio = first_roi[
            "top5_fold_ratio"
        ]

        count = first_roi[
            "top5_fold_count"
        ]

        # This is not reliable if count == 0.
        # Total models will be computed below separately.

    # --------------------------------------------------------
    # ROI aggregation
    # --------------------------------------------------------

    for roi, g in experiment_roi_df.groupby(
        "roi"
    ):

        row = {

            "roi":
                int(roi),

            # --------------------------------------------
            # 5CV mean-rank across random seeds
            # --------------------------------------------

            "mean_rank_across_random_seeds":
                g[
                    "mean_rank_5cv"
                ].mean(),

            "median_rank_across_random_seeds":
                g[
                    "mean_rank_5cv"
                ].median(),

            "rank_std_across_random_seeds":
                g[
                    "mean_rank_5cv"
                ].std(
                    ddof=0
                ),

            # --------------------------------------------
            # Weight
            # --------------------------------------------

            "mean_weight_across_random_seeds":
                g[
                    "mean_weight_5cv"
                ].mean(),

            "weight_std_across_random_seeds":
                g[
                    "mean_weight_5cv"
                ].std(
                    ddof=0
                ),

            "n_random_seed_experiments":
                n_experiments
        }

        # --------------------------------------------
        # Top-K reproducibility
        # --------------------------------------------

        for k in topks:

            fold_counts = g[
                f"top{k}_fold_count"
            ]

            # Total appearances across all models
            row[
                f"top{k}_total_fold_count"
            ] = int(
                fold_counts.sum()
            )

            # Average number of folds per random seed
            row[
                f"top{k}_mean_fold_count"
            ] = (
                fold_counts.mean()
            )

            # Number of random-seed experiments
            # where ROI appeared at least once
            row[
                f"top{k}_random_seed_count"
            ] = int(
                (
                    fold_counts > 0
                ).sum()
            )

            row[
                f"top{k}_random_seed_ratio"
            ] = (
                (
                    fold_counts > 0
                ).mean()
            )

            # Number of random-seed experiments where
            # ROI appeared in >= 3/5 folds
            row[
                f"top{k}_majority_seed_count"
            ] = int(
                (
                    fold_counts >= 3
                ).sum()
            )

        rows.append(
            row
        )

    global_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # Global consensus ranking
    #
    # Primary:
    #   mean rank across random-seed 5CV experiments
    #
    # Tie:
    #   smaller SD
    # ========================================================

    global_df = global_df.sort_values(
        by=[
            "mean_rank_across_random_seeds",
            "rank_std_across_random_seeds"
        ],
        ascending=[
            True,
            True
        ]
    ).reset_index(
        drop=True
    )

    global_df.insert(
        0,
        "global_consensus_rank",
        np.arange(
            1,
            len(global_df) + 1
        )
    )

    return global_df


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def build_performance_summary(
    experiments
):

    rows = []

    for exp in experiments:

        aucs = []

        for fold_id in sorted(
            exp["folds"].keys()
        ):

            result = exp[
                "folds"
            ][fold_id]

            auc = result.get(
                "auc",
                np.nan
            )

            aucs.append(
                float(auc)
            )

        aucs = np.asarray(
            aucs,
            dtype=float
        )

        rows.append({

            "experiment":
                exp["name"],

            "random_seed":
                exp["random_seed"],

            "n_folds":
                len(aucs),

            "mean_auc":
                np.nanmean(
                    aucs
                ),

            "std_auc":
                np.nanstd(
                    aucs,
                    ddof=0
                ),

            "min_auc":
                np.nanmin(
                    aucs
                ),

            "max_auc":
                np.nanmax(
                    aucs
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    consensus_stability_df,
    same_fold_df,
    global_df,
    performance_df
):

    # ========================================================
    # Performance
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PERFORMANCE BY RANDOM SEED")
    print("=" * 80)

    print(
        performance_df.to_string(
            index=False
        )
    )

    # ========================================================
    # MAIN: 5CV consensus stability
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "5CV CONSENSUS STABILITY "
        "ACROSS RANDOM SEEDS"
    )

    print("=" * 80)

    if len(
        consensus_stability_df
    ) == 0:

        print(
            "At least 2 experiments are required."
        )

    else:

        rho = consensus_stability_df[
            "spearman_rho"
        ]

        print(
            "\nFull-ranking Spearman rho:"
        )

        print(
            f"  mean = {rho.mean():.4f}"
        )

        print(
            f"  SD   = {rho.std(ddof=0):.4f}"
        )

        print(
            f"  min  = {rho.min():.4f}"
        )

        print(
            f"  max  = {rho.max():.4f}"
        )

        for k in TOPKS:

            overlap = (
                consensus_stability_df[
                    f"top{k}_overlap"
                ]
            )

            ratio = (
                consensus_stability_df[
                    f"top{k}_overlap_ratio"
                ]
            )

            jac = (
                consensus_stability_df[
                    f"top{k}_jaccard"
                ]
            )

            print(
                f"\nTop-{k}"
            )

            print(
                f"  overlap: "
                f"{overlap.mean():.2f}/{k}"
            )

            print(
                f"  overlap ratio: "
                f"{ratio.mean():.3f}"
            )

            print(
                f"  Jaccard: "
                f"{jac.mean():.3f}"
            )

    # ========================================================
    # Same fold
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "SAME-FOLD STABILITY "
        "ACROSS RANDOM SEEDS"
    )

    print("=" * 80)

    if len(
        same_fold_df
    ) > 0:

        rho = same_fold_df[
            "spearman_rho"
        ]

        print(
            f"\nSpearman rho = "
            f"{rho.mean():.4f} "
            f"+/- "
            f"{rho.std(ddof=0):.4f}"
        )

        for k in TOPKS:

            overlap = same_fold_df[
                f"top{k}_overlap"
            ]

            jac = same_fold_df[
                f"top{k}_jaccard"
            ]

            print(
                f"Top-{k}: "
                f"overlap="
                f"{overlap.mean():.2f}/{k} | "
                f"Jaccard="
                f"{jac.mean():.3f}"
            )

    # ========================================================
    # Global consensus
    # ========================================================

    print("\n")
    print("=" * 80)
    print("GLOBAL CONSENSUS TOP-30")
    print("=" * 80)

    columns = [

        "global_consensus_rank",

        "roi",

        "mean_rank_across_random_seeds",

        "rank_std_across_random_seeds",

        "mean_weight_across_random_seeds",

        "top5_total_fold_count",

        "top10_total_fold_count",

        "top10_random_seed_count",

        "top20_total_fold_count",

        "top20_random_seed_count"
    ]

    print(
        global_df[
            columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze HiTraNet seed stability "
            "across multiple random-seed "
            "5-fold CV experiments."
        )
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help=(
            "Path to experiments/seed_stability"
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--n_rois",
        type=int,
        default=N_ROIS
    )

    args = parser.parse_args()

    root_dir = os.path.abspath(
        args.root_dir
    )

    # --------------------------------------------------------
    # Default output directory
    # --------------------------------------------------------

    if args.output_dir is None:

        output_dir = os.path.join(
            root_dir,
            "stability_analysis"
        )

    else:

        output_dir = os.path.abspath(
            args.output_dir
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\nRoot directory:")
    print(root_dir)

    print("\nOutput directory:")
    print(output_dir)

    # ========================================================
    # 1. Discover experiments
    # ========================================================

    experiments = (
        discover_experiments(
            root_dir
        )
    )

    if len(experiments) == 0:

        raise RuntimeError(
            "No valid experiments found."
        )

    if len(experiments) < 2:

        print(
            "\nWARNING: "
            "Only one experiment found. "
            "Random-seed stability cannot be "
            "calculated."
        )

    # ========================================================
    # 2. Build 5CV consensus for each random seed
    # ========================================================

    (
        fold_roi_df,
        experiment_roi_df,
        consensus_rankings
    ) = build_5cv_consensus(
        experiments,
        n_rois=args.n_rois,
        topks=TOPKS
    )

    # ========================================================
    # 3. MAIN:
    # Random-seed stability of 5CV consensus
    # ========================================================

    consensus_stability_df = (
        compare_5cv_consensus(
            consensus_rankings,
            n_rois=args.n_rois,
            topks=TOPKS
        )
    )

    # ========================================================
    # 4. Secondary:
    # same-fold random-seed stability
    # ========================================================

    same_fold_df = (
        compare_same_fold_across_random_seeds(
            experiments,
            n_rois=args.n_rois,
            topks=TOPKS
        )
    )

    # ========================================================
    # 5. Global consensus
    # ========================================================

    global_df = (
        build_global_consensus(
            experiment_roi_df,
            topks=TOPKS
        )
    )

    # ========================================================
    # 6. Performance
    # ========================================================

    performance_df = (
        build_performance_summary(
            experiments
        )
    )

    # ========================================================
    # 7. SAVE
    # ========================================================

    fold_roi_df.to_csv(
        os.path.join(
            output_dir,
            "01_all_fold_roi_rankings.csv"
        ),
        index=False
    )

    experiment_roi_df.to_csv(
        os.path.join(
            output_dir,
            "02_5cv_consensus_by_random_seed.csv"
        ),
        index=False
    )

    consensus_stability_df.to_csv(
        os.path.join(
            output_dir,
            "03_5cv_consensus_random_seed_stability.csv"
        ),
        index=False
    )

    same_fold_df.to_csv(
        os.path.join(
            output_dir,
            "04_same_fold_random_seed_stability.csv"
        ),
        index=False
    )

    global_df.to_csv(
        os.path.join(
            output_dir,
            "05_global_consensus_seeds.csv"
        ),
        index=False
    )

    performance_df.to_csv(
        os.path.join(
            output_dir,
            "06_performance_by_random_seed.csv"
        ),
        index=False
    )

    # ========================================================
    # 8. PRINT
    # ========================================================

    print_summary(
        consensus_stability_df,
        same_fold_df,
        global_df,
        performance_df
    )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"\nResults saved to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()