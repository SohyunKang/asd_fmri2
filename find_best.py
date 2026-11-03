import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 설정
# ============================================================
ROOT_DIR = Path("/data5/sohyunkang/experiments2")
RESULT_FILENAME = "cv_results.json"

# 순위를 결정할 주요 지표
RANK_METRIC = "auc"

# 표본 표준편차
STD_DDOF = 1

# args.json에서 출력하고 비교할 hyperparameter
HYPERPARAMETER_KEYS = [
    "d_model",
    "n_hops",
    "depth",
    "top_m",
    "seed_std",
    "mlp_hidden",
    "dropout",
    "act",
    "fc_norm",
    "lr",
    "seed_lr",
    "weight_decay",
    "batch_size",
    "epochs",
    "patience",
    "inner_val_size",
    "n_splits",
]

# 같은 폴더에서 찾을 설정 파일
CONFIG_FILENAMES = [
    "args.json",
    "config.json",
    "hyperparameters.json",
]


# ============================================================
# metric 이름 대응
# ============================================================
METRIC_KEYS = {
    "auc": {
        "list": ["auc_per_fold"],
        "mean": ["auc_mean"],
        "fold": ["auc", "best_auc"],
    },
    "acc": {
        "list": [
            "acc_per_fold",
            "accuracy_per_fold",
        ],
        "mean": [
            "acc_mean",
            "accuracy_mean",
        ],
        "fold": [
            "acc",
            "accuracy",
            "best_acc",
            "best_accuracy",
        ],
    },
    "sensitivity": {
        "list": [
            "sensitivity_per_fold",
            "sens_per_fold",
        ],
        "mean": [
            "sensitivity_mean",
            "sens_mean",
        ],
        "fold": [
            "sensitivity",
            "sens",
            "best_sensitivity",
            "best_sens",
        ],
    },
    "specificity": {
        "list": [
            "specificity_per_fold",
            "spec_per_fold",
        ],
        "mean": [
            "specificity_mean",
            "spec_mean",
        ],
        "fold": [
            "specificity",
            "spec",
            "best_specificity",
            "best_spec",
        ],
    },
    "f1": {
        "list": ["f1_per_fold"],
        "mean": ["f1_mean"],
        "fold": ["f1", "best_f1"],
    },
}


def first_existing(data, keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return None


def extract_from_fold_records(data, fold_keys):
    """fold_summaries와 folds 형식 모두 지원."""
    records = data.get("fold_summaries")

    if records is None:
        records = data.get("folds")

    if not records:
        return None

    values = []

    for record in records:
        value = first_existing(record, fold_keys)

        if value is not None:
            values.append(float(value))

    return values if values else None


def extract_metric(data, metric):
    keys = METRIC_KEYS[metric]

    # 1. *_per_fold
    values = first_existing(data, keys["list"])

    if values is not None:
        values = np.asarray(values, dtype=float)

        return {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(
                np.std(values, ddof=STD_DDOF)
                if len(values) > 1
                else 0.0
            ),
        }

    # 2. fold_summaries 또는 folds
    values = extract_from_fold_records(
        data,
        keys["fold"],
    )

    if values is not None:
        values = np.asarray(values, dtype=float)

        return {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(
                np.std(values, ddof=STD_DDOF)
                if len(values) > 1
                else 0.0
            ),
        }

    # 3. mean만 존재
    mean_value = first_existing(
        data,
        keys["mean"],
    )

    if mean_value is not None:
        return {
            "values": None,
            "mean": float(mean_value),
            "std": np.nan,
        }

    return {
        "values": None,
        "mean": np.nan,
        "std": np.nan,
    }


# ============================================================
# Hyperparameter 추출
# ============================================================
def load_hyperparameters(json_path, cv_data):
    """
    우선순위:
      1. cv_results.json 내부의 args/config/hyperparameters
      2. 같은 폴더의 args.json/config.json/hyperparameters.json
    """

    for key in ["args", "config", "hyperparameters"]:
        value = cv_data.get(key)

        if isinstance(value, dict):
            return value

    for filename in CONFIG_FILENAMES:
        config_path = json_path.parent / filename

        if not config_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if isinstance(config, dict):
                return config

        except (json.JSONDecodeError, OSError) as error:
            print(
                f"[WARNING] Cannot read {config_path}: {error}"
            )

    return {}


def make_dataframe_safe(value):
    """list/dict hyperparameter를 CSV에 저장할 수 있게 변환."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return value


# ============================================================
# JSON 검색
# ============================================================
json_files = sorted(
    ROOT_DIR.glob(f"*/{RESULT_FILENAME}")
)

if not json_files:
    json_files = sorted(
        ROOT_DIR.rglob(RESULT_FILENAME)
    )

if not json_files:
    raise FileNotFoundError(
        f"No {RESULT_FILENAME} found under "
        f"{ROOT_DIR.resolve()}"
    )

print(f"Found {len(json_files)} result files")


# ============================================================
# run별 결과와 hyperparameter 추출
# ============================================================
run_rows = []

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hyperparameters = load_hyperparameters(
        json_path,
        data,
    )

    try:
        run_name = str(
            json_path.parent.relative_to(ROOT_DIR)
        )
    except ValueError:
        run_name = json_path.parent.name

    row = {
        "run": run_name,
        "path": str(json_path),
        "random_seed": first_existing(
            hyperparameters,
            ["random_seed", "seed"],
        ),
    }

    # Hyperparameter 저장
    for key in HYPERPARAMETER_KEYS:
        row[key] = make_dataframe_safe(
            hyperparameters.get(key, np.nan)
        )

    # Metric 저장
    for metric in METRIC_KEYS:
        result = extract_metric(data, metric)

        row[f"{metric}_mean"] = result["mean"]
        row[f"{metric}_fold_std"] = result["std"]

    run_rows.append(row)


run_df = pd.DataFrame(run_rows)


# ============================================================
# 각 metric별 순위 계산
# ============================================================
for metric in METRIC_KEYS:
    mean_column = f"{metric}_mean"
    rank_column = f"{metric}_rank"

    run_df[rank_column] = (
        run_df[mean_column]
        .rank(
            method="min",
            ascending=False,
            na_option="keep",
        )
        .astype("Int64")
    )


# 주요 지표(AUC) 기준 정렬
run_df = (
    run_df
    .sort_values(
        by=[
            f"{RANK_METRIC}_rank",
            f"{RANK_METRIC}_mean",
        ],
        ascending=[True, False],
        na_position="last",
    )
    .reset_index(drop=True)
)


# ============================================================
# run 순위 출력
# ============================================================
available_hyperparameters = [
    key
    for key in HYPERPARAMETER_KEYS
    if key in run_df.columns
    and run_df[key].notna().any()
]

display_columns = [
    f"{RANK_METRIC}_rank",
    "run",
    "random_seed",
]

display_columns.extend(
    available_hyperparameters
)

for metric in METRIC_KEYS:
    display_columns.extend([
        f"{metric}_mean",
        f"{metric}_fold_std",
        f"{metric}_rank",
    ])


print(
    f"\n========== Per-run ranking "
    f"(sorted by {RANK_METRIC.upper()}) =========="
)

print(
    run_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 동일 hyperparameter끼리 seed 결과 집계
# ============================================================
group_keys = available_hyperparameters

config_rows = []

if group_keys:
    grouped = run_df.groupby(
        group_keys,
        dropna=False,
        sort=False,
    )

    for config_values, group in grouped:
        if len(group_keys) == 1:
            config_values = (config_values,)

        row = {
            key: value
            for key, value in zip(
                group_keys,
                config_values,
            )
        }

        row["n_runs"] = len(group)
        row["seeds"] = ",".join(
            group["random_seed"]
            .dropna()
            .astype(int)
            .astype(str)
            .tolist()
        )

        for metric in METRIC_KEYS:
            values = (
                group[f"{metric}_mean"]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) == 0:
                metric_mean = np.nan
                metric_std = np.nan

            else:
                metric_mean = float(
                    np.mean(values)
                )

                metric_std = float(
                    np.std(values, ddof=STD_DDOF)
                    if len(values) > 1
                    else 0.0
                )

            row[f"{metric}_repeated_mean"] = metric_mean
            row[f"{metric}_repeated_std"] = metric_std

        config_rows.append(row)


config_df = pd.DataFrame(config_rows)

if not config_df.empty:
    # hyperparameter configuration 순위
    for metric in METRIC_KEYS:
        mean_column = f"{metric}_repeated_mean"
        rank_column = f"{metric}_rank"

        config_df[rank_column] = (
            config_df[mean_column]
            .rank(
                method="min",
                ascending=False,
                na_option="keep",
            )
            .astype("Int64")
        )

    config_df = (
        config_df
        .sort_values(
            by=[
                f"{RANK_METRIC}_rank",
                f"{RANK_METRIC}_repeated_mean",
            ],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    config_display_columns = [
        f"{RANK_METRIC}_rank",
        "n_runs",
        "seeds",
    ]

    config_display_columns.extend(
        group_keys
    )

    for metric in METRIC_KEYS:
        config_display_columns.extend([
            f"{metric}_repeated_mean",
            f"{metric}_repeated_std",
            f"{metric}_rank",
        ])

    print(
        f"\n========== Hyperparameter ranking "
        f"(sorted by {RANK_METRIC.upper()}) =========="
    )

    print(
        config_df[
            config_display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

else:
    print(
        "\n[WARNING] Hyperparameter information was not found."
    )


# ============================================================
# 결과 저장
# ============================================================
run_output = (
    ROOT_DIR
    / "ranked_cv_runs.csv"
)

config_output = (
    ROOT_DIR
    / "ranked_hyperparameter_summary.csv"
)

run_df.to_csv(
    run_output,
    index=False,
)

if not config_df.empty:
    config_df.to_csv(
        config_output,
        index=False,
    )


print(f"\nSaved: {run_output}")

if not config_df.empty:
    print(f"Saved: {config_output}")