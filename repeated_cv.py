import json
from pathlib import Path

import numpy as np
import pandas as pd
import shutil


# ============================================================
# 설정
# ============================================================
ROOT_DIR = Path("/data5/sohyunkang/experiments2")
RESULT_FILENAME = "cv_results.json"

# 표본 표준편차
STD_DDOF = 1

# ============================================================
# 낮은 AUC 실험 폴더 삭제 설정
# ============================================================
DELETE_LOW_AUC = True
AUC_THRESHOLD = 0.69

# True: 삭제하지 않고 대상만 출력
# False: 실제 폴더 삭제
DRY_RUN = False


# ============================================================
# 여러 결과 형식에서 metric 추출
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
        "list": ["sensitivity_per_fold", "sens_per_fold"],
        "mean": ["sensitivity_mean", "sens_mean"],
        "fold": [
            "sensitivity",
            "sens",
            "best_sensitivity",
            "best_sens",
        ],
    },
    "specificity": {
        "list": ["specificity_per_fold", "spec_per_fold"],
        "mean": ["specificity_mean", "spec_mean"],
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
    """
    다음 두 형식을 모두 지원:
      data["fold_summaries"]
      data["folds"]
    """
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

    # 1. *_per_fold가 있으면 우선 사용
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

    # 2. fold_summaries 또는 folds에서 추출
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

    # 3. mean만 있는 경우
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
# JSON 파일 검색
# ============================================================
json_files = sorted(
    ROOT_DIR.glob(f"*/{RESULT_FILENAME}")
)

if not json_files:
    # 폴더가 한 단계 이상 중첩된 경우
    json_files = sorted(
        ROOT_DIR.rglob(RESULT_FILENAME)
    )

if not json_files:
    raise FileNotFoundError(
        f"No {RESULT_FILENAME} found under {ROOT_DIR.resolve()}"
    )

print(f"Found {len(json_files)} result files")


# ============================================================
# seed/run별 결과
# ============================================================
run_rows = []
pooled_fold_values = {
    metric: []
    for metric in METRIC_KEYS
}

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {
        "run": json_path.parent.name,
        "path": str(json_path),
    }

    for metric in METRIC_KEYS:
        result = extract_metric(data, metric)

        row[f"{metric}_mean"] = result["mean"]
        row[f"{metric}_fold_std"] = result["std"]

        if result["values"] is not None:
            pooled_fold_values[metric].extend(
                result["values"].tolist()
            )

    run_rows.append(row)


run_df = pd.DataFrame(run_rows)

# ============================================================
# AUC threshold 이하 실험 폴더 삭제
# ============================================================
if DELETE_LOW_AUC:

    print("\n========== Low-AUC experiment folders ==========")
    print(f"Threshold: AUC <= {AUC_THRESHOLD:.4f}")
    print(f"DRY_RUN: {DRY_RUN}")

    delete_count = 0
    skip_count = 0

    for _, row in run_df.iterrows():

        auc_mean = row["auc_mean"]
        json_path = Path(row["path"])
        experiment_dir = json_path.parent

        # AUC가 없는 경우 삭제하지 않음
        if pd.isna(auc_mean):
            print(
                f"[SKIP] {experiment_dir.name} "
                f"(AUC unavailable)"
            )
            skip_count += 1
            continue

        # threshold 이하만 삭제
        if auc_mean <= AUC_THRESHOLD:

            if DRY_RUN:
                print(
                    f"[DELETE TARGET] "
                    f"{experiment_dir} "
                    f"| AUC={auc_mean:.4f}"
                )

            else:
                print(
                    f"[DELETE] "
                    f"{experiment_dir} "
                    f"| AUC={auc_mean:.4f}"
                )

                try:
                    shutil.rmtree(experiment_dir)
                    delete_count += 1

                except Exception as e:
                    print(
                        f"  -> ERROR: {e}"
                    )

    if DRY_RUN:
        print(
            "\n[DRY RUN] No folders were actually deleted."
        )
    else:
        print(
            f"\nDeleted {delete_count} experiment folders."
        )

print("\n========== Per-run CV results ==========")

display_columns = ["run"]

for metric in METRIC_KEYS:
    display_columns.extend([
        f"{metric}_mean",
        f"{metric}_fold_std",
    ])

print(
    run_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 반복 실행(seed) 평균과 표준편차
# ============================================================
summary_rows = []

for metric in METRIC_KEYS:
    run_means = (
        run_df[f"{metric}_mean"]
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(run_means) == 0:
        continue

    repeated_mean = float(np.mean(run_means))

    repeated_std = float(
        np.std(run_means, ddof=STD_DDOF)
        if len(run_means) > 1
        else 0.0
    )

    pooled_values = np.asarray(
        pooled_fold_values[metric],
        dtype=float,
    )

    pooled_mean = (
        float(np.mean(pooled_values))
        if len(pooled_values) > 0
        else np.nan
    )

    pooled_std = (
        float(np.std(pooled_values, ddof=STD_DDOF))
        if len(pooled_values) > 1
        else 0.0
    )

    summary_rows.append({
        "metric": metric.upper(),

        # 논문 보고에 권장:
        # 각 seed의 5-fold 평균을 구한 뒤 seed 간 mean/std
        "repeated_mean": repeated_mean,
        "repeated_std": repeated_std,
        "n_runs": len(run_means),

        # 참고용: 모든 seed×fold를 하나로 합친 값
        "pooled_fold_mean": pooled_mean,
        "pooled_fold_std": pooled_std,
        "n_fold_results": len(pooled_values),
    })


summary_df = pd.DataFrame(summary_rows)

print("\n========== Repeated-CV summary ==========")

for _, row in summary_df.iterrows():
    print(
        f"{row['metric']:12s}: "
        f"{row['repeated_mean']:.4f} "
        f"± {row['repeated_std']:.4f} "
        f"(runs={int(row['n_runs'])})"
    )


# ============================================================
# 결과 저장
# ============================================================
run_output = ROOT_DIR / "repeated_cv_per_run.csv"
summary_output = ROOT_DIR / "repeated_cv_summary.csv"

run_df.to_csv(run_output, index=False)
summary_df.to_csv(summary_output, index=False)

print(f"\nSaved: {run_output}")
print(f"Saved: {summary_output}")