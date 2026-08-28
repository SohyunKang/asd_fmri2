import subprocess
import itertools
import json
import os
from datetime import datetime

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"
# ============================================================
# 기본 설정
# ============================================================

PYTHON = "python"
MAIN = "main.py"

# # 처음에는 너무 크게 돌리지 말 것
# SEARCH_SPACE = {
#     "d_model": [64, 96, 128],
#     "n_hops": [4, 6],
#     "top_m": [5, 8, 10],
#     "depth": [2, 3],
#     "dropout": [0.1, 0.2],
#     "lr": [5e-5, 1e-4, 3e-4, 1e-3],
#     "seed_lr": [1e-4, 3e-4, 1e-3],
#     "weight_decay": [1e-4],
#     "batch_size": [8, 16, 32]
# }
SEARCH_SPACE = {
    "random_seed": [0,10,20,30,40],
}

# main.py에서 사용하는 고정 옵션
FIXED_ARGS = {
    "epochs": 50,
    "n_splits": 5,
}


# ============================================================
# output
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tune_dir = os.path.join(
    "experiments",
    f"tuning_{timestamp}"
)

os.makedirs(tune_dir, exist_ok=True)

result_file = os.path.join(
    tune_dir,
    "tuning_results.json"
)


# ============================================================
# 모든 조합 생성
# ============================================================

keys = list(SEARCH_SPACE.keys())

combinations = list(
    itertools.product(
        *[SEARCH_SPACE[k] for k in keys]
    )
)

print("=" * 70)
print("Hyperparameter tuning")
print(f"Total combinations: {len(combinations)}")
print(f"Save dir: {tune_dir}")
print("=" * 70)


all_results = []


# ============================================================
# tuning
# ============================================================

for trial_idx, values in enumerate(combinations, start=1):

    params = dict(zip(keys, values))

    print("\n")
    print("=" * 70)
    print(
        f"[TRIAL {trial_idx}/{len(combinations)}]"
    )
    print(params)
    print("=" * 70)

    cmd = [
        PYTHON,
        MAIN,
    ]

    # fixed arguments
    for key, value in FIXED_ARGS.items():
        cmd.extend([
            f"--{key}",
            str(value)
        ])

    # tuning arguments
    for key, value in params.items():
        cmd.extend([
            f"--{key}",
            str(value)
        ])

    print(
        "[COMMAND]",
        " ".join(cmd)
    )

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1
        )

        output_lines = []

        # main.py 출력 실시간 표시
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"main.py failed with return code {return_code}"
            )

        stdout = "".join(output_lines)

        # =========================================
        # CV AUC parsing
        # =========================================
        auc_mean = None
        auc_std = None

        for line in output_lines:
            if "AUC  Mean ± Std:" in line:
                tmp = line.replace(
                    "AUC  Mean ± Std:", ""
                ).strip()

                mean_str, std_str = tmp.split("±")

                auc_mean = float(mean_str.strip())
                auc_std = float(std_str.strip())

                break

        trial_result = {
            "trial": trial_idx,
            **params,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "status": "success",
        }

        print(
            f"\n>>> TRIAL {trial_idx} RESULT: "
            f"AUC={auc_mean:.4f} ± {auc_std:.4f}"
        )

    except Exception as e:

        print(f"\n[ERROR] Trial {trial_idx}")
        print(e)

        trial_result = {
            "trial": trial_idx,
            **params,
            "auc_mean": None,
            "auc_std": None,
            "status": "failed",
        }

    all_results.append(trial_result)

    # --------------------------------------------------------
    # 매 trial마다 저장
    # 중간에 프로그램 죽어도 결과 보존
    # --------------------------------------------------------

    with open(
        result_file,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )


# ============================================================
# ranking
# ============================================================

valid_results = [
    r for r in all_results
    if r["auc_mean"] is not None
]

valid_results.sort(
    key=lambda x: x["auc_mean"],
    reverse=True
)


print("\n\n")
print("=" * 70)
print("TUNING FINISHED")
print("=" * 70)

for rank, result in enumerate(
    valid_results[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"AUC={result['auc_mean']:.4f} "
        f"± {result['auc_std']:.4f} | "
        f"{ {k: result[k] for k in keys} }"
    )


# ranked result 별도 저장
ranked_file = os.path.join(
    tune_dir,
    "tuning_results_ranked.json"
)

with open(
    ranked_file,
    "w"
) as f:

    json.dump(
        valid_results,
        f,
        indent=4
    )

print(
    f"\nSaved: {ranked_file}"
)