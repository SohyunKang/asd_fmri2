import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# 기본 설정
# ============================================================

# 현재 활성화된 conda/venv의 Python을 그대로 사용
PYTHON = sys.executable
MAIN = Path("main3.py")

# 매번 같은 폴더를 사용한다. 이 폴더의 기존 결과를 불러와 이어서 실행한다.
TUNE_DIR = Path("experiments") / "random_tuning6"

# 전체 후보 중 무작위로 실행할 조합 수
# 나중에 이 값을 늘리면 기존 조합 뒤에 새로운 조합만 추가된다.
N_RANDOM_TRIALS = 100

# hyperparameter 조합 추출을 재현하기 위한 seed
# main3.py의 학습 random_seed와는 별개임
SEARCH_RANDOM_SEED = 2026

# True: 이전에 실패하거나 중단된 trial은 다음 실행에서 다시 시도
# False: 실패한 trial도 건너뜀
RETRY_FAILED = True


# ============================================================
# Random search 후보 공간
# ============================================================

SEARCH_SPACE = {
#     "n_hops": [3, 4, 5],
#     "depth": [2, 3, 4],
#     # "top_m": [3, 5, 7, 10],
#     "lr": [1e-4, 3e-4],
#     "seed_lr": [1e-4, 3e-4],
#     "batch_size": [8, 16, 32, 64],
    "random_seed": list(range(1, 1001)),
}

# 모든 trial에 공통으로 적용할 main3.py 옵션
# 공정한 hyperparameter 비교를 위해 학습 seed는 동일하게 유지
FIXED_ARGS = {
    "epochs": 30,
    "n_hops": 3,
    "depth": 2,
    "top_m": 5,
    "lr": 1e-4,
    "seed_lr": 1e-4,
    "batch_size": 8,
    "d_model": 64,
    "mlp_hidden": 64,
    "dropout": 0.4,
    "weight_decay": 0.001,
    # "random_seed": 18,
}


# ============================================================
# 보조 함수
# ============================================================

def build_all_combinations(search_space):
    """SEARCH_SPACE에서 가능한 모든 고유 조합을 생성한다."""
    keys = list(search_space.keys())
    values = itertools.product(*(search_space[key] for key in keys))
    combinations = [dict(zip(keys, row)) for row in values]
    return keys, combinations


def sample_random_combinations(all_combinations, n_trials, random_seed):
    """전체 조합을 재현 가능하게 섞은 뒤 앞에서 n_trials개를 선택한다."""
    if n_trials <= 0:
        raise ValueError("N_RANDOM_TRIALS must be greater than 0")

    rng = random.Random(random_seed)
    n_selected = min(n_trials, len(all_combinations))
    shuffled = [dict(params) for params in all_combinations]
    rng.shuffle(shuffled)

    return shuffled[:n_selected]


def params_key(params, keys):
    """Hyperparameter 딕셔너리를 비교 가능한 고유 key로 변환한다."""
    return tuple((key, params[key]) for key in keys)


def make_command(params):
    """main3.py 실행 명령을 생성한다."""
    cmd = [PYTHON, str(MAIN)]

    for key, value in FIXED_ARGS.items():
        cmd.extend([f"--{key}", str(value)])

    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    return cmd


def parse_auc_summary(output_lines):
    """main3.py 출력에서 5-fold AUC mean/std를 추출한다."""
    pattern = re.compile(
        r"AUC\s+Mean\s*[±+/-]+\s*Std:\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
        r"[±+/-]+\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    for line in output_lines:
        match = pattern.search(line)
        if match:
            return float(match.group(1)), float(match.group(2))

    return None, None


def save_json(path, data):
    """JSON 결과를 안전하게 갱신한다."""
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(temp_path, path)


def load_json(path, default):
    """파일이 없으면 default를 반환하고, 있으면 JSON을 읽는다."""
    path = Path(path)

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_extend_sampled_combinations(
    sampled_file,
    all_combinations,
    keys,
):
    """
    이전 sampled_hyperparameters.json을 우선 사용한다.

    N_RANDOM_TRIALS가 증가한 경우에만 아직 선택되지 않은 조합을
    동일한 random 순서에서 추가한다. 따라서 재실행 또는 trial 수 증가 시
    기존 조합의 순서와 번호가 바뀌지 않는다.
    """
    manifest = load_json(sampled_file, default=None)

    random_order = sample_random_combinations(
        all_combinations=all_combinations,
        n_trials=len(all_combinations),
        random_seed=SEARCH_RANDOM_SEED,
    )

    if manifest is None:
        selected = []
    else:
        if manifest.get("search_space") != SEARCH_SPACE:
            raise ValueError(
                "SEARCH_SPACE differs from the existing random_tuning run. "
                "Use a different TUNE_DIR for a new search space."
            )

        if manifest.get("fixed_args") != FIXED_ARGS:
            raise ValueError(
                "FIXED_ARGS differs from the existing random_tuning run. "
                "Use a different TUNE_DIR for a new experiment."
            )

        if manifest.get("search_random_seed") != SEARCH_RANDOM_SEED:
            raise ValueError(
                "SEARCH_RANDOM_SEED differs from the existing random_tuning "
                "run. Restore the previous seed or use a different TUNE_DIR."
            )

        selected = manifest.get("sampled_combinations", [])

    valid_keys = {
        params_key(params, keys)
        for params in all_combinations
    }
    selected_keys = set()
    cleaned_selected = []

    for params in selected:
        try:
            key = params_key(params, keys)
        except (KeyError, TypeError):
            continue

        if key in valid_keys and key not in selected_keys:
            cleaned_selected.append(params)
            selected_keys.add(key)

    # 기존 목록은 유지하고, 목표 trial 수가 커졌을 때만 새 조합을 추가한다.
    target_count = min(N_RANDOM_TRIALS, len(all_combinations))

    for params in random_order:
        if len(cleaned_selected) >= target_count:
            break

        key = params_key(params, keys)
        if key not in selected_keys:
            cleaned_selected.append(params)
            selected_keys.add(key)

    return cleaned_selected


def ordered_results(sampled_combinations, results_by_key, keys):
    """선택된 trial 순서대로 현재 결과를 정렬한다."""
    results = []

    for params in sampled_combinations:
        key = params_key(params, keys)
        if key in results_by_key:
            results.append(results_by_key[key])

    return results


# ============================================================
# Main
# ============================================================

def main():
    if not MAIN.exists():
        raise FileNotFoundError(
            f"Cannot find training script: {MAIN.resolve()}"
        )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    tune_dir = TUNE_DIR
    tune_dir.mkdir(parents=True, exist_ok=True)

    result_file = tune_dir / "tuning_results.json"
    ranked_file = tune_dir / "tuning_results_ranked.json"
    sampled_file = tune_dir / "sampled_hyperparameters.json"

    keys, all_combinations = build_all_combinations(SEARCH_SPACE)
    sampled_combinations = get_or_extend_sampled_combinations(
        sampled_file=sampled_file,
        all_combinations=all_combinations,
        keys=keys,
    )

    # 실행 전에 어떤 조합이 선택되었는지 저장
    save_json(sampled_file, {
        "search_random_seed": SEARCH_RANDOM_SEED,
        "requested_trials": N_RANDOM_TRIALS,
        "total_possible_combinations": len(all_combinations),
        "selected_trials": len(sampled_combinations),
        "fixed_args": FIXED_ARGS,
        "search_space": SEARCH_SPACE,
        "sampled_combinations": sampled_combinations,
    })

    print("=" * 70)
    print("Random hyperparameter search")
    print(f"Total possible combinations: {len(all_combinations)}")
    print(f"Randomly selected trials: {len(sampled_combinations)}")
    print(f"Search random seed: {SEARCH_RANDOM_SEED}")
    print(f"Training random seed: {FIXED_ARGS.get('random_seed')}")
    print(f"Save dir: {tune_dir}")
    print("=" * 70)

    previous_results = load_json(result_file, default=[])

    if not isinstance(previous_results, list):
        raise ValueError(f"Invalid result format: {result_file}")

    results_by_key = {}

    for result in previous_results:
        try:
            key = params_key(result, keys)
        except (KeyError, TypeError):
            continue

        # 동일 조합의 기록이 여러 개 있으면 JSON에서 뒤쪽 기록을 사용한다.
        results_by_key[key] = result

    completed_count = sum(
        results_by_key.get(params_key(params, keys), {}).get("status") == "success"
        for params in sampled_combinations
    )

    print(
        f"Resume status: {completed_count}/{len(sampled_combinations)} "
        "successful trials already completed"
    )

    for trial_idx, params in enumerate(sampled_combinations, start=1):
        key = params_key(params, keys)
        previous = results_by_key.get(key)

        if previous is not None and previous.get("status") == "success":
            print(
                f"[SKIP {trial_idx}/{len(sampled_combinations)}] "
                f"already completed | AUC={previous['auc_mean']:.4f} "
                f"± {previous['auc_std']:.4f} | {params}"
            )
            continue

        if (
            previous is not None
            and previous.get("status") in {"failed", "interrupted"}
            and not RETRY_FAILED
        ):
            print(
                f"[SKIP {trial_idx}/{len(sampled_combinations)}] "
                f"previous status={previous.get('status')} | {params}"
            )
            continue

        print("\n")
        print("=" * 70)
        print(f"[TRIAL {trial_idx}/{len(sampled_combinations)}]")
        print(params)
        print("=" * 70)

        cmd = make_command(params)
        print("[COMMAND]", " ".join(cmd), flush=True)

        attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
        trial_log_path = tune_dir / (
            f"trial_{trial_idx:03d}_attempt_{attempt:02d}.log"
        )
        start_time = time.time()
        process = None

        running_result = {
            "trial": trial_idx,
            **params,
            "auc_mean": None,
            "auc_std": None,
            "elapsed_sec": 0.0,
            "return_code": None,
            "attempt": attempt,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "log_path": str(trial_log_path),
            "status": "running",
        }
        results_by_key[key] = running_result
        save_json(
            result_file,
            ordered_results(sampled_combinations, results_by_key, keys),
        )

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )

            output_lines = []

            if process.stdout is None:
                raise RuntimeError("Failed to capture main3.py output")

            # main3.py 출력을 실시간으로 표시하면서 trial 로그도 저장
            with open(trial_log_path, "w", encoding="utf-8") as log_f:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_f.write(line)
                    log_f.flush()
                    output_lines.append(line)

            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(
                    f"main3.py failed with return code {return_code}"
                )

            auc_mean, auc_std = parse_auc_summary(output_lines)

            if auc_mean is None or auc_std is None:
                raise ValueError(
                    "Training finished, but 'AUC Mean ± Std' was not found "
                    "in the output. Check the trial log."
                )

            elapsed_sec = time.time() - start_time

            trial_result = {
                "trial": trial_idx,
                **params,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "elapsed_sec": elapsed_sec,
                "return_code": return_code,
                "attempt": attempt,
                "started_at": running_result["started_at"],
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "log_path": str(trial_log_path),
                "status": "success",
            }

            print(
                f"\n>>> TRIAL {trial_idx} RESULT: "
                f"AUC={auc_mean:.4f} ± {auc_std:.4f} | "
                f"time={elapsed_sec / 60:.1f} min"
            )

        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()

                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            print("\n[INTERRUPTED] Random search stopped by user")

            interrupted_result = {
                **running_result,
                "elapsed_sec": time.time() - start_time,
                "return_code": process.returncode if process is not None else None,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "status": "interrupted",
            }
            results_by_key[key] = interrupted_result
            save_json(
                result_file,
                ordered_results(sampled_combinations, results_by_key, keys),
            )
            raise

        except Exception as error:
            elapsed_sec = time.time() - start_time

            print(f"\n[ERROR] Trial {trial_idx}")
            print(error)

            return_code = (
                process.returncode
                if process is not None and process.returncode is not None
                else None
            )

            trial_result = {
                "trial": trial_idx,
                **params,
                "auc_mean": None,
                "auc_std": None,
                "elapsed_sec": elapsed_sec,
                "return_code": return_code,
                "attempt": attempt,
                "started_at": running_result["started_at"],
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "log_path": str(trial_log_path),
                "status": "failed",
                "error": str(error),
            }

        results_by_key[key] = trial_result
        all_results = ordered_results(
            sampled_combinations,
            results_by_key,
            keys,
        )

        # 매 trial마다 저장하여 중간 종료 시에도 결과를 보존
        save_json(result_file, all_results)

        valid_results = [
            result
            for result in all_results
            if result["auc_mean"] is not None
        ]
        valid_results.sort(
            key=lambda result: result["auc_mean"],
            reverse=True,
        )
        save_json(ranked_file, valid_results)

    all_results = ordered_results(
        sampled_combinations,
        results_by_key,
        keys,
    )

    valid_results = [
        result
        for result in all_results
        if result["auc_mean"] is not None
    ]
    valid_results.sort(
        key=lambda result: result["auc_mean"],
        reverse=True,
    )

    print("\n\n")
    print("=" * 70)
    print("RANDOM SEARCH FINISHED")
    print("=" * 70)

    if not valid_results:
        print("No successful trial was found.")
    else:
        for rank, result in enumerate(valid_results[:10], start=1):
            hyperparameters = {
                key: result[key]
                for key in keys
            }

            print(
                f"{rank:2d}. "
                f"AUC={result['auc_mean']:.4f} "
                f"± {result['auc_std']:.4f} | "
                f"{hyperparameters}"
            )

    save_json(ranked_file, valid_results)

    print(f"\nSaved all results: {result_file}")
    print(f"Saved ranked results: {ranked_file}")
    print(f"Saved sampled parameters: {sampled_file}")


if __name__ == "__main__":
    main()
