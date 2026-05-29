import json
import argparse
import subprocess
import glob
import os
import time


# -------------------------
# args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, required=True)
parser.add_argument("--n_workers", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()


# -------------------------
# load configs
# -------------------------
with open("search_configs2.json") as f:
    configs = json.load(f)

SEARCH_KEYS = list(configs[0].keys())


def normalize(v):
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"   # float 안정화
    return str(v)


def is_same_config(p1, p2):

    for k in SEARCH_KEYS:

        v1 = normalize(p1.get(k))
        v2 = normalize(p2.get(k))

        if v1 != v2:
            print(f"❌ mismatch: {k} | {v1} vs {v2}")
            return False

    return True

# -------------------------
# load existing experiments
# -------------------------
def load_existing_configs(exp_root="/data5/sohyunkang/experiments"):

    print("\n🔍 Scanning experiments in:", exp_root)

    existing = []

    exp_dirs = glob.glob(os.path.join(exp_root, "*"))

    print(f"Found {len(exp_dirs)} directories")

    for d in exp_dirs:

        args_path = os.path.join(d, "args.json")

        if not os.path.exists(args_path):
            print("❌ No args.json in:", d)
            continue

        print("✅ Found args.json:", args_path)

        try:
            with open(args_path) as f:
                saved = json.load(f)

            cleaned = {
            k: saved.get(k, None)   # 🔥 없는 것도 None으로 맞춤
            for k in SEARCH_KEYS
        }

            existing.append(cleaned)

        except Exception as e:
            print(f"⚠️ Failed to read {args_path}: {e}")

    print(f"\n🔥 Loaded {len(existing)} existing configs")
    return existing




# -------------------------
# main loop
# -------------------------
best_auc = -1.0
best_exp = None
best_params = None

existing_configs = load_existing_configs()

for i, params in enumerate(configs):

    
    if i % args.n_workers != args.worker_id:
        continue

    # -------------------------
    # 🔥 skip duplicate
    # -------------------------
    skip = False

    for ex in existing_configs:
        if is_same_config(params, ex):
            print("\n🚫 Skipping already executed config")
            print("Params:", params)
            skip = True
            break   # 🔥 중요 (더 볼 필요 없음)

    if skip:
        continue

    # -------------------------
    # run command
    # -------------------------
    cmd = [
        "python3", "main.py",
        "--csv", "./data/abide_subjects_fc_with_demo_260202.xlsx",
        "--aal_txt", "data/ROI_MNI_V4.txt",
        "--n_rois", "116",
        "--n_classes", "2",
        "--epochs", "350",
        "--fc_norm", "row",
        "--worker_id", str(args.worker_id),
    ]

    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    print("\n=================================")
    print(f"Worker {args.worker_id} running config #{i}")
    print("Params:", params)
    print("=================================")

    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={args.gpu} " + " ".join(cmd),
        shell=True,
        check=False
    )

    time.sleep(1)

    # -------------------------
    # 결과 로드
    # -------------------------
    exp_dirs = sorted(
        glob.glob(f"/data5/sohyunkang/experiments/w{args.worker_id}_hitranet_*"),
        key=os.path.getmtime
    )

    if not exp_dirs:
        print(f"[Worker {args.worker_id}] No experiment folder found.")
        continue

    latest_exp = exp_dirs[-1]
    cv_file = os.path.join(latest_exp, "cv_results.json")

    if not os.path.exists(cv_file):
        print(f"[Worker {args.worker_id}] No cv_results.json in {latest_exp}")
        continue

    with open(cv_file) as f:
        results = json.load(f)

    auc = results["auc_mean"]

    print("\nTrial finished")
    print("Experiment:", latest_exp)
    print("CV AUC:", auc)

    # -------------------------
    # best 갱신
    # -------------------------
    if auc > best_auc:
        best_auc = auc
        best_exp = latest_exp
        best_params = params

        print("\n🔥 NEW BEST RESULT 🔥")
        print("Best CV AUC:", best_auc)
        print("Best Experiment:", best_exp)
        print("Best Params:", best_params)

        best_save = {
            "worker_id": args.worker_id,
            "best_auc": best_auc,
            "best_experiment": best_exp,
            "best_params": best_params,
        }

        with open(f"best_worker{args.worker_id}.json", "w") as f:
            json.dump(best_save, f, indent=2)

    else:
        print("\nCurrent Best CV AUC:", best_auc)
        print("Best Experiment:", best_exp)


# -------------------------
# final
# -------------------------
print("\n=================================")
print(f"Worker {args.worker_id} finished all assigned jobs.")
print("Final Best CV AUC:", best_auc)
print("Final Best Experiment:", best_exp)
print("Final Best Params:", best_params)
print("=================================")