import subprocess
import random
import json
import glob
import os
import time

N_TRIALS = 60

search_space = {
    "d_model": [96, 128],
    "n_hops": [5, 6, 7],
    "depth": [2, 3],
    "top_m": [6, 8, 10],
    "seed_std": [0.3, 0.5, 0.7],
    "mlp_hidden": [128, 256],
    "dropout": [0.05, 0.1, 0.2],
    "lr": [1e-4, 3e-4, 5e-4],
    "seed_lr": [1e-3, 3e-3, 1e-2],
    "batch_size": [8, 16, 32, 64, 128],
}

BASE_CMD = [
    "python3", "main.py",
    "--csv", "./data/abide_subjects_fc_with_demo_260202.xlsx",
    "--aal_txt", "data/ROI_MNI_V4.txt",
    "--n_rois", "116",
    "--n_classes", "2",
    "--epochs", "150",
    "--fc_norm", "row"
]

best_auc = -1
best_config = None
best_exp = None

for trial in range(N_TRIALS):

    params = {k: random.choice(v) for k, v in search_space.items()}

    cmd = ["CUDA_VISIBLE_DEVICES=2"] + BASE_CMD.copy()

    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    print("\n=================================")
    print(f"Trial {trial+1}/{N_TRIALS}")
    print("Params:", params)
    print("=================================")

    subprocess.run(" ".join(cmd), shell=True)

    time.sleep(2)

    exp_dirs = sorted(glob.glob("experiments/hitranet_*"), key=os.path.getmtime)

    if len(exp_dirs) == 0:
        continue

    latest_exp = exp_dirs[-1]

    cv_file = os.path.join(latest_exp, "cv_results.json")

    if os.path.exists(cv_file):

        with open(cv_file) as f:
            results = json.load(f)

        auc = results["auc_mean"]

        print("\nTrial CV AUC:", auc)
        print("Experiment:", latest_exp)

        if auc > best_auc:

            best_auc = auc
            best_config = params
            best_exp = latest_exp

            print("\n🔥 NEW BEST RESULT 🔥")
            print("Best CV AUC:", best_auc)
            print("Best Experiment:", best_exp)
            print("Best Config:", best_config)

        else:

            print("\nCurrent Best CV AUC:", best_auc)
            print("Best Experiment:", best_exp)
            print("Best Config:", best_config)