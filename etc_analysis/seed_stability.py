import json

with open("aal_network_labels.txt", "r") as f:
    aal_roi_names = [line.strip() for line in f]

import os
import torch

base_dir = "experiments"

from collections import Counter

all_seed_names = []

for folder in os.listdir(base_dir):

    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):

        if not file.endswith("best_aux.pt"):
            continue

        aux = torch.load(os.path.join(folder_path, file), map_location="cpu")

        seeds = aux["seed_indices"].numpy()

        names = [aal_roi_names[i] for i in seeds]

        all_seed_names.extend(names)

counter = Counter(all_seed_names)

print("\nSeed frequency across folds:\n")

for name, count in counter.most_common():
    print(f"{name:25s} {count}")


import os
import json
import numpy as np

base_dir = "experiments"

metrics = {
    "auc": [],
    "acc": [],
    "sensitivity": [],
    "specificity": [],
    "epoch": []
}

print("\n=== Fold Results ===\n")

fold_id = 0

for folder in sorted(os.listdir(base_dir)):

    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):

        if not file.endswith("metrics.json"):
            continue

        path = os.path.join(folder_path, file)

        with open(path) as f:
            data = json.load(f)

        auc = data["auc"]
        acc = data["acc"]
        sens = data["sensitivity"]
        spec = data["specificity"]
        epoch = data["epoch"]

        print(f"Fold {fold_id}")
        print(f"  AUC  : {auc:.4f}")
        print(f"  ACC  : {acc:.4f}")
        print(f"  Sens : {sens:.4f}")
        print(f"  Spec : {spec:.4f}")
        print(f"  Epoch: {epoch}")
        print()

        metrics["auc"].append(auc)
        metrics["acc"].append(acc)
        metrics["sensitivity"].append(sens)
        metrics["specificity"].append(spec)
        metrics["epoch"].append(epoch)

        fold_id += 1


print("\n=== Mean ± Std ===\n")

for k,v in metrics.items():

    mean = np.mean(v)
    std = np.std(v)

    if k == "epoch":
        print(f"{k:12s}: {mean:.2f} ± {std:.2f}")
    else:
        print(f"{k:12s}: {mean:.4f} ± {std:.4f}")