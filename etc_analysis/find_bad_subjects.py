import os
import glob
import torch
import numpy as np
from collections import defaultdict

ROOT = "/data5/sohyunkang/experiments"

# subject 기준 aggregation
wrong_counter = defaultdict(int)
total_counter = defaultdict(int)

# -------------------------
# 모든 experiment 순회
# -------------------------
exp_dirs = glob.glob(os.path.join(ROOT, "w*_hitranet_20260402*"))

print(f"Found {len(exp_dirs)} experiments\n")

for exp in exp_dirs:

    aux_files = glob.glob(os.path.join(exp, "fold*_best_aux.pt"))

    for aux_path in aux_files:

        try:
            aux = torch.load(aux_path, map_location="cpu", weights_only=False)

            y_true = np.array(aux["val_y"])
            y_pred = np.array(aux["val_pred"])
            subjects = np.array(aux["val_subject"])

            wrong = (y_true != y_pred)

            for s, w in zip(subjects, wrong):
                total_counter[s] += 1
                if w:
                    wrong_counter[s] += 1

        except Exception as e:
            print(f"Error in {aux_path}: {e}")

# -------------------------
# 결과 정리
# -------------------------
results = []

for s in total_counter:
    total = total_counter[s]
    wrong = wrong_counter[s]
    ratio = wrong / total

    results.append((s, ratio, wrong, total))

# ratio 기준 정렬
results.sort(key=lambda x: x[1], reverse=True)

# -------------------------
# 출력
# -------------------------
print("\n========== MOST MISCLASSIFIED SUBJECTS (100%) ==========\n")

bad_filtered = [
    (s, ratio, wrong, total)
    for s, ratio, wrong, total in results
    if ratio == 1 and total >= 5
]

bad_filtered.sort(key=lambda x: x[0])

for s, ratio, wrong, total in bad_filtered:
    print(f"{s} | wrong {wrong}/{total} ({ratio:.2f})")

print(f"\nTotal bad: {len(bad_filtered)}")


# -------------------------
# GOOD SUBJECTS (0%)
# -------------------------
print("\n========== PERFECTLY CLASSIFIED SUBJECTS (0%) ==========\n")

good_filtered = [
    (s, ratio, wrong, total)
    for s, ratio, wrong, total in results
    if ratio == 0 and total >= 5
]

good_filtered.sort(key=lambda x: x[0])

for s, ratio, wrong, total in good_filtered:
    print(f"{s} | wrong {wrong}/{total} ({ratio:.2f})")

print(f"\nTotal good: {len(good_filtered)}")