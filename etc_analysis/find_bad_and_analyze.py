import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------
# 설정
# -------------------------
ROOT = "/data5/sohyunkang/experiments"
xlsx_path = "./data/abide_subjects_fc_with_demo_260202.xlsx"
save_dir = "./fc_analysis"
os.makedirs(save_dir, exist_ok=True)

n_rois = 116

# -------------------------
# 1. subject 성능 집계
# -------------------------
wrong_counter = defaultdict(int)
total_counter = defaultdict(int)

exp_dirs = glob.glob(os.path.join(ROOT, "w*_hitranet_20260402*"))

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
# 2. bad / good 분리
# -------------------------
bad_subjects = []
good_subjects = []

for s in total_counter:
    total = total_counter[s]
    wrong = wrong_counter[s]

    if total < 3:
        continue

    if wrong == total:
        bad_subjects.append(s)

    if wrong == 0:
        good_subjects.append(s)

print(f"🔥 bad subjects: {len(bad_subjects)}")
print(f"🔥 good subjects: {len(good_subjects)}")

# -------------------------
# 3. 데이터 로드
# -------------------------
df = pd.read_excel(xlsx_path)
roi_cols = [c for c in df.columns if c.startswith("ROI")]
df = df.set_index("Subject")

def reconstruct_fc(row):
    vals = row[roi_cols].to_numpy(dtype=np.float32)
    fc = np.zeros((n_rois, n_rois), dtype=np.float32)
    triu_idx = np.triu_indices(n_rois, k=1)
    fc[triu_idx] = vals
    return fc + fc.T

# -------------------------
# 4. FC 분리
# -------------------------
bad_fcs, good_fcs = [], []

for subj in df.index:
    fc = reconstruct_fc(df.loc[subj])

    if subj in bad_subjects:
        bad_fcs.append(fc)

    if subj in good_subjects:
        good_fcs.append(fc)

bad_fcs = np.array(bad_fcs)
good_fcs = np.array(good_fcs)

# -------------------------
# 5. 평균 계산
# -------------------------
bad_mean = bad_fcs.mean(axis=0)
good_mean = good_fcs.mean(axis=0)
diff = bad_mean - good_mean

# -------------------------
# 6. 저장 함수 (🔥 label 포함)
# -------------------------
def save_fc(mat, title, filename, vmin=-1, vmax=1):
    plt.figure(figsize=(6,5))
    plt.imshow(mat, cmap="bwr", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

# -------------------------
# 7. 평균 저장
# -------------------------
save_fc(bad_mean, f"Bad Mean FC (100% wrong, n={len(bad_fcs)})", "bad_mean.png")
save_fc(good_mean, f"Good Mean FC (0% wrong, n={len(good_fcs)})", "good_mean.png")
save_fc(diff, "Difference (Bad - Good)", "diff.png", vmin=-0.3, vmax=0.3)

# -------------------------
# 8. 개별 subject 저장
# -------------------------
def save_individual(subject_list, prefix):
    for subj in subject_list:  # 너무 많으면 제한
        if subj not in df.index:
            continue

        row = df.loc[subj]
        fc = reconstruct_fc(row)

        label = row["DX_GROUP"] if "DX_GROUP" in row else "NA"

        title = f"{prefix} | {subj} | label={label}"
        filename = f"{prefix}_{label}_{subj}.png"

        save_fc(fc, title, filename)

# bad
save_individual(bad_subjects, "BAD")

# good
save_individual(good_subjects, "GOOD")

print(f"✅ saved to {save_dir}")