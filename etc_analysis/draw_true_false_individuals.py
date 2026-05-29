import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 🔥 설정 (여기만 바꿔)
# -------------------------
exp_dir = "/data5/sohyunkang/experiments/w2_hitranet_20260402_143843_609394"
fold = 4

xlsx_path = "./data/abide_subjects_fc_with_demo_260202.xlsx"
n_rois = 116

# -------------------------
# 데이터 로드
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

def save_fc(fc, title, save_path):
    plt.figure(figsize=(5,5))
    plt.imshow(fc, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# -------------------------
# aux 로드
# -------------------------
aux_path = os.path.join(exp_dir, f"fold{fold}_best_aux.pt")

aux = torch.load(aux_path, map_location="cpu", weights_only=False)

y_true = np.array(aux["val_y"])
y_pred = np.array(aux["val_pred"])
print(np.unique(y_true), np.unique(y_pred))

subjects = np.array(aux["val_subject"])

# -------------------------
# 폴더 생성
# -------------------------
base_save_dir = os.path.join(exp_dir, f"fold{fold}_fc")

groups = ["ASD_correct", "ASD_wrong", "TD_correct", "TD_wrong"]

for g in groups:
    os.makedirs(os.path.join(base_save_dir, g), exist_ok=True)

# -------------------------
# 저장
# -------------------------
for yt, yp, subj in zip(y_true, y_pred, subjects):

    if subj not in df.index:
        continue

    row = df.loc[subj]
    fc = reconstruct_fc(row)
    fc = abs(fc)

    label = row["DX_GROUP"]  # 1=ASD, 2=TD
    correct = (yt == yp)

    if label == 1 and correct:
        group = "ASD_correct"
    elif label == 1 and not correct:
        group = "ASD_wrong"
    elif label == 2 and correct:
        group = "TD_correct"
    elif label == 2 and not correct:
        group = "TD_wrong"
    else:
        continue

    save_path = os.path.join(base_save_dir, group, f"{subj}.png")

    title = f"{subj} | label={yt} | pred={yp}"

    save_fc(fc, title, save_path)

print(f"✅ Saved all FCs to: {base_save_dir}")