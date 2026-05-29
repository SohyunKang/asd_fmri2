import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/data5/sohyunkang/experiments"
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

# -------------------------
# experiment 순회
# -------------------------
exp_dirs = glob.glob(os.path.join(ROOT, "w*_hitranet_20260402*"))

for exp in exp_dirs:

    aux_files = glob.glob(os.path.join(exp, "fold*_best_aux.pt"))

    for aux_path in aux_files:

        try:
            fold_name = os.path.basename(aux_path).replace("_best_aux.pt", "")
            print(f"\n📂 {exp} - {fold_name}")

            aux = torch.load(aux_path, map_location="cpu", weights_only=False)

            y_true = np.array(aux["val_y"])
            y_pred = np.array(aux["val_pred"])
            subjects = np.array(aux["val_subject"])

            # -------------------------
            # 그룹 분리
            # -------------------------
            groups = {
                "ASD_true": [],
                "ASD_false": [],
                "TD_true": [],
                "TD_false": [],
            }

            for yt, yp, subj in zip(y_true, y_pred, subjects):

                if subj not in df.index:
                    continue

                fc = reconstruct_fc(df.loc[subj])
                fc = abs(fc)

                label = df.loc[subj]["DX_GROUP"]  # 1=ASD, 2=TD
                correct = (yt == yp)

                if label == 1:
                    (groups["ASD_true"] if correct else groups["ASD_false"]).append(fc)
                elif label == 2:
                    (groups["TD_true"] if correct else groups["TD_false"]).append(fc)

            def safe_mean(arr):
                return np.mean(arr, axis=0) if len(arr) > 0 else None

            ASD_true = safe_mean(groups["ASD_true"])
            ASD_false = safe_mean(groups["ASD_false"])
            TD_true = safe_mean(groups["TD_true"])
            TD_false = safe_mean(groups["TD_false"])

            # -------------------------
            # diff 계산
            # -------------------------
            ASD_diff = ASD_true - ASD_false if ASD_true is not None and ASD_false is not None else None
            TD_diff = TD_true - TD_false if TD_true is not None and TD_false is not None else None

            cross_1 = ASD_true - TD_false if ASD_true is not None and TD_false is not None else None
            cross_2 = ASD_false - TD_true if ASD_false is not None and TD_true is not None else None

            # 🔥 NEW (핵심 추가)
            cross_3 = ASD_true - TD_true if ASD_true is not None and TD_true is not None else None
            
            # -------------------------
            # plot (3x3)
            # -------------------------
            plt.figure(figsize=(15,12))

            def plot_safe(mat, title, idx, vmin=-1, vmax=1):
                plt.subplot(3,3,idx)
                if mat is not None:
                    plt.imshow(mat, cmap="bwr", vmin=vmin, vmax=vmax)
                    plt.colorbar()
                plt.title(title)

            # 🔹 Row 1: ASD
            plot_safe(ASD_true, f"ASD Correct (n={len(groups['ASD_true'])})", 1)
            plot_safe(ASD_false, f"ASD Wrong (n={len(groups['ASD_false'])})", 2)
            plot_safe(ASD_diff, "ASD Diff (Correct - Wrong)", 3, vmin=-0.3, vmax=0.3)

            # 🔹 Row 2: TD
            plot_safe(TD_false, f"TD Wrong (n={len(groups['TD_false'])})", 4)
            plot_safe(TD_true, f"TD Correct (n={len(groups['TD_true'])})", 5)
            plot_safe(TD_diff, "TD Diff (Correct - Wrong)", 6, vmin=-0.3, vmax=0.3)

            # 🔹 Row 3: Cross
            plot_safe(cross_1, "ASD correct - TD wrong", 7, vmin=-0.3, vmax=0.3)
            plot_safe(cross_2, "ASD wrong - TD correct", 8, vmin=-0.3, vmax=0.3)

            # 🔥 NEW
            plot_safe(cross_3, "ASD correct - TD correct", 9, vmin=-0.3, vmax=0.3)

            # optional: 마지막 하나는 비워도 됨
            plt.subplot(3,3,9)
            plt.axis("off")

            plt.suptitle(f"{fold_name} 3x3 FC Comparison", fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(exp, f"{fold_name}_fc_compare.png")
            plt.savefig(save_path, dpi=200)
            plt.close()

            print(f"✅ saved: {save_path}")

        except Exception as e:
            print(f"❌ Error in {aux_path}: {e}")