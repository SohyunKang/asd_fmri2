import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import FCDataset

import pandas as pd
import numpy as np

df = pd.read_excel("./data/abide_subjects_fc_with_demo_260202.xlsx")

# FC column만 평균
fc_cols = [c for c in df.columns if "ROI" in c]
mean_upper = df[fc_cols].mean().to_numpy()

N = 116
fc = np.zeros((N, N), dtype=np.float32)
triu_idx = np.triu_indices(N, k=1)
fc[triu_idx] = mean_upper
fc = fc + fc.T

np.fill_diagonal(fc, 0)

# 3️⃣ Plot (단일 plot, 스타일 지정 없음)
plt.figure(figsize=(6, 5.5))
plt.imshow(fc, aspect="auto", vmin=-0.2, vmax=0.2, cmap="RdBu_r")
plt.colorbar(label="Mean Correlation")
plt.title("Group-Averaged rsFC Matrix")
plt.xlabel("ROI index")
plt.ylabel("ROI index")
plt.tight_layout()
plt.savefig("group_mean_fc.png", dpi=300)   # ← 저장
plt.close()
