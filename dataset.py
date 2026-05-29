import re
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FCDataset(Dataset):
    """
    Loads FC from xlsx where only upper-triangle values are stored as pair columns.

    Expected format:
      - label column (default: 'label')
      - FC columns named like 'ROI1_ROI2', 'ROI1_ROI3', ..., 'ROI(N-1)_ROIN'
        (Typically upper triangle, i < j)

    Notes:
      - ROI indices are assumed to be 1-based in column names (ROI1 ... ROIN).
      - Diagonal is set to 0.
      - FC is reconstructed as symmetric.
    """

    ROI_PAIR_RE = re.compile(r"^ROI(\d+)_ROI(\d+)$")

    def __init__(
        self,
        xlsx_path: str,
        n_rois: int,
        label_col: str = "label",
        dtype: np.dtype = np.float32,
        enforce_upper_only: bool = True,  # if True, require i<j for all FC columns
    ):
        super().__init__()
        self.df = pd.read_excel(xlsx_path)
        self.n_rois = int(n_rois)
        self.label_col = label_col
        self.dtype = dtype
        self.enforce_upper_only = enforce_upper_only

        if self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found.")

        self.triu_size = self.n_rois * (self.n_rois - 1) // 2

        # Parse ROI pair columns
        parsed = []
        for c in self.df.columns:
            m = self.ROI_PAIR_RE.match(str(c))
            if m:
                i = int(m.group(1))
                j = int(m.group(2))
                parsed.append((c, i, j))

        if len(parsed) != self.triu_size:
            raise ValueError(
                f"Expected {self.triu_size} ROI pair columns like ROI1_ROI2, "
                f"but found {len(parsed)}."
            )

        # Validate indices and (optionally) require upper triangle
        for c, i, j in parsed:
            if not (1 <= i <= self.n_rois and 1 <= j <= self.n_rois):
                raise ValueError(f"Column '{c}' has ROI indices out of range 1..{self.n_rois}")
            if self.enforce_upper_only and not (i < j):
                raise ValueError(
                    f"Column '{c}' violates upper-triangle convention (expected i<j)."
                )

        # Sort columns by (i, j) to map into np.triu_indices order
        parsed.sort(key=lambda x: (x[1], x[2]))
        self.fc_cols = [c for c, _, _ in parsed]

        self.label_map = None
        unique = sorted(self.df[self.label_col].dropna().unique().tolist())
        # [1: Autism, 2: Typical Development] -> {1:0, 2:1}
        self.label_map = {int(v): i for i, v in enumerate(unique)}
        print("Label map:", self.label_map)

    def __len__(self) -> int:
        return len(self.df)

    def _reconstruct_fc(self, upper_vals: np.ndarray) -> np.ndarray:
        """
        upper_vals: shape (M,) where M = N*(N-1)//2
        returns full symmetric FC (N, N)
        """
        N = self.n_rois
        fc = np.zeros((N, N), dtype=self.dtype)

        triu_idx = np.triu_indices(N, k=1)  # order: (0,1),(0,2)...(0,N-1),(1,2)...
        fc[triu_idx] = upper_vals
        fc = fc + fc.T
        return fc

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        upper_vals = row[self.fc_cols].to_numpy(dtype=self.dtype)
        fc = self._reconstruct_fc(upper_vals)

        raw_label = int(row[self.label_col])
        label = self.label_map[raw_label]

        sub_id = row['Subject']
        AGE_AT_SCAN = row['AGE_AT_SCAN']
        SEX = row['SEX']
        func_mean_fd = row['func_mean_fd']
        Site = row['Site']
        SITE_ID = row['SITE_ID']


        return {
            "fc": torch.from_numpy(fc),  # (N, N)
            "label": torch.tensor(label, dtype=torch.long),
            "sub_id": sub_id,
            "AGE_AT_SCAN": AGE_AT_SCAN,
            "SEX": SEX,
            "Site": Site,
            "SITE_ID": SITE_ID,
            
        }
