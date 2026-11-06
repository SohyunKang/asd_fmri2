# HiTraNet

**Hierarchical Trajectory Representation Learning from Resting-State Functional Connectivity for Autism Spectrum Disorder**

HiTraNet is a PyTorch-based research framework for autism spectrum disorder (ASD) classification and analysis of hierarchical latent representations derived from resting-state functional connectivity (rsFC). This README explains the data format, model architecture, training pipeline, saved artifacts, and analysis workflow used by `main3.py` and `model.py`.

> **Research use:** Predictions and propagation visualizations are intended for research analysis, not clinical diagnosis.

## Contents

1. [Project overview](#1-project-overview)
2. [Scientific motivation](#2-scientific-motivation)
3. [Implemented workflow](#3-implemented-workflow)
4. [Repository structure](#4-repository-structure)
5. [Installation and dependencies](#5-installation-and-dependencies)
6. [Dataset and preprocessing](#6-dataset-and-preprocessing)
7. [Input format and data validation](#7-input-format-and-data-validation)
8. [Model architecture](#8-model-architecture)
9. [Tensor shapes and returned values](#9-tensor-shapes-and-returned-values)
10. [Training objective and optimization](#10-training-objective-and-optimization)
11. [Site-aware cross-validation](#11-site-aware-cross-validation)
12. [Command-line arguments](#12-command-line-arguments)
13. [Running experiments](#13-running-experiments)
14. [Evaluation metrics and label conventions](#14-evaluation-metrics-and-label-conventions)
15. [Experiment directory and output files](#15-experiment-directory-and-output-files)
16. [Loading checkpoints and interpretation tensors](#16-loading-checkpoints-and-interpretation-tensors)
17. [Interpreting hierarchical trajectories](#17-interpreting-hierarchical-trajectories)

---

## 1. Project overview

**HiTraNet (Hierarchical Trajectory Network)** learns subject-specific, multi-hop latent representations from resting-state functional connectivity (rsFC) to classify **autism spectrum disorder (ASD)** versus **typically developing (TD)** controls.

Each subject is represented as a weighted graph whose nodes correspond to brain regions of interest (ROIs) and whose edge weights encode functional connectivity. The model selects informative seed ROIs and tracks both latent node states and seed influence distributions through successive propagation hops. Hop-wise representations are aggregated into a subject embedding for classification and later interpretation.

The framework contains six key components:

1. **Learnable seed selection:** top-$M$ ROI origins selected by globally learned seed scores.
2. **Seed-conditioned initialization:** separate node-state channels initialized using each subject's FC and a trainable seed token.
3. **Hop-dependent propagation:** an FC-informed transition operator generated from evolving node states.
4. **Seed influence tracking:** propagation distributions from each seed to destination ROIs.
5. **Hierarchical trajectory readout:** seed- and hop-specific pooled latent vectors.
6. **Prediction head:** classification from the concatenation of aggregated hop representations.

### Inputs and outputs

| Component | Input | Output |
|---|---|---|
| `FCDataset` | Excel table with 6,670 FC correlations per subject and phenotypic metadata | Symmetric FC matrices `(B,116,116)`, labels, subject/site fields |
| Seed selector | Trainable ROI scores | Selected ROI indices and normalized weights |
| Encoder | FC, seeds, weights | Hop-specific transition operators, influence fields and node states |
| Trajectory readout | Seed influence fields and states | Seed-specific features `(B,K,M,d)` |
| Classifier | Weighted representation concatenated across hops | ASD/TD logits `(B,2)` |
| Training driver | Dataset, CV settings and hyperparameters | Fold metrics, checkpoints, split manifest, latent tensors |

Here, **hop** denotes a step in the model's graph-based representation propagation; it is not a longitudinal time point.

---

## 2. Scientific motivation

ASD has heterogeneous effects on functional brain organization. The brain's macroscale functional hierarchy is commonly described as a progression from **sensory/unimodal systems**, through **association systems**, toward **transmodal systems** involved in higher-order integration. Functional connectivity (FC) derived from resting-state fMRI provides a network representation with which to investigate such organization.

Conventional classifiers often summarize an rsFC matrix using global features or identify discriminative regions and connections. HiTraNet additionally models how representations **originating from selected ROIs are redistributed and integrated over multiple graph propagation depths**. The resulting sequence of latent vectors constitutes a *hierarchical trajectory* in a shared feature space.

Two complementary objects make this analysis possible:

- **Seed-conditioned node states** represent the evolving information at each ROI.
- **Seed influence fields** track the model-assigned distribution of influence from an originating seed to destination ROIs.

Their influence-weighted readout produces a trajectory for each seed. The same hop representations are integrated for ASD/TD prediction, thereby connecting predictive features with the model's hierarchical propagation process.

The sensory → association → transmodal hierarchy is the **scientific context used to interpret** the learned trajectories. Functional-network assignments and biological interpretations are assessed through downstream analysis rather than imposed as fixed labels in the classifier.

---

## 3. Implemented workflow

```text
ABIDE-I preprocessed rs-fMRI regional time series
             |
             |  rs-fMRI preprocessing and quality control
             v
AAL atlas (116 ROIs) + Pearson correlation
             |
             v
One FC matrix per participant: 116 x 116
             |
             |  store upper triangle in Excel (6,670 features)
             v
FCDataset(dataset.py)
  - reconstruct symmetric FC
  - read DX_GROUP, subject ID, site, and metadata
             |
             v
Site-aware 5-fold partition (main3.py)
  - outer test fold
  - inner train / validation split
  - approximately balance diagnosis + site + diagnosis-by-site
             |
             v
HiTraNet(model.py)
  [1] Learn global top-M ROI seeds and weights
  [2] Initialize participant-specific, seed-conditioned node features
  [3] Repeat K hops:
        FC-informed, state-dependent P^(k)
        update seed-origin influence p_s^(k)
        update seed-conditioned node states X_s^(k)
        pool g_s^(k) = sum_i p_s,i^(k) X_s,i^(k)
        aggregate seeds using alpha_s
  [4] Concatenate all K hop representations
  [5] MLP -> ASD / TD logits
             |
             v
Class-weighted CE training and validation-AUC selection
             |
             v
Fold test evaluation + saved checkpoints + seed trajectories
```

**Training input:** `main3.py` starts from the prepared FC Excel file. rs-fMRI preprocessing and FC construction are upstream data preparation steps.

---

## 4. Repository structure

```text
HiTraNet/
├── README.md
├── main3.py                     # Main experiment entry point
├── model.py                     # HiTraNet architecture used by main3.py
├── model2.py                    # Additional model architecture
├── dataset.py                   # FC Excel loader and matrix reconstruction
├── utils.py                     # Utilities for seeds, metrics, FC and ROI names
├── data/
│   ├── abide_subjects_fc_with_demo_260202.xlsx
│   └── ROI_MNI_V4.txt
└── visualization/
    ├── __init__.py
    └── viz_umap.py              # Module imported by main3.py
```

### Module responsibilities

| File | Role | Main functions/classes |
|---|---|---|
| `main3.py` | Site-aware splitting, model training, validation, fold evaluation, export | `main`, `train_one_epoch`, `eval_one_epoch`, `build_site_aware_cv_splits` |
| `model.py` | HiTraNet used by `main3.py` | `HiTraNet`, `PeripherySeedness`, `SeedHopEncoder`, `EdgeGatedPropagation`, `GraphDeepBlock`, `MLPHead` |
| `dataset.py` | Reading subject-wise FC features and metadata | `FCDataset` |
| `utils.py` | Random seed control, FC normalization helper, meters, ROI-name loading | `set_seed`, `normalize_fc`, `accuracy_from_logits`, `AverageMeter`, `load_roi_names_from_txt` |
| `model2.py` | Additional trajectory encoder and transmodal-arrival architecture | `HierarchicalTrajectoryEncoderDeep`, `TransmodalArrival` |
| `visualization/viz_umap.py` | Functions referenced by the entry script for trajectory visualization | `extract_seed_step_graphrepr`, `save_umap_seed_trajectories` |

The training entry point imports `HiTraNet` with `from model import HiTraNet`. The data files and `visualization` package are expected to be available at the paths shown above. The experiment output path is specified in Section 5.3.

---

## 5. Installation and dependencies

### 5.1 Requirements

Use Python **3.9+** (`argparse.BooleanOptionalAction` is used) and install a PyTorch build compatible with the machine's CPU or CUDA environment. The main Python dependencies are:

- `torch`
- `numpy`, `pandas`, `openpyxl`
- `scikit-learn`, `matplotlib`
- `entmax`

Example setup:

```bash
conda create -n hitranet python=3.11 -y
conda activate hitranet

# Install PyTorch for your CUDA/CPU platform (https://pytorch.org/get-started/locally/).
python -m pip install numpy pandas openpyxl scikit-learn matplotlib entmax
```

Install any dependencies used by the repository's `visualization/viz_umap.py` module as well. A GPU is useful for full experiments; a CPU is sufficient for small model tests.

### 5.2 Visualization module

`main3.py` imports the following functions on startup:

```python
from visualization.viz_umap import (
    extract_seed_step_graphrepr,
    save_umap_seed_trajectories,
)
```

Place the corresponding `visualization/viz_umap.py` module in the repository before running `main3.py`. The main training loop saves site-distribution plots by default; the UMAP routines are separate analysis utilities.

For a training-only checkout that does not include the visualization module, remove or comment out the import above. The UMAP function calls inside the epoch loop are already commented out.

### 5.3 Experiment output directory

By default, `main3.py` saves experiments under:

```text
/data5/sohyunkang/experiments2/w<worker_id>_hitranet_<timestamp>/
```

This path is defined by the following line in `main3.py`:

```python
exp_dir = f"/data5/sohyunkang/experiments2/w{args.worker_id}_hitranet_{timestamp}"
```

Configure the directory for the destination server before running the first experiment. For example, replace the parent path with an accessible location while retaining the experiment-specific suffix:

```python
exp_dir = f"./experiments/w{args.worker_id}_hitranet_{timestamp}"
```

The script automatically creates `exp_dir` with `os.makedirs(exp_dir, exist_ok=True)`.

### 5.4 Basic checks

From the repository root:

```bash
python -m py_compile dataset.py main3.py model.py model2.py utils.py
python -c "import torch, pandas, sklearn, matplotlib, entmax; print('Imports OK')"
python main3.py --help
```

The `--help` check also imports `visualization.viz_umap`, so keep that module accessible.

---

## 6. Dataset and preprocessing

### 6.1 Dataset source

HiTraNet is designed for the **ABIDE-I (Autism Brain Imaging Data Exchange I)** resting-state fMRI dataset. The input to the experiment driver is a **subject-wise Excel workbook** of functional connectivity (FC) features and subject metadata. The number of subjects and their ASD/TD proportions are determined by the rows in the workbook supplied through `--csv`.

The classifier uses the reconstructed FC matrices. Diagnosis labels (`DX_GROUP`) supply the learning targets; acquisition sites (`SITE_ID`) define the site-aware cross-validation partitions. The additional subject metadata support dataset inspection and result identification.

### 6.2 Functional connectivity construction

Resting-state FC represents correlations between regional fMRI time series. For a **116-ROI AAL parcellation**, a subject's Pearson correlation matrix has shape `(116,116)` and contains **6,670 unique ROI pairs** excluding self-connections.

The data preparation process used to create the input Excel workbook is:

1. Obtain regional resting-state fMRI time series for the AAL ROIs using the chosen ABIDE-I preprocessing derivatives.
2. Compute pairwise Pearson correlation coefficients between ROI time series.
3. Store the strict upper-triangle FC values in columns named `ROI1_ROI2`, `ROI1_ROI3`, ..., `ROI115_ROI116`.
4. Add the subject identifier, diagnosis label, acquisition-site information, and required phenotypic metadata.

The executable experiment begins at this prepared workbook. The loader reconstructs a symmetric FC matrix for every subject.

### 6.3 FC representation in the model

For each subject, the matrix $A\in\mathbb{R}^{N\times N}$ contains signed FC values. `FCDataset` reconstructs the full symmetric matrix, preserves positive and negative correlations, and sets the diagonal entries to zero. The reconstructed FC matrix is passed to HiTraNet directly.

### 6.4 Dataset-level processing

If an experiment introduces transformations estimated across participants, such as harmonization or data-driven scaling, estimate their parameters on each training partition and apply them to its validation and test partitions. The training script uses the prepared FC entries from the input workbook.

---

## 7. Input format and data validation

### 7.1 Excel file

The CLI argument `--csv` is historically named but must point to an **Excel workbook** (`.xlsx`) because `dataset.py` uses `pd.read_excel()`.

Default value:

```text
data/abide_subjects_fc_with_demo_260202.xlsx
```

The experiment driver constructs the dataset with:

```python
FCDataset(
    xlsx_path=args.csv,
    n_rois=args.n_rois,
    label_col="DX_GROUP",
)
```

For experiments launched through `main3.py`, the label column is `DX_GROUP`.

### 7.2 Required columns

| Column(s) | Expected content | Use |
|---|---|---|
| `DX_GROUP` | ABIDE diagnostic code | Classification label; usually `1=ASD`, `2=TD` |
| `Subject` | Unique participant ID | Subject identifier saved with results |
| `AGE_AT_SCAN` | Numeric age at scan | Split-distribution summary |
| `SEX` | Dataset's encoded sex value | Split-distribution summary |
| `func_mean_fd` | Numeric mean framewise displacement | Read from the workbook by the dataset loader |
| `Site` | Acquisition-site description | Subject metadata |
| `SITE_ID` | Acquisition-site code | **Required** site-aware partitioning |
| `ROI1_ROI2`, `ROI1_ROI3`, ... | Pearson FC between the indicated ROIs | Reconstruct `N × N` FC |

The ROI pair columns are named using **1-based ROI indices** and, by default, must satisfy `i < j`. For `N = 116`, **exactly 6,670** matching columns must be present.

For example, the FC columns begin with:

```text
ROI1_ROI2
ROI1_ROI3
...
ROI1_ROI116
ROI2_ROI3
...
ROI115_ROI116
```

The order of columns **in the workbook** is not important. `FCDataset` parses and sorts ROI pairs into NumPy strict-upper-triangle order before reconstructing:

```python
fc = np.zeros((N, N), dtype=np.float32)
fc[np.triu_indices(N, k=1)] = upper_values
fc = fc + fc.T
```

For each subject:

- `fc.shape == (116, 116)` and `fc.dtype == float32`;
- `fc` is symmetric, `fc[i, j] == fc[j, i]`;
- diagonal entries are zero;
- `label` is mapped to a contiguous zero-based integer internally.

### 7.3 Label mapping

`FCDataset` sorts the **unique raw diagnosis values** found in the workbook and maps them to integer indices.

For an ABIDE workbook with the expected coding:

```text
Raw DX_GROUP = 1 (ASD) -> internal label 0
Raw DX_GROUP = 2 (TD)  -> internal label 1
```

The loader prints `Label map: {1: 0, 2: 1}`. The evaluation code **assumes exactly this mapping** and treats internal label `0` as the ASD-positive class. If the input labels differ, verify or explicitly fix the mapping before interpreting sensitivity, specificity, F1, or ROC-AUC.

### 7.4 ROI-name file

Default:

```text
data/ROI_MNI_V4.txt
```

`load_roi_names_from_txt()` reads the **second whitespace-separated field** on each nonempty, non-commented line:

```text
1 Precentral_L
2 Precentral_R
3 Frontal_Sup_L
...
```

`main3.py` reads the ROI-name file at startup and checks that it contains exactly `--n_rois` names. Keep ROI ordering consistent with the FC features.

### 7.5 Recommended preflight data audit

Run the following without altering the experiment code:

```bash
python - <<'PY'
from itertools import combinations
import numpy as np
import pandas as pd

path = 'data/abide_subjects_fc_with_demo_260202.xlsx'
n = 116

df = pd.read_excel(path)
required = ['DX_GROUP', 'Subject', 'AGE_AT_SCAN', 'SEX',
            'func_mean_fd', 'Site', 'SITE_ID']
missing = sorted(set(required) - set(df.columns))
assert not missing, f'Missing metadata: {missing}'

expected = {f'ROI{i}_ROI{j}' for i, j in combinations(range(1, n + 1), 2)}
actual = {c for c in df.columns if isinstance(c, str) and c.startswith('ROI')}
assert expected == actual, (
    f'Missing FC pairs: {len(expected - actual)}; '
    f'Unexpected ROI columns: {len(actual - expected)}'
)
assert not df['Subject'].duplicated().any(), 'Duplicate subject IDs found'
assert set(df['DX_GROUP'].dropna().unique()) == {1, 2}, 'Unexpected diagnostic codes'
assert not df[sorted(expected)].isna().any().any(), 'Missing FC values'
assert np.isfinite(df[sorted(expected)].to_numpy(dtype=float)).all(), 'Nonfinite FC values'

print('Participants:', len(df))
print('DX_GROUP:', df['DX_GROUP'].value_counts().to_dict())
print('Sites:', df['SITE_ID'].nunique())
print('FC pairs:', len(expected))
print('Data schema checks passed')
PY
```

This preflight assumes one row per unique subject. If a future dataset includes repeated scans per participant, **subject-grouped splitting must replace the present row-index splitting** to prevent subject leakage.

---

## 8. Model architecture

The HiTraNet forward pass in `model.py` uses the following dimensions:

- `B`: minibatch size;
- `N`: number of ROIs (default **116**);
- `M`: selected seeds (default **5**);
- `K`: propagation hops (default **3**);
- `d`: latent dimension (default **128**);
- `C`: output classes (default **2**).

Each participant is represented by a signed FC matrix $A\in\mathbb{R}^{N\times N}$. The model receives this matrix from `FCDataset`.

### 8.1 Learned global seed selection — `PeripherySeedness`

A global trainable seed score $u_i$ is maintained for each ROI and initialized as:

$$
u_i\sim\mathcal{N}(0,0.1^2),\qquad u\in\mathbb{R}^{N}.
$$

The seed module returns the score vector $u$. The **$M$ ROIs with the highest scores** are selected as seeds:

$$
\mathcal{S}=\mathrm{TopM}(u).
$$

Importance weights are computed by applying **1.5-entmax** to all ROI scores and renormalizing the weights of the selected seeds:

$$
a=\mathrm{entmax}_{1.5}(u),\qquad
\alpha_s=\frac{a_s}{\sum_{j\in\mathcal{S}}a_j},\quad s\in\mathcal{S}.
$$

The selected seed weights are nonnegative and sum to one. Seed scores, indices, and weights are shared globally across participants for a trained model. The subsequent FC-dependent node states and propagation operators are subject-specific.

The selector uses `torch.topk` ranking in both training and evaluation. Gradients propagate through the selected entmax weights; selected integer indices define the seed channels.

### 8.2 Seed-conditioned initialization — `SeedHopEncoder`

Each ROI's FC connectivity profile (one row of $A$) is projected to a $d$-dimensional latent vector:

$$
F_i=W_{\mathrm{FC}}A_{i,:}+b_{\mathrm{FC}},\qquad F\in\mathbb{R}^{N\times d}.
$$

For every selected seed $s$, a seed-conditioned node-state matrix is initialized from these features and a shared trainable seed token $t$:

$$
X_{s,i}^{(0)}=F_i+\mathbb{1}[i=s]t,
\qquad t\in\mathbb{R}^{d}.
$$

The token is added at the selected ROI within that seed's channel and initialized using `--seed_std` (default `0.5`). Each channel therefore combines the subject's FC-derived latent features with the location of its seed.

The corresponding seed influence distribution is initialized as a one-hot row vector:

$$
p_{s,i}^{(0)}=\mathbb{1}[i=s].
$$

### 8.3 Shared, FC-informed hop-dependent operator — `EdgeGatedPropagation`

At hop $k$, seed-conditioned node states are combined with their globally learned seed weights:

$$
\bar H^{(k-1)}=\sum_{s\in\mathcal{S}}\alpha_sX_s^{(k-1)}
\in\mathbb{R}^{N\times d}.
$$

Learnable query and key projections are applied:

$$
Q^{(k)}=\bar H^{(k-1)}W_q,\qquad
K^{(k)}=\bar H^{(k-1)}W_k.
$$

The FC-informed propagation operator is:

$$
P^{(k)}=\mathrm{RowSoftmax}\left(
\frac{Q^{(k)}(K^{(k)})^{\top}}{\sqrt{d_k}}+A
\right).
$$

`P^(k)` is a **dense, row-stochastic transition matrix** with shape `(B,N,N)`. It is determined by both the subject's FC values and hop-specific node states. Signed FC values contribute to the pre-softmax scores, allowing positive and negative connectivity to modulate the learned transitions.

### 8.4 Seed influence propagation

Each seed has a row-vector distribution over destination ROIs. Its update is:

$$
p_s^{(k)}=p_s^{(k-1)}P^{(k)}.
$$

Because the initial vector is one-hot and every `P` is row-stochastic, `p_s^(k)` sums to approximately one and can be interpreted as a **model-derived distribution of influence over ROIs at hop k**.

For comprehensive analysis, the model also tracks an all-source arrival matrix:

$$
P_{\mathrm{all}}^{(0)}=I,\qquad
P_{\mathrm{all}}^{(k)}=P_{\mathrm{all}}^{(k-1)}P^{(k)}.
$$

The encoder returns these values as `p_all_list` and `arrival_all` for direct model-level analysis. The experiment driver exports selected-seed influence arrays (`p`) in its auxiliary bundle.

### 8.5 Seed-conditioned node-state propagation

The model propagates node-state features using the **transpose** of the transition operator:

$$
Z_s^{(k)}=(P^{(k)})^{\top}X_s^{(k-1)}.
$$

Thus destination ROI $j$ aggregates representations from source ROIs $i$ with weights $P_{ij}^{(k)}$, in the same direction as the row-vector seed influence update.

A learnable feature transform $W\in\mathbb{R}^{d\times d}$ is shared across all propagation hops:

$$
\widetilde X_s^{(k)}=Z_s^{(k)}W.
$$

The transformed features and preceding state are combined using a weighted residual step:

$$
R_s^{(k)}=0.1\widetilde X_s^{(k)}+0.9X_s^{(k-1)}.
$$

Each hop then applies the shared stack of node-wise feed-forward blocks and an output normalization:

$$
X_s^{(k)}=\mathrm{LayerNorm}\bigl(\mathrm{DeepBlocks}(R_s^{(k)})\bigr).
$$

Each `GraphDeepBlock` performs:

```text
Input x
  -> LayerNorm(d)
  -> Linear(d, 4d)
  -> GELU or ReLU
  -> Dropout
  -> Linear(4d, d)
  -> Dropout
  -> residual output: x + sigmoid(gate) * FFN(x)
```

The trainable scalar gate is initialized at `-2.0` (initial sigmoid value approximately `0.119`). The `--depth` parameter controls how many feed-forward blocks are applied per hop. The same deep stack and the same transformation matrix $W$ are reused at every hop.

### 8.6 Seed-specific hierarchical latent trajectory

At every hop, the model forms a seed-specific pooled feature:

$$
g_s^{(k)}=\sum_{i=1}^{N}p_{s,i}^{(k)}X_{s,i}^{(k)}
\in\mathbb{R}^{d}.
$$

For each seed, the sequence

$$
\mathcal T_s=\{g_s^{(1)},g_s^{(2)},\ldots,g_s^{(K)}\}
$$

is its **hierarchical latent trajectory**. All hop representations have the same dimension, and the shared recurrence makes cross-hop analysis possible.

### 8.7 Aggregation across seeds

The seed-specific features are integrated at every hop using the learned global seed weights:

$$
\bar g^{(k)}=\sum_{s\in\mathcal{S}}\alpha_sg_s^{(k)}.
$$

The result is one participant-specific feature vector of dimension $d$ for each hop, used by the prediction head.

### 8.8 Multi-hop prediction

The model concatenates the aggregate of every hop:

$$
G=[\bar g^{(1)}\,\|\,\bar g^{(2)}\,\|\,\ldots\,\|\,\bar g^{(K)}]
\in\mathbb{R}^{Kd}.
$$

The prediction head performs:

```text
LayerNorm(K*d)
 -> Linear(K*d, mlp_hidden)
 -> GELU
 -> Dropout
 -> Linear(mlp_hidden, 2)
 -> logits [ASD, TD]
```

At default settings (`K=3`, `d=128`), the input to the MLP has **384 dimensions**. The final logits have **two dimensions** corresponding to internal labels `0=ASD`, `1=TD`.

---

## 9. Tensor shapes and returned values

### 9.1 Main forward-path tensors

| Tensor / symbol | Code name | Shape | Description |
|---|---|---|---|
| Subject FC | `A` | `(B, N, N)` | Reconstructed signed FC |
| Seed scores | `u_per` | `(N,)` | Globally trained raw ROI scores |
| Seed indices | `seed_indices` | `(M,)` | Selected **0-based** ROI indices |
| Seed weights | `seed_weights` | `(M,)` | Entmax-based weights summing to 1 |
| FC node features | `fc_feat` | `(B, N, d)` | Projection of FC rows |
| Seed states | `X` | `(B, M, N, d)` | Latent node state for each seed |
| Seed-weighted context | `H_bar` | `(B, N, d)` | Operator conditioning state |
| Hop transition | `P` | `(B, N, N)` | Dense, row-stochastic learned operator |
| Seed influence | `p` | `(B, M, N)` | Destination weights for each seed |
| All-source arrival | `p_all` | `(B, N, N)` | Composition of transitions |
| Per-seed graph features | `g_seed` | `(B, M, d)` | Trajectory position for each seed |
| Weighted hop features | `g_bar` | `(B, d)` | Seed-aggregated features |
| Subject features | `g` | `(B, K*d)` | All-hop concatenated representation |
| Class logits | `logits` | `(B, 2)` | Classification output |

### 9.2 Direct model usage

```python
import torch
from model import HiTraNet

model = HiTraNet(
    n_rois=116,
    d_model=128,
    n_hops=3,
    top_m=5,
    mlp_hidden=128,
    n_classes=2,
    depth=2,
    dropout=0.2,
    seed_std=0.5,
    act="gelu",
    use_gated_residual=True,
)
model.eval()

# Replace with a real 116 x 116 FC matrix before scientific use.
A = torch.zeros(2, 116, 116)

with torch.no_grad():
    logits, aux = model(A)
    prob_asd = logits.softmax(dim=1)[:, 0]

print("logits:", logits.shape)
print("ASD probability:", prob_asd.shape)
print("seeds:", aux["seed_indices"])
print("seed weights:", aux["seed_weights"])
print("hop-specific seed embeddings:", [x.shape for x in aux["g_seed_list"]])
print("hop-specific influence:", [x.shape for x in aux["p_list"]])
```

`A = 0` is a **shape smoke test only**; it is not a meaningful biological input.

### 9.3 `model.py` auxiliary dictionary

The `model.py` forward pass returns `logits, aux`, with the following analysis fields:

| Key | Type / shape | Meaning |
|---|---|---|
| `seed_indices` | CPU tensor `(M,)` | Selected ROI indices |
| `seed_weights` | tensor `(M,)` | Learned global seed weights |
| `seedness` | tensor `(N,)` | Raw learnable seed scores |
| `P_list` | `K` tensors `(B,N,N)` | Per-hop transition operators |
| `p_list` | `K` tensors `(B,M,N)` | Selected seed-origin influence fields |
| `p_all_list` | `K` tensors `(B,N,N)` | All-source transition compositions |
| `arrival_all` | tensor `(B,N,N)` | Final all-source arrival matrix |
| `H_list` | `K` tensors `(B,M,N,d)` | Seed-conditioned node states |
| `g_seed_list` | `K` tensors `(B,M,d)` | Raw seed-specific trajectory positions |
| `g_bar_list` | `K` tensors `(B,d)` | Seed-weighted hop representations used for classification |
| `g` | tensor `(B,K*d)` | Concatenated prediction representation |

Most auxiliary tensors are explicitly **detached from autograd** before returning. They are suitable for analysis and visualization, not as intermediate values for an additional backpropagation objective without modifying the model.

---

## 10. Training objective and optimization

### 10.1 Classification loss

Training minimizes **class-weighted cross entropy with label smoothing**:

```python
loss = torch.nn.functional.cross_entropy(
    logits,
    y,
    weight=class_weights,
    label_smoothing=0.05,
)
```

Class weights are calculated separately for each fold using counts from the **training** subset:

$$
w_c\propto\frac{1}{n_c^{\mathrm{train}}}.
$$

The weights are rescaled to sum to the number of classes. Validation and test losses are computed using unweighted cross entropy without label smoothing; the class-weighted training loss is the optimization objective.

### 10.2 Optimizer

The optimizer is **AdamW** with separate parameter groups:

| Group | Learning rate | Weight decay |
|---|---:|---:|
| `encoder` | `--lr` (default `1e-4`) | `--weight_decay` (default `1e-4`) |
| `head` | `--lr` | `--weight_decay` |
| `periphery_seedness` | `--seed_lr` (default `1e-4`) | **0** |

The script clips the global gradient norm to **1.0** before updating parameters:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

The driver performs standard full-precision, single-process PyTorch optimization with AdamW and gradient clipping.

### 10.3 Checkpoint selection and early stopping

The script selects one checkpoint per fold using **validation ROC-AUC**:

- Eligible epochs begin at **epoch 5** (`epoch >= 5`).
- An update occurs when validation ROC-AUC is strictly greater than the previously saved best value.
- `--patience` is the number of consecutive epochs without an eligible validation-AUC improvement; its counter includes the first four epochs.
- After training, the best weights are loaded and train/validation/test metrics and auxiliary tensors are collected again.

`--epochs` should be at least `5` to allow an eligible checkpoint. During training, validation and test metrics are logged each epoch in `training_history.csv`; checkpoint selection and early stopping depend on validation ROC-AUC.

## 11. Site-aware cross-validation

### 11.1 Rationale

ABIDE-I is collected across acquisition sites. Site distribution and diagnostic composition can differ across institutions and scanning protocols. A split that balances diagnosis but ignores acquisition site may inadvertently produce uneven site composition between training, validation, and test partitions.

`main3.py` implements an **iterative multilabel, site-aware partition** to approximately preserve three categorical distributions simultaneously:

1. **Diagnosis** (`label`);
2. **Acquisition site** (`SITE_ID`);
3. **Diagnosis × site combination**.

Each participant is encoded using three categorical indicator tokens, e.g.:

```text
diagnosis=0
site=NYU
diagnosis=0|site=NYU
```

The partitioner greedily prioritizes the **rarest remaining category** and assigns a sample to the split with the highest unmet category target while respecting exact split-size capacities. Random tie-breaking is controlled by a NumPy generator seeded with `--random_seed` (and fold-specific offsets for inner splits).

This approach is an **approximate balancing heuristic**; it does not guarantee identical site × diagnosis counts, especially for rare combinations, and it is **not** leave-one-site-out evaluation.

### 11.2 Outer and inner splits

The default protocol is:

```text
All subjects
   |
   +-- Outer 5-fold partition
         |
         +-- Test fold (~20% of all subjects)
         |
         +-- Remaining subjects (~80%)
                  |
                  +-- Inner training (~85% of outer train)
                  |
                  +-- Inner validation (~15% of outer train)
```

With `--n_splits 5 --inner_val_size 0.15`, the intended approximate overall fractions are:

| Partition | Fraction of all subjects |
|---|---:|
| Training | ~68% |
| Validation | ~12% |
| Test | ~20% |

Integer split sizes are assigned using the **largest-remainder method**. The actual values depend on the cohort size.

The split function checks that the train, validation, and test row-index sets are disjoint, together cover the dataset, and each contain both diagnostic classes. Input data are expected to contain one row per participant.

### 11.3 Missing acquisition-site names

The script maps missing or empty `SITE_ID` values to the categorical label `UNKNOWN` for deterministic partitioning. It does **not** otherwise perform scanner harmonization or use site as a predictive covariate.

### 11.4 Splits manifest

Every run saves `site_aware_splits.json`, containing:

```text
strategy
random_seed
n_splits
inner_val_size
folds:
  fold
  outer_train_indices
  train_indices
  validation_indices
  test_indices
  train_distribution
  validation_distribution
  test_distribution
```

Each distribution records overall count, per-label count, per-site count, and `site|label` joint counts.

The manifest records integer row positions in the input workbook. Preserve its row order when reusing split indices in subsequent analyses.

### 11.5 Single-fold runs

`--fold 1` through `--fold N` runs only the requested fold, but the script still builds and saves the entire split manifest. It does not save the across-fold `cv_results.json` when `--fold` is set.

`--worker_id` changes only the experiment-directory prefix and is **not** PyTorch distributed worker rank management. Running folds on different GPUs requires independently launching commands and assigning `CUDA_VISIBLE_DEVICES` or equivalent device controls.

---

## 12. Command-line arguments

The table below lists the CLI arguments accepted by `main3.py`.

| Argument | Default | Purpose |
|---|---|---|
| `--csv` | `data/abide_subjects_fc_with_demo_260202.xlsx` | **Excel** dataset path despite flag name |
| `--aal_txt` | `data/ROI_MNI_V4.txt` | Text file containing AAL ROI names |
| `--n_rois` | `116` | Number of ROIs; expected FC columns = `N(N-1)/2` |
| `--n_classes` | `2` | Binary ASD/TD classification; evaluation is binary-specific |
| `--d_model` | `128` | Latent dimension `d` |
| `--n_hops` | `3` | Number of graph propagation steps `K` |
| `--depth` | `2` | Number of node-wise FFN blocks per hop |
| `--top_m` | `5` | Number of selected seed ROIs `M` |
| `--seed_std` | `0.5` | Initialization std for trainable seed token |
| `--mlp_hidden` | `128` | Classifier hidden dimension |
| `--dropout` | `0.2` | Dropout inside graph blocks and classifier |
| `--act` | `gelu` | Graph block activation (`gelu`, `relu`) |
| `--fc_norm` | `row` | Parsed normalization setting; training and evaluation pass FC directly to the model |
| `--lr` | `1e-4` | Encoder/head learning rate |
| `--seed_lr` | `1e-4` | Seedness-parameter learning rate |
| `--weight_decay` | `1e-4` | Encoder/head AdamW decay |
| `--batch_size` | `8` | Train, validation, and test batch size |
| `--epochs` | `30` | Maximum epochs per fold |
| `--patience` | `7` | Early-stopping patience in epochs |
| `--inner_val_size` | `0.15` | Validation fraction within outer training pool |
| `--random_seed` | `10` | Random seed and outer site-aware partition seed |
| `--fold` | `None` | Optional 1-based selected outer fold |
| `--n_splits` | `5` | Number of outer folds |
| `--num_workers` | `0` | DataLoader worker subprocesses |
| `--worker_id` | `0` | Filename prefix only, not distributed training |
| `--vis` / `--no-vis` | `False` | Visualization CLI flag; standard training loop saves site-distribution figures |
| `--save_model` / `--no-save_model` | `True` | Save best model checkpoints (not auxiliary `.pt` exports) |

The FC matrix is supplied to `HiTraNet` directly in both training and evaluation.

---

## 13. Running experiments

### 13.1 Default five-fold run

Once all required files and imports are present and the output path is writable:

```bash
python main3.py \
  --csv data/abide_subjects_fc_with_demo_260202.xlsx \
  --aal_txt data/ROI_MNI_V4.txt \
  --n_rois 116 \
  --n_splits 5 \
  --inner_val_size 0.15 \
  --random_seed 10 \
  --epochs 30 \
  --patience 7 \
  --batch_size 8 \
  --d_model 128 \
  --n_hops 3 \
  --top_m 5 \
  --depth 2 \
  --lr 1e-4 \
  --seed_lr 1e-4 \
  --save_model
```

The command uses the default experiment settings. Each experiment directory name contains `worker_id` and a timestamp.

### 13.2 Run only fold 1

```bash
python main3.py --fold 1 --n_splits 5 --random_seed 10
```

This runs one specified outer fold and writes its fold-level outputs.

### 13.3 Run different folds independently

```bash
CUDA_VISIBLE_DEVICES=0 python main3.py --fold 1 --worker_id 1
CUDA_VISIBLE_DEVICES=1 python main3.py --fold 2 --worker_id 2
```

Run additional folds analogously. This is **independent process execution**; `main3.py` contains no multi-GPU synchronization. Output results will land in separate timestamped directories. To report one unified CV result, aggregate the final test metrics from the five fold-level JSON files and ensure all runs used the **same split manifest and input row order**.

### 13.4 Hyperparameter changes

For example:

```bash
python main3.py \
  --top_m 10 \
  --n_hops 4 \
  --d_model 64 \
  --depth 3 \
  --dropout 0.3 \
  --lr 5e-5 \
  --seed_lr 1e-4
```

`M` controls the number of seed channels, `K` controls the number of hop updates and readout dimension, and `d` controls the latent feature width. Hyperparameter selection uses validation results within the outer training partition.

### 13.5 Lightweight model-only smoke test

Use the model without needing the ABIDE spreadsheet:

```bash
python - <<'PY'
import torch
from model import HiTraNet

net = HiTraNet(
    n_rois=12, d_model=16, n_hops=2, top_m=3,
    mlp_hidden=16, n_classes=2, depth=1,
)

A = torch.randn(2, 12, 12)
A = 0.5 * (A + A.transpose(-1, -2))
A.diagonal(dim1=-2, dim2=-1).zero_()

net.eval()
with torch.no_grad():
    logits, aux = net(A)

assert logits.shape == (2, 2)
assert len(aux['p_list']) == 2
assert aux['p_list'][0].shape == (2, 3, 12)
assert torch.allclose(aux['p_list'][-1].sum(dim=-1),
                      torch.ones(2, 3), atol=1e-5)
print('Model smoke test passed')
PY
```

This is a **software shape/invariant check**, not a test of scientific validity.

---

## 14. Evaluation metrics and label conventions

`eval_one_epoch()` computes softmax probabilities from logits:

```python
probs = torch.softmax(logits, dim=1)
p_asd = probs[:, 0]          # ASD (internal label 0)
pred = logits.argmax(dim=1) # 0 = ASD, 1 = TD
```

The code defines **ASD as the positive class** when computing binary metrics:

| Term | Definition |
|---|---|
| `TP` | Actual ASD predicted ASD |
| `FN` | Actual ASD predicted TD |
| `TN` | Actual TD predicted TD |
| `FP` | Actual TD predicted ASD |
| Sensitivity | `TP / (TP + FN)` |
| Specificity | `TN / (TN + FP)` |
| Accuracy | Correct predictions / total subjects |
| F1-score | `f1_score(..., pos_label=0)` |
| ROC-AUC | `roc_auc_score((y_true == 0).astype(int), p_asd)` |

For binary softmax predictions, taking `argmax` is equivalent to an ASD probability threshold of `0.5` (except possible exact ties). The fold-level JSON reports `"threshold": 0.5`.

The validation debug log prints distributions of predicted ASD probabilities within ASD/TD groups and the fraction with `P(ASD) >= 0.5`. These are **diagnostics**, not threshold tuning.

**Reporting convention:** Always state explicitly that **sensitivity is ASD sensitivity** and **specificity is TD specificity**. Otherwise readers can misinterpret the values because internal ASD label is zero.

If one partition contains a single class, AUC is not defined; the script attempts to catch `ValueError` during `roc_auc_score` and returns `nan`. Its split construction normally rejects splits without both classes.

### Cross-validation summary

For a complete run, `main3.py` prints and saves the **mean and population standard deviation** (`np.std`, `ddof=0`) of final outer-test metrics across all folds.

---

## 15. Experiment directory and output files

A complete default run produces a structure similar to:

```text
<experiment_dir>/
├── args.json
├── training_history.csv
├── site_aware_splits.json
├── model_arch.txt
├── fold1_site_dist_pct.png
├── fold1_best_model.pt
├── fold1_metrics.json
├── fold1_best_aux.pt
├── fold2_site_dist_pct.png
├── fold2_best_model.pt
├── fold2_metrics.json
├── fold2_best_aux.pt
├── ...
├── fold5_site_dist_pct.png
├── fold5_best_model.pt
├── fold5_metrics.json
├── fold5_best_aux.pt
└── cv_results.json
```

`model_arch.txt` is written **only when fold 1 is executed**. `cv_results.json` is written only when **all folds** are run in one invocation (`--fold` is omitted). `fold{n}_best_model.pt` is not written if `--no-save_model` is used.

### 15.1 `args.json`

Contains the parsed command-line arguments and defaults. It does **not** record package versions, Git commit, exact output directory root, or upstream FC-preprocessing settings.

### 15.2 `training_history.csv`

One record per epoch and fold:

```text
fold, epoch,
train_loss, train_acc,
val_loss, val_acc, val_auc, val_sens, val_spec, val_f1,
test_loss, test_acc, test_auc, test_sens, test_spec, test_f1,
is_best, stale_epochs
```

The history includes the training, validation, and test metrics collected at every epoch.

### 15.3 `fold{k}_metrics.json`

This file is first written when a new validation-best model is found, then **overwritten after final selected-checkpoint evaluation**. The final file contains **outer-test** metrics plus the selected epoch and validation-selection AUC, for example:

```json
{
  "fold": 1,
  "split_strategy": "iterative_multilabel_diagnosis_site_and_joint",
  "best_epoch": 17,
  "selection_val_auc": 0.71,
  "auc": 0.70,
  "acc": 0.66,
  "sensitivity": 0.65,
  "specificity": 0.67,
  "f1": 0.64,
  "threshold": 0.5,
  "seed_indices": [0, 1, 2, 3, 4],
  "seed_weights": [0.25, 0.22, 0.20, 0.18, 0.15]
}
```

The JSON above shows an example format with illustrative values. In the final file, `auc/acc/sensitivity/specificity/f1` are **test** metrics, while `selection_val_auc` records the validation criterion used for checkpoint selection.

### 15.4 `fold{k}_best_model.pt`

When saving is enabled, the checkpoint contains:

```text
model                 model.state_dict()
args                  parsed CLI arguments
fold                  fold index
split_strategy        site-aware strategy string
best_epoch            selected epoch
selection_val_auc     selected validation AUC
test_metrics          final test metrics
outer_train_indices
train_indices
validation_indices
test_indices
```

The final checkpoint stores the selected model weights, training arguments, data partitions, validation selection AUC, and final test metrics.

### 15.5 `fold{k}_best_aux.pt`

This file is saved **regardless of `--save_model`** and contains additional tensors from the restored best model:

```text
best_epoch, selection_val_auc
train_subject, train_y, train_g, train_p, train_H
val_subject, val_y, val_g, val_p, val_H, val_pred, val_score
test_subject, test_y, test_g, test_p, test_H, test_pred, test_score
seed_indices, seed_weights, test_indices
```

The stored shapes for a particular split with `S` subjects are:

| Saved tensor | Shape | Interpretation |
|---|---|---|
| `{split}_g` | `(S, K, M, d)` | Raw seed-specific trajectory embedding per hop |
| `{split}_p` | `(S, K, M, N)` | Seed-origin influence distribution per hop |
| `{split}_H` | `(S, K, M, N, d)` | Node-level latent states per seed and hop |
| `{split}_y` | `(S,)` | Internal class labels |
| `val_pred`, `test_pred` | `(S,)` | Predicted classes |
| `val_score`, `test_score` | `(S,)` | Predicted ASD probability |
| `seed_indices` | `(M,)` | Ranked selected 0-based ROI indices |
| `seed_weights` | `(M,)` | Weights corresponding to ranked seed indices |

`train_*` entries contain ground-truth labels and trajectory tensors. The transition operators (`P_list`) and all-source arrival matrices (`p_all_list`, `arrival_all`) can be obtained from a direct `model(fc)` forward call.

### 15.6 File-size considerations

`H` is large because it stores full node states for **every subject, hop, seed, ROI, and latent feature**. Rough storage in float32 is:

$$
\text{bytes}(H)\approx 4\times S\times K\times M\times N\times d.
$$

For `S=816`, `K=3`, `M=5`, `N=116`, `d=128`, the `H` tensor alone is approximately **0.73 GB in decimal units** across an entire fold's train/validation/test exports. Serialization overhead and additional tensors add to this. For longer runs, consider exporting only selected subjects, seeds, or pooled trajectories if disk or RAM is constrained.

---

## 16. Loading checkpoints and interpretation tensors

### 16.1 Restore a trained model

```python
import torch
from model import HiTraNet

ckpt_path = 'PATH_TO_EXPERIMENT/fold1_best_model.pt'
# Load only checkpoints you created or obtained from a trusted source.
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
a = ckpt['args']

model = HiTraNet(
    n_rois=a['n_rois'],
    d_model=a['d_model'],
    n_hops=a['n_hops'],
    top_m=a['top_m'],
    mlp_hidden=a['mlp_hidden'],
    n_classes=a['n_classes'],
    depth=a['depth'],
    dropout=a['dropout'],
    seed_std=a['seed_std'],
    act=a['act'],
    use_gated_residual=True,
)
model.load_state_dict(ckpt['model'])
model.eval()

print('Fold:', ckpt['fold'])
print('Best epoch:', ckpt.get('best_epoch'))
print('Validation selection AUC:', ckpt.get('selection_val_auc'))
```

`torch.load` may unpickle Python objects when `weights_only=False`; **do not load untrusted checkpoint files**. The source environment and PyTorch version should also be checked when restoring historical checkpoints.

### 16.2 Read the auxiliary tensor bundle

```python
import torch

bundle_path = 'PATH_TO_EXPERIMENT/fold1_best_aux.pt'
aux = torch.load(bundle_path, map_location='cpu', weights_only=False)

for prefix in ['train', 'val', 'test']:
    g = aux[f'{prefix}_g']
    p = aux[f'{prefix}_p']
    H = aux[f'{prefix}_H']
    print(prefix, 'g:', tuple(g.shape),
          'p:', tuple(p.shape),
          'H:', tuple(H.shape))

print('Seed ROI indices (zero-based):', aux['seed_indices'].tolist())
print('Seed weights:', aux['seed_weights'].tolist())
```

### 16.3 Convert seed indices to AAL ROI names

`model.py` uses **0-based** tensor indices; the Excel columns and the ROI-name file typically use **1-based** indices. For interpretation, convert indices consistently:

```python
from utils import load_roi_names_from_txt

roi_names = load_roi_names_from_txt('data/ROI_MNI_V4.txt')
for index, weight in zip(aux['seed_indices'], aux['seed_weights']):
    i = int(index)
    print(f'ROI{i + 1}: {roi_names[i]} | alpha={float(weight):.5f}')
```

A seed with index `0` corresponds to `ROI1` and to the first parsed ROI name.

### 16.4 Extract a subject's trajectory and influence

```python
# First subject of the selected fold's test partition
subject_idx = 0
seed_rank = 0

trajectory = aux['test_g'][subject_idx, :, seed_rank, :] # (K, d)
influence = aux['test_p'][subject_idx, :, seed_rank, :]   # (K, N)
node_states = aux['test_H'][subject_idx, :, seed_rank, :, :]  # (K,N,d)

print('Trajectory:', trajectory.shape)
print('Influence:', influence.shape)
print('Node states:', node_states.shape)
print('Influence sum per hop:', influence.sum(dim=-1))
```

The row selected by `subject_idx` corresponds to `aux['test_subject'][subject_idx]`; use the saved identifier when merging with diagnostic or clinical metadata.

### 16.5 Generate a per-subject predictions table

```python
import pandas as pd

records = []
for sid, y, yp, score in zip(
    aux['test_subject'], aux['test_y'],
    aux['test_pred'], aux['test_score']
):
    records.append({
        'Subject': str(sid),
        'true_internal_label': int(y),
        'pred_internal_label': int(yp),
        'p_ASD': float(score),
    })

pd.DataFrame(records).to_csv(
    'fold1_test_predictions.csv', index=False
)
```

This script creates a separate CSV from the saved per-subject test predictions.

---

## 17. Interpreting hierarchical trajectories

HiTraNet provides saved tensors for examining seed importance, seed-to-destination influence, and the latent trajectories associated with each participant and propagation hop. The sections below explain how to access and summarize those outputs.

### 17.1 Global seed importance

`seed_indices` and `seed_weights` identify learned seed ROIs and their relative weights in each trained fold. For anatomical reporting, convert the **zero-based** indices to AAL ROI names using `data/ROI_MNI_V4.txt`.

Useful summaries include **top-ranked seeds per fold**, selection frequency across folds, and average weights for frequently selected regions. These summaries provide an overview of which ROIs the trained model uses as origins for its multi-hop representations.

### 17.2 Seed-to-destination redistribution

`p` provides a per-subject destination distribution for each seed and hop. One possible analysis is:

$$
\Delta p_{s,i}^{(k)}=p_{s,i}^{(k)}-p_{s,i}^{(k-1)}.
$$

You can rank ROIs by destination influence, visualize maps of individual subjects, or compare group-averaged distributions. Since each hop has a normalized distribution, changes across hops represent changes in the model's relative influence weights over destination ROIs.

### 17.3 Hierarchical latent trajectories

`g` has one `d`-dimensional embedding per hop and seed. The following analyses are possible:

- Seed-specific trajectories: $g_s^{(1)}\to \cdots \to g_s^{(K)}$.
- Group-level centroids and within-group dispersion at each hop.
- Distances between consecutive hop states, such as $\lVert g_s^{(k+1)}-g_s^{(k)}\rVert_2$.
- Differences in trajectory patterns between ASD and TD subjects, with appropriate uncertainty estimates.
- Relation of trajectory changes to independently defined sensory, association, and transmodal functional-network labels.

For group trajectory plots, **fit the dimensionality-reduction transform jointly** in a consistent reference space, then project all groups and all hops through that **same fitted transform**. Fitting UMAP separately for each group/hop would make apparent movements incomparable.

### 17.4 Subject-level representations used for classification

The auxiliary bundle saves the raw seed trajectories `test_g` with shape `(S,K,M,d)`. For classification, the model computes a seed-weighted feature at each hop and concatenates the hop vectors. Reconstruct the same weighted features using the restored model from Section 16.1:

```python
import torch

# model: restored HiTraNet from Section 16.1
# aux: loaded fold*_best_aux.pt from Section 16.2
seed_indices, alpha = model.topm_seeds()
g_seed = aux['test_g']                      # (S,K,M,d)
alpha = alpha.cpu().to(dtype=g_seed.dtype) # (M,)

subject_hop_embeddings = torch.einsum(
    'm,skmd->skd', alpha, g_seed
)                                           # (S,K,d)
subject_features = subject_hop_embeddings.flatten(start_dim=1)  # (S,K*d)

print(subject_hop_embeddings.shape, subject_features.shape)
```

`model.topm_seeds()` supplies seed weights in the same index order that the model uses for the `g_seed` channel dimension. For anatomical reports, `seed_indices` can be mapped to ROI names using the method in Section 16.3.

### 17.5 Relationship to functional hierarchy

The scientific interpretation considers the organization of brain systems from **sensory** through **association** to **transmodal** regions. HiTraNet provides hop-specific state representations (`H`), seed influence fields (`p`), and pooled trajectories (`g`) that can be related to these systems during downstream analysis.

To study this organization, first assign AAL ROIs to functional-network groups, then summarize seed-origin influence and latent trajectories across hops and diagnostic groups. This allows network-level descriptions of how the model integrates FC information at increasing propagation depths.

### 17.6 Trajectory visualization

The standard `main3.py` training loop writes **site-distribution plots**, fold results, and analysis tensors. The imported functions `extract_seed_step_graphrepr` and `save_umap_seed_trajectories` provide interfaces for separate UMAP-based visualization code; their calls inside the epoch loop are commented out.

For post-training trajectory visualization, use the saved `fold{k}_best_aux.pt` arrays and a common dimensionality-reduction transform across diagnostic groups and hops.
