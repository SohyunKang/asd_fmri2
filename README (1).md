# HiTraNet

**Hierarchical Trajectory Representation Learning from Resting-State Functional Connectivity for Autism Spectrum Disorder**

> **Project status:** Research code accompanying an **unpublished manuscript draft**. The manuscript is a scientific description in progress, not a finalized or peer-reviewed specification. This README documents the behavior of the **currently supplied code** (`main3.py`, `model.py`, `dataset.py`, `utils.py`, and `model2.py`) and explicitly identifies differences between the manuscript draft and the implementation.
>
> **Clinical-use disclaimer:** This is a research prototype for studying autism spectrum disorder (ASD)-related functional connectivity. It is **not** a validated medical device, a clinical diagnostic test, or a substitute for clinical assessment.

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
18. [Manuscript-versus-code correspondence](#18-manuscript-versus-code-correspondence)
19. [Reproducibility and scientific reporting](#19-reproducibility-and-scientific-reporting)
20. [Known limitations and implementation notes](#20-known-limitations-and-implementation-notes)
21. [Troubleshooting](#21-troubleshooting)
22. [Suggested extensions](#22-suggested-extensions)
23. [Citation, data use, and acknowledgments](#23-citation-data-use-and-acknowledgments)

---

## 1. Project overview

**HiTraNet (Hierarchical Trajectory Network)** is a graph-based deep learning framework developed to examine **hierarchical changes in latent representations of resting-state functional connectivity (rsFC)** and to classify **ASD versus typically developing (TD) controls**.

Unlike approaches that flatten a functional connectivity matrix and perform classification directly, HiTraNet represents each participant's brain as a weighted graph and constructs **seed-specific trajectories over successive graph propagation hops**. The central objects are:

- **Learned seed regions:** a small, globally shared set of informative brain regions selected through trainable ROI-level scores.
- **Hop-dependent propagation operators:** participant- and hop-specific transition matrices estimated from latent node states and informed by individual FC matrices.
- **Seed influence fields:** explicit distributions recording where information originating at each selected seed is propagated.
- **Seed-conditioned node states:** latent representations updated recursively through graph propagation and shared transformations.
- **Hierarchical latent trajectories:** one representation per selected seed and hop, which can be inspected for ASD-related differences.
- **ASD classification:** an MLP that integrates weighted seed summaries from all hops.

The research objective is twofold: **predict diagnostic group from rsFC** and **provide a structured, model-derived account of how seed-originated representations change across graph propagation depths**.

**Interpretation boundary:** A sequence of *model propagation hops* is not longitudinal time, a measured neuronal signal trajectory, or direct evidence that activity physically travels along a particular sensory-to-transmodal route. Such biological interpretations require additional analysis and validation.

### Principal inputs and outputs

| Component | Input | Output |
|---|---|---|
| Data loader | Excel file containing 6,670 upper-triangle FC features per subject, labels, and metadata | Symmetric `116 × 116` FC tensor and metadata |
| Seed selection | Learned global ROI-level scores | `M` selected ROI indices and normalized weights |
| Propagation encoder | FC matrix, seed indices, seed weights | Hop-wise node states, influence fields, and trajectories |
| Prediction head | Concatenated weighted hop representations | Binary classification logits |
| Training script | Dataset, parameters, site-aware CV settings | Fold checkpoints, metric files, split manifest, auxiliary interpretation tensors |

---

## 2. Scientific motivation

Autism spectrum disorder is associated with heterogeneous patterns of brain functional connectivity. Functional systems are organized across multiple spatial and functional scales, including sensory, association, and transmodal systems. The manuscript draft motivates HiTraNet by asking whether a predictive model can move beyond identifying **which ROI or edge is important** and instead characterize **how a participant-specific representation evolves across successive levels of graph-based integration**.

The proposed approach responds to three limitations of common FC-based classifiers:

1. **Vectorized FC models** can learn discriminative correlations but do not explicitly preserve an ROI-to-ROI propagation process.
2. **Conventional graph classifiers** may propagate information across a network yet often summarize their explanations as static node/edge importance.
3. **Static explanations** generally do not represent the sequence of seed-originated changes across several propagation depths.

HiTraNet therefore maintains **both** an evolving node-state tensor and an explicit seed influence distribution at each hop. It then summarizes them jointly into a representation that can be used for classification and exploratory visualization.

The manuscript draft is the **scientific motivation and terminology source** for this repository. The equations in Section 8 below describe what the supplied code actually computes, including cases where this differs from draft equations.

---

## 3. Implemented workflow

```text
ABIDE-I preprocessed rs-fMRI regional time series
             |
             |  external preprocessing / QC (not included here)
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

**Important:** The current training script starts with an **already constructed FC Excel file**. It does **not** itself download ABIDE, preprocess fMRI scans, extract ROI time series, calculate correlations, or produce the paper's figures.

---

## 4. Repository structure

The files examined for this README are:

```text
HiTraNet/
├── README.md                   # this document
├── main3.py                    # training, site-aware CV, evaluation, export
├── model.py                    # CURRENT HiTraNet architecture used by main3.py
├── model2.py                   # older/alternative HiTraNet implementation
├── dataset.py                  # Excel FC loader and symmetric reconstruction
├── utils.py                    # seeds, FC normalization helper, meters, ROI names
│
├── data/                       # REQUIRED but not included in the upload
│   ├── abide_subjects_fc_with_demo_260202.xlsx
│   └── ROI_MNI_V4.txt
│
├── visualization/              # REFERENCED but not included in the upload
│   ├── __init__.py
│   └── viz_umap.py             # imported by main3.py
│
└── experiments/                # recommended portable output location
    └── ...
```

### Responsibilities of the supplied modules

| File | Role | Entry points / main classes |
|---|---|---|
| `main3.py` | Full experiment driver | `main`, `train_one_epoch`, `eval_one_epoch`, `build_site_aware_cv_splits` |
| `model.py` | **Active** HiTraNet implementation | `HiTraNet`, `PeripherySeedness`, `SeedHopEncoder`, `EdgeGatedPropagation`, `GraphDeepBlock`, `MLPHead` |
| `dataset.py` | FC reconstruction from spreadsheet | `FCDataset` |
| `utils.py` | Shared utilities | `set_seed`, `normalize_fc`, `accuracy_from_logits`, `AverageMeter`, `load_roi_names_from_txt` |
| `model2.py` | **Legacy/alternative**, not invoked by `main3.py` | `HierarchicalTrajectoryEncoder`, `HierarchicalTrajectoryEncoderDeep`, `TransmodalArrival`, a different `HiTraNet` |

**Do not swap `model.py` and `model2.py` by filename alone.** Their constructors, forward passes, auxiliary outputs, and seed mechanisms differ. The `main3.py` import is specifically `from model import HiTraNet`.

The `visualization/viz_umap.py` import is **unconditional** in `main3.py` even though its experiment-loop visualization calls are commented out. This missing file can cause startup to fail; see [Installation](#5-installation-and-dependencies) and [Troubleshooting](#21-troubleshooting).

---

## 5. Installation and dependencies

### 5.1 Requirements

Recommended environment:

- Python **3.9 or later**, because the CLI uses `argparse.BooleanOptionalAction`; a current Python 3.10/3.11 environment is a reasonable choice.
- PyTorch installed for the machine's CPU/CUDA platform.
- `numpy`, `pandas`, `openpyxl`, `scikit-learn`, `matplotlib`, and **`entmax`**.
- Additional packages may be required by the external `visualization` module if it is restored.

A simple environment setup:

```bash
conda create -n hitranet python=3.11 -y
conda activate hitranet

# Install the appropriate PyTorch build for your platform first.
# Example: use the installation command from https://pytorch.org/get-started/locally/

python -m pip install numpy pandas openpyxl scikit-learn matplotlib entmax
```

`torch` is an additional required dependency; the command above intentionally does **not** assume the correct CUDA wheel for your GPU. A CPU-only environment is sufficient for debugging and smoke tests, although training may be significantly slower.

For a fully reproducible release, replace these unpinned packages with a tested `environment.yml` or a locked dependency file, and save `python --version`, `pip freeze`, the CUDA toolkit version, and the GPU model with each experiment.

### 5.2 Missing visualization dependency

The first lines of `main3.py` include:

```python
from visualization.viz_umap import (
    extract_seed_step_graphrepr,
    save_umap_seed_trajectories,
)
```

The `visualization/` package was **not** among the provided files. Two supported approaches:

1. **Restore** the original `visualization/viz_umap.py` package and its dependencies, **or**
2. If only training is required, **comment out the import** above. The corresponding UMAP calls in the supplied training loop are already commented out.

Passing `--no-vis` is **not enough** to bypass an import error, because imports happen before argument parsing.

### 5.3 Output-directory portability

The current `main3.py` constructs `exp_dir` from a **hard-coded absolute `/data5/.../experiments2/` path**. On another workstation this directory may be absent or inaccessible. Edit the path before running or, preferably, replace its construction with:

```python
output_root = os.environ.get("HITRANET_OUTPUT_DIR", "./experiments")
exp_dir = os.path.join(
    output_root,
    f"w{args.worker_id}_hitranet_{timestamp}",
)
```

Then choose an output directory with:

```bash
export HITRANET_OUTPUT_DIR="$PWD/experiments"
```

> This is a **recommended edit**, not a feature already present in the supplied code. Currently there is **no** `--output_dir` CLI argument.

### 5.4 Basic checks

From the repository root:

```bash
python -m py_compile dataset.py main3.py model.py model2.py utils.py
python -c "import torch, pandas, sklearn, matplotlib, entmax; print('Imports OK')"
python main3.py --help
```

The last command requires that the unconditional visualization import has been resolved.

---

## 6. Dataset and preprocessing

### 6.1 Source dataset

The manuscript draft describes an experiment on **ABIDE-I (Autism Brain Imaging Data Exchange I)**. Its stated analysis cohort consists of:

| Group | Participants in manuscript draft |
|---|---:|
| ASD | 374 |
| TD | 442 |
| **Total** | **816** |

These counts are **reported in the manuscript draft**, not computed or independently checked by the supplied training code. The actual Excel input must be inspected to confirm the cohort used for any particular run.

Subject-level covariates may include age at scan, biological sex coding, motion estimates, acquisition site, and other ABIDE phenotypic measures. The current classifier consumes **FC only**; demographic variables are loaded for partitioning, diagnostics, or identification, not concatenated to model features.

### 6.2 fMRI processing described in the draft

The following describes the manuscript's upstream data construction, **not executable steps implemented in the five uploaded Python files**:

1. Obtain rs-fMRI regional time-series derivatives from the **ABIDE Preprocessed** release (FCP-INDI).
2. Use **C-PAC** derivatives under the `filt_global` strategy, as stated in the draft.
3. Extract regional time series using the **Automated Anatomical Labeling (AAL)** atlas with **116 ROIs**.
4. Retain subjects with complete ROI coverage; exclude subjects with near-zero-variance ROI time series to avoid undefined correlations.
5. Compute the pairwise **Pearson correlation coefficient** between regional time series.
6. Form a symmetric **`116 × 116`** FC matrix for each participant.
7. Store its strict upper triangle (without the diagonal), giving `116 × 115 / 2 = 6,670` connections.
8. Merge participant records with diagnosis and site identifiers and other available phenotypic metadata.

The draft also discusses a mean-framewise-displacement criterion involving **0.2 mm**. Because the raw preprocessing/download scripts and the actual cohort spreadsheet were not provided, **verify the applied motion filtering and QC before describing the criterion as reproduced by this repository**.

### 6.3 Signed functional connectivity

Pearson correlations can take positive or negative values. `FCDataset` reconstructs the symmetric FC matrix **without discarding negative correlations, thresholding, or imposing binary graph edges**. The diagonal is set to zero. At present, the active model receives the resulting **signed, unnormalized FC values**; Section 18 details the implications for the propagation operator.

### 6.4 Preventing information leakage during FC construction

Pearson correlations for a subject are computed entirely from that subject's time series. If adding any **data-driven, cross-subject** transforms (feature selection, site harmonization, learned scaling, imputation, ComBat, normalization fitted across participants), fit them **inside each training partition only** and apply the fitted transform to validation/test subjects.

---

## 7. Input format and data validation

### 7.1 Excel file

The CLI argument `--csv` is historically named but must point to an **Excel workbook** (`.xlsx`) because `dataset.py` uses `pd.read_excel()`.

Default value:

```text
data/abide_subjects_fc_with_demo_260202.xlsx
```

The active call constructs the dataset with:

```python
FCDataset(
    xlsx_path=args.csv,
    n_rois=args.n_rois,
    label_col="DX_GROUP",
)
```

Thus, although `FCDataset` defaults to `label_col='label'` when called independently, **`main3.py` requires `DX_GROUP`**.

### 7.2 Required columns

| Column(s) | Expected content | Use |
|---|---|---|
| `DX_GROUP` | ABIDE diagnostic code | Classification label; usually `1=ASD`, `2=TD` |
| `Subject` | Unique participant ID | Subject identifier saved with results |
| `AGE_AT_SCAN` | Numeric age at scan | Split-distribution summary |
| `SEX` | Dataset's encoded sex value | Split-distribution summary |
| `func_mean_fd` | Numeric mean framewise displacement | Read by loader; currently **not returned** to `main3.py` |
| `Site` | Acquisition-site description | Returned by loader; not used for stratification |
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

The main script asserts that exactly `--n_rois` names were loaded. This file is needed even for a training-only run in the current implementation. The ROI-name ordering must match the indices used for FC construction; the script checks **length only**, not semantic consistency.

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

**This section is derived from the active `model.py` execution path.** Let:

- `B`: minibatch size;
- `N`: number of ROIs (default **116**);
- `M`: selected seeds (default **5**);
- `K`: propagation hops (default **3**);
- `d`: latent dimension (default **128**);
- `C`: output classes (default **2**).

A participant has signed FC matrix $A\in\mathbb{R}^{N\times N}$. In practice, `main3.py` sends the matrix reconstructed by `FCDataset` **directly to the model**.

### 8.1 Learned global seed selection — `PeripherySeedness`

A trainable score vector $u\in\mathbb{R}^{N}$ is initialized from a normal distribution with standard deviation `0.1`:

$$
 u_i \sim \mathcal{N}(0,\,0.1^2).
$$

**Actual implementation:** `PeripherySeedness.forward()` returns the raw scores $u$, **not** `sigmoid(u)`.

The selected seed set is the deterministic top-$M$ set:

$$
\mathcal S=\operatorname{TopM}(u).
$$

Seed weights are obtained by applying **1.5-entmax to all ROI scores**, selecting the chosen entries, and renormalizing:

$$
 a=\operatorname{entmax}_{1.5}(u), \qquad
 \alpha_s=\frac{a_s}{\sum_{j\in\mathcal S}a_j},\quad s\in\mathcal S.
$$

The weights are nonnegative and sum to one across the selected seeds. Entmax is a sparse alternative to softmax. The seeds and their importance weights are **global model parameters**, shared across subjects, whereas the FC-dependent representations and operators are **subject-specific**.

**Optimization detail:** indices are selected by a discrete `torch.topk`; gradients do not flow through the ranking operation itself. Gradients can flow through **selected entmax weights**. There is currently no auxiliary loss directly supervising seed scores.

Although `topm_seeds_train()` generates Gumbel noise internally, its noisy ranking is **not used** in the active code; training and evaluation both select the deterministic top-M scores.

### 8.2 Seed-conditioned initialization — `SeedHopEncoder`

First, each ROI's full FC row (its connectivity profile) is projected into $d$ dimensions:

$$
F_i = W_{\mathrm{FC}} A_{i,:}+b_{\mathrm{FC}},
\quad F\in\mathbb{R}^{N\times d}.
$$

For each selected seed $s$, initialize a separate node-state matrix:

$$
X_{s,i}^{(0)}=F_i+\mathbb{1}[i=s]\,t,
$$

where $t\in\mathbb{R}^{d}$ is a shared trainable **seed token** (normally initialized with `std=--seed_std`). Hence all seed channels begin with the participant's connectivity-profile features, but each receives its seed token at a different originating node.

The influence distribution starts as a one-hot vector:

$$
p_{s,i}^{(0)}=\mathbb{1}[i=s].
$$

The code also creates a trainable `id_embed` lookup but does **not use it** in the current forward computation (`deg_feat = fc_feat`). Do not describe node identity embeddings as an active input to the current classifier.

### 8.3 Shared, FC-informed hop-dependent operator — `EdgeGatedPropagation`

At each hop $k$, compute the seed-weighted latent node context:

$$
\bar H^{(k-1)}=\sum_{s\in\mathcal S}\alpha_s X_s^{(k-1)}
\in\mathbb{R}^{N\times d}.
$$

Learn attention-style queries and keys:

$$
Q^{(k)}=\bar H^{(k-1)}W_q,\qquad
K^{(k)}=\bar H^{(k-1)}W_k.
$$

The **implemented** propagation operator is:

$$
P^{(k)}=\operatorname{RowSoftmax}
\left(\frac{Q^{(k)}(K^{(k)})^\top}{\sqrt{d_k}} + A\right).
$$

`P^(k)` is nonnegative and each row sums to 1. It depends on the individual FC matrix and on **states that have changed across hops**, so it is both subject-specific and hop-dependent.

**Important topology caveat:** The actual operator uses **`score + A`**. It does **not** apply an FC-edge binary mask, does **not** use `log(A)` or elementwise multiplication with a log-scaled adjacency, and does **not** prohibit propagation over zero-valued FC entries. Thus, this release is most accurately described as **FC-informed dense learned propagation**, not strictly edge-masked message passing. Negative FC values lower a logit's contribution relative to positive entries, but they are not removed.

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

This is returned as `p_all_list` and `arrival_all` by `model.py`, but the current `main3.py` **does not save these to its auxiliary `.pt` file**.

### 8.5 Seed-conditioned node-state propagation

The code uses the **transpose** of the learned operator for the node-state message direction:

$$
Z_s^{(k)}=(P^{(k)})^\top X_s^{(k-1)}.
$$

This means destination node `j` aggregates features from originating nodes `i` using `P[i,j]`; it is consistent with a row-vector influence distribution propagated as `p @ P`.

A learnable transformation $W\in\mathbb{R}^{d\times d}$ is shared across hops:

$$
\tilde X_s^{(k)}=Z_s^{(k)}W.
$$

The implemented hop-level residual mix is **not** the simple sum shown in the early draft. Instead:

$$
R_s^{(k)}=0.1\,\tilde X_s^{(k)} +0.9\,X_s^{(k-1)}.
$$

Then the model applies a stack of shared node-wise `GraphDeepBlock`s, followed by LayerNorm:

$$
X_s^{(k)}=\operatorname{LayerNorm}\bigl(\operatorname{DeepBlocks}(R_s^{(k)})\bigr).
$$

Each `GraphDeepBlock` performs:

```text
Input x
   -> LayerNorm
   -> Linear(d, 4d)
   -> GELU (or ReLU)
   -> Dropout
   -> Linear(4d, d)
   -> Dropout
   -> residual: x + sigmoid(gate) * FFN(x)
```

The scalar residual gate is initialized to `-2.0` before the sigmoid, so its initial scale is approximately `0.119`. The `depth` argument controls how many such blocks are used **inside each hop**, without changing the number of graph propagation hops. The same deep block stack and the same transform `W` are reused at all hops.

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

For each hop:

$$
\bar g^{(k)}=\sum_{s\in\mathcal S}\alpha_s g_s^{(k)}.
$$

The current **classifier uses these fixed, globally learned seed weights** `alpha`, not subject-specific attention weights. A subject-specific attention module (`seed_proj`, `seed_attn`, `aggregate_seeds`) exists in the source, but its use in the forward path is **commented out**. It should be treated as experimental/inactive.

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

The active forward pass returns `logits, aux` where `aux` includes:

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
| `g_bar_list` | `K` tensors `(B,d)` | Seed-weighted positions actually used for classification |
| `g` | tensor `(B,K*d)` | Concatenated prediction representation |

Most auxiliary tensors are explicitly **detached from autograd** before returning. They are suitable for analysis and visualization, not as intermediate values for an additional backpropagation objective without modifying the model.

---

## 10. Training objective and optimization

### 10.1 Classification loss

For each train minibatch, `main3.py` uses class-weighted cross entropy with label smoothing:

```python
loss = torch.nn.functional.cross_entropy(
    logits,
    y,
    weight=class_weights,
    label_smoothing=0.05,
)
```

The class weights are recomputed **for each outer CV fold from its training split alone**:

$$
w_c\propto\frac{1}{n_c^{\mathrm{train}}}.
$$

They are rescaled in the code so that their sum equals the number of classes. This increases the relative contribution of the minority diagnosis class.

**Not currently active:** seed-weight entropy is computed in the training function but is **not added to the loss**. The optional entropy-regularized form is only commented out. The active training objective is classification loss alone.

Validation and test losses are calculated using **ordinary, unweighted cross entropy without label smoothing**. Consequently, train and validation loss values do not measure exactly the same objective; compare them cautiously.

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

There is no active learning-rate scheduler, mixed-precision (`autocast`) logic, gradient accumulation, pretrained weight loading, or distributed training in the provided experiment driver.

### 10.3 Checkpoint selection and early stopping

- Main selection criterion: **highest validation ROC-AUC**.
- A checkpoint is eligible for selection **starting at epoch 5** (`epoch >= 5`).
- An improvement is **strict** (`val_auc > best_val_auc`); a tie is not an improvement.
- Only the first model attaining a given best validation AUC is retained, unless a higher AUC occurs later.
- `--patience` counts consecutive epochs without a qualifying improvement, including the warm-up epochs before epoch 5.
- At the end, the script **restores the selected validation-best weights**, evaluates train/validation/test again, and saves fold-level analysis tensors.

Use `--epochs >= 5`; otherwise the script can finish without any eligible checkpoint and raise `RuntimeError: no valid checkpoint was produced`.

**Test-set monitoring warning:** The supplied code currently evaluates and logs test metrics at **every epoch**. The code does **not** use test metrics for checkpoint selection or early stopping, but repeated test monitoring can still encourage human-in-the-loop test-based tuning. For strict final benchmarking, disable per-epoch test evaluation and run the test set **once after validation-based checkpoint selection**. See [Reproducibility](#19-reproducibility-and-scientific-reporting).

---

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

**Implementation-level checks:** The split function verifies that train/validation/test index sets are mutually disjoint, cover the dataset, and contain **both diagnosis classes**. It does **not** independently check unique patient IDs across rows. That is why the data audit in Section 7 is important.

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

**Recommendation:** Reuse the **exact subject partitions** from the manifest when comparing HiTraNet with baselines. The stored indices are positions in a specific spreadsheet; preserve the spreadsheet and its row ordering, or convert them into subject-ID lists before reordering data.

### 11.5 Single-fold runs

`--fold 1` through `--fold N` runs only the requested fold, but the script still builds and saves the entire split manifest. It does not save the across-fold `cv_results.json` when `--fold` is set.

`--worker_id` changes only the experiment-directory prefix and is **not** PyTorch distributed worker rank management. Running folds on different GPUs requires independently launching commands and assigning `CUDA_VISIBLE_DEVICES` or equivalent device controls.

---

## 12. Command-line arguments

The table below reflects arguments actually defined in `main3.py`.

| Argument | Default | Meaning / active behavior |
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
| `--fc_norm` | `row` | **Currently not applied**; model receives raw FC |
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
| `--vis` / `--no-vis` | `False` | Flag present but active UMAP calls are commented out |
| `--save_model` / `--no-save_model` | `True` | Save best model checkpoints (not auxiliary `.pt` exports) |

**Important:** `--fc_norm` does not change experiments in the current code because both calls to `normalize_fc(fc, method=fc_norm)` are commented out in training and evaluation. The existence of the CLI flag should not be interpreted as evidence of row normalization.

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

If no argument is specified, these are approximately the current defaults. The experiment directory name contains `worker_id` and a timestamp.

### 13.2 Run only fold 1

```bash
python main3.py --fold 1 --n_splits 5 --random_seed 10
```

Use this for a short check, not as a substitute for evaluating the entire 5-fold CV protocol.

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

Changes in `M` alter the number of concurrently maintained seed-conditioned node-state channels; changes in `K` affect runtime, representation size, and auxiliary-file size; changes in `d` impact latent width and memory consumption. Hyperparameters should be chosen from validation results within the outer training partition, never from outer test performance.

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

For a full run, `main3.py` prints and saves the **mean and population standard deviation** (`np.std`, `ddof=0`) of final outer-test metrics across folds. Fold scores represent different test subsets; the summary is not an external independent-site validation and should not be described as such.

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

Test values are logged every epoch by the supplied implementation (see the methodological caveat in Section 10).

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

**All numbers and indices above are illustrative placeholders**, not observed results. In the final JSON, `auc/acc/sensitivity/specificity/f1` are **test** metrics. In contrast, `selection_val_auc` is the criterion used to select the checkpoint.

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

The checkpoint is updated on each eligible validation improvement and rewritten after final test evaluation. The exact keys of intermediate checkpoint versions can differ; the schema above describes the **final** saved version.

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

The `train_*` tensors do **not** include `train_pred` or `train_score` in the current export. `P_list`, `p_all_list`, and `arrival_all` are **available from direct model inference but not stored in this `best_aux.pt`**.

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

The test predictions table is a **derived artifact**; `main3.py` does not save that CSV automatically.

---

## 17. Interpreting hierarchical trajectories

The manuscript draft motivates three complementary interpretation levels. The code exposes the quantities needed for initial exploration, but **publication-quality neurobiological conclusions require additional processing and controls**.

### 17.1 Global seed importance

`seed_indices` and `seed_weights` reveal which ROI origins are selected by a trained fold and the relative contribution assigned to those seeds in the classifier. Good practice:

- Report seeds using full **AAL ROI names**, anatomical hemisphere, and `ROI1`-based index.
- Show per-fold selection frequency and weights rather than relying on one fold.
- Avoid claiming that a learned seed proves a causal signal origin or a specific anatomical afferent pathway.
- Recognize that seedness is global per fold, not individualized per participant.

The manuscript draft describes exploratory seed patterns in visual and sensorimotor regions, including superior occipital, postcentral, and precentral areas. Those are **draft analysis observations**, not guaranteed outcomes of every new run; use the actual exported `seed_indices` to verify them.

### 17.2 Seed-to-destination redistribution

`p` provides a per-subject destination distribution for each seed and hop. One possible analysis is:

$$
\Delta p_{s,i}^{(k)}=p_{s,i}^{(k)}-p_{s,i}^{(k-1)}.
$$

You can rank ROIs by destination influence, visualize maps of individual subjects, or compare group-averaged distributions. Because each hop's distribution is normalized, interpret **changes in relative allocation**, not literal transmitted signal energy.

**Warning:** This is a **model-derived influence field**. With the current dense operator, influence is not restricted to nonzero FC graph edges. A high destination weight should not be described as a directly observed anatomical route.

### 17.3 Hierarchical latent trajectories

`g` has one `d`-dimensional embedding per hop and seed. The following analyses are possible:

- Seed-specific trajectories: $g_s^{(1)}\to \cdots \to g_s^{(K)}$.
- Group-level centroids and within-group dispersion at each hop.
- Distances between consecutive hop states, such as $\lVert g_s^{(k+1)}-g_s^{(k)}\rVert_2$.
- Differences in trajectory patterns between ASD and TD subjects, with appropriate uncertainty estimates.
- Relation of trajectory changes to independently defined sensory, association, and transmodal functional-network labels.

For group trajectory plots, **fit the dimensionality-reduction transform jointly** in a consistent reference space, then project all groups and all hops through that **same fitted transform**. Fitting UMAP separately for each group/hop would make apparent movements incomparable.

### 17.4 Subject-level representations used for classification

The main `best_aux.pt` saves **raw seed trajectories** `g` of shape `(S,K,M,d)`. The classifier itself operates on the alpha-weighted aggregate per hop. It can be reconstructed from the exported tensors as:

```python
import torch

g_seed = aux['test_g']              # (S,K,M,d)
alpha = aux['seed_weights'].float() # (M,)
subject_hop_embeddings = (
    g_seed * alpha[None, None, :, None]
).sum(dim=2)                        # (S,K,d)
subject_features = subject_hop_embeddings.flatten(start_dim=1)  # (S,K*d)
```

**Precision note:** The `seed_weights` are saved in order of descending weight, and the saved `g` seed dimension is generated in the original **`torch.topk` seed-index order**. The code sorts weights and indices before export but does **not reorder `g`, `p`, or `H` to the same sorted rank**. Therefore, the multiplication shown above is **valid only if the saved seed order already matches the sorted order** (typically `torch.topk` returns sorted descending scores and entmax is monotonic, but do not rely on this silently). For rigorous reconstruction, explicitly align seed indices and weights using the original model's seed order or verify the correspondence before applying weights. Better yet, save the unsorted `seed_indices`/`seed_weights` and the selected order in a future release.

To align a sorted export with the exact original model order after restoring a checkpoint, use:

```python
# Requires the restored `model` from Section 16.1 and loaded `aux`.
raw_indices, _ = model.topm_seeds()
exported_indices = aux['seed_indices'].tolist()
exported_weights = aux['seed_weights'].float()
weight_by_roi = {int(i): float(w) for i, w in
                 zip(exported_indices, exported_weights)}
aligned_alpha = torch.tensor(
    [weight_by_roi[int(i)] for i in raw_indices.tolist()],
    dtype=aux['test_g'].dtype,
)
subject_hop_embeddings = (
    aux['test_g'] * aligned_alpha[None, None, :, None]
).sum(dim=2)
```

This maps weights by their ROI identities rather than assuming that two independently sorted arrays share the same tie-breaking order.

### 17.5 Relationship to functional hierarchy

The paper motivates a **sensory → association → transmodal** interpretation. The model does **not** explicitly impose anatomical functional-network labels, optimize against a hierarchy score, or require transitions to move monotonically toward default-mode regions. Analyses should therefore distinguish:

- **Observed model behavior:** changes in `p`, `H`, and `g` over hops;
- **External network annotations:** sensory/association/transmodal labels mapped to AAL ROIs;
- **Biological hypothesis:** whether these changes correspond to plausible macroscale functional hierarchy in ASD.

The third step is a hypothesis requiring evaluation, not an automatic inference from increasing `k`.

### 17.6 UMAP status

`main3.py` imports functions named `extract_seed_step_graphrepr` and `save_umap_seed_trajectories`, but the relevant calls in the training loop are **commented out**. The implementation was not included among the supplied files. As a result, this README does **not** promise automatically generated UMAP figures or reproduce the manuscript's figure pipelines from the present five-file snapshot.

---

## 18. Manuscript-versus-code correspondence

**This section is important for reproducibility and for editing the manuscript draft.** The manuscript draft and current source share the same central concept—seed-origin influence tracking and hop-wise latent representations—but several mathematical and experimental details differ.

| Topic | Manuscript draft / intended description | Current `model.py` / `main3.py` behavior | Consequence |
|---|---|---|---|
| Seed score | $r=\sigma(u)\in[0,1]^N$ | Returns **raw** trainable `u_per` | Score scale differs; selected ranking can remain monotonic, but reported values differ |
| Seed weight | $\alpha_s=r_s/\sum_jr_j$ | `entmax15(u)` over all ROIs, select top-M, renormalize | Sparse entmax weights, not normalized sigmoid weights |
| Initial node embedding | Learnable node embeddings + seed token | `Linear(FC row)` + seed token at selected location; `id_embed` unused | Subject FC enters node states immediately |
| FC normalization | Row-normalized FC operator | Normalization calls commented out; **raw FC** passed to model | `--fc_norm` currently has no effect |
| Propagation prior | Draft equation uses elementwise scaling / log FC prior | `softmax(QK^T/sqrt(dk) + A)` | Different operator and FC influence |
| Graph constraint | FC-constrained message passing | **Dense** softmax over all ROIs, no hard edge mask | Do not claim strict topology-masked propagation |
| Direction of node update | Draft writes $P X W+X$ | Uses **$P^\top X W$** internally | Orientation matters when interpreting flow |
| Residual | Draft writes unit residual sum | `0.1 * propagated + 0.9 * input`, then FFNs + LayerNorm | Update strength is explicitly damped |
| Seed aggregation | Global alpha per hop | Global alpha per hop (matches broad draft intent) | No subject-specific seed attention in active classifier |
| Seed selection randomness | Possible Gumbel exploration | Generated noise **unused**; deterministic top-M | No exploration-based stochastic ranking |
| Entropy regularization | Considered in comments | Entropy computed but **excluded from loss** | Classification-only objective |
| CV sampling | Diagnosis-aware/site-aware protocol described conceptually | Exact custom **diagnosis + site + interaction** iterative partition | Cite implementation details and manifest |
| Test metrics | Final held-out evaluation expected | Also monitored **each epoch**; not used in selection code | Best practice is to remove per-epoch test reporting |
| Hierarchy interpretation | Progressive hierarchical integration | Model hop sequence without anatomical hierarchy supervision | Avoid treating hop count as proof of neural hierarchy |
| Manuscript figures | Seeds, propagation patterns, UMAP trajectories | Full visualization code **not provided**; UMAP calls commented | Additional scripts/data required for figure reproduction |

### 18.1 What counts as the current implementation?

For reproducing results from `python main3.py`, the **current code is the source of truth**. Before journal submission or a public release, either:

- **Revise the manuscript's mathematical equations and descriptions** to match the implemented operator and training setup; **or**
- **Change the implementation** to match the intended manuscript formulations, then retrain, reevaluate, and regenerate every reported result and figure.

Do not silently mix manuscript equations from one version with metrics from another code version.

### 18.2 Difference from `model2.py`

The older `model2.py` provides a distinct approach:

- `PeripherySeedness` applies a sigmoid to global ROI scores.
- The encoder starts with an ROI identity basis weighted by seedness (`H0 = diag(s)`) rather than maintaining independent top-M seed-state channels.
- Its recurrent updates use `A @ H` and a learned transform.
- `TransmodalArrival` adds a softmax weighting of final-hop ROIs.
- It pools hop representations and classifies their concatenation.

These are useful as architectural-history or ablation references, but **they are not the architecture called by `main3.py`**. Their outputs (`arrival`, `H_list`, etc.) must not be confused with the current model's `p`, `p_all`, and `g_seed`.

---

## 19. Reproducibility and scientific reporting

### 19.1 Required provenance for a reproducible run

Retain, at minimum:

- **Code revision:** Git commit hash, with clear identification of active `model.py`.
- **Input provenance:** ABIDE source release; preprocessing pipeline, derivative strategy, atlas, QC criteria, subject exclusions; cryptographic hash and row order of the final Excel file.
- **ROI mapping:** exact `ROI_MNI_V4.txt` file, version, and ROI order.
- **Environment:** Python, PyTorch, CUDA/cuDNN, `entmax`, NumPy, pandas, sklearn, and device information.
- **Splits:** complete `site_aware_splits.json`, including the random seed and subject IDs for every split.
- **Hyperparameters:** `args.json` plus any code-level constants such as the `0.1/0.9` residual mix and `label_smoothing=0.05`.
- **Selection rule:** validation ROC-AUC beginning at epoch 5, patience, and checkpoint filename.
- **Final results:** outer-test per-fold metrics, mean/std convention, prediction table, confidence intervals where appropriate.
- **Figures:** analysis script, mapping of AAL to functional networks, and the train/fold/subject population used.

### 19.2 Deterministic settings

`utils.set_seed()` sets seeds for Python `random`, NumPy, CPU PyTorch, and CUDA PyTorch. It also requests deterministic cuDNN behavior and disables cuDNN benchmarking.

This improves repeatability but does **not** guarantee bit-for-bit reproducibility across different PyTorch/CUDA versions, operations, hardware, threading settings, or library versions. Moreover, `model.py` currently generates unused random Gumbel samples during training, which advances the RNG state even though it does not alter selected seed indices. Reproduction requires preserving that code path or removing it consistently across all runs.

### 19.3 Held-out test discipline

Although the supplied script selects its checkpoint only using validation AUC, it prints and stores test metrics every epoch. For a paper-quality pipeline, a recommended change is:

```python
# Inside the epoch loop:
tr = train_one_epoch(...)
va = eval_one_epoch(...)
# Do not evaluate test here.

# Once the validation-best state has been restored:
test_metrics = eval_one_epoch(model, test_loader, ...)
```

If you make this change, also update the CSV header and row writes to remove per-epoch `test_*` columns. The current README reports what the script does **as uploaded**, rather than claiming this change is already present.

### 19.4 Avoiding cohort and site leakage

- The current split is **row-index based**: verify one unique subject per row.
- Site-aware partitioning is **not** the same as train-on-sites-A/test-on-unseen-site-B evaluation.
- Site labels are only used for partitioning; the code does not regress out or harmonize scanner effects.
- Any learned preprocessing fitted across participants belongs **inside each training fold**.
- Do not use outer test results to select architecture, hyperparameters, diagnostic thresholds, or a preferred figure.

### 19.5 Reporting model comparison

For comparisons against MLP, SVM, GNN, Transformer, prototype, counter-condition, or other architectures mentioned in the manuscript, use **identical subject IDs and exact fold assignments**. Results cited in a manuscript draft should not be treated as release-validated benchmark numbers until matched to the frozen input spreadsheet, trained checkpoint files, and reporting code.

### 19.6 Paper figures and interim results

The manuscript draft contains descriptions and preliminary tables/figures. Some result-table cells are unfinished or inconsistent between different metrics (e.g., an AUC table and a narrative accuracy value). For this reason, **this README deliberately does not present manuscript draft performance figures as verified final results**. Populate final performance tables from `cv_results.json` and the per-fold saved outputs only after the experimental protocol is frozen.

---

## 20. Known limitations and implementation notes

1. **Raw FC is actually used.** `--fc_norm` is currently inert; train/test feed `fc` directly to `model(fc)`.
2. **Dense propagation.** `P = softmax(score + A)` is not strictly edge-masked graph message passing. Reproduce experiments with this precise definition.
3. **Subject-specificity is in the states and transitions, not the seed set.** Seed scores and weights are globally trained model parameters shared across participants.
4. **No explicit functional hierarchy labels.** Higher hop count is an analysis scale, not a verified sensory-to-transmodal anatomical path.
5. **Inactive modules exist.** `id_embed`, subject-specific seed attention functions, `gumbel_topk`, and optional entropy regularization are not part of the active forward/loss path.
6. **Test metrics are available during training.** This can introduce inadvertent human-driven model selection bias despite automated validation-only checkpointing.
7. **Visualization is incomplete in this snapshot.** The UMAP import is unresolved without an additional file, and actual UMAP execution is commented out.
8. **Some CLI options do not have their intuitive effect.** `--vis` does not trigger UMAP generation, `--fc_norm` does not normalize inputs, and `--worker_id` is not a distributed worker identifier.
9. **Absolute output path.** Experiment output portability requires a code change or suitable server filesystem.
10. **Potentially large auxiliary saves.** All-subject node states can consume substantial disk space and RAM.
11. **The loader is strict about schema but not all semantics.** It verifies pair-column count and allowed index ranges; use the additional exact-pair and unique-subject audit supplied above.
12. **Phenotypic metadata is not a model input.** Age, sex, site, and motion are not directly used by the classification head.
13. **`func_mean_fd` inconsistency.** The loader reads it but omits it from the returned dictionary, so `main3.py` cannot compute its intended distribution summary even though the Excel file needs the column.
14. **Legacy architecture is not interchangeable.** `model2.py` requires its own calling code and output interpretation.
15. **No finalized inference CLI.** You can restore a checkpoint and call the model programmatically, but a packaged `predict.py` command is not included.
16. **Research-only interpretation.** Visualization of propagation weights should be regarded as an exploratory representation analysis rather than a proven neural mechanism.

---

## 21. Troubleshooting

### `ModuleNotFoundError: No module named 'visualization'`

`main3.py` imports `visualization.viz_umap` at module import time. Restore the missing `visualization/` package, or comment out the import while the visualization code is disabled.

### `ModuleNotFoundError: No module named 'entmax'`

```bash
python -m pip install entmax
```

`model.py` imports `entmax15` inside its active seed-selection functions. This package is necessary even when not generating visualizations.

### `FileNotFoundError` for `data/abide_subjects_fc_with_demo_260202.xlsx`

Use `--csv /absolute/path/to/file.xlsx` or place the input workbook at the default location. Despite the name, `--csv` must currently reference an Excel file.

### `Label column 'DX_GROUP' not found`

The training entry point explicitly requests `label_col='DX_GROUP'`. Verify the workbook header and raw diagnostic coding.

### `Expected 6670 ROI pair columns ... but found ...`

Check the ROI count (`--n_rois`), the `ROIi_ROIj` naming pattern, the exact complete upper-triangle set, and whether Excel columns were renamed, omitted, or duplicated.

### `SITE_ID is required for site-aware splitting`

The dataset must return `SITE_ID`; it is required for the custom split. Check the spreadsheet's exact capitalization and the returned dictionary from `FCDataset`.

### `ROI names (...) != n_rois (...)`

Check `--aal_txt`. The parser reads the second whitespace-delimited column of noncomment lines. Its number of entries must equal `--n_rois`.

### `RuntimeError: Fold X: ... does not contain both classes`

The site-aware split rejected a train, validation, or test portion without both diagnostic groups. Check class prevalence, total cohort size, `--n_splits`, and `--inner_val_size`. Rare site × diagnosis categories are handled approximately, but each entire partition still must contain both classes.

### `RuntimeError: Fold X: no valid checkpoint was produced`

Check that `--epochs >= 5`, that validation AUC is finite from epoch 5 onward, and that folds contain both classes. Check predictions and numerical stability if all AUCs are `nan`.

### CUDA out-of-memory

Try smaller `--batch_size`, `--top_m`, `--d_model`, or `--n_hops`. Remember that states scale with `B*M*N*d`, and each hop builds a dense `(B,N,N)` propagation matrix. Auxiliary export additionally collects full-resolution `H` for every participant.

### Nothing is produced by `--vis`

The flag is parsed but the visualization block is commented out, so no UMAP figure is generated. You need the missing visualization package and must reinstate/integrate the calls to produce those outputs.

### `PermissionError` or output folder creation failure

The experiment root is hard-coded to an absolute server-specific path. Change it to a writable path; see Section 5.3.

### Unexpected ASD sensitivity / specificity

Confirm that the dataset prints `{1: 0, 2: 1}`. The evaluation code uses **internal class 0 as ASD** for probabilities, F1, TP, and sensitivity.

### Apparent mismatch between `seed_weights` and trajectories

See Section 17.4. The final export sorts seed indices/weights, while the representation tensors are stored in model order. Verify and explicitly align ordering when applying exported weights post hoc.

### Why does adding `--fc_norm sym` produce the same results?

The argument is parsed but normalization is not called. Re-enable explicit normalization consistently in **both** training and evaluation if a normalization ablation is intended. This changes the model and requires retraining.

---

## 22. Suggested extensions

The following are **recommendations**, not features already implemented in the supplied repository:

| Priority | Extension | Rationale |
|---|---|---|
| High | Expose `--output_dir` and remove absolute paths | Portability |
| High | Decide and freeze the exact `P^(k)` definition | Consistency between manuscript and code |
| High | Freeze preprocessing provenance and input hashes | Reproducibility |
| High | Stop evaluating the test fold during training | Protect held-out evaluation |
| High | Save unsorted model-order seed indices and weights | Unambiguous trajectory alignment |
| High | Add `requirements.txt` / `environment.yml` with tested versions | Reproducible installation |
| Medium | Make the UMAP import optional and provide visualization scripts | Fully functional analysis pipeline |
| Medium | Save per-subject predictions and `P_list`/`p_all` when needed | Interpretable, auditable output |
| Medium | Add unit tests for FC reconstruction, transition normalization, and saved tensor alignment | Detect regressions |
| Medium | Implement group-aware CV for repeated measurements | Prevent subject leakage in longitudinal/repeat-scan data |
| Medium | Compare signed / absolute / thresholded / masked FC operators | Test the role of graph topology |
| Medium | Evaluate seed stability across CV folds and bootstrap resamples | Interpretation robustness |
| Medium | Evaluate generalization to held-out acquisition sites | Assess multisite robustness |
| Medium | Analyze confounding by site, motion, age, and sex | Scientific validity |
| Research | Validate network-hierarchy interpretation against external atlas labels and clinical phenotypes | Move from model trajectories to biological evidence |

---

## 23. Citation, data use, and acknowledgments

### 23.1 Manuscript status and citation

The related manuscript is currently a **draft** titled:

> *HiTraNet: Hierarchical Trajectory Representation Learning from Resting-State Functional Connectivity for Autism Spectrum Disorder.*

No final author list, publication venue, DOI, repository URL, or release version was provided as part of this documentation package. **Do not invent a published citation.** Once the paper is publicly released, replace this section with the verified publication citation and BibTeX entry, and reference the exact code release corresponding to reported results.

### 23.2 Data use

This project uses data described in the manuscript draft as obtained from **ABIDE-I** and the **ABIDE Preprocessed** resource. Follow the dataset providers' current access, attribution, licensing, and data-use requirements. Do not redistribute identifiable individual-level phenotypic records or scans as part of a public code repository. The input Excel workbook and the ROI-name file were **not** included with the five uploaded source files.

Useful official resources for locating data and documentation:

- ABIDE/INDI project: https://fcon_1000.projects.nitrc.org/indi/abide/
- ABIDE Preprocessed: https://preprocessed-connectomes-project.org/abide/
- PyTorch installation: https://pytorch.org/get-started/locally/

External URLs are provided as resource identifiers; exact availability and applicable terms should be checked when preparing a public release.

### 23.3 Research scope

HiTraNet is intended for **research on latent rsFC representation learning and model-based interpretation**. Results depend on the chosen cohort, preprocessing, site split, hyperparameters, code version, and saved checkpoints. Outputs must not be presented as a clinical ASD diagnosis for an individual person.

---

## Quick reproducibility checklist

Before claiming a result reproduces the manuscript experiment, verify:

- [ ] The **exact 816-subject cohort (374 ASD / 442 TD)** or a documented alternative cohort has been confirmed from the input spreadsheet.
- [ ] FC was built from the recorded **C-PAC / `filt_global` / AAL-116** upstream pipeline with documented QC.
- [ ] Input contains **exactly 6,670** uniquely identified FC pairs and correctly mapped diagnosis labels.
- [ ] `SITE_ID` values and duplicate-subject checks are complete.
- [ ] The model used is **`model.py`**, not `model2.py`.
- [ ] The implemented **raw-FC, dense `softmax(attention + A)` operator** is recorded accurately.
- [ ] Fold assignment is preserved in `site_aware_splits.json` and compared consistently across baselines.
- [ ] Checkpoints were selected using **validation AUC**, not test metrics.
- [ ] All fold-level **final test metrics** were computed from restored validation-best weights.
- [ ] Seed indices are converted from **0-based tensor indices** to the correct AAL label order.
- [ ] Seed weights and saved trajectory tensors are explicitly **aligned** before interpretation.
- [ ] Paper figures and claims distinguish **learned graph hops** from physiological/longitudinal time or proven anatomical routes.
- [ ] The manuscript draft's equations, paper tables, and this released code version have been reconciled.

**Documentation basis:** Supplied source files `dataset.py`, `main3.py`, `model.py`, `model2.py`, `utils.py`, cross-checked against the user's unpublished HiTraNet manuscript draft. This README does not assert that training was run or that the draft's reported results were independently reproduced.
