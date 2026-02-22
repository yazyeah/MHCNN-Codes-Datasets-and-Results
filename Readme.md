# 📌 MHCNN Codes, Datasets and Results

This repository provides a **TensorFlow/Keras** implementation of the paper:

> **MHCNN: Multimodal heterogeneous CNN with modality-cross-attention for robust fault diagnosis with limited samples**

MHCNN targets **intelligent fault diagnosis (IFD)** under two practical constraints:

- **Limited labeled samples** (few-shot regime)
- **Strong measurement noise / unreliable sensors**

It addresses these challenges by combining (i) **heterogeneous modality-specific encoders**, (ii) **bidirectional cross-attention**, and (iii) **sample-wise adaptive modality re-weighting** to produce robust fused representations.

---

## 🏗️ 1. Model Explanation

### 1.1 Problem setup and inputs

For each sample $i$, MHCNN takes **two synchronized 1-D time-domain segments**:

- **Vibration**: $x_i^{(v)} \in \mathbb{R}^{L}$
- **Acoustic** (paper) / **Current** (KAIST scripts in this repo): $x_i^{(a)} \in \mathbb{R}^{L}$

In the paper, the segment length is **$L=2048$** and the classification is over **$K$** health states (Case 1 uses $K=7$).  
In this repository’s **KAIST Load-Shift** scripts, the **second modality is motor current (.tdms)**, but the network design and training protocol follow the same MHCNN pipeline.

---

### 1.2 High-level architecture (MSHEM → CAIM → AMRM → FCB)

MHCNN consists of four coupled modules:

1. **MSHEM — Modality-Specific Heterogeneous Encoder Module**  
   Two 1-D CNN branches designed with **different kernel sizes, activations, and pooling** to match the statistical characteristics of each modality.

2. **CAIM — Cross-Attention Interaction Module (bidirectional)**  
   A **bidirectional** cross-attention block that exchanges information between modalities:
   - $G^{v \leftarrow a}$: vibration features enhanced by acoustic/current  
   - $G^{a \leftarrow v}$: acoustic/current features enhanced by vibration

3. **AMRM — Adaptive Modality Re-Weighting Module**  
   Computes **sample-wise reliability weights** for each modality and performs adaptive fusion so that noisy/unreliable modalities are down-weighted.

4. **FCB — Fusion Classification Block**  
   A compact classifier (Dense → Softmax) that maps fused features to the final health-state probability vector.

---

### 1.3 MSHEM details (heterogeneous encoders)

The core idea is **asymmetry** between modality branches:

- **Vibration branch**: larger kernels (e.g., **$k=16$**) + **Average Pooling** + **LeakyReLU**  
  Motivation: vibration often contains periodic impulsive patterns and benefits from larger receptive fields and smoother pooling.

- **Acoustic branch** (paper) / **Current branch** (KAIST scripts): smaller kernels (e.g., **$k=8$**) + **Max Pooling** + **GELU**  
  Motivation: acoustic/current signals often include transient broadband components where peak-preserving pooling can be beneficial.

A representative encoder pattern is summarized below (exact tensor shapes depend on the number of blocks selected by Optuna):

| Module | Modality | Typical block pattern |
|---|---|---|
| MSHEM | Vibration | Conv1D($k=16$) → LeakyReLU → AvgPool → … (stacked) |
| MSHEM | Acoustic/Current | Conv1D($k=8$) → GELU → MaxPool → … (stacked) |

---

### 1.4 CAIM details (bidirectional cross-attention)

CAIM uses standard attention:

$$
\mathrm{Att}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

but applies it **cross-modally in both directions**:

- **$a \rightarrow v$**: $Q=\mathrm{Vib}$, $K,V=\mathrm{Aco/Cur}$
- **$v \rightarrow a$**: $Q=\mathrm{Aco/Cur}$, $K,V=\mathrm{Vib}$

The result is a pair of **cross-enhanced** representations that preserve each modality’s identity while injecting complementary cues.

---

### 1.5 AMRM details (sample-wise modality reliability)

AMRM estimates **reliability weights** per sample (e.g., two scalars $w_v, w_a$ with $w_v+w_a=1$) and constructs a fused representation:

$$
z_i = [\,w_v \cdot g_i^{(v)} \,;\, w_a \cdot g_i^{(a)}\,]
$$

This helps when one sensor becomes noisy: the model can reduce its influence at inference time.

---

### 1.6 FCB (fusion classification)

The fused vector $z_i$ is passed to a compact classifier:

- Dense (e.g., **256 units**) → activation
- Dense (**$K$ units**) → Softmax

yielding the posterior probability over health states.

---

### 1.7 Training protocol and hyperparameter selection

- Loss: **categorical cross-entropy**
- Optimizer: **Adamax**
- Few-shot evaluation: repeated runs with fixed seeding; report **mean ± std** (and trimmed statistics where enabled)
- Hyperparameters (e.g., number of CNN blocks per branch, attention embedding dimension, dropout, batch size, learning rate) are selected by an **Optuna** search.  
  (In this repo’s KAIST scripts, Optuna searches these parameters in Stage 1, then Stage 2/3 reuse the best setting.)

---

## 📂 2. Dataset Explanation

The paper evaluates MHCNN on two public datasets. This repository focuses on reproducing the **KAIST Load-Shift** setting, while also keeping the **uOttawa (UO)** case organized for reference.

> **Important update:** In this repository, datasets **ARE included** and stored under the `Datasets/` directory (as shown in your local folder structure).  
> If the dataset is large, you should use **Git LFS** to store and pull it correctly (see the reproducibility guide).

---

### 2.1 Case 1 (Paper): UORED-VAFCLS (Vibration + Acoustic)

- Collected on a University of Ottawa bearing test rig (vibration + acoustic).
- Sampling rate: **42 kHz**, duration: **10 s** per recording, speed: **1750 RPM**.
- After segmentation, each class contains **400 vibration** and **400 acoustic** samples, each sample with **2048 points**.
- The paper uses **7 classes** (Healthy / Developing fault / Faulty across defect locations such as inner race, outer race, ball, cage).

**Repository location (default):**
- `Datasets/UO/` (your repo will store the uOttawa dataset here)

---

### 2.2 Case 2 (This repo): KAIST Load-Shift (Vibration `.mat` + Current `.tdms`)

This repository’s reproducibility scripts implement **Load-Shift generalization**:

- **Train/Val:** **0 Nm** only  
- **Test:** **2 Nm** and **4 Nm**

Two synchronized modalities are used:

- **Vibration** stored as MATLAB **`.mat`**
- **Motor current** stored as NI **`.tdms`**

**Repository location (default):**
- `Datasets/KAIST/`

A common on-disk structure is:
- `Datasets/KAIST/vibration/`  (contains `0Nm_*.mat`, `2Nm_*.mat`, `4Nm_*.mat`)
- `Datasets/KAIST/current/`             (contains `0Nm_*.tdms`, `2Nm_*.tdms`, `4Nm_*.tdms`)

> The code itself does **NOT** force a fixed structure: you can place the data anywhere and simply point the script placeholders to your folders.

---

#### 2.2.1 Expected file naming

The KAIST scripts assume the following file naming convention (per load and fault):

- Vibration: `{LOAD}_{FAULT}.mat`  
- Current: `{LOAD}_{FAULT}.tdms` (or `.TDMS`)

Example (0 Nm, Normal):

- `0Nm_Normal.mat`
- `0Nm_Normal.tdms`

The load set for this repo’s Load-Shift experiment is:

- `0Nm`, `2Nm`, `4Nm`

The fault labels depend on the script’s `FAULTS` list (example below is a common 5-class setup):

- `Normal`
- `BPFI_03`, `BPFI_10` (inner race, two severities)
- `BPFO_03`, `BPFO_10` (outer race, two severities)

> If your KAIST files use different names, rename them to match the above pattern (or update the `FAULTS` list and loader accordingly).

---

#### 2.2.2 Segmentation and sampling (important for reproducibility)

- Segment length: **2048 points** (time-domain)
- Per class/per load cap: typically **400 segments**
- Splits:
  - Train/Val: only from **0 Nm**
  - Test: only from **2 Nm** and **4 Nm**
- Leakage control (recommended): a hash-based overlap check ensures train/val segments do not overlap

---

## 🚀 3. Code Reproducibility Guide (KAIST)

This section is written as a **step-by-step checklist** for reproducing the **KAIST Load-Shift** experiment on Windows.

### 0) What you need

- **Windows 10/11** (Windows 11 recommended)
- (Strongly recommended) **NVIDIA GPU with ≥ 8 GB VRAM**
- **Git** + *(Optional)* **Git LFS** (only if you store large files via LFS)
- A terminal: **Anaconda Prompt** (recommended) or **PowerShell**
- This repository cloned locally (including `Datasets/` if you keep datasets inside the repo)

---

### 0.1 Install Git (and Git LFS if needed)

#### 0.1.1 Install Git

1) Download and install Git (official):  
https://git-scm.com/downloads

2) Verify installation:
~~~bash
git --version
~~~

#### 0.1.2 (Optional) Install Git LFS

> Only needed if you track large files using Git LFS.

1) Install Git LFS:  
https://git-lfs.com/

2) Verify installation:
~~~bash
git lfs version
~~~

---

### 0.2 Clone the repository

> Recommended: clone the repo instead of downloading ZIP.

~~~bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_FOLDER>
~~~

#### 0.2.1 (Optional) Pull LFS files

If your repo uses Git LFS:
~~~bash
git lfs install
git lfs pull
~~~

---

### 1) Install Python (Miniconda recommended, If suitable, Anaconda is welcomed)

1) Install **Miniconda** (official):  
https://docs.conda.io/en/latest/miniconda.html

2) Verify Conda:
~~~bash
conda --version
~~~

> Avoid using the system Python shipped with Windows.

---

### 2) Install NVIDIA Driver + CUDA + cuDNN (order matters)

> Skip this section if you will run on CPU only (much slower for Optuna + training).

#### 2.1 Install NVIDIA GPU driver

1) Install driver (official):  
https://www.nvidia.com/Download/index.aspx

2) Verify:
~~~bash
nvidia-smi
~~~

#### 2.2 Install CUDA Toolkit

1) CUDA Toolkit Archive:  
https://developer.nvidia.com/cuda-toolkit-archive

2) Install your CUDA version, then verify:
~~~bash
nvcc --version
~~~

#### 2.3 Install cuDNN

1) cuDNN Archive:  
https://developer.nvidia.com/rdp/cudnn-archive

2) Download the cuDNN version matching your CUDA version.

3) Install cuDNN by copying files into your CUDA directory  
(example CUDA path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\`)

- Copy `bin\cudnn*.dll` → `...\CUDA\vXX.X\bin\`
- Copy `include\cudnn*.h` → `...\CUDA\vXX.X\include\`
- Copy `lib\x64\cudnn*.lib` → `...\CUDA\vXX.X\lib\x64\`

> TensorFlow GPU support depends on TF/CUDA/cuDNN compatibility.  
> If GPU is not detected later, follow TensorFlow official install notes:  
> https://www.tensorflow.org/install

---

### 3) Create and activate the Conda environment

Recommended Python version:
~~~bash
conda create -n mhcnn-kaist python=3.9.20 -y
conda activate mhcnn-kaist
python -V
~~~

---

### 4) Install dependencies

#### Option A: `requirements.txt` (recommended for GitHub)

If your repo provides `requirements.txt`:
~~~bash
pip install --upgrade pip
pip install -r requirements.txt
~~~

#### Option B: install via pip commands (explicit for beginners)

~~~bash
pip install --upgrade pip

# Core deep learning (TensorFlow/Keras)
pip install tensorflow==2.7.0 keras==2.7.0

# Scientific stack + plotting
pip install numpy scipy pandas matplotlib seaborn scikit-learn

# KAIST motor current (.tdms)
pip install nptdms

# Hyperparameter search (Optuna)
pip install optuna optuna-integration
~~~

**Notes**
- `nptdms` is **required** to read KAIST current signals stored in `.tdms`.
- `optuna-integration` may be needed for Keras pruning callbacks depending on Optuna version.

---

### 5) Check TensorFlow GPU availability (one command)

~~~bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
~~~

- If `GPUs: []`, go back to **Step 2** and check:
  - Driver: `nvidia-smi`
  - CUDA: `nvcc --version`
  - cuDNN: copied to CUDA folder correctly
  - TensorFlow GPU notes:  
    https://www.tensorflow.org/install

---

### 6) (Optional) Configure your IDE

#### PyCharm
- Settings → Python Interpreter → Add → Conda Env  
- Select the interpreter from `mhcnn-kaist`

#### VSCode
- Command Palette → `Python: Select Interpreter`  
- Choose the `mhcnn-kaist` environment

---

### 7) Configure path placeholders (MOST IMPORTANT)

Open your main KAIST script and locate placeholders such as:

- `YOUR_TEMP_PATH`
- `YOUR_VIB_MAT_DIR_0NM`, `YOUR_VIB_MAT_DIR_2NM`, `YOUR_VIB_MAT_DIR_4NM`
- `YOUR_CUR_TDMS_DIR`
- `YOUR_OUTPUT_DIR`

Replace them with **your local absolute paths** (any location you like).

#### Recommended setup if datasets are inside this repo

If your repository contains:
- `Datasets/KAIST/vibration`  (contains `0Nm_*.mat`, `2Nm_*.mat`, `4Nm_*.mat`)
- `Datasets/KAIST/current/`   (contains `0Nm_*.tdms`, `2Nm_*.tdms`, `4Nm_*.tdms`)

Then set:

- `YOUR_VIB_MAT_DIR_0NM`, `YOUR_VIB_MAT_DIR_2NM`, `YOUR_VIB_MAT_DIR_4NM` →  
  `...\<repo>\Datasets\KAIST\vibration`

- `YOUR_CUR_TDMS_DIR` →  
  `...\<repo>\Datasets\KAIST\current\`

- `YOUR_OUTPUT_DIR` → any writable folder (recommended inside repo), e.g.  
  `...\<repo>\Outputs\MHCNN_KAIST_Results\`

- `YOUR_TEMP_PATH` → any writable temp folder, e.g.  
  `D:\temp`

#### Quick checklist after replacement

- Vibration folder contains:
  - `0Nm_Normal.mat`, `2Nm_Normal.mat`, `4Nm_Normal.mat`, ...
- Current folder contains:
  - `0Nm_Normal.tdms`, `2Nm_Normal.tdms`, `4Nm_Normal.tdms`, ...
- `YOUR_OUTPUT_DIR` exists (or can be created) and is writable.

---

### 8) Run the script (Terminal + IDE)

#### Terminal
~~~bash
python your_script.py
~~~

#### IDE
Right click → Run

The KAIST benchmark script runs **three stages**:

- **Stage 1 (Optuna)**: hyperparameter search (prints best params)
- **Stage 2 (Few-shot main)**: train on 0 Nm (few-shot), test on 2 Nm & 4 Nm; repeat multiple seeds and report trimmed mean ± std
- **Stage 3 (Noise robustness)**: FAIR “SameModel” protocol — train one model once, then evaluate multiple SNR points with repeated noise draws

> First-run tip: reduce `SEARCH_TRIALS` and/or `SAMPLE_RANGE` to validate your pipeline quickly.

---

### 9) Troubleshooting (FAQ)

**(1) “Cannot find .mat / .tdms”**  
- Naming must match `{LOAD}_{FAULT}.mat/.tdms`.  
- Check that placeholders point to the correct folders.

**(2) `ModuleNotFoundError: No module named 'nptdms'`**  
~~~bash
pip install nptdms
~~~

**(3) TensorFlow cannot find GPU**  
- Re-check install order: Driver → CUDA → cuDNN.  
- Verify:
  - `nvidia-smi`
  - `nvcc --version`
  - GPU list:
    ~~~bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ~~~

**(4) Out-of-memory (OOM) on GPU**  
- Reduce `batch_size` (or rerun Optuna with smaller batch-size search range).  
- Reduce visualization cost (e.g., lower `TSNE_MAX_POINTS`).

**(5) Permission error when writing outputs**  
- Choose an output folder you can write to (avoid protected system folders).

---

## 🚀 4. Code Reproducibility Guide (UO)

This section provides a **step-by-step checklist** for reproducing the **uOttawa (UO, Case 1)** experiments (Vibration + Acoustic) on Windows.

> **Dependencies are the same as KAIST** (TensorFlow/Keras + scientific stack).  
> If you already created the `mhcnn-kaist` Conda environment, you can reuse it for UO directly.

---

### 0) What you need

- **Windows 10/11**
- (Recommended) **NVIDIA GPU with ≥ 8 GB VRAM**
- **Git** + *(Optional)* **Git LFS**
- A terminal: **Anaconda Prompt** (recommended) or **PowerShell**
- This repository cloned locally (including `Datasets/` if you keep datasets inside the repo)

---

### 1) Activate the existing Conda environment (recommended)

If you already created it for KAIST:

~~~bash
conda activate mhcnn-kaist
python -V
~~~

Otherwise, create it (same as KAIST):

~~~bash
conda create -n mhcnn-kaist python=3.9.20 -y
conda activate mhcnn-kaist
python -V
~~~

---

### 2) Install dependencies (same as KAIST)

If `requirements.txt` exists:

~~~bash
pip install --upgrade pip
pip install -r requirements.txt
~~~

Or install explicitly:

~~~bash
pip install --upgrade pip

# Core deep learning (TensorFlow/Keras)
pip install tensorflow==2.7.0 keras==2.7.0

# Scientific stack + plotting
pip install numpy scipy pandas matplotlib seaborn scikit-learn

# Hyperparameter search (Optuna) - only needed if your UO script uses Optuna
pip install optuna optuna-integration
~~~

**Notes**
- `nptdms` is **NOT required** for UO (it is only needed for KAIST motor current `.tdms`).
- UO scripts load `.mat` files only.

---

### 3) Prepare the UO dataset (expected folder layout)

The UO scripts expect the dataset root folder to be named exactly:

- `3_MatLab_Raw_Data/`

and contain subfolders like:

- `1_Healthy/`
- `2_Inner_Race_Faults/`
- `3_Outer_Race_Faults/`
- `4_Ball_Faults/`
- `5_Cage_Faults/`

The scripts typically load `.mat` files such as:

- `H_1_0.mat`, `H_2_0.mat`
- `I_1_1.mat`, `I_2_1.mat`, `I_1_2.mat`, `I_2_2.mat`
- `O_6_2.mat`, `O_7_2.mat`
- `B_11_*.mat`, `B_12_*.mat` (depending on the exact script)
- `C_16_*.mat`, `C_17_*.mat`

> If your filenames differ, either rename them to match the script convention  
> or update the file list inside the loader.

---

### 4) Configure path placeholders (MOST IMPORTANT)

Open your UO scripts and locate placeholders such as:

- `YOUR_UO_DATA_ROOT`
- `YOUR_UO_OUTPUT_DIR`
- (Optional) `YOUR_TEMP_DIR`

Replace them with **your local absolute paths**.

#### 4.1 What each placeholder means

- `YOUR_UO_DATA_ROOT`  
  Must point to the dataset folder named **`3_MatLab_Raw_Data`**.  
  Example (dataset inside this repo):
  - `...\<repo>\Datasets\UO\3_MatLab_Raw_Data`

- `YOUR_UO_OUTPUT_DIR`  
  Any writable directory for saving results (recommended inside repo):
  - `...\<repo>\Outputs\UO_Results\`

- `YOUR_TEMP_DIR` (Optional)  
  Any writable temp folder (Windows), e.g. `D:\temp`.  
  Set to empty string `""` to disable override.

#### 4.2 Quick checklist after replacement

- `YOUR_UO_DATA_ROOT` ends with `...\3_MatLab_Raw_Data`
- Class subfolders exist under `3_MatLab_Raw_Data`
- The `.mat` files referenced by the script exist
- `YOUR_UO_OUTPUT_DIR` exists (or can be created) and is writable

---

### 5) Run the UO scripts

Run the target UO script from terminal:

~~~bash
python your_uo_script.py
~~~

Common UO scripts include:
- The main MHCNN few-shot experiment (Case 1, vib+aco)
- Ablation scripts (Exp1 ~ Exp5), where each script disables specific modules:
  - Exp1: Single Vibration only
  - Exp2: Single Acoustic only
  - Exp3: No CAIM & No AMRM (naive concat)
  - Exp4: w/o Adaptive Modality Fusion (direct concatenation variant)
  - Exp5: With AMRM, No CAIM

---

### 6) Troubleshooting (FAQ)

**(1) “Cannot find .mat”**  
- Verify `YOUR_UO_DATA_ROOT` points to `3_MatLab_Raw_Data`.  
- Verify file names and subfolders match the script.

**(2) TensorFlow cannot find GPU**  
- Same as KAIST: check driver/CUDA/cuDNN installation and TF compatibility.

**(3) Permission error when writing outputs**  
- Choose a writable `YOUR_UO_OUTPUT_DIR` (avoid protected system directories).

**(4) Out-of-memory (OOM)**  
- Reduce `batch_size` and/or reduce visualization cost if applicable.

---

## 📁 5. Generated Results Layout (What You Should See)

All outputs are written under your configured output directory placeholders.

- KAIST scripts write to `YOUR_OUTPUT_DIR/ ...`
- UO scripts write to `YOUR_UO_OUTPUT_DIR/ ...`

---

### 5.1 KAIST (Load-Shift) Results Layout

All outputs are written under your configured `YOUR_OUTPUT_DIR`.

A typical directory layout produced by the KAIST script is:

```text
YOUR_OUTPUT_DIR/
├─ log.txt
├─ Time_Summary.txt
├─ Final_Summary_Stats_raw.csv
├─ Final_Summary_Stats_trimmed.csv
├─ Samples_05/
│  ├─ Run_01/
│  │  ├─ curves.png
│  │  ├─ Test_2Nm/
│  │  │  ├─ cm_2Nm.png
│  │  │  ├─ roc_2Nm.png
│  │  │  └─ tsne_2Nm.png
│  │  └─ Test_4Nm/
│  │     ├─ cm_4Nm.png
│  │     ├─ roc_4Nm.png
│  │     └─ tsne_4Nm.png
│  └─ Run_02/
│     └─ ...
├─ Samples_06/
│  └─ ...
├─ ...
└─ NoiseStudy_LoadShift_FAIR_SameModel/
   ├─ Noise_Robustness_LoadShift_SameModel_FAIR.csv
   ├─ radar_2Nm.png
   ├─ radar_4Nm.png
   └─ BaselinePlots_SNR0dB/
      ├─ cm_2Nm.png
      ├─ roc_2Nm.png
      ├─ tsne_2Nm.png
      ├─ cm_4Nm.png
      ├─ roc_4Nm.png
      └─ tsne_4Nm.png
```

---

### 5.2 UO (Case 1: Vibration + Acoustic) Results Layout

UO scripts do **not** contain the load-specific split (no `Test_2Nm/` and `Test_4Nm/`), so the directory structure is simpler.

All outputs are written under your configured `YOUR_UO_OUTPUT_DIR`.

A typical directory layout produced by **UO ablation scripts (Exp1 ~ Exp5)** is:

```text
YOUR_UO_OUTPUT_DIR/
└─ MHCNN_Ablation_Experiment/
   ├─ Exp1_Single_Vib/
   │  ├─ experiment_log_Exp1.txt
   │  ├─ Final_Summary_Stats_Exp1.csv
   │  ├─ performance_trend_exp1.png
   │  ├─ Samples_05/
   │  │  ├─ Run_01/
   │  │  │  ├─ curves.png
   │  │  │  ├─ cm.png
   │  │  │  ├─ tsne.png
   │  │  │  └─ roc.png
   │  │  └─ Run_02/
   │  │     └─ ...
   │  ├─ Samples_06/
   │  │  └─ ...
   │  └─ ...
   ├─ Exp2_Single_Aco/
   │  ├─ experiment_log_Exp2.txt
   │  ├─ Final_Summary_Stats_Exp2.csv
   │  ├─ performance_trend_exp2.png
   │  └─ Samples_05/Run_01/... (same pattern)
   ├─ Exp3_No_CAIM_No_AMRM/
   │  ├─ experiment_log_Exp3.txt
   │  ├─ Final_Summary_Stats_Exp3.csv
   │  ├─ performance_trend_exp3.png
   │  └─ Samples_05/Run_01/... (same pattern)
   ├─ Exp4_NoFusion/
   │  ├─ experiment_log_Ablation4.txt
   │  ├─ Final_Summary_Stats.csv
   │  ├─ performance_trend_ablation4.png
   │  └─ Samples_05/Run_01/... (same pattern)
   └─ Exp5_With_AMRM_No_CAIM/
      ├─ experiment_log_Exp5.txt
      ├─ Final_Summary_Stats_Exp5.csv
      ├─ performance_trend_exp5.png
      └─ Samples_05/Run_01/... (same pattern)
```

A typical directory layout produced by **UO main MHCNN few-shot experiment** (if your main script uses the same convention) is:

```text
YOUR_UO_OUTPUT_DIR/
└─ MHCNN_UO_Results/
   ├─ log.txt (or experiment_log.txt)
   ├─ Final_Summary_Stats_raw.csv (optional)
   ├─ Final_Summary_Stats_trimmed.csv (optional)
   ├─ performance_trend.png
   ├─ Samples_05/
   │  ├─ Run_01/
   │  │  ├─ curves.png
   │  │  ├─ cm.png
   │  │  ├─ tsne.png
   │  │  └─ roc.png
   │  └─ Run_02/
   │     └─ ...
   ├─ Samples_06/
   │  └─ ...
   └─ ...
```

---

### What each item means (applies to both KAIST and UO)

* `log.txt` / `experiment_log_*.txt`  
  Full run log (stdout/stderr tee), including key parameters and per-run progress.

* `Time_Summary.txt` (KAIST only)  
  A simple time breakdown for Stage 1/2/3 and total runtime.

* `Final_Summary_Stats_raw.csv` / `Final_Summary_Stats_trimmed.csv` (KAIST)  
  Aggregated metrics over repeated runs for each `Samples = N`, typically reported as **mean ± std** (and trimmed mean ± std when enabled).

* `Final_Summary_Stats_Exp*.csv` / `Final_Summary_Stats.csv` (UO ablations)  
  Aggregated metrics over repeated runs for each `Samples = N`, typically using **trimmed mean ± std**.

* `performance_trend*.png`  
  Performance curves (Accuracy/F1 vs training samples per class).

* `Samples_XX/Run_YY/`  
  Artifacts for one run:

  * `curves.png`: training curves (loss/accuracy)
  * `cm*.png`: confusion matrix
  * `roc*.png`: ROC curves (one-vs-rest + macro/micro depending on script)
  * `tsne*.png`: 2D embedding visualization of learned features

* `Test_2Nm/` and `Test_4Nm/` (KAIST only)  
  Evaluation artifacts for each target load.

* `NoiseStudy_LoadShift_FAIR_SameModel/` (KAIST only)  
  Noise robustness results under the FAIR “SameModel” protocol.

---

## 📚 Citation

If you use this code in your work, please cite the corresponding paper.
