# -*- coding: utf-8 -*-
"""
CFFN baseline experiment script (Case 1: uOttawa vibration + acoustic)

Protocol
---------------------------------------------
- Same model capacity and hyperparameters for ALL sample sizes S.
- Validation is ALWAYS held out from TRAIN (stratified per class).
- Test set is ONLY used for final evaluation (never used for model selection).

Path / Environment Convention (KAIST-style placeholders)
-------------------------------------------------------
# Paths
- Replace the placeholders below with YOUR local absolute paths.
- UO dataset root MUST point to the folder: "3_MatLab_Raw_Data"
  (it contains subfolders like "1_Healthy", "2_Inner_Race_Faults", ...).
- Output folder can be ANYWHERE you want.

# Environment (optional)
- CUDA_VISIBLE_DEVICES: GPU id string (default "0")
- TF_CPP_MIN_LOG_LEVEL: default "2"
- TEMP_DIR: optional temp directory on Windows; set to "" to disable.
"""

import os
import sys
import warnings
import random
import gc

# =========================
# 0) Runtime Environment (MUST be set before importing TensorFlow)
# =========================
GPU_ID = "0"  # <-- change if needed, e.g., "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional Windows temp dir override (set to "" to disable)
TEMP_DIR = r"YOUR_TEMP_DIR"  # e.g., r"D:\temp" ; or "" to disable
if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, MaxPooling1D, BatchNormalization, Activation,
    GlobalAveragePooling1D, Dropout
)
from keras.utils import np_utils
from keras.regularizers import l2

import seaborn as sns
from scipy.io import loadmat

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ====== Plot style ======
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 1) Global config =================
SAMPLE_RANGE  = range(5, 31)
REPEAT_TIMES  = 10

DATA_POINTS   = 2048
NUM_CLASSES   = 7

# =========================
# Paths (placeholders)
# =========================
DATA_PATH_ROOT  = r"YOUR_UO_DATA_ROOT"     # e.g., r"D:\uOttawa\3_MatLab_Raw_Data"
BASE_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"    # e.g., r"D:\Results\CFFN_UO"

# Safety checks (fail fast if user forgets to edit placeholders)
if "YOUR_" in DATA_PATH_ROOT:
    raise ValueError(
        "[Path Error] DATA_PATH_ROOT is still a placeholder.\n"
        "Please set DATA_PATH_ROOT to your local '3_MatLab_Raw_Data' folder."
    )
if "YOUR_" in BASE_OUTPUT_DIR:
    raise ValueError(
        "[Path Error] BASE_OUTPUT_DIR is still a placeholder.\n"
        "Please set BASE_OUTPUT_DIR to where you want to save outputs."
    )
if not os.path.isdir(DATA_PATH_ROOT):
    raise FileNotFoundError(
        f"[Path Error] DATA_PATH_ROOT does not exist:\n  {DATA_PATH_ROOT}\n"
        "It must point to the folder '3_MatLab_Raw_Data'."
    )
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

EPOCHS         = 80
LEARNING_RATE  = 1.3866e-3
BATCH_SIZE     = 16
OPTIMIZER_NAME = "Adamax"

CLIP_VALUE = 3.0
EPS = 1e-8

# ---- CCA numerics ----
CCA_EPS = 1e-10

# =========================
# Fixed hyperparameters (S-INVARIANT)
# =========================
# CCA (fixed)
CCA_REG_FIXED   = 3.8e-3
CCA_CAP_MAX     = 28 

# EDCN (fixed)
EDCN_HEAD_DROPOUT = 0.18
EDCN_L2W          = 1e-4
EDCN_BASE_FILTERS = 32
EDCN_FEAT_DIM     = 128

# Classifier MLP (fixed)
CLF_HIDDEN  = 40
CLF_DROPOUT = 0.40
CLF_L2W     = 3e-4

# ---- Validation ----
VAL_FRACTION_PER_CLASS = 0.2  # 20% of the per-class training set
VAL_MIN_PER_CLASS      = 1    # must be >=1 for early stopping
def get_val_train_per_class(num_samples: int) -> int:
    v = int(np.ceil(num_samples * VAL_FRACTION_PER_CLASS))
    v = max(VAL_MIN_PER_CLASS, v)
    v = min(v, num_samples - 1)  # ensure train has at least 1 per class
    return v

# ---- EarlyStopping with min-epoch protection ----
MIN_EPOCH_EDCN = 20
MIN_EPOCH_CLF  = 20
PATIENCE_EDCN  = 10
PATIENCE_CLF   = 10

# ---- t-SNE ----
TSNE_ONCE_PER_SAMPLE = True
TSNE_MAX_POINTS = 1200
TSNE_N_ITER = 600
TSNE_RUN_FOR_PLOTTING = 1

# ---- per-run plots ----
PLOT_CURVES_PER_RUN = True
PLOT_CM_PER_RUN     = True
PLOT_ROC_PER_RUN    = True

# ---- model.summary print ----
PRINT_MODEL_SUMMARY_ONCE = True
SUMMARY_AT_SAMPLE = 5
SUMMARY_AT_RUN    = 1
_printed_summary_once = False


# ================= 2) Reproducibility =================
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ================= 3) Logging =================
class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ================= 4) Data loading =================
def load_all_data():
    print("[Data] Loading dataset...")
    tot_num0 = 200

    def load_mat_category(folder, filename):
        path = os.path.join(DATA_PATH_ROOT, folder, filename)
        try:
            data = loadmat(path)
            key = filename.replace(".mat", "")
            if key not in data:
                keys = [k for k in data.keys() if not k.startswith("__")]
                if keys:
                    key = keys[0]
            val = data[key]
            vib = val[:, 0]
            aco = val[:, 1]
            vib_samples = np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            aco_samples = np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            for i in range(tot_num0):
                st = i * DATA_POINTS
                ed = (i + 1) * DATA_POINTS
                if ed <= len(vib):
                    vib_samples[i, :] = vib[st:ed]
                    aco_samples[i, :] = aco[st:ed]
            return vib_samples, aco_samples
        except Exception as e:
            print(f"[Load Error] {path}: {e}")
            return (
                np.zeros((tot_num0, DATA_POINTS), dtype=np.float32),
                np.zeros((tot_num0, DATA_POINTS), dtype=np.float32),
            )

    def load_class_data(file_list):
        v_list, a_list = [], []
        for folder, file in file_list:
            v, a = load_mat_category(folder, file)
            v_list.append(v)
            a_list.append(a)
        return np.vstack(v_list), np.vstack(a_list)

    datasets = []
    datasets.append(load_class_data([("1_Healthy", "H_1_0.mat"), ("1_Healthy", "H_2_0.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_1.mat"), ("2_Inner_Race_Faults", "I_2_1.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_2.mat"), ("2_Inner_Race_Faults", "I_2_2.mat")]))
    datasets.append(load_class_data([("3_Outer_Race_Faults", "O_6_2.mat"), ("3_Outer_Race_Faults", "O_7_2.mat")]))
    datasets.append(load_class_data([("4_Ball_Faults", "B_11_2.mat"), ("4_Ball_Faults", "B_12_2.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_1.mat"), ("5_Cage_Faults", "C_17_1.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_2.mat"), ("5_Cage_Faults", "C_17_2.mat")]))
    print("[Data] Loading complete.")
    return datasets

def pack_datasets_to_global_arrays(datasets):
    Xv_all, Xa_all, y_all = [], [], []
    class_ranges = []
    cursor = 0
    for label, (vib, aco) in enumerate(datasets):
        n = vib.shape[0]
        Xv_all.append(vib)
        Xa_all.append(aco)
        y_all.append(np.full((n,), label, dtype=np.int64))
        class_ranges.append((cursor, cursor + n))
        cursor += n
    Xv_all = np.vstack(Xv_all).astype(np.float32)
    Xa_all = np.vstack(Xa_all).astype(np.float32)
    y_all = np.concatenate(y_all)
    return Xv_all, Xa_all, y_all, class_ranges

def get_split_indices_by_class(class_ranges, seed, num_train_per_class):
    train_idx, test_idx = [], []
    for (st, ed) in class_ranges:
        idx = np.arange(st, ed)
        np.random.seed(seed)
        np.random.shuffle(idx)
        train_idx.append(idx[:num_train_per_class])
        test_idx.append(idx[num_train_per_class:])
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)
    np.random.seed(seed)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


# ================= 5) Normalize =================
def standardize_clip_train_only_1d(x_train, x_test, clip_value=3.0):
    mu = np.mean(x_train)
    sd = np.std(x_train) + EPS
    x_train_n = (x_train - mu) / sd
    x_test_n  = (x_test  - mu) / sd
    x_train_n = np.clip(x_train_n, -clip_value, clip_value)
    x_test_n  = np.clip(x_test_n,  -clip_value, clip_value)
    return x_train_n.astype(np.float32), x_test_n.astype(np.float32)


# ================= 6) EDCN =================
def build_edcn_branch(input_shape=(2048, 1), num_classes=NUM_CLASSES, prefix="V",
                      head_dropout=0.18, l2w=1e-4,
                      base_filters=32, feat_dim=128):
    bf = int(base_filters)

    inp = Input(shape=input_shape, name=f"{prefix}_in")

    x = Conv1D(bf, 64, padding="same", strides=1, kernel_regularizer=l2(l2w), name=f"{prefix}_conv1")(inp)
    x = BatchNormalization(name=f"{prefix}_bn1")(x)
    x = Activation("relu", name=f"{prefix}_relu1")(x)
    x = MaxPooling1D(pool_size=16, strides=16, padding="same", name=f"{prefix}_pool1")(x)

    x = Conv1D(bf * 2, 32, padding="same", strides=1, kernel_regularizer=l2(l2w), name=f"{prefix}_conv2")(x)
    x = BatchNormalization(name=f"{prefix}_bn2")(x)
    x = Activation("relu", name=f"{prefix}_relu2")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_pool2")(x)

    x = Conv1D(bf * 4, 32, padding="same", strides=1, kernel_regularizer=l2(l2w), name=f"{prefix}_conv3")(x)
    x = BatchNormalization(name=f"{prefix}_bn3")(x)
    x = Activation("relu", name=f"{prefix}_relu3")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_pool3")(x)

    x = Conv1D(bf * 8, 16, padding="same", strides=1, kernel_regularizer=l2(l2w), name=f"{prefix}_conv4")(x)
    x = BatchNormalization(name=f"{prefix}_bn4")(x)
    x = Activation("relu", name=f"{prefix}_relu4")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_pool4")(x)

    x = GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    feat = Dense(int(feat_dim), activation="relu", kernel_regularizer=l2(l2w), name=f"{prefix}_fc1")(x)

    head = Dropout(head_dropout, name=f"{prefix}_head_dropout")(feat)
    out  = Dense(num_classes, activation="softmax", kernel_regularizer=l2(l2w), name=f"{prefix}_out")(head)

    model_cls  = Model(inp, out,  name=f"EDCN_{prefix}_CLS")
    model_feat = Model(inp, feat, name=f"EDCN_{prefix}_FEAT")
    return model_cls, model_feat


# ================= 7) Helpers =================
def stratified_pick_indices(y, per_class, seed=42):
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    picked = []
    for c in range(NUM_CLASSES):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        k = min(int(per_class), len(idx_c))
        if k > 0:
            picked.append(idx_c[:k])
    if len(picked) == 0:
        return np.arange(len(y))
    idx = np.concatenate(picked)
    rng.shuffle(idx)
    return idx

def stratified_holdout_from_train(y_train, per_class_val, seed=42):
    """
    Return (train_keep_idx, val_idx) indices in [0..len(y_train)-1]
    """
    y_train = np.asarray(y_train, dtype=np.int64)
    val_idx = stratified_pick_indices(y_train, per_class=per_class_val, seed=seed)
    mask = np.ones(len(y_train), dtype=bool)
    mask[val_idx] = False
    train_keep_idx = np.where(mask)[0]
    return train_keep_idx, val_idx

class BestWeights2Branch(tf.keras.callbacks.Callback):
    def __init__(self, model_v, model_a):
        super().__init__()
        self.model_v = model_v
        self.model_a = model_a
        self.best_v = -np.inf
        self.best_a = -np.inf
        self.best_w_v = None
        self.best_w_a = None
        self._resolved = False
        self.key_v = None
        self.key_a = None

    def _resolve(self, logs):
        keys = list(logs.keys())
        for k in ["val_V_out_accuracy", "val_v_out_accuracy"]:
            if k in logs:
                self.key_v = k
                break
        for k in ["val_A_out_accuracy", "val_a_out_accuracy"]:
            if k in logs:
                self.key_a = k
                break
        if self.key_v is None:
            for k in keys:
                if k.startswith("val_") and ("V_out" in k or "v_out" in k) and "accuracy" in k:
                    self.key_v = k
                    break
        if self.key_a is None:
            for k in keys:
                if k.startswith("val_") and ("A_out" in k or "a_out" in k) and "accuracy" in k:
                    self.key_a = k
                    break
        self._resolved = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (not self._resolved) and len(logs) > 0:
            self._resolve(logs)

        v = logs.get(self.key_v, None) if self.key_v else None
        a = logs.get(self.key_a, None) if self.key_a else None

        if v is not None and v > self.best_v:
            self.best_v = v
            self.best_w_v = self.model_v.get_weights()
        if a is not None and a > self.best_a:
            self.best_a = a
            self.best_w_a = self.model_a.get_weights()

    def on_train_end(self, logs=None):
        if self.best_w_v is not None:
            self.model_v.set_weights(self.best_w_v)
        if self.best_w_a is not None:
            self.model_a.set_weights(self.best_w_a)

class EarlyStoppingMinEpoch(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", mode="min", patience=10, min_epoch=20, restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.best = None
        self.best_weights = None
        self._better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            return

        if self.best is None:
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        if epoch < self.min_epoch:
            if self._better(current, self.best):
                self.best = current
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            return

        if self._better(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.restore_best_weights and (self.best_weights is not None):
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True


# ================= 8) Offline CCA (fit + transform) =================
def cca_fit(Xtr, Ytr, reg, max_components):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)

    N  = Xtr.shape[0]
    dx = Xtr.shape[1]
    dy = Ytr.shape[1]
    d  = min(dx, dy)

    mx = Xtr.mean(axis=0, keepdims=True)
    my = Ytr.mean(axis=0, keepdims=True)
    Xc = Xtr - mx
    Yc = Ytr - my

    denom = max(N - 1, 1)
    Cxx = (Xc.T @ Xc) / denom + reg * np.eye(dx)
    Cyy = (Yc.T @ Yc) / denom + reg * np.eye(dy)
    Cxy = (Xc.T @ Yc) / denom
    Cyx = Cxy.T

    iCxx = np.linalg.pinv(Cxx)
    iCyy = np.linalg.pinv(Cyy)

    M = iCxx @ Cxy @ iCyy @ Cyx
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    Wx_all = eigvecs[:, idx]

    k = min(d, max(N - 1, 1))
    if max_components is not None:
        k = min(k, int(max_components))
    k = max(k, 1)

    eigvals_k = np.maximum(eigvals[:k], 0.0)
    Wx = Wx_all[:, :k]

    denom_sqrt = np.sqrt(eigvals_k + CCA_EPS)
    Wy = (iCyy @ Cyx @ Wx) / denom_sqrt[np.newaxis, :]

    params = dict(mx=mx, my=my, Wx=Wx, Wy=Wy, k=int(k))
    return params

def cca_transform(X, Y, params):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mx, my, Wx, Wy = params["mx"], params["my"], params["Wx"], params["Wy"]
    X0 = (X - mx) @ Wx
    Y0 = (Y - my) @ Wy
    return np.concatenate([X0, Y0], axis=1).astype(np.float32)


# ================= 9) best_parameters.txt =================
def write_best_parameters_txt(out_dir: str):
    txt_path = os.path.join(out_dir, "best_parameters.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("CFFN baseline hyperparameters (S-invariant; no leakage; no manual suppression)\n")
        f.write("=" * 86 + "\n\n")
        f.write(f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}\n")
        f.write("Validation policy: ALWAYS hold out from TRAIN (stratified per class).\n")
        f.write(f"Val per class = ceil(S * {VAL_FRACTION_PER_CLASS}) bounded to [1, S-1]\n\n")

        f.write("[Fixed Hyperparameters]\n")
        f.write(f"CCA: reg={CCA_REG_FIXED}, cap_max={CCA_CAP_MAX} (final k <= min(dx,dy,N-1,cap_max))\n")
        f.write(f"EDCN: base_filters={EDCN_BASE_FILTERS}, feat_dim={EDCN_FEAT_DIM}, drop={EDCN_HEAD_DROPOUT}, l2={EDCN_L2W}\n")
        f.write(f"CLF: hidden={CLF_HIDDEN}, drop={CLF_DROPOUT}, l2={CLF_L2W}\n\n")

        f.write(f"[EarlyStopMinEpoch] EDCN(min={MIN_EPOCH_EDCN},pat={PATIENCE_EDCN}), CLF(min={MIN_EPOCH_CLF},pat={PATIENCE_CLF})\n")
        f.write(f"[t-SNE] once per sample: max_points={TSNE_MAX_POINTS}, n_iter={TSNE_N_ITER}\n")


# ================= 10) Main =================
if __name__ == "__main__":
    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_CFFN.txt")
    sys.stdout = Logger(log_path, sys.stdout)

    write_best_parameters_txt(BASE_OUTPUT_DIR)

    print(">>> CFFN baseline experiment (Case 1) <<<")
    print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
    if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
        print(f"TEMP_DIR            = {TEMP_DIR}")
    print(f"BASE_OUTPUT_DIR     = {BASE_OUTPUT_DIR}")
    print(f"DATA_PATH_ROOT      = {DATA_PATH_ROOT}")
    print(f"SAMPLE_RANGE        = {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}")
    print(f"REPEAT_TIMES        = {REPEAT_TIMES}")
    print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}")
    print("Validation: ALWAYS from TRAIN (stratified holdout).")
    print(f"Fixed HP: CCA(reg={CCA_REG_FIXED}, cap_max={CCA_CAP_MAX}); "
          f"EDCN(base={EDCN_BASE_FILTERS}, feat={EDCN_FEAT_DIM}); "
          f"CLF(hidden={CLF_HIDDEN})")
    print(f"t-SNE: once per sample={TSNE_ONCE_PER_SAMPLE}, max_points={TSNE_MAX_POINTS}, n_iter={TSNE_N_ITER}\n")

    ALL_DATASETS = load_all_data()
    Xv_time, Xa_time, y_all, class_ranges = pack_datasets_to_global_arrays(ALL_DATASETS)
    print(f"[Data] Total samples: {Xv_time.shape[0]} (per modality), classes={NUM_CLASSES}")

    summary_stats = []

    for num_samples in SAMPLE_RANGE:
        # fixed hp
        cca_reg = CCA_REG_FIXED
        cap_k   = min(CCA_CAP_MAX, num_samples - 1)  # numeric feasibility (not tuning)

        sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        vpc = get_val_train_per_class(num_samples)
        print(f"\n======== Training samples per class: {num_samples} (Runs 1-{REPEAT_TIMES}) ========")
        print(f"  [ValMode] train-holdout val: {vpc}/class")
        print(f"  [FixedHP] CCA(reg={cca_reg:g}, cap_max={CCA_CAP_MAX}, cap_used={cap_k}) | "
              f"EDCN(base={EDCN_BASE_FILTERS}, feat={EDCN_FEAT_DIM}, drop={EDCN_HEAD_DROPOUT:.2f}, l2={EDCN_L2W:g}) | "
              f"CLF(hidden={CLF_HIDDEN}, drop={CLF_DROPOUT:.2f}, l2={CLF_L2W:g})")

        metrics_buffer = {"acc": [], "f1": [], "prec": [], "recall": []}
        tsne_payload = None

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            seed = num_samples * 100 + run_idx
            set_global_seed(seed)

            tr_idx, te_idx = get_split_indices_by_class(class_ranges, seed=seed, num_train_per_class=num_samples)

            x_tr_v_full = Xv_time[tr_idx]
            x_tr_a_full = Xa_time[tr_idx]
            y_tr_full   = y_all[tr_idx]

            x_te_v = Xv_time[te_idx]
            x_te_a = Xa_time[te_idx]
            y_te   = y_all[te_idx]

            # Normalize with FULL train (before holdout split; no test leakage)
            x_tr_v_full, x_te_v = standardize_clip_train_only_1d(x_tr_v_full, x_te_v, CLIP_VALUE)
            x_tr_a_full, x_te_a = standardize_clip_train_only_1d(x_tr_a_full, x_te_a, CLIP_VALUE)

            x_te_v_in = x_te_v[..., np.newaxis]
            x_te_a_in = x_te_a[..., np.newaxis]
            y_te_oh   = np_utils.to_categorical(y_te, NUM_CLASSES).astype(np.float32)

            # -------- Validation ALWAYS from TRAIN --------
            train_keep_idx, val_idx_in_train = stratified_holdout_from_train(y_tr_full, per_class_val=vpc, seed=seed)

            x_tr_v = x_tr_v_full[train_keep_idx]
            x_tr_a = x_tr_a_full[train_keep_idx]
            y_tr   = y_tr_full[train_keep_idx]

            x_val_v = x_tr_v_full[val_idx_in_train]
            x_val_a = x_tr_a_full[val_idx_in_train]
            y_val   = y_tr_full[val_idx_in_train]

            x_tr_v_in  = x_tr_v[..., np.newaxis]
            x_tr_a_in  = x_tr_a[..., np.newaxis]
            x_val_v_in = x_val_v[..., np.newaxis]
            x_val_a_in = x_val_a[..., np.newaxis]

            y_tr_oh  = np_utils.to_categorical(y_tr,  NUM_CLASSES).astype(np.float32)
            y_val_oh = np_utils.to_categorical(y_val, NUM_CLASSES).astype(np.float32)

            tf.keras.backend.clear_session()
            gc.collect()

            # -------- Stage 1: EDCN branches (fixed capacity) --------
            model_v_cls, model_v_feat = build_edcn_branch(
                input_shape=(DATA_POINTS, 1),
                num_classes=NUM_CLASSES,
                prefix="V",
                head_dropout=EDCN_HEAD_DROPOUT,
                l2w=EDCN_L2W,
                base_filters=EDCN_BASE_FILTERS,
                feat_dim=EDCN_FEAT_DIM
            )
            model_a_cls, model_a_feat = build_edcn_branch(
                input_shape=(DATA_POINTS, 1),
                num_classes=NUM_CLASSES,
                prefix="A",
                head_dropout=EDCN_HEAD_DROPOUT,
                l2w=EDCN_L2W,
                base_filters=EDCN_BASE_FILTERS,
                feat_dim=EDCN_FEAT_DIM
            )

            in_v = Input(shape=(DATA_POINTS, 1), name="in_v")
            in_a = Input(shape=(DATA_POINTS, 1), name="in_a")
            out_v = model_v_cls(in_v)
            out_a = model_a_cls(in_a)
            model_2b = Model([in_v, in_a], [out_v, out_a], name="CFFN_EDCN_2Branch")

            opt = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE) if OPTIMIZER_NAME.lower() == "adamax" \
                  else tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

            model_2b.compile(
                optimizer=opt,
                loss=["categorical_crossentropy", "categorical_crossentropy"],
                metrics=["accuracy"]
            )

            if PRINT_MODEL_SUMMARY_ONCE and (not _printed_summary_once) and (num_samples == SUMMARY_AT_SAMPLE) and (run_idx == SUMMARY_AT_RUN):
                print("\n================= Model Summaries (printed once) =================")
                print("\n[EDCN - Vibration Branch]")
                model_v_cls.summary()
                print("\n[EDCN - Acoustic Branch]")
                model_a_cls.summary()
                print("\n[CFFN 2-Branch Wrapper]")
                model_2b.summary()
                print("===================================================================\n")
                _printed_summary_once = True

            cb_best = BestWeights2Branch(model_v_cls, model_a_cls)
            cb_es = EarlyStoppingMinEpoch(
                monitor="val_loss", mode="min",
                patience=PATIENCE_EDCN, min_epoch=MIN_EPOCH_EDCN,
                restore_best_weights=False
            )

            _ = model_2b.fit(
                [x_tr_v_in, x_tr_a_in],
                [y_tr_oh, y_tr_oh],
                validation_data=([x_val_v_in, x_val_a_in], [y_val_oh, y_val_oh]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[cb_best, cb_es]
            )

            # -------- Extract features for TRAIN/VAL/TEST --------
            feat_tr_v  = model_v_feat.predict(x_tr_v_in,  verbose=0).astype(np.float32)
            feat_tr_a  = model_a_feat.predict(x_tr_a_in,  verbose=0).astype(np.float32)
            feat_val_v = model_v_feat.predict(x_val_v_in, verbose=0).astype(np.float32)
            feat_val_a = model_a_feat.predict(x_val_a_in, verbose=0).astype(np.float32)
            feat_te_v  = model_v_feat.predict(x_te_v_in,  verbose=0).astype(np.float32)
            feat_te_a  = model_a_feat.predict(x_te_a_in,  verbose=0).astype(np.float32)

            # -------- Stage 2: Offline CCA (fit on TRAIN only) --------
            cca_params = cca_fit(feat_tr_v, feat_tr_a, reg=cca_reg, max_components=cap_k)
            k = cca_params["k"]

            fused_tr  = cca_transform(feat_tr_v,  feat_tr_a,  cca_params)
            fused_val = cca_transform(feat_val_v, feat_val_a, cca_params)
            fused_te  = cca_transform(feat_te_v,  feat_te_a,  cca_params)

            tf.keras.backend.clear_session()
            gc.collect()

            # -------- Classifier MLP (fixed capacity) --------
            clf_in = Input(shape=(fused_tr.shape[1],), name="cca_fused_in")
            x = Dense(CLF_HIDDEN, activation="relu", kernel_regularizer=l2(CLF_L2W), name="clf_fc1")(clf_in)
            x = Dropout(CLF_DROPOUT, name="clf_dropout")(x)
            clf_out = Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=l2(CLF_L2W), name="softmax_out")(x)
            clf = Model(clf_in, clf_out, name="CFFN_CCA_ClassifierMLP")

            opt2 = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE) if OPTIMIZER_NAME.lower() == "adamax" \
                   else tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            clf.compile(optimizer=opt2, loss="categorical_crossentropy", metrics=["accuracy"])

            cb_es2 = EarlyStoppingMinEpoch(
                monitor="val_loss", mode="min",
                patience=PATIENCE_CLF, min_epoch=MIN_EPOCH_CLF,
                restore_best_weights=True
            )

            hist_obj = clf.fit(
                fused_tr, y_tr_oh,
                validation_data=(fused_val, y_val_oh),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[cb_es2]
            )
            hist = hist_obj.history

            # -------- Evaluate on TRUE TEST --------
            y_pred_prob = clf.predict(fused_te, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = y_te

            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            rec  = recall_score(y_true, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f} "
                  f"(CCA k={k}, cap_used={cap_k}, reg={cca_reg:g})")

            if TSNE_ONCE_PER_SAMPLE and (run_idx == TSNE_RUN_FOR_PLOTTING):
                tsne_payload = (fused_te.copy(), y_true.copy())

            if PLOT_CURVES_PER_RUN:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(hist.get("loss", []), label="Train")
                plt.plot(hist.get("val_loss", []), label="Val")
                plt.title(f"Loss - Sample {num_samples}")
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(hist.get("accuracy", []), label="Train")
                plt.plot(hist.get("val_accuracy", []), label="Val")
                plt.title(f"Accuracy - Sample {num_samples}")
                plt.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "curves.png"), dpi=300)
                plt.close()

            if PLOT_CM_PER_RUN:
                plt.figure(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - Sample {num_samples}")
                plt.savefig(os.path.join(run_dir, "cm.png"), dpi=300)
                plt.close()

            if PLOT_ROC_PER_RUN:
                y_test_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
                fpr, tpr, roc_auc = dict(), dict(), dict()

                for i in range(NUM_CLASSES):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(NUM_CLASSES):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= NUM_CLASSES
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                plt.figure(figsize=(8, 6))
                plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg (AUC={roc_auc['micro']:.4f})",
                         color="deeppink", linestyle=":", lw=3)
                plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg (AUC={roc_auc['macro']:.4f})",
                         color="navy", linestyle=":", lw=3)

                colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"])
                for i, color in zip(range(NUM_CLASSES), colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC={roc_auc[i]:.4f})")

                plt.plot([0, 1], [0, 1], "k--", lw=2)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - Sample {num_samples}")
                plt.legend(loc="lower right", fontsize="small")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "roc.png"), dpi=300)
                plt.close()

            del model_v_cls, model_v_feat, model_a_cls, model_a_feat, model_2b, clf
            gc.collect()

        # ---- t-SNE once per sample ----
        if TSNE_ONCE_PER_SAMPLE and (tsne_payload is not None):
            fused_te_rep, y_te_rep = tsne_payload
            per_c = max(1, TSNE_MAX_POINTS // NUM_CLASSES)
            idx_tsne = stratified_pick_indices(y_te_rep, per_class=per_c, seed=42)
            if len(idx_tsne) > TSNE_MAX_POINTS:
                rng = np.random.RandomState(42)
                idx_tsne = rng.choice(idx_tsne, TSNE_MAX_POINTS, replace=False)

            X_tsne = fused_te_rep[idx_tsne]
            y_tsne = y_te_rep[idx_tsne]

            tsne = TSNE(n_components=2, random_state=42, n_iter=TSNE_N_ITER, init="pca", learning_rate="auto")
            tsne_xy = tsne.fit_transform(X_tsne)

            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=tsne_xy[:, 0], y=tsne_xy[:, 1], hue=y_tsne, palette="tab10", legend="full", s=15)
            plt.title(f"t-SNE - Sample {num_samples} (Run_{TSNE_RUN_FOR_PLOTTING:02d})")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, "tsne.png"), dpi=300)
            plt.close()

        # ---- Trimmed stats ----
        stats = [num_samples]
        for metric in ["acc", "f1", "prec", "recall"]:
            values = sorted(metrics_buffer[metric])
            trimmed_values = values[1:-1] if len(values) > 2 else values
            stats.append(float(np.mean(trimmed_values)))
            stats.append(float(np.std(trimmed_values)))

        summary_stats.append(stats)
        print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

    # ---- Save summary CSV ----
    cols = [
        "Samples",
        "Acc_Mean", "Acc_Std",
        "F1_Mean", "F1_Std",
        "Prec_Mean", "Prec_Std",
        "Recall_Mean", "Recall_Std"
    ]
    df = pd.DataFrame(summary_stats, columns=cols)
    df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_CFFN.csv"), index=False)

    # ---- Trend fig ----
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o", label="Accuracy", capsize=5)
    plt.errorbar(df["Samples"], df["F1_Mean"],  yerr=df["F1_Std"],  fmt="-s", label="F1 Score", capsize=5)
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Score")
    plt.title("Performance vs. Sample Size")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend_cffn.png"), dpi=300)
    plt.close()

    print(f"\n[All Done] Results saved to: {BASE_OUTPUT_DIR}")
    print(f"[Log] {log_path}")
    print(f"[Best Params] {os.path.join(BASE_OUTPUT_DIR, 'best_parameters.txt')}")