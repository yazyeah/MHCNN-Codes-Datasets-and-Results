# -*- coding: utf-8 -*-
"""
MHCNN (Optuna) - uOttawa Vibration + Acoustic (UORED-VAFCLS) Small-Sample Study

Purpose
-------
This script reproduces the Case-1 experiments on the uOttawa vibration-acoustic bearing dataset:
- Stage 1: Optuna hyperparameter selection on a fixed split (N=15 per class, seed=42).
- Stage 2: Sensitivity study over N = 5..30 training samples per class, repeated 10 runs per N.
- Reporting: trimmed mean ± std (drop best & worst) for Accuracy / Macro-F1 / Macro-Precision / Macro-Recall.
- Artifacts per run: training curves, confusion matrix, t-SNE, ROC curve.
- Summary: Final_Summary_Stats.csv + performance_trend.png.

Dataset Assumptions
-------------------
- UO_DATA_ROOT MUST point to the folder "3_MatLab_Raw_Data" that contains subfolders like:
    1_Healthy, 2_Inner_Race_Faults, 3_Outer_Race_Faults, 4_Ball_Faults, 5_Cage_Faults
- Each .mat file contains a 2-column array [vibration, acoustic] (time-series) with enough length
  to segment into non-overlapping windows of length 2048.
- Two recordings are used per class and concatenated -> 400 segments per class (200 per file).

Reproducibility Notes
---------------------
- For each (N, run_idx), seed = 100*N + run_idx is applied to Python / NumPy / TensorFlow.
- Full determinism on GPU is not guaranteed for TF 2.7.0, but seeds stabilize sampling and init.

Path Convention (IMPORTANT)
---------------------------
- Replace the placeholders in the "Paths" section with YOUR own local ABSOLUTE paths.
- UO_DATA_ROOT must point to the dataset folder "3_MatLab_Raw_Data" (NOT its parent).
- UO_OUTPUT_DIR can be ANY folder you want (all logs/plots/csv will be saved there).
"""

import os

# =========================
# 0) Runtime Environment (MUST be set before importing TensorFlow)
# =========================
# GPU selection (optional). If you have multiple GPUs, change "0" to "1", etc.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional: Windows temp dir redirection (useful when system temp is on a slow disk).
# If you do NOT need it, set TEMP_DIR = "".
TEMP_DIR = r"YOUR_TEMP_DIR"   # e.g., r"D:\temp"   OR set to "" to disable
if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

import warnings
warnings.filterwarnings("ignore")

import random
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# Use Keras from TF for version consistency (TF 2.7.x)
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, Flatten, Conv1D,
    MaxPooling1D, AveragePooling1D, Lambda, Concatenate,
    LeakyReLU, Activation
)
import tensorflow.keras.backend as K

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

from scipy.io import loadmat
import optuna

# Optuna pruning callback compatibility
try:
    from optuna_integration import TFKerasPruningCallback
except ImportError:
    try:
        from optuna.integration import TFKerasPruningCallback
    except ImportError:
        TFKerasPruningCallback = None


# =========================
# Global Plot Settings (300 DPI, Times New Roman)
# =========================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1) Global Config
# =========================
SEARCH_TRIALS = 30
SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7


# =========================
# Paths (PLACEHOLDERS - MUST REPLACE)
# =========================
# Replace the placeholders below with YOUR own local, absolute paths.
# You can put data/output ANYWHERE; this script does NOT assume any fixed folder structure.
#
# [IMPORTANT] UO_DATA_ROOT must point to the dataset folder named "3_MatLab_Raw_Data".
# That folder must contain subfolders like:
#   1_Healthy, 2_Inner_Race_Faults, 3_Outer_Race_Faults, 4_Ball_Faults, 5_Cage_Faults
#
# Example:
#   UO_DATA_ROOT = r"D:\Datasets\uOttawa\3_MatLab_Raw_Data"
#   UO_OUTPUT_DIR = r"D:\Results\MHCNN_UO_Optuna"
UO_DATA_ROOT = r"YOUR_UO_DATA_ROOT"        # <-- MUST point to ".../3_MatLab_Raw_Data"
UO_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"      # <-- ANY output folder you want

# Create output directory early
if "YOUR_" in UO_OUTPUT_DIR:
    raise ValueError(
        "[Path Config Error] UO_OUTPUT_DIR is still a placeholder.\n"
        "Please replace UO_OUTPUT_DIR with an absolute output path."
    )
os.makedirs(UO_OUTPUT_DIR, exist_ok=True)


# =========================
# 2) Utilities: Seeds, GPU Memory Growth, Path Checks
# =========================
def configure_tensorflow_runtime() -> None:
    """Enable GPU memory growth to avoid TF grabbing all VRAM at startup."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # Safe fallback: do nothing
        pass


def set_global_seed(seed: int) -> None:
    """Set seeds for Python/NumPy/TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_paths() -> None:
    """Validate dataset path placeholder + existence."""
    if "YOUR_" in UO_DATA_ROOT:
        raise ValueError(
            "[Path Config Error] UO_DATA_ROOT is still a placeholder.\n"
            "Please replace UO_DATA_ROOT with the absolute path to '3_MatLab_Raw_Data'."
        )
    if not os.path.isdir(UO_DATA_ROOT):
        raise FileNotFoundError(
            f"[Path Error] UO_DATA_ROOT does not exist:\n  {UO_DATA_ROOT}\n"
            "UO_DATA_ROOT must point to the folder '3_MatLab_Raw_Data'."
        )


configure_tensorflow_runtime()
ensure_paths()


# =========================
# 3) Data Loading (uOttawa)
# =========================
def load_all_data():
    """
    Load and segment all classes.

    Returns
    -------
    datasets : list of tuples
        datasets[label] = (vib_samples, aco_samples)
        Each array has shape (400, 2048) if two files x 200 segments are used.
    """
    print("[Data] Loading uOttawa vibration-acoustic dataset...")
    tot_num_per_file = 200  # 200 segments per file -> 2 files -> 400 segments per class

    def load_mat_category(folder: str, filename: str):
        path = os.path.join(UO_DATA_ROOT, folder, filename)
        if not os.path.isfile(path):
            print(f"[Warning] Missing file: {path}")
            return np.zeros((tot_num_per_file, DATA_POINTS)), np.zeros((tot_num_per_file, DATA_POINTS))

        try:
            data = loadmat(path)
            key = filename.replace(".mat", "")
            if key not in data:
                keys = [k for k in data.keys() if not k.startswith("__")]
                if keys:
                    key = keys[0]

            val = data[key]
            # Expected: val[:,0]=vibration, val[:,1]=acoustic
            vib = val[:, 0]
            aco = val[:, 1]

            vib_samples = np.zeros((tot_num_per_file, DATA_POINTS), dtype=np.float32)
            aco_samples = np.zeros((tot_num_per_file, DATA_POINTS), dtype=np.float32)

            for i in range(tot_num_per_file):
                start = i * DATA_POINTS
                end = (i + 1) * DATA_POINTS
                if end <= len(vib):
                    vib_samples[i, :] = vib[start:end]
                    aco_samples[i, :] = aco[start:end]

            return vib_samples, aco_samples

        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            return np.zeros((tot_num_per_file, DATA_POINTS)), np.zeros((tot_num_per_file, DATA_POINTS))

    def load_class_data(file_list):
        v_list, a_list = [], []
        for folder, file in file_list:
            v, a = load_mat_category(folder, file)
            v_list.append(v)
            a_list.append(a)
        return np.vstack(v_list), np.vstack(a_list)

    datasets = []
    # Class mapping:
    # 0: Healthy
    # 1: Developing fault (inner race)
    # 2: Faulty (inner race)
    # 3: Faulty (outer race)
    # 4: Faulty (ball)
    # 5: Developing fault (cage)
    # 6: Faulty (cage)
    datasets.append(load_class_data([("1_Healthy", "H_1_0.mat"), ("1_Healthy", "H_2_0.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_1.mat"), ("2_Inner_Race_Faults", "I_2_1.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_2.mat"), ("2_Inner_Race_Faults", "I_2_2.mat")]))
    datasets.append(load_class_data([("3_Outer_Race_Faults", "O_6_2.mat"), ("3_Outer_Race_Faults", "O_7_2.mat")]))
    datasets.append(load_class_data([("4_Ball_Faults", "B_11_2.mat"), ("4_Ball_Faults", "B_12_2.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_1.mat"), ("5_Cage_Faults", "C_17_1.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_2.mat"), ("5_Cage_Faults", "C_17_2.mat")]))

    print("[Data] Loading complete.")
    return datasets


ALL_DATASETS = load_all_data()


def get_split_data(datasets, seed: int, num_train_per_class: int):
    """
    For each class:
      - Concatenate vib and aco to keep pairing
      - Shuffle jointly (per-class)
      - Take first N as training, remainder as held-out set

    Returns
    -------
    (x_train_vib, x_train_aco, y_train), (x_test_vib, x_test_aco, y_test)
    """
    train_list, test_list = [], []
    for label, (vib, aco) in enumerate(datasets):
        combined = np.hstack((vib, aco))  # shape: (samples, 4096)

        np.random.seed(seed)
        np.random.shuffle(combined)

        train_part = combined[:num_train_per_class, :]
        test_part = combined[num_train_per_class:, :]

        y_train_lbl = np.full((len(train_part), 1), label)
        y_test_lbl = np.full((len(test_part), 1), label)

        train_list.append(np.hstack((train_part, y_train_lbl)))
        test_list.append(np.hstack((test_part, y_test_lbl)))

    train_all = np.vstack(train_list)
    test_all = np.vstack(test_list)

    np.random.shuffle(train_all)
    np.random.shuffle(test_all)

    x_train_vib = train_all[:, 0:2048][:, :, np.newaxis]
    x_train_aco = train_all[:, 2048:4096][:, :, np.newaxis]
    y_train = to_categorical(train_all[:, 4096].astype(int), NUM_CLASSES)

    x_test_vib = test_all[:, 0:2048][:, :, np.newaxis]
    x_test_aco = test_all[:, 2048:4096][:, :, np.newaxis]
    y_test = to_categorical(test_all[:, 4096].astype(int), NUM_CLASSES)

    return (x_train_vib, x_train_aco, y_train), (x_test_vib, x_test_aco, y_test)


# =========================
# 4) Model Definition (MHCNN)
# =========================
class CrossAttention(keras.layers.Layer):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.W_q = self.add_weight(
            name="W_q",
            shape=(input_shape[0][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_k = self.add_weight(
            name="W_k",
            shape=(input_shape[1][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_v = self.add_weight(
            name="W_v",
            shape=(input_shape[1][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, key = inputs
        query_proj = K.dot(query, self.W_q)
        key_proj = K.dot(key, self.W_k)
        value_proj = K.dot(key, self.W_v)

        attention_scores = K.batch_dot(query_proj, key_proj, axes=[2, 2])
        attention_scores = K.softmax(attention_scores, axis=-1)
        return K.batch_dot(attention_scores, value_proj)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def weighted_features(inputs):
    features, weights = inputs
    return features * weights


def create_model(trial=None):
    if trial is not None:
        dropout_vib = trial.suggest_float("dropout_vib", 0.1, 0.5)
        dropout_aco = trial.suggest_float("dropout_aco", 0.1, 0.5)
        atten_dim = trial.suggest_categorical("atten_dim", [128, 256])
        n_layers_vib = trial.suggest_int("n_layers_vib", 3, 5)
        n_layers_aco = trial.suggest_int("n_layers_aco", 3, 5)
    else:
        dropout_vib = 0.2
        dropout_aco = 0.4
        atten_dim = 256
        n_layers_vib = 4
        n_layers_aco = 4

    alpha = 0.2

    # --- Vibration branch ---
    input_v = Input(shape=(DATA_POINTS, 1), name="input_vibration")
    x = input_v
    filters_vib = [32, 64, 128, 256, 256]
    for i in range(n_layers_vib):
        f = filters_vib[i] if i < len(filters_vib) else 256
        stride = 2 if i < 2 else 1
        kernel = 16

        x = Conv1D(f, kernel, strides=stride, padding="same")(x)
        x = LeakyReLU(alpha=alpha)(x)

        if i == n_layers_vib - 1:
            x = Dropout(dropout_vib)(x)
            features_v = AveragePooling1D(2, strides=2, padding="same")(x)
        else:
            x = AveragePooling1D(2, strides=2, padding="same")(x)

    # --- Acoustic branch ---
    input_a = Input(shape=(DATA_POINTS, 1), name="input_acoustic")
    y = input_a
    filters_aco = [32, 64, 128, 256, 256]
    for i in range(n_layers_aco):
        f = filters_aco[i] if i < len(filters_aco) else 256
        stride = 2 if i < 2 else 1
        kernel = 8

        y = Conv1D(f, kernel, strides=stride, padding="same")(y)
        y = Activation(tf.nn.gelu)(y)

        if i == n_layers_aco - 1:
            y = Dropout(dropout_aco)(y)
            features_a = MaxPooling1D(2, strides=2, padding="same")(y)
        else:
            y = MaxPooling1D(2, strides=2, padding="same")(y)

    # --- Bidirectional cross-attention ---
    cross_att_va = CrossAttention(output_dim=atten_dim, name="cross_att_v_from_a")
    feat_v_att = cross_att_va([features_v, features_a])

    cross_att_av = CrossAttention(output_dim=atten_dim, name="cross_att_a_from_v")
    feat_a_att = cross_att_av([features_a, features_v])

    flat_v = Flatten(name="flat_v")(feat_v_att)
    flat_a = Flatten(name="flat_a")(feat_a_att)

    w_logits = Concatenate(name="w_concat")([
        Dense(1, activation="sigmoid", name="w_sig_v")(flat_v),
        Dense(1, activation="sigmoid", name="w_sig_a")(flat_a),
    ])
    w_final = Dense(2, activation="softmax", name="w_softmax")(w_logits)

    fused = Concatenate(name="fused_concat")([
        Lambda(weighted_features, name="scale_v")([flat_v, w_final[:, 0:1]]),
        Lambda(weighted_features, name="scale_a")([flat_a, w_final[:, 1:2]]),
    ])

    fused = Dense(256, name="fcb_dense")(fused)
    fused = LeakyReLU(alpha=0.2, name="fcb_lrelu")(fused)

    output = Dense(NUM_CLASSES, activation="softmax", name="cls_head")(fused)

    clf_model = Model(inputs=[input_v, input_a], outputs=output, name="MHCNN_UO")
    feat_model = Model(inputs=[input_v, input_a], outputs=fused, name="MHCNN_UO_Features")
    return clf_model, feat_model


# =========================
# 5) Optuna Objective
# =========================
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    seed = 42
    set_global_seed(seed)
    (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(
        ALL_DATASETS, seed=seed, num_train_per_class=15
    )

    try:
        tf.keras.backend.clear_session()
        model, _ = create_model(trial)
        model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [TFKerasPruningCallback(trial, "val_accuracy")] if TFKerasPruningCallback else []
        hist = model.fit(
            [x_tr_v, x_tr_a], y_tr,
            epochs=25,
            batch_size=batch_size,
            validation_data=([x_te_v, x_te_a], y_te),
            verbose=0,
            callbacks=callbacks,
        )
        return float(hist.history["val_accuracy"][-1])
    except Exception:
        return 0.0


# =========================
# 6) Plot Helpers (Matplotlib-only)
# =========================
def plot_curves(history, save_path: str, title_prefix: str):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Held-out")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Held-out")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, save_path: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, [str(i) for i in range(NUM_CLASSES)], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(NUM_CLASSES)])

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne(features, labels, save_path: str, title: str):
    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, s=15, cmap="tab10", alpha=0.9)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    legend1 = plt.legend(
        *scatter.legend_elements(),
        title="Class",
        loc="best",
        fontsize="small"
    )
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(y_true, y_prob, save_path: str, title: str):
    y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
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
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg (AUC={roc_auc['micro']:.4f})", linestyle=":", lw=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg (AUC={roc_auc['macro']:.4f})", linestyle=":", lw=3)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"])
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC={roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 7) Main
# =========================
def main():
    print(f"[Path] UO_DATA_ROOT  = {UO_DATA_ROOT}")
    print(f"[Path] UO_OUTPUT_DIR = {UO_OUTPUT_DIR}")

    # -------- Stage 1: Optuna search --------
    print("\n[Stage 1] Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=SEARCH_TRIALS)
    print(f"[Optuna] Best params: {study.best_params}")

    from optuna.trial import FixedTrial
    best_trial_context = FixedTrial(study.best_params)
    best_lr = float(study.best_params["lr"])
    best_bs = int(study.best_params["batch_size"])

    print("\n[Info] Best MHCNN model summary:")
    print("=" * 80)
    try:
        tf.keras.backend.clear_session()
        set_global_seed(42)
        temp_model, _ = create_model(best_trial_context)
        temp_model.summary()
        del temp_model
    except Exception as e:
        print(f"[Warning] Failed to print model summary: {e}")
    print("=" * 80)

    # -------- Stage 2: Sensitivity over N=5..30 --------
    print("\n[Stage 2] Sensitivity experiment (N=5..30 samples per class)...")
    summary_stats = []

    for num_samples in SAMPLE_RANGE:
        sample_dir = os.path.join(UO_OUTPUT_DIR, f"Samples_{num_samples:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n========== N={num_samples} (Runs 1..{REPEAT_TIMES}) ==========")
        metrics_buffer = {"acc": [], "f1": [], "prec": [], "recall": []}

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            seed = num_samples * 100 + run_idx
            set_global_seed(seed)

            (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(
                ALL_DATASETS, seed=seed, num_train_per_class=num_samples
            )

            tf.keras.backend.clear_session()
            model, feat_model = create_model(best_trial_context)
            model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=best_lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            history = model.fit(
                [x_tr_v, x_tr_a], y_tr,
                epochs=80,
                batch_size=best_bs,
                validation_data=([x_te_v, x_te_a], y_te),
                verbose=0,
            )

            y_prob = model.predict([x_te_v, x_te_a], verbose=0)
            y_pred = np.argmax(y_prob, axis=1)
            y_true = np.argmax(y_te, axis=1)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            rec = recall_score(y_true, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"[N={num_samples:02d} | Run={run_idx:02d} | seed={seed}] Acc={acc:.4f}, F1={f1:.4f}")

            plot_curves(history, os.path.join(run_dir, "curves.png"), title_prefix=f"N={num_samples}, Run={run_idx}")
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, os.path.join(run_dir, "cm.png"),
                                  title=f"Confusion Matrix (N={num_samples}, Run={run_idx})")

            feats = feat_model.predict([x_te_v, x_te_a], verbose=0)
            plot_tsne(feats, y_true, os.path.join(run_dir, "tsne.png"),
                      title=f"t-SNE (N={num_samples}, Run={run_idx})")

            plot_roc(y_true, y_prob, os.path.join(run_dir, "roc.png"),
                     title=f"ROC (N={num_samples}, Run={run_idx})")

        stats = [num_samples]
        for metric in ["acc", "f1", "prec", "recall"]:
            values = sorted(metrics_buffer[metric])
            trimmed = values[1:-1] if len(values) > 2 else values
            stats.append(float(np.mean(trimmed)))
            stats.append(float(np.std(trimmed)))

        summary_stats.append(stats)
        print(f"[Done] N={num_samples:02d} trimmed mean Acc={stats[1]:.4f} (std={stats[2]:.4f})")

    cols = [
        "Samples",
        "Acc_Mean", "Acc_Std",
        "F1_Mean", "F1_Std",
        "Prec_Mean", "Prec_Std",
        "Recall_Mean", "Recall_Std"
    ]
    df = pd.DataFrame(summary_stats, columns=cols)
    df.to_csv(os.path.join(UO_OUTPUT_DIR, "Final_Summary_Stats.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o", label="Accuracy", capsize=5)
    plt.errorbar(df["Samples"], df["F1_Mean"], yerr=df["F1_Std"], fmt="-s", label="Macro-F1", capsize=5)
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Score")
    plt.title("Performance vs. Sample Size (uOttawa)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(UO_OUTPUT_DIR, "performance_trend.png"), dpi=300)
    plt.close()

    print(f"\n[All Done] Results saved to:\n  {UO_OUTPUT_DIR}")


if __name__ == "__main__":
    main()