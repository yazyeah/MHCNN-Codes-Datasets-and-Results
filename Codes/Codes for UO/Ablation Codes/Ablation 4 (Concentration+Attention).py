# -*- coding: utf-8 -*-
"""
MHCNN Ablation Experiment 4 (Case 1: uOttawa vibration + acoustic)

Ablation Exp4: w/o Adaptive Modality Fusion (Direct Concatenation)
- Kept:
  1) Two-branch feature extractors (vibration CNN branch + acoustic CNN branch)
  2) Classifier head (Dense -> Dropout -> Softmax)
- Removed / Disabled:
  - Adaptive Modality Fusion / Re-weighting module (AMRM-like adaptive fusion)
- Fusion strategy:
  - Direct concatenation of two modality features (Concat(v_feat, a_feat))

Notes:
- This script follows the same experiment protocol as your main experiments:
  - SAMPLE_RANGE = 5..30, REPEAT_TIMES = 10, seed = num_samples*100 + run_idx
  - Metrics: Accuracy / Macro-F1 / Macro-Precision / Macro-Recall
  - Per-run artifacts: curves.png, cm.png, tsne.png, roc.png
  - Plot dpi & font: 300 dpi, Times New Roman

Portable path paradigm (IMPORTANT):
- You MUST set:
  - YOUR_UO_DATA_ROOT   : points to the folder named exactly '3_MatLab_Raw_Data'
  - YOUR_UO_OUTPUT_DIR  : any directory for saving outputs
- Optional:
  - TEMP_DIR            : Windows temp directory; set "" to disable override
"""

import os
import sys

# ===================== 1) Runtime Environment (set BEFORE importing TensorFlow) =====================
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional Windows temp dir override (set "" to disable)
TEMP_DIR = r"YOUR_TEMP_DIR"  # e.g., r"D:\temp" ; or "" to disable
if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
    if not os.path.exists(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
        except Exception:
            pass
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

import warnings
warnings.filterwarnings("ignore")

import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from itertools import cycle

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv1D, BatchNormalization, Activation, MaxPooling1D,
    GlobalAveragePooling1D, Concatenate, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2  # kept (even if unused) to preserve your environment

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize


# ===================== 2) Global settings =====================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10
DATA_POINTS = 2048
NUM_CLASSES = 7

EPOCHS = 80
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
OPTIMIZER_NAME = "Adamax"
TRAIN_VERBOSE = 0


# =========================
# Paths (portable placeholders)
# =========================
YOUR_UO_DATA_ROOT = r"YOUR_UO_DATA_ROOT"      # must point to ...\3_MatLab_Raw_Data
YOUR_UO_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"    # any output directory

DATA_PATH_ROOT = None
BASE_OUTPUT_DIR = None


def _is_placeholder(p: str) -> bool:
    if p is None:
        return True
    s = str(p).strip()
    return (s == "") or ("YOUR_" in s)


def _validate_and_prepare_paths():
    global DATA_PATH_ROOT, BASE_OUTPUT_DIR

    if _is_placeholder(YOUR_UO_DATA_ROOT):
        raise ValueError(
            "Path not set: YOUR_UO_DATA_ROOT. "
            "Please set it to your uOttawa dataset folder named '3_MatLab_Raw_Data'."
        )
    if not os.path.isdir(YOUR_UO_DATA_ROOT):
        raise FileNotFoundError(f"YOUR_UO_DATA_ROOT does not exist: {YOUR_UO_DATA_ROOT}")

    base_name = os.path.basename(os.path.normpath(YOUR_UO_DATA_ROOT))
    if base_name != "3_MatLab_Raw_Data":
        raise ValueError(
            "YOUR_UO_DATA_ROOT must point to the folder named exactly '3_MatLab_Raw_Data'.\n"
            f"Current folder name = '{base_name}'\n"
            f"Current path        = {YOUR_UO_DATA_ROOT}"
        )

    if _is_placeholder(YOUR_UO_OUTPUT_DIR):
        raise ValueError(
            "Path not set: YOUR_UO_OUTPUT_DIR. "
            "Please set it to any directory where you want experiment outputs to be saved."
        )
    os.makedirs(YOUR_UO_OUTPUT_DIR, exist_ok=True)

    DATA_PATH_ROOT = YOUR_UO_DATA_ROOT
    BASE_OUTPUT_DIR = os.path.join(
        YOUR_UO_OUTPUT_DIR,
        "MHCNN_Ablation_Experiment",
        "Exp4_No_AdaptiveFusion_DirectConcat"
    )
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ===================== 3) Helpers =====================
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def print_model_summary_once(model):
    print("\n" + "=" * 90)
    print("Model Summary: MHCNN Ablation Exp4 (No Adaptive Fusion; Direct Concat)")
    print("=" * 90)
    model.summary(print_fn=lambda x: print(x))
    print("=" * 90 + "\n")


# ===================== 4) Data loading =====================
def load_all_data():
    print("[Data] Loading dataset ...")
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
            vib, aco = val[:, 0], val[:, 1]

            vib_samples = np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            aco_samples = np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            for i in range(tot_num0):
                if (i + 1) * DATA_POINTS <= len(vib):
                    vib_samples[i, :] = vib[i * DATA_POINTS: (i + 1) * DATA_POINTS]
                    aco_samples[i, :] = aco[i * DATA_POINTS: (i + 1) * DATA_POINTS]
            return vib_samples, aco_samples
        except Exception as e:
            print(f"[Load Error] {path}: {e}")
            return np.zeros((tot_num0, DATA_POINTS), dtype=np.float32), np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)

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
    return np.vstack(Xv_all), np.vstack(Xa_all), np.concatenate(y_all), class_ranges


def get_split_indices_by_class(class_ranges, seed, num_train_per_class):
    train_idx, test_idx = [], []
    for (st, ed) in class_ranges:
        idx = np.arange(st, ed)
        np.random.seed(seed)
        np.random.shuffle(idx)
        train_idx.append(idx[:num_train_per_class])
        test_idx.append(idx[num_train_per_class:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    np.random.seed(seed)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


# ===================== 5) Network (Ablation Exp4) =====================
def conv_block(x, filters, kernel_size, strides=1, name=None):
    x = Conv1D(filters, kernel_size, strides=strides, padding="same",
               use_bias=False, name=f"{name}_conv")(x)
    x = BatchNormalization(name=f"{name}_bn")(x)
    x = Activation("relu", name=f"{name}_act")(x)
    x = MaxPooling1D(2, padding="same", name=f"{name}_pool")(x)
    return x


def build_mhcnn_ablation4(num_classes=7):
    inp_v = Input(shape=(DATA_POINTS, 1), name="vib_input")
    inp_a = Input(shape=(DATA_POINTS, 1), name="aco_input")

    # 1) Vib Branch
    v = conv_block(inp_v, 16, 64, strides=2, name="vib_b1")
    v = conv_block(v, 32, 16, strides=1, name="vib_b2")
    v = conv_block(v, 64, 3, strides=1, name="vib_b3")
    v_feat = GlobalAveragePooling1D(name="vib_gap")(v)

    # 2) Aco Branch
    a = conv_block(inp_a, 16, 3, strides=1, name="aco_b1")
    a = conv_block(a, 32, 3, strides=1, name="aco_b2")
    a = conv_block(a, 64, 3, strides=1, name="aco_b3")
    a = MaxPooling1D(4)(a)
    a_feat = GlobalAveragePooling1D(name="aco_gap")(a)

    # 3) [ABLATION] Direct Concatenation instead of Adaptive Fusion
    fused = Concatenate(name="concat_fusion")([v_feat, a_feat])

    # 4) Classifier
    x = Dense(128, activation="relu", name="fc1")(fused)
    x = Dropout(0.5)(x)
    logits = Dense(num_classes, activation="softmax", name="softmax")(x)

    model = Model(inputs=[inp_v, inp_a], outputs=logits, name="MHCNN_Ablation4")
    feat_model = Model(inputs=[inp_v, inp_a], outputs=fused, name="Feature_Extractor")
    return model, feat_model


# ===================== 6) Visualization =====================
def plot_curves(history, out_path, num_samples):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Test")
    plt.title(f"Loss - Sample {num_samples}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Test")
    plt.title(f"Accuracy - Sample {num_samples}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_cm(y_true, y_pred, out_path, num_samples):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_tsne(feat_te, y_true, out_path, num_samples):
    if feat_te.shape[0] > 1000:
        idx = np.random.choice(feat_te.shape[0], 1000, replace=False)
        feat_te = feat_te[idx]
        y_true = y_true[idx]

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(feat_te)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x=tsne[:, 0], y=tsne[:, 1], hue=y_true,
        palette=sns.color_palette("hls", NUM_CLASSES),
        legend="full", s=50, alpha=0.8
    )
    plt.title("t-SNE")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_roc(y_true_indices, y_pred_prob, out_path, num_samples):
    y_test_bin = label_binarize(y_true_indices, classes=np.arange(NUM_CLASSES))
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro-avg (AUC={roc_auc['macro']:.2f})",
        color="navy", linestyle=":", lw=3
    )

    colors = cycle(sns.color_palette("hls", NUM_CLASSES))
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC={roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ===================== 7) Main =====================
if __name__ == "__main__":
    _validate_and_prepare_paths()

    sys.stdout = Logger(os.path.join(BASE_OUTPUT_DIR, "experiment_log_Ablation4.txt"), sys.stdout)

    print(">>> MHCNN Ablation Exp4: w/o Adaptive Fusion (Direct Concatenation) <<<")
    print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
    if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
        print(f"TEMP_DIR           = {TEMP_DIR}")
    print(f"DATA_PATH_ROOT     = {DATA_PATH_ROOT}")
    print(f"BASE_OUTPUT_DIR    = {BASE_OUTPUT_DIR}")
    print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}")
    print()

    # 1) Print model summary once
    tf.keras.backend.clear_session()
    _tmp, _ = build_mhcnn_ablation4(NUM_CLASSES)
    print_model_summary_once(_tmp)
    del _tmp
    gc.collect()

    # 2) Load data
    ALL_DATASETS = load_all_data()
    Xv_time, Xa_time, y_all, class_ranges = pack_datasets_to_global_arrays(ALL_DATASETS)

    summary_stats = []

    # 3) Loop experiments
    for num_samples in SAMPLE_RANGE:
        sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n======== Training samples per class: {num_samples} (Run 1-{REPEAT_TIMES}) ========")

        metrics_buffer = {"acc": [], "f1": [], "prec": [], "recall": []}

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            seed = num_samples * 100 + run_idx
            set_global_seed(seed)
            tf.keras.backend.clear_session()
            gc.collect()

            tr_idx, te_idx = get_split_indices_by_class(class_ranges, seed, num_samples)

            x_tr_v = Xv_time[tr_idx][:, :, np.newaxis]
            x_tr_a = Xa_time[tr_idx][:, :, np.newaxis]
            y_tr = to_categorical(y_all[tr_idx], NUM_CLASSES)

            x_te_v = Xv_time[te_idx][:, :, np.newaxis]
            x_te_a = Xa_time[te_idx][:, :, np.newaxis]
            y_te = to_categorical(y_all[te_idx], NUM_CLASSES)
            y_te_labels = y_all[te_idx]

            model, feat_model = build_mhcnn_ablation4(NUM_CLASSES)

            if OPTIMIZER_NAME.lower() == "adamax":
                opt = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE)
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

            hist = model.fit(
                [x_tr_v, x_tr_a], y_tr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=([x_te_v, x_te_a], y_te),
                verbose=TRAIN_VERBOSE
            )

            y_pred_prob = model.predict([x_te_v, x_te_a], verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            feat_te = feat_model.predict([x_te_v, x_te_a], verbose=0)

            acc = accuracy_score(y_te_labels, y_pred)
            f1 = f1_score(y_te_labels, y_pred, average="macro")
            prec = precision_score(y_te_labels, y_pred, average="macro")
            rec = recall_score(y_te_labels, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}")

            plot_curves(hist, os.path.join(run_dir, "curves.png"), num_samples)
            plot_cm(y_te_labels, y_pred, os.path.join(run_dir, "cm.png"), num_samples)
            plot_tsne(feat_te, y_te_labels, os.path.join(run_dir, "tsne.png"), num_samples)
            plot_roc(y_te_labels, y_pred_prob, os.path.join(run_dir, "roc.png"), num_samples)

            del model, feat_model, hist

        stats = [num_samples]
        for metric in ["acc", "f1", "prec", "recall"]:
            values = sorted(metrics_buffer[metric])
            trimmed_values = values[1:-1] if len(values) > 2 else values
            stats.append(float(np.mean(trimmed_values)))
            stats.append(float(np.std(trimmed_values)))
        summary_stats.append(stats)
        print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f}")

    cols = ["Samples", "Acc_Mean", "Acc_Std", "F1_Mean", "F1_Std", "Prec_Mean", "Prec_Std", "Recall_Mean", "Recall_Std"]
    df = pd.DataFrame(summary_stats, columns=cols)
    df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o", label="Accuracy", capsize=5)
    plt.errorbar(df["Samples"], df["F1_Mean"], yerr=df["F1_Std"], fmt="-s", label="F1 Score", capsize=5)
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Score")
    plt.title("Performance vs. Sample Size (Ablation Exp4)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend_ablation4.png"), dpi=300)
    plt.close()

    print(f"\nAll done! Results saved to: {BASE_OUTPUT_DIR}")
    print(f"Log file: {os.path.join(BASE_OUTPUT_DIR, 'experiment_log_Ablation4.txt')}")
    print(f"Summary CSV: {os.path.join(BASE_OUTPUT_DIR, 'Final_Summary_Stats.csv')}")