# -*- coding: utf-8 -*-
"""
MHCNN Ablation Experiment 5 (Case 1: uOttawa vibration + acoustic)

Ablation Exp5: With AMRM, No CAIM
- Kept:
  1) Two MSHEM modality-specific branches (Vibration branch + Acoustic branch)
  2) AMRM (Adaptive Modality Re-weighting Module) on flattened modality features
  3) FCB classifier head (Dense(256) + LeakyReLU + Softmax)
- Removed / Disabled:
  - CAIM (Cross-Attention / Cross-Interaction Enhancement) module
- Purpose:
  Evaluate whether AMRM alone (without cross-modality enhancement) still brings gains.

Protocol (aligned to your main benchmark style):
- SAMPLE_RANGE = 5..30, REPEAT_TIMES = 10, seed = num_samples*100 + run_idx
- Metrics: Accuracy / Macro-F1 / Macro-Precision / Macro-Recall
- Per-run artifacts: curves.png, cm.png, tsne.png, roc.png
- Summary: trimmed mean±std CSV + performance trend plot

Portable path paradigm (IMPORTANT):
- You MUST set:
  - YOUR_UO_DATA_ROOT   : points to the folder named exactly '3_MatLab_Raw_Data'
  - YOUR_UO_OUTPUT_DIR  : any directory for saving outputs
- Optional:
  - YOUR_TEMP_DIR       : Windows temp directory; set "" to disable override
"""

import os
import sys

# ===================== 1) Runtime Environment (set BEFORE importing TensorFlow) =====================
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional Windows temp dir override (set "" to disable)
YOUR_TEMP_DIR = r"YOUR_TEMP_DIR"  # e.g., r"D:\temp" ; or "" to disable
if YOUR_TEMP_DIR and ("YOUR_" not in YOUR_TEMP_DIR):
    if not os.path.exists(YOUR_TEMP_DIR):
        try:
            os.makedirs(YOUR_TEMP_DIR, exist_ok=True)
        except Exception:
            pass
    os.environ["TEMP"] = YOUR_TEMP_DIR
    os.environ["TMP"] = YOUR_TEMP_DIR
    os.environ["TMPDIR"] = YOUR_TEMP_DIR

import warnings
warnings.filterwarnings("ignore")

import random
import gc
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
    Dense, Input, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D,
    Lambda, Concatenate, Activation, LeakyReLU
)
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize


# ===================== 2) Global plotting style =====================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 3) Experiment config =====================
SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7

EPOCHS = 80
BEST_PARAMS = {
    "lr": 0.0013,
    "batch_size": 32,
    "n_layers_vib": 3,
    "dropout_vib": 0.45,
    "n_layers_aco": 5,
    "dropout_aco": 0.27
}

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
    """
    Enforce the portable path paradigm:
    - YOUR_UO_DATA_ROOT must exist and end with folder name '3_MatLab_Raw_Data'
    - YOUR_UO_OUTPUT_DIR must be set (any existing/creatable folder)
    """
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
        "Exp5_With_AMRM_No_CAIM"
    )
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ===================== 4) Reproducibility & Logging =====================
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
    print("Model Summary: Ablation Exp5 (With AMRM, No CAIM)")
    print("=" * 90)
    model.summary(print_fn=lambda x: print(x))
    print("=" * 90 + "\n")


# ===================== 5) Data loading =====================
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
            vib = val[:, 0]
            aco = val[:, 1]

            vib_samples = np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            aco_samples = np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            for i in range(tot_num0):
                if (i + 1) * DATA_POINTS <= len(vib):
                    vib_samples[i, :] = vib[i * DATA_POINTS:(i + 1) * DATA_POINTS]
                    aco_samples[i, :] = aco[i * DATA_POINTS:(i + 1) * DATA_POINTS]
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


def get_split_data(datasets, seed, num_train_per_class):
    """
    Keep your original split protocol:
    - per class: shuffle combined(vib, aco) sample-wise, take first N for train, rest for test
    - then shuffle train_all and test_all
    """
    train_list, test_list = [], []
    for label, (vib, aco) in enumerate(datasets):
        combined = np.hstack((vib, aco))
        np.random.seed(seed)
        np.random.shuffle(combined)
        train_part = combined[:num_train_per_class, :]
        test_part = combined[num_train_per_class:, :]

        y_train = np.full((len(train_part), 1), label)
        y_test = np.full((len(test_part), 1), label)

        train_list.append(np.hstack((train_part, y_train)))
        test_list.append(np.hstack((test_part, y_test)))

    train_all = np.vstack(train_list)
    test_all = np.vstack(test_list)
    np.random.shuffle(train_all)
    np.random.shuffle(test_all)

    x_train_vib = train_all[:, 0:2048][:, :, np.newaxis]
    x_train_aco = train_all[:, 2048:4096][:, :, np.newaxis]
    y_train = to_categorical(train_all[:, 4096], NUM_CLASSES)

    x_test_vib = test_all[:, 0:2048][:, :, np.newaxis]
    x_test_aco = test_all[:, 2048:4096][:, :, np.newaxis]
    y_test = to_categorical(test_all[:, 4096], NUM_CLASSES)

    return (x_train_vib, x_train_aco, y_train), (x_test_vib, x_test_aco, y_test)


# ===================== 6) Model (Ablation Exp5: With AMRM, No CAIM) =====================
def weighted_features(inputs):
    """Aux function for AMRM weighting: elementwise multiply features by scalar weight."""
    features, weights = inputs
    return features * weights


def create_model_exp5():
    """
    Ablation Exp5:
    - MSHEM (Vib) + MSHEM (Aco)
    - NO CAIM (no cross-attention / cross enhancement)
    - AMRM applied on flattened modality features
    - Concat(weighted_v, weighted_a) -> FCB
    """
    # 1) MSHEM - Vib
    input_vib = Input(shape=(2048, 1), name="input_vib")
    x = input_vib
    filters_v = [32, 64, 128, 256, 256]

    for i in range(BEST_PARAMS["n_layers_vib"]):
        f = filters_v[i] if i < len(filters_v) else 256
        stride = 2 if i < 2 else 1
        x = Conv1D(f, 16, strides=stride, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        if i == BEST_PARAMS["n_layers_vib"] - 1:
            x = Dropout(BEST_PARAMS["dropout_vib"])(x)
            feat_v = AveragePooling1D(2, strides=2, padding="same")(x)
        else:
            x = AveragePooling1D(2, strides=2, padding="same")(x)

    # 2) MSHEM - Aco
    input_aco = Input(shape=(2048, 1), name="input_aco")
    y = input_aco
    filters_a = [32, 64, 128, 256, 256]

    for i in range(BEST_PARAMS["n_layers_aco"]):
        f = filters_a[i] if i < len(filters_a) else 256
        stride = 2 if i < 2 else 1
        y = Conv1D(f, 8, strides=stride, padding="same")(y)
        y = Activation("gelu")(y)
        if i == BEST_PARAMS["n_layers_aco"] - 1:
            y = Dropout(BEST_PARAMS["dropout_aco"])(y)
            feat_a = MaxPooling1D(2, strides=2, padding="same")(y)
        else:
            y = MaxPooling1D(2, strides=2, padding="same")(y)

    # 3) [ABLATION] No CAIM -> flatten directly
    flat_v = Flatten()(feat_v)
    flat_a = Flatten()(feat_a)

    # 4) AMRM (Adaptive Modality Re-weighting)
    score_v = Dense(1, activation="sigmoid")(flat_v)
    score_a = Dense(1, activation="sigmoid")(flat_a)
    scores_concat = Concatenate()([score_v, score_a])
    attention_weights = Dense(2, activation="softmax")(scores_concat)

    w_v = Lambda(lambda z: z[:, 0:1])(attention_weights)
    w_a = Lambda(lambda z: z[:, 1:2])(attention_weights)

    weighted_v = Lambda(weighted_features)([flat_v, w_v])
    weighted_a = Lambda(weighted_features)([flat_a, w_a])

    fused = Concatenate()([weighted_v, weighted_a])

    # 5) FCB
    fc = Dense(256)(fused)
    fc = LeakyReLU(alpha=0.2)(fc)
    output = Dense(NUM_CLASSES, activation="softmax")(fc)

    model = Model(inputs=[input_vib, input_aco], outputs=output, name="Ablation5_WithAMRM_NoCAIM")
    feature_model = Model(inputs=[input_vib, input_aco], outputs=fc, name="Ablation5_Feature")
    return model, feature_model


# ===================== 7) Visualization =====================
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
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cm(y_true, y_pred, out_path, num_samples):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Sample {num_samples}")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tsne(feats, y_true, out_path, num_samples):
    tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_true, palette="tab10", legend="full", s=15)
    plt.title(f"t-SNE - Sample {num_samples}")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc(y_true, y_pred_prob, out_path, num_samples):
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
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-avg (AUC={roc_auc['micro']:.4f})",
        color="deeppink", linestyle=":", lw=3
    )
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro-avg (AUC={roc_auc['macro']:.4f})",
        color="navy", linestyle=":", lw=3
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"])
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC={roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Sample {num_samples}")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ===================== 8) Main =====================
if __name__ == "__main__":
    _validate_and_prepare_paths()

    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_Exp5.txt")
    sys.stdout = Logger(log_path, sys.stdout)

    print(">>> MHCNN Ablation Exp5: With AMRM, No CAIM (uOttawa vib+aco) <<<")
    print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
    if YOUR_TEMP_DIR and ("YOUR_" not in YOUR_TEMP_DIR):
        print(f"TEMP_DIR           = {YOUR_TEMP_DIR}")
    print(f"DATA_PATH_ROOT     = {DATA_PATH_ROOT}")
    print(f"BASE_OUTPUT_DIR    = {BASE_OUTPUT_DIR}")
    print(f"Protocol: SAMPLE_RANGE={SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}, REPEAT_TIMES={REPEAT_TIMES}")
    print(f"Training: epochs={EPOCHS}, batch_size={BEST_PARAMS['batch_size']}, lr={BEST_PARAMS['lr']}, optimizer={OPTIMIZER_NAME}")
    print(f"Model: vib_layers={BEST_PARAMS['n_layers_vib']}, aco_layers={BEST_PARAMS['n_layers_aco']} | "
          f"drop_v={BEST_PARAMS['dropout_vib']}, drop_a={BEST_PARAMS['dropout_aco']}")
    print(f"Log file: {log_path}\n")

    # Print model summary once
    tf.keras.backend.clear_session()
    _tmp_model, _ = create_model_exp5()
    print_model_summary_once(_tmp_model)
    del _tmp_model
    tf.keras.backend.clear_session()
    gc.collect()

    # Load data AFTER paths are ready
    ALL_DATASETS = load_all_data()

    summary_stats = []

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

            # Split
            (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(
                ALL_DATASETS, seed=seed, num_train_per_class=num_samples
            )

            # Train
            model, feat_model = create_model_exp5()
            if OPTIMIZER_NAME.lower() == "adamax":
                opt = tf.keras.optimizers.Adamax(learning_rate=BEST_PARAMS["lr"])
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=BEST_PARAMS["lr"])

            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

            history = model.fit(
                [x_tr_v, x_tr_a], y_tr,
                epochs=EPOCHS,
                batch_size=BEST_PARAMS["batch_size"],
                validation_data=([x_te_v, x_te_a], y_te),
                verbose=TRAIN_VERBOSE
            )

            # Predict
            y_pred_prob = model.predict([x_te_v, x_te_a], verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_te, axis=1)

            # Features for t-SNE
            feats = feat_model.predict([x_te_v, x_te_a], verbose=0)

            # Metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            rec = recall_score(y_true, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}")

            # Visualizations
            plot_curves(history, os.path.join(run_dir, "curves.png"), num_samples)
            plot_cm(y_true, y_pred, os.path.join(run_dir, "cm.png"), num_samples)
            plot_tsne(feats, y_true, os.path.join(run_dir, "tsne.png"), num_samples)
            plot_roc(y_true, y_pred_prob, os.path.join(run_dir, "roc.png"), num_samples)

            del model, feat_model, history
            tf.keras.backend.clear_session()
            gc.collect()

        # Trimmed statistics
        stats = [num_samples]
        for metric in ["acc", "f1", "prec", "recall"]:
            values = sorted(metrics_buffer[metric])
            trimmed_values = values[1:-1] if len(values) > 2 else values
            stats.append(float(np.mean(trimmed_values)))
            stats.append(float(np.std(trimmed_values)))
        summary_stats.append(stats)

        print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

    # Save summary CSV
    cols = [
        "Samples",
        "Acc_Mean", "Acc_Std",
        "F1_Mean", "F1_Std",
        "Prec_Mean", "Prec_Std",
        "Recall_Mean", "Recall_Std"
    ]
    df = pd.DataFrame(summary_stats, columns=cols)
    out_csv = os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_Exp5.csv")
    df.to_csv(out_csv, index=False)

    # Performance trend plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o", label="Accuracy", capsize=5)
    plt.errorbar(df["Samples"], df["F1_Mean"], yerr=df["F1_Std"], fmt="-s", label="F1 Score", capsize=5)
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Score")
    plt.title("Performance vs. Sample Size (Ablation Exp5)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend_exp5.png"), dpi=300)
    plt.close()

    print(f"\nAll done! Results saved to: {BASE_OUTPUT_DIR}")
    print(f"Log file: {log_path}")
    print(f"Summary CSV: {out_csv}")