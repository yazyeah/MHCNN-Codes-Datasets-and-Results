# -*- coding: utf-8 -*-
"""
MHCNN Ablation Experiment Script (Case 1: uOttawa vibration + acoustic)

Ablation Exp3: No CAIM & No AMRM (Naive Concatenation Fusion)
- Kept:
  1) Vibration branch encoder (MSHEM-style 1D CNN stack)
  2) Acoustic  branch encoder (MSHEM-style 1D CNN stack)
  3) Classifier head (FCB-style Dense->LeakyReLU->Softmax)
- Removed / Disabled:
  1) CAIM (Cross-Attention Interaction Module; cross-modal attention)
  2) AMRM (Adaptive Modality Re-weighting Module; dynamic modality weighting)
- Fusion strategy:
  - Flatten(vib_features) concat Flatten(aco_features) -> FCB

Protocol alignment (same as main experiments):
- SAMPLE_RANGE = 5..30, REPEAT_TIMES = 10, seed = num_samples*100 + run_idx
- Metrics: Accuracy / Macro-F1 / Macro-Precision / Macro-Recall
- Per-run artifacts: curves.png, cm.png, tsne.png, roc.png
- Plot style: 300 dpi, Times New Roman

IMPORTANT (Portable path paradigm):
- You MUST set YOUR_UO_DATA_ROOT and YOUR_UO_OUTPUT_DIR below.
- YOUR_UO_DATA_ROOT must point to the dataset folder named: '3_MatLab_Raw_Data'.
"""

import os
import sys
import warnings
import random

# =========================
# 0) Runtime Environment (set before importing TensorFlow)
# =========================
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional Windows temp dir override (set to "" to disable)
TEMP_DIR = r"YOUR_TEMP_DIR"  # e.g., r"D:\temp" ; or "" to disable
if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import (
    Dense, Input, Dropout, Flatten, Conv1D,
    MaxPooling1D, AveragePooling1D, Concatenate,
    Activation, LeakyReLU
)
from keras.utils import np_utils
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from itertools import cycle

# [Global plotting style]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 1) Global config =================
SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7

# =========================
# Paths (portable placeholders)
# =========================
YOUR_UO_DATA_ROOT = r"YOUR_UO_DATA_ROOT"      # must point to ...\3_MatLab_Raw_Data
YOUR_UO_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"    # any output folder you want


def _is_placeholder(p: str) -> bool:
    if p is None:
        return True
    s = str(p).strip()
    return (s == "") or ("YOUR_" in s)


def _validate_and_prepare_paths():
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


DATA_PATH_ROOT = None
BASE_OUTPUT_DIR = None

# =========================
# Hyperparameters
# =========================
BEST_PARAMS = {
    "lr": 0.0013866,
    "batch_size": 16,
    "n_layers_vib": 4,
    "dropout_vib": 0.4223,
    "n_layers_aco": 5,
    "dropout_aco": 0.3497
}
EPOCHS = 80


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
                st = i * DATA_POINTS
                ed = (i + 1) * DATA_POINTS
                if ed <= len(vib):
                    vib_samples[i, :] = vib[st:ed]
                    aco_samples[i, :] = aco[st:ed]
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
    y_train = np_utils.to_categorical(train_all[:, 4096], NUM_CLASSES)

    x_test_vib = test_all[:, 0:2048][:, :, np.newaxis]
    x_test_aco = test_all[:, 2048:4096][:, :, np.newaxis]
    y_test = np_utils.to_categorical(test_all[:, 4096], NUM_CLASSES)

    return (x_train_vib, x_train_aco, y_train), (x_test_vib, x_test_aco, y_test)


# ================= 5) Model definition (Ablation Exp3: No CAIM & No AMRM) =================
def create_naive_fusion_model():
    """
    Ablation Exp3 model: naive concatenation fusion (no CAIM, no AMRM).
    Structure: [MSHEM(Vib), MSHEM(Aco)] -> Flatten -> Concat -> FCB
    """
    vib_layers = BEST_PARAMS["n_layers_vib"]
    vib_drop = BEST_PARAMS["dropout_vib"]
    aco_layers = BEST_PARAMS["n_layers_aco"]
    aco_drop = BEST_PARAMS["dropout_aco"]

    # --- Branch 1: Vibration ---
    input_vib = Input(shape=(2048, 1), name="input_vib")
    x = input_vib
    filters_v = [32, 64, 128, 256, 256]
    for i in range(vib_layers):
        f = filters_v[i] if i < len(filters_v) else 256
        stride = 2 if i < 2 else 1
        x = Conv1D(f, 16, strides=stride, padding="same")(x)  # k=16
        x = LeakyReLU(alpha=0.2)(x)
        if i == vib_layers - 1:
            x = Dropout(vib_drop)(x)
            feat_v = AveragePooling1D(2, strides=2, padding="same")(x)  # AvgPool
        else:
            x = AveragePooling1D(2, strides=2, padding="same")(x)

    # --- Branch 2: Acoustic ---
    input_aco = Input(shape=(2048, 1), name="input_aco")
    y = input_aco
    filters_a = [32, 64, 128, 256, 256]
    for i in range(aco_layers):
        f = filters_a[i] if i < len(filters_a) else 256
        stride = 2 if i < 2 else 1
        y = Conv1D(f, 8, strides=stride, padding="same")(y)  # k=8
        y = Activation(tf.keras.activations.gelu)(y)
        if i == aco_layers - 1:
            y = Dropout(aco_drop)(y)
            feat_a = MaxPooling1D(2, strides=2, padding="same")(y)  # MaxPool
        else:
            y = MaxPooling1D(2, strides=2, padding="same")(y)

    # --- Naive Fusion (Concat) ---
    flat_v = Flatten()(feat_v)
    flat_a = Flatten()(feat_a)
    fused = Concatenate()([flat_v, flat_a])

    # --- FCB ---
    fc = Dense(256)(fused)
    fc = LeakyReLU(alpha=0.2)(fc)
    output = Dense(NUM_CLASSES, activation="softmax")(fc)

    model = Model(inputs=[input_vib, input_aco], outputs=output)
    feature_model = Model(inputs=[input_vib, input_aco], outputs=fc)
    return model, feature_model


# ================= 6) Main =================
if __name__ == "__main__":
    _validate_and_prepare_paths()
    DATA_PATH_ROOT = YOUR_UO_DATA_ROOT
    BASE_OUTPUT_DIR = os.path.join(YOUR_UO_OUTPUT_DIR, "MHCNN_Ablation_Experiment", "Exp3_No_CAIM_No_AMRM")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Logging
    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_Exp3.txt")
    sys.stdout = Logger(log_path, sys.stdout)

    print(">>> Ablation Exp3: No CAIM & No AMRM (Naive Concatenation) <<<")
    print("Config: two modalities; naive concat fusion; no cross-attention; no adaptive weighting.")
    print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
    if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
        print(f"TEMP_DIR            = {TEMP_DIR}")
    print(f"DATA_PATH_ROOT      = {DATA_PATH_ROOT}")
    print(f"BASE_OUTPUT_DIR     = {BASE_OUTPUT_DIR}")
    print(f"BEST_PARAMS (fixed) = {BEST_PARAMS}")
    print(f"Log file            = {log_path}\n")

    # Load data
    ALL_DATASETS = load_all_data()

    # Print model summary once
    print("\n>>> Model Architecture Summary (Ablation Exp3):")
    temp_model, _ = create_naive_fusion_model()
    temp_model.summary()
    print("=" * 60 + "\n")
    tf.keras.backend.clear_session()

    summary_stats = []

    for num_samples in SAMPLE_RANGE:
        sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n======== Training samples per class: {num_samples} (Run 1-{REPEAT_TIMES}) ========")

        metrics_buffer = {"acc": [], "f1": [], "prec": [], "recall": []}

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            # 1) split data
            seed = num_samples * 100 + run_idx
            set_global_seed(seed)
            (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(
                ALL_DATASETS, seed=seed, num_train_per_class=num_samples
            )

            # 2) train
            model, feat_model = create_naive_fusion_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adamax(learning_rate=BEST_PARAMS["lr"]),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            history = model.fit(
                [x_tr_v, x_tr_a], y_tr,
                epochs=EPOCHS,
                batch_size=BEST_PARAMS["batch_size"],
                validation_data=([x_te_v, x_te_a], y_te),
                verbose=0
            )

            # 3) predict
            y_pred_prob = model.predict([x_te_v, x_te_a], verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_te, axis=1)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            rec = recall_score(y_true, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}")

            # 4) Visualizations (Title - Sample XX)

            # Curves
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history["loss"], label="Train")
            plt.plot(history.history["val_loss"], label="Test")
            plt.title(f"Loss - Sample {num_samples}")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history["accuracy"], label="Train")
            plt.plot(history.history["val_accuracy"], label="Test")
            plt.title(f"Accuracy - Sample {num_samples}")
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "curves.png"), dpi=300)
            plt.close()

            # CM
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - Sample {num_samples}")
            plt.savefig(os.path.join(run_dir, "cm.png"), dpi=300)
            plt.close()

            # t-SNE
            feats = feat_model.predict([x_te_v, x_te_a], verbose=0)
            tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_true, palette="tab10", legend="full", s=15)
            plt.title(f"t-SNE - Sample {num_samples}")
            plt.savefig(os.path.join(run_dir, "tsne.png"), dpi=300)
            plt.close()

            # ROC
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
            plt.savefig(os.path.join(run_dir, "roc.png"), dpi=300)
            plt.close()

            tf.keras.backend.clear_session()

        # Trimmed statistics
        stats = [num_samples]
        for metric in ["acc", "f1", "prec", "recall"]:
            values = sorted(metrics_buffer[metric])
            trimmed_values = values[1:-1] if len(values) > 2 else values
            stats.append(float(np.mean(trimmed_values)))
            stats.append(float(np.std(trimmed_values)))
        summary_stats.append(stats)

        print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

    # Summary CSV
    cols = ["Samples", "Acc_Mean", "Acc_Std", "F1_Mean", "F1_Std", "Prec_Mean", "Prec_Std", "Recall_Mean", "Recall_Std"]
    df = pd.DataFrame(summary_stats, columns=cols)
    df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_Exp3.csv"), index=False)

    # Trend plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o", label="Naive Fusion Accuracy", capsize=5)
    plt.errorbar(df["Samples"], df["F1_Mean"], yerr=df["F1_Std"], fmt="-s", label="Naive Fusion F1 Score", capsize=5)
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Score")
    plt.title("Performance vs. Sample Size")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend_exp3.png"), dpi=300)
    plt.close()

    print(f"\nAll done! Results saved to: {BASE_OUTPUT_DIR}")
    print(f"Log file: {log_path}")
    print(f"Summary CSV: {os.path.join(BASE_OUTPUT_DIR, 'Final_Summary_Stats_Exp3.csv')}")