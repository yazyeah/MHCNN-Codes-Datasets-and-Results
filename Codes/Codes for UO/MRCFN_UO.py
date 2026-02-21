# -*- coding: utf-8 -*-
"""
MRCFN baseline experiment script (Case 1: uOttawa vibration + acoustic)

Goal: replicate the *output protocol* of MHCNN_Oputuna_Experiment.py:
- Same data loading + splitting logic (Samples_05..Samples_30, Run_01..Run_10)
- Same metrics: Accuracy / Macro-F1 / Macro-Precision / Macro-Recall
- Same visualization artifacts per run: curves.png, cm.png, tsne.png, roc.png
- Same aggregated outputs: Final_Summary_Stats.csv, performance_trend.png
- Same plotting style: 300 dpi, Times New Roman
- Same logging behavior: mirror all prints to experiment_log_MRCFN.txt
- Add a MHCNN-like best_parameters.txt (fixed baseline config, no Optuna)

Difference: model is MRCFN (CPM + 2×DRRM + SCRM per branch, then GIPFM fusion, then CB).

Path / Environment Convention (KAIST-style placeholders)
-------------------------------------------------------
# Paths
- Replace the placeholders below with YOUR local absolute paths.
- UO_DATA_ROOT MUST point to the dataset folder: "3_MatLab_Raw_Data"
  (i.e., it contains subfolders like "1_Healthy", "2_Inner_Race_Faults", ...).
- Output folder can be ANYWHERE you want.

# Environment (optional)
- CUDA_VISIBLE_DEVICES: GPU id string (default "0")
- TF_CPP_MIN_LOG_LEVEL: default "2"
- TEMP_DIR: optional temp directory on Windows; set to "" to disable.
"""

import os
import sys
import warnings

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

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D,
                          GlobalAveragePooling1D, GlobalMaxPooling1D,
                          SeparableConv1D, BatchNormalization, Dropout,
                          Activation, Add, Multiply, Concatenate, Lambda)
from keras.utils import np_utils
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_curve, auc, accuracy_score)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import seaborn as sns
from scipy.io import loadmat
from itertools import cycle

# ====== Global plotting style: 300 DPI, Times New Roman ======
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. Global config =================
SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7

# =========================
# Paths 
# =========================
# Replace these placeholders with YOUR local absolute paths.
# UO_DATA_ROOT MUST point to: ...\3_MatLab_Raw_Data
DATA_PATH_ROOT = r"YOUR_UO_DATA_ROOT"       # <-- e.g., r"D:\uOttawa\3_MatLab_Raw_Data"
BASE_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"     # <-- e.g., r"D:\Results\MRCFN_UO"

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

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# ================= 2. Logging helper =================
class TeeLogger:
    """Mirror stdout/stderr to a log file, while still printing to console."""
    def __init__(self, log_path: str):
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, msg):
        self.stdout.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass


# ================= 3. Data loading =================
def load_all_data():
    print("[Data] Loading dataset...")
    tot_num0 = 200

    def load_mat_category(folder, filename):
        path = os.path.join(DATA_PATH_ROOT, folder, filename)
        try:
            data = loadmat(path)
            key = filename.replace('.mat', '')
            if key not in data:
                keys = [k for k in data.keys() if not k.startswith('__')]
                if keys:
                    key = keys[0]
            val = data[key]
            vib = val[:, 0]
            aco = val[:, 1]
            vib_samples = np.zeros((tot_num0, DATA_POINTS))
            aco_samples = np.zeros((tot_num0, DATA_POINTS))
            for i in range(tot_num0):
                if (i + 1) * DATA_POINTS <= len(vib):
                    vib_samples[i, :] = vib[i * DATA_POINTS:(i + 1) * DATA_POINTS]
                    aco_samples[i, :] = aco[i * DATA_POINTS:(i + 1) * DATA_POINTS]
            return vib_samples, aco_samples
        except Exception as e:
            print(f"[Load Error] {path}: {e}")
            return np.zeros((tot_num0, DATA_POINTS)), np.zeros((tot_num0, DATA_POINTS))

    datasets = []

    def load_class_data(file_list):
        v_list, a_list = [], []
        for folder, file in file_list:
            v, a = load_mat_category(folder, file)
            v_list.append(v)
            a_list.append(a)
        return np.vstack(v_list), np.vstack(a_list)

    # NOTE: keep exactly the same file list as your MHCNN script for strict comparability.
    datasets.append(load_class_data([("1_Healthy", "H_1_0.mat"), ("1_Healthy", "H_2_0.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_1.mat"), ("2_Inner_Race_Faults", "I_2_1.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_2.mat"), ("2_Inner_Race_Faults", "I_2_2.mat")]))
    datasets.append(load_class_data([("3_Outer_Race_Faults", "O_6_2.mat"), ("3_Outer_Race_Faults", "O_7_2.mat")]))
    datasets.append(load_class_data([("4_Ball_Faults", "B_11_2.mat"), ("4_Ball_Faults", "B_12_2.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_1.mat"), ("5_Cage_Faults", "C_17_1.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_2.mat"), ("5_Cage_Faults", "C_17_2.mat")])))

    print("[Data] Loading complete.")
    return datasets


ALL_DATASETS = load_all_data()


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


# ================= 4. MRCFN model definition =================
def cpm_block(x, filters, kernel_size=16, pool_size=2, name=None):
    """Convolutional Pooling Module (CPM): Conv + ReLU + MaxPool."""
    x = Conv1D(filters, kernel_size, padding='same', name=None if name is None else f"{name}_conv")(x)
    x = Activation('relu', name=None if name is None else f"{name}_relu")(x)
    x = MaxPooling1D(pool_size=pool_size, strides=2, padding='same', name=None if name is None else f"{name}_pool")(x)
    return x


def drrm_block(x, filters, kernel_size=3, name=None):
    """
    Double Ring Residual Module (DRRM) — a practical 1D adaptation:
    - Use depthwise separable convs for parameter efficiency.
    - Use nested residual connections to mimic 'double-ring' behavior.
    """
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv1D(filters, 1, padding='same', name=None if name is None else f"{name}_proj")(shortcut)

    y = SeparableConv1D(filters, kernel_size, padding='same', name=None if name is None else f"{name}_sep1")(x)
    y = BatchNormalization(name=None if name is None else f"{name}_bn1")(y)
    y = Activation('relu', name=None if name is None else f"{name}_relu1")(y)

    y = SeparableConv1D(filters, kernel_size, padding='same', name=None if name is None else f"{name}_sep2")(y)
    y = BatchNormalization(name=None if name is None else f"{name}_bn2")(y)

    y = Add(name=None if name is None else f"{name}_add1")([shortcut, y])
    y = Activation('relu', name=None if name is None else f"{name}_relu2")(y)

    z = SeparableConv1D(filters, kernel_size, padding='same', name=None if name is None else f"{name}_sep3")(y)
    z = BatchNormalization(name=None if name is None else f"{name}_bn3")(z)
    z = Activation('relu', name=None if name is None else f"{name}_relu3")(z)

    z = SeparableConv1D(filters, kernel_size, padding='same', name=None if name is None else f"{name}_sep4")(z)
    z = BatchNormalization(name=None if name is None else f"{name}_bn4")(z)

    z = Add(name=None if name is None else f"{name}_add2")([y, z])
    z = Activation('relu', name=None if name is None else f"{name}_relu4")(z)
    return z


def scrm_block(x, reduction=8, spatial_kernel=7, name=None):
    """
    Spatial Channel Reconstruction Module (SCRM) — 1D CBAM-style reconstruction.
    """
    ch = int(x.shape[-1])

    gap = GlobalAveragePooling1D(name=None if name is None else f"{name}_gap")(x)
    gmp = GlobalMaxPooling1D(name=None if name is None else f"{name}_gmp")(x)

    shared1 = Dense(max(ch // reduction, 1), activation='relu',
                    name=None if name is None else f"{name}_ca_fc1")
    shared2 = Dense(ch, activation=None, name=None if name is None else f"{name}_ca_fc2")

    ca1 = shared2(shared1(gap))
    ca2 = shared2(shared1(gmp))
    ca = Add(name=None if name is None else f"{name}_ca_add")([ca1, ca2])
    ca = Activation('sigmoid', name=None if name is None else f"{name}_ca_sigmoid")(ca)
    ca = Lambda(lambda t: tf.expand_dims(t, axis=1),
                name=None if name is None else f"{name}_ca_expand")(ca)
    x_ca = Multiply(name=None if name is None else f"{name}_ca_mul")([x, ca])

    avg_pool = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                      name=None if name is None else f"{name}_sa_avg")(x_ca)
    max_pool = Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                      name=None if name is None else f"{name}_sa_max")(x_ca)
    sa = Concatenate(axis=-1, name=None if name is None else f"{name}_sa_cat")([avg_pool, max_pool])
    sa = Conv1D(1, spatial_kernel, padding='same', activation='sigmoid',
                name=None if name is None else f"{name}_sa_conv")(sa)
    x_sa = Multiply(name=None if name is None else f"{name}_sa_mul")([x_ca, sa])
    return x_sa


def gipfm_fusion(f_v, f_a, name=None):
    """
    Global Interactive Perception Fusion Module (GIPFM) — gated fusion with softmax weights.
    """
    joint = Concatenate(axis=-1, name=None if name is None else f"{name}_cat")([f_v, f_a])
    g = GlobalAveragePooling1D(name=None if name is None else f"{name}_gap")(joint)
    w = Dense(2, activation='softmax', name=None if name is None else f"{name}_w")(g)

    def scale_feat(args):
        feat, w_scalar = args
        w_scalar = tf.reshape(w_scalar, (-1, 1, 1))
        return feat * w_scalar

    w_v = Lambda(lambda t: t[:, 0], name=None if name is None else f"{name}_wv")(w)
    w_a = Lambda(lambda t: t[:, 1], name=None if name is None else f"{name}_wa")(w)

    fv_scaled = Lambda(scale_feat, name=None if name is None else f"{name}_scale_v")([f_v, w_v])
    fa_scaled = Lambda(scale_feat, name=None if name is None else f"{name}_scale_a")([f_a, w_a])

    fused = Add(name=None if name is None else f"{name}_add")([fv_scaled, fa_scaled])
    return fused


def create_mrcfn_model():
    """
    MRCFN: CPM -> DRRM1 -> DRRM2 -> SCRM (per sensor), then GIPFM fusion, then CB (GAP + FC + softmax).
    """
    inp_v = Input(shape=(DATA_POINTS, 1), name="input_vib")
    inp_a = Input(shape=(DATA_POINTS, 1), name="input_aco")

    v = cpm_block(inp_v, filters=32, kernel_size=16, name="v_cpm")
    a = cpm_block(inp_a, filters=32, kernel_size=16, name="a_cpm")

    v = drrm_block(v, filters=64, kernel_size=3, name="v_drrm1")
    v = drrm_block(v, filters=128, kernel_size=3, name="v_drrm2")

    a = drrm_block(a, filters=64, kernel_size=3, name="a_drrm1")
    a = drrm_block(a, filters=128, kernel_size=3, name="a_drrm2")

    v = scrm_block(v, reduction=8, spatial_kernel=7, name="v_scrm")
    a = scrm_block(a, reduction=8, spatial_kernel=7, name="a_scrm")

    fused_map = gipfm_fusion(v, a, name="gipfm")

    feat_vec = GlobalAveragePooling1D(name="cb_gap")(fused_map)  # used for t-SNE
    logits = Dense(NUM_CLASSES, name="cb_fc")(feat_vec)
    out = Activation("softmax", name="softmax")(logits)

    model = Model(inputs=[inp_v, inp_a], outputs=out, name="MRCFN")
    feat_model = Model(inputs=[inp_v, inp_a], outputs=feat_vec, name="MRCFN_feat")
    return model, feat_model


def write_best_parameters_txt(out_dir: str):
    """
    Write a MHCNN-like 'best_parameters.txt' for baselines (fixed config, no Optuna).
    Includes: key hyperparameters + model.summary() text.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_parameters.txt")
    model, _ = create_mrcfn_model()

    summary_lines = []
    model.summary(print_fn=lambda s: summary_lines.append(s))

    with open(path, "w", encoding="utf-8") as f:
        f.write("Best MRCFN Model Architecture Summary (fixed baseline config)\n")
        f.write("=" * 70 + "\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"learning_rate: {LEARNING_RATE}\n")
        f.write(f"num_classes: {NUM_CLASSES}\n")
        f.write(f"sample_range: {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop - 1}\n")
        f.write(f"repeat_times: {REPEAT_TIMES}\n")
        f.write("\nModel Summary:\n")
        f.write("-" * 70 + "\n")
        f.write("\n".join(summary_lines) + "\n")

    del model


# ================= 5. Main experiment =================
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_MRCFN.txt")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(">>> MRCFN baseline experiment (Case 1) <<<")
        print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
        print(f"TEMP_DIR            = {TEMP_DIR}")
        print(f"BASE_OUTPUT_DIR     = {BASE_OUTPUT_DIR}")
        print(f"DATA_PATH_ROOT      = {DATA_PATH_ROOT}")
        print(f"SAMPLE_RANGE        = {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}")
        print(f"REPEAT_TIMES        = {REPEAT_TIMES}")
        print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}\n")

        # --- write best_parameters.txt (fixed config) ---
        write_best_parameters_txt(BASE_OUTPUT_DIR)

        print(">>> MRCFN Model Architecture Summary:")
        print("=" * 60)
        temp_model, _ = create_mrcfn_model()
        temp_model.summary()
        del temp_model
        print("=" * 60 + "\n")

        summary_stats = []

        for num_samples in SAMPLE_RANGE:
            sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            print(f"\n======== Training samples per class: {num_samples} (Runs 1-{REPEAT_TIMES}) ========")

            metrics_buffer = {'acc': [], 'f1': [], 'prec': [], 'recall': []}

            for run_idx in range(1, REPEAT_TIMES + 1):
                run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
                os.makedirs(run_dir, exist_ok=True)

                seed = num_samples * 100 + run_idx
                (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(
                    ALL_DATASETS, seed=seed, num_train_per_class=num_samples
                )

                model, feat_model = create_mrcfn_model()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                history = model.fit(
                    [x_tr_v, x_tr_a], y_tr,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=([x_te_v, x_te_a], y_te),
                    verbose=0
                )

                y_pred_prob = model.predict([x_te_v, x_te_a], verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_true = np.argmax(y_te, axis=1)

                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                prec = precision_score(y_true, y_pred, average='macro')
                rec = recall_score(y_true, y_pred, average='macro')

                metrics_buffer['acc'].append(acc)
                metrics_buffer['f1'].append(f1)
                metrics_buffer['prec'].append(prec)
                metrics_buffer['recall'].append(rec)

                print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}")

                # --- Visualizations ---
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Train')
                plt.plot(history.history['val_loss'], label='Test')
                plt.title(f'Loss - Sample {num_samples}')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(history.history['accuracy'], label='Train')
                plt.plot(history.history['val_accuracy'], label='Test')
                plt.title(f'Accuracy - Sample {num_samples}')
                plt.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, 'curves.png'), dpi=300)
                plt.close()

                plt.figure(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - Sample {num_samples}')
                plt.savefig(os.path.join(run_dir, 'cm.png'), dpi=300)
                plt.close()

                feats = feat_model.predict([x_te_v, x_te_a], verbose=0)
                tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
                plt.figure(figsize=(6, 5))
                sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_true, palette='tab10',
                                legend='full', s=15)
                plt.title(f't-SNE - Sample {num_samples}')
                plt.savefig(os.path.join(run_dir, 'tsne.png'), dpi=300)
                plt.close()

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
                plt.plot(fpr["micro"], tpr["micro"],
                         label=f'Micro-avg (AUC={roc_auc["micro"]:.4f})',
                         color='deeppink', linestyle=':', lw=3)
                plt.plot(fpr["macro"], tpr["macro"],
                         label=f'Macro-avg (AUC={roc_auc["macro"]:.4f})',
                         color='navy', linestyle=':', lw=3)

                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
                for i, color in zip(range(NUM_CLASSES), colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC={roc_auc[i]:.4f})')

                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Sample {num_samples}')
                plt.legend(loc="lower right", fontsize='small')
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, 'roc.png'), dpi=300)
                plt.close()

                del model, feat_model

            stats = [num_samples]
            for metric in ['acc', 'f1', 'prec', 'recall']:
                values = sorted(metrics_buffer[metric])
                trimmed_values = values[1:-1] if len(values) > 2 else values
                stats.append(float(np.mean(trimmed_values)))
                stats.append(float(np.std(trimmed_values)))

            summary_stats.append(stats)
            print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

        cols = ['Samples', 'Acc_Mean', 'Acc_Std', 'F1_Mean', 'F1_Std', 'Prec_Mean', 'Prec_Std', 'Recall_Mean', 'Recall_Std']
        df = pd.DataFrame(summary_stats, columns=cols)
        df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'Final_Summary_Stats.csv'), index=False)

        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Samples'], df['Acc_Mean'], yerr=df['Acc_Std'], fmt='-o', label='Accuracy', capsize=5)
        plt.errorbar(df['Samples'], df['F1_Mean'], yerr=df['F1_Std'], fmt='-s', label='F1 Score', capsize=5)
        plt.xlabel('Training Samples per Class')
        plt.ylabel('Score')
        plt.title('Performance vs. Sample Size')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'performance_trend.png'), dpi=300)
        plt.close()

        print(f"\nAll done! Results saved to: {BASE_OUTPUT_DIR}")
        print(f"Log file: {log_path}")

    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()