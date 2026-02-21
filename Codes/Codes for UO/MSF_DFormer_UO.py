# -*- coding: utf-8 -*-
"""
MSF-DFormer baseline experiment script (Case 1: uOttawa vibration + acoustic)  [TensorFlow/Keras]

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

Fixes applied:
1. REMOVED `jit_compile=True` to fix TypeError on your TF version.
2. Kept Lightweight Params (Filters=32, DModel=64) for speed.
3. Kept Visualization style STRICTLY aligned with MHCNN.
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
import keras

# ===== Mixed precision (Safe check) =====
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

import seaborn as sns
from scipy.io import loadmat
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from itertools import cycle

# ----- Global plotting style (align to MHCNN) -----
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 1) Global config =================
SAMPLE_RANGE = range(5, 31)  # 5..30
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7

# =========================
# Paths (placeholders)
# =========================
# Replace these placeholders with YOUR local absolute paths.
# UO_DATA_ROOT MUST point to: ...\3_MatLab_Raw_Data
DATA_PATH_ROOT = r"YOUR_UO_DATA_ROOT"      # e.g., r"D:\uOttawa\3_MatLab_Raw_Data"
BASE_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"    # e.g., r"D:\Results\MSF_DFormer_UO"

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
LEARNING_RATE = 1.3866e-3
BATCH_SIZE = 16
OPTIMIZER_NAME = "Adamax"

# ----- MSF-DFormer Hyperparams (Lightweight Mode) -----
MDSCB_FILTERS = 32
ADFF_FILTERS = 64
DOT_DMODEL = 64
DOT_NHEAD = 2
DOT_NLAYERS = 2
SEQ_LEN_TOKENS = 3

# ----- Progress / speed knobs -----
TRAIN_VERBOSE = 0
VALIDATION_BATCH_SIZE = 16
VALIDATION_FREQ = 5
TSNE_MAX_POINTS = 1400
TSNE_SEED = 42


# ======================= Reproducibility =======================
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================= Logging helper =======================
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
    print("Model Summary (Lightweight Version)")
    print("=" * 90)
    model.summary(print_fn=lambda x: print(x))
    print("=" * 90)
    try:
        total_params = model.count_params()
        print(f"Total params: {total_params:,}")
    except Exception:
        pass
    print()


# ======================= Data loading =======================
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
    cursor = 0
    class_ranges = []
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
    test_idx = np.concatenate(test_idx)

    np.random.seed(seed)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


# ======================= MSF-DFormer =======================
class GroupNorm1D(keras.layers.Layer):
    def __init__(self, groups=4, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.eps = eps

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.gamma = self.add_weight(name="gn_gamma", shape=(c,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="gn_beta", shape=(c,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        C = tf.shape(x)[2]
        G = tf.minimum(self.groups, C)
        x_ = tf.reshape(x, [B, L, G, C // G])
        mean, var = tf.nn.moments(x_, axes=[1, 3], keepdims=True)
        x_ = (x_ - mean) / tf.sqrt(var + self.eps)
        x_ = tf.reshape(x_, [B, L, C])
        return x_ * self.gamma + self.beta


class SCRM(keras.layers.Layer):
    def __init__(self, groups=4, init_tau=0.5, sharpness=10.0, **kwargs):
        super().__init__(**kwargs)
        self.gn = GroupNorm1D(groups=groups)
        self.init_tau = init_tau
        self.sharpness = sharpness

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.gamma = self.add_weight(name="scrm_gamma", shape=(c,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="scrm_beta", shape=(c,), initializer="zeros", trainable=True)
        self.tau = self.add_weight(
            name="scrm_tau",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_tau),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        x_gn = self.gn(x)
        w = tf.sigmoid(x_gn * self.gamma + self.beta)
        mask = tf.sigmoid((w - self.tau) * self.sharpness)
        w1 = mask * w
        w2 = (1.0 - mask) * (1.0 - w)
        x_high = x * w1
        x_low = x * w2
        c = tf.shape(x)[-1]
        c_half = c // 2
        xh1, xh2 = x_high[:, :, :c_half], x_high[:, :, c_half:]
        xl1, xl2 = x_low[:, :, :c_half], x_low[:, :, c_half:]
        y1 = xh1 + xl2
        y2 = xh2 + xl1
        y = tf.concat([y1, y2], axis=-1)
        return y


class SFFM(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.fc = keras.layers.Dense(filters, activation="sigmoid")

    def call(self, xa, xb):
        fa_avg = tf.reduce_mean(xa, axis=1)
        fa_max = tf.reduce_max(xa, axis=1)
        fb_avg = tf.reduce_mean(xb, axis=1)
        fb_max = tf.reduce_max(xb, axis=1)
        z = tf.concat([fa_avg, fa_max, fb_avg, fb_max], axis=-1)
        wc = self.fc(z)
        wc = tf.expand_dims(wc, axis=1)
        pa = wc
        pb = 1.0 - wc
        fused = (xa * pa + xa) + (xb * pb + xb)
        return fused


class MDSCB(keras.layers.Layer):
    def __init__(self, filters=128, expansion=2, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.expansion = expansion
        self.pool = pool
        self.pw_expand = keras.layers.Conv1D(filters * expansion, kernel_size=1, padding="same")
        self.bn0 = keras.layers.BatchNormalization()
        self.act0 = keras.layers.Activation("relu")
        self.b3 = keras.layers.SeparableConv1D(filters * expansion, kernel_size=3, padding="same")
        self.b5 = keras.layers.SeparableConv1D(filters * expansion, kernel_size=5, padding="same")
        self.b7 = keras.layers.SeparableConv1D(filters * expansion, kernel_size=7, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.bn5 = keras.layers.BatchNormalization()
        self.bn7 = keras.layers.BatchNormalization()
        self.act = keras.layers.Activation("relu")
        self.merge_pw = keras.layers.Conv1D(filters, kernel_size=1, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.proj = None
        self.pool_layer = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same") if pool else None
        self.out_act = keras.layers.Activation("relu")

    def build(self, input_shape):
        in_c = int(input_shape[-1])
        if in_c != self.filters:
            self.proj = keras.layers.Conv1D(self.filters, kernel_size=1, padding="same")
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.pw_expand(x)
        h = self.bn0(h, training=training)
        h = self.act0(h)
        h3 = self.act(self.bn3(self.b3(h), training=training))
        h5 = self.act(self.bn5(self.b5(h), training=training))
        h7 = self.act(self.bn7(self.b7(h), training=training))
        h = tf.concat([h3, h5, h7], axis=-1)
        h = self.merge_pw(h)
        h = self.bn1(h, training=training)
        res = self.proj(x) if self.proj is not None else x
        y = self.out_act(h + res)
        if self.pool_layer is not None:
            y = self.pool_layer(y)
        return y


class ADFF(keras.layers.Layer):
    def __init__(self, out_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.alpha = self.add_weight(name="adff_alpha", shape=(), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="adff_beta", shape=(), initializer="ones", trainable=True)
        self.proj = keras.layers.Dense(out_dim, activation="relu")
        self.conv = keras.layers.Conv1D(out_dim, kernel_size=3, padding="same")
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.Activation("relu")

    def call(self, feats_list, training=False):
        tokens = []
        for fi in feats_list:
            avg_vec = tf.reduce_mean(fi, axis=1)
            max_vec = tf.reduce_max(fi, axis=1)
            w = tf.sigmoid(self.alpha * avg_vec + self.beta * max_vec)
            vec = w * avg_vec + (1.0 - w) * max_vec
            vec = self.proj(vec)
            tokens.append(vec)
        x = tf.stack(tokens, axis=1)
        h = self.conv(x)
        h = self.bn(h, training=training)
        h = self.act(h)
        y = x + h
        return y


class DOT(keras.layers.Layer):
    def __init__(self, d_model=256, nhead=4, n_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.dropout = dropout
        self.offset_fc = keras.layers.Dense(SEQ_LEN_TOKENS)
        self.mha = [keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead, dropout=dropout)
                    for _ in range(n_layers)]
        self.ln1 = [keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(n_layers)]
        self.ln2 = [keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(n_layers)]
        self.do1 = [keras.layers.Dropout(dropout) for _ in range(n_layers)]
        self.do2 = [keras.layers.Dropout(dropout) for _ in range(n_layers)]
        self.ffn1 = [keras.layers.Dense(d_model * 4, activation=tf.keras.activations.gelu) for _ in range(n_layers)]
        self.ffn2 = [keras.layers.Dense(d_model) for _ in range(n_layers)]

    def call(self, x, training=False):
        x_bar = tf.reduce_mean(x, axis=-1)
        delta = self.offset_fc(x_bar)
        x = x + tf.expand_dims(delta, axis=-1)
        for i in range(self.n_layers):
            attn_out = self.mha[i](x, x, training=training)
            x = self.ln1[i](x + self.do1[i](attn_out, training=training))
            f = self.ffn2[i](self.ffn1[i](x))
            x = self.ln2[i](x + self.do2[i](f, training=training))
        return x


def build_msf_dformer(num_classes=NUM_CLASSES):
    inp_v = keras.layers.Input(shape=(DATA_POINTS, 1), name="vib_in")
    inp_a = keras.layers.Input(shape=(DATA_POINTS, 1), name="aco_in")

    mdscb1_v = MDSCB(filters=MDSCB_FILTERS, expansion=2, pool=True, name="MDSCB1_v")
    mdscb1_a = MDSCB(filters=MDSCB_FILTERS, expansion=2, pool=True, name="MDSCB1_a")
    mdscb2_v = MDSCB(filters=MDSCB_FILTERS, expansion=4, pool=True, name="MDSCB2_v")
    mdscb2_a = MDSCB(filters=MDSCB_FILTERS, expansion=4, pool=True, name="MDSCB2_a")
    mdscb3_v = MDSCB(filters=MDSCB_FILTERS, expansion=6, pool=True, name="MDSCB3_v")
    mdscb3_a = MDSCB(filters=MDSCB_FILTERS, expansion=6, pool=True, name="MDSCB3_a")

    scrm1_v = SCRM(groups=4, name="SCRM1_v")
    scrm1_a = SCRM(groups=4, name="SCRM1_a")
    scrm2_v = SCRM(groups=4, name="SCRM2_v")
    scrm2_a = SCRM(groups=4, name="SCRM2_a")
    scrm3_v = SCRM(groups=4, name="SCRM3_v")
    scrm3_a = SCRM(groups=4, name="SCRM3_a")

    sffm1 = SFFM(filters=MDSCB_FILTERS, name="SFFM1")
    sffm2 = SFFM(filters=MDSCB_FILTERS, name="SFFM2")
    sffm3 = SFFM(filters=MDSCB_FILTERS, name="SFFM3")

    xv = mdscb1_v(inp_v)
    xa = mdscb1_a(inp_a)
    xv = scrm1_v(xv)
    xa = scrm1_a(xa)
    f1 = sffm1(xv, xa)

    xv = mdscb2_v(xv)
    xa = mdscb2_a(xa)
    xv = scrm2_v(xv)
    xa = scrm2_a(xa)
    f2 = sffm2(xv, xa)

    xv = mdscb3_v(xv)
    xa = mdscb3_a(xa)
    xv = scrm3_v(xv)
    xa = scrm3_a(xa)
    f3 = sffm3(xv, xa)

    adff = ADFF(out_dim=ADFF_FILTERS, name="ADFF")
    tokens = adff([f1, f2, f3])

    dot = DOT(d_model=DOT_DMODEL, nhead=DOT_NHEAD, n_layers=DOT_NLAYERS, dropout=0.1, name="DOT")
    tokens = dot(tokens)

    feat = keras.layers.GlobalAveragePooling1D(name="global_token_pool")(tokens)

    # IMPORTANT: output logits in float32 for mixed_precision stability
    logits = keras.layers.Dense(num_classes, activation="softmax", name="cls", dtype="float32")(feat)

    model = keras.Model([inp_v, inp_a], logits, name="MSF_DFormer_TF")
    feat_model = keras.Model([inp_v, inp_a], feat, name="MSF_DFormer_Feature")
    return model, feat_model


# ======================= Plotting =======================
def plot_curves(hist, out_path, num_samples):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    loss = hist.history.get("loss", [])
    plt.plot(loss, label="Train")

    if "val_loss" in hist.history:
        val_loss = hist.history["val_loss"]
        if len(val_loss) == len(loss):
            plt.plot(val_loss, label="Test")
        else:
            k = int(VALIDATION_FREQ) if isinstance(VALIDATION_FREQ, int) and VALIDATION_FREQ > 0 else 1
            val_epochs = np.arange(k, k * len(val_loss) + 1, k)
            plt.plot(val_epochs - 1, val_loss, label="Test")

    plt.title(f"Loss - Sample {num_samples}")
    plt.legend()

    plt.subplot(1, 2, 2)
    acc = hist.history.get("accuracy", [])
    plt.plot(acc, label="Train")

    if "val_accuracy" in hist.history:
        val_acc = hist.history["val_accuracy"]
        if len(val_acc) == len(acc):
            plt.plot(val_acc, label="Test")
        else:
            k = int(VALIDATION_FREQ) if isinstance(VALIDATION_FREQ, int) and VALIDATION_FREQ > 0 else 1
            val_epochs = np.arange(k, k * len(val_acc) + 1, k)
            plt.plot(val_epochs - 1, val_acc, label="Test")

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


def plot_tsne(feat_te, y_true, out_path, num_samples, max_points=TSNE_MAX_POINTS, seed=TSNE_SEED):
    y_true = np.asarray(y_true).astype(int)
    n = feat_te.shape[0]
    if n > max_points:
        rng = np.random.RandomState(seed)
        per_cls = max(1, max_points // NUM_CLASSES)
        idx_keep = []
        for c in range(NUM_CLASSES):
            idx_c = np.where(y_true == c)[0]
            rng.shuffle(idx_c)
            idx_keep.extend(idx_c[:per_cls])
        idx_keep = np.array(idx_keep, dtype=int)
        feat_te = feat_te[idx_keep]
        y_true = y_true[idx_keep]

    tsne = TSNE(n_components=2, random_state=42).fit_transform(feat_te)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=tsne[:, 0],
        y=tsne[:, 1],
        hue=y_true,
        palette="tab10",
        legend="full",
        s=15
    )
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


# ======================= best_parameters.txt =======================
def write_best_parameters_txt(out_dir: str):
    txt_path = os.path.join(out_dir, "best_parameters.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MSF-DFormer fixed hyperparameters (baseline, TensorFlow/Keras)\n")
        f.write("=" * 78 + "\n\n")
        f.write("Paper: MSF-DFormer (Multisensor Multiscale Fusion Network with Deformable Transformer)\n")
        f.write("Key idea: MDSCB -> SCAFM(SCRM+SFFM) x3 -> ADFF -> DOT -> classifier\n\n")
        f.write("[Training protocol aligned to MHCNN]\n")
        f.write(f"  Sample range: {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop - 1}\n")
        f.write(f"  Repeats per N: {REPEAT_TIMES}\n")
        f.write(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}\n")
        f.write(f"  train_verbose={TRAIN_VERBOSE}, validation_batch_size={VALIDATION_BATCH_SIZE}\n")
        f.write(f"  validation_freq={VALIDATION_FREQ}\n")
        f.write("  mixed_precision=mixed_float16, jit_compile=False (Compatibility)\n")
        f.write(f"  tsne_max_points={TSNE_MAX_POINTS}\n\n")
        f.write("[Model (Lightweight)]\n")
        f.write(f"  MDSCB filters={MDSCB_FILTERS}, expansion rates=2/4/6, 3 stages\n")
        f.write(f"  ADFF filters={ADFF_FILTERS}, seq_len(tokens)={SEQ_LEN_TOKENS}\n")
        f.write(f"  DOT d_model={DOT_DMODEL}, nhead={DOT_NHEAD}, n_layers={DOT_NLAYERS}\n")
        f.write(f"  Classes={NUM_CLASSES}\n")


# ======================= Main =======================
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_MSF_DFormer.txt")
    sys.stdout = Logger(log_path, sys.stdout)

    gpus = tf.config.list_physical_devices("GPU")
    print("[TF] GPUs:", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("[TF] set_memory_growth failed:", e)

    write_best_parameters_txt(BASE_OUTPUT_DIR)

    print(">>> MSF-DFormer baseline experiment (Case 1, TensorFlow/Keras) <<<")
    print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
    if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
        print(f"TEMP_DIR            = {TEMP_DIR}")
    print(f"BASE_OUTPUT_DIR     = {BASE_OUTPUT_DIR}")
    print(f"DATA_PATH_ROOT      = {DATA_PATH_ROOT}")
    print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}")
    print(f"[Speed knobs] verbose={TRAIN_VERBOSE}, val_batch={VALIDATION_BATCH_SIZE}, val_freq={VALIDATION_FREQ}, tsne_max_points={TSNE_MAX_POINTS}")
    print("[Speed] mixed_precision=mixed_float16, jit_compile=False (Compatibility)")

    # ----- Load data -----
    ALL_DATASETS = load_all_data()
    Xv_time, Xa_time, y_all, class_ranges = pack_datasets_to_global_arrays(ALL_DATASETS)
    per_class = class_ranges[0][1] - class_ranges[0][0]
    print(f"[Data] classes={NUM_CLASSES}, per-class samples={per_class} (per modality)")

    # ----- Print model summary ONCE -----
    tf.keras.backend.clear_session()
    _model_for_summary, _ = build_msf_dformer(num_classes=NUM_CLASSES)
    print_model_summary_once(_model_for_summary)
    del _model_for_summary
    tf.keras.backend.clear_session()
    gc.collect()

    summary_stats = []

    for num_samples in SAMPLE_RANGE:
        sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n======== Training samples per class: {num_samples} (Runs 1-{REPEAT_TIMES}) ========")

        metrics_buffer = {"acc": [], "f1": [], "prec": [], "recall": []}

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            seed = num_samples * 100 + run_idx
            set_global_seed(seed)
            tf.keras.backend.clear_session()
            gc.collect()

            tr_idx, te_idx = get_split_indices_by_class(class_ranges, seed=seed, num_train_per_class=num_samples)

            x_tr_v = Xv_time[tr_idx][:, :, np.newaxis]
            x_tr_a = Xa_time[tr_idx][:, :, np.newaxis]
            y_tr = y_all[tr_idx]

            x_te_v = Xv_time[te_idx][:, :, np.newaxis]
            x_te_a = Xa_time[te_idx][:, :, np.newaxis]
            y_te = y_all[te_idx]

            y_tr_oh = tf.keras.utils.to_categorical(y_tr, NUM_CLASSES).astype(np.float32)
            y_te_oh = tf.keras.utils.to_categorical(y_te, NUM_CLASSES).astype(np.float32)

            print(f"  -> Start Run_{run_idx:02d}, seed={seed}, train={x_tr_v.shape[0]}, test={x_te_v.shape[0]}")
            sys.stdout.flush()

            model, feat_model = build_msf_dformer(num_classes=NUM_CLASSES)

            if OPTIMIZER_NAME.lower() == "adamax":
                opt = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE, clipnorm=1.0)
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

            # [FIXED] Removed jit_compile=True
            model.compile(
                optimizer=opt,
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            # ----- OOM Safe Training -----
            try:
                hist = model.fit(
                    [x_tr_v, x_tr_a], y_tr_oh,
                    validation_data=([x_te_v, x_te_a], y_te_oh),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=TRAIN_VERBOSE,
                    validation_batch_size=VALIDATION_BATCH_SIZE,
                    validation_freq=VALIDATION_FREQ
                )
            except tf.errors.ResourceExhaustedError:
                print(f"[WARN] OOM at batch={BATCH_SIZE}, val_batch={VALIDATION_BATCH_SIZE}. Retrying with val_batch=4...")
                tf.keras.backend.clear_session()
                gc.collect()
                model, feat_model = build_msf_dformer(num_classes=NUM_CLASSES)

                model.compile(
                    optimizer=opt,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

                hist = model.fit(
                    [x_tr_v, x_tr_a], y_tr_oh,
                    validation_data=([x_te_v, x_te_a], y_te_oh),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=TRAIN_VERBOSE,
                    validation_batch_size=4,
                    validation_freq=VALIDATION_FREQ
                )

            # ----- predict -----
            y_pred_prob = model.predict([x_te_v, x_te_a], verbose=0, batch_size=VALIDATION_BATCH_SIZE)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = y_te

            # ----- features for t-SNE -----
            feat_te = feat_model.predict([x_te_v, x_te_a], verbose=0, batch_size=VALIDATION_BATCH_SIZE)
            feat_te = np.nan_to_num(feat_te, nan=0.0, posinf=0.0, neginf=0.0)

            # ----- metrics -----
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            rec = recall_score(y_true, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}")

            # ----- Visualizations -----
            plot_curves(hist, os.path.join(run_dir, "curves.png"), num_samples)
            plot_cm(y_true, y_pred, os.path.join(run_dir, "cm.png"), num_samples)
            plot_tsne(feat_te, y_true, os.path.join(run_dir, "tsne.png"), num_samples)
            plot_roc(y_true, y_pred_prob, os.path.join(run_dir, "roc.png"), num_samples)

            del model, feat_model
            gc.collect()

        # ----- Trimmed statistics -----
        stats = [num_samples]
        for metric in ["acc", "f1", "prec", "recall"]:
            values = sorted(metrics_buffer[metric])
            trimmed_values = values[1:-1] if len(values) > 2 else values
            stats.append(float(np.mean(trimmed_values)))
            stats.append(float(np.std(trimmed_values)))
        summary_stats.append(stats)

        print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

    # ----- Save summary CSV -----
    cols = [
        "Samples",
        "Acc_Mean", "Acc_Std",
        "F1_Mean", "F1_Std",
        "Prec_Mean", "Prec_Std",
        "Recall_Mean", "Recall_Std"
    ]
    df = pd.DataFrame(summary_stats, columns=cols)
    out_csv = os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_MSF_DFormer.csv")
    df.to_csv(out_csv, index=False)

    # ----- Performance trend plot -----
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o", label="Accuracy", capsize=5)
    plt.errorbar(df["Samples"], df["F1_Mean"], yerr=df["F1_Std"], fmt="-s", label="F1 Score", capsize=5)
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Score")
    plt.title("Performance vs. Sample Size")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend_msf_dformer.png"), dpi=300)
    plt.close()

    print(f"\n[All Done] Results saved to: {BASE_OUTPUT_DIR}")
    print(f"[Log] {log_path}")
    print(f"[Best Params] {os.path.join(BASE_OUTPUT_DIR, 'best_parameters.txt')}")
    print(f"[Summary CSV] {out_csv}")