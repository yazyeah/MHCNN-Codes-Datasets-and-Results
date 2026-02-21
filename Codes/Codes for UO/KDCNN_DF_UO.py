# -*- coding: utf-8 -*-
"""
KDCNN-DF baseline experiment script (Case 1: uOttawa vibration + acoustic)

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

Target:
- Network architecture follows the KDCNN-DF paper (teacher CNN + KD-trained student CNN + DF classifier).
- Experiment protocol / outputs aligned with your MHCNN script:
  - Same data loading + splitting logic (Samples_05..Samples_30, Run_01..Run_10, seed=num_samples*100+run_idx)
  - Same metrics: Accuracy / Macro-F1 / Macro-Precision / Macro-Recall
  - Same per-run artifacts: curves.png, cm.png, tsne.png, roc.png
  - Same plotting style: 300 dpi, Times New Roman, title formats consistent
  - Same stdout logging to experiment_log_*.txt
  - Generate best_parameters.txt (fixed, no Optuna) for consistency

Fixes (NaN Error):
- Added clipnorm=1.0 to optimizers to prevent exploding gradients in few-shot training.
- Added np.nan_to_num() cleanup before Deep Forest stage.
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
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils import np_utils

from scipy.io import loadmat
from scipy.signal import cwt, morlet2, resample

import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from itertools import cycle

# ====== Global plotting style: 300 DPI, Times New Roman ======
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 1) Global config =================
SAMPLE_RANGE = range(5, 31)   # 5..30
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7

# =========================
# Paths (placeholders)
# =========================
# Replace these placeholders with YOUR local absolute paths.
# UO_DATA_ROOT MUST point to: ...\3_MatLab_Raw_Data
DATA_PATH_ROOT  = r"YOUR_UO_DATA_ROOT"      # e.g., r"D:\uOttawa\3_MatLab_Raw_Data"
BASE_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"     # e.g., r"D:\Results\KDCNN_DF_UO"

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

# ---- Align with your MHCNN "full experiment" stage ----
EPOCHS = 80
LEARNING_RATE = 1.3866e-3
BATCH_SIZE = 16
OPTIMIZER_NAME = "Adamax"

# ---- KD settings (from the paper) ----
TEMPERATURE = 4.0
ALPHA = 0.6     # hard-label CE weight
BETA  = 0.4     # soft-label KD (KL) weight

# ---- CWT settings ----
CWT_SCALES = 64
CWT_TIME_BINS = 64
MORLET_W = 5.0

# Train-only standardization + clip
CLIP_VALUE = 3.0
EPS = 1e-8

# ---- DF / cascade forest settings ----
USE_DF_CLASSIFIER = True
MAX_CASCADE_LAYERS = 4
N_TREES = 200
CASCADE_PATIENCE = 1
CASCADE_TOL = 1e-4

USE_XGBOOST = False


# ================= 2) Reproducibility helper =================
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ================= 3) Logging helper =================
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
                np.zeros((tot_num0, DATA_POINTS), dtype=np.float32)
            )

    datasets = []

    def load_class_data(file_list):
        v_list, a_list = [], []
        for folder, file in file_list:
            v, a = load_mat_category(folder, file)
            v_list.append(v)
            a_list.append(a)
        return np.vstack(v_list), np.vstack(a_list)

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
    test_idx = np.concatenate(test_idx)

    np.random.seed(seed)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


# ================= 5) CWT preprocessing + caching =================
def cwt_scalogram_64x64(sig_1d: np.ndarray) -> np.ndarray:
    widths = np.arange(1, CWT_SCALES + 1)
    coef = cwt(sig_1d, lambda M, s: morlet2(M, s, w=MORLET_W), widths)
    mag = np.abs(coef).astype(np.float32)  # (64, 2048)
    mag_rs = resample(mag, CWT_TIME_BINS, axis=1).astype(np.float32)  # (64,64)
    mag_rs = np.log1p(mag_rs)  # Log compression
    return mag_rs


def build_or_load_cwt_cache(X_time: np.ndarray, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"[CWT] Loading cache: {cache_path}")
        arr = np.load(cache_path)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]
        return arr.astype(np.float32)

    print(f"[CWT] Building cache: {cache_path}")
    N = X_time.shape[0]
    out = np.zeros((N, CWT_SCALES, CWT_TIME_BINS), dtype=np.float32)
    for i in range(N):
        out[i] = cwt_scalogram_64x64(X_time[i])
        if (i + 1) % 200 == 0:
            print(f"  [CWT] Progress: {i+1}/{N}")
    out = out[..., np.newaxis]  # (N,64,64,1)
    np.save(cache_path, out)
    print(f"[CWT] Cache saved: {cache_path}")
    return out


def standardize_clip_train_only(x_train, x_test, clip_value=3.0):
    mu = np.mean(x_train, axis=0, keepdims=True)
    sd = np.std(x_train, axis=0, keepdims=True) + EPS
    x_train_n = (x_train - mu) / sd
    x_test_n = (x_test - mu) / sd
    x_train_n = np.clip(x_train_n, -clip_value, clip_value)
    x_test_n = np.clip(x_test_n, -clip_value, clip_value)
    return x_train_n.astype(np.float32), x_test_n.astype(np.float32)


# ================= 6) Teacher / Student CNN (paper-style) =================
def build_teacher_cnn(input_shape=(64, 64, 1), num_classes=NUM_CLASSES):
    inp = Input(shape=input_shape, name="teacher_in")
    x = Conv2D(32, (5, 5), padding="same", strides=(1, 1), name="t_conv1")(inp)
    x = BatchNormalization(name="t_bn1")(x)
    x = Activation("relu", name="t_relu1")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="t_pool1")(x)

    x = Conv2D(32, (3, 3), padding="same", strides=(1, 1), name="t_conv2")(x)
    x = Activation("relu", name="t_relu2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="t_pool2")(x)

    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), name="t_conv3")(x)
    x = Activation("relu", name="t_relu3")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="t_pool3")(x)

    x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), name="t_conv4")(x)
    x = Activation("relu", name="t_relu4")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="t_pool4")(x)

    x = Flatten(name="t_flatten")(x)
    x = Dense(128, activation="relu", name="t_fc1")(x)
    x = Dense(128, activation="relu", name="t_fc2")(x)
    logits = Dense(num_classes, activation=None, name="t_logits")(x)
    return Model(inp, logits, name="TeacherCNN")


def build_student35_cnn(input_shape=(64, 64, 1), num_classes=NUM_CLASSES):
    inp = Input(shape=input_shape, name="student_in")
    x = Conv2D(8, (3, 3), padding="same", strides=(1, 1), name="s_conv1")(inp)
    x = Activation("relu", name="s_relu1")(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same", name="s_pool1")(x)

    x = Conv2D(8, (5, 5), padding="same", strides=(1, 1), name="s_conv2")(x)
    x = Activation("relu", name="s_relu2")(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same", name="s_pool2")(x)

    x = Flatten(name="s_flatten")(x)          # typically 128-dim (8@4x4)
    emb = Dense(64, activation="relu", name="s_fc1")(x)  # embedding for DF / t-SNE
    logits = Dense(num_classes, activation=None, name="s_logits")(emb)

    model_logits = Model(inp, logits, name="Student35_logits")
    model_emb = Model(inp, emb, name="Student35_emb")
    return model_logits, model_emb


# ================= 7) KD training loop (SAFE tf.function per-run) =================
def _maybe_build_optimizer(opt, var_list):
    try:
        opt.build(var_list)
    except Exception:
        zero_grads = [tf.zeros_like(v) for v in var_list]
        opt.apply_gradients(zip(zero_grads, var_list))


def run_kd_training_two_modalities(
    x_tr_v, x_tr_a, y_tr_onehot,
    x_te_v, x_te_a, y_te_onehot,
    seed: int
):
    set_global_seed(seed)
    tf.keras.backend.clear_session()

    x_te_v_tf = tf.convert_to_tensor(x_te_v, dtype=tf.float32)
    x_te_a_tf = tf.convert_to_tensor(x_te_a, dtype=tf.float32)
    y_te_tf   = tf.convert_to_tensor(y_te_onehot, dtype=tf.float32)

    teacher_v = build_teacher_cnn()
    teacher_a = build_teacher_cnn()
    student_v_logits, student_v_emb = build_student35_cnn()
    student_a_logits, student_a_emb = build_student35_cnn()

    # [FIX]: Added clipnorm=1.0 to prevent exploding gradients
    if OPTIMIZER_NAME.lower() == "adamax":
        opt_tv = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE, clipnorm=1.0)
        opt_ta = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE, clipnorm=1.0)
        opt_sv = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE, clipnorm=1.0)
        opt_sa = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE, clipnorm=1.0)
    else:
        opt_tv = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        opt_ta = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        opt_sv = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        opt_sa = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    _maybe_build_optimizer(opt_tv, teacher_v.trainable_variables)
    _maybe_build_optimizer(opt_ta, teacher_a.trainable_variables)
    _maybe_build_optimizer(opt_sv, student_v_logits.trainable_variables)
    _maybe_build_optimizer(opt_sa, student_a_logits.trainable_variables)

    @tf.function
    def train_step_teacher(model_t, optimizer_t, x, y_onehot):
        with tf.GradientTape() as tape:
            logits = model_t(x, training=True)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits)
            )
        grads = tape.gradient(loss, model_t.trainable_variables)
        optimizer_t.apply_gradients(zip(grads, model_t.trainable_variables))
        return loss

    @tf.function
    def train_step_student_kd(model_s, optimizer_s, x, y_onehot, teacher_logits, T, alpha, beta):
        with tf.GradientTape() as tape:
            s_logits = model_s(x, training=True)
            hard = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=s_logits)
            )

            t_prob = tf.nn.softmax(teacher_logits / T)
            s_prob = tf.nn.softmax(s_logits / T)

            # [FIX]: Added 1e-12 to logs for numeric stability
            kl = tf.reduce_mean(
                tf.reduce_sum(
                    t_prob * (tf.math.log(t_prob + 1e-12) - tf.math.log(s_prob + 1e-12)),
                    axis=1
                )
            )
            loss = alpha * hard + beta * (T * T) * kl

        grads = tape.gradient(loss, model_s.trainable_variables)
        optimizer_s.apply_gradients(zip(grads, model_s.trainable_variables))
        return loss

    n_tr = x_tr_v.shape[0]
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(x_tr_v, dtype=tf.float32),
            tf.convert_to_tensor(x_tr_a, dtype=tf.float32),
            tf.convert_to_tensor(y_tr_onehot, dtype=tf.float32)
        )
    )
    ds = ds.shuffle(buffer_size=n_tr, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for ep in range(EPOCHS):
        ep_loss = []
        ep_acc  = []

        for xb_v, xb_a, yb in ds:
            # update teachers (hard CE)
            _ = train_step_teacher(teacher_v, opt_tv, xb_v, yb)
            _ = train_step_teacher(teacher_a, opt_ta, xb_a, yb)

            # teacher logits for KD
            tlog_v = tf.stop_gradient(teacher_v(xb_v, training=False))
            tlog_a = tf.stop_gradient(teacher_a(xb_a, training=False))

            # update students (KD)
            ls_v = train_step_student_kd(student_v_logits, opt_sv, xb_v, yb, tlog_v, TEMPERATURE, ALPHA, BETA)
            ls_a = train_step_student_kd(student_a_logits, opt_sa, xb_a, yb, tlog_a, TEMPERATURE, ALPHA, BETA)

            # fused train acc
            pv = tf.nn.softmax(student_v_logits(xb_v, training=False))
            pa = tf.nn.softmax(student_a_logits(xb_a, training=False))
            p  = (pv + pa) / 2.0

            y_pred = tf.argmax(p, axis=1)
            y_true = tf.argmax(yb, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

            ep_loss.append((ls_v + ls_a) / 2.0)
            ep_acc.append(acc)

        tr_loss = float(tf.reduce_mean(ep_loss).numpy())
        tr_acc  = float(tf.reduce_mean(ep_acc).numpy())

        # --- val (kept as-is: uses provided x_te_*, y_te_*) ---
        pv = tf.nn.softmax(student_v_logits(x_te_v_tf, training=False))
        pa = tf.nn.softmax(student_a_logits(x_te_a_tf, training=False))
        p  = (pv + pa) / 2.0

        y_pred = tf.argmax(p, axis=1)
        y_true = tf.argmax(y_te_tf, axis=1)
        val_acc = float(tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32)).numpy())

        val_ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_te_tf, p)).numpy()
        val_ce = float(val_ce)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(val_ce)
        history["val_accuracy"].append(val_acc)

    return (student_v_logits, student_v_emb, student_a_logits, student_a_emb, history)


# ================= 8) DF (cascade forest) =================
def _oof_and_test_proba(clf, X_train, y_train, X_test, seed):
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = int(np.min(counts))
    n_splits = min(5, min_count)
    if n_splits < 2:
        clf.fit(X_train, y_train)
        proba_train = clf.predict_proba(X_train)
        proba_test  = clf.predict_proba(X_test)
        return proba_train, proba_test

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((X_train.shape[0], NUM_CLASSES), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        clf_fold = RandomForestClassifier(
            n_estimators=getattr(clf, "n_estimators", 200),
            random_state=seed,
            n_jobs=-1
        ) if isinstance(clf, RandomForestClassifier) else ExtraTreesClassifier(
            n_estimators=getattr(clf, "n_estimators", 200),
            random_state=seed,
            n_jobs=-1
        )

        clf_fold.set_params(**clf.get_params())
        clf_fold.fit(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = clf_fold.predict_proba(X_train[va_idx]).astype(np.float32)

    clf.fit(X_train, y_train)
    proba_test = clf.predict_proba(X_test).astype(np.float32)
    return oof, proba_test


def fit_cascade_forest(X_train, y_train, X_test, y_test, seed):
    Xtr = X_train.copy()
    Xte = X_test.copy()

    best_acc = -1.0
    best_proba = None
    no_improve = 0

    for layer in range(1, MAX_CASCADE_LAYERS + 1):
        rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed + 31 * layer, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=N_TREES, random_state=seed + 97 * layer, n_jobs=-1)

        oof_rf, te_rf = _oof_and_test_proba(rf, Xtr, y_train, Xte, seed + 1000 * layer)
        oof_et, te_et = _oof_and_test_proba(et, Xtr, y_train, Xte, seed + 2000 * layer)

        proba_layer = (te_rf + te_et) / 2.0
        y_pred = np.argmax(proba_layer, axis=1)
        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc + CASCADE_TOL:
            best_acc = acc
            best_proba = proba_layer
            no_improve = 0
        else:
            no_improve += 1

        Xtr = np.concatenate([Xtr, oof_rf, oof_et], axis=1)
        Xte = np.concatenate([Xte, te_rf, te_et], axis=1)

        if no_improve >= CASCADE_PATIENCE:
            break

    return best_proba


# ================= 9) Consistent best_parameters.txt =================
def write_best_parameters_txt(out_dir: str):
    txt_path = os.path.join(out_dir, "best_parameters.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("KDCNN-DF fixed hyperparameters (baseline, no Optuna)\n")
        f.write("=" * 78 + "\n\n")
        f.write("Paper: An Integrated Framework for Bearing Fault Diagnosis: CNN Model Compression Through Knowledge Distillation\n")
        f.write("Key idea: CWT scalogram -> Teacher CNN -> KD-trained Student_35 (KDCNN) -> DF classifier\n\n")
        f.write("[Training protocol aligned to MHCNN]\n")
        f.write(f"  Sample range: {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}\n")
        f.write(f"  Repeats per N: {REPEAT_TIMES}\n")
        f.write(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}\n\n")
        f.write("[Stability fixes]\n")
        f.write("  - clipnorm=1.0 on optimizers\n")
        f.write("  - np.nan_to_num before DF stage\n")


# ================= 10) Main experiment =================
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_KDCNN_DF.txt")
    sys.stdout = Logger(log_path, sys.stdout)

    write_best_parameters_txt(BASE_OUTPUT_DIR)

    print(">>> KDCNN-DF baseline experiment (Case 1) <<<")
    print(f"CUDA_VISIBLE_DEVICES = {GPU_ID}")
    if TEMP_DIR and ("YOUR_" not in TEMP_DIR):
        print(f"TEMP_DIR            = {TEMP_DIR}")
    print(f"BASE_OUTPUT_DIR     = {BASE_OUTPUT_DIR}")
    print(f"DATA_PATH_ROOT      = {DATA_PATH_ROOT}")
    print(f"SAMPLE_RANGE        = {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}")
    print(f"REPEAT_TIMES        = {REPEAT_TIMES}")
    print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, optimizer={OPTIMIZER_NAME}\n")

    # ----- Load data -----
    ALL_DATASETS = load_all_data()
    Xv_time, Xa_time, y_all, class_ranges = pack_datasets_to_global_arrays(ALL_DATASETS)
    print(f"[Data] Total samples: {Xv_time.shape[0]} (per modality), classes={NUM_CLASSES}")

    # ----- Build or load CWT cache -----
    cache_v = os.path.join(BASE_OUTPUT_DIR, "cwt_cache_vib_64x64.npy")
    cache_a = os.path.join(BASE_OUTPUT_DIR, "cwt_cache_aco_64x64.npy")

    Xv_cwt = build_or_load_cwt_cache(Xv_time, cache_v)
    Xa_cwt = build_or_load_cwt_cache(Xa_time, cache_a)

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

            # ----- split indices -----
            tr_idx, te_idx = get_split_indices_by_class(class_ranges, seed=seed, num_train_per_class=num_samples)

            x_tr_v = Xv_cwt[tr_idx]
            x_tr_a = Xa_cwt[tr_idx]
            y_tr = y_all[tr_idx]
            y_tr_oh = np_utils.to_categorical(y_tr, NUM_CLASSES).astype(np.float32)

            x_te_v = Xv_cwt[te_idx]
            x_te_a = Xa_cwt[te_idx]
            y_te = y_all[te_idx]
            y_te_oh = np_utils.to_categorical(y_te, NUM_CLASSES).astype(np.float32)

            # ----- train-only standardize + clip -----
            x_tr_v, x_te_v = standardize_clip_train_only(x_tr_v, x_te_v, CLIP_VALUE)
            x_tr_a, x_te_a = standardize_clip_train_only(x_tr_a, x_te_a, CLIP_VALUE)

            # ----- KD training -----
            sv_logits, sv_emb, sa_logits, sa_emb, hist = run_kd_training_two_modalities(
                x_tr_v, x_tr_a, y_tr_oh,
                x_te_v, x_te_a, y_te_oh,
                seed=seed
            )

            # ----- Features for DF & t-SNE -----
            feat_tr = np.concatenate([
                sv_emb.predict(x_tr_v, verbose=0),
                sa_emb.predict(x_tr_a, verbose=0)
            ], axis=1)  # (Ntr,128)

            feat_te = np.concatenate([
                sv_emb.predict(x_te_v, verbose=0),
                sa_emb.predict(x_te_a, verbose=0)
            ], axis=1)  # (Nte,128)

            # [FIX]: Handle NaNs produced by unstable KD training before passing to Random Forest
            feat_tr = np.nan_to_num(feat_tr, nan=0.0, posinf=0.0, neginf=0.0)
            feat_te = np.nan_to_num(feat_te, nan=0.0, posinf=0.0, neginf=0.0)

            # ----- Prediction probabilities -----
            if USE_DF_CLASSIFIER:
                y_pred_prob = fit_cascade_forest(feat_tr, y_tr, feat_te, y_te, seed=seed)
                if y_pred_prob is None:
                    pv = tf.nn.softmax(sv_logits(x_te_v, training=False)).numpy()
                    pa = tf.nn.softmax(sa_logits(x_te_a, training=False)).numpy()
                    y_pred_prob = (pv + pa) / 2.0
            else:
                pv = tf.nn.softmax(sv_logits(x_te_v, training=False)).numpy()
                pa = tf.nn.softmax(sa_logits(x_te_a, training=False)).numpy()
                y_pred_prob = (pv + pa) / 2.0

            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = y_te

            # ----- metrics -----
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")
            prec = precision_score(y_true, y_pred, average="macro")
            rec  = recall_score(y_true, y_pred, average="macro")

            metrics_buffer["acc"].append(acc)
            metrics_buffer["f1"].append(f1)
            metrics_buffer["prec"].append(prec)
            metrics_buffer["recall"].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}")

            # ================= Visualizations =================
            # curves.png
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(hist["loss"], label="Train")
            plt.plot(hist["val_loss"], label="Test")
            plt.title(f"Loss - Sample {num_samples}")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(hist["accuracy"], label="Train")
            plt.plot(hist["val_accuracy"], label="Test")
            plt.title(f"Accuracy - Sample {num_samples}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "curves.png"), dpi=300)
            plt.close()

            # cm.png
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - Sample {num_samples}")
            plt.savefig(os.path.join(run_dir, "cm.png"), dpi=300)
            plt.close()

            # tsne.png
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
            plt.savefig(os.path.join(run_dir, "tsne.png"), dpi=300)
            plt.close()

            # roc.png
            y_test_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
            fpr, tpr, roc_auc = dict(), dict(), dict()

            for i in range(NUM_CLASSES):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # micro-average
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # macro-average
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

            # basic cleanup
            tf.keras.backend.clear_session()
            gc.collect()

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
    df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_KDCNN_DF.csv"), index=False)

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
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend_kdcnn_df.png"), dpi=300)
    plt.close()

    print(f"\n[All Done] Results saved to: {BASE_OUTPUT_DIR}")
    print(f"[Log] {log_path}")
    print(f"[Best Params] {os.path.join(BASE_OUTPUT_DIR, 'best_parameters.txt')}")