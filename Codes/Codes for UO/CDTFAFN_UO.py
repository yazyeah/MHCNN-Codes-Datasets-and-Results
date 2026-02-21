# -*- coding: utf-8 -*-
"""
CDTFAFN baseline experiment script (Case 1: uOttawa vibration + acoustic)

Goal: replicate the *output protocol* of MHCNN_Oputuna_Experiment.py:
- Same data loading + splitting logic (Samples_05..Samples_30, Run_01..Run_10)
- Same metrics: Accuracy / Macro-F1 / Macro-Precision / Macro-Recall
- Same visualization artifacts per run: curves.png, cm.png, tsne.png, roc.png
- Same aggregated outputs: Final_Summary_Stats.csv, performance_trend.png
- Same plotting style: 300 dpi, Times New Roman
- Same logging: experiment_log_CDTFAFN.txt
- Same "best_parameters.txt" existence (for traceability; no Optuna here)

Difference: model is CDTFAFN (SIEU -> coarse dual-scale conv -> fine dual-branch + TFA-FFU -> concat -> FC -> softmax).

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Flatten, Activation, Add, Multiply, Concatenate, Lambda,
    Conv2D, BatchNormalization,
    GlobalAveragePooling2D, GlobalMaxPooling2D
)
from keras.utils import np_utils

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

import seaborn as sns
from scipy.io import loadmat
from scipy import signal as sp_signal
from itertools import cycle
from PIL import Image

# ====== Global plotting style: 300 DPI, Times New Roman ======
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 1. Global config =================
SAMPLE_RANGE = range(5, 31)     # 5..30
REPEAT_TIMES = 10               # Run_01..Run_10

DATA_POINTS = 2048
NUM_CLASSES = 7                 # <-- Your Case 1 (uOttawa) in this project uses 7 classes

# =========================
# Paths 
# =========================
# Replace these placeholders with YOUR local absolute paths.
# UO_DATA_ROOT MUST point to: ...\3_MatLab_Raw_Data
DATA_PATH_ROOT = r"YOUR_UO_DATA_ROOT"       # <-- e.g., r"D:\uOttawa\3_MatLab_Raw_Data"
BASE_OUTPUT_DIR = r"YOUR_UO_OUTPUT_DIR"     # <-- e.g., r"D:\Results\CDTFAFN_UO"

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

# Training hyper-params
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# SIEU (signal-to-image encoding) config (default: STFT -> 64x64x3)
TFR_H, TFR_W, TFR_C = 64, 64, 3
STFT_NPERSEG = 128
STFT_NOVERLAP = 96
STFT_NFFT = 256

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
    print("[Data] Loading uOttawa vibration-acoustic dataset...")
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
            vib_samples = np.zeros((tot_num0, DATA_POINTS))
            aco_samples = np.zeros((tot_num0, DATA_POINTS))
            for i in range(tot_num0):
                if (i + 1) * DATA_POINTS <= len(vib):
                    vib_samples[i, :] = vib[i * DATA_POINTS:(i + 1) * DATA_POINTS]
                    aco_samples[i, :] = aco[i * DATA_POINTS:(i + 1) * DATA_POINTS]
            return vib_samples, aco_samples
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
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
    datasets.append(load_class_data([("4_Ball_Faults", "B_11_2.mat"), ("4_Ball_Faults", "B_11_2.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_1.mat"), ("5_Cage_Faults", "C_17_1.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_2.mat"), ("5_Cage_Faults", "C_17_2.mat")]))

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

# ================= 4. SIEU (signal -> TFR image) =================
def signal_to_tfr(sig_1d: np.ndarray) -> np.ndarray:
    """
    Default SIEU implementation: STFT magnitude -> log1p -> min-max -> resize to 64x64 -> 3 channels.
    Replace this function with ICQ-NSGT-based TFR if you have that implementation.
    """
    x = sig_1d.astype(np.float32).copy()
    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-8)

    f, t, Zxx = sp_signal.stft(
        x,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
        boundary=None
    )
    S = np.abs(Zxx)
    S = np.log1p(S)

    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    img_u8 = (S * 255.0).astype(np.uint8)

    pil = Image.fromarray(img_u8)
    pil = pil.resize((TFR_W, TFR_H), resample=Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0

    arr3 = np.stack([arr, arr, arr], axis=-1)  # 64x64x3
    return arr3

def batch_to_tfr(X: np.ndarray) -> np.ndarray:
    """
    X: (N, 2048, 1) -> (N, 64, 64, 3)
    """
    N = X.shape[0]
    out = np.zeros((N, TFR_H, TFR_W, TFR_C), dtype=np.float32)
    for i in range(N):
        out[i] = signal_to_tfr(X[i, :, 0])
    return out

# ================= 5. CDTFAFN building blocks =================
def conv_bn_relu(x, filters, k, s=1, name=None):
    x = Conv2D(filters, kernel_size=k, strides=s, padding="same",
               name=None if name is None else f"{name}_conv")(x)
    x = BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = Activation("relu", name=None if name is None else f"{name}_relu")(x)
    return x

def dual_scale_block(x, name=None):
    """
    Dual-scale feature extraction block (Table 2):
      Conv2d 64, 5x5, stride 2 + BN + ReLU -> 32x32x64
      Conv2d 32, 3x3, stride 1            -> 32x32x32 (no BN/ReLU in table)
    """
    x = conv_bn_relu(x, filters=64, k=(5, 5), s=2, name=None if name is None else f"{name}_5x5")
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding="same",
               name=None if name is None else f"{name}_3x3")(x)
    return x  # 32x32x32

def cam_module(F, reduction=8, name=None):
    """
    Channel Attention Module (CAM), CBAM-style:
      Mc(F) = sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    """
    C = int(F.shape[-1])
    hidden = max(C // reduction, 1)

    avg = GlobalAveragePooling2D(name=None if name is None else f"{name}_avg")(F)
    mx  = GlobalMaxPooling2D(name=None if name is None else f"{name}_max")(F)

    mlp1 = Dense(hidden, activation="relu", name=None if name is None else f"{name}_mlp1")
    mlp2 = Dense(C, activation=None, name=None if name is None else f"{name}_mlp2")

    avg_out = mlp2(mlp1(avg))
    max_out = mlp2(mlp1(mx))
    att = Add(name=None if name is None else f"{name}_add")([avg_out, max_out])
    att = Activation("sigmoid", name=None if name is None else f"{name}_sigmoid")(att)
    att = Lambda(lambda t: tf.reshape(t, (-1, 1, 1, C)),
                 name=None if name is None else f"{name}_reshape")(att)

    return Multiply(name=None if name is None else f"{name}_mul")([F, att])

def axis_attention(Fc, axis="freq", name=None):
    """
    Engineering implementation of FAM/TAM:
    - FAM: focus on frequency axis (y-axis). Pool over time+channel -> 1 x W x 1, conv along W.
    - TAM: focus on time axis (x-axis). Pool over freq+channel  -> H x 1 x 1, conv along H.
    """
    if axis == "freq":
        avg = Lambda(lambda t: tf.reduce_mean(t, axis=[1, 3], keepdims=True),
                     name=None if name is None else f"{name}_avg")(Fc)  # 1 x W x 1
        mx  = Lambda(lambda t: tf.reduce_max(t, axis=[1, 3], keepdims=True),
                     name=None if name is None else f"{name}_max")(Fc)
        x = Concatenate(axis=-1, name=None if name is None else f"{name}_cat")([avg, mx])  # 1 x W x 2
        # 3 cascaded convs (paper uses 3x3; here use (1,3) to emphasize frequency axis)
        x = Conv2D(8, (1, 3), padding="same", activation="relu",
                   name=None if name is None else f"{name}_c1")(x)
        x = Conv2D(8, (1, 3), padding="same", activation="relu",
                   name=None if name is None else f"{name}_c2")(x)
        x = Conv2D(1, (1, 3), padding="same", activation="sigmoid",
                   name=None if name is None else f"{name}_c3")(x)  # 1 x W x 1
        return Multiply(name=None if name is None else f"{name}_mul")([Fc, x])

    else:  # axis == "time"
        avg = Lambda(lambda t: tf.reduce_mean(t, axis=[2, 3], keepdims=True),
                     name=None if name is None else f"{name}_avg")(Fc)  # H x 1 x 1
        mx  = Lambda(lambda t: tf.reduce_max(t, axis=[2, 3], keepdims=True),
                     name=None if name is None else f"{name}_max")(Fc)
        x = Concatenate(axis=-1, name=None if name is None else f"{name}_cat")([avg, mx])  # H x 1 x 2
        # 3 cascaded convs (use (3,1) to emphasize time axis)
        x = Conv2D(8, (3, 1), padding="same", activation="relu",
                   name=None if name is None else f"{name}_c1")(x)
        x = Conv2D(8, (3, 1), padding="same", activation="relu",
                   name=None if name is None else f"{name}_c2")(x)
        x = Conv2D(1, (3, 1), padding="same", activation="sigmoid",
                   name=None if name is None else f"{name}_c3")(x)  # H x 1 x 1
        return Multiply(name=None if name is None else f"{name}_mul")([Fc, x])

def ms_cam(F, reduction=8, name=None):
    """
    MS-CAM style attention for AFF:
    produce weight W in [0,1] with both global & local context, then broadcast.
    """
    C = int(F.shape[-1])
    hidden = max(C // reduction, 1)

    # local
    local = Conv2D(hidden, (1, 1), padding="same", activation="relu",
                   name=None if name is None else f"{name}_l1")(F)
    local = BatchNormalization(name=None if name is None else f"{name}_lbn1")(local)
    local = Conv2D(C, (1, 1), padding="same", activation=None,
                   name=None if name is None else f"{name}_l2")(local)
    local = BatchNormalization(name=None if name is None else f"{name}_lbn2")(local)

    # global
    g = GlobalAveragePooling2D(name=None if name is None else f"{name}_gap")(F)
    g = Dense(hidden, activation="relu", name=None if name is None else f"{name}_g1")(g)
    g = Dense(C, activation=None, name=None if name is None else f"{name}_g2")(g)
    g = Lambda(lambda t: tf.reshape(t, (-1, 1, 1, C)),
               name=None if name is None else f"{name}_greshape")(g)

    w = Add(name=None if name is None else f"{name}_add")([local, g])
    w = Activation("sigmoid", name=None if name is None else f"{name}_sigmoid")(w)
    return w  # HxWxC (broadcastable)

def aff_fuse(Ff, Ft, name=None):
    """
    AFF-like fusion with two-stage MS-CAM gating.
    """
    one_minus = lambda w: 1.0 - w

    s1 = Add(name=None if name is None else f"{name}_s1")([Ff, Ft])
    w1 = ms_cam(s1, reduction=8, name=None if name is None else f"{name}_ms1")
    Ff1 = Multiply(name=None if name is None else f"{name}_Ff1")([Ff, w1])
    Ft1 = Multiply(name=None if name is None else f"{name}_Ft1")([Ft, Lambda(one_minus, name=None if name is None else f"{name}_om1")(w1)])

    s2 = Add(name=None if name is None else f"{name}_s2")([Ff1, Ft1])
    w2 = ms_cam(s2, reduction=8, name=None if name is None else f"{name}_ms2")
    out = Add(name=None if name is None else f"{name}_out")([
        Multiply()([Ff1, w2]),
        Multiply()([Ft1, Lambda(one_minus)(w2)])
    ])
    return out

def tfa_ffu(F, name=None):
    """
    TFA-FFU (engineering replica):
      Fc = CAM(F)
      Ff = FAM(Fc)
      Ft = TAM(Fc)
      Fft = AFF(Ff, Ft)
      output = Conv1x1(Fft) + BN + ReLU (keep same channels)
    """
    C = int(F.shape[-1])

    Fc = cam_module(F, reduction=8, name=None if name is None else f"{name}_cam")
    Ff = axis_attention(Fc, axis="freq", name=None if name is None else f"{name}_fam")
    Ft = axis_attention(Fc, axis="time", name=None if name is None else f"{name}_tam")

    Fft = aff_fuse(Ff, Ft, name=None if name is None else f"{name}_aff")

    out = Conv2D(C, (1, 1), padding="same", name=None if name is None else f"{name}_pw")(Fft)
    out = BatchNormalization(name=None if name is None else f"{name}_bn")(out)
    out = Activation("relu", name=None if name is None else f"{name}_relu")(out)
    return out

def feature_learning_block(x, ksize, filters, name=None):
    """
    One feature learning block in Table 2:
      Conv2d(filters, ksize, stride=1) + BN + ReLU
      + TFA-FFU (same channel)
    """
    x = conv_bn_relu(x, filters=filters, k=ksize, s=1, name=None if name is None else f"{name}_conv")
    x = tfa_ffu(x, name=None if name is None else f"{name}_tfa")
    return x

# ================= 6. CDTFAFN model definition =================
def create_cdtfafn_model():
    """
    CDTFAFN (Case 1 in this project uses 7 classes):
      Inputs: vib_tfr (64,64,3), aco_tfr (64,64,3)
      Coarse: dual_scale(v), dual_scale(a) -> F1,F2 (32,32,32)
      Fine: 2 branches on (F1+F2)
        branch1: 5x5 blocks -> 8ch
        branch2: 3x3 blocks -> 8ch
      Fusion: concat(F1,F2,F1',F2') -> 32x32x80
      Classifier: Flatten -> Dense(128, ReLU) -> Dense(NUM_CLASSES) -> Softmax
    """
    inp_v = Input(shape=(TFR_H, TFR_W, TFR_C), name="input_vib_tfr")
    inp_a = Input(shape=(TFR_H, TFR_W, TFR_C), name="input_aco_tfr")

    F1 = dual_scale_block(inp_v, name="v_dual")
    F2 = dual_scale_block(inp_a, name="a_dual")

    Ff1 = Add(name="coarse_add")([F1, F2])  # coarse fused (32,32,32)

    # Fine-grained stage: two branches
    # Branch 1: 5x5 convs (32->16->8), each followed by TFA-FFU
    b1 = feature_learning_block(Ff1, ksize=(5, 5), filters=32, name="b1_blk1")
    b1 = feature_learning_block(b1,  ksize=(5, 5), filters=16, name="b1_blk2")
    F1p = feature_learning_block(b1, ksize=(5, 5), filters=8,  name="b1_blk3")  # (32,32,8)

    # Branch 2: 3x3 convs (32->16->8), each followed by TFA-FFU
    b2 = feature_learning_block(Ff1, ksize=(3, 3), filters=32, name="b2_blk1")
    b2 = feature_learning_block(b2,  ksize=(3, 3), filters=16, name="b2_blk2")
    F2p = feature_learning_block(b2, ksize=(3, 3), filters=8,  name="b2_blk3")  # (32,32,8)

    # Final fusion: cat along channels -> 32x32x80
    Ff2 = Concatenate(axis=-1, name="final_concat")([F1, F2, F1p, F2p])

    x = Flatten(name="flatten")(Ff2)
    feat_vec = Dense(128, activation="relu", name="fc1")(x)   # used for t-SNE
    logits = Dense(NUM_CLASSES, name="fc2")(feat_vec)
    out = Activation("softmax", name="softmax")(logits)

    model = Model(inputs=[inp_v, inp_a], outputs=out, name="CDTFAFN")
    feat_model = Model(inputs=[inp_v, inp_a], outputs=feat_vec, name="CDTFAFN_feat")
    return model, feat_model

# ================= 7. best_parameters.txt (for traceability) =================
def write_best_parameters_txt(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Best CDTFAFN Parameters (fixed, no Optuna)\n")
        f.write("=" * 70 + "\n")
        f.write(f"EPOCHS={EPOCHS}\nBATCH_SIZE={BATCH_SIZE}\nLEARNING_RATE={LEARNING_RATE}\n")
        f.write(f"SIEU=STFT(default)\nTFR={TFR_H}x{TFR_W}x{TFR_C}\n")
        f.write(f"STFT_NPERSEG={STFT_NPERSEG}\nSTFT_NOVERLAP={STFT_NOVERLAP}\nSTFT_NFFT={STFT_NFFT}\n")
        f.write("=" * 70 + "\n\n")
        f.write("CDTFAFN Model Architecture Summary:\n")
        f.write("=" * 70 + "\n")
        temp_model, _ = create_cdtfafn_model()
        temp_model.summary(print_fn=lambda s: f.write(s + "\n"))
        del temp_model

# ================= 8. Main experiment =================
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(BASE_OUTPUT_DIR, "experiment_log_CDTFAFN.txt")
    best_params_path = os.path.join(BASE_OUTPUT_DIR, "best_parameters.txt")

    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(">>> CDTFAFN baseline experiment (Case 1) <<<")
        print(f"BASE_OUTPUT_DIR = {BASE_OUTPUT_DIR}")
        print(f"DATA_PATH_ROOT  = {DATA_PATH_ROOT}")
        print(f"SAMPLE_RANGE    = {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}")
        print(f"REPEAT_TIMES    = {REPEAT_TIMES}")
        print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}\n")

        # Write best_parameters.txt (fixed)
        write_best_parameters_txt(best_params_path)
        print(f">>> best_parameters.txt written: {best_params_path}\n")

        # Also print model summary into log
        print(">>> CDTFAFN Model Architecture Summary:")
        print("=" * 60)
        temp_model, _ = create_cdtfafn_model()
        temp_model.summary()
        del temp_model
        print("=" * 60 + "\n")

        summary_stats = []

        for num_samples in SAMPLE_RANGE:
            sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            print(f"\n========== N={num_samples} (Runs 1..{REPEAT_TIMES}) ==========")

            metrics_buffer = {"acc": [], "f1": [], "prec": [], "recall": []}

            for run_idx in range(1, REPEAT_TIMES + 1):
                tf.keras.backend.clear_session()
                gc.collect()

                run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
                os.makedirs(run_dir, exist_ok=True)

                seed = num_samples * 100 + run_idx
                (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(
                    ALL_DATASETS, seed=seed, num_train_per_class=num_samples
                )

                # --- SIEU: convert raw 2048x1 to 64x64x3 TFR images ---
                x_tr_v_img = batch_to_tfr(x_tr_v)
                x_tr_a_img = batch_to_tfr(x_tr_a)
                x_te_v_img = batch_to_tfr(x_te_v)
                x_te_a_img = batch_to_tfr(x_te_a)

                model, feat_model = create_cdtfafn_model()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

                history = model.fit(
                    [x_tr_v_img, x_tr_a_img], y_tr,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=([x_te_v_img, x_te_a_img], y_te),
                    verbose=0
                )

                y_pred_prob = model.predict([x_te_v_img, x_te_a_img], verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_true = np.argmax(y_te, axis=1)

                acc = accuracy_score(y_true, y_pred)
                f1  = f1_score(y_true, y_pred, average="macro")
                prec = precision_score(y_true, y_pred, average="macro")
                rec  = recall_score(y_true, y_pred, average="macro")

                metrics_buffer["acc"].append(acc)
                metrics_buffer["f1"].append(f1)
                metrics_buffer["prec"].append(prec)
                metrics_buffer["recall"].append(rec)

                print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

                # --- Visualizations (same filenames as MHCNN protocol) ---
                # curves.png
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

                # cm.png
                plt.figure(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - Sample {num_samples}")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "cm.png"), dpi=300)
                plt.close()

                # tsne.png
                feats = feat_model.predict([x_te_v_img, x_te_a_img], verbose=0)
                tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
                plt.figure(figsize=(6, 5))
                sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_true, palette="tab10",
                                legend="full", s=15)
                plt.title(f"t-SNE - Sample {num_samples}")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "tsne.png"), dpi=300)
                plt.close()

                # roc.png (OvR per class + micro/macro)
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
                         label=f"Micro-avg (AUC={roc_auc['micro']:.4f})",
                         color="deeppink", linestyle=":", lw=3)
                plt.plot(fpr["macro"], tpr["macro"],
                         label=f"Macro-avg (AUC={roc_auc['macro']:.4f})",
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

                del model, feat_model

            # --- Trimmed mean/std (drop min & max) ---
            stats = [num_samples]
            for metric in ["acc", "f1", "prec", "recall"]:
                values = sorted(metrics_buffer[metric])
                trimmed = values[1:-1] if len(values) > 2 else values
                stats.append(float(np.mean(trimmed)))
                stats.append(float(np.std(trimmed)))
            summary_stats.append(stats)

            print(f"  >>> S{num_samples} done (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

        # --- Save Final_Summary_Stats.csv ---
        cols = ["Samples",
                "Acc_Mean", "Acc_Std",
                "F1_Mean", "F1_Std",
                "Prec_Mean", "Prec_Std",
                "Recall_Mean", "Recall_Std"]
        df = pd.DataFrame(summary_stats, columns=cols)
        df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats.csv"), index=False)

        # --- Save performance_trend.png (Accuracy + F1, same as MHCNN trend style) ---
        plt.figure(figsize=(10, 6))
        plt.errorbar(df["Samples"], df["Acc_Mean"], yerr=df["Acc_Std"], fmt="-o",
                     label="Accuracy", capsize=5)
        plt.errorbar(df["Samples"], df["F1_Mean"], yerr=df["F1_Std"], fmt="-s",
                     label="F1 Score", capsize=5)
        plt.xlabel("Training Samples per Class")
        plt.ylabel("Score")
        plt.title("Performance vs. Sample Size")
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, "performance_trend.png"), dpi=300)
        plt.close()

        print(f"\n[All Done] Results saved to:\n  {BASE_OUTPUT_DIR}")
        print(f"Log file: {log_path}")
        print(f"best_parameters.txt: {best_params_path}")

    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()