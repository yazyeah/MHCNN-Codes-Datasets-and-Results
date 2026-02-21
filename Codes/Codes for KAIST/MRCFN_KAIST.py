"""
MRCFN — KAIST Load-Shift Case Study 
Vibration (.mat) + Current (.tdms) | Segment length L=2048

[Task / Setting]
- Train/Val: 0Nm only (hash-overlap leakage check enabled)
- Test: 2Nm & 4Nm
- Few-shot protocol: N = 5..30 samples/class, repeated runs = 10
- Seed rule: seed = N*100 + run_idx

[Network (paper-consistent, MRCFN)]
- CPM: Conv1D(k=3,s=2,f=32) + MaxPool(k=2,s=2)
- DRRM1: out=64  (2 residual rings; DSConv; MP; Conv)
- DRRM2: out=128
- SCRM: gate threshold=0.5; GroupConv groups=64; sigmoid+softmax gates
- GIPFM: FC1=32, FC2=128, relation vector + channel attention
- CB: GAP + FC -> NUM_CLASSES

[Noise / Normalization Policy]
- Baseline noise environment: 0 dB Gaussian applied to train/val/test (Stage2 baseline)
- Robustness Stage3 (FAIR, SameModel): train ONE NoiseStudy_model once, then evaluate:
  NoNoise + 0 + (-2,-4,-6,-8,-10), report mean±std over repeated noise draws
- IMPORTANT: ENABLE_POST_ZSCORE_AFTER_NOISE MUST remain False to preserve SNR semantics.

[Outputs]
- log.txt (tee stdout/stderr)
- curves.png per run
- Confusion matrix / t-SNE / ROC per test load (2Nm, 4Nm)
- Final_Summary_Stats_raw.csv + Final_Summary_Stats_trimmed.csv
- performance_trend.png
- NoiseStudy_LoadShift_FAIR_SameModel/: CSV + radar plots + baseline plots

[Open-source friendly paths]
- This script does NOT assume any fixed folder structure.
- Replace the placeholders in the PATHS section (YOUR_*) with your local absolute paths.
"""

import os
import gc
import random
import hashlib
import warnings
import traceback
from itertools import cycle
import sys
import time

warnings.filterwarnings("ignore")

# ---------------------------
# Environment (open-source friendly placeholders)
# ---------------------------
YOUR_TEMP_PATH = r"YOUR_TEMP_PATH"  # e.g., r"D:\temp" (replace). Leave as-is to skip auto-setting.
YOUR_CUDA_VISIBLE_DEVICES = "0"     # optional: "0" / "1" / "" (empty = default)
YOUR_TF_CPP_MIN_LOG_LEVEL = "2"     # optional: "2" to suppress INFO/WARN

# TEMP/TMP (optional)
if "YOUR_TEMP_PATH" not in YOUR_TEMP_PATH:
    os.environ["TEMP"] = YOUR_TEMP_PATH
    os.environ["TMP"] = YOUR_TEMP_PATH
    os.makedirs(YOUR_TEMP_PATH, exist_ok=True)

# GPU selection (optional)
if YOUR_CUDA_VISIBLE_DEVICES is not None and str(YOUR_CUDA_VISIBLE_DEVICES) != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(YOUR_CUDA_VISIBLE_DEVICES)

# TensorFlow logging (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(YOUR_TF_CPP_MIN_LOG_LEVEL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (
    Dense, Input, Dropout, Flatten, Conv1D,
    MaxPooling1D, AveragePooling1D,
    Lambda, Concatenate, Activation, LeakyReLU,
    GlobalAveragePooling1D, Add
)
import keras.backend as K
from keras.utils import np_utils

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# TDMS reader
try:
    from nptdms import TdmsFile
except ImportError as e:
    raise ImportError("Missing dependency: nptdms. Please run: pip install nptdms") from e


def _fmt_sec(sec: float) -> str:
    sec = float(sec)
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m{s:05.2f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{s:05.2f}s"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------
# Global plot style (same)
# ---------------------------
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1) Config (benchmark-aligned)
# ============================================================
CUR_MODE = "U"  # "U","V","W","MAG"
DEBUG_PRINT_TDMS_CHANNELS = False

SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

# Paper max iteration=200; keep as default for "A strict"
EPOCHS = 200

DATA_POINTS = 2048
MAX_SEGMENTS_PER_CLASS = 400

TRAIN_LOAD = "0Nm"
TEST_LOADS = ["2Nm", "4Nm"]

FAULTS = [
    ("NC-0", "Normal"),
    ("IF-1", "BPFI_03"),
    ("IF-2", "BPFI_10"),
    ("OF-1", "BPFO_03"),
    ("OF-2", "BPFO_10"),
]
NUM_CLASSES = len(FAULTS)

# -------- Paths (OPEN-SOURCE friendly placeholders; replace with your absolute paths) ----------
VIB_MAT_DIR_0NM = r"YOUR_VIB_MAT_DIR_0NM"   # must contain files like: 0Nm_Normal.mat, 0Nm_BPFO_03.mat, ...
VIB_MAT_DIR_2NM = r"YOUR_VIB_MAT_DIR_2NM"   # must contain files like: 2Nm_Normal.mat, ...
VIB_MAT_DIR_4NM = r"YOUR_VIB_MAT_DIR_4NM"   # must contain files like: 4Nm_Normal.mat, ...
CUR_TDMS_DIR = r"YOUR_CUR_TDMS_DIR"         # must contain files like: 0Nm_Normal.tdms, 2Nm_*.tdms, 4Nm_*.tdms
BASE_OUTPUT_DIR = r"YOUR_OUTPUT_DIR"        # all results will be written here
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

TSNE_MAX_POINTS = 2000
ENABLE_HASH_OVERLAP_CHECK = True
VAL_PER_CLASS_MAIN = 50

# -------- Noise settings (benchmark rule) --------
TRAIN_BASE_SNR_DB = 0  # baseline environment: 0 dB
NO_NOISE_DB = float("inf")
TEST_SNR_DB_LIST = [NO_NOISE_DB, 0, -2, -4, -6, -8, -10]

ENABLE_ZSCORE_PER_SEGMENT = True          # pre z-score at load-time
ENABLE_POST_ZSCORE_AFTER_NOISE = False    # MUST remain False for meaningful SNR sweep
POST_ZSCORE_EPS = 1e-8

# FAIR eval repeat per SNR (benchmark rule)
EVAL_NOISE_REPEATS = 5

# Optional TTA (keep off)
ENABLE_TTA = False
TTA_K = 3

# Training hyperparams (paper Table 3 default)
LR = 1e-3
BATCH_SIZE = 32
GRAD_CLIPNORM = 1.0

# -------- Visualization control --------
CM_CMAP_2NM = "Blues"
CM_CMAP_4NM = "Oranges"
ROC_COLORS_2NM = ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"]
ROC_COLORS_4NM = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]


# ============================================================
# 2) Reproducibility
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# 3) Helpers: hash / normalize (numpy side)
# ============================================================
def _hash_sample(vib_1x2048: np.ndarray, cur_1x2048: np.ndarray) -> str:
    b = vib_1x2048.astype(np.float32).tobytes() + cur_1x2048.astype(np.float32).tobytes()
    return hashlib.sha1(b).hexdigest()


def zscore_per_segment_np(segs: np.ndarray, eps: float = POST_ZSCORE_EPS) -> np.ndarray:
    segs = np.asarray(segs, dtype=np.float32)
    mu = np.mean(segs, axis=1, keepdims=True)
    sd = np.std(segs, axis=1, keepdims=True) + eps
    return (segs - mu) / sd


def add_gaussian_noise_np(x, snr_db, rng: np.random.RandomState):
    """
    x: (N, 2048, 1)
    noise_power = signal_power / 10^(snr/10)
    """
    if np.isinf(snr_db):
        return x
    x2 = x.astype(np.float32)
    sig_power = np.mean(np.square(x2), axis=(1, 2), keepdims=True) + 1e-12
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    noise_power = sig_power / snr_lin
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=x2.shape).astype(np.float32)
    return x2 + noise


def add_noise_then_optional_post_zscore_np(x, snr_db, rng: np.random.RandomState):
    x_n = add_gaussian_noise_np(x, snr_db, rng)
    # default False; keep false for meaningful SNR
    if ENABLE_POST_ZSCORE_AFTER_NOISE and ENABLE_ZSCORE_PER_SEGMENT:
        x_n = zscore_per_segment_np(x_n.squeeze(-1))[:, :, None]
    return x_n


def estimate_empirical_snr_db(x_clean, x_noisy) -> float:
    x_clean = np.asarray(x_clean, dtype=np.float32)
    x_noisy = np.asarray(x_noisy, dtype=np.float32)
    n = x_noisy - x_clean
    ps = float(np.mean(x_clean**2) + 1e-12)
    pn = float(np.mean(n**2) + 1e-12)
    return 10.0 * np.log10(ps / pn)


# ============================================================
# 4) Vibration .mat parsing (KAIST style)
# ============================================================
def _extract_values_matrix(mat_dict: dict) -> np.ndarray:
    if "Signal" not in mat_dict:
        keys = [k for k in mat_dict.keys() if not k.startswith("__")]
        raise KeyError(f"Missing 'Signal'. Available keys: {keys[:10]}")
    sig = mat_dict["Signal"]
    try:
        yv = sig["y_values"][0, 0]
        vals = yv["values"][0, 0]
        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        return vals
    except Exception as e:
        raise RuntimeError(f"Failed to parse 'Signal.y_values.values': {e}")


def _pick_vib_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        values = values.reshape(-1, 1)
    ncol = values.shape[1]
    if ncol >= 5:
        return values[:, 1].astype(np.float32)  # xA
    if ncol == 4:
        return values[:, 0].astype(np.float32)
    return values[:, 0].astype(np.float32)


# ============================================================
# 5) Current TDMS parsing
# ============================================================
def _norm_name(name: str) -> str:
    return (name or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")

def _is_time_channel(name: str) -> bool:
    n = _norm_name(name)
    return ("time" in n) or ("timestamp" in n) or ("timestmp" in n)

def _is_temp_channel(name: str) -> bool:
    n = _norm_name(name)
    return ("temp" in n) or ("temperature" in n) or ("housing" in n)

def _is_phase_u(name: str) -> bool:
    n = _norm_name(name)
    return ("uphase" in n) or (n == "u") or n.endswith("uphase")

def _is_phase_v(name: str) -> bool:
    n = _norm_name(name)
    return ("vphase" in n) or (n == "v") or n.endswith("vphase")

def _is_phase_w(name: str) -> bool:
    n = _norm_name(name)
    return ("wphase" in n) or (n == "w") or n.endswith("wphase")


def read_current_1d_from_tdms(tdms_path: str, mode: str = "U", min_len: int = DATA_POINTS * 2) -> np.ndarray:
    tdms = TdmsFile.read(tdms_path)

    channels = []
    for g in tdms.groups():
        for ch in g.channels():
            name = ch.name or ""
            try:
                x = np.asarray(ch[:], dtype=np.float32).squeeze()
            except Exception:
                continue
            if x.ndim != 1:
                if x.ndim == 2 and x.shape[1] >= 1:
                    x = x[:, 0]
                else:
                    continue
            if x.size < min_len:
                continue
            channels.append((name, x.astype(np.float32)))

    if DEBUG_PRINT_TDMS_CHANNELS:
        print(f"[TDMS] {os.path.basename(tdms_path)} channels:")
        for name, x in channels:
            print(f"  - {name:30s} | len={len(x)} | std={np.std(x):.6f}")

    if len(channels) == 0:
        raise RuntimeError(f"[TDMS ERROR] No valid channels found in: {tdms_path}")

    u = v = w = None
    pool = []
    for name, x in channels:
        if _is_time_channel(name) or _is_temp_channel(name):
            continue
        if _is_phase_u(name):
            u = x
        elif _is_phase_v(name):
            v = x
        elif _is_phase_w(name):
            w = x
        pool.append((name, x))

    def _max_var(arr_list):
        best_x, best_var = None, -1.0
        for nm, xx in arr_list:
            vv = float(np.var(xx))
            if vv > best_var:
                best_var = vv
                best_x = xx
        if best_x is None:
            raise RuntimeError(f"[TDMS ERROR] No non-time/temp channels left in: {tdms_path}")
        return best_x

    mode = (mode or "U").upper()

    if mode == "U":
        return u if u is not None else _max_var(pool)
    if mode == "V":
        return v if v is not None else _max_var(pool)
    if mode == "W":
        return w if w is not None else _max_var(pool)
    if mode == "MAG":
        if (u is None) or (v is None) or (w is None):
            if len(pool) >= 3:
                pool_sorted = sorted(pool, key=lambda t: float(np.var(t[1])), reverse=True)
                u_, v_, w_ = pool_sorted[0][1], pool_sorted[1][1], pool_sorted[2][1]
                L = min(len(u_), len(v_), len(w_))
                return np.sqrt(u_[:L]**2 + v_[:L]**2 + w_[:L]**2).astype(np.float32)
            raise RuntimeError(f"[TDMS ERROR] MAG mode needs 3 channels, but not found in: {tdms_path}")
        L = min(len(u), len(v), len(w))
        return np.sqrt(u[:L]**2 + v[:L]**2 + w[:L]**2).astype(np.float32)

    raise ValueError(f"Unknown CUR_MODE={mode}. Use 'U','V','W','MAG'.")


# ============================================================
# 6) Segmentation
# ============================================================
def segment_2048(x_1d: np.ndarray, max_segments: int) -> np.ndarray:
    x_1d = np.asarray(x_1d, dtype=np.float32).flatten()
    n_total = len(x_1d) // DATA_POINTS
    n_use = min(n_total, max_segments)
    if n_use <= 0:
        raise ValueError(f"Signal too short for segmentation: len={len(x_1d)}")
    segs = np.zeros((n_use, DATA_POINTS), dtype=np.float32)
    for i in range(n_use):
        segs[i, :] = x_1d[i * DATA_POINTS:(i + 1) * DATA_POINTS]
    return segs


# ============================================================
# 7) Load datasets: vib(.mat) + current(.tdms)
# ============================================================
def _mat_path(vib_dir: str, load_tag: str, fault: str) -> str:
    p = os.path.join(vib_dir, f"{load_tag}_{fault}.mat")
    if not os.path.exists(p):
        raise FileNotFoundError(f"[VIB MAT MISSING] {p}")
    return p

def _tdms_path(cur_dir: str, load_tag: str, fault: str) -> str:
    for ext in [".tdms", ".TDMS"]:
        p = os.path.join(cur_dir, f"{load_tag}_{fault}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"[CUR TDMS MISSING] Tried {load_tag}_{fault}.tdms/.TDMS in {cur_dir}")

def load_dataset_for_load(load_tag: str, vib_dir: str):
    datasets = []
    for cls_name, fault in FAULTS:
        vib_file = _mat_path(vib_dir, load_tag, fault)
        cur_file = _tdms_path(CUR_TDMS_DIR, load_tag, fault)

        vib_mat = loadmat(vib_file)
        vib_vals = _extract_values_matrix(vib_mat)
        vib_1d = _pick_vib_1d(vib_vals)

        cur_1d = read_current_1d_from_tdms(cur_file, mode=CUR_MODE)

        vib_segs = segment_2048(vib_1d, MAX_SEGMENTS_PER_CLASS)
        cur_segs = segment_2048(cur_1d, MAX_SEGMENTS_PER_CLASS)

        n_use = min(len(vib_segs), len(cur_segs))
        vib_segs = vib_segs[:n_use]
        cur_segs = cur_segs[:n_use]

        if np.std(vib_segs) < 1e-8:
            raise RuntimeError(f"[DATA ERROR] vib near-constant: {vib_file}")
        if np.std(cur_segs) < 1e-8:
            raise RuntimeError(f"[DATA ERROR] current near-constant: {cur_file}")

        # pre z-score
        if ENABLE_ZSCORE_PER_SEGMENT:
            vib_segs = zscore_per_segment_np(vib_segs)
            cur_segs = zscore_per_segment_np(cur_segs)

        datasets.append((vib_segs, cur_segs))
    return datasets

def print_dataset_stats(load_tag: str, datasets):
    print(f"Load {load_tag}:")
    for i, (cls_name, fault) in enumerate(FAULTS):
        vib_segs, cur_segs = datasets[i]
        print(f"  Class {cls_name:<4} | {load_tag}_{fault:<8} : {vib_segs.shape[0]} segments")


print("Loading Vibration(.mat)+Current(.tdms) ...")
DATA_0 = load_dataset_for_load("0Nm", VIB_MAT_DIR_0NM)
DATA_2 = load_dataset_for_load("2Nm", VIB_MAT_DIR_2NM)
DATA_4 = load_dataset_for_load("4Nm", VIB_MAT_DIR_4NM)
print_dataset_stats("0Nm", DATA_0)
print_dataset_stats("2Nm", DATA_2)
print_dataset_stats("4Nm", DATA_4)
print("Data loaded.\n")


# ============================================================
# 8) Split: 0Nm train/val; 2Nm/4Nm test
# ============================================================
def build_train_val_from_0Nm(datasets_0, seed: int, n_train: int, n_val: int):
    rng = np.random.RandomState(seed)
    tr_list, va_list = [], []
    tr_hash, va_hash = set(), set()

    for label, (vib, cur) in enumerate(datasets_0):
        M = vib.shape[0]
        need = n_train + n_val + 1
        assert M >= need, f"[DATA ERROR] class {label} has {M} segments, need >= {need}"

        combined = np.hstack([vib, cur])  # (M, 4096)
        rng.shuffle(combined)

        tr = combined[:n_train]
        va = combined[n_train:n_train + n_val]

        y_tr = np.full((len(tr), 1), label)
        y_va = np.full((len(va), 1), label)

        if ENABLE_HASH_OVERLAP_CHECK:
            for i in range(len(tr)):
                tr_hash.add(_hash_sample(tr[i, :2048], tr[i, 2048:]))
            for i in range(len(va)):
                va_hash.add(_hash_sample(va[i, :2048], va[i, 2048:]))

        tr_list.append(np.hstack([tr, y_tr]))
        va_list.append(np.hstack([va, y_va]))

    if ENABLE_HASH_OVERLAP_CHECK:
        assert len(tr_hash.intersection(va_hash)) == 0, "[LEAKAGE ERROR] 0Nm train/val overlap!"

    tr_all = np.vstack(tr_list)
    va_all = np.vstack(va_list)
    rng.shuffle(tr_all)
    rng.shuffle(va_all)

    def pack(arr):
        xv = arr[:, 0:2048][:, :, np.newaxis].astype(np.float32)
        xc = arr[:, 2048:4096][:, :, np.newaxis].astype(np.float32)
        y = np_utils.to_categorical(arr[:, 4096], NUM_CLASSES).astype(np.float32)
        return xv, xc, y

    return pack(tr_all), pack(va_all)

def build_test_from_load(datasets_load, seed: int = 0):
    rng = np.random.RandomState(seed)
    te_list = []
    for label, (vib, cur) in enumerate(datasets_load):
        combined = np.hstack([vib, cur])
        y = np.full((len(combined), 1), label)
        te_list.append(np.hstack([combined, y]))
    te_all = np.vstack(te_list)
    rng.shuffle(te_all)

    xv = te_all[:, 0:2048][:, :, np.newaxis].astype(np.float32)
    xc = te_all[:, 2048:4096][:, :, np.newaxis].astype(np.float32)
    y = np_utils.to_categorical(te_all[:, 4096], NUM_CLASSES).astype(np.float32)
    return xv, xc, y


# ============================================================
# 9) MRCFN Network (paper-consistent modules)
# ============================================================
class GroupNorm1D(keras.layers.Layer):
    """Simple GroupNorm for (B,L,C).

    FIX (A strict): In SCRM you use hard-threshold mask (cast(gate>=0.5)), which is non-differentiable.
    That makes GN gamma/beta gradients not exist -> TF warning.
    So we freeze gamma/beta (trainable=False) to match your strict-hard-mask design and silence warnings.
    """
    def __init__(self, groups=32, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.eps = float(eps)

    def build(self, input_shape):
        C = int(input_shape[-1])
        G = min(self.groups, C)
        if C % G != 0:
            # fallback to LayerNorm-like behavior
            G = 1
        self.groups = G

        # ===== KEY FIX: freeze gamma/beta to avoid "no gradients" warning =====
        self.gamma = self.add_weight(name="gamma", shape=(1, 1, C), initializer="ones", trainable=False)
        self.beta = self.add_weight(name="beta", shape=(1, 1, C), initializer="zeros", trainable=False)

        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        shape = tf.shape(x)
        B, L = shape[0], shape[1]
        C = x.shape[-1]
        G = self.groups
        x_ = tf.reshape(x, [B, L, G, C // G])
        mean, var = tf.nn.moments(x_, axes=[1, 3], keepdims=True)
        x_ = (x_ - mean) / tf.sqrt(var + self.eps)
        x_ = tf.reshape(x_, [B, L, C])
        return x_ * self.gamma + self.beta


def CPM_block(x):
    # Paper Table2: Conv1D(k=3,s=2,f=32) + MaxPool(k=2,s=2)
    x = Conv1D(32, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same")(x)
    return x


def DRRM_block(x, out_ch: int, name_prefix="DRRM"):
    """
    Deep Residual Reconstruction Module (paper Sec 2.2.2):
    - Two residual rings
    - Use DSConv (SeparableConv1D) + MP + Conv
    - Shortcut projection on MP path to match channels (necessary for valid Add)
    """
    # DSConv1 (stride=2) -> X1
    x1 = keras.layers.SeparableConv1D(out_ch, kernel_size=3, strides=2, padding="same", use_bias=False,
                                     name=f"{name_prefix}_ds1")(x)
    x1 = keras.layers.PReLU(shared_axes=[1], name=f"{name_prefix}_prelu1")(x1)

    # DSConv2 (stride=1) -> X2
    x2 = keras.layers.SeparableConv1D(out_ch, kernel_size=3, strides=1, padding="same", use_bias=False,
                                     name=f"{name_prefix}_ds2")(x1)
    x2 = keras.layers.PReLU(shared_axes=[1], name=f"{name_prefix}_prelu2")(x2)

    # MP(X0) + projection to out_ch -> X_MP
    x_mp = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{name_prefix}_mp")(x)
    x_mp = Conv1D(out_ch, kernel_size=1, strides=1, padding="same", use_bias=False, name=f"{name_prefix}_proj")(x_mp)

    # Residual ring 1: X_res1 = X2 + X_MP
    x_res1 = Add(name=f"{name_prefix}_add1")([x2, x_mp])
    x_res1 = keras.layers.PReLU(shared_axes=[1], name=f"{name_prefix}_prelu_res1")(x_res1)

    # Residual ring 2:
    # X3 = Conv(X1, k=3,s=2), X4 = Conv(X_res1,k=3,s=2), X_res2 = X3 + X4
    x3 = Conv1D(out_ch, kernel_size=3, strides=2, padding="same", use_bias=False, name=f"{name_prefix}_conv_x1")(x1)
    x3 = keras.layers.PReLU(shared_axes=[1], name=f"{name_prefix}_prelu3")(x3)

    x4 = Conv1D(out_ch, kernel_size=3, strides=2, padding="same", use_bias=False, name=f"{name_prefix}_conv_res1")(x_res1)
    x4 = keras.layers.PReLU(shared_axes=[1], name=f"{name_prefix}_prelu4")(x4)

    x_res2 = Add(name=f"{name_prefix}_add2")([x3, x4])
    x_res2 = keras.layers.PReLU(shared_axes=[1], name=f"{name_prefix}_prelu_res2")(x_res2)

    return x_res2


def SCRM_block(x, gate_th=0.5, groups=64, name_prefix="SCRM"):
    """
    Spatial-Channel Reconstruction Module (paper Sec 2.2.3):
    Implement SRU + CRU style:
    - SRU: GN -> sigmoid gate -> threshold(0.5) -> swap/reconstruct
    - CRU: split -> 1x1 conv expand -> group conv (groups=64) -> softmax channel weights -> weighted sum
    """
    C = int(x.shape[-1])
    assert C % 2 == 0, "SCRM expects even channels."

    # ---- SRU (spatial reconstruction) ----
    x1, x2 = Lambda(lambda t: tf.split(t, 2, axis=-1), name=f"{name_prefix}_split_sru")(x)

    gn = GroupNorm1D(groups=groups, name=f"{name_prefix}_gn")(x1)
    gate = Activation("sigmoid", name=f"{name_prefix}_sigmoid_gate")(gn)
    mask = Lambda(lambda t: tf.cast(t >= gate_th, tf.float32), name=f"{name_prefix}_th_mask")(gate)
    inv = Lambda(lambda t: 1.0 - t, name=f"{name_prefix}_inv_mask")(mask)

    y1 = Add(name=f"{name_prefix}_sru_y1")([x1 * mask, x2 * inv])
    y2 = Add(name=f"{name_prefix}_sru_y2")([x2 * mask, x1 * inv])
    y_sr = Concatenate(axis=-1, name=f"{name_prefix}_concat_sru")([y1, y2])

    # ---- CRU (channel reconstruction) ----
    y_u, y_l = Lambda(lambda t: tf.split(t, 2, axis=-1), name=f"{name_prefix}_split_cru")(y_sr)

    # expand both to 128 channels (paper Table2: output channels=128; group size=64)
    y1_hat = Conv1D(128, kernel_size=1, strides=1, padding="same", use_bias=False, name=f"{name_prefix}_y1_1x1")(y_u)
    y1_hat = Activation("relu", name=f"{name_prefix}_y1_relu")(y1_hat)

    # group conv (kernel=3). To keep length unchanged, use stride=1.
    y1_g = Conv1D(128, kernel_size=3, strides=1, padding="same", groups=64, dilation_rate=2,
                 use_bias=False, name=f"{name_prefix}_gconv")(y1_hat)
    y1_g = Activation("relu", name=f"{name_prefix}_gconv_relu")(y1_g)

    y1_p = Conv1D(128, kernel_size=1, strides=1, padding="same", use_bias=False, name=f"{name_prefix}_y1_pconv")(y1_g)

    y2_hat = Conv1D(128, kernel_size=1, strides=1, padding="same", use_bias=False, name=f"{name_prefix}_y2_1x1")(y_l)
    y2_hat = Activation("relu", name=f"{name_prefix}_y2_relu")(y2_hat)

    # channel importance (softmax)
    gap1 = GlobalAveragePooling1D(name=f"{name_prefix}_gap1")(y1_p)  # (B,128)
    gap2 = GlobalAveragePooling1D(name=f"{name_prefix}_gap2")(y2_hat)

    g = Concatenate(name=f"{name_prefix}_g_concat")([gap1, gap2])    # (B,256)
    g = Activation("softmax", name=f"{name_prefix}_softmax_g")(g)

    g1 = Lambda(lambda t: t[:, :128], name=f"{name_prefix}_g1")(g)
    g2 = Lambda(lambda t: t[:, 128:], name=f"{name_prefix}_g2")(g)

    g1e = Lambda(lambda t: tf.expand_dims(t, 1), name=f"{name_prefix}_g1e")(g1)  # (B,1,128)
    g2e = Lambda(lambda t: tf.expand_dims(t, 1), name=f"{name_prefix}_g2e")(g2)

    y_cr = Add(name=f"{name_prefix}_out")([y1_p * g1e, y2_hat * g2e])  # (B,L,128)
    return y_cr


def channel_att_vector(s, name_prefix="CA"):
    # Paper GIPFM: GAP -> FC1(32, ReLU) -> FC2(128)
    v = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(s)
    v = Dense(32, activation="relu", name=f"{name_prefix}_fc1")(v)
    v = Dense(128, activation=None, name=f"{name_prefix}_fc2")(v)
    return v


def GIPFM_fusion(s1, s2, name_prefix="GIPFM"):
    """
    Paper Sec 2.3:
    - v1, v2 from channel attention MLP
    - relationship vector vr from (v1+v2) through same MLP shape
    - weights: sigmoid(vr + vi)
    - S_fuse = S1*W1 + S2*W2
    - PConv(1x1) after fusion
    """
    v1 = channel_att_vector(s1, name_prefix=f"{name_prefix}_v1")
    v2 = channel_att_vector(s2, name_prefix=f"{name_prefix}_v2")

    vr = Add(name=f"{name_prefix}_vsum")([v1, v2])
    vr = Dense(32, activation="relu", name=f"{name_prefix}_vr_fc1")(vr)
    vr = Dense(128, activation=None, name=f"{name_prefix}_vr_fc2")(vr)

    w1 = Activation("sigmoid", name=f"{name_prefix}_w1_sigmoid")(Add()([vr, v1]))
    w2 = Activation("sigmoid", name=f"{name_prefix}_w2_sigmoid")(Add()([vr, v2]))

    w1e = Lambda(lambda t: tf.expand_dims(t, 1), name=f"{name_prefix}_w1e")(w1)
    w2e = Lambda(lambda t: tf.expand_dims(t, 1), name=f"{name_prefix}_w2e")(w2)

    s1w = keras.layers.Multiply(name=f"{name_prefix}_s1w")([s1, w1e])
    s2w = keras.layers.Multiply(name=f"{name_prefix}_s2w")([s2, w2e])

    s_fuse = Add(name=f"{name_prefix}_sum")([s1w, s2w])

    # PCConv (1x1)
    f = Conv1D(128, kernel_size=1, strides=1, padding="same", use_bias=False, name=f"{name_prefix}_pconv")(s_fuse)
    f = Activation("relu", name=f"{name_prefix}_pconv_relu")(f)
    return f


def sensor_subnet(x_in, name_prefix="Sensor"):
    x = CPM_block(x_in)
    x = DRRM_block(x, out_ch=64, name_prefix=f"{name_prefix}_DRRM1")
    x = DRRM_block(x, out_ch=128, name_prefix=f"{name_prefix}_DRRM2")
    x = SCRM_block(x, gate_th=0.5, groups=64, name_prefix=f"{name_prefix}_SCRM")
    return x


def create_mrcfn():
    in_v = Input(shape=(DATA_POINTS, 1), name="Input_Vib")
    in_c = Input(shape=(DATA_POINTS, 1), name="Input_Cur")

    f1 = sensor_subnet(in_v, name_prefix="Vib")
    f2 = sensor_subnet(in_c, name_prefix="Cur")

    f_fuse = GIPFM_fusion(f1, f2, name_prefix="GIPFM")

    feat_vec = GlobalAveragePooling1D(name="CB_GAP")(f_fuse)  # 128-d
    out = Dense(NUM_CLASSES, activation="softmax", name="CB_FC")(feat_vec)

    model = Model(inputs=[in_v, in_c], outputs=out, name="MRCFN")
    feat_model = Model(inputs=[in_v, in_c], outputs=feat_vec, name="MRCFN_feat")
    return model, feat_model


# ============================================================
# 10) Visualization (same as benchmark)
# ============================================================
def save_curves(history, out_path, title_prefix, val_name="Val"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label=val_name)
    plt.title(f"Loss - {title_prefix}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label=val_name)
    plt.title(f"Accuracy - {title_prefix}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_cm(y_true, y_pred, out_path, title_prefix, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True)
    plt.title(f"Confusion Matrix - {title_prefix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_tsne(feat_model, x_te_v, x_te_c, y_true, out_path, title_prefix, seed=42):
    feats = feat_model.predict([x_te_v, x_te_c], verbose=0)
    n = len(feats)
    if n > TSNE_MAX_POINTS:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=TSNE_MAX_POINTS, replace=False)
        feats = feats[idx]
        y_vis = y_true[idx]
    else:
        y_vis = y_true

    tsne = TSNE(n_components=2, random_state=seed).fit_transform(feats)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_vis, palette="tab10",
                    legend="full", s=15)
    plt.title(f"t-SNE - {title_prefix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_roc(y_true, y_prob, out_path, title_prefix, class_colors=None):
    y_test_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg (AUC={roc_auc['micro']:.4f})",
             linestyle=":", lw=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg (AUC={roc_auc['macro']:.4f})",
             linestyle=":", lw=3)

    if class_colors is None:
        class_colors = ROC_COLORS_2NM

    colors = cycle(class_colors)
    for i, c in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=c, lw=2, label=f"Class {i} (AUC={roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title_prefix}")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 11) Trimmed stats + radar + trend
# ============================================================
def trimmed_mean_std(values):
    v = sorted(list(values))
    if len(v) > 2:
        v = v[1:-1]
    return float(np.mean(v)), float(np.std(v))

def plot_radar(snr_or_labels, acc_list, f1_list, out_path, title):
    def _lab(s):
        if isinstance(s, str):
            return s
        if np.isinf(s):
            return "NoNoise"
        return str(int(s))

    labels = [_lab(s) for s in snr_or_labels]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    acc = list(acc_list) + [acc_list[0]]
    f1 = list(f1_list) + [f1_list[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.plot(angles, acc, linewidth=2, label="Accuracy")
    ax.fill(angles, acc, alpha=0.10)
    ax.plot(angles, f1, linewidth=2, label="Macro-F1")
    ax.fill(angles, f1, alpha=0.10)

    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, y=1.08)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_performance_trend(trim_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(9, 5))
    x = trim_df["N"].values

    plt.plot(x, trim_df["Acc2_Mean"].values, label="Acc@2Nm")
    plt.plot(x, trim_df["F12_Mean"].values, label="F1@2Nm")
    plt.plot(x, trim_df["Acc4_Mean"].values, label="Acc@4Nm")
    plt.plot(x, trim_df["F14_Mean"].values, label="F1@4Nm")

    plt.xlabel("N (train samples per class, 0Nm)")
    plt.ylabel("Metric")
    plt.title("Performance Trend (Trimmed Mean)")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 12) Evaluate helper (benchmark style)
# ============================================================
def eval_on_test(model, feat_model, x_te_v, x_te_c, y_te,
                out_dir, tag, snr_db, seed, tta_k=1,
                roc_colors=None, cm_cmap="Blues"):
    """
    Evaluate with optional TTA:
    - snr_db=inf: no noise
    - else: add noise (and optional post-zscore depending on flag)
    """
    y_true = np.argmax(y_te, axis=1)

    use_tta = (tta_k is not None and int(tta_k) > 1 and (snr_db is not None)
               and (not (np.isscalar(snr_db) and np.isinf(snr_db))))

    t_inf0 = time.perf_counter()

    if not use_tta:
        rng = np.random.RandomState(seed + 999)
        x_tv = add_noise_then_optional_post_zscore_np(x_te_v, snr_db, rng)
        x_tc = add_noise_then_optional_post_zscore_np(x_te_c, snr_db, rng)
        y_prob = model.predict([x_tv, x_tc], verbose=0)
    else:
        Kk = int(tta_k)
        prob_sum = None
        for kk in range(Kk):
            rng = np.random.RandomState(seed + 999 + kk * 131)
            x_tv = add_noise_then_optional_post_zscore_np(x_te_v, snr_db, rng)
            x_tc = add_noise_then_optional_post_zscore_np(x_te_c, snr_db, rng)
            p = model.predict([x_tv, x_tc], verbose=0)
            prob_sum = p.astype(np.float64) if prob_sum is None else (prob_sum + p.astype(np.float64))
        y_prob = (prob_sum / float(Kk)).astype(np.float32)

    y_pred = np.argmax(y_prob, axis=1)
    t_inf1 = time.perf_counter()
    infer_time_s = float(t_inf1 - t_inf0)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")

    viz_time_s = 0.0
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        t_viz0 = time.perf_counter()

        title_snr = "NoNoise" if (snr_db is None or (np.isscalar(snr_db) and np.isinf(snr_db))) else f"{snr_db}dB"
        save_cm(y_true, y_pred, os.path.join(out_dir, f"cm_{tag}.png"),
                f"{tag} | SNR={title_snr}", cmap=cm_cmap)
        save_tsne(feat_model, x_tv, x_tc, y_true, os.path.join(out_dir, f"tsne_{tag}.png"),
                  f"{tag} | SNR={title_snr}")
        save_roc(y_true, y_prob, os.path.join(out_dir, f"roc_{tag}.png"),
                 f"{tag} | SNR={title_snr}", class_colors=roc_colors)

        t_viz1 = time.perf_counter()
        viz_time_s = float(t_viz1 - t_viz0)

    return acc, f1, prec, rec, infer_time_s, viz_time_s


# ============================================================
# 13) Full experiment (Stage2)
# ============================================================
def run_full_experiment():
    x2_v, x2_c, y2 = build_test_from_load(DATA_2, seed=123)
    x4_v, x4_c, y4 = build_test_from_load(DATA_4, seed=456)

    print("\n>>> MRCFN Model Architecture Summary:")
    print("=" * 60)
    tmp_model, _ = create_mrcfn()
    tmp_model.summary()
    del tmp_model
    print("=" * 60 + "\n")

    raw_rows = []
    trim_rows = []

    for n_train in SAMPLE_RANGE:
        sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{n_train:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n======== Train(0Nm) N={n_train} (Run 1-{REPEAT_TIMES}) ========")

        buf = {
            "acc_2": [], "f1_2": [], "prec_2": [], "rec_2": [],
            "acc_4": [], "f1_4": [], "prec_4": [], "rec_4": [],
            "train_s": [],
            "t2_inf_s": [], "t2_viz_s": [],
            "t4_inf_s": [], "t4_viz_s": [],
        }

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            seed = n_train * 100 + run_idx
            set_global_seed(seed)

            n_val = min(VAL_PER_CLASS_MAIN, MAX_SEGMENTS_PER_CLASS - n_train - 1)
            (x_tr_v, x_tr_c, y_tr), (x_va_v, x_va_c, y_va) = build_train_val_from_0Nm(
                DATA_0, seed=seed, n_train=n_train, n_val=n_val
            )

            # add baseline noise (0dB) once per run (benchmark env)
            rng_tr = np.random.RandomState(seed + 111)
            rng_va = np.random.RandomState(seed + 222)
            x_tr_vn = add_noise_then_optional_post_zscore_np(x_tr_v, TRAIN_BASE_SNR_DB, rng_tr)
            x_tr_cn = add_noise_then_optional_post_zscore_np(x_tr_c, TRAIN_BASE_SNR_DB, rng_tr)
            x_va_vn = add_noise_then_optional_post_zscore_np(x_va_v, TRAIN_BASE_SNR_DB, rng_va)
            x_va_cn = add_noise_then_optional_post_zscore_np(x_va_c, TRAIN_BASE_SNR_DB, rng_va)

            tf.keras.backend.clear_session()
            gc.collect()

            model, feat_model = create_mrcfn()
            opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=GRAD_CLIPNORM)
            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

            t_fit0 = time.perf_counter()
            history = model.fit(
                [x_tr_vn, x_tr_cn], y_tr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=([x_va_vn, x_va_cn], y_va),
                verbose=0
            )
            t_fit1 = time.perf_counter()
            train_time_s = float(t_fit1 - t_fit0)

            save_curves(history, os.path.join(run_dir, "curves.png"),
                        f"Train0Nm N={n_train} Run={run_idx}", val_name="Val(0Nm)")

            out2 = os.path.join(run_dir, "Test_2Nm")
            out4 = os.path.join(run_dir, "Test_4Nm")

            acc2, f12, p2, r2, t2_inf_s, t2_viz_s = eval_on_test(
                model, feat_model, x2_v, x2_c, y2, out2, "2Nm", TRAIN_BASE_SNR_DB, seed,
                tta_k=(TTA_K if ENABLE_TTA else 1), roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            acc4, f14, p4, r4, t4_inf_s, t4_viz_s = eval_on_test(
                model, feat_model, x4_v, x4_c, y4, out4, "4Nm", TRAIN_BASE_SNR_DB, seed,
                tta_k=(TTA_K if ENABLE_TTA else 1), roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
            )

            buf["acc_2"].append(acc2); buf["f1_2"].append(f12); buf["prec_2"].append(p2); buf["rec_2"].append(r2)
            buf["acc_4"].append(acc4); buf["f1_4"].append(f14); buf["prec_4"].append(p4); buf["rec_4"].append(r4)

            buf["train_s"].append(train_time_s)
            buf["t2_inf_s"].append(t2_inf_s); buf["t2_viz_s"].append(t2_viz_s)
            buf["t4_inf_s"].append(t4_inf_s); buf["t4_viz_s"].append(t4_viz_s)

            raw_rows.append([
                n_train, run_idx,
                acc2, f12, p2, r2,
                acc4, f14, p4, r4,
                train_time_s, t2_inf_s, t2_viz_s, t4_inf_s, t4_viz_s
            ])

            print(
                f"  [N{n_train}-R{run_idx}] "
                f"Train={_fmt_sec(train_time_s)} | "
                f"2Nm Acc={acc2:.4f} F1={f12:.4f} | "
                f"4Nm Acc={acc4:.4f} F1={f14:.4f}"
            )

            del model, feat_model
            tf.keras.backend.clear_session()
            gc.collect()

        acc2_m, acc2_s = trimmed_mean_std(buf["acc_2"])
        f12_m, f12_s = trimmed_mean_std(buf["f1_2"])
        p2_m, p2_s = trimmed_mean_std(buf["prec_2"])
        r2_m, r2_s = trimmed_mean_std(buf["rec_2"])

        acc4_m, acc4_s = trimmed_mean_std(buf["acc_4"])
        f14_m, f14_s = trimmed_mean_std(buf["f1_4"])
        p4_m, p4_s = trimmed_mean_std(buf["prec_4"])
        r4_m, r4_s = trimmed_mean_std(buf["rec_4"])

        tr_m, tr_s = trimmed_mean_std(buf["train_s"])
        t2i_m, t2i_s = trimmed_mean_std(buf["t2_inf_s"])
        t2v_m, t2v_s = trimmed_mean_std(buf["t2_viz_s"])
        t4i_m, t4i_s = trimmed_mean_std(buf["t4_inf_s"])
        t4v_m, t4v_s = trimmed_mean_std(buf["t4_viz_s"])

        trim_rows.append([
            n_train,
            acc2_m, acc2_s, f12_m, f12_s, p2_m, p2_s, r2_m, r2_s,
            acc4_m, acc4_s, f14_m, f14_s, p4_m, p4_s, r4_m, r4_s,
            tr_m, tr_s,
            t2i_m, t2i_s, t2v_m, t2v_s,
            t4i_m, t4i_s, t4v_m, t4v_s
        ])

        print(f"  >>> Trimmed N={n_train}: 2Nm Acc={acc2_m:.4f}±{acc2_s:.4f} F1={f12_m:.4f}±{f12_s:.4f} | "
              f"4Nm Acc={acc4_m:.4f}±{acc4_s:.4f} F1={f14_m:.4f}±{f14_s:.4f}")

    raw_df = pd.DataFrame(
        raw_rows,
        columns=[
            "N", "Run",
            "Acc_2Nm", "F1_2Nm", "Prec_2Nm", "Recall_2Nm",
            "Acc_4Nm", "F1_4Nm", "Prec_4Nm", "Recall_4Nm",
            "TrainTime_s",
            "Test2_Infer_s", "Test2_Viz_s",
            "Test4_Infer_s", "Test4_Viz_s",
        ]
    )
    trim_df = pd.DataFrame(
        trim_rows,
        columns=[
            "N",
            "Acc2_Mean", "Acc2_Std", "F12_Mean", "F12_Std", "Prec2_Mean", "Prec2_Std", "Rec2_Mean", "Rec2_Std",
            "Acc4_Mean", "Acc4_Std", "F14_Mean", "F14_Std", "Prec4_Mean", "Prec4_Std", "Rec4_Mean", "Rec4_Std",
            "TrainTime_Mean_s", "TrainTime_Std_s",
            "Test2_Infer_Mean_s", "Test2_Infer_Std_s",
            "Test2_Viz_Mean_s", "Test2_Viz_Std_s",
            "Test4_Infer_Mean_s", "Test4_Infer_Std_s",
            "Test4_Viz_Mean_s", "Test4_Viz_Std_s",
        ]
    )

    raw_csv = os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_raw.csv")
    trim_csv = os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_trimmed.csv")
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    trim_df.to_csv(trim_csv, index=False, encoding="utf-8-sig")

    trend_png = os.path.join(BASE_OUTPUT_DIR, "performance_trend.png")
    plot_performance_trend(trim_df, trend_png)

    print("[Stage2] Saved:", raw_csv)
    print("[Stage2] Saved:", trim_csv)
    print("[Stage2] Saved:", trend_png)


# ============================================================
# 14) Noise robustness study (Stage3) - FAIR VERSION (SameModel)
# ============================================================
def run_noise_study():
    """
    FAIR robustness (SameModel):
    - Train ONE model (N=30, fixed seed) under baseline env (0 dB)
    - Evaluate across SNR points:
        NoNoise + 0 + (-2..-10 step=2)
      For each noisy SNR point: repeat EVAL_NOISE_REPEATS draws, report mean±std.
    """
    noise_dir = os.path.join(BASE_OUTPUT_DIR, "NoiseStudy_LoadShift_FAIR_SameModel")
    os.makedirs(noise_dir, exist_ok=True)

    x2_v, x2_c, y2 = build_test_from_load(DATA_2, seed=123)
    x4_v, x4_c, y4 = build_test_from_load(DATA_4, seed=456)

    # Train NoiseStudy model ONCE (N=30, fixed seed)
    seed = 2026
    set_global_seed(seed)

    n_train = 30
    n_val = min(VAL_PER_CLASS_MAIN, MAX_SEGMENTS_PER_CLASS - n_train - 1)
    (x_tr_v, x_tr_c, y_tr), (x_va_v, x_va_c, y_va) = build_train_val_from_0Nm(
        DATA_0, seed=seed, n_train=n_train, n_val=n_val
    )

    # baseline 0dB noise for training/val
    rng_tr = np.random.RandomState(seed + 111)
    rng_va = np.random.RandomState(seed + 222)
    x_tr_vn = add_noise_then_optional_post_zscore_np(x_tr_v, TRAIN_BASE_SNR_DB, rng_tr)
    x_tr_cn = add_noise_then_optional_post_zscore_np(x_tr_c, TRAIN_BASE_SNR_DB, rng_tr)
    x_va_vn = add_noise_then_optional_post_zscore_np(x_va_v, TRAIN_BASE_SNR_DB, rng_va)
    x_va_cn = add_noise_then_optional_post_zscore_np(x_va_c, TRAIN_BASE_SNR_DB, rng_va)

    tf.keras.backend.clear_session()
    gc.collect()

    model, feat_model = create_mrcfn()
    opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=GRAD_CLIPNORM)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    t_fit0 = time.perf_counter()
    _ = model.fit(
        [x_tr_vn, x_tr_cn], y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([x_va_vn, x_va_cn], y_va),
        verbose=0
    )
    t_fit1 = time.perf_counter()
    print(f"[Stage3-FAIR] NoiseStudy_model trained ONCE. time={_fmt_sec(t_fit1 - t_fit0)}")

    unified_tta_k = (int(TTA_K) if ENABLE_TTA else 1)

    rows = []
    acc2_mean_list, f12_mean_list, acc4_mean_list, f14_mean_list = [], [], [], []

    for snr_db in TEST_SNR_DB_LIST:
        label = "NoNoise" if (np.isscalar(snr_db) and np.isinf(snr_db)) else str(int(snr_db))
        tta_k = 1 if (np.isscalar(snr_db) and np.isinf(snr_db)) else unified_tta_k

        acc2_buf, f12_buf = [], []
        acc4_buf, f14_buf = [], []

        reps = 1 if (np.isscalar(snr_db) and np.isinf(snr_db)) else EVAL_NOISE_REPEATS
        for rep in range(reps):
            rep_seed2 = seed + 1000 + rep * 17
            rep_seed4 = seed + 2000 + rep * 17

            a2, f2, _, _, _, _ = eval_on_test(
                model, feat_model, x2_v, x2_c, y2,
                out_dir=None, tag="2Nm", snr_db=snr_db, seed=rep_seed2,
                tta_k=1, roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            a4, f4, _, _, _, _ = eval_on_test(
                model, feat_model, x4_v, x4_c, y4,
                out_dir=None, tag="4Nm", snr_db=snr_db, seed=rep_seed4,
                tta_k=1, roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
            )

            acc2_buf.append(a2); f12_buf.append(f2)
            acc4_buf.append(a4); f14_buf.append(f4)

        acc2_m, acc2_s = float(np.mean(acc2_buf)), float(np.std(acc2_buf))
        f12_m, f12_s = float(np.mean(f12_buf)), float(np.std(f12_buf))
        acc4_m, acc4_s = float(np.mean(acc4_buf)), float(np.std(acc4_buf))
        f14_m, f14_s = float(np.mean(f14_buf)), float(np.std(f14_buf))

        rows.append([
            label, (np.nan if np.isinf(snr_db) else float(snr_db)),
            acc2_m, acc2_s, f12_m, f12_s,
            acc4_m, acc4_s, f14_m, f14_s,
            tta_k, reps
        ])

        acc2_mean_list.append(acc2_m); f12_mean_list.append(f12_m)
        acc4_mean_list.append(acc4_m); f14_mean_list.append(f14_m)

        print(f"[Stage3-FAIR] SNR={label} | 2Nm Acc={acc2_m:.4f}±{acc2_s:.4f} F1={f12_m:.4f}±{f12_s:.4f} | "
              f"4Nm Acc={acc4_m:.4f}±{acc4_s:.4f} F1={f14_m:.4f}±{f14_s:.4f}")

        # optional sanity check: empirical SNR
        if not (np.isscalar(snr_db) and np.isinf(snr_db)):
            rng = np.random.RandomState(seed + 555)
            subset = slice(0, min(256, x2_v.shape[0]))
            x_clean = x2_v[subset]
            x_noisy = add_noise_then_optional_post_zscore_np(x_clean, snr_db, rng)
            emp = estimate_empirical_snr_db(x_clean, x_noisy)
            print(f"    [Sanity] empirical SNR≈{emp:.2f} dB (vib subset, after your pipeline)")

    out_csv = os.path.join(noise_dir, "Noise_Robustness_LoadShift_SameModel_FAIR.csv")
    pd.DataFrame(rows, columns=[
        "label", "SNR_dB",
        "Acc_2Nm_mean", "Acc_2Nm_std", "F1_2Nm_mean", "F1_2Nm_std",
        "Acc_4Nm_mean", "Acc_4Nm_std", "F1_4Nm_mean", "F1_4Nm_std",
        "tta_k_reported", "eval_repeats"
    ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[Stage3-FAIR] Saved:", out_csv)

    plot_radar(TEST_SNR_DB_LIST, acc2_mean_list, f12_mean_list, os.path.join(noise_dir, "radar_2Nm.png"),
               title="2Nm Robustness (FAIR, SameModel)  NoNoise/0/-2..-10 (mean over noise draws)")
    plot_radar(TEST_SNR_DB_LIST, acc4_mean_list, f14_mean_list, os.path.join(noise_dir, "radar_4Nm.png"),
               title="4Nm Robustness (FAIR, SameModel)  NoNoise/0/-2..-10 (mean over noise draws)")

    # baseline plots only (0 dB, single draw)
    base_plots_dir = os.path.join(noise_dir, "BaselinePlots_SNR0dB")
    os.makedirs(base_plots_dir, exist_ok=True)
    _ = eval_on_test(model, feat_model, x2_v, x2_c, y2, base_plots_dir, "2Nm", 0, seed + 10, tta_k=1,
                     roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM)
    _ = eval_on_test(model, feat_model, x4_v, x4_c, y4, base_plots_dir, "4Nm", 0, seed + 20, tta_k=1,
                     roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM)

    del model, feat_model
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# 15) Main
# ============================================================
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # log tee
    log_path = os.path.join(BASE_OUTPUT_DIR, "log.txt")
    f_log = open(log_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, f_log)
    sys.stderr = _Tee(old_err, f_log)

    try:
        t0_all = time.perf_counter()

        print("========== MRCFN (Paper-Structure) | KAIST Load-Shift | Vib(.mat)+Current(.tdms) ==========")
        print("VIB_MAT_DIR_0NM:", VIB_MAT_DIR_0NM)
        print("VIB_MAT_DIR_2NM:", VIB_MAT_DIR_2NM)
        print("VIB_MAT_DIR_4NM:", VIB_MAT_DIR_4NM)
        print("CUR_TDMS_DIR   :", CUR_TDMS_DIR)
        print("OUTPUT_DIR     :", BASE_OUTPUT_DIR)

        robust_envs = ["NoNoise" if np.isinf(s) else int(s) for s in TEST_SNR_DB_LIST]
        print(f"Base noise: {TRAIN_BASE_SNR_DB}dB | Robust envs={robust_envs}")
        print(f"[FIX] post-zscore(after noise)={ENABLE_POST_ZSCORE_AFTER_NOISE} (must be False)")
        print(f"[TTA] ENABLE_TTA={ENABLE_TTA} | TTA_K={TTA_K}")
        print(f"[EVAL] repeats per SNR={EVAL_NOISE_REPEATS}")
        print(f"[Train] epochs={EPOCHS} batch={BATCH_SIZE} Adam lr={LR} clipnorm={GRAD_CLIPNORM}")
        print("[LOG] Writing terminal output to:", log_path)

        # Save a "best_parameters.txt" for baseline consistency
        best_txt = os.path.join(BASE_OUTPUT_DIR, "best_parameters.txt")
        with open(best_txt, "w", encoding="utf-8") as f:
            f.write(f"optimizer: Adam\n")
            f.write(f"lr: {LR}\n")
            f.write(f"batch_size: {BATCH_SIZE}\n")
            f.write(f"epochs: {EPOCHS}\n")
            f.write(f"base_snr_db: {TRAIN_BASE_SNR_DB}\n")
            f.write(f"note: network structure follows MRCFN paper Table2; pipeline aligns to benchmark.\n")
        print("[INFO] Saved:", best_txt)

        print("\n>>> Stage 2: Full experiment (Train 0Nm@base | Test 2Nm&4Nm@base) ...")
        t_s2_0 = time.perf_counter()
        run_full_experiment()
        t_s2_1 = time.perf_counter()
        stage2_s = float(t_s2_1 - t_s2_0)
        print(f"[Time] Stage2(FullExp) = {_fmt_sec(stage2_s)}")

        print("\n>>> Stage 3: Noise robustness study (FAIR, SameModel, mean±std over noise draws) ...")
        t_s3_0 = time.perf_counter()
        run_noise_study()
        t_s3_1 = time.perf_counter()
        stage3_s = float(t_s3_1 - t_s3_0)
        print(f"[Time] Stage3(NoiseStudy) = {_fmt_sec(stage3_s)}")

        total_s = float(time.perf_counter() - t0_all)
        print(f"[Time] TOTAL = {_fmt_sec(total_s)}")

        time_txt = os.path.join(BASE_OUTPUT_DIR, "Time_Summary.txt")
        with open(time_txt, "w", encoding="utf-8") as f:
            f.write(f"Stage2_FullExp_s: {stage2_s:.6f}\n")
            f.write(f"Stage3_NoiseStudy_s: {stage3_s:.6f}\n")
            f.write(f"TOTAL_s: {total_s:.6f}\n")
        print("[Time] Saved:", time_txt)

        print("\nALL DONE. Outputs saved to:", BASE_OUTPUT_DIR)
        print("[LOG] Saved:", log_path)

    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f_log.close()


if __name__ == "__main__":
    main()
