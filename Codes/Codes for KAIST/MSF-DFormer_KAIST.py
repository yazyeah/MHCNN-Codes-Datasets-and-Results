"""
MSF-DFormer (NO Optuna) — KAIST Load-Shift Case Study (Benchmark-aligned)
Vibration (.mat) + Current (.tdms) | Segment length L=2048

[Task / Setting]
- Train/Val: 0Nm only (hash-overlap leakage check enabled)
- Test: 2Nm & 4Nm
- Few-shot protocol: N = 5..30 samples/class, repeated runs = 10
- Seed rule: seed = N*100 + run_idx
- Segments per class cap: 400 (per load)

[Network (paper-consistent, MSF-DFormer)]
- Two-branch encoder (Vibration branch + Current branch)
  - MSF: multi-scale Conv1D feature blocks (parallel kernels/dilations + residual)
  - DFormer: token projection + Transformer encoder blocks (MHA + FFN)
- Fusion: concat(v_vec, c_vec) -> MLP -> Softmax(NUM_CLASSES)
- NOTE: Only create_model() / network structure is the baseline-specific part.
  All other protocol components stay aligned to the benchmark conventions.

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
import sys
import time
import math
import random
import hashlib
import warnings
import traceback
from itertools import cycle

warnings.filterwarnings("ignore")

# ---------------------------
# Environment (open-source friendly placeholders)
# NOTE: keep this BEFORE importing TensorFlow.
# ---------------------------
YOUR_TEMP_PATH = r"YOUR_TEMP_PATH"                 # e.g., r"D:\temp" (replace). Leave as-is to not override.
YOUR_CUDA_VISIBLE_DEVICES = "0"                    # optional: "0"/"1"/"" (empty = default)
YOUR_TF_CPP_MIN_LOG_LEVEL = "2"                    # optional: "2" suppress INFO/WARN

# TEMP/TMP (optional)
if "YOUR_TEMP_PATH" not in YOUR_TEMP_PATH:
    os.environ["TEMP"] = YOUR_TEMP_PATH
    os.environ["TMP"] = YOUR_TEMP_PATH
    os.makedirs(YOUR_TEMP_PATH, exist_ok=True)

# GPU selection (optional)
if YOUR_CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(YOUR_CUDA_VISIBLE_DEVICES)

# TensorFlow logging (optional)
if YOUR_TF_CPP_MIN_LOG_LEVEL is not None:
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
    MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention,
    Concatenate, Activation, LeakyReLU, Add
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


# ---------------------------
# GPU memory growth (safe)
# ---------------------------
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


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
# Global plot style (match benchmark)
# ---------------------------
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1) Config (aligned)
# ============================================================
CUR_MODE = "U"  # "U","V","W","MAG"
DEBUG_PRINT_TDMS_CHANNELS = False

SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

# --- Training hyperparams (fixed, NO Optuna) ---
EPOCHS = 70
BATCH_SIZE = 16
LR = 4.0e-4
GRAD_CLIPNORM = 1.0

DATA_POINTS = 2048
MAX_SEGMENTS_PER_CLASS = 400  # cap per class per load

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

# ---------------------------
# Paths (OPEN-SOURCE friendly placeholders)
# Replace the placeholders below with YOUR local absolute paths.
# This script does NOT assume any fixed folder structure.
# ---------------------------

# Vibration .mat folders for 0Nm/2Nm/4Nm:
# Must contain files like: "0Nm_<fault>.mat", "2Nm_<fault>.mat", "4Nm_<fault>.mat"
VIB_MAT_DIR_0NM = r"YOUR_VIB_MAT_DIR_0NM"
VIB_MAT_DIR_2NM = r"YOUR_VIB_MAT_DIR_2NM"
VIB_MAT_DIR_4NM = r"YOUR_VIB_MAT_DIR_4NM"

# Current .tdms folder:
# Must contain files like: "0Nm_<fault>.tdms", "2Nm_<fault>.tdms", "4Nm_<fault>.tdms"
CUR_TDMS_DIR = r"YOUR_CUR_TDMS_DIR"

# Output folder (all results will be saved here)
# Recommended (benchmark-aligned): ...\KAIST\MSF-DFormer_Comparison_KAIST
BASE_OUTPUT_DIR = r"YOUR_OUTPUT_DIR"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

TSNE_MAX_POINTS = 2000

ENABLE_HASH_OVERLAP_CHECK = True
VAL_PER_CLASS_MAIN = 50

# -------- Noise settings --------
TRAIN_BASE_SNR_DB = 0  # baseline environment: 0dB

NO_NOISE_DB = float("inf")
TEST_SNR_DB_LIST = [NO_NOISE_DB, 0, -2, -4, -6, -8, -10]

# ===== IMPORTANT (aligned): do NOT post-zscore AFTER adding noise =====
ENABLE_ZSCORE_PER_SEGMENT = True         # pre z-score at load-time
ENABLE_POST_ZSCORE_AFTER_NOISE = False   # MUST stay False for meaningful SNR sweep
POST_ZSCORE_EPS = 1e-8

# -------- Stage3 NoiseStudy_model training (aligned) --------
STEP3_BASE_SNR_DB = 0.0
STEP3_NOISY_LOW_DB = -10.0
STEP3_NOISY_HIGH_DB = 0.0
STEP3_SNR_BIAS_P = 3.0

STEP3_NOISY_CE_WEIGHT = 0.35
STEP3_CONSIST_MAX_LAMBDA = 0.10
STEP3_CONSIST_WARMUP_EPOCHS = 15

STEP3_HARD_SNR_DB = -10.0
STEP3_HARD_CE_WEIGHT = 0.25
STEP3_HARD_KL_WEIGHT = 0.50
STEP3_EPOCHS = 70

# -------- TTA (FAIR) --------
ENABLE_TTA = False
TTA_K = 3

# ===== evaluation repeats per SNR =====
EVAL_NOISE_REPEATS = 5
EVAL_TTA_PER_REPEAT = 1

# -------- Consistency regularization (Stage2) --------
ENABLE_CONSISTENCY = True
CONSIST_MAX_LAMBDA = 0.12
CONSIST_WARMUP_EPOCHS = 15
CONSIST_SNR_LOW_DB = -10.0
CONSIST_SNR_HIGH_DB = 0.0
CONSIST_SNR_BIAS_P = 2.0
CONSIST_USE_PER_SAMPLE_SNR = True

# -------- Modality Drop/Corrupt (noisy-view) --------
ENABLE_MODALITY_CORRUPT = True
MOD_CORRUPT_PROB = 0.08
MOD_CORRUPT_MODE = "extra_noise"          # "zero" or "extra_noise"
MOD_CORRUPT_EXTRA_SNR_DB = -12.0

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
# 3.1) TF noise + optional post-zscore
# ============================================================
def _tf_zscore_per_segment(x, eps=POST_ZSCORE_EPS):
    mu = tf.reduce_mean(x, axis=1, keepdims=True)
    sd = tf.math.reduce_std(x, axis=1, keepdims=True) + eps
    return (x - mu) / sd


def _tf_add_gaussian_noise_snr(x, snr_db):
    x = tf.cast(x, tf.float32)
    sig_power = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + 1e-12
    snr_lin = tf.pow(10.0, snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(noise_power), dtype=tf.float32)
    return x + noise


def _tf_add_noise_then_optional_post_zscore(x, snr_db):
    x_n = _tf_add_gaussian_noise_snr(x, snr_db)
    if ENABLE_POST_ZSCORE_AFTER_NOISE and ENABLE_ZSCORE_PER_SEGMENT:
        x_n = _tf_zscore_per_segment(x_n, eps=POST_ZSCORE_EPS)
    return x_n


def _tf_sample_snr_db_custom(batch_size,
                             low_db: float,
                             high_db: float,
                             bias_p: float,
                             per_sample: bool = True):
    low_db = float(low_db)
    high_db = float(high_db)
    p = float(bias_p)

    if per_sample:
        u = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
        u_b = tf.pow(u, p)  # bias toward low SNR
        return low_db + (high_db - low_db) * u_b

    u = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)
    u_b = tf.pow(u, p)
    return low_db + (high_db - low_db) * u_b


def _tf_sample_snr_db(batch_size):
    return _tf_sample_snr_db_custom(
        batch_size,
        low_db=CONSIST_SNR_LOW_DB,
        high_db=CONSIST_SNR_HIGH_DB,
        bias_p=CONSIST_SNR_BIAS_P,
        per_sample=CONSIST_USE_PER_SAMPLE_SNR
    )


def _tf_modality_corrupt(xv, xc):
    if (not ENABLE_MODALITY_CORRUPT) or (MOD_CORRUPT_PROB <= 0.0):
        return xv, xc

    B = tf.shape(xv)[0]
    trigger = tf.random.uniform([B, 1, 1], 0.0, 1.0, dtype=tf.float32) < float(MOD_CORRUPT_PROB)
    choose_v = tf.random.uniform([B, 1, 1], 0.0, 1.0, dtype=tf.float32) < 0.5

    drop_v = tf.logical_and(trigger, choose_v)
    drop_c = tf.logical_and(trigger, tf.logical_not(choose_v))

    if MOD_CORRUPT_MODE.lower() == "zero":
        xv2 = tf.where(drop_v, tf.zeros_like(xv), xv)
        xc2 = tf.where(drop_c, tf.zeros_like(xc), xc)
        return xv2, xc2

    if MOD_CORRUPT_MODE.lower() == "extra_noise":
        extra_snr = tf.constant(float(MOD_CORRUPT_EXTRA_SNR_DB), dtype=tf.float32)
        xv_extra = _tf_add_gaussian_noise_snr(xv, extra_snr)
        xc_extra = _tf_add_gaussian_noise_snr(xc, extra_snr)

        if ENABLE_POST_ZSCORE_AFTER_NOISE and ENABLE_ZSCORE_PER_SEGMENT:
            xv_extra = _tf_zscore_per_segment(xv_extra, eps=POST_ZSCORE_EPS)
            xc_extra = _tf_zscore_per_segment(xc_extra, eps=POST_ZSCORE_EPS)

        xv2 = tf.where(drop_v, xv_extra, xv)
        xc2 = tf.where(drop_c, xc_extra, xc)
        return xv2, xc2

    return xv, xc


# ============================================================
# 4) Vibration .mat parsing (KAIST style)  [FIXED HERE]
# ============================================================
def _extract_values_matrix_kaist(mat_dict: dict) -> np.ndarray:
    """
    Benchmark-consistent KAIST parsing:
      mat['Signal']['y_values']['values']
    """
    if "Signal" not in mat_dict:
        raise KeyError("Missing 'Signal' key.")
    sig = mat_dict["Signal"]
    yv = sig["y_values"][0, 0]
    vals = yv["values"][0, 0]
    vals = np.asarray(vals)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    return vals


def _fallback_find_numeric_array(mat_dict: dict) -> np.ndarray:
    """
    Fallback: search for the first 1D/2D numeric ndarray not in __ keys.
    """
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    for k in keys:
        v = mat_dict.get(k, None)
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            if v.ndim == 1:
                return v.reshape(-1, 1)
            if v.ndim == 2:
                return v
    raise RuntimeError(f"[MAT] No numeric array found in keys={keys[:20]}")


def load_mat_values(mat_path: str) -> np.ndarray:
    md = loadmat(mat_path)
    # try KAIST structure first (this is the FIX)
    try:
        return _extract_values_matrix_kaist(md)
    except Exception:
        # fallback
        try:
            return _fallback_find_numeric_array(md)
        except Exception as e2:
            keys = [k for k in md.keys() if not k.startswith("__")]
            raise RuntimeError(f"[MAT] Parse failed for: {mat_path}\n"
                               f"Available keys: {keys[:30]}\n"
                               f"Fallback error: {e2}") from e2


def _pick_vib_1d(values: np.ndarray) -> np.ndarray:
    """
    Match your benchmark policy:
    - if >=5 cols: use col 1 (xA)
    - else use col 0
    """
    values = np.asarray(values)
    if values.ndim != 2:
        values = values.reshape(-1, 1)
    ncol = values.shape[1]
    if ncol >= 5:
        return values[:, 1].astype(np.float32)
    if ncol == 4:
        return values[:, 0].astype(np.float32)
    return values[:, 0].astype(np.float32)


# ============================================================
# 5) Current TDMS parsing (aligned)
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
# 6) Segmentation (aligned)
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
# 7) Load datasets: vib(.mat) + current(.tdms)  (aligned)
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

        vib_vals = load_mat_values(vib_file)     # <-- FIXED MAT parsing
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


def preload_kaist():
    print(">>> Pre-loading KAIST loads into memory...")
    print("Loading Vibration(.mat)+Current(.tdms) ...")
    data_0 = load_dataset_for_load("0Nm", VIB_MAT_DIR_0NM)
    data_2 = load_dataset_for_load("2Nm", VIB_MAT_DIR_2NM)
    data_4 = load_dataset_for_load("4Nm", VIB_MAT_DIR_4NM)
    print_dataset_stats("0Nm", data_0)
    print_dataset_stats("2Nm", data_2)
    print_dataset_stats("4Nm", data_4)
    print("Data loaded.\n")
    return {"0Nm": data_0, "2Nm": data_2, "4Nm": data_4}


# ============================================================
# 8) Split: 0Nm train/val; 2Nm/4Nm test  (aligned)
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
# 9) MSF-DFormer model (ONLY PART allowed to differ)
# ============================================================
def _msf_block(x, filters: int, name: str, dropout: float = 0.10):
    """
    Multi-Scale Feature block (1D):
    - parallel conv with different kernel sizes + dilations
    - concat -> 1x1 fuse -> residual
    """
    shortcut = x

    b1 = Conv1D(filters, kernel_size=3, padding="same", dilation_rate=1, name=f"{name}_k3_d1")(x)
    b1 = Activation("gelu", name=f"{name}_b1_gelu")(b1)

    b2 = Conv1D(filters, kernel_size=5, padding="same", dilation_rate=2, name=f"{name}_k5_d2")(x)
    b2 = Activation("gelu", name=f"{name}_b2_gelu")(b2)

    b3 = Conv1D(filters, kernel_size=7, padding="same", dilation_rate=4, name=f"{name}_k7_d4")(x)
    b3 = Activation("gelu", name=f"{name}_b3_gelu")(b3)

    m = Concatenate(name=f"{name}_concat")([b1, b2, b3])
    m = Conv1D(filters, kernel_size=1, padding="same", name=f"{name}_fuse1x1")(m)
    m = Dropout(dropout, name=f"{name}_drop")(m)

    # residual (project if needed)
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same", name=f"{name}_proj")(shortcut)

    out = Add(name=f"{name}_add")([shortcut, m])
    out = Activation("gelu", name=f"{name}_out_gelu")(out)
    return out


def _dformer_block(x, embed_dim: int, num_heads: int, ff_dim: int, dropout: float, name: str):
    """
    Transformer encoder block for 1D tokens.
    """
    # LN + MHA
    x1 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads,
                             dropout=dropout, name=f"{name}_mha")(x1, x1)
    x2 = Add(name=f"{name}_add1")([x, attn])

    # LN + FFN
    x3 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x2)
    ff = Dense(ff_dim, activation="gelu", name=f"{name}_ff1")(x3)
    ff = Dropout(dropout, name=f"{name}_ffdrop")(ff)
    ff = Dense(embed_dim, name=f"{name}_ff2")(ff)
    out = Add(name=f"{name}_add2")([x2, ff])
    return out


def _msf_dformer_encoder(inp, prefix: str,
                         stem_filters: int = 64,
                         msf_filters: int = 96,
                         embed_dim: int = 128,
                         num_heads: int = 4,
                         ff_dim: int = 256,
                         n_blocks: int = 2,
                         dropout: float = 0.20):
    """
    1) stem downsample
    2) MSF blocks
    3) token projection + DFormer blocks
    4) global pooling -> vector
    """
    x = Conv1D(stem_filters, kernel_size=9, strides=2, padding="same", name=f"{prefix}_stem_conv")(inp)
    x = Activation("gelu", name=f"{prefix}_stem_gelu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_stem_pool")(x)  # 2048 -> 512

    x = _msf_block(x, msf_filters, name=f"{prefix}_msf1", dropout=0.10)
    x = _msf_block(x, msf_filters, name=f"{prefix}_msf2", dropout=0.10)

    # tokens: project to embed_dim and downsample to manageable length
    # 512 -> 128 tokens
    x = Conv1D(embed_dim, kernel_size=4, strides=4, padding="same", name=f"{prefix}_tokenize")(x)
    x = Dropout(dropout, name=f"{prefix}_tok_drop")(x)

    for i in range(n_blocks):
        x = _dformer_block(x, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                           dropout=dropout, name=f"{prefix}_dformer{i+1}")

    x = LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_out")(x)
    vec = GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    vec = Dropout(dropout, name=f"{prefix}_vec_drop")(vec)
    return vec


def create_model():
    """
    Return:
      - base_model: softmax classifier
      - feat_model: fused feature vector (for t-SNE)
    """
    # modest capacity (avoid too high/too low; also closer to KAIST few-shot)
    embed_dim = 128
    num_heads = 4
    ff_dim = 256
    n_blocks = 2
    dropout = 0.25

    input_v = Input(shape=(DATA_POINTS, 1), name="vib_input")
    input_c = Input(shape=(DATA_POINTS, 1), name="cur_input")

    v_vec = _msf_dformer_encoder(
        input_v, prefix="vib",
        stem_filters=64, msf_filters=96,
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
        n_blocks=n_blocks, dropout=dropout
    )
    c_vec = _msf_dformer_encoder(
        input_c, prefix="cur",
        stem_filters=64, msf_filters=96,
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
        n_blocks=n_blocks, dropout=dropout
    )

    fused = Concatenate(name="fusion_concat")([v_vec, c_vec])
    fused = Dense(256, activation="gelu", name="fusion_fc1")(fused)
    fused = Dropout(0.30, name="fusion_drop")(fused)
    fused = Dense(256, activation="gelu", name="fusion_fc2")(fused)
    feat = Dropout(0.10, name="feat_drop")(fused)

    out = Dense(NUM_CLASSES, activation="softmax", name="pred")(feat)

    base_model = Model(inputs=[input_v, input_c], outputs=out, name="MSF_DFormer")
    feat_model = Model(inputs=[input_v, input_c], outputs=feat, name="MSF_DFormer_feat")
    return base_model, feat_model


# ============================================================
# 9.1) Trainer: CE(base) + CE(noisy) + KL(base||noisy)  (aligned)
# ============================================================
class NoiseStableTrainer(tf.keras.Model):
    def __init__(self,
                 base_model: tf.keras.Model,
                 base_snr_db: float,
                 noisy_low_db: float = CONSIST_SNR_LOW_DB,
                 noisy_high_db: float = CONSIST_SNR_HIGH_DB,
                 bias_p: float = CONSIST_SNR_BIAS_P,
                 per_sample_snr: bool = CONSIST_USE_PER_SAMPLE_SNR,
                 noisy_ce_weight: float = 0.0,
                 hard_snr_db: float = None,
                 hard_ce_weight: float = 0.0,
                 hard_kl_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_snr_db = tf.constant(float(base_snr_db), dtype=tf.float32)

        self.noisy_low_db = float(noisy_low_db)
        self.noisy_high_db = float(noisy_high_db)
        self.bias_p = float(bias_p)
        self.per_sample_snr = bool(per_sample_snr)

        self.noisy_ce_weight = tf.constant(float(noisy_ce_weight), dtype=tf.float32)
        self.use_noisy_ce = (float(noisy_ce_weight) > 0.0)

        self.use_hard = (hard_snr_db is not None) and ((float(hard_ce_weight) > 0.0) or ENABLE_CONSISTENCY)
        self.hard_snr_db = tf.constant(float(hard_snr_db), dtype=tf.float32) if (hard_snr_db is not None) else None
        self.hard_ce_weight = tf.constant(float(hard_ce_weight), dtype=tf.float32)
        self.hard_kl_weight = tf.constant(float(hard_kl_weight), dtype=tf.float32)

        self.cons_lambda = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        self._ce_metric = tf.keras.metrics.Mean(name="ce_loss")
        self._noisy_ce_metric = tf.keras.metrics.Mean(name="noisy_ce_loss")
        self._hard_ce_metric = tf.keras.metrics.Mean(name="hard_ce_loss")
        self._kl_metric = tf.keras.metrics.Mean(name="kl_loss")
        self._kl_hard_metric = tf.keras.metrics.Mean(name="kl_hard_loss")

    @property
    def metrics(self):
        return [self._ce_metric, self._noisy_ce_metric, self._hard_ce_metric, self._kl_metric, self._kl_hard_metric] + super().metrics

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    @staticmethod
    def _kl_stopgrad_teacher(p_teacher, p_student):
        eps = 1e-7
        p_t = tf.stop_gradient(tf.clip_by_value(p_teacher, eps, 1.0 - eps))
        p_s = tf.clip_by_value(p_student, eps, 1.0 - eps)
        kl_vec = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(p_t, p_s)
        return tf.reduce_mean(kl_vec)

    def train_step(self, data):
        (xv, xc), y = data
        y = tf.cast(y, tf.float32)
        B = tf.shape(xv)[0]

        # base-view (anchor)
        xv_base = _tf_add_noise_then_optional_post_zscore(xv, self.base_snr_db)
        xc_base = _tf_add_noise_then_optional_post_zscore(xc, self.base_snr_db)

        need_rand = bool(ENABLE_CONSISTENCY) or self.use_noisy_ce
        if need_rand:
            snr_rnd = _tf_sample_snr_db_custom(
                B,
                low_db=self.noisy_low_db,
                high_db=self.noisy_high_db,
                bias_p=self.bias_p,
                per_sample=self.per_sample_snr
            )
            xv_noisy = _tf_add_gaussian_noise_snr(tf.cast(xv, tf.float32), snr_rnd)
            xc_noisy = _tf_add_gaussian_noise_snr(tf.cast(xc, tf.float32), snr_rnd)
            if ENABLE_POST_ZSCORE_AFTER_NOISE and ENABLE_ZSCORE_PER_SEGMENT:
                xv_noisy = _tf_zscore_per_segment(xv_noisy, eps=POST_ZSCORE_EPS)
                xc_noisy = _tf_zscore_per_segment(xc_noisy, eps=POST_ZSCORE_EPS)
            xv_noisy, xc_noisy = _tf_modality_corrupt(xv_noisy, xc_noisy)
        else:
            xv_noisy, xc_noisy = None, None

        if self.use_hard:
            xv_hard = _tf_add_noise_then_optional_post_zscore(xv, self.hard_snr_db)
            xc_hard = _tf_add_noise_then_optional_post_zscore(xc, self.hard_snr_db)
            xv_hard, xc_hard = _tf_modality_corrupt(xv_hard, xc_hard)
        else:
            xv_hard, xc_hard = None, None

        with tf.GradientTape() as tape:
            p_base = self.base_model([xv_base, xc_base], training=True)
            ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, p_base))
            reg = tf.add_n(self.base_model.losses) if self.base_model.losses else 0.0

            total_loss = ce + reg

            noisy_ce = tf.constant(0.0, dtype=tf.float32)
            hard_ce = tf.constant(0.0, dtype=tf.float32)
            kl = tf.constant(0.0, dtype=tf.float32)
            kl_hard = tf.constant(0.0, dtype=tf.float32)

            if (xv_noisy is not None) and (xc_noisy is not None):
                p_noisy = self.base_model([xv_noisy, xc_noisy], training=True)
                noisy_ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, p_noisy))
                total_loss = total_loss + tf.cast(self.noisy_ce_weight, tf.float32) * noisy_ce
                if ENABLE_CONSISTENCY:
                    kl = self._kl_stopgrad_teacher(p_base, p_noisy)
                    total_loss = total_loss + self.cons_lambda * kl

            if (xv_hard is not None) and (xc_hard is not None):
                p_hard = self.base_model([xv_hard, xc_hard], training=True)
                hard_ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, p_hard))
                total_loss = total_loss + tf.cast(self.hard_ce_weight, tf.float32) * hard_ce
                if ENABLE_CONSISTENCY:
                    kl_hard = self._kl_stopgrad_teacher(p_base, p_hard)
                    total_loss = total_loss + self.cons_lambda * tf.cast(self.hard_kl_weight, tf.float32) * kl_hard

            loss = total_loss

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.compiled_metrics.update_state(y, p_base)

        self._ce_metric.update_state(ce)
        self._noisy_ce_metric.update_state(noisy_ce)
        self._hard_ce_metric.update_state(hard_ce)
        self._kl_metric.update_state(kl)
        self._kl_hard_metric.update_state(kl_hard)

        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        out["cons_lambda"] = self.cons_lambda
        return out

    def test_step(self, data):
        (xv, xc), y = data
        y = tf.cast(y, tf.float32)

        xv_base = _tf_add_noise_then_optional_post_zscore(xv, self.base_snr_db)
        xc_base = _tf_add_noise_then_optional_post_zscore(xc, self.base_snr_db)

        p = self.base_model([xv_base, xc_base], training=False)
        ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, p))
        reg = tf.add_n(self.base_model.losses) if self.base_model.losses else 0.0
        loss = ce + reg

        self.compiled_metrics.update_state(y, p)
        self._ce_metric.update_state(ce)
        self._noisy_ce_metric.update_state(tf.constant(0.0, dtype=tf.float32))
        self._hard_ce_metric.update_state(tf.constant(0.0, dtype=tf.float32))
        self._kl_metric.update_state(tf.constant(0.0, dtype=tf.float32))
        self._kl_hard_metric.update_state(tf.constant(0.0, dtype=tf.float32))

        out = {m.name: m.result() for m in self.metrics}
        out["loss"] = loss
        out["cons_lambda"] = self.cons_lambda
        return out


class ConsistencyLambdaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_lambda: float, warmup_epochs: int):
        super().__init__()
        self.max_lambda = float(max_lambda)
        self.warmup_epochs = int(max(1, warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "cons_lambda"):
            return
        if (not ENABLE_CONSISTENCY) or (self.max_lambda <= 0.0):
            self.model.cons_lambda.assign(0.0)
            return
        if epoch < self.warmup_epochs:
            lam = self.max_lambda * (float(epoch + 1) / float(self.warmup_epochs))
        else:
            lam = self.max_lambda
        self.model.cons_lambda.assign(lam)


# ============================================================
# 11) Visualization (aligned)
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
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_vis, palette="tab10", legend="full", s=15)
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
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg (AUC={roc_auc['micro']:.4f})", linestyle=":", lw=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg (AUC={roc_auc['macro']:.4f})", linestyle=":", lw=3)

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
# 12) Trimmed stats + radar + trend plot (aligned style)
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

def plot_performance_trend(n_list, acc2, f12, acc4, f14, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(n_list, acc2, marker="o", label="2Nm Acc")
    plt.plot(n_list, f12, marker="o", label="2Nm F1")
    plt.plot(n_list, acc4, marker="s", label="4Nm Acc")
    plt.plot(n_list, f14, marker="s", label="4Nm F1")
    plt.xlabel("N per class (0Nm train)")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.05)
    plt.title("Performance Trend (Trimmed Mean) - MSF-DFormer Load Shift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# 13) Evaluate helper (aligned)
# ============================================================
def eval_on_test(model, feat_model, x_te_v, x_te_c, y_te,
                out_dir, tag, snr_db, seed, tta_k=1,
                roc_colors=None, cm_cmap="Blues"):
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
# 14) Full experiment (Stage2)  (aligned)
# ============================================================
def run_full_experiment(DATA_0, DATA_2, DATA_4):
    x2_v, x2_c, y2 = build_test_from_load(DATA_2, seed=123)
    x4_v, x4_c, y4 = build_test_from_load(DATA_4, seed=456)

    print("\n>>> Best MSF-DFormer Model Architecture Summary:")
    print("=" * 60)
    tmp_model, _ = create_model()
    tmp_model.summary()
    del tmp_model
    print("=" * 60 + "\n")

    raw_rows = []
    trim_rows = []

    trend_n = []
    trend_acc2 = []
    trend_f12 = []
    trend_acc4 = []
    trend_f14 = []

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

            tf.keras.backend.clear_session()
            gc.collect()

            base_model, feat_model = create_model()
            trainer = NoiseStableTrainer(base_model, base_snr_db=TRAIN_BASE_SNR_DB, name="NoiseStableTrainer")

            opt = tf.keras.optimizers.Adamax(learning_rate=LR, clipnorm=GRAD_CLIPNORM)
            trainer.compile(optimizer=opt, metrics=["accuracy"])

            callbacks = [ConsistencyLambdaScheduler(CONSIST_MAX_LAMBDA, CONSIST_WARMUP_EPOCHS)]

            t_fit0 = time.perf_counter()
            history = trainer.fit(
                [x_tr_v, x_tr_c], y_tr,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=([x_va_v, x_va_c], y_va),
                verbose=0,
                callbacks=callbacks
            )
            t_fit1 = time.perf_counter()
            train_time_s = float(t_fit1 - t_fit0)

            save_curves(history, os.path.join(run_dir, "curves.png"),
                        f"Train0Nm N={n_train} Run={run_idx}", val_name="Val(0Nm)")

            out2 = os.path.join(run_dir, "Test_2Nm")
            out4 = os.path.join(run_dir, "Test_4Nm")

            acc2, f12, p2, r2, t2_inf_s, t2_viz_s = eval_on_test(
                base_model, feat_model, x2_v, x2_c, y2, out2, "2Nm", TRAIN_BASE_SNR_DB, seed,
                tta_k=1, roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            acc4, f14, p4, r4, t4_inf_s, t4_viz_s = eval_on_test(
                base_model, feat_model, x4_v, x4_c, y4, out4, "4Nm", TRAIN_BASE_SNR_DB, seed,
                tta_k=1, roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
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

            del trainer, base_model, feat_model
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

        trend_n.append(n_train)
        trend_acc2.append(acc2_m); trend_f12.append(f12_m)
        trend_acc4.append(acc4_m); trend_f14.append(f14_m)

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

    raw_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_raw.csv"), index=False, encoding="utf-8-sig")
    trim_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_trimmed.csv"), index=False, encoding="utf-8-sig")

    plot_performance_trend(trend_n, trend_acc2, trend_f12, trend_acc4, trend_f14,
                           os.path.join(BASE_OUTPUT_DIR, "performance_trend.png"))

    return


# ============================================================
# 15) Noise robustness study (Stage3) - FAIR VERSION (SameModel)
# ============================================================
def run_noise_study(DATA_0, DATA_2, DATA_4):
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

    tf.keras.backend.clear_session()
    gc.collect()

    base_model, feat_model = create_model()

    trainer = NoiseStableTrainer(
        base_model,
        base_snr_db=STEP3_BASE_SNR_DB,
        noisy_low_db=STEP3_NOISY_LOW_DB,
        noisy_high_db=STEP3_NOISY_HIGH_DB,
        bias_p=STEP3_SNR_BIAS_P,
        per_sample_snr=True,
        noisy_ce_weight=STEP3_NOISY_CE_WEIGHT,
        hard_snr_db=STEP3_HARD_SNR_DB,
        hard_ce_weight=STEP3_HARD_CE_WEIGHT,
        hard_kl_weight=STEP3_HARD_KL_WEIGHT,
        name="NoiseStableTrainer_NoiseStudyModel_FAIR"
    )

    opt = tf.keras.optimizers.Adamax(learning_rate=LR, clipnorm=GRAD_CLIPNORM)
    trainer.compile(optimizer=opt, metrics=["accuracy"])
    callbacks = [ConsistencyLambdaScheduler(STEP3_CONSIST_MAX_LAMBDA, STEP3_CONSIST_WARMUP_EPOCHS)]

    t_fit0 = time.perf_counter()
    _ = trainer.fit(
        [x_tr_v, x_tr_c], y_tr,
        epochs=STEP3_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([x_va_v, x_va_c], y_va),
        verbose=0,
        callbacks=callbacks
    )
    t_fit1 = time.perf_counter()
    print(f"[Step3-FAIR] NoiseStudy_model trained ONCE. time={_fmt_sec(t_fit1 - t_fit0)}")

    unified_tta_k = (int(TTA_K) if ENABLE_TTA else 1)

    rows = []
    acc2_mean_list, f12_mean_list, acc4_mean_list, f14_mean_list = [], [], [], []

    for snr_db in TEST_SNR_DB_LIST:
        label = "NoNoise" if (np.isscalar(snr_db) and np.isinf(snr_db)) else str(int(snr_db))
        tta_k = 1 if (np.isscalar(snr_db) and np.isinf(snr_db)) else unified_tta_k

        acc2_buf, f12_buf = [], []
        acc4_buf, f14_buf = [], []

        for rep in range(EVAL_NOISE_REPEATS if not (np.isscalar(snr_db) and np.isinf(snr_db)) else 1):
            rep_seed2 = seed + 1000 + rep * 17
            rep_seed4 = seed + 2000 + rep * 17
            k_inside = EVAL_TTA_PER_REPEAT if tta_k > 1 else 1

            a2, f2, _, _, _, _ = eval_on_test(
                base_model, feat_model, x2_v, x2_c, y2,
                out_dir=None, tag="2Nm", snr_db=snr_db, seed=rep_seed2,
                tta_k=k_inside, roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            a4, f4, _, _, _, _ = eval_on_test(
                base_model, feat_model, x4_v, x4_c, y4,
                out_dir=None, tag="4Nm", snr_db=snr_db, seed=rep_seed4,
                tta_k=k_inside, roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
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
            tta_k, EVAL_NOISE_REPEATS
        ])

        acc2_mean_list.append(acc2_m); f12_mean_list.append(f12_m)
        acc4_mean_list.append(acc4_m); f14_mean_list.append(f14_m)

        print(f"[Step3-FAIR] SNR={label} | 2Nm Acc={acc2_m:.4f}±{acc2_s:.4f} F1={f12_m:.4f}±{f12_s:.4f} | "
              f"4Nm Acc={acc4_m:.4f}±{acc4_s:.4f} F1={f14_m:.4f}±{f14_s:.4f} | tta={tta_k}")

        if not (np.isscalar(snr_db) and np.isinf(snr_db)):
            rng = np.random.RandomState(seed + 555)
            subset = slice(0, min(256, x2_v.shape[0]))
            x_clean = x2_v[subset]
            x_noisy = add_noise_then_optional_post_zscore_np(x_clean, snr_db, rng)
            emp = estimate_empirical_snr_db(x_clean, x_noisy)
            print(f"    [Sanity] empirical SNR≈{emp:.2f} dB (vib subset, after pipeline)")

    out_csv = os.path.join(noise_dir, "Noise_Robustness_LoadShift_SameModel_FAIR.csv")
    pd.DataFrame(rows, columns=[
        "label", "SNR_dB",
        "Acc_2Nm_mean", "Acc_2Nm_std", "F1_2Nm_mean", "F1_2Nm_std",
        "Acc_4Nm_mean", "Acc_4Nm_std", "F1_4Nm_mean", "F1_4Nm_std",
        "tta_k_reported", "eval_repeats"
    ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[Step3-FAIR] Saved:", out_csv)

    plot_radar(TEST_SNR_DB_LIST, acc2_mean_list, f12_mean_list, os.path.join(noise_dir, "radar_2Nm.png"),
               title="2Nm Robustness (FAIR, SameModel)  NoNoise/0/-2..-10 (mean over noise draws)")
    plot_radar(TEST_SNR_DB_LIST, acc4_mean_list, f14_mean_list, os.path.join(noise_dir, "radar_4Nm.png"),
               title="4Nm Robustness (FAIR, SameModel)  NoNoise/0/-2..-10 (mean over noise draws)")

    # Baseline plots only (0 dB, single draw)
    base_plots_dir = os.path.join(noise_dir, "BaselinePlots_SNR0dB")
    os.makedirs(base_plots_dir, exist_ok=True)
    _ = eval_on_test(base_model, feat_model, x2_v, x2_c, y2, base_plots_dir, "2Nm", 0, seed + 10, tta_k=1,
                     roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM)
    _ = eval_on_test(base_model, feat_model, x4_v, x4_c, y4, base_plots_dir, "4Nm", 0, seed + 20, tta_k=1,
                     roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM)

    del trainer, base_model, feat_model
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# 16) Main (aligned)
# ============================================================
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(BASE_OUTPUT_DIR, "log.txt")
    f_log = open(log_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, f_log)
    sys.stderr = _Tee(old_err, f_log)

    try:
        t0_all = time.perf_counter()

        print("========== MSF-DFormer (Baseline, NO Optuna) Load Shift | Vib(.mat)+Current(.tdms) ==========")
        print("VIB_MAT_DIR_0NM:", VIB_MAT_DIR_0NM)
        print("VIB_MAT_DIR_2NM:", VIB_MAT_DIR_2NM)
        print("VIB_MAT_DIR_4NM:", VIB_MAT_DIR_4NM)
        print("CUR_TDMS_DIR   :", CUR_TDMS_DIR)
        print("OUTPUT_DIR     :", BASE_OUTPUT_DIR)
        print(f"NUM_CLASSES={NUM_CLASSES} | SAMPLE_RANGE=5..30 | REPEAT={REPEAT_TIMES}")
        print(f"TRAIN/VAL: {TRAIN_LOAD} | TEST: {TEST_LOADS}")
        robust_envs = ["NoNoise" if np.isinf(s) else int(s) for s in TEST_SNR_DB_LIST]
        print(f"Base noise: {TRAIN_BASE_SNR_DB}dB | Robust envs={robust_envs}")
        print(f"[post-zscore(after noise)] {ENABLE_POST_ZSCORE_AFTER_NOISE}  (should be False)")
        print(f"[TTA] ENABLE_TTA={ENABLE_TTA} | TTA_K={TTA_K}")
        print(f"[EVAL] repeats per SNR={EVAL_NOISE_REPEATS} | tta per repeat={EVAL_TTA_PER_REPEAT}")
        print(f"[Train] epochs={EPOCHS}, batch={BATCH_SIZE}, Adamax lr={LR}, clipnorm={GRAD_CLIPNORM}")
        print("[LOG] Writing terminal output to:", log_path)

        # save fixed params for consistency
        bp = os.path.join(BASE_OUTPUT_DIR, "best_parameters.txt")
        with open(bp, "w", encoding="utf-8") as f:
            f.write("MSF-DFormer (fixed, NO Optuna)\n")
            f.write(f"LR={LR}\nBATCH_SIZE={BATCH_SIZE}\nEPOCHS={EPOCHS}\n")
            f.write(f"CONSIST_MAX_LAMBDA={CONSIST_MAX_LAMBDA}\nCONSIST_WARMUP_EPOCHS={CONSIST_WARMUP_EPOCHS}\n")
            f.write(f"TRAIN_BASE_SNR_DB={TRAIN_BASE_SNR_DB}\n")
            f.write(f"ENABLE_POST_ZSCORE_AFTER_NOISE={ENABLE_POST_ZSCORE_AFTER_NOISE}\n")
        print("[OK] Saved:", bp)

        # ---- Preload data ----
        data_all = preload_kaist()
        DATA_0 = data_all["0Nm"]
        DATA_2 = data_all["2Nm"]
        DATA_4 = data_all["4Nm"]

        # ---- Stage2 ----
        print("\n>>> Stage 2: Full experiment (Train 0Nm@base | Test 2Nm&4Nm@base) ...")
        t_s2_0 = time.perf_counter()
        run_full_experiment(DATA_0, DATA_2, DATA_4)
        t_s2_1 = time.perf_counter()
        stage2_s = float(t_s2_1 - t_s2_0)
        print(f"[Time] Stage2(FullExp) = {_fmt_sec(stage2_s)}")

        # ---- Stage3 ----
        print("\n>>> Stage 3: Noise robustness study (FAIR, SameModel, mean±std over noise draws) ...")
        t_s3_0 = time.perf_counter()
        run_noise_study(DATA_0, DATA_2, DATA_4)
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

    except Exception as e:
        print("\n[ERROR] Unhandled exception:", type(e).__name__, str(e))
        traceback.print_exc()
        raise
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f_log.close()


if __name__ == "__main__":
    main()
