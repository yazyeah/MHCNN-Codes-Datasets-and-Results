"""
KDCNN-DF — KAIST Load-Shift Case Study
Vibration (.mat) + Current (.tdms) | Segment length L=2048

[Protocol (STRICT, benchmark-aligned)]
- Train/Val: 0Nm only (hash-overlap leakage check enabled)
- Test: 2Nm & 4Nm
- Few-shot: N = 5..30 samples/class, repeated runs = 10
- Seed rule: seed = N*100 + run_idx
- Segments per class cap: 400 (per load)
- Baseline noise environment: 0 dB Gaussian for train/val/test (Stage2 baseline)
- IMPORTANT: ENABLE_POST_ZSCORE_AFTER_NOISE MUST remain False (preserve meaningful SNR semantics)

[Baseline rule]
- ONLY the KDCNN-DF model architecture follows the paper:
  CWT -> Teacher CNN / Student CNN (KD) -> simplified MGS -> Cascade Deep Forest
- Everything else (data loading / splits / seeds / noise policy / outputs / plots / CSV / logging)
  stays aligned to MHCNN_Comparison_KAIST_Advanced conventions.

[Robustness Stage3 (FAIR, SameModel)]
- Train ONE NoiseStudy_model once (fixed seed, N=30), then evaluate SNR points:
  NoNoise + 0 + (-2, -4, -6, -8, -10)
- Each SNR point: mean±std over repeated noise draws

[Outputs]
- log.txt (tee stdout/stderr)
- curves.png per run (Teacher + Student-KD)
- Confusion matrix / t-SNE / ROC per test load (2Nm, 4Nm)
- Final_Summary_Stats_raw.csv + Final_Summary_Stats_trimmed.csv
- NoiseStudy_LoadShift_FAIR_SameModel/: CSV + radar plots
- CWT cache dir (BASE 0dB only) under BASE_OUTPUT_DIR

[Open-source friendly paths]
- Replace placeholders in the PATHS section (YOUR_*) with your local absolute paths.
- This script does NOT assume any fixed folder structure.
"""


import os
import sys
import gc
import time
import math
import random
import hashlib
import warnings
import traceback
from itertools import cycle
from typing import List, Tuple, Dict, Any

warnings.filterwarnings("ignore")

# ---------------------------
# Environment (open-source friendly placeholders)
# NOTE: keep this BEFORE importing TensorFlow.
# ---------------------------
YOUR_TEMP_PATH = r"YOUR_TEMP_PATH"                 # e.g., r"D:\temp_fix" (replace). Keep placeholder to NOT override.
YOUR_CUDA_VISIBLE_DEVICES = "0"                    # optional: "0"/"1"/"" (empty = default)
YOUR_TF_CPP_MIN_LOG_LEVEL = "2"                    # optional: "2" suppress INFO/WARN

# TEMP/TMP/TMPDIR (optional)
if "YOUR_TEMP_PATH" not in YOUR_TEMP_PATH:
    os.environ["TEMP"] = YOUR_TEMP_PATH
    os.environ["TMP"] = YOUR_TEMP_PATH
    os.environ["TMPDIR"] = YOUR_TEMP_PATH
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
from keras import layers
from keras.utils import np_utils

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# TDMS reader
try:
    from nptdms import TdmsFile
except ImportError as e:
    raise ImportError("Missing dependency: nptdms. Please run: pip install nptdms") from e

# CWT dependency
try:
    import pywt
except ImportError as e:
    raise ImportError("Missing dependency: PyWavelets. Please run: pip install PyWavelets") from e


# ---------------------------
# Utils
# ---------------------------
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
# Global plot style (match your template)
# ---------------------------
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1) Config (align to benchmark)
# ============================================================
CUR_MODE = "U"  # "U","V","W","MAG"
DEBUG_PRINT_TDMS_CHANNELS = False

SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

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
# ---------------------------

# Vibration .mat folders for 0Nm/2Nm/4Nm:
# Must contain files like: "0Nm_<fault>.mat", "2Nm_<fault>.mat", "4Nm_<fault>.mat"
VIB_MAT_DIR_0NM = r"YOUR_VIB_MAT_DIR_0NM"
VIB_MAT_DIR_2NM = r"YOUR_VIB_MAT_DIR_2NM"
VIB_MAT_DIR_4NM = r"YOUR_VIB_MAT_DIR_4NM"

# Current .tdms folder:
# Must contain files like: "0Nm_<fault>.tdms", "2Nm_<fault>.tdms", "4Nm_<fault>.tdms"
CUR_TDMS_DIR = r"YOUR_CUR_TDMS_DIR"

# Output root (benchmark-aligned suggestion):
# ...\KAIST\KDCNN_DF_Comparison_KAIST
BASE_OUTPUT_DIR = r"YOUR_OUTPUT_DIR"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

TSNE_MAX_POINTS = 2000
ENABLE_HASH_OVERLAP_CHECK = True
VAL_PER_CLASS_MAIN = 50

# -------- Noise settings --------
TRAIN_BASE_SNR_DB = 0  # baseline environment: 0dB
NO_NOISE_DB = float("inf")
TEST_SNR_DB_LIST = [NO_NOISE_DB, 0, -2, -4, -6, -8, -10]

# z-score policy (match your template)
ENABLE_ZSCORE_PER_SEGMENT = True
ENABLE_POST_ZSCORE_AFTER_NOISE = False
POST_ZSCORE_EPS = 1e-8

# FAIR robustness repeats (match your template)
EVAL_NOISE_REPEATS = 5

# -------- KDCNN-DF (paper) key hyperparams --------
# Paper reports for Ottawa datasets: alpha=0.45, T=2.5, teacher epochs=35, student epochs=40
KD_ALPHA = 0.45
KD_TEMPERATURE = 2.5

TEACHER_EPOCHS = 35
STUDENT_EPOCHS = 40
BATCH_SIZE = 32
LR_TEACHER = 1e-3
LR_STUDENT = 1e-3
GRAD_CLIPNORM = 1.0
EARLY_STOP_PATIENCE = 6  # helps small-N stability

# -------- Deep Forest / MGS (practical, paper-aligned spirit) --------
MGS_M = 8  # because student FC1=64 => 8x8
MGS_WINDOWS = [(4, 2), (2, 1)]  # (J, stride S)
DF_MAX_LAYERS = 6
DF_EARLY_STOP_PATIENCE = 2

DF_N_ESTIMATORS = 300
DF_MIN_SAMPLES_LEAF = 1
DF_MAX_FEATURES = "sqrt"

# -------- Visualization control --------
CM_CMAP_2NM = "Blues"
CM_CMAP_4NM = "Oranges"
ROC_COLORS_2NM = ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"]
ROC_COLORS_4NM = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

# -------- CWT cache for BASE ENV (0dB) --------
# To keep runtime feasible across N=5..30 x 10 runs, we cache 0dB CWT images for all loads.
CWT_SCALES = np.arange(1, 65, dtype=np.float32)  # 64 scales
CWT_WAVELET = "morl"
CWT_IMG_SIZE = 64
CWT_CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, "_cache_cwt_base0dB")
os.makedirs(CWT_CACHE_DIR, exist_ok=True)

# Global deterministic seed for cached 0dB noise (per-segment deterministic)
CWT_CACHE_SEED_BASE = 20260215


# ============================================================
# 2) Reproducibility
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _configure_tf_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# ============================================================
# 3) Hash / normalize / noise (numpy)
# ============================================================
def _hash_sample(vib_1x2048: np.ndarray, cur_1x2048: np.ndarray) -> str:
    b = vib_1x2048.astype(np.float32).tobytes() + cur_1x2048.astype(np.float32).tobytes()
    return hashlib.sha1(b).hexdigest()


def zscore_per_segment_np(segs: np.ndarray, eps: float = POST_ZSCORE_EPS) -> np.ndarray:
    segs = np.asarray(segs, dtype=np.float32)
    mu = np.mean(segs, axis=1, keepdims=True)
    sd = np.std(segs, axis=1, keepdims=True) + eps
    return (segs - mu) / sd


def add_gaussian_noise_1d(sig_1d: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    sig = np.asarray(sig_1d, dtype=np.float32).reshape(-1)
    if np.isinf(snr_db):
        return sig
    ps = float(np.mean(sig ** 2) + 1e-12)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    pn = ps / snr_lin
    noise = rng.normal(0.0, math.sqrt(pn), size=sig.shape).astype(np.float32)
    out = sig + noise
    # keep your policy: no post-zscore after noise by default
    if ENABLE_POST_ZSCORE_AFTER_NOISE and ENABLE_ZSCORE_PER_SEGMENT:
        mu = float(np.mean(out))
        sd = float(np.std(out) + POST_ZSCORE_EPS)
        out = (out - mu) / sd
    return out


def estimate_empirical_snr_db(x_clean, x_noisy) -> float:
    x_clean = np.asarray(x_clean, dtype=np.float32)
    x_noisy = np.asarray(x_noisy, dtype=np.float32)
    n = x_noisy - x_clean
    ps = float(np.mean(x_clean ** 2) + 1e-12)
    pn = float(np.mean(n ** 2) + 1e-12)
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
                return np.sqrt(u_[:L] ** 2 + v_[:L] ** 2 + w_[:L] ** 2).astype(np.float32)
            raise RuntimeError(f"[TDMS ERROR] MAG mode needs 3 channels, but not found in: {tdms_path}")
        L = min(len(u), len(v), len(w))
        return np.sqrt(u[:L] ** 2 + v[:L] ** 2 + w[:L] ** 2).astype(np.float32)

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
# 7) Load datasets: vib(.mat) + current(.tdms)  (same as your template)
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


# ============================================================
# 8) Split indices: 0Nm train/val; 2Nm/4Nm test  (NO leakage)
# ============================================================
def build_train_val_indices_from_0Nm(datasets_0, seed: int, n_train: int, n_val: int):
    rng = np.random.RandomState(seed)
    tr_idx_all = []
    va_idx_all = []
    tr_hash, va_hash = set(), set()

    for label, (vib, cur) in enumerate(datasets_0):
        M = vib.shape[0]
        need = n_train + n_val + 1
        assert M >= need, f"[DATA ERROR] class {label} has {M} segments, need >= {need}"
        idx = np.arange(M, dtype=np.int64)
        rng.shuffle(idx)

        tr_idx = idx[:n_train]
        va_idx = idx[n_train:n_train + n_val]

        if ENABLE_HASH_OVERLAP_CHECK:
            for i in tr_idx:
                tr_hash.add(_hash_sample(vib[i], cur[i]))
            for i in va_idx:
                va_hash.add(_hash_sample(vib[i], cur[i]))

        tr_idx_all.append(tr_idx)
        va_idx_all.append(va_idx)

    if ENABLE_HASH_OVERLAP_CHECK:
        assert len(tr_hash.intersection(va_hash)) == 0, "[LEAKAGE ERROR] 0Nm train/val overlap!"

    return tr_idx_all, va_idx_all


# ============================================================
# 9) CWT -> 64x64 image (paper-style, practical downsample)
# ============================================================
def cwt_to_64x64(sig_1d: np.ndarray) -> np.ndarray:
    """
    Paper uses CWT coefficient matrix representation (64x64).
    We compute 64 scales, then downsample time axis to 64 by mean pooling.
    """
    x = np.asarray(sig_1d, dtype=np.float32).reshape(-1)
    coeffs, _freqs = pywt.cwt(x, CWT_SCALES, CWT_WAVELET)
    mat = np.abs(coeffs).astype(np.float32)  # (64, L)

    L = mat.shape[1]
    if L == CWT_IMG_SIZE:
        out = mat
    elif L % CWT_IMG_SIZE == 0:
        factor = L // CWT_IMG_SIZE
        out = mat.reshape(CWT_IMG_SIZE, CWT_IMG_SIZE, factor).mean(axis=2)
    else:
        # fallback: linear interpolate each scale to 64 points
        xs = np.linspace(0, 1, L, endpoint=True)
        xt = np.linspace(0, 1, CWT_IMG_SIZE, endpoint=True)
        out = np.zeros((CWT_IMG_SIZE, CWT_IMG_SIZE), dtype=np.float32)
        for i in range(CWT_IMG_SIZE):
            out[i, :] = np.interp(xt, xs, mat[i, :]).astype(np.float32)

    # per-sample min-max normalize to [0,1]
    mn = float(out.min())
    mx = float(out.max())
    out = (out - mn) / (mx - mn + 1e-8)
    return out.astype(np.float32)


def build_cwt_image_pair(vib_seg: np.ndarray, cur_seg: np.ndarray,
                         snr_db: float, rng_v: np.random.RandomState, rng_c: np.random.RandomState) -> np.ndarray:
    """
    Add noise at snr_db in time domain (per segment), then CWT->64x64, stack channels => (64,64,2)
    """
    vib_n = add_gaussian_noise_1d(vib_seg, snr_db, rng_v)
    cur_n = add_gaussian_noise_1d(cur_seg, snr_db, rng_c)
    img_v = cwt_to_64x64(vib_n)
    img_c = cwt_to_64x64(cur_n)
    img = np.stack([img_v, img_c], axis=-1)  # (64,64,2)
    return img.astype(np.float32)


def _seg_seed(load_tag: str, fault: str, seg_idx: int, seed_base: int) -> int:
    s = f"{seed_base}|{load_tag}|{fault}|{seg_idx}".encode("utf-8")
    h = hashlib.sha1(s).hexdigest()
    return int(h[:8], 16)  # 32-bit


def load_or_build_cwt_cache_for_load(datasets_load, load_tag: str, snr_db: float,
                                     cache_dir: str, seed_base: int) -> List[np.ndarray]:
    """
    Cache ONLY the BASE ENV (0dB) CWT images for all segments in this load.
    Output: list per class; each item: (M,64,64,2) float16
    """
    out = []
    snr_label = "NoNoise" if np.isinf(snr_db) else f"{int(snr_db)}dB"
    for (cls_name, fault), (vib_segs, cur_segs) in zip(FAULTS, datasets_load):
        cache_path = os.path.join(cache_dir, f"cwt_{load_tag}_{fault}_snr{snr_label}.npz")
        if os.path.exists(cache_path):
            arr = np.load(cache_path)["arr"]
            out.append(arr)
            continue

        M = vib_segs.shape[0]
        imgs = np.zeros((M, CWT_IMG_SIZE, CWT_IMG_SIZE, 2), dtype=np.float16)

        print(f"[CWT-CACHE] Building {load_tag}_{fault} | M={M} | SNR={snr_label} ...")
        t0 = time.perf_counter()
        for i in range(M):
            sd = _seg_seed(load_tag, fault, i, seed_base)
            rng_v = np.random.RandomState(sd + 11)
            rng_c = np.random.RandomState(sd + 29)
            img = build_cwt_image_pair(vib_segs[i], cur_segs[i], snr_db, rng_v, rng_c)
            imgs[i] = img.astype(np.float16)
            if (i + 1) % 100 == 0 or (i + 1) == M:
                print(f"  ... {i+1}/{M}")

        t1 = time.perf_counter()
        np.savez_compressed(cache_path, arr=imgs)
        print(f"[CWT-CACHE] Saved: {cache_path} | time={_fmt_sec(t1 - t0)}")
        out.append(imgs)

    return out


# ============================================================
# 10) KDCNN Teacher / Student models (paper table I & II, Student_53)
# ============================================================
def build_teacher_cnn(num_classes: int) -> keras.Model:
    """
    Table I (teacher): Conv1 32@5x5 -> BN -> Max(2)
                      Conv2 32@3x3 -> Max(2)
                      Conv3 64@3x3 -> Max(2)
                      Conv4 64@3x3 -> Max(2)
                      Flatten 1024 -> FC1 128 -> FC2 128 -> logits
    Adaptation: input has 2 channels (vib+current CWT stacked), still 64x64.
    """
    inp = layers.Input(shape=(CWT_IMG_SIZE, CWT_IMG_SIZE, 2), name="img_in")
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)              # ~1024
    x = layers.Dense(128, activation="relu")(x)  # FC1
    feat = layers.Dense(128, activation="relu", name="teacher_fc2")(x)  # FC2
    logits = layers.Dense(num_classes, name="teacher_logits")(feat)
    out = layers.Softmax(name="teacher_softmax")(logits)
    model = keras.Model(inp, out, name="TeacherCNN")
    return model


def build_student_cnn(num_classes: int) -> Tuple[keras.Model, keras.Model]:
    """
    Table II (student_53): Conv1 8@5x5 -> Max(4)
                          Conv2 8@3x3 -> Max(4)
                          Flatten 128 -> FC1 64 -> logits
    Return: (student_model_for_probs, student_feat_model (FC1 features))
    """
    inp = layers.Input(shape=(CWT_IMG_SIZE, CWT_IMG_SIZE, 2), name="img_in")
    x = layers.Conv2D(8, (5, 5), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D((4, 4))(x)  # 64->16

    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((4, 4))(x)  # 16->4

    x = layers.Flatten()(x)  # 8*4*4 = 128
    feat = layers.Dense(64, activation="relu", name="student_fc1")(x)  # FC1 (64)
    logits = layers.Dense(num_classes, name="student_logits")(feat)
    out = layers.Softmax(name="student_softmax")(logits)

    student = keras.Model(inp, out, name="StudentCNN")
    feat_model = keras.Model(inp, feat, name="StudentFeat")
    return student, feat_model


# ============================================================
# 10.1) KD training wrapper (teacher frozen, student trained)
# ============================================================
class KDTrainer(keras.Model):
    def __init__(self, teacher: keras.Model, student: keras.Model, alpha: float, temperature: float):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_tracker = keras.metrics.Mean(name="ce")
        self.kd_tracker = keras.metrics.Mean(name="kd")
        self.acc = keras.metrics.CategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.ce_tracker, self.kd_tracker, self.acc]

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y_true = data
        y_true = tf.cast(y_true, tf.float32)

        # teacher preds (stopgrad)
        y_teacher = tf.stop_gradient(self.teacher(x, training=False))

        with tf.GradientTape() as tape:
            y_student = self.student(x, training=True)

            # Hard-label CE
            ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_student))

            # Soft-label KD (KL) with temperature
            eps = 1e-7
            t = self.temperature
            # Use logits-like by applying log on softmax outputs (stable enough here)
            p_t = tf.clip_by_value(y_teacher, eps, 1.0 - eps)
            p_s = tf.clip_by_value(y_student, eps, 1.0 - eps)

            # temperature smoothing
            p_t_T = tf.nn.softmax(tf.math.log(p_t) / t, axis=-1)
            p_s_T = tf.nn.softmax(tf.math.log(p_s) / t, axis=-1)

            kd = tf.reduce_mean(tf.keras.losses.KLDivergence()(p_t_T, p_s_T)) * (t ** 2)

            loss = self.alpha * ce + (1.0 - self.alpha) * kd

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.ce_tracker.update_state(ce)
        self.kd_tracker.update_state(kd)
        self.acc.update_state(y_true, y_student)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_true = data
        y_true = tf.cast(y_true, tf.float32)
        y_student = self.student(x, training=False)
        ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_student))

        # teacher preds (stopgrad)
        y_teacher = tf.stop_gradient(self.teacher(x, training=False))
        eps = 1e-7
        t = self.temperature
        p_t = tf.clip_by_value(y_teacher, eps, 1.0 - eps)
        p_s = tf.clip_by_value(y_student, eps, 1.0 - eps)
        p_t_T = tf.nn.softmax(tf.math.log(p_t) / t, axis=-1)
        p_s_T = tf.nn.softmax(tf.math.log(p_s) / t, axis=-1)
        kd = tf.reduce_mean(tf.keras.losses.KLDivergence()(p_t_T, p_s_T)) * (t ** 2)

        loss = self.alpha * ce + (1.0 - self.alpha) * kd

        self.loss_tracker.update_state(loss)
        self.ce_tracker.update_state(ce)
        self.kd_tracker.update_state(kd)
        self.acc.update_state(y_true, y_student)

        return {m.name: m.result() for m in self.metrics}


# ============================================================
# 11) Simplified MGS + Cascade Deep Forest (paper spirit)
# ============================================================
def _extract_mgs_features(feat64: np.ndarray, windows: List[Tuple[int, int]]) -> np.ndarray:
    """
    feat64: (N,64) -> reshape (N,8,8), then sliding windows JxJ with stride S,
    flatten patches and concat -> (N, sum_patches * J^2)
    """
    X = np.asarray(feat64, dtype=np.float32)
    assert X.ndim == 2 and X.shape[1] == 64, f"Expected (N,64), got {X.shape}"
    N = X.shape[0]
    M = MGS_M
    X2 = X.reshape(N, M, M)

    feats_all = []
    for (J, S) in windows:
        patches = []
        for r in range(0, M - J + 1, S):
            for c in range(0, M - J + 1, S):
                p = X2[:, r:r + J, c:c + J].reshape(N, J * J)
                patches.append(p)
        feats_all.append(np.concatenate(patches, axis=1))

    return np.concatenate(feats_all, axis=1).astype(np.float32)


class CascadeDeepForest:
    """
    Minimal cascade forest:
    - Each layer has 2 forests (RF + ExtraTrees), outputs 2*C class-proba
    - Next layer input = [original_mgs_features, prev_layer_proba]
    - Early stop by validation accuracy
    """
    def __init__(self,
                 num_classes: int,
                 max_layers: int = DF_MAX_LAYERS,
                 early_stop_patience: int = DF_EARLY_STOP_PATIENCE,
                 n_estimators: int = DF_N_ESTIMATORS,
                 max_features: Any = DF_MAX_FEATURES,
                 min_samples_leaf: int = DF_MIN_SAMPLES_LEAF,
                 random_state: int = 0):
        self.num_classes = int(num_classes)
        self.max_layers = int(max_layers)
        self.early_stop_patience = int(early_stop_patience)
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

        self.layers_: List[Tuple[Any, Any]] = []
        self.best_layer_idx_ = -1
        self.best_val_acc_ = -1.0

    def _make_pair(self, layer_idx: int):
        rs1 = self.random_state + layer_idx * 100 + 1
        rs2 = self.random_state + layer_idx * 100 + 2
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=rs1
        )
        et = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=rs2
        )
        return rf, et

    @staticmethod
    def _proba_concat(rf, et, X):
        p1 = rf.predict_proba(X)
        p2 = et.predict_proba(X)
        return np.concatenate([p1, p2], axis=1).astype(np.float32)

    def fit(self, X_train: np.ndarray, y_train_int: np.ndarray,
            X_val: np.ndarray, y_val_int: np.ndarray, verbose: bool = True):
        X0_tr = np.asarray(X_train, dtype=np.float32)
        X0_va = np.asarray(X_val, dtype=np.float32)
        y_tr = np.asarray(y_train_int, dtype=np.int64)
        y_va = np.asarray(y_val_int, dtype=np.int64)

        self.layers_.clear()
        self.best_layer_idx_ = -1
        self.best_val_acc_ = -1.0

        patience = 0
        cur_tr = X0_tr
        cur_va = X0_va

        for layer_idx in range(self.max_layers):
            rf, et = self._make_pair(layer_idx)
            rf.fit(cur_tr, y_tr)
            et.fit(cur_tr, y_tr)

            proba_va = self._proba_concat(rf, et, cur_va)
            # aggregate proba for prediction (avg of two forests)
            p_avg = 0.5 * (proba_va[:, :self.num_classes] + proba_va[:, self.num_classes:])
            y_pred = np.argmax(p_avg, axis=1)
            val_acc = float(np.mean(y_pred == y_va))

            self.layers_.append((rf, et))

            if verbose:
                print(f"    [DF] layer={layer_idx+1}/{self.max_layers} val_acc={val_acc:.4f}")

            if val_acc > self.best_val_acc_ + 1e-6:
                self.best_val_acc_ = val_acc
                self.best_layer_idx_ = layer_idx
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    if verbose:
                        print(f"    [DF] early stop at layer {layer_idx+1} (best={self.best_layer_idx_+1}, acc={self.best_val_acc_:.4f})")
                    break

            # next layer input
            proba_tr = self._proba_concat(rf, et, cur_tr)
            cur_tr = np.concatenate([X0_tr, proba_tr], axis=1)
            cur_va = np.concatenate([X0_va, proba_va], axis=1)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert len(self.layers_) > 0, "CascadeDeepForest not fitted."
        X0 = np.asarray(X, dtype=np.float32)
        cur = X0
        proba = None
        use_layers = self.best_layer_idx_ + 1 if self.best_layer_idx_ >= 0 else len(self.layers_)

        for i in range(use_layers):
            rf, et = self.layers_[i]
            proba = self._proba_concat(rf, et, cur)
            # next input
            cur = np.concatenate([X0, proba], axis=1)

        p_avg = 0.5 * (proba[:, :self.num_classes] + proba[:, self.num_classes:])
        return p_avg.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.argmax(p, axis=1).astype(np.int64)


# ============================================================
# 12) Visualization (same style as your template)
# ============================================================
def save_curves_teacher_student(hist_t, hist_s, out_path, title_prefix):
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    if hist_t is not None:
        plt.plot(hist_t.history.get("loss", []), label="Teacher-Train")
        plt.plot(hist_t.history.get("val_loss", []), label="Teacher-Val")
    if hist_s is not None:
        plt.plot(hist_s.history.get("loss", []), label="StudentKD-Train")
        plt.plot(hist_s.history.get("val_loss", []), label="StudentKD-Val")
    plt.title(f"Loss - {title_prefix}")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    if hist_t is not None:
        plt.plot(hist_t.history.get("accuracy", []), label="Teacher-Train")
        plt.plot(hist_t.history.get("val_accuracy", []), label="Teacher-Val")
    if hist_s is not None:
        plt.plot(hist_s.history.get("accuracy", []), label="StudentKD-Train")
        plt.plot(hist_s.history.get("val_accuracy", []), label="StudentKD-Val")
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


def save_tsne(feats, y_true, out_path, title_prefix, seed=42):
    feats = np.asarray(feats, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

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
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
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
# 13) Trimmed stats + radar (same as your template)
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


# ============================================================
# 14) Build baseline (0dB) train/val/test images from cached CWT
# ============================================================
def pack_train_val_from_indices(imgs_0: List[np.ndarray],
                                tr_idx_all: List[np.ndarray],
                                va_idx_all: List[np.ndarray],
                                seed: int):
    rng = np.random.RandomState(seed)
    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []

    for label in range(NUM_CLASSES):
        X_tr_list.append(imgs_0[label][tr_idx_all[label]].astype(np.float32))
        X_va_list.append(imgs_0[label][va_idx_all[label]].astype(np.float32))
        y_tr_list.append(np.full((len(tr_idx_all[label]),), label, dtype=np.int64))
        y_va_list.append(np.full((len(va_idx_all[label]),), label, dtype=np.int64))

    X_tr = np.concatenate(X_tr_list, axis=0)
    X_va = np.concatenate(X_va_list, axis=0)
    y_tr = np.concatenate(y_tr_list, axis=0)
    y_va = np.concatenate(y_va_list, axis=0)

    # shuffle
    p_tr = rng.permutation(len(X_tr))
    p_va = rng.permutation(len(X_va))
    X_tr, y_tr = X_tr[p_tr], y_tr[p_tr]
    X_va, y_va = X_va[p_va], y_va[p_va]

    y_tr_oh = np_utils.to_categorical(y_tr, NUM_CLASSES).astype(np.float32)
    y_va_oh = np_utils.to_categorical(y_va, NUM_CLASSES).astype(np.float32)

    return (X_tr, y_tr, y_tr_oh), (X_va, y_va, y_va_oh)


def pack_test_from_cached(imgs_load: List[np.ndarray], seed: int = 0):
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for label in range(NUM_CLASSES):
        X_list.append(imgs_load[label].astype(np.float32))
        y_list.append(np.full((imgs_load[label].shape[0],), label, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    p = rng.permutation(len(X))
    X, y = X[p], y[p]
    y_oh = np_utils.to_categorical(y, NUM_CLASSES).astype(np.float32)
    return X, y, y_oh


# ============================================================
# 15) Evaluate helper (student feat -> MGS -> DF -> metrics + plots)
# ============================================================
def eval_on_test_df(student_feat_model: keras.Model,
                    df_model: CascadeDeepForest,
                    X_te: np.ndarray,
                    y_te_int: np.ndarray,
                    out_dir: str,
                    tag: str,
                    seed: int,
                    roc_colors=None,
                    cm_cmap="Blues"):
    y_true = np.asarray(y_te_int, dtype=np.int64)

    t0 = time.perf_counter()
    feats = student_feat_model.predict(X_te, verbose=0)
    feats = np.asarray(feats, dtype=np.float32)
    X_mgs = _extract_mgs_features(feats, MGS_WINDOWS)

    y_prob = df_model.predict_proba(X_mgs)
    y_pred = np.argmax(y_prob, axis=1)
    t1 = time.perf_counter()
    infer_time_s = float(t1 - t0)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")

    viz_time_s = 0.0
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        t2 = time.perf_counter()

        save_cm(y_true, y_pred, os.path.join(out_dir, f"cm_{tag}.png"),
                f"{tag} | BASE(0dB)", cmap=cm_cmap)
        save_tsne(feats, y_true, os.path.join(out_dir, f"tsne_{tag}.png"),
                  f"{tag} | BASE(0dB)", seed=seed + 7)
        save_roc(y_true, y_prob, os.path.join(out_dir, f"roc_{tag}.png"),
                 f"{tag} | BASE(0dB)", class_colors=roc_colors)

        t3 = time.perf_counter()
        viz_time_s = float(t3 - t2)

    return acc, f1, prec, rec, infer_time_s, viz_time_s


# ============================================================
# 16) Noise robustness eval (compute CWT on-the-fly for SNR points)
# ============================================================
def build_cwt_batch_from_raw_segments(vib_segs: np.ndarray, cur_segs: np.ndarray,
                                      snr_db: float, seed: int) -> np.ndarray:
    """
    vib_segs, cur_segs: (N,2048) clean (already pre-zscored)
    return: (N,64,64,2) float32
    """
    vib_segs = np.asarray(vib_segs, dtype=np.float32)
    cur_segs = np.asarray(cur_segs, dtype=np.float32)
    N = vib_segs.shape[0]
    X = np.zeros((N, CWT_IMG_SIZE, CWT_IMG_SIZE, 2), dtype=np.float32)

    rng_master = np.random.RandomState(seed)
    for i in range(N):
        sd = int(rng_master.randint(0, 2**31 - 1))
        rng_v = np.random.RandomState(sd + 11)
        rng_c = np.random.RandomState(sd + 29)
        X[i] = build_cwt_image_pair(vib_segs[i], cur_segs[i], snr_db, rng_v, rng_c)
        if (i + 1) % 200 == 0 or (i + 1) == N:
            pass
    return X


def eval_noise_point(student_feat_model: keras.Model,
                     df_model: CascadeDeepForest,
                     datasets_load,   # list per class: (vib_segs, cur_segs)
                     snr_db: float,
                     seed: int) -> Tuple[float, float]:
    """
    Evaluate one SNR point on a given load:
    - build noisy CWT on-the-fly from raw segments
    - predict via student->MGS->DF
    returns: (acc, macro_f1)
    """
    # pack raw segments + labels
    vib_list, cur_list, y_list = [], [], []
    for label, (vib, cur) in enumerate(datasets_load):
        vib_list.append(vib)
        cur_list.append(cur)
        y_list.append(np.full((vib.shape[0],), label, dtype=np.int64))
    vib_all = np.concatenate(vib_list, axis=0)
    cur_all = np.concatenate(cur_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)

    # shuffle deterministic
    rng = np.random.RandomState(seed + 1)
    p = rng.permutation(len(y_true))
    vib_all, cur_all, y_true = vib_all[p], cur_all[p], y_true[p]

    X = build_cwt_batch_from_raw_segments(vib_all, cur_all, snr_db, seed=seed + 9)
    feats = student_feat_model.predict(X, verbose=0)
    X_mgs = _extract_mgs_features(feats, MGS_WINDOWS)
    y_prob = df_model.predict_proba(X_mgs)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


# ============================================================
# 17) Stage2 Full experiment (train per N & run)
# ============================================================
def run_full_experiment(DATA_0, DATA_2, DATA_4, IMG_0, IMG_2, IMG_4):
    # Fixed test packs (cached 0dB CWT)
    X2, y2_int, _y2_oh = pack_test_from_cached(IMG_2, seed=123)
    X4, y4_int, _y4_oh = pack_test_from_cached(IMG_4, seed=456)

    # Print model summaries once
    print("\n>>> KDCNN-DF Model Summaries (Teacher / Student_53):")
    print("=" * 60)
    teacher_tmp = build_teacher_cnn(NUM_CLASSES)
    teacher_tmp.summary()
    student_tmp, feat_tmp = build_student_cnn(NUM_CLASSES)
    student_tmp.summary()
    del teacher_tmp, student_tmp, feat_tmp
    print("=" * 60 + "\n")

    # Save fixed parameters
    best_txt = os.path.join(BASE_OUTPUT_DIR, "best_parameters.txt")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("KDCNN-DF baseline (NO Optuna)\n")
        f.write("Base env: 0 dB Gaussian (cached CWT)\n")
        f.write(f"KD_ALPHA: {KD_ALPHA}\n")
        f.write(f"KD_TEMPERATURE: {KD_TEMPERATURE}\n")
        f.write(f"TEACHER_EPOCHS: {TEACHER_EPOCHS}\n")
        f.write(f"STUDENT_EPOCHS: {STUDENT_EPOCHS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LR_TEACHER: {LR_TEACHER}\n")
        f.write(f"LR_STUDENT: {LR_STUDENT}\n")
        f.write(f"MGS_M: {MGS_M} | MGS_WINDOWS: {MGS_WINDOWS}\n")
        f.write(f"DF_N_ESTIMATORS: {DF_N_ESTIMATORS}\n")
        f.write(f"DF_MAX_LAYERS: {DF_MAX_LAYERS} | DF_EARLY_STOP_PATIENCE: {DF_EARLY_STOP_PATIENCE}\n")
    print("[OK] Saved:", best_txt)

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
            tr_idx_all, va_idx_all = build_train_val_indices_from_0Nm(
                DATA_0, seed=seed, n_train=n_train, n_val=n_val
            )
            (X_tr, y_tr_int, y_tr_oh), (X_va, y_va_int, y_va_oh) = pack_train_val_from_indices(
                IMG_0, tr_idx_all, va_idx_all, seed=seed
            )

            tf.keras.backend.clear_session()
            gc.collect()
            _configure_tf_memory_growth()

            # ----- Train Teacher -----
            teacher = build_teacher_cnn(NUM_CLASSES)
            teacher.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR_TEACHER, clipnorm=GRAD_CLIPNORM),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            cb_teacher = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
                )
            ]
            t_fit0 = time.perf_counter()
            hist_t = teacher.fit(
                X_tr, y_tr_oh,
                epochs=TEACHER_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_va, y_va_oh),
                verbose=0,
                callbacks=cb_teacher
            )

            # ----- Train Student with KD -----
            student, student_feat = build_student_cnn(NUM_CLASSES)
            kd = KDTrainer(teacher=teacher, student=student, alpha=KD_ALPHA, temperature=KD_TEMPERATURE)
            kd.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STUDENT, clipnorm=GRAD_CLIPNORM))
            cb_student = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
                )
            ]
            hist_s = kd.fit(
                X_tr, y_tr_oh,
                epochs=STUDENT_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_va, y_va_oh),
                verbose=0,
                callbacks=cb_student
            )
            t_fit1 = time.perf_counter()
            train_time_s = float(t_fit1 - t_fit0)

            # Save curves
            save_curves_teacher_student(hist_t, hist_s, os.path.join(run_dir, "curves.png"),
                                        f"Train0Nm N={n_train} Run={run_idx}")

            # ----- Deep Forest training on Student features (train/val) -----
            tr_feats = student_feat.predict(X_tr, verbose=0)
            va_feats = student_feat.predict(X_va, verbose=0)
            X_tr_mgs = _extract_mgs_features(tr_feats, MGS_WINDOWS)
            X_va_mgs = _extract_mgs_features(va_feats, MGS_WINDOWS)

            df = CascadeDeepForest(num_classes=NUM_CLASSES, random_state=seed)
            print(f"  [N{n_train}-R{run_idx}] Training DF (cascade) ...")
            df.fit(X_tr_mgs, y_tr_int, X_va_mgs, y_va_int, verbose=True)

            # ----- Evaluate on cached BASE(0dB) tests -----
            out2 = os.path.join(run_dir, "Test_2Nm")
            out4 = os.path.join(run_dir, "Test_4Nm")

            acc2, f12, p2, r2, t2_inf_s, t2_viz_s = eval_on_test_df(
                student_feat, df, X2, y2_int, out2, "2Nm", seed,
                roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            acc4, f14, p4, r4, t4_inf_s, t4_viz_s = eval_on_test_df(
                student_feat, df, X4, y4_int, out4, "4Nm", seed,
                roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
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

            # cleanup
            del df, kd, student, student_feat, teacher
            tf.keras.backend.clear_session()
            gc.collect()

        # Trimmed summary per N
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

    raw_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_raw.csv"), index=False, encoding="utf-8-sig")
    trim_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_trimmed.csv"), index=False, encoding="utf-8-sig")
    print("[OK] Saved summary CSVs to:", BASE_OUTPUT_DIR)


# ============================================================
# 18) Stage3 Noise robustness (FAIR SameModel) - train ONCE at N=30 then evaluate SNR points
# ============================================================
def run_noise_study(DATA_0, DATA_2, DATA_4, IMG_0):
    noise_dir = os.path.join(BASE_OUTPUT_DIR, "NoiseStudy_LoadShift_FAIR_SameModel")
    os.makedirs(noise_dir, exist_ok=True)

    # Train NoiseStudy model ONCE (N=30, fixed seed)
    seed = 2026
    set_global_seed(seed)
    _configure_tf_memory_growth()

    n_train = 30
    n_val = min(VAL_PER_CLASS_MAIN, MAX_SEGMENTS_PER_CLASS - n_train - 1)
    tr_idx_all, va_idx_all = build_train_val_indices_from_0Nm(DATA_0, seed=seed, n_train=n_train, n_val=n_val)
    (X_tr, y_tr_int, y_tr_oh), (X_va, y_va_int, y_va_oh) = pack_train_val_from_indices(IMG_0, tr_idx_all, va_idx_all, seed=seed)

    tf.keras.backend.clear_session()
    gc.collect()

    # Teacher
    teacher = build_teacher_cnn(NUM_CLASSES)
    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_TEACHER, clipnorm=GRAD_CLIPNORM),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    cb_teacher = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
    t0 = time.perf_counter()
    _ = teacher.fit(X_tr, y_tr_oh, epochs=TEACHER_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_va, y_va_oh), verbose=0, callbacks=cb_teacher)

    # Student KD
    student, student_feat = build_student_cnn(NUM_CLASSES)
    kd = KDTrainer(teacher=teacher, student=student, alpha=KD_ALPHA, temperature=KD_TEMPERATURE)
    kd.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STUDENT, clipnorm=GRAD_CLIPNORM))
    cb_student = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
    _ = kd.fit(X_tr, y_tr_oh, epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_va, y_va_oh), verbose=0, callbacks=cb_student)

    # Deep forest trained ONCE on N=30 features
    tr_feats = student_feat.predict(X_tr, verbose=0)
    va_feats = student_feat.predict(X_va, verbose=0)
    X_tr_mgs = _extract_mgs_features(tr_feats, MGS_WINDOWS)
    X_va_mgs = _extract_mgs_features(va_feats, MGS_WINDOWS)

    df = CascadeDeepForest(num_classes=NUM_CLASSES, random_state=seed)
    print("[Step3-FAIR] Training DF (cascade) for NoiseStudy_model ...")
    df.fit(X_tr_mgs, y_tr_int, X_va_mgs, y_va_int, verbose=True)

    t1 = time.perf_counter()
    print(f"[Step3-FAIR] NoiseStudy_model trained ONCE. time={_fmt_sec(t1 - t0)}")

    rows = []
    acc2_mean_list, f12_mean_list = [], []
    acc4_mean_list, f14_mean_list = [], []

    for snr_db in TEST_SNR_DB_LIST:
        label = "NoNoise" if (np.isscalar(snr_db) and np.isinf(snr_db)) else str(int(snr_db))

        acc2_buf, f12_buf = [], []
        acc4_buf, f14_buf = [], []

        reps = 1 if (np.isscalar(snr_db) and np.isinf(snr_db)) else EVAL_NOISE_REPEATS
        for rep in range(reps):
            rep_seed2 = seed + 1000 + rep * 17
            rep_seed4 = seed + 2000 + rep * 17

            a2, f2 = eval_noise_point(student_feat, df, DATA_2, snr_db, rep_seed2)
            a4, f4 = eval_noise_point(student_feat, df, DATA_4, snr_db, rep_seed4)

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
            reps
        ])

        acc2_mean_list.append(acc2_m); f12_mean_list.append(f12_m)
        acc4_mean_list.append(acc4_m); f14_mean_list.append(f14_m)

        print(f"[Step3-FAIR] SNR={label} | 2Nm Acc={acc2_m:.4f}±{acc2_s:.4f} F1={f12_m:.4f}±{f12_s:.4f} | "
              f"4Nm Acc={acc4_m:.4f}±{acc4_s:.4f} F1={f14_m:.4f}±{f14_s:.4f} | repeats={reps}")

        if not (np.isscalar(snr_db) and np.isinf(snr_db)):
            # sanity check on small subset (vib only, quick)
            rng = np.random.RandomState(seed + 555)
            v0 = DATA_2[0][0][:256]  # class0 vib subset
            v_clean = v0[0]
            v_noisy = add_gaussian_noise_1d(v_clean, snr_db, rng)
            emp = estimate_empirical_snr_db(v_clean, v_noisy)
            print(f"    [Sanity] empirical SNR≈{emp:.2f} dB (vib sample)")

    out_csv = os.path.join(noise_dir, "Noise_Robustness_LoadShift_SameModel_FAIR.csv")
    pd.DataFrame(rows, columns=[
        "label", "SNR_dB",
        "Acc_2Nm_mean", "Acc_2Nm_std", "F1_2Nm_mean", "F1_2Nm_std",
        "Acc_4Nm_mean", "Acc_4Nm_std", "F1_4Nm_mean", "F1_4Nm_std",
        "eval_repeats"
    ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[Step3-FAIR] Saved:", out_csv)

    plot_radar(TEST_SNR_DB_LIST, acc2_mean_list, f12_mean_list, os.path.join(noise_dir, "radar_2Nm.png"),
               title="2Nm Robustness (FAIR, SameModel)  NoNoise/0/-2..-10 (mean over noise draws)")
    plot_radar(TEST_SNR_DB_LIST, acc4_mean_list, f14_mean_list, os.path.join(noise_dir, "radar_4Nm.png"),
               title="4Nm Robustness (FAIR, SameModel)  NoNoise/0/-2..-10 (mean over noise draws)")

    # cleanup
    del df, kd, student, student_feat, teacher
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# 19) Main
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
        _configure_tf_memory_growth()

        print("========== KDCNN-DF (Baseline) Load Shift | Vib(.mat)+Current(.tdms) ==========")
        print("VIB_MAT_DIR_0NM:", VIB_MAT_DIR_0NM)
        print("VIB_MAT_DIR_2NM:", VIB_MAT_DIR_2NM)
        print("VIB_MAT_DIR_4NM:", VIB_MAT_DIR_4NM)
        print("CUR_TDMS_DIR   :", CUR_TDMS_DIR)
        print("OUTPUT_DIR     :", BASE_OUTPUT_DIR)

        robust_envs = ["NoNoise" if np.isinf(s) else int(s) for s in TEST_SNR_DB_LIST]
        print(f"Base noise: {TRAIN_BASE_SNR_DB}dB | Robust envs={robust_envs}")
        print(f"[FIX] post-zscore(after noise)={ENABLE_POST_ZSCORE_AFTER_NOISE}  (should be False for meaningful SNR sweep)")
        print(f"[KD] alpha={KD_ALPHA} | T={KD_TEMPERATURE} | teacher_epochs={TEACHER_EPOCHS} | student_epochs={STUDENT_EPOCHS}")
        print(f"[DF] MGS_M={MGS_M} windows={MGS_WINDOWS} | estimators={DF_N_ESTIMATORS} | max_layers={DF_MAX_LAYERS}")
        print(f"[EVAL] repeats per SNR={EVAL_NOISE_REPEATS}")
        print("[LOG] Writing terminal output to:", log_path)

        # -------- Load raw segments (zscored) --------
        print("\n>>> Loading Vibration(.mat)+Current(.tdms) ...")
        DATA_0 = load_dataset_for_load("0Nm", VIB_MAT_DIR_0NM)
        DATA_2 = load_dataset_for_load("2Nm", VIB_MAT_DIR_2NM)
        DATA_4 = load_dataset_for_load("4Nm", VIB_MAT_DIR_4NM)
        print_dataset_stats("0Nm", DATA_0)
        print_dataset_stats("2Nm", DATA_2)
        print_dataset_stats("4Nm", DATA_4)
        print("Data loaded.\n")

        # -------- Build/Load cached BASE(0dB) CWT images (critical for runtime) --------
        print(">>> Building/Loading BASE(0dB) CWT caches (all loads, all segments) ...")
        t_c0 = time.perf_counter()
        IMG_0 = load_or_build_cwt_cache_for_load(DATA_0, "0Nm", TRAIN_BASE_SNR_DB, CWT_CACHE_DIR, CWT_CACHE_SEED_BASE)
        IMG_2 = load_or_build_cwt_cache_for_load(DATA_2, "2Nm", TRAIN_BASE_SNR_DB, CWT_CACHE_DIR, CWT_CACHE_SEED_BASE)
        IMG_4 = load_or_build_cwt_cache_for_load(DATA_4, "4Nm", TRAIN_BASE_SNR_DB, CWT_CACHE_DIR, CWT_CACHE_SEED_BASE)
        t_c1 = time.perf_counter()
        print(f"[CWT-CACHE] Done. time={_fmt_sec(t_c1 - t_c0)}")

        # -------- Stage2 --------
        print("\n>>> Stage 2: Full experiment (Train 0Nm@BASE(0dB) | Test 2Nm&4Nm@BASE(0dB)) ...")
        t_s2_0 = time.perf_counter()
        run_full_experiment(DATA_0, DATA_2, DATA_4, IMG_0, IMG_2, IMG_4)
        t_s2_1 = time.perf_counter()
        stage2_s = float(t_s2_1 - t_s2_0)
        print(f"[Time] Stage2(FullExp) = {_fmt_sec(stage2_s)}")

        # -------- Stage3 --------
        print("\n>>> Stage 3: Noise robustness study (FAIR, SameModel, mean±std over noise draws) ...")
        t_s3_0 = time.perf_counter()
        run_noise_study(DATA_0, DATA_2, DATA_4, IMG_0)
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
        print("\n[FATAL] Exception:", type(e).__name__, e)
        traceback.print_exc()

    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f_log.close()


if __name__ == "__main__":
    main()
