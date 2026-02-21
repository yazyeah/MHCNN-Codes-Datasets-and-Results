"""
CDTFAFN — KAIST Load-Shift Case Study
Vibration (.mat) + Current (.tdms) | Segment length L=2048

[Task / Setting]
- Train/Val: 0Nm only (hash-overlap leakage check enabled)
- Test: 2Nm & 4Nm
- Few-shot protocol: N = 5..30 samples/class, repeated runs = 10
- Seed rule: seed = N*100 + run_idx
- Segments per class cap: 400 (per load)

[Network (paper-consistent, CDTFAFN)]
- SIEU: end-to-end time-frequency mapping (STFT -> optional mel/CQT-like -> log -> minmax -> resize 64x64)
- 3-channel TF image: [static, delta, delta-delta]
- Coarse feature block (per modality): Conv5x5(stride2,64)+BN+ReLU -> Conv3x3(32)
- Coarse fusion: Add(F_v, F_c)
- Two fine branches from fused coarse map:
  - Branch-1: k=3 Conv blocks + TFAFFU (with MSCAM)
  - Branch-2: k=5 Conv blocks + TFAFFU (with MSCAM)
- Final fusion: Concat([F1, F2, F1p, F2p]) -> Flatten -> FC(128) -> Dropout -> Softmax(NUM_CLASSES)

[Noise / Normalization Policy]
- Baseline noise environment: 0 dB Gaussian applied to train/val/test (Stage2 baseline)
- Robustness Stage3 (FAIR, SameModel): train ONE NoiseStudy_model once, then evaluate:
  NoNoise + 0 + (-2,-4,-6,-8,-10), report mean±std over repeated noise draws
- IMPORTANT: ENABLE_POST_ZSCORE_AFTER_NOISE MUST remain False to preserve SNR semantics.

[Implementation Notes]
- Fix: TFAFFU frequency/time attention broadcast bug (keep rank-4 weights; avoid rank explosion)
- Speed optimizations (do NOT change protocol/outputs):
  (1) One-pass inference model outputs [prob, feat]
  (2) t-SNE uses PCA pre-reduction (same points, same saved figure type)
  (3) Strong cleanup: del arrays + plt.close('all') + clear_session

[Outputs]
- log.txt (tee stdout/stderr)
- curves.png per run
- Confusion matrix / t-SNE / ROC per test load (2Nm, 4Nm)
- Final_Summary_Stats_raw.csv + Final_Summary_Stats_trimmed.csv
- NoiseStudy_LoadShift_FAIR_SameModel/: CSV + radar plots + baseline plots

[Open-source friendly paths]
- This script does NOT assume any fixed folder structure.
- Replace the placeholders in the PATHS section (YOUR_*) with your local absolute paths.
"""

import os, sys, gc, time, random, hashlib, warnings
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
from scipy.io import loadmat

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (
    Dense, Input, Dropout, Flatten,
    Conv2D, BatchNormalization, Activation,
    Concatenate, Add
)
from keras.utils import np_utils

from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, auc, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

try:
    from nptdms import TdmsFile
except ImportError as e:
    raise ImportError("Missing dependency: nptdms. Please run: pip install nptdms") from e


# ---------------------------
# Plot style (aligned)
# ---------------------------
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------
# TF GPU config (stability + avoid fragmentation)
# ---------------------------
def _setup_tf_runtime():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    pass
    except Exception:
        pass

_setup_tf_runtime()


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


# ============================================================
# [FIX] NaN root-cause guard helpers (singleton tail batch)
# ============================================================
def get_safe_batch_size(n_samples: int, preferred_bs: int, phase_tag: str = "") -> int:
    """
    Avoid the dangerous case: last batch size == 1 (n_samples % batch_size == 1),
    which can destabilize BatchNorm / custom normalization modules in few-shot training.
    Keeps all samples (does NOT drop remainder).
    """
    n_samples = int(n_samples)
    preferred_bs = int(preferred_bs)

    if n_samples <= 1:
        return 1
    if preferred_bs < 2:
        return max(1, min(preferred_bs, n_samples))

    # Do not exceed dataset size
    bs = min(preferred_bs, n_samples)

    # Safe directly
    if (n_samples % bs) != 1:
        return bs

    # Try smaller sizes first (minimal behavior change)
    for cand in range(bs - 1, 1, -1):
        if (n_samples % cand) != 1:
            print(f"[SAFE-BS] {phase_tag} n_samples={n_samples}, preferred_bs={preferred_bs} "
                  f"-> use batch_size={cand} (avoid remainder=1)")
            return cand

    # Fallback (very rare)
    print(f"[SAFE-BS][WARN] {phase_tag} fallback to batch_size=2 for n_samples={n_samples}")
    return 2


def warn_if_singleton_tail_batch(n_samples: int, batch_size: int, phase_tag: str = ""):
    r = int(n_samples) % int(batch_size)
    if r == 1:
        print(f"[WARN][TAIL-1] {phase_tag}: n_samples={n_samples}, batch_size={batch_size}, "
              f"remainder=1 -> may cause NaN with BN/custom norm. A safe batch size will be used.")


# ============================================================
# 1) Config (A-G aligned)
# ============================================================
CUR_MODE = "U"  # "U","V","W","MAG"
DEBUG_PRINT_TDMS_CHANNELS = False

SAMPLE_RANGE = range(5,31)
REPEAT_TIMES = 10
EPOCHS = 80
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
BASE_OUTPUT_DIR = r"YOUR_OUTPUT_DIR"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

TSNE_MAX_POINTS = 2000

# t-SNE speed knobs
TSNE_PCA_DIM = 50          # PCA pre-reduction (same points)
TSNE_N_ITER = 750          # faster than 1000, still stable (change only visualization embedding)
TSNE_PERPLEXITY = 30.0

ENABLE_HASH_OVERLAP_CHECK = True
VAL_PER_CLASS_MAIN = 50

# -------- Noise settings --------
TRAIN_BASE_SNR_DB = 0
NO_NOISE_DB = float("inf")
TEST_SNR_DB_LIST = [NO_NOISE_DB, 0, -2, -4, -6, -8, -10]

ENABLE_ZSCORE_PER_SEGMENT = True
ENABLE_POST_ZSCORE_AFTER_NOISE = False
POST_ZSCORE_EPS = 1e-8

# -------- Stage3 NoiseStudy (FAIR SameModel) --------
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

ENABLE_TTA = False
TTA_K = 3

EVAL_NOISE_REPEATS = 5
EVAL_TTA_PER_REPEAT = 1

GRAD_CLIPNORM = 1.0
BASELINE_LR = 0.0004156294449523281
BASELINE_BATCH_SIZE = 16

ENABLE_CONSISTENCY = True
CONSIST_MAX_LAMBDA = 0.12
CONSIST_WARMUP_EPOCHS = 15
CONSIST_SNR_LOW_DB = -10.0
CONSIST_SNR_HIGH_DB = 0.0
CONSIST_SNR_BIAS_P = 2.0
CONSIST_USE_PER_SAMPLE_SNR = True

ENABLE_MODALITY_CORRUPT = True
MOD_CORRUPT_PROB = 0.08
MOD_CORRUPT_MODE = "extra_noise"
MOD_CORRUPT_EXTRA_SNR_DB = -12.0

CM_CMAP_2NM = "Blues"
CM_CMAP_4NM = "Oranges"
ROC_COLORS_2NM = ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"]
ROC_COLORS_4NM = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

# ============================================================
# SIEU (CQT-like) parameters
# ============================================================
SIEU_MODE = "mel"            # "mel" (CQT-like) or "stft"
SIEU_OUT_HW = 64
SIEU_ADD_DELTAS_3CH = True   # output 64x64x3 via [static, delta, delta-delta]

# IMPORTANT: set this to the real sampling rate if you know it
SIEU_SAMPLE_RATE_HZ = 25600

SIEU_FRAME_LENGTH = 128
SIEU_FRAME_STEP = 64
SIEU_FFT_LENGTH = 256
SIEU_N_MELS = 64
SIEU_FMIN_HZ = 20.0
SIEU_FMAX_HZ = None          # None => sr/2
SIEU_POWER = 2.0             # power spectrogram


# ============================================================
# 2) Reproducibility
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# 3) Hash / normalize
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
# 3.1) TF noise helpers
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

def _tf_sample_snr_db_custom(batch_size, low_db, high_db, bias_p, per_sample=True):
    low_db = float(low_db); high_db = float(high_db); p = float(bias_p)
    if per_sample:
        u = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
        u_b = tf.pow(u, p)
        return low_db + (high_db - low_db) * u_b
    u = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)
    u_b = tf.pow(u, p)
    return low_db + (high_db - low_db) * u_b

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
# 4) Vibration .mat parsing
# ============================================================
def _extract_values_matrix(mat_dict: dict) -> np.ndarray:
    if "Signal" not in mat_dict:
        keys = [k for k in mat_dict.keys() if not k.startswith("__")]
        raise KeyError(f"Missing 'Signal'. Available keys: {keys[:10]}")
    sig = mat_dict["Signal"]
    yv = sig["y_values"][0, 0]
    vals = yv["values"][0, 0]
    vals = np.asarray(vals)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    return vals

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
# 7) Load datasets
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


print("Loading Vibration(.mat)+Current(.tdms) ...")
DATA_0 = load_dataset_for_load("0Nm", VIB_MAT_DIR_0NM)
DATA_2 = load_dataset_for_load("2Nm", VIB_MAT_DIR_2NM)
DATA_4 = load_dataset_for_load("4Nm", VIB_MAT_DIR_4NM)
print_dataset_stats("0Nm", DATA_0)
print_dataset_stats("2Nm", DATA_2)
print_dataset_stats("4Nm", DATA_4)
print("Data loaded.\n")


# ============================================================
# 8) Split
# ============================================================
def build_train_val_from_0Nm(datasets_0, seed: int, n_train: int, n_val: int):
    rng = np.random.RandomState(seed)
    tr_list, va_list = [], []
    tr_hash, va_hash = set(), set()

    for label, (vib, cur) in enumerate(datasets_0):
        M = vib.shape[0]
        need = n_train + n_val + 1
        assert M >= need, f"[DATA ERROR] class {label} has {M} segments, need >= {need}"

        combined = np.hstack([vib, cur])
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
# 9) CDTFAFN model (ONLY PART THAT DIFFERS)
# ============================================================
class SIEULayer(keras.layers.Layer):
    """
    End-to-end SIEU:
      1) STFT (fast)
      2) Optional mel filterbank (CQT-like, log-frequency)
      3) log + per-sample minmax
      4) resize to 64x64
      5) output 3ch: [static, delta, delta-delta]
    """
    def __init__(self,
                 mode="mel",
                 out_hw=64,
                 sample_rate=25600,
                 frame_length=128,
                 frame_step=64,
                 fft_length=256,
                 n_mels=64,
                 fmin=20.0,
                 fmax=None,
                 power=2.0,
                 add_deltas_3ch=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.mode = str(mode).lower()
        self.out_hw = int(out_hw)
        self.sample_rate = float(sample_rate)
        self.frame_length = int(frame_length)
        self.frame_step = int(frame_step)
        self.fft_length = int(fft_length)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = None if fmax is None else float(fmax)
        self.power = float(power)
        self.add_deltas_3ch = bool(add_deltas_3ch)

        self._mel_w = None

    def build(self, input_shape):
        if self.mode == "mel":
            num_spec_bins = self.fft_length // 2 + 1
            fmax = (self.sample_rate / 2.0) if (self.fmax is None) else self.fmax
            mel_w = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.n_mels,
                num_spectrogram_bins=num_spec_bins,
                sample_rate=self.sample_rate,
                lower_edge_hertz=self.fmin,
                upper_edge_hertz=fmax
            )
            mel_np = mel_w.numpy().astype(np.float32)
            self._mel_w = self.add_weight(
                name="mel_w",
                shape=mel_np.shape,
                initializer=tf.constant_initializer(mel_np),
                trainable=False,
            )
        super().build(input_shape)

    def call(self, x):
        x = tf.squeeze(tf.cast(x, tf.float32), axis=-1)  # (B,2048)
        stft = tf.signal.stft(
            x,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=tf.signal.hann_window,
            pad_end=False
        )
        mag = tf.abs(stft)
        spec = tf.pow(mag + 1e-12, self.power)  # (B,T,F)

        if self.mode == "mel":
            mel = tf.tensordot(spec, self._mel_w, axes=1)  # (B,T,M)
            mel.set_shape([None, None, self.n_mels])
            feat = mel
        else:
            feat = spec

        feat = tf.transpose(feat, perm=[0, 2, 1])  # (B,F,T)
        feat = tf.math.log(feat + 1e-7)

        mn = tf.reduce_min(feat, axis=[1, 2], keepdims=True)
        mx = tf.reduce_max(feat, axis=[1, 2], keepdims=True)
        feat = (feat - mn) / (mx - mn + 1e-8)

        feat = feat[..., tf.newaxis]  # (B,F,T,1)
        feat = tf.image.resize(feat, [self.out_hw, self.out_hw], method="bilinear")

        base = feat
        d1 = base[:, :, 1:, :] - base[:, :, :-1, :]
        d1 = tf.pad(d1, [[0, 0], [0, 0], [0, 1], [0, 0]])
        d2 = d1[:, :, 1:, :] - d1[:, :, :-1, :]
        d2 = tf.pad(d2, [[0, 0], [0, 0], [0, 1], [0, 0]])

        if self.add_deltas_3ch:
            return tf.concat([base, d1, d2], axis=-1)  # (B,64,64,3)
        return tf.tile(base, [1, 1, 1, 3])


class MSCAM(keras.layers.Layer):
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = int(max(1, reduction))

    def build(self, input_shape):
        C = int(input_shape[-1])
        Cr = max(1, C // self.reduction)

        self.l_pw1 = Conv2D(Cr, kernel_size=1, padding="same", use_bias=False)
        self.l_bn1 = BatchNormalization()
        self.l_pw2 = Conv2D(C, kernel_size=1, padding="same", use_bias=False)
        self.l_bn2 = BatchNormalization()

        self.g_pw1 = Conv2D(Cr, kernel_size=1, padding="same", use_bias=False)
        self.g_bn1 = BatchNormalization()
        self.g_pw2 = Conv2D(C, kernel_size=1, padding="same", use_bias=False)
        self.g_bn2 = BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=False):
        l = self.l_pw1(x)
        l = self.l_bn1(l, training=training)
        l = tf.nn.relu(l)
        l = self.l_pw2(l)
        l = self.l_bn2(l, training=training)

        g = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        g = self.g_pw1(g)
        g = self.g_bn1(g, training=training)
        g = tf.nn.relu(g)
        g = self.g_pw2(g)
        g = self.g_bn2(g, training=training)

        return tf.nn.sigmoid(l + g)


class TFAFFU(keras.layers.Layer):
    """
    FIXED: _freq_att/_time_att broadcast must keep rank=4 weights.
    """
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = int(max(1, reduction))

    def build(self, input_shape):
        C = int(input_shape[-1])
        Cr = max(1, C // self.reduction)

        self.cam_fc1 = Dense(Cr, activation="relu", use_bias=True)
        self.cam_fc2 = Dense(C, activation=None, use_bias=True)

        self.fam_conv = keras.layers.Conv1D(1, kernel_size=7, padding="same", use_bias=True)
        self.tam_conv = keras.layers.Conv1D(1, kernel_size=7, padding="same", use_bias=True)

        self.mscam1 = MSCAM(reduction=self.reduction)
        self.mscam2 = MSCAM(reduction=self.reduction)

        self.out_pw = Conv2D(C, kernel_size=1, padding="same", use_bias=False)
        self.out_bn = BatchNormalization()
        super().build(input_shape)

    def _cam(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2])
        mx = tf.reduce_max(x, axis=[1, 2])
        avg_m = self.cam_fc2(self.cam_fc1(avg))
        mx_m = self.cam_fc2(self.cam_fc1(mx))
        w = tf.nn.sigmoid(avg_m + mx_m)[:, tf.newaxis, tf.newaxis, :]
        return x * w

    def _freq_att(self, x):
        avg = tf.reduce_mean(x, axis=[2, 3])  # (B,H)
        mx = tf.reduce_max(x, axis=[2, 3])    # (B,H)
        s = tf.concat([avg[..., tf.newaxis], mx[..., tf.newaxis]], axis=-1)  # (B,H,2)
        w = tf.nn.sigmoid(self.fam_conv(s))  # (B,H,1)
        w = tf.expand_dims(w, axis=2)        # (B,H,1,1)
        return x * w

    def _time_att(self, x):
        avg = tf.reduce_mean(x, axis=[1, 3])  # (B,W)
        mx = tf.reduce_max(x, axis=[1, 3])    # (B,W)
        s = tf.concat([avg[..., tf.newaxis], mx[..., tf.newaxis]], axis=-1)  # (B,W,2)
        w = tf.nn.sigmoid(self.tam_conv(s))  # (B,W,1)
        w = tf.expand_dims(w, axis=1)        # (B,1,W,1)
        return x * w

    def _aff(self, f, t, training=False):
        w1 = self.mscam1(f + t, training=training)
        f1 = w1 * f
        t1 = (1.0 - w1) * t

        w2 = self.mscam2(f1 + t1, training=training)
        return w2 * f1 + (1.0 - w2) * t1

    def call(self, x, training=False):
        fc = self._cam(x)
        ff = self._freq_att(fc)
        ft = self._time_att(fc)
        fused = self._aff(ff, ft, training=training)
        y = self.out_pw(fused)
        y = self.out_bn(y, training=training)
        return tf.nn.relu(y)


def create_model():
    input_v = Input(shape=(DATA_POINTS, 1), name="vib_input")
    input_c = Input(shape=(DATA_POINTS, 1), name="cur_input")

    sieu = SIEULayer(
        mode=SIEU_MODE,
        out_hw=SIEU_OUT_HW,
        sample_rate=SIEU_SAMPLE_RATE_HZ,
        frame_length=SIEU_FRAME_LENGTH,
        frame_step=SIEU_FRAME_STEP,
        fft_length=SIEU_FFT_LENGTH,
        n_mels=SIEU_N_MELS,
        fmin=SIEU_FMIN_HZ,
        fmax=SIEU_FMAX_HZ,
        power=SIEU_POWER,
        add_deltas_3ch=SIEU_ADD_DELTAS_3CH,
        name="SIEU"
    )

    tfr_v = sieu(input_v)  # (64,64,3)
    tfr_c = sieu(input_c)

    def coarse_block(x, name_prefix):
        x = Conv2D(64, kernel_size=5, strides=2, padding="same", use_bias=False, name=f"{name_prefix}_conv5")(x)
        x = BatchNormalization(name=f"{name_prefix}_bn5")(x)
        x = Activation("relu", name=f"{name_prefix}_relu5")(x)
        x = Conv2D(32, kernel_size=3, strides=1, padding="same", use_bias=True, name=f"{name_prefix}_conv3")(x)
        return x  # (32,32,32)

    F1 = coarse_block(tfr_v, "coarse_v")
    F2 = coarse_block(tfr_c, "coarse_c")
    Ff1 = Add(name="coarse_fuse_add")([F1, F2])

    def feature_learning_branch(x, ksize, name_prefix):
        filters = [32, 16, 8]
        out = x
        for i, f in enumerate(filters, start=1):
            out = Conv2D(f, kernel_size=ksize, strides=1, padding="same", use_bias=False,
                         name=f"{name_prefix}_conv_{i}")(out)
            out = BatchNormalization(name=f"{name_prefix}_bn_{i}")(out)
            out = Activation("relu", name=f"{name_prefix}_relu_{i}")(out)
            out = TFAFFU(reduction=4, name=f"{name_prefix}_tfaffu_{i}")(out)
        return out  # (32,32,8)

    F1p = feature_learning_branch(Ff1, ksize=3, name_prefix="fine_b1_k3")
    F2p = feature_learning_branch(Ff1, ksize=5, name_prefix="fine_b2_k5")

    Ff2 = Concatenate(axis=-1, name="final_fuse_concat")([F1, F2, F1p, F2p])  # (32,32,80)
    x = Flatten(name="flatten")(Ff2)

    feat = Dense(128, activation="relu", name="fc1_128")(x)
    feat = Dropout(0.25, name="drop_fc1")(feat)
    out = Dense(NUM_CLASSES, activation="softmax", name="softmax")(feat)

    base_model = Model(inputs=[input_v, input_c], outputs=out, name="CDTFAFN")
    # SPEED: one-pass inference outputs both prob and feat
    infer_model = Model(inputs=[input_v, input_c], outputs=[out, feat], name="CDTFAFN_infer")
    return base_model, infer_model


# ============================================================
# 9.1) Trainer
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
        lam = self.max_lambda * (float(epoch + 1) / float(self.warmup_epochs)) if epoch < self.warmup_epochs else self.max_lambda
        self.model.cons_lambda.assign(lam)


# ============================================================
# 11) Visualization
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
    plt.close("all")

def save_cm(y_true, y_pred, out_path, title_prefix, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(f"Confusion Matrix - {title_prefix}")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, [f"{i}" for i in range(NUM_CLASSES)], rotation=0)
    plt.yticks(tick_marks, [f"{i}" for i in range(NUM_CLASSES)], rotation=0)

    thresh = cm.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    plt.close("all")

def save_tsne_from_feats(feats, y_true, out_path, title_prefix, seed=42):
    feats = np.asarray(feats, dtype=np.float32)
    n = feats.shape[0]
    if n > TSNE_MAX_POINTS:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=TSNE_MAX_POINTS, replace=False)
        feats = feats[idx]
        y_vis = y_true[idx]
    else:
        y_vis = y_true

    # PCA pre-reduction (speed, no reduction in points/outputs)
    if TSNE_PCA_DIM is not None and int(TSNE_PCA_DIM) > 0 and feats.shape[1] > int(TSNE_PCA_DIM):
        pca_dim = int(min(TSNE_PCA_DIM, feats.shape[1]))
        feats = PCA(n_components=pca_dim, random_state=seed).fit_transform(feats)

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        n_iter=int(TSNE_N_ITER),
        perplexity=float(TSNE_PERPLEXITY),
        method="barnes_hut",
        angle=0.5
    ).fit_transform(feats)

    plt.figure(figsize=(6, 5))
    # fast scatter by class
    for c in range(NUM_CLASSES):
        idx = (y_vis == c)
        if np.any(idx):
            plt.scatter(tsne[idx, 0], tsne[idx, 1], s=10, label=str(c), alpha=0.85)
    plt.title(f"t-SNE - {title_prefix}")
    plt.legend(loc="best", fontsize="small", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    plt.close("all")

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

    if class_colors is None:
        class_colors = ROC_COLORS_2NM

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg (AUC={roc_auc['micro']:.4f})", linestyle=":", lw=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg (AUC={roc_auc['macro']:.4f})", linestyle=":", lw=3)

    for i in range(NUM_CLASSES):
        c = class_colors[i % len(class_colors)]
        plt.plot(fpr[i], tpr[i], color=c, lw=2, label=f"Class {i} (AUC={roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title_prefix}")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    plt.close("all")


# ============================================================
# 12) Trimmed stats + radar
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
    plt.close("all")


# ============================================================
# [FIX] 12.1) Non-finite quick intercept
# ============================================================
def has_non_finite(arr) -> bool:
    arr = np.asarray(arr)
    return (not np.all(np.isfinite(arr)))


# ============================================================
# 13) Evaluate helper
# ============================================================
def eval_on_test(infer_model, x_te_v, x_te_c, y_te,
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
        y_prob, feats = infer_model.predict([x_tv, x_tc], verbose=0)
    else:
        Kk = int(tta_k)
        prob_sum = None
        feat_sum = None
        for kk in range(Kk):
            rng = np.random.RandomState(seed + 999 + kk * 131)
            x_tv = add_noise_then_optional_post_zscore_np(x_te_v, snr_db, rng)
            x_tc = add_noise_then_optional_post_zscore_np(x_te_c, snr_db, rng)
            p, f = infer_model.predict([x_tv, x_tc], verbose=0)
            p = p.astype(np.float64)
            f = f.astype(np.float64)
            prob_sum = p if prob_sum is None else (prob_sum + p)
            feat_sum = f if feat_sum is None else (feat_sum + f)
        y_prob = (prob_sum / float(Kk)).astype(np.float32)
        feats = (feat_sum / float(Kk)).astype(np.float32)

    # -------- [FIX] non-finite quick intercept: fail fast (do NOT nan_to_num) --------
    title_snr_dbg = "NoNoise" if (snr_db is None or (np.isscalar(snr_db) and np.isinf(snr_db))) else str(snr_db)
    run_tag_dbg = f"{tag}|SNR={title_snr_dbg}|seed={seed}"
    if has_non_finite(y_prob) or has_non_finite(feats):
        print(f"[ERROR][NONFINITE] Detected NaN/Inf in outputs. Mark this run as invalid. ({run_tag_dbg})")
        raise FloatingPointError("Non-finite outputs detected.")

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
        save_tsne_from_feats(feats, y_true, os.path.join(out_dir, f"tsne_{tag}.png"),
                             f"{tag} | SNR={title_snr}", seed=seed + 7)
        save_roc(y_true, y_prob, os.path.join(out_dir, f"roc_{tag}.png"),
                 f"{tag} | SNR={title_snr}", class_colors=roc_colors)

        t_viz1 = time.perf_counter()
        viz_time_s = float(t_viz1 - t_viz0)

    # cleanup big arrays (does NOT change outputs)
    del y_prob, feats
    try:
        del x_tv, x_tc
    except Exception:
        pass
    gc.collect()
    plt.close("all")

    return acc, f1, prec, rec, infer_time_s, viz_time_s


# ============================================================
# 14) Full experiment (Stage2)
# ============================================================
def run_full_experiment():
    best_lr = BASELINE_LR
    best_bs = BASELINE_BATCH_SIZE

    x2_v, x2_c, y2 = build_test_from_load(DATA_2, seed=123)
    x4_v, x4_c, y4 = build_test_from_load(DATA_4, seed=456)

    print("\n>>> CDTFAFN Model Architecture Summary:")
    print("=" * 60)
    tmp_model, _ = create_model()
    tmp_model.summary()
    del tmp_model
    tf.keras.backend.clear_session()
    gc.collect()
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

            tf.keras.backend.clear_session()
            gc.collect()

            base_model, infer_model = create_model()
            trainer = NoiseStableTrainer(base_model, base_snr_db=TRAIN_BASE_SNR_DB, name="NoiseStableTrainer")
            opt = tf.keras.optimizers.Adamax(learning_rate=best_lr, clipnorm=GRAD_CLIPNORM)
            trainer.compile(optimizer=opt, metrics=["accuracy"])
            callbacks = [ConsistencyLambdaScheduler(CONSIST_MAX_LAMBDA, CONSIST_WARMUP_EPOCHS)]

            # [FIX] Safe batch size for few-shot train samples (avoid tail batch == 1)
            n_train_samples = int(len(y_tr))  # balanced 5 classes => 5 * n_train
            warn_if_singleton_tail_batch(n_train_samples, best_bs, phase_tag=f"Stage2|N={n_train}|Run={run_idx}")
            effective_bs = get_safe_batch_size(
                n_samples=n_train_samples,
                preferred_bs=best_bs,
                phase_tag=f"Stage2|N={n_train}|Run={run_idx}"
            )

            t_fit0 = time.perf_counter()
            history = trainer.fit(
                [x_tr_v, x_tr_c], y_tr,
                epochs=EPOCHS,
                batch_size=effective_bs,
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
                infer_model, x2_v, x2_c, y2, out2, "2Nm", TRAIN_BASE_SNR_DB, seed,
                tta_k=1, roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            acc4, f14, p4, r4, t4_inf_s, t4_viz_s = eval_on_test(
                infer_model, x4_v, x4_c, y4, out4, "4Nm", TRAIN_BASE_SNR_DB, seed,
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

            # cleanup
            del history
            del trainer, base_model, infer_model
            del x_tr_v, x_tr_c, y_tr, x_va_v, x_va_c, y_va
            tf.keras.backend.clear_session()
            gc.collect()
            plt.close("all")

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


# ============================================================
# 15) Noise robustness study (Stage3) - FAIR SameModel
# ============================================================
def run_noise_study():
    noise_dir = os.path.join(BASE_OUTPUT_DIR, "NoiseStudy_LoadShift_FAIR_SameModel")
    os.makedirs(noise_dir, exist_ok=True)

    best_lr = BASELINE_LR
    best_bs = BASELINE_BATCH_SIZE

    x2_v, x2_c, y2 = build_test_from_load(DATA_2, seed=123)
    x4_v, x4_c, y4 = build_test_from_load(DATA_4, seed=456)

    seed = 2026
    set_global_seed(seed)

    n_train = 30
    n_val = min(VAL_PER_CLASS_MAIN, MAX_SEGMENTS_PER_CLASS - n_train - 1)
    (x_tr_v, x_tr_c, y_tr), (x_va_v, x_va_c, y_va) = build_train_val_from_0Nm(
        DATA_0, seed=seed, n_train=n_train, n_val=n_val
    )

    tf.keras.backend.clear_session()
    gc.collect()

    base_model, infer_model = create_model()

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

    opt = tf.keras.optimizers.Adamax(learning_rate=best_lr, clipnorm=GRAD_CLIPNORM)
    trainer.compile(optimizer=opt, metrics=["accuracy"])
    callbacks = [ConsistencyLambdaScheduler(STEP3_CONSIST_MAX_LAMBDA, STEP3_CONSIST_WARMUP_EPOCHS)]

    # [FIX] Safe batch size for Stage3 training (uniform guard, even if usually not triggered)
    n_stage3_train = int(len(y_tr))
    warn_if_singleton_tail_batch(n_stage3_train, best_bs, phase_tag="Stage3|NoiseStudyTrain")
    effective_bs_stage3 = get_safe_batch_size(
        n_samples=n_stage3_train,
        preferred_bs=best_bs,
        phase_tag="Stage3|NoiseStudyTrain"
    )

    t_fit0 = time.perf_counter()
    _ = trainer.fit(
        [x_tr_v, x_tr_c], y_tr,
        epochs=STEP3_EPOCHS,
        batch_size=effective_bs_stage3,
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

        acc2_buf, f12_buf, acc4_buf, f14_buf = [], [], [], []

        for rep in range(EVAL_NOISE_REPEATS if not (np.isscalar(snr_db) and np.isinf(snr_db)) else 1):
            rep_seed2 = seed + 1000 + rep * 17
            rep_seed4 = seed + 2000 + rep * 17
            k_inside = EVAL_TTA_PER_REPEAT if tta_k > 1 else 1

            a2, f2, _, _, _, _ = eval_on_test(
                infer_model, x2_v, x2_c, y2,
                out_dir=None, tag="2Nm", snr_db=snr_db, seed=rep_seed2,
                tta_k=k_inside, roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            a4, f4, _, _, _, _ = eval_on_test(
                infer_model, x4_v, x4_c, y4,
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
            print(f"    [Sanity] empirical SNR≈{emp:.2f} dB (vib subset, after your current pipeline)")

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

    base_plots_dir = os.path.join(noise_dir, "BaselinePlots_SNR0dB")
    os.makedirs(base_plots_dir, exist_ok=True)
    _ = eval_on_test(infer_model, x2_v, x2_c, y2, base_plots_dir, "2Nm", 0, seed + 10, tta_k=1,
                     roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM)
    _ = eval_on_test(infer_model, x4_v, x4_c, y4, base_plots_dir, "4Nm", 0, seed + 20, tta_k=1,
                     roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM)

    del trainer, base_model, infer_model
    del x_tr_v, x_tr_c, y_tr, x_va_v, x_va_c, y_va
    tf.keras.backend.clear_session()
    gc.collect()
    plt.close("all")


# ============================================================
# 16) Main
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

        print("========== CDTFAFN Load Shift | Vib(.mat)+Current(.tdms) ==========")
        print("VIB_MAT_DIR_0NM:", VIB_MAT_DIR_0NM)
        print("VIB_MAT_DIR_2NM:", VIB_MAT_DIR_2NM)
        print("VIB_MAT_DIR_4NM:", VIB_MAT_DIR_4NM)
        print("CUR_TDMS_DIR   :", CUR_TDMS_DIR)
        print("OUTPUT_DIR     :", BASE_OUTPUT_DIR)

        robust_envs = ["NoNoise" if np.isinf(s) else int(s) for s in TEST_SNR_DB_LIST]
        print(f"Base noise: {TRAIN_BASE_SNR_DB}dB | Robust envs={robust_envs}")
        print(f"[FIX(A)] post-zscore(after noise)={ENABLE_POST_ZSCORE_AFTER_NOISE}")
        print(f"[SIEU] mode={SIEU_MODE} sr={SIEU_SAMPLE_RATE_HZ} fl={SIEU_FRAME_LENGTH} fs={SIEU_FRAME_STEP} "
              f"fft={SIEU_FFT_LENGTH} mel={SIEU_N_MELS} 3ch_deltas={SIEU_ADD_DELTAS_3CH}")
        print(f"[TTA] ENABLE_TTA={ENABLE_TTA} | TTA_K={TTA_K}")
        print(f"[EVAL] repeats per SNR={EVAL_NOISE_REPEATS} | tta per repeat={EVAL_TTA_PER_REPEAT}")
        print(f"[BASELINE TRAIN] lr={BASELINE_LR} batch_size={BASELINE_BATCH_SIZE} epochs={EPOCHS} opt=Adamax clipnorm={GRAD_CLIPNORM}")
        print(f"[STEP3] epochs={STEP3_EPOCHS} noisy_ce_w={STEP3_NOISY_CE_WEIGHT} hard_ce_w={STEP3_HARD_CE_WEIGHT} "
              f"lambda_max={STEP3_CONSIST_MAX_LAMBDA} warmup={STEP3_CONSIST_WARMUP_EPOCHS} bias_p={STEP3_SNR_BIAS_P}")
        print(f"[TSNE] PCA_DIM={TSNE_PCA_DIM} n_iter={TSNE_N_ITER} perplexity={TSNE_PERPLEXITY} max_points={TSNE_MAX_POINTS}")
        print("[LOG] Writing terminal output to:", log_path)

        print("\n>>> Stage 2: Full experiment ...")
        t_s2_0 = time.perf_counter()
        run_full_experiment()
        t_s2_1 = time.perf_counter()
        stage2_s = float(t_s2_1 - t_s2_0)
        print(f"[Time] Stage2(FullExp) = {_fmt_sec(stage2_s)}")

        print("\n>>> Stage 3: Noise robustness study (FAIR SameModel) ...")
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
