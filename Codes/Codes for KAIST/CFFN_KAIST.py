"""
CFFN (NO Optuna) — KAIST Load-Shift Case Study
Vibration (.mat) + Current (.tdms) | Segment length L=2048

[Task / Setting]
- Train/Val: 0Nm only (hash-overlap leakage check enabled)
- Test: 2Nm & 4Nm
- Few-shot protocol: N = 5..30 samples/class, repeated runs = 10
- Seed rule: seed = N*100 + run_idx
- Segments per class cap: 400 (per load)

[Network (paper-consistent, CFFN)]
- Two-branch EDCN feature extractor (Vibration branch + Current branch)
  - Conv1D blocks + BN+ReLU + Pooling -> GAP -> FC feature
- Offline CCA fusion (fit on train features; transform train/val/test features)
- Classifier head: Softmax on concatenated CCA-projected features

[Baseline Mode]
- EDCN_BASE_FILTERS = 16
- EDCN_FEAT_DIM     = 64
- EDCN_HEAD_DROPOUT = 0.35
- EDCN_L2           = 5e-4
- CCA_MAX_COMPONENTS= 32
- CCA_REG           = 1e-2
- Label smoothing   = 0.10
- NOTE: Only the network-specific knobs differ; the COMMON protocol is unchanged.

[Noise / Normalization Policy]
- Baseline noise environment: 0 dB Gaussian applied to train/val/test (Stage2 baseline)
- Robustness Stage3 (FAIR, SameModel): train ONE NoiseStudy_model once, then evaluate:
  NoNoise + 0 + (-2,-4,-6,-8,-10), report mean±std over repeated noise draws
- IMPORTANT: POST_ZSCORE_AFTER_NOISE MUST remain False to preserve SNR semantics.

[Outputs]
- log.txt (tee stdout/stderr)
- curves.png per run (EDCN two-branch + CCA+Softmax)
- Confusion matrix / t-SNE / ROC per test load (2Nm, 4Nm)
- Final_Summary_Stats_raw.csv + Final_Summary_Stats_trimmed.csv
- performance_trend.png
- NoiseStudy_LoadShift_FAIR_SameModel/: CSV + radar plots + baseline plots

[Open-source friendly paths]
- This script does NOT assume any fixed folder structure.
- Replace the placeholders in the PATHS section (YOUR_*) with your local absolute paths.
"""

import os
import sys
import gc
import time
import random
import warnings
from typing import Dict, Tuple, List, Any

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Activation,
    GlobalAveragePooling1D, Dense, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adamax

try:
    from nptdms import TdmsFile
except Exception as e:
    raise ImportError("Missing dependency: nptdms. Install via: pip install nptdms") from e


# ============================================================
# 0) GLOBAL VISUAL STYLE (must match COMMON)
# ============================================================
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


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
# Recommended (benchmark-aligned): ...\KAIST\CFFN_Comparison_KAIST
BASE_OUTPUT_DIR = r"YOUR_OUTPUT_DIR"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

FAULTS = [
    ("NC-0", "Normal"),
    ("IF-1", "BPFI_03"),
    ("IF-2", "BPFI_10"),
    ("OF-1", "BPFO_03"),
    ("OF-2", "BPFO_10"),
]
NUM_CLASSES = len(FAULTS)

SEG_LEN = 2048
ENABLE_ZSCORE_PER_SEGMENT = True
MAX_SEGMENTS_PER_CLASS = 400

SAMPLE_RANGE = range(5, 31)   # 5..30
REPEAT_TIMES = 10
VAL_PER_CLASS_MAIN = 50

ENABLE_HASH_OVERLAP_CHECK = True

EPOCHS_FULL = 80
BATCH_SIZE = 16
GRAD_CLIPNORM = 1.0
BEST_LR = 0.0008468950740919305

# --- Baseline knobs ---
LABEL_SMOOTHING = 0.10
EDCN_BASE_FILTERS = 16         
EDCN_FEAT_DIM = 64              
EDCN_HEAD_DROPOUT = 0.35        
EDCN_L2 = 5e-4                  
CCA_REG = 1e-2                  
CCA_EPS = 1e-9
CCA_MAX_COMPONENTS = 32         

LOSS_FN = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
METRIC_NAME = "accuracy"

# --- Noise env (MHCNN-aligned) ---
TRAIN_BASE_SNR_DB = 0.0

NO_NOISE_DB = float("inf")                 # == "NoNoise"
TEST_SNR_DB_LIST = [NO_NOISE_DB, 0, -2, -4, -6, -8, -10]

EVAL_NOISE_REPEATS = 5
POST_ZSCORE_AFTER_NOISE = False

# --- Visualization fixed params (COMMON) ---
FIGSIZE_CURVES = (10, 4)
FIGSIZE_CM = (6, 5)
FIGSIZE_TSNE = (6, 5)
FIGSIZE_ROC = (8, 6)
FIGSIZE_RADAR = (7, 7)

CM_CMAP_2NM = "Blues"
CM_CMAP_4NM = "Oranges"
ROC_COLORS_2NM = ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "brown"]
ROC_COLORS_4NM = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

TSNE_MAX_POINTS = 2000
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1200


# ============================================================
# 2) LOG TEE
# ============================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# 3) SEED / UTILS
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def trimmed_mean_std(values: List[float]) -> Tuple[float, float]:
    v = [float(x) for x in values]
    if len(v) < 3:
        m = float(np.mean(v)) if len(v) else 0.0
        s = float(np.std(v)) if len(v) else 0.0
        return m, s
    v_sorted = sorted(v)
    core = v_sorted[1:-1]
    return float(np.mean(core)), float(np.std(core))


def zscore_per_segment(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd


def add_noise_then_optional_post_zscore_np(x_clean: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    if snr_db is None or (np.isscalar(snr_db) and np.isinf(snr_db)):
        return x_clean.astype(np.float32)

    x = x_clean.astype(np.float32)
    p_sig = np.mean(np.square(x), axis=1, keepdims=True) + 1e-12
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    p_noise = p_sig / snr_lin
    noise = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)
    p_n0 = np.mean(np.square(noise), axis=1, keepdims=True) + 1e-12
    noise = noise * np.sqrt(p_noise / p_n0)
    x_noisy = x + noise

    if POST_ZSCORE_AFTER_NOISE:
        x_noisy = zscore_per_segment(x_noisy)

    return x_noisy.astype(np.float32)


# ============================================================
# 4) HASH OVERLAP CHECK
# ============================================================
def _quick_hash_vec(vec: np.ndarray) -> int:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return 0
    idx = np.linspace(0, v.size - 1, num=min(64, v.size), dtype=int)
    sub = v[idx]
    h = 1469598103934665603
    for f in sub:
        u = np.float32(f).tobytes()
        for b in u:
            h ^= b
            h *= 1099511628211
            h &= 0xFFFFFFFFFFFFFFFF
    return int(h)


def hash_overlap_check(x_tr_a, x_tr_b, x_va_a, x_va_b, prefix=""):
    if not ENABLE_HASH_OVERLAP_CHECK:
        return
    tr_hash = set()
    for i in range(x_tr_a.shape[0]):
        tr_hash.add((_quick_hash_vec(x_tr_a[i]), _quick_hash_vec(x_tr_b[i])))
    ov = 0
    for j in range(x_va_a.shape[0]):
        key = (_quick_hash_vec(x_va_a[j]), _quick_hash_vec(x_va_b[j]))
        if key in tr_hash:
            ov += 1
    if ov > 0:
        print(f"[WARN][{prefix}] Hash overlap detected between train/val: {ov} samples")


# ============================================================
# 5) KAIST LOADING
# ============================================================
def _norm_name(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def _squeeze_mat_obj(obj: Any) -> Any:
    if isinstance(obj, np.ndarray) and obj.size == 1:
        try:
            return obj.reshape(-1)[0]
        except Exception:
            return obj
    return obj


def _get_field(obj: Any, field: str) -> Any:
    obj = _squeeze_mat_obj(obj)
    if isinstance(obj, dict):
        return obj[field]
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        return obj[field]
    if isinstance(obj, np.ndarray) and getattr(obj.dtype, "names", None):
        obj2 = _squeeze_mat_obj(obj)
        if hasattr(obj2, "dtype") and getattr(obj2.dtype, "names", None):
            return obj2[field]
        return obj[field]
    raise KeyError(f"Cannot access field '{field}' on type={type(obj)}")


def _kaist_try_extract_values_matrix(mat_dict: Dict) -> np.ndarray:
    if "Signal" not in mat_dict:
        raise KeyError("Missing key 'Signal'")
    sig = _squeeze_mat_obj(mat_dict["Signal"])
    yv = _squeeze_mat_obj(_get_field(sig, "y_values"))
    vals = _squeeze_mat_obj(_get_field(yv, "values"))
    arr = np.asarray(vals)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _pick_vib_1d_from_values_matrix(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if v.ndim != 2:
        v = v.reshape(-1, 1)
    ncol = v.shape[1]
    out = v[:, 1] if ncol >= 5 else v[:, 0]
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    return out


def _collect_numeric_arrays_recursive(obj: Any, path: str, out: List[Tuple[str, np.ndarray]]):
    obj = _squeeze_mat_obj(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            _collect_numeric_arrays_recursive(v, f"{path}.{k}" if path else str(k), out)
        return

    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            out.append((path, obj))
            return
        if getattr(obj.dtype, "names", None):
            for name in obj.dtype.names:
                try:
                    _collect_numeric_arrays_recursive(obj[name], f"{path}.{name}" if path else name, out)
                except Exception:
                    pass
            return
        if obj.dtype == object:
            for idx, item in np.ndenumerate(obj):
                _collect_numeric_arrays_recursive(item, f"{path}[{idx}]", out)
            return
        return

    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        for name in obj.dtype.names:
            try:
                _collect_numeric_arrays_recursive(obj[name], f"{path}.{name}" if path else name, out)
            except Exception:
                pass
        return


def _pick_vib_1d_from_mat(mat: Dict) -> np.ndarray:
    if "Signal" in mat:
        try:
            values = _kaist_try_extract_values_matrix(mat)
            vib_1d = _pick_vib_1d_from_values_matrix(values)
            if vib_1d.size > 1000 and np.std(vib_1d) > 1e-12:
                return vib_1d.astype(np.float32)
        except Exception:
            pass

    collected: List[Tuple[str, np.ndarray]] = []
    _collect_numeric_arrays_recursive(mat, "", collected)

    candidates = []
    for p, arr in collected:
        try:
            a = np.asarray(arr)
            if not np.issubdtype(a.dtype, np.number):
                continue
            a = a.squeeze()
            if a.ndim == 2:
                a1 = a[:, 0] if a.shape[0] >= a.shape[1] else a[0, :]
            else:
                a1 = a.reshape(-1)
            a1 = np.asarray(a1, dtype=np.float32).reshape(-1)
            if a1.size > 1000 and np.std(a1) > 1e-12:
                candidates.append((p, a1))
        except Exception:
            pass

    if not candidates:
        keys = [k for k in mat.keys() if not str(k).startswith("__")]
        raise ValueError(f"No usable numeric long vector found in .mat. Available keys: {keys[:20]}")

    candidates.sort(key=lambda x: x[1].size, reverse=True)
    return candidates[0][1].astype(np.float32)


def segment_signal(x_1d: np.ndarray, seg_len=2048, max_segments=400) -> np.ndarray:
    x = np.asarray(x_1d, dtype=np.float32).reshape(-1)
    total = x.shape[0]
    nseg = total // seg_len
    nseg = min(nseg, int(max_segments))
    if nseg <= 0:
        raise ValueError("Signal too short for segmentation.")
    x = x[: nseg * seg_len]
    segs = x.reshape(nseg, seg_len, 1)
    if ENABLE_ZSCORE_PER_SEGMENT:
        segs = zscore_per_segment(segs)
    return segs.astype(np.float32)


def read_current_1d_from_tdms(tdms_path: str) -> np.ndarray:
    tdms = TdmsFile.read(tdms_path)
    for g in tdms.groups():
        for ch in g.channels():
            name = _norm_name(ch.name)
            if "curr" in name or "current" in name or "ia" in name or "ib" in name or "ic" in name:
                data = np.asarray(ch[:], dtype=np.float32).reshape(-1)
                if data.size > 1000:
                    return data
    best = None
    for g in tdms.groups():
        for ch in g.channels():
            data = np.asarray(ch[:], dtype=np.float32).reshape(-1)
            if data.size > 1000 and (best is None or data.size > best.size):
                best = data
    if best is None:
        raise ValueError(f"No usable channel found in tdms: {tdms_path}")
    return best.astype(np.float32)


def mat_path(load_nm: int, tag: str) -> str:
    return f"{load_nm}Nm_{tag}.mat"


def tdms_path(load_nm: int, tag: str) -> str:
    return f"{load_nm}Nm_{tag}.tdms"


def load_kaist_load(load_nm: int, vib_dir: str, cur_dir: str) -> Dict[str, Dict]:
    data = {}
    for cls, tag in FAULTS:
        mpath = os.path.join(vib_dir, mat_path(load_nm, tag))
        tpath = os.path.join(cur_dir, tdms_path(load_nm, tag))

        mat = loadmat(mpath)
        vib_1d = _pick_vib_1d_from_mat(mat)
        cur_1d = read_current_1d_from_tdms(tpath)

        vib_seg = segment_signal(vib_1d, SEG_LEN, MAX_SEGMENTS_PER_CLASS)
        cur_seg = segment_signal(cur_1d, SEG_LEN, MAX_SEGMENTS_PER_CLASS)

        n = min(vib_seg.shape[0], cur_seg.shape[0], MAX_SEGMENTS_PER_CLASS)
        vib_seg = vib_seg[:n]
        cur_seg = cur_seg[:n]

        data[cls] = {"vib": vib_seg, "cur": cur_seg}
        print(f"  Class {cls:5s} | {load_nm}Nm_{tag:<8s} : {n} segments")
    return data


def build_train_val_from_0Nm(
    data0: Dict[str, Dict], seed: int, n_train: int, n_val: int
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)

    xtr_v_list, xtr_c_list, ytr_list = [], [], []
    xva_v_list, xva_c_list, yva_list = [], [], []

    for ci, (cls, _) in enumerate(FAULTS):
        vib = data0[cls]["vib"]
        cur = data0[cls]["cur"]
        n_total = vib.shape[0]

        need = n_train + n_val
        if n_total < need:
            raise ValueError(f"Not enough segments for {cls}: have {n_total}, need {need}")

        idx = np.arange(n_total)
        rng.shuffle(idx)
        tr_idx = idx[:n_train]
        va_idx = idx[n_train:n_train + n_val]

        xtr_v_list.append(vib[tr_idx])
        xtr_c_list.append(cur[tr_idx])
        ytr_list.append(np.full((n_train,), ci, dtype=np.int64))

        xva_v_list.append(vib[va_idx])
        xva_c_list.append(cur[va_idx])
        yva_list.append(np.full((n_val,), ci, dtype=np.int64))

    xtr_v = np.concatenate(xtr_v_list, axis=0)
    xtr_c = np.concatenate(xtr_c_list, axis=0)
    ytr = np.concatenate(ytr_list, axis=0)

    xva_v = np.concatenate(xva_v_list, axis=0)
    xva_c = np.concatenate(xva_c_list, axis=0)
    yva = np.concatenate(yva_list, axis=0)

    hash_overlap_check(xtr_v, xtr_c, xva_v, xva_c, prefix=f"N{n_train}-seed{seed}")

    ytr_oh = tf.keras.utils.to_categorical(ytr, NUM_CLASSES)
    yva_oh = tf.keras.utils.to_categorical(yva, NUM_CLASSES)

    return (xtr_v, xtr_c, ytr_oh), (xva_v, xva_c, yva_oh)


def build_test_from_load(dataL: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xt_v_list, xt_c_list, yt_list = [], [], []
    for ci, (cls, _) in enumerate(FAULTS):
        vib = dataL[cls]["vib"][:MAX_SEGMENTS_PER_CLASS]
        cur = dataL[cls]["cur"][:MAX_SEGMENTS_PER_CLASS]
        n = min(vib.shape[0], cur.shape[0])
        vib = vib[:n]
        cur = cur[:n]
        xt_v_list.append(vib)
        xt_c_list.append(cur)
        yt_list.append(np.full((n,), ci, dtype=np.int64))

    xt_v = np.concatenate(xt_v_list, axis=0).astype(np.float32)
    xt_c = np.concatenate(xt_c_list, axis=0).astype(np.float32)
    yt = np.concatenate(yt_list, axis=0).astype(np.int64)
    yt_oh = tf.keras.utils.to_categorical(yt, NUM_CLASSES)
    return xt_v, xt_c, yt_oh


# ============================================================
# 6) CFFN MODEL (EDCN + Offline CCA + Softmax)
# ============================================================
def build_edcn_branch(input_shape=(2048, 1), num_classes=NUM_CLASSES, prefix="V") -> Model:
    inp = Input(shape=input_shape, name=f"{prefix}_in")

    x = Conv1D(EDCN_BASE_FILTERS, 64, padding="same", kernel_regularizer=l2(EDCN_L2), name=f"{prefix}_conv1")(inp)
    x = BatchNormalization(name=f"{prefix}_bn1")(x)
    x = Activation("relu", name=f"{prefix}_relu1")(x)
    x = MaxPooling1D(pool_size=16, strides=16, padding="same", name=f"{prefix}_pool1")(x)

    x = Conv1D(EDCN_BASE_FILTERS * 2, 32, padding="same", kernel_regularizer=l2(EDCN_L2), name=f"{prefix}_conv2")(x)
    x = BatchNormalization(name=f"{prefix}_bn2")(x)
    x = Activation("relu", name=f"{prefix}_relu2")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_pool2")(x)

    x = Conv1D(EDCN_BASE_FILTERS * 4, 32, padding="same", kernel_regularizer=l2(EDCN_L2), name=f"{prefix}_conv3")(x)
    x = BatchNormalization(name=f"{prefix}_bn3")(x)
    x = Activation("relu", name=f"{prefix}_relu3")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_pool3")(x)

    x = Conv1D(EDCN_BASE_FILTERS * 8, 16, padding="same", kernel_regularizer=l2(EDCN_L2), name=f"{prefix}_conv4")(x)
    x = BatchNormalization(name=f"{prefix}_bn4")(x)
    x = Activation("relu", name=f"{prefix}_relu4")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"{prefix}_pool4")(x)

    x = GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    feat = Dense(EDCN_FEAT_DIM, kernel_regularizer=l2(EDCN_L2), name=f"{prefix}_feat")(x)
    feat = Activation("relu", name=f"{prefix}_feat_relu")(feat)

    if EDCN_HEAD_DROPOUT > 0:
        feat = Dropout(EDCN_HEAD_DROPOUT, name=f"{prefix}_drop")(feat)

    out = Dense(num_classes, activation="softmax", name=f"{prefix}_out")(feat)
    return Model(inp, out, name=f"EDCN_{prefix}")


def build_two_branch_train_model() -> Tuple[Model, Model, Model]:
    edcn_v = build_edcn_branch(prefix="V")
    edcn_c = build_edcn_branch(prefix="C")

    model_2b = Model([edcn_v.input, edcn_c.input], [edcn_v.output, edcn_c.output], name="CFFN_TwoBranch_EDCN")

    feat_v = Model(edcn_v.input, edcn_v.get_layer("V_feat_relu").output, name="feat_V")
    feat_c = Model(edcn_c.input, edcn_c.get_layer("C_feat_relu").output, name="feat_C")

    return model_2b, feat_v, feat_c


def cca_fit(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3, max_components: int = 128) -> Dict:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mx = X.mean(axis=0, keepdims=True)
    my = Y.mean(axis=0, keepdims=True)
    X0 = X - mx
    Y0 = Y - my
    N = X0.shape[0]
    d = X0.shape[1]

    Cxx = (X0.T @ X0) / max(N - 1, 1) + reg * np.eye(d)
    Cyy = (Y0.T @ Y0) / max(N - 1, 1) + reg * np.eye(d)
    Cxy = (X0.T @ Y0) / max(N - 1, 1)
    Cyx = Cxy.T

    iCxx = np.linalg.pinv(Cxx)
    iCyy = np.linalg.pinv(Cyy)

    M = iCxx @ Cxy @ iCyy @ Cyx
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    Wx_all = eigvecs[:, idx]

    k = min(d, max(N - 1, 1))
    k = min(k, int(max_components))
    k = max(int(k), 1)

    eigvals_k = np.maximum(eigvals[:k], 0.0)
    Wx = Wx_all[:, :k]
    denom_sqrt = np.sqrt(eigvals_k + CCA_EPS)
    Wy = (iCyy @ Cyx @ Wx) / denom_sqrt[np.newaxis, :]

    return dict(mx=mx, my=my, Wx=Wx, Wy=Wy, k=k)


def cca_transform(X: np.ndarray, Y: np.ndarray, params: Dict) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mx, my, Wx, Wy = params["mx"], params["my"], params["Wx"], params["Wy"]
    X0 = (X - mx) @ Wx
    Y0 = (Y - my) @ Wy
    return np.concatenate([X0, Y0], axis=1).astype(np.float32)


def build_softmax_classifier(input_dim: int, num_classes: int) -> Model:
    inp = Input(shape=(input_dim,), name="CCA_in")
    out = Dense(num_classes, activation="softmax", name="Softmax")(inp)
    return Model(inp, out, name="CCA_Softmax_Classifier")


# ============================================================
# 7) PLOTTING
# ============================================================
def save_curves(history, out_path, title_prefix, val_name="Val"):
    plt.figure(figsize=FIGSIZE_CURVES)
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
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion(y_true, y_pred, out_path, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=FIGSIZE_CM)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=[c for c, _ in FAULTS],
        yticklabels=[c for c, _ in FAULTS]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_tsne_from_features(features: np.ndarray, y_true: np.ndarray, out_path: str, title: str, seed: int):
    feats = np.asarray(features, dtype=np.float32)
    y = np.asarray(y_true, dtype=np.int64)

    if feats.shape[0] > TSNE_MAX_POINTS:
        rng = np.random.RandomState(seed)
        idx = rng.choice(feats.shape[0], TSNE_MAX_POINTS, replace=False)
        feats = feats[idx]
        y = y[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, max(5, (feats.shape[0] // 10))),
        n_iter=TSNE_N_ITER,
        random_state=seed,
        init="pca",
        learning_rate="auto"
    )
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=FIGSIZE_TSNE)
    for ci, (cls, _) in enumerate(FAULTS):
        m = (y == ci)
        plt.scatter(emb[m, 0], emb[m, 1], s=15, label=cls, alpha=0.85)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_multiclass(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, title: str, colors: List[str]):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=FIGSIZE_ROC)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        col = colors[i % len(colors)]
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.3f}", color=col)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar(snr_list: List[float], acc_list: List[float], f1_list: List[float], out_path: str, title: str):
    def _lab(s):
        if s is None:
            return "NoNoise"
        if np.isscalar(s) and np.isinf(s):
            return "NoNoise"
        try:
            return str(int(s))
        except Exception:
            return str(s)

    labels = [_lab(s) for s in snr_list]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    acc = acc_list + acc_list[:1]
    f1 = f1_list + f1_list[:1]

    plt.figure(figsize=FIGSIZE_RADAR)
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, acc, linewidth=2, label="Accuracy")
    ax.fill(angles, acc, alpha=0.15)
    ax.plot(angles, f1, linewidth=2, label="Macro-F1")
    ax.fill(angles, f1, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.0, 1.0)
    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_performance_trend(trimmed_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(9, 5))
    x = trimmed_df["n_train"].values
    plt.plot(x, trimmed_df["Acc_2Nm_mean"].values, label="2Nm Acc")
    plt.plot(x, trimmed_df["F1_2Nm_mean"].values, label="2Nm F1")
    plt.plot(x, trimmed_df["Acc_4Nm_mean"].values, label="4Nm Acc")
    plt.plot(x, trimmed_df["F1_4Nm_mean"].values, label="4Nm F1")
    plt.ylim(0.0, 1.02)
    plt.xlabel("N (train samples per class @0Nm)")
    plt.ylabel("Score")
    plt.title("Performance Trend (Trimmed Mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 8) EVAL
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def eval_on_test_cffn(
    feat_v: Model, feat_c: Model,
    cca_params: Dict, clf: Model,
    x_v: np.ndarray, x_c: np.ndarray, y_onehot: np.ndarray,
    out_dir: str, tag: str,
    snr_db: float, seed: int,
    roc_colors: List[str], cm_cmap: str
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    x_vn = add_noise_then_optional_post_zscore_np(x_v, snr_db, rng)
    x_cn = add_noise_then_optional_post_zscore_np(x_c, snr_db, rng)

    y_true = np.argmax(y_onehot, axis=1)

    t0_inf = time.perf_counter()
    fv = feat_v.predict(x_vn, batch_size=256, verbose=0)
    fc = feat_c.predict(x_cn, batch_size=256, verbose=0)
    fused = cca_transform(fv, fc, cca_params)
    y_prob = clf.predict(fused, batch_size=256, verbose=0)
    t1_inf = time.perf_counter()

    y_pred = np.argmax(y_prob, axis=1)
    m = compute_metrics(y_true, y_pred)

    t0_viz = time.perf_counter()
    save_confusion(y_true, y_pred, os.path.join(out_dir, f"cm_{tag}.png"),
                   title=f"Confusion Matrix ({tag})", cmap=cm_cmap)
    save_tsne_from_features(fused, y_true, os.path.join(out_dir, f"tsne_{tag}.png"),
                            title=f"t-SNE ({tag})", seed=seed + 123)
    save_roc_multiclass(y_true, y_prob, os.path.join(out_dir, f"roc_{tag}.png"),
                        title=f"ROC ({tag})", colors=roc_colors)
    t1_viz = time.perf_counter()

    m["infer_time"] = float(t1_inf - t0_inf)
    m["viz_time"] = float(t1_viz - t0_viz)
    return m


# ============================================================
# 9) best_parameters.txt
# ============================================================
def write_best_parameters_txt(out_dir: str):
    p = os.path.join(out_dir, "best_parameters.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("CFFN (KAIST Load-Shift) - FIXED TRAINING SETTINGS (Moderate Baseline)\n")
        f.write("=" * 80 + "\n")
        f.write(f"EPOCHS_FULL = {EPOCHS_FULL}\n")
        f.write(f"OPTIMIZER   = Adamax(lr={BEST_LR}, clipnorm={GRAD_CLIPNORM})\n")
        f.write(f"BATCH_SIZE  = {BATCH_SIZE}\n")
        f.write(f"LOSS        = CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})\n")
        f.write(f"METRIC      = {METRIC_NAME}\n\n")

        f.write("DATA SETTINGS\n")
        f.write("-" * 80 + "\n")
        f.write(f"SEG_LEN = {SEG_LEN}\n")
        f.write(f"ENABLE_ZSCORE_PER_SEGMENT = {ENABLE_ZSCORE_PER_SEGMENT}\n")
        f.write(f"MAX_SEGMENTS_PER_CLASS = {MAX_SEGMENTS_PER_CLASS}\n")
        f.write(f"SAMPLE_RANGE = {SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}\n")
        f.write(f"REPEAT_TIMES = {REPEAT_TIMES}\n")
        f.write(f"VAL_PER_CLASS_MAIN = {VAL_PER_CLASS_MAIN}\n")
        f.write(f"seed rule = n_train*100 + run_idx\n")
        f.write(f"ENABLE_HASH_OVERLAP_CHECK = {ENABLE_HASH_OVERLAP_CHECK}\n\n")

        f.write("NOISE SETTINGS (MHCNN-aligned)\n")
        f.write("-" * 80 + "\n")
        f.write(f"TRAIN_BASE_SNR_DB = {TRAIN_BASE_SNR_DB}\n")
        f.write(f"TEST_SNR_DB_LIST  = {TEST_SNR_DB_LIST}\n")
        f.write(f"EVAL_NOISE_REPEATS = {EVAL_NOISE_REPEATS}\n")
        f.write(f"POST_ZSCORE_AFTER_NOISE = {POST_ZSCORE_AFTER_NOISE}\n\n")

        f.write("MODEL (Moderate EDCN)\n")
        f.write("-" * 80 + "\n")
        f.write("EDCN blocks: base=24 -> 24/48/96/192, kernel 64/32/32/16, pool 16/2/2/2 + BN+ReLU + GAP + FC(96)\n")
        f.write(f"EDCN_L2={EDCN_L2}, EDCN_HEAD_DROPOUT={EDCN_HEAD_DROPOUT}, LABEL_SMOOTHING={LABEL_SMOOTHING}\n")
        f.write(f"CCA_REG={CCA_REG}, CCA_MAX_COMPONENTS={CCA_MAX_COMPONENTS}\n")


# ============================================================
# 10) TRAIN ONE RUN (CFFN two-stage)
# ============================================================
def train_one_run_cffn(
    xtr_v, xtr_c, ytr,
    xva_v, xva_c, yva,
    seed: int,
    run_dir: str
) -> Tuple[Model, Model, Dict, Model, float, float]:
    os.makedirs(run_dir, exist_ok=True)

    rng_tr = np.random.RandomState(seed + 11)
    rng_va = np.random.RandomState(seed + 22)
    xtr_vn = add_noise_then_optional_post_zscore_np(xtr_v, TRAIN_BASE_SNR_DB, rng_tr)
    xtr_cn = add_noise_then_optional_post_zscore_np(xtr_c, TRAIN_BASE_SNR_DB, rng_tr)
    xva_vn = add_noise_then_optional_post_zscore_np(xva_v, TRAIN_BASE_SNR_DB, rng_va)
    xva_cn = add_noise_then_optional_post_zscore_np(xva_c, TRAIN_BASE_SNR_DB, rng_va)

    model_2b, feat_v, feat_c = build_two_branch_train_model()
    opt = Adamax(learning_rate=BEST_LR, clipnorm=GRAD_CLIPNORM)

    model_2b.compile(
        optimizer=opt,
        loss={"V_out": LOSS_FN, "C_out": LOSS_FN},
        metrics={"V_out": [METRIC_NAME], "C_out": [METRIC_NAME]},
    )

    t0 = time.perf_counter()
    hist_edcn = model_2b.fit(
        [xtr_vn, xtr_cn], {"V_out": ytr, "C_out": ytr},
        validation_data=([xva_vn, xva_cn], {"V_out": yva, "C_out": yva}),
        epochs=EPOCHS_FULL,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    t1 = time.perf_counter()

    # curves_edcn.png
    try:
        avg_acc = (np.array(hist_edcn.history["V_out_accuracy"]) + np.array(hist_edcn.history["C_out_accuracy"])) / 2.0
        avg_vacc = (np.array(hist_edcn.history["val_V_out_accuracy"]) + np.array(hist_edcn.history["val_C_out_accuracy"])) / 2.0
        tmp_hist = type("H", (), {})()
        tmp_hist.history = {
            "loss": hist_edcn.history["loss"],
            "val_loss": hist_edcn.history["val_loss"],
            "accuracy": avg_acc.tolist(),
            "val_accuracy": avg_vacc.tolist(),
        }
        save_curves(tmp_hist, os.path.join(run_dir, "curves_edcn.png"), title_prefix="EDCN(two-branch)")
    except Exception:
        pass

    # offline CCA
    fv_tr = feat_v.predict(xtr_vn, batch_size=256, verbose=0)
    fc_tr = feat_c.predict(xtr_cn, batch_size=256, verbose=0)
    cca_params = cca_fit(fv_tr, fc_tr, reg=CCA_REG, max_components=CCA_MAX_COMPONENTS)

    fused_tr = cca_transform(fv_tr, fc_tr, cca_params)

    fv_va = feat_v.predict(xva_vn, batch_size=256, verbose=0)
    fc_va = feat_c.predict(xva_cn, batch_size=256, verbose=0)
    fused_va = cca_transform(fv_va, fc_va, cca_params)

    clf = build_softmax_classifier(fused_tr.shape[1], NUM_CLASSES)
    opt2 = Adamax(learning_rate=BEST_LR, clipnorm=GRAD_CLIPNORM)
    clf.compile(optimizer=opt2, loss=LOSS_FN, metrics=[METRIC_NAME])

    hist_clf = clf.fit(
        fused_tr, ytr,
        validation_data=(fused_va, yva),
        epochs=EPOCHS_FULL,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    save_curves(hist_clf, os.path.join(run_dir, "curves.png"), title_prefix="CCA+Softmax")

    t2 = time.perf_counter()
    train_time_total = float(t2 - t0)
    return feat_v, feat_c, cca_params, clf, train_time_total, float(t2 - t1)


# ============================================================
# 11) STEP3: NoiseStudy (FAIR SameModel) - MHCNN aligned
# ============================================================
def run_noise_study_same_model(data0, x2_v, x2_c, y2, x4_v, x4_c, y4):
    noise_dir = os.path.join(BASE_OUTPUT_DIR, "NoiseStudy_LoadShift_FAIR_SameModel")
    os.makedirs(noise_dir, exist_ok=True)

    seed = 2026
    set_global_seed(seed)

    n_train = 30
    n_val = min(VAL_PER_CLASS_MAIN, MAX_SEGMENTS_PER_CLASS - n_train - 1)
    (xtr_v, xtr_c, ytr), (xva_v, xva_c, yva) = build_train_val_from_0Nm(
        data0, seed=seed, n_train=n_train, n_val=n_val
    )

    tf.keras.backend.clear_session()
    gc.collect()

    print("\n>>> Step3 (NoiseStudy FAIR SameModel): Train once @0dB, eval multi-SNR ...")
    feat_v, feat_c, cca_params, clf, train_time_total, _ = train_one_run_cffn(
        xtr_v, xtr_c, ytr, xva_v, xva_c, yva, seed=seed, run_dir=noise_dir
    )
    print(f"[Step3] TrainTime={train_time_total:.2f}s")

    rows = []
    acc2_mean_list, f12_mean_list = [], []
    acc4_mean_list, f14_mean_list = [], []

    def _is_nonoise(s):
        try:
            return np.isscalar(s) and np.isinf(s)
        except Exception:
            return False

    for snr_db in TEST_SNR_DB_LIST:
        is_nonoise = _is_nonoise(snr_db)
        label = "NoNoise" if is_nonoise else str(int(snr_db))
        repeats = 1 if is_nonoise else EVAL_NOISE_REPEATS

        acc2_list, f12_list = [], []
        acc4_list, f14_list = [], []

        for r in range(repeats):
            s = seed + 1000 + r * 17
            m2 = eval_on_test_cffn(
                feat_v, feat_c, cca_params, clf,
                x2_v, x2_c, y2,
                out_dir=os.path.join(noise_dir, "_tmp2"), tag="2Nm",
                snr_db=snr_db, seed=s,
                roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
            )
            m4 = eval_on_test_cffn(
                feat_v, feat_c, cca_params, clf,
                x4_v, x4_c, y4,
                out_dir=os.path.join(noise_dir, "_tmp4"), tag="4Nm",
                snr_db=snr_db, seed=s + 9,
                roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
            )
            acc2_list.append(m2["acc"]); f12_list.append(m2["f1"])
            acc4_list.append(m4["acc"]); f14_list.append(m4["f1"])

        acc2_m, acc2_s = float(np.mean(acc2_list)), float(np.std(acc2_list))
        f12_m, f12_s = float(np.mean(f12_list)), float(np.std(f12_list))
        acc4_m, acc4_s = float(np.mean(acc4_list)), float(np.std(acc4_list))
        f14_m, f14_s = float(np.mean(f14_list)), float(np.std(f14_list))

        rows.append([
            label,
            (np.nan if is_nonoise else float(snr_db)),
            acc2_m, acc2_s, f12_m, f12_s,
            acc4_m, acc4_s, f14_m, f14_s,
            repeats
        ])

        acc2_mean_list.append(acc2_m); f12_mean_list.append(f12_m)
        acc4_mean_list.append(acc4_m); f14_mean_list.append(f14_m)

        print(f"[Step3] {label:>7s} | 2Nm Acc={acc2_m:.4f}±{acc2_s:.4f} F1={f12_m:.4f}±{f12_s:.4f} | "
              f"4Nm Acc={acc4_m:.4f}±{acc4_s:.4f} F1={f14_m:.4f}±{f14_s:.4f}")

    # cleanup tmp dirs
    for d in [os.path.join(noise_dir, "_tmp2"), os.path.join(noise_dir, "_tmp4")]:
        if os.path.isdir(d):
            try:
                for fn in os.listdir(d):
                    os.remove(os.path.join(d, fn))
                os.rmdir(d)
            except Exception:
                pass

    out_csv = os.path.join(noise_dir, "Noise_Robustness_LoadShift_SameModel_FAIR.csv")
    pd.DataFrame(rows, columns=[
        "label", "SNR_dB",
        "Acc_2Nm_mean", "Acc_2Nm_std", "F1_2Nm_mean", "F1_2Nm_std",
        "Acc_4Nm_mean", "Acc_4Nm_std", "F1_4Nm_mean", "F1_4Nm_std",
        "eval_repeats"
    ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[Step3] Saved:", out_csv)

    plot_radar(
        TEST_SNR_DB_LIST, acc2_mean_list, f12_mean_list,
        os.path.join(noise_dir, "radar_2Nm.png"),
        title="2Nm Robustness (FAIR, SameModel)  NoNoise/0/-2/-4/-6/-8/-10 dB"
    )
    plot_radar(
        TEST_SNR_DB_LIST, acc4_mean_list, f14_mean_list,
        os.path.join(noise_dir, "radar_4Nm.png"),
        title="4Nm Robustness (FAIR, SameModel)  NoNoise/0/-2/-4/-6/-8/-10 dB"
    )

    # baseline plots @0dB (must output CM/tSNE/ROC)
    base_dir = os.path.join(noise_dir, "BaselineSNR_0dB")
    out2 = os.path.join(base_dir, "Test_2Nm")
    out4 = os.path.join(base_dir, "Test_4Nm")
    _ = eval_on_test_cffn(
        feat_v, feat_c, cca_params, clf,
        x2_v, x2_c, y2,
        out_dir=out2, tag="2Nm",
        snr_db=0, seed=seed + 10,
        roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
    )
    _ = eval_on_test_cffn(
        feat_v, feat_c, cca_params, clf,
        x4_v, x4_c, y4,
        out_dir=out4, tag="4Nm",
        snr_db=0, seed=seed + 20,
        roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
    )

    del feat_v, feat_c, clf
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# 12) MAIN
# ============================================================
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    write_best_parameters_txt(BASE_OUTPUT_DIR)

    log_path = os.path.join(BASE_OUTPUT_DIR, "log.txt")
    f_log = open(log_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, f_log)
    sys.stderr = _Tee(old_err, f_log)

    t_all0 = time.perf_counter()

    try:
        print("========== CFFN (NO Optuna) Load Shift | Vib(.mat)+Current(.tdms) ==========")
        print("VIB_MAT_DIR_0NM:", VIB_MAT_DIR_0NM)
        print("VIB_MAT_DIR_2NM:", VIB_MAT_DIR_2NM)
        print("VIB_MAT_DIR_4NM:", VIB_MAT_DIR_4NM)
        print("CUR_TDMS_DIR   :", CUR_TDMS_DIR)
        print("OUTPUT_DIR     :", BASE_OUTPUT_DIR)
        print(f"Base noise: {TRAIN_BASE_SNR_DB} dB | Robust envs={TEST_SNR_DB_LIST}")
        print(f"[FIX] post-zscore(after noise)={POST_ZSCORE_AFTER_NOISE}")
        print(f"[Protocol] N={SAMPLE_RANGE.start}..{SAMPLE_RANGE.stop-1}, runs={REPEAT_TIMES}, seed=n_train*100+run_idx")
        print(f"[Train] epochs={EPOCHS_FULL}, bs={BATCH_SIZE}, opt=Adamax(lr={BEST_LR}, clipnorm={GRAD_CLIPNORM})")
        print(f"[Moderate] filters={EDCN_BASE_FILTERS}, feat_dim={EDCN_FEAT_DIM}, drop={EDCN_HEAD_DROPOUT}, l2={EDCN_L2}, "
              f"cca_k<= {CCA_MAX_COMPONENTS}, cca_reg={CCA_REG}, label_smooth={LABEL_SMOOTHING}")
        print("============================================================\n")

        print(">>> Pre-loading KAIST data into memory...")
        print("Loading 0Nm ...")
        data0 = load_kaist_load(0, VIB_MAT_DIR_0NM, CUR_TDMS_DIR)
        print("Loading 2Nm ...")
        data2 = load_kaist_load(2, VIB_MAT_DIR_2NM, CUR_TDMS_DIR)
        print("Loading 4Nm ...")
        data4 = load_kaist_load(4, VIB_MAT_DIR_4NM, CUR_TDMS_DIR)

        x2_v, x2_c, y2 = build_test_from_load(data2)
        x4_v, x4_c, y4 = build_test_from_load(data4)
        print(f"[TestSet] 2Nm: {x2_v.shape} | 4Nm: {x4_v.shape}\n")

        tf.keras.backend.clear_session()
        model_2b, feat_v_tmp, feat_c_tmp = build_two_branch_train_model()
        print(">>> CFFN Two-Branch EDCN Summary:")
        model_2b.summary()
        del model_2b, feat_v_tmp, feat_c_tmp
        tf.keras.backend.clear_session()
        gc.collect()

        raw_rows = []
        trimmed_rows = []

        t_stage2_0 = time.perf_counter()
        for n_train in SAMPLE_RANGE:
            print(f"\n======== Train(0Nm) N={n_train} (Run 1-{REPEAT_TIMES}) ========")
            sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{n_train:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            acc2_list, f12_list, p2_list, r2_list = [], [], [], []
            acc4_list, f14_list, p4_list, r4_list = [], [], [], []
            train_time_list, infer2_list, viz2_list, infer4_list, viz4_list = [], [], [], [], []

            n_val = min(VAL_PER_CLASS_MAIN, MAX_SEGMENTS_PER_CLASS - n_train - 1)

            for run_idx in range(1, REPEAT_TIMES + 1):
                seed = n_train * 100 + run_idx
                set_global_seed(seed)

                run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
                os.makedirs(run_dir, exist_ok=True)

                (xtr_v, xtr_c, ytr), (xva_v, xva_c, yva) = build_train_val_from_0Nm(
                    data0, seed=seed, n_train=n_train, n_val=n_val
                )

                tf.keras.backend.clear_session()
                gc.collect()

                feat_v, feat_c, cca_params, clf, train_time, _ = train_one_run_cffn(
                    xtr_v, xtr_c, ytr, xva_v, xva_c, yva, seed=seed, run_dir=run_dir
                )

                out2 = os.path.join(run_dir, "Test_2Nm")
                out4 = os.path.join(run_dir, "Test_4Nm")

                m2 = eval_on_test_cffn(
                    feat_v, feat_c, cca_params, clf,
                    x2_v, x2_c, y2,
                    out_dir=out2, tag="2Nm",
                    snr_db=TRAIN_BASE_SNR_DB, seed=seed + 10,
                    roc_colors=ROC_COLORS_2NM, cm_cmap=CM_CMAP_2NM
                )
                m4 = eval_on_test_cffn(
                    feat_v, feat_c, cca_params, clf,
                    x4_v, x4_c, y4,
                    out_dir=out4, tag="4Nm",
                    snr_db=TRAIN_BASE_SNR_DB, seed=seed + 20,
                    roc_colors=ROC_COLORS_4NM, cm_cmap=CM_CMAP_4NM
                )

                acc2_list.append(m2["acc"]); f12_list.append(m2["f1"]); p2_list.append(m2["prec"]); r2_list.append(m2["rec"])
                acc4_list.append(m4["acc"]); f14_list.append(m4["f1"]); p4_list.append(m4["prec"]); r4_list.append(m4["rec"])

                train_time_list.append(train_time)
                infer2_list.append(m2["infer_time"]); viz2_list.append(m2["viz_time"])
                infer4_list.append(m4["infer_time"]); viz4_list.append(m4["viz_time"])

                t2_total = float(m2["infer_time"] + m2["viz_time"])
                t4_total = float(m4["infer_time"] + m4["viz_time"])

                print(
                    f"  [N{n_train}-R{run_idx}] "
                    f"Train={train_time:.2f}s | "
                    f"2Nm Acc={m2['acc']:.4f} F1={m2['f1']:.4f} "
                    f"(Infer={m2['infer_time']:.2f}s, Viz={m2['viz_time']:.2f}s, Total={t2_total:.2f}s) | "
                    f"4Nm Acc={m4['acc']:.4f} F1={m4['f1']:.4f} "
                    f"(Infer={m4['infer_time']:.2f}s, Viz={m4['viz_time']:.2f}s, Total={t4_total:.2f}s)",
                    flush=True
                )

                del feat_v, feat_c, clf
                tf.keras.backend.clear_session()
                gc.collect()

            raw_rows.append([
                n_train,
                float(np.mean(acc2_list)), float(np.std(acc2_list)),
                float(np.mean(f12_list)), float(np.std(f12_list)),
                float(np.mean(p2_list)), float(np.std(p2_list)),
                float(np.mean(r2_list)), float(np.std(r2_list)),
                float(np.mean(acc4_list)), float(np.std(acc4_list)),
                float(np.mean(f14_list)), float(np.std(f14_list)),
                float(np.mean(p4_list)), float(np.std(p4_list)),
                float(np.mean(r4_list)), float(np.std(r4_list)),
                float(np.mean(train_time_list)), float(np.mean(infer2_list)), float(np.mean(viz2_list)),
                float(np.mean(infer4_list)), float(np.mean(viz4_list)),
            ])

            acc2_m, acc2_s = trimmed_mean_std(acc2_list)
            f12_m, f12_s = trimmed_mean_std(f12_list)
            p2_m, p2_s = trimmed_mean_std(p2_list)
            r2_m, r2_s = trimmed_mean_std(r2_list)

            acc4_m, acc4_s = trimmed_mean_std(acc4_list)
            f14_m, f14_s = trimmed_mean_std(f14_list)
            p4_m, p4_s = trimmed_mean_std(p4_list)
            r4_m, r4_s = trimmed_mean_std(r4_list)

            trimmed_rows.append([
                n_train,
                acc2_m, acc2_s, f12_m, f12_s, p2_m, p2_s, r2_m, r2_s,
                acc4_m, acc4_s, f14_m, f14_s, p4_m, p4_s, r4_m, r4_s
            ])

            print(f"  >>> Trimmed N={n_train}: "
                  f"2Nm Acc={acc2_m:.4f}±{acc2_s:.4f} F1={f12_m:.4f}±{f12_s:.4f} | "
                  f"4Nm Acc={acc4_m:.4f}±{acc4_s:.4f} F1={f14_m:.4f}±{f14_s:.4f}")

        t_stage2_1 = time.perf_counter()

        raw_csv = os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_raw.csv")
        raw_cols = [
            "n_train",
            "Acc_2Nm_mean", "Acc_2Nm_std", "F1_2Nm_mean", "F1_2Nm_std",
            "Prec_2Nm_mean", "Prec_2Nm_std", "Recall_2Nm_mean", "Recall_2Nm_std",
            "Acc_4Nm_mean", "Acc_4Nm_std", "F1_4Nm_mean", "F1_4Nm_std",
            "Prec_4Nm_mean", "Prec_4Nm_std", "Recall_4Nm_mean", "Recall_4Nm_std",
            "Train_time_mean", "Infer_time_2Nm_mean", "Viz_time_2Nm_mean",
            "Infer_time_4Nm_mean", "Viz_time_4Nm_mean"
        ]
        pd.DataFrame(raw_rows, columns=raw_cols).to_csv(raw_csv, index=False, encoding="utf-8-sig")

        trimmed_csv = os.path.join(BASE_OUTPUT_DIR, "Final_Summary_Stats_trimmed.csv")
        trimmed_cols = [
            "n_train",
            "Acc_2Nm_mean", "Acc_2Nm_std", "F1_2Nm_mean", "F1_2Nm_std",
            "Prec_2Nm_mean", "Prec_2Nm_std", "Recall_2Nm_mean", "Recall_2Nm_std",
            "Acc_4Nm_mean", "Acc_4Nm_std", "F1_4Nm_mean", "F1_4Nm_std",
            "Prec_4Nm_mean", "Prec_4Nm_std", "Recall_4Nm_mean", "Recall_4Nm_std",
        ]
        trimmed_df = pd.DataFrame(trimmed_rows, columns=trimmed_cols)
        trimmed_df.to_csv(trimmed_csv, index=False, encoding="utf-8-sig")

        plot_performance_trend(trimmed_df, os.path.join(BASE_OUTPUT_DIR, "performance_trend.png"))

        print("\n[Saved] ", raw_csv)
        print("[Saved] ", trimmed_csv)
        print("[Saved] performance_trend.png")

        # Step3 NoiseStudy
        t_stage3_0 = time.perf_counter()
        run_noise_study_same_model(data0, x2_v, x2_c, y2, x4_v, x4_c, y4)
        t_stage3_1 = time.perf_counter()

        t_all1 = time.perf_counter()
        with open(os.path.join(BASE_OUTPUT_DIR, "Time_Summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Stage2(full N=5..30 x10 runs) = {t_stage2_1 - t_stage2_0:.2f} s\n")
            f.write(f"Stage3(NoiseStudy same model) = {t_stage3_1 - t_stage3_0:.2f} s\n")
            f.write(f"TOTAL = {t_all1 - t_all0:.2f} s\n")

        print("\n[Time] Stage2 =", f"{t_stage2_1 - t_stage2_0:.2f}s")
        print("[Time] Stage3 =", f"{t_stage3_1 - t_stage3_0:.2f}s")
        print("[Time] TOTAL  =", f"{t_all1 - t_all0:.2f}s")

    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f_log.close()


if __name__ == "__main__":
    main()
