# -*- coding: utf-8 -*-
"""
AgentSeg — Agentic Histopathology Segmentation
N1 -> N2 -> N6 Full Agentic Pipeline with SAM ViT-B
"""

# ── stdlib imports (always available) ────────────────────────
import gc
import io
import os
import sys
import time
import warnings
import urllib.request
warnings.filterwarnings("ignore")

# ── Streamlit (installed via pip) ────────────────────────────
import streamlit as st

# ── Page config must be FIRST streamlit call ─────────────────
st.set_page_config(
    page_title="AgentSeg | Histopathology Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# DEPENDENCY CHECK — show install instructions if missing
# ════════════════════════════════════════════════════════════
_missing = []
try:
    import numpy as np
except ImportError:
    _missing.append("numpy")

try:
    import cv2  # noqa: F401
except ImportError:
    _missing.append("opencv-python-headless")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    _missing.append("torch")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    _missing.append("matplotlib")

try:
    from PIL import Image
except ImportError:
    _missing.append("Pillow")

try:
    from skimage.measure import label, regionprops
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage.morphology import (
        remove_small_objects, disk, binary_closing, binary_dilation
    )
except ImportError:
    _missing.append("scikit-image")

try:
    from scipy import ndimage as ndi
except ImportError:
    _missing.append("scipy")

try:
    from segment_anything import (
        sam_model_registry,
        SamAutomaticMaskGenerator,
        SamPredictor,
    )
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    _missing.append("segment-anything")

if _missing:
    st.error(
        f"**Missing dependencies:** `{'`, `'.join(_missing)}`\n\n"
        "Run this command in your terminal, then restart the app:\n\n"
        "```bash\n"
        "pip3 install --break-system-packages \\\n"
        "    opencv-python-headless torch torchvision \\\n"
        "    scikit-image scipy matplotlib scikit-learn \\\n"
        "    segment-anything\n"
        "```"
    )
    st.stop()

# ════════════════════════════════════════════════════════════
# SAM CHECKPOINT — automatically download if missing
# ════════════════════════════════════════════════════════════

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_CHECKPOINT = os.path.join(BASE_DIR, "sam_vit_b_01ec64.pth")


def ensure_sam_checkpoint():
    """Download SAM ViT-B checkpoint automatically if it is missing."""

    if os.path.exists(SAM_CHECKPOINT):
        return SAM_CHECKPOINT

    st.info("📥 SAM ViT-B model not found. Downloading model (~375 MB)...")

    try:
        urllib.request.urlretrieve(
            SAM_URL,
            SAM_CHECKPOINT
        )

        st.success("✅ SAM ViT-B model downloaded successfully.")

        return SAM_CHECKPOINT

    except Exception as e:
        st.error(
            f"❌ Could not download SAM ViT-B checkpoint.\n\n"
            f"Error: {e}"
        )
        st.stop()

    # ════════════════════════════════════════════════════════════
# AGENTIC AUTO-CALIBRATION — no human input needed
# ════════════════════════════════════════════════════════════
def auto_calibrate_amg(image_rgb):
    """
    Automatically select AMG + morphology thresholds
    based on image statistics. Fully agentic — zero human input.
    """
    gray        = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_int    = float(gray.mean())
    contrast    = float(gray.std())
    h, w        = gray.shape
    img_area    = h * w

    # Estimate nucleus size from image area
    # Small image (256px) → small nuclei → small min_area
    est_nucleus_area = max(30, int(img_area * 0.0008))

    if contrast < 30:
        # Low contrast — dark/faint staining → very relaxed
        cfg = dict(pts=32, iou=0.70, stability=0.78,
                   min_area=max(30, est_nucleus_area // 2),
                   area_min=20, area_max=4000, solidity_min=0.40,
                   reason="low contrast → relaxed thresholds")
    elif contrast > 60 and mean_int > 150:
        # High contrast, bright → can be stricter
        cfg = dict(pts=16, iou=0.82, stability=0.88,
                   min_area=est_nucleus_area,
                   area_min=50, area_max=3500, solidity_min=0.50,
                   reason="high contrast → balanced thresholds")
    else:
        # Medium — general histopathology
        cfg = dict(pts=24, iou=0.75, stability=0.83,
                   min_area=max(40, est_nucleus_area // 2),
                   area_min=30, area_max=4000, solidity_min=0.45,
                   reason="medium contrast → standard thresholds")

    return cfg

# ── all imports succeeded — safe to use everywhere ───────────

# ════════════════════════════════════════════════════════════
# CSS — premium dark theme
# ════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:   #0a0e1a;
    --bg-card:   #111827;
    --bg-panel:  #1a2235;
    --blue:      #3b82f6;
    --cyan:      #06b6d4;
    --violet:    #8b5cf6;
    --green:     #10b981;
    --amber:     #f59e0b;
    --rose:      #f43f5e;
    --text-1:    #f1f5f9;
    --text-2:    #94a3b8;
    --border:    rgba(99,102,241,0.18);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-1) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1400px; }

/* Hero */
.hero {
    background: linear-gradient(135deg,#0f172a 0%,#1e1b4b 40%,#0c1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 60% 80% at 70% 50%,rgba(99,102,241,.15) 0%,transparent 70%);
    pointer-events:none;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg,#818cf8 0%,#38bdf8 50%,#34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 .5rem; line-height:1.1;
}
.hero-subtitle { color:var(--text-2); font-size:1.05rem; margin:0; }
.hero-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.3);
    border-radius:999px; padding:4px 14px; font-size:.78rem; font-weight:600;
    color:#a5b4fc; margin-bottom:1rem; text-transform:uppercase; letter-spacing:.08em;
}

/* Pipeline cards */
.pipeline-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:14px; padding:1rem 1.25rem; margin-bottom:.6rem;
    display:flex; align-items:center; gap:.85rem;
    transition:border-color .25s,box-shadow .25s;
}
.pipeline-card.active {
    border-color:var(--blue);
    box-shadow:0 0 18px rgba(59,130,246,.25);
}
.pipeline-card.done {
    border-color:var(--green);
    box-shadow:0 0 10px rgba(16,185,129,.18);
}
.step-badge {
    min-width:30px; height:30px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:.72rem; font-weight:700; flex-shrink:0;
}
.step-badge.pending { background:rgba(148,163,184,.15); color:var(--text-2); border:1px solid rgba(148,163,184,.3); }
.step-badge.active  { background:rgba(59,130,246,.2);  color:var(--blue);   border:1px solid var(--blue); animation:pulse-b 1.5s infinite; }
.step-badge.done    { background:rgba(16,185,129,.2);  color:var(--green);  border:1px solid var(--green); }
@keyframes pulse-b { 0%,100%{box-shadow:0 0 0 0 rgba(59,130,246,.4);} 50%{box-shadow:0 0 0 8px rgba(59,130,246,0);} }

.step-label { font-size:.86rem; font-weight:600; color:var(--text-1); }
.step-desc  { font-size:.73rem; color:var(--text-2); margin-top:2px; }

/* Metric cards */
.metric-row { display:flex; gap:1rem; margin:1rem 0; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:130px;
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:14px; padding:1.1rem; text-align:center;
}
.metric-val { font-size:1.75rem; font-weight:700; color:var(--text-1); }
.metric-lbl { font-size:.7rem; font-weight:600; color:var(--text-2);
              text-transform:uppercase; letter-spacing:.08em; margin-top:4px; }
.metric-card.blue   { border-color:rgba(59,130,246,.4);  }
.metric-card.green  { border-color:rgba(16,185,129,.4);  }
.metric-card.violet { border-color:rgba(139,92,246,.4);  }
.metric-card.amber  { border-color:rgba(245,158,11,.4);  }
.metric-card.cyan   { border-color:rgba(6,182,212,.4);   }

/* Explain box */
.explain-box {
    background:linear-gradient(135deg,#0f172a 0%,#1a1a2e 100%);
    border:1px solid rgba(99,102,241,.25); border-radius:14px; padding:1.5rem;
    font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#c4b5fd;
    white-space:pre-wrap; line-height:1.7; margin-top:1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:var(--bg-card) !important; border-radius:12px !important;
    padding:4px !important; border:1px solid var(--border) !important; gap:2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px !important; padding:8px 18px !important;
    font-size:.85rem !important; font-weight:600 !important;
    color:var(--text-2) !important; background:transparent !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(99,102,241,.2) !important;
    color:var(--text-1) !important; border:1px solid rgba(99,102,241,.4) !important;
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    border:none !important; border-radius:10px !important; color:white !important;
    font-weight:600 !important; font-size:.95rem !important;
    padding:.65rem 1.8rem !important; letter-spacing:.03em !important;
    transition:all .2s ease !important; box-shadow:0 4px 15px rgba(99,102,241,.3) !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 6px 20px rgba(99,102,241,.5) !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background:var(--bg-card) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color:var(--text-2) !important; font-size:.72rem !important;
    text-transform:uppercase !important; letter-spacing:.12em !important;
    font-weight:700 !important; margin-bottom:.5rem !important;
}

/* Progress */
.stProgress > div > div {
    background:linear-gradient(90deg,#4f46e5,#06b6d4) !important;
    border-radius:999px !important;
}

/* Misc */
hr { border-color:var(--border) !important; margin:1.5rem 0 !important; }
[data-testid="stFileUploader"] {
    background:var(--bg-card) !important;
    border:2px dashed rgba(99,102,241,.35) !important;
    border-radius:16px !important; padding:1.5rem !important;
}
[data-testid="stExpander"] {
    background:var(--bg-card) !important;
    border:1px solid var(--border) !important; border-radius:12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hero">
  <div class="hero-badge">🔬 Agentic AI &middot; SAM ViT-B &middot; Histopathology</div>
  <div class="hero-title">AgentSeg</div>
  <p class="hero-subtitle">
    Uncertainty-Guided Iterative Nucleus Segmentation &nbsp;&middot;&nbsp;
    N1 Epistemic Decomposition &nbsp;&middot;&nbsp;
    N2 Autonomous Prompt Generation &nbsp;&middot;&nbsp;
    N6 Adaptive Controller
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════
# SIDEBAR — Configuration
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Model")
    sam_ckpt = ensure_sam_checkpoint()

    st.caption("SAM ViT-B checkpoint ready ✓")
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"])

    st.markdown("---")
    st.markdown("### 🧬 AMG Settings")
    pts_per_side     = st.slider("Points per side",         8,   32,  16, 4)
    pred_iou_thresh  = st.slider("Pred IoU threshold",      0.50, 0.95, 0.86, 0.01)
    stability_thresh = st.slider("Stability score thresh",  0.70, 0.98, 0.90, 0.01)
    min_mask_area    = st.slider("Min mask area (px²)",     30,  500, 150, 10)

    st.markdown("---")
    st.markdown("### 🔄 Morphology Filter")
    area_min     = st.slider("Area min",       20,  200,  60, 10)
    area_max     = st.slider("Area max",     1000, 6000, 3500, 100)
    solidity_min = st.slider("Solidity min",  0.30, 0.90, 0.50, 0.05)

    st.markdown("---")
    st.markdown("### 🤖 Agentic (N1→N2→N6)")
    n_passes        = st.slider("N1 stochastic passes",      2, 8, 4, 1)
    noise_std_val   = st.slider("N1 score noise std",       0.01, 0.20, 0.08, 0.01)
    n2_max_regions  = st.slider("N2 max clusters",          1, 5, 3, 1)
    n6_max_iters    = st.slider("N6 max correction iters",  1, 5, 3, 1)
    run_agentic     = st.checkbox("Run N1→N2→N6 Agentic loop", value=True)

    st.markdown("---")
    st.markdown("### 📊 Display")
    show_entropy  = st.checkbox("Show entropy maps",  value=True)
    show_pipe_log = st.checkbox("Show pipeline log",  value=True)

    st.markdown("---")
    st.markdown(
        "<p style='color:#475569;font-size:.7rem;text-align:center;'>"
        "SAM ViT-B &middot; RAM-optimised &middot; CPU/GPU<br>"
        "N1 &middot; N2 &middot; N6 modules enabled</p>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════
# PIPELINE METADATA
# ════════════════════════════════════════════════════════════
STEPS = [
    ("Macenko Stain Normalisation",      "Normalises H&E stain variation across slides",            ""),
    ("SAM Automatic Mask Generation",    "Auto-generates candidate nucleus masks (AMG)",             ""),
    ("Non-Maximum Suppression",          "Removes overlapping low-confidence masks",                 ""),
    ("Morphology Filtering",             "Rejects artifacts by shape, solidity & intensity",         ""),
    ("Watershed Separation",             "Splits touching nuclei into individual instances",          ""),
    ("Initial Segmentation Ready",       "First-pass instance map complete",                         ""),
    ("N1: Monte Carlo Inference",        "Stochastic passes — score perturbation sampling",          "N1"),
    ("N1: Predictive Entropy Map",       "H[p(y|x)] — total uncertainty quantification",            "N1"),
    ("N1: BALD Decomposition",           "Epistemic vs aleatoric uncertainty separation",            "N1"),
    ("N1: Epistemic Region Selection",   "Boundary-focused correctable-region mask",                 "N1"),
    ("N2: Automatic Prompt Generation",  "Cluster centroids + bboxes from N1 epistemic map",        "N2"),
    ("N2: Local Patch SAM Refinement",   "Corrective SAM calls on uncertain boundary regions",      "N2"),
    ("N6: Iterative Evaluation",         "IoU tracking, adaptive param updates, convergence",       "N6"),
    ("N6: Explainability Layer",         "Natural language explanation of each decision",            "N6"),
    ("Final Refined Segmentation",       "Complete agentic output with explainability report",       ""),
]

MOD_COLOR = {"N1": "#f59e0b", "N2": "#06b6d4", "N6": "#10b981", "": "#94a3b8"}


def render_pipeline(current: int) -> str:
    html = ""
    for i, (name, desc, mod) in enumerate(STEPS):
        if i < current:
            cls, badge_cls, icon = "done", "done", "&#10003;"
        elif i == current:
            cls, badge_cls, icon = "active", "active", str(i + 1)
        else:
            cls, badge_cls, icon = "", "pending", str(i + 1)

        mod_span = (
            f'<span style="color:{MOD_COLOR[mod]};font-size:.68rem;font-weight:700;">[{mod}]</span> '
            if mod else ""
        )
        html += (
            f'<div class="pipeline-card {cls}">'
            f'<div class="step-badge {badge_cls}">{icon}</div>'
            f'<div><div class="step-label">{mod_span}{name}</div>'
            f'<div class="step-desc">{desc}</div></div></div>'
        )
    return html


# ════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS  (ported verbatim from research notebook)
# ════════════════════════════════════════════════════════════

class MacenkoNormalizer:
    """Macenko H&E stain normaliser."""

    def __init__(self, beta: float = 0.15, alpha: float = 1.0):
        self.beta = beta
        self.alpha = alpha
        self.HERef = None
        self.maxCRef = None

    def _to_od(self, img_flat):
        img_flat = img_flat.astype(np.float64) + 1e-6
        OD = -np.log(img_flat / 255.0)
        OD[(OD < self.beta).any(axis=1)] = 0
        return OD

    def _get_stain_matrix(self, OD):
        OD_valid = OD[~(OD == 0).any(axis=1)]
        if len(OD_valid) < 10:
            return np.eye(3, 2)
        _, V = np.linalg.eigh(np.cov(OD_valid.T))
        V = V[:, [2, 1]]
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        that = OD_valid @ V
        phi = np.arctan2(that[:, 1], that[:, 0])
        vMin = V @ np.array(
            [[np.cos(np.percentile(phi, self.alpha))],
             [np.sin(np.percentile(phi, self.alpha))]]
        )
        vMax = V @ np.array(
            [[np.cos(np.percentile(phi, 100 - self.alpha))],
             [np.sin(np.percentile(phi, 100 - self.alpha))]]
        )
        HE = np.hstack([vMin, vMax]) if vMin[0] > vMax[0] else np.hstack([vMax, vMin])
        return HE

    def fit(self, target_img):
        OD = self._to_od(target_img.reshape(-1, 3))
        self.HERef = self._get_stain_matrix(OD)
        C = np.linalg.lstsq(self.HERef, OD.T, rcond=None)[0]
        self.maxCRef = np.percentile(C, 99, axis=1)

    def transform(self, source_img):
        if self.HERef is None:
            return source_img
        h, w, _ = source_img.shape
        OD = self._to_od(source_img.reshape(-1, 3))
        try:
            HE = self._get_stain_matrix(OD)
            C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
            maxC = np.percentile(C, 99, axis=1)
            C = C / (maxC[:, None] + 1e-6) * self.maxCRef[:, None]
            out = np.clip(np.exp(-(self.HERef @ C).T) * 255, 0, 255).astype(np.uint8)
            return out.reshape(h, w, 3)
        except Exception:
            return source_img


def mask_nms(sam_masks, iou_threshold=0.3, max_masks=50):
    """Non-maximum suppression over SAM masks."""
    if not sam_masks:
        return []
    sorted_m = sorted(sam_masks, key=lambda m: m["predicted_iou"], reverse=True)
    kept = []
    suppressed = set()
    for i, mi in enumerate(sorted_m):
        if i in suppressed or len(kept) >= max_masks:
            continue
        kept.append(mi)
        si = mi["segmentation"]
        ai = si.sum()
        for j in range(i + 1, len(sorted_m)):
            if j in suppressed:
                continue
            sj = sorted_m[j]["segmentation"]
            inter = np.logical_and(si, sj).sum()
            if inter / (min(ai, sj.sum()) + 1e-6) > iou_threshold:
                suppressed.add(j)
    return kept


class BalancedMorphologyFilter:
    """3-stage morphology filter with fallback guarantee."""

    def __init__(
        self,
        area_min=60, area_max=3500,
        solidity_min=0.50, eccentricity_max=0.95,
        extent_min=0.28, intensity_margin=0.35,
        min_keep=3, fallback_k=8,
    ):
        self.area_min = area_min
        self.area_max = area_max
        self.solidity_min = solidity_min
        self.eccentricity_max = eccentricity_max
        self.extent_min = extent_min
        self.intensity_margin = intensity_margin
        self.min_keep = min_keep
        self.fallback_k = fallback_k

    def _intensity_thresh(self, gray):
        if gray is None:
            return 220
        med = float(np.median(gray))
        return 255 if med < 80 else min(230, med * (1 + self.intensity_margin))

    def _passes(self, seg, gray, i_thresh):
        props = regionprops(seg.astype(int))
        if not props:
            return False
        p = props[0]
        if not (self.area_min < p.area < self.area_max):
            return False
        if p.solidity < self.solidity_min:
            return False
        if p.eccentricity > self.eccentricity_max:
            return False
        if p.extent < self.extent_min:
            return False
        if gray is not None and i_thresh < 255:
            if gray[seg.astype(bool)].mean() > i_thresh:
                return False
        return True

    def filter_masks(self, sam_masks, gray=None):
        if not sam_masks:
            return [], []
        i_thresh = self._intensity_thresh(gray)
        accepted, neg_pts = [], []
        for m in sam_masks:
            seg = m["segmentation"]
            if self._passes(seg, gray, i_thresh):
                accepted.append(m)
            else:
                yx = np.argwhere(seg)
                if len(yx):
                    cy, cx = yx.mean(axis=0)
                    neg_pts.append((int(cx), int(cy)))
        if len(accepted) < self.min_keep:
            accepted2 = [
                m for m in sam_masks
                if self._passes(m["segmentation"], None, 255)
            ]
            if len(accepted2) > len(accepted):
                accepted = accepted2
                neg_pts = []
        if len(accepted) < self.min_keep and sam_masks:
            k = min(self.fallback_k, len(sam_masks))
            accepted = sorted(
                sam_masks, key=lambda m: m["predicted_iou"], reverse=True
            )[:k]
            neg_pts = []
        return accepted, neg_pts


def balanced_watershed(combined_mask, min_area=60):
    """Watershed with adaptive footprint and fallback markers."""
    binary = combined_mask.astype(bool)
    binary = remove_small_objects(binary, min_size=min_area)
    binary = ndi.binary_fill_holes(binary)
    binary = binary_closing(binary, disk(2))
    if not binary.any():
        return np.zeros_like(combined_mask, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary)
    vals = distance[binary]
    if len(vals) == 0:
        return np.zeros_like(combined_mask, dtype=np.int32)
    typical_r = max(3, int(np.percentile(vals, 65)))
    min_dist  = max(4, int(0.6 * typical_r))
    coords = peak_local_max(
        distance,
        footprint=disk(typical_r),
        labels=binary.astype(int),
        min_distance=min_dist,
    )
    if len(coords) == 0:
        yx = np.argwhere(binary)
        if len(yx):
            coords = np.array([yx.mean(axis=0).astype(int)])
        else:
            return np.zeros_like(combined_mask, dtype=np.int32)
    markers = np.zeros_like(distance, dtype=np.int32)
    for k, (y, x) in enumerate(coords):
        markers[y, x] = k + 1
    inst = watershed(-distance, markers, mask=binary, compactness=0.01)
    del distance, markers, vals, binary
    if inst.max() == 0:
        inst, _ = label(combined_mask.astype(bool), return_num=True)
    return inst.astype(np.int32)


def compute_mask_entropy(masks_pred, scores):
    """Pixel-wise entropy from SAM's 3 candidates."""
    if masks_pred is None or len(masks_pred) == 0 or scores is None:
        return np.zeros((1, 1), dtype=np.float32)
    with torch.no_grad():
        probs  = F.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
        masks_t = torch.tensor(masks_pred.astype(np.float32))
        p_fg   = (probs[:, None, None] * masks_t).sum(0).clamp(1e-6, 1 - 1e-6)
        ent    = -(p_fg * torch.log(p_fg) + (1 - p_fg) * torch.log(1 - p_fg))
        result = ent.numpy()
        del probs, masks_t, p_fg, ent
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def get_bbox(mask, padding=4):
    """Bounding box of a binary mask with padding."""
    yx = np.argwhere(mask)
    if len(yx) == 0:
        return None
    y1, x1 = yx.min(axis=0)
    y2, x2 = yx.max(axis=0)
    H, W = mask.shape
    return np.array(
        [max(0, x1 - padding), max(0, y1 - padding),
         min(W - 1, x2 + padding), min(H - 1, y2 + padding)],
        dtype=np.float32,
    )


# ── N1: Uncertainty decomposition ────────────────────────────
def n1_stochastic_passes(predictor, point_coords, point_labels,
                          bbox=None, n_passes=4, noise_std=0.08):
    """1 real SAM call + (n_passes-1) score perturbations."""
    prob_maps  = []
    base_mask  = None
    base_score = 0.0
    try:
        with torch.no_grad():
            if bbox is not None:
                mp, sc, _ = predictor.predict(box=bbox, multimask_output=True)
            else:
                mp, sc, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
        best_idx   = int(np.argmax(sc))
        base_mask  = mp[best_idx].astype(bool)
        base_score = float(sc[best_idx])
        probs_r = F.softmax(torch.tensor(sc, dtype=torch.float32), dim=0)
        masks_t = torch.tensor(mp.astype(np.float32))
        p_fg    = (probs_r[:, None, None] * masks_t).sum(0).clamp(1e-7, 1 - 1e-7)
        prob_maps.append(p_fg.numpy().copy())
        del p_fg
        for _ in range(n_passes - 1):
            noise  = np.random.normal(0, noise_std, sc.shape).astype(np.float32)
            sc_n   = sc + noise
            probs_n = F.softmax(torch.tensor(sc_n, dtype=torch.float32), dim=0)
            p_n    = (probs_n[:, None, None] * masks_t).sum(0).clamp(1e-7, 1 - 1e-7)
            prob_maps.append(p_n.numpy().copy())
            del probs_n, p_n, noise, sc_n
        del probs_r, masks_t, mp, sc
    except Exception:
        pass
    return prob_maps, base_mask, base_score


def n1_predictive_entropy(prob_maps):
    """H[p(y|x)] — total uncertainty."""
    if not prob_maps:
        return np.zeros((1, 1), dtype=np.float32)
    p_mean = np.stack(prob_maps).mean(axis=0).clip(1e-7, 1 - 1e-7)
    ent = -(p_mean * np.log(p_mean) + (1 - p_mean) * np.log(1 - p_mean))
    return ent.astype(np.float32)


def n1_expected_entropy(prob_maps):
    """E[H[p(y|x,w)]] — aleatoric component."""
    if not prob_maps:
        return np.zeros((1, 1), dtype=np.float32)
    per = []
    for p in prob_maps:
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        per.append(-(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)))
    return np.stack(per).mean(axis=0).astype(np.float32)


def n1_normalize(m):
    """Scale map to [0, 1]. Handles flat maps safely."""
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-10:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def n1_analyze(epi_map, ale_map, base_mask):
    """Boundary-focused epistemic vs aleatoric classification."""
    H, W = epi_map.shape[:2]
    if base_mask is not None and base_mask.any():
        dilated       = binary_dilation(base_mask, disk(4))
        boundary_ring = dilated & ~base_mask
    else:
        boundary_ring = np.ones((H, W), dtype=bool)
    epi_n = n1_normalize(epi_map)
    ale_n = n1_normalize(ale_map)
    epi_boundary = epi_n[boundary_ring]
    ale_boundary = ale_n[boundary_ring]
    if len(epi_boundary) == 0:
        epi_mask = epi_n > 0.5
        return False, "no boundary ring pixels found", 0.0, epi_mask.astype(bool)
    n_epi_dom = int((epi_boundary > ale_boundary).sum())
    epi_ratio = n_epi_dom / (len(epi_boundary) + 1e-6)
    if epi_boundary.max() > 0:
        epi_thresh = float(np.percentile(epi_boundary, 70))
        epi_mask   = (epi_n >= epi_thresh) & boundary_ring & (epi_n >= ale_n * 0.3)
    else:
        epi_mask  = boundary_ring.copy()
        epi_ratio = 0.5
    if epi_ratio >= 0.40:
        reason = (
            f"epistemic dominant in boundary: {epi_ratio:.1%} "
            "of boundary pixels are correctable"
        )
    else:
        reason = (
            f"aleatoric dominant in boundary: only {epi_ratio:.1%} "
            "of boundary pixels are correctable (still passing to N2)"
        )
    return True, reason, epi_ratio, epi_mask.astype(bool)


# ── N2: Automatic prompt generation ──────────────────────────
def n2_threshold_regions(epi_map, epi_mask, threshold_percentile=80):
    """Threshold N1 epistemic map within N1-approved mask."""
    if epi_map is None or epi_map.size == 0 or not epi_mask.any():
        shape = epi_map.shape if epi_map is not None else (1, 1)
        return np.zeros(shape, dtype=np.uint8)
    epi_vals = epi_map[epi_mask]
    if len(epi_vals) == 0 or float(epi_vals.max()) < 1e-10:
        return np.zeros(epi_map.shape, dtype=np.uint8)
    thresh = float(np.percentile(epi_vals, threshold_percentile))
    return ((epi_map >= thresh) & epi_mask).astype(np.uint8)


def n2_extract_clusters(binary_mask, min_area=10, max_regions=3):
    """Connected-component analysis on N1-gated epistemic mask."""
    if binary_mask is None or int(binary_mask.sum()) == 0:
        return []
    labeled, n = label(binary_mask.astype(bool), return_num=True)
    if n == 0:
        return []
    clusters = []
    for p in regionprops(labeled):
        if p.area < min_area:
            continue
        y0, x0, y1, x1 = p.bbox
        cy, cx = p.centroid
        clusters.append({
            "centroid": (int(cx), int(cy)),
            "bbox":     (int(x0), int(y0), int(x1), int(y1)),
            "area":     p.area,
        })
    return sorted(clusters, key=lambda c: c["area"], reverse=True)[:max_regions]


def n2_point_prompts(clusters, current_mask):
    """Centroid point prompts from N1 epistemic clusters."""
    if not clusters:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)
    pts, lbls = [], []
    for c in clusters:
        x, y = c["centroid"]
        x, y = max(0, x), max(0, y)
        pts.append([float(x), float(y)])
        if (current_mask is not None
                and y < current_mask.shape[0]
                and x < current_mask.shape[1]):
            lbls.append(int(current_mask[y, x]))
        else:
            lbls.append(1)
    return np.array(pts, dtype=np.float32), np.array(lbls, dtype=np.int32)


def n2_box_prompts(clusters, img_shape, padding=3):
    """Bounding box prompts around N1 epistemic clusters."""
    if not clusters:
        return []
    H, W = img_shape[:2]
    boxes = []
    for c in clusters:
        x0, y0, x1, y1 = c["bbox"]
        box = np.array(
            [max(0, x0 - padding), max(0, y0 - padding),
             min(W - 1, x1 + padding), min(H - 1, y1 + padding)],
            dtype=np.float32,
        )
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def n2_validate_prompts(point_coords, point_labels, boxes, img_shape):
    """Clip all coords to image bounds; drop degenerate boxes."""
    H, W = img_shape[:2]
    if point_coords is not None and len(point_coords) > 0:
        point_coords = np.clip(point_coords, 0, [W - 1, H - 1])
    clean = []
    for b in (boxes or []):
        b = np.clip(b, 0, [W - 1, H - 1, W - 1, H - 1])
        if b[2] > b[0] and b[3] > b[1]:
            clean.append(b)
    return point_coords, point_labels, clean


def n2_corrective_sam(predictor, anchor_pt, point_coords, point_labels,
                       boxes, current_mask, current_score, n_iters=3):
    """SAM calls with anchor + N2-generated prompts."""
    best_mask  = current_mask.copy() if current_mask is not None else None
    best_score = current_score
    all_pts  = [[float(anchor_pt[0]), float(anchor_pt[1])]]
    all_lbls = [1]
    for pt, lbl in zip(point_coords, point_labels):
        all_pts.append([float(pt[0]), float(pt[1])])
        all_lbls.append(int(lbl))
    for _ in range(n_iters):
        try:
            with torch.no_grad():
                mp, sc, _ = predictor.predict(
                    point_coords=np.array(all_pts, dtype=np.float32),
                    point_labels=np.array(all_lbls, dtype=np.int32),
                    multimask_output=True,
                )
            idx = int(np.argmax(sc))
            if float(sc[idx]) > best_score:
                best_score = float(sc[idx])
                best_mask  = mp[idx].astype(bool)
            del mp, sc
        except Exception:
            break
    for box in (boxes or []):
        try:
            with torch.no_grad():
                mp, sc, _ = predictor.predict(box=box, multimask_output=True)
            idx = int(np.argmax(sc))
            if float(sc[idx]) > best_score:
                best_score = float(sc[idx])
                best_mask  = mp[idx].astype(bool)
            del mp, sc
        except Exception:
            continue
    return best_mask, best_score


# ── N6: Adaptive controller ───────────────────────────────────
def n6_evaluate(iou_before, iou_after, epi_ratio, n_clusters,
                entropy_before, entropy_after, iteration, max_iters):
    """Assess N2 refinement quality. Returns state dict."""
    return {
        "iou_delta":     iou_after - iou_before,
        "entropy_delta": entropy_after - entropy_before,
        "epi_ratio":     epi_ratio,
        "n_clusters":    n_clusters,
        "converged": (
            abs(iou_after - iou_before) < 0.003
            or entropy_after < 0.03
            or iteration >= max_iters - 1
        ),
        "iteration": iteration,
    }


def n6_decide(state, params):
    """Agentic policy engine: read N1+N2 state, output action + updated params."""
    p = dict(params)
    action = "continue"
    reason = []
    if state["converged"]:
        action = "stop_early"
        reason.append("convergence criterion met")
    elif state["iou_delta"] < -0.02:
        action = "fallback"
        reason.append(f'IoU degraded {state["iou_delta"]:+.3f} — reverting to N1 mask')
    elif state["iou_delta"] > 0.03:
        action = "increase_iters"
        p["n_iters"] = min(p.get("n_iters", 3) + 1, 5)
        reason.append(f'strong N2 gain {state["iou_delta"]:+.3f} — extending budget')
    elif state["epi_ratio"] < 0.3 and state["n_clusters"] > 0:
        action = "reduce_iters"
        p["n_iters"] = max(p.get("n_iters", 3) - 1, 1)
        reason.append(f'N1 epi_ratio={state["epi_ratio"]:.1%} low — reducing N2 budget')
    elif state["n_clusters"] >= 3:
        action = "increase_patch"
        p["box_padding"] = min(p.get("box_padding", 3) + 2, 8)
        reason.append(f'{state["n_clusters"]} N2 clusters — expanding box padding')
    elif state["n_clusters"] == 0:
        action = "lower_threshold"
        p["percentile"] = max(p.get("percentile", 80) - 10, 50)
        reason.append("no N2 clusters — lowering N2 percentile threshold")
    return action, reason, p


def n6_explain(img_label, n1_reason, epi_ratio, n_clusters,
               n2_npts, n2_nboxes, n6_action, n6_reasons,
               iou_before, iou_after, n_epi_refined, n_ale_skipped, n_fallbacks):
    """Natural-language explanation of N1->N2->N6 pipeline decisions."""
    action_text = {
        "stop_early":      "Stopped early — convergence criterion satisfied.",
        "fallback":        "Reverted to N1 base mask — N2 degraded quality.",
        "increase_iters":  "Extended N2 iteration budget — strong improvement detected.",
        "reduce_iters":    "Reduced N2 iterations — N1 showed low epistemic ratio.",
        "increase_patch":  "Expanded N2 box padding — multiple clusters needed context.",
        "lower_threshold": "Lowered N2 percentile threshold — no clusters found.",
        "continue":        "Kept current parameters — steady progress.",
        "N/A":             "No N6 decision recorded.",
    }.get(n6_action, n6_action)
    n1_gate = (
        "REFINE -> N2 runs" if "epistemic" in n1_reason
        else "ALEATORIC -> N2 reduced budget"
    )
    lines = [
        f"=== N6 Explanation --- {img_label} ===\n",
        "[N1 - WHERE to refine]",
        f"  Classification : {n1_gate}",
        f"  Reason         : {n1_reason}",
        f"  Epistemic ratio: {epi_ratio:.1%} of boundary pixels are correctable\n",
        "[N2 - HOW to refine]",
        f"  Clusters detected : {n_clusters} epistemic boundary regions",
        f"  Point prompts     : {n2_npts} auto-generated from N1 cluster centroids",
        f"  Box prompts       : {n2_nboxes} auto-generated from N1 cluster extents",
        f"  IoU change        : {iou_before:.3f} -> {iou_after:.3f} ({iou_after - iou_before:+.3f})\n",
        "[N6 - HOW MUCH / STRATEGY]",
        f"  Action    : {action_text}",
        f"  Reasoning : {'; '.join(n6_reasons) if n6_reasons else 'steady progress'}\n",
        "[Image Summary]",
        f"  N2 epistemic refined  : {n_epi_refined}",
        f"  N2 reduced budget     : {n_ale_skipped}  (N1 aleatoric signal)",
        f"  Fallbacks activated   : {n_fallbacks}",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ════════════════════════════════════════════════════════════

def fig_to_pil(fig):
    """Render matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def draw_masks_overlay(image_rgb, masks, alpha=0.45):
    """Colour-coded mask overlay on image."""
    overlay = image_rgb.copy().astype(np.float32)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(masks), 1)))
    for i, m in enumerate(masks):
        seg = m["segmentation"] if isinstance(m, dict) else (m > 0)
        c = (np.array(cmap[i % len(cmap)][:3]) * 255).astype(np.uint8)
        overlay[seg] = overlay[seg] * (1 - alpha) + c * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_instance_rgb(inst_map):
    """Convert integer instance map to false-colour RGB."""
    n = int(inst_map.max())
    if n == 0:
        return np.zeros((*inst_map.shape, 3), dtype=np.uint8)
    cmap = plt.cm.nipy_spectral
    rgb  = np.zeros((*inst_map.shape, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        c = (np.array(cmap(i / max(n, 1))[:3]) * 255).astype(np.uint8)
        rgb[inst_map == i] = c
    return rgb


def make_n1_fig(epi_map, ale_map, image_rgb):
    """Side-by-side N1 uncertainty visualisation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#111827")
    for ax in axes:
        ax.set_facecolor("#111827")
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input", color="#f1f5f9", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    im1 = axes[1].imshow(n1_normalize(epi_map), cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[1], fraction=0.046).ax.tick_params(colors="#94a3b8")
    axes[1].set_title("N1: Epistemic Uncertainty\n(correctable — N2 targets this)",
                      color="#fbbf24", fontsize=9, fontweight="bold")
    axes[1].axis("off")
    im2 = axes[2].imshow(n1_normalize(ale_map), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[2], fraction=0.046).ax.tick_params(colors="#94a3b8")
    axes[2].set_title("N1: Aleatoric Uncertainty\n(inherent data noise)",
                      color="#38bdf8", fontsize=9, fontweight="bold")
    axes[2].axis("off")
    plt.tight_layout(pad=0.5)
    return fig


def make_comparison_fig(original, normalized, masks_vis, inst_vis):
    """4-panel comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.patch.set_facecolor("#111827")
    panels = [
        (original,   "Input Image"),
        (normalized, "Stain Normalised"),
        (masks_vis,  "SAM Masks"),
        (inst_vis,   "Instance Map"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.set_facecolor("#111827")
        ax.imshow(img)
        ax.set_title(title, color="#f1f5f9", fontsize=9, fontweight="bold")
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    return fig


# ════════════════════════════════════════════════════════════
# SAM MODEL LOADER (cached — loads once per session)
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_sam_model(checkpoint_path: str, device_str: str):
    """Load SAM ViT-B and return (sam, predictor, error_string)."""
    if not SAM_AVAILABLE:
        return None, None, "segment-anything not installed"
    if not os.path.exists(checkpoint_path):
        return None, None, (
            f"Checkpoint not found: `{checkpoint_path}`\n\n"
            "Download with:\n"
            "```bash\n"
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "```"
        )
    try:
        dev = (
            device_str if device_str != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(dev)
        sam.eval()
        predictor = SamPredictor(sam)
        return sam, predictor, None
    except Exception as exc:
        return None, None, str(exc)


# ════════════════════════════════════════════════════════════
# LAYOUT — Upload | Pipeline tracker
# ════════════════════════════════════════════════════════════
col_up, col_pipe = st.columns([1, 1], gap="large")

with col_up:
    st.markdown("#### 📂 Upload Histopathology Image")
    uploaded = st.file_uploader(
        "Drop an H&E stained image (PNG, JPG, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded:
        raw_bytes = np.frombuffer(uploaded.read(), np.uint8)
        raw_bgr   = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            st.error("Could not decode image. Please upload a valid image file.")
            st.stop()
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        # Resize if very large
        H0, W0 = raw_rgb.shape[:2]
        if max(H0, W0) > 1024:
            scale   = 1024 / max(H0, W0)
            raw_rgb = cv2.resize(
                raw_rgb, (int(W0 * scale), int(H0 * scale)),
                interpolation=cv2.INTER_AREA,
            )
            st.info(f"Resized to {raw_rgb.shape[1]}x{raw_rgb.shape[0]} px for processing.")
        st.image(
            raw_rgb,
            caption=f"{uploaded.name}  |  {raw_rgb.shape[1]}x{raw_rgb.shape[0]} px",
            use_container_width=True,
        )
        run_btn = st.button(
            "🚀  Run Agentic Segmentation Pipeline",
            use_container_width=True,
        )
    else:
        st.markdown(
            """
<div style="text-align:center;padding:3rem 1rem;color:#475569;">
  <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
  <div style="font-size:1rem;font-weight:600;color:#64748b;">
    Upload an H&amp;E stained histopathology image
  </div>
  <div style="font-size:.82rem;margin-top:.5rem;">PNG · JPG · TIFF supported</div>
</div>""",
            unsafe_allow_html=True,
        )
        run_btn = False

with col_pipe:
    st.markdown("#### 🔄 Pipeline Progress")
    pipe_slot = st.empty()
    pipe_slot.markdown(render_pipeline(-1), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ════════════════════════════════════════════════════════════
if uploaded and run_btn:
    st.markdown("---")

    # ── Load SAM ─────────────────────────────────────────────
    with st.spinner("Loading SAM ViT-B..."):
        sam, predictor, sam_err = load_sam_model(sam_ckpt, device_choice)

    if sam_err:
        st.error(f"**SAM error:** {sam_err}")
        st.stop()

    # ── AGENTIC: Auto-calibrate thresholds from image ─────────
    amg_cfg = auto_calibrate_amg(raw_rgb)
    st.info(
        f"🤖 **Agentic Auto-Calibration:** {amg_cfg['reason']} | "
        f"pts={amg_cfg['pts']} iou={amg_cfg['iou']} "
        f"stability={amg_cfg['stability']} min_area={amg_cfg['min_area']}"
    )

    amg_strict = SamAutomaticMaskGenerator(
        sam,
        points_per_side        = amg_cfg['pts'],
        pred_iou_thresh        = amg_cfg['iou'],
        stability_score_thresh = amg_cfg['stability'],
        min_mask_region_area   = amg_cfg['min_area'],
        box_nms_thresh         = 0.5,
        crop_n_layers          = 0,
    )
    amg_permissive = SamAutomaticMaskGenerator(
        sam,
        points_per_side        = max(8, amg_cfg['pts'] - 8),
        pred_iou_thresh        = max(0.60, amg_cfg['iou'] - 0.08),
        stability_score_thresh = max(0.70, amg_cfg['stability'] - 0.05),
        min_mask_region_area   = max(20, amg_cfg['min_area'] // 2),
        box_nms_thresh         = 0.6,
        crop_n_layers          = 0,
    )
    morph_filter = BalancedMorphologyFilter(
        area_min     = amg_cfg['area_min'],
        area_max     = amg_cfg['area_max'],
        solidity_min = amg_cfg['solidity_min'],
    )
    normalizer = MacenkoNormalizer()

    # ── progress helpers ─────────────────────────────────────
    prog_bar = st.progress(0, text="Initialising...")
    log_lines = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    def step(idx: int, msg: str = "") -> None:
        pipe_slot.markdown(render_pipeline(idx), unsafe_allow_html=True)
        pct = min(100, int(idx / len(STEPS) * 100))
        prog_bar.progress(pct, text=msg)

    results = {}

    try:
        # STEP 0 — Stain normalisation ────────────────────────
        step(0, "Step 1 — Macenko stain normalisation...")
        t0 = time.time()
        normalizer.fit(raw_rgb)
        norm_rgb = normalizer.transform(raw_rgb)
        gray = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2GRAY)
        log(f"[E1] Stain normalisation: {time.time()-t0:.2f}s")

        # STEP 1 — SAM AMG ────────────────────────────────────
        step(1, "Step 2 — SAM automatic mask generation...")
        t0 = time.time()
        with torch.no_grad():
            raw_strict = amg_strict.generate(norm_rgb)
        log(f"[AMG] {len(raw_strict)} candidate masks in {time.time()-t0:.2f}s")

        # STEP 2 — NMS ────────────────────────────────────────
        step(2, "Step 3 — Non-maximum suppression...")
        nms_masks = mask_nms(raw_strict, iou_threshold=0.3, max_masks=50)
        del raw_strict
        log(f"[NMS] After NMS: {len(nms_masks)} masks")

        # STEP 3 — Morphology filter ──────────────────────────
        step(3, "Step 4 — Morphology filtering...")
        acc_masks, _ = morph_filter.filter_masks(nms_masks, gray)
        del nms_masks
        if len(acc_masks) < 5:
            with torch.no_grad():
                raw_perm = amg_permissive.generate(norm_rgb)
            nms_perm = mask_nms(raw_perm, iou_threshold=0.4, max_masks=50)
            del raw_perm
            acc_perm, _ = morph_filter.filter_masks(nms_perm, gray)
            del nms_perm
            if len(acc_perm) > len(acc_masks):
                acc_masks = acc_perm
                log("[Fallback] Permissive AMG used")
        log(f"[Filter] Accepted: {len(acc_masks)} masks")

        # STEP 4 — Watershed ──────────────────────────────────
        step(4, "Step 5 — Watershed instance separation...")
        combined = np.zeros(norm_rgb.shape[:2], dtype=np.float32)
        for m in acc_masks:
            combined[m["segmentation"]] = 1.0
        inst_map  = balanced_watershed(combined, min_area=60)
        pred_bin  = (inst_map > 0).astype(np.uint8)
        del combined
        log(f"[Watershed] Instances: {inst_map.max()}")

        # STEP 5 — Initial segmentation ready ─────────────────
        step(5, "Step 6 — Initial segmentation complete...")
        masks_vis = draw_masks_overlay(norm_rgb, acc_masks)
        inst_vis  = draw_instance_rgb(inst_map)
        results.update(
            norm_rgb=norm_rgb, masks_vis=masks_vis,
            inst_vis=inst_vis, n_initial=int(inst_map.max()),
        )
        gc.collect()

        # ── AGENTIC LOOP: N1->N2->N6 ─────────────────────────
        agentic = dict(
            ran=False, n_epi=0, n_ale=0, n_fallbacks=0,
            epi_maps=[], ale_maps=[], final_inst=inst_map,
            final_vis=inst_vis, mean_entropy=0.0,
            n_final=int(inst_map.max()), explain="",
            all_diags=[],
        )

        if run_agentic:
            step(6, "N1 — Monte Carlo stochastic inference...")

            # Build anchors from AMG mask centroids
            anchors = []
            for m in acc_masks:
                yx = np.argwhere(m["segmentation"])
                if len(yx) == 0:
                    continue
                cy, cx = yx.mean(axis=0)
                anchors.append({
                    "centroid": (int(cx), int(cy)),
                    "mask":     m["segmentation"],
                    "bbox":     get_bbox(m["segmentation"], padding=3),
                })

            with torch.no_grad():
                predictor.set_image(norm_rgb)

            refined_masks = []
            epi_maps_out  = []
            ale_maps_out  = []
            all_diags     = []
            last_diag     = {}
            last_n6_log   = []
            n_epi = n_ale = n_fb = 0
            entropy_vals  = []

            n6_params = {
                "n_iters":     n6_max_iters,
                "box_padding": 3,
                "max_regions": n2_max_regions,
                "percentile":  80,
            }

            total = len(anchors)
            for ni, anchor in enumerate(anchors):
                frac    = 6 + 8 * ni / max(total, 1)
                cur_idx = min(int(frac), len(STEPS) - 2)
                step(cur_idx,
                     f"N1->N2->N6 — nucleus {ni+1}/{total}...")

                anchor_pt = anchor["centroid"]
                bbox      = anchor["bbox"]
                pt_arr    = np.array([[anchor_pt[0], anchor_pt[1]]], dtype=np.float32)
                lbl_arr   = np.array([1], dtype=np.int32)

                # N1 stochastic passes
                prob_maps, base_mask, base_score = n1_stochastic_passes(
                    predictor, pt_arr, lbl_arr,
                    bbox=bbox, n_passes=n_passes, noise_std=noise_std_val,
                )

                if not prob_maps or base_mask is None:
                    n_fb += 1
                    if bbox is not None:
                        try:
                            with torch.no_grad():
                                mp, sc, _ = predictor.predict(
                                    box=bbox, multimask_output=True)
                            refined_masks.append(mp[int(np.argmax(sc))].astype(np.uint8))
                            del mp, sc
                        except Exception:
                            pass
                    all_diags.append({"stage": "N1_fail", "fallback": True})
                    continue

                # N1 BALD decomposition
                pred_ent = n1_predictive_entropy(prob_maps)
                exp_ent  = n1_expected_entropy(prob_maps)
                epi_map  = np.clip(pred_ent - exp_ent, 0, None)
                ale_map  = np.clip(exp_ent, 0, None)
                entropy_vals.append(float(pred_ent.mean()))
                del pred_ent, exp_ent, prob_maps

                if len(epi_maps_out) < 3:
                    epi_maps_out.append(epi_map.copy())
                    ale_maps_out.append(ale_map.copy())

                _, n1_reason, epi_ratio, epi_mask = n1_analyze(
                    epi_map, ale_map, base_mask)

                if epi_ratio >= 0.20:
                    n_epi += 1
                else:
                    n_ale += 1
                    n6_params["n_iters"] = max(1, n6_params["n_iters"] - 1)

                # N2 prompt generation
                binary   = n2_threshold_regions(
                    epi_map, epi_mask,
                    threshold_percentile=n6_params["percentile"],
                )
                clusters = n2_extract_clusters(
                    binary, min_area=10,
                    max_regions=n6_params["max_regions"],
                )
                del binary
                pt_coords, pt_labels = n2_point_prompts(clusters, base_mask)
                boxes_n2 = n2_box_prompts(
                    clusters, norm_rgb.shape,
                    padding=n6_params["box_padding"],
                )
                pt_coords, pt_labels, boxes_n2 = n2_validate_prompts(
                    pt_coords, pt_labels, boxes_n2, norm_rgb.shape)

                iou_before = float(
                    base_mask.sum()
                    / (np.logical_or(base_mask, anchor["mask"]).sum() + 1e-6)
                )
                try:
                    refined_mask, refined_score = n2_corrective_sam(
                        predictor, anchor_pt,
                        pt_coords, pt_labels, boxes_n2,
                        base_mask, base_score,
                        n_iters=n6_params["n_iters"],
                    )
                except Exception:
                    refined_mask  = base_mask.copy()
                    refined_score = base_score
                    n_fb += 1

                iou_after = float(
                    refined_mask.sum()
                    / (np.logical_or(refined_mask, anchor["mask"]).sum() + 1e-6)
                )

                # N6 evaluate & adapt
                ent_now = float(epi_map.mean())
                state   = n6_evaluate(
                    iou_before, iou_after, epi_ratio, len(clusters),
                    ent_now, ent_now * 0.75, ni, total,
                )
                n6_action, n6_reasons, n6_params = n6_decide(state, n6_params)
                last_n6_log = n6_reasons

                final_mask = base_mask if n6_action == "fallback" else refined_mask
                refined_masks.append(final_mask.astype(np.uint8))

                last_diag = {
                    "stage":      "N1_N2_N6",
                    "n1_reason":  n1_reason,
                    "epi_ratio":  epi_ratio,
                    "n_clusters": len(clusters),
                    "n2_pts":     len(pt_coords) if pt_coords is not None else 0,
                    "n2_boxes":   len(boxes_n2),
                    "iou_before": iou_before,
                    "iou_after":  iou_after,
                    "n6_action":  n6_action,
                }
                all_diags.append(last_diag)
                del epi_map, ale_map, epi_mask, base_mask, refined_mask, final_mask
                del clusters, pt_coords, pt_labels, boxes_n2

            # Build final refined map
            final_bin = np.zeros(norm_rgb.shape[:2], dtype=np.float32)
            for rm in refined_masks:
                final_bin[rm.astype(bool)] = 1.0
            final_inst = balanced_watershed(final_bin, min_area=60)
            final_vis  = draw_instance_rgb(final_inst)

            mean_ent = float(np.mean(entropy_vals)) if entropy_vals else 0.0
            expl = n6_explain(
                img_label    = "uploaded image",
                n1_reason    = last_diag.get("n1_reason",  "N/A"),
                epi_ratio    = last_diag.get("epi_ratio",  0.0),
                n_clusters   = last_diag.get("n_clusters", 0),
                n2_npts      = last_diag.get("n2_pts",     0),
                n2_nboxes    = last_diag.get("n2_boxes",   0),
                n6_action    = last_diag.get("n6_action",  "N/A"),
                n6_reasons   = last_n6_log,
                iou_before   = last_diag.get("iou_before", 0.0),
                iou_after    = last_diag.get("iou_after",  0.0),
                n_epi_refined = n_epi,
                n_ale_skipped = n_ale,
                n_fallbacks   = n_fb,
            )
            log(f"[N1->N2->N6] {total} nuclei | Epi={n_epi} Ale={n_ale} FB={n_fb}")

            agentic.update(
                ran=True, n_epi=n_epi, n_ale=n_ale, n_fallbacks=n_fb,
                epi_maps=epi_maps_out, ale_maps=ale_maps_out,
                final_inst=final_inst, final_vis=final_vis,
                mean_entropy=mean_ent, n_final=int(final_inst.max()),
                explain=expl, all_diags=all_diags,
            )
            gc.collect()

        # Done
        step(len(STEPS), "Pipeline complete!")
        prog_bar.progress(100, text="Done!")

        # ════════════════════════════════════════════════════
        # RESULTS
        # ════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📊 Results")

        n_init  = results["n_initial"]
        n_final = agentic["n_final"]

        st.markdown(
            f"""
<div class="metric-row">
  <div class="metric-card blue">
    <div class="metric-val">{n_init}</div>
    <div class="metric-lbl">Initial Instances</div>
  </div>
  <div class="metric-card green">
    <div class="metric-val">{n_final}</div>
    <div class="metric-lbl">Final Instances</div>
  </div>
  <div class="metric-card amber">
    <div class="metric-val">{agentic['n_epi']}</div>
    <div class="metric-lbl">N1 Epistemic (N2 ran)</div>
  </div>
  <div class="metric-card violet">
    <div class="metric-val">{agentic['n_ale']}</div>
    <div class="metric-lbl">N1 Aleatoric (reduced)</div>
  </div>
  <div class="metric-card cyan">
    <div class="metric-val">{agentic['mean_entropy']:.4f}</div>
    <div class="metric-lbl">Mean Entropy</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        tab_seg, tab_ent, tab_n6, tab_log = st.tabs(
            ["🖼️ Segmentation", "🧪 Entropy Maps", "🤖 N1→N2→N6", "📋 Pipeline Log"]
        )

        with tab_seg:
            st.markdown("#### Segmentation Comparison")
            fig = make_comparison_fig(
                raw_rgb, norm_rgb, results["masks_vis"], results["inst_vis"])
            st.image(fig_to_pil(fig), use_container_width=True)
            plt.close(fig)
            if agentic["ran"]:
                st.markdown("#### Before vs After Agentic Refinement")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(results["inst_vis"],
                             caption=f"Initial SAM Watershed ({n_init} instances)",
                             use_container_width=True)
                with c2:
                    st.image(agentic["final_vis"],
                             caption=f"After N1->N2->N6 Refinement ({n_final} instances)",
                             use_container_width=True)

        with tab_ent:
            if show_entropy and agentic["ran"] and agentic["epi_maps"]:
                st.markdown("#### N1 Uncertainty Maps (sample nuclei)")
                n_show = min(3, len(agentic["epi_maps"]))
                ecols  = st.columns(n_show)
                for i in range(n_show):
                    with ecols[i]:
                        fig_n1 = make_n1_fig(
                            agentic["epi_maps"][i],
                            agentic["ale_maps"][i],
                            norm_rgb,
                        )
                        st.image(fig_to_pil(fig_n1),
                                 caption=f"Nucleus {i+1}",
                                 use_container_width=True)
                        plt.close(fig_n1)
                if agentic["epi_maps"]:
                    st.markdown("#### Aggregate Epistemic Map")
                    agg = np.stack(agentic["epi_maps"]).mean(axis=0)
                    fig_agg, ax = plt.subplots(figsize=(6, 4))
                    fig_agg.patch.set_facecolor("#111827")
                    ax.set_facecolor("#111827")
                    im = ax.imshow(n1_normalize(agg), cmap="inferno", vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(colors="#94a3b8")
                    ax.set_title("Aggregate Epistemic Uncertainty",
                                 color="#f1f5f9", fontsize=10, fontweight="bold")
                    ax.axis("off")
                    plt.tight_layout()
                    ca, cb = st.columns([2, 1])
                    with ca:
                        st.image(fig_to_pil(fig_agg), use_container_width=True)
                    with cb:
                        st.markdown(
                            "**Interpretation**\n"
                            "- 🔴 High entropy — model uncertain\n"
                            "- 🟡 Medium — N2 targets this\n"
                            "- 🔵 Low — confident, no refinement needed"
                        )
                    plt.close(fig_agg)
            else:
                st.info("Enable the agentic loop in the sidebar to see entropy maps.")

        with tab_n6:
            if agentic["ran"]:
                st.markdown("#### N6 Explainability Report")
                st.markdown(
                    f'<div class="explain-box">{agentic["explain"]}</div>',
                    unsafe_allow_html=True,
                )
                if agentic["all_diags"]:
                    st.markdown("#### Per-Nucleus Diagnostics")
                    import pandas as pd
                    rows = [
                        {
                            "#":             i + 1,
                            "N1 Reason":     d.get("n1_reason", "")[:60],
                            "Epi Ratio":     f'{d.get("epi_ratio", 0):.1%}',
                            "# Clusters":    d.get("n_clusters", 0),
                            "N2 Pts":        d.get("n2_pts",     0),
                            "N2 Boxes":      d.get("n2_boxes",   0),
                            "IoU Before":    f'{d.get("iou_before", 0):.3f}',
                            "IoU After":     f'{d.get("iou_after",  0):.3f}',
                            "N6 Action":     d.get("n6_action",  ""),
                        }
                        for i, d in enumerate(agentic["all_diags"])
                        if d.get("stage") == "N1_N2_N6"
                    ]
                    if rows:
                        st.dataframe(pd.DataFrame(rows),
                                     use_container_width=True, height=320)
            else:
                st.info("Enable the agentic loop in the sidebar to see N6 analysis.")

        with tab_log:
            if show_pipe_log:
                st.markdown("#### Execution Log")
                st.code("\n".join(f"[{i+1:02d}] {l}" for i, l in enumerate(log_lines)))

        st.success("Agentic segmentation pipeline completed successfully!")

    except Exception as exc:
        import traceback
        st.error(f"**Pipeline error:** {exc}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())
        prog_bar.empty()

# ════════════════════════════════════════════════════════════
# INFO PAGE — shown when no image has been uploaded
# ════════════════════════════════════════════════════════════
elif not uploaded:
    st.markdown("---")
    st.markdown("## 📖 About the Pipeline")
    ca, cb, cc = st.columns(3)
    _card_style = (
        "background:#111827;border-radius:14px;padding:1.5rem;margin-bottom:.5rem;"
    )
    with ca:
        st.markdown(
            f'<div style="{_card_style}border:1px solid rgba(245,158,11,.3);">'
            '<div style="font-size:1.4rem;margin-bottom:.4rem;">🔍</div>'
            '<div style="font-size:.9rem;font-weight:700;color:#fbbf24;">[N1] Uncertainty Decomposition</div>'
            '<div style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin-top:.5rem;">'
            "Uses the <strong>BALD criterion</strong> to separate:<br>"
            "&#8226; <strong>Epistemic</strong> — correctable with better prompts<br>"
            "&#8226; <strong>Aleatoric</strong> — inherent noise, cannot be fixed<br><br>"
            "<em>4 stochastic passes on the cached SAM embedding.</em>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown(
            f'<div style="{_card_style}border:1px solid rgba(6,182,212,.3);">'
            '<div style="font-size:1.4rem;margin-bottom:.4rem;">🤖</div>'
            '<div style="font-size:.9rem;font-weight:700;color:#06b6d4;">[N2] Auto Prompt Generation</div>'
            '<div style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin-top:.5rem;">'
            "<strong>Autonomous</strong> SAM prompt generation from N1 epistemic maps:<br>"
            "&#8226; Cluster centroids &rarr; <strong>point prompts</strong><br>"
            "&#8226; Cluster extents &rarr; <strong>bounding box prompts</strong><br>"
            "&#8226; Zero human input required"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with cc:
        st.markdown(
            f'<div style="{_card_style}border:1px solid rgba(16,185,129,.3);">'
            '<div style="font-size:1.4rem;margin-bottom:.4rem;">&#9881;&#65039;</div>'
            '<div style="font-size:.9rem;font-weight:700;color:#10b981;">[N6] Adaptive Controller</div>'
            '<div style="font-size:.78rem;color:#94a3b8;line-height:1.6;margin-top:.5rem;">'
            "Reads N1 + N2 signals to decide <strong>how much</strong> to refine:<br>"
            "&#8226; Tracks IoU delta across iterations<br>"
            "&#8226; Adapts N2 budget, box padding, percentile<br>"
            "&#8226; Generates <strong>natural language explanations</strong>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="{_card_style}border:1px solid rgba(99,102,241,.25);margin-top:.5rem;">'
        '<div style="font-size:.9rem;font-weight:700;color:#a5b4fc;margin-bottom:.75rem;">'
        "&#128260; Full Pipeline</div>"
        '<div style="display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;font-size:.78rem;color:#94a3b8;">'
        + " &rarr; ".join([
            '<span style="background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);'
            f'border-radius:6px;padding:3px 9px;">{s}</span>'
            for s in [
                "Input Image", "Macenko Norm", "SAM AMG", "NMS",
                "Morph Filter", "Watershed",
                '<span style="color:#fbbf24;">N1 BALD</span>',
                '<span style="color:#06b6d4;">N2 Prompts</span>',
                '<span style="color:#10b981;">N6 Control</span>',
                "Final Output",
            ]
        ])
        + "</div></div>",
        unsafe_allow_html=True,
    )
