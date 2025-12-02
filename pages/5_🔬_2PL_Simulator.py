import itertools
import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from PIL import Image
import streamlit as st

import dose_response as dp

# ===================== Auto-populate 2PL parameters from Upload Page =====================
params_df = st.session_state.get("model_input", None)
selected_model = st.session_state.get("selected_model", None)

# Default values for 2PL
b_default, c_default = 0.0, 50.0

if params_df is not None and selected_model == "2PL":
    st.info("Using parameter ranges from Upload Page for 2PL model")
    st.dataframe(params_df)

    numeric_cols = [c for c in params_df.columns if c != "sample"]

    if 'Average' in params_df['sample'].values:
        row = params_df[params_df['sample'] == 'Average']
        b = row[numeric_cols[0]].values[0] if len(numeric_cols) > 0 else b_default
        c = row[numeric_cols[1]].values[0] if len(numeric_cols) > 1 else c_default

    elif 'Min' in params_df['sample'].values and 'Max' in params_df['sample'].values:
        row_min = params_df[params_df['sample'] == 'Min']
        row_max = params_df[params_df['sample'] == 'Max']
        # Midpoint of min/max for numeric columns
        b = (row_min[numeric_cols[0]].values[0] + row_max[numeric_cols[0]].values[0]) / 2 if len(numeric_cols) > 0 else b_default
        c = (row_min[numeric_cols[1]].values[0] + row_max[numeric_cols[1]].values[0]) / 2 if len(numeric_cols) > 1 else c_default

    else:
        st.warning("Uploaded data format not recognized. Using default 2PL values.")
        b, c = b_default, c_default

else:
    # No uploaded data or model not selected -> use defaults
    b, c = b_default, c_default



# ---- Fix Plotly annotation xref/yref for subplot #1 vs #2.. ----
def axis_domain_ref(ax: str, i: int) -> str:
    return f"{ax} domain" if i == 1 else f"{ax}{i} domain"


# ===================== Page config (must be first Streamlit call) ===========

st.set_page_config(
    page_title="2PL Quantification Tool",
    layout="wide",
)

# ===================== Defaults (2PL-only, namespaced) ======================

DEFAULTS = {
    # Ranges (min/max) for 2PL: only slope (B) and EC50 (C)
    "b_min_2pl": -2.0,
    "b_max_2pl": -0.5,
    "c_min_2pl": 0.1,
    "c_max_2pl": 3.0,

    # Dilution controls – aligned with 4PL defaults
    "top_conc_2pl": 100.0,              # 10**2
    "even_dil_factor_2pl": 10 ** 0.5,   # √10 ≈ 3.162
    "dilution_str_2pl": "",             # optional custom 7 factors

    # App state
    "curves_2pl": [],
    "next_label_idx_2pl": 1,

    # RP input text (optional)
    "rps_str_2pl": "",
}

FLOAT_KEYS = [
    "b_min_2pl",
    "b_max_2pl",
    "c_min_2pl",
    "c_max_2pl",
    "top_conc_2pl",
    "even_dil_factor_2pl",
]

# ===================== Early session-state sync =============================

if "initialized_2pl" not in st.session_state:
    st.session_state["initialized_2pl"] = True
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# Ensure keys exist and have consistent float types
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k in FLOAT_KEYS:
    st.session_state[k] = float(st.session_state[k])

# Recommender interaction flags (already namespaced)
st.session_state.setdefault("apply_rec_factor_2pl", False)
st.session_state.setdefault("rec_factor_value_2pl", None)
st.session_state.setdefault("rec_df_2pl", None)

# If a recommended factor was chosen, apply it before widgets render
if st.session_state.get("apply_rec_factor_2pl", False):
    rec_val = st.session_state.get("rec_factor_value_2pl", None)
    if rec_val is not None:
        st.session_state["even_dil_factor_2pl"] = float(rec_val)
        st.session_state["dilution_str_2pl"] = ""
    st.session_state["apply_rec_factor_2pl"] = False

# ===================== Sidebar Logo (works on all pages) ====================

LOGO_PATH = "graphics/Logo.jpg"

try:
    logo = Image.open(LOGO_PATH)
    buffer = io.BytesIO()
    logo.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"]::before {{
                content: "";
                display: block;
                height: 200px;
                margin-bottom: 1rem;
                background-image: url("data:image/png;base64,{img_b64}");
                background-position: center;
                background-repeat: no-repeat;
                background-size: contain;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    # Fail silently if logo is missing in dev environments
    pass

# ===================== 2PL model (fixed lower=0, upper=1) ===================

A_FIXED = 1.0   # upper asymptote
D_FIXED = 0.0   # lower asymptote


def two_param_logistic_logx(x_log10, b, c, A=A_FIXED, D=D_FIXED):
    """
    Two-parameter logistic in log10(x) space with fixed lower (D) and upper (A):

        y = D + (A - D) / (1 + 10^(b * (x_log10 - log10(c))))

    - x_log10 : log10 concentration
    - b       : slope parameter (steepness) in log10 space
    - c       : EC50 on the original concentration scale

    For A=1, D=0 this is a normalized response in [0, 1].
    """
    return D + (A - D) / (1 + 10 ** (b * (x_log10 - np.log10(c))))


def compute_curve_2pl(b, c, x_log10):
    return two_param_logistic_logx(x_log10, b, c, A=A_FIXED, D=D_FIXED)


# ===================== Title & short description ============================

st.title("2PL Quantification Tool")

st.caption(
    "Parallel-curve **two-parameter logistic** model with fixed lower "
    "asymptote **D = 0** and upper asymptote **A = 1**. "
    "Only slope (**B**) and EC₅₀ (**C**) vary."
)

# ===================== Tunable region thresholds (anchor / linear) ==========
# For 2PL with A=1, D=0, the normalized response t is just y itself.
LINEAR_LOW, LINEAR_HIGH = 0.2, 0.8
LOWER_ANCHOR_MAX = 0.10
UPPER_ANCHOR_MIN = 0.90


# ===================== Helper utilities ====================================

def _rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()


def _parse_list(raw: str):
    if not raw or not raw.strip():
        return []
    parts = [p for chunk in raw.split(",") for p in chunk.strip().split()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return vals


def _parse_rps(raw: str):
    vals = _parse_list(raw)
    return sorted(set(vals))


def _list_to_str(xs):
    return " ".join(f"{v:.6g}" for v in xs) if xs else ""


def classify_region(y_val: float) -> str:
    if y_val <= LOWER_ANCHOR_MAX:
        return "Lower anchor"
    if y_val >= UPPER_ANCHOR_MIN:
        return "Upper anchor"
    if LINEAR_LOW <= y_val <= LINEAR_HIGH:
        return "Linear band"
    return "Transition"


# ===================== Data model for saved curves =========================

def curves_to_dataframe(curves):
    rows = []
    for cv in curves:
        grid = cv.get("grid", {})
        rows.append({
            "label": cv["label"],
            "b": cv["b"],
            "c": cv["c"],
            "rp": cv.get("rp"),
            "top_conc": grid.get("top_conc"),
            "even_factor": grid.get("even_factor"),
            "custom_factors": _list_to_str(grid.get("custom_factors", [])),
            "x_log10_points": _list_to_str(cv.get("x_log10_sparse", [])),
            "conc_points": _list_to_str(cv.get("conc_sparse", [])),
            "y_points": _list_to_str(cv.get("y_sparse", [])),
        })
    return pd.DataFrame(rows)


def _lock_curve(label, b, c, rp=None, grid=None):
    """
    Persist an 8-point curve using the current dilution grid.

    For RP ≠ 1, store the effective EC50 C_eff = C / RP
    (parallel-curve model: same shape, horizontal shift only).
    """
    c_eff = c / rp if (rp is not None and rp != 0) else c

    top_conc = float(grid.get("top_conc")) if grid else 100.0
    even_factor = float(grid.get("even_factor")) if grid else 10 ** 0.5
    custom_factors = list(grid.get("custom_factors", [])) if grid else []

    x_sparse_locked = dp.generate_log_conc(
        top_conc=top_conc,
        dil_factor=even_factor,
        n_points=8,
        dense=False,
        dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
    )

    conc_sparse = (10 ** x_sparse_locked).astype(float)
    y_sparse_locked = compute_curve_2pl(b, c_eff, x_sparse_locked)

    entry = {
        "label": label,
        "b": float(b),
        "c": float(c_eff),
        "rp": 1.0 if rp in (None, 0) else float(rp),
        "grid": {
            "top_conc": float(top_conc),
            "even_factor": float(even_factor),
            "custom_factors": list(custom_factors),
        },
        "x_log10_sparse": [float(v) for v in x_sparse_locked],
        "conc_sparse": [float(v) for v in conc_sparse],
        "y_sparse": [float(v) for v in y_sparse_locked],
    }
    st.session_state["curves_2pl"].append(entry)


# ===================== Recommender internals (2PL: B & C vary) =============

@st.cache_data(show_spinner=False)
def _evaluate_factor_2pl(
    factor,
    top_conc,
    b_min,
    b_max,
    c_min,
    c_max,
    rps_list,
):
    """
    Evaluate a single even dilution factor for 2PL with fixed A=1, D=0.

    - Builds 8-point dilution series
    - Evaluates 4 edge combinations: (b_min/max × c_min/max)
    - For each combo & RP, classifies points into lower/linear/upper bands
    - Returns worst-case coverage across all combos and RPs
    """
    x_sparse = dp.generate_log_conc(
        top_conc=top_conc,
        dil_factor=factor,
        n_points=8,
        dense=False,
    )

    combos = list(itertools.product([b_min, b_max], [c_min, c_max]))

    worst_case = {"lower_min": 8, "linear_min": 8, "upper_min": 8}

    for bb, cc in combos:
        rows = []
        for rp in (rps_list or [1.0]):
            c_eff = cc / rp
            y = compute_curve_2pl(bb, c_eff, x_sparse)
            t = y  # normalized response

            lower = int(np.sum(t <= LOWER_ANCHOR_MAX))
            linear = int(np.sum((t >= LINEAR_LOW) & (t <= LINEAR_HIGH)))
            upper = int(np.sum(t >= UPPER_ANCHOR_MIN))

            rows.append(
                {"rp": float(rp), "lower": lower, "linear": linear, "upper": upper}
            )

        lower_min_combo = min(r["lower"] for r in rows)
        linear_min_combo = min(r["linear"] for r in rows)
        upper_min_combo = min(r["upper"] for r in rows)

        worst_case["lower_min"] = min(worst_case["lower_min"], lower_min_combo)
        worst_case["linear_min"] = min(worst_case["linear_min"], linear_min_combo)
        worst_case["upper_min"] = min(worst_case["upper_min"], upper_min_combo)

    min_ok = (
        worst_case["linear_min"] >= 2
        and worst_case["lower_min"] >= 1
        and worst_case["upper_min"] >= 1
    )

    pref_ok = (
        worst_case["linear_min"] >= 3
        and worst_case["lower_min"] >= 2
        and worst_case["upper_min"] >= 2
    )

    score = (
        min(worst_case["linear_min"], 3)
        + min(worst_case["lower_min"], 2)
        + min(worst_case["upper_min"], 2)
    )

    return {
        "factor": float(factor),
        "worst_linear": int(worst_case["linear_min"]),
        "worst_lower": int(worst_case["lower_min"]),
        "worst_upper": int(worst_case["upper_min"]),
        "meets_min": bool(min_ok),
        "meets_preferred": bool(pref_ok),
        "score": int(score),
    }


# ========================================================================== #
# ===================== Layout: main panels =================================
# ========================================================================== #

st.markdown("---")

left_panel, graph_col = st.columns([1.15, 1.85], gap="large")

# --------------------- Left: Dilution series controls ----------------------

with left_panel:
    st.subheader("Dilution series")
    with st.expander("What is a dilution series?", expanded=False):
        st.markdown(
            "The **dilution factor** is how much each step is diluted from the previous one, "
            "creating a series of decreasing doses for the curve. "
            "For example, a factor of **3** means each well is **1/3** of the previous well. "
            "Providing exactly **7** custom factors overrides the even factor."
        )

    st.number_input(
        "Top concentration",
        min_value=1e-12,
        max_value=1e12,
        value=float(st.session_state["top_conc_2pl"]),
        step=1.0,
        format="%.6g",
        key="top_conc_2pl",
        help=(
            "Highest concentration in the dilution series (original scale). "
            "All other wells are generated by repeated division by the "
            "dilution factor(s)."
        ),
    )

    st.number_input(
        "Even dilution factor (applied 7×)",
        min_value=1.0001,
        max_value=1e9,
        value=float(st.session_state["even_dil_factor_2pl"]),
        step=0.01,
        format="%.6g",
        key="even_dil_factor_2pl",
        help=(
            "Factor between adjacent wells (e.g., 2 for 2-fold, 3 for 3-fold). "
            "√10 ≈ 3.162 is common for dose–response assays."
        ),
    )

    st.text_input(
        "Custom 7 dilution factors "
        "(override even factor if exactly 7 numbers)",
        key="dilution_str_2pl",
        value=st.session_state["dilution_str_2pl"],
        placeholder="e.g., 3.162 3.162 3.162 3.162 3.162 3.162 3.162",
        help=(
            "Provide 7 step-wise multipliers (high → low). "
            "If blank or not exactly 7 values, the even factor is used."
        ),
    )

    custom_factors = _parse_list(st.session_state["dilution_str_2pl"])
    if len(custom_factors) == 0:
        st.caption("Using even dilution factor.")
    elif len(custom_factors) == 7:
        st.caption(f"Using custom factors: {custom_factors}")
    else:
        st.warning(
            f"Provide exactly 7 factors (got {len(custom_factors)}). "
            "Falling back to even factor."
        )
        custom_factors = []

# --------------------- Right: Graph header & placeholder -------------------

with graph_col:
    st.subheader("Dose–Response Curves (2PL)")
    plot_placeholder = st.empty()

# --------------------- Parameter ranges (2PL) ------------------------------

with left_panel:
    st.markdown("### Parameter ranges (2PL)")
    st.caption(
        "Fixed lower asymptote **D = 0** and upper asymptote **A = 1**; "
        "only slope (**B**) and EC₅₀ (**C**) vary."
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.number_input(
            "b_min (slope, B)",
            min_value=-10.0,
            max_value=10.0,
            value=float(st.session_state["b_min_2pl"]),
            step=0.01,
            key="b_min_2pl",
        )
        st.number_input(
            "c_min (EC₅₀, C)",
            min_value=1e-6,
            max_value=1e6,
            value=float(st.session_state["c_min_2pl"]),
            step=0.01,
            format="%.6g",
            key="c_min_2pl",
        )
    with r1c2:
        st.number_input(
            "b_max (slope, B)",
            min_value=-10.0,
            max_value=10.0,
            value=float(st.session_state["b_max_2pl"]),
            step=0.01,
            key="b_max_2pl",
        )
        st.number_input(
            "c_max (EC₅₀, C)",
            min_value=1e-6,
            max_value=1e6,
            value=float(st.session_state["c_max_2pl"]),
            step=0.01,
            format="%.6g",
            key="c_max_2pl",
        )

    st.text_input(
        "Relative potencies (RP; comma/space separated)",
        key="rps_str_2pl",
        value=st.session_state["rps_str_2pl"],
        placeholder="e.g., 0.4, 0.5  1, 1.5, 2",
        help=(
            "Relative potency shifts the curve horizontally via C_eff = C / RP. "
            "Curves are assumed parallel; only EC₅₀ changes."
        ),
    )

    user_rps = _parse_rps(st.session_state["rps_str_2pl"])
    if not user_rps:
        rps = [0.4, 1.0, 1.6]
        st.caption(
            "Using default RP values: **0.4** (40%), **1.0** (reference), "
            "**1.6** (160%)."
        )
    else:
        rps = user_rps
        st.caption(f"Parsed RP values: {user_rps}")

# --------------------- Base parameters for main graph ----------------------

b_min = float(st.session_state["b_min_2pl"])
b_max = float(st.session_state["b_max_2pl"])
c_min = float(st.session_state["c_min_2pl"])
c_max = float(st.session_state["c_max_2pl"])

b = (b_min + b_max) / 2.0
c = (c_min + c_max) / 2.0

top_conc = float(st.session_state["top_conc_2pl"])
even_factor = float(st.session_state["even_dil_factor_2pl"])

# Live grids used for plotting and preview
x_sparse_live = dp.generate_log_conc(
    top_conc=top_conc,
    dil_factor=even_factor,
    n_points=8,
    dense=False,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)
x_dense_live = dp.generate_log_conc(
    top_conc=top_conc,
    dil_factor=even_factor,
    n_points=8,
    dense=True,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)

# Color map for RPs
palette = qualitative.Plotly
rp_sorted = sorted(set(rps))
rp_color_map = {}
color_cursor = 1
if 1.0 in rp_sorted:
    rp_color_map[1.0] = palette[0]
for rp_val in rp_sorted:
    if rp_val == 1.0:
        continue
    rp_color_map[rp_val] = palette[color_cursor % len(palette)]
    color_cursor += 1

# ===================== Main figure =========================================

fig = go.Figure()

# --- Reference curve (RP = 1.0) -------------------------------------------

base_color = rp_color_map.get(1.0, palette[0])
y_dense_ref = compute_curve_2pl(b, c, x_dense_live)
fig.add_scatter(
    x=x_dense_live,
    y=y_dense_ref,
    mode="lines",
    name="Reference (RP=1.0)",
    line=dict(width=2, color=base_color),
    hovertemplate=(
        "log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Reference</extra>"
    ),
)
y_sparse_ref = compute_curve_2pl(b, c, x_sparse_live)
fig.add_scatter(
    x=x_sparse_live,
    y=y_sparse_ref,
    mode="markers",
    marker=dict(size=8, symbol="circle", color=base_color),
    showlegend=False,
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
)

# EC50 for reference curve
ec50_log10 = np.log10(c)
fig.add_vline(
    x=ec50_log10,
    line=dict(color=base_color, width=1.5, dash="dot"),
    annotation_text="EC₅₀ (reference)",
    annotation_position="top",
    annotation_font=dict(size=11),
)

# --- RP curves (assumed parallel) ------------------------------------------

for rp_val in rp_sorted:
    if rp_val == 1.0:
        continue
    color = rp_color_map[rp_val]
    y_dense = compute_curve_2pl(b, c / rp_val, x_dense_live)
    fig.add_scatter(
        x=x_dense_live,
        y=y_dense,
        mode="lines",
        name=f"Sample (RP={rp_val:g})",
        line=dict(width=2, color=color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )
    y_sparse = compute_curve_2pl(b, c / rp_val, x_sparse_live)
    fig.add_scatter(
        x=x_sparse_live,
        y=y_sparse,
        mode="markers",
        marker=dict(size=8, symbol="circle", color=color),
        showlegend=False,
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

# --- Locked curves ---------------------------------------------------------

locked_start_index = 6
for idx, cv in enumerate(st.session_state["curves_2pl"]):
    grid = cv.get("grid", {}) or {}
    tc = grid.get("top_conc", top_conc)
    ef = grid.get("even_factor", even_factor)
    cf = grid.get("custom_factors", [])
    lock_color = palette[(locked_start_index + idx) % len(palette)]

    x_dense_locked = dp.generate_log_conc(
        top_conc=float(tc),
        dil_factor=float(ef),
        n_points=8,
        dense=True,
        dilution_factors=(cf if isinstance(cf, list) and len(cf) == 7 else None),
    )
    y_locked_line = compute_curve_2pl(cv["b"], cv["c"], x_dense_locked)
    fig.add_scatter(
        x=x_dense_locked,
        y=y_locked_line,
        mode="lines",
        name=f'{cv["label"]} (locked)',
        line=dict(dash="dash", color=lock_color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

    xs = cv.get("x_log10_sparse", [])
    ys = cv.get("y_sparse", [])
    if xs and ys and len(xs) == 8 and len(ys) == 8:
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="markers",
            showlegend=False,
            marker=dict(size=7, color=lock_color, symbol="circle-open"),
            hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
        )

fig.update_layout(
    title="Dose–Response Curves (2PL, normalized A=1, D=0)",
    xaxis_title="log₁₀(concentration)",
    yaxis_title="Response (fraction of max)",
    legend_title=None,
    margin=dict(l=10, r=10, t=50, b=10),
)
fig.update_yaxes(range=[-0.05, 1.05])

# Render main graph & controls
with graph_col:
    plot_placeholder.plotly_chart(
        fig,
        config={"responsive": True, "displayModeBar": True},
        use_container_width=True,
    )

    # Curve labeling & buttons
    default_label = f"Curve {st.session_state['next_label_idx_2pl']}"
    label = st.text_input(
        "Label for base curve (reference 2PL)",
        value=default_label,
        help="Base curve name; RP variants are saved as parallel curves.",
        key="label_input_2pl",
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Add curve", type="primary", key="btn_add_curve_2pl"):
            base_rps = rp_sorted[:] if rp_sorted else [1.0]
            if 1.0 not in base_rps:
                base_rps = [1.0] + base_rps

            grid = {
                "top_conc": top_conc,
                "even_factor": even_factor,
                "custom_factors": list(custom_factors)
                if len(custom_factors) == 7
                else [],
            }

            for rp_val in base_rps:
                lbl = label if rp_val == 1.0 else f"{label} (RP={rp_val:g})"
                _lock_curve(label=lbl, b=b, c=c, rp=rp_val, grid=grid)

            st.session_state["next_label_idx_2pl"] += 1
            st.success(f"Saved '{label}' with {len(base_rps)} curve(s).")
            _rerun()

    with col_btn2:
        if st.button("Clear all saved curves", key="btn_clear_curves_2pl"):
            st.session_state["curves_2pl"] = []
            st.session_state["next_label_idx_2pl"] = 1
            st.info("Cleared all saved curves.")
            _rerun()

    st.markdown("---")
    
# ========================================================================== #

# ===================== Edge-case subplots (B,C extremes) ===================

st.markdown("### Edge cases (2PL): all min/max combinations of B, C")
with st.expander("What is Edge cases?", expanded=False):
    st.markdown(
        "These plots show all the edge-case combinations of parameters **A, B, C, D, and/or E**. "
        "Each panel represents a minimum/maximum setting of those parameters so users can see how "
        "extreme values change the shape and behavior of the response curve."
    )

x_sparse_edge = dp.generate_log_conc(
    top_conc=top_conc,
    dil_factor=even_factor,
    n_points=8,
    dense=False,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)
x_dense_edge = dp.generate_log_conc(
    top_conc=top_conc,
    dil_factor=even_factor,
    n_points=8,
    dense=True,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)

b_min = float(st.session_state["b_min_2pl"])
b_max = float(st.session_state["b_max_2pl"])
c_min = float(st.session_state["c_min_2pl"])
c_max = float(st.session_state["c_max_2pl"])

choices = list(itertools.product([b_min, b_max], [c_min, c_max]))

edge_fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.06,
    vertical_spacing=0.10,
    subplot_titles=[f"B={bb:.3g}, C={cc:.3g}" for (bb, cc) in choices],
)

for idx, (bb, cc) in enumerate(choices, start=1):
    r = ((idx - 1) // 2) + 1
    k = ((idx - 1) % 2) + 1
    for rp_val in rp_sorted if rp_sorted else [1.0]:
        color = rp_color_map.get(rp_val, palette[0] if rp_val == 1.0 else palette[1])
        edge_fig.add_scatter(
            x=x_dense_edge,
            y=compute_curve_2pl(bb, cc / rp_val, x_dense_edge),
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False,
            row=r,
            col=k,
        )
        edge_fig.add_scatter(
            x=x_sparse_edge,
            y=compute_curve_2pl(bb, cc / rp_val, x_sparse_edge),
            mode="markers",
            marker=dict(size=5, color=color),
            showlegend=False,
            row=r,
            col=k,
        )

for r in range(1, 3):
    for c_ in range(1, 3):
        i = (r - 1) * 2 + c_
        edge_fig.layout[f"xaxis{i}"].title.text = ""
        edge_fig.layout[f"yaxis{i}"].title.text = ""
        edge_fig.update_xaxes(showticklabels=(r == 2), row=r, col=c_)
        edge_fig.update_yaxes(showticklabels=(c_ == 1), row=r, col=c_)

edge_fig.update_layout(margin=dict(l=80, r=20, t=50, b=80), height=600)
edge_fig.add_annotation(
    x=0.5,
    y=-0.10,
    xref="paper",
    yref="paper",
    text="log₁₀(concentration)",
    showarrow=False,
    font=dict(size=14),
)
edge_fig.add_annotation(
    x=0.0,
    y=0.5,
    xref="paper",
    yref="paper",
    text="response (fraction of max)",
    showarrow=False,
    textangle=-90,
    xanchor="right",
    yanchor="middle",
    xshift=-40,
    font=dict(size=14),
)

# --------- Add tooltips (question mark in top-right corner of each subplot) -------
import html  # for safe escaping of tooltip text

# Row/column-specific explanations for each subplot (2x2 grid)
# 2PL MODEL
tooltip_texts_2pl = [
    # Row 1 (col1, col2)
    [
        "Steep & potent: sharp rise at low concentration",
        "Steep but low potency: sharp rise only at high concentration",
    ],
    # Row 2 (col1, col2)
    [
        "Shallow but potent: gradual increase starting early",
        "Shallow & weak: slow rise and needs high concentration",
    ],
]




# ---------- Info icons for 2PL (same placement as 4PL/5PL: right edge at y=0) ----------
N_COLS_2PL = 2

# 用当前 2PL 边缘图的 x 轴范围
x_min_2pl = float(np.min(x_dense_edge))
x_max_2pl = float(np.max(x_dense_edge))
x_pad_2pl = 0.02 * (x_max_2pl - x_min_2pl)

# hover 白底更清晰
edge_fig.update_layout(hoverlabel=dict(bgcolor="white"))

# tooltip 文本拍平成一维（行优先顺序：row1 col1, row1 col2, row2 col1, row2 col2）
tooltips_flat_2pl = [t for row in tooltip_texts_2pl for t in row]

for idx in range(1, 4 + 1):  # 2x2 = 4 subplots
    row0 = (idx - 1) // N_COLS_2PL   # 0..1
    col0 = (idx - 1) %  N_COLS_2PL   # 0..1
    r, c = row0 + 1, col0 + 1        # plotly 1-based

    # 放在 y=0 轴线上，x 轴最右侧留点边距
    y_icon = 0.0
    x_icon = x_max_2pl - x_pad_2pl
    tip = tooltips_flat_2pl[idx - 1]

    # 圆圈（处理 hover）
    edge_fig.add_scatter(
        x=[x_icon], y=[y_icon],
        mode="markers",
        marker=dict(
            symbol="circle",
            size=20,
            color="white",
            line=dict(color="#475569", width=1.5)
        ),
        hovertext=[tip],
        hoverinfo="text",
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
        row=r, col=c,
    )

    # 圆圈里的 “i”
    edge_fig.add_scatter(
        x=[x_icon], y=[y_icon],
        mode="text",
        text=["i"],
        textfont=dict(size=13, color="#475569"),
        hoverinfo="skip",
        showlegend=False,
        row=r, col=c,
    )
# ---------- end info icons for 2PL ----------





st.plotly_chart(edge_fig, config={"responsive": True}, use_container_width=True)

# ========================================================================== #
st.markdown("---")
# ========================================================================== #

# ===================== Saved curves + Dilution preview =====================

col_saved, col_preview = st.columns([1.15, 1.85], gap="large")

with col_saved:
    st.subheader("Saved curves (2PL)")

    df_saved = curves_to_dataframe(st.session_state["curves_2pl"])
    if not df_saved.empty:
        st.dataframe(df_saved, use_container_width=True, height=320)
        st.download_button(
            "Export Saved Curves CSV",
            data=df_saved.to_csv(index=False).encode("utf-8"),
            file_name="dose_response_curves_2pl.csv",
            mime="text/csv",
            key="btn_export_saved_csv_2pl",
        )
    else:
        st.info("No saved curves yet. Use **Add curve** above to lock a set.")

with col_preview:
    st.subheader("Dilution preview (current settings, 2PL)")

    conc_sparse_live = (10 ** x_sparse_live).astype(float)
    y_sparse_live = compute_curve_2pl(b, c, x_sparse_live)
    regions = [classify_region(y) for y in y_sparse_live]

    df_preview = pd.DataFrame(
        {
            "Well": np.arange(1, len(x_sparse_live) + 1, dtype=int),
            "log10(conc)": x_sparse_live,
            "conc": conc_sparse_live,
            "response (reference)": y_sparse_live,
            "region (ref)": regions,
        }
    )

    dfp = df_preview.copy()
    dfp["log10(conc)"] = dfp["log10(conc)"].map(lambda v: f"{v:.6f}")
    dfp["conc"] = dfp["conc"].map(lambda v: f"{v:.6g}")
    dfp["response (reference)"] = dfp["response (reference)"].map(
        lambda v: f"{v:.4f}"
    )

    st.dataframe(dfp, use_container_width=True, height=320)

    st.download_button(
        "Export Dilution Preview CSV",
        data=df_preview.to_csv(index=False).encode("utf-8"),
        file_name="current_dilution_preview_2pl.csv",
        mime="text/csv",
        key="btn_export_preview_csv_2pl",
    )

    if len(custom_factors) == 7:
        st.caption(f"Using custom factors: {custom_factors}")
    else:
        st.caption(f"Using even dilution factor: {even_factor:.6g}")

st.markdown("---")
st.markdown("**References**")
st.markdown(
    """
- Duda, J., Hengstler, J. G., & Rahnenführer, J. (2022). *td2pLL: An intuitive time-dose-response model for cytotoxicity data with varying exposure durations.* Computational Toxicology, 23, 100234. https://doi.org/10.1016/j.comtox.2022.100234  
- Kuster Lab. (2024, Jul 17). *Statistical analysis of dose-response curves.* Wiley Analytical Science. https://analyticalscience.wiley.com/content/article-do/statistical-analysis-dose-response-curves  
- O’Brien, T. E., & Silcox, J. (2021). *Efficient experimental design for dose response modelling.* Journal of Applied Statistics, 48(13–15), 2864–2888. https://doi.org/10.1080/02664763.2021.1880556
    """
)
