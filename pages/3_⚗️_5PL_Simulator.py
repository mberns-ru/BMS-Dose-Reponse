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
from utils.param_loader import load_param_bounds


# ===================== Helpers to read bounds from Upload page =====================

def _normalize_bounds_5pl(bounds: dict | None):
    """
    Normalize a bounds dict into:
        {"a": (lo, hi), "b": ..., "c": ..., "d": ..., "e": ...?}

    Requires at least A–D; E is optional.
    """
    if not isinstance(bounds, dict):
        return None

    out: dict[str, tuple[float, float]] = {}
    for k in ["a", "b", "c", "d", "e"]:
        pair = bounds.get(k) or bounds.get(k.upper())
        if not pair:
            continue
        if len(pair) != 2:
            return None
        lo, hi = map(float, pair)
        if lo > hi:
            lo, hi = hi, lo
        out[k] = (lo, hi)

    # Require at least A–D
    if not {"a", "b", "c", "d"}.issubset(out.keys()):
        return None
    return out


def _bounds_from_params_df_5pl(df: pd.DataFrame | None):
    """
    Infer bounds from st.session_state['model_input'], which comes from the Upload page.

    Uses Min/Max rows if present; otherwise min/max of numeric columns.
    A–D are required; E is optional (if a 5th numeric column exists).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    try:
        if "sample" in df.columns and {"Min", "Max"}.issubset(set(df["sample"])):
            num_cols = [c for c in df.columns if c != "sample"]
            if len(num_cols) < 4:
                return None

            row_min = df.loc[df["sample"] == "Min", num_cols].iloc[0]
            row_max = df.loc[df["sample"] == "Max", num_cols].iloc[0]

            out = {
                "a": (float(row_min.iloc[0]), float(row_max.iloc[0])),
                "b": (float(row_min.iloc[1]), float(row_max.iloc[1])),
                "c": (float(row_min.iloc[2]), float(row_max.iloc[2])),
                "d": (float(row_min.iloc[3]), float(row_max.iloc[3])),
            }
            if len(num_cols) >= 5:
                out["e"] = (float(row_min.iloc[4]), float(row_max.iloc[4]))
            return out

        # Fallback: min/max of first 4–5 numeric columns
        num_cols = [
            c
            for c in df.columns
            if c != "sample" and np.issubdtype(df[c].dtype, np.number)
        ]
        if len(num_cols) < 4:
            return None

        mins = [float(df[c].min()) for c in num_cols[:5]]
        maxs = [float(df[c].max()) for c in num_cols[:5]]

        out = {
            "a": (mins[0], maxs[0]),
            "b": (mins[1], maxs[1]),
            "c": (mins[2], maxs[2]),
            "d": (mins[3], maxs[3]),
        }
        if len(num_cols) >= 5:
            out["e"] = (mins[4], maxs[4])
        return out
    except Exception:
        return None


def _defaults_5pl():
    return {
        "a": (0.80, 1.20),
        "b": (-2.00, -0.50),
        "c": (0.10, 3.00),
        "d": (0.00, 0.20),
        "e": (0.80, 1.20),
    }


# --- Read ranges + detect whether they came from the Upload page (like 4PL) ---

_raw_bounds = load_param_bounds()
_model_input = st.session_state.get("model_input")

# Treat any usable bounds/model_input as "from upload"
_FROM_UPLOAD_5pl = (_raw_bounds is not None) or (
    _model_input is not None and not getattr(_model_input, "empty", False)
)

_bounds_5pl = _normalize_bounds_5pl(_raw_bounds)
if _bounds_5pl is None:
    _bounds_5pl = _normalize_bounds_5pl(_bounds_from_params_df_5pl(_model_input))
if _bounds_5pl is None:
    _bounds_5pl = _defaults_5pl()

# Allow re-prefill when bounds change
if st.session_state.get("_last_bounds_5pl") != _bounds_5pl:
    st.session_state.pop("prefilled_5pl", None)
    st.session_state["_last_bounds_5pl"] = _bounds_5pl

# Prefill only once so user edits don’t get stomped
if "prefilled_5pl" not in st.session_state:
    defaults_all = _defaults_5pl()
    mapping = {
        "a": ("a_min_5pl", "a_max_5pl"),
        "b": ("b_min_5pl", "b_max_5pl"),
        "c": ("c_min_5pl", "c_max_5pl"),
        "d": ("d_min_5pl", "d_max_5pl"),
        "e": ("e_min_5pl", "e_max_5pl"),
    }
    for k, (k_min, k_max) in mapping.items():
        lo, hi = _bounds_5pl.get(k, defaults_all[k])
        st.session_state[k_min] = lo
        st.session_state[k_max] = hi

    st.session_state["prefilled_5pl"] = True

    # Build the "Loaded ranges → ..." message, but only if Upload was used
    if _FROM_UPLOAD_5pl:
        st.session_state["_5pl_loaded_ranges_msg"] = (
            f"Loaded ranges → A({st.session_state['a_min_5pl']},"
            f"{st.session_state['a_max_5pl']}), "
            f"B({st.session_state['b_min_5pl']},"
            f"{st.session_state['b_max_5pl']}), "
            f"C({st.session_state['c_min_5pl']},"
            f"{st.session_state['c_max_5pl']}), "
            f"D({st.session_state['d_min_5pl']},"
            f"{st.session_state['d_max_5pl']}), "
            f"E({st.session_state['e_min_5pl']},"
            f"{st.session_state['e_max_5pl']})"
        )
    else:
        st.session_state.pop("_5pl_loaded_ranges_msg", None)


# ===================== Page config =====================

st.set_page_config(
    page_title="5-Parameter Logistic (5PL) Simulator",
    layout="wide",
)

# ===================== Sidebar Logo =====================

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
    pass

# ===================== Title & help text =====================

st.title("5-Parameter Logistic (5PL) Simulator")

with st.expander("How to use this tool", expanded=False):

    # --- Parameters (left) + equation (right), matching 4PL layout ---
    params_col, eqn_col = st.columns([1.7, 1.3])

    with params_col:
        st.markdown(
            r"""
### **Parameters**

- **A** — *Lower asymptote*  
  The minimum response value the curve approaches at very low concentrations.

- **B** — *Slope / Hill slope*  
  Controls the steepness of the curve; higher magnitude = sharper transition.

- **C** — *EC₅₀ (Midpoint)*  
  The concentration at which the response is halfway between A and D.

- **D** — *Upper asymptote*  
  The maximum response the curve approaches at high concentrations.

- **E** — *Asymmetry parameter*  
  Controls how symmetric vs. skewed the curve is around the midpoint.
            """
        )

    with eqn_col:
        st.markdown("### **Equation**")
        st.latex(
            r"""
            y = D + \frac{A - D}
                     {\left(1 + 10^{\,B\,(\log_{10}(x) - \log_{10}(C))}\right)^{E}}
            """
        )

    # --- Edge-case envelope explanation ---
    st.markdown(
        r"""
### Edge cases

To show how sensitive the dose–response curve is to parameter uncertainty, the tool plots all  
**32 combinations** of minimum and maximum values of the five parameters:

$$
(A_{\min/\max},\ B_{\min/\max},\ C_{\min/\max},\ D_{\min/\max},\ E_{\min/\max})
$$

Each line represents one extreme configuration.  
Together, these curves form an **edge-case envelope** showing:

- The steepest and shallowest possible slopes  
- The earliest and latest possible transitions  
- The lowest and highest asymptotes  
- The most symmetric and most asymmetric shapes  

This helps visualize the full range of behaviors the assay could produce under uncertainty.
        """
    )


# ===================== Session State / Defaults =====================

DEFAULTS = {
    "a_min_5pl": 0.8,
    "a_max_5pl": 1.2,
    "b_min_5pl": -2.0,
    "b_max_5pl": -0.5,
    "c_min_5pl": 0.1,
    "c_max_5pl": 3.0,
    "d_min_5pl": 0.0,
    "d_max_5pl": 0.2,
    "e_min_5pl": 0.8,
    "e_max_5pl": 1.2,
    "top_conc_5pl": 10**2,
    "even_dil_factor_5pl": 10**0.5,
    "dilution_str_5pl": "",
    "curves_5pl": [],
    "next_label_idx_5pl": 1,
    "rps_str_5pl": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault("apply_rec_factor_5pl", False)
st.session_state.setdefault("rec_factor_value_5pl", None)
st.session_state.setdefault("rec_df_5pl", None)


def _rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()


# Apply any pending recommended factor before building widgets
if st.session_state.get("apply_rec_factor_5pl", False):
    rec_val = st.session_state.get("rec_factor_value_5pl", None)
    if rec_val is not None:
        st.session_state["even_dil_factor_5pl"] = float(rec_val)
        st.session_state["dilution_str_5pl"] = ""
    st.session_state["apply_rec_factor_5pl"] = False


# ===================== Small helpers =====================

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


# ===================== 5PL model & boundaries =====================

def _five_pl_logistic(x_log10, a, b, c, d, e):
    c = float(c)
    if c <= 0:
        c = 1e-12
    return d + (a - d) / (1.0 + np.power(10.0, b * (x_log10 - np.log10(c)))) ** e


def _region_bounds_5pl(a, b, c, d, e, frac_low=0.1, frac_high=0.9):
    """
    Solve for log10(x) where the 5PL reaches given fractions of its dynamic range.

    frac_low / frac_high are in [0,1] and refer to
        y = d + frac * (a - d).
    """
    if e == 0:
        return -np.inf, np.inf

    c = float(c)
    if c <= 0:
        c = 1e-12

    def solve(frac):
        if frac <= 0.0 or frac >= 1.0:
            return np.nan
        t = frac ** (-1.0 / e) - 1.0
        if t <= 0 or np.isnan(t):
            return np.nan
        try:
            return np.log10(c) + (np.log10(t) / b)
        except Exception:
            return np.nan

    x_low = solve(frac_low)
    x_high = solve(frac_high)

    if np.isnan(x_low):
        x_low = -np.inf
    if np.isnan(x_high):
        x_high = np.inf

    if x_low > x_high:
        x_low, x_high = x_high, x_low

    return x_low, x_high


# ===================== Curve saving =====================

def curves_5pl_to_dataframe(curves_5pl):
    rows = []
    for cv in curves_5pl:
        grid = cv.get("grid", {})
        rows.append(
            {
                "label": cv["label"],
                "a": cv["a"],
                "b": cv["b"],
                "c": cv["c"],
                "d": cv["d"],
                "e": cv["e"],
                "rp": cv.get("rp"),
                "top_conc_5pl": grid.get("top_conc_5pl"),
                "even_factor": grid.get("even_factor"),
                "custom_factors": _list_to_str(grid.get("custom_factors", [])),
                "x_log10_points": _list_to_str(cv.get("x_log10_sparse", [])),
                "conc_points": _list_to_str(cv.get("conc_sparse", [])),
                "y_points": _list_to_str(cv.get("y_sparse", [])),
            }
        )
    return pd.DataFrame(rows)


def _lock_curve(label, a, b, c, d, e, rp=None, grid=None):
    c_eff = c / rp if (rp is not None and rp != 0) else c

    top_conc_5pl = float(grid.get("top_conc_5pl")) if grid else 10**2
    even_factor = float(grid.get("even_factor")) if grid else 10**0.5
    custom_factors = list(grid.get("custom_factors", [])) if grid else []

    x_sparse_locked = dp.generate_log_conc(
        top_conc=top_conc_5pl,
        dil_factor=even_factor,
        n_points=8,
        dense=False,
        dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
    )

    conc_sparse = (10**x_sparse_locked).astype(float)
    y_sparse_locked = _five_pl_logistic(x_sparse_locked, a, b, c_eff, d, e)

    entry = {
        "label": label,
        "a": float(a),
        "b": float(b),
        "c": float(c_eff),
        "d": float(d),
        "e": float(e),
        "rp": 1.0 if rp in (None, 0) else float(rp),
        "grid": {
            "top_conc_5pl": float(top_conc_5pl),
            "even_factor": float(even_factor),
            "custom_factors": list(custom_factors),
        },
        "x_log10_sparse": [float(v) for v in x_sparse_locked],
        "conc_sparse": [float(v) for v in conc_sparse],
        "y_sparse": [float(v) for v in y_sparse_locked],
    }
    st.session_state["curves_5pl"].append(entry)


# ======= Recommender internals (5PL, same logic as 4PL) =======

def _score_pattern(n_bottom, n_linear, n_top):
    """
    Asymmetric score:

    - Target anchors: bottom = 2, top = 2
    - Target linear: at least 3 (extra linear points are OK)
    """
    target_bottom = 2
    target_linear_min = 3
    target_top = 2

    anchor_weight = 4.0

    bottom_dev = n_bottom - target_bottom
    top_dev = n_top - target_top

    penalty_bottom = anchor_weight * (bottom_dev**2)
    penalty_top = anchor_weight * (top_dev**2)

    if n_linear >= target_linear_min:
        penalty_linear = 0.0
    else:
        shortfall = target_linear_min - n_linear
        penalty_linear = float(shortfall**2)

    return penalty_bottom + penalty_linear + penalty_top


@st.cache_data(show_spinner=False)
def _evaluate_factor_5pl(
    factor,
    top,
    A,
    B,
    C,
    D,
    e_val,
    rps_list,
    combos16=None,
):
    x_points = dp.generate_log_conc(
        top_conc=top, dil_factor=factor, n_points=8, dense=False
    )

    def eval_one(a_, b_, c_, d_):
        rows = []
        for rp in (rps_list or [1.0]):
            c_eff = c_ / rp
            y = _five_pl_logistic(x_points, a_, b_, c_eff, d_, e_val)

            x_low, x_high = _region_bounds_5pl(a_, b_, c_eff, d_, e_val)

            n_bottom = int(np.sum(x_points < x_low))
            n_top = int(np.sum(x_points > x_high))
            n_linear = len(x_points) - n_bottom - n_top

            J = _score_pattern(n_bottom, n_linear, n_top)

            rows.append(
                {
                    "rp": float(rp),
                    "bottom": n_bottom,
                    "linear": n_linear,
                    "top": n_top,
                    "score": J,
                }
            )

        worst = max(rows, key=lambda r: r["score"])
        return rows, worst

    per_combo = []
    worst_overall = None

    if combos16 is None:
        rows, worst = eval_one(A, B, C, D)
        per_combo.append({"combo": "avg", "rows": rows, "worst": worst})
        worst_overall = worst
    else:
        for i, (aa, bb, cc, dd) in enumerate(combos16):
            rows, worst = eval_one(aa, bb, cc, dd)
            per_combo.append({"combo": f"edge{i+1}", "rows": rows, "worst": worst})
            if worst_overall is None or worst["score"] > worst_overall["score"]:
                worst_overall = worst

    wb = int(worst_overall["bottom"])
    wl = int(worst_overall["linear"])
    wt = int(worst_overall["top"])
    score = float(worst_overall["score"])

    meets_min = wl >= 2 and wb >= 1 and wb <= 2 and wt >= 1
    meets_preferred = wl >= 3 and wb == 2 and wt == 2

    return {
        "factor": float(factor),
        "worst_lower": wb,
        "worst_linear": wl,
        "worst_upper": wt,
        "score": score,
        "meets_min": bool(meets_min),
        "meets_preferred": bool(meets_preferred),
        "detail": per_combo,
    }


@st.cache_data(show_spinner=False)
def suggest_factor_from_ranges(
    top_conc_5pl,
    a_min_5pl,
    a_max_5pl,
    b_min_5pl,
    b_max_5pl,
    c_min_5pl,
    c_max_5pl,
    d_min_5pl,
    d_max_5pl,
    e_min_5pl,
    e_max_5pl,
    rps_list,
    n_candidates=80,
):
    e_mid = 0.5 * (e_min_5pl + e_max_5pl)

    choices = [
        (a_min_5pl, a_max_5pl),
        (b_min_5pl, b_max_5pl),
        (c_min_5pl, c_max_5pl),
        (d_min_5pl, d_max_5pl),
    ]
    combos16 = list(itertools.product(*choices))

    factors = np.unique(
        np.round(
            np.logspace(np.log10(1.2), np.log10(10.0), int(n_candidates)),
            6,
        )
    )

    results = []
    best = None
    best_key = None

    for f in factors:
        res = _evaluate_factor_5pl(
            factor=f,
            top=top_conc_5pl,
            A=0.0,
            B=0.0,
            C=1.0,
            D=0.0,
            e_val=e_mid,
            rps_list=rps_list,
            combos16=combos16,
        )
        results.append(res)

        key = (
            res["meets_min"],
            res["meets_preferred"],
            -res["score"],
            -res["factor"],
        )
        if best is None or key > best_key:
            best = res
            best_key = key

    if results:
        rec_df_5pl = pd.DataFrame(
            [
                {
                    "factor": r["factor"],
                    "worst_lower": r["worst_lower"],
                    "worst_linear": r["worst_linear"],
                    "worst_upper": r["worst_upper"],
                    "score": r["score"],
                    "meets_min": r["meets_min"],
                    "meets_preferred": r["meets_preferred"],
                }
                for r in results
            ]
        )
        rec_df_5pl = rec_df_5pl.sort_values(
            by=["meets_min", "meets_preferred", "score", "factor"],
            ascending=[False, False, True, True],
        )
        st.session_state["rec_df_5pl"] = rec_df_5pl
    else:
        st.session_state["rec_df_5pl"] = None

    return best


# ========================================================================== #
# ===================== Layout: main panels ===================== #
# ========================================================================== #

left_panel, graph_col = st.columns([1.15, 1.85], gap="large")

# -------------------- Left: Dilution series controls -----------------------

with left_panel:
    st.subheader("Dilution series")

    st.number_input(
        "Top concentration",
        min_value=1e-12,
        max_value=1e12,
        step=1.0,
        format="%.6g",
        key="top_conc_5pl",
        help=(
            "The top concentration is the highest dose you start with, "
            "setting the upper limit of the dose–response curve."
        ),
    )

    st.number_input(
        "Even dilution factor (applied 7×)",
        min_value=1.0001,
        max_value=1e9,
        step=0.01,
        format="%.6g",
        key="even_dil_factor_5pl",
        help="Example: 2 halves each step; √10≈3.162 is common.",
    )

    st.text_input(
        "Custom 7 dilution factors (override even factor if exactly 7 numbers)",
        key="dilution_str_5pl",
        placeholder="e.g., 3.162 3.162 3.162 3.162 3.162 3.162 3.162",
        help="Provide 7 step-wise multipliers (high→low).",
    )
    custom_factors = _parse_list(st.session_state["dilution_str_5pl"])
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

# -------------------- Right: Graph header & placeholder --------------------

with graph_col:
    plot_placeholder = st.empty()

# -------------------- Left: Parameter ranges (min/max) ---------------------

with left_panel:
    st.markdown("### Parameter ranges (min/max)")

    # --- Upload-derived info (only if Upload was used), like 4PL ---
    if _FROM_UPLOAD_5pl:
        with st.expander("Loaded from Upload page", expanded=False):
            st.write("param_bounds:", st.session_state.get("param_bounds"))
            mi = st.session_state.get("model_input")
            if isinstance(mi, pd.DataFrame):
                st.dataframe(mi)

        msg = st.session_state.get("_5pl_loaded_ranges_msg")
        if msg:
            st.info(msg)

        clear_col, _ = st.columns([1, 3])
        with clear_col:
            if st.button("Clear load", key="clear_5pl_load"):
                st.session_state.pop("param_bounds", None)
                st.session_state.pop("model_input", None)
                st.session_state.pop("_5pl_loaded_ranges_msg", None)
                st.session_state.pop("_last_bounds_5pl", None)
                st.session_state.pop("prefilled_5pl", None)
                st.rerun()

    # Ensure defaults exist (in case user cleared things)
    defaults_ranges = {
        "a_min_5pl": 0.8,
        "a_max_5pl": 1.2,
        "b_min_5pl": -2.0,
        "b_max_5pl": -0.5,
        "c_min_5pl": 0.1,
        "c_max_5pl": 3.0,
        "d_min_5pl": 0.0,
        "d_max_5pl": 0.2,
        "e_min_5pl": 0.8,
        "e_max_5pl": 1.2,
    }
    for k, v in defaults_ranges.items():
        if k not in st.session_state:
            st.session_state[k] = v

    colA, colMin, colMax = st.columns([0.6, 1, 1])
    with colA:
        st.markdown("**Parameter**")
    with colMin:
        st.markdown("**Min**")
    with colMax:
        st.markdown("**Max**")

    def row(param_label, k_min, k_max, min_cfg, max_cfg):
        cA, cMin, cMax = st.columns([0.6, 1, 1])
        with cA:
            st.markdown(param_label)
        with cMin:
            st.number_input("", key=k_min, label_visibility="collapsed", **min_cfg)
        with cMax:
            st.number_input("", key=k_max, label_visibility="collapsed", **max_cfg)

    row(
        "A",
        "a_min_5pl",
        "a_max_5pl",
        {"min_value": 0.0, "max_value": 2.0, "step": 0.01},
        {"min_value": 0.0, "max_value": 2.0, "step": 0.01},
    )

    row(
        "B",
        "b_min_5pl",
        "b_max_5pl",
        {"min_value": -10.0, "max_value": 10.0, "step": 0.01},
        {"min_value": -10.0, "max_value": 10.0, "step": 0.01},
    )

    row(
        "C",
        "c_min_5pl",
        "c_max_5pl",
        {"min_value": 1e-6, "max_value": 1e6, "step": 0.01, "format": "%.6g"},
        {"min_value": 1e-6, "max_value": 1e6, "step": 0.01, "format": "%.6g"},
    )

    row(
        "D",
        "d_min_5pl",
        "d_max_5pl",
        {"min_value": 0.0, "max_value": 1.0, "step": 0.01},
        {"min_value": 0.0, "max_value": 1.0, "step": 0.01},
    )

    row(
        "E",
        "e_min_5pl",
        "e_max_5pl",
        {"min_value": 0.1, "max_value": 5.0, "step": 0.01},
        {"min_value": 0.1, "max_value": 5.0, "step": 0.01},
    )

    st.text_input(
        "Relative potencies (comma/space separated)",
        key="rps_str_5pl",
        placeholder="e.g., 0.25, 0.5  1, 1.5, 2",
    )
    user_rps = _parse_rps(st.session_state["rps_str_5pl"])
    if not user_rps:
        rps = [0.4, 1.0, 1.6]
        st.caption("Using default RP values: 0.4 (40%), 1.0 (reference), 1.6 (160%).")
    else:
        rps = user_rps
        st.caption(f"Parsed RP values: {user_rps}")

# -------------------- Compute main graph data ------------------------

a_min_5pl = float(st.session_state["a_min_5pl"])
a_max_5pl = float(st.session_state["a_max_5pl"])
b_min_5pl = float(st.session_state["b_min_5pl"])
b_max_5pl = float(st.session_state["b_max_5pl"])
c_min_5pl = float(st.session_state["c_min_5pl"])
c_max_5pl = float(st.session_state["c_max_5pl"])
d_min_5pl = float(st.session_state["d_min_5pl"])
d_max_5pl = float(st.session_state["d_max_5pl"])
e_min_5pl = float(st.session_state["e_min_5pl"])
e_max_5pl = float(st.session_state["e_max_5pl"])

a = (a_min_5pl + a_max_5pl) / 2.0
b = (b_min_5pl + b_max_5pl) / 2.0
c = (c_min_5pl + c_max_5pl) / 2.0
d = (d_min_5pl + d_max_5pl) / 2.0
e = (e_min_5pl + e_max_5pl) / 2.0

top_conc_5pl = float(st.session_state["top_conc_5pl"])
even_factor = float(st.session_state["even_dil_factor_5pl"])

x_sparse_live = dp.generate_log_conc(
    top_conc=top_conc_5pl,
    dil_factor=even_factor,
    n_points=8,
    dense=False,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)
x_dense_live = dp.generate_log_conc(
    top_conc=top_conc_5pl,
    dil_factor=even_factor,
    n_points=8,
    dense=True,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)

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

# ===================== Main figure =====================

fig = go.Figure()

base_color = rp_color_map.get(1.0, palette[0])
y_dense_ref = _five_pl_logistic(x_dense_live, a, b, c, d, e)
fig.add_scatter(
    x=x_dense_live,
    y=y_dense_ref,
    mode="lines",
    name="Reference (RP=1.0)",
    line=dict(width=2, color=base_color),
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Reference</extra>",
)
y_sparse_ref = _five_pl_logistic(x_sparse_live, a, b, c, d, e)
fig.add_scatter(
    x=x_sparse_live,
    y=y_sparse_ref,
    mode="markers",
    marker=dict(size=7, color=base_color),
    showlegend=False,
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
)

for rp_val in rp_sorted:
    if rp_val == 1.0:
        continue
    color = rp_color_map[rp_val]
    y_dense = _five_pl_logistic(x_dense_live, a, b, c / rp_val, d, e)
    fig.add_scatter(
        x=x_dense_live,
        y=y_dense,
        mode="lines",
        name=f"Sample (RP={rp_val:g})",
        line=dict(width=2, color=color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )
    y_sparse = _five_pl_logistic(x_sparse_live, a, b, c / rp_val, d, e)
    fig.add_scatter(
        x=x_sparse_live,
        y=y_sparse,
        mode="markers",
        marker=dict(size=7, color=color),
        showlegend=False,
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

locked_start_index = 6
for idx, cv in enumerate(st.session_state["curves_5pl"]):
    grid = cv.get("grid", {}) or {}
    tc = grid.get("top_conc_5pl", top_conc_5pl)
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
    y_locked_line = _five_pl_logistic(
        x_dense_locked, cv["a"], cv["b"], cv["c"], cv["d"], cv["e"]
    )
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
            marker=dict(size=7, color=lock_color),
            hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
        )

fig.update_layout(
    title="Dose-Response Curves",
    xaxis_title="Log Concentration",
    yaxis_title="Response",
    legend_title=None,
    margin=dict(l=10, r=10, t=40, b=10),
)

with graph_col:
    plot_placeholder.plotly_chart(
        fig,
        config={"responsive": True, "displayModeBar": True},
        use_container_width=True,
    )

    st.markdown("### Add curve")
    default_label = f"Curve {st.session_state['next_label_idx_5pl']}"
    label = st.text_input(
        "Label for base curve",
        value=default_label,
        help="Base curve name; RP curves_5pl add ' (RP=...)'.",
        key="label_input_5pl",
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Add curve", type="primary", key="btn_save_curve_5pl"):
            grid_info = {
                "top_conc_5pl": top_conc_5pl,
                "even_factor": even_factor,
                "custom_factors": list(custom_factors)
                if len(custom_factors) == 7
                else [],
            }

            _lock_curve(label, a, b, c, d, e, rp=1.0, grid=grid_info)
            for rp_val in rp_sorted:
                if rp_val == 1.0:
                    continue
                rp_label = f"{label} (RP={rp_val:g})"
                _lock_curve(rp_label, a, b, c, d, e, rp=rp_val, grid=grid_info)

            st.session_state["next_label_idx_5pl"] += 1
            st.success("Saved current curve(s).")
            _rerun()

    with col_btn2:
        if st.button(
            "Clear all saved curves", key="btn_clear_curves_5pl_5pl"
        ):
            st.session_state["curves_5pl"] = []
            st.session_state["next_label_idx_5pl"] = 1
            st.info("Cleared all saved curves.")
            _rerun()

    st.markdown("---")

    st.markdown("### Dilution scheme recommender (from A, B, C, D, E ranges)")
    with st.expander("How this suggestion works", expanded=False):
        st.markdown(
            r"""
This tool recommends **even dilution factor(s)** using the current A, B, C, D, E
parameter ranges and the selected relative potencies (RPs).

For each candidate factor (log-spaced between **1.2** and **10**), it:

1. Builds an 8-point dilution series in log₁₀(concentration) space.  
2. Evaluates all **32 min/max combinations** of (A, B, C, D, E).  
3. For each combination & RP, computes the 5-parameter logistic curve and finds the
   log₁₀-concentrations where the curve reaches **10%** and **90%** of its dynamic range
   between the lower (D) and upper (A) asymptotes.  
4. Classifies each dilution point as:  

   - **Bottom anchor**: below the 10% point  
   - **Linear region**: between 10% and 90%  
   - **Top anchor**: above the 90% point  

For every curve/RP, it then compares the pattern  
(bottom, linear, top) to the target **(2, 3, 2)** using:
            """
        )
        st.latex(r"""
            J = (n_{\text{bottom}} - 2)^2
            + (n_{\text{linear}} - 3)^2
            + (n_{\text{top}} - 2)^2
        """)
        st.markdown(
            """
For a given dilution factor, we look at the **worst-case** \(J\) across all
32 parameter corners and all RPs. Factors with smaller worst-case \(J\) are better
(they are closest to the 2–3–2 pattern in every case).

The table below shows, for each candidate factor:

- The **worst-case counts** (bottom, linear, top)  
- The corresponding score \(J\)  
- Flags indicating whether it meets simple minimum / preferred coverage rules.
"""
        )

    n_candidates = st.slider(
        "Search resolution (# of candidate factors between 1.2 and 10, log-spaced)",
        min_value=10,
        max_value=200,
        value=80,
        step=10,
        key="n_candidates_5pl",
    )

    if st.button(
        "Recommend even dilution factors from A,B,C,D,E ranges",
        type="primary",
        key="btn_run_rec_5pl",
    ):
        suggest_factor_from_ranges(
            top_conc_5pl=top_conc_5pl,
            a_min_5pl=a_min_5pl,
            a_max_5pl=a_max_5pl,
            b_min_5pl=b_min_5pl,
            b_max_5pl=b_max_5pl,
            c_min_5pl=c_min_5pl,
            c_max_5pl=c_max_5pl,
            d_min_5pl=d_min_5pl,
            d_max_5pl=d_max_5pl,
            e_min_5pl=e_min_5pl,
            e_max_5pl=e_max_5pl,
            rps_list=rp_sorted,
            n_candidates=n_candidates,
        )
        st.success("Computed recommendations.")

# ===================== Full-width recommendations table =====================

rec_df_5pl = st.session_state.get("rec_df_5pl", None)
st.markdown("---")
st.markdown("### Recommended even dilution factors")

if rec_df_5pl is not None and not rec_df_5pl.empty:
    rec_df_5pl_disp = rec_df_5pl.copy()
    rec_df_5pl_disp["factor"] = rec_df_5pl_disp["factor"].map(lambda v: f"{v:.6g}")
    rec_df_5pl_disp["score"] = rec_df_5pl_disp["score"].map(lambda v: f"{v:.3g}")
    st.dataframe(rec_df_5pl_disp.head(20), use_container_width=True, height=320)

    st.download_button(
        "Download all recommendations (CSV)",
        data=rec_df_5pl.to_csv(index=False).encode("utf-8"),
        file_name="dilution_factor_recommendations_5pl.csv",
        mime="text/csv",
        key="btn_dl_rec_csv_5pl",
    )

    top_20 = rec_df_5pl.head(20)["factor"].tolist()
    st.markdown("#### Load a recommended factor into the current dilution settings")

    selected_factor = st.selectbox(
        "Choose a recommended factor to apply:",
        options=top_20,
        format_func=lambda v: f"{v:.6g}",
        key="rec_factor_select_5pl",
    )

    if st.button("Use selected factor", key="btn_use_rec_factor_5pl"):
        st.session_state["rec_factor_value_5pl"] = float(selected_factor)
        st.session_state["apply_rec_factor_5pl"] = True
        st.success(
            f"Will set even dilution factor to {selected_factor:.6g} "
            "and clear custom factors."
        )
        _rerun()
else:
    st.info(
        "Click **Recommend even dilution factors from A,B,C,D,E ranges** to generate "
        "a ranked table of candidate dilution factors."
    )

st.markdown("---")

# ===================== Edge cases (32 panels) =====================

st.markdown("### Edge cases: all min/max combinations of A, B, C, D, E")

x_sparse_edge = dp.generate_log_conc(
    top_conc=top_conc_5pl,
    dil_factor=even_factor,
    n_points=8,
    dense=False,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)
x_dense_edge = dp.generate_log_conc(
    top_conc=top_conc_5pl,
    dil_factor=even_factor,
    n_points=8,
    dense=True,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)

a_min_5pl_ec = float(st.session_state["a_min_5pl"])
a_max_5pl_ec = float(st.session_state["a_max_5pl"])
b_min_5pl_ec = float(st.session_state["b_min_5pl"])
b_max_5pl_ec = float(st.session_state["b_max_5pl"])
c_min_5pl_ec = float(st.session_state["c_min_5pl"])
c_max_5pl_ec = float(st.session_state["c_max_5pl"])
d_min_5pl_ec = float(st.session_state["d_min_5pl"])
d_max_5pl_ec = float(st.session_state["d_max_5pl"])
e_min_5pl_ec = float(st.session_state["e_min_5pl"])
e_max_5pl_ec = float(st.session_state["e_max_5pl"])

choices_edge = [
    (a_min_5pl_ec, a_max_5pl_ec),
    (b_min_5pl_ec, b_max_5pl_ec),
    (c_min_5pl_ec, c_max_5pl_ec),
    (d_min_5pl_ec, d_max_5pl_ec),
    (e_min_5pl_ec, e_max_5pl_ec),
]
combos_edge = list(itertools.product(*choices_edge))

edge_fig = make_subplots(
    rows=4,
    cols=8,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.06,
    subplot_titles=[
        f"A={a_:.3g}, B={b_:.3g}, C={c_:.3g}<br>D={d_:.3g}, E={e_:.3g}"
        for (a_, b_, c_, d_, e_) in combos_edge
    ],
)

for idx, (aa, bb, cc, dd, ee_) in enumerate(combos_edge, start=1):
    r = ((idx - 1) // 8) + 1
    c_idx = ((idx - 1) % 8) + 1
    for rp_val in rp_sorted if rp_sorted else [1.0]:
        color = rp_color_map.get(rp_val, palette[0] if rp_val == 1.0 else palette[1])
        edge_fig.add_scatter(
            x=x_dense_edge,
            y=_five_pl_logistic(x_dense_edge, aa, bb, cc / rp_val, dd, ee_),
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False,
            row=r,
            col=c_idx,
        )
        edge_fig.add_scatter(
            x=x_sparse_edge,
            y=_five_pl_logistic(x_sparse_edge, aa, bb, cc / rp_val, dd, ee_),
            mode="markers",
            marker=dict(size=5, color=color),
            showlegend=False,
            row=r,
            col=c_idx,
        )

for r in range(1, 5):
    for c_idx in range(1, 9):
        i = (r - 1) * 8 + c_idx
        edge_fig.layout[f"xaxis{i}"].title.text = ""
        edge_fig.layout[f"yaxis{i}"].title.text = ""
        edge_fig.update_xaxes(showticklabels=(r == 4), row=r, col=c_idx)
        edge_fig.update_yaxes(showticklabels=(c_idx == 1), row=r, col=c_idx)

tooltip_texts_5pl = [
    "Low baseline, highly sensitive, shallow slope, weak max response, symmetric",
    "Low baseline, highly sensitive, shallow slope, weak max response, asymmetric",
    "Low baseline, highly sensitive, shallow slope, strong max response, symmetric",
    "Low baseline, highly sensitive, shallow slope, strong max response, asymmetric",
    "Low baseline, low sensitivity, shallow slope, weak max response, symmetric",
    "Low baseline, low sensitivity, shallow slope, weak max response, asymmetric",
    "Low baseline, low sensitivity, shallow slope, strong max response, symmetric",
    "Low baseline, low sensitivity, shallow slope, strong max response, asymmetric",
    "Low baseline, highly sensitive, steep slope, weak max response, symmetric",
    "Low baseline, highly sensitive, steep slope, weak max response, asymmetric",
    "Low baseline, highly sensitive, steep slope, strong max response, symmetric",
    "Low baseline, highly sensitive, steep slope, strong max response, asymmetric",
    "Low baseline, low sensitivity, steep slope, weak max response, symmetric",
    "Low baseline, low sensitivity, steep slope, weak max response, asymmetric",
    "Low baseline, low sensitivity, steep slope, strong max response, symmetric",
    "Low baseline, low sensitivity, steep slope, strong max response, asymmetric",
    "High baseline, highly sensitive, shallow slope, weak max response, symmetric",
    "High baseline, highly sensitive, shallow slope, weak max response, asymmetric",
    "High baseline, highly sensitive, shallow slope, strong max response, symmetric",
    "High baseline, highly sensitive, shallow slope, strong max response, asymmetric",
    "High baseline, low sensitivity, shallow slope, weak max response, symmetric",
    "High baseline, low sensitivity, shallow slope, weak max response, asymmetric",
    "High baseline, low sensitivity, shallow slope, strong max response, symmetric",
    "High baseline, low sensitivity, shallow slope, strong max response, asymmetric",
    "High baseline, highly sensitive, steep slope, weak max response, symmetric",
    "High baseline, highly sensitive, steep slope, weak max response, asymmetric",
    "High baseline, highly sensitive, steep slope, strong max response, symmetric",
    "High baseline, highly sensitive, steep slope, strong max response, asymmetric",
    "High baseline, low sensitivity, steep slope, weak max response, symmetric",
    "High baseline, low sensitivity, steep slope, weak max response, asymmetric",
    "High baseline, low sensitivity, steep slope, strong max response, symmetric",
    "High baseline, low sensitivity, steep slope, strong max response, asymmetric",
]

edge_fig.update_layout(
    margin=dict(l=80, r=20, t=50, b=80),
    height=900,
    hoverlabel=dict(bgcolor="white"),
)

x_min_ec = float(np.min(x_dense_edge))
x_max_ec = float(np.max(x_dense_edge))
x_pad_ec = 0.02 * (x_max_ec - x_min_ec)
x_icon = x_max_ec - x_pad_ec
y_icon = 0.0

for idx, (_aa, _bb, _cc, _dd, _ee_) in enumerate(combos_edge, start=1):
    row0 = (idx - 1) // 8
    col0 = (idx - 1) % 8
    r = row0 + 1
    c_idx = col0 + 1

    tip = (
        tooltip_texts_5pl[idx - 1]
        if 1 <= idx <= len(tooltip_texts_5pl)
        else f"Edge case {idx}"
    )

    edge_fig.add_scatter(
        x=[x_icon],
        y=[y_icon],
        mode="markers",
        marker=dict(
            symbol="circle",
            size=20,
            color="white",
            line=dict(color="#475569", width=1.5),
        ),
        hovertext=[tip],
        hoverinfo="text",
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
        row=r,
        col=c_idx,
    )

    edge_fig.add_scatter(
        x=[x_icon],
        y=[y_icon],
        mode="text",
        text=["i"],
        textfont=dict(size=12, color="#475569"),
        hoverinfo="skip",
        showlegend=False,
        row=r,
        col=c_idx,
    )

edge_fig.add_annotation(
    x=0.5,
    y=-0.10,
    xref="paper",
    yref="paper",
    text="Log Concentration",
    showarrow=False,
    font=dict(size=14),
)
edge_fig.add_annotation(
    x=0.0,
    y=0.5,
    xref="paper",
    yref="paper",
    text="Response",
    showarrow=False,
    textangle=-90,
    xanchor="right",
    yanchor="middle",
    xshift=-40,
    font=dict(size=14),
)

st.plotly_chart(edge_fig, use_container_width=True)

# ======= RULE =======
st.markdown("---")

# ======= Row: Saved curves_5pl (left) | Dilution preview (right) =======
col_saved, col_preview = st.columns([1.15, 1.85], gap="large")

with col_saved:
    st.subheader("Saved curves")
    df_saved = curves_5pl_to_dataframe(st.session_state["curves_5pl"])
    if not df_saved.empty:
        st.dataframe(df_saved, use_container_width=True, height=320)
        st.download_button(
            "Export Saved curves_5pl CSV",
            data=df_saved.to_csv(index=False).encode("utf-8"),
            file_name="dose_response_curves_5pl_5pl.csv",
            mime="text/csv",
            key="btn_export_saved_csv_5pl",
        )
    else:
        st.info("No saved curves yet.")

with col_preview:
    st.subheader("Dilution preview (current settings)")
    conc_sparse_live = (10**x_sparse_live).astype(float)
    y_sparse_live = _five_pl_logistic(x_sparse_live, a, b, c, d, e)

    df_preview = pd.DataFrame(
        {
            "Well": np.arange(1, len(x_sparse_live) + 1, dtype=int),
            "log10(conc)": x_sparse_live,
            "conc": conc_sparse_live,
            "response (current)": y_sparse_live,
        }
    )
    dfp = df_preview.copy()
    dfp["log10(conc)"] = dfp["log10(conc)"].map(lambda v: f"{v:.6f}")
    dfp["conc"] = dfp["conc"].map(lambda v: f"{v:.6g}")
    dfp["response (current)"] = dfp["response (current)"].map(lambda v: f"{v:.4f}")

    st.dataframe(dfp, use_container_width=True, height=320)

    st.download_button(
        "Export Dilution Preview CSV",
        data=df_preview.to_csv(index=False).encode("utf-8"),
        file_name="dilution_preview_5pl.csv",
        mime="text/csv",
        key="btn_export_preview_csv_5pl",
    )

    if custom_factors and len(custom_factors) == 7:
        st.caption(
            "Using custom 7 dilution factors; even factor "
            f"(for dense grid) = {even_factor:.6g}"
        )
    else:
        st.caption(f"Using even dilution factor: {even_factor:.6g}")

# ===================== References =====================
st.markdown("---")
st.markdown("**References**")

st.markdown(
    """
- Brendan Bioanalytics. (n.d.). *4PL, 5PL, calibrators: Comparing 4PL & 5PL curve fitting and optimizing calibrator doses.*  
https://www.brendan.com/4pl-curve-fitting/

- Cumberland, W. N., Fong, Y., Yu, X., Defawe, O., Frahm, N., & De Rosa, S. (2015).  
*Nonlinear calibration model choice between the four and five-parameter logistic models.*  
Journal of Biopharmaceutical Statistics, 25(5), 972–983.  
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4263697/

- Stephenson, M., & Segall, J. (2024, Aug 6).  
*What is the 5PL formula?* Quantics Biostatistics.  
https://www.quantics.co.uk/blog/what-is-the-5pl-formula/
"""
)
