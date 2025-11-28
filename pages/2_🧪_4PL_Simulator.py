import itertools
import streamlit as st
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import qualitative
import dose_response as dp
from PIL import Image
import base64, io

# ===================== Sidebar Logo (works on all pages) =====================

LOGO_PATH = "graphics/Logo.jpg"

# Load and convert image
logo = Image.open(LOGO_PATH)
buffer = io.BytesIO()
logo.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode()

# Inject logo above navigation menu
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


st.set_page_config(page_title="4PL Quantification Tool", layout="wide")

# ======= Title =======
st.title("4PL Quantification Tool")

with st.expander("How to use this tool", expanded=False):
    st.markdown(
        """
**Parameters**

- **A** is the lower asymptote  
- **B** indicates steepness (proportional to the slope of the curve at the mid-point)  
- **C** is the concentration corresponding to 50% of the response (**EC₅₀**)  
- **D** is the upper asymptote  


**Edge cases**

These plots show all the edge-case combinations of parameters **A**, **B**, **C**, and **D**.  
Each panel represents a minimum/maximum setting of those parameters so users can see how extreme
values change the shape and behavior of the response curve.  


**Add curve**

Enter a name for the base curve. Once submitted, it will be saved and displayed in the curve list
at the bottom-left.
"""
    )


# ======= Session State / Defaults =======
DEFAULTS = {
    # Ranges (min/max) only; main graph uses averages
    "a_min": 0.8, "a_max": 1.2,
    "b_min": -2.0, "b_max": -0.5,
    "c_min": 0.1, "c_max": 3.0,
    "d_min": 0.0, "d_max": 0.2,

    # Dilution controls
    "top_conc": 10**2,            # 100
    "even_dil_factor": 10**(1/2), # √10 ≈ 3.162
    "dilution_str": "",           # optional custom 7 factors

    # App state
    "curves": [],
    "next_label_idx": 1,

    # RP input text (optional)
    "rps_str": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# For talking between recommender and main UI
st.session_state.setdefault("apply_rec_factor", False)
st.session_state.setdefault("rec_factor_value", None)
st.session_state.setdefault("rec_df", None)

# ---- Apply any pending recommended factor BEFORE widgets are created ----
if st.session_state.get("apply_rec_factor", False):
    rec_val = st.session_state.get("rec_factor_value", None)
    if rec_val is not None:
        # These keys back widgets, so only touch them here at the top
        st.session_state["even_dil_factor"] = float(rec_val)
        st.session_state["dilution_str"] = ""  # ensure even factor is used
    st.session_state["apply_rec_factor"] = False

# ======= Tunable region thresholds for the recommender =======
# These were originally for t-normalization; we now use true logistic
# boundaries at ~10% and ~90% of the dynamic range between D and A.
LINEAR_LOW, LINEAR_HIGH = 0.2, 0.8      # kept for documentation only
LOWER_ANCHOR_MAX = 0.10
UPPER_ANCHOR_MIN = 0.90

# ======= Helpers =======
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

# ======= Data model for saved curves =======
def curves_to_dataframe(curves):
    rows = []
    for idx, cv in enumerate(curves):
        grid = cv.get("grid", {})
        rows.append({
            "label": cv["label"],
            "a": cv["a"], "b": cv["b"], "c": cv["c"], "d": cv["d"],
            "rp": cv.get("rp"),
            "top_conc": grid.get("top_conc"),
            "even_factor": grid.get("even_factor"),
            "custom_factors": _list_to_str(grid.get("custom_factors", [])),
            "x_log10_points": _list_to_str(cv.get("x_log10_sparse", [])),
            "conc_points": _list_to_str(cv.get("conc_sparse", [])),
            "y_points": _list_to_str(cv.get("y_sparse", [])),
        })
    return pd.DataFrame(rows)

def _lock_curve(label, a, b, c, d, rp=None, grid=None):
    # Effective EC50 with RP
    c_eff = c / rp if (rp is not None and rp != 0) else c

    # Build the sparse x for THIS curve’s locked grid
    top_conc = float(grid.get("top_conc")) if grid else 10**2
    even_factor = float(grid.get("even_factor")) if grid else 10**0.5
    custom_factors = list(grid.get("custom_factors", [])) if grid else []

    x_sparse_locked = dp.generate_log_conc(
        top_conc=top_conc,
        dil_factor=even_factor,
        n_points=8,
        dense=False,
        dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
    )

    # Save the 8-point coordinates (log10 and linear conc), and the y at those points
    conc_sparse = (10 ** x_sparse_locked).astype(float)
    y_sparse_locked = dp.compute_curve(a, b, c_eff, d, x_sparse_locked)

    entry = {
        "label": label,
        "a": float(a), "b": float(b), "c": float(c_eff), "d": float(d),
        "rp": 1.0 if rp in (None, 0) else float(rp),

        # Persist the dilution grid used at lock time
        "grid": {
            "top_conc": float(top_conc),
            "even_factor": float(even_factor),
            "custom_factors": list(custom_factors),
        },

        # Persist the 8 locked points
        "x_log10_sparse": [float(v) for v in x_sparse_locked],
        "conc_sparse": [float(v) for v in conc_sparse],
        "y_sparse": [float(v) for v in y_sparse_locked],
    }
    st.session_state["curves"].append(entry)

# ======= Recommender internals (from parameter ranges) =======

def _four_pl_logistic(x_log10, a, b, c, d):
    """4PL in log10(x) space (same form as dp.compute_curve)."""
    return d + (a - d) / (1 + 10 ** (b * (x_log10 - np.log10(c))))

def _region_bounds_4pl(a, b, c, d, frac_low=0.1, frac_high=0.9):
    """
    Solve for log10(x) where the 4PL reaches given fractions of its dynamic range.

    frac_low / frac_high are in [0,1] and refer to
        y = d + frac * (a - d).
    """
    denom = (a - d)
    if np.allclose(denom, 0.0):
        # Degenerate: no usable dynamic range
        return -np.inf, np.inf

    def solve(frac):
        y = d + frac * denom  # target y
        # From 4PL inversion:
        # 10^(b*(x - log10(c))) = (a - y) / (y - d)
        num = (a - y)
        den = (y - d)
        if np.allclose(den, 0.0):
            return np.nan
        t = num / den
        if t <= 0 or np.isnan(t):
            return np.nan
        return np.log10(c) + (np.log10(t) / b)

    x_low = solve(frac_low)
    x_high = solve(frac_high)

    # If inversion fails, fall back to very wide bounds
    if np.isnan(x_low):
        x_low = -np.inf
    if np.isnan(x_high):
        x_high = np.inf

    return x_low, x_high

def _score_pattern(n_bottom, n_linear, n_top, target=(2, 3, 2)):
    """
    Squared-error score to the ideal pattern (bottom, linear, top) = (2,3,2).
    """
    tb, tl, tt = target
    return (
        (n_bottom - tb) ** 2 +
        (n_linear - tl) ** 2 +
        (n_top    - tt) ** 2
    )

@st.cache_data(show_spinner=False)
def _evaluate_factor(factor, top, A, B, C, D, rps_list, combos16=None):
    """
    Evaluate a single even dilution factor using the (2,3,2) squared-error pattern.

    For each candidate factor:
      - Build 8 log10-concentration points.
      - For each (A,B,C,D) combo and each RP:
          * Compute the 4PL curve.
          * Find x where the curve is at 10% and 90% of its dynamic range.
          * Count points in: bottom (below 10%), linear (10–90%), top (above 90%).
          * Compute J = (bottom-2)^2 + (linear-3)^2 + (top-2)^2.
      - Use the *worst-case* J over all combos and RPs as the factor's score.

    Returns:
      - worst_lower, worst_linear, worst_upper: counts at the worst-case pattern
      - score: worst-case squared error J
      - meets_min / meets_preferred: flags derived from counts (for UI sorting)
    """
    # 8 log-spaced x-points for this factor
    x_points = dp.generate_log_conc(
        top_conc=top, dil_factor=factor, n_points=8, dense=False
    )

    def eval_one(a_, b_, c_, d_):
        rows = []
        for rp in (rps_list or [1.0]):
            c_eff = c_ / rp
            y = dp.compute_curve(a_, b_, c_eff, d_, x_points)

            # region boundaries in log10 space, based on 10% and 90% of (A-D)
            x_low, x_high = _region_bounds_4pl(a_, b_, c_eff, d_)

            n_bottom = int(np.sum(x_points < x_low))
            n_top    = int(np.sum(x_points > x_high))
            n_linear = len(x_points) - n_bottom - n_top

            J = _score_pattern(n_bottom, n_linear, n_top)

            rows.append({
                "rp": float(rp),
                "bottom": n_bottom,
                "linear": n_linear,
                "top": n_top,
                "score": J,
            })

        # worst case = largest pattern error
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

    # Unpack worst-case counts & error
    wb = int(worst_overall["bottom"])
    wl = int(worst_overall["linear"])
    wt = int(worst_overall["top"])
    score = float(worst_overall["score"])

    # Flags used just for UI convenience (can be tuned)
    meets_min = (wl >= 2 and wb >= 1 and wt >= 1)
    meets_preferred = (wl >= 3 and wb >= 2 and wt >= 2)

    return {
        "factor": float(factor),

        # For the table + sorting
        "worst_lower": wb,
        "worst_linear": wl,
        "worst_upper": wt,
        "score": score,                # squared error to (2,3,2)

        "meets_min": bool(meets_min),
        "meets_preferred": bool(meets_preferred),

        "detail": per_combo,
    }

@st.cache_data(show_spinner=False)
def suggest_factor_from_ranges(
    top_conc,
    a_min, a_max,
    b_min, b_max,
    c_min, c_max,
    d_min, d_max,
    rps_list,
    n_candidates=80,
):
    """
    Use the current A,B,C,D ranges to suggest a single even dilution factor.

    - Builds all 16 min/max combinations
    - Tests log-spaced factors in [1.2, 10]
    - Picks the best factor based on:
        meets_min, meets_preferred, *smallest* score (pattern error),
        then smaller factor.
    """
    choices = [(a_min, a_max), (b_min, b_max), (c_min, c_max), (d_min, d_max)]
    combos16 = list(itertools.product(*choices))

    factors = np.unique(
        np.round(
            np.logspace(np.log10(1.2), np.log10(10.0), int(n_candidates)),
            6,
        )
    )

    best = None
    best_key = None

    for f in factors:
        res = _evaluate_factor(
            factor=f,
            top=top_conc,
            A=0.0, B=0.0, C=1.0, D=0.0,  # ignored when combos16 is used
            rps_list=rps_list,
            combos16=combos16,
        )
        # We want: meets_min=True, meets_preferred=True, smallest score, smallest factor
        key = (
            res["meets_min"],
            res["meets_preferred"],
            -res["score"],   # smaller score -> larger -score
            -res["factor"],  # smaller factor -> larger -factor
        )
        if best is None or key > best_key:
            best = res
            best_key = key

    return best

# ======= Row 1: Dilution series (left) | Graph (right) =======
left_panel, graph_col = st.columns([1.15, 1.85], gap="large")

with left_panel:
    st.subheader("Dilution series")

    with st.expander("What does 'Dilution series' mean?", expanded=False):
        st.markdown(
                "The **dilution factor** is how much each step is diluted from the previous one, "
                "creating a series of decreasing doses for the curve. For example, a factor of 3 "
                "means each well is 1/3 of the concentration of the prior well."
        )
        

    st.number_input(
    "Top concentration",
    min_value=1e-12, max_value=1e12,
    step=1.0, format="%.6g", key="top_conc",
    help="The top concentration is the highest dose you start with, setting the upper limit of the dose–response curve."
)

    st.number_input(
        "Even dilution factor (applied 7×)",
        min_value=1.0001, max_value=1e9,
        step=0.01, format="%.6g", key="even_dil_factor",
        help="Example: 2 halves each step; √10≈3.162 is common."
    )
    st.text_input(
        "Custom 7 dilution factors (override even factor if exactly 7 numbers)",
        key="dilution_str",
        placeholder="e.g., 3.162 3.162 3.162 3.162 3.162 3.162 3.162",
        help="Provide 7 step-wise multipliers (high→low)."
    )
    custom_factors = _parse_list(st.session_state["dilution_str"])
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



# Graph header and placeholder begin in Row 1 and continue in Row 2
with graph_col:
    st.subheader("Dose-Response Curves")
    plot_placeholder = st.empty()

# ======= Row 2: Parameter ranges (min/max) | Graph continues (right) =======
with left_panel:
    st.markdown("### Parameter ranges (min/max)")
    with st.expander("What do A, B, C, and D mean?", expanded=False):
        st.markdown(
            """
**A** – Lower asymptote  
**B** – Steepness (slope around the midpoint)  
**C** – EC₅₀: concentration giving 50% response  
**D** – Upper asymptote  
            """
        )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.number_input("a_min", 0.0, 2.0, step=0.01, key="a_min")
        st.number_input("c_min", 1e-6, 1e6, step=0.01, format="%.6g", key="c_min")
    with r1c2:
        st.number_input("a_max", 0.0, 2.0, step=0.01, key="a_max")
        st.number_input("c_max", 1e-6, 1e6, step=0.01, format="%.6g", key="c_max")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.number_input("b_min", -10.0, 10.0, step=0.01, key="b_min")
        st.number_input("d_min", 0.0, 1.0, step=0.01, key="d_min")
    with r2c2:
        st.number_input("b_max", -10.0, 10.0, step=0.01, key="b_max")
        st.number_input("d_max", 0.0, 1.0, step=0.01, key="d_max")

    # ---- Relative potencies (applied to c only) ----
    st.text_input(
        "Relative potencies (comma/space separated)",
        key="rps_str",
        placeholder="e.g., 0.25, 0.5  1, 1.5, 2"
    )
    user_rps = _parse_rps(st.session_state["rps_str"])
    # Default to 40%, reference, 160% if empty
    if not user_rps:
        rps = [0.4, 1.0, 1.6]
        st.caption("Using default RP values: 0.4 (40%), 1.0 (reference), 1.6 (160%)")
    else:
        rps = user_rps
        st.caption(f"Parsed RP values: {user_rps}")

# ======= Compute main-graph parameters as averages of min/max =======
a_min, a_max = float(st.session_state["a_min"]), float(st.session_state["a_max"])
b_min, b_max = float(st.session_state["b_min"]), float(st.session_state["b_max"])
c_min, c_max = float(st.session_state["c_min"]), float(st.session_state["c_max"])
d_min, d_max = float(st.session_state["d_min"]), float(st.session_state["d_max"])

# Averages for main graph
a = (a_min + a_max) / 2.0
b = (b_min + b_max) / 2.0
c = (c_min + c_max) / 2.0
d = (d_min + d_max) / 2.0

top_conc = float(st.session_state["top_conc"])
even_factor = float(st.session_state["even_dil_factor"])

# ======= Build live grids once =======
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

# ======= Color map for RP curves (consistent across main & edge plots) =======
palette = qualitative.Plotly  # length >= 10
rp_sorted = sorted(set(rps))
rp_color_map = {}
# Reserve index 0 for RP=1.0 "Reference"
color_cursor = 1
if 1.0 in rp_sorted:
    rp_color_map[1.0] = palette[0]
for rp in rp_sorted:
    if rp == 1.0:
        continue
    rp_color_map[rp] = palette[color_cursor % len(palette)]
    color_cursor += 1

# ======= Make live plot (main) =======
fig = go.Figure()

# Reference curve (RP=1.0)
base_color = rp_color_map.get(1.0, palette[0])
y_dense_ref = dp.compute_curve(a, b, c, d, x_dense_live)
fig.add_scatter(
    x=x_dense_live, y=y_dense_ref, mode="lines",
    name="Reference (RP=1.0)",
    line=dict(width=2, color=base_color),
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Reference</extra>",
)
y_sparse_ref = dp.compute_curve(a, b, c, d, x_sparse_live)
fig.add_scatter(
    x=x_sparse_live, y=y_sparse_ref, mode="markers",
    marker=dict(size=7, color=base_color),
    showlegend=False,
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
)

# RP curves
for rp in rp_sorted:
    if rp == 1.0:
        continue
    color = rp_color_map[rp]
    y_dense = dp.compute_curve(a, b, c / rp, d, x_dense_live)
    fig.add_scatter(
        x=x_dense_live, y=y_dense, mode="lines",
        name=f"Sample (RP={rp:g})",
        line=dict(width=2, color=color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )
    y_sparse = dp.compute_curve(a, b, c / rp, d, x_sparse_live)
    fig.add_scatter(
        x=x_sparse_live, y=y_sparse, mode="markers",
        marker=dict(size=7, color=color),
        showlegend=False,
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

# Locked curves
locked_start_index = 6
for idx, cv in enumerate(st.session_state["curves"]):
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
    y_locked_line = dp.compute_curve(cv["a"], cv["b"], cv["c"], cv["d"], x_dense_locked)
    fig.add_scatter(
        x=x_dense_locked, y=y_locked_line, mode="lines",
        name=f'{cv["label"]} (locked)',
        line=dict(dash="dash", color=lock_color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

    xs = cv.get("x_log10_sparse", [])
    ys = cv.get("y_sparse", [])
    if xs and ys and len(xs) == 8 and len(ys) == 8:
        fig.add_scatter(
            x=xs, y=ys, mode="markers",
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

# Render main graph & controls
with graph_col:
    plot_placeholder = st.empty()

    # Main graph
    plot_placeholder.plotly_chart(
        fig,
        config={"responsive": True, "displayModeBar": True},
        use_container_width=True,
    )

    # Curve labeling & buttons
    default_label = f"Curve {st.session_state['next_label_idx']}"
    label = st.text_input(
        "Label for base curve",
        value=default_label,
        help="Base curve name; RP curves add ' (RP=...)'.",
        key="label_input",
    )
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Add curve", type="primary", key="btn_add_curve"):
            base_rps = rp_sorted[:] if rp_sorted else [1.0]
            if 1.0 not in base_rps:
                base_rps = [1.0] + base_rps
            grid = {
                "top_conc": top_conc,
                "even_factor": even_factor,
                "custom_factors": list(custom_factors) if len(custom_factors) == 7 else [],
            }
            for rp in base_rps:
                lbl = label if rp == 1.0 else f"{label} (RP={rp:g})"
                _lock_curve(label=lbl, a=a, b=b, c=c, d=d, rp=rp, grid=grid)
            st.session_state["next_label_idx"] += 1
            st.success(f"Saved '{label}' with {len(base_rps)} curve(s).")
            _rerun()
    with col_btn2:
        if st.button("Clear all saved curves", key="btn_clear_curves"):
            st.session_state["curves"] = []
            st.session_state["next_label_idx"] = 1
            st.info("Cleared all saved curves.")
            _rerun()

    # --- Recommender lives under the graph, in this same column ---
    st.markdown("---")
    st.markdown("### Dilution scheme recommender (from A, B, C, D ranges)")

    with st.expander("How this suggestion works", expanded=False):

        st.markdown(
            """
    This tool recommends **even dilution factor(s)** using the current **A, B, C, D**
    parameter ranges and the selected relative potencies (RPs).

    For each candidate factor (log-spaced between **1.2** and **10**), it:

    1. Builds an 8-point dilution series in log10(concentration) space.  
    2. Evaluates **all 16 min/max combinations** of (A, B, C, D).  
    3. For each combination & RP, computes the 4-parameter logistic curve and finds the
    log10-concentrations where the curve reaches **10%** and **90%** of its dynamic range
    between the lower (D) and upper (A) asymptotes.  
    4. Classifies each dilution point as:  
    - **Bottom anchor**: below the 10% point  
    - **Linear region**: between 10% and 90%  
    - **Top anchor**: above the 90% point  

    For every curve/RP, it then compares the pattern  
    **(bottom, linear, top)** to the target **(2, 3, 2)** using:
    """
        )

        st.latex(r"""
            J = (n_{\text{bottom}} - 2)^2
            + (n_{\text{linear}} - 3)^2
            + (n_{\text{top}} - 2)^2
        """)

        st.markdown(
            """
    For a given dilution factor, we look at the **worst-case** \(J\) across all 16 parameter
    corners and all RPs. Factors with smaller worst-case \(J\) are better (they are closest
    to the 2–3–2 pattern in every case).

    The table below shows, for each candidate factor:

    - The **worst-case counts** (bottom, linear, top)  
    - The corresponding score \(J\)  
    - Flags indicating whether it meets simple minimum / preferred coverage rules.
    """
        )


        n_cand = st.slider(
            "Search resolution (# of candidate factors between 1.2 and 10, log-spaced)",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
            help="More candidates = more precise but slower.",
            key="n_cand_recommender",
        )

    # Run recommender on click, based on A,B,C,D rangess
    if st.button(
        "Recommend even dilution factors from A,B,C,D ranges",
        type="primary",
        key="btn_recommend_factors",
    ):
        # Rebuild ranges in case user just changed them
        a_min_local, a_max_local = float(st.session_state["a_min"]), float(st.session_state["a_max"])
        b_min_local, b_max_local = float(st.session_state["b_min"]), float(st.session_state["b_max"])
        c_min_local, c_max_local = float(st.session_state["c_min"]), float(st.session_state["c_max"])
        d_min_local, d_max_local = float(st.session_state["d_min"]), float(st.session_state["d_max"])

        choices_local = [
            (a_min_local, a_max_local),
            (b_min_local, b_max_local),
            (c_min_local, c_max_local),
            (d_min_local, d_max_local),
        ]
        combos16 = list(itertools.product(*choices_local))

        factors = np.unique(
            np.round(
                np.logspace(np.log10(1.2), np.log10(10.0), int(n_cand)),
                6,
            )
        )

        rows = []
        for f in factors:
            res = _evaluate_factor(
                factor=f,
                top=top_conc,
                A=0.0, B=0.0, C=1.0, D=0.0,  # ignored because combos16 is used
                rps_list=rp_sorted,
                combos16=combos16,
            )
            rows.append(res)

        if rows:
            rec_df = pd.DataFrame(
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
                    for r in rows
                ]
            )
            # Ranking: prefer factors that meet the minimum rules and
            # have the smallest pattern-error score, then smaller factor.
            rec_df = rec_df.sort_values(
                by=["meets_min", "meets_preferred", "score", "factor"],
                ascending=[False, False, True, True],
            )
            st.session_state["rec_df"] = rec_df
        else:
            st.session_state["rec_df"] = None
            st.warning("No candidate factors were evaluated (check settings).")

# ======= Full-width recommendations table =======
rec_df = st.session_state.get("rec_df", None)
st.markdown("---")
st.markdown("### Recommended even dilution factors")

if rec_df is not None and not rec_df.empty:
    rec_df_disp = rec_df.copy()
    rec_df_disp["factor"] = rec_df_disp["factor"].map(lambda v: f"{v:.6g}")
    rec_df_disp["score"] = rec_df_disp["score"].map(lambda v: f"{v:.3g}")
    st.dataframe(rec_df_disp.head(20), use_container_width=True, height=320)

    st.download_button(
        "Download all recommendations (CSV)",
        data=rec_df.to_csv(index=False).encode("utf-8"),
        file_name="dilution_factor_recommendations.csv",
        mime="text/csv",
        key="btn_dl_rec_csv",
    )

    # Select + apply a recommended factor
    top_20 = rec_df.head(20)["factor"].tolist()
    st.markdown("#### Load a recommended factor into the current dilution settings")

    selected_factor = st.selectbox(
        "Choose a recommended factor to apply:",
        options=top_20,
        format_func=lambda v: f"{v:.6g}",
        key="rec_factor_select",
    )

    if st.button("Use selected factor", key="btn_use_rec_factor"):
        st.session_state["rec_factor_value"] = float(selected_factor)
        st.session_state["apply_rec_factor"] = True
        st.success(
            f"Will set even dilution factor to {selected_factor:.6g} and clear custom factors."
        )
        _rerun()
else:
    st.info(
        "Click **Recommend even dilution factors from A,B,C,D ranges** to generate "
        "a ranked table of candidate dilution factors."
    )

# ======= RULE =======
st.markdown("---")

# ======= 16 Edge-case subplots =======
st.markdown("### Edge cases: all min/max combinations of A, B, C, D")
with st.expander("What is Edge cases?", expanded=False):
    st.markdown(
        "These plots show all the edge-case combinations of parameters **A, B, C, D, and/or E**. "
        "Each panel represents a minimum/maximum setting of those parameters so users can see how "
        "extreme values change the shape and behavior of the response curve."
    )



x_sparse_edge = dp.generate_log_conc(
    top_conc=top_conc, dil_factor=even_factor, n_points=8, dense=False,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)
x_dense_edge = dp.generate_log_conc(
    top_conc=top_conc, dil_factor=even_factor, n_points=8, dense=True,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)

# Refresh min/max in case user changed them
a_min, a_max = float(st.session_state["a_min"]), float(st.session_state["a_max"])
b_min, b_max = float(st.session_state["b_min"]), float(st.session_state["b_max"])
c_min, c_max = float(st.session_state["c_min"]), float(st.session_state["c_max"])
d_min, d_max = float(st.session_state["d_min"]), float(st.session_state["d_max"])

choices = [(a_min, a_max), (b_min, b_max), (c_min, c_max), (d_min, d_max)]
combos = list(itertools.product(*choices))

edge_fig = make_subplots(
    rows=4, cols=4,
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.03, vertical_spacing=0.06,
    subplot_titles=[
        f"A={a_:.3g}, B={b_:.3g}, C={c_:.3g}, D={d_:.3g}"
        for (a_, b_, c_, d_) in combos
    ],
)

for idx, (aa, bb, cc, dd) in enumerate(combos, start=1):
    r = ((idx - 1) // 4) + 1
    k = ((idx - 1) % 4) + 1
    for rp in rp_sorted if rp_sorted else [1.0]:
        color = rp_color_map.get(rp, palette[0] if rp == 1.0 else palette[1])
        edge_fig.add_scatter(
            x=x_dense_edge,
            y=dp.compute_curve(aa, bb, cc / rp, dd, x_dense_edge),
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False,
            row=r,
            col=k,
        )
        edge_fig.add_scatter(
            x=x_sparse_edge,
            y=dp.compute_curve(aa, bb, cc / rp, dd, x_sparse_edge),
            mode="markers",
            marker=dict(size=5, color=color),
            showlegend=False,
            row=r,
            col=k,
        )

for r in range(1, 5):
    for c in range(1, 5):
        i = (r - 1) * 4 + c
        edge_fig.layout[f"xaxis{i}"].title.text = ""
        edge_fig.layout[f"yaxis{i}"].title.text = ""
        edge_fig.update_xaxes(showticklabels=(r == 4), row=r, col=c)
        edge_fig.update_yaxes(showticklabels=(c == 1), row=r, col=c)

edge_fig.update_layout(margin=dict(l=80, r=20, t=50, b=80), height=850)
edge_fig.add_annotation(
    x=0.5, y=-0.10, xref="paper", yref="paper",
    text="log10(conc)", showarrow=False, font=dict(size=14),
)
edge_fig.add_annotation(
    x=0.0, y=0.5, xref="paper", yref="paper",
    text="response", showarrow=False, textangle=-90,
    xanchor="right", yanchor="middle", xshift=-40,
    font=dict(size=14),
)

# ----- 4PL edge-case hover badges: stable circle+i with hover text -----

edge_fig.update_layout(hovermode="closest")

# 禁用已有曲线/散点的 hover（避免抢气泡）
for t in list(edge_fig.data):
    mode = getattr(t, "mode", "") or ""
    if "text" not in mode:
        t.update(hoverinfo="skip", hovertemplate=None)


# 2) 你的 4x4 说明文本矩阵（按行给出的 16 条）
tooltip_texts_4pl = [
    [
        "Shallow slope, highly sensitive drug (low EC50), low top response, low baseline — almost flat; barely measurable effect",
        "Shallow slope, sensitive drug, low top, high baseline — small dynamic range (baseline starts high)",
        "Shallow slope, weak drug (high EC50), low top, low baseline — minimal effect overall",
        "Shallow slope, weak drug, low top, high baseline — needs high dose to see a modest effect",
    ],
    [
        "Steep slope, sensitive drug, low top, low baseline — sharp jump but small ceiling",
        "Steep slope, sensitive, low top, high baseline — steep, compressed dynamic range",
        "Steep slope, weak drug, low top, low baseline — sharp transition but cannot reach high effect",
        "Steep slope, weak drug, low top, high baseline — steep curve but needs high dose; limited max effect",
    ],
    [
        "Shallow slope, sensitive drug, high top, low baseline — gradual rise; small slope",
        "Shallow slope, sensitive, high top, high baseline — shallow dynamic response",
        "Shallow slope, weak drug, high top, low baseline — fairly flat; drug ineffective at normal doses",
        "Shallow slope, weak drug, high top, high baseline — small rise from an already-high baseline",
    ],
    [
        "Steep slope, sensitive drug, high top, low baseline — strong sigmoidal curve",
        "Steep slope, sensitive drug, high top, high baseline — steep rise starting from high baseline",
        "Steep slope, weak drug, high top, low baseline — needs high dose but transitions sharply",
        "Steep slope, weak drug, high top, high baseline — steep curve but requires very high dose",
    ],
]
# ----- 4PL edge-case hover badges：把“ⓘ”放在右下角 -----

N_COLS = 4
x_min = float(np.min(x_dense_edge))
x_max = float(np.max(x_dense_edge))
x_pad = 0.02 * (x_max - x_min)  # 往左缩一点，避免贴死

for idx, (aa, bb, cc, dd) in enumerate(combos, start=1):
    row0 = (idx - 1) // N_COLS
    col0 = (idx - 1) %  N_COLS
    r, c = row0 + 1, col0 + 1

    # ---- ★ y 坐标统一放 y=0 ----
    y_icon = 0.0

    # ---- ★ x 坐标放在最右边（往左缩一点 padding） ----
    x_icon = x_max - x_pad

    tip = tooltip_texts_4pl[row0][col0]

    # ---- 圆圈（负责 hover）----
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

    # ---- i 字 ----
    edge_fig.add_scatter(
        x=[x_icon], y=[y_icon],
        mode="text",
        text=["i"],
        textfont=dict(size=13, color="#475569"),
        hoverinfo="skip",
        showlegend=False,
        row=r, col=c,
    )


# 让 hover 气泡底色统一
edge_fig.update_layout(hoverlabel=dict(bgcolor="white"))


st.plotly_chart(edge_fig, config={"responsive": True}, use_container_width=True)





# ======= RULE =======
st.markdown("---")

# ======= Row 3: Saved curves (left) | Dilution preview (right) =======
col_saved, col_preview = st.columns([1.15, 1.85], gap="large")

with col_saved:
    st.subheader("Saved curves")
    df_saved = curves_to_dataframe(st.session_state["curves"])
    if not df_saved.empty:
        st.dataframe(df_saved, use_container_width=True, height=320)
        st.download_button(
            "Export Saved Curves CSV",
            data=df_saved.to_csv(index=False).encode("utf-8"),
            file_name="dose_response_curves.csv",
            mime="text/csv",
            key="btn_export_saved_csv",
        )
    else:
        st.info("No saved curves yet.")

with col_preview:
    st.subheader("Dilution preview (current settings)")
    conc_sparse_live = (10 ** x_sparse_live).astype(float)
    y_sparse_live = dp.compute_curve(a, b, c, d, x_sparse_live)

    df_preview = pd.DataFrame({
        "Well": np.arange(1, len(x_sparse_live) + 1, dtype=int),
        "log10(conc)": x_sparse_live,
        "conc": conc_sparse_live,
        "response (current)": y_sparse_live,
    })
    dfp = df_preview.copy()
    dfp["log10(conc)"] = dfp["log10(conc)"].map(lambda v: f"{v:.6f}")
    dfp["conc"] = dfp["conc"].map(lambda v: f"{v:.6g}")
    dfp["response (current)"] = dfp["response (current)"].map(lambda v: f"{v:.4f}")

    st.dataframe(dfp, use_container_width=True, height=320)

    st.download_button(
        "Export Dilution Preview CSV",
        data=df_preview.to_csv(index=False).encode("utf-8"),
        file_name="current_dilution_preview.csv",
        mime="text/csv",
        key="btn_export_preview_csv",
    )

    if len(custom_factors) == 7:
        st.caption(f"Using custom factors: {custom_factors}")
    else:
        st.caption(f"Using even dilution factor: {even_factor:.6g}")

st.markdown("---")
st.markdown("**References**")
st.markdown(
    """
- MyAssays. (n.d.). *Four parameter logistic regression.* MyAssays.  
  https://www.myassays.com/four-parameter-logistic-regression.html  

- Smirnov, P., Kofia, V., Maru, A., Freeman, M., Ho, C., El-Hachem, N., Adam, G.-A., Ba-alawi, W., Safikhani, Z., & Haibe-Kains, B. (2018).  
  *PharmacoDB: An integrative database for mining in vitro anticancer drug screening studies.*  
  Nucleic Acids Research, 46(D1), D994–D1002.  
  https://doi.org/10.1093/nar/gkx911  

- United States Pharmacopeial Convention. (2012).  
  *Analysis of biological assays (USP 35 General Chapter 〈1034〉).*  
    """
)
