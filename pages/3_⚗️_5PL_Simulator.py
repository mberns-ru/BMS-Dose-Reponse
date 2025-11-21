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

st.set_page_config(page_title="5PL Quantification Tool", layout="wide")

# ======= Title =======
st.title("5PL Quantification Tool")

with st.expander("How to use this tool", expanded=False):
    st.markdown(
        r"""
**Parameters**

- **A** is the lower asymptote  
- **B** indicates steepness (proportional to the slope of the curve at the mid-point)  
- **C** is the concentration corresponding to 50% of the response (**EC₅₀**)  
- **D** is the upper asymptote  
- **E** is the asymmetry parameter  

In log₁₀ concentration space, the 5-parameter logistic (5PL) used here is:

\[
y = D + \frac{A - D}{\left(1 + 10^{\,B\,(\log_{10}(x) - \log_{10}(C))}\right)^{E}}
\]

When \(E = 1\), this reduces to the usual 4PL.

**Edge cases**

These plots show all the edge-case combinations of parameters **A**, **B**, **C**, **D**, and **E**  
(using min/max for each). Each panel represents a minimum/maximum setting of those
parameters so you can see how extreme values change the shape and behavior of the response
curve.

**Add curve**

Enter a name for the base curve. Once submitted, it will be saved and displayed in the curve
list at the bottom-left.
"""
    )

# ======= Session State / Defaults =======
DEFAULTS = {
    # Ranges (min/max) only; main graph uses averages
    "a_min": 0.8, "a_max": 1.2,
    "b_min": -2.0, "b_max": -0.5,
    "c_min": 0.1, "c_max": 3.0,
    "d_min": 0.0, "d_max": 0.2,
    "e_min": 0.8, "e_max": 1.2,   # asymmetry around 1 (≈4PL)

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

# ======= 5PL core =======

def _five_pl_logistic(x_log10, a, b, c, d, e):
    """
    5PL in log10(x) space:

        y = D + (A - D) / (1 + 10^{B (x - log10(C))})^E
    """
    # Avoid log10 issues if c <= 0
    c = float(c)
    if c <= 0:
        c = 1e-12
    return d + (a - d) / (1.0 + np.power(10.0, b * (x_log10 - np.log10(c))))**e

def _region_bounds_5pl(a, b, c, d, e, frac_low=0.1, frac_high=0.9):
    """
    Solve for log10(x) where the 5PL reaches given fractions of its dynamic range.

    frac_low / frac_high are in [0,1] and refer to
        y = d + frac * (a - d).

    Inverting:

        y - d = (a - d) / (1 + T)^E,  T = 10^{B (x - log10(C))}
        frac = (y - d) / (a - d) = 1 / (1 + T)^E
        (1 + T)^E = 1 / frac
        T = frac^{-1/E} - 1
        x = log10(C) + log10(T) / B
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

    # If inversion fails, fall back to wide bounds
    if np.isnan(x_low):
        x_low = -np.inf
    if np.isnan(x_high):
        x_high = np.inf

    # Ensure ordering
    if x_low > x_high:
        x_low, x_high = x_high, x_low

    return x_low, x_high

# ======= Data model for saved curves =======
def curves_to_dataframe(curves):
    rows = []
    for idx, cv in enumerate(curves):
        grid = cv.get("grid", {})
        rows.append({
            "label": cv["label"],
            "a": cv["a"], "b": cv["b"], "c": cv["c"], "d": cv["d"], "e": cv["e"],
            "rp": cv.get("rp"),
            "top_conc": grid.get("top_conc"),
            "even_factor": grid.get("even_factor"),
            "custom_factors": _list_to_str(grid.get("custom_factors", [])),
            "x_log10_points": _list_to_str(cv.get("x_log10_sparse", [])),
            "conc_points": _list_to_str(cv.get("conc_sparse", [])),
            "y_points": _list_to_str(cv.get("y_sparse", [])),
        })
    return pd.DataFrame(rows)

def _lock_curve(label, a, b, c, d, e, rp=None, grid=None):
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
    y_sparse_locked = _five_pl_logistic(x_sparse_locked, a, b, c_eff, d, e)

    entry = {
        "label": label,
        "a": float(a), "b": float(b), "c": float(c_eff), "d": float(d), "e": float(e),
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
def _evaluate_factor(factor, top, A, B, C, D, E, rps_list, combos=None):
    """
    Evaluate a single even dilution factor using the (2,3,2) squared-error pattern.

    - factor: even dilution factor to test
    - top: top concentration (linear units)
    - A,B,C,D,E: nominal/logical parameter ranges
    - rps_list: list of relative potencies
    - combos: list of (a,b,c,d,e) tuples representing edge-case parameter
              combinations; if None, use a single (A,B,C,D,E).
    """
    # Build the sparse x for this candidate factor (for pattern evaluation)
    x_sparse = dp.generate_log_conc(
        top_conc=top,
        dil_factor=factor,
        n_points=8,
        dense=False,
        dilution_factors=None,
    )

    if combos is None:
        combos_iter = [(A, B, C, D, E)]
    else:
        combos_iter = combos

    worst_lower = 0
    worst_linear = 0
    worst_upper = 0
    worst_score = np.inf

    # Evaluate each combination + each RP, track worst pattern
    for (a_, b_, c_, d_, e_) in combos_iter:
        x_low, x_high = _region_bounds_5pl(a_, b_, c_, d_, e_, frac_low=0.1, frac_high=0.9)

        for rp in rps_list if rps_list else [1.0]:
            c_eff = c_ / rp if rp != 0 else c_

            # Evaluate y at the sparse points to determine region membership
            y_sparse = _five_pl_logistic(x_sparse, a_, b_, c_eff, d_, e_)

            # Determine which x's are in bottom/linear/top based on x_low/x_high
            # Bottom: x <= x_low
            # Linear: x_low < x < x_high
            # Top:    x >= x_high
            n_bottom = int(np.sum(x_sparse <= x_low))
            n_linear = int(np.sum((x_sparse > x_low) & (x_sparse < x_high)))
            n_top = int(np.sum(x_sparse >= x_high))

            score = _score_pattern(n_bottom, n_linear, n_top)

            worst_lower = max(worst_lower, n_bottom)
            worst_linear = min(worst_linear if worst_linear else n_linear, n_linear)
            worst_upper = max(worst_upper, n_top)
            worst_score = max(worst_score, score)

    meets_min = (worst_linear >= 1) and (worst_lower >= 1) and (worst_upper >= 1)
    meets_preferred = (worst_linear >= 3) and (worst_lower >= 2) and (worst_upper >= 2)

    return {
        "factor": float(factor),
        "worst_lower": int(worst_lower),
        "worst_linear": int(worst_linear),
        "worst_upper": int(worst_upper),
        "score": float(worst_score),
        "meets_min": bool(meets_min),
        "meets_preferred": bool(meets_preferred),
    }

def suggest_factor_from_ranges(
    top_conc,
    a_min, a_max,
    b_min, b_max,
    c_min, c_max,
    d_min, d_max,
    e_min, e_max,
    rps_list,
    n_cand=80,
):
    """
    Use the range of A,B,C,D,E and user RP list to propose even dilution factors.
    """
    # Build edge-case combinations (min/max for each parameter)
    choices = [
        (a_min, a_max),
        (b_min, b_max),
        (c_min, c_max),
        (d_min, d_max),
        (e_min, e_max),
    ]
    combos = list(itertools.product(*choices))  # 2^5 = 32 combinations

    rp_sorted = sorted(set(rps_list)) if rps_list else [1.0]

    # Log-spaced factors from ~1.2 to 10 on logarithmic scale
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
            A=0.0, B=0.0, C=1.0, D=0.0, E=1.0,  # ignored because combos is used
            rps_list=rp_sorted,
            combos=combos,
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

# ======= Layout: Row 1 =======
left_panel, graph_col = st.columns([0.9, 1.6], gap="large")

with left_panel:
    st.markdown("### Dilution settings")

    st.number_input(
        "Top concentration",
        min_value=1e-6,
        max_value=1e12,
        value=float(st.session_state["top_conc"]),
        step=0.01,
        format="%.6g",
        key="top_conc",
    )

    st.number_input(
        "Even dilution factor",
        min_value=1.01,
        max_value=100.0,
        value=float(st.session_state["even_dil_factor"]),
        step=0.01,
        format="%.6g",
        key="even_dil_factor",
    )

    st.text_input(
        "Custom dilution factors (7 numbers, comma/space separated)",
        key="dilution_str",
        placeholder="e.g., 3 3 3 3 3 3 3",
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

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.number_input("e_min", 0.1, 5.0, step=0.01, key="e_min")
    with r3c2:
        st.number_input("e_max", 0.1, 5.0, step=0.01, key="e_max")

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
e_min, e_max = float(st.session_state["e_min"]), float(st.session_state["e_max"])

# Averages for main graph
a = (a_min + a_max) / 2.0
b = (b_min + b_max) / 2.0
c = (c_min + c_max) / 2.0
d = (d_min + d_max) / 2.0
e = (e_min + e_max) / 2.0

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
y_dense_ref = _five_pl_logistic(x_dense_live, a, b, c, d, e)
fig.add_scatter(
    x=x_dense_live, y=y_dense_ref, mode="lines",
    name="Reference (RP=1.0)",
    line=dict(width=2, color=base_color),
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Reference</extra>",
)
y_sparse_ref = _five_pl_logistic(x_sparse_live, a, b, c, d, e)
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
    y_dense = _five_pl_logistic(x_dense_live, a, b, c / rp, d, e)
    fig.add_scatter(
        x=x_dense_live, y=y_dense, mode="lines",
        name=f"Sample (RP={rp:g})",
        line=dict(width=2, color=color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )
    y_sparse = _five_pl_logistic(x_sparse_live, a, b, c / rp, d, e)
    fig.add_scatter(
        x=x_sparse_live, y=y_sparse, mode="markers",
        marker=dict(size=7, color=color),
        showlegend=False,
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

fig.update_layout(
    xaxis_title="log10(concentration)",
    yaxis_title="Response",
    height=600,
    margin=dict(l=60, r=20, t=50, b=60),
)
plot_placeholder.plotly_chart(fig, config={"responsive": True}, use_container_width=True)

# ======= Row 4: Add curve + Recommender =======
col_add_curve, col_rec = st.columns([1.0, 1.4], gap="large")

with col_add_curve:
    st.markdown("### Add curve")
    label_default = f"Curve {st.session_state['next_label_idx']}"
    label = st.text_input("Curve label", value=label_default, key="new_curve_label_5pl")

    if st.button("Save current curve(s) & fit (locked)", key="btn_save_curve_5pl"):
        # Lock reference curve
        grid_info = {
            "top_conc": top_conc,
            "even_factor": even_factor,
            "custom_factors": custom_factors,
        }
        _lock_curve(label, a, b, c, d, e, rp=1.0, grid=grid_info)

        # Lock RP curves
        for rp in rp_sorted:
            if rp == 1.0:
                continue
            rp_label = f"{label} (RP={rp:g})"
            _lock_curve(rp_label, a, b, c, d, e, rp=rp, grid=grid_info)

        st.session_state["next_label_idx"] += 1
        st.success("Saved current curve(s).")

    if st.button("Clear all saved curves", key="btn_clear_curves_5pl"):
        st.session_state["curves"] = []
        st.session_state["next_label_idx"] = 1
        st.success("Cleared all saved curves.")

with col_rec:
    st.markdown("### Dilution Scheme Recommender (5PL)")
    st.markdown(
        "Recommends even dilution factors that give ≈2 points in each asymptote "
        "and ≈3 points in the linear region, across all min/max combinations of "
        "A, B, C, D, and E and all specified relative potencies."
    )

    if st.button("Recommend even dilution factors from A,B,C,D,E ranges", key="btn_run_rec_5pl"):
        suggest_factor_from_ranges(
            top_conc=top_conc,
            a_min=a_min, a_max=a_max,
            b_min=b_min, b_max=b_max,
            c_min=c_min, c_max=c_max,
            d_min=d_min, d_max=d_max,
            e_min=e_min, e_max=e_max,
            rps_list=rp_sorted,
            n_cand=80,
        )
        st.success("Computed recommendations.")

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
        file_name="dilution_factor_recommendations_5pl.csv",
        mime="text/csv",
        key="btn_dl_rec_csv_5pl",
    )

    # Select + apply a recommended factor
    top_20 = rec_df.head(20)["factor"].tolist()
    st.markdown("#### Load a recommended factor into the current dilution settings")

    selected_factor = st.selectbox(
        "Choose a recommended factor to apply:",
        options=top_20,
        format_func=lambda v: f"{v:.6g}",
        key="rec_factor_select_5pl",
    )

    if st.button("Use selected factor", key="btn_use_rec_factor_5pl"):
        st.session_state["rec_factor_value"] = float(selected_factor)
        st.session_state["apply_rec_factor"] = True
        st.success(
            f"Will set even dilution factor to {selected_factor:.6g} and clear custom factors."
        )
        _rerun()
else:
    st.info(
        "Click **Recommend even dilution factors from A,B,C,D,E ranges** to generate "
        "a ranked table of candidate dilution factors."
    )

# ======= RULE =======
st.markdown("---")

# ======= 32 Edge-case subplots (A,B,C,D,E min/max) =======
st.markdown("### Edge cases: all min/max combinations of A, B, C, D, E (32 panels)")

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
e_min, e_max = float(st.session_state["e_min"]), float(st.session_state["e_max"])

choices_edge = [(a_min, a_max), (b_min, b_max), (c_min, c_max), (d_min, d_max), (e_min, e_max)]
combos_edge = list(itertools.product(*choices_edge))  # 32 combos

edge_fig = make_subplots(
    rows=4, cols=8,
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.02, vertical_spacing=0.06,
    subplot_titles=[
        f"A={a_:.3g}, B={b_:.3g}, C={c_:.3g}, D={d_:.3g}, E={e_:.3g}"
        for (a_, b_, c_, d_, e_) in combos_edge
    ],
)

for idx, (aa, bb, cc, dd, ee_) in enumerate(combos_edge, start=1):
    r = ((idx - 1) // 8) + 1
    k = ((idx - 1) % 8) + 1
    for rp in rp_sorted if rp_sorted else [1.0]:
        color = rp_color_map.get(rp, palette[0] if rp == 1.0 else palette[1])
        edge_fig.add_scatter(
            x=x_dense_edge,
            y=_five_pl_logistic(x_dense_edge, aa, bb, cc / rp, dd, ee_),
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False,
            row=r,
            col=k,
        )
        edge_fig.add_scatter(
            x=x_sparse_edge,
            y=_five_pl_logistic(x_sparse_edge, aa, bb, cc / rp, dd, ee_),
            mode="markers",
            marker=dict(size=5, color=color),
            showlegend=False,
            row=r,
            col=k,
        )

for r in range(1, 5):
    for c_ in range(1, 9):
        i = (r - 1) * 8 + c_
        edge_fig.layout[f"xaxis{i}"].title.text = ""
        edge_fig.layout[f"yaxis{i}"].title.text = ""
        edge_fig.update_xaxes(showticklabels=(r == 4), row=r, col=c_)
        edge_fig.update_yaxes(showticklabels=(c_ == 1), row=r, col=c_)

edge_fig.update_layout(margin=dict(l=80, r=20, t=50, b=80), height=900)
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

st.plotly_chart(edge_fig, config={"responsive": True}, use_container_width=True)

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
            file_name="dose_response_curves_5pl.csv",
            mime="text/csv",
            key="btn_export_saved_csv_5pl",
        )
    else:
        st.info("No saved curves yet.")

with col_preview:
    st.subheader("Dilution preview (current settings)")
    conc_sparse_live = (10 ** x_sparse_live).astype(float)
    y_sparse_live = _five_pl_logistic(x_sparse_live, a, b, c, d, e)

    preview_df = pd.DataFrame(
        {
            "log10_conc": x_sparse_live,
            "conc": conc_sparse_live,
            "response_ref": y_sparse_live,
        }
    )
    st.dataframe(preview_df, use_container_width=True, height=320)