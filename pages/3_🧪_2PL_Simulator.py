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

st.set_page_config(page_title="2PL Quantification Tool", layout="wide")

# ======= 2PL model (fixed lower=0, upper=1) ================================

A_FIXED = 1.0   # upper asymptote
D_FIXED = 0.0   # lower asymptote

def two_param_logistic_logx(x_log10, b, c, A=A_FIXED, D=D_FIXED):
    """
    Standard 2PL logistic in log10(x) space with fixed lower (D) and upper (A):
        y = D + (A - D) / (1 + 10^(b * (x_log10 - log10(c))))
    """
    return D + (A - D) / (1 + 10 ** (b * (x_log10 - np.log10(c))))

def compute_curve_2pl(b, c, x_log10):
    return two_param_logistic_logx(x_log10, b, c, A=A_FIXED, D=D_FIXED)

# ======= Title =============================================================
st.title("2PL Quantification Tool")

# ======= Session State / Defaults ==========================================
DEFAULTS = {
    # Ranges (min/max) for 2PL: only slope (B) and EC50 (C)
    "b_min": -2.0, "b_max": -0.5,
    "c_min": 0.1, "c_max": 3.0,

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
st.session_state.setdefault("apply_rec_factor_2pl", False)
st.session_state.setdefault("rec_factor_value_2pl", None)
st.session_state.setdefault("rec_df_2pl", None)

# ---- Apply any pending recommended factor BEFORE widgets are created ----
if st.session_state.get("apply_rec_factor_2pl", False):
    rec_val = st.session_state.get("rec_factor_value_2pl", None)
    if rec_val is not None:
        st.session_state["even_dil_factor"] = float(rec_val)
        st.session_state["dilution_str"] = ""  # ensure even factor is used
    st.session_state["apply_rec_factor_2pl"] = False

# ======= Tunable region thresholds for the recommender =====================
# For 2PL with A=1, D=0, the normalized response t is just y itself.
LINEAR_LOW, LINEAR_HIGH = 0.2, 0.8
LOWER_ANCHOR_MAX = 0.10
UPPER_ANCHOR_MIN = 0.90

# ======= Helpers ===========================================================
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

# ======= Data model for saved curves ======================================
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
    y_sparse_locked = compute_curve_2pl(b, c_eff, x_sparse_locked)

    entry = {
        "label": label,
        "b": float(b),
        "c": float(c_eff),
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

# ======= Recommender internals (2PL: only B and C vary) ====================
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

    # 4 edge combos for 2PL
    combos = list(itertools.product([b_min, b_max], [c_min, c_max]))

    worst_case = {"lower_min": 8, "linear_min": 8, "upper_min": 8}

    for bb, cc in combos:
        rows = []
        for rp in (rps_list or [1.0]):
            c_eff = cc / rp
            y = compute_curve_2pl(bb, c_eff, x_sparse)
            t = y  # already normalized between ~0 and 1 for 2PL

            lower = int(np.sum(t <= LOWER_ANCHOR_MAX))
            linear = int(np.sum((t >= LINEAR_LOW) & (t <= LINEAR_HIGH)))
            upper = int(np.sum(t >= UPPER_ANCHOR_MIN))

            rows.append(
                {"rp": float(rp), "lower": lower, "linear": linear, "upper": upper}
            )

        # For this combo, min across RPs
        lower_min_combo = min(r["lower"] for r in rows)
        linear_min_combo = min(r["linear"] for r in rows)
        upper_min_combo = min(r["upper"] for r in rows)

        # Update global worst case across combos
        worst_case["lower_min"] = min(worst_case["lower_min"], lower_min_combo)
        worst_case["linear_min"] = min(worst_case["linear_min"], linear_min_combo)
        worst_case["upper_min"] = min(worst_case["upper_min"], upper_min_combo)

    min_ok = (
        worst_case["linear_min"] >= 2    # minimum requirement
        and worst_case["lower_min"] >= 1
        and worst_case["upper_min"] >= 1
    )

    # Preferred: 3 linear + 2 anchors on each side
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

# ======= RULE ==============================================================
st.markdown("---")

# ======= Row 1: Dilution series (left) | Graph (right) =====================
left_panel, graph_col = st.columns([1.15, 1.85], gap="large")

with left_panel:
    st.subheader("Dilution series")
    st.number_input(
        "Top concentration",
        min_value=1e-12, max_value=1e12,
        step=1.0, format="%.6g", key="top_conc"
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
    st.subheader("Dose-Response Curves (2PL)")
    plot_placeholder = st.empty()

# ======= Row 2: Parameter ranges (min/max) | Graph continues (right) =======
with left_panel:
    st.markdown("### Parameter ranges (2PL)")

    st.markdown(
        "_This page assumes a fixed lower asymptote **D = 0** and upper "
        "asymptote **A = 1** (normalized response). Only the slope (B) and "
        "EC₅₀ (C) vary._"
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.number_input("b_min (slope)", -10.0, 10.0, step=0.01, key="b_min")
        st.number_input("c_min (EC₅₀)", 1e-6, 1e6, step=0.01, format="%.6g", key="c_min")
    with r1c2:
        st.number_input("b_max (slope)", -10.0, 10.0, step=0.01, key="b_max")
        st.number_input("c_max (EC₅₀)", 1e-6, 1e6, step=0.01, format="%.6g", key="c_max")

    # ---- Relative potencies (applied to C only) ----
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

# ======= Compute main-graph parameters as averages of min/max =============
b_min, b_max = float(st.session_state["b_min"]), float(st.session_state["b_max"])
c_min, c_max = float(st.session_state["c_min"]), float(st.session_state["c_max"])

# Averages for main graph
b = (b_min + b_max) / 2.0
c = (c_min + c_max) / 2.0

top_conc = float(st.session_state["top_conc"])
even_factor = float(st.session_state["even_dil_factor"])

# ======= Build live grids once ============================================
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

# ======= Color map for RP curves ==========================================
palette = qualitative.Plotly  # length >= 10
rp_sorted = sorted(set(rps))
rp_color_map = {}
color_cursor = 1
if 1.0 in rp_sorted:
    rp_color_map[1.0] = palette[0]
for rp in rp_sorted:
    if rp == 1.0:
        continue
    rp_color_map[rp] = palette[color_cursor % len(palette)]
    color_cursor += 1

# ======= Make live plot (main) ============================================
fig = go.Figure()

# Reference curve (RP=1.0)
base_color = rp_color_map.get(1.0, palette[0])
y_dense_ref = compute_curve_2pl(b, c, x_dense_live)
fig.add_scatter(
    x=x_dense_live, y=y_dense_ref, mode="lines",
    name="Reference (RP=1.0)",
    line=dict(width=2, color=base_color),
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Reference</extra>",
)
y_sparse_ref = compute_curve_2pl(b, c, x_sparse_live)
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
    y_dense = compute_curve_2pl(b, c / rp, x_dense_live)
    fig.add_scatter(
        x=x_dense_live, y=y_dense, mode="lines",
        name=f"Sample (RP={rp:g})",
        line=dict(width=2, color=color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )
    y_sparse = compute_curve_2pl(b, c / rp, x_sparse_live)
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
    y_locked_line = compute_curve_2pl(cv["b"], cv["c"], x_dense_locked)
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
    title="Dose-Response Curves (2PL)",
    xaxis_title="Log Concentration",
    yaxis_title="Response (normalized)",
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
                "custom_factors": list(custom_factors) if len(custom_factors) == 7 else [],
            }
            for rp in base_rps:
                lbl = label if rp == 1.0 else f"{label} (RP={rp:g})"
                _lock_curve(label=lbl, b=b, c=c, rp=rp, grid=grid)
            st.session_state["next_label_idx"] += 1
            st.success(f"Saved '{label}' with {len(base_rps)} curve(s).")
            _rerun()
    with col_btn2:
        if st.button("Clear all saved curves", key="btn_clear_curves_2pl"):
            st.session_state["curves"] = []
            st.session_state["next_label_idx"] = 1
            st.info("Cleared all saved curves.")
            _rerun()

    # --- Recommender lives under the graph, in this same column ---
    st.markdown("---")
    st.markdown("### Dilution scheme recommender (from B, C ranges)")

    with st.expander("How this suggestion works (2PL)", expanded=False):
        st.markdown(
            f"""
This tool recommends **even dilution factor(s)** using the current **2PL** parameter
ranges (slope **B** and EC₅₀ **C**) and the selected relative potencies (RPs).

For each candidate factor (log-spaced between **1.2** and **10**), it:

1. Builds an 8-point dilution series.  
2. Evaluates **4 edge combinations** of (B, C):  
   (b_min, c_min), (b_min, c_max), (b_max, c_min), (b_max, c_max).  
3. For each combination & RP, classifies points using the normalized response
   (which, for 2PL with A=1, D=0, is just **y**):

   - Lower anchor: y ≤ {LOWER_ANCHOR_MAX:.2f}  
   - Linear region: {LINEAR_LOW:.2f} ≤ y ≤ {LINEAR_HIGH:.2f}  
   - Upper anchor: y ≥ {UPPER_ANCHOR_MIN:.2f}  

A factor is considered **acceptable** if in the *worst case*:

- ≥ 2 points lie in the linear band, and  
- ≥ 1 point lies in **each** anchor.

Among candidates, the algorithm prefers:

1. Those meeting the minimum rule  
2. Then those with stronger anchor coverage (2+ per anchor)  
3. Then higher total coverage score  
4. Then smaller dilution factors (finer spacing)
"""
        )

    n_cand = st.slider(
        "Search resolution (# of candidate factors between 1.2 and 10, log-spaced)",
        min_value=20,
        max_value=200,
        value=80,
        step=10,
        help="More candidates = more precise but slower.",
        key="n_cand_recommender_2pl",
    )

    # Run recommender on click, based on B,C ranges
    if st.button(
        "Recommend even dilution factors from B,C ranges",
        type="primary",
        key="btn_recommend_factors_2pl",
    ):
        b_min_local, b_max_local = float(st.session_state["b_min"]), float(st.session_state["b_max"])
        c_min_local, c_max_local = float(st.session_state["c_min"]), float(st.session_state["c_max"])

        factors = np.unique(
            np.round(
                np.logspace(np.log10(1.2), np.log10(10.0), int(n_cand)),
                6,
            )
        )

        rows = []
        for f_ in factors:
            res = _evaluate_factor_2pl(
                factor=f_,
                top_conc=top_conc,
                b_min=b_min_local,
                b_max=b_max_local,
                c_min=c_min_local,
                c_max=c_max_local,
                rps_list=rp_sorted,
            )
            rows.append(res)

        if rows:
            rec_df = pd.DataFrame(
                [
                    {
                        "factor": r["factor"],
                        "worst_linear": r["worst_linear"],
                        "worst_lower": r["worst_lower"],
                        "worst_upper": r["worst_upper"],
                        "meets_min": r["meets_min"],
                        "meets_preferred": r["meets_preferred"],
                        "score": r["score"],
                    }
                    for r in rows
                ]
            )
            # Ranking: meets_min, meets_preferred, score, factor ascending
            rec_df = rec_df.sort_values(
                by=["meets_min", "meets_preferred", "score", "factor"],
                ascending=[False, False, False, True],
            )
            st.session_state["rec_df_2pl"] = rec_df
        else:
            st.session_state["rec_df_2pl"] = None
            st.warning("No candidate factors were evaluated (check settings).")

    # Show table + picker if we have recommendations
    rec_df = st.session_state.get("rec_df_2pl", None)
    if rec_df is not None and not rec_df.empty:
        st.markdown("**Recommended even dilution factors** (top 20)")
        rec_df_disp = rec_df.copy()
        rec_df_disp["factor"] = rec_df_disp["factor"].map(lambda v: f"{v:.6g}")
        st.dataframe(rec_df_disp.head(20), use_container_width=True, height=320)

        st.download_button(
            "Download all recommendations (CSV)",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name="dilution_factor_recommendations_2pl.csv",
            mime="text/csv",
            key="btn_dl_rec_csv_2pl",
        )

        # Select + apply a recommended factor
        top_20 = rec_df.head(20)["factor"].tolist()
        st.markdown("#### Load a recommended factor into the current dilution settings")

        selected_factor = st.selectbox(
            "Choose a recommended factor to apply:",
            options=top_20,
            format_func=lambda v: f"{v:.6g}",
            key="rec_factor_select_2pl",
        )

        if st.button("Use selected factor", key="btn_use_rec_factor_2pl"):
            st.session_state["rec_factor_value_2pl"] = float(selected_factor)
            st.session_state["apply_rec_factor_2pl"] = True
            st.success(
                f"Will set even dilution factor to {selected_factor:.6g} and clear custom factors."
            )
            _rerun()
    else:
        st.info(
            "Click **Recommend even dilution factors from B,C ranges** to generate "
            "a ranked table of candidate dilution factors."
        )

# ======= RULE ==============================================================
st.markdown("---")

# ======= Edge-case subplots (2PL: B,C min/max combinations) ===============
st.markdown("### Edge cases (2PL): all min/max combinations of B, C")

x_sparse_edge = dp.generate_log_conc(
    top_conc=top_conc, dil_factor=even_factor, n_points=8, dense=False,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)
x_dense_edge = dp.generate_log_conc(
    top_conc=top_conc, dil_factor=even_factor, n_points=8, dense=True,
    dilution_factors=(custom_factors if len(custom_factors) == 7 else None),
)

# Refresh min/max in case user changed them
b_min, b_max = float(st.session_state["b_min"]), float(st.session_state["b_max"])
c_min, c_max = float(st.session_state["c_min"]), float(st.session_state["c_max"])

choices = list(itertools.product([b_min, b_max], [c_min, c_max]))  # 4 combos

edge_fig = make_subplots(
    rows=2, cols=2,
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.06, vertical_spacing=0.10,
    subplot_titles=[
        f"B={bb:.3g}, C={cc:.3g}"
        for (bb, cc) in choices
    ],
)

for idx, (bb, cc) in enumerate(choices, start=1):
    r = ((idx - 1) // 2) + 1
    k = ((idx - 1) % 2) + 1
    for rp in rp_sorted if rp_sorted else [1.0]:
        color = rp_color_map.get(rp, palette[0] if rp == 1.0 else palette[1])
        edge_fig.add_scatter(
            x=x_dense_edge,
            y=compute_curve_2pl(bb, cc / rp, x_dense_edge),
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False,
            row=r,
            col=k,
        )
        edge_fig.add_scatter(
            x=x_sparse_edge,
            y=compute_curve_2pl(bb, cc / rp, x_sparse_edge),
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
    x=0.5, y=-0.10, xref="paper", yref="paper",
    text="log10(conc)", showarrow=False, font=dict(size=14),
)
edge_fig.add_annotation(
    x=0.0, y=0.5, xref="paper", yref="paper",
    text="response (normalized)", showarrow=False, textangle=-90,
    xanchor="right", yanchor="middle", xshift=-40,
    font=dict(size=14),
)

st.plotly_chart(edge_fig, config={"responsive": True}, use_container_width=True)

# ======= RULE ==============================================================
st.markdown("---")

# ======= Row 3: Saved curves (left) | Dilution preview (right) ============
col_saved, col_preview = st.columns([1.15, 1.85], gap="large")

with col_saved:
    st.subheader("Saved curves (2PL)")
    df_saved = curves_to_dataframe(st.session_state["curves"])
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
        st.info("No saved curves yet.")

with col_preview:
    st.subheader("Dilution preview (current settings, 2PL)")
    conc_sparse_live = (10 ** x_sparse_live).astype(float)
    y_sparse_live = compute_curve_2pl(b, c, x_sparse_live)

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
        file_name="current_dilution_preview_2pl.csv",
        mime="text/csv",
        key="btn_export_preview_csv_2pl",
    )

    if len(custom_factors) == 7:
        st.caption(f"Using custom factors: {custom_factors}")
    else:
        st.caption(f"Using even dilution factor: {even_factor:.6g}")
