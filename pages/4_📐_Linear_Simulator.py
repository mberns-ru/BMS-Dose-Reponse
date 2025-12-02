# 5_📈_Linear_Quantification.py

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

import dose_response as dp  # uses generate_log_conc for dilution grid


# ===================== Auto-populate from Upload Page =====================
# If the user uploaded data in the Upload Page and selected Linear model,
# store it in session_state["model_input"] with columns: ['Well','log10(conc)','conc','response']

uploaded_df = st.session_state.get("model_input", None)
use_uploaded_data = uploaded_df is not None

if use_uploaded_data:
    st.info("Using data uploaded from Upload Page for Linear model")

    # Try to pick numeric columns automatically
    numeric_cols = uploaded_df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Uploaded data does not have enough numeric columns for Linear model")
    else:
        # Assume first column is X (log10 conc), second column is Y (response)
        x_sparse_live = uploaded_df[numeric_cols[0]].astype(float).to_numpy()
        y_sparse_live_lin = uploaded_df[numeric_cols[1]].astype(float).to_numpy()

        # Interpolate for dense points
        from scipy.interpolate import interp1d
        x_dense_live = np.linspace(x_sparse_live.min(), x_sparse_live.max(), 50)
        interp_fn = interp1d(x_sparse_live, y_sparse_live_lin, kind="linear", fill_value="extrapolate")
        y_dense_base = interp_fn(x_dense_live)

        # Compute approximate slope/intercept for plotting
        m = (y_sparse_live_lin[-1] - y_sparse_live_lin[0]) / (x_sparse_live[-1] - x_sparse_live[0])
        b = y_sparse_live_lin[0] - m * x_sparse_live[0]


# ===================== Check for uploaded data =========================
uploaded_df = st.session_state.get('model_input', None)
use_uploaded_data = uploaded_df is not None


# ===================== Sidebar Logo (same as other pages) ===================

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
    # Fail silently if logo is missing in dev envs
    pass


# ===================== Page config =========================================

st.set_page_config(
    page_title="Linear Quantification Tool",
    layout="wide",
)

st.title("Linear Quantification Tool")

with st.expander("How to use this tool", expanded=False):
    st.markdown(
        r"""
This page mirrors the **4PL Quantification Tool**, but uses a **linear model**
for the dose–response relationship in log₁₀(concentration) space.

**Model**

We assume:

\[
y = m \cdot \log_{10}(x) + b
\]

- \(m\) – slope (change in response per log₁₀ unit change in concentration)  
- \(b\) – intercept (response when \(\log_{10}(x) = 0\), i.e., \(x = 1\))

**Relative potency (RP)**  

For each RP value, the sample curve is a **scaled version** of the base line:

\[
y_\text{RP}(x) = \text{RP} \cdot (m \cdot \log_{10}(x) + b)
\]

This gives a family of parallel-like lines that can be used for simple
parallel-line bioassay visualizations.

**Edge cases**

The edge-case panel shows all min/max combinations of \(m\) and \(b\) so you
can see how extreme parameter values affect the linear response.

**Add curve**

Enter a label for the base line. When you click **Add curve**, the base line
and all RP variants are "locked": their 8 dilution points are saved and
displayed alongside future curves.
"""
    )


# ===================== Session State / Defaults (namespaced) ===============

DEFAULTS_LIN = {
    # Parameter ranges (used to compute central line)
    "m_min_lin": 0.2,
    "m_max_lin": 1.0,
    "b_min_lin": 0.0,
    "b_max_lin": 0.2,

    # Dilution controls
    "top_conc_lin": 10**2,
    "even_dil_factor_lin": 10**0.5,
    "dilution_str_lin": "",

    # App state
    "curves_lin": [],
    "next_label_idx_lin": 1,

    # RP text input
    "rps_str_lin": "",
}

for k, v in DEFAULTS_LIN.items():
    st.session_state.setdefault(k, v)


# ===================== Helpers =============================================

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


# ===================== Linear model in log10(x) space ======================

def linear_logx(x_log10, m, b):
    """Simple linear response in log10(concentration) space."""
    return m * x_log10 + b


def compute_curve_linear(m, b, x_log10):
    return linear_logx(x_log10, m, b)


# ===================== Data model for saved curves =========================

def curves_to_dataframe(curves):
    rows = []
    for cv in curves:
        grid = cv.get("grid", {})
        rows.append({
            "label": cv["label"],
            "m": cv["m"],
            "b": cv["b"],
            "rp": cv.get("rp"),
            "top_conc": grid.get("top_conc"),
            "even_factor": grid.get("even_factor"),
            "custom_factors": _list_to_str(grid.get("custom_factors", [])),
            "x_log10_points": _list_to_str(cv.get("x_log10_sparse", [])),
            "conc_points": _list_to_str(cv.get("conc_sparse", [])),
            "y_points": _list_to_str(cv.get("y_sparse", [])),
        })
    return pd.DataFrame(rows)


def _lock_curve(label, m, b, rp=None, grid=None):
    """
    Persist an 8-point linear curve using the current dilution grid.

    For RP ≠ 1, store the scaled line y_RP = RP * (m x + b).
    """
    rp_eff = 1.0 if rp in (None, 0) else float(rp)

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

    conc_sparse = (10 ** x_sparse_locked).astype(float)
    y_base = compute_curve_linear(m, b, x_sparse_locked)
    y_sparse_locked = rp_eff * y_base

    entry = {
        "label": label,
        "m": float(m),
        "b": float(b),
        "rp": rp_eff,
        "grid": {
            "top_conc": float(top_conc),
            "even_factor": float(even_factor),
            "custom_factors": list(custom_factors),
        },
        "x_log10_sparse": [float(v) for v in x_sparse_locked],
        "conc_sparse": [float(v) for v in conc_sparse],
        "y_sparse": [float(v) for v in y_sparse_locked],
    }
    st.session_state["curves_lin"].append(entry)


# ===================== Layout: Row 1 (dilution vs graph) ===================

left_panel, graph_col = st.columns([1.15, 1.85], gap="large")

with left_panel:
    st.subheader("Dilution series")

    with st.expander("What is a dilution series?", expanded=False):
        st.markdown(
            "The **dilution factor** is how much each step is diluted from the "
            "previous one, creating a series of decreasing doses. "
            "For example, a factor of **3** means each well is 1/3 of the "
            "previous concentration. Providing exactly **7** custom factors "
            "overrides the even factor."
        )

    st.number_input(
        "Top concentration",
        min_value=1e-12,
        max_value=1e12,
        value=float(st.session_state["top_conc_lin"]),
        step=1.0,
        format="%.6g",
        key="top_conc_lin",
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
        value=float(st.session_state["even_dil_factor_lin"]),
        step=0.01,
        format="%.6g",
        key="even_dil_factor_lin",
        help=(
            "Factor between adjacent wells (e.g., 2 for 2-fold, 3 for 3-fold). "
            "√10 ≈ 3.162 is common for dose–response assays."
        ),
    )

    st.text_input(
        "Custom 7 dilution factors "
        "(override even factor if exactly 7 numbers)",
        key="dilution_str_lin",
        value=st.session_state["dilution_str_lin"],
        placeholder="e.g., 3.162 3.162 3.162 3.162 3.162 3.162 3.162",
        help=(
            "Provide 7 step-wise multipliers (high → low). "
            "If blank or not exactly 7 values, the even factor is used."
        ),
    )

    custom_factors = _parse_list(st.session_state["dilution_str_lin"])
    if len(custom_factors) == 0:
        st.caption("Using even dilution factor.")
        custom_factors = []
    elif len(custom_factors) == 7:
        st.caption(f"Using custom factors: {custom_factors}")
    else:
        st.warning(
            f"Provide exactly 7 factors (got {len(custom_factors)}). "
            "Falling back to even factor."
        )
        custom_factors = []


with graph_col:
    st.subheader("Dose–Response Curves (Linear)")
    plot_placeholder = st.empty()


# ===================== Row 2: parameter ranges + graph =====================

with left_panel:
    st.markdown("### Parameter ranges (min/max)")

    with st.expander("What do m and b mean?", expanded=False):
        st.markdown(
            r"""
**m** – Slope of the line in log₁₀(concentration) space  
**b** – Intercept (response when log₁₀(concentration) = 0, i.e., \(x = 1\))

The central line in the main graph uses the midpoints of these ranges.
"""
        )

    # Ensure defaults present
    for k, v in {
        "m_min_lin": -1.0,
        "m_max_lin": 1.0,
        "b_min_lin": 0.0,
        "b_max_lin": 1.0,
    }.items():
        st.session_state.setdefault(k, v)

    colParam, colMin, colMax = st.columns([0.6, 1, 1])
    with colParam:
        st.markdown("**Parameter**")
    with colMin:
        st.markdown("**Min**")
    with colMax:
        st.markdown("**Max**")

    def row(label, k_min, k_max, min_cfg, max_cfg):
        cA, cMin, cMax = st.columns([0.6, 1, 1])
        with cA:
            st.markdown(label)
        with cMin:
            st.number_input(
                "", key=k_min, label_visibility="collapsed", **min_cfg
            )
        with cMax:
            st.number_input(
                "", key=k_max, label_visibility="collapsed", **max_cfg
            )

    row(
        "m (slope)",
        "m_min_lin",
        "m_max_lin",
        {"min_value": -10.0, "max_value": 10.0, "step": 0.01},
        {"min_value": -10.0, "max_value": 10.0, "step": 0.01},
    )
    row(
        "b (intercept)",
        "b_min_lin",
        "b_max_lin",
        {"min_value": -10.0, "max_value": 10.0, "step": 0.01},
        {"min_value": -10.0, "max_value": 10.0, "step": 0.01},
    )

    st.text_input(
        "Relative potencies (RP; comma/space separated)",
        key="rps_str_lin",
        value=st.session_state["rps_str_lin"],
        placeholder="e.g., 0.5, 1, 1.5, 2",
        help=(
            "RP scales the response: y_RP(x) = RP · (m · log10(x) + b). "
            "Use values like 0.5, 1, 2 to represent half / equal / double effect."
        ),
    )

    user_rps = _parse_rps(st.session_state["rps_str_lin"])
    if not user_rps:
        rps = [0.5, 1.0, 2.0]
        st.caption("Using default RP values: 0.5, 1.0 (reference), 2.0")
    else:
        rps = user_rps
        st.caption(f"Parsed RP values: {user_rps}")


# ===================== Main-graph parameters ===============================

m_min = float(st.session_state["m_min_lin"])
m_max = float(st.session_state["m_max_lin"])
b_min = float(st.session_state["b_min_lin"])
b_max = float(st.session_state["b_max_lin"])

m = (m_min + m_max) / 2.0
b = (b_min + b_max) / 2.0

top_conc = float(st.session_state["top_conc_lin"])
even_factor = float(st.session_state["even_dil_factor_lin"])

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


# ===================== Main figure ========================================

fig = go.Figure()

# Base line (RP=1.0)
base_color = rp_color_map.get(1.0, palette[0])
y_dense_base = compute_curve_linear(m, b, x_dense_live)
fig.add_scatter(
    x=x_dense_live,
    y=y_dense_base,
    mode="lines",
    name="Reference (RP=1.0)",
    line=dict(width=2, color=base_color),
    hovertemplate=(
        "log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Reference</extra>"
    ),
)
y_sparse_base = compute_curve_linear(m, b, x_sparse_live)
fig.add_scatter(
    x=x_sparse_live,
    y=y_sparse_base,
    mode="markers",
    marker=dict(size=7, color=base_color),
    showlegend=False,
    hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
)

# RP-scaled lines
for rp_val in rp_sorted:
    if rp_val == 1.0:
        continue
    color = rp_color_map[rp_val]
    y_dense = rp_val * y_dense_base
    fig.add_scatter(
        x=x_dense_live,
        y=y_dense,
        mode="lines",
        name=f"Sample (RP={rp_val:g})",
        line=dict(width=2, color=color),
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )
    y_sparse = rp_val * y_sparse_base
    fig.add_scatter(
        x=x_sparse_live,
        y=y_sparse,
        mode="markers",
        marker=dict(size=7, color=color),
        showlegend=False,
        hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>",
    )

# Locked curves
locked_start_index = 6
for idx, cv in enumerate(st.session_state["curves_lin"]):
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

    y_base_locked = compute_curve_linear(cv["m"], cv["b"], x_dense_locked)
    y_locked_line = cv["rp"] * y_base_locked

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
    title="Dose–Response Curves (Linear in log₁₀ concentration)",
    xaxis_title="log₁₀(concentration)",
    yaxis_title="Response",
    legend_title=None,
    margin=dict(l=10, r=10, t=50, b=10),
)

with graph_col:
    plot_placeholder.plotly_chart(
        fig,
        config={"responsive": True, "displayModeBar": True},
        use_container_width=True,
    )

    default_label = f"Curve {st.session_state['next_label_idx_lin']}"
    label = st.text_input(
        "Label for base line",
        value=default_label,
        help="Base line name; RP variants add ' (RP=...)'.",
        key="label_input_lin",
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Add curve", type="primary", key="btn_add_curve_lin"):
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
                _lock_curve(label=lbl, m=m, b=b, rp=rp_val, grid=grid)

            st.session_state["next_label_idx_lin"] += 1
            st.success(f"Saved '{label}' with {len(base_rps)} line(s).")
            _rerun()

    with col_btn2:
        if st.button("Clear all saved curves", key="btn_clear_curves_lin"):
            st.session_state["curves_lin"] = []
            st.session_state["next_label_idx_lin"] = 1
            st.info("Cleared all saved lines.")
            _rerun()

# ===================== Edge-case subplots (m, b extremes) ==================

st.markdown("### Edge cases: all min/max combinations of slope (m) and intercept (b)")

with st.expander("What are edge cases?", expanded=False):
    st.markdown(
        "These plots show all the edge-case combinations of **m** and **b**. "
        "Each panel represents a minimum/maximum setting so you can see how "
        "extreme values change the linear response pattern."
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

m_min = float(st.session_state["m_min_lin"])
m_max = float(st.session_state["m_max_lin"])
b_min = float(st.session_state["b_min_lin"])
b_max = float(st.session_state["b_max_lin"])

choices_edge = list(itertools.product([m_min, m_max], [b_min, b_max]))

edge_fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.06,
    vertical_spacing=0.10,
    subplot_titles=[f"m={mm:.3g}, b={bb:.3g}" for (mm, bb) in choices_edge],
)

for idx, (mm, bb) in enumerate(choices_edge, start=1):
    r = ((idx - 1) // 2) + 1
    c_ = ((idx - 1) % 2) + 1
    for rp_val in rp_sorted if rp_sorted else [1.0]:
        color = rp_color_map.get(
            rp_val, palette[0] if rp_val == 1.0 else palette[1]
        )
        y_dense = rp_val * compute_curve_linear(mm, bb, x_dense_edge)
        edge_fig.add_scatter(
            x=x_dense_edge,
            y=y_dense,
            mode="lines",
            line=dict(width=2, color=color),
            showlegend=False,
            row=r,
            col=c_,
        )
        y_sparse = rp_val * compute_curve_linear(mm, bb, x_sparse_edge)
        edge_fig.add_scatter(
            x=x_sparse_edge,
            y=y_sparse,
            mode="markers",
            marker=dict(size=5, color=color),
            showlegend=False,
            row=r,
            col=c_,
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
    text="response",
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

# ======= Row: Saved lines (left) | Dilution preview (right) =======
col_saved_lin, col_preview_lin = st.columns([1.15, 1.85], gap="large")

with col_saved_lin:
    st.subheader("Saved curves")
    df_saved_lin = curves_to_dataframe(st.session_state["curves_lin"])
    if not df_saved_lin.empty:
        st.dataframe(df_saved_lin, use_container_width=True, height=320)
        st.download_button(
            "Export Saved Curves CSV",
            data=df_saved_lin.to_csv(index=False).encode("utf-8"),
            file_name="dose_response_curves_linear.csv",
            mime="text/csv",
            key="btn_export_saved_csv_lin",
        )
    else:
        st.info("No saved curves yet.")

with col_preview_lin:
    st.subheader("Dilution preview (current settings)")
    conc_sparse_live_lin = (10 ** x_sparse_live).astype(float)
    y_sparse_live_lin = compute_curve_linear(m, b, x_sparse_live)

    df_preview_lin = pd.DataFrame({
        "Well": np.arange(1, len(x_sparse_live) + 1, dtype=int),
        "log10(conc)": x_sparse_live,
        "conc": conc_sparse_live_lin,
        "response (current)": y_sparse_live_lin,
    })
    dfp_lin = df_preview_lin.copy()
    dfp_lin["log10(conc)"] = dfp_lin["log10(conc)"].map(lambda v: f"{v:.6f}")
    dfp_lin["conc"] = dfp_lin["conc"].map(lambda v: f"{v:.6g}")
    dfp_lin["response (current)"] = dfp_lin["response (current)"].map(
        lambda v: f"{v:.4f}"
    )



    st.dataframe(dfp_lin, use_container_width=True, height=320)

    st.download_button(
        "Export Dilution Preview CSV",
        data=df_preview_lin.to_csv(index=False).encode("utf-8"),
        file_name="dilution_preview_linear.csv",
        mime="text/csv",
        key="btn_export_preview_csv_lin",
    )

    if custom_factors and len(custom_factors) == 7:
        st.caption(
            "Using custom 7 dilution factors; even factor "
            f"(for dense grid) = {even_factor:.6g}"
        )
    else:
        st.caption(f"Using even dilution factor: {even_factor:.6g}")

st.markdown("---")
st.markdown("**References**")
st.markdown(
    """
- United States Pharmacopeial Convention. (2012).  
  *Analysis of biological assays (USP 35 General Chapter 〈1034〉).*  
    """
)
