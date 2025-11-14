# dose_response.py
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

# ===== Core 4PL in log10(x) space ======================================
def four_param_logistic_logx(x_log10, a, b, c, d):
    return d + (a - d) / (1 + 10 ** (b * (x_log10 - np.log10(c))))

def compute_curve(a, b, c, d, x_log10):
    return four_param_logistic_logx(x_log10, a, b, c, d)

def generate_log_conc(
    top_conc=10**2,
    dil_factor=10**0.5,
    n_points=8,
    dense=False,
    dilution_factors=None,
):
    """
    Generate log10(conc) points for an n-point serial dilution.

    Modes:
      - Even dilution: specify `dil_factor` (applied n_points-1 times).
      - Custom 7 factors: pass `dilution_factors` list of length n_points-1.
      - dense=True: return a dense grid spanning the sparse range (high→low).
    """
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if dilution_factors is not None and len(dilution_factors) != (n_points - 1):
        raise ValueError(f"dilution_factors must have length {n_points - 1}")

    conc = [float(top_conc)]
    if dilution_factors is not None:
        for f in dilution_factors:
            conc.append(conc[-1] / float(f))
    else:
        for _ in range(n_points - 1):
            conc.append(conc[-1] / float(dil_factor))

    x_sparse = np.log10(np.array(conc, dtype=float))
    if dense:
        x_min, x_max = x_sparse.min(), x_sparse.max()
        return np.linspace(x_max, x_min, 300)  # high → low

    return x_sparse

# ===== Plot helpers =====================================================
def _plot_sparse_markers(a, b, c, d, x_sparse, fig, name="Current"):
    y_sparse = four_param_logistic_logx(x_sparse, a, b, c, d)
    fig.add_trace(
        go.Scatter(
            x=x_sparse, y=y_sparse,
            mode="markers",
            showlegend=False,
            marker=dict(size=7, color=qualitative.Plotly[0]),
            legendgroup=name,
            hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>"
        )
    )
    return fig

def plot_ref_curve(a, b, c, d, x_sparse, x_dense, fig):
    y_dense  = four_param_logistic_logx(x_dense, a, b, c, d)
    color = qualitative.Plotly[0]
    fig.add_trace(
        go.Scatter(
            x=x_dense, y=y_dense,
            mode="lines",
            name="Current Curve",
            line=dict(width=2, color=color),
            legendgroup="Current",
            hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra>Current</extra>"
        )
    )
    _plot_sparse_markers(a, b, c, d, x_sparse, fig, name="Current")
    return fig

def plot_sample_curve_smooth(a, b, c, d, x_dense, rp, fig):
    y_dense = four_param_logistic_logx(x_dense, a, b, c / rp, d)
    fig.add_trace(
        go.Scatter(
            x=x_dense, y=y_dense, mode='lines',
            name=f"Sample (RP={rp})",
            hovertemplate="log10(conc)=%{x:.3f}<br>Response=%{y:.3f}<extra></extra>"
        )
    )
    return fig

def plot_all_curves(
    a, b, c, d,
    top_conc=None,
    dil_factor=None,
    rps=None,
    dilution_factors=None,
    n_points=8
):
    if top_conc is None:
        top_conc = 10 ** 2
    if dil_factor is None:
        dil_factor = 10 ** (1 / 2)
    if rps is None:
        rps = []

    x_sparse = generate_log_conc(
        top_conc=top_conc,
        dil_factor=dilution_factors[0] if dilution_factors else dil_factor,
        n_points=n_points,
        dense=False,
        dilution_factors=dilution_factors,
    )
    x_dense  = generate_log_conc(
        top_conc=top_conc,
        dil_factor=dilution_factors[0] if dilution_factors else dil_factor,
        n_points=n_points,
        dense=True,
        dilution_factors=dilution_factors,
    )

    fig = go.Figure()
    plot_ref_curve(a, b, c, d, x_sparse, x_dense, fig)

    if rps:
        for rp in rps:
            fig = plot_sample_curve_smooth(a, b, c, d, x_dense, rp, fig)

    fig.update_layout(
        title="Dose-Response Curves",
        xaxis_title="Log Concentration",
        yaxis_title="Response",
        legend_title=None,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
