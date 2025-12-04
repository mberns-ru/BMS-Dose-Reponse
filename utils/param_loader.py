# utils/param_loader.py
import pandas as pd
import streamlit as st

# ----------------- Defaults -----------------
def _defaults_5pl():
    return {
        "a": (0.80, 1.20),
        "b": (-2.00, -0.50),
        "c": (0.10, 3.00),
        "d": (0.00, 0.20),
        "e": (0.80, 1.20),
    }

# ----------------- Normalize -----------------
def _normalize_bounds_5pl(bounds: dict | None):
    if not isinstance(bounds, dict):
        return None
    out = {}
    for k in ["a", "b", "c", "d", "e"]:
        pair = bounds.get(k) or bounds.get(k.upper())
        if not pair or len(pair) != 2:
            continue
        lo, hi = map(float, pair)
        if lo > hi:
            lo, hi = hi, lo
        out[k] = (lo, hi)
    if not {"a","b","c","d"}.issubset(out.keys()):
        return None
    return out

# ----------------- Extract Pairs from Rows -----------------
def _pairs_from_rows(rmin: pd.Series, rmax: pd.Series, cols: list[str]):
    out = {}
    keys = ["a", "b", "c", "d", "e"]
    for i, c in enumerate(cols[:5]):
        lo, hi = float(rmin[c]), float(rmax[c])
        if lo > hi:
            lo, hi = hi, lo
        out[keys[i]] = (lo, hi)
    # Fill missing e if less than 5 columns
    if len(cols) < 5:
        out["e"] = _defaults_5pl()["e"]
    return out

# ----------------- Load Param Bounds -----------------
def load_param_bounds():
    pb = st.session_state.get("param_bounds")
    if isinstance(pb, dict):
        norm = _normalize_bounds_5pl(pb)
        if norm:
            return norm

    df = st.session_state.get("model_input")
    if df is None or getattr(df, "empty", False):
        return None

    df2 = df.copy()
    df2.columns = [str(c).strip().lower() for c in df2.columns]

    # Prefer Min/Max rows if 'sample' exists
    if "sample" in df2.columns:
        s = df2["sample"].astype(str).str.strip().str.lower()
        if "min" in s.values and "max" in s.values:
            rmin = df2.loc[s=="min"].iloc[0]
            rmax = df2.loc[s=="max"].iloc[0]
            num_cols = [c for c in df2.columns if c != "sample" and pd.api.types.is_numeric_dtype(df2[c])]
            return _pairs_from_rows(rmin, rmax, num_cols[:5])

    # Fallback: min/max of first 4–5 numeric columns
    num_cols = [c for c in df2.columns if c != "sample" and pd.api.types.is_numeric_dtype(df2[c])]
    if len(num_cols) < 4:
        return None

    mins = [float(df2[c].min()) for c in num_cols[:5]]
    maxs = [float(df2[c].max()) for c in num_cols[:5]]
    out = {
        "a": (mins[0], maxs[0]),
        "b": (mins[1], maxs[1]),
        "c": (mins[2], maxs[2]),
        "d": (mins[3], maxs[3]),
    }
    if len(num_cols) >= 5:
        out["e"] = (mins[4], maxs[4])
    else:
        out["e"] = _defaults_5pl()["e"]
    return out
