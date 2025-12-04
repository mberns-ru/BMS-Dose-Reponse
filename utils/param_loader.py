# utils/param_loader.py
import pandas as pd
import streamlit as st

def load_param_bounds():
    """
    Produce {'a':(lo,hi), 'b':(lo,hi), 'c':(lo,hi), 'd':(lo,hi)} from either:
      1) st.session_state['param_bounds'] if already set, or
      2) st.session_state['model_input'] using Min/Max rows (preferred), or
      3) first 4 numeric columns of model_input (fallback).
    Returns None if nothing usable.
    """
    # 1) Honor already-computed bounds
    pb = st.session_state.get("param_bounds")
    if isinstance(pb, dict):
        out = _normalize(pb)
        if out:
            return out

    # 2) Try to derive from model_input
    df = st.session_state.get("model_input")
    if df is None or getattr(df, "empty", False):
        return None

    df2 = df.copy()
    df2.columns = [str(c).strip().lower() for c in df2.columns]

    # 2a) Prefer Min/Max rows if 'sample' exists
    if "sample" in df2.columns:
        s = df2["sample"].astype(str).str.strip().str.lower()
        if "min" in s.values and "max" in s.values:
            rmin = df2.loc[s == "min"].iloc[0]
            rmax = df2.loc[s == "max"].iloc[0]

            # Use named a,b,c,d if present; else first 4 numeric columns (excluding 'sample')
            cols_named = [c for c in ["a", "b", "c", "d", "e"] if c in df2.columns] #added e
            if len(cols_named) >= 4:
                return _pairs_from_rows(rmin, rmax, cols_named)

            num_cols = [c for c in df2.columns
                        if c != "sample" and pd.api.types.is_numeric_dtype(df2[c])]
            if len(num_cols) >= 4:
                return _pairs_from_rows(rmin, rmax, num_cols[:5])

    # 2b) Fallback: per-column min/max across first 4 numeric columns
    num_cols = [c for c in df2.columns
                if c != "sample" and pd.api.types.is_numeric_dtype(df2[c])]
    if len(num_cols) >= 4:
        max_cols = num_cols[:5]
        mins = [float(df2[c].min()) for c in num_cols[:4]]
        maxs = [float(df2[c].max()) for c in num_cols[:4]]
        keys = ["a", "b", "c", "d", "e"]  
        return {keys[i]: (mins[i], maxs[i]) for i in range(len(max_cols))}

    return None


def _normalize(d: dict | None):
    if not isinstance(d, dict):
        return None
    out = {}
    try:
        for k in ["a", "b", "c", "d", "e"]:
            v = d.get(k) or d.get(k.upper())
            if v is None or len(v) != 2:
                continue
            lo, hi = float(v[0]), float(v[1])
            if lo > hi:
                lo, hi = hi, lo
            out[k] = (lo, hi)
    except Exception:
        return None
    return out


def _pairs_from_rows(rmin: pd.Series, rmax: pd.Series, cols: list[str]):
    try:
        pairs = []
        for c in cols[:5]:
            lo = float(rmin[c]); hi = float(rmax[c])
            if lo > hi:
                lo, hi = hi, lo
            pairs.append((lo, hi))
        keys = ["a", "b", "c", "d", "e"]
        return {keys[i]: pairs[i] for i in range(4)}
    except Exception:
        return None
