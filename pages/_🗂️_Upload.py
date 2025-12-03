import streamlit as st
import pandas as pd
from PIL import Image
import io, base64, os

# --- Logo ---
LOGO_PATH = "graphics/Logo.jpg"
try:
    if os.path.exists(LOGO_PATH):
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
    st.warning("Logo could not be loaded.")

# --- Helpers: detect model + route ---
def _norm_cols(cols):
    return [str(c).strip().lower() for c in cols]

def detect_model_from_df(df: pd.DataFrame) -> str | None:
    """
    Heuristics:
    1) If columns named A,B,C,D,(E) exist (case-insensitive), use their count.
    2) Else, if exactly 2/4/5 numeric parameter columns exist, map to 2PL/4PL/5PL.
    3) Else fallback to Linear.
    """
    if df is None or df.empty:
        return None

    cols_lower = _norm_cols(df.columns)
    named_cnt = sum(1 for p in ["a","b","c","d","e"] if p in cols_lower)

    if named_cnt == 5:
        return "5PL"
    if named_cnt == 4:
        return "4PL"
    if named_cnt == 2:
        return "2PL"

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cnt = len(numeric_cols)

    if cnt == 5:
        return "5PL"
    if cnt == 4:
        return "4PL"
    if cnt == 2:
        return "2PL"
    return "Linear"

def route_to_model(model: str):
    # Remember choice for other pages
    st.session_state["selected_model"] = model
    try:
        if model == "4PL":
            st.switch_page("pages/4PL.py")
        elif model == "5PL":
            st.switch_page("pages/5PL.py")
        elif model == "2PL":
            st.switch_page("pages/2PL.py")
        else:
            st.switch_page("pages/Linear.py")
    except Exception:
        st.info(f"Auto navigation not available here. Open **{model}** from the sidebar.")


def _find_page_target(preferred_key: str) -> tuple[str | None, str | None]:
    """
    Try to find the correct page by scanning ./pages for filenames that include the key (case-insensitive).
    Returns (path_form, title_form) candidates for st.switch_page().
    """
    try:
        pages_dir = "pages"
        if not os.path.isdir(pages_dir):
            return None, None

        files = os.listdir(pages_dir)
        # Case-insensitive match on filename (without extension) containing the key
        matches = []
        for f in files:
            if not f.lower().endswith(".py"):
                continue
            name_wo_ext = os.path.splitext(f)[0]
            if preferred_key.lower() in name_wo_ext.lower():
                matches.append(f)

        if not matches:
            return None, None

        # Pick the shortest matching filename (usually the cleanest)
        best = sorted(matches, key=len)[0]
        path_form = f"{pages_dir}/{best}"

        # Derive a title form Streamlit would show in sidebar:
        #  - strip a leading numeric/underscore prefix like "01_" or "1_"
        #  - title is the filename without extension
        title_form = os.path.splitext(best)[0]
        # trim leading numbering patterns like "01_", "1_", "001_"
        import re
        title_form = re.sub(r"^\d+_+", "", title_form)

        return path_form, title_form
    except Exception:
        return None, None





# --- Page Info ---
with st.expander("What this page does and how to use it"):
    st.markdown("""
    Upload your experimental data (Excel or CSV) containing replicate columns (A, B, C, D).  
    **Min/Max** across replicates will be calculated.  
    These processed parameter ranges will be automatically available in any dose-response model page (Linear, 4PL, 5PL, 2PL).
    """)

st.title("Upload Data for Dose-Response Analysis")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            import openpyxl
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include='number').columns
        if 'sample' in df.columns:
            return df[['sample'] + numeric_cols.tolist()]
        else:
            return df[numeric_cols]
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = read_uploaded_file(uploaded_file)

if df is not None:
    st.subheader("Preview of uploaded data:")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include='number').columns

    # --- Only Min/Max ---
    df_summary = pd.DataFrame([df[numeric_cols].min(), df[numeric_cols].max()], columns=numeric_cols)
    df_summary.insert(0, 'sample', ['Min', 'Max'])
    st.write("Summary Data (Min/Max across replicates):")
    st.dataframe(df_summary)

    # --- Save to session_state for model pages ---
    st.session_state['model_input'] = df_summary.copy()
    st.success("Parameter ranges are now ready for use in any model page.")


  # --- Decide target page but DO NOT auto-jump; show a button instead ---
detected = detect_model_from_df(df)

# Map to your real filenames (matches your screenshot, incl. emojis)
page_map = {
    "4PL":   "pages/2_🧪_4PL_Simulator.py",
    "5PL":   "pages/3_⚗️_5PL_Simulator.py",
    "Linear":"pages/4_📐_Linear_Simulator.py",
    "2PL":   "pages/5_🔬_2PL_Simulator.py",
}

if detected:
    st.info(f"Detected model: **{detected}**")

    target = page_map.get(detected)
    st.session_state["selected_model"] = detected

    # Primary action: click to navigate
    if st.button(f"Process & open {detected} page", type="primary"):
        if hasattr(st, "switch_page") and target:
            try:
                st.switch_page(target)
            except Exception:
                st.warning("Couldn’t navigate automatically. Use the link below.")
        else:
            st.warning("This Streamlit version doesn’t support st.switch_page. Use the link below.")

    # Always show a reliable link as a backup
    try:
        if target:
            st.page_link(target, label=f"Go to {detected}", icon="➡️")
    except Exception:
        # If page_link not available, at least tell the user the path
        st.caption(f"Open from sidebar: {target or detected}")
else:
    st.info("Could not auto-detect a model. Please choose one below.")





    # --- Model Guidance ---
    st.subheader("Which Dose-Response Model to Use?")
    st.info("""
    - **4PL**: Symmetric sigmoidal curves.
    - **5PL**: Adds asymmetry.
    - **2PL**: Simpler, slope + EC50.
    - **Linear**: Linear trend in log10(concentration) space.
    """)









