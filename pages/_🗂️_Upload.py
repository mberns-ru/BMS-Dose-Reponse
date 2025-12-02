import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import os

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

    # --- Model Guidance ---
    st.subheader("Which Dose-Response Model to Use?")
    st.info("""
    - **4PL**: Symmetric sigmoidal curves.
    - **5PL**: Adds asymmetry.
    - **2PL**: Simpler, slope + EC50.
    - **Linear**: Linear trend in log10(concentration) space.
    """)

# --- Optional Model Selection (only after upload) ---
if df is not None:
    st.subheader("Select Model for Analysis")
    model_choice = st.selectbox(
        "Choose the dose-response model to use", 
        ["Linear", "4PL", "5PL", "2PL"]
    )

    # Save selected model to session_state
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = None
    st.session_state["selected_model"] = model_choice

    # --- Navigation Message ---
    if model_choice:
        st.info(f"Navigate to the {model_choice} page from the sidebar.")

