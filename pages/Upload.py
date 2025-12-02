import streamlit as st
import pandas as pd
from PIL import Image

# --- Logo ---

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


# --- Page Info Dropdown ---
with st.expander("What this page does and how to use it"):
    st.markdown("""
    This page allows you to upload your experimental data (Excel or CSV) for dose-response analysis.  

    **How to use the summary statistics options:**

    - **Average**: Computes the mean of all replicates for each sample.  
    - **Min/Max**: Shows minimum and maximum values across replicates.

    After selecting your preferred summary statistic, the processed data will be available for any dose-response model (Linear, 4PL, 5PL, 2PL).
    """)

st.title("Upload Data for Dose-Response Analysis")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

def _read_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include='number').columns
        return df[['sample'] + numeric_cols.tolist()] if 'sample' in df.columns else df[numeric_cols]
    return None

df = _read_uploaded_file(uploaded_file)

if df is not None:
    st.write("Preview of your uploaded data:")
    st.dataframe(df)

    # --- Min/Max vs Average ---
    st.subheader("Select Summary Statistics")
    summary_option = st.radio(
        "Choose summary statistic",
        ("Average", "Min/Max")
    )

    numeric_cols = df.select_dtypes(include='number').columns

    if summary_option == "Average":
        df_summary = pd.DataFrame([df[numeric_cols].mean()], columns=numeric_cols)
        df_summary.insert(0, 'sample', 'Average')
        st.write("Summary Data (Average across replicates):")
        st.dataframe(df_summary)
    else:
        df_summary = pd.DataFrame([df[numeric_cols].min(), df[numeric_cols].max()], columns=numeric_cols)
        df_summary.insert(0, 'sample', ['Min', 'Max'])
        st.write("Summary Data (Min and Max across replicates):")
        st.dataframe(df_summary)

    # --- Save to session_state for model pages ---
    st.session_state['model_input'] = df_summary.copy()

    # --- Model Guidance ---
    st.subheader("Which Dose-Response Model to Use?")
    st.info("""
    - **4PL**: Most common, symmetric sigmoidal curves. Use if your data has a typical S-shape.
    - **5PL**: Adds asymmetry, good for curves that are not symmetric.
    - **2PL**: Simpler, only slope and EC50. Use for basic dose-response data.
    - **Linear**: Use if your data shows a linear trend rather than sigmoidal.
    """)

    # --- Model Selection (optional) ---
    st.subheader("Select Model for Analysis")
    model_choice = st.selectbox(
        "Choose the dose-response model to use",
        ["Linear", "4PL", "5PL", "2PL"]
    )

    st.info(f"Selected model: {model_choice}. Go to the corresponding page to continue with these input values.")

