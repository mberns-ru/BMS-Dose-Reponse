import streamlit as st
import pandas as pd
from PIL import Image

# --- Logo ---
LOGO_PATH = "C:/Users/tanvi/OneDrive/Desktop/Python/Logo.jpg"
logo = Image.open(LOGO_PATH)
st.image(logo, width=200)

st.title("Upload Data for Dose-Response Analysis")

# --- Info Section ---
with st.expander("How to use this feature"):
    st.markdown("""
    This page allows you to upload your experimental data (Excel or CSV) for dose-response analysis.  

    **How to use the summary statistics options:**

    - **Average**: Computes the mean of all replicates for each sample.  
      Use this when your replicates are consistent and you want a single representative value per sample.

    - **Min/Max**: Shows two rows – one with the minimum and one with the maximum value across replicates.  
      Use this when your replicates vary significantly and you want to capture the full range of values.

    After selecting your preferred summary statistic, the processed data will be displayed below for preview.  
    You can then use these values in downstream analyses, such as fitting 4PL, 5PL, 2PL, or linear models.
    """)

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "xls", "csv"])

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
    st.write("Preview of your data:")
    st.dataframe(df)

    # --- Min/Max vs Average ---
    st.subheader("Select Summary Statistics")
    summary_option = st.radio(
        "Choose whether to use Average or Min/Max values for downstream analysis",
        ("Average", "Min/Max")
    )

    numeric_cols = df.select_dtypes(include='number').columns

    if summary_option == "Average":
        df_summary = pd.DataFrame([df[numeric_cols].mean()])
        st.write("Summary (Average) Data:")
        st.dataframe(df_summary)

    else:  # Min/Max
        min_row = df[numeric_cols].min()
        max_row = df[numeric_cols].max()
        df_summary = pd.DataFrame([min_row, max_row])
        df_summary.index = ['Min', 'Max']
        st.write("Summary (Min/Max) Data:")
        st.dataframe(df_summary)

    # --- Model Guidance ---
    st.subheader("Which Dose-Response Model to Use?")
    st.info("""
    - **4PL**: Most common, symmetric sigmoidal curves. Use if your data has a typical S-shape.
    - **5PL**: Adds asymmetry, good for curves that are not symmetric.
    - **2PL**: Simpler, only slope and EC50. Use for basic dose-response data.
    - **Linear**: Use if your data shows a linear trend rather than sigmoidal. For linear data, summarize across replicates using Average or Min/Max per dose.
    """)

