import streamlit as st
from PIL import Image
import base64
import io

# ===================== Page config =====================
st.set_page_config(
    page_title="Dose–Response Curve Simulator",
    layout="wide",
)

# ===================== Logo: load & inject into sidebar =====================
LOGO_PATH = "graphics/Logo.jpg"

# Load the image once
logo = Image.open(LOGO_PATH)

# Encode as base64 so we can insert it via CSS above the nav
buffer = io.BytesIO()
logo.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode()

# Put logo at the very top of the sidebar, above the page navigation
st.markdown(
    f"""
    <style>
        [data-testid="stSidebarNav"]::before {{
            content: "";
            display: block;
            margin: 0 auto 1rem;
            height: 200px;
            background-image: url("data:image/png;base64,{img_b64}");
            background-repeat: no-repeat;
            background-position: center center;
            background-size: contain;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.success("Select a page above ☝️")

# ===================== Home page header =====================

header_logo_col, header_text_col = st.columns([1, 3])

with header_logo_col:
    st.markdown(
        f"""
        <div style="
            height: 300px;
            width: 300px;
            background-image: url('data:image/png;base64,{img_b64}');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            border-radius: 12px;
        ">
        </div>
        """,
        unsafe_allow_html=True
    )
with header_text_col:
    st.title(" Dose–Response Curve Simulator")
    st.markdown("""
    ### Welcome!
    Use the sidebar to navigate between pages.

    - **Home**: Overview of the simulator and usage notes  
    - **4PL Simulator**: Interactive tool to visualize and fit 4-parameter logistic (4PL) curves
    - **2PL Simulator**: Interactive tool to visualize and fit 2-parameter logistic (2PL) curves
    """)


st.markdown("---")
st.markdown("### 📋 Usage Notes")
st.markdown(
    """
**Set the dilution series**

Begin by entering the **top concentration** and the **dilution factor**. 
These values define the concentration levels used to generate the dose–response curves.

**Specify parameter ranges**

Each model includes a set of parameters that control curve shape and behavior. 
Specify the minimum and maximum values for these parameters to explore their full range and generate the corresponding dose–response curves.

**Run and save experiment**

After entering your first set of inputs, the tool will generate the corresponding dose-response curves. 
You may save this run, then enter additional configurations to compare multiple rounds of experiments.

**Use the Dilution Scheme Recommendations (Optional)** 

Instead of manually entering dilution values, you may choose to use the Dilution Scheme Recommender, which proposes a scheme based on your parameter ranges. 

**Review and export Result**

All experiment runs are logged in the table at the bottom of the page. You may export the full table as a CSV file for further analysis or documentation. 

"""
)



# ==============================================================
#                     MODEL COMPARISON TABLES
# ==============================================================

st.markdown("## 📌 Model Comparison")

# ----------- Table 1: Models & Parameters ----------- #
st.markdown("""
### 1. Models & Parameters

| Model | Parameters | Shape |
|-------|------------|--------|
| **Linear Model** | slope, intercept | Straight line |
| **2PL (simple logistic)** | slope, EC₅₀ | Symmetric S-curve |
| **4PL** | lower, upper, slope, EC₅₀ | Symmetric S-curve (more flexible) |
| **5PL** | lower, upper, slope, EC₅₀, asymmetry (E) | Asymmetric S-curve |

*Fitting difficulty increases from **Linear → 2PL → 4PL → 5PL**.*
""")

# ----------- Table 2: Usage ----------- #
st.markdown("""
### 2. Model Usage

| Model | Advantages | Weaknesses |
|--------|------------|-------------|
| **Linear Model** | Simple linear trends | Cannot describe biological dose response |
| **2PL** | Good for normalized data | Cannot fit lower/upper asymptotes |
| **4PL** | Standard bioassay model | Cannot model asymmetry |
| **5PL** | Captures asymmetric curves | Needs more data; can be unstable |
""")

# ----------- Table 3: Interpretation ----------- #
st.markdown("""
### 3. Interpretation

| Model | EC₅₀? | Lower/Upper? | Asymmetry? |
|--------|-------|----------------|-------------|
| **Linear Model** | No | No | No |
| **2PL** | Yes | No (fixed) | No |
| **4PL** | Yes | Yes | No |
| **5PL** | Yes | Yes | Yes |
""")

st.markdown("---")

st.markdown("""
### Need help?
Navigate to **4PL Quantification** in the sidebar to interactively explore dilution schemes,  
response curves, and assay behavior.
""")
