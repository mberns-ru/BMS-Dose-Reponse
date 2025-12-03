import streamlit as st
from PIL import Image
import base64
import io
import os
import base64

# ===================== Page config =====================
st.set_page_config(
    page_title="Dose–Response Curve Simulator",
    layout="wide",
)

# ===================== Sidebar logo (consistent with other pages) =====================

LOGO_PATH = "graphics/Logo.jpg"

img_b64 = None
if os.path.exists(LOGO_PATH):
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

st.sidebar.success("Select a page above ☝️")

# ===================== Header =====================

# 👉 Change this one number to resize the logo + shift layout
LOGO_SIZE = 260  # in pixels; try 220, 280, 320, etc.

st.markdown(
    f"""
    <style>
        .header-container {{
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;  /* allows wrap on smaller screens */
        }}

        .header-logo {{
            flex: 0 0 auto;
            width: {LOGO_SIZE}px;
            height: {LOGO_SIZE}px;
            border-radius: 16px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
        }}

        .header-text {{
            flex: 1 1 280px;  /* grows/shrinks with remaining space */
            min-width: 260px;
        }}

        .header-text h1 {{
            margin-bottom: 0.4rem;
        }}

        .header-text p {{
            margin-bottom: 0.6rem;
        }}

        .header-text ul {{
            margin-top: 0.2rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if img_b64 is not None:
    st.markdown(
        f"""
        <div class="header-container">
            <div class="header-logo"
                 style="background-image: url('data:image/png;base64,{img_b64}');">
            </div>
            <div class="header-text">
                <h1>Dose–Response Curve Simulator</h1>
                <p>
                    This app helps you design and explore
                    <strong>dose–response curves</strong> using
                    <strong>Linear</strong>, <strong>2PL</strong>,
                    <strong>4PL</strong>, and <strong>5PL</strong> models.
                </p>
                <p>Use the sidebar to navigate between pages:</p>
                <ul>
                    <li>🏠 <strong>Home</strong> – Overview &amp; quick-start guide</li>
                    <li>🗂️ <strong>Upload</strong> – Import experimental data and compute parameter ranges</li>
                    <li>🧪 <strong>4PL Simulator</strong> – Standard 4-parameter logistic curves</li>
                    <li>⚗️ <strong>5PL Simulator</strong> – 5-parameter logistic with asymmetry</li>
                    <li>🔬 <strong>2PL Simulator</strong> – Normalized 2-parameter logistic curves</li>
                    <li>📐 <strong>Linear Simulator</strong> – Linear model in log₁₀(concentration) space</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ===================== Quick workflow + model flow chart =====================

left_col, right_col = st.columns([1.1, 1.3])

with left_col:
    st.subheader("🧭 Quick workflow")

    st.markdown(
        """
        **Typical analysis flow**

        1. (_Optional_) **Upload data** on the *Upload* page  
           - Compute **min / max** parameter ranges from your experimental replicates.
        2. **Choose a model page** (Linear, 2PL, 4PL, or 5PL) from the sidebar.  
        3. **Set the dilution series**  
           - Top concentration  
           - Even or custom 7-step dilution factors  
        4. **Specify parameter ranges** for that model  
           - The main curve uses the **midpoint** of each range.  
        5. (**Optional**) Enter **relative potencies (RP)**  
           - Generates parallel sample curves (e.g. 40%, 100%, 160% of reference).  
        6. **Add & lock curves**  
           - Save curve families and compare them across different parameter sets.  
        7. **Review tables / edge cases**  
           - Inspect well-by-well values and edge-case panels to stress-test your design.
        """
    )

with right_col:
    st.subheader("📊 Model flow chart")

    st.markdown(
        """
        <div style="display:flex; flex-direction:column; gap:0.6rem;">

          <div style="
              padding:0.6rem 0.8rem;
              border-radius:12px;
              background-color:#f5f5f9;
              text-align:center;
              font-weight:600;
          ">
            Start
          </div>

          <div style="text-align:center; font-size:1.3rem;">⬇</div>

          <div style="
              padding:0.6rem 0.8rem;
              border-radius:12px;
              background-color:#eef6ff;
              text-align:center;
              font-weight:600;
          ">
            Upload experimental data (optional)
            <br><span style="font-weight:400;">&ldquo;Upload&rdquo; page</span>
          </div>

          <div style="text-align:center; font-size:1.3rem;">⬇</div>

          <div style="
              padding:0.6rem 0.8rem;
              border-radius:12px;
              background-color:#e8fdf5;
              text-align:center;
              font-weight:600;
          ">
            Choose model family
          </div>

          <div style="
              display:flex;
              flex-wrap:wrap;
              gap:0.6rem;
              justify-content:space-between;
              margin-top:0.4rem;
          ">
            <div style="
                flex:1 1 48%;
                padding:0.6rem;
                border-radius:10px;
                background-color:#ffffff;
                border:1px solid #d0d7ff;
                font-size:0.9rem;
            ">
              <b>📐 Linear</b><br/>
              Straight line in log₁₀(conc).  
              Useful sanity-check or rough trend.
            </div>
            <div style="
                flex:1 1 48%;
                padding:0.6rem;
                border-radius:10px;
                background-color:#ffffff;
                border:1px solid #d0d7ff;
                font-size:0.9rem;
            ">
              <b>🔬 2PL</b><br/>
              Normalized S-curve, fixed lower/upper.  
              Slope + EC₅₀ only.
            </div>
            <div style="
                flex:1 1 48%;
                padding:0.6rem;
                border-radius:10px;
                background-color:#ffffff;
                border:1px solid #d0d7ff;
                font-size:0.9rem;
            ">
              <b>🧪 4PL</b><br/>
              Full symmetric logistic: lower, upper, slope, EC₅₀.
            </div>
            <div style="
                flex:1 1 48%;
                padding:0.6rem;
                border-radius:10px;
                background-color:#ffffff;
                border:1px solid #d0d7ff;
                font-size:0.9rem;
            ">
              <b>⚗️ 5PL</b><br/>
              4PL + asymmetry parameter E for skewed curves.
            </div>
          </div>

          <div style="text-align:center; font-size:1.3rem; margin-top:0.6rem;">⬇</div>

          <div style="
              padding:0.6rem 0.8rem;
              border-radius:12px;
              background-color:#fff7e6;
              text-align:center;
              font-weight:600;
          ">
            Set dilution series → tune parameters → view curves & edge cases → save runs
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ===================== Usage notes (full width, at bottom) =====================

st.markdown("### 📋 Usage notes")
st.markdown(
        """
**Set the dilution series**

Begin by entering the **top concentration** and the **dilution factor**.  
These values define the concentration levels used to generate the dose–response curves.  
You can optionally override with **7 custom dilution factors**.

**Specify parameter ranges**

Each model includes parameters that control curve shape and behavior  
(e.g., asymptotes, slope, EC₅₀, asymmetry).  
Specify **min** and **max** for each; the main plot uses the **midpoint** of each range.

**Relative potencies (RP)**

Many pages allow you to specify **RP values** to create a family of parallel curves.  
For example, RP = 0.4, 1.0, 1.6 corresponds roughly to **40%**, **100%**, and **160%** potency.

**Run and save experiments**

After entering your inputs, the tool generates the corresponding dose–response curves.  
Click **“Add curve”** to *lock in* the current curve family (including RP variants).  
You can then adjust parameters and compare multiple saved experiments.

**Edge-case panels**

Each model page includes an **edge-case view** that combines min/max parameter values.  
This lets you see how extreme settings influence the curve and whether your dilution scheme  
still covers anchors and the linear region.

**Review & export (future extension)**

Saved curves and well values are kept in tables that can be exported  
for downstream analysis or documentation.
"""
    )

st.markdown("---")


# ===================== Optimization + Scoring (two-column layout) =====================

opt_col, detail_col = st.columns([1.1, 1.2])

with opt_col:
    st.markdown("### ⚙️ Optimization / Dilution Scheme Scoring")

    st.markdown(
        """
The optimization panel helps you choose an **even dilution factor** that gives a good
distribution of wells across:

- **Lower anchors** (near the lower asymptote)  
- **Linear region** (steep, informative part of the curve)  
- **Upper anchors** (near the upper asymptote)

On the 4PL/5PL pages, the recommender:

1. Takes your **top concentration** and parameter ranges (**A, B, C, D** and **E** for 5PL).  
2. Builds **7 log-spaced wells** for each candidate even dilution factor.  
3. For each parameter edge case and each RP, it:
   - Computes the curve at those wells.  
   - Finds boundaries where the curve reaches ~10% and ~90% of its dynamic range.  
   - Counts how many wells fall:
     - below the 10% boundary (**bottom anchors**)  
     - between 10% and 90% (**linear region**)  
     - above the 90% boundary (**top anchors**)
"""
    )

with detail_col:
    st.markdown("### 🔍 Anchor pattern, score, and thresholds")

    st.markdown(
        r"""
**Target pattern**

For the 7 wells, the **target pattern** is roughly:

- 2 bottom anchors  
- 3 linear points  
- 2 top anchors  

For each candidate factor and each parameter/RP combination, the app computes a
**penalty score**:
        """
    )

    # --- Equation on its own line, like the 4PL page ---
    st.latex(r"""
        J = (n_{\text{bottom}} - 2)^2
        + (n_{\text{linear}} - 3)^2
        + (n_{\text{top}} - 2)^2
    """)

    st.markdown(
        r"""
- *bottom* = wells below the lower anchor boundary  
- *linear* = wells between ~10–90% of the dynamic range  
- *top* = wells above the upper anchor boundary  

For a given factor, the tool looks across **all** edge-case parameter
combinations (and RPs) and takes the **worst-case** \(J\).  
That worst-case pattern is summarized in the table as:

- `worst_lower`  
- `worst_linear`  
- `worst_upper`  
- `score` (the corresponding \(J\))
        """
    )