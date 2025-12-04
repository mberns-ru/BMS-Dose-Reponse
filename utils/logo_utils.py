# logo_utils.py
import os
import base64

import streamlit as st


LOGO_PATH = "graphics/Logo.jpg"


@st.cache_resource
def get_sidebar_logo_css(height: int = 200) -> str:
    """
    Load Logo.jpg once, convert to base64, and return the CSS to inject it
    into the sidebar nav. Because this is cached as a resource, the file is
    only opened once per process and the handle is immediately closed.
    """
    if not os.path.exists(LOGO_PATH):
        return ""

    # Open-and-close immediately: no descriptor leak.
    with open(LOGO_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"""
    <style>
        [data-testid="stSidebarNav"]::before {{
            content: "";
            display: block;
            height: {height}px;
            margin-bottom: 1rem;
            background-image: url("data:image/jpeg;base64,{img_b64}");
            background-position: center;
            background-repeat: no-repeat;
            background-size: contain;
        }}
    </style>
    """
