import sys
from pathlib import Path

import streamlit as st

# Ensure repo root is on sys.path so imports like `from src...` work
# even when Streamlit runs this file from inside `src/dashboard/`.
# Home.py lives at <repo>/src/dashboard/Home.py
# To import the top-level package `src.*`, we need <repo> on sys.path.
# Home.py is at <repo>/src/dashboard/Home.py
# parents[0]=dashboard, [1]=src, [2]=<repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Single-page main app UI (admin/pipeline/search).
# We keep Streamlit's sidebar, but repurpose pages to:
# - Back to PortalRecruit homepage
# - Member Login (placeholder)

WORDMARK_DARK_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_WORDMARK_DARK.jpg"
WORDMARK_LIGHT_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_WORDMARK_LIGHT.jpg"
BG_VIDEO_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_ANIMATED_LOGO.mp4"

st.set_page_config(
    page_title="PortalRecruit | Setup & Search",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get help": "https://portalrecruit.com",
        "About": "PortalRecruit — recruiting intelligence platform.",
    },
)

# Background video (hosted on GitHub Pages to avoid base64 + slow loads)
# IMPORTANT: avoid leading indentation in this HTML/CSS string.
# Streamlit's markdown renderer can treat indented lines as a code block and print the CSS.
from src.dashboard.theme import inject_background

inject_background()

# Top wordmark (replace visible "PortalRecruit" text with the image)
st.markdown(
    f"""
<div class="pr-hero">
  <img src="{WORDMARK_DARK_URL}" style="max-width:560px; width:min(560px, 92vw); height:auto; object-fit:contain;" />
  <div class="pr-hero-sub">Connect your Synergy key → ingest data → start searching.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Execute the Admin/Pipeline UI directly on the home page.
# This keeps one source of truth without trying to import a module named "1_Admin_Settings".
admin_path = Path(__file__).with_name("admin_content.py")
code = admin_path.read_text(encoding="utf-8")
exec(compile(code, str(admin_path), "exec"), globals(), globals())
