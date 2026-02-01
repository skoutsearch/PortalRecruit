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

# One fucking page: Home renders the Admin/Pipeline UI.
# (No multipage nav, no clicking around.)

LOGO_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_LOGO.png"
BG_VIDEO_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_ANIMATED_LOGO.mp4"

st.set_page_config(
    page_title="PortalRecruit | Setup & Search",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="collapsed",
)

# Background video (hosted on GitHub Pages to avoid base64 + slow loads)
st.markdown(
    f"""
<div class="bg-video-wrap">
  <video class="bg-video" autoplay loop muted playsinline>
    <source src="{BG_VIDEO_URL}" type="video/mp4" />
  </video>
  <div class="bg-video-overlay"></div>
</div>
<style>
  .bg-video-wrap {{
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    z-index: 0; /* avoid negative z-index quirks */
    pointer-events: none; /* never block clicks */
  }}
  .bg-video-wrap * {{ pointer-events: none; }}

  /* Force ALL Streamlit UI layers above the background video */
  html, body { background: #020617; }

  /* Streamlit paints opaque backgrounds on several wrappers; make them transparent
     so the fixed video can actually show through. */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  .stApp, .stApp > div, .stApp main {
    background: transparent !important;
    position: relative;
    z-index: 2;
  }

  .bg-video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    min-width: 100%;
    min-height: 100%;
    object-fit: cover;
    opacity: 0.32;
    filter: saturate(1.05) contrast(1.02) brightness(0.92) blur(3px);
    z-index: 0;
  }

  .bg-video-overlay {{
    position: absolute;
    inset: 0;
    z-index: 1;
    background: linear-gradient(
      180deg,
      rgba(2, 6, 23, 0.88) 0%,
      rgba(2, 6, 23, 0.82) 45%,
      rgba(2, 6, 23, 0.90) 100%
    );
  }}
  .bg-video-overlay::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(closest-side, rgba(0,0,0,0) 62%, rgba(0,0,0,0.40) 100%);
  }}

  /* Hide sidebar + multipage nav completely */
  section[data-testid="stSidebar"],
  div[data-testid="stSidebarNav"],
  button[data-testid="collapsedControl"] {{
    display: none !important;
  }}

  h1, h2, h3, h4, h5, h6, p, div, span, label, li {{
    color: #f8fafc !important;
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# Minimal top brand bar
st.markdown(
    f"""
<div style="display:flex; align-items:center; gap:14px; margin: 6px 0 18px 0;">
  <img src="{LOGO_URL}" style="width:56px; height:56px; object-fit:contain;" />
  <div>
    <div style="font-size:28px; font-weight:900; line-height:1.1;">PortalRecruit</div>
    <div style="opacity:0.75; font-size:14px;">Connect your Synergy key → ingest data → start searching.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Execute the Admin/Pipeline UI directly on the home page.
# This keeps one source of truth without trying to import a module named "1_Admin_Settings".
admin_path = Path(__file__).with_name("pages") / "1_Admin_Settings.py"
code = admin_path.read_text(encoding="utf-8")
exec(compile(code, str(admin_path), "exec"), globals(), globals())
