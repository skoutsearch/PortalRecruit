import os
from pathlib import Path

import streamlit as st

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
  .stApp {{ position: relative; z-index: 1; }}

  .bg-video {{
    position: absolute;
    top: 50%;
    left: 50%;
    min-width: 100%;
    min-height: 100%;
    width: auto;
    height: auto;
    transform: translate(-50%, -50%) scale(1.03);
    object-fit: cover;
    opacity: 0.32;
    filter: saturate(1.05) contrast(1.02) brightness(0.92) blur(3px);
  }}

  .bg-video-overlay {{
    position: absolute;
    inset: 0;
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
