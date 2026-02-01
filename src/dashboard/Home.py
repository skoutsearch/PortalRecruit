import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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
# IMPORTANT: avoid leading indentation in this HTML/CSS string.
# Streamlit's markdown renderer can treat indented lines as a code block and print the CSS.
BG_HTML = f"""<div class=\"bg-video-wrap\">
<video class=\"bg-video\" autoplay loop muted playsinline>
  <source src=\"{BG_VIDEO_URL}\" type=\"video/mp4\" />
</video>
<div class=\"bg-video-overlay\"></div>
</div>
<style>
.bg-video-wrap {{
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  z-index: 0;
  pointer-events: none;
}}
.bg-video-wrap * {{ pointer-events: none; }}

/* Force ALL Streamlit UI layers above the background video */
html, body {{ background: #020617; }}

/* Streamlit paints opaque backgrounds on several wrappers; make them transparent
   so the fixed video can actually show through. */
[data-testid=\"stAppViewContainer\"],
[data-testid=\"stMain\"],
[data-testid=\"stHeader\"],
[data-testid=\"stToolbar\"],
.stApp, .stApp > div, .stApp main {{
  background: transparent !important;
  position: relative;
  z-index: 2;
}}

.bg-video {{
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
}}

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
  content: \"\";
  position: absolute;
  inset: 0;
  background: radial-gradient(closest-side, rgba(0,0,0,0) 62%, rgba(0,0,0,0.40) 100%);
}}

/* Hide sidebar + multipage nav completely */
section[data-testid=\"stSidebar\"],
div[data-testid=\"stSidebarNav\"],
button[data-testid=\"collapsedControl\"] {{
  display: none !important;
}}

h1, h2, h3, h4, h5, h6, p, div, span, label, li {{
  color: #f8fafc !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}

/* -----------------------------
   PortalRecruit “glassy” theme
   ----------------------------- */
:root {{
  --pr-glass-bg: rgba(15, 23, 42, 0.55);
  --pr-glass-bg-2: rgba(30, 41, 59, 0.35);
  --pr-glass-border: rgba(255, 255, 255, 0.08);
  --pr-glow: rgba(234, 88, 12, 0.16);
}}

/* Main content as a single glass panel */
section.main > div.block-container {{
  background: var(--pr-glass-bg);
  border: 1px solid var(--pr-glass-border);
  border-radius: 22px;
  padding: 24px 28px 34px 28px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.42), 0 0 28px var(--pr-glow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  margin-top: 18px;
}}

/* Sub-panels (forms / status / expanders) */
div[data-testid="stForm"],
div[data-testid="stStatusWidget"],
div[data-testid="stExpander"],
div[data-testid="stAlert"],
div[data-testid="stNotification"],
div[data-testid="stMarkdownContainer"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--pr-glass-bg-2);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 14px 14px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}}

/* Make Streamlit's default horizontal separators subtler */
hr {{
  border: none;
  height: 1px;
  background: rgba(255,255,255,0.08);
}}

/* Buttons */
.stButton > button {{
  border-radius: 14px !important;
  font-weight: 800 !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: linear-gradient(135deg, rgba(234,88,12,0.92) 0%, rgba(249,115,22,0.88) 100%) !important;
  color: #0b1120 !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.35);
}}
.stButton > button:hover {{
  filter: brightness(1.05);
  box-shadow: 0 12px 34px rgba(0,0,0,0.42);
}}

/* Inputs */
input, textarea, select {{
  border-radius: 12px !important;
}}

</style>"""

# Use components.html instead of st.markdown to avoid any markdown/code-block rendering quirks
# that can cause CSS to be printed as text.
components.html(BG_HTML, height=0, width=0)

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
