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

# Single-page main app UI (admin/pipeline/search).
# We keep Streamlit's sidebar, but repurpose pages to:
# - Back to PortalRecruit homepage
# - Member Login (placeholder)

LOGO_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_LOGO.png"
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

/* Sidebar: keep it, just style it */
section[data-testid=\"stSidebar\"] {{
  background: rgba(2, 6, 23, 0.55) !important;
  border-right: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
}}

/* Nav links */
div[data-testid=\"stSidebarNav\"] a {{
  border-radius: 12px;
  padding: 10px 12px;
}}
div[data-testid=\"stSidebarNav\"] a:hover {{
  background: rgba(255,255,255,0.06);
}}

/* Collapsed control button */
button[data-testid=\"collapsedControl\"] {{
  border-radius: 12px !important;
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}}

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600;700;900&display=swap');

h1, h2, h3, h4, h5, h6 {{
  color: #f8fafc !important;
  font-family: "Space Grotesk", Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif !important;
  letter-spacing: -0.01em;
}}

p, div, span, label, li {{
  color: #f8fafc !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif !important;
}}

/* PortalRecruit hero title */
.pr-hero {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin: 10px 0 18px 0;
  text-align: center;
}}
.pr-hero-title {{
  font-size: 56px;
  font-weight: 800;
  line-height: 1.05;
}}
.pr-hero-sub {{
  font-size: 16px;
  opacity: 0.78;
}}

/* -----------------------------
   PortalRecruit “glassy” theme
   ----------------------------- */
:root {{
  --pr-glass-bg: rgba(15, 23, 42, 0.28);
  --pr-glass-bg-2: rgba(30, 41, 59, 0.38);
  --pr-glass-border: rgba(255, 255, 255, 0.08);
  --pr-glow: rgba(234, 88, 12, 0.16);
}}

/* Whole-page glass overlay (subtle) */
.bg-video-overlay {{
  /* keep existing gradient, but add a subtle frosted layer */
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}}

/* Remove the big "single panel" look; let content float */
section.main > div.block-container {{
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 18px 22px 30px 22px;
  box-shadow: none !important;
  margin-top: 10px;
}}

/* Sub-panels (forms / status / expanders) */
div[data-testid="stForm"] {
  border: 2px solid rgba(255,255,255,0.14);
  box-shadow: 0 0 0 1px rgba(234,88,12,0.10), 0 18px 60px rgba(0,0,0,0.42);
}

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

# Centered hero header (bigger than any section below)
st.markdown(
    f"""
<div class="pr-hero">
  <img src="{LOGO_URL}" style="width:92px; height:92px; object-fit:contain; filter: drop-shadow(0 0 12px rgba(234,88,12,0.25));" />
  <div class="pr-hero-title">PortalRecruit</div>
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
