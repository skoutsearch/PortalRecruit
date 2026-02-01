"""Shared UI helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st


CSS_URL = "https://skoutsearch.github.io/PortalRecruit/streamlit.css"
BG_VIDEO_URL = "https://skoutsearch.github.io/PortalRecruit/PORTALRECRUIT_ANIMATED_LOGO.mp4"


def inject_background(video_url: str | None = None) -> None:
    """Inject the PortalRecruit video background + shared CSS.

    Uses external CSS (GitHub Pages) to avoid Streamlit stripping <style> blocks.
    """

    vid = video_url or BG_VIDEO_URL

    bg_html = f"""<link rel=\"stylesheet\" href=\"{CSS_URL}\" />
<div style=\"position:fixed; inset:0; width:100vw; height:100vh; overflow:hidden; z-index:0; pointer-events:none;\">
  <video autoplay loop muted playsinline style=\"position:absolute; top:50%; left:50%; min-width:100%; min-height:100%; width:auto; height:auto; transform:translate(-50%,-50%) scale(1.03); object-fit:cover; opacity:0.32; filter:saturate(1.05) contrast(1.02) brightness(0.92) blur(3px);\">
    <source src=\"{vid}\" type=\"video/mp4\" />
  </video>
  <div style=\"position:absolute; inset:0; background:linear-gradient(180deg, rgba(2,6,23,0.88) 0%, rgba(2,6,23,0.82) 45%, rgba(2,6,23,0.90) 100%); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);\"></div>
  <div style=\"position:absolute; inset:0; background:radial-gradient(closest-side, rgba(0,0,0,0) 62%, rgba(0,0,0,0.40) 100%);\"></div>
</div>"""

    st.markdown(bg_html, unsafe_allow_html=True)
