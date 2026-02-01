import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Back to Homepage", page_icon="↩️", layout="wide")

# Auto-redirect (best effort). If the browser blocks it, user still has a button.
components.html(
    """
<script>
  try {
    window.top.location.href = "https://portalrecruit.com";
  } catch (e) {}
</script>
""",
    height=0,
    width=0,
)

st.markdown(
    """
<div style="max-width:760px; margin: 8vh auto 0 auto; text-align:center;">
  <h1 style="font-size:44px; font-weight:900; margin-bottom:10px;">Returning to Homepage</h1>
  <p style="opacity:0.8; font-size:16px; margin-bottom:26px;">If you weren't redirected automatically, use the button below.</p>
</div>
""",
    (True),
)

st.link_button("Go to portalrecruit.com", "https://portalrecruit.com", use_container_width=True)
