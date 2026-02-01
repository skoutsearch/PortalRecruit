import streamlit as st

st.set_page_config(page_title="Member Login", page_icon="🔒", layout="wide")

st.markdown(
    """
<style>
  .login-wrap { max-width: 520px; margin: 7vh auto 0 auto; }
  .login-card {
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 26px 22px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 18px 60px rgba(0,0,0,0.42);
  }
  .login-title { font-size: 40px; font-weight: 950; margin: 0 0 6px 0; text-align:center; }
  .login-sub { opacity: 0.78; text-align:center; margin: 0 0 18px 0; }
</style>
<div class="login-wrap">
  <div class="login-card">
    <div class="login-title">Member Login</div>
    <div class="login-sub">Placeholder for a future subscription / member portal.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.container():
    st.text_input("Email", placeholder="coach@school.edu")
    st.text_input("Password", type="password", placeholder="••••••••")
    st.button("Sign in (placeholder)", use_container_width=True)

st.info("This page is a visual placeholder only — no authentication is implemented yet.")
