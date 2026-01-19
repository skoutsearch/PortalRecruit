import streamlit as st
import os
import sys
import threading
import subprocess

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

st.set_page_config(page_title="SKOUT Admin", layout="wide", page_icon="⚙️")

ENV_PATH = os.path.join(os.getcwd(), ".env")

def save_api_key(key):
    """Writes the API key to the .env file."""
    with open(ENV_PATH, "w") as f:
        f.write(f"SYNERGY_API_KEY={key}\n")
    os.environ["SYNERGY_API_KEY"] = key # Update current session
    st.toast("API Key Saved Successfully!", icon="✅")

def run_ingestion_script(script_name):
    """Runs a python script as a subprocess."""
    script_path = os.path.join(os.getcwd(), "src", script_name)
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

st.title("⚙️ System Configuration")
st.markdown("Manage your Synergy Data connection and database status.")

# --- SECTION 1: API CREDENTIALS ---
st.subheader("1. Synergy API Connection")
current_key = os.getenv("SYNERGY_API_KEY", "")

with st.form("api_key_form"):
    user_key = st.text_input("Enter Synergy API Key", value=current_key, type="password")
    submit = st.form_submit_button("Save Credentials")
    
    if submit:
        save_api_key(user_key)
        st.rerun()

if current_key:
    st.success("System is connected to Synergy Sports API.")
else:
    st.error("System is disconnected. Please enter a key.")

st.divider()

# --- SECTION 2: DATA INGESTION ---
st.subheader("2. Data Pipeline")
st.caption("Trigger data updates manually. In production, this runs automatically overnight.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Step 1: Game Schedule**")
    st.info("Fetches latest games from API.")
    if st.button("Sync Schedule"):
        with st.spinner("Fetching games..."):
            log = run_ingestion_script("ingestion/ingest_acc_schedule.py")
            st.text_area("Log", log, height=150)

with col2:
    st.markdown("**Step 2: Play Data**")
    st.info("Downloads PBP data for games.")
    if st.button("Sync Plays"):
        with st.spinner("Downloading plays..."):
            log = run_ingestion_script("ingestion/ingest_game_events.py")
            st.text_area("Log", log, height=150)

with col3:
    st.markdown("**Step 3: AI Intelligence**")
    st.info("Generates Tags & Embeddings.")
    if st.button("Build AI Index"):
        with st.spinner("Thinking..."):
            # Run Tagger then Embeddings
            log1 = run_ingestion_script("processing/apply_tags.py")
            log2 = run_ingestion_script("processing/generate_embeddings.py")
            st.text_area("Log", log1 + "\n" + log2, height=150)
