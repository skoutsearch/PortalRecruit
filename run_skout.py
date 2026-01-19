import subprocess
import sys
import time
import webbrowser
import os
from threading import Thread

def run_landing_page():
    """Hosts the 'www' directory on Port 8000"""
    print("🚀 Host: Landing Page running at http://localhost:8000")
    # Using Python's built-in simple HTTP server
    subprocess.run([sys.executable, "-m", "http.server", "8000", "--directory", "www"])

def run_app():
    """Hosts the Streamlit Engine on Port 8501"""
    print("🏀 App: SKOUT Engine running at http://localhost:8501")
    # Running Streamlit as a module
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/Home.py"])

if __name__ == "__main__":
    print("\n------------------------------------------------")
    print("   STARTING SKOUT INTELLIGENCE SYSTEM")
    print("------------------------------------------------")

    # 1. Start Marketing Site (Background Thread)
    landing_thread = Thread(target=run_landing_page)
    landing_thread.daemon = True 
    landing_thread.start()

    # 2. Start Application (Main Process)
    # We launch this first to ensure the backend is ready
    app_thread = Thread(target=run_app)
    app_thread.daemon = True
    app_thread.start()

    # 3. Wait a moment for servers to spin up, then open browser
    time.sleep(3)
    print("✨ Systems Online. Opening Browser...")
    webbrowser.open("http://localhost:8000")

    # Keep script running to maintain servers
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SKOUT...")
        sys.exit(0)
