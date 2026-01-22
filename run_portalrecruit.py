import subprocess
import sys
import time
import webbrowser
import os
from threading import Thread

def run_landing_page():
    # Silently run the landing page server
    subprocess.run([sys.executable, "-m", "http.server", "8000", "--directory", "www"], 
                  stdout=subprocess.DEVNULL, 
                  stderr=subprocess.DEVNULL)

def run_app():
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/Home.py", 
                    "--server.headless", "true", 
                    "--server.port", "8501"])

if __name__ == "__main__":
    print("\n------------------------------------------------")
    print("   🚀 STARTING SKOUT INTELLIGENCE SYSTEM")
    print("------------------------------------------------")

    # 1. Start Landing Page (Marketing Site)
    print("   • Hosting Landing Page on Port 8000...")
    t1 = Thread(target=run_landing_page)
    t1.daemon = True
    t1.start()

    # 2. Start Intelligence Engine (App)
    print("   • Booting AI Engine on Port 8501...")
    t2 = Thread(target=run_app)
    t2.daemon = True
    t2.start()

    # 3. Launch
    print("   • Waiting for systems...")
    time.sleep(3) 
    
    url = "http://localhost:8000"
    print(f"\n   ✅ LAUNCH SUCCESSFUL: {url}")
    print("   (If browser does not open, click the link above)\n")
    
    try:
        webbrowser.open(url)
    except:
        pass

    # Keep alive
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n   🛑 Shutting down...")
        sys.exit(0)
