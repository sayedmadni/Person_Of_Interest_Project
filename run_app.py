#!/usr/bin/env python3
"""
Launch script for the Celebrity Search AI Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app with the beautiful theme"""
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Launching Celebrity Search AI...")
    print("📱 Opening in your default browser...")
    print("🎨 Theme: Modern gradient design with enhanced UI")
    print("⚡ Features: AI-powered search, gallery view, real-time analysis")
    print("\n" + "="*50)
    
    try:
        # Run the Streamlit app using uv
        subprocess.run([
            "uv", "run", "streamlit", "run", 
            "src/ad_poi.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
