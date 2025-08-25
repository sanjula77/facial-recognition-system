#!/usr/bin/env python3
"""
Launcher script for the Facial Recognition Streamlit Application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application."""
    
    # Check if we're in the right directory
    if not os.path.exists("src/streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found in src/ directory")
        print("💡 Make sure you're running this from the project root directory")
        sys.exit(1)
    
    print("🚀 Launching Facial Recognition Web Application...")
    print("📱 Opening in your default web browser...")
    print("💡 Press Ctrl+C to stop the application")
    print()
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
