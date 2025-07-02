#!/usr/bin/env python3
"""
Alternative launcher for the Quantum Art Streamlit App
Use this if the standard 'streamlit run' command doesn't work.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🚀 Launching Quantum Art Streamlit App...")
    print("=" * 50)
    
    # Check if Image_Generation.py exists
    app_file = Path("Image_Generation.py")
    if not app_file.exists():
        print("❌ Error: Image_Generation.py not found!")
        print("Please make sure you're in the correct directory.")
        return
    
    print("✅ Found Image_Generation.py")
    
    # Try to launch Streamlit
    try:
        print("🔄 Starting Streamlit server...")
        
        # Try different methods
        methods = [
            [sys.executable, "-m", "streamlit", "run", "Image_Generation.py", "--server.port", "8501"],
            [sys.executable, "-m", "streamlit", "run", "Image_Generation.py", "--server.port", "8502"],
            ["streamlit", "run", "Image_Generation.py", "--server.port", "8501"],
            ["streamlit", "run", "Image_Generation.py", "--server.port", "8502"]
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"📡 Trying method {i}...")
            try:
                # Start the process
                process = subprocess.Popen(
                    method,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait a moment to see if it starts successfully
                time.sleep(3)
                
                if process.poll() is None:  # Process is still running
                    port = method[-1] if "--server.port" in method else "8501"
                    url = f"http://localhost:{port}"
                    
                    print(f"✅ Streamlit server started successfully!")
                    print(f"🌐 URL: {url}")
                    print(f"📋 To stop the server, press Ctrl+C in the terminal")
                    print()
                    print("🔴 IMPORTANT: If the browser doesn't open automatically,")
                    print(f"   manually open: {url}")
                    print()
                    
                    # Try to open browser
                    try:
                        print("🌐 Attempting to open browser...")
                        webbrowser.open(url)
                        print("✅ Browser should open now!")
                    except Exception as e:
                        print(f"⚠️  Could not open browser automatically: {e}")
                        print(f"👉 Please manually open: {url}")
                    
                    # Wait for the process to finish
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping Streamlit server...")
                        process.terminate()
                        print("✅ Server stopped.")
                    
                    return
                else:
                    # Process ended, check for errors
                    stdout, stderr = process.communicate()
                    print(f"❌ Method {i} failed:")
                    if stderr:
                        print(f"   Error: {stderr.strip()}")
                    continue
                    
            except FileNotFoundError:
                print(f"❌ Method {i} failed: Command not found")
                continue
            except Exception as e:
                print(f"❌ Method {i} failed: {e}")
                continue
        
        # If we get here, all methods failed
        print("\n❌ All launch methods failed!")
        print("\n🔧 Manual Steps:")
        print("1. Open Command Prompt or PowerShell")
        print("2. Navigate to this directory")
        print("3. Run: streamlit run Image_Generation.py")
        print("4. Open http://localhost:8501 in your browser")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 Try manual launch:")
        print("streamlit run Image_Generation.py")


if __name__ == "__main__":
    main() 