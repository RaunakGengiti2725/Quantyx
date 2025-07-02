# 🚀 How to Launch the Quantum Art Streamlit App

## ✅ Quick Steps to Launch

### **Method 1: Standard Launch**
1. Open **Command Prompt** or **PowerShell**
2. Navigate to the project folder:
   ```
   cd "C:\Users\rgeng\Dropbox\PC\Downloads\quantumproject-main (8)\quantumproject-main"
   ```
3. Run the app:
   ```
   streamlit run streamlit_quantum_app.py
   ```
4. **Manually open your browser** and go to: `http://localhost:8501`

### **Method 2: Force Browser Opening**
```
streamlit run streamlit_quantum_app.py --server.headless false
```

### **Method 3: Different Port**
```
streamlit run streamlit_quantum_app.py --server.port 8502
```
Then open: `http://localhost:8502`

## 🔧 Troubleshooting

### **Issue: Command runs but browser doesn't open**
**Solution**: Manually open your web browser and navigate to:
- `http://localhost:8501` (default)
- `http://127.0.0.1:8501` (alternative)

### **Issue: Port already in use**
**Solution**: Try a different port:
```
streamlit run streamlit_quantum_app.py --server.port 8503
```

### **Issue: Firewall blocking**
**Solution**: When Windows asks about firewall, click "Allow access"

### **Issue: Browser shows "This site can't be reached"**
**Solutions**:
1. Wait 30 seconds for the server to fully start
2. Try refreshing the page
3. Check if the command is still running in terminal
4. Try `http://127.0.0.1:8501` instead

## 🌐 Manual Browser Launch

If automatic browser opening fails:

1. **Copy this URL**: `http://localhost:8501`
2. **Open any browser** (Chrome, Firefox, Edge, Safari)
3. **Paste the URL** in the address bar
4. **Press Enter**

## ⚡ Quick Test

First, let's test with a simple app:

1. Run this command:
   ```
   streamlit run simple_streamlit_test.py
   ```
2. Open `http://localhost:8501` in your browser
3. If you see a simple test page with a sine wave, Streamlit is working!

## 🎨 Full Quantum Art App

Once the simple test works, run the full app:
```
streamlit run streamlit_quantum_app.py
```

## 📋 What You Should See

When successful, you'll see:
1. **Terminal output** with "You can now view your Streamlit app in the browser."
2. **Local URL** showing `http://localhost:8501`
3. **Web browser** opens automatically (or you open it manually)
4. **Beautiful dark-themed art generator** with:
   - Animated gradient background
   - Sidebar with art style selector
   - Parameter sliders
   - Generate button

## 🐛 Common Issues & Fixes

### **Python/Streamlit Not Found**
```
pip install streamlit numpy matplotlib scipy pillow
```

### **Port Issues**
Add `--server.port XXXX` where XXXX is a different number (8502, 8503, etc.)

### **Permission Issues**
Run PowerShell as Administrator

### **Antivirus Blocking**
Temporarily disable antivirus or add Python to exceptions

## 🔄 Alternative: Run from Python Directly

If Streamlit command doesn't work, try this Python approach:

1. Create a file called `launch_app.py`:
```python
import subprocess
import sys

def main():
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_quantum_app.py", 
            "--server.port", "8501"
        ])
    except Exception as e:
        print(f"Error: {e}")
        print("Try opening http://localhost:8501 manually")

if __name__ == "__main__":
    main()
```

2. Run: `python launch_app.py`

## 📱 Mobile Access

Once running, you can also access from your phone:
1. Find your computer's IP address (usually 192.168.x.x)
2. Open `http://YOUR_IP:8501` on your phone's browser

## ✨ Expected App Features

When working correctly, you'll have:
- **🎨 5 Art Styles**: Quantum Bloom, Singularity Core, Entanglement Field, Crystal Spire, Tunneling Veil
- **⚛️ 4 Parameter Sliders**: Energy, Symmetry, Deformation, Color Variation
- **🚀 Generate Button**: Creates art in 1-3 seconds
- **💾 Download Button**: Saves high-res PNG files
- **🌌 Beautiful UI**: Dark theme with animations

## 🆘 Still Not Working?

1. **Check the terminal** for error messages
2. **Try the simple test app first**: `streamlit run simple_streamlit_test.py`
3. **Use a different browser** (Chrome recommended)
4. **Clear browser cache** and try again
5. **Restart your computer** and try again
6. **Check Windows Defender/Firewall** settings

---

**🎯 Key Point**: The Streamlit server might start successfully but not open the browser automatically. Always try manually navigating to `http://localhost:8501` in your browser! 