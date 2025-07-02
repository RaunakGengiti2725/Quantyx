# ⚛️ Quantyx - AI Image Generation Platform

> **Physics-Inspired Creativity • Export Ready • Professional Quality**

![Quantyx Banner](https://img.shields.io/badge/Quantyx-AI%20Art%20Platform-blueviolet?style=for-the-badge&logo=atom)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Quantyx is a revolutionary **Research-to-Art Platform** that transforms real quantum many-body simulations into stunning visualizations. Unlike other generators that use "physics-inspired" algorithms, Quantyx renders **actual quantum entanglement data** from your research, creating artwork that represents genuine scientific discoveries.

## 🌟 Features

### 🔬 **Real Quantum Research Integration** ⭐ **NEW!**
- **Live Quantum Simulations** - PennyLane-powered TFIM, XXZ, Heisenberg Hamiltonians
- **Entanglement Analysis** - Von Neumann entropy across all contiguous intervals
- **Holographic Correspondence** - AdS/CFT-inspired bulk-boundary mapping
- **Graph Neural Networks** - Learn bulk geometry from boundary entanglement
- **Curvature-Energy Correlations** - Real physics results in every visualization
- **Research Data Export** - Download complete quantum simulation data

### 🎨 **5 Unique Artistic Styles**
- **Quantum Bloom** - Blooming quantum interference patterns
- **Singularity Core** - Gravitational collapse visualizations  
- **Entanglement Field** - Quantum entanglement dynamics
- **Crystal Spire** - Geometric crystalline structures
- **Tunneling Veil** - Quantum tunneling probability clouds

### 🎛️ **Real-Time Physics Controls**
- **Energy Flux** (0-10) - Control quantum field intensity
- **Geometric Order** (0-100%) - Adjust symmetry levels
- **Field Distortion** (0-1) - Spacetime curvature effects
- **Spectral Blend** (0-1) - Color variation across fields

### 🌈 **8 Professional Color Palettes**
- Deep Plasma, Quantum Aurora, Neon Dreams
- Cosmic Fire, Ocean Depths, Forest Mystique
- Desert Heat, Arctic Glow

### 📐 **Export Options**
- **Standard** (512×512) - Quick previews
- **HD** (1024×1024) - Social media ready
- **4K** (3840×2160) - Professional displays
- **Print** (4096×4096) - Ultra high-resolution

### 🎬 **Animation Studio**
- Create looping GIF animations
- Parameter morphing over time
- Perfect for VJing and social media
- Professional export quality

### 🧬 **Quantum Research Presets**
- **🔬 Quantum Criticality** - Critical point phase transitions (TFIM)
- **🌌 Deep Entanglement** - Maximum entropy regimes (Heisenberg)
- **🕳️ Holographic Duality** - Strong bulk-boundary correspondence (XXZ)
- **⚡ Quantum Quench** - Sudden parameter change dynamics
- **📐 Geometric Phase** - Adiabatic evolution with Berry phases

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RaunakGengiti2725/Quantyx.git
   cd Quantyx
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Quantyx**
   ```bash
   streamlit run "Image_Generation.py"
   ```
   
   Or use the launcher:
   ```bash
   python launch_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 💡 Usage Guide

### **Quick Start with Presets**
1. Choose a **Quick Preset** from the sidebar (Ethereal Dream, Cosmic Storm, etc.)
2. Click **⚛️ Run Simulation**
3. Download your artwork in your preferred resolution

### **Manual Fine-Tuning**
1. Select your **Quantum Structure** style
2. Adjust **Field Parameters**:
   - **Energy Flux** for intensity
   - **Geometric Order** for symmetry
   - **Field Distortion** for warping effects
   - **Spectral Blend** for color variation
3. Choose your **Spectral Palette**
4. Set **Output Resolution**
5. Generate and export

### **Creating Animations**
1. Navigate to **Animate Fields** page
2. Set base parameters
3. Choose animation parameter (Energy, Symmetry, etc.)
4. Set frame count and variation range
5. Generate looping GIF animation

## 🔬 Technical Details

### **Quantum Research Foundation**
Quantyx uses your actual research modules:
- **PennyLane Simulations** - Real quantum circuits and Hamiltonians  
- **Graph Neural Networks** - Bulk geometry reconstruction from entanglement
- **Holographic Duality** - AdS/CFT correspondence principles
- **Many-Body Physics** - TFIM, XXZ, Heisenberg model simulations
- **Entanglement Analysis** - Von Neumann entropy distributions
- **Curvature-Energy Mapping** - Boundary-bulk holographic correspondence

### **Rendering Pipeline**
- NumPy arrays for field calculations
- SciPy for Gaussian filtering and transformations
- Matplotlib colormaps for scientific visualization
- PIL for high-quality image processing

### **Performance**
- **512px**: ~1 second generation
- **1024px**: ~3 seconds generation  
- **4K**: ~15 seconds generation
- Optimized algorithms for real-time parameter adjustment

## 🎯 Use Cases

### **Research & Science**
- **Physics Research** - Visualize your quantum simulation results
- **Scientific Papers** - Publication-ready quantum art figures  
- **Conference Presentations** - Stunning physics visualization
- **Grant Proposals** - Visual representation of quantum concepts
- **Educational Content** - Teaching quantum mechanics through art

### **Creative & Commercial**
- **Digital Art** - Physics-accurate generative art
- **VJ Performances** - Live quantum-driven visuals
- **Music Videos** - Science-inspired backgrounds
- **Social Media** - Unique quantum content
- **Print Media** - High-resolution scientific art
- **Game Development** - Quantum-themed procedural textures
- **Web Design** - Physics-based design elements

## 📁 Project Structure

```
Quantyx/
├── Image_Generation.py          # Main Streamlit app
├── pages/
│   └── Animate_Fields.py        # Animation studio
├── quantumproject/              # Core modules
│   ├── models/                  # Neural network models
│   ├── quantum/                 # Physics simulations
│   ├── utils/                   # Utility functions
│   └── visualization/           # Plotting tools
├── requirements.txt             # Python dependencies
├── launch_app.py               # Application launcher
└── .streamlit/                 # Streamlit configuration
```

## 🛠️ Development

### **Adding New Styles**
1. Implement your field equation in `QuantumArtGenerator`
2. Add style mapping in `_render_field` method
3. Update color scheme definitions
4. Test with various parameters

### **Custom Color Palettes**
1. Define your palette in `COLOR_PALETTES`
2. Include description and color mappings
3. Test across all artistic styles

### **Performance Optimization**
- Use NumPy vectorization for field calculations
- Implement caching for repeated computations
- Consider multiprocessing for animation rendering

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎭 Gallery

*Coming soon - showcase of artwork created with Quantyx*

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/RaunakGengiti2725/Quantyx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RaunakGengiti2725/Quantyx/discussions)
- **Documentation**: Check the `/docs` folder for detailed guides

## 🌟 Acknowledgments

- Built with ❤️ using quantum physics and mathematics
- Powered by Streamlit, NumPy, and SciPy
- Inspired by the beauty of quantum mechanics
- Special thanks to the scientific visualization community

---

**⚛️ Quantyx - Where Physics Meets Art**

*Transform quantum equations into visual masterpieces*
