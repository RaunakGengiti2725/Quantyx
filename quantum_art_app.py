#!/usr/bin/env python3
"""
Quantum Geometry Art Generator - Streamlit App

A modern, visually stunning web app for generating physics-inspired art
using curvature-energy fields from quantum geometry simulations.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import io
import base64
from PIL import Image
import time
import random

# Configure page
st.set_page_config(
    page_title="Quantum Geometry Art Generator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, elegant styling
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 35%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated background glow */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.1) 0%, transparent 50%);
        animation: glow 20s ease-in-out infinite alternate;
        z-index: -1;
    }
    
    @keyframes glow {
        0% { transform: rotate(0deg) scale(1); opacity: 0.3; }
        50% { transform: rotate(180deg) scale(1.1); opacity: 0.5; }
        100% { transform: rotate(360deg) scale(1); opacity: 0.3; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInUp 1.5s ease-out;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1.8s ease-out;
        font-weight: 300;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Custom button styling */
    .generate-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .generate-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .generate-button:active {
        transform: translateY(0);
    }
    
    /* Loading spinner */
    .spinner {
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 2px solid #667eea;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 8px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Image container */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        animation: fadeIn 1s ease-out;
    }
    
    .image-container img {
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        max-width: 100%;
        height: auto;
    }
    
    .image-container img:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.7);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Download button */
    .download-button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        text-decoration: none;
        display: inline-block;
    }
    
    .download-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
        color: white;
        text-decoration: none;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #764ba2;
    }
    
    /* Streamlit widget overrides */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style parameters section */
    .parameter-section {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .parameter-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


class QuantumArtGenerator:
    """Advanced quantum geometry art generator with multiple styles."""
    
    def __init__(self):
        self.styles = {
            "Quantum Bloom": self._quantum_bloom,
            "Singularity Core": self._singularity_core,
            "Entanglement Field": self._entanglement_field,
            "Crystal Spire": self._crystal_spire,
            "Tunneling Veil": self._tunneling_veil
        }
    
    def generate_quantum_image(self, style, energy_intensity, symmetry_level, 
                             deformation_amount, color_variation, resolution=512):
        """Generate quantum geometry art based on parameters."""
        
        # Create coordinate system
        x = np.linspace(-4, 4, resolution)
        y = np.linspace(-4, 4, resolution)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Generate base field using selected style
        field = self.styles[style](X, Y, R, Theta, energy_intensity, 
                                 symmetry_level, deformation_amount)
        
        # Apply color variation and final processing
        field = self._apply_color_variation(field, color_variation)
        
        # Create the visualization
        return self._render_field(field, style, energy_intensity, color_variation)
    
    def _quantum_bloom(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Quantum Bloom: Flower-like quantum interference patterns."""
        petals = max(3, int(symmetry / 10))
        field = energy * np.sin(petals * Theta) * np.exp(-0.3 * R)
        
        # Add quantum interference rings
        rings = np.sin(4 * np.pi * R) * np.exp(-0.1 * R)
        field += 0.5 * energy * rings
        
        # Deformation effects
        if deformation > 0:
            warp = deformation * np.sin(3 * Theta + R) * np.exp(-0.2 * R)
            field += warp
        
        return gaussian_filter(field, sigma=1.0)
    
    def _singularity_core(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Singularity Core: Black hole-like gravitational field effects."""
        # Central singularity
        singularity = energy * np.exp(-2 * R) / (R + 0.1)
        
        # Accretion disk patterns
        disk = np.sin(symmetry * Theta / 10) * np.exp(-0.5 * R) * (1 / (R + 0.1))
        
        # Spacetime curvature effects
        curvature = energy * 0.3 * np.sin(8 * np.pi * R) / (R + 0.5)
        
        field = singularity + 0.7 * disk + 0.4 * curvature
        
        # Add deformation warping
        if deformation > 0:
            warp = deformation * np.sin(R * 2 * np.pi) * np.cos(4 * Theta)
            field += warp
        
        return gaussian_filter(field, sigma=0.8)
    
    def _entanglement_field(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Entanglement Field: Quantum entanglement visualization."""
        # Create entangled particle pairs
        pairs = int(symmetry / 20) + 2
        field = np.zeros_like(X)
        
        for i in range(pairs):
            angle1 = 2 * np.pi * i / pairs
            angle2 = angle1 + np.pi  # Entangled pair
            
            x1, y1 = 2 * np.cos(angle1), 2 * np.sin(angle1)
            x2, y2 = 2 * np.cos(angle2), 2 * np.sin(angle2)
            
            # Particle 1 field
            r1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
            field1 = energy * np.exp(-2 * r1) * np.sin(4 * np.pi * r1)
            
            # Particle 2 field (entangled - opposite phase)
            r2 = np.sqrt((X - x2)**2 + (Y - y2)**2)
            field2 = -energy * np.exp(-2 * r2) * np.sin(4 * np.pi * r2)
            
            field += field1 + field2
        
        # Add connection lines between entangled pairs
        connection = energy * 0.2 * np.sin(6 * Theta) * np.exp(-0.3 * R)
        field += connection
        
        # Deformation
        if deformation > 0:
            field += deformation * np.sin(X + Y) * np.cos(X - Y)
        
        return gaussian_filter(field, sigma=1.2)
    
    def _crystal_spire(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Crystal Spire: Crystalline growth patterns."""
        # Crystal lattice base
        lattice_spacing = 4.0 / max(1, symmetry / 10)
        crystal_x = np.sin(2 * np.pi * X / lattice_spacing)
        crystal_y = np.sin(2 * np.pi * Y / lattice_spacing)
        lattice = energy * crystal_x * crystal_y * np.exp(-0.2 * R)
        
        # Growth spires
        spires = energy * 0.7 * np.sin(4 * Theta) * np.exp(-0.4 * R) / (R + 0.1)
        
        # Faceted surfaces
        facets = np.sin(6 * Theta) * np.cos(3 * np.pi * R) * np.exp(-0.3 * R)
        
        field = lattice + spires + 0.4 * energy * facets
        
        # Crystal deformation
        if deformation > 0:
            strain = deformation * np.sin(8 * Theta) * np.exp(-0.5 * R)
            field += strain
        
        return gaussian_filter(field, sigma=0.9)
    
    def _tunneling_veil(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Tunneling Veil: Quantum tunneling probability clouds."""
        # Potential barriers
        barriers = energy * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        
        # Tunneling probability clouds
        tunneling = energy * 0.8 * np.exp(-R) * np.sin(symmetry * Theta / 10)
        
        # Wave function interference
        k = 2 * np.pi / 2.0
        wave1 = np.sin(k * (X + Y)) * np.exp(-0.1 * R)
        wave2 = np.sin(k * (X - Y)) * np.exp(-0.1 * R)
        interference = energy * 0.5 * (wave1 + wave2)
        
        field = barriers + tunneling + interference
        
        # Add deformation effects
        if deformation > 0:
            veil = deformation * np.sin(3 * R) * np.cos(5 * Theta) * np.exp(-0.2 * R)
            field += veil
        
        return gaussian_filter(field, sigma=1.1)
    
    def _apply_color_variation(self, field, variation):
        """Apply color variation effects to the field."""
        if variation > 0:
            # Add spectral variations
            variation_field = variation * np.random.normal(0, 0.1, field.shape)
            field = field + gaussian_filter(variation_field, sigma=2.0)
        
        return field
    
    def _render_field(self, field, style, energy, color_variation):
        """Render the quantum field as a beautiful visualization."""
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        
        # Choose color scheme based on style and parameters
        cmap = self._get_color_scheme(style, color_variation)
        
        # Main field visualization
        im = ax.imshow(field, extent=[-4, 4, -4, 4], cmap=cmap, 
                      interpolation='bilinear', alpha=0.9)
        
        # Add contour lines for quantum levels
        levels = np.linspace(field.min(), field.max(), 12)
        contours = ax.contour(np.linspace(-4, 4, field.shape[0]), 
                            np.linspace(-4, 4, field.shape[1]), 
                            field, levels=levels, colors='white', 
                            alpha=0.2, linewidths=0.5)
        
        # Style the plot
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis('off')
        
        # Add subtle glow effect
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
    
    def _get_color_scheme(self, style, variation):
        """Get appropriate color scheme for each style."""
        schemes = {
            "Quantum Bloom": self._create_bloom_colormap(variation),
            "Singularity Core": self._create_singularity_colormap(variation),
            "Entanglement Field": self._create_entanglement_colormap(variation),
            "Crystal Spire": self._create_crystal_colormap(variation),
            "Tunneling Veil": self._create_veil_colormap(variation)
        }
        return schemes.get(style, plt.cm.plasma)
    
    def _create_bloom_colormap(self, variation):
        if variation > 0.5:
            colors = ['#000011', '#1a0033', '#4d0080', '#8000ff', '#bf80ff', '#e6ccff']
        else:
            colors = ['#000022', '#003366', '#0066cc', '#3399ff', '#80ccff', '#cceeff']
        return LinearSegmentedColormap.from_list('bloom', colors)
    
    def _create_singularity_colormap(self, variation):
        if variation > 0.5:
            colors = ['#000000', '#1a0000', '#660000', '#cc0000', '#ff6600', '#ffcc00']
        else:
            colors = ['#000000', '#0a0a0a', '#333333', '#666666', '#999999', '#ffffff']
        return LinearSegmentedColormap.from_list('singularity', colors)
    
    def _create_entanglement_colormap(self, variation):
        colors = ['#000033', '#1a0066', '#4d00cc', '#8000ff', '#cc66ff', '#ffccff']
        return LinearSegmentedColormap.from_list('entanglement', colors)
    
    def _create_crystal_colormap(self, variation):
        if variation > 0.5:
            colors = ['#001122', '#003366', '#0066aa', '#0099dd', '#33bbff', '#66ddff']
        else:
            colors = ['#002211', '#004433', '#006655', '#009977', '#33cc99', '#66ffbb']
        return LinearSegmentedColormap.from_list('crystal', colors)
    
    def _create_veil_colormap(self, variation):
        colors = ['#110022', '#330044', '#660088', '#9900cc', '#cc33ff', '#ff99ff']
        return LinearSegmentedColormap.from_list('veil', colors)


def create_download_link(img, filename):
    """Create a download link for the generated image."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">✨ Download PNG</a>'
    return href


def main():
    """Main Streamlit app."""
    
    # Load custom CSS
    load_css()
    
    # Initialize the generator
    if 'generator' not in st.session_state:
        st.session_state.generator = QuantumArtGenerator()
    
    # Initialize generated image state
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    
    # Main title and subtitle
    st.markdown('<h1 class="main-title">Quantum Geometry Art Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Create visuals powered by quantum curvature and emergent geometry.</p>', unsafe_allow_html=True)
    
    # Sidebar with controls
    with st.sidebar:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown('<div class="parameter-title">🎨 Art Style</div>', unsafe_allow_html=True)
        
        style = st.selectbox(
            "Choose base geometry:",
            ["Quantum Bloom", "Singularity Core", "Entanglement Field", "Crystal Spire", "Tunneling Veil"],
            help="Each style represents different quantum phenomena"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.markdown('<div class="parameter-title">⚛️ Quantum Parameters</div>', unsafe_allow_html=True)
        
        energy_intensity = st.slider(
            "Energy Intensity",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Controls the amplitude of quantum field fluctuations"
        )
        
        symmetry_level = st.slider(
            "Symmetry Level",
            min_value=0,
            max_value=100,
            value=50,
            help="Adjusts rotational and translational symmetries"
        )
        
        deformation_amount = st.slider(
            "Deformation Amount",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Amount of spacetime curvature deformation"
        )
        
        color_variation = st.slider(
            "Color Variation",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Spectral variation and color complexity"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🚀 Generate Artwork", type="primary", use_container_width=True):
            with st.spinner("Calculating quantum fields..."):
                # Add a small delay for the loading effect
                time.sleep(0.5)
                
                # Generate the artwork
                st.session_state.generated_image = st.session_state.generator.generate_quantum_image(
                    style=style,
                    energy_intensity=energy_intensity,
                    symmetry_level=symmetry_level,
                    deformation_amount=deformation_amount,
                    color_variation=color_variation,
                    resolution=512
                )
                
                st.success("✨ Artwork generated!")
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.generated_image is not None:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.generated_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            filename = f"quantum_art_{style.lower().replace(' ', '_')}_{int(time.time())}.png"
            download_link = create_download_link(st.session_state.generated_image, filename)
            st.markdown(f'<div style="text-align: center; margin-top: 1rem;">{download_link}</div>', 
                       unsafe_allow_html=True)
            
            # Display current parameters
            st.markdown("---")
            st.markdown("### Current Parameters")
            
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                st.metric("Style", style)
                st.metric("Energy", f"{energy_intensity:.1f}")
            with param_col2:
                st.metric("Symmetry", f"{symmetry_level}")
                st.metric("Deformation", f"{deformation_amount:.2f}")
        
        else:
            # Placeholder content
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #666;">
                <h3>🌌 Ready to Create Quantum Art</h3>
                <p>Adjust the parameters in the sidebar and click "Generate Artwork" to create your unique quantum geometry visualization.</p>
                <br>
                <p><em>Each artwork is generated using real physics equations from quantum field theory and differential geometry.</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Built with ⚛️ by the Quantum Art Team | 
        <a href="https://github.com/quantumproject" target="_blank">View on GitHub</a></p>
        <p><em>Powered by quantum field theory, differential geometry, and emergent physics</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 