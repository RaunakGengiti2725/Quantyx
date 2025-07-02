#!/usr/bin/env python3
"""
Quantum Geometry Art Generator - Professional Creative Platform
A cutting-edge web app for generating physics-inspired art with professional features.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import io
import base64
from PIL import Image, ImageEnhance
import time
import json
from datetime import datetime
import imageio
from io import BytesIO
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Quantyx - AI Image Generation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with animations
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 35%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%);
        animation: glow 20s ease-in-out infinite alternate;
        z-index: -1;
    }
    
    @keyframes glow {
        0% { opacity: 0.3; }
        50% { opacity: 0.5; }
        100% { opacity: 0.3; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes rocket {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInUp 1.5s ease-out;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .image-container {
        animation: slideIn 0.8s ease-out;
    }
    
    .image-container img {
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .image-container img:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.7);
    }
    
    .generate-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .generate-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: rocket 1s ease-in-out infinite;
    }
    
    .generate-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .generate-button:hover::before {
        left: 100%;
    }
    
    .download-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 16px;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .download-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .download-button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.5);
        text-decoration: none;
        color: white;
    }
    
    .download-button:hover::before {
        left: 100%;
    }
    
    .download-button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    .preset-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: slideIn 0.6s ease-out;
    }
    
    .preset-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .resolution-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        margin-top: 3rem;
    }
    
    .slider-active {
        animation: pulse 1s ease-in-out 3;
    }
    
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        animation: slideIn 0.5s ease-out, pulse 2s ease-in-out infinite;
        margin-bottom: 1rem;
    }
    
    .caption-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        font-style: italic;
        color: #c0c0c0;
        animation: slideIn 0.7s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)


# Color Palettes
COLOR_PALETTES = {
    "Deep Plasma": {
        "description": "High energy fields (purple, neon)",
        "colors": {
            "Quantum Bloom": ['#000022', '#1a0066', '#4d00cc', '#8000ff', '#cc66ff', '#ffccff'],
            "Singularity Core": ['#000033', '#330066', '#660099', '#9900cc', '#cc33ff', '#ff99ff'],
            "Entanglement Field": ['#110033', '#330077', '#5500bb', '#7700ff', '#aa44ff', '#dd88ff'],
            "Crystal Spire": ['#001144', '#002288', '#0044cc', '#0066ff', '#4499ff', '#88ccff'],
            "Tunneling Veil": ['#220044', '#440088', '#6600cc', '#8844ff', '#aa66ff', '#cc99ff']
        }
    },
    "Ice Flux": {
        "description": "Cold, symmetrical structures (blue/white)",
        "colors": {
            "Quantum Bloom": ['#000033', '#001166', '#002299', '#0044cc', '#3377ff', '#66aaff'],
            "Singularity Core": ['#001122', '#003355', '#005588', '#0077bb', '#2299dd', '#55bbff'],
            "Entanglement Field": ['#000044', '#001177', '#0033aa', '#0055dd', '#3388ff', '#66bbff'],
            "Crystal Spire": ['#001155', '#002288', '#0044bb', '#0066ee', '#3399ff', '#66ccff'],
            "Tunneling Veil": ['#000066', '#002299', '#0044cc', '#0066ff', '#4499ff', '#77ccff']
        }
    },
    "Solar Burst": {
        "description": "Warm, chaotic deformation (orange/red)",
        "colors": {
            "Quantum Bloom": ['#220000', '#550000', '#880011', '#bb2200', '#ee5500', '#ff8833'],
            "Singularity Core": ['#330000', '#660000', '#990000', '#cc2200', '#ff5500', '#ff8844'],
            "Entanglement Field": ['#331100', '#662200', '#993300', '#cc5500', '#ff7700', '#ff9955'],
            "Crystal Spire": ['#221100', '#553300', '#885500', '#bb7700', '#ee9900', '#ffbb44'],
            "Tunneling Veil": ['#442200', '#775500', '#aa7700', '#dd9900', '#ffbb00', '#ffdd66']
        }
    }
}

# Visual Presets
VISUAL_PRESETS = {
    "Fractal Bloom": {
        "style": "Quantum Bloom",
        "energy_intensity": 8.5,
        "symmetry_level": 85,
        "deformation_amount": 0.6,
        "color_variation": 0.8,
        "description": "Organic fractal patterns with vibrant energy blooms"
    },
    "Energetic Grid": {
        "style": "Crystal Spire",
        "energy_intensity": 9.0,
        "symmetry_level": 95,
        "deformation_amount": 0.2,
        "color_variation": 0.3,
        "description": "High-energy crystalline lattice structures"
    },
    "Liquid Collapse": {
        "style": "Singularity Core",
        "energy_intensity": 7.5,
        "symmetry_level": 15,
        "deformation_amount": 0.9,
        "color_variation": 0.7,
        "description": "Fluid dynamics meeting gravitational collapse"
    },
    "Frozen Singularity": {
        "style": "Singularity Core",
        "energy_intensity": 6.0,
        "symmetry_level": 75,
        "deformation_amount": 0.1,
        "color_variation": 0.2,
        "description": "Crystallized black hole with minimal deformation"
    },
    "Ethereal Entanglement": {
        "style": "Entanglement Field",
        "energy_intensity": 5.5,
        "symmetry_level": 60,
        "deformation_amount": 0.7,
        "color_variation": 0.9,
        "description": "Flowing quantum entanglement with ethereal colors"
    },
    "Mystic Veil": {
        "style": "Tunneling Veil",
        "energy_intensity": 4.0,
        "symmetry_level": 40,
        "deformation_amount": 0.8,
        "color_variation": 0.6,
        "description": "Mysterious quantum tunneling effects"
    }
}


class QuantumArtGenerator:
    """Advanced quantum geometry art generator with professional features."""
    
    def __init__(self):
        self.styles = {
            "Quantum Bloom": self._quantum_bloom,
            "Singularity Core": self._singularity_core,
            "Entanglement Field": self._entanglement_field,
            "Crystal Spire": self._crystal_spire,
            "Tunneling Veil": self._tunneling_veil
        }
    
    def generate_quantum_image(self, style, energy_intensity, symmetry_level, 
                             deformation_amount, color_variation, resolution=512, color_palette="Deep Plasma"):
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
        
        # Apply color variation
        if color_variation > 0:
            variation_field = color_variation * np.random.normal(0, 0.1, field.shape)
            field = field + gaussian_filter(variation_field, sigma=2.0)
        
        # Create visualization
        return self._render_field(field, style, color_variation, resolution, color_palette)
    
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
        """Singularity Core: Black hole-like gravitational effects."""
        singularity = energy * np.exp(-2 * R) / (R + 0.1)
        disk = np.sin(symmetry * Theta / 10) * np.exp(-0.5 * R) * (1 / (R + 0.1))
        curvature = energy * 0.3 * np.sin(8 * np.pi * R) / (R + 0.5)
        
        field = singularity + 0.7 * disk + 0.4 * curvature
        
        if deformation > 0:
            warp = deformation * np.sin(R * 2 * np.pi) * np.cos(4 * Theta)
            field += warp
        
        return gaussian_filter(field, sigma=0.8)
    
    def _entanglement_field(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Entanglement Field: Quantum entanglement visualization."""
        pairs = int(symmetry / 20) + 2
        field = np.zeros_like(X)
        
        for i in range(pairs):
            angle1 = 2 * np.pi * i / pairs
            angle2 = angle1 + np.pi
            
            x1, y1 = 2 * np.cos(angle1), 2 * np.sin(angle1)
            x2, y2 = 2 * np.cos(angle2), 2 * np.sin(angle2)
            
            r1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
            field1 = energy * np.exp(-2 * r1) * np.sin(4 * np.pi * r1)
            
            r2 = np.sqrt((X - x2)**2 + (Y - y2)**2)
            field2 = -energy * np.exp(-2 * r2) * np.sin(4 * np.pi * r2)
            
            field += field1 + field2
        
        connection = energy * 0.2 * np.sin(6 * Theta) * np.exp(-0.3 * R)
        field += connection
        
        if deformation > 0:
            field += deformation * np.sin(X + Y) * np.cos(X - Y)
        
        return gaussian_filter(field, sigma=1.2)
    
    def _crystal_spire(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Crystal Spire: Crystalline growth patterns."""
        lattice_spacing = 4.0 / max(1, symmetry / 10)
        crystal_x = np.sin(2 * np.pi * X / lattice_spacing)
        crystal_y = np.sin(2 * np.pi * Y / lattice_spacing)
        lattice = energy * crystal_x * crystal_y * np.exp(-0.2 * R)
        
        spires = energy * 0.7 * np.sin(4 * Theta) * np.exp(-0.4 * R) / (R + 0.1)
        facets = np.sin(6 * Theta) * np.cos(3 * np.pi * R) * np.exp(-0.3 * R)
        
        field = lattice + spires + 0.4 * energy * facets
        
        if deformation > 0:
            strain = deformation * np.sin(8 * Theta) * np.exp(-0.5 * R)
            field += strain
        
        return gaussian_filter(field, sigma=0.9)
    
    def _tunneling_veil(self, X, Y, R, Theta, energy, symmetry, deformation):
        """Tunneling Veil: Quantum tunneling probability clouds."""
        barriers = energy * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        tunneling = energy * 0.8 * np.exp(-R) * np.sin(symmetry * Theta / 10)
        
        k = 2 * np.pi / 2.0
        wave1 = np.sin(k * (X + Y)) * np.exp(-0.1 * R)
        wave2 = np.sin(k * (X - Y)) * np.exp(-0.1 * R)
        interference = energy * 0.5 * (wave1 + wave2)
        
        field = barriers + tunneling + interference
        
        if deformation > 0:
            veil = deformation * np.sin(3 * R) * np.cos(5 * Theta) * np.exp(-0.2 * R)
            field += veil
        
        return gaussian_filter(field, sigma=1.1)
    
    def _render_field(self, field, style, color_variation, resolution, color_palette="Deep Plasma"):
        """Render the quantum field as a visualization."""
        dpi = 150 if resolution <= 512 else 300 if resolution <= 1024 else 450
        fig_size = (12, 12) if resolution > 512 else (10, 10)
        
        fig, ax = plt.subplots(figsize=fig_size, facecolor='black')
        ax.set_facecolor('black')
        
        # Choose color scheme
        cmap = self._get_color_scheme(style, color_variation, color_palette)
        
        # Main field visualization
        im = ax.imshow(field, extent=[-4, 4, -4, 4], cmap=cmap, 
                      interpolation='bilinear', alpha=0.9)
        
        # Add contour lines
        levels = np.linspace(field.min(), field.max(), 12)
        contours = ax.contour(np.linspace(-4, 4, field.shape[0]), 
                            np.linspace(-4, 4, field.shape[1]), 
                            field, levels=levels, colors='white', 
                            alpha=0.2, linewidths=0.5)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        plt.close()
        
        image = Image.open(buf)
        
        # Apply post-processing for higher resolutions
        if resolution > 512:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
        
        return image
    
    def _get_color_scheme(self, style, variation, palette="Deep Plasma"):
        """Get color scheme for each style using selected palette."""
        # Get colors from selected palette
        if palette in COLOR_PALETTES and style in COLOR_PALETTES[palette]["colors"]:
            colors = COLOR_PALETTES[palette]["colors"][style]
        else:
            # Fallback to Deep Plasma
            colors = COLOR_PALETTES["Deep Plasma"]["colors"][style]
        
        return LinearSegmentedColormap.from_list(f'{style.lower().replace(" ", "_")}', colors)
    
    def generate_caption(self, style, energy, symmetry, deformation, color_variation):
        """Generate artistic caption for the artwork."""
        style_descriptions = {
            "Quantum Bloom": "blooming with quantum interference",
            "Singularity Core": "collapsing into gravitational singularity",
            "Entanglement Field": "dancing with quantum entanglement",
            "Crystal Spire": "crystallizing in perfect geometric harmony",
            "Tunneling Veil": "tunneling through quantum probability"
        }
        
        energy_desc = "intense" if energy > 7 else "moderate" if energy > 4 else "gentle"
        symmetry_desc = "perfectly ordered" if symmetry > 80 else "balanced" if symmetry > 40 else "chaotically beautiful"
        deform_desc = "with flowing deformations" if deformation > 0.6 else "with subtle warping" if deformation > 0.3 else "in pristine form"
        color_desc = "vibrant colors" if color_variation > 0.7 else "harmonious hues" if color_variation > 0.4 else "subtle tones"
        
        return f"An {energy_desc} quantum field {style_descriptions[style]}, {symmetry_desc} {deform_desc}, painted in {color_desc}."


def create_download_link(img, filename, resolution):
    """Create download link for generated image."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    file_size = len(buf.getvalue())
    size_mb = file_size / (1024 * 1024)
    
    b64 = base64.b64encode(buf.read()).decode()
    
    res_badge = ""
    if resolution >= 3840:
        res_badge = '<span class="resolution-badge">4K</span>'
    elif resolution >= 1024:
        res_badge = '<span class="resolution-badge">HD</span>'
    else:
        res_badge = '<span class="resolution-badge">STD</span>'
    
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">✨ Download PNG {res_badge} ({size_mb:.1f}MB)</a>'
    return href


def apply_preset(preset_name):
    """Apply a visual preset to the session state."""
    preset = VISUAL_PRESETS[preset_name]
    st.session_state.current_style = preset["style"]
    st.session_state.current_energy = preset["energy_intensity"]
    st.session_state.current_symmetry = preset["symmetry_level"]
    st.session_state.current_deformation = preset["deformation_amount"]
    st.session_state.current_color_variation = preset["color_variation"]
    st.session_state.preset_applied = preset_name

def randomize_parameters():
    """Randomize all parameters for creative exploration."""
    import random
    styles = ["Quantum Bloom", "Singularity Core", "Entanglement Field", "Crystal Spire", "Tunneling Veil"]
    
    st.session_state.current_style = random.choice(styles)
    st.session_state.current_energy = round(random.uniform(2.0, 9.0), 1)
    st.session_state.current_symmetry = random.randint(10, 95)
    st.session_state.current_deformation = round(random.uniform(0.1, 0.9), 2)
    st.session_state.current_color_variation = round(random.uniform(0.2, 0.8), 2)
    st.session_state.randomized = True

def generate_animation_frames(generator, style, base_energy, base_symmetry, base_deformation, 
                            base_color_variation, color_palette, resolution, animation_params):
    """Generate multiple frames for animation by varying parameters over time."""
    frames = []
    frame_count = animation_params['frame_count']
    animate_param = animation_params['animate_param']
    variation_range = animation_params['variation_range']
    
    for i in range(frame_count):
        # Calculate animation progress (0 to 1 and back for seamless loop)
        if i < frame_count // 2:
            progress = i / (frame_count // 2)
        else:
            progress = (frame_count - i) / (frame_count // 2)
        
        # Vary the selected parameter
        if animate_param == "Energy Flux":
            energy = base_energy + (progress - 0.5) * variation_range * 2
            energy = max(0.1, min(10.0, energy))
            frame = generator.generate_quantum_image(style, energy, base_symmetry, 
                                                   base_deformation, base_color_variation, 
                                                   resolution, color_palette)
        elif animate_param == "Geometric Order":
            symmetry = base_symmetry + (progress - 0.5) * variation_range * 100
            symmetry = max(0, min(100, int(symmetry)))
            frame = generator.generate_quantum_image(style, base_energy, symmetry, 
                                                   base_deformation, base_color_variation, 
                                                   resolution, color_palette)
        elif animate_param == "Field Distortion":
            deformation = base_deformation + (progress - 0.5) * variation_range
            deformation = max(0.0, min(1.0, deformation))
            frame = generator.generate_quantum_image(style, base_energy, base_symmetry, 
                                                   deformation, base_color_variation, 
                                                   resolution, color_palette)
        elif animate_param == "Spectral Blend":
            color_var = base_color_variation + (progress - 0.5) * variation_range
            color_var = max(0.0, min(1.0, color_var))
            frame = generator.generate_quantum_image(style, base_energy, base_symmetry, 
                                                   base_deformation, color_var, 
                                                   resolution, color_palette)
        
        frames.append(frame)
    
    return frames

def create_gif_from_frames(frames, duration_per_frame=0.1):
    """Create an animated GIF from a list of PIL images."""
    # Convert PIL images to numpy arrays for imageio
    frame_arrays = []
    for frame in frames:
        frame_array = np.array(frame)
        frame_arrays.append(frame_array)
    
    # Create GIF in memory
    gif_buffer = BytesIO()
    imageio.mimsave(gif_buffer, frame_arrays, format='GIF', duration=duration_per_frame, loop=0)
    gif_buffer.seek(0)
    
    return gif_buffer

def create_download_link_gif(gif_buffer, filename):
    """Create download link for animated GIF."""
    gif_data = gif_buffer.getvalue()
    file_size_mb = len(gif_data) / (1024 * 1024)
    
    b64 = base64.b64encode(gif_data).decode()
    href = f'<a href="data:image/gif;base64,{b64}" download="{filename}" class="download-button">🎬 Download Animated GIF ({file_size_mb:.1f}MB)</a>'
    return href


def main():
    """Main Streamlit app."""
    
    # Load custom CSS
    load_css()
    
    # Initialize generator and session state
    if 'generator' not in st.session_state:
        st.session_state.generator = QuantumArtGenerator()
    
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    
    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False
    
    if 'artwork_caption' not in st.session_state:
        st.session_state.artwork_caption = ""
    
    # Initialize animation session state
    if 'animation_frames' not in st.session_state:
        st.session_state.animation_frames = None
    
    if 'animation_gif' not in st.session_state:
        st.session_state.animation_gif = None
    
    # Initialize current parameters
    for param in ['current_style', 'current_energy', 'current_symmetry', 'current_deformation', 'current_color_variation']:
        if param not in st.session_state:
            if param == 'current_style':
                st.session_state[param] = "Quantum Bloom"
            elif param == 'current_energy':
                st.session_state[param] = 5.0
            elif param == 'current_symmetry':
                st.session_state[param] = 50
            elif param == 'current_deformation':
                st.session_state[param] = 0.3
            elif param == 'current_color_variation':
                st.session_state[param] = 0.5
    
    # Main title with animated background
    st.markdown("""
    <div style="position: relative; overflow: hidden; text-align: center; padding: 3rem 0;
                -webkit-mask: linear-gradient(90deg, transparent 0%, black 15%, black 85%, transparent 100%);
                mask: linear-gradient(90deg, transparent 0%, black 15%, black 85%, transparent 100%);">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                    background: radial-gradient(circle at 30% 70%, rgba(255, 0, 150, 0.2) 0%, transparent 50%),
                               radial-gradient(circle at 70% 30%, rgba(0, 255, 200, 0.2) 0%, transparent 50%),
                               radial-gradient(circle at center, rgba(102, 126, 234, 0.15) 0%, transparent 70%);
                    animation: pulse 4s ease-in-out infinite;"></div>
        <h1 style="font-size: 4rem; font-weight: 700; 
                   background: linear-gradient(135deg, #ff0099 0%, #00ffcc 25%, #667eea 50%, #ff6b35 75%, #f093fb 100%);
                   background-size: 300% 300%;
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   margin-bottom: 1rem; position: relative; z-index: 1;
                   animation: fadeInUp 1.5s ease-out, gradientShift 6s ease-in-out infinite;
                   text-shadow: 0 0 30px rgba(255, 0, 150, 0.3), 0 0 60px rgba(0, 255, 200, 0.2);
                   filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.4));">
            ⚛️ Quantyx
        </h1>
        <p style="font-size: 1.3rem; 
                  background: linear-gradient(90deg, #ffffff 0%, #00ffcc 50%, #ffffff 100%);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  margin-bottom: 0.5rem; position: relative; z-index: 1;
                  animation: fadeInUp 1.8s ease-out;">
            AI Image Generation Platform
        </p>
        <p style="font-size: 1rem; color: #e0e0ff; position: relative; z-index: 1;
                  animation: fadeInUp 2.1s ease-out;
                  text-shadow: 0 0 10px rgba(224, 224, 255, 0.3);">
            Physics-Inspired Creativity • Export Ready • Professional Quality
        </p>
    </div>
    <style>
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar controls for art generation
    with st.sidebar:
        st.markdown("### 🎨 Quick Presets")
        
        # Visual presets in a more organized layout
        preset_col1, preset_col2 = st.columns(2)
        preset_items = list(VISUAL_PRESETS.items())
        
        for i in range(0, len(preset_items), 2):
            with preset_col1:
                if i < len(preset_items):
                    preset_name, preset_data = preset_items[i]
                    if st.button(preset_name, key=f"preset_{i}", help=preset_data["description"], use_container_width=True):
                        apply_preset(preset_name)
                        st.rerun()
            
            with preset_col2:
                if i + 1 < len(preset_items):
                    preset_name, preset_data = preset_items[i + 1]
                    if st.button(preset_name, key=f"preset_{i+1}", help=preset_data["description"], use_container_width=True):
                        apply_preset(preset_name)
                        st.rerun()
        
        # Randomize button
        if st.button("🎲 Surprise Me", help="Randomize all parameters for creative exploration", use_container_width=True, type="secondary"):
            randomize_parameters()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🌈 Spectral Palette")
        
        # Color palette selector
        color_palette = st.selectbox(
            "Choose Color Mapping:",
            list(COLOR_PALETTES.keys()),
            help="Select the color palette that maps to your quantum field energy"
        )
        
        # Show palette description
        st.info(f"💡 {COLOR_PALETTES[color_palette]['description']}")
        
        st.markdown("---")
        st.markdown("### 🎛️ Geometry Blueprint")
        
        style = st.selectbox(
            "Quantum Structure:",
            ["Quantum Bloom", "Singularity Core", "Entanglement Field", "Crystal Spire", "Tunneling Veil"],
            index=["Quantum Bloom", "Singularity Core", "Entanglement Field", "Crystal Spire", "Tunneling Veil"].index(st.session_state.current_style)
        )
        st.session_state.current_style = style
        
        st.markdown("### ⚛️ Field Parameters")
        
        energy_intensity = st.slider("Energy Flux", 0.0, 10.0, st.session_state.current_energy, 0.1, help="Controls the intensity of quantum field energy")
        st.session_state.current_energy = energy_intensity
        
        symmetry_level = st.slider("Geometric Order", 0, 100, st.session_state.current_symmetry, help="Symmetry level in the quantum structure")
        st.session_state.current_symmetry = symmetry_level
        
        deformation_amount = st.slider("Field Distortion", 0.0, 1.0, st.session_state.current_deformation, 0.01, help="Amount of spacetime curvature deformation")
        st.session_state.current_deformation = deformation_amount
        
        color_variation = st.slider("Spectral Blend", 0.0, 1.0, st.session_state.current_color_variation, 0.01, help="Color variation across the quantum field")
        st.session_state.current_color_variation = color_variation
        
        st.markdown("### 📐 Output Resolution")
        
        resolution_option = st.selectbox(
            "Quality:",
            ["Standard (512×512)", "HD (1024×1024)", "4K (3840×2160)", "Print (4096×4096)"]
        )
        
        resolution_map = {
            "Standard (512×512)": 512,
            "HD (1024×1024)": 1024,
            "4K (3840×2160)": 3840,
            "Print (4096×4096)": 4096
        }
        resolution = resolution_map[resolution_option]
        
        # Show estimated generation time
        time_estimates = {512: "~1s", 1024: "~3s", 3840: "~15s", 4096: "~20s"}
        st.info(f"⏱️ Est. time: {time_estimates.get(resolution, '~?s')}")
        
        # Animated generate button
        st.markdown("---")
        if st.button("⚛️ Run Simulation", type="primary", use_container_width=True):
            with st.spinner("🌌 Simulating quantum field dynamics..."):
                start_time = time.time()
                
                # Store current palette for generation
                st.session_state.current_palette = color_palette
                
                st.session_state.generated_image = st.session_state.generator.generate_quantum_image(
                    style=style,
                    energy_intensity=energy_intensity,
                    symmetry_level=symmetry_level,
                    deformation_amount=deformation_amount,
                    color_variation=color_variation,
                    resolution=resolution,
                    color_palette=color_palette
                )
                
                # Generate caption
                st.session_state.artwork_caption = st.session_state.generator.generate_caption(
                    style, energy_intensity, symmetry_level, deformation_amount, color_variation
                )
                
                generation_time = time.time() - start_time
                st.session_state.generation_time = generation_time
                st.session_state.generation_complete = True
                st.session_state.current_resolution = resolution
                
                st.rerun()
    
    # Main content area for art generation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Show image only when generation is complete
        if st.session_state.generation_complete and st.session_state.generated_image is not None:
            # Image display first
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.generated_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Success message with timing (appears after image)
            st.markdown(f"""
            <div class="success-message">
                ⚛️ Quantum Structure Simulated! ({st.session_state.generation_time:.2f}s)
            </div>
            """, unsafe_allow_html=True)
            
            # Caption
            if st.session_state.artwork_caption:
                st.markdown(f"""
                <div class="caption-box">
                    "{st.session_state.artwork_caption}"
                </div>
                """, unsafe_allow_html=True)
            
            # Download button
            timestamp = int(time.time())
            filename = f"quantyx_{style.lower().replace(' ', '_')}_{timestamp}.png"
            download_link = create_download_link(st.session_state.generated_image, filename, st.session_state.current_resolution)
            st.markdown(f'<div style="text-align: center; margin: 1.5rem 0;">{download_link}</div>', 
                       unsafe_allow_html=True)
            
            # Export metadata option
            if st.checkbox("📋 Include Metadata", help="Export generation parameters as JSON"):
                try:
                    metadata = {
                        "style": str(style),
                        "parameters": {
                            "energy_intensity": float(energy_intensity),
                            "symmetry_level": int(symmetry_level),
                            "deformation_amount": float(deformation_amount),
                            "color_variation": float(color_variation)
                        },
                        "resolution": f"{st.session_state.current_resolution}×{st.session_state.current_resolution}",
                        "caption": str(st.session_state.artwork_caption),
                        "generated_at": datetime.now().isoformat(),
                        "generation_time": f"{st.session_state.generation_time:.2f}s"
                    }
                    
                    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Download Metadata (JSON)",
                        data=metadata_json,
                        file_name=f"quantyx_metadata_{timestamp}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error creating metadata: {str(e)}")
            
            # Technical parameters display
            st.markdown("---")
            st.markdown("### 🔬 Technical Parameters")
            
            param_col1, param_col2, param_col3, param_col4 = st.columns(4)
            with param_col1:
                st.metric("Style", style)
                st.metric("Energy", f"{energy_intensity:.1f}")
            with param_col2:
                st.metric("Symmetry", f"{symmetry_level}")
                st.metric("Deformation", f"{deformation_amount:.2f}")
            with param_col3:
                st.metric("Color Var.", f"{color_variation:.2f}")
                st.metric("Resolution", f"{st.session_state.current_resolution}px")
            with param_col4:
                st.metric("Gen. Time", f"{st.session_state.generation_time:.2f}s")
                pixels = st.session_state.current_resolution ** 2
                st.metric("Total Pixels", f"{pixels:,}")
        
        else:
            # Welcome screen with instructions (only shown before first generation)
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #fff;">
                <h3 style="color: #fff;">🌌 Welcome to Quantyx</h3>
                <p style="font-size: 1.1rem; margin: 1.5rem 0; color: #fff;">Choose a preset or adjust parameters manually, then generate your quantum masterpiece.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature cards in columns
            feat_col1, feat_col2, feat_col3 = st.columns(3)
            
            with feat_col1:
                st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;">
                    <h4 style="color: #fff; margin-bottom: 1rem;">⚡ Quick Presets</h4>
                    <p style="color: #fff;">Professional presets designed by artists</p>
                </div>
                """, unsafe_allow_html=True)
            
            with feat_col2:
                st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;">
                    <h4 style="color: #fff; margin-bottom: 1rem;">🎛️ Manual Control</h4>
                    <p style="color: #fff;">Fine-tune every physics parameter</p>
                </div>
                """, unsafe_allow_html=True)
            
            with feat_col3:
                st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;">
                    <h4 style="color: #fff; margin-bottom: 1rem;">📐 Pro Export</h4>
                    <p style="color: #fff;">HD, 4K, and print-ready resolutions</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #fff;">
                <p style="color: #fff;"><em>Each artwork uses real equations from quantum field theory and differential geometry.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Footer (only shown on welcome screen)
            st.markdown("---")
            st.markdown("""
            <div class="footer">
                <p style="color: #fff;">Built with ⚛️ Quantum Physics & ❤️ by the Quantyx Team</p>
                <p style="color: #fff;"><em>Quantyx • Professional Quantum Art Platform • Export Ready</em></p>
                <p style="font-size: 0.9rem; color: #fff;">
                    Powered by quantum field theory, differential geometry, and advanced visualization algorithms
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Navigation hint for animation page
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin: 1rem 0;">
        <h4 style="color: #fff; margin-bottom: 1rem;">🎬 Unlock Quantyx's Animation Power</h4>
        <p style="color: #e6e6ff; margin-bottom: 0;">Visit the <strong>Animate Fields</strong> page to create mesmerizing looping animations perfect for VJing and social sharing!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show notifications
    if 'preset_applied' in st.session_state and st.session_state.preset_applied:
        st.success(f"🎨 Applied preset: {st.session_state.preset_applied}")
        del st.session_state.preset_applied
    
    if 'randomized' in st.session_state and st.session_state.randomized:
        st.info(f"🎲 Parameters randomized! Try generating to see your surprise creation.")
        del st.session_state.randomized
    



if __name__ == "__main__":
    main() 