#!/usr/bin/env python3
"""
Quantum Geometry Art Generator - Animation Mode
Multi-Frame Animation Mode for creating looping quantum field visualizations.
Perfect for VJing, music videos, and social sharing.
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
import sys

# Add parent directory to path to import from main app
import os.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from streamlit_quantum_app import (
        COLOR_PALETTES, QuantumArtGenerator, load_css,
        create_gif_from_frames, create_download_link_gif
    )
except ImportError:
    st.error("Error importing required modules. Please ensure the main app file is available.")
    st.stop()

# Note: st.set_page_config() is handled by the main app, not individual pages

def main():
    """Animation Studio main interface."""
    
    # Load custom CSS
    load_css()
    
    # Initialize generator and session state
    if 'generator' not in st.session_state:
        st.session_state.generator = QuantumArtGenerator()
    
    # Initialize animation session state
    if 'animation_frames' not in st.session_state:
        st.session_state.animation_frames = None
    
    if 'animation_gif' not in st.session_state:
        st.session_state.animation_gif = None
    

    
    # Animation controls in sidebar
    with st.sidebar:
        st.markdown("## 🎬 Animation Settings")
        
        # Base parameters for animation
        st.markdown("### 🎯 Base Structure")
        anim_style = st.selectbox(
            "Quantum Structure:",
            ["Quantum Bloom", "Singularity Core", "Entanglement Field", "Crystal Spire", "Tunneling Veil"],
            help="Base quantum field geometry for animation"
        )
        
        anim_palette = st.selectbox(
            "Color Palette:",
            list(COLOR_PALETTES.keys()),
            help="Color scheme for the animation"
        )
        
        # Show palette description
        st.info(f"💡 {COLOR_PALETTES[anim_palette]['description']}")
        
        st.markdown("### ⚛️ Base Parameters")
        
        base_energy = st.slider("Base Energy Flux", 0.0, 10.0, 5.0, 0.1, help="Starting energy level")
        base_symmetry = st.slider("Base Geometric Order", 0, 100, 50, help="Starting symmetry level")
        base_deformation = st.slider("Base Field Distortion", 0.0, 1.0, 0.3, 0.01, help="Starting deformation amount")
        base_color_variation = st.slider("Base Spectral Blend", 0.0, 1.0, 0.5, 0.01, help="Starting color variation")
        
        st.markdown("### 🎭 Animation Control")
        
        animate_param = st.selectbox(
            "Animate Parameter:",
            ["Energy Flux", "Geometric Order", "Field Distortion", "Spectral Blend"],
            help="Which parameter to animate over time"
        )
        
        variation_range = st.slider(
            "Animation Intensity", 
            0.1, 2.0, 0.8, 0.1,
            help="How dramatically the parameter changes"
        )
        
        st.markdown("### ⏱️ Timing Control")
        
        frame_count = st.slider(
            "Frame Count", 
            8, 30, 16,
            help="More frames = smoother animation but larger file"
        )
        
        frame_duration = st.slider(
            "Frame Duration (seconds)", 
            0.05, 0.5, 0.1, 0.01,
            help="How long each frame displays"
        )
        
        total_duration = frame_count * frame_duration
        st.info(f"🕐 Total loop: {total_duration:.1f} seconds")
        
        st.markdown("### 📐 Export Settings")
        
        anim_resolution = st.selectbox(
            "Resolution:",
            ["Standard (512×512)", "HD (1024×1024)"],
            help="Higher resolution = larger file size"
        )
        
        anim_resolution_map = {
            "Standard (512×512)": 512,
            "HD (1024×1024)": 1024
        }
        anim_res = anim_resolution_map[anim_resolution]
        
        # Estimate file size and generation time
        estimated_time = frame_count * (2 if anim_res == 1024 else 1)
        estimated_size = frame_count * (0.5 if anim_res == 1024 else 0.2)
        
        st.markdown(f"""
        **📊 Estimates:**
        - ⏱️ Generation: ~{estimated_time}s
        - 📁 File size: ~{estimated_size:.1f}MB
        - 🔄 Loop duration: {total_duration:.1f}s
        """)
        
        # Generate animation button
        st.markdown("---")
        if st.button("🎬 Generate Animation", type="primary", use_container_width=True, key="generate_anim"):
            generate_animation(
                st.session_state.generator, anim_style, anim_palette,
                base_energy, base_symmetry, base_deformation, base_color_variation,
                animate_param, variation_range, frame_count, frame_duration, anim_res
            )
    
    # Main content area
    if st.session_state.animation_gif is not None:
        # Success message with elegant design
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 3rem 2rem; 
                    border-radius: 24px; 
                    text-align: center; 
                    margin: 3rem 0;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                    backdrop-filter: blur(10px);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem; font-weight: 300;">Animation Complete</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">
                {len(st.session_state.animation_frames)} frames • {st.session_state.animation_time:.1f}s generation time
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered animation preview
        preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
        
        with preview_col2:
            st.markdown("""
            <div style="text-align: center; margin: 3rem 0;">
                <h2 style="color: #fff; font-weight: 300; margin-bottom: 2rem;">✨ Your Creation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the first frame as preview with elegant styling
            st.markdown('<div style="border-radius: 16px; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.5);">', unsafe_allow_html=True)
            st.image(st.session_state.animation_frames[0], 
                    caption="", 
                    use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button with modern design
            timestamp = int(time.time())
            gif_filename = f"quantyx_animation_{st.session_state.animation_params['style'].lower().replace(' ', '_')}_{timestamp}.gif"
            download_link_gif = create_download_link_gif(st.session_state.animation_gif, gif_filename)
            st.markdown(f'''
            <div style="text-align: center; margin: 3rem 0;">
                {download_link_gif}
            </div>
            ''', unsafe_allow_html=True)
        
        # Frame gallery with modern design
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 2rem 0;">
            <h2 style="color: #fff; font-weight: 300; margin-bottom: 1rem;">🎞️ Frame Sequence</h2>
            <p style="color: #aaa; font-size: 1.1rem;">Individual frames that compose your animation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show frames in a modern grid
        frames_per_row = 4
        frame_rows = [st.session_state.animation_frames[i:i+frames_per_row] 
                     for i in range(0, len(st.session_state.animation_frames), frames_per_row)]
        
        for row_idx, frame_row in enumerate(frame_rows):
            cols = st.columns(len(frame_row))
            for col_idx, frame in enumerate(frame_row):
                frame_num = row_idx * frames_per_row + col_idx + 1
                with cols[col_idx]:
                    st.markdown(f'<div style="border-radius: 8px; overflow: hidden; margin: 0.5rem 0;">', unsafe_allow_html=True)
                    st.image(frame, caption=f"Frame {frame_num}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Beautiful stats section at the bottom
        st.markdown("""
        <div style="margin: 4rem 0 2rem 0;">
            <div style="text-align: center; margin-bottom: 3rem;">
                <h2 style="color: #fff; font-weight: 300;">📊 Animation Statistics</h2>
                <p style="color: #aaa;">Technical details of your quantum animation</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Symmetrical stats grid
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        # Animation metrics with beautiful cards
        stats_data = [
            ("🎞️ Frames", len(st.session_state.animation_frames)),
            ("⏱️ Duration", f"{len(st.session_state.animation_frames) * st.session_state.animation_params['frame_duration']:.1f}s"),
            ("🎛️ Parameter", st.session_state.animation_params['animate_param']),
            ("🎨 Style", st.session_state.animation_params['style']),
            ("🌈 Palette", st.session_state.animation_params['palette']),
            ("📐 Resolution", f"{st.session_state.animation_params['resolution']}px"),
            ("⚡ Gen Time", f"{st.session_state.animation_time:.1f}s"),
            ("💾 File Size", f"{len(st.session_state.animation_gif.getvalue()) / (1024*1024):.1f}MB")
        ]
        
        # Display stats in modern cards across two rows
        for i in range(0, len(stats_data), 4):
            row_cols = st.columns(4)
            for j in range(4):
                if i + j < len(stats_data):
                    label, value = stats_data[i + j]
                    with row_cols[j]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
                                    border: 1px solid rgba(255,255,255,0.1);
                                    border-radius: 20px;
                                    padding: 2.5rem 1rem;
                                    text-align: center;
                                    margin: 0.8rem 0;
                                    backdrop-filter: blur(15px);
                                    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                                    cursor: pointer;
                                    position: relative;
                                    overflow: hidden;">
                            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        opacity: 0; transition: opacity 0.3s ease;"></div>
                            <div style="font-size: 2.5rem; margin-bottom: 0.8rem; position: relative; z-index: 1;">{label.split()[0]}</div>
                            <div style="color: #aaa; font-size: 0.9rem; margin-bottom: 0.8rem; position: relative; z-index: 1; text-transform: uppercase; letter-spacing: 1px;">{label.split(' ', 1)[1] if ' ' in label else label}</div>
                            <div style="color: #fff; font-size: 1.6rem; font-weight: 600; position: relative; z-index: 1;">{value}</div>
                        </div>
                        <style>
                        div:hover {{
                            transform: translateY(-8px) scale(1.02);
                            box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
                        }}
                        div:hover::before {{
                            opacity: 1;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
    
    else:
        # Animation studio header
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 16px; margin-bottom: 2rem;">
            <h1 style="color: #fff; font-size: 3rem; margin-bottom: 1rem;">🎬 Quantyx Animation Studio</h1>
            <p style="color: #cce7ff; font-size: 1.3rem; margin-bottom: 0.5rem;">Multi-Frame Animation Mode</p>
            <p style="color: #99d6ff; font-size: 1rem;">Create looping animations • Perfect for VJing • Music videos • Social sharing</p>
        </div>
        """, unsafe_allow_html=True)
        

        
        # Feature showcase
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                <h3 style="color: #fff; margin-bottom: 1rem;">🎭 Parameter Animation</h3>
                <p style="color: #e6e6ff;">Smoothly animate energy, symmetry, distortion, or color over time</p>
                <ul style="color: #ccccff; list-style: none; padding: 0;">
                    <li>• Energy Flux: Pulsing effects</li>
                    <li>• Geometric Order: Morphing shapes</li>
                    <li>• Field Distortion: Flowing waves</li>
                    <li>• Spectral Blend: Color shifts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                <h3 style="color: #fff; margin-bottom: 1rem;">🔄 Seamless Loops</h3>
                <p style="color: #e6ffe6;">Perfect loops designed for continuous playback</p>
                <ul style="color: #ccffcc; list-style: none; padding: 0;">
                    <li>• Smooth transitions</li>
                    <li>• No jarring cuts</li>
                    <li>• Mathematical precision</li>
                    <li>• Infinite replay value</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                <h3 style="color: #fff; margin-bottom: 1rem;">📱 Export Ready</h3>
                <p style="color: #ffe6e6;">Optimized for social sharing and professional use</p>
                <ul style="color: #ffcccc; list-style: none; padding: 0;">
                    <li>• GIF format support</li>
                    <li>• Multiple resolutions</li>
                    <li>• Optimized file sizes</li>
                    <li>• Cross-platform compatible</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # How it works section
        st.markdown("---")
        st.markdown("### 🛠️ How Animation Works")
        
        steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
        
        with steps_col1:
            st.markdown("""
            **1️⃣ Choose Base**
            - Select quantum structure
            - Set starting parameters
            - Pick color palette
            """)
        
        with steps_col2:
            st.markdown("""
            **2️⃣ Select Animation**
            - Choose parameter to animate
            - Set animation intensity
            - Configure timing
            """)
        
        with steps_col3:
            st.markdown("""
            **3️⃣ Generate Frames**
            - Automatically creates sequence
            - Progress tracking
            - Seamless loop calculation
            """)
        
        with steps_col4:
            st.markdown("""
            **4️⃣ Export & Share**
            - Download as GIF
            - Preview frame gallery
            - Share on social media
            """)
        
        # Footer (only shown on welcome screen)
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
                    <p style="color: #fff;">🎬 Quantyx Animation Studio • Built with ⚛️ Quantum Physics</p>
        <p style="color: #ccc;"><em>Professional quantum animation tools • Part of the Quantyx platform</em></p>
            <p style="font-size: 0.9rem; color: #888;">
                Perfect for VJ performances, music videos, digital art, and social media content
            </p>
        </div>
        """, unsafe_allow_html=True)
    



def generate_animation(generator, style, palette, base_energy, base_symmetry, 
                      base_deformation, base_color_variation, animate_param, 
                      variation_range, frame_count, frame_duration, resolution):
    """Generate animation with progress tracking."""
    
    with st.spinner(f"🌌 Generating {frame_count} quantum animation frames..."):
        start_time = time.time()
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frames = []
        
        for i in range(frame_count):
            status_text.text(f"Generating frame {i+1}/{frame_count}...")
            
            # Calculate animation progress (0 to 1 and back for seamless loop)
            if i < frame_count // 2:
                progress = i / (frame_count // 2)
            else:
                progress = (frame_count - i) / (frame_count // 2)
            
            # Vary the selected parameter
            if animate_param == "Energy Flux":
                energy = base_energy + (progress - 0.5) * variation_range * 2
                energy = max(0.1, min(10.0, energy))
                frame = generator.generate_quantum_image(
                    style, energy, base_symmetry, base_deformation, 
                    base_color_variation, resolution, palette)
            elif animate_param == "Geometric Order":
                symmetry = base_symmetry + (progress - 0.5) * variation_range * 100
                symmetry = max(0, min(100, int(symmetry)))
                frame = generator.generate_quantum_image(
                    style, base_energy, symmetry, base_deformation, 
                    base_color_variation, resolution, palette)
            elif animate_param == "Field Distortion":
                deformation = base_deformation + (progress - 0.5) * variation_range
                deformation = max(0.0, min(1.0, deformation))
                frame = generator.generate_quantum_image(
                    style, base_energy, base_symmetry, deformation, 
                    base_color_variation, resolution, palette)
            elif animate_param == "Spectral Blend":
                color_var = base_color_variation + (progress - 0.5) * variation_range
                color_var = max(0.0, min(1.0, color_var))
                frame = generator.generate_quantum_image(
                    style, base_energy, base_symmetry, base_deformation, 
                    color_var, resolution, palette)
            
            frames.append(frame)
            progress_bar.progress((i + 1) / frame_count)
        
        # Create GIF
        status_text.text("Creating animated GIF...")
        gif_buffer = create_gif_from_frames(frames, frame_duration)
        
        generation_time = time.time() - start_time
        
        # Store results in session state
        st.session_state.animation_frames = frames
        st.session_state.animation_gif = gif_buffer
        st.session_state.animation_time = generation_time
        st.session_state.animation_params = {
            'style': style,
            'palette': palette,
            'animate_param': animate_param,
            'frame_count': frame_count,
            'variation_range': variation_range,
            'frame_duration': frame_duration,
            'resolution': resolution
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()


if __name__ == "__main__":
    main() 