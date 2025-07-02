#!/usr/bin/env python3
"""
Quantyx Animation Studio - Physics-Accurate Quantum Animation
Multi-Frame Animation Mode for creating looping quantum field visualizations.
Now with REAL quantum physics simulations for scientific accuracy!
"""

import streamlit as st

# Configure page for animation studio
st.set_page_config(
    page_title="Quantyx - Animation Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
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
    from Image_Generation import VISUAL_PRESETS
    
    # Import quantum research capabilities
    try:
        from quantum_art_bridge import QuantumArtBridge, quick_quantum_art, generate_quantum_report
        QUANTUM_RESEARCH_AVAILABLE = True
    except ImportError:
        QUANTUM_RESEARCH_AVAILABLE = False
        
except ImportError:
    st.error("Error importing required modules. Please ensure the main app file is available.")
    st.stop()

# Note: st.set_page_config() is handled by the main app, not individual pages

def reset_animation_wizard():
    """Reset the animation wizard to step 1 and clear all selections."""
    st.session_state.animation_wizard_step = 1
    st.session_state.selected_animation_preset = None
    st.session_state.animation_color_palette = "Deep Plasma"
    st.session_state.animation_frames = 20
    st.session_state.animation_duration = 2.0
    
    # Clear any quantum animation state
    st.session_state.quantum_animation_mode = False
    if hasattr(st.session_state, 'current_quantum_anim_preset'):
        del st.session_state.current_quantum_anim_preset

def generate_quantum_animation_from_wizard():
    """Generate quantum animation using the wizard selections."""
    # Get settings from wizard
    color_palette = st.session_state.animation_color_palette
    resolution = st.session_state.animation_resolution
    frames = st.session_state.animation_frames
    duration = st.session_state.animation_duration
    frame_duration = duration / frames
    
    # Calculate individual frame duration
    frame_duration = duration / frames
    
    quantum_mode = getattr(st.session_state, 'animation_quantum_mode', False)
    
    with st.spinner("🎬 Creating your quantum animation masterpiece..."):
        if quantum_mode and QUANTUM_RESEARCH_AVAILABLE:
            # Get time range from preset or default
            preset_data = st.session_state.animation_preset_data
            time_range = preset_data.get("evolution_time", 2.0)
            
            generate_quantum_animation(
                st.session_state.generator, color_palette, frames, 
                frame_duration, resolution, preset_data, time_range
            )
        else:
            # For artistic mode, use defaults
            generate_animation(
                st.session_state.generator, "Quantum Bloom", color_palette,
                5.0, 50, 0.3, 0.5, "Energy Flux", 0.8, frames, frame_duration, resolution
            )
        
        # Reset wizard after successful generation
        st.session_state.animation_wizard_step = 1
        
        st.success("🎉 **Quantum Animation Generated!** Your masterpiece is ready below!")

def main():
    """Animation Studio main interface with quantum research integration."""
    
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
    
    if 'animation_quantum_data' not in st.session_state:
        st.session_state.animation_quantum_data = None

    # Initialize animation wizard state
    if 'animation_wizard_step' not in st.session_state:
        st.session_state.animation_wizard_step = 1
    if 'selected_animation_preset' not in st.session_state:
        st.session_state.selected_animation_preset = None
    if 'animation_color_palette' not in st.session_state:
        st.session_state.animation_color_palette = "Deep Plasma"
    if 'animation_frames' not in st.session_state:
        st.session_state.animation_frames = 20
    if 'animation_duration' not in st.session_state:
        st.session_state.animation_duration = 2.0

    # Perfect Animation Wizard (4-Step Flow)
    with st.sidebar:
        st.markdown("## 🎬 Animation Wizard")
        
        # Progress indicator (Perfect 4-step flow)
        animation_steps = ["🎯 Animation Type", "🌈 Color Palette", "⚙️ Settings", "🎬 Generate"]
        current_step = st.session_state.animation_wizard_step
        
        st.markdown("### 📋 Progress")
        for i, step in enumerate(animation_steps, 1):
            if i < current_step:
                st.markdown(f"✅ **Step {i}**: {step}")
            elif i == current_step:
                st.markdown(f"🔄 **Step {i}**: {step}")
            else:
                st.markdown(f"⏸️ **Step {i}**: {step}")
        
        st.markdown("---")
        
        # Step 1: Choose Animation Type
        if st.session_state.animation_wizard_step == 1:
            st.markdown("### 🎯 Step 1: Choose Animation")
            st.markdown("*Select your quantum animation type*")
            
            # Animation preset dropdown
            animation_presets = {
                "🌊 Quantum Phase Evolution": {
                    "description": "Watch quantum phases evolve through time with real TFIM Hamiltonian dynamics",
                    "hamiltonian_type": "tfim",
                    "evolution_time": 3.14,
                    "hamiltonian_params": {"h": 0.5, "J": 1.0},
                    "style": "Quantum Bloom",
                    "animation_type": "phase_evolution",
                    "frames": 20,
                    "speed": "medium"
                },
                "🔗 Entanglement Dynamics": {
                    "description": "Visualize quantum entanglement growing and spreading across the system",
                    "hamiltonian_type": "xxz",
                    "evolution_time": 2.5,
                    "hamiltonian_params": {"Jz": 1.0, "Jxy": 0.5},
                    "style": "Entanglement Field", 
                    "animation_type": "entanglement_growth",
                    "frames": 16,
                    "speed": "slow"
                },
                "⚡ Quantum Quench Animation": {
                    "description": "Dramatic quantum state transitions with sudden parameter changes",
                    "hamiltonian_type": "heisenberg",
                    "evolution_time": 4.0,
                    "hamiltonian_params": {"J": 1.2},
                    "style": "Crystal Spire",
                    "animation_type": "quench_dynamics",
                    "frames": 24,
                    "speed": "fast"
                },
                "🌌 Holographic Flow": {
                    "description": "AdS/CFT inspired bulk-boundary correspondence flowing through time",
                    "hamiltonian_type": "tfim",
                    "evolution_time": 6.28,
                    "hamiltonian_params": {"h": 1.0, "J": 0.8},
                    "style": "Singularity Core",
                    "animation_type": "holographic_flow",
                    "frames": 32,
                    "speed": "medium"
                }
            }
            
            selected_animation_preset = st.selectbox(
                "Choose animation preset:",
                list(animation_presets.keys()),
                help="Select a quantum animation with real physics evolution"
            )
            
            if selected_animation_preset:
                preset_data = animation_presets[selected_animation_preset]
                st.info(f"🎬 **{selected_animation_preset}**\n\n{preset_data['description']}")
                st.session_state.selected_animation_preset = selected_animation_preset
                st.session_state.animation_preset_data = preset_data
                st.session_state.animation_quantum_mode = True
                
                # Set default frames and duration from preset
                st.session_state.animation_frames = preset_data["frames"]
                if preset_data["speed"] == "slow":
                    st.session_state.animation_duration = 3.0
                elif preset_data["speed"] == "fast":
                    st.session_state.animation_duration = 1.5
                else:  # medium
                    st.session_state.animation_duration = 2.0
            
            if st.button("Continue to Colors ➡️", type="primary", use_container_width=True):
                if st.session_state.selected_animation_preset:
                    st.session_state.animation_wizard_step = 2
                    st.rerun()
        
        # Step 2: Choose Color Palette
        elif st.session_state.animation_wizard_step == 2:
            st.markdown("### 🌈 Step 2: Animation Colors")
            st.markdown("*Choose the color palette for your animation*")
            
            color_palette = st.selectbox(
                "Animation Palette:",
                list(COLOR_PALETTES.keys()),
                index=list(COLOR_PALETTES.keys()).index(st.session_state.animation_color_palette),
                help="Color mapping for the animation"
            )
            
            st.info(f"💡 {COLOR_PALETTES[color_palette]['description']}")
            st.session_state.animation_color_palette = color_palette
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.animation_wizard_step = 1
                    st.rerun()
            with col2:
                if st.button("Continue to Settings ➡️", type="primary", use_container_width=True):
                    st.session_state.animation_wizard_step = 3
                    st.rerun()
        
        # Step 3: Animation Settings
        elif st.session_state.animation_wizard_step == 3:
            st.markdown("### ⚙️ Step 3: Animation Settings")
            st.markdown("*Fine-tune your animation parameters*")
            
            # Show current preset
            if st.session_state.selected_animation_preset:
                st.markdown(f"**Animation:** {st.session_state.selected_animation_preset}")
            
            frames = st.slider("Number of Frames", 8, 60, st.session_state.animation_frames, help="More frames = smoother animation")
            st.session_state.animation_frames = frames
            
            duration = st.slider("Total Duration (s)", 0.5, 5.0, st.session_state.animation_duration, 0.1, help="Animation length in seconds")
            st.session_state.animation_duration = duration
            
            resolution_option = st.selectbox(
                "Quality:",
                ["Standard (512×512)", "HD (1024×1024)", "4K (3840×2160)"]
            )
            
            resolution_map = {
                "Standard (512×512)": 512,
                "HD (1024×1024)": 1024, 
                "4K (3840×2160)": 3840
            }
            resolution = resolution_map[resolution_option]
            st.session_state.animation_resolution = resolution
            
            # Estimated time
            est_time = frames * (resolution // 256) * 0.5
            st.info(f"⏱️ Est. time: ~{est_time:.0f}s")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.animation_wizard_step = 2
                    st.rerun()
            with col2:
                if st.button("Ready to Generate ➡️", type="primary", use_container_width=True):
                    st.session_state.animation_wizard_step = 4
                    st.rerun()
        
        # Step 4: Final Settings & Generate (Perfect Butterfly Symmetry)
        elif st.session_state.animation_wizard_step == 4:
            st.markdown("### 🎬 Step 4: Create Your Animation")
            st.markdown("*Perfect your animation settings and generate*")
            
            # Beautiful symmetrical layout
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.session_state.selected_animation_preset:
                    st.success(f"🎬 **Animation Type**\n{st.session_state.selected_animation_preset}")
            with preset_col2:
                if st.session_state.animation_color_palette:
                    st.info(f"🌈 **Color Palette**\n{st.session_state.animation_color_palette}")
            
            st.markdown("#### ⚛️ Perfect Animation Parameters")
            
            # Beautiful clean vertical layout - each parameter gets its own line
            frames = st.slider("Number of Frames", 8, 60, st.session_state.animation_frames, help="More frames = smoother animation")
            st.session_state.animation_frames = frames
            
            duration = st.slider("Total Duration (s)", 0.5, 5.0, st.session_state.animation_duration, 0.1, help="Animation length in seconds")
            st.session_state.animation_duration = duration
            
            st.markdown("#### 📐 Output Resolution")
            
            resolution_option = st.selectbox(
                "Quality:",
                ["Standard (512×512)", "HD (1024×1024)", "4K (3840×2160)"]
            )
            
            resolution_map = {
                "Standard (512×512)": 512,
                "HD (1024×1024)": 1024, 
                "4K (3840×2160)": 3840
            }
            resolution = resolution_map[resolution_option]
            st.session_state.animation_resolution = resolution
            
            # Show estimated generation time
            est_time = frames * (resolution // 256) * 0.5
            st.info(f"⏱️ Est. time: ~{est_time:.0f}s")
            
            # Perfect symmetrical navigation
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.animation_wizard_step = 3
                    st.rerun()
            with nav_col2:
                # Check if we're in quantum research mode for button text
                quantum_mode = getattr(st.session_state, 'animation_quantum_mode', False)
                button_text = "🔬 Generate Quantum Animation" if quantum_mode else "🎬 Generate Animation"
                
                if st.button(button_text, type="primary", use_container_width=True):
                    # Generate immediately
                    generate_quantum_animation_from_wizard()
                    st.rerun()
        
        # Reset wizard button (always visible)
        st.markdown("---")
        if st.button("🔄 Start Over", help="Reset the animation wizard", use_container_width=True, type="secondary"):
            reset_animation_wizard()
            st.rerun()
    
    # Main content area
    if st.session_state.animation_gif is not None:
        # Success message with elegant design
        quantum_mode_text = " (Quantum Research)" if st.session_state.get('quantum_animation_mode', False) else ""
        
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
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem; font-weight: 300;">Animation Complete{quantum_mode_text}</h1>
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
            animation_type = "quantum" if st.session_state.get('quantum_animation_mode', False) else "artistic"
            gif_filename = f"quantyx_animation_{animation_type}_{timestamp}.gif"
            download_link_gif = create_download_link_gif(st.session_state.animation_gif, gif_filename)
            st.markdown(f'''
            <div style="text-align: center; margin: 3rem 0;">
                {download_link_gif}
            </div>
            ''', unsafe_allow_html=True)
        
        # Quantum Research Report (if applicable)
        if (st.session_state.get('quantum_animation_mode', False) and 
            st.session_state.animation_quantum_data is not None and
            QUANTUM_RESEARCH_AVAILABLE):
            
            st.markdown("---")
            st.markdown("### 🔬 Quantum Animation Research Report")
            
            quantum_data_list = st.session_state.animation_quantum_data
            
            # Average quantum metrics across all frames
            avg_correlation = np.mean([qd.correlation_coefficient for qd in quantum_data_list])
            avg_entropy = np.mean([qd.metadata['max_entropy'] for qd in quantum_data_list])
            evolution_range = max([qd.evolution_time for qd in quantum_data_list]) - min([qd.evolution_time for qd in quantum_data_list])
            
            research_col1, research_col2, research_col3 = st.columns(3)
            with research_col1:
                st.metric("Avg. Curvature-Energy Correlation", f"{avg_correlation:.4f}")
                st.metric("Avg. Max Entanglement Entropy", f"{avg_entropy:.3f}")
            with research_col2:
                st.metric("Hamiltonian Type", quantum_data_list[0].hamiltonian_type.upper())
                st.metric("Evolution Time Range", f"{evolution_range:.2f}")
            with research_col3:
                st.metric("Quantum System Size", f"{quantum_data_list[0].n_qubits} qubits")
                st.metric("Total Frames", len(quantum_data_list))
            
            # Research report expander
            with st.expander("📋 Complete Animation Physics Report", expanded=False):
                st.markdown("**Frame-by-Frame Quantum Analysis:**")
                
                for i, qd in enumerate(quantum_data_list):
                    st.markdown(f"""
                    **Frame {i+1}** (t = {qd.evolution_time:.2f}):
                    - Correlation: {qd.correlation_coefficient:.4f}
                    - Max Entropy: {qd.metadata['max_entropy']:.3f}
                    - Mean Curvature: {qd.metadata['mean_curvature']:.3f}
                    """)
                
                # Download complete research data
                animation_research_data = {
                    "animation_metadata": {
                        "animation_type": "quantum_research",
                        "frame_count": len(quantum_data_list),
                        "hamiltonian_type": quantum_data_list[0].hamiltonian_type,
                        "evolution_time_range": [qd.evolution_time for qd in quantum_data_list],
                        "avg_correlation": avg_correlation,
                        "avg_entropy": avg_entropy
                    },
                    "frame_data": [
                        {
                            "frame": i,
                            "evolution_time": qd.evolution_time,
                            "correlation_coefficient": qd.correlation_coefficient,
                            "entanglement_entropies": qd.entanglement_entropies.tolist(),
                            "curvatures": qd.curvatures.tolist(),
                            "energy_deltas": qd.energy_deltas.tolist(),
                            "metadata": qd.metadata
                        }
                        for i, qd in enumerate(quantum_data_list)
                    ]
                }
                
                research_json = json.dumps(animation_research_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download Complete Animation Research Data (JSON)",
                    data=research_json,
                    file_name=f"quantyx_animation_research_{timestamp}.json",
                    mime="application/json"
                )
        
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
                    
                    # Show quantum data for this frame if available
                    if (st.session_state.get('quantum_animation_mode', False) and 
                        st.session_state.animation_quantum_data is not None and
                        frame_num <= len(st.session_state.animation_quantum_data)):
                        qd = st.session_state.animation_quantum_data[frame_num - 1]
                        caption = f"Frame {frame_num}\nt={qd.evolution_time:.2f}\nρ={qd.correlation_coefficient:.3f}"
                    else:
                        caption = f"Frame {frame_num}"
                    
                    st.image(frame, caption=caption, use_container_width=True)
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
        animation_type = "Quantum Research" if st.session_state.get('quantum_animation_mode', False) else "Artistic"
        
        stats_data = [
            ("🎞️ Frames", len(st.session_state.animation_frames)),
            ("⏱️ Duration", f"{len(st.session_state.animation_frames) * st.session_state.animation_params['frame_duration']:.1f}s"),
            ("🔬 Type", animation_type),
            ("🎨 Style", st.session_state.animation_params.get('style', 'Quantum')),
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
        # Wizard-aware welcome screen for animation
        current_step = st.session_state.animation_wizard_step
        
        if current_step == 1:
            # Step 1 welcome
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #fff;">
                <h3 style="color: #fff;">🎬 Welcome to the Quantyx Animation Wizard</h3>
                <p style="font-size: 1.3rem; margin: 1.5rem 0; color: #fff;">Let's create your quantum animation step by step!</p>
                <p style="font-size: 1.1rem; margin: 1.5rem 0; color: #ccc;">👈 Start by choosing your animation type in the sidebar</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature cards for animation types
            feat_col1, feat_col2 = st.columns(2)
            
            with feat_col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h3 style="color: #fff; margin-bottom: 1rem;">🌊 Phase Evolution</h3>
                    <p style="color: #e6e6ff;">Watch quantum phases transition with TFIM dynamics</p>
                    <ul style="color: #ccccff; list-style: none; padding: 0;">
                        <li>• Real quantum computing</li>
                        <li>• Phase transitions</li>
                        <li>• Scientific accuracy</li>
                        <li>• Research export</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with feat_col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h3 style="color: #fff; margin-bottom: 1rem;">🔗 Entanglement Dynamics</h3>
                    <p style="color: #e6ffe6;">Visualize quantum entanglement spreading</p>
                    <ul style="color: #ccffcc; list-style: none; padding: 0;">
                        <li>• Growing entanglement</li>
                        <li>• XXZ Hamiltonian</li>
                        <li>• Beautiful transitions</li>
                        <li>• Frame-by-frame data</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        elif current_step == 2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; color: #fff;">
                <h3 style="color: #fff;">🌈 Step 2: Choose Animation Colors</h3>
                <p style="font-size: 1.1rem; margin: 1.5rem 0; color: #ccc;">Select the color palette that will bring your quantum animation to life</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif current_step == 3:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; color: #fff;">
                <h3 style="color: #fff;">⚙️ Step 3: Animation Settings</h3>
                <p style="font-size: 1.1rem; margin: 1.5rem 0; color: #ccc;">Fine-tune your animation frames and timing</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif current_step == 4:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; color: #fff;">
                <h3 style="color: #fff;">🎬 Step 4: Create Your Animation</h3>
                <p style="font-size: 1.1rem; margin: 1.5rem 0; color: #ccc;">Perfect your settings and generate your quantum animation</p>
                <p style="font-size: 1.0rem; margin: 1.5rem 0; color: #aaa;">👈 Click generate in the sidebar when ready</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Beautiful creation summary for final step
            st.markdown("### ✨ Your Perfect Animation Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h4 style="color: #fff; margin-bottom: 1rem;">🎬 Your Animation</h4>
                    <p style="color: #e6e6ff; font-size: 1.1rem;">""" + str(st.session_state.selected_animation_preset) + """</p>
                    <p style="color: #ccccff; font-size: 0.9rem;">""" + str(st.session_state.animation_frames) + """ frames</p>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h4 style="color: #fff; margin-bottom: 1rem;">🌈 Your Style</h4>
                    <p style="color: #e6ffe6; font-size: 1.1rem;">""" + str(st.session_state.animation_color_palette) + """</p>
                    <p style="color: #ccffcc; font-size: 0.9rem;">""" + str(st.session_state.animation_duration) + """s duration</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show animation progress for steps 2-3 (Perfect Butterfly Layout)
        if current_step > 1 and current_step < 4:
            st.markdown("### 📋 Your Animation Selections So Far")
            
            progress_col1, progress_col2 = st.columns(2)
            
            with progress_col1:
                if st.session_state.selected_animation_preset:
                    st.success(f"**Animation Type:** {st.session_state.selected_animation_preset}")
                else:
                    st.info("**Animation Type:** Not selected")
            
            with progress_col2:
                if current_step >= 3:
                    st.success(f"**Colors:** {st.session_state.animation_color_palette}")
                else:
                    st.info("**Colors:** Not selected")
        
        # Footer for first step only
        if current_step == 1:
            st.markdown("---")
            st.markdown("""
            <div class="footer">
                <p style="color: #fff;">🎬 Quantyx Animation Wizard • Built with ⚛️ Quantum Physics</p>
                <p style="color: #fff;"><em>Step-by-step quantum animation creation • Perfect for VJing & social sharing</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature showcase (only for step 1)
        if current_step == 1:
            st.markdown("---")
            st.markdown("### 🎬 Animation Studio Features")
            feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                <h3 style="color: #fff; margin-bottom: 1rem;">🔬 Quantum Animations</h3>
                <p style="color: #e6e6ff;">Real quantum physics simulations animated over time</p>
                <ul style="color: #ccccff; list-style: none; padding: 0;">
                    <li>• Phase transitions</li>
                    <li>• Entanglement evolution</li>
                    <li>• Quantum quench dynamics</li>
                    <li>• Holographic flow</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                <h3 style="color: #fff; margin-bottom: 1rem;">🎭 Parameter Animation</h3>
                <p style="color: #e6ffe6;">Smoothly animate energy, symmetry, distortion, or color over time</p>
                <ul style="color: #ccffcc; list-style: none; padding: 0;">
                    <li>• Energy Flux: Pulsing effects</li>
                    <li>• Geometric Order: Morphing shapes</li>
                    <li>• Field Distortion: Flowing waves</li>
                    <li>• Spectral Blend: Color shifts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                <h3 style="color: #fff; margin-bottom: 1rem;">📱 Export Ready</h3>
                <p style="color: #ffe6e6;">Optimized for social sharing and professional use</p>
                <ul style="color: #ffcccc; list-style: none; padding: 0;">
                    <li>• Perfect seamless loops</li>
                    <li>• Multiple resolutions</li>
                    <li>• Research data export</li>
                    <li>• Cross-platform GIFs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # How it works section
        st.markdown("---")
        st.markdown("### 🛠️ How Quantum Animation Works")
        
        steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
        
        with steps_col1:
            st.markdown("""
            **1️⃣ Choose Mode**
            - Quantum research presets
            - Manual parameter control
            - Select physics model
            """)
        
        with steps_col2:
            st.markdown("""
            **2️⃣ Set Animation**
            - Time evolution range
            - Frame count & timing
            - Animation intensity
            """)
        
        with steps_col3:
            st.markdown("""
            **3️⃣ Generate Frames**
            - Real quantum simulations
            - Physics-accurate evolution
            - Seamless loop calculation
            """)
        
        with steps_col4:
            st.markdown("""
            **4️⃣ Export & Research**
            - Download as GIF
            - Export quantum data
            - Scientific analysis
            """)
        
        # Footer (only shown on welcome screen)
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p style="color: #fff;">🎬 Quantyx Animation Studio • Built with ⚛️ Real Quantum Physics</p>
            <p style="color: #ccc;"><em>Scientific quantum animation tools • Research-grade accuracy</em></p>
            <p style="font-size: 0.9rem; color: #888;">
                Perfect for VJ performances, scientific visualization, music videos, and educational content
            </p>
        </div>
        """, unsafe_allow_html=True)


def generate_quantum_animation(generator, palette, frame_count, frame_duration, 
                              resolution, quantum_preset, time_range):
    """Generate animation using real quantum physics simulations."""
    
    with st.spinner(f"🌌 Running {frame_count} quantum simulations for physics-accurate animation..."):
        start_time = time.time()
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frames = []
        quantum_data_list = []
        
        # Initialize quantum bridge
        if QUANTUM_RESEARCH_AVAILABLE:
            bridge = QuantumArtBridge(n_qubits=6)  # Smaller for faster animation generation
        else:
            st.error("Quantum research not available!")
            return
        
        for i in range(frame_count):
            status_text.text(f"Running quantum simulation for frame {i+1}/{frame_count}...")
            
            # Calculate evolution time for this frame
            evolution_time = (i / (frame_count - 1)) * time_range
            
            try:
                # Run quantum simulation for this time point
                quantum_data = bridge.run_quantum_simulation(
                    hamiltonian_type=quantum_preset["hamiltonian_type"],
                    evolution_time=evolution_time,
                    hamiltonian_params=quantum_preset.get("hamiltonian_params"),
                    trotter_steps=10  # Faster for animation
                )
                
                # Map quantum data to art parameters
                art_params = bridge.quantum_to_art_mapping(
                    quantum_data, 
                    style_preference=quantum_preset.get("style_preference", "auto")
                )
                
                # Generate image using quantum-derived parameters
                frame = generator.generate_quantum_image(
                    style=art_params["style"],
                    energy_intensity=art_params["energy_intensity"],
                    symmetry_level=int(art_params["symmetry_level"]),
                    deformation_amount=art_params["deformation_amount"],
                    color_variation=art_params["color_variation"],
                    resolution=resolution,
                    color_palette=palette
                )
                
                frames.append(frame)
                quantum_data_list.append(quantum_data)
                
            except Exception as e:
                st.warning(f"Quantum simulation failed for frame {i+1}, using fallback: {e}")
                # Fallback to basic generation
                frame = generator.generate_quantum_image(
                    "Quantum Bloom", 5.0, 50, 0.3, 0.5, resolution, palette
                )
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
        st.session_state.animation_quantum_data = quantum_data_list
        st.session_state.animation_params = {
            'style': 'Quantum Research',
            'palette': palette,
            'animate_param': 'Quantum Evolution',
            'frame_count': frame_count,
            'frame_duration': frame_duration,
            'resolution': resolution,
            'quantum_preset': quantum_preset,
            'time_range': time_range
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()


def generate_animation(generator, style, palette, base_energy, base_symmetry, 
                      base_deformation, base_color_variation, animate_param, 
                      variation_range, frame_count, frame_duration, resolution):
    """Generate animation with progress tracking (classic mode)."""
    
    with st.spinner(f"🌌 Generating {frame_count} artistic animation frames..."):
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
        st.session_state.animation_quantum_data = None  # No quantum data for artistic mode
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