"""
Quantum-Inspired Generative Art Engine

This module transforms curvature-energy mappings from synthetic manifolds into 
high-fidelity generative art, simulating quantum-like fields through purely 
classical geometry. Each image or animation frame is derived from spatial 
curvature gradients and energy analogs computed using discrete differential geometry.

The system assigns color, opacity, particle effects, and field lines based on 
local energy density and curvature intensity. Users can manipulate topological 
seeds, noise levels, and smoothing coefficients to generate infinite visual 
permutations.

Technically structured around a modular Python pipeline:
1. Simulate curvature-energy fields
2. Map results to color-space and geometry overlay schema  
3. Export visuals (PNG/MP4)

This engine emphasizes explainable art—each image is backed by actual 
mathematical data, offering both aesthetic intrigue and physical meaning.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.interpolate import griddata, RBFInterpolator
import seaborn as sns

# Import from existing modules
from curvature_energy_analysis import (
    compute_curvature, 
    compute_energy_deltas,
    safe_pearson_correlation
)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Ultra-modern artistic configuration
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'xtick.bottom': False,
    'xtick.top': False,
    'ytick.left': False,
    'ytick.right': False,
    'figure.facecolor': 'black',
    'axes.facecolor': 'black'
})


@dataclass
class ArtisticParameters:
    """Configuration for generative art parameters."""
    # Topological parameters
    grid_size: int = 256
    noise_scale: float = 0.1
    smoothing_sigma: float = 2.0
    topology_seed: int = 42
    
    # Visual mapping parameters
    color_scheme: str = 'quantum_plasma'  # 'quantum_plasma', 'cosmic', 'neon', 'aurora'
    opacity_mapping: str = 'energy_density'  # 'energy_density', 'curvature', 'gradient'
    particle_density: float = 0.02
    field_line_density: float = 0.05
    
    # Animation parameters
    n_frames: int = 120
    evolution_speed: float = 0.1
    rotation_enabled: bool = True
    
    # Output parameters
    output_format: str = 'both'  # 'png', 'mp4', 'both'
    output_dir: str = 'generative_art_output'
    high_resolution: bool = True
    
    # Advanced parameters
    enable_3d: bool = False
    interactive_mode: bool = False
    apply_post_processing: bool = True


class QuantumManifoldGenerator:
    """Generates synthetic manifolds with curvature-energy mappings."""
    
    def __init__(self, params: ArtisticParameters):
        self.params = params
        self.rng = np.random.default_rng(params.topology_seed)
        
    def generate_base_manifold(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base 2D manifold using multiple noise sources."""
        size = self.params.grid_size
        x = np.linspace(-4, 4, size)
        y = np.linspace(-4, 4, size)
        X, Y = np.meshgrid(x, y)
        
        # Multi-scale Perlin-like noise
        Z = np.zeros_like(X)
        frequencies = [1, 2, 4, 8, 16]
        amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]
        
        for freq, amp in zip(frequencies, amplitudes):
            # Sinusoidal base with noise perturbations
            noise_x = self.rng.normal(0, self.params.noise_scale, X.shape)
            noise_y = self.rng.normal(0, self.params.noise_scale, Y.shape)
            
            Z += amp * np.sin(freq * X + noise_x) * np.cos(freq * Y + noise_y)
            Z += amp * 0.3 * np.exp(-(X**2 + Y**2) / (4 / freq))
        
        # Add quantum-inspired interference patterns
        Z += 0.2 * np.sin(np.sqrt(X**2 + Y**2) * 3) * np.exp(-0.1 * (X**2 + Y**2))
        
        # Smooth the manifold
        Z = gaussian_filter(Z, sigma=self.params.smoothing_sigma)
        
        return X, Y, Z
    
    def compute_manifold_curvature(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Compute discrete Gaussian curvature on the manifold."""
        # Compute gradients
        dz_dx, dz_dy = np.gradient(Z)
        
        # Compute second derivatives
        d2z_dx2, _ = np.gradient(dz_dx)
        _, d2z_dy2 = np.gradient(dz_dy)
        d2z_dxdy, _ = np.gradient(dz_dy)
        
        # Gaussian curvature formula for height field z = f(x,y)
        # K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2
        numerator = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        denominator = (1 + dz_dx**2 + dz_dy**2)**2
        
        # Avoid division by zero
        curvature = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator != 0)
        
        return curvature
    
    def compute_energy_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Compute quantum-inspired energy field."""
        # Energy density based on manifold geometry and quantum-like potentials
        dz_dx, dz_dy = np.gradient(Z)
        gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
        
        # Kinetic energy analog (gradient term)
        kinetic = 0.5 * gradient_magnitude**2
        
        # Potential energy (position-dependent)
        potential = 0.25 * (X**2 + Y**2) + 0.1 * Z**2
        
        # Quantum pressure (Laplacian term)
        laplacian = np.gradient(np.gradient(Z, axis=0), axis=0) + \
                   np.gradient(np.gradient(Z, axis=1), axis=1)
        quantum_pressure = 0.1 * np.abs(laplacian)
        
        energy = kinetic + potential + quantum_pressure
        
        # Add noise for artistic variation
        energy += self.rng.normal(0, 0.05 * np.std(energy), energy.shape)
        
        return energy


class ArtisticRenderer:
    """Renders curvature-energy fields as generative art."""
    
    def __init__(self, params: ArtisticParameters):
        self.params = params
        self.color_maps = self._create_custom_colormaps()
        
    def _create_custom_colormaps(self) -> Dict[str, Any]:
        """Create custom artistic color schemes."""
        color_maps = {}
        
        # Quantum plasma scheme
        colors_plasma = ['#0a0a0a', '#1e0a3e', '#3d1a78', '#7209b7', 
                        '#a663cc', '#d4a5db', '#ffeaa7', '#00b894']
        color_maps['quantum_plasma'] = LinearSegmentedColormap.from_list(
            'quantum_plasma', colors_plasma, N=256)
        
        # Cosmic scheme
        colors_cosmic = ['#000000', '#0f0f23', '#1a1a3a', '#2d1b69', 
                        '#aa3366', '#ff6b6b', '#ffd93d', '#ffffff']
        color_maps['cosmic'] = LinearSegmentedColormap.from_list(
            'cosmic', colors_cosmic, N=256)
        
        # Neon scheme
        colors_neon = ['#000a0a', '#001122', '#002244', '#0033aa', 
                      '#00ffaa', '#66ff66', '#ffff00', '#ff00ff']
        color_maps['neon'] = LinearSegmentedColormap.from_list(
            'neon', colors_neon, N=256)
        
        # Aurora scheme
        colors_aurora = ['#0d1b2a', '#415a77', '#778da9', '#7209b7',
                        '#e0aaff', '#c77dff', '#e9c46a', '#f4a261']
        color_maps['aurora'] = LinearSegmentedColormap.from_list(
            'aurora', colors_aurora, N=256)
        
        return color_maps
    
    def render_static_art(self, X: np.ndarray, Y: np.ndarray, 
                         curvature: np.ndarray, energy: np.ndarray,
                         title: str = "Quantum Curvature-Energy Manifold") -> plt.Figure:
        """Render a static artistic visualization."""
        if self.params.enable_3d:
            return self._render_3d_art(X, Y, curvature, energy, title)
        else:
            return self._render_2d_art(X, Y, curvature, energy, title)
    
    def _render_2d_art(self, X: np.ndarray, Y: np.ndarray, 
                      curvature: np.ndarray, energy: np.ndarray,
                      title: str) -> plt.Figure:
        """Render 2D artistic visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')
        fig.suptitle(title, fontsize=20, color='white', y=0.95)
        
        # Main curvature-energy composite
        ax_main = axes[0, 0]
        ax_main.set_facecolor('black')
        
        # Normalize fields for artistic rendering
        curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min())
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min())
        
        # Create artistic composite field
        composite = 0.6 * curvature_norm + 0.4 * energy_norm
        
        # Main field visualization with artistic styling
        cmap = self.color_maps[self.params.color_scheme]
        im_main = ax_main.imshow(composite, extent=[X.min(), X.max(), Y.min(), Y.max()],
                                cmap=cmap, alpha=0.9, interpolation='bilinear')
        
        # Add field lines
        self._add_field_lines(ax_main, X, Y, curvature, energy)
        
        # Add particle effects
        self._add_particle_effects(ax_main, X, Y, energy_norm)
        
        ax_main.set_title('Curvature-Energy Field', color='white', fontsize=14)
        ax_main.axis('off')
        
        # Pure curvature field
        ax_curv = axes[0, 1]
        ax_curv.set_facecolor('black')
        im_curv = ax_curv.imshow(curvature, extent=[X.min(), X.max(), Y.min(), Y.max()],
                                cmap='RdBu_r', alpha=0.9, interpolation='bilinear')
        ax_curv.set_title('Gaussian Curvature Field', color='white', fontsize=14)
        ax_curv.axis('off')
        
        # Pure energy field
        ax_energy = axes[1, 0]
        ax_energy.set_facecolor('black')
        im_energy = ax_energy.imshow(energy, extent=[X.min(), X.max(), Y.min(), Y.max()],
                                    cmap='plasma', alpha=0.9, interpolation='bilinear')
        ax_energy.set_title('Quantum Energy Density', color='white', fontsize=14)
        ax_energy.axis('off')
        
        # Correlation visualization
        ax_corr = axes[1, 1]
        ax_corr.set_facecolor('black')
        self._render_correlation_art(ax_corr, curvature.flatten(), energy.flatten())
        
        plt.tight_layout()
        return fig
    
    def _render_3d_art(self, X: np.ndarray, Y: np.ndarray, 
                      curvature: np.ndarray, energy: np.ndarray,
                      title: str) -> plt.Figure:
        """Render 3D artistic visualization."""
        fig = plt.figure(figsize=(20, 15), facecolor='black')
        
        # Main 3D surface
        ax_3d = fig.add_subplot(221, projection='3d')
        ax_3d.set_facecolor('black')
        
        # Create artistic height field (combine curvature and energy)
        height = 0.7 * curvature + 0.3 * energy
        
        # Artistic surface with custom coloring
        surf = ax_3d.plot_surface(X, Y, height, 
                                 facecolors=plt.cm.plasma((energy - energy.min()) / (energy.max() - energy.min())),
                                 alpha=0.8, linewidth=0, antialiased=True)
        
        # Add contour lines at the base
        ax_3d.contour(X, Y, height, levels=15, cmap='white', alpha=0.3, offset=height.min())
        
        ax_3d.set_title('3D Curvature-Energy Manifold', color='white', fontsize=14)
        ax_3d.grid(False)
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        
        # 2D projections with artistic styling
        for i, (field, name, cmap) in enumerate([(curvature, 'Curvature', 'RdBu_r'),
                                                 (energy, 'Energy', 'plasma')]):
            ax = fig.add_subplot(2, 2, i + 3)
            ax.set_facecolor('black')
            im = ax.imshow(field, extent=[X.min(), X.max(), Y.min(), Y.max()],
                          cmap=cmap, alpha=0.9, interpolation='bilinear')
            ax.set_title(f'{name} Field', color='white', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _add_field_lines(self, ax: plt.Axes, X: np.ndarray, Y: np.ndarray,
                        curvature: np.ndarray, energy: np.ndarray) -> None:
        """Add artistic field lines to the visualization."""
        # Compute gradient field for field lines
        grad_x, grad_y = np.gradient(energy)
        
        # Sample points for field lines
        n_lines = int(self.params.field_line_density * X.shape[0])
        step = max(1, X.shape[0] // n_lines)
        
        x_samples = X[::step, ::step]
        y_samples = Y[::step, ::step]
        u_samples = grad_x[::step, ::step]
        v_samples = grad_y[::step, ::step]
        
        # Normalize for consistent line lengths
        magnitude = np.sqrt(u_samples**2 + v_samples**2)
        u_norm = np.divide(u_samples, magnitude, out=np.zeros_like(u_samples), where=magnitude != 0)
        v_norm = np.divide(v_samples, magnitude, out=np.zeros_like(v_samples), where=magnitude != 0)
        
        # Create artistic field lines
        scale = 0.3
        for i in range(x_samples.shape[0]):
            for j in range(x_samples.shape[1]):
                if magnitude[i, j] > 0.1:  # Only draw significant field lines
                    x_start = x_samples[i, j]
                    y_start = y_samples[i, j]
                    x_end = x_start + scale * u_norm[i, j]
                    y_end = y_start + scale * v_norm[i, j]
                    
                    # Color based on field strength
                    alpha = min(1.0, magnitude[i, j] / np.max(magnitude))
                    ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                            head_width=0.05, head_length=0.05, fc='cyan', ec='cyan',
                            alpha=alpha * 0.6, linewidth=1)
    
    def _add_particle_effects(self, ax: plt.Axes, X: np.ndarray, Y: np.ndarray,
                             energy_field: np.ndarray) -> None:
        """Add particle effects based on energy density."""
        # Sample particle positions based on energy density
        n_particles = int(self.params.particle_density * X.size)
        
        # Create probability distribution from energy field
        energy_flat = energy_field.flatten()
        energy_prob = (energy_flat - energy_flat.min()) / (energy_flat.max() - energy_flat.min())
        energy_prob = energy_prob / energy_prob.sum()
        
        # Sample positions
        indices = np.random.choice(len(energy_flat), size=n_particles, p=energy_prob)
        i_coords, j_coords = np.unravel_index(indices, X.shape)
        
        particle_x = X[i_coords, j_coords]
        particle_y = Y[i_coords, j_coords]
        particle_energy = energy_field[i_coords, j_coords]
        
        # Artistic particle rendering
        sizes = 20 + 100 * (particle_energy - particle_energy.min()) / (particle_energy.max() - particle_energy.min())
        colors = plt.cm.hot(particle_energy / particle_energy.max())
        
        scatter = ax.scatter(particle_x, particle_y, s=sizes, c=colors, alpha=0.7, 
                           edgecolors='white', linewidths=0.5)
    
    def _render_correlation_art(self, ax: plt.Axes, curvature_flat: np.ndarray, 
                               energy_flat: np.ndarray) -> None:
        """Create artistic visualization of curvature-energy correlation."""
        ax.set_facecolor('black')
        
        # Sample points for artistic scatter
        n_sample = min(5000, len(curvature_flat))
        indices = np.random.choice(len(curvature_flat), n_sample, replace=False)
        
        x_sample = curvature_flat[indices]
        y_sample = energy_flat[indices]
        
        # Create artistic scatter with density-based coloring
        scatter = ax.scatter(x_sample, y_sample, s=20, alpha=0.6, 
                           c=range(len(x_sample)), cmap='viridis')
        
        # Add correlation statistics
        r, p = safe_pearson_correlation(curvature_flat, energy_flat)
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3e}', 
               transform=ax.transAxes, color='white', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.set_xlabel('Curvature', color='white')
        ax.set_ylabel('Energy', color='white')
        ax.set_title('Curvature-Energy Correlation', color='white', fontsize=14)
        ax.tick_params(colors='white')
    
    def create_animation(self, manifold_generator: QuantumManifoldGenerator) -> animation.FuncAnimation:
        """Create animated visualization of evolving curvature-energy fields."""
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        ax.axis('off')
        
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            ax.axis('off')
            
            # Evolve the manifold parameters
            t = frame * self.params.evolution_speed
            
            # Temporarily modify parameters for animation
            original_seed = manifold_generator.params.topology_seed
            manifold_generator.params.topology_seed = original_seed + frame
            
            # Generate evolved manifold
            X, Y, Z = manifold_generator.generate_base_manifold()
            
            # Add time evolution
            Z += 0.1 * np.sin(t) * np.exp(-(X**2 + Y**2) / 8)
            
            curvature = manifold_generator.compute_manifold_curvature(X, Y, Z)
            energy = manifold_generator.compute_energy_field(X, Y, Z)
            
            # Artistic rendering
            composite = 0.6 * ((curvature - curvature.min()) / (curvature.max() - curvature.min())) + \
                       0.4 * ((energy - energy.min()) / (energy.max() - energy.min()))
            
            cmap = self.color_maps[self.params.color_scheme]
            im = ax.imshow(composite, extent=[X.min(), X.max(), Y.min(), Y.max()],
                          cmap=cmap, alpha=0.9, interpolation='bilinear')
            
            # Add evolving particle effects
            self._add_particle_effects(ax, X, Y, (energy - energy.min()) / (energy.max() - energy.min()))
            
            ax.set_title(f'Evolving Quantum Fields (t={t:.2f})', 
                        color='white', fontsize=16, pad=20)
            
            # Restore original seed
            manifold_generator.params.topology_seed = original_seed
        
        anim = animation.FuncAnimation(fig, animate, frames=self.params.n_frames,
                                      interval=50, blit=False, repeat=True)
        
        return anim


class GenerativeArtEngine:
    """Main engine orchestrating the generative art pipeline."""
    
    def __init__(self, params: Optional[ArtisticParameters] = None):
        self.params = params or ArtisticParameters()
        self.manifold_generator = QuantumManifoldGenerator(self.params)
        self.renderer = ArtisticRenderer(self.params)
        
        # Ensure output directory exists
        os.makedirs(self.params.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Generative Art Engine initialized with {self.params.color_scheme} color scheme")
    
    def generate_single_artwork(self, title: Optional[str] = None) -> str:
        """Generate a single piece of generative art."""
        start_time = time.time()
        
        # Step 1: Simulate curvature-energy fields
        logger.info("Generating quantum manifold...")
        X, Y, Z = self.manifold_generator.generate_base_manifold()
        
        logger.info("Computing curvature field...")
        curvature = self.manifold_generator.compute_manifold_curvature(X, Y, Z)
        
        logger.info("Computing energy field...")
        energy = self.manifold_generator.compute_energy_field(X, Y, Z)
        
        # Step 2: Render artistic visualization
        if title is None:
            title = f"Quantum Art #{self.params.topology_seed}"
        
        logger.info("Rendering artwork...")
        fig = self.renderer.render_static_art(X, Y, curvature, energy, title)
        
        # Step 3: Save output
        timestamp = int(time.time())
        filename = f"quantum_art_{timestamp}_{self.params.topology_seed}.png"
        filepath = os.path.join(self.params.output_dir, filename)
        
        dpi = 300 if self.params.high_resolution else 150
        fig.savefig(filepath, dpi=dpi, facecolor='black', edgecolor='none')
        plt.close(fig)
        
        elapsed = time.time() - start_time
        logger.info(f"Artwork generated in {elapsed:.2f}s: {filepath}")
        
        return filepath
    
    def generate_animation(self, title: Optional[str] = None) -> str:
        """Generate animated artwork."""
        start_time = time.time()
        
        if title is None:
            title = f"Quantum Animation #{self.params.topology_seed}"
        
        logger.info("Creating animated visualization...")
        anim = self.renderer.create_animation(self.manifold_generator)
        
        # Save animation
        timestamp = int(time.time())
        filename = f"quantum_animation_{timestamp}_{self.params.topology_seed}.mp4"
        filepath = os.path.join(self.params.output_dir, filename)
        
        # Use different writers based on availability
        try:
            writer = animation.FFMpegWriter(fps=24, bitrate=2000)
            anim.save(filepath, writer=writer)
        except Exception as e:
            logger.warning(f"FFMpeg not available: {e}")
            try:
                anim.save(filepath, writer='pillow', fps=12)
            except Exception as e2:
                logger.error(f"Could not save animation: {e2}")
                return ""
        
        plt.close()
        
        elapsed = time.time() - start_time
        logger.info(f"Animation generated in {elapsed:.2f}s: {filepath}")
        
        return filepath
    
    def generate_series(self, n_artworks: int = 5, 
                       include_animations: bool = False) -> List[str]:
        """Generate a series of artworks with different parameters."""
        filepaths = []
        
        original_seed = self.params.topology_seed
        
        for i in range(n_artworks):
            # Vary parameters for each artwork
            self.params.topology_seed = original_seed + i * 42
            self.params.noise_scale = 0.05 + 0.1 * np.random.random()
            self.params.smoothing_sigma = 1.0 + 3.0 * np.random.random()
            
            # Recreate generators with new parameters
            self.manifold_generator = QuantumManifoldGenerator(self.params)
            
            # Generate static artwork
            filepath = self.generate_single_artwork(f"Quantum Series #{i+1}")
            filepaths.append(filepath)
            
            # Optionally generate animation
            if include_animations and i % 2 == 0:  # Every other artwork
                anim_path = self.generate_animation(f"Quantum Animation Series #{i+1}")
                if anim_path:
                    filepaths.append(anim_path)
        
        # Restore original parameters
        self.params.topology_seed = original_seed
        self.manifold_generator = QuantumManifoldGenerator(self.params)
        
        logger.info(f"Generated series of {len(filepaths)} artworks")
        return filepaths
    
    def generate_interactive_demo(self) -> Optional[str]:
        """Generate interactive visualization using Plotly (if available)."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive demos")
            return None
        
        # Generate data
        X, Y, Z = self.manifold_generator.generate_base_manifold()
        curvature = self.manifold_generator.compute_manifold_curvature(X, Y, Z)
        energy = self.manifold_generator.compute_energy_field(X, Y, Z)
        
        # Create interactive visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Curvature Field', 'Energy Field', 
                          'Combined Field', 'Correlation'],
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                  [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )
        
        # Add heatmaps
        fig.add_trace(go.Heatmap(z=curvature, colorscale='RdBu'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=energy, colorscale='Plasma'), row=1, col=2)
        fig.add_trace(go.Heatmap(z=0.6 * curvature + 0.4 * energy, 
                                colorscale='Viridis'), row=2, col=1)
        
        # Add correlation scatter
        n_sample = min(1000, curvature.size)
        indices = np.random.choice(curvature.size, n_sample, replace=False)
        curv_flat = curvature.flatten()[indices]
        energy_flat = energy.flatten()[indices]
        
        fig.add_trace(go.Scatter(x=curv_flat, y=energy_flat, mode='markers',
                                marker=dict(size=3, opacity=0.6)), row=2, col=2)
        
        fig.update_layout(title="Interactive Quantum Curvature-Energy Visualization",
                         height=800, showlegend=False)
        
        # Save interactive plot
        timestamp = int(time.time())
        filename = f"interactive_quantum_art_{timestamp}.html"
        filepath = os.path.join(self.params.output_dir, filename)
        fig.write_html(filepath)
        
        logger.info(f"Interactive visualization saved: {filepath}")
        return filepath


def create_demo_artworks():
    """Create demonstration artworks with different styles."""
    color_schemes = ['quantum_plasma', 'cosmic', 'neon', 'aurora']
    
    for scheme in color_schemes:
        params = ArtisticParameters(
            color_scheme=scheme,
            topology_seed=42 + hash(scheme) % 1000,
            grid_size=256,
            noise_scale=0.15,
            high_resolution=True,
            output_dir=f'demo_art_{scheme}'
        )
        
        engine = GenerativeArtEngine(params)
        
        # Generate static artwork
        engine.generate_single_artwork(f"Demo: {scheme.title()} Style")
        
        # Generate small animation for demo
        params.n_frames = 60
        engine.generate_animation(f"Demo Animation: {scheme.title()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum-Inspired Generative Art Engine")
    parser.add_argument("--mode", choices=['single', 'series', 'demo', 'interactive'], 
                       default='single', help="Generation mode")
    parser.add_argument("--color-scheme", choices=['quantum_plasma', 'cosmic', 'neon', 'aurora'],
                       default='quantum_plasma', help="Color scheme")
    parser.add_argument("--grid-size", type=int, default=256, help="Grid resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--3d", action='store_true', help="Enable 3D rendering")
    parser.add_argument("--animate", action='store_true', help="Generate animation")
    parser.add_argument("--output-dir", default='generative_art_output', help="Output directory")
    
    args = parser.parse_args()
    
    # Setup parameters
    params = ArtisticParameters(
        color_scheme=args.color_scheme,
        grid_size=args.grid_size,
        topology_seed=args.seed,
        enable_3d=args.three_d,
        output_dir=args.output_dir
    )
    
    engine = GenerativeArtEngine(params)
    
    if args.mode == 'single':
        if args.animate:
            engine.generate_animation()
        else:
            engine.generate_single_artwork()
    elif args.mode == 'series':
        engine.generate_series(n_artworks=5, include_animations=args.animate)
    elif args.mode == 'demo':
        create_demo_artworks()
    elif args.mode == 'interactive':
        engine.generate_interactive_demo()
    
    print(f"✨ Generative art completed! Check {args.output_dir}/ for results.") 