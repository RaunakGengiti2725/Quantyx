"""
Quantum-Conditioned 3D Curvature Art Generator

Advanced generative system that transforms semantic prompts into high-resolution 3D 
structures through quantum-curvature simulations. Maps natural language descriptions 
to geometric parameters including curvature intensity, field coherence, nodal symmetry, 
and gradient entropy.

Features:
- Semantic prompt → geometric parameter mapping
- Toroidal and multi-connected manifold generation
- Quantum interference simulation with multi-scale curvature modulation
- Surface deformation tensors and energy gradient fields
- Topological phase shifts based on descriptive tokens
- 3D mesh export (GLB/OBJ) with embedded metadata
- Photonic glow effects and void-like transitions
"""

from __future__ import annotations

import logging
import os
import re
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
from scipy.spatial import SphericalVoronoi, distance_matrix
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize

# Conditional imports for 3D processing
try:
    from skimage import measure
    from skimage.filters import gaussian
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available - some 3D processing features disabled")

# Import from existing modules
from curvature_energy_analysis import (
    compute_curvature, 
    compute_energy_deltas,
    safe_pearson_correlation
)

try:
    import trimesh
    import pygltflib
    MESH_EXPORT_AVAILABLE = True
except ImportError:
    MESH_EXPORT_AVAILABLE = False
    logging.warning("trimesh/pygltflib not available - 3D mesh export disabled")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QuantumGeometryParameters:
    """Parameters derived from semantic prompt interpretation."""
    
    # Core geometric properties
    curvature_intensity: float = 1.0  # [0.1, 5.0]
    field_coherence: float = 0.5      # [0.0, 1.0] 
    nodal_symmetry: int = 6           # [3, 12]
    gradient_entropy: float = 0.3     # [0.0, 1.0]
    
    # Topological properties
    genus: int = 1                    # Topological genus (0=sphere, 1=torus, 2=double-torus, etc.)
    connectivity: str = "torus"       # "sphere", "torus", "klein_bottle", "multi_torus"
    phase_shift_magnitude: float = 0.2 # [0.0, 1.0]
    
    # Quantum field properties  
    interference_scale: float = 1.0   # Multi-scale interference strength
    quantum_noise: float = 0.1        # Quantum fluctuation amplitude
    coherence_length: float = 2.0     # Spatial coherence scale
    
    # Energy landscape
    energy_well_depth: float = 1.0    # Potential well strength
    barrier_height: float = 0.5       # Energy barrier magnitude
    dissipation_rate: float = 0.1     # Energy dissipation
    
    # Aesthetic properties
    glow_intensity: float = 0.8       # High-energy photonic glow
    void_contrast: float = 0.9        # Low-energy void darkness
    transition_sharpness: float = 0.5 # Wavefunction transition width
    
    # Resolution and quality
    resolution: int = 128             # Base grid resolution
    detail_level: int = 3             # Multi-scale detail levels
    mesh_quality: str = "high"        # "low", "medium", "high", "ultra"


class SemanticParameterMapper:
    """Maps natural language prompts to quantum geometry parameters."""
    
    def __init__(self):
        # Semantic keyword mappings
        self.intensity_keywords = {
            'intense': 3.0, 'violent': 4.0, 'explosive': 5.0, 'powerful': 2.5,
            'gentle': 0.3, 'soft': 0.2, 'subtle': 0.5, 'mild': 0.4,
            'dramatic': 3.5, 'extreme': 4.5, 'minimal': 0.1
        }
        
        self.coherence_keywords = {
            'coherent': 0.9, 'organized': 0.8, 'structured': 0.7, 'ordered': 0.8,
            'chaotic': 0.1, 'random': 0.2, 'turbulent': 0.15, 'disordered': 0.1,
            'crystalline': 0.95, 'fractal': 0.6, 'geometric': 0.85
        }
        
        self.symmetry_keywords = {
            'triangular': 3, 'square': 4, 'pentagonal': 5, 'hexagonal': 6,
            'octagonal': 8, 'symmetric': 6, 'asymmetric': 7, 'irregular': 9,
            'complex': 12, 'simple': 3, 'elaborate': 10
        }
        
        self.topology_keywords = {
            'spherical': 'sphere', 'toroidal': 'torus', 'twisted': 'klein_bottle',
            'connected': 'multi_torus', 'knotted': 'torus', 'simple': 'sphere',
            'complex': 'multi_torus', 'intertwined': 'multi_torus'
        }
        
        self.energy_keywords = {
            'energetic': 2.0, 'explosive': 3.0, 'calm': 0.3, 'serene': 0.2,
            'dynamic': 2.5, 'static': 0.4, 'flowing': 1.5, 'stagnant': 0.3,
            'pulsing': 2.8, 'stable': 0.5, 'volatile': 3.5
        }
        
        self.aesthetic_keywords = {
            'glowing': (1.0, 0.3), 'bright': (0.8, 0.4), 'luminous': (0.9, 0.2),
            'dark': (0.2, 0.9), 'void': (0.1, 1.0), 'shadowy': (0.3, 0.8),
            'ethereal': (0.7, 0.6), 'solid': (0.4, 0.3), 'translucent': (0.6, 0.5)
        }
    
    def parse_prompt(self, prompt: str) -> QuantumGeometryParameters:
        """Parse semantic prompt and extract geometry parameters."""
        prompt_lower = prompt.lower()
        tokens = re.findall(r'\b\w+\b', prompt_lower)
        
        params = QuantumGeometryParameters()
        
        # Extract curvature intensity
        intensity_scores = [self.intensity_keywords.get(token, 0) for token in tokens]
        if any(intensity_scores):
            params.curvature_intensity = max(intensity_scores)
        
        # Extract field coherence
        coherence_scores = [self.coherence_keywords.get(token, -1) for token in tokens]
        coherence_scores = [s for s in coherence_scores if s >= 0]
        if coherence_scores:
            params.field_coherence = np.mean(coherence_scores)
        
        # Extract nodal symmetry
        symmetry_scores = [self.symmetry_keywords.get(token, 0) for token in tokens]
        if any(symmetry_scores):
            params.nodal_symmetry = max(symmetry_scores)
        
        # Extract topology
        for token in tokens:
            if token in self.topology_keywords:
                params.connectivity = self.topology_keywords[token]
                break
        
        # Determine genus from connectivity and descriptors
        if params.connectivity == 'sphere':
            params.genus = 0
        elif params.connectivity == 'torus':
            params.genus = 1
        elif params.connectivity == 'klein_bottle':
            params.genus = 1
        elif params.connectivity == 'multi_torus':
            # Count complexity indicators
            complexity_count = sum(1 for token in tokens if token in ['complex', 'multiple', 'intertwined'])
            params.genus = min(3, max(1, 1 + complexity_count))
        
        # Extract energy characteristics
        energy_scores = [self.energy_keywords.get(token, 0) for token in tokens]
        if any(energy_scores):
            params.energy_well_depth = max(energy_scores)
        
        # Extract aesthetic properties
        glow_sum, void_sum = 0, 0
        aesthetic_count = 0
        for token in tokens:
            if token in self.aesthetic_keywords:
                glow, void = self.aesthetic_keywords[token]
                glow_sum += glow
                void_sum += void
                aesthetic_count += 1
        
        if aesthetic_count > 0:
            params.glow_intensity = glow_sum / aesthetic_count
            params.void_contrast = void_sum / aesthetic_count
        
        # Derive secondary parameters
        params.gradient_entropy = 1.0 - params.field_coherence
        params.phase_shift_magnitude = params.gradient_entropy * 0.5
        params.interference_scale = 0.5 + 1.5 * params.curvature_intensity / 5.0
        params.quantum_noise = 0.05 + 0.2 * params.gradient_entropy
        
        # Adjust resolution based on complexity
        complexity_factor = (params.curvature_intensity + params.genus + params.nodal_symmetry / 12) / 3
        params.resolution = int(64 + 192 * complexity_factor)
        params.resolution = min(512, max(64, params.resolution))
        
        if any(word in prompt_lower for word in ['detailed', 'intricate', 'complex', 'high-resolution']):
            params.mesh_quality = "ultra"
            params.detail_level = 4
        elif any(word in prompt_lower for word in ['simple', 'basic', 'minimal']):
            params.mesh_quality = "low"
            params.detail_level = 1
        
        logger.info(f"Parsed prompt parameters: intensity={params.curvature_intensity:.2f}, "
                   f"coherence={params.field_coherence:.2f}, genus={params.genus}, "
                   f"connectivity={params.connectivity}")
        
        return params


class QuantumManifoldGenerator3D:
    """Generate 3D quantum-curvature manifolds with advanced topologies."""
    
    def __init__(self, params: QuantumGeometryParameters):
        self.params = params
        self.rng = np.random.default_rng(42)
        
    def generate_base_topology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate base 3D topology based on connectivity type."""
        if self.params.connectivity == "sphere":
            return self._generate_spherical_topology()
        elif self.params.connectivity == "torus":
            return self._generate_toroidal_topology()
        elif self.params.connectivity == "klein_bottle":
            return self._generate_klein_bottle_topology()
        elif self.params.connectivity == "multi_torus":
            return self._generate_multi_torus_topology()
        else:
            return self._generate_toroidal_topology()  # Default fallback
    
    def _generate_spherical_topology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate spherical base manifold."""
        phi = np.linspace(0, np.pi, self.params.resolution)
        theta = np.linspace(0, 2*np.pi, self.params.resolution)
        Phi, Theta = np.meshgrid(phi, theta)
        
        # Base sphere with quantum fluctuations
        R = 1.0 + self.params.quantum_noise * self.rng.normal(0, 0.1, Phi.shape)
        
        X = R * np.sin(Phi) * np.cos(Theta)
        Y = R * np.sin(Phi) * np.sin(Theta)
        Z = R * np.cos(Phi)
        
        return X, Y, Z
    
    def _generate_toroidal_topology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate toroidal base manifold."""
        u = np.linspace(0, 2*np.pi, self.params.resolution)
        v = np.linspace(0, 2*np.pi, self.params.resolution)
        U, V = np.meshgrid(u, v)
        
        # Torus parameters
        R = 3.0  # Major radius
        r = 1.0  # Minor radius
        
        # Add quantum deformations
        R_eff = R + self.params.quantum_noise * self.rng.normal(0, 0.2, U.shape)
        r_eff = r + self.params.quantum_noise * self.rng.normal(0, 0.1, U.shape)
        
        X = (R_eff + r_eff * np.cos(V)) * np.cos(U)
        Y = (R_eff + r_eff * np.cos(V)) * np.sin(U)
        Z = r_eff * np.sin(V)
        
        return X, Y, Z
    
    def _generate_klein_bottle_topology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Klein bottle topology."""
        u = np.linspace(0, 2*np.pi, self.params.resolution)
        v = np.linspace(0, 2*np.pi, self.params.resolution)
        U, V = np.meshgrid(u, v)
        
        # Klein bottle parametrization
        X = (2 + np.cos(V/2) * np.sin(U) - np.sin(V/2) * np.sin(2*U)) * np.cos(V)
        Y = (2 + np.cos(V/2) * np.sin(U) - np.sin(V/2) * np.sin(2*U)) * np.sin(V)
        Z = np.sin(V/2) * np.sin(U) + np.cos(V/2) * np.sin(2*U)
        
        # Add quantum fluctuations
        noise_amplitude = self.params.quantum_noise * 0.1
        X += noise_amplitude * self.rng.normal(0, 1, X.shape)
        Y += noise_amplitude * self.rng.normal(0, 1, Y.shape)
        Z += noise_amplitude * self.rng.normal(0, 1, Z.shape)
        
        return X, Y, Z
    
    def _generate_multi_torus_topology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate multi-connected torus topology."""
        # Create multiple connected tori
        n_tori = self.params.genus
        
        combined_X, combined_Y, combined_Z = [], [], []
        
        for i in range(n_tori):
            # Offset and scale each torus
            offset_x = 4.0 * i * np.cos(2*np.pi*i/n_tori)
            offset_y = 4.0 * i * np.sin(2*np.pi*i/n_tori)
            offset_z = 0.5 * i
            
            X_torus, Y_torus, Z_torus = self._generate_toroidal_topology()
            
            # Scale and translate
            scale = 0.8 + 0.4 * np.sin(np.pi * i / n_tori)
            X_torus = scale * X_torus + offset_x
            Y_torus = scale * Y_torus + offset_y
            Z_torus = scale * Z_torus + offset_z
            
            combined_X.append(X_torus)
            combined_Y.append(Y_torus)
            combined_Z.append(Z_torus)
        
        # Combine all tori (use first one as base, add others as perturbations)
        X = combined_X[0]
        Y = combined_Y[0]
        Z = combined_Z[0]
        
        # Add influence from other tori as field perturbations
        for i in range(1, n_tori):
            influence = 0.3 / i  # Decreasing influence
            X += influence * (combined_X[i] - combined_X[0])
            Y += influence * (combined_Y[i] - combined_Y[0])
            Z += influence * (combined_Z[i] - combined_Z[0])
        
        return X, Y, Z
    
    def compute_surface_deformation_tensor(self, X: np.ndarray, Y: np.ndarray, 
                                         Z: np.ndarray) -> np.ndarray:
        """Compute surface deformation tensor for the manifold."""
        # Compute gradients
        dz_dx, dz_dy = np.gradient(Z)
        
        # Compute second derivatives
        d2z_dx2, _ = np.gradient(dz_dx)
        _, d2z_dy2 = np.gradient(dz_dy)
        d2z_dxdy, _ = np.gradient(dz_dy)
        
        # Gaussian curvature formula
        numerator = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        denominator = (1 + dz_dx**2 + dz_dy**2)**2
        
        # Avoid division by zero
        curvature = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator != 0)
        
        # Surface deformation tensor (combines curvatures with quantum parameters)
        deformation = (
            self.params.curvature_intensity * curvature +
            self.params.phase_shift_magnitude * np.sin(4*np.pi*X) * np.cos(4*np.pi*Y)
        )
        
        return deformation
    
    def compute_quantum_energy_field(self, X: np.ndarray, Y: np.ndarray, 
                                   Z: np.ndarray, deformation: np.ndarray) -> np.ndarray:
        """Compute quantum energy field with localized gradients."""
        # Base potential energy
        r_squared = X**2 + Y**2 + Z**2
        potential = 0.5 * self.params.energy_well_depth * r_squared
        
        # Kinetic energy from deformation gradients
        grad_x, grad_y = np.gradient(deformation)
        kinetic = 0.5 * (grad_x**2 + grad_y**2)
        
        # Quantum interference effects
        interference = np.zeros_like(X)
        scales = [1, 2, 4]
        
        for scale in scales:
            k = 2*np.pi*scale / self.params.coherence_length
            phase = k * (X + Y + Z) + self.params.phase_shift_magnitude * deformation
            interference += (1.0 / scale) * np.sin(phase) * np.exp(-0.1 * scale * r_squared)
        
        # Localized energy wells/barriers
        energy_landscape = np.zeros_like(X)
        
        # Create nodal points based on symmetry
        for i in range(self.params.nodal_symmetry):
            angle = 2*np.pi*i / self.params.nodal_symmetry
            node_x = 2.0 * np.cos(angle)
            node_y = 2.0 * np.sin(angle)
            
            dist_to_node = np.sqrt((X - node_x)**2 + (Y - node_y)**2 + Z**2)
            energy_landscape += -self.params.energy_well_depth * np.exp(-dist_to_node**2)
        
        # Total energy field
        total_energy = potential + kinetic + interference + energy_landscape
        
        # Apply coherence filtering
        if self.params.field_coherence < 1.0:
            noise_strength = 1.0 - self.params.field_coherence
            coherence_filter = gaussian_filter(total_energy, sigma=self.params.coherence_length)
            total_energy = (
                self.params.field_coherence * coherence_filter +
                noise_strength * total_energy
            )
        
        return total_energy


class Quantum3DMeshGenerator:
    """Generate exportable 3D meshes from quantum fields."""
    
    def __init__(self, params: QuantumGeometryParameters):
        self.params = params
        
    def generate_mesh_from_fields(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                 deformation: np.ndarray, energy: np.ndarray) -> Optional[Any]:
        """Generate 3D mesh from quantum field data."""
        if not MESH_EXPORT_AVAILABLE:
            logger.warning("Mesh export libraries not available")
            return None
        
        # Apply deformation to create final surface
        deformation_strength = 0.3 * self.params.curvature_intensity
        X_final = X + deformation_strength * deformation * (X / np.sqrt(X**2 + Y**2 + Z**2 + 1e-8))
        Y_final = Y + deformation_strength * deformation * (Y / np.sqrt(X**2 + Y**2 + Z**2 + 1e-8))
        Z_final = Z + deformation_strength * deformation * (Z / np.sqrt(X**2 + Y**2 + Z**2 + 1e-8))
        
        # Create point cloud
        points = np.column_stack([
            X_final.ravel(),
            Y_final.ravel(), 
            Z_final.ravel()
        ])
        
        # Add energy values as vertex colors
        energy_normalized = (energy - energy.min()) / (energy.max() - energy.min())
        colors = self._energy_to_color(energy_normalized.ravel())
        
        try:
            # Create mesh using alpha shapes or Delaunay triangulation
            mesh = trimesh.Trimesh()
            
            if len(points) > 1000:  # Subsample for performance
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Use convex hull as a starting point, then apply smoothing
            hull = trimesh.convex.convex_hull(points)
            
            # Apply smoothing and subdivision based on quality settings
            if self.params.mesh_quality in ["high", "ultra"]:
                hull = hull.subdivide()
                if self.params.mesh_quality == "ultra":
                    hull = hull.subdivide()
            
            # Apply vertex colors
            hull.visual.vertex_colors = (colors * 255).astype(np.uint8)
            
            # Add metadata
            hull.metadata['quantum_parameters'] = {
                'curvature_intensity': self.params.curvature_intensity,
                'field_coherence': self.params.field_coherence,
                'nodal_symmetry': self.params.nodal_symmetry,
                'connectivity': self.params.connectivity,
                'genus': self.params.genus
            }
            
            return hull
            
        except Exception as e:
            logger.error(f"Mesh generation failed: {e}")
            return None
    
    def _energy_to_color(self, energy_normalized: np.ndarray) -> np.ndarray:
        """Convert energy values to RGB colors based on aesthetic parameters."""
        # Create color mapping based on glow/void parameters
        colors = np.zeros((len(energy_normalized), 4))  # RGBA
        
        # High energy = photonic glow
        high_energy_mask = energy_normalized > 0.7
        colors[high_energy_mask] = [
            1.0 * self.params.glow_intensity,  # R
            0.8 * self.params.glow_intensity,  # G  
            0.4 * self.params.glow_intensity,  # B
            1.0  # A
        ]
        
        # Low energy = void-like
        low_energy_mask = energy_normalized < 0.3
        void_color = 1.0 - self.params.void_contrast
        colors[low_energy_mask] = [void_color, void_color, void_color, 1.0]
        
        # Transition regions = wavefunction superposition
        transition_mask = (energy_normalized >= 0.3) & (energy_normalized <= 0.7)
        transition_energy = energy_normalized[transition_mask]
        
        # Smooth transition with quantum interference patterns
        transition_phase = 2*np.pi*transition_energy
        interference = 0.5 * (1 + np.sin(transition_phase * self.params.nodal_symmetry))
        
        colors[transition_mask, 0] = 0.3 + 0.4 * interference  # R
        colors[transition_mask, 1] = 0.1 + 0.6 * transition_energy  # G
        colors[transition_mask, 2] = 0.8 + 0.2 * interference  # B
        colors[transition_mask, 3] = 0.7 + 0.3 * transition_energy  # A
        
        return colors
    
    def export_mesh(self, mesh: Any, filepath: str, format: str = "glb") -> bool:
        """Export mesh to file."""
        if mesh is None:
            return False
        
        try:
            if format.lower() == "glb":
                mesh.export(filepath)
            elif format.lower() == "obj":
                mesh.export(filepath)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Mesh exported to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")
            return False


class Quantum3DArtEngine:
    """Main engine for quantum-conditioned 3D art generation."""
    
    def __init__(self, output_dir: str = "quantum_3d_art"):
        self.output_dir = output_dir
        self.semantic_mapper = SemanticParameterMapper()
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        logger.info("Quantum 3D Art Engine initialized")
    
    def generate_from_prompt(self, prompt: str, 
                           export_formats: List[str] = ["glb", "png"]) -> Dict[str, str]:
        """Generate 3D quantum art from semantic prompt."""
        start_time = time.time()
        
        logger.info(f"Processing prompt: '{prompt[:50]}...'")
        
        # Step 1: Parse prompt to geometry parameters
        params = self.semantic_mapper.parse_prompt(prompt)
        
        # Step 2: Generate quantum manifold
        manifold_gen = QuantumManifoldGenerator3D(params)
        X, Y, Z = manifold_gen.generate_base_topology()
        
        # Step 3: Compute surface deformation tensor
        logger.info("Computing surface deformation tensor...")
        deformation = manifold_gen.compute_surface_deformation_tensor(X, Y, Z)
        
        # Step 4: Compute quantum energy field
        logger.info("Computing quantum energy field...")
        energy = manifold_gen.compute_quantum_energy_field(X, Y, Z, deformation)
        
        # Step 5: Generate and export 3D mesh
        mesh_gen = Quantum3DMeshGenerator(params)
        mesh = mesh_gen.generate_mesh_from_fields(X, Y, Z, deformation, energy)
        
        # Step 6: Export files
        timestamp = int(time.time())
        safe_prompt = re.sub(r'[^a-zA-Z0-9_\-]', '_', prompt[:30])
        
        exported_files = {}
        
        # Export 3D mesh
        if mesh and any(fmt in export_formats for fmt in ["glb", "obj"]):
            for fmt in ["glb", "obj"]:
                if fmt in export_formats:
                    mesh_file = f"quantum_3d_{safe_prompt}_{timestamp}.{fmt}"
                    mesh_path = os.path.join(self.output_dir, mesh_file)
                    if mesh_gen.export_mesh(mesh, mesh_path, fmt):
                        exported_files[fmt] = mesh_path
        
        # Export visualization
        if "png" in export_formats:
            vis_file = f"quantum_3d_vis_{safe_prompt}_{timestamp}.png"
            vis_path = os.path.join(self.output_dir, vis_file)
            if self._create_visualization(X, Y, Z, deformation, energy, params, vis_path):
                exported_files["png"] = vis_path
        
        # Export metadata
        metadata_file = f"quantum_3d_metadata_{safe_prompt}_{timestamp}.json"
        metadata_path = os.path.join(self.output_dir, metadata_file)
        self._export_metadata(prompt, params, metadata_path)
        exported_files["metadata"] = metadata_path
        
        elapsed = time.time() - start_time
        logger.info(f"Quantum 3D art generated in {elapsed:.2f}s")
        
        return exported_files
    
    def _create_visualization(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                            deformation: np.ndarray, energy: np.ndarray,
                            params: QuantumGeometryParameters, filepath: str) -> bool:
        """Create and save visualization of the quantum fields."""
        try:
            fig = plt.figure(figsize=(20, 15), facecolor='black')
            
            # 3D surface plot
            ax_3d = fig.add_subplot(221, projection='3d')
            ax_3d.set_facecolor('black')
            
            # Apply deformation for visualization
            deformation_strength = 0.2
            X_deformed = X + deformation_strength * deformation * np.sign(X)
            Y_deformed = Y + deformation_strength * deformation * np.sign(Y)
            Z_deformed = Z + deformation_strength * deformation * np.sign(Z)
            
            # Color based on energy
            energy_norm = (energy - energy.min()) / (energy.max() - energy.min())
            
            # Subsample for visualization performance
            step = max(1, params.resolution // 50)
            surf = ax_3d.plot_surface(
                X_deformed[::step, ::step], 
                Y_deformed[::step, ::step], 
                Z_deformed[::step, ::step],
                facecolors=plt.cm.plasma(energy_norm[::step, ::step]),
                alpha=0.8, linewidth=0, antialiased=True
            )
            
            ax_3d.set_title('3D Quantum Manifold', color='white', fontsize=14)
            ax_3d.grid(False)
            
            # Energy field visualization
            ax_energy = fig.add_subplot(222)
            ax_energy.set_facecolor('black')
            im_energy = ax_energy.imshow(energy, cmap='plasma', interpolation='bilinear')
            ax_energy.set_title('Quantum Energy Field', color='white', fontsize=14)
            ax_energy.axis('off')
            
            # Deformation field visualization  
            ax_deform = fig.add_subplot(223)
            ax_deform.set_facecolor('black')
            im_deform = ax_deform.imshow(deformation, cmap='RdBu_r', interpolation='bilinear')
            ax_deform.set_title('Surface Deformation Tensor', color='white', fontsize=14)
            ax_deform.axis('off')
            
            # Parameter summary
            ax_text = fig.add_subplot(224)
            ax_text.set_facecolor('black')
            ax_text.axis('off')
            
            param_text = f"""Quantum Geometry Parameters:
            
Curvature Intensity: {params.curvature_intensity:.2f}
Field Coherence: {params.field_coherence:.2f}
Nodal Symmetry: {params.nodal_symmetry}
Gradient Entropy: {params.gradient_entropy:.2f}
Topology: {params.connectivity} (genus {params.genus})
Resolution: {params.resolution}x{params.resolution}
            
Energy Characteristics:
Well Depth: {params.energy_well_depth:.2f}
Barrier Height: {params.barrier_height:.2f}
Interference Scale: {params.interference_scale:.2f}
            
Aesthetic Properties:
Glow Intensity: {params.glow_intensity:.2f}
Void Contrast: {params.void_contrast:.2f}
Transition Sharpness: {params.transition_sharpness:.2f}"""
            
            ax_text.text(0.05, 0.95, param_text, transform=ax_text.transAxes,
                        fontsize=10, color='white', verticalalignment='top',
                        fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, facecolor='black', edgecolor='none')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return False
    
    def _export_metadata(self, prompt: str, params: QuantumGeometryParameters, 
                        filepath: str) -> None:
        """Export metadata JSON file."""
        metadata = {
            "prompt": prompt,
            "generation_timestamp": time.time(),
            "parameters": {
                "curvature_intensity": params.curvature_intensity,
                "field_coherence": params.field_coherence,
                "nodal_symmetry": params.nodal_symmetry,
                "gradient_entropy": params.gradient_entropy,
                "genus": params.genus,
                "connectivity": params.connectivity,
                "phase_shift_magnitude": params.phase_shift_magnitude,
                "interference_scale": params.interference_scale,
                "quantum_noise": params.quantum_noise,
                "coherence_length": params.coherence_length,
                "energy_well_depth": params.energy_well_depth,
                "barrier_height": params.barrier_height,
                "glow_intensity": params.glow_intensity,
                "void_contrast": params.void_contrast,
                "resolution": params.resolution,
                "mesh_quality": params.mesh_quality
            },
            "physical_interpretation": {
                "description": f"Quantum manifold with {params.connectivity} topology (genus {params.genus})",
                "energy_scale": "Normalized quantum energy units",
                "curvature_type": "Gaussian curvature with quantum deformation",
                "field_coherence_meaning": "Spatial correlation length of quantum field",
                "symmetry_explanation": f"{params.nodal_symmetry}-fold rotational symmetry in energy landscape"
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Example usage and CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum-Conditioned 3D Art Generator")
    parser.add_argument("prompt", help="Semantic prompt for art generation")
    parser.add_argument("--output-dir", default="quantum_3d_art", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["glb", "png"], 
                       choices=["glb", "obj", "png"], help="Export formats")
    
    args = parser.parse_args()
    
    # Create engine and generate art
    engine = Quantum3DArtEngine(args.output_dir)
    files = engine.generate_from_prompt(args.prompt, args.formats)
    
    print("\n✨ Quantum 3D Art Generation Complete!")
    print(f"📁 Output directory: {args.output_dir}")
    print("📋 Generated files:")
    for format_type, filepath in files.items():
        print(f"  {format_type}: {filepath}")


if __name__ == "__main__":
    main() 