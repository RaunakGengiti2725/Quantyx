# Quantum-Conditioned 3D Curvature Art Generator

An advanced generative art system that transforms semantic prompts into high-resolution 3D structures through quantum-curvature simulations. This system maps natural language descriptions to sophisticated geometric parameters and generates exportable 3D meshes with embedded quantum-state metadata.

![Quantum Art Banner](https://img.shields.io/badge/Quantum-3D%20Art-purple?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge) ![Math](https://img.shields.io/badge/Differential-Geometry-green?style=for-the-badge)

## 🌟 Features

### Core Capabilities
- **Semantic Prompt Interpretation**: Maps natural language to geometric parameters
- **Advanced Topology Generation**: Spherical, toroidal, Klein bottle, and multi-connected manifolds
- **Quantum Field Simulation**: Multi-scale interference effects and energy landscapes
- **3D Mesh Export**: GLB/OBJ format with embedded metadata
- **Physical Interpretability**: Each artwork backed by actual mathematical physics

### Semantic Parameter Mapping
- **Curvature Intensity**: `gentle`, `intense`, `explosive`, `dramatic`
- **Field Coherence**: `crystalline`, `chaotic`, `organized`, `turbulent`
- **Nodal Symmetry**: `triangular`, `hexagonal`, `complex`, `symmetric`
- **Topology**: `spherical`, `toroidal`, `twisted`, `intertwined`
- **Energy Aesthetics**: `glowing`, `void`, `luminous`, `ethereal`

### Advanced Features
- **Multi-scale Curvature Modulation**: Simulates quantum interference
- **Surface Deformation Tensors**: Physics-based geometric constraints
- **Photonic Glow Effects**: High-energy zones emit luminous effects
- **Void Transitions**: Low-energy regions appear minimal/void-like
- **Topological Phase Shifts**: Based on descriptive tokens

## 🚀 Quick Start

### Installation

1. **Install base dependencies:**
```bash
cd quantumproject-main
pip install -r requirements.txt
```

2. **Install 3D art dependencies:**
```bash
pip install -r requirements_3d_art.txt
```

### Basic Usage

```python
from quantum_3d_art_generator import Quantum3DArtEngine

# Initialize the engine
engine = Quantum3DArtEngine(output_dir="my_quantum_art")

# Generate art from a semantic prompt
files = engine.generate_from_prompt(
    "A crystalline quantum structure with hexagonal symmetry, glowing with intense coherent energy",
    export_formats=["glb", "png"]
)

print("Generated files:", files)
```

### Command Line Interface

```bash
# Generate art from a prompt
python quantum_3d_art_generator.py "explosive volcanic quantum terrain with asymmetric energy eruptions"

# Specify output formats
python quantum_3d_art_generator.py "gentle toroidal flow" --formats glb obj png

# Set custom output directory
python quantum_3d_art_generator.py "complex intertwined topology" --output-dir my_custom_art
```

## 🎨 Demo and Examples

### Run the Interactive Demo

```bash
# Full demo with example prompts
python demo_quantum_3d_art.py

# Interactive mode
python demo_quantum_3d_art.py --interactive

# Analyze semantic mappings
python demo_quantum_3d_art.py --analysis

# Parameter showcase
python demo_quantum_3d_art.py --showcase
```

### Example Prompts

| Prompt | Description | Key Features |
|--------|-------------|--------------|
| `"crystalline quantum structure with hexagonal symmetry"` | High coherence, structured | 🔷 Hexagonal symmetry, crystalline order |
| `"chaotic turbulent manifold with swirling void regions"` | Low coherence, dramatic | 🌪️ Chaotic topology, void aesthetics |
| `"gentle toroidal flow with ethereal transitions"` | Smooth, minimal | 🍩 Toroidal topology, soft transitions |
| `"explosive volcanic quantum terrain"` | High intensity, dramatic | 🌋 Volcanic energy landscape |
| `"twisted Klein bottle with fractal interference"` | Complex topology | 🌀 Klein bottle, fractal patterns |

## 🔬 Technical Architecture

### Parameter Mapping System

The semantic mapper translates natural language into geometric parameters:

```python
# Intensity Keywords → Curvature Intensity (0.1-5.0)
intensity_keywords = {
    'gentle': 0.3, 'intense': 3.0, 'explosive': 5.0,
    'dramatic': 3.5, 'subtle': 0.5, 'violent': 4.0
}

# Coherence Keywords → Field Coherence (0.0-1.0)
coherence_keywords = {
    'crystalline': 0.95, 'chaotic': 0.1, 'organized': 0.8,
    'turbulent': 0.15, 'structured': 0.7, 'random': 0.2
}
```

### Quantum Field Generation

1. **Base Topology**: Generate manifold (sphere, torus, Klein bottle, multi-torus)
2. **Surface Deformation**: Compute Gaussian curvature tensor
3. **Energy Field**: Quantum interference + localized wells/barriers
4. **Aesthetic Mapping**: Energy → color, opacity, glow effects

### Mathematical Foundation

The system computes physical quantities:

- **Gaussian Curvature**: `K = (f_xx * f_yy - f_xy²) / (1 + f_x² + f_y²)²`
- **Quantum Energy**: `E = T_kinetic + V_potential + E_interference`
- **Surface Deformation**: `δ = α·K + β·sin(4πx)cos(4πy)`

## 📊 Output Formats

### 3D Mesh Files
- **GLB**: Binary glTF format (recommended for web/AR/VR)
- **OBJ**: Wavefront OBJ format (widely compatible)
- **Embedded metadata**: Quantum parameters and physical interpretation

### Visualizations
- **PNG**: High-resolution visualization with parameter summary
- **Multi-panel layout**: 3D surface, energy field, deformation tensor, parameters

### Metadata
- **JSON**: Complete parameter set and physical interpretation
- **Timestamps**: Generation metadata
- **Provenance**: Original prompt and parameter mapping

## 🎯 Advanced Usage

### Custom Parameter Control

```python
from quantum_3d_art_generator import QuantumGeometryParameters

# Create custom parameters
params = QuantumGeometryParameters(
    curvature_intensity=2.5,
    field_coherence=0.8,
    nodal_symmetry=8,
    connectivity="klein_bottle",
    resolution=256,
    mesh_quality="ultra"
)

# Use with generator
manifold_gen = QuantumManifoldGenerator3D(params)
X, Y, Z = manifold_gen.generate_base_topology()
```

### Batch Generation

```python
prompts = [
    "intense crystalline hexagonal structure",
    "chaotic void with explosive energy",
    "gentle spherical quantum bubble"
]

engine = Quantum3DArtEngine()

for prompt in prompts:
    files = engine.generate_from_prompt(prompt)
    print(f"Generated: {files}")
```

### Performance Optimization

```python
# For high-performance generation
params = QuantumGeometryParameters(
    resolution=128,        # Balance quality vs. speed
    mesh_quality="medium", # Reduce subdivision levels
    detail_level=2        # Fewer multi-scale levels
)
```

## 🏗️ Architecture Overview

```
Semantic Prompt
     ↓
Parameter Mapper → QuantumGeometryParameters
     ↓
Manifold Generator → 3D Topology (X, Y, Z)
     ↓
Physics Engine → Curvature + Energy Fields
     ↓
Mesh Generator → 3D Mesh with Colors
     ↓
Export System → GLB/OBJ + PNG + JSON
```

### Core Classes

- **`SemanticParameterMapper`**: NLP → geometry parameters
- **`QuantumManifoldGenerator3D`**: Generate base topologies
- **`Quantum3DMeshGenerator`**: Convert fields → 3D meshes
- **`Quantum3DArtEngine`**: Main orchestration engine

## 🔧 Dependencies

### Required
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
networkx>=2.6.0
scikit-image>=0.19.0
```

### Optional (Enhanced Features)
```
trimesh>=3.15.0          # 3D mesh processing
pygltflib>=1.15.0        # GLB export
open3d>=0.16.0           # Advanced mesh operations
plotly>=5.0.0            # Interactive visualization
cupy>=10.0.0             # GPU acceleration
```

## 🎨 Gallery Examples

### Crystalline Structures
- **Prompt**: `"crystalline quantum structure with hexagonal symmetry"`
- **Features**: High coherence, structured energy wells, geometric precision

### Chaotic Manifolds  
- **Prompt**: `"chaotic turbulent manifold with explosive energy"`
- **Features**: Low coherence, dramatic deformations, complex topology

### Ethereal Flows
- **Prompt**: `"gentle toroidal flow with ethereal transitions"`
- **Features**: Smooth gradients, minimal intensity, soft aesthetics

## 🚧 Advanced Features

### Quantum Interference Simulation
Multi-scale interference patterns simulate quantum field effects:

```python
# Multi-scale interference computation
scales = [1, 2, 4, 8]
for scale in scales:
    k = 2π * scale / coherence_length
    phase = k * (x + y + z) + phase_shift * deformation
    interference += (1/scale) * sin(phase) * exp(-0.1 * scale * r²)
```

### Topological Phase Shifts
Parameter-dependent phase transitions based on descriptive tokens:
- `entangled` → increased interference coupling
- `chaotic` → reduced field coherence  
- `crystalline` → enhanced structural symmetry
- `decaying` → time-dependent energy dissipation

### Aesthetic Mapping Rules
- **High-energy zones** (E > 0.7): Photonic glow effects
- **Low-energy zones** (E < 0.3): Void-like darkness
- **Transition regions**: Wavefunction superposition visualization

## 📈 Performance & Scaling

| Resolution | Generation Time | Memory Usage | Quality |
|------------|-----------------|--------------|---------|
| 64×64      | ~2-5 seconds   | ~100 MB     | Preview |
| 128×128    | ~5-15 seconds  | ~400 MB     | Standard |
| 256×256    | ~15-45 seconds | ~1.5 GB     | High |
| 512×512    | ~1-3 minutes   | ~6 GB       | Ultra |

### Optimization Tips
- Use `resolution=128` for interactive development
- Enable GPU acceleration with CuPy for large resolutions
- Set `mesh_quality="medium"` for faster generation
- Use `detail_level=2` to reduce computation time

## 🤝 Contributing

We welcome contributions! Areas of interest:
- **New topology types**: Additional manifold geometries
- **Enhanced semantics**: Improved NLP → parameter mapping
- **Aesthetic algorithms**: Novel color/glow mapping functions
- **Performance**: GPU optimization and parallel processing
- **Export formats**: Additional 3D file format support

## 📖 Citation

```bibtex
@software{quantum_3d_art_2024,
  title={Quantum-Conditioned 3D Curvature Art Generator},
  author={QuantumProject Team},
  year={2024},
  url={https://github.com/quantumproject/3d-art},
  description={Semantic prompt to 3D quantum art generation}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Differential geometry foundations from modern mathematical physics
- Quantum field theory inspiration for interference patterns
- Generative art community for aesthetic principles
- Open source 3D graphics ecosystem (trimesh, Open3D, etc.)

---

**✨ Create infinite quantum-inspired 3D artworks from simple text prompts!**

For questions, issues, or collaboration: [Create an issue](https://github.com/quantumproject/issues) or reach out to the development team. 