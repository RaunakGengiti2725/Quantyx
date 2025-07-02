#!/usr/bin/env python3
"""
Simple Quantum Art Demo - Core Dependencies Only

Generates quantum-inspired art from semantic prompts using only numpy, matplotlib, and scipy.
Creates beautiful 2D visualizations that demonstrate the quantum field principles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import re
import os
import time
from pathlib import Path

# Set dark artistic style
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'savefig.facecolor': 'black',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white'
})


class SimpleSemanticMapper:
    """Maps semantic prompts to quantum art parameters."""
    
    def __init__(self):
        self.intensity_keywords = {
            'explosive': 5.0, 'intense': 3.5, 'violent': 4.0, 'dramatic': 3.0,
            'powerful': 2.5, 'strong': 2.0, 'moderate': 1.5, 'gentle': 0.5,
            'soft': 0.3, 'subtle': 0.4, 'minimal': 0.2, 'weak': 0.3
        }
        
        self.coherence_keywords = {
            'crystalline': 0.95, 'structured': 0.85, 'organized': 0.80, 'ordered': 0.75,
            'geometric': 0.90, 'symmetric': 0.85, 'regular': 0.80, 'coherent': 0.90,
            'chaotic': 0.1, 'turbulent': 0.15, 'random': 0.2, 'disordered': 0.1,
            'irregular': 0.25, 'scattered': 0.3, 'wild': 0.15
        }
        
        self.topology_keywords = {
            'vortex': 'spiral', 'spiral': 'spiral', 'swirling': 'spiral',
            'wave': 'wave', 'ripple': 'wave', 'flowing': 'wave',
            'radial': 'radial', 'circular': 'radial', 'centered': 'radial',
            'linear': 'linear', 'parallel': 'linear', 'striped': 'linear'
        }
        
        self.color_keywords = {
            'luminous': 'plasma', 'glowing': 'plasma', 'bright': 'plasma',
            'void': 'viridis_r', 'dark': 'viridis_r', 'shadow': 'viridis_r',
            'fire': 'hot', 'flame': 'hot', 'volcanic': 'hot',
            'ice': 'cool', 'frozen': 'cool', 'crystal': 'cool',
            'electric': 'electric', 'energy': 'electric', 'quantum': 'electric'
        }
    
    def parse_prompt(self, prompt):
        """Parse prompt and extract parameters."""
        tokens = re.findall(r'\b\w+\b', prompt.lower())
        
        # Extract intensity
        intensity_scores = [self.intensity_keywords.get(token, 0) for token in tokens]
        intensity = max(intensity_scores) if any(intensity_scores) else 1.5
        
        # Extract coherence
        coherence_scores = [self.coherence_keywords.get(token, -1) for token in tokens]
        coherence_scores = [s for s in coherence_scores if s >= 0]
        coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        # Extract topology
        topology = 'radial'  # default
        for token in tokens:
            if token in self.topology_keywords:
                topology = self.topology_keywords[token]
                break
        
        # Extract color scheme
        color_scheme = 'plasma'  # default
        for token in tokens:
            if token in self.color_keywords:
                color_scheme = self.color_keywords[token]
                break
        
        return {
            'intensity': intensity,
            'coherence': coherence,
            'topology': topology,
            'color_scheme': color_scheme,
            'tokens': tokens
        }


class QuantumArtGenerator:
    """Generate quantum-inspired 2D art."""
    
    def __init__(self, resolution=256):
        self.resolution = resolution
        self.x = np.linspace(-4, 4, resolution)
        self.y = np.linspace(-4, 4, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)
    
    def generate_quantum_field(self, params):
        """Generate quantum field based on parameters."""
        intensity = params['intensity']
        coherence = params['coherence']
        topology = params['topology']
        
        # Base quantum field
        if topology == 'spiral':
            # Spiral vortex pattern
            spiral_field = intensity * np.sin(3 * self.Theta + 0.5 * self.R) * np.exp(-0.1 * self.R)
            wave_field = np.sin(2 * np.pi * self.R / 2) * np.cos(4 * self.Theta)
        elif topology == 'wave':
            # Wave interference pattern
            spiral_field = intensity * np.sin(2 * np.pi * self.X / 2) * np.sin(2 * np.pi * self.Y / 2)
            wave_field = np.sin(np.sqrt(self.X**2 + self.Y**2) * np.pi) * np.exp(-0.05 * self.R)
        elif topology == 'radial':
            # Radial energy pattern
            spiral_field = intensity * np.sin(4 * np.pi * self.R / 3) * np.exp(-0.08 * self.R)
            wave_field = np.cos(6 * self.Theta) * np.exp(-0.1 * self.R)
        else:  # linear
            # Linear wave pattern
            spiral_field = intensity * np.sin(2 * np.pi * (self.X + self.Y) / 2)
            wave_field = np.sin(2 * np.pi * (self.X - self.Y) / 2)
        
        # Combine fields
        quantum_field = spiral_field + 0.5 * wave_field
        
        # Add quantum interference
        interference_scales = [1, 2, 4]
        for scale in interference_scales:
            k = 2 * np.pi * scale
            phase = k * (self.X + self.Y) / 4
            interference = (1.0 / scale) * np.sin(phase) * np.exp(-0.1 * scale * self.R**2 / 16)
            quantum_field += 0.3 * intensity * interference
        
        # Apply coherence filtering
        if coherence < 1.0:
            noise_strength = 1.0 - coherence
            noise = noise_strength * np.random.normal(0, 0.2, quantum_field.shape)
            coherent_field = gaussian_filter(quantum_field, sigma=coherence * 2)
            quantum_field = coherence * coherent_field + noise_strength * (quantum_field + noise)
        else:
            quantum_field = gaussian_filter(quantum_field, sigma=1.0)
        
        return quantum_field
    
    def create_color_scheme(self, scheme_name):
        """Create custom color schemes."""
        if scheme_name == 'electric':
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['#000033', '#000066', '#0033cc', '#0066ff', '#33aaff', '#66ccff', '#ffffff']
            return LinearSegmentedColormap.from_list('electric', colors)
        elif scheme_name == 'cool':
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['#000022', '#002244', '#004488', '#0066aa', '#3399cc', '#66ccee', '#aaeeff']
            return LinearSegmentedColormap.from_list('cool', colors)
        else:
            return plt.cm.get_cmap(scheme_name)
    
    def render_art(self, params, title="Quantum Field Art"):
        """Render the quantum art visualization."""
        # Generate quantum field
        field = self.generate_quantum_field(params)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        
        # Main quantum field visualization
        ax_main = plt.subplot(2, 2, 1)
        ax_main.set_facecolor('black')
        
        # Get color scheme
        if params['color_scheme'] in ['electric', 'cool']:
            cmap = self.create_color_scheme(params['color_scheme'])
        else:
            cmap = plt.cm.get_cmap(params['color_scheme'])
        
        # Main field plot
        im_main = ax_main.imshow(field, extent=[-4, 4, -4, 4], cmap=cmap, 
                               interpolation='bilinear', alpha=0.9)
        
        # Add contour lines for quantum levels
        contours = ax_main.contour(self.X, self.Y, field, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        ax_main.set_title(f'Quantum Field: {title}', color='white', fontsize=14, pad=20)
        ax_main.axis('off')
        
        # Energy density plot
        ax_energy = plt.subplot(2, 2, 2)
        ax_energy.set_facecolor('black')
        
        energy_density = field**2
        im_energy = ax_energy.imshow(energy_density, extent=[-4, 4, -4, 4], 
                                   cmap='hot', interpolation='bilinear', alpha=0.8)
        ax_energy.set_title('Energy Density |ψ|²', color='white', fontsize=12)
        ax_energy.axis('off')
        
        # Phase visualization
        ax_phase = plt.subplot(2, 2, 3)
        ax_phase.set_facecolor('black')
        
        phase = np.angle(field + 1j * np.roll(field, 1, axis=0))
        im_phase = ax_phase.imshow(phase, extent=[-4, 4, -4, 4], 
                                 cmap='hsv', interpolation='bilinear', alpha=0.8)
        ax_phase.set_title('Quantum Phase', color='white', fontsize=12)
        ax_phase.axis('off')
        
        # Parameter summary
        ax_params = plt.subplot(2, 2, 4)
        ax_params.set_facecolor('black')
        ax_params.axis('off')
        
        param_text = f"""Quantum Art Parameters:

Intensity: {params['intensity']:.2f}
Coherence: {params['coherence']:.2f}
Topology: {params['topology']}
Color Scheme: {params['color_scheme']}

Field Statistics:
Mean Energy: {np.mean(energy_density):.3f}
Max Amplitude: {np.max(np.abs(field)):.3f}
Coherence Length: {1/params['coherence']:.2f}

Aesthetic Properties:
High-energy regions: {np.sum(energy_density > np.percentile(energy_density, 80))} pixels
Low-energy regions: {np.sum(energy_density < np.percentile(energy_density, 20))} pixels
        """
        
        ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes,
                      fontsize=10, color='white', verticalalignment='top',
                      fontfamily='monospace')
        
        plt.tight_layout()
        return fig


def generate_art_from_prompt(prompt, output_dir="quantum_art_output"):
    """Generate quantum art from a semantic prompt."""
    
    print(f"🎨 Generating quantum art from prompt:")
    print(f"📝 \"{prompt}\"")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse prompt
    mapper = SimpleSemanticMapper()
    params = mapper.parse_prompt(prompt)
    
    print("⚙️  Parsed parameters:")
    print(f"   Intensity: {params['intensity']:.2f}")
    print(f"   Coherence: {params['coherence']:.2f}")
    print(f"   Topology: {params['topology']}")
    print(f"   Color Scheme: {params['color_scheme']}")
    print()
    
    # Generate art
    print("🌌 Generating quantum field...")
    generator = QuantumArtGenerator(resolution=256)
    
    start_time = time.time()
    fig = generator.render_art(params, title=f"'{prompt[:30]}...'")
    
    # Save the artwork
    timestamp = int(time.time())
    safe_prompt = re.sub(r'[^a-zA-Z0-9_\-]', '_', prompt[:40])
    filename = f"quantum_art_{safe_prompt}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='black')
    
    elapsed = time.time() - start_time
    print(f"✅ Artwork generated in {elapsed:.2f}s")
    print(f"💾 Saved to: {filepath}")
    
    # Show the plot
    plt.show()
    
    return filepath, params


def main():
    """Main demo function."""
    
    print("🌟 Simple Quantum Art Generator Demo")
    print("=" * 40)
    print()
    
    # Example prompts to demonstrate different styles
    example_prompts = [
        "explosive crystalline quantum vortex with chaotic energy flows and luminous void boundaries",
        "gentle ethereal wave patterns with structured geometric harmony",
        "intense volcanic fire energy with dramatic radial symmetry",
        "soft ice crystal formations with minimal coherent flows"
    ]
    
    # Let's use the first exciting prompt
    chosen_prompt = example_prompts[0]
    
    try:
        filepath, params = generate_art_from_prompt(chosen_prompt)
        
        print()
        print("🎉 Quantum art generation complete!")
        print(f"🖼️  View your artwork: {filepath}")
        print()
        print("💡 Try more prompts:")
        for i, prompt in enumerate(example_prompts[1:], 2):
            print(f"   {i}. \"{prompt}\"")
        
    except Exception as e:
        print(f"❌ Error generating art: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 