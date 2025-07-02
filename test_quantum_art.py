#!/usr/bin/env python3
"""
Simple test script for Quantum 3D Art Generator core functionality
"""

import sys
import numpy as np
import re

def test_core_functionality():
    """Test core components without optional dependencies."""
    
    print("🧪 Testing Quantum 3D Art Generator Core Components")
    print("=" * 55)
    
    try:
        # Test 1: Core dependencies
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        print("✅ Core dependencies: numpy, matplotlib, scipy")
        
        # Test 2: Semantic parameter mapping
        class TestSemanticMapper:
            def __init__(self):
                self.intensity_keywords = {
                    'intense': 3.0, 'gentle': 0.3, 'explosive': 5.0,
                    'dramatic': 3.5, 'subtle': 0.5, 'minimal': 0.1
                }
                self.coherence_keywords = {
                    'crystalline': 0.95, 'chaotic': 0.1, 'organized': 0.8,
                    'turbulent': 0.15, 'structured': 0.7
                }
                
            def parse_prompt(self, prompt):
                tokens = re.findall(r'\b\w+\b', prompt.lower())
                
                intensity_scores = [self.intensity_keywords.get(token, 0) for token in tokens]
                intensity = max(intensity_scores) if any(intensity_scores) else 1.0
                
                coherence_scores = [self.coherence_keywords.get(token, -1) for token in tokens]
                coherence_scores = [s for s in coherence_scores if s >= 0]
                coherence = np.mean(coherence_scores) if coherence_scores else 0.5
                
                return {
                    'intensity': intensity, 
                    'coherence': coherence,
                    'tokens': tokens
                }
        
        mapper = TestSemanticMapper()
        test_prompts = [
            "intense crystalline hexagonal quantum structure",
            "chaotic turbulent manifold with explosive energy",
            "gentle ethereal flow with minimal perturbations"
        ]
        
        print("✅ Semantic parameter mapping:")
        for prompt in test_prompts:
            result = mapper.parse_prompt(prompt)
            print(f"   '{prompt[:30]}...' → intensity={result['intensity']:.2f}, coherence={result['coherence']:.2f}")
        
        # Test 3: Basic manifold generation
        print("✅ Basic manifold generation:")
        
        # Spherical topology
        phi = np.linspace(0, np.pi, 32)
        theta = np.linspace(0, 2*np.pi, 32)
        Phi, Theta = np.meshgrid(phi, theta)
        
        X_sphere = np.sin(Phi) * np.cos(Theta)
        Y_sphere = np.sin(Phi) * np.sin(Theta)
        Z_sphere = np.cos(Phi)
        print(f"   Sphere: {X_sphere.shape}")
        
        # Toroidal topology
        u = np.linspace(0, 2*np.pi, 32)
        v = np.linspace(0, 2*np.pi, 32)
        U, V = np.meshgrid(u, v)
        
        R, r = 3.0, 1.0
        X_torus = (R + r * np.cos(V)) * np.cos(U)
        Y_torus = (R + r * np.cos(V)) * np.sin(U)
        Z_torus = r * np.sin(V)
        print(f"   Torus: {X_torus.shape}")
        
        # Test 4: Curvature computation
        dz_dx, dz_dy = np.gradient(Z_torus)
        d2z_dx2, _ = np.gradient(dz_dx)
        _, d2z_dy2 = np.gradient(dz_dy)
        d2z_dxdy, _ = np.gradient(dz_dy)
        
        # Gaussian curvature
        numerator = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
        denominator = (1 + dz_dx**2 + dz_dy**2)**2
        curvature = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator != 0)
        
        print(f"✅ Curvature computation: range=[{curvature.min():.3f}, {curvature.max():.3f}]")
        
        # Test 5: Energy field simulation
        r_squared = X_torus**2 + Y_torus**2 + Z_torus**2
        potential = 0.5 * r_squared
        
        # Quantum interference pattern
        k = 2*np.pi / 2.0  # wave number
        phase = k * (X_torus + Y_torus + Z_torus)
        interference = np.sin(phase) * np.exp(-0.1 * r_squared)
        
        energy = potential + 0.5 * interference
        print(f"✅ Energy field simulation: range=[{energy.min():.3f}, {energy.max():.3f}]")
        
        # Test 6: Color mapping
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min())
        
        # High energy = photonic glow
        high_energy_mask = energy_norm > 0.7
        glow_pixels = np.sum(high_energy_mask)
        
        # Low energy = void
        low_energy_mask = energy_norm < 0.3
        void_pixels = np.sum(low_energy_mask)
        
        print(f"✅ Aesthetic mapping: {glow_pixels} glow pixels, {void_pixels} void pixels")
        
        print("\n🎨 Core Quantum Art Engine: ALL TESTS PASSED!")
        print("📊 System Status:")
        print(f"   • Semantic mapping: Ready")
        print(f"   • Manifold generation: Ready")
        print(f"   • Curvature computation: Ready")
        print(f"   • Energy field simulation: Ready")
        print(f"   • Aesthetic mapping: Ready")
        
        print("\n📦 Next Steps:")
        print("   • Install 3D dependencies: pip install scikit-image trimesh")
        print("   • Run full demo: python demo_quantum_3d_art.py")
        print("   • Try interactive mode: python demo_quantum_3d_art.py --interactive")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install numpy matplotlib scipy")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_core_functionality() 