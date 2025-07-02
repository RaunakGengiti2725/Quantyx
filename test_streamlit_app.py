#!/usr/bin/env python3
"""
Test script for Streamlit Quantum Art App
Generates sample artwork to demonstrate functionality.
"""

from streamlit_quantum_app import QuantumArtGenerator
import time

def test_quantum_art_generation():
    """Test the quantum art generation functionality."""
    
    print("🎨 Testing Quantum Geometry Art Generator")
    print("=" * 45)
    print()
    
    # Initialize generator
    generator = QuantumArtGenerator()
    
    # Test each style with different parameters
    test_cases = [
        {
            "style": "Quantum Bloom",
            "energy_intensity": 7.0,
            "symmetry_level": 60,
            "deformation_amount": 0.4,
            "color_variation": 0.7,
            "description": "Organic flower-like patterns with interference rings"
        },
        {
            "style": "Singularity Core", 
            "energy_intensity": 8.5,
            "symmetry_level": 30,
            "deformation_amount": 0.8,
            "color_variation": 0.3,
            "description": "Black hole with dramatic accretion disk"
        },
        {
            "style": "Crystal Spire",
            "energy_intensity": 6.0,
            "symmetry_level": 80,
            "deformation_amount": 0.1,
            "color_variation": 0.4,
            "description": "Geometric crystalline structures"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🌟 Test {i}: {test_case['style']}")
        print(f"📝 {test_case['description']}")
        print(f"⚙️  Parameters:")
        print(f"   Energy: {test_case['energy_intensity']}")
        print(f"   Symmetry: {test_case['symmetry_level']}")
        print(f"   Deformation: {test_case['deformation_amount']}")
        print(f"   Color Variation: {test_case['color_variation']}")
        
        start_time = time.time()
        
        try:
            # Generate artwork
            sample_image = generator.generate_quantum_image(
                style=test_case['style'],
                energy_intensity=test_case['energy_intensity'],
                symmetry_level=test_case['symmetry_level'], 
                deformation_amount=test_case['deformation_amount'],
                color_variation=test_case['color_variation'],
                resolution=256  # Smaller for testing
            )
            
            # Save sample
            timestamp = int(time.time())
            safe_style = test_case['style'].lower().replace(' ', '_')
            filename = f'sample_{safe_style}_{timestamp}.png'
            sample_image.save(filename)
            
            elapsed = time.time() - start_time
            print(f"✅ Generated in {elapsed:.2f}s: {filename}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
        print()
    
    print("🎉 Streamlit App Test Complete!")
    print()
    print("🚀 To run the full interactive app:")
    print("   streamlit run \"🖼️ Image Generation.py\"")
    print()
    print("🌐 The app will open at: http://localhost:8501")
    print()
    print("✨ App Features:")
    print("   • Modern dark UI with animations")
    print("   • 5 quantum art styles")
    print("   • Real-time parameter controls")
    print("   • High-resolution PNG downloads")
    print("   • Physics-based field generation")


if __name__ == "__main__":
    test_quantum_art_generation() 