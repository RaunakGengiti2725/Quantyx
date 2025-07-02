#!/usr/bin/env python3
"""
Quantum 3D Art Generation Demo

This script demonstrates the capabilities of the quantum-conditioned 3D art generator
with various example prompts showing different semantic mappings to geometric parameters.

Usage:
    python demo_quantum_3d_art.py
    python demo_quantum_3d_art.py --interactive
    python demo_quantum_3d_art.py --prompt "Your custom prompt here"
"""

import os
import time
import argparse
from pathlib import Path
from typing import List, Dict

# Import the quantum 3D art generator
from quantum_3d_art_generator import (
    Quantum3DArtEngine,
    SemanticParameterMapper,
    QuantumGeometryParameters
)


def demo_example_prompts():
    """Run through a series of example prompts to showcase different capabilities."""
    
    # Example prompts demonstrating different semantic mappings
    example_prompts = [
        {
            "prompt": "A crystalline quantum structure with hexagonal symmetry, glowing with intense coherent energy",
            "description": "High coherence, hexagonal symmetry, intense energy with crystalline structure"
        },
        {
            "prompt": "Chaotic turbulent manifold with swirling void regions and explosive energy bursts",
            "description": "Low coherence, high intensity, chaotic topology with void aesthetics"
        },
        {
            "prompt": "Gentle toroidal flow with soft ethereal transitions and minimal perturbations",
            "description": "Toroidal topology, low intensity, smooth transitions"
        },
        {
            "prompt": "Complex intertwined multi-dimensional structure with dramatic phase shifts",
            "description": "Multi-torus topology, high complexity, strong phase shifts"
        },
        {
            "prompt": "Spherical quantum bubble with pulsing symmetric energy wells and luminous boundaries",
            "description": "Spherical topology, symmetric energy landscape, high glow"
        },
        {
            "prompt": "Twisted Klein bottle geometry with fractal interference patterns",
            "description": "Klein bottle topology, fractal interference, quantum geometry"
        },
        {
            "prompt": "Explosive volcanic quantum terrain with asymmetric energy eruptions",
            "description": "High intensity, asymmetric features, volcanic energy landscape"
        },
        {
            "prompt": "Serene minimal void space with subtle geometric harmonics",
            "description": "Minimal intensity, high void contrast, simple geometry"
        }
    ]
    
    print("🎨 Quantum 3D Art Generation Demo")
    print("=" * 60)
    print()
    
    # Initialize the art engine
    engine = Quantum3DArtEngine(output_dir="demo_quantum_art")
    
    print(f"📁 Output directory: {engine.output_dir}")
    print(f"🔬 Using semantic parameter mapping")
    print()
    
    for i, example in enumerate(example_prompts, 1):
        print(f"🌟 Example {i}: {example['description']}")
        print(f"📝 Prompt: \"{example['prompt']}\"")
        print("⚙️  Processing...")
        
        start_time = time.time()
        
        try:
            # Generate the artwork
            files = engine.generate_from_prompt(
                example['prompt'], 
                export_formats=["png", "glb"]  # Skip GLB if trimesh not available
            )
            
            elapsed = time.time() - start_time
            
            print(f"✅ Generated in {elapsed:.2f}s")
            print("📋 Output files:")
            for format_type, filepath in files.items():
                print(f"   {format_type}: {Path(filepath).name}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
        print()
    
    print("🎉 Demo complete! Check the output directory for generated artworks.")


def analyze_semantic_mapping():
    """Demonstrate how different words map to geometric parameters."""
    
    print("🔍 Semantic Parameter Mapping Analysis")
    print("=" * 50)
    print()
    
    mapper = SemanticParameterMapper()
    
    # Test prompts for parameter analysis
    test_prompts = [
        "intense crystalline structure",
        "chaotic turbulent manifold", 
        "gentle ethereal flow",
        "explosive dramatic eruption",
        "minimal subtle harmony",
        "complex intertwined topology",
        "spherical symmetric bubble",
        "toroidal twisted geometry"
    ]
    
    print("Prompt → Parameter Mapping:")
    print("-" * 30)
    
    for prompt in test_prompts:
        params = mapper.parse_prompt(prompt)
        
        print(f"\nPrompt: \"{prompt}\"")
        print(f"  Curvature Intensity: {params.curvature_intensity:.2f}")
        print(f"  Field Coherence:     {params.field_coherence:.2f}")
        print(f"  Nodal Symmetry:      {params.nodal_symmetry}")
        print(f"  Topology:            {params.connectivity} (genus {params.genus})")
        print(f"  Resolution:          {params.resolution}")
        print(f"  Mesh Quality:        {params.mesh_quality}")
    
    print("\n📊 Parameter Ranges:")
    print("  Curvature Intensity: 0.1 - 5.0 (geometric deformation strength)")
    print("  Field Coherence:     0.0 - 1.0 (spatial correlation)")
    print("  Nodal Symmetry:      3 - 12 (rotational symmetry order)")
    print("  Gradient Entropy:    0.0 - 1.0 (field disorder)")


def interactive_mode():
    """Interactive mode for testing custom prompts."""
    
    print("🎯 Interactive Quantum 3D Art Generator")
    print("=" * 45)
    print()
    print("Enter semantic prompts to generate quantum 3D art.")
    print("Type 'quit' or 'exit' to stop, 'help' for examples.")
    print()
    
    engine = Quantum3DArtEngine(output_dir="interactive_quantum_art")
    
    help_text = """
💡 Example prompts to try:
  - "intense crystalline hexagonal quantum structure"
  - "chaotic swirling void with explosive energy"
  - "gentle toroidal flow with ethereal glow"
  - "complex multi-dimensional twisted geometry"
  - "minimal spherical harmony with subtle patterns"
  - "dramatic volcanic quantum terrain"
  - "serene geometric void space"
  - "fractal interference with phase transitions"
    """
    
    while True:
        try:
            prompt = input("🎨 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif prompt.lower() in ['help', 'h']:
                print(help_text)
                continue
            elif not prompt:
                print("Please enter a prompt or 'help' for examples.")
                continue
            
            print(f"⚙️  Processing: \"{prompt}\"...")
            
            start_time = time.time()
            files = engine.generate_from_prompt(prompt, export_formats=["png"])
            elapsed = time.time() - start_time
            
            print(f"✅ Generated in {elapsed:.2f}s")
            print("📋 Files created:")
            for format_type, filepath in files.items():
                print(f"   {format_type}: {Path(filepath).name}")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


def create_parameter_showcase():
    """Create artworks showcasing different parameter ranges."""
    
    print("📊 Parameter Range Showcase")
    print("=" * 30)
    print()
    
    engine = Quantum3DArtEngine(output_dir="parameter_showcase")
    
    # Test different parameter combinations
    showcase_configs = [
        ("Low Intensity", "gentle subtle minimal soft flow"),
        ("High Intensity", "intense explosive dramatic violent power"),
        ("High Coherence", "crystalline organized structured geometric ordered"),
        ("Low Coherence", "chaotic random turbulent disordered fractal"),
        ("Simple Topology", "spherical simple basic minimal"),
        ("Complex Topology", "complex intertwined multi-dimensional twisted"),
        ("High Symmetry", "hexagonal octagonal symmetric geometric regular"),
        ("Low Symmetry", "asymmetric irregular complex elaborate")
    ]
    
    for name, prompt in showcase_configs:
        print(f"🎯 {name}: \"{prompt}\"")
        
        try:
            files = engine.generate_from_prompt(prompt, export_formats=["png"])
            print(f"   ✅ Generated: {Path(files['png']).name}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("🎉 Parameter showcase complete!")


def benchmark_performance():
    """Benchmark the generation performance with different settings."""
    
    print("⚡ Performance Benchmark")
    print("=" * 25)
    print()
    
    test_cases = [
        ("Low Resolution", "simple minimal spherical structure", 64),
        ("Medium Resolution", "moderate crystalline geometric pattern", 128),
        ("High Resolution", "complex detailed intricate quantum manifold", 256)
    ]
    
    for name, prompt, expected_res in test_cases:
        print(f"🏁 {name} ({expected_res}x{expected_res}):")
        print(f"   Prompt: \"{prompt}\"")
        
        start_time = time.time()
        
        try:
            engine = Quantum3DArtEngine(output_dir="benchmark_results")
            files = engine.generate_from_prompt(prompt, export_formats=["png"])
            
            elapsed = time.time() - start_time
            print(f"   ⏱️  Generation time: {elapsed:.2f}s")
            print(f"   📄 Output: {Path(files['png']).name}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()


def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Quantum 3D Art Generation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_quantum_3d_art.py                    # Run all demos
  python demo_quantum_3d_art.py --interactive      # Interactive mode
  python demo_quantum_3d_art.py --prompt "..."     # Single prompt
  python demo_quantum_3d_art.py --analysis         # Parameter analysis
  python demo_quantum_3d_art.py --showcase         # Parameter showcase
  python demo_quantum_3d_art.py --benchmark        # Performance test
        """
    )
    
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Generate art from a single prompt"
    )
    
    parser.add_argument(
        "--analysis", "-a",
        action="store_true", 
        help="Show semantic parameter mapping analysis"
    )
    
    parser.add_argument(
        "--showcase", "-s",
        action="store_true",
        help="Create parameter range showcase"
    )
    
    parser.add_argument(
        "--benchmark", "-b", 
        action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--examples", "-e",
        action="store_true",
        help="Run example prompts demo"
    )
    
    parser.add_argument(
        "--formats", 
        nargs="+",
        default=["png"],
        choices=["png", "glb", "obj"],
        help="Output formats (default: png)"
    )
    
    args = parser.parse_args()
    
    # If no specific mode specified, run examples
    if not any([args.interactive, args.prompt, args.analysis, 
                args.showcase, args.benchmark, args.examples]):
        args.examples = True
    
    try:
        if args.analysis:
            analyze_semantic_mapping()
            
        elif args.interactive:
            interactive_mode()
            
        elif args.prompt:
            print(f"🎨 Generating art from prompt: \"{args.prompt}\"")
            print()
            
            engine = Quantum3DArtEngine(output_dir="custom_prompt_art")
            files = engine.generate_from_prompt(args.prompt, args.formats)
            
            print("✅ Generation complete!")
            print("📋 Output files:")
            for format_type, filepath in files.items():
                print(f"   {format_type}: {filepath}")
                
        elif args.showcase:
            create_parameter_showcase()
            
        elif args.benchmark:
            benchmark_performance()
            
        elif args.examples:
            demo_example_prompts()
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 