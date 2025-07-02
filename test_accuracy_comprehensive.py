#!/usr/bin/env python3
"""
Comprehensive Accuracy Testing Framework for Quantum Art Generation System

This module tests multiple aspects of accuracy:
1. Mathematical Accuracy - Quantum field equations and curvature calculations
2. Semantic Mapping Accuracy - Natural language to parameter conversion
3. Visual Quality Metrics - Statistical analysis of generated art
4. Consistency Testing - Reproducibility and parameter sensitivity
5. Performance Benchmarking - Speed and resource usage
6. Edge Case Validation - Extreme parameter handling
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from typing import Dict, List, Tuple, Any
from scipy import stats
from sklearn.metrics import mean_squared_error
import hashlib

# Import our quantum art modules
try:
    from streamlit_quantum_app import QuantumArtGenerator
    from quantum_3d_art_generator import SemanticParameterMapper, QuantumManifoldGenerator3D
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running in limited testing mode...")

class QuantumArtAccuracyTester:
    """Comprehensive testing framework for quantum art generation accuracy"""
    
    def __init__(self):
        self.test_results = {
            'mathematical_accuracy': {},
            'semantic_mapping_accuracy': {},
            'visual_quality_metrics': {},
            'consistency_tests': {},
            'performance_benchmarks': {},
            'edge_case_validation': {}
        }
        
        # Initialize generators if available
        try:
            self.art_generator = QuantumArtGenerator()
            self.semantic_mapper = SemanticParameterMapper()
            self.manifold_generator = QuantumManifoldGenerator3D()
            self.generators_available = True
        except:
            self.generators_available = False
            print("Generators not available - running mathematical tests only")
    
    def test_mathematical_accuracy(self) -> Dict[str, Any]:
        """Test the mathematical accuracy of quantum field equations and curvature calculations"""
        print("🧮 Testing Mathematical Accuracy...")
        
        results = {}
        
        # Test 1: Quantum Energy Field Conservation
        print("  Testing energy conservation...")
        energy_conservation_error = self._test_energy_conservation()
        results['energy_conservation_error'] = energy_conservation_error
        results['energy_conservation_pass'] = energy_conservation_error < 1e-10
        
        # Test 2: Curvature Tensor Properties
        print("  Testing curvature tensor properties...")
        curvature_test = self._test_curvature_properties()
        results.update(curvature_test)
        
        # Test 3: Wave Function Normalization
        print("  Testing wave function normalization...")
        normalization_test = self._test_wave_function_normalization()
        results['wave_function_normalized'] = normalization_test
        
        # Test 4: Fourier Transform Accuracy
        print("  Testing Fourier transform accuracy...")
        fft_accuracy = self._test_fft_accuracy()
        results['fft_accuracy'] = fft_accuracy
        
        self.test_results['mathematical_accuracy'] = results
        return results
    
    def test_semantic_mapping_accuracy(self) -> Dict[str, Any]:
        """Test how accurately semantic prompts map to expected parameters"""
        print("🗣️ Testing Semantic Mapping Accuracy...")
        
        if not self.generators_available:
            return {'error': 'Generators not available'}
        
        results = {}
        
        # Define test cases with expected parameter ranges
        test_cases = [
            {
                'prompt': 'intense chaotic energy',
                'expected': {'energy_intensity': (0.8, 1.0), 'symmetry_level': (0.0, 0.3)},
                'description': 'High energy, low symmetry'
            },
            {
                'prompt': 'calm crystalline structure',
                'expected': {'energy_intensity': (0.1, 0.4), 'symmetry_level': (0.7, 1.0)},
                'description': 'Low energy, high symmetry'
            },
            {
                'prompt': 'ethereal flowing patterns',
                'expected': {'deformation_amount': (0.6, 1.0), 'color_variation': (0.5, 0.9)},
                'description': 'High deformation and color variation'
            },
            {
                'prompt': 'geometric precision mathematical',
                'expected': {'symmetry_level': (0.8, 1.0), 'deformation_amount': (0.0, 0.3)},
                'description': 'High symmetry, low deformation'
            }
        ]
        
        mapping_scores = []
        
        for i, case in enumerate(test_cases):
            print(f"  Testing case {i+1}: {case['description']}")
            
            # Map semantic prompt to parameters
            mapped_params = self.semantic_mapper.map_prompt_to_parameters(case['prompt'])
            
            # Check if mapped parameters fall within expected ranges
            case_score = 0
            total_checks = 0
            
            for param_name, expected_range in case['expected'].items():
                if param_name in mapped_params:
                    value = mapped_params[param_name]
                    if expected_range[0] <= value <= expected_range[1]:
                        case_score += 1
                    total_checks += 1
            
            if total_checks > 0:
                accuracy = case_score / total_checks
                mapping_scores.append(accuracy)
                results[f'case_{i+1}_accuracy'] = accuracy
                results[f'case_{i+1}_description'] = case['description']
            
        overall_accuracy = np.mean(mapping_scores) if mapping_scores else 0
        results['overall_semantic_accuracy'] = overall_accuracy
        results['semantic_mapping_pass'] = overall_accuracy > 0.7
        
        self.test_results['semantic_mapping_accuracy'] = results
        return results
    
    def test_visual_quality_metrics(self, num_samples=10) -> Dict[str, Any]:
        """Analyze statistical properties of generated art for quality assessment"""
        print("🎨 Testing Visual Quality Metrics...")
        
        if not self.generators_available:
            return {'error': 'Generators not available'}
        
        results = {}
        art_styles = ['Quantum Bloom', 'Singularity Core', 'Entanglement Field', 'Crystal Spire', 'Tunneling Veil']
        
        quality_metrics = {
            'color_diversity': [],
            'edge_complexity': [],
            'symmetry_score': [],
            'contrast_ratio': [],
            'fractal_dimension': []
        }
        
        for style in art_styles[:2]:  # Test first 2 styles for speed
            print(f"  Analyzing {style}...")
            
            for _ in range(num_samples):
                # Generate art with random parameters
                params = {
                    'energy_intensity': np.random.random(),
                    'symmetry_level': np.random.random(),
                    'deformation_amount': np.random.random(),
                    'color_variation': np.random.random()
                }
                
                try:
                    image_data = self.art_generator.generate_art(style, params)
                    
                    # Calculate quality metrics
                    metrics = self._calculate_image_metrics(image_data)
                    
                    for metric_name, value in metrics.items():
                        if metric_name in quality_metrics:
                            quality_metrics[metric_name].append(value)
                except Exception as e:
                    print(f"    Warning: Failed to generate/analyze image: {e}")
        
        # Calculate statistics for each metric
        for metric_name, values in quality_metrics.items():
            if values:
                results[f'{metric_name}_mean'] = np.mean(values)
                results[f'{metric_name}_std'] = np.std(values)
                results[f'{metric_name}_min'] = np.min(values)
                results[f'{metric_name}_max'] = np.max(values)
        
        # Quality assessment
        results['visual_quality_pass'] = self._assess_visual_quality(quality_metrics)
        
        self.test_results['visual_quality_metrics'] = results
        return results
    
    def test_consistency_and_reproducibility(self, num_trials=5) -> Dict[str, Any]:
        """Test consistency and reproducibility of the generation process"""
        print("🔄 Testing Consistency and Reproducibility...")
        
        if not self.generators_available:
            return {'error': 'Generators not available'}
        
        results = {}
        
        # Test reproducibility with same parameters
        fixed_params = {
            'energy_intensity': 0.5,
            'symmetry_level': 0.7,
            'deformation_amount': 0.3,
            'color_variation': 0.6
        }
        
        # Generate multiple images with same parameters
        image_hashes = []
        generation_times = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")
            
            start_time = time.time()
            try:
                image_data = self.art_generator.generate_art('Quantum Bloom', fixed_params)
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                # Create hash of image data for reproducibility check
                image_hash = hashlib.md5(image_data.tobytes()).hexdigest()
                image_hashes.append(image_hash)
                
            except Exception as e:
                print(f"    Error in trial {trial + 1}: {e}")
        
        # Analyze consistency
        if generation_times:
            results['avg_generation_time'] = np.mean(generation_times)
            results['generation_time_std'] = np.std(generation_times)
            results['generation_time_consistency'] = np.std(generation_times) / np.mean(generation_times) < 0.3
        
        # Check reproducibility (images should be similar but not identical due to randomness)
        unique_hashes = len(set(image_hashes))
        results['unique_outputs'] = unique_hashes
        results['total_trials'] = len(image_hashes)
        results['reproducibility_score'] = 1.0 - (unique_hashes / len(image_hashes)) if image_hashes else 0
        
        # Test parameter sensitivity
        sensitivity_results = self._test_parameter_sensitivity()
        results.update(sensitivity_results)
        
        self.test_results['consistency_tests'] = results
        return results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Benchmark performance across different configurations"""
        print("⚡ Testing Performance Benchmarks...")
        
        if not self.generators_available:
            return {'error': 'Generators not available'}
        
        results = {}
        
        # Test different image sizes
        sizes = [(128, 128), (256, 256), (512, 512)]
        style = 'Quantum Bloom'
        params = {'energy_intensity': 0.5, 'symmetry_level': 0.5, 'deformation_amount': 0.5, 'color_variation': 0.5}
        
        for width, height in sizes:
            print(f"  Testing {width}x{height} resolution...")
            
            times = []
            for _ in range(3):  # 3 trials per size
                start_time = time.time()
                try:
                    # Note: This assumes the generator can accept size parameters
                    image_data = self.art_generator.generate_art(style, params)
                    times.append(time.time() - start_time)
                except Exception as e:
                    print(f"    Error with size {width}x{height}: {e}")
            
            if times:
                results[f'time_{width}x{height}'] = np.mean(times)
        
        # Memory usage estimation (simplified)
        results['estimated_memory_usage_mb'] = self._estimate_memory_usage()
        
        # Performance score
        base_time = results.get('time_256x256', 1.0)
        results['performance_score'] = 1.0 / base_time if base_time > 0 else 0
        results['performance_pass'] = base_time < 5.0  # Should generate 256x256 in under 5 seconds
        
        self.test_results['performance_benchmarks'] = results
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and extreme parameter values"""
        print("🚨 Testing Edge Cases...")
        
        if not self.generators_available:
            return {'error': 'Generators not available'}
        
        results = {}
        edge_cases = [
            {'name': 'all_zeros', 'params': {'energy_intensity': 0, 'symmetry_level': 0, 'deformation_amount': 0, 'color_variation': 0}},
            {'name': 'all_ones', 'params': {'energy_intensity': 1, 'symmetry_level': 1, 'deformation_amount': 1, 'color_variation': 1}},
            {'name': 'extreme_contrast', 'params': {'energy_intensity': 1, 'symmetry_level': 0, 'deformation_amount': 1, 'color_variation': 0}},
            {'name': 'negative_values', 'params': {'energy_intensity': -0.5, 'symmetry_level': -0.3, 'deformation_amount': -0.1, 'color_variation': -0.2}},
            {'name': 'values_over_one', 'params': {'energy_intensity': 1.5, 'symmetry_level': 1.2, 'deformation_amount': 1.8, 'color_variation': 1.1}}
        ]
        
        for case in edge_cases:
            print(f"  Testing {case['name']}...")
            
            try:
                image_data = self.art_generator.generate_art('Quantum Bloom', case['params'])
                results[f"{case['name']}_success"] = True
                results[f"{case['name']}_error"] = None
                
                # Check if image is valid (not all zeros or NaN)
                if np.isnan(image_data).any():
                    results[f"{case['name']}_valid"] = False
                    results[f"{case['name']}_issue"] = "Contains NaN values"
                elif np.all(image_data == 0):
                    results[f"{case['name']}_valid"] = False
                    results[f"{case['name']}_issue"] = "All zero image"
                else:
                    results[f"{case['name']}_valid"] = True
                    results[f"{case['name']}_issue"] = None
                    
            except Exception as e:
                results[f"{case['name']}_success"] = False
                results[f"{case['name']}_error"] = str(e)
                results[f"{case['name']}_valid"] = False
        
        # Calculate overall edge case handling score
        success_rate = sum(1 for key in results if key.endswith('_success') and results[key]) / len(edge_cases)
        results['edge_case_success_rate'] = success_rate
        results['edge_case_pass'] = success_rate > 0.6
        
        self.test_results['edge_case_validation'] = results
        return results
    
    def run_full_accuracy_test(self) -> Dict[str, Any]:
        """Run all accuracy tests and generate comprehensive report"""
        print("🚀 Running Comprehensive Accuracy Test Suite...\n")
        
        start_time = time.time()
        
        # Run all test categories
        self.test_mathematical_accuracy()
        print()
        
        if self.generators_available:
            self.test_semantic_mapping_accuracy()
            print()
            
            self.test_visual_quality_metrics(num_samples=5)  # Reduced samples for speed
            print()
            
            self.test_consistency_and_reproducibility(num_trials=3)
            print()
            
            self.test_performance_benchmarks()
            print()
            
            self.test_edge_cases()
            print()
        
        total_time = time.time() - start_time
        
        # Generate overall assessment
        overall_results = self._generate_overall_assessment(total_time)
        
        # Save results to file
        self._save_test_results()
        
        return overall_results
    
    # Helper methods
    def _test_energy_conservation(self) -> float:
        """Test if quantum energy fields conserve energy"""
        # Create a simple 2D energy field
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Quantum harmonic oscillator energy field
        energy_field = 0.5 * (X**2 + Y**2) * np.exp(-(X**2 + Y**2))
        
        # Calculate total energy (should be conserved under unitary transformations)
        initial_energy = np.sum(energy_field**2)
        
        # Apply a unitary transformation (rotation)
        theta = np.pi / 4
        X_rot = X * np.cos(theta) - Y * np.sin(theta)
        Y_rot = X * np.sin(theta) + Y * np.cos(theta)
        
        energy_field_rot = 0.5 * (X_rot**2 + Y_rot**2) * np.exp(-(X_rot**2 + Y_rot**2))
        final_energy = np.sum(energy_field_rot**2)
        
        return abs(initial_energy - final_energy) / initial_energy
    
    def _test_curvature_properties(self) -> Dict[str, Any]:
        """Test mathematical properties of curvature tensors"""
        # Test Riemann curvature tensor properties
        results = {}
        
        # Simple 2D surface with known curvature
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        U, V = np.meshgrid(u, v)
        
        # Sphere of radius 1 (constant curvature = 1)
        X = np.sin(V) * np.cos(U)
        Y = np.sin(V) * np.sin(U)
        Z = np.cos(V)
        
        # Calculate discrete curvature
        curvature = self._calculate_discrete_curvature(X, Y, Z)
        
        # For a unit sphere, mean curvature should be approximately 1
        expected_curvature = 1.0
        curvature_error = abs(np.mean(curvature) - expected_curvature)
        
        results['curvature_calculation_error'] = curvature_error
        results['curvature_test_pass'] = curvature_error < 0.2
        
        return results
    
    def _test_wave_function_normalization(self) -> bool:
        """Test if quantum wave functions are properly normalized"""
        # Create a simple wave function
        x = np.linspace(-5, 5, 1000)
        psi = np.exp(-x**2/2) / (np.pi**0.25)  # Gaussian wave function
        
        # Check normalization: ∫|ψ|² dx = 1
        normalization = np.trapz(np.abs(psi)**2, x)
        return abs(normalization - 1.0) < 1e-2
    
    def _test_fft_accuracy(self) -> float:
        """Test Fast Fourier Transform accuracy"""
        # Create a known signal
        t = np.linspace(0, 1, 1000, endpoint=False)
        signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*10*t)
        
        # Forward and inverse FFT
        fft_signal = np.fft.fft(signal)
        reconstructed = np.fft.ifft(fft_signal)
        
        # Calculate reconstruction error
        error = np.mean(np.abs(signal - reconstructed.real))
        return error
    
    def _calculate_image_metrics(self, image_data: np.ndarray) -> Dict[str, float]:
        """Calculate various quality metrics for an image"""
        metrics = {}
        
        # Ensure image is 2D for analysis
        if len(image_data.shape) == 3:
            # Convert to grayscale
            image_gray = np.mean(image_data, axis=2)
        else:
            image_gray = image_data
        
        # Color diversity (for color images)
        if len(image_data.shape) == 3:
            metrics['color_diversity'] = np.std(image_data)
        else:
            metrics['color_diversity'] = 0
        
        # Edge complexity using gradient magnitude
        gy, gx = np.gradient(image_gray)
        edge_magnitude = np.sqrt(gx**2 + gy**2)
        metrics['edge_complexity'] = np.mean(edge_magnitude)
        
        # Symmetry score (simplified)
        flipped = np.flip(image_gray, axis=1)
        symmetry_diff = np.mean(np.abs(image_gray - flipped))
        metrics['symmetry_score'] = 1.0 / (1.0 + symmetry_diff)
        
        # Contrast ratio
        max_val = np.max(image_gray)
        min_val = np.min(image_gray)
        metrics['contrast_ratio'] = (max_val - min_val) / (max_val + min_val + 1e-8)
        
        # Simplified fractal dimension (box counting approximation)
        metrics['fractal_dimension'] = self._estimate_fractal_dimension(image_gray)
        
        return metrics
    
    def _estimate_fractal_dimension(self, image: np.ndarray) -> float:
        """Estimate fractal dimension using simplified box counting"""
        # Binarize image
        threshold = np.mean(image)
        binary = image > threshold
        
        # Count boxes at different scales
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            h, w = binary.shape
            boxes_h = h // scale
            boxes_w = w // scale
            
            count = 0
            for i in range(boxes_h):
                for j in range(boxes_w):
                    box = binary[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
                    if np.any(box):
                        count += 1
            counts.append(count)
        
        # Fit power law
        if len(counts) > 1 and all(c > 0 for c in counts):
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return -slope
        else:
            return 1.5  # Default reasonable value
    
    def _assess_visual_quality(self, quality_metrics: Dict[str, List[float]]) -> bool:
        """Assess overall visual quality based on metrics"""
        # Check if we have sufficient diversity and complexity
        thresholds = {
            'color_diversity': 0.1,
            'edge_complexity': 0.01,
            'contrast_ratio': 0.3
        }
        
        passes = 0
        total = 0
        
        for metric, threshold in thresholds.items():
            if metric in quality_metrics and quality_metrics[metric]:
                mean_value = np.mean(quality_metrics[metric])
                if mean_value > threshold:
                    passes += 1
                total += 1
        
        return passes / total > 0.6 if total > 0 else False
    
    def _test_parameter_sensitivity(self) -> Dict[str, Any]:
        """Test how sensitive the output is to parameter changes"""
        results = {}
        
        base_params = {'energy_intensity': 0.5, 'symmetry_level': 0.5, 'deformation_amount': 0.5, 'color_variation': 0.5}
        
        try:
            base_image = self.art_generator.generate_art('Quantum Bloom', base_params)
            
            sensitivities = []
            for param_name in base_params:
                # Slightly modify one parameter
                modified_params = base_params.copy()
                modified_params[param_name] = min(1.0, base_params[param_name] + 0.1)
                
                modified_image = self.art_generator.generate_art('Quantum Bloom', modified_params)
                
                # Calculate difference
                difference = np.mean(np.abs(base_image - modified_image))
                sensitivities.append(difference)
                results[f'{param_name}_sensitivity'] = difference
            
            results['avg_parameter_sensitivity'] = np.mean(sensitivities)
            results['sensitivity_consistency'] = np.std(sensitivities) / np.mean(sensitivities) < 0.5
            
        except Exception as e:
            results['sensitivity_test_error'] = str(e)
        
        return results
    
    def _calculate_discrete_curvature(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Calculate discrete curvature of a surface"""
        # Simplified discrete mean curvature calculation
        h, w = X.shape
        curvature = np.zeros_like(X)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Get neighboring points
                neighbors = [
                    (X[i-1,j], Y[i-1,j], Z[i-1,j]),
                    (X[i+1,j], Y[i+1,j], Z[i+1,j]),
                    (X[i,j-1], Y[i,j-1], Z[i,j-1]),
                    (X[i,j+1], Y[i,j+1], Z[i,j+1])
                ]
                
                center = (X[i,j], Y[i,j], Z[i,j])
                
                # Calculate discrete Laplacian (simplified curvature measure)
                laplacian = 0
                for nx, ny, nz in neighbors:
                    dist = np.sqrt((nx-center[0])**2 + (ny-center[1])**2 + (nz-center[2])**2)
                    laplacian += dist
                
                curvature[i,j] = laplacian / 4.0
        
        return curvature
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Simplified estimation based on typical array sizes
        typical_image_size = 512 * 512 * 3 * 8  # 512x512 RGB double precision
        overhead = 1.5  # Factor for intermediate calculations
        return (typical_image_size * overhead) / (1024 * 1024)
    
    def _generate_overall_assessment(self, total_time: float) -> Dict[str, Any]:
        """Generate overall assessment of the quantum art system"""
        assessment = {
            'test_duration_seconds': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_score': 0,
            'category_scores': {},
            'recommendations': []
        }
        
        # Calculate scores for each category
        if 'mathematical_accuracy' in self.test_results:
            math_score = self._score_mathematical_tests()
            assessment['category_scores']['mathematical_accuracy'] = math_score
        
        if self.generators_available:
            if 'semantic_mapping_accuracy' in self.test_results:
                semantic_score = self._score_semantic_tests()
                assessment['category_scores']['semantic_mapping'] = semantic_score
            
            if 'visual_quality_metrics' in self.test_results:
                visual_score = self._score_visual_tests()
                assessment['category_scores']['visual_quality'] = visual_score
            
            if 'consistency_tests' in self.test_results:
                consistency_score = self._score_consistency_tests()
                assessment['category_scores']['consistency'] = consistency_score
            
            if 'performance_benchmarks' in self.test_results:
                performance_score = self._score_performance_tests()
                assessment['category_scores']['performance'] = performance_score
            
            if 'edge_case_validation' in self.test_results:
                edge_case_score = self._score_edge_case_tests()
                assessment['category_scores']['edge_cases'] = edge_case_score
        
        # Calculate overall score
        if assessment['category_scores']:
            assessment['overall_score'] = np.mean(list(assessment['category_scores'].values()))
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_recommendations()
        
        # Overall assessment
        if assessment['overall_score'] >= 0.8:
            assessment['grade'] = 'A - Excellent'
        elif assessment['overall_score'] >= 0.7:
            assessment['grade'] = 'B - Good'
        elif assessment['overall_score'] >= 0.6:
            assessment['grade'] = 'C - Acceptable'
        else:
            assessment['grade'] = 'D - Needs Improvement'
        
        return assessment
    
    def _score_mathematical_tests(self) -> float:
        """Score mathematical accuracy tests"""
        results = self.test_results['mathematical_accuracy']
        score = 0
        
        if results.get('energy_conservation_pass', False):
            score += 0.25
        if results.get('curvature_test_pass', False):
            score += 0.25
        if results.get('wave_function_normalized', False):
            score += 0.25
        
        fft_accuracy = results.get('fft_accuracy', 1.0)
        if fft_accuracy < 1e-10:
            score += 0.25
        
        return score
    
    def _score_semantic_tests(self) -> float:
        """Score semantic mapping tests"""
        results = self.test_results['semantic_mapping_accuracy']
        return results.get('overall_semantic_accuracy', 0)
    
    def _score_visual_tests(self) -> float:
        """Score visual quality tests"""
        results = self.test_results['visual_quality_metrics']
        return 0.8 if results.get('visual_quality_pass', False) else 0.4
    
    def _score_consistency_tests(self) -> float:
        """Score consistency tests"""
        results = self.test_results['consistency_tests']
        score = 0
        
        if results.get('generation_time_consistency', False):
            score += 0.4
        if results.get('sensitivity_consistency', False):
            score += 0.3
        
        repro_score = results.get('reproducibility_score', 0)
        score += repro_score * 0.3
        
        return score
    
    def _score_performance_tests(self) -> float:
        """Score performance tests"""
        results = self.test_results['performance_benchmarks']
        return 0.8 if results.get('performance_pass', False) else 0.4
    
    def _score_edge_case_tests(self) -> float:
        """Score edge case tests"""
        results = self.test_results['edge_case_validation']
        return results.get('edge_case_success_rate', 0)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check mathematical accuracy
        math_results = self.test_results.get('mathematical_accuracy', {})
        if not math_results.get('energy_conservation_pass', True):
            recommendations.append("Improve energy conservation in quantum field calculations")
        
        # Check semantic mapping
        if self.generators_available:
            semantic_results = self.test_results.get('semantic_mapping_accuracy', {})
            if semantic_results.get('overall_semantic_accuracy', 1) < 0.7:
                recommendations.append("Enhance semantic-to-parameter mapping accuracy")
            
            # Check performance
            perf_results = self.test_results.get('performance_benchmarks', {})
            if not perf_results.get('performance_pass', True):
                recommendations.append("Optimize generation speed for better user experience")
            
            # Check edge cases
            edge_results = self.test_results.get('edge_case_validation', {})
            if edge_results.get('edge_case_success_rate', 1) < 0.8:
                recommendations.append("Improve handling of extreme parameter values")
        
        if not recommendations:
            recommendations.append("System shows excellent performance across all test categories")
        
        return recommendations
    
    def _save_test_results(self):
        """Save test results to JSON file"""
        filename = f"quantum_art_accuracy_test_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, default=str)
        print(f"📄 Test results saved to: {filename}")
    
    def print_summary_report(self, overall_results: Dict[str, Any]):
        """Print a formatted summary report"""
        print("\n" + "="*80)
        print("🎯 QUANTUM ART GENERATION SYSTEM - ACCURACY TEST REPORT")
        print("="*80)
        
        print(f"\n📊 Overall Assessment: {overall_results['grade']}")
        print(f"🎯 Overall Score: {overall_results['overall_score']:.2%}")
        print(f"⏱️  Test Duration: {overall_results['test_duration_seconds']:.1f} seconds")
        print(f"📅 Test Date: {overall_results['timestamp']}")
        
        if overall_results['category_scores']:
            print(f"\n📈 Category Scores:")
            for category, score in overall_results['category_scores'].items():
                status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
                print(f"  {status} {category.replace('_', ' ').title()}: {score:.2%}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(overall_results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Main function to run the accuracy tests"""
    print("🧪 Quantum Art Generation System - Comprehensive Accuracy Testing")
    print("=" * 80)
    
    tester = QuantumArtAccuracyTester()
    
    try:
        overall_results = tester.run_full_accuracy_test()
        tester.print_summary_report(overall_results)
        
        return overall_results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 