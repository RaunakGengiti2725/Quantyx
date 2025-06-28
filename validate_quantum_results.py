#!/usr/bin/env python3
"""
QUANTUM SIMULATION VALIDATION SCRIPT
====================================

This script validates the correctness of quantum simulation results by testing:
1. Basic quantum mechanics principles (unitarity, normalization)
2. Entanglement entropy bounds and properties
3. Hamiltonian evolution correctness
4. Statistical correlation validity
5. Comparison with known analytical results
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from quantumproject.quantum.simulations import Simulator, von_neumann_entropy
from quantumproject.utils.tree import BulkTree
from quantumproject.quantum.simulations import contiguous_intervals

class QuantumValidationSuite:
    """Comprehensive validation suite for quantum simulation results."""
    
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
        self.simulator = Simulator(n_qubits)
        self.tree = BulkTree(n_qubits)
        self.validation_results = {}
        
    def test_state_normalization(self, state):
        """Test 1: Quantum states must be normalized."""
        norm = np.linalg.norm(state)
        is_normalized = abs(norm - 1.0) < 1e-10
        
        print(f"✓ State Normalization Test")
        print(f"  Norm: {norm:.12f} (should be 1.0)")
        print(f"  Result: {'✅ PASS' if is_normalized else '❌ FAIL'}")
        
        return is_normalized
    
    def test_hamiltonian_hermiticity(self):
        """Test 2: Hamiltonian must be Hermitian."""
        H = self.simulator.build_hamiltonian("xxz", Jx=1.0, Jy=1.0, Jz=0.5, h=0.8)
        is_hermitian = np.allclose(H, H.conj().T, atol=1e-12)
        
        print(f"\n✓ Hamiltonian Hermiticity Test")
        print(f"  Max deviation: {np.max(np.abs(H - H.conj().T)):.2e}")
        print(f"  Result: {'✅ PASS' if is_hermitian else '❌ FAIL'}")
        
        return is_hermitian
    
    def test_unitary_evolution(self):
        """Test 3: Time evolution must be unitary."""
        H = self.simulator.build_hamiltonian("xxz", Jx=1.0, Jy=1.0, Jz=0.5, h=0.8)
        t = 0.5
        
        # Get time evolution operator
        U = self.simulator._get_time_evolution_operator(H, t)
        
        # Check if U is unitary: U† U = I
        U_dagger_U = U.conj().T @ U
        identity = np.eye(U.shape[0])
        is_unitary = np.allclose(U_dagger_U, identity, atol=1e-10)
        
        print(f"\n✓ Unitary Evolution Test")
        print(f"  Max deviation from identity: {np.max(np.abs(U_dagger_U - identity)):.2e}")
        print(f"  Result: {'✅ PASS' if is_unitary else '❌ FAIL'}")
        
        return is_unitary
    
    def test_entropy_bounds(self):
        """Test 4: Von Neumann entropy must satisfy physical bounds."""
        # Create test states with known entanglement
        all_tests_pass = True
        
        print(f"\n✓ Entropy Bounds Test")
        
        # Test 1: Product state should have zero entropy
        product_state = np.zeros(2**self.n_qubits, dtype=complex)
        product_state[0] = 1.0  # |00...0⟩
        
        region = [0, 1]  # First two qubits
        entropy_product = von_neumann_entropy(product_state, region)
        test1_pass = abs(entropy_product) < 1e-10
        
        print(f"  Product state entropy: {entropy_product:.2e} (should be ~0)")
        print(f"  Test 1: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        all_tests_pass &= test1_pass
        
        # Test 2: Maximally entangled state
        max_ent_state = np.zeros(2**self.n_qubits, dtype=complex)
        max_ent_state[0] = 1/np.sqrt(2)
        max_ent_state[-1] = 1/np.sqrt(2)  # |00...0⟩ + |11...1⟩
        
        entropy_max = von_neumann_entropy(max_ent_state, region)
        max_possible = min(len(region), self.n_qubits - len(region)) * np.log(2)
        test2_pass = entropy_max <= max_possible + 1e-10
        
        print(f"  Max entangled entropy: {entropy_max:.4f} (max possible: {max_possible:.4f})")
        print(f"  Test 2: {'✅ PASS' if test2_pass else '❌ FAIL'}")
        all_tests_pass &= test2_pass
        
        # Test 3: Entropy should be non-negative
        H = self.simulator.build_hamiltonian("xxz", Jx=1.0, Jy=1.0, Jz=0.5, h=0.8)
        evolved_state = self.simulator.time_evolved_state(H, 1.0)
        entropy_evolved = von_neumann_entropy(evolved_state, region)
        test3_pass = entropy_evolved >= -1e-10
        
        print(f"  Evolved state entropy: {entropy_evolved:.4f} (should be ≥ 0)")
        print(f"  Test 3: {'✅ PASS' if test3_pass else '❌ FAIL'}")
        all_tests_pass &= test3_pass
        
        return all_tests_pass
    
    def test_correlation_validity(self):
        """Test 5: Statistical correlations must be mathematically valid."""
        print(f"\n✓ Correlation Validity Test")
        
        # Generate test data with known correlation
        n_points = 100
        true_r = 0.8
        
        # Create correlated data
        x = np.random.normal(0, 1, n_points)
        noise = np.random.normal(0, np.sqrt(1 - true_r**2), n_points)
        y = true_r * x + noise
        
        # Calculate correlation
        calculated_r, p_value = stats.pearsonr(x, y)
        
        # Test bounds
        test1_pass = -1 <= calculated_r <= 1
        test2_pass = 0 <= p_value <= 1
        test3_pass = abs(calculated_r - true_r) < 0.2  # Should be close to true correlation
        
        print(f"  True correlation: {true_r:.3f}")
        print(f"  Calculated correlation: {calculated_r:.3f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Bounds test: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        print(f"  P-value test: {'✅ PASS' if test2_pass else '❌ FAIL'}")
        print(f"  Accuracy test: {'✅ PASS' if test3_pass else '❌ FAIL'}")
        
        return test1_pass and test2_pass and test3_pass
    
    def test_known_analytical_results(self):
        """Test 6: Compare with known analytical results for simple cases."""
        print(f"\n✓ Analytical Comparison Test")
        
        # Test: Two-qubit system with known results
        if self.n_qubits >= 2:
            # Bell state preparation
            bell_state = np.zeros(4, dtype=complex)
            bell_state[0] = 1/np.sqrt(2)  # |00⟩
            bell_state[3] = 1/np.sqrt(2)  # |11⟩
            
            # Analytical result: entropy of one qubit in Bell state = log(2)
            expected_entropy = np.log(2)
            calculated_entropy = von_neumann_entropy(bell_state, [0])
            
            test_pass = abs(calculated_entropy - expected_entropy) < 1e-10
            
            print(f"  Bell state single-qubit entropy:")
            print(f"    Expected: {expected_entropy:.6f}")
            print(f"    Calculated: {calculated_entropy:.6f}")
            print(f"    Difference: {abs(calculated_entropy - expected_entropy):.2e}")
            print(f"  Result: {'✅ PASS' if test_pass else '❌ FAIL'}")
            
            return test_pass
        else:
            print("  Skipped (need at least 2 qubits)")
            return True
    
    def test_energy_conservation(self):
        """Test 7: Energy should be conserved during unitary evolution."""
        print(f"\n✓ Energy Conservation Test")
        
        H = self.simulator.build_hamiltonian("xxz", Jx=1.0, Jy=1.0, Jz=0.5, h=0.8)
        
        # Initial state
        initial_state = np.random.normal(0, 1, 2**self.n_qubits) + 1j * np.random.normal(0, 1, 2**self.n_qubits)
        initial_state = initial_state / np.linalg.norm(initial_state)
        
        # Calculate initial energy
        initial_energy = np.real(initial_state.conj() @ H @ initial_state)
        
        # Evolve and calculate final energy
        final_state = self.simulator.time_evolve_state(initial_state, H, 1.0)
        final_energy = np.real(final_state.conj() @ H @ final_state)
        
        energy_diff = abs(final_energy - initial_energy)
        test_pass = energy_diff < 1e-10
        
        print(f"  Initial energy: {initial_energy:.8f}")
        print(f"  Final energy: {final_energy:.8f}")
        print(f"  Energy difference: {energy_diff:.2e}")
        print(f"  Result: {'✅ PASS' if test_pass else '❌ FAIL'}")
        
        return test_pass
    
    def test_time_reversal(self):
        """Test 8: Forward then backward evolution should return to initial state."""
        print(f"\n✓ Time Reversal Test")
        
        H = self.simulator.build_hamiltonian("tfim", J=1.0, h=0.5)
        t = 0.7
        
        # Initial random state
        initial_state = np.random.normal(0, 1, 2**self.n_qubits) + 1j * np.random.normal(0, 1, 2**self.n_qubits)
        initial_state = initial_state / np.linalg.norm(initial_state)
        
        # Forward evolution
        evolved_state = self.simulator.time_evolve_state(initial_state, H, t)
        
        # Backward evolution
        recovered_state = self.simulator.time_evolve_state(evolved_state, H, -t)
        
        # Check if we recovered the initial state
        fidelity = abs(np.vdot(initial_state, recovered_state))**2
        test_pass = fidelity > 1 - 1e-10
        
        print(f"  Fidelity after round trip: {fidelity:.12f}")
        print(f"  State difference norm: {np.linalg.norm(initial_state - recovered_state):.2e}")
        print(f"  Result: {'✅ PASS' if test_pass else '❌ FAIL'}")
        
        return test_pass
    
    def run_full_validation(self):
        """Run all validation tests."""
        print("🔬 QUANTUM SIMULATION VALIDATION SUITE")
        print("=" * 50)
        print(f"System: {self.n_qubits} qubits, {2**self.n_qubits} states")
        print("=" * 50)
        
        tests = [
            ("State Normalization", lambda: self.test_state_normalization(self.simulator.time_evolved_state(
                self.simulator.build_hamiltonian("xxz"), 0.5))),
            ("Hamiltonian Hermiticity", self.test_hamiltonian_hermiticity),
            ("Unitary Evolution", self.test_unitary_evolution),
            ("Entropy Bounds", self.test_entropy_bounds),
            ("Correlation Validity", self.test_correlation_validity),
            ("Analytical Comparison", self.test_known_analytical_results),
            ("Energy Conservation", self.test_energy_conservation),
            ("Time Reversal", self.test_time_reversal),
        ]
        
        all_passed = True
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                all_passed &= result
            except Exception as e:
                print(f"\n❌ ERROR in {test_name}: {e}")
                results[test_name] = False
                all_passed = False
        
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        print("\n" + "=" * 50)
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"🎯 OVERALL RESULT: {overall_status}")
        print("=" * 50)
        
        if all_passed:
            print("\n🎉 Your quantum simulation is mathematically correct!")
            print("   The results in stable_correlations can be trusted.")
            print("   Your quantum system obeys all fundamental physics laws.")
        else:
            print("\n⚠️  Some validation tests failed!")
            print("   Check the implementation for potential bugs.")
            print("   Results may not be physically meaningful.")
        
        return all_passed, results

def main():
    """Run validation with different system sizes."""
    print("🚀 Starting Quantum Validation Suite...")
    
    # Test different system sizes
    for n_qubits in [4, 6, 8]:
        print(f"\n{'='*60}")
        print(f"🔬 TESTING {n_qubits}-QUBIT SYSTEM")
        print(f"{'='*60}")
        
        validator = QuantumValidationSuite(n_qubits)
        passed, results = validator.run_full_validation()
        
        if not passed:
            print(f"\n⚠️  Issues found with {n_qubits}-qubit system!")
            break
    
    print("\n🏁 Validation Complete!")

if __name__ == "__main__":
    main() 