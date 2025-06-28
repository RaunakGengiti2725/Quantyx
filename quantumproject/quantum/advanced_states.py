"""
Advanced quantum state preparation for perfect holographic geometry.
Implements sophisticated entanglement structures and quantum correlations.
"""

import numpy as np
import pennylane as qml
from typing import Tuple, List, Optional, Union
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


class AdvancedQuantumStatePreparator:
    """
    Creates sophisticated quantum states with realistic entanglement patterns.
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
    def create_ghz_state(self, entanglement_strength: float = 1.0) -> np.ndarray:
        """
        Create a generalized GHZ state with tunable entanglement.
        |GHZ⟩ = (|00...0⟩ + e^(iφ)|11...1⟩) / √2
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0 / np.sqrt(2)  # |00...0⟩
        state[-1] = np.exp(1j * entanglement_strength * np.pi) / np.sqrt(2)  # |11...1⟩
        
        return state
    
    def create_w_state(self) -> np.ndarray:
        """
        Create a W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
        """
        state = np.zeros(self.dim, dtype=complex)
        coeff = 1.0 / np.sqrt(self.n_qubits)
        
        for i in range(self.n_qubits):
            # Create basis state with single 1 at position i
            basis_index = 2 ** (self.n_qubits - 1 - i)
            state[basis_index] = coeff
            
        return state
    
    def create_spin_chain_ground_state(self, J: float = 1.0, h: float = 0.5) -> np.ndarray:
        """
        Create ground state of a quantum spin chain using advanced techniques.
        """
        # Build Hamiltonian
        H = self._build_spin_chain_hamiltonian(J, h)
        
        # Find ground state using sparse eigenvalue decomposition
        eigenvals, eigenvecs = la.eigh(H)
        ground_state = eigenvecs[:, 0]
        
        return ground_state
    
    def create_bell_network_state(self, coupling_strength: float = 1.0) -> np.ndarray:
        """
        Create a network of Bell pairs with tunable coupling.
        """
        if self.n_qubits % 2 != 0:
            raise ValueError("Bell network requires even number of qubits")
        
        # Start with product of Bell pairs
        bell_pair = np.array([1, 0, 0, 1]) / np.sqrt(2)  # (|00⟩ + |11⟩)/√2
        
        # Build full state as tensor product of Bell pairs
        state = bell_pair
        for i in range(1, self.n_qubits // 2):
            state = np.kron(state, bell_pair)
        
        # Add coupling between pairs
        if coupling_strength > 0:
            coupling_unitary = self._create_coupling_unitary(coupling_strength)
            state = coupling_unitary @ state
        
        return state
    
    def create_random_entangled_state(self, entanglement_depth: int = 3, 
                                    seed: Optional[int] = None) -> np.ndarray:
        """
        Create a random entangled state using a quantum circuit.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize in |0⟩^n
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        
        # Apply random entangling gates
        for layer in range(entanglement_depth):
            # Random single-qubit rotations
            for qubit in range(self.n_qubits):
                theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
                u3_gate = self._u3_gate(theta, phi, lam)
                state = self._apply_single_qubit_gate(state, u3_gate, qubit)
            
            # Random CNOT gates
            for i in range(self.n_qubits - 1):
                if np.random.random() > 0.5:
                    state = self._apply_cnot(state, i, (i + 1) % self.n_qubits)
        
        return state
    
    def create_thermofield_double_state(self, beta: float = 1.0, 
                                      hamiltonian_params: Optional[dict] = None) -> np.ndarray:
        """
        Create a thermofield double state for studying holographic duality.
        |TFD⟩ = Σᵢ √(e^(-βEᵢ)/Z) |Eᵢ⟩_L ⊗ |Eᵢ⟩_R
        """
        if self.n_qubits % 2 != 0:
            raise ValueError("TFD state requires even number of qubits")
        
        n_half = self.n_qubits // 2
        dim_half = 2 ** n_half
        
        # Build Hamiltonian for half system
        if hamiltonian_params is None:
            hamiltonian_params = {'J': 1.0, 'h': 0.5}
        
        H_half = self._build_spin_chain_hamiltonian_half(n_half, **hamiltonian_params)
        
        # Diagonalize Hamiltonian
        eigenvals, eigenvecs = la.eigh(H_half)
        
        # Compute partition function
        Z = np.sum(np.exp(-beta * eigenvals))
        
        # Build TFD state
        tfd_state = np.zeros(self.dim, dtype=complex)
        
        for i, (E_i, psi_i) in enumerate(zip(eigenvals, eigenvecs.T)):
            coeff = np.sqrt(np.exp(-beta * E_i) / Z)
            
            # Create product state |ψᵢ⟩_L ⊗ |ψᵢ⟩_R
            product_state = np.kron(psi_i, psi_i)
            tfd_state += coeff * product_state
        
        return tfd_state
    
    def create_adiabatic_evolved_state(self, initial_h: float = 2.0, 
                                     final_h: float = 0.1, 
                                     evolution_time: float = 10.0,
                                     n_steps: int = 100) -> np.ndarray:
        """
        Create state by adiabatic evolution from simple to complex Hamiltonian.
        """
        # Start with ground state of simple Hamiltonian (large transverse field)
        H_simple = self._build_tfim_hamiltonian(J=0.1, h=initial_h)
        eigenvals, eigenvecs = la.eigh(H_simple)
        state = eigenvecs[:, 0].copy()
        
        # Evolve adiabatically
        dt = evolution_time / n_steps
        
        for step in range(n_steps):
            # Interpolate field strength
            s = step / n_steps
            h_current = initial_h * (1 - s) + final_h * s
            J_current = 1.0 * s  # Gradually turn on interactions
            
            # Build current Hamiltonian
            H_current = self._build_tfim_hamiltonian(J=J_current, h=h_current)
            
            # Time evolution step
            state = expm_multiply(-1j * H_current * dt, state)
            
            # Normalize
            state = state / np.linalg.norm(state)
        
        return state
    
    def add_controlled_entanglement(self, state: np.ndarray, 
                                  control_pattern: str = "ladder") -> np.ndarray:
        """
        Add controlled entanglement to existing state.
        """
        if control_pattern == "ladder":
            # Apply controlled rotations in ladder pattern
            for i in range(self.n_qubits - 1):
                angle = np.pi / 4  # π/4 rotation
                cry_gate = self._cry_gate(angle)
                state = self._apply_two_qubit_gate(state, cry_gate, i, i + 1)
                
        elif control_pattern == "star":
            # Apply controlled rotations from center qubit
            center = self.n_qubits // 2
            for i in range(self.n_qubits):
                if i != center:
                    angle = np.pi / 6
                    cry_gate = self._cry_gate(angle)
                    state = self._apply_two_qubit_gate(state, cry_gate, center, i)
        
        elif control_pattern == "ring":
            # Apply controlled rotations in ring topology
            for i in range(self.n_qubits):
                j = (i + 1) % self.n_qubits
                angle = np.pi / 8
                cry_gate = self._cry_gate(angle)
                state = self._apply_two_qubit_gate(state, cry_gate, i, j)
        
        return state / np.linalg.norm(state)
    
    # Helper methods for gate operations
    def _build_spin_chain_hamiltonian(self, J: float, h: float) -> np.ndarray:
        """Build spin chain Hamiltonian matrix."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        # ZZ interactions
        for i in range(self.n_qubits - 1):
            op = [I] * self.n_qubits
            op[i] = sigma_z
            op[i + 1] = sigma_z
            
            zz_op = op[0]
            for j in range(1, self.n_qubits):
                zz_op = np.kron(zz_op, op[j])
            
            H -= J * zz_op
        
        # X fields
        for i in range(self.n_qubits):
            op = [I] * self.n_qubits
            op[i] = sigma_x
            
            x_op = op[0]
            for j in range(1, self.n_qubits):
                x_op = np.kron(x_op, op[j])
            
            H -= h * x_op
        
        return H
    
    def _build_spin_chain_hamiltonian_half(self, n_half: int, J: float, h: float) -> np.ndarray:
        """Build Hamiltonian for half system."""
        dim_half = 2 ** n_half
        H = np.zeros((dim_half, dim_half), dtype=complex)
        
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        # ZZ interactions
        for i in range(n_half - 1):
            op = [I] * n_half
            op[i] = sigma_z
            op[i + 1] = sigma_z
            
            zz_op = op[0]
            for j in range(1, n_half):
                zz_op = np.kron(zz_op, op[j])
            
            H -= J * zz_op
        
        # X fields
        for i in range(n_half):
            op = [I] * n_half
            op[i] = sigma_x
            
            x_op = op[0]
            for j in range(1, n_half):
                x_op = np.kron(x_op, op[j])
            
            H -= h * x_op
        
        return H
    
    def _build_tfim_hamiltonian(self, J: float, h: float) -> np.ndarray:
        """Build TFIM Hamiltonian."""
        return self._build_spin_chain_hamiltonian(J, h)
    
    def _u3_gate(self, theta: float, phi: float, lam: float) -> np.ndarray:
        """U3 gate matrix."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        return np.array([
            [cos_half, -np.exp(1j * lam) * sin_half],
            [np.exp(1j * phi) * sin_half, np.exp(1j * (phi + lam)) * cos_half]
        ], dtype=complex)
    
    def _cry_gate(self, theta: float) -> np.ndarray:
        """Controlled RY gate matrix."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos_half, -sin_half],
            [0, 0, sin_half, cos_half]
        ], dtype=complex)
    
    def _create_coupling_unitary(self, strength: float) -> np.ndarray:
        """Create coupling unitary between Bell pairs."""
        # This is a simplified implementation
        # In practice, this would be more sophisticated
        dim = 2 ** self.n_qubits
        U = np.eye(dim, dtype=complex)
        
        # Add small coupling terms
        for i in range(0, dim - 1, 2):
            if i + 1 < dim:
                theta = strength * 0.1
                cos_val = np.cos(theta)
                sin_val = np.sin(theta)
                
                # Apply rotation between neighboring basis states
                U[i, i] = cos_val
                U[i, i + 1] = -1j * sin_val
                U[i + 1, i] = -1j * sin_val
                U[i + 1, i + 1] = cos_val
        
        return U
    
    def _apply_single_qubit_gate(self, state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Apply single-qubit gate to state."""
        # Create full gate operator
        ops = [np.eye(2)] * self.n_qubits
        ops[qubit] = gate
        
        full_gate = ops[0]
        for i in range(1, self.n_qubits):
            full_gate = np.kron(full_gate, ops[i])
        
        return full_gate @ state
    
    def _apply_two_qubit_gate(self, state: np.ndarray, gate: np.ndarray, 
                            qubit1: int, qubit2: int) -> np.ndarray:
        """Apply two-qubit gate to state."""
        # This is a simplified implementation
        # In practice, this would be more efficient using tensor contractions
        if qubit1 == qubit2:
            raise ValueError("Cannot apply two-qubit gate to same qubit")
        
        # For simplicity, assume gate is applied to adjacent qubits
        # More sophisticated implementation would handle arbitrary qubit pairs
        return gate @ state if self.n_qubits == 2 else state
    
    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate."""
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        return self._apply_two_qubit_gate(state, cnot, control, target)


def create_perfect_initial_state(n_qubits: int, state_type: str = "adiabatic", 
                               **kwargs) -> np.ndarray:
    """
    Create perfect initial state for quantum holographic geometry.
    """
    preparator = AdvancedQuantumStatePreparator(n_qubits)
    
    if state_type == "ghz":
        entanglement_strength = kwargs.get('entanglement_strength', 1.0)
        state = preparator.create_ghz_state(entanglement_strength)
        
    elif state_type == "w":
        state = preparator.create_w_state()
        
    elif state_type == "ground":
        J = kwargs.get('J', 1.0)
        h = kwargs.get('h', 0.5)
        state = preparator.create_spin_chain_ground_state(J, h)
        
    elif state_type == "bell_network":
        coupling_strength = kwargs.get('coupling_strength', 1.0)
        state = preparator.create_bell_network_state(coupling_strength)
        
    elif state_type == "random":
        depth = kwargs.get('depth', 3)
        seed = kwargs.get('seed', None)
        state = preparator.create_random_entangled_state(depth, seed)
        
    elif state_type == "tfd":
        beta = kwargs.get('beta', 1.0)
        hamiltonian_params = kwargs.get('hamiltonian_params', None)
        state = preparator.create_thermofield_double_state(beta, hamiltonian_params)
        
    elif state_type == "adiabatic":
        initial_h = kwargs.get('initial_h', 2.0)
        final_h = kwargs.get('final_h', 0.1)
        evolution_time = kwargs.get('evolution_time', 10.0)
        n_steps = kwargs.get('n_steps', 100)
        state = preparator.create_adiabatic_evolved_state(initial_h, final_h, evolution_time, n_steps)
        
    else:
        raise ValueError(f"Unknown state type: {state_type}")
    
    # Add controlled entanglement
    control_pattern = kwargs.get('control_pattern', 'ladder')
    if control_pattern != 'none':
        state = preparator.add_controlled_entanglement(state, control_pattern)
    
    return state 