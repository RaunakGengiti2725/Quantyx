# File: quantumproject/quantum/simulations.py

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ───────────────────────────────
# 1) TIME EVOLUTION QNODE
# ───────────────────────────────

# We will override the default device on demand inside run_one_time_step,
# so here we define a "placeholder" device. The actual device should be created
# in the training pipeline when needed (e.g., with 5 Trotter steps).
dev_placeholder = None

@qml.qnode(qml.device("default.qubit", wires=1), interface="autograd")
def time_evolved_state(t):
    """
    Placeholder QNode. This function is redefined at runtime in run_one_time_step
    with the appropriate number of wires and Hamiltonian. If you call this
    version directly, it will simply return |0⟩ for a single qubit.
    """
    # Default: do nothing, return state of a 1-qubit |0⟩
    return qml.state()


# ───────────────────────────────
# 2) HAMILTONIAN BUILDERS
# ───────────────────────────────

def make_tfim_hamiltonian(n, J=1.0, h=1.0):
    """
    Construct a transverse‐field Ising model (TFIM) Hamiltonian on n qubits:
      H = -J * Σ Z_i Z_{i+1}  - h * Σ X_i
    with periodic boundary conditions.
    """
    coeffs = []
    ops = []
    for i in range(n):
        Zi = qml.PauliZ(i)
        Zj = qml.PauliZ((i + 1) % n)
        coeffs.append(-J)
        ops.append(Zi @ Zj)
    for i in range(n):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def make_xxz_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    """
    Construct an XXZ Hamiltonian on n qubits:
      H = Σ [ Jx * (X_i X_{i+1}) + Jy * (Y_i Y_{i+1}) + Jz * (Z_i Z_{i+1}) ]
          + h * Σ Z_i   (optional longitudinal field)
    with periodic boundary conditions.
    """
    coeffs = []
    ops = []
    for i in range(n):
        ip1 = (i + 1) % n
        Xi = qml.PauliX(i)
        Xj = qml.PauliX(ip1)
        Yi = qml.PauliY(i)
        Yj = qml.PauliY(ip1)
        Zi = qml.PauliZ(i)
        Zj = qml.PauliZ(ip1)

        # X_i X_{i+1}
        coeffs.append(Jx)
        ops.append(Xi @ Xj)
        # Y_i Y_{i+1}
        coeffs.append(Jy)
        ops.append(Yi @ Yj)
        # Z_i Z_{i+1}
        coeffs.append(Jz)
        ops.append(Zi @ Zj)

    # Optional longitudinal field term (on Z)
    if abs(h) > 1e-12:
        for i in range(n):
            coeffs.append(h)
            ops.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, ops)


# ───────────────────────────────
# 3) CONTIGUOUS INTERVAL ENUMERATION
# ───────────────────────────────

def contiguous_intervals(n_qubits, max_interval_size=None):
    """
    Return a list of all contiguous intervals of boundary qubits [0..n_qubits-1].
    If max_interval_size is provided (an integer between 1 and n_qubits-1),
    only intervals up to that length will be returned.
    """
    regions = []
    max_len = max_interval_size if max_interval_size is not None else n_qubits - 1
    for length in range(1, max_len + 1):
        for start in range(0, n_qubits - length + 1):
            regions.append(tuple(range(start, start + length)))
    return regions


# ───────────────────────────────
# 4) VON NEUMANN ENTROPY
# ───────────────────────────────

def reduced_density_matrix(state, subsys):
    """
    Given a full state vector `state` of length 2^n_qubits and a list `subsys`
    of qubit indices, compute the reduced density matrix on that subsystem
    by tracing out all other qubits.
    """
    n = int(np.log2(len(state)))
    dims = [2] * n
    psi = pnp.reshape(state, dims)
    keep = list(subsys)
    trace_out = [i for i in range(n) if i not in keep]
    # Partial trace
    rho = pnp.tensordot(psi, pnp.conj(psi), axes=(trace_out, trace_out))
    dim_sub = 2 ** len(subsys)
    return pnp.reshape(rho, (dim_sub, dim_sub))


def von_neumann_entropy(state, subsys):
    """
    Compute the von Neumann entropy S(ρ) = -Tr(ρ log ρ) of the subsystem 'subsys'
    from the full state vector `state`.
    """
    rho = reduced_density_matrix(state, subsys)
    # Compute eigenvalues
    evs = qml.math.eigvalsh(rho)
    # Clip for numerical stability
    evs = pnp.clip(evs, 1e-12, 1.0)
    return float(-pnp.sum(evs * pnp.log(evs)))


# ───────────────────────────────
# 5) REDEFINE time_evolved_state ON THE FLY
# ───────────────────────────────

def get_time_evolution_qnode(n_qubits, hamiltonian, trotter_steps=1):
    """
    Return a QNode that, given a time t, prepares |+>^n, then applies
    e^{-i H t} with `trotter_steps` slices, and returns the state vector.
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, interface="autograd")
    def evolve(t):
        # Prepare |+> on each qubit
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # Apply Trotterized time evolution
        qml.templates.ApproxTimeEvolution(hamiltonian, t, trotter_steps)
        return qml.state()

    return evolve


# ───────────────────────────────
# 6) SIMULATOR CLASS
# ───────────────────────────────

class Simulator:
    """
    Quantum simulator class that wraps the quantum evolution functionality.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None)
    
    def build_hamiltonian(self, hamiltonian_type, **kwargs):
        """
        Build a Hamiltonian of the specified type.
        
        Args:
            hamiltonian_type: "tfim", "xxz", or "heisenberg"
            **kwargs: Parameters for the Hamiltonian
        """
        if hamiltonian_type.lower() in ["tfim", "transverse_field_ising"]:
            J = kwargs.get("J", 1.0)
            h = kwargs.get("h", 1.0)
            return make_tfim_hamiltonian(self.n_qubits, J, h)
        elif hamiltonian_type.lower() in ["xxz"]:
            Jx = kwargs.get("Jx", 1.0)
            Jy = kwargs.get("Jy", 1.0)
            Jz = kwargs.get("Jz", 1.0)
            h = kwargs.get("h", 0.0)
            return make_xxz_hamiltonian(self.n_qubits, Jx, Jy, Jz, h)
        elif hamiltonian_type.lower() in ["heisenberg"]:
            # Heisenberg is XXZ with Jx=Jy=Jz
            J = kwargs.get("J", 1.0)
            h = kwargs.get("h", 0.0)
            return make_xxz_hamiltonian(self.n_qubits, J, J, J, h)
        else:
            raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
    
    def time_evolved_state(self, hamiltonian, t, trotter_steps=1):
        """
        Compute the time-evolved state from the |+>^n initial state.
        
        Args:
            hamiltonian: PennyLane Hamiltonian
            t: Evolution time
            trotter_steps: Number of Trotter steps for time evolution
            
        Returns:
            State vector after time evolution
        """
        evolve_qnode = get_time_evolution_qnode(self.n_qubits, hamiltonian, trotter_steps)
        return evolve_qnode(t)

    def add_entanglement_layer(self, state: np.ndarray) -> np.ndarray:
        """
        Add an entanglement layer to create more realistic quantum correlations.
        This applies controlled operations to increase entanglement in the state.
        """
        n_qubits = int(np.log2(len(state)))
        
        # Create a simple entangling circuit
        # Apply controlled rotations between adjacent qubits
        for i in range(n_qubits - 1):
            # Create a controlled rotation matrix
            ctrl_rot = np.eye(len(state), dtype=complex)
            
            # Apply small rotations to create entanglement
            angle = 0.2  # Small angle for gentle entanglement
            cos_half = np.cos(angle / 2)
            sin_half = np.sin(angle / 2)
            
            # Create entangling unitary (simplified Bell-state-like transformation)
            for j in range(0, len(state), 4):
                if j + 3 < len(state):
                    # Apply controlled Y rotation
                    ctrl_rot[j + 1, j + 1] = cos_half
                    ctrl_rot[j + 1, j + 3] = -1j * sin_half
                    ctrl_rot[j + 3, j + 1] = -1j * sin_half
                    ctrl_rot[j + 3, j + 3] = cos_half
            
            state = ctrl_rot @ state
        
        # Normalize the state
        state = state / np.linalg.norm(state)
        return state


# ───────────────────────────────
# 7) BOUNDARY ENERGY DELTA
# ───────────────────────────────

def boundary_energy_delta(state_base, state_t):
    """
    Compute ΔE_i = <Z_i>(t) - <Z_i>(0) for each boundary qubit i.
    Returns numpy array of length n_qubits.
    """
    n_qubits = int(np.log2(len(state_base)))
    
    def z_expectation(state_vector, wire):
        """Compute ⟨Z⟩ on wire from state vector."""
        n = int(np.log2(len(state_vector)))
        dims = [2] * n
        psi = state_vector.reshape(dims)
        subsys = [wire]
        trace_out = [i for i in range(n) if i not in subsys]
        rho = np.tensordot(psi, psi.conj(), axes=(trace_out, trace_out))
        # ⟨Z⟩ = Tr(ρ * σ_z) where σ_z = [[1, 0], [0, -1]]
        z_op = np.array([[1, 0], [0, -1]], dtype=complex)
        return float(np.real(np.trace(rho @ z_op)))
    
    deltas = []
    for i in range(n_qubits):
        E0 = z_expectation(state_base, i)
        Et = z_expectation(state_t, i)
        deltas.append(Et - E0)
    
    return np.array(deltas)
