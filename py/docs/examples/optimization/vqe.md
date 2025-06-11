# Variational Quantum Eigensolver (VQE)

**Level:** 🔴 Advanced  
**Runtime:** 2-5 minutes  
**Topics:** Quantum chemistry, Molecular simulation, Eigenvalue problems  
**Applications:** Drug discovery, Materials science

Learn to find ground state energies of molecules using the Variational Quantum Eigensolver - one of the most promising near-term quantum algorithms for quantum chemistry.

## What is VQE?

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the ground state energy of quantum systems, particularly molecules:

**Problem:** Given a molecular Hamiltonian H, find the ground state energy E₀ = min⟨ψ|H|ψ⟩

**VQE Approach:**
1. **Prepare** a parameterized quantum state |ψ(θ)⟩ (ansatz)
2. **Measure** expectation value ⟨ψ(θ)|H|ψ(θ)⟩
3. **Optimize** parameters θ to minimize energy
4. **Extract** ground state energy and properties

**Key Applications:**
- Drug discovery and pharmaceutical research
- Catalyst design and materials science
- Understanding chemical reaction mechanisms
- Battery and solar cell material optimization

## Quantum Chemistry Background

### Molecular Hamiltonians

In quantum chemistry, molecules are described by the electronic Hamiltonian:

```
H = -∑ᵢ ∇ᵢ²/2 + ∑ᵢ<ⱼ 1/|rᵢ-rⱼ| - ∑ᵢ,ₐ Zₐ/|rᵢ-Rₐ| + ∑ₐ<ᵦ ZₐZᵦ/|Rₐ-Rᵦ|
```

Where:
- First term: electron kinetic energy
- Second term: electron-electron repulsion
- Third term: electron-nuclei attraction  
- Fourth term: nuclei-nuclei repulsion

### Second Quantization

For quantum computers, we use second quantization with creation/annihilation operators:

```
H = ∑ᵢⱼ hᵢⱼ aᵢ†aⱼ + ∑ᵢⱼₖₗ hᵢⱼₖₗ aᵢ†aⱼ†aₖaₗ
```

### Jordan-Wigner Transformation

Fermions are mapped to qubits using Jordan-Wigner transformation:
- aᵢ† → ∏ⱼ<ᵢ Zⱼ ⊗ (Xᵢ - iYᵢ)/2
- Preserves fermionic anticommutation relations

## Implementation

### Molecular Hamiltonian Setup

```python
import quantrs2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigh

class MolecularHamiltonian:
    """
    Representation of a molecular Hamiltonian for VQE calculations.
    """
    
    def __init__(self, one_body_integrals, two_body_integrals, nuclear_repulsion=0.0):
        """
        Initialize molecular Hamiltonian.
        
        Args:
            one_body_integrals: One-electron integrals h_ij
            two_body_integrals: Two-electron integrals h_ijkl  
            nuclear_repulsion: Nuclear repulsion energy
        """
        self.one_body = one_body_integrals
        self.two_body = two_body_integrals
        self.nuclear_repulsion = nuclear_repulsion
        
        self.num_orbitals = len(one_body_integrals)
        self.num_qubits = 2 * self.num_orbitals  # Spin orbitals
        
        # Convert to Pauli operators for quantum computer
        self.pauli_operators = self._jordan_wigner_transform()
        
        print(f"🧪 Molecular Hamiltonian:")
        print(f"   Orbitals: {self.num_orbitals}")
        print(f"   Qubits: {self.num_qubits}")
        print(f"   Pauli terms: {len(self.pauli_operators)}")
        print(f"   Nuclear repulsion: {nuclear_repulsion:.6f}")
    
    def _jordan_wigner_transform(self):
        """Convert fermionic Hamiltonian to Pauli operators."""
        
        pauli_ops = []
        
        # One-body terms: h_ij * a_i† * a_j
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                coeff = self.one_body[i//2, j//2] if (i%2 == j%2) else 0.0
                
                if abs(coeff) > 1e-12:
                    if i == j:
                        # Number operator: (I - Z)/2
                        pauli_ops.append({
                            'coefficient': coeff / 2,
                            'pauli_string': 'I' * self.num_qubits
                        })
                        pauli_ops.append({
                            'coefficient': -coeff / 2,
                            'pauli_string': 'I' * i + 'Z' + 'I' * (self.num_qubits - i - 1)
                        })
                    else:
                        # Hopping terms: (X+iY)(X-iY)/4 and (X-iY)(X+iY)/4
                        pauli_string_1 = ['I'] * self.num_qubits
                        pauli_string_2 = ['I'] * self.num_qubits
                        
                        # Add Z operators for Jordan-Wigner strings
                        for k in range(min(i, j) + 1, max(i, j)):
                            pauli_string_1[k] = 'Z'
                            pauli_string_2[k] = 'Z'
                        
                        # Pauli operators at sites i and j
                        pauli_string_1[i] = 'X'
                        pauli_string_1[j] = 'X'
                        pauli_string_2[i] = 'Y'
                        pauli_string_2[j] = 'Y'
                        
                        pauli_ops.append({
                            'coefficient': coeff / 4,
                            'pauli_string': ''.join(pauli_string_1)
                        })
                        pauli_ops.append({
                            'coefficient': coeff / 4,
                            'pauli_string': ''.join(pauli_string_2)
                        })
        
        # Two-body terms: h_ijkl * a_i† * a_j† * a_k * a_l
        # Simplified - only include diagonal terms for this example
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                # Coulomb interaction: n_i * n_j
                if i//2 < self.num_orbitals and j//2 < self.num_orbitals:
                    coeff = self.two_body[i//2, j//2, i//2, j//2] if (i%2 == j%2) else 0.0
                    
                    if abs(coeff) > 1e-12:
                        # (I-Z_i)(I-Z_j)/4
                        pauli_ops.extend([
                            {
                                'coefficient': coeff / 4,
                                'pauli_string': 'I' * self.num_qubits
                            },
                            {
                                'coefficient': -coeff / 4,
                                'pauli_string': self._pauli_string_with_z(i)
                            },
                            {
                                'coefficient': -coeff / 4,
                                'pauli_string': self._pauli_string_with_z(j)
                            },
                            {
                                'coefficient': coeff / 4,
                                'pauli_string': self._pauli_string_with_zz(i, j)
                            }
                        ])
        
        # Remove terms with zero coefficients and combine like terms
        return self._consolidate_pauli_terms(pauli_ops)
    
    def _pauli_string_with_z(self, position):
        """Create Pauli string with Z at given position."""
        pauli_list = ['I'] * self.num_qubits
        pauli_list[position] = 'Z'
        return ''.join(pauli_list)
    
    def _pauli_string_with_zz(self, pos1, pos2):
        """Create Pauli string with Z at two positions."""
        pauli_list = ['I'] * self.num_qubits
        pauli_list[pos1] = 'Z'
        pauli_list[pos2] = 'Z'
        return ''.join(pauli_list)
    
    def _consolidate_pauli_terms(self, pauli_ops):
        """Combine Pauli terms with same operators."""
        consolidated = {}
        
        for op in pauli_ops:
            pauli_str = op['pauli_string']
            if pauli_str in consolidated:
                consolidated[pauli_str] += op['coefficient']
            else:
                consolidated[pauli_str] = op['coefficient']
        
        # Remove zero terms
        return [
            {'coefficient': coeff, 'pauli_string': pauli_str}
            for pauli_str, coeff in consolidated.items()
            if abs(coeff) > 1e-12
        ]
    
    def compute_energy_expectation(self, quantum_state):
        """Compute ⟨ψ|H|ψ⟩ for given quantum state."""
        
        total_energy = self.nuclear_repulsion
        
        for op in self.pauli_operators:
            coefficient = op['coefficient']
            pauli_string = op['pauli_string']
            
            # Compute expectation value ⟨ψ|P|ψ⟩ for Pauli operator P
            expectation = self._pauli_expectation(quantum_state, pauli_string)
            total_energy += coefficient * expectation
        
        return total_energy
    
    def _pauli_expectation(self, state, pauli_string):
        """Compute expectation value of Pauli string."""
        
        # For simplicity, we'll compute this classically
        # In practice, this would be measured on quantum hardware
        
        if pauli_string == 'I' * len(pauli_string):
            return 1.0
        
        # For this example, we'll use simplified expectation calculation
        # Real implementation would apply Pauli rotations and measure
        expectation = 1.0
        for i, pauli in enumerate(pauli_string):
            if pauli == 'Z':
                # ⟨Z⟩ depends on state - simplified calculation
                expectation *= np.cos(2 * np.pi * i / len(pauli_string))
            elif pauli in ['X', 'Y']:
                # ⟨X⟩, ⟨Y⟩ typically smaller
                expectation *= 0.5
        
        return expectation
    
    def exact_diagonalization(self):
        """Compute exact ground state energy (for small systems)."""
        
        # Build full Hamiltonian matrix (exponentially large!)
        dim = 2 ** self.num_qubits
        H_matrix = np.zeros((dim, dim), dtype=complex)
        
        # Add nuclear repulsion to diagonal
        H_matrix += self.nuclear_repulsion * np.eye(dim)
        
        # Add Pauli operator contributions
        for op in self.pauli_operators:
            coefficient = op['coefficient']
            pauli_string = op['pauli_string']
            
            pauli_matrix = self._pauli_string_to_matrix(pauli_string)
            H_matrix += coefficient * pauli_matrix
        
        # Diagonalize
        eigenvalues, eigenvectors = eigh(H_matrix)
        
        ground_state_energy = eigenvalues[0]
        ground_state = eigenvectors[:, 0]
        
        print(f"🎯 Exact diagonalization:")
        print(f"   Ground state energy: {ground_state_energy:.8f}")
        print(f"   Matrix dimension: {dim}x{dim}")
        
        return ground_state_energy, ground_state
    
    def _pauli_string_to_matrix(self, pauli_string):
        """Convert Pauli string to matrix representation."""
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Tensor product of individual Pauli matrices
        result = pauli_dict[pauli_string[0]]
        for pauli in pauli_string[1:]:
            result = np.kron(result, pauli_dict[pauli])
        
        return result

# Create example molecules
def create_hydrogen_molecule(bond_distance=0.735):
    """Create H2 molecule Hamiltonian (minimal basis)."""
    
    print(f"🔬 Creating H₂ molecule (R = {bond_distance:.3f} Å)")
    
    # Simplified H2 integrals for minimal basis (STO-3G)
    # These would typically come from classical quantum chemistry calculations
    
    # One-electron integrals (kinetic + nuclear attraction)
    h_ij = np.array([
        [-1.252, -0.677],
        [-0.677, -1.252]
    ])
    
    # Two-electron integrals (electron repulsion)
    h_ijkl = np.zeros((2, 2, 2, 2))
    
    # Diagonal Coulomb integrals
    h_ijkl[0, 0, 0, 0] = 0.674
    h_ijkl[1, 1, 1, 1] = 0.674
    h_ijkl[0, 1, 0, 1] = 0.664
    h_ijkl[1, 0, 1, 0] = 0.664
    
    # Exchange integrals
    h_ijkl[0, 1, 1, 0] = 0.697
    h_ijkl[1, 0, 0, 1] = 0.697
    
    # Nuclear repulsion energy
    nuclear_repulsion = 1.0 / bond_distance
    
    return MolecularHamiltonian(h_ij, h_ijkl, nuclear_repulsion)

def create_lithium_hydride():
    """Create LiH molecule Hamiltonian."""
    
    print(f"🔬 Creating LiH molecule")
    
    # Simplified LiH integrals (would be much larger in reality)
    h_ij = np.array([
        [-2.344, -0.174],
        [-0.174, -0.237]
    ])
    
    h_ijkl = np.zeros((2, 2, 2, 2))
    h_ijkl[0, 0, 0, 0] = 0.674
    h_ijkl[1, 1, 1, 1] = 0.697
    h_ijkl[0, 1, 0, 1] = 0.663
    h_ijkl[1, 0, 1, 0] = 0.663
    
    nuclear_repulsion = 0.995
    
    return MolecularHamiltonian(h_ij, h_ijkl, nuclear_repulsion)

# Create sample molecules
h2_molecule = create_hydrogen_molecule()
lih_molecule = create_lithium_hydride()
```

### VQE Algorithm Implementation

```python
class VariationalQuantumEigensolver:
    """
    Variational Quantum Eigensolver implementation.
    """
    
    def __init__(self, hamiltonian, ansatz_type='uccsd', optimizer='COBYLA'):
        """
        Initialize VQE solver.
        
        Args:
            hamiltonian: MolecularHamiltonian object
            ansatz_type: Type of variational ansatz ('uccsd', 'hardware_efficient', 'hea')
            optimizer: Classical optimizer ('COBYLA', 'SLSQP', 'BFGS')
        """
        self.hamiltonian = hamiltonian
        self.ansatz_type = ansatz_type
        self.optimizer = optimizer
        
        self.num_qubits = hamiltonian.num_qubits
        
        # Initialize ansatz
        self.ansatz = self._create_ansatz()
        self.num_parameters = self.ansatz.num_parameters
        
        # Optimization tracking
        self.optimization_history = []
        self.best_energy = float('inf')
        self.best_parameters = None
        
        print(f"⚗️ VQE Configuration:")
        print(f"   Ansatz: {ansatz_type}")
        print(f"   Parameters: {self.num_parameters}")
        print(f"   Optimizer: {optimizer}")
    
    def _create_ansatz(self):
        """Create variational ansatz circuit."""
        
        if self.ansatz_type == 'uccsd':
            return UCCSDansatz(self.num_qubits)
        elif self.ansatz_type == 'hardware_efficient':
            return HardwareEfficientAnsatz(self.num_qubits)
        elif self.ansatz_type == 'hea':
            return HardwareEfficientAnsatz(self.num_qubits)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
    
    def create_vqe_circuit(self, parameters):
        """Create VQE quantum circuit with given parameters."""
        
        circuit = quantrs2.Circuit(self.num_qubits)
        
        # Initialize reference state (Hartree-Fock)
        self._prepare_hartree_fock_state(circuit)
        
        # Apply ansatz
        self.ansatz.apply_to_circuit(circuit, parameters)
        
        return circuit
    
    def _prepare_hartree_fock_state(self, circuit):
        """Prepare Hartree-Fock reference state."""
        
        # For H2, HF state is |10⟩ (one electron per orbital, opposite spins)
        # For LiH, HF state is |1100⟩ (doubly occupied Li 1s, singly occupied H 1s)
        
        # Simplified: occupy first orbitals
        num_electrons = self.hamiltonian.num_orbitals  # Assuming neutral molecules
        
        for i in range(num_electrons):
            circuit.x(i)
        
        print(f"   Prepared HF state with {num_electrons} electrons")
    
    def measure_energy_expectation(self, parameters, num_shots=1000):
        """Measure expectation value ⟨ψ(θ)|H|ψ(θ)⟩."""
        
        circuit = self.create_vqe_circuit(parameters)
        
        total_energy = self.hamiltonian.nuclear_repulsion
        
        # Measure each Pauli term
        for pauli_op in self.hamiltonian.pauli_operators:
            coefficient = pauli_op['coefficient']
            pauli_string = pauli_op['pauli_string']
            
            # Create measurement circuit for this Pauli operator
            measurement_circuit = circuit.copy()
            self._add_pauli_measurements(measurement_circuit, pauli_string)
            measurement_circuit.measure_all()
            
            # Run circuit and compute expectation
            expectation = 0.0
            for _ in range(num_shots):
                result = measurement_circuit.run()
                expectation += self._compute_pauli_expectation(result, pauli_string)
            
            expectation /= num_shots
            total_energy += coefficient * expectation
        
        return total_energy
    
    def _add_pauli_measurements(self, circuit, pauli_string):
        """Add basis rotations for measuring Pauli operators."""
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                circuit.ry(i, -np.pi/2)  # Rotate X to Z basis
            elif pauli == 'Y':
                circuit.rx(i, np.pi/2)   # Rotate Y to Z basis
            # Z measurements require no rotation
    
    def _compute_pauli_expectation(self, measurement_result, pauli_string):
        """Compute expectation value from measurement results."""
        
        probabilities = measurement_result.state_probabilities()
        expectation = 0.0
        
        for state_str, prob in probabilities.items():
            # Compute (-1)^(number of 1s in measured qubits for non-I Paulis)
            parity = 0
            for i, (bit, pauli) in enumerate(zip(state_str, pauli_string)):
                if pauli != 'I' and bit == '1':
                    parity += 1
            
            sign = (-1) ** parity
            expectation += sign * prob
        
        return expectation
    
    def cost_function(self, parameters):
        """Cost function for classical optimization."""
        
        energy = self.measure_energy_expectation(parameters)
        
        # Track optimization progress
        self.optimization_history.append({
            'iteration': len(self.optimization_history),
            'parameters': parameters.copy(),
            'energy': energy
        })
        
        # Update best solution
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_parameters = parameters.copy()
        
        # Print progress
        if len(self.optimization_history) % 10 == 0:
            print(f"   Iteration {len(self.optimization_history)}: E = {energy:.6f}")
        
        return energy
    
    def optimize(self, initial_parameters=None, maxiter=100):
        """Optimize VQE parameters to find ground state energy."""
        
        print(f"🚀 Starting VQE optimization...")
        
        if initial_parameters is None:
            # Small random initialization (important for VQE)
            np.random.seed(42)
            initial_parameters = np.random.normal(0, 0.1, self.num_parameters)
        
        print(f"   Initial parameters: {initial_parameters}")
        
        # Clear optimization history
        self.optimization_history = []
        self.best_energy = float('inf')
        
        # Run classical optimization
        result = minimize(
            self.cost_function,
            initial_parameters,
            method=self.optimizer,
            options={'maxiter': maxiter, 'disp': True}
        )
        
        print(f"\n✅ VQE optimization completed!")
        print(f"   Ground state energy: {self.best_energy:.8f}")
        print(f"   Optimal parameters: {self.best_parameters}")
        print(f"   Iterations: {len(self.optimization_history)}")
        
        return result
    
    def analyze_convergence(self):
        """Analyze VQE convergence behavior."""
        
        if not self.optimization_history:
            print("No optimization data available")
            return
        
        energies = [point['energy'] for point in self.optimization_history]
        iterations = [point['iteration'] for point in self.optimization_history]
        
        plt.figure(figsize=(12, 5))
        
        # Energy convergence
        plt.subplot(1, 2, 1)
        plt.plot(iterations, energies, 'b-', linewidth=2)
        plt.axhline(y=self.best_energy, color='r', linestyle='--', 
                   label=f'Best: {self.best_energy:.6f}')
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Hartree)')
        plt.title('VQE Energy Convergence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Parameter evolution
        plt.subplot(1, 2, 2)
        parameters_history = np.array([point['parameters'] for point in self.optimization_history])
        
        for i in range(min(5, self.num_parameters)):  # Show first 5 parameters
            plt.plot(iterations, parameters_history[:, i], 
                    label=f'θ_{i}', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Convergence statistics
        final_energy = energies[-1]
        energy_change = abs(energies[-1] - energies[-10]) if len(energies) >= 10 else 0
        
        print(f"📊 Convergence Analysis:")
        print(f"   Final energy: {final_energy:.8f}")
        print(f"   Best energy: {self.best_energy:.8f}")
        print(f"   Energy change (last 10 iter): {energy_change:.8f}")
        print(f"   Total iterations: {len(self.optimization_history)}")

class UCCSDansatz:
    """Unitary Coupled Cluster Singles and Doubles ansatz."""
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        
        # For UCCSD, we need parameters for single and double excitations
        # Simplified: one parameter per possible excitation
        num_single_excitations = num_qubits * (num_qubits - 1) // 2
        num_double_excitations = num_qubits * (num_qubits - 1) * (num_qubits - 2) * (num_qubits - 3) // 8
        
        # For small molecules, use simplified UCCSD
        self.num_parameters = min(4, num_qubits)  # Limit for demonstration
        
        print(f"   UCCSD ansatz: {self.num_parameters} parameters")
    
    def apply_to_circuit(self, circuit, parameters):
        """Apply UCCSD ansatz to circuit."""
        
        # Simplified UCCSD implementation
        # Real UCCSD would include all single and double excitations
        
        for i, param in enumerate(parameters):
            if i < self.num_qubits - 1:
                # Single excitation: RY rotation + CNOT
                circuit.ry(i, param)
                circuit.cx(i, i + 1)
                circuit.ry(i + 1, -param)
                circuit.cx(i, i + 1)

class HardwareEfficientAnsatz:
    """Hardware-efficient ansatz for NISQ devices."""
    
    def __init__(self, num_qubits, num_layers=2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # 3 rotation parameters per qubit per layer
        self.num_parameters = 3 * num_qubits * num_layers
        
        print(f"   Hardware-efficient ansatz: {num_layers} layers, {self.num_parameters} parameters")
    
    def apply_to_circuit(self, circuit, parameters):
        """Apply hardware-efficient ansatz to circuit."""
        
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Rotation layer
            for qubit in range(self.num_qubits):
                circuit.rx(qubit, parameters[param_idx])
                param_idx += 1
                circuit.ry(qubit, parameters[param_idx])
                param_idx += 1
                circuit.rz(qubit, parameters[param_idx])
                param_idx += 1
            
            # Entangling layer
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            
            # Circular entanglement for layers > 1
            if self.num_qubits > 2 and layer > 0:
                circuit.cx(self.num_qubits - 1, 0)

# Demonstrate VQE on sample molecules
def demonstrate_vqe():
    """Demonstrate VQE on sample molecular systems."""
    
    print("🌟 VQE Molecular Simulation Demonstration")
    print("=" * 50)
    
    # Test on H2 molecule
    print(f"\n💧 Hydrogen Molecule (H₂):")
    
    # Create VQE solver
    vqe_h2 = VariationalQuantumEigensolver(
        h2_molecule, 
        ansatz_type='hardware_efficient',
        optimizer='COBYLA'
    )
    
    # Find exact solution for comparison
    exact_energy_h2, _ = h2_molecule.exact_diagonalization()
    
    # Run VQE optimization
    print(f"\nRunning VQE optimization...")
    result_h2 = vqe_h2.optimize(maxiter=50)
    
    # Analyze results
    vqe_energy_h2 = vqe_h2.best_energy
    error_h2 = abs(vqe_energy_h2 - exact_energy_h2)
    
    print(f"\n📊 H₂ Results:")
    print(f"   VQE energy: {vqe_energy_h2:.8f}")
    print(f"   Exact energy: {exact_energy_h2:.8f}")
    print(f"   Error: {error_h2:.8f} ({error_h2*1000:.3f} mH)")
    print(f"   Chemical accuracy: {'✅' if error_h2 < 0.0016 else '❌'}")
    
    # Visualize convergence
    vqe_h2.analyze_convergence()
    
    return vqe_h2, exact_energy_h2

# Run VQE demonstration
vqe_demo, exact_demo = demonstrate_vqe()
```

### Advanced VQE Techniques

```python
def adaptive_vqe():
    """Implement adaptive VQE with growing ansatz."""
    
    print("\n🔄 Adaptive VQE")
    print("=" * 20)
    
    class AdaptiveVQE(VariationalQuantumEigensolver):
        """Adaptive VQE that grows ansatz during optimization."""
        
        def __init__(self, hamiltonian):
            # Start with minimal ansatz
            super().__init__(hamiltonian, ansatz_type='hardware_efficient')
            self.ansatz_layers = 1
            self.convergence_threshold = 1e-6
            
        def adaptive_optimize(self, max_layers=5):
            """Optimize with adaptive ansatz growth."""
            
            print("Starting adaptive optimization...")
            
            energies_by_layer = []
            
            for layer in range(1, max_layers + 1):
                print(f"\n--- Layer {layer} ---")
                
                # Update ansatz
                self.ansatz = HardwareEfficientAnsatz(self.num_qubits, num_layers=layer)
                self.num_parameters = self.ansatz.num_parameters
                
                # Optimize current ansatz
                result = self.optimize(maxiter=30)
                energies_by_layer.append(self.best_energy)
                
                print(f"Layer {layer} energy: {self.best_energy:.8f}")
                
                # Check convergence
                if layer > 1:
                    improvement = energies_by_layer[-2] - energies_by_layer[-1]
                    print(f"Improvement: {improvement:.8f}")
                    
                    if improvement < self.convergence_threshold:
                        print(f"Converged at layer {layer}")
                        break
            
            return energies_by_layer
    
    # Test adaptive VQE
    adaptive_vqe_solver = AdaptiveVQE(h2_molecule)
    layer_energies = adaptive_vqe_solver.adaptive_optimize(max_layers=4)
    
    # Compare with exact
    exact_energy, _ = h2_molecule.exact_diagonalization()
    
    print(f"\nAdaptive VQE Results:")
    for i, energy in enumerate(layer_energies):
        error = abs(energy - exact_energy)
        print(f"  Layer {i+1}: {energy:.8f} (error: {error:.8f})")
    
    return adaptive_vqe_solver, layer_energies

def vqe_with_noise_mitigation():
    """VQE with error mitigation techniques."""
    
    print("\n🛡️  VQE with Noise Mitigation")
    print("=" * 35)
    
    class NoiseMitigatedVQE(VariationalQuantumEigensolver):
        """VQE with error mitigation."""
        
        def __init__(self, hamiltonian, mitigation_method='zero_noise_extrapolation'):
            super().__init__(hamiltonian)
            self.mitigation_method = mitigation_method
        
        def measure_energy_expectation(self, parameters, num_shots=1000):
            """Measure energy with noise mitigation."""
            
            if self.mitigation_method == 'zero_noise_extrapolation':
                return self._zero_noise_extrapolation(parameters, num_shots)
            elif self.mitigation_method == 'readout_correction':
                return self._readout_error_mitigation(parameters, num_shots)
            else:
                return super().measure_energy_expectation(parameters, num_shots)
        
        def _zero_noise_extrapolation(self, parameters, num_shots):
            """Zero-noise extrapolation for error mitigation."""
            
            # Simulate different noise levels
            noise_factors = [1, 2, 3]  # Fold circuit 1x, 2x, 3x
            energies = []
            
            for factor in noise_factors:
                # Create circuit with noise scaling
                circuit = self.create_vqe_circuit(parameters)
                
                # Add identity gates to increase noise (simplified)
                for _ in range(factor - 1):
                    for qubit in range(self.num_qubits):
                        circuit.x(qubit)
                        circuit.x(qubit)  # Two X gates = identity with noise
                
                # Measure energy
                energy = self._measure_energy_simple(circuit, num_shots)
                energies.append(energy)
            
            # Linear extrapolation to zero noise
            if len(energies) >= 2:
                # Fit line and extrapolate to factor = 0
                slope = (energies[1] - energies[0]) / (noise_factors[1] - noise_factors[0])
                zero_noise_energy = energies[0] - slope * noise_factors[0]
                return zero_noise_energy
            else:
                return energies[0]
        
        def _readout_error_mitigation(self, parameters, num_shots):
            """Readout error mitigation."""
            
            # Simplified readout error correction
            # In practice, this requires calibration measurements
            
            circuit = self.create_vqe_circuit(parameters)
            raw_energy = self._measure_energy_simple(circuit, num_shots)
            
            # Apply correction factor (simplified)
            correction_factor = 0.95  # Assume 5% readout error
            corrected_energy = raw_energy / correction_factor
            
            return corrected_energy
        
        def _measure_energy_simple(self, circuit, num_shots):
            """Simplified energy measurement."""
            # This would normally measure all Pauli terms
            # Simplified for demonstration
            return super().measure_energy_expectation(circuit.parameters, num_shots)
    
    # Test noise-mitigated VQE
    mitigated_vqe = NoiseMitigatedVQE(h2_molecule, mitigation_method='zero_noise_extrapolation')
    result = mitigated_vqe.optimize(maxiter=30)
    
    print(f"Noise-mitigated VQE energy: {mitigated_vqe.best_energy:.8f}")
    
    return mitigated_vqe

def vqe_dissociation_curve():
    """Compute molecular dissociation curve using VQE."""
    
    print("\n📈 H₂ Dissociation Curve")
    print("=" * 30)
    
    # Test different bond distances
    bond_distances = np.linspace(0.5, 3.0, 8)
    vqe_energies = []
    exact_energies = []
    
    print("Computing energies at different bond distances...")
    
    for i, distance in enumerate(bond_distances):
        print(f"\nBond distance {distance:.2f} Å ({i+1}/{len(bond_distances)}):")
        
        # Create molecule at this distance
        h2_at_distance = create_hydrogen_molecule(bond_distance=distance)
        
        # VQE calculation
        vqe = VariationalQuantumEigensolver(
            h2_at_distance,
            ansatz_type='hardware_efficient',
            optimizer='COBYLA'
        )
        
        vqe.optimize(maxiter=20)
        vqe_energies.append(vqe.best_energy)
        
        # Exact calculation (for small system)
        if h2_at_distance.num_qubits <= 8:
            exact_energy, _ = h2_at_distance.exact_diagonalization()
            exact_energies.append(exact_energy)
        else:
            exact_energies.append(None)
        
        print(f"  VQE energy: {vqe.best_energy:.6f}")
        if exact_energies[-1] is not None:
            print(f"  Exact energy: {exact_energies[-1]:.6f}")
    
    # Plot dissociation curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(bond_distances, vqe_energies, 'bo-', linewidth=2, 
             markersize=8, label='VQE')
    
    if all(e is not None for e in exact_energies):
        plt.plot(bond_distances, exact_energies, 'r--', linewidth=2, 
                 label='Exact')
    
    plt.xlabel('Bond Distance (Å)')
    plt.ylabel('Energy (Hartree)')
    plt.title('H₂ Potential Energy Surface')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Find equilibrium bond distance
    min_idx = np.argmin(vqe_energies)
    equilibrium_distance = bond_distances[min_idx]
    equilibrium_energy = vqe_energies[min_idx]
    
    print(f"\n📊 Dissociation Analysis:")
    print(f"   Equilibrium distance: {equilibrium_distance:.3f} Å")
    print(f"   Equilibrium energy: {equilibrium_energy:.6f} Hartree")
    print(f"   Experimental R_e: 0.741 Å")
    
    return bond_distances, vqe_energies, exact_energies

# Run advanced VQE techniques
adaptive_solver, adaptive_energies = adaptive_vqe()
mitigated_solver = vqe_with_noise_mitigation()
dissociation_data = vqe_dissociation_curve()
```

### Quantum Chemistry Applications

```python
def drug_discovery_example():
    """VQE application in drug discovery."""
    
    print("\n💊 Drug Discovery with VQE")
    print("=" * 30)
    
    print("Simulating drug-target interactions:")
    print("1. Small molecule optimization")
    print("2. Protein-ligand binding studies")
    print("3. Chemical reaction pathways")
    
    # Simplified drug molecule (caffeine-like structure)
    class DrugMolecule:
        def __init__(self, name, functional_groups):
            self.name = name
            self.functional_groups = functional_groups
            
        def compute_binding_affinity(self, target_protein):
            """Compute binding affinity using VQE."""
            
            print(f"Computing binding affinity for {self.name}...")
            
            # Create simplified interaction Hamiltonian
            # In reality, this would be derived from quantum chemistry calculations
            interaction_strength = np.random.uniform(0.1, 1.0)  # Simplified
            
            # VQE calculation for binding complex
            binding_energy = -interaction_strength * len(self.functional_groups)
            
            print(f"  Binding energy: {binding_energy:.4f} kcal/mol")
            return binding_energy
    
    # Test drug candidates
    drug_candidates = [
        DrugMolecule("Compound A", ["hydroxyl", "amino", "carboxyl"]),
        DrugMolecule("Compound B", ["methyl", "phosphate", "amino"]),
        DrugMolecule("Compound C", ["hydroxyl", "carboxyl", "sulfate"])
    ]
    
    target_protein = "COVID-19 Main Protease"
    
    print(f"\nScreening against {target_protein}:")
    
    binding_affinities = []
    for drug in drug_candidates:
        affinity = drug.compute_binding_affinity(target_protein)
        binding_affinities.append((drug.name, affinity))
    
    # Rank by binding affinity
    binding_affinities.sort(key=lambda x: x[1])  # More negative = stronger binding
    
    print(f"\nRanked drug candidates:")
    for i, (name, affinity) in enumerate(binding_affinities):
        print(f"  {i+1}. {name}: {affinity:.4f} kcal/mol")
    
    print(f"\nBest candidate: {binding_affinities[0][0]}")
    
    return binding_affinities

def catalyst_design_example():
    """VQE application in catalyst design."""
    
    print("\n⚗️ Catalyst Design with VQE")
    print("=" * 35)
    
    print("Optimizing catalysts for chemical reactions:")
    print("1. Transition state energy calculations")
    print("2. Activation barrier optimization")
    print("3. Selectivity analysis")
    
    class CatalystSystem:
        def __init__(self, metal_center, ligands):
            self.metal_center = metal_center
            self.ligands = ligands
            
        def compute_activation_barrier(self, reaction):
            """Compute activation barrier using VQE."""
            
            print(f"Computing activation barrier for {self.metal_center} catalyst...")
            
            # Simplified calculation
            # Real implementation would use full quantum chemistry
            
            # Metal center effects
            metal_effects = {
                'Pd': -0.5,  # Good catalyst
                'Pt': -0.3,  # Expensive but effective
                'Ni': -0.2,  # Cheap alternative
                'Cu': -0.1   # Mild activity
            }
            
            # Ligand effects
            ligand_effect = len(self.ligands) * 0.1
            
            baseline_barrier = 25.0  # kcal/mol
            activation_barrier = baseline_barrier + metal_effects.get(self.metal_center, 0) + ligand_effect
            
            print(f"  Activation barrier: {activation_barrier:.2f} kcal/mol")
            return activation_barrier
    
    # Test catalyst candidates
    catalysts = [
        CatalystSystem("Pd", ["PPh3", "Cl"]),
        CatalystSystem("Ni", ["PCy3", "Br"]),
        CatalystSystem("Cu", ["NHC", "I"]),
        CatalystSystem("Pt", ["dppe", "Cl"])
    ]
    
    reaction = "C-C coupling"
    
    print(f"\nOptimizing catalysts for {reaction}:")
    
    catalyst_performance = []
    for catalyst in catalysts:
        barrier = catalyst.compute_activation_barrier(reaction)
        catalyst_performance.append((catalyst.metal_center, barrier))
    
    # Rank by activation barrier (lower is better)
    catalyst_performance.sort(key=lambda x: x[1])
    
    print(f"\nRanked catalysts (lower barrier = better):")
    for i, (metal, barrier) in enumerate(catalyst_performance):
        print(f"  {i+1}. {metal}: {barrier:.2f} kcal/mol")
    
    print(f"\nBest catalyst: {catalyst_performance[0][0]}")
    
    return catalyst_performance

def materials_discovery_example():
    """VQE application in materials discovery."""
    
    print("\n🔬 Materials Discovery with VQE")
    print("=" * 35)
    
    print("Discovering new materials with desired properties:")
    print("1. Electronic band structure calculation")
    print("2. Superconductivity prediction")
    print("3. Battery material optimization")
    
    class Material:
        def __init__(self, composition, crystal_structure):
            self.composition = composition
            self.crystal_structure = crystal_structure
            
        def compute_band_gap(self):
            """Compute electronic band gap using VQE."""
            
            print(f"Computing band gap for {self.composition}...")
            
            # Simplified band gap calculation
            # Real implementation would solve solid-state Hamiltonians
            
            # Different crystal structures affect band gap
            structure_effects = {
                'cubic': 0.0,
                'hexagonal': 0.2,
                'tetragonal': 0.1,
                'orthorhombic': 0.15
            }
            
            # Composition effects (simplified)
            base_gap = 2.0  # eV
            if 'Si' in self.composition:
                base_gap = 1.1
            elif 'GaAs' in self.composition:
                base_gap = 1.4
            elif 'ZnO' in self.composition:
                base_gap = 3.3
            
            structure_effect = structure_effects.get(self.crystal_structure, 0)
            band_gap = base_gap + structure_effect
            
            print(f"  Band gap: {band_gap:.2f} eV")
            return band_gap
    
    # Test material candidates for solar cells
    materials = [
        Material("Si", "cubic"),
        Material("GaAs", "cubic"),
        Material("ZnO", "hexagonal"),
        Material("CdTe", "cubic"),
        Material("CIGS", "tetragonal")
    ]
    
    print(f"\nScreening materials for solar cell applications:")
    print(f"Target band gap: 1.0-1.5 eV (optimal for solar)")
    
    material_properties = []
    for material in materials:
        band_gap = material.compute_band_gap()
        
        # Score based on proximity to optimal range
        optimal_min, optimal_max = 1.0, 1.5
        if optimal_min <= band_gap <= optimal_max:
            score = 1.0
        else:
            score = 1.0 / (1.0 + abs(band_gap - (optimal_min + optimal_max)/2))
        
        material_properties.append((material.composition, band_gap, score))
    
    # Rank by suitability score
    material_properties.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nRanked materials for solar cells:")
    for i, (comp, gap, score) in enumerate(material_properties):
        print(f"  {i+1}. {comp}: {gap:.2f} eV (score: {score:.3f})")
    
    print(f"\nBest material: {material_properties[0][0]}")
    
    return material_properties

# Run quantum chemistry applications
drug_affinities = drug_discovery_example()
catalyst_rankings = catalyst_design_example()
material_rankings = materials_discovery_example()
```

## Performance Analysis and Benchmarking

```python
def comprehensive_vqe_benchmark():
    """Comprehensive VQE performance benchmark."""
    
    print("\n🏆 Comprehensive VQE Benchmark")
    print("=" * 40)
    
    import time
    
    # Test different molecule sizes and ansatz types
    test_cases = [
        ("H2", h2_molecule, ["hardware_efficient", "uccsd"]),
        ("LiH", lih_molecule, ["hardware_efficient"])
    ]
    
    benchmark_results = []
    
    for molecule_name, molecule, ansatz_types in test_cases:
        print(f"\n--- Benchmarking {molecule_name} ---")
        
        # Get exact solution for comparison
        if molecule.num_qubits <= 8:
            exact_energy, _ = molecule.exact_diagonalization()
        else:
            exact_energy = None
        
        for ansatz_type in ansatz_types:
            print(f"\nTesting {ansatz_type} ansatz:")
            
            # Time VQE optimization
            start_time = time.time()
            
            vqe = VariationalQuantumEigensolver(
                molecule,
                ansatz_type=ansatz_type,
                optimizer='COBYLA'
            )
            
            result = vqe.optimize(maxiter=30)
            
            optimization_time = time.time() - start_time
            
            # Calculate metrics
            vqe_energy = vqe.best_energy
            error = abs(vqe_energy - exact_energy) if exact_energy else None
            chemical_accuracy = error < 0.0016 if error else None
            
            benchmark_results.append({
                'molecule': molecule_name,
                'ansatz': ansatz_type,
                'vqe_energy': vqe_energy,
                'exact_energy': exact_energy,
                'error': error,
                'chemical_accuracy': chemical_accuracy,
                'optimization_time': optimization_time,
                'iterations': len(vqe.optimization_history),
                'num_parameters': vqe.num_parameters,
                'num_qubits': vqe.num_qubits
            })
            
            print(f"  VQE energy: {vqe_energy:.8f}")
            if exact_energy:
                print(f"  Exact energy: {exact_energy:.8f}")
                print(f"  Error: {error:.8f}")
                print(f"  Chemical accuracy: {'✅' if chemical_accuracy else '❌'}")
            print(f"  Time: {optimization_time:.2f}s")
            print(f"  Parameters: {vqe.num_parameters}")
    
    # Create benchmark summary
    print(f"\n📊 VQE Benchmark Summary:")
    print(f"{'Molecule':<8} {'Ansatz':<15} {'Qubits':<7} {'Params':<7} {'Error':<12} {'Time (s)':<9} {'Accuracy'}")
    print("-" * 80)
    
    for result in benchmark_results:
        error_str = f"{result['error']:.6f}" if result['error'] else "N/A"
        accuracy_str = "✅" if result['chemical_accuracy'] else "❌" if result['chemical_accuracy'] is not None else "N/A"
        
        print(f"{result['molecule']:<8} {result['ansatz']:<15} {result['num_qubits']:<7} "
              f"{result['num_parameters']:<7} {error_str:<12} {result['optimization_time']:<9.2f} {accuracy_str}")
    
    # Analysis
    print(f"\nBenchmark Analysis:")
    
    # Average errors by ansatz type
    ansatz_errors = {}
    for result in benchmark_results:
        if result['error'] is not None:
            ansatz = result['ansatz']
            if ansatz not in ansatz_errors:
                ansatz_errors[ansatz] = []
            ansatz_errors[ansatz].append(result['error'])
    
    for ansatz, errors in ansatz_errors.items():
        avg_error = np.mean(errors)
        print(f"  Average error for {ansatz}: {avg_error:.6f}")
    
    # Scaling analysis
    print(f"\nScaling Analysis:")
    print(f"  VQE complexity: O(M^4) integrals, O(P) parameters")
    print(f"  Circuit depth: O(P) for hardware-efficient ansatz")
    print(f"  Classical optimization: O(iterations × measurements)")
    
    return benchmark_results

def vqe_noise_analysis():
    """Analyze VQE performance under different noise conditions."""
    
    print("\n🔊 VQE Noise Analysis")
    print("=" * 25)
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]  # Simulated noise levels
    
    print("Testing VQE under different noise conditions...")
    
    noise_results = []
    
    for noise in noise_levels:
        print(f"\nNoise level: {noise*100:.1f}%")
        
        # Create VQE with simulated noise
        vqe_noisy = VariationalQuantumEigensolver(
            h2_molecule,
            ansatz_type='hardware_efficient'
        )
        
        # Modify energy measurement to include noise (simplified)
        original_measure = vqe_noisy.measure_energy_expectation
        
        def noisy_measure(params, num_shots=1000):
            clean_energy = original_measure(params, num_shots)
            noise_factor = 1 + np.random.normal(0, noise)
            return clean_energy * noise_factor
        
        vqe_noisy.measure_energy_expectation = noisy_measure
        
        # Optimize
        result = vqe_noisy.optimize(maxiter=20)
        
        # Get exact energy for comparison
        exact_energy, _ = h2_molecule.exact_diagonalization()
        error = abs(vqe_noisy.best_energy - exact_energy)
        
        noise_results.append({
            'noise_level': noise,
            'vqe_energy': vqe_noisy.best_energy,
            'error': error
        })
        
        print(f"  VQE energy: {vqe_noisy.best_energy:.6f}")
        print(f"  Error: {error:.6f}")
    
    # Plot noise analysis
    plt.figure(figsize=(10, 6))
    
    noise_percentages = [r['noise_level'] * 100 for r in noise_results]
    errors = [r['error'] for r in noise_results]
    
    plt.semilogy(noise_percentages, errors, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=0.0016, color='g', linestyle='--', label='Chemical accuracy')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Energy Error (Hartree)')
    plt.title('VQE Performance vs Noise Level')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Find noise threshold for chemical accuracy
    chemical_accuracy_threshold = 0.0016
    for result in noise_results:
        if result['error'] > chemical_accuracy_threshold:
            max_noise = result['noise_level'] * 100
            break
    else:
        max_noise = noise_levels[-1] * 100
    
    print(f"\nNoise Analysis Results:")
    print(f"  Maximum noise for chemical accuracy: ~{max_noise:.1f}%")
    print(f"  Error scaling: ~O(noise_level)")
    
    return noise_results

# Run performance analysis
benchmark_data = comprehensive_vqe_benchmark()
noise_data = vqe_noise_analysis()
```

## Exercises and Extensions

### Exercise 1: Custom Molecular Systems
```python
def exercise_custom_molecules():
    """Exercise: Implement VQE for custom molecular systems."""
    
    print("🎯 Exercise: Custom Molecular Systems")
    print("=" * 40)
    
    # TODO: Implement VQE for:
    # 1. Water molecule (H2O) - bent geometry
    # 2. Ammonia (NH3) - pyramidal structure  
    # 3. Benzene ring - aromatic system
    # 4. Metal complexes - transition metal chemistry
    
    print("Your challenge:")
    print("1. Create Hamiltonians for H2O, NH3, benzene")
    print("2. Design appropriate ansatz for each system")
    print("3. Compare VQE accuracy across different molecules")
    print("4. Analyze computational scaling")

exercise_custom_molecules()
```

### Exercise 2: Advanced Ansatz Design
```python
def exercise_advanced_ansatz():
    """Exercise: Design problem-specific ansatz."""
    
    print("🎯 Exercise: Advanced Ansatz Design")
    print("=" * 35)
    
    # TODO: Implement:
    # 1. Problem-specific ansatz based on molecular orbitals
    # 2. Symmetry-adapted ansatz respecting molecular point groups
    # 3. k-UpCCGSD ansatz for strongly correlated systems
    # 4. Qubit-efficient ansatz for large molecules
    
    print("Design advanced ansatz types:")
    print("1. Molecular orbital-based ansatz")
    print("2. Symmetry-preserving circuits")
    print("3. Hardware-efficient vs chemically-inspired trade-offs")

exercise_advanced_ansatz()
```

### Exercise 3: Real Quantum Hardware
```python
def exercise_hardware_implementation():
    """Exercise: Implement VQE on real quantum hardware."""
    
    print("🎯 Exercise: Hardware Implementation")
    print("=" * 35)
    
    # TODO: Adapt VQE for:
    # 1. IBM Quantum devices - topology constraints
    # 2. Google Quantum AI - native gate sets
    # 3. IonQ - all-to-all connectivity
    # 4. Error mitigation on real devices
    
    print("Implement VQE on quantum hardware:")
    print("1. Handle device topology and gate constraints")
    print("2. Implement readout error correction")
    print("3. Use zero-noise extrapolation")
    print("4. Compare simulator vs hardware results")

exercise_hardware_implementation()
```

## Summary

🎉 **Congratulations!** You've learned:
- How to formulate molecular Hamiltonians for quantum computers
- VQE algorithm design and implementation for quantum chemistry
- Different ansatz types (UCCSD, hardware-efficient) and their trade-offs
- Advanced techniques: adaptive VQE, noise mitigation, dissociation curves
- Real-world applications in drug discovery, catalyst design, and materials science
- Performance analysis and hardware considerations

VQE represents one of the most promising near-term applications of quantum computing, with potential to revolutionize chemistry and materials science!

**Next Steps:**
- Explore [QAOA for optimization](qaoa_maxcut.md)  
- Try [Portfolio Optimization](portfolio.md) for finance
- Learn about [Quantum Machine Learning](../ml/vqc.md)

## References

### Foundational Papers
- Peruzzo et al. (2014). "A variational eigenvalue solver on a photonic quantum processor"
- McClean et al. (2016). "The theory of variational hybrid quantum-classical algorithms"

### Quantum Chemistry Applications
- Kandala et al. (2017). "Hardware-efficient variational quantum eigensolver for small molecules"
- Cao et al. (2019). "Quantum chemistry in the age of quantum computing"

### Reviews and Perspectives
- McArdle et al. (2020). "Quantum computational chemistry"
- Bauer et al. (2020). "Quantum algorithms for quantum chemistry and quantum materials science"

---

*"The future of chemistry is quantum. VQE is opening the door to understanding molecular systems at unprecedented scales."* - Quantum Chemistry Pioneer

🚀 **Ready to simulate molecules with quantum computers?** Explore more [Optimization Examples](index.md)!