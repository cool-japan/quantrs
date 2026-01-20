# Interactive Quantum Algorithms Workshop

Welcome to the intermediate quantum algorithms workshop! This comprehensive tutorial explores famous quantum algorithms with hands-on implementations and interactive visualizations.

## Workshop Overview

- **Duration**: 2-3 hours
- **Prerequisites**: Basic quantum computing knowledge, completed beginner tutorial
- **Learning Goals**: Implement and understand Grover's, QFT, VQE, and QAOA algorithms

## Setup and Imports

```python
# Install requirements
!pip install quantrs2 matplotlib numpy scipy jupyter-widgets plotly

# Core imports
import quantrs2
from quantrs2 import Circuit
from quantrs2.gates import H, X, Y, Z, CNOT, RY, RZ
from quantrs2.algorithms import grovers_search, quantum_fourier_transform
from quantrs2.visualization import visualize_circuit, plot_algorithm_comparison
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

## Algorithm 1: Grover's Quantum Search 🔍

### Understanding the Problem

Grover's algorithm searches an unsorted database quadratically faster than classical algorithms.

```python
class GroverSimulator:
    """Interactive Grover's algorithm simulator."""
    
    def __init__(self, n_qubits, marked_item):
        self.n_qubits = n_qubits
        self.n_items = 2**n_qubits
        self.marked_item = marked_item
        self.optimal_iterations = int(np.pi/4 * np.sqrt(self.n_items))
        
    def create_oracle(self, circuit, marked_state):
        """Create oracle that marks the target state."""
        # Convert marked state to binary
        binary_state = format(marked_state, f'0{self.n_qubits}b')
        
        # Flip qubits that should be 0 in target state
        for i, bit in enumerate(binary_state):
            if bit == '0':
                circuit.x(i)
        
        # Multi-controlled Z gate
        if self.n_qubits == 1:
            circuit.z(0)
        elif self.n_qubits == 2:
            circuit.cz(0, 1)
        else:
            # For more qubits, use multi-controlled Z
            circuit.multi_controlled_z(list(range(self.n_qubits)))
        
        # Flip back
        for i, bit in enumerate(binary_state):
            if bit == '0':
                circuit.x(i)
    
    def create_diffusion_operator(self, circuit):
        """Create the diffusion operator (inversion about average)."""
        # Apply H to all qubits
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Apply X to all qubits
        for i in range(self.n_qubits):
            circuit.x(i)
        
        # Multi-controlled Z
        if self.n_qubits == 1:
            circuit.z(0)
        elif self.n_qubits == 2:
            circuit.cz(0, 1)
        else:
            circuit.multi_controlled_z(list(range(self.n_qubits)))
        
        # Apply X to all qubits
        for i in range(self.n_qubits):
            circuit.x(i)
        
        # Apply H to all qubits
        for i in range(self.n_qubits):
            circuit.h(i)
    
    def run_grover(self, iterations):
        """Run Grover's algorithm for specified iterations."""
        circuit = Circuit(self.n_qubits)
        
        # Initialize superposition
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle
            self.create_oracle(circuit, self.marked_item)
            # Diffusion operator
            self.create_diffusion_operator(circuit)
        
        # Get final state
        result = circuit.run()
        return result.state_vector, circuit

# Interactive Grover's demonstration
@interact(
    n_qubits=widgets.Dropdown(options=[2, 3, 4], value=3, description='Qubits:'),
    marked_item=widgets.IntSlider(min=0, max=7, value=5, description='Target:'),
    iterations=widgets.IntSlider(min=0, max=10, value=2, description='Iterations:')
)
def interactive_grover(n_qubits, marked_item, iterations):
    """Interactive Grover's algorithm explorer."""
    
    # Adjust marked_item range based on n_qubits
    max_item = 2**n_qubits - 1
    if marked_item > max_item:
        marked_item = max_item
    
    # Create simulator
    grover = GroverSimulator(n_qubits, marked_item)
    
    # Run algorithm
    state_vector, circuit = grover.run_grover(iterations)
    
    # Calculate probabilities
    probabilities = np.abs(state_vector)**2
    
    # Display results
    print(f"🎯 Searching for item {marked_item} in database of {2**n_qubits} items")
    print(f"📊 Optimal iterations: {grover.optimal_iterations}")
    print(f"🔄 Current iterations: {iterations}")
    print(f"🎲 Success probability: {probabilities[marked_item]:.3f}")
    
    # Visualize probabilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Probability distribution
    items = list(range(2**n_qubits))
    colors = ['red' if i == marked_item else 'blue' for i in items]
    ax1.bar(items, probabilities, color=colors, alpha=0.7)
    ax1.set_xlabel('Database Item')
    ax1.set_ylabel('Probability')
    ax1.set_title('Search Probabilities')
    ax1.axhline(y=1/2**n_qubits, color='gray', linestyle='--', 
                label='Classical random guess')
    ax1.legend()
    
    # Success probability vs iterations
    iterations_range = range(0, 2*grover.optimal_iterations + 1)
    success_probs = []
    
    for iter_count in iterations_range:
        test_state, _ = grover.run_grover(iter_count)
        success_prob = np.abs(test_state[marked_item])**2
        success_probs.append(success_prob)
    
    ax2.plot(iterations_range, success_probs, 'b-', linewidth=2)
    ax2.axvline(x=grover.optimal_iterations, color='red', linestyle='--',
                label=f'Optimal: {grover.optimal_iterations}')
    ax2.axvline(x=iterations, color='green', linestyle='-',
                label=f'Current: {iterations}')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Success Probability')
    ax2.set_title('Success Probability vs Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return circuit, state_vector
```

### Try It Yourself! 🎯

**Challenge 1**: Grover's Optimization
1. For 3 qubits, what's the optimal number of iterations?
2. What happens if you use too many iterations?
3. How does the success probability change with database size?

## Algorithm 2: Quantum Fourier Transform 🌊

The QFT is the quantum analog of the discrete Fourier transform, essential for many quantum algorithms.

```python
class QFTVisualizer:
    """Interactive QFT demonstration."""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    
    def create_qft_circuit(self):
        """Create QFT circuit."""
        circuit = Circuit(self.n_qubits)
        
        for i in range(self.n_qubits):
            # Apply Hadamard gate
            circuit.h(i)
            
            # Apply controlled rotation gates
            for j in range(i + 1, self.n_qubits):
                angle = 2 * np.pi / (2 ** (j - i + 1))
                circuit.controlled_rz(j, i, angle)
        
        # Reverse the order of qubits (SWAP gates)
        for i in range(self.n_qubits // 2):
            circuit.swap(i, self.n_qubits - 1 - i)
        
        return circuit
    
    def apply_qft_to_state(self, input_state):
        """Apply QFT to an input state."""
        circuit = Circuit(self.n_qubits)
        
        # Prepare input state
        circuit.initialize_state(input_state)
        
        # Apply QFT
        qft_circuit = self.create_qft_circuit()
        circuit.append(qft_circuit)
        
        return circuit.run().state_vector

# Interactive QFT demonstration
@interact(
    n_qubits=widgets.Dropdown(options=[2, 3, 4], value=3, description='Qubits:'),
    input_pattern=widgets.Dropdown(
        options=[
            ('Computational basis |000⟩', 'basis_0'),
            ('Computational basis |001⟩', 'basis_1'),
            ('Superposition |+++⟩', 'superposition'),
            ('Custom pattern', 'custom')
        ],
        value='basis_0',
        description='Input:'
    )
)
def interactive_qft(n_qubits, input_pattern):
    """Interactive QFT explorer."""
    
    # Create QFT visualizer
    qft_viz = QFTVisualizer(n_qubits)
    
    # Prepare input state
    if input_pattern == 'basis_0':
        input_state = np.zeros(2**n_qubits)
        input_state[0] = 1.0
        state_name = "|000...⟩"
    elif input_pattern == 'basis_1':
        input_state = np.zeros(2**n_qubits)
        input_state[1] = 1.0
        state_name = "|001...⟩"
    elif input_pattern == 'superposition':
        input_state = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
        state_name = "|+++...⟩"
    else:
        # Custom pattern - alternating
        input_state = np.zeros(2**n_qubits)
        for i in range(0, 2**n_qubits, 2):
            input_state[i] = 1.0
        input_state = input_state / np.linalg.norm(input_state)
        state_name = "Custom"
    
    # Apply QFT
    output_state = qft_viz.apply_qft_to_state(input_state)
    
    # Visualize transformation
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Input Amplitudes', 'Output Amplitudes',
                       'Input Phases', 'Output Phases'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Input and output amplitudes
    x_labels = [f"|{format(i, f'0{n_qubits}b')}⟩" for i in range(2**n_qubits)]
    
    fig.add_trace(go.Bar(x=x_labels, y=np.abs(input_state), 
                        name='Input', marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Bar(x=x_labels, y=np.abs(output_state), 
                        name='Output', marker_color='red'), row=1, col=2)
    
    # Phases
    input_phases = np.angle(input_state)
    output_phases = np.angle(output_state)
    
    fig.add_trace(go.Scatter(x=x_labels, y=input_phases, 
                            mode='markers+lines', name='Input Phase',
                            marker_color='blue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=output_phases, 
                            mode='markers+lines', name='Output Phase',
                            marker_color='red'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False,
                     title_text=f"QFT Transformation: {state_name}")
    fig.show()
    
    # Display key insights
    print(f"🌊 Quantum Fourier Transform Applied to {state_name}")
    print(f"📊 Input state has {np.count_nonzero(input_state)} non-zero amplitudes")
    print(f"📊 Output state has {np.count_nonzero(output_state)} non-zero amplitudes")
    
    # Frequency analysis
    dominant_frequencies = np.argsort(np.abs(output_state))[-3:][::-1]
    print(f"🎵 Dominant frequency components:")
    for i, freq in enumerate(dominant_frequencies):
        if np.abs(output_state[freq]) > 1e-10:
            print(f"  {i+1}. Frequency {freq}: amplitude {np.abs(output_state[freq]):.3f}")
    
    return input_state, output_state
```

## Algorithm 3: Variational Quantum Eigensolver (VQE) ⚛️

VQE is a hybrid quantum-classical algorithm for finding ground state energies.

```python
class VQESimulator:
    """Interactive VQE algorithm demonstration."""
    
    def __init__(self, n_qubits, hamiltonian_type='h2'):
        self.n_qubits = n_qubits
        self.hamiltonian = self._create_hamiltonian(hamiltonian_type)
        self.optimization_history = []
    
    def _create_hamiltonian(self, hamiltonian_type):
        """Create example Hamiltonians."""
        if hamiltonian_type == 'h2':
            # Simplified H2 molecule Hamiltonian
            return {
                'ZZ': -1.0523732,
                'Z0': 0.39793742,
                'Z1': -0.39793742,
                'XX': -0.01128010,
                'YY': 0.01128010
            }
        elif hamiltonian_type == 'ising':
            # Ising model Hamiltonian
            return {
                'ZZ': -1.0,
                'Z0': -0.5,
                'Z1': -0.5
            }
        else:
            # Simple Pauli Z
            return {'Z0': -1.0}
    
    def create_ansatz(self, parameters):
        """Create parameterized ansatz circuit."""
        circuit = Circuit(self.n_qubits)
        
        # Layer 1: RY rotations
        for i in range(self.n_qubits):
            circuit.ry(i, parameters[i])
        
        # Layer 2: Entangling gates
        for i in range(self.n_qubits - 1):
            circuit.cnot(i, i + 1)
        
        # Layer 3: More RY rotations
        for i in range(self.n_qubits):
            if len(parameters) > self.n_qubits + i:
                circuit.ry(i, parameters[self.n_qubits + i])
        
        return circuit
    
    def compute_expectation_value(self, parameters):
        """Compute expectation value of Hamiltonian."""
        circuit = self.create_ansatz(parameters)
        state = circuit.run().state_vector
        
        # Compute expectation value for each Hamiltonian term
        expectation = 0.0
        
        for pauli_string, coefficient in self.hamiltonian.items():
            # Apply Pauli operators and compute expectation
            if pauli_string == 'ZZ':
                # Z⊗Z measurement
                z_z_expectation = self._compute_zz_expectation(state)
                expectation += coefficient * z_z_expectation
            elif pauli_string == 'XX':
                # X⊗X measurement
                x_x_expectation = self._compute_xx_expectation(state)
                expectation += coefficient * x_x_expectation
            elif pauli_string == 'YY':
                # Y⊗Y measurement
                y_y_expectation = self._compute_yy_expectation(state)
                expectation += coefficient * y_y_expectation
            elif pauli_string.startswith('Z'):
                # Single Z measurement
                qubit = int(pauli_string[1])
                z_expectation = self._compute_z_expectation(state, qubit)
                expectation += coefficient * z_expectation
        
        return expectation
    
    def _compute_zz_expectation(self, state):
        """Compute ⟨Z⊗Z⟩ expectation value."""
        # For 2-qubit system: |00⟩ and |11⟩ have +1, |01⟩ and |10⟩ have -1
        return (np.abs(state[0])**2 - np.abs(state[1])**2 - 
                np.abs(state[2])**2 + np.abs(state[3])**2)
    
    def _compute_xx_expectation(self, state):
        """Compute ⟨X⊗X⟩ expectation value."""
        # Need to transform to X basis and compute
        # Simplified calculation for demonstration
        return np.real(state[0]*np.conj(state[3]) + state[1]*np.conj(state[2]) +
                      state[2]*np.conj(state[1]) + state[3]*np.conj(state[0]))
    
    def _compute_yy_expectation(self, state):
        """Compute ⟨Y⊗Y⟩ expectation value."""
        # Simplified calculation for demonstration
        return np.real(-1j*state[0]*np.conj(state[3]) + 1j*state[1]*np.conj(state[2]) +
                      1j*state[2]*np.conj(state[1]) - 1j*state[3]*np.conj(state[0]))
    
    def _compute_z_expectation(self, state, qubit):
        """Compute ⟨Z⟩ expectation value for single qubit."""
        if qubit == 0:
            return (np.abs(state[0])**2 + np.abs(state[1])**2 - 
                   np.abs(state[2])**2 - np.abs(state[3])**2)
        else:
            return (np.abs(state[0])**2 - np.abs(state[1])**2 + 
                   np.abs(state[2])**2 - np.abs(state[3])**2)
    
    def optimize(self, initial_parameters, method='COBYLA'):
        """Run VQE optimization."""
        self.optimization_history = []
        
        def cost_function(params):
            energy = self.compute_expectation_value(params)
            self.optimization_history.append(energy)
            return energy
        
        result = minimize(cost_function, initial_parameters, method=method)
        return result

# Interactive VQE demonstration
@interact(
    hamiltonian_type=widgets.Dropdown(
        options=['h2', 'ising', 'simple'],
        value='h2',
        description='Hamiltonian:'
    ),
    n_parameters=widgets.IntSlider(min=2, max=8, value=4, description='Parameters:'),
    optimizer=widgets.Dropdown(
        options=['COBYLA', 'BFGS', 'Powell'],
        value='COBYLA',
        description='Optimizer:'
    )
)
def interactive_vqe(hamiltonian_type, n_parameters, optimizer):
    """Interactive VQE demonstration."""
    
    # Create VQE simulator
    vqe = VQESimulator(2, hamiltonian_type)
    
    print(f"🧪 VQE Simulation: {hamiltonian_type.upper()} Hamiltonian")
    print(f"⚙️ Parameters: {n_parameters}, Optimizer: {optimizer}")
    
    # Random initial parameters
    np.random.seed(42)  # For reproducibility
    initial_params = np.random.uniform(0, 2*np.pi, n_parameters)
    
    print(f"🎲 Initial parameters: {initial_params[:4]}")  # Show first 4
    
    # Run optimization
    result = vqe.optimize(initial_params, method=optimizer)
    
    print(f"✅ Optimization completed!")
    print(f"🎯 Ground state energy: {result.fun:.6f}")
    print(f"🔄 Iterations: {len(vqe.optimization_history)}")
    
    # Plot optimization progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Energy convergence
    ax1.plot(vqe.optimization_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('VQE Energy Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Final ansatz circuit visualization
    final_circuit = vqe.create_ansatz(result.x)
    # Note: This would show circuit diagram in real implementation
    ax2.text(0.5, 0.5, f'Final Circuit\n{n_parameters} parameters\nDepth: {len(result.x)//2 + 1}',
             ha='center', va='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax2.set_title('Optimized Ansatz Circuit')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Parameter analysis
    print(f"📊 Optimized parameters:")
    for i, param in enumerate(result.x):
        print(f"  θ_{i}: {param:.3f}")
    
    return result, vqe

# Run VQE demonstration
vqe_result = interactive_vqe('h2', 4, 'COBYLA')
```

## Algorithm 4: Quantum Approximate Optimization Algorithm (QAOA) 🎯

QAOA solves combinatorial optimization problems using quantum computers.

```python
class QAOASimulator:
    """Interactive QAOA demonstration for MaxCut problem."""
    
    def __init__(self, graph_edges):
        self.edges = graph_edges
        self.n_qubits = max(max(edge) for edge in graph_edges) + 1
    
    def create_cost_hamiltonian(self):
        """Create cost Hamiltonian for MaxCut."""
        # For MaxCut: H_C = Σ (1 - Z_i Z_j) / 2 for each edge (i,j)
        hamiltonian = {}
        for i, j in self.edges:
            edge_key = f"ZZ_{i}_{j}"
            hamiltonian[edge_key] = -0.5  # Coefficient for ZiZj term
        return hamiltonian
    
    def create_qaoa_circuit(self, gamma, beta):
        """Create QAOA circuit with given parameters."""
        circuit = Circuit(self.n_qubits)
        
        # Initial superposition
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Apply cost unitary exp(-i*gamma*H_C)
        for i, j in self.edges:
            circuit.cnot(i, j)
            circuit.rz(j, 2 * gamma)
            circuit.cnot(i, j)
        
        # Apply mixer unitary exp(-i*beta*H_M)
        for i in range(self.n_qubits):
            circuit.rx(i, 2 * beta)
        
        return circuit
    
    def compute_cost_expectation(self, gamma, beta):
        """Compute expectation value of cost function."""
        circuit = self.create_qaoa_circuit(gamma, beta)
        state = circuit.run().state_vector
        
        cost = 0.0
        for bitstring_int in range(2**self.n_qubits):
            probability = np.abs(state[bitstring_int])**2
            bitstring = format(bitstring_int, f'0{self.n_qubits}b')
            
            # Calculate cost for this bitstring
            cut_value = 0
            for i, j in self.edges:
                if bitstring[i] != bitstring[j]:  # Different sides of cut
                    cut_value += 1
            
            cost += probability * cut_value
        
        return cost
    
    def optimize_qaoa(self, p_layers=1):
        """Optimize QAOA parameters."""
        # Initial parameters
        initial_params = np.random.uniform(0, 2*np.pi, 2*p_layers)
        
        def objective(params):
            # Extract gamma and beta parameters
            gammas = params[:p_layers]
            betas = params[p_layers:]
            
            # For simplicity, use single layer
            total_cost = 0
            for gamma, beta in zip(gammas, betas):
                total_cost += self.compute_cost_expectation(gamma, beta)
            
            return -total_cost  # Minimize negative (maximize cost)
        
        result = minimize(objective, initial_params, method='COBYLA')
        return result

# Interactive QAOA demonstration
@interact(
    graph_type=widgets.Dropdown(
        options=[
            ('Triangle (3 nodes)', 'triangle'),
            ('Square (4 nodes)', 'square'), 
            ('Complete 4-graph', 'complete4'),
            ('Linear chain', 'linear')
        ],
        value='triangle',
        description='Graph:'
    ),
    p_layers=widgets.IntSlider(min=1, max=3, value=1, description='QAOA layers:'),
    gamma=widgets.FloatSlider(min=0, max=np.pi, step=0.1, value=0.5, description='γ (gamma):'),
    beta=widgets.FloatSlider(min=0, max=np.pi, step=0.1, value=0.5, description='β (beta):')
)
def interactive_qaoa(graph_type, p_layers, gamma, beta):
    """Interactive QAOA demonstration."""
    
    # Define graphs
    graphs = {
        'triangle': [(0, 1), (1, 2), (2, 0)],
        'square': [(0, 1), (1, 2), (2, 3), (3, 0)],
        'complete4': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        'linear': [(0, 1), (1, 2), (2, 3)]
    }
    
    edges = graphs[graph_type]
    qaoa = QAOASimulator(edges)
    
    print(f"🔗 Graph: {graph_type.upper()} with {len(edges)} edges")
    print(f"📊 Nodes: {qaoa.n_qubits}, Parameters: γ={gamma:.2f}, β={beta:.2f}")
    
    # Compute current expectation value
    cost_expectation = qaoa.compute_cost_expectation(gamma, beta)
    print(f"🎯 Expected cut value: {cost_expectation:.3f}")
    
    # Create parameter sweep for visualization
    gamma_range = np.linspace(0, np.pi, 20)
    beta_range = np.linspace(0, np.pi, 20)
    cost_landscape = np.zeros((len(gamma_range), len(beta_range)))
    
    for i, g in enumerate(gamma_range):
        for j, b in enumerate(beta_range):
            cost_landscape[i, j] = qaoa.compute_cost_expectation(g, b)
    
    # Visualize cost landscape
    fig = go.Figure(data=go.Heatmap(
        z=cost_landscape,
        x=beta_range,
        y=gamma_range,
        colorscale='Viridis',
        hovertemplate='γ: %{y:.2f}<br>β: %{x:.2f}<br>Cost: %{z:.3f}<extra></extra>'
    ))
    
    # Add current point
    fig.add_trace(go.Scatter(
        x=[beta], y=[gamma],
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name='Current point'
    ))
    
    fig.update_layout(
        title=f'QAOA Cost Landscape: {graph_type.upper()}',
        xaxis_title='β (beta)',
        yaxis_title='γ (gamma)',
        height=500
    )
    fig.show()
    
    # Show measurement probabilities
    circuit = qaoa.create_qaoa_circuit(gamma, beta)
    state = circuit.run().state_vector
    probabilities = np.abs(state)**2
    
    # Display top solutions
    print(f"\n🏆 Top measurement outcomes:")
    sorted_indices = np.argsort(probabilities)[::-1]
    for i, idx in enumerate(sorted_indices[:4]):
        if probabilities[idx] > 1e-6:
            bitstring = format(idx, f'0{qaoa.n_qubits}b')
            cut_value = sum(1 for edge_i, edge_j in edges 
                           if bitstring[edge_i] != bitstring[edge_j])
            print(f"  {i+1}. |{bitstring}⟩: {probabilities[idx]:.3f} (cut = {cut_value})")
    
    return qaoa, cost_landscape

# Run QAOA demonstration
qaoa_result = interactive_qaoa('triangle', 1, 0.5, 0.5)
```

## Algorithm Comparison Workshop 📊

Let's compare all algorithms we've learned:

```python
def algorithm_comparison_workshop():
    """Compare quantum algorithms on different metrics."""
    
    algorithms = {
        'Grover': {
            'speedup': 'Quadratic',
            'problem_type': 'Search',
            'classical_complexity': 'O(N)',
            'quantum_complexity': 'O(√N)',
            'practical_advantage': 'Moderate'
        },
        'QFT': {
            'speedup': 'Exponential',
            'problem_type': 'Transform',
            'classical_complexity': 'O(N log N)',
            'quantum_complexity': 'O(log²N)',
            'practical_advantage': 'High'
        },
        'VQE': {
            'speedup': 'Polynomial*',
            'problem_type': 'Optimization',
            'classical_complexity': 'Exponential',
            'quantum_complexity': 'Polynomial',
            'practical_advantage': 'NISQ-era'
        },
        'QAOA': {
            'speedup': 'Problem-dependent',
            'problem_type': 'Combinatorial',
            'classical_complexity': 'NP-hard',
            'quantum_complexity': 'Polynomial',
            'practical_advantage': 'NISQ-era'
        }
    }
    
    # Create comparison visualization
    metrics = ['Speedup Type', 'Problem Domain', 'NISQ Suitability', 'Theoretical Advantage']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Algorithm characteristics radar chart
    categories = ['Speed', 'Practical', 'NISQ-Ready', 'Proven']
    
    # Dummy scores for visualization (0-5 scale)
    scores = {
        'Grover': [4, 3, 5, 5],
        'QFT': [5, 4, 4, 5],
        'VQE': [3, 5, 5, 4],
        'QAOA': [3, 4, 5, 3]
    }
    
    for i, (alg_name, score) in enumerate(scores.items()):
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        score_cycle = score + [score[0]]  # Complete the circle
        angles_cycle = np.concatenate([angles, [angles[0]]])
        
        axes[i].plot(angles_cycle, score_cycle, 'o-', linewidth=2, label=alg_name)
        axes[i].fill(angles_cycle, score_cycle, alpha=0.25)
        axes[i].set_xticks(angles)
        axes[i].set_xticklabels(categories)
        axes[i].set_ylim(0, 5)
        axes[i].set_title(f'{alg_name} Algorithm Profile')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print("🔬 Quantum Algorithm Comparison")
    print("=" * 50)
    
    for alg_name, props in algorithms.items():
        print(f"\n{alg_name}:")
        for key, value in props.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return algorithms

# Run comparison workshop
comparison_results = algorithm_comparison_workshop()
```

## Final Project: Hybrid Algorithm Design 🚀

Design your own hybrid quantum-classical algorithm:

```python
def final_project_template():
    """Template for designing your own quantum algorithm."""
    
    print("🚀 Final Project: Design Your Quantum Algorithm!")
    print("=" * 50)
    
    # Project ideas
    ideas = [
        "Quantum-enhanced machine learning classifier",
        "Hybrid optimization for portfolio management", 
        "Quantum algorithm for graph coloring",
        "NISQ-era chemistry simulation",
        "Quantum data compression scheme"
    ]
    
    print("💡 Project Ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"  {i}. {idea}")
    
    print("\n📋 Implementation Checklist:")
    checklist = [
        "[ ] Define the problem clearly",
        "[ ] Design quantum circuit or ansatz",
        "[ ] Implement classical optimization loop",
        "[ ] Add measurement and post-processing",
        "[ ] Compare with classical baseline",
        "[ ] Analyze quantum advantage",
        "[ ] Document algorithm steps",
        "[ ] Test with different parameters"
    ]
    
    for item in checklist:
        print(f"  {item}")
    
    # Template code structure
    template_code = '''
# Your Algorithm Template
class YourQuantumAlgorithm:
    def __init__(self, problem_parameters):
        self.params = problem_parameters
        self.n_qubits = self._determine_qubit_count()
    
    def create_quantum_circuit(self, parameters):
        """Design your quantum circuit here."""
        circuit = Circuit(self.n_qubits)
        # Add your gates here
        return circuit
    
    def classical_cost_function(self, parameters):
        """Define your optimization objective."""
        # Implement your cost function
        pass
    
    def optimize(self):
        """Hybrid optimization loop."""
        # Use scipy.optimize or custom optimizer
        pass
    
    def analyze_results(self, result):
        """Analyze and visualize results."""
        # Create plots and metrics
        pass
'''
    
    print(f"\n💻 Template Code Structure:")
    print(template_code)
    
    print("\n🎯 Success Criteria:")
    print("  ✓ Algorithm runs without errors")
    print("  ✓ Shows quantum-classical interaction")  
    print("  ✓ Produces meaningful results")
    print("  ✓ Includes performance analysis")
    print("  ✓ Demonstrates potential quantum advantage")
    
    return template_code

# Start your final project
project_template = final_project_template()

# Your implementation space:
# Create your algorithm here!
```

## Summary and Next Steps 🎓

Congratulations! You've completed the quantum algorithms workshop.

### What You've Mastered:
- ✅ Grover's quantum search algorithm
- ✅ Quantum Fourier Transform (QFT)
- ✅ Variational Quantum Eigensolver (VQE)
- ✅ Quantum Approximate Optimization Algorithm (QAOA)
- ✅ Algorithm comparison and analysis
- ✅ Hybrid quantum-classical programming

### Advanced Topics to Explore:
1. **Quantum Error Correction**: Learn about fault-tolerant quantum computation
2. **Advanced VQE**: Explore different ansätze and optimization strategies
3. **Quantum Machine Learning**: Implement quantum neural networks
4. **Real Hardware**: Run algorithms on actual quantum computers
5. **Quantum Advantage**: Analyze when quantum algorithms outperform classical

### Resources for Continued Learning:
- [Advanced Quantum Algorithms](../advanced/)
- [Hardware Integration Guide](../../hardware/)
- [Quantum Machine Learning Tutorials](../applications/ml/)
- [Research Papers and References](../../community/resources.md)

### Share Your Work! 🌟
- Upload your algorithms to the [QuantRS2 Algorithm Marketplace](../../dev-tools/marketplace.md)
- Contribute to the [QuantRS2 Community](https://github.com/cool-japan/quantrs)
- Share your results on social media with #QuantRS2

---

*Keep exploring the quantum realm! The future of computing is in your hands.* 🚀✨