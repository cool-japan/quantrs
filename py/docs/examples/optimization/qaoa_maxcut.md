# QAOA for Max-Cut Problem

**Level:** 🟡 Intermediate  
**Runtime:** 1-3 minutes  
**Topics:** Combinatorial optimization, QAOA, Graph problems  
**Problem:** Graph partitioning

Learn to solve the Maximum Cut problem using the Quantum Approximate Optimization Algorithm (QAOA) - a key algorithm for quantum advantage in optimization.

## What is the Max-Cut Problem?

The Maximum Cut (Max-Cut) problem is a classic NP-hard combinatorial optimization problem:

**Given:** An undirected graph with weighted edges  
**Goal:** Partition vertices into two sets to maximize total weight of edges between sets

**Applications:**
- Network design and clustering
- VLSI circuit layout optimization  
- Portfolio optimization
- Image segmentation
- Social network analysis

## What is QAOA?

The Quantum Approximate Optimization Algorithm (QAOA) is a variational hybrid quantum-classical algorithm designed for combinatorial optimization problems:

1. **Encode** the problem as a cost Hamiltonian
2. **Prepare** a parameterized quantum state using alternating unitaries
3. **Measure** expectation value of the cost function
4. **Optimize** parameters classically to maximize/minimize the cost
5. **Extract** the approximate solution from the optimized quantum state

**Key Benefits:**
- Polynomial depth quantum circuits (suitable for NISQ devices)
- Provable approximation guarantees
- Potential quantum advantage for certain problem instances

## Implementation

### Graph Representation and Max-Cut Formulation

```python
import quantrs2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import combinations

class MaxCutProblem:
    """
    Representation of the Maximum Cut problem.
    """
    
    def __init__(self, graph=None, adjacency_matrix=None):
        """
        Initialize Max-Cut problem instance.
        
        Args:
            graph: NetworkX graph object
            adjacency_matrix: 2D numpy array representing the graph
        """
        if graph is not None:
            self.graph = graph
            self.adjacency_matrix = nx.adjacency_matrix(graph).toarray()
        elif adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
            self.graph = nx.from_numpy_array(adjacency_matrix)
        else:
            raise ValueError("Must provide either graph or adjacency_matrix")
        
        self.num_vertices = len(self.adjacency_matrix)
        self.num_edges = len(self.graph.edges())
        
        print(f"📊 Max-Cut Problem Instance:")
        print(f"   Vertices: {self.num_vertices}")
        print(f"   Edges: {self.num_edges}")
        print(f"   Total weight: {self.total_weight}")
    
    @property
    def total_weight(self):
        """Total weight of all edges."""
        return np.sum(self.adjacency_matrix) / 2  # Divide by 2 for undirected
    
    def evaluate_cut(self, assignment):
        """
        Evaluate the cut value for a given vertex assignment.
        
        Args:
            assignment: List/array of 0s and 1s indicating partition
            
        Returns:
            Cut value (total weight of edges crossing the partition)
        """
        cut_value = 0
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if assignment[i] != assignment[j]:  # Vertices in different sets
                    cut_value += self.adjacency_matrix[i][j]
        return cut_value
    
    def brute_force_solution(self):
        """Find optimal solution by brute force (exponential time)."""
        
        print("🔍 Finding optimal solution via brute force...")
        
        best_cut = 0
        best_assignment = None
        
        # Try all possible 2^n assignments
        for assignment_int in range(2**(self.num_vertices - 1)):  # Fix first vertex to reduce symmetry
            assignment = [0] + [(assignment_int >> i) & 1 for i in range(self.num_vertices - 1)]
            cut_value = self.evaluate_cut(assignment)
            
            if cut_value > best_cut:
                best_cut = cut_value
                best_assignment = assignment
        
        print(f"   Optimal cut value: {best_cut}")
        print(f"   Optimal assignment: {best_assignment}")
        
        return best_assignment, best_cut
    
    def visualize_graph(self, assignment=None, title="Graph"):
        """Visualize the graph with optional cut visualization."""
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        
        if assignment is None:
            # Draw graph without partition
            nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, font_size=16, font_weight='bold')
            
            # Draw edge labels (weights)
            edge_labels = nx.get_edge_attributes(self.graph, 'weight')
            if not edge_labels:  # If no weights, show 1s
                edge_labels = {edge: 1 for edge in self.graph.edges()}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        else:
            # Draw graph with partition coloring
            colors = ['red' if assignment[node] == 0 else 'blue' for node in self.graph.nodes()]
            nx.draw(self.graph, pos, with_labels=True, node_color=colors,
                   node_size=500, font_size=16, font_weight='bold')
            
            # Highlight cut edges
            cut_edges = []
            regular_edges = []
            
            for edge in self.graph.edges():
                if assignment[edge[0]] != assignment[edge[1]]:
                    cut_edges.append(edge)
                else:
                    regular_edges.append(edge)
            
            # Draw regular edges in gray
            nx.draw_networkx_edges(self.graph, pos, regular_edges, edge_color='gray', width=1)
            
            # Draw cut edges in thick red
            nx.draw_networkx_edges(self.graph, pos, cut_edges, edge_color='red', width=3)
            
            cut_value = self.evaluate_cut(assignment)
            title += f" (Cut Value: {cut_value})"
        
        plt.title(title, size=16, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Create example Max-Cut instances
def create_sample_graphs():
    """Create sample graphs for testing QAOA."""
    
    graphs = {}
    
    # Triangle graph (3 vertices)
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1, {'weight': 1}), (1, 2, {'weight': 1}), (2, 0, {'weight': 1})])
    graphs['triangle'] = MaxCutProblem(triangle)
    
    # Square graph (4 vertices)
    square = nx.Graph()
    square.add_edges_from([(0, 1, {'weight': 1}), (1, 2, {'weight': 1}), 
                          (2, 3, {'weight': 1}), (3, 0, {'weight': 1})])
    graphs['square'] = MaxCutProblem(square)
    
    # Complete graph K4
    k4 = nx.complete_graph(4)
    graphs['k4'] = MaxCutProblem(k4)
    
    # Random weighted graph
    np.random.seed(42)
    random_graph = nx.erdos_renyi_graph(5, 0.6, seed=42)
    for edge in random_graph.edges():
        random_graph[edge[0]][edge[1]]['weight'] = np.random.randint(1, 5)
    graphs['random'] = MaxCutProblem(random_graph)
    
    return graphs

# Test Max-Cut problem setup
sample_graphs = create_sample_graphs()

print("Sample Max-Cut instances:")
for name, problem in sample_graphs.items():
    print(f"\n{name.upper()}:")
    if problem.num_vertices <= 4:  # Only solve small instances
        _, optimal_cut = problem.brute_force_solution()
```

### QAOA Implementation

```python
class QAOAMaxCut:
    """
    QAOA implementation for the Max-Cut problem.
    """
    
    def __init__(self, problem, num_layers=1):
        """
        Initialize QAOA solver.
        
        Args:
            problem: MaxCutProblem instance
            num_layers: Number of QAOA layers (p parameter)
        """
        self.problem = problem
        self.num_layers = num_layers
        self.num_qubits = problem.num_vertices
        
        # QAOA parameters: [beta_1, gamma_1, beta_2, gamma_2, ...]
        self.num_parameters = 2 * num_layers
        
        # Optimization history
        self.optimization_history = []
        self.best_parameters = None
        self.best_cost = -np.inf
        
        print(f"🔧 QAOA Configuration:")
        print(f"   Qubits: {self.num_qubits}")
        print(f"   Layers: {self.num_layers}")
        print(f"   Parameters: {self.num_parameters}")
    
    def create_qaoa_circuit(self, parameters):
        """
        Create QAOA quantum circuit for given parameters.
        
        Args:
            parameters: Array of [beta_1, gamma_1, beta_2, gamma_2, ...]
            
        Returns:
            Quantum circuit implementing QAOA
        """
        circuit = quantrs2.Circuit(self.num_qubits)
        
        # Initial state: uniform superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        
        # Apply QAOA layers
        for layer in range(self.num_layers):
            gamma = parameters[2 * layer]
            beta = parameters[2 * layer + 1]
            
            # Cost unitary: exp(-i * gamma * C)
            self.apply_cost_unitary(circuit, gamma)
            
            # Mixing unitary: exp(-i * beta * B) 
            self.apply_mixing_unitary(circuit, beta)
        
        return circuit
    
    def apply_cost_unitary(self, circuit, gamma):
        """Apply the cost unitary encoding the Max-Cut objective."""
        
        # For Max-Cut, the cost Hamiltonian is:
        # C = Σ_{(i,j) ∈ E} w_{ij} * (1 - Z_i * Z_j) / 2
        # The unitary is exp(-i * gamma * C)
        
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                weight = self.problem.adjacency_matrix[i][j]
                if weight > 0:  # There's an edge
                    # Apply exp(-i * gamma * weight * (1 - Z_i * Z_j) / 2)
                    # This is equivalent to RZ gates and CNOT
                    
                    # Single qubit rotations
                    circuit.rz(i, gamma * weight)
                    circuit.rz(j, gamma * weight)
                    
                    # Two-qubit ZZ interaction
                    circuit.cx(i, j)
                    circuit.rz(j, -gamma * weight)
                    circuit.cx(i, j)
    
    def apply_mixing_unitary(self, circuit, beta):
        """Apply the mixing unitary for exploring the solution space."""
        
        # The mixing Hamiltonian is B = Σ_i X_i
        # The unitary is exp(-i * beta * B) = Π_i exp(-i * beta * X_i)
        
        for qubit in range(self.num_qubits):
            circuit.rx(qubit, 2 * beta)  # RX(2*beta) = exp(-i * beta * X)
    
    def measure_expectation_value(self, parameters, num_shots=1000):
        """
        Measure expectation value of the cost function.
        
        Args:
            parameters: QAOA parameters
            num_shots: Number of measurement shots
            
        Returns:
            Expectation value of the Max-Cut objective
        """
        circuit = self.create_qaoa_circuit(parameters)
        circuit.measure_all()
        
        # Run circuit multiple times to estimate expectation
        total_cost = 0
        
        for _ in range(num_shots):
            result = circuit.run()
            probabilities = result.state_probabilities()
            
            # For each measured state, compute its contribution to cost
            for state_str, prob in probabilities.items():
                # Convert binary string to assignment
                assignment = [int(bit) for bit in state_str]
                cut_value = self.problem.evaluate_cut(assignment)
                total_cost += prob * cut_value
        
        return total_cost / num_shots
    
    def cost_function(self, parameters):
        """
        Cost function for classical optimization.
        Returns negative expectation value (for minimization).
        """
        expectation = self.measure_expectation_value(parameters)
        cost = -expectation  # Negative because we want to maximize
        
        # Store optimization history
        self.optimization_history.append({
            'parameters': parameters.copy(),
            'cost': cost,
            'expectation': expectation
        })
        
        # Update best solution
        if expectation > self.best_cost:
            self.best_cost = expectation
            self.best_parameters = parameters.copy()
        
        return cost
    
    def optimize(self, initial_parameters=None, method='COBYLA', maxiter=100):
        """
        Optimize QAOA parameters using classical optimization.
        
        Args:
            initial_parameters: Starting point for optimization
            method: Optimization method ('COBYLA', 'SLSQP', 'Nelder-Mead')
            maxiter: Maximum number of iterations
            
        Returns:
            Optimization result
        """
        print(f"🚀 Starting QAOA optimization...")
        print(f"   Method: {method}")
        print(f"   Max iterations: {maxiter}")
        
        if initial_parameters is None:
            # Random initialization
            np.random.seed(42)
            initial_parameters = np.random.uniform(0, 2*np.pi, self.num_parameters)
        
        print(f"   Initial parameters: {initial_parameters}")
        
        # Clear optimization history
        self.optimization_history = []
        self.best_cost = -np.inf
        
        # Run optimization
        result = minimize(
            self.cost_function,
            initial_parameters,
            method=method,
            options={'maxiter': maxiter, 'disp': True}
        )
        
        print(f"\n✅ Optimization completed!")
        print(f"   Best expectation value: {self.best_cost:.4f}")
        print(f"   Optimal parameters: {self.best_parameters}")
        print(f"   Function evaluations: {len(self.optimization_history)}")
        
        return result
    
    def get_solution_probabilities(self, parameters=None, num_shots=1000):
        """Get probability distribution over all possible solutions."""
        
        if parameters is None:
            parameters = self.best_parameters
        
        circuit = self.create_qaoa_circuit(parameters)
        circuit.measure_all()
        
        # Accumulate probabilities over multiple runs
        all_probabilities = {}
        
        for _ in range(num_shots):
            result = circuit.run()
            probabilities = result.state_probabilities()
            
            for state, prob in probabilities.items():
                if state in all_probabilities:
                    all_probabilities[state] += prob
                else:
                    all_probabilities[state] = prob
        
        # Normalize
        total_prob = sum(all_probabilities.values())
        return {state: prob / total_prob for state, prob in all_probabilities.items()}
    
    def extract_solution(self, parameters=None, top_k=5):
        """Extract the most likely solutions from QAOA."""
        
        print(f"🎯 Extracting QAOA solutions...")
        
        probs = self.get_solution_probabilities(parameters)
        
        # Sort by probability
        sorted_solutions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   Top {top_k} solutions:")
        solutions = []
        
        for i, (state_str, prob) in enumerate(sorted_solutions[:top_k]):
            assignment = [int(bit) for bit in state_str]
            cut_value = self.problem.evaluate_cut(assignment)
            
            solutions.append({
                'assignment': assignment,
                'cut_value': cut_value,
                'probability': prob,
                'state': state_str
            })
            
            print(f"   {i+1}. {state_str} → Cut: {cut_value:4.1f}, Prob: {prob:.3f}")
        
        return solutions

# Demonstrate QAOA on sample problems
def demonstrate_qaoa():
    """Demonstrate QAOA on sample Max-Cut instances."""
    
    print("🌟 QAOA Max-Cut Demonstration")
    print("=" * 50)
    
    # Test on triangle graph
    triangle_problem = sample_graphs['triangle']
    print(f"\n📐 Triangle Graph:")
    triangle_problem.visualize_graph(title="Triangle Graph")
    
    # Solve with QAOA
    qaoa_triangle = QAOAMaxCut(triangle_problem, num_layers=1)
    result = qaoa_triangle.optimize(maxiter=50)
    
    # Extract solutions
    solutions = qaoa_triangle.extract_solution(top_k=3)
    best_solution = solutions[0]
    
    # Visualize best solution
    triangle_problem.visualize_graph(
        best_solution['assignment'], 
        f"Triangle QAOA Solution"
    )
    
    # Compare with optimal
    _, optimal_cut = triangle_problem.brute_force_solution()
    qaoa_cut = best_solution['cut_value']
    approximation_ratio = qaoa_cut / optimal_cut if optimal_cut > 0 else 1
    
    print(f"\n📊 Triangle Results:")
    print(f"   QAOA cut: {qaoa_cut}")
    print(f"   Optimal cut: {optimal_cut}")
    print(f"   Approximation ratio: {approximation_ratio:.3f}")
    
    return qaoa_triangle, solutions

# Run QAOA demonstration
qaoa_demo, demo_solutions = demonstrate_qaoa()
```

### Advanced QAOA Features

```python
def analyze_qaoa_performance():
    """Analyze QAOA performance characteristics."""
    
    print("\n📈 QAOA Performance Analysis")
    print("=" * 40)
    
    # Test different numbers of layers
    layer_counts = [1, 2, 3, 4]
    square_problem = sample_graphs['square']
    
    print(f"Testing QAOA layers on square graph:")
    
    results = {}
    for p in layer_counts:
        print(f"\n  Testing p = {p} layers...")
        
        qaoa = QAOAMaxCut(square_problem, num_layers=p)
        qaoa.optimize(maxiter=30)
        
        solutions = qaoa.extract_solution(top_k=1)
        best_cut = solutions[0]['cut_value']
        
        results[p] = {
            'best_cut': best_cut,
            'parameters': qaoa.best_parameters,
            'evaluations': len(qaoa.optimization_history)
        }
        
        print(f"    Best cut: {best_cut}")
        print(f"    Function evaluations: {results[p]['evaluations']}")
    
    # Find theoretical maximum
    _, optimal_cut = square_problem.brute_force_solution()
    
    print(f"\n📊 Layer Analysis Results:")
    print(f"{'Layers':<8} {'Cut Value':<12} {'Ratio':<8} {'Evaluations'}")
    print("-" * 40)
    
    for p in layer_counts:
        cut = results[p]['best_cut']
        ratio = cut / optimal_cut
        evals = results[p]['evaluations']
        print(f"{p:<8} {cut:<12.1f} {ratio:<8.3f} {evals}")
    
    print(f"\nOptimal cut value: {optimal_cut}")
    
    return results

def visualize_optimization_landscape():
    """Visualize the QAOA optimization landscape."""
    
    print("\n🗺️  QAOA Optimization Landscape")
    print("=" * 35)
    
    triangle_problem = sample_graphs['triangle']
    qaoa = QAOAMaxCut(triangle_problem, num_layers=1)
    
    # Create parameter grid for visualization
    gamma_range = np.linspace(0, 2*np.pi, 20)
    beta_range = np.linspace(0, np.pi, 20)
    
    print("Computing landscape (this may take a moment)...")
    
    landscape = np.zeros((len(gamma_range), len(beta_range)))
    
    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            parameters = np.array([gamma, beta])
            expectation = qaoa.measure_expectation_value(parameters, num_shots=100)
            landscape[i, j] = expectation
    
    # Plot landscape
    plt.figure(figsize=(12, 5))
    
    # 2D heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(landscape, extent=[0, np.pi, 0, 2*np.pi], aspect='auto', cmap='viridis')
    plt.colorbar(label='Expectation Value')
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    plt.title('QAOA Optimization Landscape')
    
    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    
    plt.subplot(1, 2, 2, projection='3d')
    G, B = np.meshgrid(gamma_range, beta_range)
    plt.plot_surface(B, G, landscape.T, cmap='viridis', alpha=0.8)
    plt.xlabel('Beta')
    plt.ylabel('Gamma')
    plt.zlabel('Expectation Value')
    plt.title('3D Landscape')
    
    plt.tight_layout()
    plt.show()
    
    # Find maximum
    max_idx = np.unravel_index(np.argmax(landscape), landscape.shape)
    optimal_gamma = gamma_range[max_idx[0]]
    optimal_beta = beta_range[max_idx[1]]
    max_expectation = landscape[max_idx]
    
    print(f"\nLandscape Analysis:")
    print(f"  Maximum expectation: {max_expectation:.3f}")
    print(f"  Optimal gamma: {optimal_gamma:.3f}")
    print(f"  Optimal beta: {optimal_beta:.3f}")
    
    return landscape, (optimal_gamma, optimal_beta)

def compare_qaoa_with_classical():
    """Compare QAOA with classical algorithms."""
    
    print("\n🏁 QAOA vs Classical Comparison")
    print("=" * 40)
    
    # Test on multiple graph instances
    test_problems = [
        ('Triangle', sample_graphs['triangle']),
        ('Square', sample_graphs['square']),
        ('K4', sample_graphs['k4']),
        ('Random', sample_graphs['random'])
    ]
    
    results = []
    
    for name, problem in test_problems:
        print(f"\n{name} Graph:")
        
        # QAOA solution
        qaoa = QAOAMaxCut(problem, num_layers=2)
        qaoa.optimize(maxiter=40)
        qaoa_solutions = qaoa.extract_solution(top_k=1)
        qaoa_cut = qaoa_solutions[0]['cut_value']
        
        # Classical solutions
        if problem.num_vertices <= 5:  # Only for small graphs
            _, optimal_cut = problem.brute_force_solution()
        else:
            optimal_cut = None
        
        # Random solution (baseline)
        np.random.seed(42)
        random_assignment = np.random.randint(0, 2, problem.num_vertices)
        random_cut = problem.evaluate_cut(random_assignment)
        
        # Greedy solution
        greedy_cut = greedy_max_cut(problem)
        
        results.append({
            'name': name,
            'qaoa': qaoa_cut,
            'optimal': optimal_cut,
            'greedy': greedy_cut,
            'random': random_cut,
            'num_vertices': problem.num_vertices
        })
        
        print(f"  QAOA: {qaoa_cut:.1f}")
        if optimal_cut:
            print(f"  Optimal: {optimal_cut:.1f}")
            print(f"  QAOA ratio: {qaoa_cut/optimal_cut:.3f}")
        print(f"  Greedy: {greedy_cut:.1f}")
        print(f"  Random: {random_cut:.1f}")
    
    # Summary table
    print(f"\n📊 Comparison Summary:")
    print(f"{'Graph':<10} {'Vertices':<9} {'QAOA':<6} {'Greedy':<7} {'Random':<7} {'Ratio'}")
    print("-" * 55)
    
    for result in results:
        ratio = result['qaoa'] / result['optimal'] if result['optimal'] else 'N/A'
        ratio_str = f"{ratio:.3f}" if isinstance(ratio, float) else ratio
        
        print(f"{result['name']:<10} {result['num_vertices']:<9} "
              f"{result['qaoa']:<6.1f} {result['greedy']:<7.1f} "
              f"{result['random']:<7.1f} {ratio_str}")
    
    return results

def greedy_max_cut(problem):
    """Simple greedy algorithm for Max-Cut."""
    
    assignment = np.zeros(problem.num_vertices, dtype=int)
    
    for vertex in range(1, problem.num_vertices):
        # Assign vertex to partition that maximizes cut
        cut_0 = problem.evaluate_cut(assignment)
        
        assignment[vertex] = 1
        cut_1 = problem.evaluate_cut(assignment)
        
        if cut_0 > cut_1:
            assignment[vertex] = 0
    
    return problem.evaluate_cut(assignment)

# Run performance analysis
layer_results = analyze_qaoa_performance()
landscape, optimal_params = visualize_optimization_landscape()
comparison_results = compare_qaoa_with_classical()
```

### Real-World Applications

```python
def network_clustering_example():
    """Demonstrate QAOA for network clustering."""
    
    print("\n🌐 Network Clustering with QAOA")
    print("=" * 35)
    
    # Create a network with community structure
    # Two communities connected by fewer edges
    
    community_graph = nx.Graph()
    
    # Community 1: vertices 0, 1, 2
    community_graph.add_edges_from([
        (0, 1, {'weight': 3}),
        (1, 2, {'weight': 3}),
        (2, 0, {'weight': 3})
    ])
    
    # Community 2: vertices 3, 4, 5
    community_graph.add_edges_from([
        (3, 4, {'weight': 3}),
        (4, 5, {'weight': 3}),
        (5, 3, {'weight': 3})
    ])
    
    # Inter-community edges (weaker)
    community_graph.add_edges_from([
        (1, 3, {'weight': 1}),
        (2, 4, {'weight': 1})
    ])
    
    problem = MaxCutProblem(community_graph)
    
    print("Community network structure:")
    problem.visualize_graph(title="Community Network")
    
    # Apply QAOA
    qaoa = QAOAMaxCut(problem, num_layers=3)
    qaoa.optimize(maxiter=60)
    
    solutions = qaoa.extract_solution(top_k=3)
    
    print("\nCluster analysis:")
    for i, sol in enumerate(solutions):
        assignment = sol['assignment']
        cluster_0 = [j for j, x in enumerate(assignment) if x == 0]
        cluster_1 = [j for j, x in enumerate(assignment) if x == 1]
        
        print(f"  Solution {i+1}:")
        print(f"    Cluster 0: {cluster_0}")
        print(f"    Cluster 1: {cluster_1}")
        print(f"    Cut value: {sol['cut_value']}")
        print(f"    Probability: {sol['probability']:.3f}")
    
    # Visualize best clustering
    best_assignment = solutions[0]['assignment']
    problem.visualize_graph(best_assignment, "QAOA Network Clustering")
    
    return problem, solutions

def portfolio_optimization_example():
    """Use QAOA for portfolio optimization (quadratic unconstrained binary optimization)."""
    
    print("\n💰 Portfolio Optimization with QAOA")
    print("=" * 38)
    
    # Portfolio optimization problem:
    # Maximize: expected_returns^T * x - lambda * x^T * covariance * x
    # Subject to: x_i ∈ {0, 1} (binary: invest or not)
    
    # Sample data: 4 assets
    expected_returns = np.array([0.12, 0.08, 0.15, 0.10])  # Expected returns
    
    # Covariance matrix (risk)
    covariance = np.array([
        [0.04, 0.01, 0.02, 0.01],
        [0.01, 0.02, 0.01, 0.01],
        [0.02, 0.01, 0.05, 0.02],
        [0.01, 0.01, 0.02, 0.03]
    ])
    
    risk_aversion = 0.5  # Lambda parameter
    
    print(f"Assets: {len(expected_returns)}")
    print(f"Expected returns: {expected_returns}")
    print(f"Risk aversion: {risk_aversion}")
    
    # Convert to Max-Cut like problem
    # We need to formulate as QUBO: maximize x^T * Q * x + linear^T * x
    
    # Q matrix combines covariance (negative) and interaction terms
    Q = -risk_aversion * covariance
    linear = expected_returns
    
    # Create equivalent graph for QAOA
    # This is a heuristic mapping - in practice, more sophisticated encoding needed
    portfolio_graph = nx.Graph()
    
    for i in range(len(expected_returns)):
        for j in range(i + 1, len(expected_returns)):
            weight = abs(Q[i, j])
            if weight > 0.001:  # Only add significant connections
                portfolio_graph.add_edge(i, j, weight=weight * 10)  # Scale for visualization
    
    # Add self-loops as node weights (approximation)
    for i in range(len(expected_returns)):
        portfolio_graph.add_edge(i, i, weight=abs(linear[i]) * 5)
    
    problem = MaxCutProblem(portfolio_graph)
    
    print(f"\nPortfolio graph:")
    problem.visualize_graph(title="Portfolio Correlation Network")
    
    # Apply QAOA
    qaoa = QAOAMaxCut(problem, num_layers=2)
    qaoa.optimize(maxiter=40)
    
    solutions = qaoa.extract_solution(top_k=5)
    
    print(f"\nPortfolio solutions:")
    for i, sol in enumerate(solutions):
        assignment = sol['assignment']
        selected_assets = [j for j, x in enumerate(assignment) if x == 1]
        
        if selected_assets:  # Don't invest in nothing
            portfolio_return = sum(expected_returns[j] for j in selected_assets)
            portfolio_risk = sum(covariance[j, k] for j in selected_assets for k in selected_assets)
            objective = portfolio_return - risk_aversion * portfolio_risk
            
            print(f"  Portfolio {i+1}:")
            print(f"    Assets: {selected_assets}")
            print(f"    Return: {portfolio_return:.3f}")
            print(f"    Risk: {portfolio_risk:.3f}")
            print(f"    Objective: {objective:.3f}")
            print(f"    Probability: {sol['probability']:.3f}")
    
    return problem, solutions

# Run real-world examples
clustering_problem, clustering_solutions = network_clustering_example()
portfolio_problem, portfolio_solutions = portfolio_optimization_example()
```

## Algorithm Variants and Extensions

### Multi-Level QAOA

```python
def multilevel_qaoa():
    """Implement multi-level QAOA for larger graphs."""
    
    print("\n🏗️  Multi-Level QAOA")
    print("=" * 25)
    
    # Create larger random graph
    np.random.seed(42)
    large_graph = nx.erdos_renyi_graph(8, 0.4, seed=42)
    
    # Add random weights
    for edge in large_graph.edges():
        large_graph[edge[0]][edge[1]]['weight'] = np.random.randint(1, 4)
    
    large_problem = MaxCutProblem(large_graph)
    
    print(f"Large graph: {large_problem.num_vertices} vertices, {large_problem.num_edges} edges")
    
    # Standard QAOA (may struggle with larger graphs)
    print("\nStandard QAOA:")
    standard_qaoa = QAOAMaxCut(large_problem, num_layers=2)
    standard_qaoa.optimize(maxiter=30)
    standard_solutions = standard_qaoa.extract_solution(top_k=1)
    standard_cut = standard_solutions[0]['cut_value']
    
    print(f"  Standard QAOA cut: {standard_cut}")
    
    # Multi-level approach: coarsen graph, solve, refine
    print("\nMulti-level approach (conceptual):")
    
    # Step 1: Graph coarsening (simplified)
    coarse_graph = nx.Graph()
    
    # Merge vertex pairs to create smaller graph
    vertex_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for i, (v1, v2) in enumerate(vertex_pairs):
        if large_graph.has_node(v1) and large_graph.has_node(v2):
            # Merge vertices v1 and v2 into super-vertex i
            for neighbor in large_graph.neighbors(v1):
                if neighbor not in [v1, v2]:
                    weight = large_graph[v1][neighbor].get('weight', 1)
                    if coarse_graph.has_edge(i, neighbor // 2):
                        coarse_graph[i][neighbor // 2]['weight'] += weight
                    else:
                        coarse_graph.add_edge(i, neighbor // 2, weight=weight)
    
    coarse_problem = MaxCutProblem(coarse_graph)
    
    print(f"  Coarse graph: {coarse_problem.num_vertices} vertices")
    
    # Step 2: Solve coarse problem
    coarse_qaoa = QAOAMaxCut(coarse_problem, num_layers=3)
    coarse_qaoa.optimize(maxiter=40)
    coarse_solutions = coarse_qaoa.extract_solution(top_k=1)
    
    # Step 3: Project solution back (simplified)
    coarse_assignment = coarse_solutions[0]['assignment']
    projected_assignment = []
    for i, coarse_bit in enumerate(coarse_assignment):
        projected_assignment.extend([coarse_bit, coarse_bit])  # Assign same value to merged vertices
    
    # Ensure correct length
    while len(projected_assignment) < large_problem.num_vertices:
        projected_assignment.append(0)
    projected_assignment = projected_assignment[:large_problem.num_vertices]
    
    projected_cut = large_problem.evaluate_cut(projected_assignment)
    
    print(f"  Coarse QAOA cut: {coarse_solutions[0]['cut_value']}")
    print(f"  Projected cut: {projected_cut}")
    
    # Step 4: Local refinement (optional)
    print(f"  Improvement: {projected_cut - standard_cut:+.1f}")
    
    return standard_cut, projected_cut

# Run multi-level QAOA
standard_result, multilevel_result = multilevel_qaoa()
```

### Warm-Start QAOA

```python
def warm_start_qaoa():
    """Implement warm-start QAOA using classical preprocessing."""
    
    print("\n🔥 Warm-Start QAOA")
    print("=" * 20)
    
    # Use square graph for clear demonstration
    square_problem = sample_graphs['square']
    
    # Classical preprocessing: find good initial solution
    print("Step 1: Classical preprocessing")
    
    # Greedy algorithm
    greedy_assignment = np.zeros(square_problem.num_vertices, dtype=int)
    for vertex in range(1, square_problem.num_vertices):
        # Try both assignments and pick better one
        greedy_assignment[vertex] = 0
        cut_0 = square_problem.evaluate_cut(greedy_assignment)
        
        greedy_assignment[vertex] = 1
        cut_1 = square_problem.evaluate_cut(greedy_assignment)
        
        if cut_0 > cut_1:
            greedy_assignment[vertex] = 0
    
    greedy_cut = square_problem.evaluate_cut(greedy_assignment)
    print(f"  Greedy solution: {greedy_assignment}")
    print(f"  Greedy cut: {greedy_cut}")
    
    # Standard QAOA (random initialization)
    print("\nStep 2: Standard QAOA")
    standard_qaoa = QAOAMaxCut(square_problem, num_layers=2)
    standard_qaoa.optimize(maxiter=30)
    standard_solutions = standard_qaoa.extract_solution(top_k=1)
    standard_cut = standard_solutions[0]['cut_value']
    print(f"  Standard QAOA cut: {standard_cut}")
    
    # Warm-start QAOA
    print("\nStep 3: Warm-start QAOA")
    
    # Initialize QAOA parameters to favor greedy solution
    # This is a heuristic - in practice, more sophisticated methods exist
    warm_start_params = np.array([np.pi/4, np.pi/8, np.pi/6, np.pi/12])  # Biased initialization
    
    warm_qaoa = QAOAMaxCut(square_problem, num_layers=2)
    warm_qaoa.optimize(initial_parameters=warm_start_params, maxiter=30)
    warm_solutions = warm_qaoa.extract_solution(top_k=1)
    warm_cut = warm_solutions[0]['cut_value']
    
    print(f"  Warm-start QAOA cut: {warm_cut}")
    
    # Comparison
    print(f"\nComparison:")
    print(f"  Greedy: {greedy_cut}")
    print(f"  Standard QAOA: {standard_cut}")
    print(f"  Warm-start QAOA: {warm_cut}")
    print(f"  Improvement over standard: {warm_cut - standard_cut:+.1f}")
    
    return greedy_cut, standard_cut, warm_cut

# Run warm-start QAOA
greedy_result, standard_result, warm_result = warm_start_qaoa()
```

## Performance Analysis and Benchmarking

```python
def comprehensive_qaoa_benchmark():
    """Comprehensive QAOA performance benchmark."""
    
    print("\n🏆 Comprehensive QAOA Benchmark")
    print("=" * 40)
    
    import time
    
    # Test on graphs of different sizes
    graph_sizes = [3, 4, 5, 6]
    layer_counts = [1, 2, 3]
    
    results = []
    
    for n in graph_sizes:
        print(f"\nTesting {n}-vertex graphs:")
        
        # Generate random graph
        test_graph = nx.erdos_renyi_graph(n, 0.6, seed=42)
        for edge in test_graph.edges():
            test_graph[edge[0]][edge[1]]['weight'] = 1
        
        test_problem = MaxCutProblem(test_graph)
        
        # Brute force optimal (for small graphs)
        if n <= 6:
            start_time = time.time()
            _, optimal_cut = test_problem.brute_force_solution()
            brute_force_time = time.time() - start_time
        else:
            optimal_cut = None
            brute_force_time = None
        
        for p in layer_counts:
            print(f"  Testing p={p} layers...")
            
            # Time QAOA
            start_time = time.time()
            qaoa = QAOAMaxCut(test_problem, num_layers=p)
            qaoa.optimize(maxiter=20)
            qaoa_time = time.time() - start_time
            
            solutions = qaoa.extract_solution(top_k=1)
            qaoa_cut = solutions[0]['cut_value']
            
            # Calculate metrics
            approximation_ratio = qaoa_cut / optimal_cut if optimal_cut else None
            
            results.append({
                'vertices': n,
                'layers': p,
                'qaoa_cut': qaoa_cut,
                'optimal_cut': optimal_cut,
                'approximation_ratio': approximation_ratio,
                'qaoa_time': qaoa_time,
                'brute_force_time': brute_force_time,
                'function_evaluations': len(qaoa.optimization_history)
            })
    
    # Create benchmark table
    print(f"\n📊 Benchmark Results:")
    print(f"{'Vertices':<9} {'Layers':<7} {'QAOA Cut':<10} {'Optimal':<8} {'Ratio':<7} {'Time (s)':<9} {'Evals'}")
    print("-" * 70)
    
    for result in results:
        ratio_str = f"{result['approximation_ratio']:.3f}" if result['approximation_ratio'] else "N/A"
        optimal_str = f"{result['optimal_cut']:.1f}" if result['optimal_cut'] else "N/A"
        
        print(f"{result['vertices']:<9} {result['layers']:<7} {result['qaoa_cut']:<10.1f} "
              f"{optimal_str:<8} {ratio_str:<7} {result['qaoa_time']:<9.2f} {result['function_evaluations']}")
    
    # Analysis
    print(f"\nBenchmark Analysis:")
    
    # Average approximation ratios by layer count
    for p in layer_counts:
        p_results = [r for r in results if r['layers'] == p and r['approximation_ratio']]
        if p_results:
            avg_ratio = np.mean([r['approximation_ratio'] for r in p_results])
            print(f"  Average ratio for p={p}: {avg_ratio:.3f}")
    
    # Scaling analysis
    print(f"\nScaling Analysis:")
    print(f"  QAOA complexity: O(p * E) circuit depth, O(iterations) total time")
    print(f"  Brute force complexity: O(2^n)")
    
    # Find optimal layer count
    best_layer_results = {}
    for n in graph_sizes:
        n_results = [r for r in results if r['vertices'] == n]
        if n_results:
            best = max(n_results, key=lambda x: x['qaoa_cut'])
            best_layer_results[n] = best['layers']
    
    print(f"\nOptimal layer counts by graph size:")
    for n, best_p in best_layer_results.items():
        print(f"  {n} vertices: p = {best_p}")
    
    return results

# Run comprehensive benchmark
benchmark_results = comprehensive_qaoa_benchmark()
```

## Exercises and Extensions

### Exercise 1: Custom Graph Problems
```python
def exercise_custom_graphs():
    """Exercise: Solve Max-Cut on custom graph topologies."""
    
    print("🎯 Exercise: Custom Graph Topologies")
    print("=" * 35)
    
    # TODO: Create and solve Max-Cut on:
    # 1. Star graph (one central vertex connected to all others)
    # 2. Path graph (vertices connected in a line)
    # 3. Cycle graph (vertices connected in a circle)
    # 4. Bipartite graph (two disconnected groups)
    
    print("Your challenge:")
    print("1. Create star, path, cycle, and bipartite graphs")
    print("2. Apply QAOA to each topology")
    print("3. Compare approximation ratios")
    print("4. Analyze which topologies are easier for QAOA")

exercise_custom_graphs()
```

### Exercise 2: QAOA Parameter Analysis
```python
def exercise_parameter_analysis():
    """Exercise: Analyze QAOA parameter patterns."""
    
    print("🎯 Exercise: Parameter Pattern Analysis")
    print("=" * 40)
    
    # TODO: Investigate:
    # 1. How do optimal parameters change with graph size?
    # 2. Are there universal parameter patterns?
    # 3. Can you predict good initial parameters?
    # 4. How sensitive is QAOA to parameter initialization?
    
    print("Investigate QAOA parameter patterns:")
    print("1. Track optimal parameters across different graphs")
    print("2. Look for patterns in gamma/beta ratios")
    print("3. Test parameter transfer between similar graphs")

exercise_parameter_analysis()
```

### Exercise 3: Hybrid Optimization
```python
def exercise_hybrid_optimization():
    """Exercise: Implement hybrid QAOA-classical approaches."""
    
    print("🎯 Exercise: Hybrid Optimization")
    print("=" * 30)
    
    # TODO: Combine QAOA with:
    # 1. Local search refinement
    # 2. Genetic algorithm population
    # 3. Simulated annealing
    # 4. Machine learning parameter prediction
    
    print("Create hybrid algorithms:")
    print("1. QAOA + local search post-processing")
    print("2. ML-predicted parameter initialization")
    print("3. Population-based QAOA variants")

exercise_hybrid_optimization()
```

## Common Mistakes and Troubleshooting

### Mistake 1: Incorrect Cost Hamiltonian
```python
# ❌ Wrong: Minimizing instead of maximizing
def wrong_cost_hamiltonian(circuit, gamma):
    # This minimizes the cut instead of maximizing
    for i, j in edges:
        circuit.rz(i, -gamma)  # Wrong sign!

# ✅ Correct: Proper Max-Cut Hamiltonian
def correct_cost_hamiltonian(circuit, gamma):
    # Maximize cut by using correct Hamiltonian
    for i, j in edges:
        circuit.rz(i, gamma * weight)
        circuit.rz(j, gamma * weight)
        circuit.cx(i, j)
        circuit.rz(j, -gamma * weight)
        circuit.cx(i, j)
```

### Mistake 2: Poor Parameter Initialization
```python
# ❌ Wrong: All parameters start at zero
initial_params = np.zeros(2 * num_layers)

# ✅ Correct: Random or heuristic initialization
initial_params = np.random.uniform(0, 2*np.pi, 2 * num_layers)
# Or use known good starting points
initial_params = np.array([np.pi/4, np.pi/8] * num_layers)
```

### Mistake 3: Insufficient Optimization Budget
```python
# ❌ Wrong: Too few optimization steps
result = minimize(cost_function, params, options={'maxiter': 10})

# ✅ Correct: Adequate optimization budget
result = minimize(cost_function, params, options={'maxiter': 100})
# Consider using multiple restarts
```

## Summary

🎉 **Congratulations!** You've learned:
- How to formulate Max-Cut as a quantum optimization problem
- QAOA algorithm design and implementation
- Parameter optimization and classical-quantum hybrid approaches
- Performance analysis and benchmarking techniques
- Real-world applications in network clustering and portfolio optimization
- Advanced QAOA variants and improvements

QAOA represents a cornerstone of near-term quantum advantage, providing a bridge between classical optimization and quantum computing. Master these techniques to unlock quantum optimization!

**Next Steps:**
- Explore [Variational Quantum Eigensolver (VQE)](vqe.md)
- Try [Portfolio Optimization](portfolio.md) for financial applications
- Learn about [Quantum Machine Learning](../ml/vqc.md)

## References

### Foundational Papers
- Farhi et al. (2014). "A Quantum Approximate Optimization Algorithm"
- Hadfield et al. (2019). "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz"

### Performance Studies
- Zhou et al. (2020). "Quantum approximate optimization algorithm: Performance, mechanism, and implementation on near-term devices"
- Crooks (2018). "Performance of the Quantum Approximate Optimization Algorithm on the Maximum Cut Problem"

### Applications
- Bengtsson et al. (2020). "Improved Success Probability with Greater Circuit Depth for the Quantum Approximate Optimization Algorithm"
- Willsch et al. (2020). "Benchmarking the quantum approximate optimization algorithm"

---

*"The quantum approximate optimization algorithm opens the door to near-term quantum advantage in combinatorial optimization."* - Edward Farhi

🚀 **Ready to optimize with quantum algorithms?** Explore more [Optimization Examples](index.md)!