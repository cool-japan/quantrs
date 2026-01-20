"""
Quantum Annealing Module

This module provides quantum annealing functionality including QUBO/Ising models,
penalty optimization, and layout-aware graph embedding.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = hasattr(_quantrs2, 'anneal')
except ImportError:
    _NATIVE_AVAILABLE = False

if _NATIVE_AVAILABLE:
    # Import native implementations
    QuboModel = _quantrs2.anneal.PyQuboModel
    IsingModel = _quantrs2.anneal.PyIsingModel
    PenaltyOptimizer = _quantrs2.anneal.PyPenaltyOptimizer
    LayoutAwareEmbedder = _quantrs2.anneal.PyLayoutAwareEmbedder
    ChimeraGraph = _quantrs2.anneal.PyChimeraGraph
else:
    # Enhanced fallback implementations
    class QuboModel:
        """QUBO (Quadratic Unconstrained Binary Optimization) model implementation."""
        
        def __init__(self, n_vars: int):
            """
            Initialize a QUBO model.
            
            Args:
                n_vars: Number of binary variables
            """
            self.n_vars = n_vars
            self.Q = np.zeros((n_vars, n_vars))
            self.linear = np.zeros(n_vars)
            self.offset = 0.0
            
        def add_linear(self, var: int, coeff: float):
            """Add linear coefficient for a variable."""
            if 0 <= var < self.n_vars:
                self.linear[var] += coeff
            else:
                raise ValueError(f"Variable {var} out of range [0, {self.n_vars})")
                
        def add_quadratic(self, var1: int, var2: int, coeff: float):
            """Add quadratic coefficient between two variables."""
            if not (0 <= var1 < self.n_vars and 0 <= var2 < self.n_vars):
                raise ValueError("Variables out of range")
            self.Q[var1, var2] += coeff
            if var1 != var2:
                self.Q[var2, var1] += coeff
                
        def set_offset(self, offset: float):
            """Set constant offset term."""
            self.offset = offset
            
        def energy(self, assignment: List[int]) -> float:
            """Calculate energy for a given binary assignment."""
            x = np.array(assignment)
            if len(x) != self.n_vars:
                raise ValueError(f"Assignment length {len(x)} != n_vars {self.n_vars}")
            
            energy = self.offset
            energy += np.dot(self.linear, x)
            energy += 0.5 * np.dot(x, np.dot(self.Q, x))
            
            return energy
            
        def to_ising(self) -> 'IsingModel':
            """Convert QUBO to Ising model using transformation x = (s + 1) / 2."""
            n_spins = self.n_vars
            ising = IsingModel(n_spins)
            
            # Transform QUBO to Ising
            # x_i = (s_i + 1) / 2  =>  s_i = 2*x_i - 1
            # E_QUBO = sum_i h_i x_i + sum_{i,j} J_{ij} x_i x_j + offset
            # E_Ising = sum_i h'_i s_i + sum_{i,j} J'_{ij} s_i s_j + offset'
            
            for i in range(n_spins):
                # Linear terms
                h_i = self.linear[i] / 2.0
                for j in range(n_spins):
                    if i != j:
                        h_i += self.Q[i, j] / 4.0
                ising.set_field(i, h_i)
                
                # Quadratic terms
                for j in range(i + 1, n_spins):
                    J_ij = self.Q[i, j] / 4.0
                    ising.set_coupling(i, j, J_ij)
            
            # Offset term
            offset_correction = sum(self.linear) / 2.0 + sum(sum(self.Q)) / 4.0
            ising.set_offset(self.offset + offset_correction)
            
            return ising
            
        def solve_simulated_annealing(self, max_iter: int = 1000, 
                                    initial_temp: float = 10.0,
                                    final_temp: float = 0.01) -> Tuple[List[int], float]:
            """
            Solve QUBO using simulated annealing.
            
            Args:
                max_iter: Maximum number of iterations
                initial_temp: Initial temperature
                final_temp: Final temperature
                
            Returns:
                (best_assignment, best_energy)
            """
            import random
            
            # Initialize random solution
            current = [random.randint(0, 1) for _ in range(self.n_vars)]
            current_energy = self.energy(current)
            
            best = current.copy()
            best_energy = current_energy
            
            for iteration in range(max_iter):
                # Temperature schedule
                temp = initial_temp * (final_temp / initial_temp) ** (iteration / max_iter)
                
                # Propose a flip
                flip_var = random.randint(0, self.n_vars - 1)
                new_solution = current.copy()
                new_solution[flip_var] = 1 - new_solution[flip_var]
                
                new_energy = self.energy(new_solution)
                delta_energy = new_energy - current_energy
                
                # Accept or reject
                if delta_energy <= 0 or random.random() < np.exp(-delta_energy / temp):
                    current = new_solution
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy
            
            return best, best_energy
    
    class IsingModel:
        """Ising model implementation."""
        
        def __init__(self, n_spins: int):
            """
            Initialize an Ising model.
            
            Args:
                n_spins: Number of spins
            """
            self.n_spins = n_spins
            self.fields = np.zeros(n_spins)
            self.couplings = {}
            self.offset = 0.0
            
        def set_field(self, spin: int, field: float):
            """Set magnetic field for a spin."""
            if 0 <= spin < self.n_spins:
                self.fields[spin] = field
            else:
                raise ValueError(f"Spin {spin} out of range")
                
        def set_coupling(self, spin1: int, spin2: int, coupling: float):
            """Set coupling between two spins."""
            if not (0 <= spin1 < self.n_spins and 0 <= spin2 < self.n_spins):
                raise ValueError("Spins out of range")
            if spin1 != spin2:
                key = (min(spin1, spin2), max(spin1, spin2))
                self.couplings[key] = coupling
                
        def set_offset(self, offset: float):
            """Set constant offset term."""
            self.offset = offset
            
        def energy(self, spins: List[int]) -> float:
            """Calculate energy for given spin configuration."""
            s = np.array(spins)
            if len(s) != self.n_spins:
                raise ValueError(f"Spin array length {len(s)} != n_spins {self.n_spins}")
            
            energy = self.offset
            
            # Field terms
            energy -= np.dot(self.fields, s)
            
            # Coupling terms
            for (i, j), J in self.couplings.items():
                energy -= J * s[i] * s[j]
                
            return energy
            
        def to_qubo(self) -> QuboModel:
            """Convert Ising to QUBO model using transformation s = 2*x - 1."""
            qubo = QuboModel(self.n_spins)
            
            # Transform s_i = 2*x_i - 1  =>  x_i = (s_i + 1) / 2
            # E_Ising = -sum_i h_i s_i - sum_{i,j} J_{ij} s_i s_j + offset
            # E_QUBO = sum_i h'_i x_i + sum_{i,j} J'_{ij} x_i x_j + offset'
            
            for i in range(self.n_spins):
                # Linear terms: -h_i s_i = -h_i (2*x_i - 1) = -2*h_i*x_i + h_i
                qubo.add_linear(i, -2.0 * self.fields[i])
                
            for (i, j), J in self.couplings.items():
                # Coupling terms: -J s_i s_j = -J (2*x_i - 1)(2*x_j - 1)
                # = -J (4*x_i*x_j - 2*x_i - 2*x_j + 1)
                # = -4*J*x_i*x_j + 2*J*x_i + 2*J*x_j - J
                qubo.add_quadratic(i, j, -4.0 * J)
                qubo.add_linear(i, 2.0 * J)
                qubo.add_linear(j, 2.0 * J)
                
            # Offset correction
            offset_correction = sum(self.fields) - sum(self.couplings.values())
            qubo.set_offset(self.offset + offset_correction)
            
            return qubo
    
    class PenaltyOptimizer:
        """Penalty optimization for constrained problems."""
        
        def __init__(self, penalty_strength: float = 1.0):
            """
            Initialize penalty optimizer.
            
            Args:
                penalty_strength: Strength of penalty terms
            """
            self.penalty_strength = penalty_strength
            self.constraints = []
            
        def add_equality_constraint(self, variables: List[int], target: int):
            """Add equality constraint: sum(variables) = target."""
            self.constraints.append(('eq', variables, target))
            
        def add_inequality_constraint(self, variables: List[int], max_value: int):
            """Add inequality constraint: sum(variables) <= max_value."""
            self.constraints.append(('ineq', variables, max_value))
            
        def apply_penalties(self, qubo: QuboModel) -> QuboModel:
            """Apply penalty terms to QUBO model."""
            penalized = QuboModel(qubo.n_vars)
            
            # Copy original QUBO
            penalized.Q = qubo.Q.copy()
            penalized.linear = qubo.linear.copy()
            penalized.offset = qubo.offset
            
            # Add penalty terms
            for constraint_type, variables, target in self.constraints:
                if constraint_type == 'eq':
                    # Penalty: P * (sum(x_i) - target)^2
                    # = P * (sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2*target*sum_i x_i + target^2)
                    # For binary x_i: x_i^2 = x_i
                    
                    # Linear terms: P * (1 - 2*target) * x_i
                    for var in variables:
                        penalized.add_linear(var, self.penalty_strength * (1 - 2 * target))
                        
                    # Quadratic terms: 2*P * x_i * x_j for i < j
                    for i in range(len(variables)):
                        for j in range(i + 1, len(variables)):
                            penalized.add_quadratic(variables[i], variables[j], 
                                                  2 * self.penalty_strength)
                    
                    # Constant term
                    penalized.offset += self.penalty_strength * target ** 2
                    
                elif constraint_type == 'ineq':
                    # Simple penalty for inequality: P * max(0, sum(x_i) - max_value)^2
                    # Approximated by P * (sum(x_i) - max_value)^2 when violated
                    # This is a simplified implementation
                    pass
                    
            return penalized
    
    class LayoutAwareEmbedder:
        """Layout-aware graph embedding for quantum annealers."""
        
        def __init__(self, target_graph: str = "chimera"):
            """
            Initialize embedder.
            
            Args:
                target_graph: Target hardware graph type ("chimera", "pegasus", "zephyr")
            """
            self.target_graph = target_graph
            self.embedding_cache = {}
            
        def embed_problem(self, problem_graph: Dict[Tuple[int, int], float]) -> Dict[int, List[int]]:
            """
            Embed a problem graph onto the target hardware graph.
            
            Args:
                problem_graph: Dictionary of {(node1, node2): weight} edges
                
            Returns:
                Dictionary mapping logical variables to chains of physical qubits
            """
            # Simplified embedding - in practice this is a complex graph theory problem
            logical_vars = set()
            for edge in problem_graph.keys():
                logical_vars.update(edge)
            
            embedding = {}
            for i, var in enumerate(sorted(logical_vars)):
                # Simple 1-to-1 mapping for demonstration
                # Real embedders would create chains of physical qubits
                embedding[var] = [i]
                
            return embedding
            
        def generate_chimera_graph(self, m: int, n: int, t: int = 4) -> List[Tuple[int, int]]:
            """
            Generate edges for a Chimera graph.
            
            Args:
                m: Number of rows of unit cells
                n: Number of columns of unit cells  
                t: Number of qubits per side of unit cell
                
            Returns:
                List of (qubit1, qubit2) edges
            """
            edges = []
            
            for row in range(m):
                for col in range(n):
                    # Internal cell connections (bipartite)
                    cell_offset = (row * n + col) * 2 * t
                    for i in range(t):
                        for j in range(t):
                            left_qubit = cell_offset + i
                            right_qubit = cell_offset + t + j
                            edges.append((left_qubit, right_qubit))
                    
                    # External connections
                    if col < n - 1:  # Horizontal connections
                        for i in range(t):
                            curr_qubit = cell_offset + t + i
                            next_qubit = cell_offset + 2 * t + i
                            edges.append((curr_qubit, next_qubit))
                            
                    if row < m - 1:  # Vertical connections
                        for i in range(t):
                            curr_qubit = cell_offset + i
                            next_qubit = cell_offset + n * 2 * t + i
                            edges.append((curr_qubit, next_qubit))
            
            return edges
    
    class ChimeraGraph:
        """Chimera graph utilities."""
        
        @staticmethod
        def generate_edges(m: int, n: int, t: int = 4) -> List[Tuple[int, int]]:
            """Generate edges for a Chimera graph."""
            embedder = LayoutAwareEmbedder()
            return embedder.generate_chimera_graph(m, n, t)
            
        @staticmethod
        def get_adjacency_matrix(m: int, n: int, t: int = 4) -> np.ndarray:
            """Get adjacency matrix for Chimera graph."""
            edges = ChimeraGraph.generate_edges(m, n, t)
            max_node = max(max(edge) for edge in edges) + 1
            
            adj_matrix = np.zeros((max_node, max_node), dtype=int)
            for i, j in edges:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                
            return adj_matrix


class QUBOBuilder:
    """
    Helper class for building QUBO models.
    """
    
    def __init__(self, n_vars: int):
        """
        Initialize QUBO builder.
        
        Args:
            n_vars: Number of binary variables
        """
        self.model = QuboModel(n_vars)
        self.n_vars = n_vars
    
    def add_linear(self, var: int, coeff: float) -> 'QUBOBuilder':
        """Add linear term."""
        self.model.add_linear(var, coeff)
        return self
    
    def add_quadratic(self, var1: int, var2: int, coeff: float) -> 'QUBOBuilder':
        """Add quadratic term."""
        self.model.add_quadratic(var1, var2, coeff)
        return self
    
    def add_constraint(self, variables: List[int], coefficients: List[float], 
                      rhs: float, penalty: float = 1.0) -> 'QUBOBuilder':
        """
        Add equality constraint: sum(c_i * x_i) = rhs
        
        Args:
            variables: Variable indices
            coefficients: Coefficients for each variable
            rhs: Right-hand side value
            penalty: Penalty weight
        """
        # Convert to QUBO form: penalty * (sum - rhs)^2
        # Expand: penalty * (sum^2 - 2*sum*rhs + rhs^2)
        
        # Quadratic terms
        for i, (v1, c1) in enumerate(zip(variables, coefficients)):
            for j, (v2, c2) in enumerate(zip(variables, coefficients)):
                if i <= j:
                    coeff = penalty * c1 * c2
                    if i == j:
                        self.add_linear(v1, coeff)
                    else:
                        self.add_quadratic(v1, v2, coeff)
        
        # Linear terms from -2*sum*rhs
        for v, c in zip(variables, coefficients):
            self.add_linear(v, -2 * penalty * c * rhs)
        
        # Note: Constant term penalty * rhs^2 is ignored as it doesn't affect optimization
        
        return self
    
    def to_ising(self) -> Tuple[IsingModel, float]:
        """Convert to Ising model."""
        return self.model.to_ising()
    
    def get_model(self) -> QuboModel:
        """Get the QUBO model."""
        return self.model


class GraphEmbeddingHelper:
    """
    Helper class for graph embedding with penalty optimization.
    """
    
    def __init__(self, target_topology: str = "chimera", **kwargs):
        """
        Initialize graph embedding helper.
        
        Args:
            target_topology: Target hardware topology ('chimera', 'pegasus', etc.)
            **kwargs: Additional configuration options
        """
        self.embedder = LayoutAwareEmbedder(
            target_topology=target_topology,
            use_coordinates=kwargs.get('use_coordinates', True),
            chain_strength_factor=kwargs.get('chain_strength_factor', 1.2),
            metric=kwargs.get('metric', 'euclidean')
        )
        
        self.penalty_optimizer = PenaltyOptimizer(
            learning_rate=kwargs.get('learning_rate', 0.1),
            momentum=kwargs.get('momentum', 0.9),
            adaptive_strategy=kwargs.get('adaptive_strategy', 'break_frequency')
        )
    
    def embed_graph(self, source_edges: List[Tuple[int, int]], 
                   target_graph: Optional[List[Tuple[int, int]]] = None,
                   initial_chains: Optional[Dict[int, List[int]]] = None) -> Dict[int, List[int]]:
        """
        Find embedding for source graph.
        
        Args:
            source_edges: Edges in the source graph
            target_graph: Target hardware graph (auto-generated if None)
            initial_chains: Initial chain mapping (optional)
            
        Returns:
            Dictionary mapping logical qubits to physical qubit chains
        """
        if target_graph is None:
            # Generate default Chimera 16x16 graph
            target_graph = ChimeraGraph.generate_edges(16, 16, 4)
        
        embedding = self.embedder.find_embedding(source_edges, target_graph, initial_chains)
        return embedding
    
    def optimize_penalties(self, samples: List[Dict[str, Union[bool, float]]], 
                         chains: Dict[int, List[int]]) -> Dict[str, float]:
        """
        Optimize penalty weights based on sample results.
        
        Args:
            samples: List of sample results with 'chain_breaks' info
            chains: Current embedding chains
            
        Returns:
            Updated penalty weights
        """
        # Extract chain break information
        chain_breaks = []
        for chain_id, qubits in chains.items():
            broken = any(sample.get(f'chain_break_{chain_id}', False) for sample in samples)
            chain_breaks.append((chain_id, broken))
        
        # Update penalties
        penalties = self.penalty_optimizer.update_penalties(chain_breaks, None)
        return penalties
    
    def get_embedding_metrics(self) -> Dict[str, float]:
        """Get embedding quality metrics."""
        return self.embedder.get_metrics()


# Problem-specific helpers

def create_tsp_qubo(distances: np.ndarray, penalty: float = 10.0) -> QUBOBuilder:
    """
    Create QUBO for Traveling Salesman Problem.
    
    Args:
        distances: Distance matrix (n_cities x n_cities)
        penalty: Constraint penalty weight
        
    Returns:
        QUBOBuilder with TSP QUBO
    """
    n_cities = distances.shape[0]
    n_vars = n_cities * n_cities
    
    builder = QUBOBuilder(n_vars)
    
    # Objective: minimize total distance
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                for t in range(n_cities - 1):
                    # x_{i,t} * x_{j,t+1} * d_{i,j}
                    var1 = i * n_cities + t
                    var2 = j * n_cities + ((t + 1) % n_cities)
                    builder.add_quadratic(var1, var2, distances[i, j])
    
    # Constraint: each city visited exactly once
    for i in range(n_cities):
        variables = [i * n_cities + t for t in range(n_cities)]
        coefficients = [1.0] * n_cities
        builder.add_constraint(variables, coefficients, 1.0, penalty)
    
    # Constraint: each time slot has exactly one city
    for t in range(n_cities):
        variables = [i * n_cities + t for i in range(n_cities)]
        coefficients = [1.0] * n_cities
        builder.add_constraint(variables, coefficients, 1.0, penalty)
    
    return builder


def create_max_cut_qubo(edges: List[Tuple[int, int]], weights: Optional[List[float]] = None) -> QUBOBuilder:
    """
    Create QUBO for Max Cut problem.
    
    Args:
        edges: Graph edges
        weights: Edge weights (default: all 1.0)
        
    Returns:
        QUBOBuilder with Max Cut QUBO
    """
    # Find number of nodes
    n_nodes = max(max(u, v) for u, v in edges) + 1
    
    if weights is None:
        weights = [1.0] * len(edges)
    
    builder = QUBOBuilder(n_nodes)
    
    # For each edge (u, v), we want to maximize x_u + x_v - 2*x_u*x_v
    # This equals 1 when nodes are in different partitions, 0 otherwise
    for (u, v), w in zip(edges, weights):
        builder.add_linear(u, w)
        builder.add_linear(v, w)
        builder.add_quadratic(u, v, -2 * w)
    
    return builder


def create_graph_coloring_qubo(n_nodes: int, edges: List[Tuple[int, int]], 
                               n_colors: int, penalty: float = 10.0) -> QUBOBuilder:
    """
    Create QUBO for Graph Coloring problem.
    
    Args:
        n_nodes: Number of nodes
        edges: Graph edges
        n_colors: Number of colors
        penalty: Constraint penalty weight
        
    Returns:
        QUBOBuilder with Graph Coloring QUBO
    """
    n_vars = n_nodes * n_colors
    builder = QUBOBuilder(n_vars)
    
    # Constraint: each node has exactly one color
    for node in range(n_nodes):
        variables = [node * n_colors + c for c in range(n_colors)]
        coefficients = [1.0] * n_colors
        builder.add_constraint(variables, coefficients, 1.0, penalty)
    
    # Constraint: adjacent nodes have different colors
    for u, v in edges:
        for c in range(n_colors):
            var1 = u * n_colors + c
            var2 = v * n_colors + c
            builder.add_quadratic(var1, var2, penalty)
    
    return builder


# Example usage
def example_chimera_embedding():
    """Example: Embed a small graph on Chimera topology."""
    if not _NATIVE_AVAILABLE:
        print("Anneal features not available")
        return
    
    # Create a simple graph to embed
    source_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    
    # Create embedding helper
    helper = GraphEmbeddingHelper(target_topology="chimera")
    
    # Find embedding on 2x2 Chimera
    target_graph = ChimeraGraph.generate_edges(2, 2, 4)
    embedding = helper.embed_graph(source_edges, target_graph)
    
    print("Embedding found:")
    for logical, chain in embedding.items():
        print(f"  Logical qubit {logical} -> Physical qubits {chain}")
    
    # Get metrics
    metrics = helper.get_embedding_metrics()
    print("\nEmbedding metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return helper, embedding


__all__ = [
    'QuboModel',
    'IsingModel',
    'PenaltyOptimizer',
    'LayoutAwareEmbedder',
    'ChimeraGraph',
    'QUBOBuilder',
    'GraphEmbeddingHelper',
    'create_tsp_qubo',
    'create_max_cut_qubo',
    'create_graph_coloring_qubo',
    'example_chimera_embedding',
]