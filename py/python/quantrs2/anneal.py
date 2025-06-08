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
    # Provide stubs
    class QuboModel:
        """QUBO model (stub)"""
        def __init__(self, n_vars: int):
            raise ImportError("Anneal features not available. Install with: pip install quantrs2[anneal]")
    
    class IsingModel:
        """Ising model (stub)"""
        def __init__(self, n_spins: int):
            raise ImportError("Anneal features not available")
    
    class PenaltyOptimizer:
        """Penalty optimizer (stub)"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Anneal features not available")
    
    class LayoutAwareEmbedder:
        """Layout-aware embedder (stub)"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Anneal features not available")
    
    class ChimeraGraph:
        """Chimera graph utilities (stub)"""
        @staticmethod
        def generate_edges(m: int, n: int, t: int):
            raise ImportError("Anneal features not available")


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