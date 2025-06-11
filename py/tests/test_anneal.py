#!/usr/bin/env python3
"""
Test suite for quantum annealing functionality.
"""

import pytest
import numpy as np

try:
    from quantrs2.anneal import (
        QuboModel, IsingModel, PenaltyOptimizer, LayoutAwareEmbedder,
        ChimeraGraph, QUBOBuilder, GraphEmbeddingHelper,
        create_tsp_qubo, create_max_cut_qubo, create_graph_coloring_qubo,
        example_chimera_embedding
    )
    HAS_ANNEAL = True
except ImportError:
    HAS_ANNEAL = False


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestQUBOBuilder:
    """Test QUBOBuilder functionality."""
    
    def test_qubo_builder_initialization(self):
        """Test QUBO builder initialization."""
        builder = QUBOBuilder(n_vars=5)
        assert builder.n_vars == 5
        assert builder.model is not None
    
    def test_add_linear_terms(self):
        """Test adding linear terms to QUBO."""
        builder = QUBOBuilder(n_vars=3)
        
        # Add linear terms
        builder.add_linear(0, 1.5)
        builder.add_linear(1, -2.0)
        builder.add_linear(2, 0.5)
        
        # Should not raise exceptions
        model = builder.get_model()
        assert model is not None
    
    def test_add_quadratic_terms(self):
        """Test adding quadratic terms to QUBO."""
        builder = QUBOBuilder(n_vars=4)
        
        # Add quadratic terms
        builder.add_quadratic(0, 1, 1.0)
        builder.add_quadratic(1, 2, -0.5)
        builder.add_quadratic(2, 3, 2.0)
        
        model = builder.get_model()
        assert model is not None
    
    def test_add_constraints(self):
        """Test adding constraints to QUBO."""
        builder = QUBOBuilder(n_vars=4)
        
        # Add equality constraint: x0 + x1 + x2 = 1
        variables = [0, 1, 2]
        coefficients = [1.0, 1.0, 1.0]
        builder.add_constraint(variables, coefficients, 1.0, penalty=5.0)
        
        model = builder.get_model()
        assert model is not None
    
    def test_method_chaining(self):
        """Test that QUBO builder methods can be chained."""
        builder = QUBOBuilder(n_vars=3)
        
        # Chain multiple operations
        result = (builder
                 .add_linear(0, 1.0)
                 .add_linear(1, 2.0)
                 .add_quadratic(0, 1, -1.0)
                 .add_constraint([0, 1], [1.0, 1.0], 1.0, penalty=3.0))
        
        assert result is builder  # Should return self for chaining
        
        model = builder.get_model()
        assert model is not None
    
    def test_to_ising_conversion(self):
        """Test conversion from QUBO to Ising model."""
        builder = QUBOBuilder(n_vars=2)
        builder.add_linear(0, 1.0)
        builder.add_quadratic(0, 1, 2.0)
        
        ising_model, offset = builder.to_ising()
        
        assert ising_model is not None
        assert isinstance(offset, (int, float))
    
    def test_empty_qubo(self):
        """Test QUBO with no terms."""
        builder = QUBOBuilder(n_vars=2)
        
        # No terms added
        model = builder.get_model()
        assert model is not None
        
        # Should still be able to convert to Ising
        ising_model, offset = builder.to_ising()
        assert ising_model is not None


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestGraphEmbeddingHelper:
    """Test GraphEmbeddingHelper functionality."""
    
    def test_embedding_helper_initialization(self):
        """Test embedding helper initialization."""
        helper = GraphEmbeddingHelper(target_topology="chimera")
        
        assert helper.embedder is not None
        assert helper.penalty_optimizer is not None
    
    def test_embedding_helper_with_parameters(self):
        """Test embedding helper with custom parameters."""
        helper = GraphEmbeddingHelper(
            target_topology="chimera",
            use_coordinates=False,
            chain_strength_factor=1.5,
            learning_rate=0.05,
            momentum=0.8
        )
        
        assert helper.embedder is not None
        assert helper.penalty_optimizer is not None
    
    def test_embed_simple_graph(self):
        """Test embedding a simple graph."""
        helper = GraphEmbeddingHelper(target_topology="chimera")
        
        # Simple triangle graph
        source_edges = [(0, 1), (1, 2), (2, 0)]
        
        # Generate target graph
        target_graph = ChimeraGraph.generate_edges(4, 4, 4)
        
        embedding = helper.embed_graph(source_edges, target_graph)
        
        assert isinstance(embedding, dict)
        # Should have embeddings for all logical qubits (0, 1, 2)
        for logical_qubit in [0, 1, 2]:
            assert logical_qubit in embedding
            assert isinstance(embedding[logical_qubit], list)
            assert len(embedding[logical_qubit]) > 0
    
    def test_embed_graph_without_target(self):
        """Test embedding with default target graph."""
        helper = GraphEmbeddingHelper(target_topology="chimera")
        
        # Simple path graph
        source_edges = [(0, 1), (1, 2), (2, 3)]
        
        # Let helper generate default target graph
        embedding = helper.embed_graph(source_edges)
        
        assert isinstance(embedding, dict)
        for logical_qubit in [0, 1, 2, 3]:
            assert logical_qubit in embedding
    
    def test_optimize_penalties(self):
        """Test penalty optimization."""
        helper = GraphEmbeddingHelper()
        
        # Mock sample data with chain break information
        samples = [
            {'chain_break_0': False, 'chain_break_1': True},
            {'chain_break_0': False, 'chain_break_1': False},
            {'chain_break_0': True, 'chain_break_1': False}
        ]
        
        chains = {0: [0, 1], 1: [2, 3]}
        
        penalties = helper.optimize_penalties(samples, chains)
        
        assert isinstance(penalties, dict)
        # Should contain penalty information
    
    def test_get_embedding_metrics(self):
        """Test getting embedding metrics."""
        helper = GraphEmbeddingHelper()
        
        metrics = helper.get_embedding_metrics()
        
        assert isinstance(metrics, dict)
        # Metrics should contain numerical values


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestChimeraGraph:
    """Test ChimeraGraph functionality."""
    
    def test_generate_chimera_edges(self):
        """Test generating Chimera graph edges."""
        # Generate a small Chimera graph
        edges = ChimeraGraph.generate_edges(2, 2, 4)
        
        assert isinstance(edges, list)
        assert len(edges) > 0
        
        # Each edge should be a tuple of two integers
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], int)
            assert isinstance(edge[1], int)
    
    def test_chimera_edge_count(self):
        """Test that Chimera graphs have expected properties."""
        # Test different sizes
        for m, n, t in [(1, 1, 4), (2, 1, 4), (2, 2, 4)]:
            edges = ChimeraGraph.generate_edges(m, n, t)
            assert isinstance(edges, list)
            
            # Should have some edges
            assert len(edges) > 0
    
    def test_chimera_edge_validity(self):
        """Test that generated edges are valid."""
        edges = ChimeraGraph.generate_edges(2, 2, 4)
        
        # All edge endpoints should be non-negative
        for u, v in edges:
            assert u >= 0
            assert v >= 0
            assert u != v  # No self-loops


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestTSPQUBO:
    """Test TSP QUBO creation."""
    
    def test_create_tsp_qubo_small(self):
        """Test creating QUBO for small TSP instance."""
        # 3-city TSP
        distances = np.array([
            [0, 2, 3],
            [2, 0, 1],
            [3, 1, 0]
        ])
        
        builder = create_tsp_qubo(distances, penalty=10.0)
        
        assert isinstance(builder, QUBOBuilder)
        assert builder.n_vars == 9  # 3 cities * 3 time slots
        
        model = builder.get_model()
        assert model is not None
    
    def test_create_tsp_qubo_different_sizes(self):
        """Test TSP QUBO creation for different sizes."""
        for n_cities in [3, 4, 5]:
            # Create random distance matrix
            distances = np.random.rand(n_cities, n_cities)
            # Make symmetric and zero diagonal
            distances = (distances + distances.T) / 2
            np.fill_diagonal(distances, 0)
            
            builder = create_tsp_qubo(distances, penalty=5.0)
            
            assert builder.n_vars == n_cities * n_cities
            
            model = builder.get_model()
            assert model is not None
    
    def test_tsp_qubo_to_ising(self):
        """Test converting TSP QUBO to Ising."""
        distances = np.array([[0, 1], [1, 0]])  # 2-city (trivial)
        
        builder = create_tsp_qubo(distances)
        ising_model, offset = builder.to_ising()
        
        assert ising_model is not None
        assert isinstance(offset, (int, float))


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestMaxCutQUBO:
    """Test Max Cut QUBO creation."""
    
    def test_create_max_cut_qubo_simple(self):
        """Test creating QUBO for simple Max Cut instance."""
        # Triangle graph
        edges = [(0, 1), (1, 2), (2, 0)]
        
        builder = create_max_cut_qubo(edges)
        
        assert isinstance(builder, QUBOBuilder)
        assert builder.n_vars == 3  # 3 nodes
        
        model = builder.get_model()
        assert model is not None
    
    def test_create_max_cut_qubo_with_weights(self):
        """Test creating weighted Max Cut QUBO."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        weights = [1.0, 2.0, 1.5, 0.5]
        
        builder = create_max_cut_qubo(edges, weights)
        
        assert builder.n_vars == 4  # 4 nodes
        
        model = builder.get_model()
        assert model is not None
    
    def test_max_cut_qubo_path_graph(self):
        """Test Max Cut QUBO for path graph."""
        # Path: 0-1-2-3-4
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        builder = create_max_cut_qubo(edges)
        
        assert builder.n_vars == 5  # 5 nodes
        
        # Convert to Ising
        ising_model, offset = builder.to_ising()
        assert ising_model is not None
    
    def test_max_cut_edge_cases(self):
        """Test Max Cut QUBO edge cases."""
        # Single edge
        edges = [(0, 1)]
        builder = create_max_cut_qubo(edges)
        assert builder.n_vars == 2
        
        # Self-contained component
        edges = [(0, 1), (2, 3)]
        builder = create_max_cut_qubo(edges)
        assert builder.n_vars == 4


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestGraphColoringQUBO:
    """Test Graph Coloring QUBO creation."""
    
    def test_create_graph_coloring_qubo_simple(self):
        """Test creating QUBO for simple graph coloring."""
        # Triangle graph, 3 colors
        n_nodes = 3
        edges = [(0, 1), (1, 2), (2, 0)]
        n_colors = 3
        
        builder = create_graph_coloring_qubo(n_nodes, edges, n_colors, penalty=10.0)
        
        assert isinstance(builder, QUBOBuilder)
        assert builder.n_vars == 9  # 3 nodes * 3 colors
        
        model = builder.get_model()
        assert model is not None
    
    def test_graph_coloring_different_parameters(self):
        """Test graph coloring with different parameters."""
        test_cases = [
            (3, [(0, 1), (1, 2)], 2),  # Path graph, 2 colors
            (4, [(0, 1), (1, 2), (2, 3), (3, 0)], 2),  # Cycle, 2 colors
            (4, [(0, 1), (0, 2), (0, 3)], 3),  # Star graph, 3 colors
        ]
        
        for n_nodes, edges, n_colors in test_cases:
            builder = create_graph_coloring_qubo(n_nodes, edges, n_colors)
            
            assert builder.n_vars == n_nodes * n_colors
            
            model = builder.get_model()
            assert model is not None
    
    def test_graph_coloring_to_ising(self):
        """Test converting graph coloring QUBO to Ising."""
        n_nodes = 2
        edges = [(0, 1)]
        n_colors = 2
        
        builder = create_graph_coloring_qubo(n_nodes, edges, n_colors)
        ising_model, offset = builder.to_ising()
        
        assert ising_model is not None
        assert isinstance(offset, (int, float))


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestExampleFunctions:
    """Test example functions."""
    
    def test_example_chimera_embedding(self):
        """Test Chimera embedding example."""
        try:
            result = example_chimera_embedding()
            
            # Should return helper and embedding
            if result is not None:
                helper, embedding = result
                assert helper is not None
                assert isinstance(embedding, dict)
        except ImportError:
            # Expected when anneal features are not available
            pytest.skip("Anneal features not available in example")


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestAnnealIntegration:
    """Test integration between annealing components."""
    
    def test_qubo_to_ising_pipeline(self):
        """Test full QUBO to Ising conversion pipeline."""
        # Create a QUBO
        builder = QUBOBuilder(n_vars=3)
        builder.add_linear(0, 1.0).add_linear(1, -1.0).add_quadratic(0, 1, 2.0)
        
        # Convert to Ising
        ising_model, offset = builder.to_ising()
        
        assert ising_model is not None
        assert isinstance(offset, (int, float))
    
    def test_embedding_with_optimization_problem(self):
        """Test embedding with a real optimization problem."""
        # Create Max Cut problem
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        builder = create_max_cut_qubo(edges)
        
        # Get the graph for embedding
        helper = GraphEmbeddingHelper(target_topology="chimera")
        
        # The source edges are the same as the problem edges
        target_graph = ChimeraGraph.generate_edges(2, 2, 4)
        embedding = helper.embed_graph(edges, target_graph)
        
        assert isinstance(embedding, dict)
        # Should have embeddings for all nodes in the Max Cut problem
        for node in range(4):  # 4 nodes in the graph
            assert node in embedding
    
    def test_multiple_problems_workflow(self):
        """Test workflow with multiple optimization problems."""
        problems = []
        
        # TSP problem
        distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        tsp_builder = create_tsp_qubo(distances)
        problems.append(("TSP", tsp_builder))
        
        # Max Cut problem
        edges = [(0, 1), (1, 2), (2, 0)]
        max_cut_builder = create_max_cut_qubo(edges)
        problems.append(("MaxCut", max_cut_builder))
        
        # Graph Coloring problem
        gc_builder = create_graph_coloring_qubo(3, edges, 3)
        problems.append(("GraphColoring", gc_builder))
        
        # All problems should convert to Ising
        for name, builder in problems:
            model = builder.get_model()
            assert model is not None
            
            ising_model, offset = builder.to_ising()
            assert ising_model is not None
            assert isinstance(offset, (int, float))


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestAnnealErrorHandling:
    """Test error handling in annealing components."""
    
    def test_qubo_builder_invalid_variables(self):
        """Test QUBO builder with invalid variable indices."""
        builder = QUBOBuilder(n_vars=3)
        
        # Try to add terms for variables outside the range
        try:
            builder.add_linear(5, 1.0)  # Variable 5 doesn't exist
            # Some implementations might handle this gracefully
        except (IndexError, ValueError):
            # This is expected behavior
            pass
    
    def test_embedding_empty_graph(self):
        """Test embedding with empty graphs."""
        helper = GraphEmbeddingHelper()
        
        # Empty source graph
        try:
            embedding = helper.embed_graph([])
            assert isinstance(embedding, dict)
        except (ValueError, IndexError):
            # Empty graphs might not be supported
            pass
    
    def test_invalid_distance_matrix(self):
        """Test TSP with invalid distance matrix."""
        # Non-square matrix
        try:
            distances = np.array([[0, 1], [1, 0], [2, 1]])  # 3x2 matrix
            builder = create_tsp_qubo(distances)
            # Might handle gracefully or raise an error
        except (ValueError, IndexError):
            # Expected for invalid input
            pass
    
    def test_negative_parameters(self):
        """Test handling of negative parameters."""
        # Negative number of variables
        try:
            builder = QUBOBuilder(n_vars=-1)
            # Should either handle gracefully or raise error
        except ValueError:
            # Expected behavior
            pass
        
        # Negative number of colors
        try:
            builder = create_graph_coloring_qubo(3, [(0, 1)], -1)
        except ValueError:
            # Expected behavior
            pass


@pytest.mark.skipif(not HAS_ANNEAL, reason="anneal module not available")
class TestAnnealPerformance:
    """Test performance characteristics of annealing components."""
    
    def test_large_qubo_creation(self):
        """Test creating large QUBO models."""
        import time
        
        start_time = time.time()
        
        # Create moderately large QUBO
        builder = QUBOBuilder(n_vars=50)
        
        # Add many terms
        for i in range(50):
            builder.add_linear(i, np.random.randn())
        
        for i in range(100):
            var1 = np.random.randint(0, 50)
            var2 = np.random.randint(0, 50)
            if var1 != var2:
                builder.add_quadratic(var1, var2, np.random.randn())
        
        model = builder.get_model()
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert model is not None
    
    def test_embedding_performance(self):
        """Test embedding performance for moderately sized graphs."""
        import time
        
        # Create a moderately connected graph
        edges = []
        n_nodes = 10
        for i in range(n_nodes):
            for j in range(i + 1, min(i + 3, n_nodes)):
                edges.append((i, j))
        
        helper = GraphEmbeddingHelper()
        
        start_time = time.time()
        
        # Generate target and find embedding
        target_graph = ChimeraGraph.generate_edges(4, 4, 4)
        embedding = helper.embed_graph(edges, target_graph)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds max
        assert isinstance(embedding, dict)
    
    def test_multiple_problems_performance(self):
        """Test performance when creating multiple problems."""
        import time
        
        start_time = time.time()
        
        # Create multiple optimization problems
        for i in range(5):
            # Different TSP instances
            n_cities = 4 + i
            distances = np.random.rand(n_cities, n_cities)
            distances = (distances + distances.T) / 2
            np.fill_diagonal(distances, 0)
            
            builder = create_tsp_qubo(distances)
            model = builder.get_model()
            assert model is not None
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 3.0


if __name__ == "__main__":
    pytest.main([__file__])