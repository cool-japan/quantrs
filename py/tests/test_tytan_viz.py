#!/usr/bin/env python3
"""
Test suite for Tytan visualization functionality.
"""

import pytest
import numpy as np

try:
    from quantrs2.tytan_viz import (
        SampleResult, EnergyLandscapeVisualizer, SolutionAnalyzer,
        ProblemVisualizer, ConvergenceAnalyzer, VisualizationHelper,
        ProblemSpecificVisualizer, analyze_convergence,
        example_visualization_workflow
    )
    HAS_TYTAN_VIZ = True
except ImportError:
    HAS_TYTAN_VIZ = False


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestSampleResult:
    """Test SampleResult functionality."""
    
    def test_sample_result_creation(self):
        """Test creating sample results."""
        assignments = {'x0': True, 'x1': False, 'x2': True}
        energy = -5.2
        occurrences = 10
        
        result = SampleResult(assignments, energy, occurrences)
        
        assert result.assignments == assignments
        assert result.energy == energy
        assert result.occurrences == occurrences
    
    def test_sample_result_default_occurrences(self):
        """Test sample result with default occurrences."""
        assignments = {'x0': False, 'x1': True}
        energy = 1.5
        
        result = SampleResult(assignments, energy)
        
        assert result.assignments == assignments
        assert result.energy == energy
        assert result.occurrences == 1  # Default value
    
    def test_sample_result_empty_assignments(self):
        """Test sample result with empty assignments."""
        result = SampleResult({}, 0.0, 1)
        
        assert result.assignments == {}
        assert result.energy == 0.0
        assert result.occurrences == 1
    
    def test_sample_result_different_data_types(self):
        """Test sample result with different data types."""
        # String keys, boolean values
        assignments1 = {'var_a': True, 'var_b': False}
        result1 = SampleResult(assignments1, -1.0, 5)
        assert result1.assignments == assignments1
        
        # Integer keys (converted to strings)
        assignments2 = {0: True, 1: False, 2: True}
        result2 = SampleResult(assignments2, 2.5, 3)
        assert result2.assignments == assignments2


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestVisualizationHelper:
    """Test VisualizationHelper functionality."""
    
    def create_sample_results(self, n_samples=20):
        """Helper to create sample results for testing."""
        results = []
        for i in range(n_samples):
            assignments = {
                'x0': i % 2 == 0,
                'x1': i % 3 == 0,
                'x2': i % 5 == 0,
                'x3': i % 7 == 0,
            }
            energy = -10.0 + i * 0.1 + np.random.normal(0, 0.5)
            occurrences = np.random.randint(1, 5)
            results.append(SampleResult(assignments, energy, occurrences))
        return results
    
    def test_visualization_helper_initialization(self):
        """Test visualization helper initialization."""
        results = self.create_sample_results(10)
        helper = VisualizationHelper(results)
        
        assert helper.results == results
        assert helper._energy_data is None
        assert helper._solution_data is None
    
    def test_prepare_energy_landscape(self):
        """Test preparing energy landscape data."""
        results = self.create_sample_results(15)
        helper = VisualizationHelper(results)
        
        data = helper.prepare_energy_landscape(num_bins=10, compute_kde=True)
        
        assert isinstance(data, dict)
        assert helper._energy_data is not None
        # Should contain various data fields for visualization
    
    def test_prepare_energy_landscape_parameters(self):
        """Test energy landscape with different parameters."""
        results = self.create_sample_results(25)
        helper = VisualizationHelper(results)
        
        # Test with different parameters
        data1 = helper.prepare_energy_landscape(num_bins=5, compute_kde=False)
        assert isinstance(data1, dict)
        
        data2 = helper.prepare_energy_landscape(num_bins=20, compute_kde=True, kde_points=100)
        assert isinstance(data2, dict)
    
    def test_analyze_solutions(self):
        """Test solution analysis."""
        results = self.create_sample_results(20)
        helper = VisualizationHelper(results)
        
        data = helper.analyze_solutions(compute_correlations=True, compute_pca=True)
        
        assert isinstance(data, dict)
        assert helper._solution_data is not None
    
    def test_analyze_solutions_parameters(self):
        """Test solution analysis with different parameters."""
        results = self.create_sample_results(30)
        helper = VisualizationHelper(results)
        
        # Test without correlations and PCA
        data1 = helper.analyze_solutions(compute_correlations=False, compute_pca=False)
        assert isinstance(data1, dict)
        
        # Test with different PCA components
        data2 = helper.analyze_solutions(compute_pca=True, n_components=3)
        assert isinstance(data2, dict)
    
    def test_plot_energy_landscape(self):
        """Test plotting energy landscape."""
        pytest.importorskip("matplotlib")
        
        results = self.create_sample_results(15)
        helper = VisualizationHelper(results)
        
        # Should not raise exceptions
        helper.plot_energy_landscape(show=False)
    
    def test_plot_solution_heatmap(self):
        """Test plotting solution heatmap."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        
        results = self.create_sample_results(20)
        helper = VisualizationHelper(results)
        
        # Should not raise exceptions
        helper.plot_solution_heatmap(show=False)
    
    def test_get_variable_statistics(self):
        """Test getting variable statistics."""
        results = self.create_sample_results(25)
        helper = VisualizationHelper(results)
        
        stats = helper.get_variable_statistics()
        
        assert isinstance(stats, dict)
        # Should have statistics for each variable
        for var_name in ['x0', 'x1', 'x2', 'x3']:
            assert var_name in stats
            var_stats = stats[var_name]
            assert 'frequency' in var_stats
            assert 'probability_one' in var_stats
            assert 'probability_zero' in var_stats
            assert 0 <= var_stats['frequency'] <= 1
            assert 0 <= var_stats['probability_one'] <= 1
            assert 0 <= var_stats['probability_zero'] <= 1
    
    def test_visualization_workflow(self):
        """Test complete visualization workflow."""
        results = self.create_sample_results(30)
        helper = VisualizationHelper(results)
        
        # Prepare energy landscape
        energy_data = helper.prepare_energy_landscape()
        assert isinstance(energy_data, dict)
        
        # Analyze solutions
        solution_data = helper.analyze_solutions()
        assert isinstance(solution_data, dict)
        
        # Get statistics
        stats = helper.get_variable_statistics()
        assert isinstance(stats, dict)
        
        # Data should be cached
        assert helper._energy_data is not None
        assert helper._solution_data is not None


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestProblemSpecificVisualizer:
    """Test ProblemSpecificVisualizer functionality."""
    
    def test_visualize_tsp_solution(self):
        """Test TSP solution visualization."""
        pytest.importorskip("matplotlib")
        
        # Create a TSP solution (4 cities)
        assignments = {
            'x_0_0': True, 'x_0_1': False, 'x_0_2': False, 'x_0_3': False,
            'x_1_0': False, 'x_1_1': True, 'x_1_2': False, 'x_1_3': False,
            'x_2_0': False, 'x_2_1': False, 'x_2_2': True, 'x_2_3': False,
            'x_3_0': False, 'x_3_1': False, 'x_3_2': False, 'x_3_3': True,
        }
        result = SampleResult(assignments, -10.0, 1)
        
        # City coordinates
        cities = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        
        tour_length = ProblemSpecificVisualizer.visualize_tsp_solution(
            result, cities, show=False
        )
        
        assert isinstance(tour_length, float)
        assert tour_length >= 0
    
    def test_visualize_graph_coloring(self):
        """Test graph coloring visualization."""
        pytest.importorskip("matplotlib")
        
        # Create a graph coloring solution (3 nodes, 3 colors)
        assignments = {
            'x_0_0': True, 'x_0_1': False, 'x_0_2': False,  # Node 0 -> Color 0
            'x_1_0': False, 'x_1_1': True, 'x_1_2': False,  # Node 1 -> Color 1
            'x_2_0': False, 'x_2_1': False, 'x_2_2': True,  # Node 2 -> Color 2
        }
        result = SampleResult(assignments, -5.0, 1)
        
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle graph
        n_colors = 3
        
        node_colors, conflicts = ProblemSpecificVisualizer.visualize_graph_coloring(
            result, edges, n_colors, show=False
        )
        
        assert isinstance(node_colors, list)
        assert isinstance(conflicts, list)
        assert len(node_colors) == 3  # 3 nodes
    
    def test_tsp_different_sizes(self):
        """Test TSP visualization with different problem sizes."""
        pytest.importorskip("matplotlib")
        
        # Test different numbers of cities
        for n_cities in [3, 4, 5]:
            # Create identity tour (0 -> 1 -> 2 -> ... -> 0)
            assignments = {}
            for i in range(n_cities):
                for t in range(n_cities):
                    assignments[f'x_{i}_{t}'] = (i == t)
            
            result = SampleResult(assignments, 0.0, 1)
            
            # Generate random city coordinates
            cities = [(np.random.rand(), np.random.rand()) for _ in range(n_cities)]
            
            tour_length = ProblemSpecificVisualizer.visualize_tsp_solution(
                result, cities, show=False
            )
            
            assert isinstance(tour_length, float)
            assert tour_length >= 0
    
    def test_graph_coloring_different_graphs(self):
        """Test graph coloring with different graph types."""
        pytest.importorskip("matplotlib")
        
        test_cases = [
            # Path graph: 0-1-2
            ([(0, 1), (1, 2)], 3, 2),
            # Complete graph K3
            ([(0, 1), (1, 2), (2, 0)], 3, 3),
            # Star graph
            ([(0, 1), (0, 2), (0, 3)], 4, 2),
        ]
        
        for edges, n_nodes, n_colors in test_cases:
            # Create valid coloring
            assignments = {}
            for node in range(n_nodes):
                for color in range(n_colors):
                    assignments[f'x_{node}_{color}'] = (node % n_colors == color)
            
            result = SampleResult(assignments, 0.0, 1)
            
            node_colors, conflicts = ProblemSpecificVisualizer.visualize_graph_coloring(
                result, edges, n_colors, show=False
            )
            
            assert len(node_colors) == n_nodes


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestConvergenceAnalysis:
    """Test convergence analysis functionality."""
    
    def create_iteration_results(self, n_iterations=10, n_samples_per_iter=20):
        """Helper to create iteration results for testing."""
        iteration_results = []
        
        for iter_idx in range(n_iterations):
            results = []
            base_energy = -10.0 + iter_idx * 0.5  # Gradually improving
            
            for i in range(n_samples_per_iter):
                assignments = {
                    'x0': i % 2 == 0,
                    'x1': i % 3 == 0,
                }
                energy = base_energy + np.random.normal(0, 1.0)
                results.append(SampleResult(assignments, energy, 1))
            
            iteration_results.append(results)
        
        return iteration_results
    
    def test_analyze_convergence(self):
        """Test convergence analysis."""
        iteration_results = self.create_iteration_results(8, 15)
        
        data = analyze_convergence(iteration_results, ma_window=3, show=False)
        
        assert isinstance(data, dict)
        # Should contain convergence metrics
        expected_keys = ['iterations', 'best_energies', 'avg_energies', 'std_devs']
        for key in expected_keys:
            assert key in data
            assert isinstance(data[key], np.ndarray)
    
    def test_analyze_convergence_with_plot(self):
        """Test convergence analysis with plotting."""
        pytest.importorskip("matplotlib")
        
        iteration_results = self.create_iteration_results(6, 10)
        
        data = analyze_convergence(iteration_results, ma_window=2, show=False)
        
        assert isinstance(data, dict)
        # Should complete without errors
    
    def test_convergence_different_parameters(self):
        """Test convergence analysis with different parameters."""
        iteration_results = self.create_iteration_results(12, 25)
        
        # Test different moving average windows
        for ma_window in [2, 4, 6]:
            data = analyze_convergence(iteration_results, ma_window=ma_window, show=False)
            assert isinstance(data, dict)
    
    def test_convergence_single_iteration(self):
        """Test convergence with single iteration."""
        # Edge case: only one iteration
        single_iteration = [self.create_iteration_results(1, 10)[0]]
        
        data = analyze_convergence(single_iteration, ma_window=1, show=False)
        
        assert isinstance(data, dict)
        assert len(data['iterations']) == 1
    
    def test_convergence_empty_iterations(self):
        """Test convergence with empty iterations."""
        # Edge case: empty iteration list
        try:
            data = analyze_convergence([], show=False)
            # Some implementations might handle this gracefully
            assert isinstance(data, dict)
        except (ValueError, IndexError):
            # This is also acceptable behavior
            pass


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestEnergyLandscapeVisualizer:
    """Test EnergyLandscapeVisualizer functionality."""
    
    def test_prepare_landscape(self):
        """Test energy landscape preparation."""
        # Create sample results
        results = []
        for i in range(30):
            assignments = {'x0': i % 2 == 0, 'x1': i % 3 == 0}
            energy = -5.0 + i * 0.1 + np.random.normal(0, 0.3)
            results.append(SampleResult(assignments, energy, 1))
        
        data = EnergyLandscapeVisualizer.prepare_landscape(
            results, num_bins=10, compute_kde=True, kde_points=50
        )
        
        assert isinstance(data, dict)
        # Should contain visualization data


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestSolutionAnalyzer:
    """Test SolutionAnalyzer functionality."""
    
    def test_analyze_distribution(self):
        """Test solution distribution analysis."""
        # Create sample results
        results = []
        for i in range(40):
            assignments = {
                'x0': i % 2 == 0,
                'x1': i % 3 == 0,
                'x2': i % 5 == 0,
            }
            energy = np.random.normal(0, 1)
            results.append(SampleResult(assignments, energy, 1))
        
        data = SolutionAnalyzer.analyze_distribution(
            results, compute_correlations=True, compute_pca=True, n_components=2
        )
        
        assert isinstance(data, dict)
        # Should contain analysis results


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestExampleFunctions:
    """Test example functions."""
    
    def test_example_visualization_workflow(self):
        """Test example visualization workflow."""
        try:
            result = example_visualization_workflow()
            
            # Should return a VisualizationHelper or None
            if result is not None:
                assert isinstance(result, VisualizationHelper)
        except ImportError:
            # Expected when tytan features are not available
            pytest.skip("Tytan features not available in example")


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestTytanVizIntegration:
    """Test integration between Tytan visualization components."""
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline."""
        # Create comprehensive sample results
        results = []
        for i in range(50):
            assignments = {
                'x0': i % 2 == 0,
                'x1': i % 3 == 0,
                'x2': i % 5 == 0,
                'x3': i % 7 == 0,
            }
            energy = -15.0 + i * 0.2 + np.random.normal(0, 1.0)
            occurrences = np.random.randint(1, 4)
            results.append(SampleResult(assignments, energy, occurrences))
        
        # Create visualization helper
        helper = VisualizationHelper(results)
        
        # Prepare energy landscape
        energy_data = helper.prepare_energy_landscape(num_bins=15, compute_kde=True)
        assert isinstance(energy_data, dict)
        
        # Analyze solutions
        solution_data = helper.analyze_solutions(compute_correlations=True, compute_pca=True)
        assert isinstance(solution_data, dict)
        
        # Get variable statistics
        stats = helper.get_variable_statistics()
        assert isinstance(stats, dict)
        assert len(stats) == 4  # 4 variables
        
        # Verify all data is properly connected
        assert helper._energy_data is not None
        assert helper._solution_data is not None
    
    def test_visualization_with_different_problem_types(self):
        """Test visualization with different optimization problems."""
        # TSP-like results
        tsp_results = []
        for i in range(20):
            # 3-city TSP variables
            assignments = {}
            for city in range(3):
                for time in range(3):
                    assignments[f'x_{city}_{time}'] = (city == time)  # Identity permutation
            energy = np.random.normal(-5.0, 0.5)
            tsp_results.append(SampleResult(assignments, energy, 1))
        
        helper_tsp = VisualizationHelper(tsp_results)
        energy_data_tsp = helper_tsp.prepare_energy_landscape()
        assert isinstance(energy_data_tsp, dict)
        
        # Max Cut-like results
        maxcut_results = []
        for i in range(25):
            assignments = {f'x{j}': j % 2 == i % 2 for j in range(5)}
            energy = np.random.normal(0.0, 1.0)
            maxcut_results.append(SampleResult(assignments, energy, 1))
        
        helper_maxcut = VisualizationHelper(maxcut_results)
        solution_data_maxcut = helper_maxcut.analyze_solutions()
        assert isinstance(solution_data_maxcut, dict)
    
    def test_convergence_with_visualization(self):
        """Test convergence analysis integrated with visualization."""
        # Create iteration results that show convergence
        iteration_results = []
        
        for iter_idx in range(8):
            results = []
            # Energy improves over iterations
            base_energy = -5.0 - iter_idx * 0.5
            
            for i in range(15):
                assignments = {'x0': i % 2 == 0, 'x1': i % 3 == 0}
                energy = base_energy + np.random.normal(0, 0.3)
                results.append(SampleResult(assignments, energy, 1))
            
            iteration_results.append(results)
        
        # Analyze convergence
        convergence_data = analyze_convergence(iteration_results, ma_window=3, show=False)
        assert isinstance(convergence_data, dict)
        
        # Use the last iteration for detailed visualization
        final_results = iteration_results[-1]
        helper = VisualizationHelper(final_results)
        
        energy_data = helper.prepare_energy_landscape()
        solution_data = helper.analyze_solutions()
        
        assert isinstance(energy_data, dict)
        assert isinstance(solution_data, dict)


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestTytanVizErrorHandling:
    """Test error handling in Tytan visualization components."""
    
    def test_empty_results_handling(self):
        """Test handling of empty result sets."""
        # Empty results
        helper = VisualizationHelper([])
        
        try:
            energy_data = helper.prepare_energy_landscape()
            # Might handle gracefully with empty data
            assert isinstance(energy_data, dict)
        except (ValueError, IndexError):
            # This is also acceptable
            pass
        
        try:
            solution_data = helper.analyze_solutions()
            assert isinstance(solution_data, dict)
        except (ValueError, IndexError):
            pass
    
    def test_single_result_handling(self):
        """Test handling of single result."""
        single_result = [SampleResult({'x0': True}, -1.0, 1)]
        helper = VisualizationHelper(single_result)
        
        # Should handle single result gracefully
        energy_data = helper.prepare_energy_landscape()
        assert isinstance(energy_data, dict)
        
        solution_data = helper.analyze_solutions()
        assert isinstance(solution_data, dict)
    
    def test_invalid_sample_results(self):
        """Test handling of invalid sample results."""
        # Result with missing fields
        try:
            result = SampleResult(None, None, None)
            # Implementation might handle this gracefully
        except (TypeError, ValueError):
            # Expected for invalid input
            pass
    
    def test_plotting_without_matplotlib(self):
        """Test plotting when matplotlib is not available."""
        results = [SampleResult({'x0': True}, -1.0, 1)]
        helper = VisualizationHelper(results)
        
        # Mock matplotlib unavailability is difficult to test directly
        # This test is for documentation purposes


@pytest.mark.skipif(not HAS_TYTAN_VIZ, reason="tytan_viz module not available")
class TestTytanVizPerformance:
    """Test performance characteristics of Tytan visualization."""
    
    def test_large_result_set_performance(self):
        """Test visualization with large result sets."""
        import time
        
        # Create large result set
        results = []
        for i in range(200):
            assignments = {f'x{j}': (i + j) % 2 == 0 for j in range(10)}
            energy = np.random.normal(0, 1)
            results.append(SampleResult(assignments, energy, 1))
        
        helper = VisualizationHelper(results)
        
        start_time = time.time()
        
        # Perform analysis
        energy_data = helper.prepare_energy_landscape()
        solution_data = helper.analyze_solutions()
        stats = helper.get_variable_statistics()
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds max
        assert isinstance(energy_data, dict)
        assert isinstance(solution_data, dict)
        assert isinstance(stats, dict)
    
    def test_convergence_analysis_performance(self):
        """Test convergence analysis performance."""
        import time
        
        # Create many iterations
        iteration_results = []
        for iter_idx in range(20):
            results = []
            for i in range(30):
                assignments = {'x0': i % 2 == 0, 'x1': i % 3 == 0}
                energy = np.random.normal(-iter_idx, 1)
                results.append(SampleResult(assignments, energy, 1))
            iteration_results.append(results)
        
        start_time = time.time()
        
        data = analyze_convergence(iteration_results, ma_window=5, show=False)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert isinstance(data, dict)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of visualization operations."""
        # Create multiple visualization helpers
        helpers = []
        
        for i in range(10):
            results = []
            for j in range(20):
                assignments = {f'x{k}': (i + j + k) % 2 == 0 for k in range(5)}
                energy = np.random.normal(0, 1)
                results.append(SampleResult(assignments, energy, 1))
            
            helper = VisualizationHelper(results)
            helper.prepare_energy_landscape()
            helper.analyze_solutions()
            helpers.append(helper)
        
        # Should complete without memory issues
        assert len(helpers) == 10
        
        # Verify all helpers work
        for helper in helpers:
            stats = helper.get_variable_statistics()
            assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__])