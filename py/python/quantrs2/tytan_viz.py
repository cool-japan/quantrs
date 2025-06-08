"""
Tytan Visualization Module

This module provides advanced visualization capabilities for quantum annealing results,
including energy landscapes, solution distributions, and convergence analysis.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = hasattr(_quantrs2, 'tytan')
except ImportError:
    _NATIVE_AVAILABLE = False

if _NATIVE_AVAILABLE:
    # Import native implementations
    SampleResult = _quantrs2.tytan.PySampleResult
    EnergyLandscapeVisualizer = _quantrs2.tytan.PyEnergyLandscapeVisualizer
    SolutionAnalyzer = _quantrs2.tytan.PySolutionAnalyzer
    ProblemVisualizer = _quantrs2.tytan.PyProblemVisualizer
    ConvergenceAnalyzer = _quantrs2.tytan.PyConvergenceAnalyzer
else:
    # Provide stubs
    class SampleResult:
        """Sample result (stub)"""
        def __init__(self, assignments: Dict[str, bool], energy: float, occurrences: int = 1):
            self.assignments = assignments
            self.energy = energy
            self.occurrences = occurrences
    
    class EnergyLandscapeVisualizer:
        """Energy landscape visualizer (stub)"""
        @staticmethod
        def prepare_landscape(*args, **kwargs):
            raise ImportError("Tytan features not available. Install with: pip install quantrs2[tytan]")
    
    class SolutionAnalyzer:
        """Solution analyzer (stub)"""
        @staticmethod
        def analyze_distribution(*args, **kwargs):
            raise ImportError("Tytan features not available")
    
    class ProblemVisualizer:
        """Problem visualizer (stub)"""
        @staticmethod
        def extract_tsp_tour(*args, **kwargs):
            raise ImportError("Tytan features not available")
    
    class ConvergenceAnalyzer:
        """Convergence analyzer (stub)"""
        @staticmethod
        def analyze_convergence(*args, **kwargs):
            raise ImportError("Tytan features not available")


class VisualizationHelper:
    """
    Helper class for quantum annealing visualization workflows.
    """
    
    def __init__(self, results: List[SampleResult]):
        """
        Initialize visualization helper.
        
        Args:
            results: List of sample results from quantum annealing
        """
        self.results = results
        self._energy_data = None
        self._solution_data = None
    
    def prepare_energy_landscape(self, num_bins: int = 50, 
                               compute_kde: bool = True,
                               kde_points: int = 200) -> Dict[str, np.ndarray]:
        """
        Prepare energy landscape visualization data.
        
        Args:
            num_bins: Number of histogram bins
            compute_kde: Whether to compute kernel density estimation
            kde_points: Number of KDE points
            
        Returns:
            Dictionary with visualization data
        """
        data = EnergyLandscapeVisualizer.prepare_landscape(
            self.results, num_bins, compute_kde, kde_points
        )
        self._energy_data = data
        return data
    
    def analyze_solutions(self, compute_correlations: bool = True,
                         compute_pca: bool = True,
                         n_components: int = 2) -> Dict[str, Any]:
        """
        Analyze solution distributions.
        
        Args:
            compute_correlations: Whether to compute variable correlations
            compute_pca: Whether to compute PCA
            n_components: Number of PCA components
            
        Returns:
            Dictionary with analysis results
        """
        data = SolutionAnalyzer.analyze_distribution(
            self.results, compute_correlations, compute_pca, n_components
        )
        self._solution_data = data
        return data
    
    def plot_energy_landscape(self, save_path: Optional[str] = None, 
                            show: bool = True) -> None:
        """
        Plot energy landscape using matplotlib.
        
        Args:
            save_path: Path to save figure
            show: Whether to show the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")
        
        if self._energy_data is None:
            self.prepare_energy_landscape()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy scatter plot
        ax1.scatter(self._energy_data['indices'], self._energy_data['energies'], 
                   alpha=0.6, s=30)
        ax1.set_xlabel('Solution Index (sorted)')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Landscape')
        ax1.grid(True, alpha=0.3)
        
        # Energy histogram with KDE
        ax2.hist(self._energy_data['energies'], bins=self._energy_data['histogram_bins'][:-1],
                weights=self._energy_data['histogram_counts'], alpha=0.7, 
                density=True, label='Histogram')
        
        if 'kde_x' in self._energy_data and 'kde_y' in self._energy_data:
            ax2.plot(self._energy_data['kde_x'], self._energy_data['kde_y'], 
                    'r-', lw=2, label='KDE')
        
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('Density')
        ax2.set_title('Energy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_solution_heatmap(self, save_path: Optional[str] = None,
                            show: bool = True) -> None:
        """
        Plot solution matrix heatmap.
        
        Args:
            save_path: Path to save figure
            show: Whether to show the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required. Install with: pip install matplotlib seaborn")
        
        if self._solution_data is None:
            self.analyze_solutions()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(self._solution_data['solution_matrix'].T, 
                   cmap='RdBu_r', center=0.5,
                   yticklabels=self._solution_data['variable_names'],
                   xticklabels=False,
                   cbar_kws={'label': 'Variable Value'},
                   ax=ax)
        
        ax.set_xlabel('Sample')
        ax.set_ylabel('Variable')
        ax.set_title('Solution Matrix Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_variable_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each variable.
        
        Returns:
            Dictionary with variable statistics
        """
        if self._solution_data is None:
            self.analyze_solutions()
        
        stats = {}
        for var_name in self._solution_data['variable_names']:
            freq = self._solution_data['variable_frequencies'][var_name]
            stats[var_name] = {
                'frequency': freq,
                'probability_one': freq,
                'probability_zero': 1.0 - freq,
            }
        
        # Add correlation information if available
        if 'correlations' in self._solution_data:
            for var in stats:
                correlations = {}
                for key, value in self._solution_data['correlations'].items():
                    if var in key:
                        other_var = key.replace(var, '').strip('_')
                        if other_var and other_var != var:
                            correlations[other_var] = value
                stats[var]['correlations'] = correlations
        
        return stats


class ProblemSpecificVisualizer:
    """
    Visualizations for specific optimization problems.
    """
    
    @staticmethod
    def visualize_tsp_solution(result: SampleResult, cities: List[Tuple[float, float]],
                             save_path: Optional[str] = None, show: bool = True) -> float:
        """
        Visualize TSP solution.
        
        Args:
            result: Sample result containing TSP solution
            cities: List of (x, y) coordinates for each city
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Tour length
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        n_cities = len(cities)
        tour = ProblemVisualizer.extract_tsp_tour(result, n_cities)
        
        # Calculate tour length
        tour_length = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            x1, y1 = cities[tour[i]]
            x2, y2 = cities[tour[j]]
            tour_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot cities
        x_coords = [c[0] for c in cities]
        y_coords = [c[1] for c in cities]
        ax.scatter(x_coords, y_coords, s=200, c='red', zorder=3)
        
        # Label cities
        for i, (x, y) in enumerate(cities):
            ax.annotate(f'{i}', (x, y), ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        
        # Plot tour
        tour_x = [cities[tour[i]][0] for i in range(len(tour) + 1)]
        tour_y = [cities[tour[i]][1] for i in range(len(tour) + 1)]
        tour_x[-1] = tour_x[0]  # Close the tour
        tour_y[-1] = tour_y[0]
        
        ax.plot(tour_x, tour_y, 'b-', linewidth=2, alpha=0.7)
        
        ax.set_title(f'TSP Solution (Length: {tour_length:.2f})')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return tour_length
    
    @staticmethod
    def visualize_graph_coloring(result: SampleResult, edges: List[Tuple[int, int]],
                               n_colors: int, save_path: Optional[str] = None,
                               show: bool = True) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Visualize graph coloring solution.
        
        Args:
            result: Sample result containing coloring solution
            edges: Graph edges
            n_colors: Number of colors
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Tuple of (node colors, conflict edges)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        # Find number of nodes
        n_nodes = max(max(u, v) for u, v in edges) + 1
        
        # Extract coloring and conflicts
        node_colors, conflicts = ProblemVisualizer.extract_graph_coloring(
            result, n_nodes, n_colors, edges
        )
        
        # Get node positions using spring layout
        positions = ProblemVisualizer.spring_layout(n_nodes, edges)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create color map
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_colors))
        
        # Plot edges
        for u, v in edges:
            x = [positions[u][0], positions[v][0]]
            y = [positions[u][1], positions[v][1]]
            if (u, v) in conflicts or (v, u) in conflicts:
                ax.plot(x, y, 'r-', linewidth=3, alpha=0.8, zorder=1)
            else:
                ax.plot(x, y, 'k-', linewidth=1, alpha=0.3, zorder=1)
        
        # Plot nodes
        for i in range(n_nodes):
            color = colors[node_colors[i]]
            ax.scatter(positions[i][0], positions[i][1], s=500, c=[color],
                      edgecolor='black', linewidth=2, zorder=2)
            ax.annotate(f'{i}', (positions[i][0], positions[i][1]),
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_title(f'Graph Coloring ({n_colors} colors, {len(conflicts)} conflicts)')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return node_colors, conflicts


def analyze_convergence(iteration_results: List[List[SampleResult]], 
                       ma_window: int = 5,
                       save_path: Optional[str] = None,
                       show: bool = True) -> Dict[str, np.ndarray]:
    """
    Analyze and plot convergence behavior.
    
    Args:
        iteration_results: Results for each iteration
        ma_window: Moving average window size
        save_path: Path to save figure
        show: Whether to show the plot
        
    Returns:
        Dictionary with convergence data
    """
    data = ConvergenceAnalyzer.analyze_convergence(iteration_results, ma_window)
    
    if show or save_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Best energy plot
        ax1.plot(data['iterations'], data['best_energies'], 'b-', alpha=0.5, label='Best Energy')
        if 'ma_best' in data:
            ma_x = data['iterations'][ma_window-1:]
            ax1.plot(ma_x, data['ma_best'], 'b-', linewidth=2, label=f'MA({ma_window})')
        ax1.set_ylabel('Best Energy')
        ax1.set_title('Convergence Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Average energy with error bars
        ax2.errorbar(data['iterations'], data['avg_energies'], yerr=data['std_devs'],
                    fmt='o-', alpha=0.5, label='Avg ± Std')
        if 'ma_avg' in data:
            ax2.plot(ma_x, data['ma_avg'], 'r-', linewidth=2, label=f'MA({ma_window})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Energy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    return data


# Example usage
def example_visualization_workflow():
    """Example workflow for visualizing quantum annealing results."""
    if not _NATIVE_AVAILABLE:
        print("Tytan features not available")
        return
    
    # Create sample results (normally these would come from quantum annealing)
    results = []
    for i in range(100):
        assignments = {
            'x0': i % 2 == 0,
            'x1': i % 3 == 0,
            'x2': i % 5 == 0,
            'x3': i % 7 == 0,
        }
        energy = -10.0 + i * 0.1 + np.random.normal(0, 0.5)
        results.append(SampleResult(assignments, energy, 1))
    
    # Create visualization helper
    viz = VisualizationHelper(results)
    
    # Prepare and plot energy landscape
    print("Analyzing energy landscape...")
    energy_data = viz.prepare_energy_landscape()
    viz.plot_energy_landscape(save_path='energy_landscape.png', show=False)
    
    # Analyze solutions
    print("Analyzing solution distribution...")
    solution_data = viz.analyze_solutions()
    viz.plot_solution_heatmap(save_path='solution_heatmap.png', show=False)
    
    # Get variable statistics
    stats = viz.get_variable_statistics()
    print("\nVariable Statistics:")
    for var, var_stats in stats.items():
        print(f"  {var}: frequency={var_stats['frequency']:.3f}")
    
    print("\nVisualization complete! Check the saved images.")
    
    return viz


__all__ = [
    'SampleResult',
    'EnergyLandscapeVisualizer',
    'SolutionAnalyzer',
    'ProblemVisualizer',
    'ConvergenceAnalyzer',
    'VisualizationHelper',
    'ProblemSpecificVisualizer',
    'analyze_convergence',
    'example_visualization_workflow',
]