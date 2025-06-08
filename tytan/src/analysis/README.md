# Advanced Visualization for Quantum Annealing Results

This module provides sophisticated data analysis and visualization preparation for quantum annealing solutions. The visualization capabilities are designed to work with external plotting libraries while providing comprehensive data analysis within Rust.

## Features

### 1. Energy Landscape Analysis
- **Energy Distribution**: Analyze the distribution of solution energies
- **Histogram Generation**: Customizable binning for energy histograms
- **Kernel Density Estimation**: Smooth energy landscape visualization using KDE
- **Sorted Energy Values**: Track solution quality progression

```rust
use quantrs2_tytan::analysis::visualization::*;

let config = EnergyLandscapeConfig {
    num_bins: 50,
    compute_kde: true,
    kde_points: 200,
};

let landscape_data = prepare_energy_landscape(&results, Some(config))?;
export_to_csv(&landscape_data, "energy_landscape.csv")?;
```

### 2. Solution Distribution Analysis
- **Variable Frequencies**: Analyze how often each variable is set to 1
- **Correlation Analysis**: Identify correlations between variables
- **PCA Analysis**: Dimensionality reduction for high-dimensional solutions
- **Solution Matrix Export**: Export binary solution matrix for further analysis

```rust
let dist_config = SolutionDistributionConfig {
    compute_correlations: true,
    compute_pca: true,
    n_components: 2,
};

let dist_data = analyze_solution_distribution(&results, Some(dist_config))?;
export_solution_matrix(&dist_data, "solutions.csv")?;
```

### 3. Problem-Specific Visualizations

#### Traveling Salesman Problem (TSP)
- Extract tour from binary edge variables
- Calculate tour length
- Prepare data for route visualization

```rust
let tour = extract_tsp_tour(&result, n_cities)?;
let length = calculate_tour_length(&tour, &city_coordinates);
```

#### Graph Coloring
- Extract color assignments from binary variables
- Identify coloring conflicts
- Use spring layout for graph visualization

```rust
let (node_colors, conflicts) = extract_graph_coloring(&result, n_nodes, n_colors, &edges)?;
let positions = spring_layout(n_nodes, &edges);
```

#### Max Cut
- Extract partition assignments
- Identify cut edges
- Calculate cut size

#### Number Partitioning
- Extract partition assignments
- Calculate partition sums and differences

### 4. Convergence Analysis
- Track best energy evolution over iterations
- Calculate average energy and standard deviation
- Compute moving averages for smoother trends
- Identify convergence behavior

```rust
let conv_data = analyze_convergence(&iteration_results, Some(5))?;
// Access: best_energies, avg_energies, std_devs, ma_best, ma_avg
```

## Data Export Formats

### CSV Export
All visualization data can be exported to CSV format for use with external tools:
- Energy landscape data
- Solution matrices
- Convergence metrics

### Integration with Plotting Libraries

The module is designed to work with various plotting libraries:

**Python (matplotlib/plotly)**:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('energy_landscape.csv')
plt.scatter(df['index'], df['energy'])
plt.xlabel('Solution Index')
plt.ylabel('Energy')
plt.show()
```

**Gnuplot**:
```gnuplot
set xlabel "Solution Index"
set ylabel "Energy"
plot 'energy_landscape.csv' using 1:3 with points
```

## Advanced Features

### Custom Configurations
All analysis functions support custom configurations:
- Adjust histogram bins
- Control KDE parameters
- Select correlation thresholds
- Configure PCA components

### Performance Considerations
- Efficient data structures for large result sets
- Optional features can be disabled for performance
- Streaming export for large datasets

### Extensibility
The module is designed to be extended with:
- New problem types
- Additional statistical analyses
- Custom export formats
- Integration with more visualization libraries

## Examples

See `tests/visualization_tests.rs` for comprehensive examples including:
- Complete visualization workflow
- Problem-specific examples
- Integration with external tools
- Performance benchmarks