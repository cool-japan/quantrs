# QuantRS2-Tytan API Reference

## Overview

QuantRS2-Tytan is a high-level quantum annealing library for the QuantRS2 framework, providing easy-to-use interfaces for formulating and solving quantum annealing problems with support for multiple backend solvers.

## Core Modules

### `quantrs2_tytan::symbol` (requires `dwave` feature)

Symbolic expression building for quantum annealing problems.

#### Functions

- **`symbols(name: &str) -> Symbol`**
  - Creates a single symbolic variable
  - Example: `let x = symbols("x");`

- **`symbols_list(names: Vec<&str>) -> Vec<Symbol>`**
  - Creates multiple symbolic variables
  - Example: `let vars = symbols_list(vec!["x", "y", "z"]);`

- **`symbols_nbit(name: &str, n: usize) -> Vec<Symbol>`**
  - Creates n-bit encoded symbolic variables
  - Example: `let bits = symbols_nbit("q", 3);`

- **`symbols_define(shape: (usize, usize), name: &str) -> Array2<Symbol>`**
  - Creates a 2D array of symbolic variables
  - Example: `let matrix = symbols_define((3, 3), "m");`

### `quantrs2_tytan::compile` (requires `dwave` feature)

Compilation of symbolic expressions to QUBO/HOBO format.

#### Structs

- **`Compile`**
  - Main compilation interface
  - Methods:
    - `new(expr: &Expression) -> Self`
    - `get_qubo() -> Result<((Array2<f64>, HashMap<String, usize>), f64), CompileError>`
    - `get_hobo() -> Result<((ArrayD<f64>, HashMap<String, usize>), f64), CompileError>`

- **`PieckCompile`**
  - Alternative compiler with constraint handling
  - Methods:
    - `new(expr: &Expression) -> Self`
    - `get_compiled_hobo() -> Result<CompiledHOBO, CompileError>`

### `quantrs2_tytan::sampler`

Quantum annealing samplers for solving QUBO/HOBO problems.

#### Trait

- **`Sampler`**
  - Common interface for all samplers
  - Methods:
    - `run_qubo(&self, qubo: &(Array2<f64>, HashMap<String, usize>), num_samples: usize) -> Result<Vec<SampleResult>, SamplerError>`
    - `run_hobo(&self, hobo: &(ArrayD<f64>, HashMap<String, usize>), num_samples: usize) -> Result<Vec<SampleResult>, SamplerError>`

#### Implementations

- **`SASampler`** - Simulated Annealing
  ```rust
  let solver = SASampler::new(Some(Config {
      num_sweeps: 1000,
      temperature_schedule: Schedule::Linear(100.0, 0.1),
      seed: Some(42),
  }));
  ```

- **`GASampler`** - Genetic Algorithm
  ```rust
  let solver = GASampler::new(Some(GAConfig {
      population_size: 100,
      generations: 500,
      mutation_rate: 0.01,
      crossover_rate: 0.8,
      elitism: 5,
  }));
  ```

- **`ArminSampler`** - GPU-accelerated annealing (requires `gpu` feature)
  ```rust
  let solver = ArminSampler::new(Some(GPUConfig {
      work_groups: 256,
      threads_per_group: 64,
      temperature_steps: 1000,
  }));
  ```

- **`MIKASAmpler`** - HOBO-specialized solver
  ```rust
  let solver = MIKASAmpler::new(Some(MIKASConfig {
      tensor_decomposition: true,
      adaptive_sampling: true,
      num_chains: 10,
  }));
  ```

- **`DWaveSampler`** - D-Wave quantum annealer interface (requires `dwave` feature)
  ```rust
  let solver = DWaveSampler::new(Some(DWaveConfig {
      solver_name: "DW_2000Q_6",
      num_reads: 100,
      annealing_time: 20,
  }));
  ```

### `quantrs2_tytan::auto_array` (requires `dwave` feature)

Automatic result processing and array conversion.

#### Struct

- **`Auto_array`**
  - Automatic multi-dimensional array conversion
  - Methods:
    - `new() -> Self`
    - `set_nbit(nbit: Vec<String>) -> Self`
    - `set_int_range(int_range: HashMap<String, Range<i32>>) -> Self`
    - `calc_score(results: Vec<SampleResult>) -> Vec<(ArrayD<i32>, f64, usize)>`

### `quantrs2_tytan::optimize`

Energy calculation and optimization utilities.

#### Functions

- **`calculate_energy(solution: &[bool], matrix: &Array2<f64>) -> f64`**
  - Calculate QUBO energy for a solution

- **`optimize_qubo(qubo: &Array2<f64>, max_iterations: usize) -> Vec<bool>`**
  - Local optimization for QUBO problems

- **`optimize_hobo(hobo: &ArrayD<f64>, max_iterations: usize) -> Vec<bool>`**
  - Local optimization for HOBO problems

### `quantrs2_tytan::analysis`

Solution analysis and visualization tools.

#### Functions

- **`calculate_diversity(solutions: &[Vec<bool>]) -> f64`**
  - Calculate diversity metric for solution set

- **`cluster_solutions(solutions: &[Vec<bool>], k: usize) -> Vec<Vec<usize>>`**
  - Cluster similar solutions

- **`visualize_energy_distribution(energies: &[f64], path: &str) -> Result<(), VisualizationError>`**
  - Create energy distribution plot

### `quantrs2_tytan::parallel_tempering`

Advanced parallel tempering sampler.

#### Struct

- **`ParallelTemperingSampler`**
  ```rust
  let solver = ParallelTemperingSampler::new(PTConfig {
      num_replicas: 10,
      temperature_range: (0.1, 100.0),
      swap_interval: 10,
      num_sweeps: 10000,
  });
  ```

### `quantrs2_tytan::ml_guided_sampling`

Machine learning guided sampling strategies.

#### Struct

- **`MLGuidedSampler`**
  ```rust
  let solver = MLGuidedSampler::new(MLConfig {
      base_sampler: Box::new(SASampler::new(None)),
      learning_rate: 0.01,
      history_size: 1000,
      use_reinforcement: true,
  });
  ```

### `quantrs2_tytan::constraints`

Constraint handling for optimization problems.

#### Functions

- **`add_equality_constraint(qubo: &mut Array2<f64>, vars: &[usize], value: f64, penalty: f64)`**
  - Add equality constraint to QUBO

- **`add_inequality_constraint(qubo: &mut Array2<f64>, vars: &[usize], upper_bound: f64, penalty: f64)`**
  - Add inequality constraint to QUBO

### `quantrs2_tytan::encoding`

Variable encoding schemes.

#### Functions

- **`one_hot_encoding(n: usize) -> Array2<f64>`**
  - Generate one-hot encoding matrix

- **`binary_encoding(n_bits: usize, range: Range<i32>) -> Array2<f64>`**
  - Generate binary encoding matrix

- **`unary_encoding(max_value: usize) -> Array2<f64>`**
  - Generate unary encoding matrix

### `quantrs2_tytan::hybrid_algorithms`

Hybrid classical-quantum algorithms.

#### Struct

- **`HybridSolver`**
  ```rust
  let solver = HybridSolver::new(HybridConfig {
      classical_solver: Box::new(SASampler::new(None)),
      quantum_solver: Box::new(DWaveSampler::new(None)),
      partition_size: 50,
      iterations: 10,
  });
  ```

### `quantrs2_tytan::problem_decomposition`

Problem decomposition strategies.

#### Functions

- **`decompose_large_qubo(qubo: &Array2<f64>, max_size: usize) -> Vec<Array2<f64>>`**
  - Decompose large QUBO into smaller subproblems

- **`merge_solutions(subproblems: Vec<Vec<bool>>, overlap_map: &HashMap<usize, Vec<usize>>) -> Vec<bool>`**
  - Merge solutions from subproblems

### `quantrs2_tytan::applications`

Pre-built application templates.

#### Submodules

- **`finance`** - Portfolio optimization, risk analysis
- **`logistics`** - Route optimization, scheduling
- **`drug_discovery`** - Molecular docking, drug design
- **`materials`** - Material discovery, crystal structure
- **`ml_tools`** - Feature selection, clustering

### `quantrs2_tytan::gpu` (requires `gpu` feature)

GPU acceleration utilities.

#### Functions

- **`is_gpu_available() -> bool`**
  - Check GPU availability

- **`gpu_solve_qubo(qubo: &Array2<f64>, config: GPUConfig) -> Result<Vec<bool>, GPUError>`**
  - Direct GPU QUBO solver

- **`gpu_solve_hobo(hobo: &ArrayD<f64>, config: GPUConfig) -> Result<Vec<bool>, GPUError>`**
  - Direct GPU HOBO solver

### `quantrs2_tytan::benchmark`

Performance benchmarking tools.

#### Structs

- **`BenchmarkRunner`**
  ```rust
  let mut runner = BenchmarkRunner::new();
  runner.add_problem("test_qubo", qubo);
  runner.add_solver("SA", Box::new(SASampler::new(None)));
  let results = runner.run(100);
  ```

### `quantrs2_tytan::visualization`

Advanced visualization tools.

#### Functions

- **`plot_convergence(history: &[(usize, f64)], path: &str) -> Result<(), VisualizationError>`**
  - Plot optimization convergence

- **`plot_solution_landscape(solutions: &[Vec<bool>], energies: &[f64], path: &str) -> Result<(), VisualizationError>`**
  - Visualize solution landscape

- **`export_problem_graph(qubo: &Array2<f64>, path: &str) -> Result<(), VisualizationError>`**
  - Export problem structure as graph

## Error Types

### `SamplerError`
- `InvalidInput(String)` - Invalid input parameters
- `SolverFailed(String)` - Solver execution failed
- `Timeout` - Solver timed out
- `NotImplemented` - Feature not implemented

### `CompileError`
- `InvalidExpression(String)` - Invalid symbolic expression
- `VariableNotFound(String)` - Variable not defined
- `UnsupportedOperation(String)` - Unsupported operation

### `GPUError`
- `DeviceNotFound` - No GPU device available
- `KernelCompilationFailed(String)` - GPU kernel compilation failed
- `OutOfMemory` - GPU out of memory
- `ExecutionFailed(String)` - GPU execution failed

## Examples

### Basic QUBO Solving
```rust
use quantrs2_tytan::sampler::{SASampler, Sampler};
use ndarray::Array2;
use std::collections::HashMap;

// Create QUBO matrix
let mut qubo = Array2::zeros((3, 3));
qubo[[0, 0]] = -1.0;
qubo[[1, 1]] = -1.0;
qubo[[2, 2]] = -1.0;
qubo[[0, 1]] = 2.0;
qubo[[1, 0]] = 2.0;

// Create variable map
let mut var_map = HashMap::new();
var_map.insert("x".to_string(), 0);
var_map.insert("y".to_string(), 1);
var_map.insert("z".to_string(), 2);

// Solve
let solver = SASampler::new(None);
let results = solver.run_qubo(&(qubo, var_map), 100).unwrap();

// Print best solution
let best = results.iter().min_by_key(|r| r.energy as i32).unwrap();
println!("Best energy: {}, solution: {:?}", best.energy, best.solution);
```

### Symbolic Problem Construction (requires `dwave` feature)
```rust
use quantrs2_tytan::{symbols, Compile};
use quantrs2_tytan::sampler::{GASampler, Sampler};

// Define problem symbolically
let x = symbols("x");
let y = symbols("y");
let z = symbols("z");

// Constraint: exactly one variable should be 1
let expr = (x + y + z - 1).pow(2);

// Compile to QUBO
let (qubo, offset) = Compile::new(&expr).get_qubo().unwrap();

// Solve with genetic algorithm
let solver = GASampler::new(None);
let results = solver.run_qubo(&qubo, 200).unwrap();
```

### Advanced Parallel Tempering
```rust
use quantrs2_tytan::parallel_tempering::{ParallelTemperingSampler, PTConfig};

let solver = ParallelTemperingSampler::new(PTConfig {
    num_replicas: 20,
    temperature_range: (0.01, 50.0),
    swap_interval: 5,
    num_sweeps: 50000,
});

let results = solver.run_qubo(&qubo, 10).unwrap();
```

### Problem Decomposition for Large QUBOs
```rust
use quantrs2_tytan::problem_decomposition::{decompose_large_qubo, merge_solutions};

// Decompose large problem
let subproblems = decompose_large_qubo(&large_qubo, 50);

// Solve each subproblem
let mut sub_solutions = Vec::new();
for sub_qubo in subproblems {
    let result = solver.run_qubo(&sub_qubo, 100).unwrap();
    sub_solutions.push(result[0].solution.clone());
}

// Merge solutions
let final_solution = merge_solutions(sub_solutions, &overlap_map);
```

## Feature Flags

- `parallel` - Enable multi-threading (default)
- `gpu` - Enable GPU acceleration
- `dwave` - Enable symbolic math and D-Wave support
- `scirs` - Enable SciRS2 integration
- `advanced_optimization` - Advanced optimization algorithms
- `gpu_accelerated` - Full GPU acceleration
- `clustering` - Solution clustering tools
- `plotters` - Visualization support

## Performance Tips

1. **Use appropriate sampler for problem type**:
   - SA for general QUBO problems
   - GA for problems with good solution structure
   - PT for complex energy landscapes
   - MIKAS for HOBO problems

2. **Leverage GPU acceleration** for large problems (>100 variables)

3. **Use problem decomposition** for very large problems (>1000 variables)

4. **Enable SciRS2 features** for maximum performance

5. **Tune sampler parameters** based on problem characteristics

## Advanced Quantum Computing Modules 🆕

### `quantrs2_tytan::quantum_neural_networks`

Hybrid quantum-classical neural network architectures for optimization.

#### Key Types

- **`QuantumNeuralNetwork`** - Main QNN implementation
- **`QNNConfig`** - Configuration for QNN architecture
- **`QNNArchitecture`** - Architecture types (PureQuantum, HybridSequential, HybridDense)
- **`EntanglementPattern`** - Entanglement patterns (Linear, Circular, AllToAll, Custom)
- **`TrainingResult`** - Results from QNN training

#### Functions

- **`create_qnn_for_optimization(num_qubits: usize) -> Result<QuantumNeuralNetwork, QNNError>`**
  - Creates a pre-configured QNN for optimization problems

### `quantrs2_tytan::quantum_state_tomography`

Comprehensive quantum state reconstruction and characterization.

#### Key Types

- **`QuantumStateTomography`** - Main tomography system
- **`TomographyConfig`** - Configuration for tomography
- **`TomographyType`** - Types (QuantumState, ShadowTomography, CompressedSensing, AdaptiveTomography)
- **`ReconstructedState`** - Reconstructed quantum state with metadata
- **`EntanglementMeasures`** - Concurrence, negativity, and other measures

#### Functions

- **`create_tomography_system(num_qubits: usize) -> QuantumStateTomography`**
  - Creates a tomography system with default configuration
- **`create_shadow_tomography_config(num_shadows: usize) -> TomographyConfig`**
  - Creates configuration for shadow tomography

### `quantrs2_tytan::quantum_error_correction`

Advanced quantum error correction for optimization problems.

#### Key Types

- **`QuantumErrorCorrection`** - Main QEC system
- **`QECConfig`** - Configuration for error correction
- **`QuantumCodeType`** - Code types (SurfaceCode, ColorCode, StabilizerCode, TopologicalCode)
- **`DecodingAlgorithm`** - Decoding methods (MWPM, BeliefPropagation, NeuralNetwork, MachineLearning)
- **`QECMetrics`** - Performance metrics for error correction

#### Functions

- **`create_optimization_qec(num_logical_qubits: usize) -> QuantumErrorCorrection`**
  - Creates QEC system optimized for quantum annealing
- **`create_adaptive_qec_config() -> QECConfig`**
  - Creates adaptive QEC configuration

### `quantrs2_tytan::tensor_network_sampler`

Tensor network algorithms for quantum optimization.

#### Key Types

- **`TensorNetworkSampler`** - Main tensor network sampler
- **`TensorNetworkConfig`** - Configuration for tensor networks
- **`TensorNetworkType`** - Network types (MPS, PEPS, MERA, TTN, iMPS, iPEPS)
- **`TensorNetwork`** - Internal tensor network representation
- **`TensorNetworkMetrics`** - Performance metrics

#### Functions

- **`create_mps_sampler(bond_dimension: usize) -> TensorNetworkSampler`**
  - Creates Matrix Product State sampler
- **`create_peps_sampler(bond_dimension: usize, lattice_shape: (usize, usize)) -> TensorNetworkSampler`**
  - Creates Projected Entangled Pair State sampler
- **`create_mera_sampler(layers: usize) -> TensorNetworkSampler`**
  - Creates MERA sampler

### `quantrs2_tytan::advanced_performance_analysis`

Comprehensive performance monitoring and analysis.

#### Key Types

- **`AdvancedPerformanceAnalyzer`** - Main performance analyzer
- **`AnalysisConfig`** - Configuration for analysis
- **`MetricsLevel`** - Collection levels (Basic, Detailed, Comprehensive)
- **`AnalysisDepth`** - Analysis depth (Surface, Deep, Exhaustive, Adaptive)
- **`AnalysisResults`** - Comprehensive analysis results

#### Functions

- **`create_comprehensive_analyzer() -> AdvancedPerformanceAnalyzer`**
  - Creates analyzer with full monitoring capabilities
- **`create_lightweight_analyzer() -> AdvancedPerformanceAnalyzer`**
  - Creates lightweight analyzer for basic monitoring

## Complete API Usage Example

```rust
use quantrs2_tytan::{
    // Core features
    symbols, Compile,
    sampler::{SASampler, Sampler},
    
    // Advanced features
    quantum_neural_networks::create_qnn_for_optimization,
    tensor_network_sampler::create_mps_sampler,
    advanced_performance_analysis::create_comprehensive_analyzer,
};

fn advanced_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Set up performance monitoring
    let mut analyzer = create_comprehensive_analyzer();
    analyzer.start_analysis()?;
    
    // Create symbolic problem
    let x = symbols("x");
    let y = symbols("y");
    let expr = x * y + (x - y).pow(2);
    
    // Compile to QUBO
    let (qubo, offset) = Compile::new(&expr).get_qubo()?;
    
    // Use tensor network sampler
    let sampler = create_mps_sampler(32);
    let results = sampler.run_qubo(&qubo, 1000)?;
    
    // Analyze performance
    analyzer.perform_comprehensive_analysis()?;
    
    Ok(())
}
```