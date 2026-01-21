# QuantRS2-Anneal: Comprehensive Quantum Annealing Framework

[![Crates.io](https://img.shields.io/crates/v/quantrs2-anneal.svg)](https://crates.io/crates/quantrs2-anneal)
[![Documentation](https://docs.rs/quantrs2-anneal/badge.svg)](https://docs.rs/quantrs2-anneal)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Anneal is the premier quantum annealing module of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing comprehensive support for quantum annealing, optimization problems, and quantum-inspired algorithms.

## Version 0.1.1

This release features:
- Stable APIs for D-Wave, AWS Braket, and Fujitsu integrations
- Enhanced performance using SciRS2 v0.1.0's parallel algorithms for large-scale optimization
- Improved minor graph embedding with refined SciRS2 graph algorithms
- Memory-efficient sparse matrix operations via SciRS2

## 🚀 Quick Start

```rust
use quantrs2_anneal::{
    ising::IsingModel,
    simulator::{ClassicalAnnealingSimulator, AnnealingParams}
};

// Create and solve a simple Max-Cut problem
let mut model = IsingModel::new(4);
model.set_coupling(0, 1, -1.0)?;
model.set_coupling(1, 2, -1.0)?;
model.set_coupling(2, 3, -1.0)?;
model.set_coupling(3, 0, -1.0)?;

let simulator = ClassicalAnnealingSimulator::new(AnnealingParams::default())?;
let result = simulator.solve(&model)?;

println!("Best energy: {}", result.best_energy);
println!("Solution: {:?}", result.best_spins);
```

## 🌟 Key Features

### Core Problem Formulations
- **🔗 Ising Models**: Complete sparse matrix support with biases and couplings
- **🎯 QUBO Formulations**: Quadratic Unconstrained Binary Optimization with constraint handling
- **🏗️ Problem Builder**: Intuitive DSL for complex optimization problem construction
- **🔄 Format Conversion**: Seamless conversion between Ising and QUBO representations

### Quantum Annealing Simulators
- **🌡️ Classical Annealing**: Simulated annealing with configurable temperature schedules
- **🌊 Path Integral Monte Carlo**: Quantum annealing simulation with tunneling effects
- **👥 Population Annealing**: Parallel annealing with population-based sampling
- **🎪 Coherent Ising Machine**: Quantum-inspired optimization using coherent states
- **📊 Non-Stoquastic Simulators**: Support for non-stoquastic Hamiltonians

### Cloud Quantum Hardware Integration
- **🌐 D-Wave Leap**: Comprehensive client with auto-embedding and advanced parameters
- **☁️ AWS Braket**: Full integration with cost tracking and device optimization
- **🗾 Fujitsu DAU**: Digital annealer unit integration for hybrid optimization
- **🤖 Hybrid Solvers**: Classical-quantum hybrid algorithms for large-scale problems

### Advanced Algorithms and Techniques
- **🗺️ Graph Embedding**: Minor graph embedding with optimization and layout algorithms
- **⚖️ Penalty Optimization**: Automatic constraint penalty weight optimization
- **🔄 Reverse Annealing**: Advanced annealing schedules starting from known solutions
- **🌈 Flux Bias Optimization**: Per-qubit flux bias tuning for enhanced performance
- **🌟 QAOA Integration**: Bridge to variational quantum algorithms via circuit module

### Specialized Applications
- **⚡ Energy Systems**: Power grid optimization, smart grid scheduling, renewable integration
- **💰 Finance**: Portfolio optimization, risk management, algorithmic trading
- **🏥 Healthcare**: Drug discovery, treatment planning, resource allocation
- **📦 Logistics**: Vehicle routing, supply chain optimization, warehouse management
- **🏭 Manufacturing**: Production scheduling, quality control, resource allocation
- **📡 Telecommunications**: Network optimization, spectrum allocation, routing protocols

### Visualization and Analysis
- **📈 Energy Landscapes**: Interactive visualization of optimization energy surfaces
- **📊 Performance Analytics**: Comprehensive metrics, timing analysis, and benchmarking
- **🎨 Solution Visualization**: Graphical representation of optimization solutions
- **📉 Convergence Analysis**: Real-time monitoring of annealing convergence behavior

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
quantrs2-anneal = "0.1.1"

# Optional features for cloud integration
quantrs2-anneal = { version = "0.1.0", features = ["dwave", "braket", "fujitsu"] }
```

## 🎯 Feature Flags

- **`default`**: Core functionality (Ising models, QUBO, classical simulators)
- **`dwave`**: D-Wave Leap cloud service integration
- **`braket`**: AWS Braket quantum computing platform
- **`fujitsu`**: Fujitsu Digital Annealer Unit integration

## 📚 Module Overview

### Core Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| `ising` | Ising model representation | Sparse matrices, efficient operations |
| `qubo` | QUBO formulation and constraints | Penalty methods, constraint handling |
| `simulator` | Classical annealing simulators | Multiple algorithms, configurable schedules |
| `dsl` | Problem construction DSL | Intuitive problem building syntax |

### Advanced Algorithms

| Module | Description | Use Cases |
|--------|-------------|-----------|
| `coherent_ising_machine` | Quantum-inspired optimization | Large-scale continuous optimization |
| `population_annealing` | Parallel population-based annealing | High-quality solution sampling |
| `reverse_annealing` | Reverse annealing schedules | Solution refinement and local search |
| `variational_quantum_annealing` | VQA-based optimization | Hybrid classical-quantum optimization |

### Embedding and Optimization

| Module | Description | Applications |
|--------|-------------|--------------|
| `embedding` | Minor graph embedding | Hardware topology mapping |
| `layout_embedding` | Layout-aware embedding | Optimized hardware utilization |
| `penalty_optimization` | Constraint penalty tuning | Enhanced solution quality |
| `compression` | Problem compression techniques | Large problem reduction |

### Cloud Integration

| Module | Description | Capabilities |
|--------|-------------|--------------|
| `dwave` | D-Wave Leap integration | QPU access, hybrid solvers, embedding |
| `braket` | AWS Braket integration | Multi-provider access, cost management |
| `fujitsu` | Fujitsu DAU integration | Digital annealer optimization |

### Applications

| Module | Description | Domain Expertise |
|--------|-------------|------------------|
| `applications::energy` | Energy system optimization | Grid operations, renewable integration |
| `applications::finance` | Financial optimization | Portfolio management, risk analysis |
| `applications::healthcare` | Healthcare optimization | Resource allocation, treatment planning |
| `applications::logistics` | Logistics and supply chain | Routing, scheduling, inventory |

## 🔧 Usage Examples

### 1. Basic QUBO Problem with Constraints

```rust
use quantrs2_anneal::{
    qubo::{QuboBuilder, QuboFormulation},
    simulator::ClassicalAnnealingSimulator
};

// Portfolio optimization: select assets with target return and risk constraints
let mut qubo = QuboBuilder::new(10); // 10 assets

// Objective: minimize risk (maximize return)
let risk_matrix = vec![vec![0.1; 10]; 10]; // Simplified risk matrix
qubo.add_quadratic_objective(&risk_matrix, 1.0)?;

// Constraint: select exactly 5 assets
let selection = vec![1.0; 10];
qubo.add_constraint_eq(&(0..10).collect::<Vec<_>>(), &selection, 5.0, 10.0)?;

// Solve
let formulation = qubo.build()?;
let ising_model = formulation.to_ising_model()?;
let simulator = ClassicalAnnealingSimulator::new(Default::default())?;
let result = simulator.solve(&ising_model)?;

let portfolio = formulation.to_binary_solution(&result.best_spins)?;
println!("Selected assets: {:?}", portfolio);
```

### 2. Advanced D-Wave Leap Integration

```rust
#[cfg(feature = "dwave")]
use quantrs2_anneal::dwave::{
    DWaveClient, SolverSelector, SolverCategory, EmbeddingConfig, 
    ChainStrengthMethod, AdvancedProblemParams, AnnealingSchedule
};

#[cfg(feature = "dwave")]
async fn solve_with_dwave() -> Result<(), Box<dyn std::error::Error>> {
    // Advanced solver selection
    let selector = SolverSelector {
        category: SolverCategory::QPU,
        min_qubits: Some(2000),
        max_queue_time: Some(60.0),
        topology_preference: Some("pegasus".to_string()),
        ..Default::default()
    };

    // Custom embedding configuration
    let embedding_config = EmbeddingConfig {
        auto_embed: true,
        chain_strength_method: ChainStrengthMethod::Adaptive(1.5),
        optimization_level: 3,
        ..Default::default()
    };

    // Advanced annealing parameters
    let schedule = AnnealingSchedule::pause_and_ramp(200.0, 100.0, 20.0);
    let params = AdvancedProblemParams {
        num_reads: 10000,
        anneal_schedule: Some(schedule),
        programming_thermalization: Some(2000),
        auto_scale: Some(true),
        ..Default::default()
    };

    let client = DWaveClient::new(std::env::var("DWAVE_TOKEN")?, None)?;
    let solution = client.submit_ising_with_embedding(
        &ising_model,
        None, // Auto-select best solver
        Some(params),
        Some(&embedding_config),
    )?;

    Ok(())
}
```

### 3. AWS Braket with Cost Management

```rust
#[cfg(feature = "braket")]
use quantrs2_anneal::braket::{
    BraketClient, DeviceSelector, DeviceType, CostTracker, AdvancedAnnealingParams
};

#[cfg(feature = "braket")]
async fn solve_with_braket() -> Result<(), Box<dyn std::error::Error>> {
    // Cost tracking setup
    let cost_tracker = CostTracker {
        cost_limit: Some(100.0), // $100 budget
        current_cost: 0.0,
        cost_estimates: HashMap::new(),
    };

    // Device selection for cost optimization
    let device_selector = DeviceSelector {
        device_type: Some(DeviceType::QuantumProcessor),
        max_cost_per_shot: Some(0.001), // Prefer cheaper devices
        required_capabilities: vec!["ANNEALING".to_string()],
        ..Default::default()
    };

    let client = BraketClient::with_config(
        std::env::var("AWS_ACCESS_KEY_ID")?,
        std::env::var("AWS_SECRET_ACCESS_KEY")?,
        None,
        "us-east-1",
        device_selector,
        cost_tracker,
    )?;

    // Submit with cost monitoring
    let task_result = client.submit_ising(&ising_model, None, None)?;
    let metrics = client.get_task_metrics(&task_result.task_arn)?;
    
    println!("Cost: ${:.4}", metrics.cost);
    println!("Best energy: {:.6}", metrics.best_energy.unwrap_or(0.0));

    Ok(())
}
```

### 4. Coherent Ising Machine for Large Problems

```rust
use quantrs2_anneal::{
    coherent_ising_machine::{CoherentIsingMachine, CIMParams},
    ising::IsingModel
};

// Large-scale continuous optimization problem
let mut large_model = IsingModel::new(10000);
// ... populate with problem data ...

let cim_params = CIMParams {
    pump_power: 2.0,
    detuning: 0.1,
    coupling_strength: 0.5,
    num_iterations: 1000,
    convergence_threshold: 1e-6,
    ..Default::default()
};

let cim = CoherentIsingMachine::new(cim_params)?;
let result = cim.solve(&large_model)?;

println!("CIM found energy: {}", result.best_energy);
```

### 5. Multi-Objective Optimization

```rust
use quantrs2_anneal::{
    multi_objective::{MultiObjectiveOptimizer, ObjectiveWeight},
    ising::IsingModel
};

// Portfolio optimization with multiple objectives: return, risk, diversity
let mut optimizer = MultiObjectiveOptimizer::new();

// Add objectives with weights
optimizer.add_objective("return", return_model, ObjectiveWeight::Maximize(0.4))?;
optimizer.add_objective("risk", risk_model, ObjectiveWeight::Minimize(0.4))?;
optimizer.add_objective("diversity", diversity_model, ObjectiveWeight::Maximize(0.2))?;

// Pareto frontier analysis
let pareto_solutions = optimizer.find_pareto_frontier(100)?;
for solution in &pareto_solutions {
    println!("Return: {:.3}, Risk: {:.3}, Diversity: {:.3}", 
             solution.objectives["return"],
             solution.objectives["risk"], 
             solution.objectives["diversity"]);
}
```

### 6. Energy System Optimization

```rust
use quantrs2_anneal::applications::energy::{
    PowerGridOptimizer, GridConstraints, RenewableIntegration
};

// Smart grid optimization with renewable integration
let grid_config = GridConstraints {
    total_demand: 1000.0, // MW
    renewable_capacity: 300.0, // MW
    storage_capacity: 100.0, // MWh
    transmission_limits: vec![200.0, 150.0, 250.0], // MW per line
    cost_coefficients: vec![50.0, 80.0, 120.0], // $/MWh per generator
};

let renewable_config = RenewableIntegration {
    solar_forecast: vec![0.0, 50.0, 150.0, 200.0, 150.0, 50.0], // Hourly MW
    wind_forecast: vec![80.0, 90.0, 70.0, 60.0, 85.0, 95.0], // Hourly MW
    storage_schedule: true,
    demand_response: true,
};

let optimizer = PowerGridOptimizer::new(grid_config, renewable_config)?;
let schedule = optimizer.optimize_24_hour_schedule()?;

println!("Optimized power generation schedule:");
for (hour, power) in schedule.generation_schedule.iter().enumerate() {
    println!("Hour {}: {:.1} MW", hour, power);
}
```

## 🧪 Testing and Benchmarking

The framework includes comprehensive testing and benchmarking capabilities:

```rust
use quantrs2_anneal::{
    applications::performance_benchmarks::{BenchmarkSuite, Algorithm},
    applications::integration_tests::IntegrationTestSuite
};

// Run performance benchmarks
let mut benchmark = BenchmarkSuite::new();
benchmark.add_algorithm(Algorithm::ClassicalAnnealing);
benchmark.add_algorithm(Algorithm::PopulationAnnealing);
benchmark.add_algorithm(Algorithm::CoherentIsingMachine);

let results = benchmark.run_all_benchmarks()?;
results.print_comparison_report();

// Run integration tests
let test_suite = IntegrationTestSuite::new();
let test_results = test_suite.run_all_tests()?;
println!("Integration tests: {}/{} passed", 
         test_results.passed, test_results.total);
```

## 📊 Visualization

```rust
use quantrs2_anneal::visualization::{
    EnergyLandscapeVisualizer, SolutionVisualizer, ConvergenceAnalyzer
};

// Visualize energy landscape
let visualizer = EnergyLandscapeVisualizer::new(&ising_model)?;
visualizer.plot_2d_projection("energy_landscape.svg")?;

// Analyze solution quality
let solution_viz = SolutionVisualizer::new(&result);
solution_viz.plot_energy_histogram("solution_quality.svg")?;

// Monitor convergence
let convergence = ConvergenceAnalyzer::new(&annealing_trace);
convergence.plot_convergence_curve("convergence.svg")?;
```

## 🔗 Integration with QuantRS2 Ecosystem

QuantRS2-Anneal integrates seamlessly with other QuantRS2 modules:

- **[quantrs2-core](../core/README.md)**: Shared types, error handling, and utilities
- **[quantrs2-circuit](../circuit/README.md)**: QAOA bridge for variational quantum algorithms
- **[quantrs2-ml](../ml/README.md)**: Quantum machine learning integration
- **[quantrs2-sim](../sim/README.md)**: Advanced quantum simulation backends

## 🚀 Performance Characteristics

| Algorithm | Problem Size | Time Complexity | Memory Usage | Best Use Case |
|-----------|--------------|-----------------|--------------|---------------|
| Classical Annealing | < 10,000 vars | O(n²·t) | O(n²) | General optimization |
| Population Annealing | < 1,000 vars | O(p·n²·t) | O(p·n) | High-quality sampling |
| Coherent Ising Machine | < 100,000 vars | O(n·t) | O(n) | Large continuous problems |
| D-Wave QPU | < 5,000 qubits | Constant | Cloud | Real quantum annealing |
| AWS Braket | < 30 qubits | Variable | Cloud | Multi-vendor quantum access |
| Hybrid Solvers | > 10,000 vars | O(n·log(n)·t) | Cloud | Very large problems |

## 📖 Documentation

- [API Documentation](https://docs.rs/quantrs2-anneal)
- [D-Wave Leap Client Guide](docs/dwave_leap_client.md)
- [AWS Braket Client Guide](docs/aws_braket_client.md)
- [Embedding Algorithms Guide](docs/embedding_guide.md)
- [Application Tutorials](docs/tutorials/)
- [Performance Optimization](docs/performance.md)

## 🛠️ Examples

Comprehensive examples are available in the [`examples/`](examples/) directory:

- **Basic Usage**: `simple_penalty_optimization.rs`, `advanced_embedding.rs`
- **Cloud Integration**: `dwave_leap_client_example.rs`, `aws_braket_client_example.rs`
- **Advanced Algorithms**: `coherent_ising_machine_example.rs`, `population_annealing_example.rs`
- **Applications**: `energy_optimization.rs`, `portfolio_optimization.rs`
- **Visualization**: `energy_landscape_visualization.rs`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.

## 🔬 Research and Citations

If you use QuantRS2-Anneal in your research, please cite:

```bibtex
@software{quantrs2_anneal,
  title={QuantRS2-Anneal: Comprehensive Quantum Annealing Framework},
  author={QuantRS2 Development Team},
  year={2024},
  url={https://github.com/cool-japan/quantrs}
}
```

---

**🌟 Ready to solve optimization problems with quantum annealing? Check out our [Quick Start Guide](docs/quickstart.md) and [Examples](examples/)!**