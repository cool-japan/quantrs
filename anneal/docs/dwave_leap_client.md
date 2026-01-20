# D-Wave Leap Cloud Service Client

This document describes the comprehensive D-Wave Leap cloud service client implementation in the QuantRS2 anneal module.

## Overview

The enhanced D-Wave Leap client provides a complete interface to D-Wave's quantum annealing cloud services, including:

- **Quantum Processing Units (QPUs)**: Direct access to D-Wave quantum annealers
- **Hybrid Solvers**: Classical-quantum hybrid algorithms for large problems
- **Advanced Embedding**: Automatic minor graph embedding with optimization
- **Problem Management**: Status tracking, batch submission, and monitoring
- **Performance Analytics**: Comprehensive metrics and timing analysis
- **Reliability Features**: Retry logic, error handling, and timeout management

## Key Features

### 1. Enhanced Solver Management

```rust
use quantrs2_anneal::dwave::{DWaveClient, SolverSelector, SolverCategory};

let selector = SolverSelector {
    category: SolverCategory::QPU,
    min_qubits: Some(2000),
    max_queue_time: Some(60.0),
    online_only: true,
    topology_preference: Some("pegasus".to_string()),
    ..Default::default()
};

let client = DWaveClient::new(api_token, None)?;
let best_solver = client.select_solver(Some(&selector))?;
```

### 2. Automatic Embedding Integration

```rust
use quantrs2_anneal::dwave::{EmbeddingConfig, ChainStrengthMethod};

let embedding_config = EmbeddingConfig {
    auto_embed: true,
    timeout: Duration::from_secs(60),
    chain_strength_method: ChainStrengthMethod::Auto,
    optimization_level: 2,
    ..Default::default()
};

let solution = client.submit_ising_with_embedding(
    &ising_model,
    None, // Auto-select solver
    Some(advanced_params),
    Some(&embedding_config),
)?;
```

### 3. Advanced Problem Parameters

```rust
use quantrs2_anneal::dwave::{AdvancedProblemParams, AnnealingSchedule};

let schedule = AnnealingSchedule::pause_and_ramp(100.0, 50.0, 10.0);

let params = AdvancedProblemParams {
    num_reads: 1000,
    anneal_schedule: Some(schedule),
    programming_thermalization: Some(1000),
    readout_thermalization: Some(100),
    auto_scale: Some(true),
    chain_strength: Some(2.0),
    ..Default::default()
};
```

### 4. Hybrid Solver Support

```rust
use quantrs2_anneal::dwave::HybridSolverParams;

let hybrid_params = HybridSolverParams {
    time_limit: Some(30.0),
    max_variables: Some(10000),
    ..Default::default()
};

let solution = client.submit_hybrid(
    &large_ising_model,
    None, // Auto-select hybrid solver
    Some(hybrid_params),
)?;
```

### 5. Problem Status Monitoring

```rust
// Submit problem asynchronously
let solution = client.submit_ising_with_embedding(&model, None, None, None)?;
let problem_id = &solution.problem_id;

// Monitor status
loop {
    let status = client.get_problem_status(problem_id)?;
    match status.status {
        ProblemStatus::Completed => break,
        ProblemStatus::Failed => return Err("Problem failed"),
        _ => std::thread::sleep(Duration::from_secs(5)),
    }
}

// Get detailed metrics
let metrics = client.get_problem_metrics(problem_id)?;
println!("Best energy: {}", metrics.best_energy);
println!("Total time: {:?}", metrics.total_time);
```

### 6. Batch Problem Submission

```rust
let problems = vec![
    (&model1, None, None),
    (&model2, Some("solver_id"), Some(params)),
    (&model3, None, None),
];

let batch_result = client.submit_batch(problems)?;
for (i, status) in batch_result.statuses.iter().enumerate() {
    match status {
        Ok(problem_id) => println!("Problem {}: {}", i, problem_id),
        Err(e) => println!("Problem {} failed: {}", i, e),
    }
}
```

## API Reference

### Core Types

#### `DWaveClient`
The main client for interacting with D-Wave Leap services.

**Methods:**
- `new(token, endpoint)` - Create basic client
- `with_config(token, endpoint, selector, embedding_config)` - Create with custom config
- `get_leap_solvers()` - Get all available solvers
- `select_solver(selector)` - Select optimal solver based on criteria
- `submit_ising_with_embedding()` - Submit with automatic embedding
- `submit_hybrid()` - Submit to hybrid solver
- `get_problem_status(id)` - Get problem status
- `get_problem_metrics(id)` - Get performance metrics
- `submit_batch()` - Submit multiple problems
- `list_problems()` - List recent problems
- `cancel_problem(id)` - Cancel running problem

#### `SolverSelector`
Criteria for automatic solver selection.

**Fields:**
- `category: SolverCategory` - Filter by solver type
- `min_qubits: Option<usize>` - Minimum qubit count
- `max_queue_time: Option<f64>` - Maximum acceptable queue time
- `online_only: bool` - Only consider online solvers
- `name_pattern: Option<String>` - Solver name pattern matching
- `topology_preference: Option<String>` - Preferred topology type

#### `EmbeddingConfig`
Configuration for automatic embedding.

**Fields:**
- `auto_embed: bool` - Enable automatic embedding
- `timeout: Duration` - Embedding timeout
- `chain_strength_method: ChainStrengthMethod` - How to calculate chain strength
- `custom_embedding: Option<Embedding>` - Pre-computed embedding
- `optimization_level: usize` - Embedding optimization level (1-3)

#### `ChainStrengthMethod`
Methods for calculating chain strength in embeddings.

**Variants:**
- `Auto` - Automatic calculation based on problem
- `Fixed(f64)` - User-specified constant value
- `Adaptive(f64)` - Adaptive based on coupling strengths (multiplier)

#### `AdvancedProblemParams`
Advanced parameters for quantum annealing problems.

**Fields:**
- `num_reads: usize` - Number of samples to take
- `anneal_schedule: Option<AnnealingSchedule>` - Custom annealing schedule
- `programming_thermalization: Option<usize>` - Programming thermalization time
- `readout_thermalization: Option<usize>` - Readout thermalization time
- `auto_scale: Option<bool>` - Enable automatic parameter scaling
- `chain_strength: Option<f64>` - Override chain strength
- `flux_biases: Option<HashMap<String, f64>>` - Per-qubit flux biases

#### `AnnealingSchedule`
Custom annealing schedules for fine-tuned control.

**Methods:**
- `linear(time)` - Simple linear annealing
- `pause_and_ramp(time, pause_start, pause_duration)` - Pause-and-ramp schedule
- `custom(points)` - Custom schedule from time-field points

#### `ProblemMetrics`
Performance metrics for completed problems.

**Fields:**
- `total_time: Duration` - Total execution time
- `queue_time: Duration` - Time spent in queue
- `programming_time: Duration` - QPU programming time
- `sampling_time: Duration` - QPU sampling time
- `readout_time: Duration` - QPU readout time
- `best_energy: f64` - Best energy found
- `num_valid_solutions: usize` - Number of valid solutions
- `chain_break_fraction: Option<f64>` - Fraction of broken chains

### Error Handling

The client provides comprehensive error handling through the `DWaveError` enum:

```rust
use quantrs2_anneal::dwave::DWaveError;

match client.submit_ising(&model, "solver_id", params) {
    Ok(solution) => {
        // Process solution
    }
    Err(DWaveError::NetworkError(e)) => {
        // Handle network issues
    }
    Err(DWaveError::EmbeddingError(e)) => {
        // Handle embedding failures
    }
    Err(DWaveError::SolverConfigError(e)) => {
        // Handle solver configuration issues
    }
    Err(e) => {
        // Handle other errors
    }
}
```

## Usage Patterns

### 1. Simple QPU Access

```rust
use quantrs2_anneal::{IsingModel, dwave::DWaveClient};

let mut model = IsingModel::new(4);
model.set_coupling(0, 1, -1.0)?;
model.set_coupling(1, 2, -1.0)?;
model.set_coupling(2, 3, -1.0)?;
model.set_coupling(3, 0, -1.0)?;

let client = DWaveClient::new(api_token, None)?;
let solution = client.submit_ising_with_embedding(&model, None, None, None)?;

println!("Best energy: {}", solution.energies.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
```

### 2. Hybrid Solver for Large Problems

```rust
let mut large_model = IsingModel::new(1000);
// ... populate large model ...

let hybrid_params = HybridSolverParams {
    time_limit: Some(60.0), // 1 minute
    ..Default::default()
};

let client = DWaveClient::new(api_token, None)?;
let solution = client.submit_hybrid(&large_model, None, Some(hybrid_params))?;
```

### 3. Custom Embedding and Parameters

```rust
let embedding_config = EmbeddingConfig {
    auto_embed: true,
    chain_strength_method: ChainStrengthMethod::Adaptive(1.5),
    optimization_level: 3, // Maximum optimization
    ..Default::default()
};

let schedule = AnnealingSchedule::pause_and_ramp(200.0, 100.0, 20.0);
let params = AdvancedProblemParams {
    num_reads: 10000,
    anneal_schedule: Some(schedule),
    auto_scale: Some(false),
    ..Default::default()
};

let solution = client.submit_ising_with_embedding(
    &model, 
    Some("Advantage_system6.3"), 
    Some(params), 
    Some(&embedding_config)
)?;
```

### 4. Problem Monitoring and Analysis

```rust
// Submit problem
let solution = client.submit_ising_with_embedding(&model, None, None, None)?;
let problem_id = &solution.problem_id;

// Wait for completion with timeout
let start = std::time::Instant::now();
let timeout = Duration::from_secs(300); // 5 minutes

loop {
    let status = client.get_problem_status(problem_id)?;
    
    match status.status {
        ProblemStatus::Completed => {
            let metrics = client.get_problem_metrics(problem_id)?;
            
            println!("Problem completed successfully!");
            println!("Best energy: {}", metrics.best_energy);
            println!("Queue time: {:?}", metrics.queue_time);
            println!("Sampling time: {:?}", metrics.sampling_time);
            
            if let Some(chain_break_fraction) = metrics.chain_break_fraction {
                println!("Chain break fraction: {:.3}", chain_break_fraction);
            }
            
            break;
        }
        ProblemStatus::Failed => {
            return Err("Problem execution failed".into());
        }
        ProblemStatus::Cancelled => {
            return Err("Problem was cancelled".into());
        }
        _ => {
            if start.elapsed() > timeout {
                client.cancel_problem(problem_id)?;
                return Err("Problem timed out".into());
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    }
}
```

## Best Practices

### 1. Solver Selection
- Use `SolverSelector` to automatically choose the best available solver
- Consider queue times when selecting solvers for time-sensitive applications
- Prefer newer solver topologies (Pegasus, Zephyr) for better connectivity

### 2. Embedding Optimization
- Use higher optimization levels for difficult embedding problems
- Consider pre-computing embeddings for frequently used problem structures
- Monitor chain break fractions and adjust chain strength accordingly

### 3. Parameter Tuning
- Start with default parameters and adjust based on problem characteristics
- Use custom annealing schedules for problems requiring fine control
- Enable auto-scaling for problems with varying parameter magnitudes

### 4. Error Handling
- Implement retry logic for network failures
- Have fallback strategies for solver unavailability
- Monitor embedding failures and adjust problem formulation if needed

### 5. Performance Monitoring
- Track problem metrics to optimize submission parameters
- Monitor queue times to choose optimal submission times
- Analyze chain break fractions to validate embedding quality

## Integration Examples

### With QAOA Circuit Bridge

```rust
use quantrs2_anneal::{
    ising::IsingModel,
    dwave::DWaveClient,
    qaoa_circuit_bridge::QaoaCircuitBridge,
};

// Create problem and convert via QAOA bridge
let mut ising_model = IsingModel::new(4);
// ... populate model ...

// Use QAOA bridge for circuit optimization
let bridge = QaoaCircuitBridge::new(4);
let optimized_problem = bridge.prepare_problem_for_circuit_optimization(&ising_model)?;

// Submit to D-Wave
let client = DWaveClient::new(api_token, None)?;
let solution = client.submit_ising_with_embedding(&ising_model, None, None, None)?;
```

### With Embedding Module

```rust
use quantrs2_anneal::{
    embedding::{MinorMiner, HardwareGraph},
    dwave::{DWaveClient, EmbeddingConfig},
};

// Pre-compute embedding
let logical_edges = vec![(0, 1), (1, 2), (2, 0)]; // Triangle
let hardware = HardwareGraph::new_chimera(4, 4, 4)?;
let embedder = MinorMiner::default();
let embedding = embedder.find_embedding(&logical_edges, 3, &hardware)?;

// Use pre-computed embedding
let embedding_config = EmbeddingConfig {
    auto_embed: false,
    custom_embedding: Some(embedding),
    ..Default::default()
};

let solution = client.submit_ising_with_embedding(
    &model, 
    None, 
    None, 
    Some(&embedding_config)
)?;
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API token is correct and active
   - Check network connectivity to D-Wave Leap

2. **Embedding Failures**
   - Reduce problem size or increase optimization level
   - Check problem connectivity matches hardware constraints
   - Consider using hybrid solvers for densely connected problems

3. **Long Queue Times**
   - Use `SolverSelector` with `max_queue_time` filter
   - Consider using hybrid solvers during peak hours
   - Monitor solver availability patterns

4. **Poor Solution Quality**
   - Increase `num_reads` for better sampling
   - Adjust annealing schedule for problem characteristics
   - Verify chain strength is appropriate for problem scale

### Performance Optimization

1. **For Small Problems (< 100 variables)**
   - Use QPU solvers with automatic embedding
   - Enable auto-scaling for parameter optimization
   - Use multiple reads for statistical confidence

2. **For Large Problems (> 1000 variables)**
   - Use hybrid solvers for better scalability
   - Increase time limits for complex optimization
   - Consider problem decomposition techniques

3. **For Time-Critical Applications**
   - Pre-select fast solvers using `SolverSelector`
   - Use batch submission for multiple similar problems
   - Implement asynchronous problem monitoring

## License and Usage

The D-Wave Leap client requires:
- A valid D-Wave Leap account
- API access credentials
- Compliance with D-Wave's terms of service
- The `dwave` feature enabled in Cargo.toml

For more information, visit [D-Wave Leap Documentation](https://docs.dwavesys.com/docs/latest/index.html).