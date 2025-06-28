# QuantRS2 1.0 API Usage Examples

This document demonstrates the new organized API structure introduced in QuantRS2 1.0, showing how to use the hierarchical modules for different use cases.

## Migration from Beta to 1.0

### Old Way (Beta - Still Works but Deprecated)
```rust
use quantrs2_core::prelude::*;
use quantrs2_sim::prelude::*;

// This still works but shows deprecation warnings
let qubit = QubitId::new(0);
let simulator = StateVectorSimulator::new();
```

### New Way (1.0 - Recommended)
```rust
// For basic quantum programming
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

let qubit = QubitId::new(0);
let simulator = StateVectorSimulator::new();
```

## Use Case Examples

### 1. Basic Quantum Circuit Simulation

```rust
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

fn basic_simulation_example() -> Result<()> {
    // Create a quantum register
    let mut register = Register::<2>::new();
    
    // Create and run a simple circuit
    let mut simulator = StateVectorSimulator::new();
    
    // Basic gate operations would go here
    // let result = simulator.run(&circuit)?;
    
    Ok(())
}
```

### 2. Algorithm Development

```rust
use quantrs2_core::v1::algorithms::*;
use quantrs2_sim::v1::algorithms::*;

fn vqe_algorithm_example() -> Result<()> {
    // Use variational algorithm tools
    let vqe_config = VQEConfig::default();
    let optimizer = VQEWithAutodiff::new(vqe_config);
    
    // QAOA for optimization problems
    let qaoa_config = QAOAConfig::default();
    let qaoa_optimizer = QAOAOptimizer::new(qaoa_config);
    
    Ok(())
}
```

### 3. Hardware Programming

```rust
use quantrs2_core::v1::hardware::*;
use quantrs2_sim::v1::gpu::*;

fn hardware_programming_example() -> Result<()> {
    // Hardware abstraction layer
    let hardware_config = HardwareCapabilities::detect();
    
    // GPU-accelerated simulation
    #[cfg(feature = "gpu")]
    let gpu_simulator = GpuLinearAlgebra::new()?;
    
    // Pulse-level control
    let pulse_config = PulseSequence::new();
    
    Ok(())
}
```

### 4. Large-Scale Distributed Simulation

```rust
use quantrs2_sim::v1::distributed::*;

fn distributed_simulation_example() -> Result<()> {
    // Large-scale simulator for 40+ qubits
    let config = LargeScaleSimulatorConfig::default();
    let mut simulator = LargeScaleQuantumSimulator::new(config)?;
    
    // Distributed cluster simulation
    let distributed_config = DistributedSimulatorConfig::default();
    let mut distributed_sim = DistributedQuantumSimulator::new(distributed_config)?;
    
    Ok(())
}
```

### 5. Quantum Machine Learning

```rust
use quantrs2_sim::v1::algorithms::*;  // Includes quantum_ml

fn quantum_ml_example() -> Result<()> {
    // Quantum neural networks
    let qnn_config = QMLConfig::default();
    let quantum_network = QuantumNeuralNetwork::new(qnn_config);
    
    // Quantum reservoir computing
    let reservoir_config = QuantumReservoirConfig::default();
    let qrc = QuantumReservoirComputer::new(reservoir_config);
    
    // Variational quantum algorithms for ML
    let vqa_trainer = AdvancedVQATrainer::new();
    
    Ok(())
}
```

### 6. Advanced Research Applications

```rust
use quantrs2_core::v1::research::*;
use quantrs2_sim::v1::simulation::*;

fn research_example() -> Result<()> {
    // Tensor network methods
    let tensor_config = TensorNetworkSimulator::new();
    
    // Topological quantum computing
    let topological_config = TopologicalGate::new();
    
    // ZX-calculus optimization
    let zx_optimizer = ZXOptimizer::new();
    
    // Quantum networking
    let quantum_network = QuantumInternet::new();
    
    Ok(())
}
```

### 7. Developer Tools and Debugging

```rust
use quantrs2_core::v1::dev_tools::*;
use quantrs2_sim::v1::dev_tools::*;

fn debugging_example() -> Result<()> {
    // Quantum circuit debugger
    let debugger_config = DebugConfig::default();
    let mut debugger = QuantumDebugger::new(debugger_config);
    
    // Performance profiling
    let profiler_config = TelemetryConfig::default();
    let profiler = TelemetryCollector::new(profiler_config);
    
    // SciRS2 enhanced tools
    let linter = SciRS2QuantumLinter::new();
    let formatter = SciRS2QuantumFormatter::new();
    
    Ok(())
}
```

### 8. Performance Optimization

```rust
use quantrs2_sim::v1::simulation::*;  // Includes optimization

fn optimization_example() -> Result<()> {
    // Automatic backend selection
    let mut auto_optimizer = AutoOptimizer::new(AutoOptimizerConfig::default())?;
    
    // Performance prediction
    let mut predictor = create_performance_predictor()?;
    
    // Circuit optimization
    let optimizer = CircuitOptimizer::new();
    
    // Memory optimization
    let memory_optimizer = MemoryBandwidthOptimizer::new();
    
    Ok(())
}
```

## API Organization Benefits

### 1. Clear Intent
- `essentials::*` - For basic quantum programming
- `algorithms::*` - For algorithm development
- `hardware::*` - For hardware programming
- `distributed::*` - For large-scale simulation

### 2. No Naming Conflicts
```rust
// Before: Ambiguous
use quantrs2_sim::PerformanceMetrics; // Which one?

// After: Clear
use quantrs2_sim::v1::simulation::AutoOptimizerPerformanceMetrics;
use quantrs2_sim::v1::profiling::BenchmarkMemoryStats;
```

### 3. Logical Grouping
- All GPU-related functionality in `gpu::`
- All noise modeling in `noise_modeling::`
- All developer tools in `dev_tools::`

### 4. Easy Discovery
```rust
// Explore available simulation backends
use quantrs2_sim::v1::simulation::*;

// Find all optimization tools
use quantrs2_sim::v1::simulation::*;  // Includes optimization

// Access all developer tools
use quantrs2_sim::v1::dev_tools::*;
```

## Backward Compatibility

The new API maintains 100% backward compatibility:

```rust
// This still works (with deprecation warnings)
use quantrs2_core::prelude::*;
use quantrs2_sim::prelude::*;

// Migration path is gradual
use quantrs2_core::v1::essentials::*;  // Start with essentials
use quantrs2_sim::prelude::*;          // Keep old imports temporarily

// Eventually move to full new API
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;
```

## Best Practices

1. **Start with essentials** for basic quantum programming
2. **Use specific modules** for specialized functionality  
3. **Prefer v1 modules** for new code
4. **Migrate gradually** from old prelude imports
5. **Check deprecation warnings** for migration guidance

This organized API structure makes QuantRS2 more professional, easier to learn, and suitable for production quantum computing applications.