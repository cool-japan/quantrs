# QuantRS2 1.0 Comprehensive Examples Collection

This collection provides practical, runnable examples for all major features of QuantRS2 1.0, demonstrating the new organized API structure and best practices.

## Table of Contents

1. [Essential Quantum Programming Examples](#essential-quantum-programming-examples)
2. [Advanced Simulation Examples](#advanced-simulation-examples)
3. [GPU and High-Performance Examples](#gpu-and-high-performance-examples)
4. [Large-Scale and Distributed Examples](#large-scale-and-distributed-examples)
5. [Algorithm Development Examples](#algorithm-development-examples)
6. [Noise Modeling and Error Correction Examples](#noise-modeling-and-error-correction-examples)
7. [Developer Tools Examples](#developer-tools-examples)
8. [Hardware Programming Examples](#hardware-programming-examples)
9. [Research Applications Examples](#research-applications-examples)
10. [Performance Optimization Examples](#performance-optimization-examples)

## Essential Quantum Programming Examples

### Example 1: Basic Quantum States and Gates

```rust
// File: examples/essential/basic_states_gates.rs
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

/// Demonstrates basic quantum state creation and gate operations
fn main() -> Result<()> {
    println!("=== Basic Quantum States and Gates ===");
    
    // Create Bell state
    create_bell_state()?;
    
    // Create GHZ state
    create_ghz_state(4)?;
    
    // Demonstrate superposition
    demonstrate_superposition()?;
    
    // Show quantum interference
    demonstrate_interference()?;
    
    Ok(())
}

fn create_bell_state() -> Result<()> {
    println!("\n1. Creating Bell State |00⟩ + |11⟩");
    
    let mut simulator = StateVectorSimulator::new();
    
    // Apply Hadamard to create superposition
    simulator.h(0)?;
    
    // Apply CNOT to create entanglement
    simulator.cnot(0, 1)?;
    
    // Measure probabilities
    let probabilities = simulator.probabilities();
    for (state, prob) in probabilities.iter().enumerate() {
        if *prob > 1e-10 {
            println!("  |{:02b}⟩: {:.6}", state, prob);
        }
    }
    
    // Verify Bell state properties
    let entanglement = simulator.entanglement_entropy(vec![0])?;
    println!("  Entanglement entropy: {:.6}", entanglement);
    
    Ok(())
}

fn create_ghz_state(n_qubits: usize) -> Result<()> {
    println!("\n2. Creating {}-qubit GHZ State", n_qubits);
    
    let mut simulator = StateVectorSimulator::new();
    
    // Create GHZ state: |000...⟩ + |111...⟩
    simulator.h(0)?;
    for i in 1..n_qubits {
        simulator.cnot(0, i)?;
    }
    
    // Show probability distribution
    let probabilities = simulator.probabilities();
    println!("  Probability distribution:");
    for (state, prob) in probabilities.iter().enumerate() {
        if *prob > 1e-10 {
            println!("    |{:0width$b}⟩: {:.6}", state, prob, width = n_qubits);
        }
    }
    
    Ok(())
}

fn demonstrate_superposition() -> Result<()> {
    println!("\n3. Quantum Superposition with Different Angles");
    
    for angle in &[0.0, std::f64::consts::PI/4.0, std::f64::consts::PI/2.0, std::f64::consts::PI] {
        let mut simulator = StateVectorSimulator::new();
        
        // Rotate around Y-axis
        simulator.ry(*angle, 0)?;
        
        let probabilities = simulator.probabilities();
        println!("  Angle {:.2}π: |0⟩={:.3}, |1⟩={:.3}", 
                 angle / std::f64::consts::PI, probabilities[0], probabilities[1]);
    }
    
    Ok(())
}

fn demonstrate_interference() -> Result<()> {
    println!("\n4. Quantum Interference (Mach-Zehnder)");
    
    for phase in &[0.0, std::f64::consts::PI/2.0, std::f64::consts::PI] {
        let mut simulator = StateVectorSimulator::new();
        
        // Mach-Zehnder interferometer
        simulator.h(0)?;              // First beam splitter
        simulator.rz(*phase, 0)?;     // Phase shift
        simulator.h(0)?;              // Second beam splitter
        
        let probabilities = simulator.probabilities();
        println!("  Phase {:.2}π: Probability |0⟩ = {:.6}", 
                 phase / std::f64::consts::PI, probabilities[0]);
    }
    
    Ok(())
}
```

### Example 2: Parameterized Quantum Circuits

```rust
// File: examples/essential/parameterized_circuits.rs
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

/// Demonstrates parameterized quantum circuits and optimization
fn main() -> Result<()> {
    println!("=== Parameterized Quantum Circuits ===");
    
    // Basic parameterized circuit
    basic_parameterized_circuit()?;
    
    // Hardware-efficient ansatz
    hardware_efficient_ansatz()?;
    
    // Parameter optimization
    parameter_optimization_example()?;
    
    Ok(())
}

fn basic_parameterized_circuit() -> Result<()> {
    println!("\n1. Basic Parameterized Circuit");
    
    let parameters = vec![0.5, 1.2, 0.8, 2.1];
    
    let mut simulator = StateVectorSimulator::new();
    
    // Build parameterized circuit
    simulator.ry(parameters[0], 0)?;
    simulator.rz(parameters[1], 0)?;
    simulator.ry(parameters[2], 1)?;
    simulator.rz(parameters[3], 1)?;
    simulator.cnot(0, 1)?;
    
    // Measure expectation value of Pauli-Z
    let expectation_z = simulator.expectation_value_pauli_z(0)?;
    println!("  Expectation ⟨Z₀⟩ = {:.6}", expectation_z);
    
    Ok(())
}

fn hardware_efficient_ansatz() -> Result<()> {
    println!("\n2. Hardware-Efficient Ansatz");
    
    let n_qubits = 4;
    let n_layers = 3;
    let mut parameters = Vec::new();
    
    // Generate random parameters
    for _ in 0..(n_qubits * n_layers * 3) {
        parameters.push(rand::random::<f64>() * 2.0 * std::f64::consts::PI);
    }
    
    let mut simulator = StateVectorSimulator::new();
    
    let mut param_idx = 0;
    for layer in 0..n_layers {
        // Rotation layer
        for qubit in 0..n_qubits {
            simulator.rx(parameters[param_idx], qubit)?;
            param_idx += 1;
            simulator.ry(parameters[param_idx], qubit)?;
            param_idx += 1;
            simulator.rz(parameters[param_idx], qubit)?;
            param_idx += 1;
        }
        
        // Entangling layer
        for qubit in 0..n_qubits {
            simulator.cnot(qubit, (qubit + 1) % n_qubits)?;
        }
    }
    
    // Calculate circuit properties
    let avg_entanglement = calculate_average_entanglement(&simulator, n_qubits)?;
    println!("  Average entanglement: {:.6}", avg_entanglement);
    
    Ok(())
}

fn parameter_optimization_example() -> Result<()> {
    println!("\n3. Parameter Optimization Example");
    
    // Target: maximize ⟨Z₀⟩ for single qubit
    let mut best_params = vec![0.0, 0.0];
    let mut best_expectation = -1.0;
    
    // Simple grid search
    for theta in 0..20 {
        for phi in 0..20 {
            let theta_val = (theta as f64) * std::f64::consts::PI / 10.0;
            let phi_val = (phi as f64) * std::f64::consts::PI / 10.0;
            
            let mut simulator = StateVectorSimulator::new();
            simulator.ry(theta_val, 0)?;
            simulator.rz(phi_val, 0)?;
            
            let expectation = simulator.expectation_value_pauli_z(0)?;
            
            if expectation > best_expectation {
                best_expectation = expectation;
                best_params = vec![theta_val, phi_val];
            }
        }
    }
    
    println!("  Optimal parameters: θ={:.3}, φ={:.3}", best_params[0], best_params[1]);
    println!("  Maximum ⟨Z₀⟩ = {:.6}", best_expectation);
    
    Ok(())
}

fn calculate_average_entanglement(simulator: &StateVectorSimulator, n_qubits: usize) -> Result<f64> {
    let mut total_entanglement = 0.0;
    let mut count = 0;
    
    for i in 0..n_qubits {
        let partition = vec![i];
        let entanglement = simulator.entanglement_entropy(partition)?;
        total_entanglement += entanglement;
        count += 1;
    }
    
    Ok(total_entanglement / count as f64)
}
```

### Example 3: Circuit Optimization and Analysis

```rust
// File: examples/essential/circuit_optimization.rs
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

/// Demonstrates circuit optimization techniques
fn main() -> Result<()> {
    println!("=== Circuit Optimization and Analysis ===");
    
    // Gate fusion optimization
    gate_fusion_example()?;
    
    // Circuit depth reduction
    circuit_depth_reduction()?;
    
    // Gate count optimization
    gate_count_optimization()?;
    
    // Circuit equivalence checking
    circuit_equivalence_checking()?;
    
    Ok(())
}

fn gate_fusion_example() -> Result<()> {
    println!("\n1. Gate Fusion Optimization");
    
    // Create circuit with adjacent rotation gates
    let mut original_circuit = QuantumCircuit::new(2);
    original_circuit.add_gate(RZGate::new(0.1), 0)?;
    original_circuit.add_gate(RZGate::new(0.2), 0)?;
    original_circuit.add_gate(RZGate::new(0.3), 0)?;
    original_circuit.add_gate(RYGate::new(0.4), 1)?;
    original_circuit.add_gate(RYGate::new(0.5), 1)?;
    
    println!("  Original circuit: {} gates", original_circuit.gate_count());
    
    // Apply gate fusion optimization
    let optimizer = CircuitOptimizer::new();
    let optimized_circuit = optimize_circuit(&original_circuit)?;
    
    println!("  Optimized circuit: {} gates", optimized_circuit.gate_count());
    println!("  Reduction: {:.1}%", 
        100.0 * (1.0 - optimized_circuit.gate_count() as f64 / original_circuit.gate_count() as f64));
    
    // Verify equivalence
    let are_equivalent = verify_circuit_equivalence(&original_circuit, &optimized_circuit)?;
    println!("  Circuits equivalent: {}", are_equivalent);
    
    Ok(())
}

fn circuit_depth_reduction() -> Result<()> {
    println!("\n2. Circuit Depth Reduction");
    
    // Create circuit with parallelizable gates
    let mut circuit = QuantumCircuit::new(4);
    
    // Sequential gates that can be parallelized
    circuit.add_gate(HGate::new(), 0)?;
    circuit.add_gate(HGate::new(), 1)?;
    circuit.add_gate(HGate::new(), 2)?;
    circuit.add_gate(HGate::new(), 3)?;
    
    // More gates
    circuit.add_gate(RZGate::new(0.5), 0)?;
    circuit.add_gate(RZGate::new(0.6), 1)?;
    circuit.add_gate(RZGate::new(0.7), 2)?;
    circuit.add_gate(RZGate::new(0.8), 3)?;
    
    let original_depth = circuit.depth();
    println!("  Original depth: {}", original_depth);
    
    // Optimize for depth
    let depth_optimizer = DepthOptimizer::new();
    let depth_optimized = depth_optimizer.optimize(&circuit)?;
    
    println!("  Optimized depth: {}", depth_optimized.depth());
    println!("  Depth reduction: {:.1}%", 
        100.0 * (1.0 - depth_optimized.depth() as f64 / original_depth as f64));
    
    Ok(())
}

fn gate_count_optimization() -> Result<()> {
    println!("\n3. Gate Count Optimization");
    
    // Create circuit with redundant gates
    let mut circuit = QuantumCircuit::new(2);
    
    // Add gates that can be simplified
    circuit.add_gate(XGate::new(), 0)?;
    circuit.add_gate(XGate::new(), 0)?; // Double X = Identity
    circuit.add_gate(HGate::new(), 1)?;
    circuit.add_gate(ZGate::new(), 1)?;
    circuit.add_gate(HGate::new(), 1)?; // HZH = X
    
    println!("  Original gates: {}", circuit.gate_count());
    
    // Apply peephole optimization
    let peephole_optimizer = PeepholeOptimizer::new();
    let optimized = peephole_optimizer.optimize(&circuit)?;
    
    println!("  After peephole optimization: {}", optimized.gate_count());
    
    // Apply algebraic simplification
    let algebraic_optimizer = AlgebraicOptimizer::new();
    let final_circuit = algebraic_optimizer.optimize(&optimized)?;
    
    println!("  Final gates: {}", final_circuit.gate_count());
    println!("  Total reduction: {:.1}%", 
        100.0 * (1.0 - final_circuit.gate_count() as f64 / circuit.gate_count() as f64));
    
    Ok(())
}

fn circuit_equivalence_checking() -> Result<()> {
    println!("\n4. Circuit Equivalence Checking");
    
    // Create two circuits that should be equivalent
    let mut circuit1 = QuantumCircuit::new(2);
    circuit1.add_gate(HGate::new(), 0)?;
    circuit1.add_gate(CXGate::new(), (0, 1))?;
    
    let mut circuit2 = QuantumCircuit::new(2);
    circuit2.add_gate(RYGate::new(std::f64::consts::PI/2.0), 0)?;
    circuit2.add_gate(CXGate::new(), (0, 1))?;
    
    // Check equivalence
    let equivalence_checker = EquivalenceChecker::new();
    let result = equivalence_checker.check_equivalence(&circuit1, &circuit2)?;
    
    println!("  Circuits equivalent: {}", result.are_equivalent);
    println!("  Fidelity: {:.6}", result.fidelity);
    
    if !result.are_equivalent {
        println!("  Difference at qubit: {:?}", result.differing_qubits);
    }
    
    Ok(())
}
```

## Advanced Simulation Examples

### Example 4: Backend Selection and Performance

```rust
// File: examples/simulation/backend_selection.rs
use quantrs2_sim::v1::simulation::*;

/// Demonstrates automatic backend selection and performance comparison
fn main() -> Result<()> {
    println!("=== Backend Selection and Performance ===");
    
    // Automatic backend selection
    automatic_backend_selection()?;
    
    // Performance comparison
    performance_comparison()?;
    
    // Memory-efficient simulation
    memory_efficient_simulation()?;
    
    Ok(())
}

fn automatic_backend_selection() -> Result<()> {
    println!("\n1. Automatic Backend Selection");
    
    // Create circuits with different characteristics
    let circuits = vec![
        ("Small dense circuit", create_dense_circuit(8)),
        ("Large sparse circuit", create_sparse_circuit(20)),
        ("Clifford circuit", create_clifford_circuit(15)),
        ("Low entanglement", create_low_entanglement_circuit(25)),
    ];
    
    for (name, circuit) in circuits {
        let recommendation = recommend_backend_for_circuit(&circuit)?;
        
        println!("  {}: {:?}", name, recommendation.backend_type);
        println!("    Estimated time: {:.2}ms", recommendation.estimated_time_ms);
        println!("    Memory usage: {:.1}MB", recommendation.estimated_memory_mb);
        println!("    Confidence: {:.1}%", recommendation.confidence * 100.0);
    }
    
    Ok(())
}

fn performance_comparison() -> Result<()> {
    println!("\n2. Performance Comparison Across Backends");
    
    let circuit = create_benchmark_circuit(12);
    let backends = vec![
        ("State Vector", BackendType::StateVector),
        ("Tensor Network", BackendType::TensorNetwork),
        ("MPS", BackendType::MPS),
    ];
    
    for (name, backend_type) in backends {
        let start_time = std::time::Instant::now();
        
        match backend_type {
            BackendType::StateVector => {
                let mut simulator = StateVectorSimulator::new();
                simulator.run_circuit(&circuit)?;
            },
            BackendType::TensorNetwork => {
                let mut simulator = TensorNetworkSimulator::new();
                simulator.run_circuit(&circuit)?;
            },
            BackendType::MPS => {
                let mut simulator = MPSSimulator::new();
                simulator.run_circuit(&circuit)?;
            },
            _ => {}
        }
        
        let execution_time = start_time.elapsed();
        println!("  {}: {:.2}ms", name, execution_time.as_secs_f64() * 1000.0);
    }
    
    Ok(())
}

fn memory_efficient_simulation() -> Result<()> {
    println!("\n3. Memory-Efficient Large-Scale Simulation");
    
    let config = LargeScaleSimulatorConfig {
        max_qubits: 25,
        memory_limit_gb: 4,
        compression_algorithm: CompressionAlgorithm::LZ4,
        use_memory_mapping: true,
    };
    
    let mut simulator = LargeScaleQuantumSimulator::new(config)?;
    
    // Create large circuit
    for i in 0..25 {
        simulator.h(i)?;
        if i > 0 {
            simulator.cnot(i-1, i)?;
        }
    }
    
    // Monitor memory usage during execution
    let start_memory = simulator.memory_statistics();
    let result = simulator.run()?;
    let end_memory = simulator.memory_statistics();
    
    println!("  Initial memory: {:.1}MB", start_memory.total_memory_mb);
    println!("  Peak memory: {:.1}MB", end_memory.peak_memory_mb);
    println!("  Compression ratio: {:.2}", end_memory.compression_ratio);
    println!("  Final state vector size: {:.1}MB", end_memory.state_vector_size_mb);
    
    Ok(())
}

// Helper functions for creating different types of circuits
fn create_dense_circuit(n_qubits: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Dense all-to-all connectivity
    for i in 0..n_qubits {
        circuit.add_gate(HGate::new(), i).unwrap();
        for j in (i+1)..n_qubits {
            circuit.add_gate(CXGate::new(), (i, j)).unwrap();
        }
    }
    
    circuit
}

fn create_sparse_circuit(n_qubits: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Linear connectivity only
    for i in 0..n_qubits {
        circuit.add_gate(RYGate::new(0.5), i).unwrap();
        if i > 0 {
            circuit.add_gate(CXGate::new(), (i-1, i)).unwrap();
        }
    }
    
    circuit
}

fn create_clifford_circuit(n_qubits: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Only Clifford gates
    for i in 0..n_qubits {
        circuit.add_gate(HGate::new(), i).unwrap();
        circuit.add_gate(SGate::new(), i).unwrap();
        if i > 0 {
            circuit.add_gate(CXGate::new(), (i-1, i)).unwrap();
        }
    }
    
    circuit
}

fn create_low_entanglement_circuit(n_qubits: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Circuit with limited entanglement
    for i in 0..n_qubits {
        circuit.add_gate(RXGate::new(0.3), i).unwrap();
        circuit.add_gate(RZGate::new(0.4), i).unwrap();
    }
    
    // Only nearest-neighbor entanglement
    for i in 0..n_qubits-1 {
        circuit.add_gate(CXGate::new(), (i, i+1)).unwrap();
    }
    
    circuit
}

fn create_benchmark_circuit(n_qubits: usize) -> QuantumCircuit {
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Standard benchmark circuit
    for layer in 0..3 {
        for i in 0..n_qubits {
            circuit.add_gate(RYGate::new(0.1 * layer as f64), i).unwrap();
        }
        for i in 0..n_qubits-1 {
            circuit.add_gate(CXGate::new(), (i, i+1)).unwrap();
        }
    }
    
    circuit
}
```

## GPU and High-Performance Examples

### Example 5: GPU Acceleration

```rust
// File: examples/gpu/gpu_acceleration.rs
use quantrs2_sim::v1::gpu::*;

/// Demonstrates GPU-accelerated quantum simulation
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== GPU-Accelerated Quantum Simulation ===");
    
    // Check GPU availability
    if !check_gpu_availability() {
        println!("GPU not available, skipping GPU examples");
        return Ok(());
    }
    
    // Basic GPU simulation
    basic_gpu_simulation().await?;
    
    // GPU vs CPU performance comparison
    gpu_cpu_performance_comparison().await?;
    
    // Large-scale GPU simulation
    large_scale_gpu_simulation().await?;
    
    // Multi-GPU simulation
    multi_gpu_simulation().await?;
    
    Ok(())
}

fn check_gpu_availability() -> bool {
    GpuLinearAlgebra::is_available()
}

async fn basic_gpu_simulation() -> Result<()> {
    println!("\n1. Basic GPU Simulation");
    
    let gpu_simulator = GpuLinearAlgebra::new().await?;
    
    // Create Bell state on GPU
    let n_qubits = 15;
    let mut gpu_state = gpu_simulator.create_state_vector(n_qubits)?;
    
    // Apply Hadamard to first qubit
    gpu_simulator.apply_hadamard(&mut gpu_state, 0)?;
    
    // Apply CNOT gates to create entanglement
    for i in 1..n_qubits {
        gpu_simulator.apply_cnot(&mut gpu_state, 0, i)?;
    }
    
    // Measure probabilities
    let probabilities = gpu_simulator.get_probabilities(&gpu_state)?;
    
    // Count non-zero probabilities
    let non_zero_count = probabilities.iter().filter(|&&p| p > 1e-10).count();
    println!("  {} qubits simulated on GPU", n_qubits);
    println!("  {} non-zero probability amplitudes", non_zero_count);
    
    // GPU memory usage
    let memory_usage = gpu_simulator.memory_usage()?;
    println!("  GPU memory usage: {:.1}MB", memory_usage.used_mb);
    
    Ok(())
}

async fn gpu_cpu_performance_comparison() -> Result<()> {
    println!("\n2. GPU vs CPU Performance Comparison");
    
    let qubit_counts = vec![10, 15, 20];
    
    for n_qubits in qubit_counts {
        println!("  Testing {} qubits:", n_qubits);
        
        // CPU simulation
        let cpu_start = std::time::Instant::now();
        {
            let mut cpu_simulator = StateVectorSimulator::new();
            for i in 0..n_qubits {
                cpu_simulator.h(i)?;
                if i > 0 {
                    cpu_simulator.cnot(i-1, i)?;
                }
            }
            let _result = cpu_simulator.run()?;
        }
        let cpu_time = cpu_start.elapsed();
        
        // GPU simulation
        let gpu_start = std::time::Instant::now();
        {
            let gpu_simulator = GpuLinearAlgebra::new().await?;
            let mut gpu_state = gpu_simulator.create_state_vector(n_qubits)?;
            
            for i in 0..n_qubits {
                gpu_simulator.apply_hadamard(&mut gpu_state, i)?;
                if i > 0 {
                    gpu_simulator.apply_cnot(&mut gpu_state, i-1, i)?;
                }
            }
        }
        let gpu_time = gpu_start.elapsed();
        
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("    CPU: {:.2}ms, GPU: {:.2}ms, Speedup: {:.2}x", 
                 cpu_time.as_secs_f64() * 1000.0,
                 gpu_time.as_secs_f64() * 1000.0,
                 speedup);
    }
    
    Ok(())
}

async fn large_scale_gpu_simulation() -> Result<()> {
    println!("\n3. Large-Scale GPU Simulation");
    
    let n_qubits = 22; // Large for single machine
    
    let gpu_config = GpuConfig {
        memory_pool_size_mb: 4096,
        use_unified_memory: true,
        optimization_level: GpuOptimizationLevel::Aggressive,
    };
    
    let gpu_simulator = GpuLinearAlgebra::new_with_config(gpu_config).await?;
    
    // Check available GPU memory
    let available_memory = gpu_simulator.available_memory()?;
    println!("  Available GPU memory: {:.1}GB", available_memory.total_gb);
    
    if available_memory.free_gb < 2.0 {
        println!("  Insufficient GPU memory for {}-qubit simulation", n_qubits);
        return Ok(());
    }
    
    let mut gpu_state = gpu_simulator.create_state_vector(n_qubits)?;
    
    // Create quantum circuit with significant entanglement
    let start_time = std::time::Instant::now();
    
    // Random quantum circuit
    for layer in 0..10 {
        // Random single-qubit gates
        for qubit in 0..n_qubits {
            let angle = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
            gpu_simulator.apply_ry(&mut gpu_state, qubit, angle)?;
        }
        
        // Random two-qubit gates
        for qubit in 0..n_qubits-1 {
            if rand::random::<bool>() {
                gpu_simulator.apply_cnot(&mut gpu_state, qubit, qubit + 1)?;
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    
    // Memory usage analysis
    let final_memory = gpu_simulator.memory_usage()?;
    println!("  {} qubits, 10 layers executed in {:.2}s", n_qubits, execution_time.as_secs_f64());
    println!("  Peak GPU memory: {:.1}MB", final_memory.peak_mb);
    println!("  Memory efficiency: {:.1}%", final_memory.efficiency * 100.0);
    
    Ok(())
}

async fn multi_gpu_simulation() -> Result<()> {
    println!("\n4. Multi-GPU Simulation");
    
    let available_gpus = GpuBackendFactory::detect_gpu_count();
    println!("  Available GPUs: {}", available_gpus);
    
    if available_gpus < 2 {
        println!("  Multi-GPU simulation requires at least 2 GPUs");
        return Ok(());
    }
    
    let n_qubits = 25; // Larger than single GPU can handle
    
    let multi_gpu_config = MultiGpuConfig {
        gpu_count: available_gpus.min(4),
        distribution_strategy: GpuDistributionStrategy::Balanced,
        communication_backend: GpuCommunicationBackend::NCCL,
    };
    
    let multi_gpu_simulator = MultiGpuQuantumSimulator::new(multi_gpu_config).await?;
    
    // Distribute state vector across GPUs
    let mut distributed_state = multi_gpu_simulator.create_distributed_state_vector(n_qubits)?;
    
    // Apply gates across multiple GPUs
    let start_time = std::time::Instant::now();
    
    for i in 0..n_qubits {
        multi_gpu_simulator.apply_hadamard(&mut distributed_state, i)?;
        if i > 0 {
            multi_gpu_simulator.apply_cnot(&mut distributed_state, i-1, i)?;
        }
    }
    
    let execution_time = start_time.elapsed();
    
    // Analyze multi-GPU performance
    let gpu_stats = multi_gpu_simulator.get_gpu_statistics()?;
    println!("  {} qubits across {} GPUs in {:.2}s", n_qubits, available_gpus, execution_time.as_secs_f64());
    
    for (gpu_id, stats) in gpu_stats.iter() {
        println!("    GPU {}: {:.1}MB used, {:.1}% utilization", 
                 gpu_id, stats.memory_used_mb, stats.utilization_percent);
    }
    
    let total_memory = gpu_stats.values().map(|s| s.memory_used_mb).sum::<f64>();
    println!("  Total GPU memory: {:.1}MB", total_memory);
    
    Ok(())
}
```

## Continue with Additional Examples...

This comprehensive examples collection demonstrates the practical usage of QuantRS2 1.0's organized API structure. Each example is designed to be runnable and educational, showing best practices for different use cases.

The examples progress from basic quantum operations to advanced research applications, providing a complete learning path for users at all levels. The new API organization makes it easy to find relevant functionality and understand the intent behind each module.

For the complete collection of examples covering all features, see the individual example files in the repository's examples directory.