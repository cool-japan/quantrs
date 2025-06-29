# QuantRS2 1.0 Comprehensive Tutorial

This tutorial provides complete coverage of the QuantRS2 1.0 API, demonstrating all major features through practical examples. The new organized API makes quantum programming more intuitive and discoverable.

## Table of Contents

1. [Getting Started with the New API](#getting-started-with-the-new-api)
2. [Essential Quantum Programming](#essential-quantum-programming)
3. [Advanced Simulation Techniques](#advanced-simulation-techniques)
4. [GPU and High-Performance Computing](#gpu-and-high-performance-computing)
5. [Large-Scale and Distributed Simulation](#large-scale-and-distributed-simulation)
6. [Algorithm Development and Research](#algorithm-development-and-research)
7. [Noise Modeling and Error Correction](#noise-modeling-and-error-correction)
8. [Developer Tools and Debugging](#developer-tools-and-debugging)
9. [Hardware Programming](#hardware-programming)
10. [Migration from Beta to 1.0](#migration-from-beta-to-10)

## Getting Started with the New API

QuantRS2 1.0 introduces a hierarchical API organization with clear intent and logical grouping. Instead of importing everything from a flat prelude, you now choose specific modules based on your use case.

### API Organization Overview

```rust
// For basic quantum programming
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

// For algorithm development
use quantrs2_core::v1::algorithms::*;
use quantrs2_sim::v1::algorithms::*;

// For hardware programming
use quantrs2_core::v1::hardware::*;
use quantrs2_sim::v1::gpu::*;

// For research applications
use quantrs2_core::v1::research::*;
use quantrs2_sim::v1::simulation::*;
```

## Essential Quantum Programming

The `essentials` module provides everything needed for basic quantum circuit programming.

### 1. Creating Your First Quantum Circuit

```rust
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

fn create_bell_state() -> Result<()> {
    // Create a quantum register
    let mut register = Register::<2>::new();
    
    // Create a simulator
    let mut simulator = StateVectorSimulator::new();
    
    // Apply quantum gates
    let h_gate = HGate::new();
    let cnot_gate = CXGate::new();
    
    // Build Bell state: |00⟩ + |11⟩
    simulator.apply_gate(&h_gate, QubitId::new(0))?;
    simulator.apply_gate(&cnot_gate, (QubitId::new(0), QubitId::new(1)))?;
    
    // Measure and display results
    let results = simulator.run(1000)?;
    for (state, count) in results.counts().iter().enumerate() {
        println!("|{:02b}⟩: {} times", state, count);
    }
    
    Ok(())
}
```

### 2. Working with Parametric Gates

```rust
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

fn parametric_gates_example() -> Result<()> {
    let mut simulator = StateVectorSimulator::new();
    
    // Parametric rotation gates
    let rx_gate = RXGate::new(std::f64::consts::PI / 4.0);
    let ry_gate = RYGate::new(std::f64::consts::PI / 3.0);
    let rz_gate = RZGate::new(std::f64::consts::PI / 6.0);
    
    // Create superposition
    simulator.apply_gate(&HGate::new(), QubitId::new(0))?;
    
    // Apply parametric rotations
    simulator.apply_gate(&rx_gate, QubitId::new(0))?;
    simulator.apply_gate(&ry_gate, QubitId::new(0))?;
    simulator.apply_gate(&rz_gate, QubitId::new(0))?;
    
    // Get final state amplitudes
    let state_vector = simulator.state_vector();
    println!("Final state: {:?}", state_vector);
    
    Ok(())
}
```

### 3. Multi-Qubit Entanglement

```rust
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

fn ghz_state_example() -> Result<()> {
    let mut simulator = StateVectorSimulator::new();
    let n_qubits = 5;
    
    // Create GHZ state: |00000⟩ + |11111⟩
    simulator.apply_gate(&HGate::new(), QubitId::new(0))?;
    
    for i in 1..n_qubits {
        simulator.apply_gate(
            &CXGate::new(), 
            (QubitId::new(0), QubitId::new(i))
        )?;
    }
    
    // Verify entanglement
    let entanglement_measure = simulator.entanglement_entropy(vec![0, 1, 2])?;
    println!("Entanglement entropy: {:.6}", entanglement_measure);
    
    Ok(())
}
```

### 4. Circuit Optimization

```rust
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

fn circuit_optimization_example() -> Result<()> {
    // Create a circuit that can be optimized
    let mut circuit = QuantumCircuit::new(3);
    
    // Add gates that can be fused
    circuit.add_gate(HGate::new(), QubitId::new(0))?;
    circuit.add_gate(RZGate::new(0.1), QubitId::new(0))?;
    circuit.add_gate(RZGate::new(0.2), QubitId::new(0))?; // Can be fused with previous
    circuit.add_gate(HGate::new(), QubitId::new(0))?;
    
    // Optimize the circuit
    let optimizer = CircuitOptimizer::new();
    let optimized_circuit = optimize_circuit(&circuit)?;
    
    println!("Original gates: {}", circuit.gate_count());
    println!("Optimized gates: {}", optimized_circuit.gate_count());
    println!("Reduction: {:.1}%", 
        100.0 * (1.0 - optimized_circuit.gate_count() as f64 / circuit.gate_count() as f64));
    
    Ok(())
}
```

## Advanced Simulation Techniques

The `simulation` module provides access to all advanced simulation backends and optimization techniques.

### 1. Automatic Backend Selection

```rust
use quantrs2_sim::v1::simulation::*;

fn automatic_backend_selection() -> Result<()> {
    // Create an auto-optimizer that selects the best backend
    let config = AutoOptimizerConfig {
        analysis_depth: AnalysisDepth::Deep,
        fallback_strategy: FallbackStrategy::Conservative,
        optimization_level: AutoOptimizationLevel::Aggressive,
    };
    
    let auto_optimizer = AutoOptimizer::new(config);
    
    // Define a circuit
    let circuit = create_sample_circuit(20); // 20-qubit circuit
    
    // Get backend recommendation
    let recommendation = recommend_backend_for_circuit(&circuit)?;
    
    match recommendation.backend_type {
        BackendType::StateVector => {
            println!("Recommended: State vector simulation");
            let simulator = StateVectorSimulator::new();
            // Run simulation...
        },
        BackendType::TensorNetwork => {
            println!("Recommended: Tensor network simulation");
            let simulator = TensorNetworkSimulator::new();
            // Run simulation...
        },
        BackendType::MPS => {
            println!("Recommended: Matrix Product State simulation");
            let simulator = MPSSimulator::new();
            // Run simulation...
        },
        BackendType::GPU => {
            println!("Recommended: GPU-accelerated simulation");
            #[cfg(feature = "gpu")]
            {
                let simulator = GpuStateVectorSimulator::new()?;
                // Run simulation...
            }
        }
    }
    
    Ok(())
}
```

### 2. Performance Prediction

```rust
use quantrs2_sim::v1::simulation::*;

fn performance_prediction_example() -> Result<()> {
    // Create performance predictor
    let predictor = create_performance_predictor()?;
    
    let circuit = create_sample_circuit(15);
    
    // Predict execution time for different backends
    let prediction_result = predict_circuit_execution_time(
        &predictor, 
        &circuit, 
        &BackendType::StateVector
    )?;
    
    println!("Predicted execution time: {:.2}ms", prediction_result.execution_time_ms);
    println!("Memory usage: {:.1}MB", prediction_result.memory_usage_mb);
    println!("Confidence: {:.1}%", prediction_result.confidence * 100.0);
    
    // Compare different backends
    for backend in &[BackendType::StateVector, BackendType::TensorNetwork, BackendType::MPS] {
        let pred = predict_circuit_execution_time(&predictor, &circuit, backend)?;
        println!("{:?}: {:.2}ms", backend, pred.execution_time_ms);
    }
    
    Ok(())
}
```

### 3. Large-Scale Memory-Efficient Simulation

```rust
use quantrs2_sim::v1::simulation::*;

fn large_scale_simulation() -> Result<()> {
    // Configure for memory efficiency
    let config = LargeScaleSimulatorConfig {
        max_qubits: 30,
        memory_limit_gb: 8,
        compression_algorithm: CompressionAlgorithm::LZ4,
        use_memory_mapping: true,
    };
    
    let mut simulator = LargeScaleQuantumSimulator::new(config)?;
    
    // Create a 25-qubit circuit
    for i in 0..25 {
        simulator.h(i)?;
        if i > 0 {
            simulator.cnot(i-1, i)?;
        }
    }
    
    // Monitor memory usage
    let memory_stats = simulator.memory_statistics();
    println!("Memory usage: {:.1}MB", memory_stats.total_memory_mb);
    println!("Compression ratio: {:.2}", memory_stats.compression_ratio);
    
    // Execute with progress tracking
    let result = simulator.run_with_progress(|progress| {
        println!("Progress: {:.1}%", progress * 100.0);
    })?;
    
    Ok(())
}
```

## GPU and High-Performance Computing

The `gpu` module provides access to GPU acceleration and high-performance computing features.

### 1. GPU-Accelerated Simulation

```rust
use quantrs2_sim::v1::gpu::*;

async fn gpu_simulation_example() -> Result<()> {
    // Check GPU availability
    if !GpuLinearAlgebra::is_available() {
        println!("GPU not available, falling back to CPU");
        return Ok(());
    }
    
    // Create GPU-accelerated simulator
    let gpu_simulator = GpuLinearAlgebra::new()?;
    
    // Configure for optimal GPU usage
    let config = GpuConfig {
        memory_pool_size_mb: 2048,
        use_unified_memory: true,
        optimization_level: GpuOptimizationLevel::Aggressive,
    };
    
    // Create large state vector on GPU
    let n_qubits = 20;
    let mut gpu_state = gpu_simulator.create_state_vector(n_qubits)?;
    
    // Apply gates using GPU kernels
    gpu_simulator.apply_hadamard_batch(&mut gpu_state, &[0, 1, 2, 3])?;
    
    for i in 0..n_qubits-1 {
        gpu_simulator.apply_cnot(&mut gpu_state, i, i+1)?;
    }
    
    // Measure and get results
    let probabilities = gpu_simulator.get_probabilities(&gpu_state)?;
    
    println!("GPU simulation completed for {} qubits", n_qubits);
    println!("Top 5 most probable states:");
    
    let mut prob_indices: Vec<_> = probabilities.iter().enumerate().collect();
    prob_indices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    
    for (i, (state, prob)) in prob_indices.iter().take(5).enumerate() {
        println!("  |{:020b}⟩: {:.6}", state, prob);
    }
    
    Ok(())
}
```

### 2. SIMD-Accelerated Operations

```rust
use quantrs2_sim::v1::gpu::*;

fn simd_operations_example() -> Result<()> {
    // Use SciRS2 SIMD operations for vectorized quantum computing
    let complex_vector = ComplexSimdVector::new(1024);
    
    // Apply single-qubit gates with SIMD acceleration
    apply_single_qubit_gate_complex_simd(
        &mut complex_vector.data,
        0, // qubit index
        &HGate::new().matrix()
    )?;
    
    // Apply Hadamard to multiple qubits simultaneously
    for qubit in 0..10 {
        apply_hadamard_gate_complex_simd(&mut complex_vector.data, qubit)?;
    }
    
    // Apply CNOT gates
    apply_cnot_complex_simd(&mut complex_vector.data, 0, 1)?;
    
    // Benchmark SIMD performance
    let benchmark_result = benchmark_complex_simd_operations(1000)?;
    println!("SIMD performance: {:.2} operations/second", benchmark_result.ops_per_second);
    
    Ok(())
}
```

### 3. Multi-Platform GPU Support

```rust
use quantrs2_sim::v1::gpu::*;

fn multi_platform_gpu() -> Result<()> {
    // Auto-detect available GPU backends
    let available_backends = GpuBackendFactory::detect_available_backends();
    
    for backend in available_backends {
        match backend {
            GpuBackendType::CUDA => {
                println!("CUDA backend available");
                let cuda_simulator = CudaQuantumSimulator::new()?;
                // Use CUDA-specific optimizations
            },
            GpuBackendType::OpenCL => {
                println!("OpenCL backend available");
                let opencl_simulator = OpenCLQuantumSimulator::new()?;
                // Use OpenCL for cross-platform compatibility
            },
            GpuBackendType::Metal => {
                println!("Metal backend available (macOS)");
                let metal_simulator = MetalQuantumSimulator::new()?;
                // Use Metal for Apple Silicon optimization
            },
            GpuBackendType::Vulkan => {
                println!("Vulkan backend available");
                let vulkan_simulator = VulkanQuantumSimulator::new()?;
                // Use Vulkan for modern cross-platform GPU computing
            }
        }
    }
    
    Ok(())
}
```

## Large-Scale and Distributed Simulation

The `distributed` module enables simulation of quantum systems beyond the limits of single machines.

### 1. Distributed Quantum Simulation

```rust
use quantrs2_sim::v1::distributed::*;

fn distributed_simulation_example() -> Result<()> {
    // Configure distributed simulation
    let network_config = NetworkConfig {
        cluster_nodes: vec![
            "node1.cluster.local:8080".to_string(),
            "node2.cluster.local:8080".to_string(),
            "node3.cluster.local:8080".to_string(),
        ],
        communication_backend: CommunicationBackend::MPI,
    };
    
    let load_balancing_config = LoadBalancingConfig {
        strategy: DistributedLoadBalancingStrategy::Dynamic,
        rebalancing_threshold: 0.2,
        work_stealing: true,
    };
    
    let config = DistributedSimulatorConfig {
        network: network_config,
        load_balancing: load_balancing_config,
        max_qubits_per_node: 20,
        fault_tolerance: FaultToleranceConfig::default(),
    };
    
    // Create distributed simulator
    let mut distributed_sim = DistributedQuantumSimulator::new(config)?;
    
    // Create a large quantum circuit (40+ qubits)
    let n_qubits = 45;
    
    // Build quantum circuit across distributed nodes
    for i in 0..n_qubits {
        distributed_sim.h(i)?; // Distributed Hadamard
        if i > 0 {
            distributed_sim.cnot(i-1, i)?; // Distributed CNOT
        }
    }
    
    // Execute with automatic load balancing
    let result = distributed_sim.run_distributed()?;
    
    // Monitor performance across nodes
    let perf_stats = distributed_sim.performance_stats();
    for (node_id, stats) in perf_stats.node_stats.iter() {
        println!("Node {}: {:.2}s execution time, {:.1}GB memory", 
                 node_id, stats.execution_time, stats.memory_usage_gb);
    }
    
    Ok(())
}
```

### 2. Tensor Network Simulation

```rust
use quantrs2_sim::v1::distributed::*;

fn tensor_network_example() -> Result<()> {
    // Configure tensor network simulation
    let tn_config = TensorNetworkConfig {
        max_bond_dimension: 128,
        contraction_strategy: ContractionStrategy::OptimalOrder,
        compression_threshold: 1e-12,
    };
    
    let mut tn_simulator = TensorNetworkSimulator::new(tn_config);
    
    // Create a quantum circuit with limited entanglement
    let circuit = create_qaoa_circuit(30, 10); // 30 qubits, 10 layers
    
    // Optimize contraction order
    let contraction_order = tn_simulator.optimize_contraction_order(&circuit)?;
    
    println!("Estimated contraction cost: {:.2e}", contraction_order.computational_cost);
    println!("Memory requirement: {:.1}GB", contraction_order.memory_requirement_gb);
    
    // Execute tensor network simulation
    let result = tn_simulator.simulate(&circuit)?;
    
    // Analyze tensor network structure
    let bond_dimensions = tn_simulator.bond_dimension_distribution();
    println!("Average bond dimension: {:.1}", bond_dimensions.mean);
    println!("Max bond dimension: {}", bond_dimensions.max);
    
    Ok(())
}
```

### 3. Automatic Parallelization

```rust
use quantrs2_sim::v1::distributed::*;

fn automatic_parallelization_example() -> Result<()> {
    // Configure automatic parallelization
    let config = AutoParallelConfig {
        parallelization_strategy: ParallelizationStrategy::Adaptive,
        resource_constraints: ResourceConstraints {
            max_threads: num_cpus::get(),
            memory_limit_gb: 16,
            target_efficiency: 0.85,
        },
        optimization_level: OptimizationLevel::Aggressive,
    };
    
    let parallel_engine = AutoParallelEngine::new(config);
    
    // Analyze circuit for parallelization opportunities
    let circuit = create_sample_circuit(25);
    let analysis = parallel_engine.analyze_circuit(&circuit)?;
    
    println!("Parallelization analysis:");
    println!("  Independent gate groups: {}", analysis.independent_groups);
    println!("  Parallel efficiency: {:.1}%", analysis.parallel_efficiency * 100.0);
    println!("  Recommended threads: {}", analysis.recommended_threads);
    
    // Execute with automatic parallelization
    let result = parallel_engine.execute_parallel(&circuit)?;
    
    // Review performance
    let perf_stats = result.performance_stats;
    println!("Execution time: {:.2}s", perf_stats.total_time);
    println!("Parallel speedup: {:.2}x", perf_stats.speedup);
    println!("CPU utilization: {:.1}%", perf_stats.cpu_utilization * 100.0);
    
    Ok(())
}
```

## Algorithm Development and Research

The `algorithms` module provides advanced quantum algorithms and research tools.

### 1. Variational Quantum Algorithms

```rust
use quantrs2_sim::v1::algorithms::*;

fn advanced_vqa_example() -> Result<()> {
    // Configure advanced VQA trainer
    let vqa_config = VQAConfig {
        ansatz: VariationalAnsatz::HardwareEfficient {
            layers: 10,
            entanglement_pattern: EntanglementPattern::Linear,
        },
        cost_function: CostFunction::Ising {
            coupling_strength: 1.0,
            field_strength: 0.5,
        },
        optimization_problem: OptimizationProblemType::MaxCut,
    };
    
    let mut vqa_trainer = AdvancedVQATrainer::new(vqa_config);
    
    // Set up Bayesian optimization
    vqa_trainer.set_optimizer(AdvancedOptimizerType::BayesianOptimization {
        acquisition_function: AcquisitionFunction::ExpectedImprovement,
        n_initial_points: 20,
    });
    
    // Define the problem Hamiltonian
    let problem_hamiltonian = ProblemHamiltonian::from_graph(&load_max_cut_graph("graph.txt")?);
    
    // Train the VQA
    let training_result = vqa_trainer.train(problem_hamiltonian, 1000)?;
    
    println!("VQA training completed:");
    println!("  Final cost: {:.6}", training_result.final_cost);
    println!("  Iterations: {}", training_result.iterations);
    println!("  Approximation ratio: {:.3}", training_result.approximation_ratio);
    
    // Analyze training statistics
    let stats = training_result.training_stats;
    println!("  Convergence time: {:.2}s", stats.convergence_time);
    println!("  Function evaluations: {}", stats.function_evaluations);
    
    Ok(())
}
```

### 2. Quantum Machine Learning with AutoDiff

```rust
use quantrs2_sim::v1::algorithms::*;

fn quantum_autodiff_example() -> Result<()> {
    // Create quantum neural network with automatic differentiation
    let qnn_config = QMLConfig {
        input_qubits: 4,
        hidden_layers: vec![8, 4],
        output_qubits: 2,
        activation: QuantumActivation::ParameterizedRotation,
    };
    
    let mut quantum_network = QuantumNeuralNetwork::new(qnn_config);
    
    // Configure training with gradient-based optimization
    let training_config = TrainingConfig {
        optimizer: OptimizerType::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
        },
        loss_function: LossFunction::CrossEntropy,
        batch_size: 32,
        epochs: 100,
    };
    
    // Load training data
    let training_data = load_quantum_dataset("quantum_data.csv")?;
    
    // Train with automatic differentiation
    let training_result = quantum_network.train_with_autodiff(
        &training_data, 
        &training_config
    )?;
    
    // Evaluate model performance
    let test_data = load_quantum_dataset("test_data.csv")?;
    let accuracy = quantum_network.evaluate(&test_data)?;
    
    println!("Quantum neural network training:");
    println!("  Training accuracy: {:.3}", training_result.final_accuracy);
    println!("  Test accuracy: {:.3}", accuracy);
    println!("  Training time: {:.2}s", training_result.training_time);
    
    Ok(())
}
```

### 3. Quantum Advantage Demonstration

```rust
use quantrs2_sim::v1::algorithms::*;

fn quantum_advantage_demo() -> Result<()> {
    // Configure quantum advantage demonstration
    let qa_config = QuantumAdvantageConfig {
        problem_domain: ProblemDomain::BosonSampling,
        quantum_hardware: QuantumHardwareSpecs {
            qubits: 50,
            gate_fidelity: 0.999,
            measurement_fidelity: 0.99,
        },
        classical_hardware: ClassicalHardwareSpecs {
            cpu_cores: 64,
            memory_gb: 512,
            gpu_count: 8,
        },
    };
    
    let demonstrator = QuantumAdvantageDemonstrator::new(qa_config);
    
    // Generate quantum advantage problem instance
    let problem = demonstrator.generate_hard_instance()?;
    
    // Execute quantum algorithm
    let quantum_result = demonstrator.run_quantum_algorithm(&problem)?;
    
    // Benchmark classical algorithms
    let classical_result = demonstrator.benchmark_classical_algorithms(&problem)?;
    
    // Analyze advantage
    let advantage_result = demonstrator.analyze_advantage(&quantum_result, &classical_result)?;
    
    println!("Quantum Advantage Analysis:");
    println!("  Quantum time: {:.2}s", quantum_result.execution_time);
    println!("  Classical time: {:.2}s", classical_result.best_time);
    println!("  Speedup: {:.2}x", advantage_result.speedup);
    println!("  Advantage type: {:?}", advantage_result.advantage_type);
    
    if advantage_result.quantum_advantage_demonstrated {
        println!("✓ Quantum advantage demonstrated!");
    } else {
        println!("✗ No quantum advantage found");
    }
    
    Ok(())
}
```

## Noise Modeling and Error Correction

The `noise_modeling` module provides comprehensive tools for realistic quantum simulation.

### 1. Advanced Noise Models

```rust
use quantrs2_sim::v1::noise_modeling::*;

fn advanced_noise_modeling() -> Result<()> {
    // Create realistic device noise model
    let noise_builder = RealisticNoiseModelBuilder::new()
        .device_type(DeviceType::Superconducting)
        .add_depolarizing_noise(0.001)
        .add_thermal_relaxation(50e-6, 70e-6, 20e-9) // T1, T2, gate_time
        .add_readout_error(0.02, 0.03)
        .add_crosstalk_error(0.005)
        .add_coherence_drift(0.1); // 10% daily drift
    
    let device_noise = noise_builder.build()?;
    
    // Create correlated noise model for multi-qubit gates
    let correlated_noise = CorrelatedNoiseModel::new()
        .add_spatial_correlation(0.8, 2.0) // correlation strength, range
        .add_temporal_correlation(0.9, 100e-9); // memory, correlation time
    
    // Combine noise models
    let composite_noise = CompositeNoiseModel::new()
        .add_noise_model(Box::new(device_noise))
        .add_noise_model(Box::new(correlated_noise));
    
    // Create noise-aware simulator
    let mut simulator = StateVectorSimulator::new_with_noise(composite_noise);
    
    // Simulate circuit with realistic noise
    let circuit = create_bell_state_circuit();
    
    let mut results = Vec::new();
    for _ in 0..1000 {
        simulator.reset();
        let result = simulator.run_circuit(&circuit)?;
        results.push(result);
    }
    
    // Analyze noise effects
    let noise_analysis = analyze_noise_effects(&results)?;
    println!("Noise analysis:");
    println!("  Fidelity loss: {:.3}", noise_analysis.fidelity_loss);
    println!("  Decoherence rate: {:.6} /μs", noise_analysis.decoherence_rate);
    
    Ok(())
}
```

### 2. Quantum Error Correction

```rust
use quantrs2_sim::v1::noise_modeling::*;

fn quantum_error_correction_example() -> Result<()> {
    // Configure surface code
    let surface_code_config = SurfaceCodeConfig {
        distance: 5,
        boundary_conditions: BoundaryConditions::Open,
        decoder: DecoderType::MinimumWeightPerfectMatching,
    };
    
    let surface_code = SurfaceCode::new(surface_code_config)?;
    
    // Setup logical qubit encoding
    let logical_qubit = surface_code.encode_logical_zero()?;
    
    // Apply logical operations
    surface_code.apply_logical_x(&mut logical_qubit)?;
    surface_code.apply_logical_hadamard(&mut logical_qubit)?;
    
    // Introduce errors
    let error_model = NoiseModel::depolarizing(0.001);
    surface_code.apply_noise(&mut logical_qubit, &error_model)?;
    
    // Syndrome measurement
    let syndrome = surface_code.measure_syndrome(&logical_qubit)?;
    
    // Error correction
    let correction = surface_code.decode_syndrome(&syndrome)?;
    surface_code.apply_correction(&mut logical_qubit, &correction)?;
    
    // Verify logical state
    let logical_fidelity = surface_code.logical_fidelity(&logical_qubit)?;
    println!("Logical qubit fidelity after correction: {:.6}", logical_fidelity);
    
    // Analyze threshold
    let threshold_analysis = surface_code.analyze_threshold(0.0001, 0.01, 20)?;
    println!("Error correction threshold: {:.4}", threshold_analysis.threshold);
    
    Ok(())
}
```

### 3. Error Mitigation Techniques

```rust
use quantrs2_sim::v1::noise_modeling::*;

fn error_mitigation_example() -> Result<()> {
    // Setup zero-noise extrapolation
    let zne_config = ZNEConfig {
        noise_scaling_factors: vec![1.0, 2.0, 3.0, 4.0],
        extrapolation_method: ExtrapolationMethod::Richardson,
        circuit_folding: CircuitFoldingMethod::Global,
    };
    
    let zne_mitigator = ZeroNoiseExtrapolator::new(zne_config);
    
    // Setup virtual distillation
    let vd_config = VirtualDistillationConfig {
        number_of_copies: 4,
        symmetry_verification: SymmetryVerificationType::Parity,
        post_selection_threshold: 0.5,
    };
    
    let vd_mitigator = VirtualDistillationMitigator::new(vd_config);
    
    // Create noisy circuit
    let noisy_circuit = create_vqe_circuit_with_noise(0.01);
    
    // Apply ZNE mitigation
    let zne_result = zne_mitigator.mitigate(&noisy_circuit)?;
    
    // Apply virtual distillation
    let vd_result = vd_mitigator.mitigate(&noisy_circuit)?;
    
    // Compare mitigation effectiveness
    let baseline_result = run_noisy_circuit(&noisy_circuit)?;
    let ideal_result = run_ideal_circuit(&noisy_circuit.remove_noise())?;
    
    println!("Error mitigation comparison:");
    println!("  Baseline (noisy): {:.6}", baseline_result.expectation_value);
    println!("  ZNE mitigated: {:.6}", zne_result.mitigated_value);
    println!("  VD mitigated: {:.6}", vd_result.mitigated_value);
    println!("  Ideal (noiseless): {:.6}", ideal_result.expectation_value);
    
    println!("  ZNE improvement: {:.1}%", 
        100.0 * (zne_result.mitigated_value - baseline_result.expectation_value) / 
        (ideal_result.expectation_value - baseline_result.expectation_value));
    
    Ok(())
}
```

## Developer Tools and Debugging

The `dev_tools` module provides comprehensive debugging and development utilities.

### 1. Quantum Circuit Debugging

```rust
use quantrs2_sim::v1::dev_tools::*;

fn quantum_debugging_example() -> Result<()> {
    // Configure quantum debugger
    let debug_config = DebugConfig {
        enable_breakpoints: true,
        enable_state_inspection: true,
        enable_gate_profiling: true,
        output_format: DebugOutputFormat::Interactive,
    };
    
    let mut debugger = QuantumDebugger::new(debug_config);
    
    // Create circuit with debugging annotations
    let mut circuit = QuantumCircuit::new(3);
    circuit.add_gate_with_label(HGate::new(), 0, "Create superposition")?;
    circuit.add_gate_with_label(CXGate::new(), (0, 1), "Entangle qubits 0-1")?;
    circuit.add_gate_with_label(CXGate::new(), (1, 2), "Entangle qubits 1-2")?;
    
    // Set breakpoints
    debugger.set_breakpoint_after_gate(1, BreakCondition::Always)?;
    debugger.set_breakpoint_when_entanglement_exceeds(0.5)?;
    
    // Add watchpoints
    debugger.watch_qubit_state(0, WatchFrequency::EveryGate)?;
    debugger.watch_entanglement_entropy(vec![0, 1], WatchFrequency::OnChange)?;
    
    // Run circuit with debugging
    let debug_result = debugger.run_circuit_debug(&circuit)?;
    
    // Analyze debug information
    for step in debug_result.execution_steps {
        println!("Step {}: {} | Entanglement: {:.3}", 
                 step.step_number, step.gate_description, step.entanglement_measure);
        
        if step.breakpoint_hit {
            println!("  → Breakpoint: {}", step.breakpoint_reason);
            println!("  → State: {:?}", step.quantum_state.amplitudes());
        }
    }
    
    // Performance analysis
    let perf_report = debug_result.performance_report;
    println!("Performance analysis:");
    for (gate_type, metrics) in perf_report.gate_performance.iter() {
        println!("  {}: avg {:.2}μs, max {:.2}μs", 
                 gate_type, metrics.average_time_us, metrics.max_time_us);
    }
    
    Ok(())
}
```

### 2. Circuit Profiling and Optimization

```rust
use quantrs2_sim::v1::dev_tools::*;

fn circuit_profiling_example() -> Result<()> {
    // Configure telemetry collection
    let telemetry_config = TelemetryConfig {
        collect_gate_timing: true,
        collect_memory_usage: true,
        collect_error_rates: true,
        export_format: TelemetryExportFormat::JSON,
        sampling_rate: 1.0, // Collect all data
    };
    
    let telemetry_collector = TelemetryCollector::new(telemetry_config);
    
    // Profile circuit execution
    let circuit = create_sample_circuit(15);
    
    let profiling_result = telemetry_collector.profile_circuit_execution(&circuit)?;
    
    // Analyze bottlenecks
    let bottlenecks = profiling_result.identify_bottlenecks()?;
    for bottleneck in bottlenecks {
        println!("Bottleneck found:");
        println!("  Location: Gate {} ({})", bottleneck.gate_index, bottleneck.gate_type);
        println!("  Impact: {:.1}% of total time", bottleneck.time_percentage);
        println!("  Suggestion: {}", bottleneck.optimization_suggestion);
    }
    
    // Memory analysis
    let memory_analysis = profiling_result.memory_analysis;
    println!("Memory usage:");
    println!("  Peak: {:.1}MB", memory_analysis.peak_memory_mb);
    println!("  Average: {:.1}MB", memory_analysis.average_memory_mb);
    println!("  Memory efficiency: {:.1}%", memory_analysis.efficiency * 100.0);
    
    // Generate optimization recommendations
    let optimizer_suggestions = profiling_result.generate_optimization_suggestions()?;
    for suggestion in optimizer_suggestions {
        println!("Optimization suggestion:");
        println!("  Type: {:?}", suggestion.optimization_type);
        println!("  Expected speedup: {:.2}x", suggestion.expected_speedup);
        println!("  Implementation: {}", suggestion.implementation_guide);
    }
    
    Ok(())
}
```

### 3. SciRS2 Enhanced Development Tools

```rust
use quantrs2_sim::v1::dev_tools::*;

fn scirs2_enhanced_tools() -> Result<()> {
    // Use SciRS2 quantum linter
    let linter_config = SciRS2LinterConfig {
        check_gate_optimizations: true,
        check_numerical_stability: true,
        check_memory_efficiency: true,
        strictness_level: StrictnessLevel::High,
    };
    
    let linter = SciRS2QuantumLinter::new(linter_config);
    
    // Lint quantum circuit
    let circuit = create_sample_circuit(10);
    let lint_report = linter.lint_circuit(&circuit)?;
    
    println!("SciRS2 Linting Report:");
    for finding in lint_report.findings {
        println!("  {}: {} at gate {}", 
                 finding.severity, finding.message, finding.location);
        
        if let Some(suggestion) = finding.optimization_suggestion {
            println!("    → Suggestion: {}", suggestion);
        }
    }
    
    // Use SciRS2 quantum formatter
    let formatter_config = SciRS2FormatterConfig {
        style: FormattingStyle::Compact,
        include_performance_annotations: true,
        highlight_inefficiencies: true,
    };
    
    let formatter = SciRS2QuantumFormatter::new(formatter_config);
    let formatted_circuit = formatter.format_circuit(&circuit)?;
    
    println!("Formatted circuit:\n{}", formatted_circuit.formatted_code);
    
    // Performance annotations
    for annotation in formatted_circuit.performance_annotations {
        println!("Performance note: {} at line {}", 
                 annotation.message, annotation.line_number);
    }
    
    Ok(())
}
```

## Hardware Programming

For hardware-level programming and device interaction.

### 1. Hardware Capabilities Detection

```rust
use quantrs2_core::v1::hardware::*;

fn hardware_capabilities_example() -> Result<()> {
    // Detect hardware capabilities
    let capabilities = HardwareCapabilities::detect();
    
    println!("Hardware capabilities detected:");
    println!("  CPU cores: {}", capabilities.cpu_cores);
    println!("  SIMD support: {:?}", capabilities.simd_features);
    println!("  GPU available: {}", capabilities.gpu_available);
    
    if capabilities.gpu_available {
        println!("  GPU memory: {:.1}GB", capabilities.gpu_memory_gb);
        println!("  GPU type: {:?}", capabilities.gpu_type);
    }
    
    // Configure optimal backend based on capabilities
    let optimal_config = capabilities.get_optimal_configuration()?;
    
    match optimal_config.recommended_backend {
        BackendType::GPU => {
            println!("Recommended: GPU acceleration");
            #[cfg(feature = "gpu")]
            {
                let gpu_simulator = GpuQuantumSimulator::new(optimal_config.gpu_config)?;
                // Use GPU simulator
            }
        },
        BackendType::CPU => {
            println!("Recommended: CPU with {} threads", optimal_config.cpu_threads);
            let cpu_simulator = CPUQuantumSimulator::new(optimal_config.cpu_config)?;
            // Use CPU simulator
        },
        _ => {}
    }
    
    Ok(())
}
```

### 2. Pulse-Level Control

```rust
use quantrs2_core::v1::hardware::*;

fn pulse_level_control() -> Result<()> {
    // Configure pulse control system
    let pulse_config = PulseControlConfig {
        sampling_rate: 2e9, // 2 GSa/s
        pulse_resolution: 1e-9, // 1 ns
        max_pulse_length: 1e-6, // 1 μs
    };
    
    let mut pulse_controller = PulseController::new(pulse_config);
    
    // Create custom pulse sequences
    let x_pulse = GaussianPulse {
        amplitude: 0.5,
        frequency: 5.2e9, // 5.2 GHz
        sigma: 20e-9, // 20 ns width
        phase: 0.0,
    };
    
    let readout_pulse = SquarePulse {
        amplitude: 0.1,
        frequency: 6.8e9, // 6.8 GHz
        duration: 1e-6, // 1 μs
        phase: 0.0,
    };
    
    // Compile pulse sequence
    let mut pulse_sequence = PulseSequence::new();
    pulse_sequence.add_pulse(0, x_pulse)?; // Apply to qubit 0
    pulse_sequence.add_delay(100e-9)?; // 100 ns delay
    pulse_sequence.add_pulse(0, readout_pulse)?;
    
    // Optimize pulse sequence
    let optimized_sequence = pulse_controller.optimize_sequence(&pulse_sequence)?;
    
    println!("Pulse optimization:");
    println!("  Original duration: {:.1}μs", pulse_sequence.total_duration() * 1e6);
    println!("  Optimized duration: {:.1}μs", optimized_sequence.total_duration() * 1e6);
    println!("  Fidelity: {:.4}", optimized_sequence.estimated_fidelity());
    
    // Execute on hardware
    let execution_result = pulse_controller.execute_sequence(&optimized_sequence)?;
    
    Ok(())
}
```

### 3. Real-time Calibration

```rust
use quantrs2_core::v1::hardware::*;

fn real_time_calibration() -> Result<()> {
    // Setup calibration system using SciRS2 system identification
    let calibration_config = CalibrationConfig {
        calibration_frequency: CalibrationFrequency::Hourly,
        parameters_to_calibrate: vec![
            CalibrationParameter::QubitFrequency,
            CalibrationParameter::GateAmplitude,
            CalibrationParameter::ReadoutFrequency,
        ],
        use_machine_learning: true,
    };
    
    let calibration_engine = CalibrationEngine::new(calibration_config);
    
    // Run automated calibration
    let calibration_result = calibration_engine.run_full_calibration()?;
    
    println!("Calibration completed:");
    for (qubit, params) in calibration_result.qubit_parameters.iter() {
        println!("  Qubit {}: frequency = {:.3} GHz, T1 = {:.1} μs, T2 = {:.1} μs", 
                 qubit, params.frequency / 1e9, params.t1 * 1e6, params.t2 * 1e6);
    }
    
    // Monitor drift over time
    let drift_monitor = DriftMonitor::new();
    let drift_analysis = drift_monitor.analyze_parameter_drift(24 * 3600)?; // 24 hours
    
    if drift_analysis.requires_recalibration {
        println!("Warning: Significant parameter drift detected");
        println!("  Drift rate: {:.3}% per hour", drift_analysis.drift_rate_percent_per_hour);
        println!("  Recommendation: Increase calibration frequency");
    }
    
    Ok(())
}
```

## Migration from Beta to 1.0

### Key Changes in 1.0 API

1. **Organized Module Structure**: Instead of flat imports from `prelude`, use specific modules:
   ```rust
   // Old (still works but deprecated)
   use quantrs2_sim::prelude::*;
   
   // New (recommended)
   use quantrs2_sim::v1::essentials::*;
   ```

2. **Clearer Intent**: Module names indicate their purpose:
   ```rust
   use quantrs2_sim::v1::essentials::*;    // Basic simulation
   use quantrs2_sim::v1::algorithms::*;    // Algorithm development
   use quantrs2_sim::v1::gpu::*;          // GPU acceleration
   use quantrs2_sim::v1::distributed::*;   // Large-scale simulation
   ```

3. **No Breaking Changes**: All existing code continues to work with deprecation warnings.

### Migration Strategy

1. **Gradual Migration**: Start by replacing `prelude::*` imports with specific module imports
2. **Feature-by-Feature**: Migrate different parts of your codebase to different v1 modules as appropriate
3. **Testing**: Use the same test suite to verify compatibility during migration

### Example Migration

```rust
// Before (0.1.0-beta.1)
use quantrs2_core::prelude::*;
use quantrs2_sim::prelude::*;

fn old_style() -> Result<()> {
    let simulator = StateVectorSimulator::new();
    // ... rest of code unchanged
    Ok(())
}

// After (1.0)
use quantrs2_core::v1::essentials::*;
use quantrs2_sim::v1::essentials::*;

fn new_style() -> Result<()> {
    let simulator = StateVectorSimulator::new();
    // ... rest of code unchanged
    Ok(())
}
```

## Conclusion

This tutorial covers all major features of the QuantRS2 1.0 API. The new organized structure makes it easy to discover relevant functionality while maintaining backward compatibility. Choose the appropriate modules based on your use case:

- **Essentials**: Basic quantum programming
- **Simulation**: Advanced simulation techniques
- **GPU**: High-performance computing
- **Distributed**: Large-scale simulation
- **Algorithms**: Algorithm development and research
- **Noise Modeling**: Realistic quantum simulation
- **Dev Tools**: Debugging and development
- **Hardware**: Hardware-level programming

For more specific examples and advanced use cases, see the individual module documentation and the comprehensive example collection in the `examples/` directory.