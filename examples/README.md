# quantrs2-examples

Example programs for the QuantRS2 quantum computing framework.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

## Overview

`quantrs2-examples` contains runnable example programs demonstrating the capabilities of the QuantRS2 quantum computing framework. These examples cover quantum circuits, simulators, algorithms, error correction, machine learning, annealing, and hardware integration.

This crate is not published to crates.io; it is intended for local development and learning.

## Examples

| Example | Description |
|---------|-------------|
| `bell_state` | Create and measure a Bell state |
| `circuit_optimizer_demo` | Circuit optimization passes |
| `enhanced_qgan_demo` | Quantum GAN demonstration |
| `error_correction` | Quantum error correction basics |
| `error_correction_comparison` | Compare error correction codes |
| `extended_gates` | Extended gate set usage |
| `extended_gates_with_noise` | Gates under noise models |
| `five_qubit_code` | Five-qubit error correcting code |
| `gate_decomposition` | Gate decomposition techniques |
| `gpu_linalg_demo` | GPU-accelerated linear algebra |
| `graph_coloring` | Graph coloring via quantum annealing |
| `graph_optimizer_demo` | Graph-based circuit optimization |
| `grovers_algorithm` | Grover's search algorithm |
| `grovers_algorithm_noisy` | Grover's algorithm with noise |
| `hardware_topology_demo` | Hardware topology mapping |
| `hhl_demo` | HHL linear systems algorithm |
| `ibm_quantum_example` | IBM Quantum backend usage |
| `max_cut` | Max-Cut optimization problem |
| `noisy_simulator` | Noisy quantum simulation |
| `optimized_sim_demo` | Optimized simulator demonstration |
| `optimized_sim_small` | Small optimized simulation |
| `parametric_gates_demo` | Parametric gate circuits |
| `phase_error_correction` | Phase error correction |
| `qaoa_demo` | QAOA optimization algorithm |
| `qcnn_demo` | Quantum convolutional neural network |
| `qsvm_demo` | Quantum support vector machine |
| `quantum_counting_demo` | Quantum counting algorithm |
| `quantum_fourier_transform` | Quantum Fourier transform |
| `quantum_pca_demo` | Quantum PCA demonstration |
| `quantum_phase_estimation` | Quantum phase estimation |
| `quantum_teleportation` | Quantum teleportation protocol |
| `quantum_testing_demo` | Quantum testing utilities |
| `quantum_walk_demo` | Quantum walk simulation |
| `qubit_routing_demo` | Qubit routing and mapping |
| `qvae_demo` | Quantum variational autoencoder |
| `realistic_noise` | Realistic noise model simulation |
| `scirs2_integration_demo` | SciRS2 integration patterns |
| `shors_algorithm_simplified` | Simplified Shor's algorithm |
| `simd_optimization_demo` | SIMD optimization techniques |
| `sparse_clifford_demo` | Sparse Clifford simulation |
| `stabilizer_demo` | Stabilizer simulator usage |
| `tensor_network_optimization` | Tensor network optimization |
| `tensor_network_sim` | Tensor network simulator |
| `traveling_salesman` | TSP via quantum annealing |
| `tytan_3rooks` | 3-rooks problem with Tytan |
| `tytan_maxcut_scirs` | Max-Cut with Tytan and SciRS2 |
| `ultrathink_core_demo` | Core ultrathink demonstration |
| `ultrathink_demo` | Ultrathink deep analysis demo |
| `ultrathink_simple_demo` | Simple ultrathink example |

## Running Examples

```bash
# Run a specific example
cargo run -p quantrs2-examples --bin bell_state

# Run with optional features
cargo run -p quantrs2-examples --bin tytan_3rooks --features tytan
```

## Part of QuantRS2

This crate is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework (v0.1.3).

## License

Licensed under the Apache License, Version 2.0.

## Author

COOLJAPAN OU (Team Kitasan)
