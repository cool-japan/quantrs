# Quantrs Development Roadmap

This document outlines the planned development tasks for the Quantrs project, organized by milestones and priority.

## Milestone 1: Core + CPU Simulator (Current)

- [x] Create workspace structure
- [x] Implement basic types and traits in `quantrs-core`
- [x] Implement circuit builder in `quantrs-circuit` 
- [x] Implement state vector simulator in `quantrs-sim`
- [ ] Optimize state vector simulation for 30+ qubits
- [ ] Add comprehensive test suite for core functionality
- [ ] Benchmark against existing simulators
- [ ] Release v0.1.0 to crates.io

## Milestone 2: Quantum Annealing

- [ ] Implement Ising model representation
- [ ] Add QUBO problem formulation
- [ ] Implement simulated quantum annealing
- [ ] Add D-Wave API client
  - [ ] Authentication and connection management
  - [ ] Problem submission
  - [ ] Result retrieval and processing
- [ ] Provide examples of optimization problems
- [ ] Release `quantrs-anneal` v0.1.0

## Milestone 3: Remote IBM Quantum

- [ ] Implement IBM Quantum API client
  - [ ] Authentication and device selection
  - [ ] Circuit transpilation for IBM devices
  - [ ] Job submission
  - [ ] Result retrieval and processing
- [ ] Support for IBM Heron/Condor QPUs
- [ ] Add parallel job submission capabilities
- [ ] Optimize for high-throughput batch operations
- [ ] Release `quantrs-device` v0.1.0

## Milestone 4: GPU Acceleration & Python Bindings

- [ ] Implement GPU-accelerated state vector simulation using WGPU
- [ ] Add SIMD optimizations for CPU simulator
- [ ] Create PyO3 bindings for Python integration
- [ ] Package as Python module with pip installation
- [ ] Document Python API
- [ ] Release `quantrs-py` v0.1.0

## Milestone 5: Documentation & Community

- [ ] Create comprehensive API documentation
- [ ] Write tutorials and examples
- [ ] Develop user guide
- [ ] Set up CI/CD with GitHub Actions
- [ ] Create contribution guidelines
- [ ] Add benchmarking suite
- [ ] Release v1.0.0

## Ongoing Tasks

- [ ] Optimize performance for large qubit counts
- [ ] Expand gate set
- [ ] Add noise models and error correction
- [ ] Implement additional algorithms
- [ ] Support for Azure Quantum and AWS Braket
- [ ] Add tensor network simulator backend