# Quantrs Development Roadmap

This document outlines the planned development tasks for the Quantrs project, organized by milestones and priority.

## Milestone 1: Core + CPU Simulator (Current)

- [x] Create workspace structure
- [x] Implement basic types and traits in `quantrs-core`
- [x] Implement circuit builder in `quantrs-circuit` 
- [x] Implement state vector simulator in `quantrs-sim`
- [x] Optimize state vector simulation for 30+ qubits
- [x] Add comprehensive test suite for core functionality
- [x] Benchmark against existing simulators

## Milestone 2: Quantum Annealing

- [x] Implement Ising model representation
- [x] Add QUBO problem formulation
- [x] Implement simulated quantum annealing
- [x] Add D-Wave API client
  - [x] Authentication and connection management
  - [x] Problem submission
  - [x] Result retrieval and processing
- [x] Provide examples of optimization problems
  - [x] Maximum Cut (MaxCut) problem
  - [x] Graph Coloring problem
  - [x] Traveling Salesman Problem (TSP)

## Milestone 3: Remote IBM Quantum

- [ ] Implement IBM Quantum API client
  - [ ] Authentication and device selection
  - [ ] Circuit transpilation for IBM devices
  - [ ] Job submission
  - [ ] Result retrieval and processing
- [ ] Support for IBM Heron/Condor QPUs
- [ ] Add parallel job submission capabilities
- [ ] Optimize for high-throughput batch operations

## Milestone 4: GPU Acceleration & Python Bindings

- [ ] Implement GPU-accelerated state vector simulation using WGPU
- [x] Add SIMD optimizations for CPU simulator
- [ ] Create PyO3 bindings for Python integration
- [ ] Package as Python module with pip installation
- [ ] Document Python API

## Milestone 5: Documentation & Community

- [ ] Create comprehensive API documentation
- [x] Write tutorials and examples
- [ ] Develop user guide
- [ ] Set up CI/CD with GitHub Actions
- [ ] Create contribution guidelines
- [x] Add benchmarking suite
- [ ] Release v1.0.0-alpha.1
- [ ] Release `quantrs-core` v0.1.0-alpha.1
- [x] Prepare `quantrs-anneal` v0.1.0-alpha.1 for release
  - [x] Complete implementation of core annealing features
  - [x] Add examples of optimization problems
  - [x] Fix build issues with example dependencies
- [ ] Release `quantrs-py` v0.1.0-alpha.1
- [ ] Release `quantrs-device` v0.1.0-alpha.1

## Ongoing Tasks

- [x] Optimize performance for large qubit counts
- [ ] Expand gate set
- [ ] Add noise models and error correction
- [ ] Implement additional algorithms
- [ ] Support for Azure Quantum and AWS Braket
- [ ] Add tensor network simulator backend
- [ ] Fork symengine to cool-japan/symengine and maintain it there to resolve dependency conflicts in the tytan crate