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
- [x] Fork symengine-sys to cool-japan/symengine-sys and apply patches for macOS compatibility
- [x] Fork symengine to cool-japan/symengine-rs for future integration with patched symengine-sys
- [ ] Complete symengine-rs compatibility updates to enable dwave feature

## SymEngine Integration Notes

### Current Status

- Successfully patched `symengine-sys` for macOS compatibility
- Forked repositories are available at:
  - https://github.com/cool-japan/symengine-sys (branch: fixed-macos)
  - https://github.com/cool-japan/symengine-rs (branch: fixed-macos)
- The `parallel` feature works correctly with the patched symengine-sys
- The `dwave` feature is currently disabled due to incompatibilities between `symengine` and our patched `symengine-sys`

### Build Requirements

When building with symengine dependencies on macOS, set these environment variables:

```bash
export SYMENGINE_DIR=$(brew --prefix symengine)
export GMP_DIR=$(brew --prefix gmp)
export MPFR_DIR=$(brew --prefix mpfr)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix symengine)/include -I$(brew --prefix gmp)/include -I$(brew --prefix mpfr)/include"
```

### SymEngine TODO Tasks

1. **Enable the `dwave` Feature**
   - Complete the patching of the `symengine` crate to work with our patched `symengine-sys`
   - Fix type and function reference issues in `symengine` crate
   - Test compatibility with the D-Wave system

2. **Update Dependencies**
   - Regularly update the forked repositories with upstream changes
   - Test compatibility after each update
   - Consider submitting PRs to upstream repositories once fixes are stable

3. **Documentation**
   - Document the build process with symengine dependencies
   - Add a section to README.md explaining the symengine integration

### Notes for Engineers

- When adding features that require symengine, ensure they gracefully degrade when the `dwave` feature is disabled
- Use feature flags to conditionally compile code that depends on symengine
- If you encounter build issues on macOS related to symengine:
  1. Ensure you've set all the environment variables listed above
  2. Check that you're using the patched versions from cool-japan repositories
  3. Update the patched repositories if necessary

### Current Workarounds

- The `dwave` feature is temporarily disabled in tytan/Cargo.toml
- For development requiring D-Wave integration on macOS, consider using Docker with Linux