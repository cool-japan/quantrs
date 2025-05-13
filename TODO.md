# QuantRS2 Release Plans and Roadmap

This document outlines the release plans and future roadmap for the QuantRS2 project.

## Module-Specific Roadmaps

For more detailed development plans for each module, see the individual TODO files:

- [quantrs2-core](core/TODO.md): Core types and abstractions
- [quantrs2-circuit](circuit/TODO.md): Circuit builder and DSL
- [quantrs2-sim](sim/TODO.md): Quantum simulators
- [quantrs2-anneal](anneal/TODO.md): Quantum annealing
- [quantrs2-device](device/TODO.md): Hardware connectivity
- [quantrs2-py](py/TODO.md): Python bindings
- [quantrs2-tytan](tytan/TODO.md): High-level quantum annealing

## v0.1.0-alpha.1 Release Status

The first alpha release of QuantRS2 is ready with all core functionality completed.

### Completed Features

- **Core Framework**
  - ✅ Type-safe quantum circuit implementation with const generics
  - ✅ Comprehensive gate set including extended gates (S/T-dagger, √X, etc.)
  - ✅ Circuit builder API with fluent interface

- **Simulation**
  - ✅ High-performance CPU state vector simulator supporting 30+ qubits
  - ✅ GPU-accelerated state vector simulation
  - ✅ SIMD-optimized operations
  - ✅ Tensor network simulator with specialized optimizations
  - ✅ Advanced noise models (bit flip, phase flip, depolarizing, etc.)
  - ✅ IBM-specific T1/T2 relaxation models

- **Hardware Integration**
  - ✅ IBM Quantum API client
  - ✅ D-Wave quantum annealing interface

- **Quantum Algorithms**
  - ✅ Grover's search algorithm
  - ✅ Quantum Fourier Transform
  - ✅ Quantum Phase Estimation
  - ✅ Shor's algorithm (simplified)

- **Error Correction**
  - ✅ Bit flip code
  - ✅ Phase flip code
  - ✅ Shor code
  - ✅ 5-qubit perfect code

- **Documentation**
  - ✅ API documentation
  - ✅ User guides
  - ✅ Algorithm-specific documentation
  - ✅ Interactive tutorials

- **Python Bindings**
  - ✅ Full Python API via PyO3
  - ✅ GPU acceleration support in Python
  - ✅ Python package structure

## Pre-Release Checklist

- [ ] Final testing of all crates
- [ ] Version number update in all Cargo.toml files
- [ ] Release v0.1.0-alpha.1 on crates.io
- [ ] Add CHANGELOG.md
- [ ] Create GitHub release

## Future Roadmap

### v0.1.0 (Stable Release)

- [ ] Add contribution guidelines
- [ ] Implement automated testing in CI pipeline
- [ ] Complete SymEngine compatibility updates for enhanced D-Wave support
- [ ] Publish Python package to PyPI

### v0.2.0 (Future Release)

- [ ] Add support for Azure Quantum and AWS Braket
- [ ] Add visualization tools for circuits and simulation results
- [ ] Implement more advanced error correction techniques (surface codes)
- [ ] Further optimize tensor network contraction algorithms
- [ ] Add support for fermionic simulation and quantum chemistry
- [ ] Implement quantum machine learning algorithms

## SymEngine Integration Notes

- Successfully patched `symengine-sys` for macOS compatibility
- The `dwave` feature is properly gated and optional
- All SymEngine-dependent functionality is behind `#[cfg(feature = "dwave")]` gates

### Build Requirements for SymEngine

When building with symengine dependencies on macOS, set these environment variables:

```bash
export SYMENGINE_DIR=$(brew --prefix symengine)
export GMP_DIR=$(brew --prefix gmp)
export MPFR_DIR=$(brew --prefix mpfr)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix symengine)/include -I$(brew --prefix gmp)/include -I$(brew --prefix mpfr)/include"
```

### Remaining SymEngine Tasks

- [ ] Complete the patching of the `symengine` crate to work with our patched `symengine-sys`
- [ ] Fix type and function reference issues in `symengine` crate
- [ ] Test compatibility with the D-Wave system

For more details on the D-Wave integration, refer to the documentation in the `quantrs-anneal` crate.