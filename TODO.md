# QuantRS2 Roadmap

## Current Version: 0.1.3

## Completed in v0.1.3 (2026-03-27)

- [x] ZYZ decomposition fix (`core/src/synthesis.rs`)
- [x] Holonomic gate synthesis fix (`core/src/holonomic.rs`)
- [x] KAK/Cartan decomposition real eigensolver (`core/src/cartan.rs`)
- [x] Adiabatic eigenvalue solver (inverse power iteration + deflation) (`core/src/adiabatic.rs`)
- [x] QPE bug fix — controlled-U power application (`core/src/quantum_counting.rs`)
- [x] No-unwrap policy compliance — ~210 `unwrap()` calls eliminated across production code
- [x] QML encoding: Mottonen amplitude encoding + IQP ZZ interaction (`core/src/qml/encoding.rs`)
- [x] QML layer gradients: parameter-shift rule (`core/src/qml/layers.rs`)
- [x] QML NLP `parameters()` / `parameters_mut()` accessors (`core/src/qml/nlp.rs`)
- [x] License migration: MIT OR Apache-2.0 → Apache-2.0 only (COOLJAPAN Policy 2026+)
- [x] SciRS2 upgraded from 0.3.4 → 0.4.0
- [x] PyO3 0.28.2 compatibility fixes (`PyObject`→`Py<PyAny>`, `with_gil`→`attach_unchecked`)
- [x] Circuit ML optimization (Q-learning, genetic algorithm, neural network optimizers)
- [x] Tensor network enhancements (SVD compress, MPS `from_circuit`, MPS compress)
- [x] VQE enhancement (`set_parameters` fix, `ParameterizedGateRecord`)
- [x] Quantum supremacy simulation
- [x] SABRE routing, noise-aware optimization
- [x] ZX-calculus optimization (Clifford spider rewrite rules)
- [x] Quantum walk eigenvalue solver
- [x] Python mitigation bindings (PEC, virtual distillation, symmetry verification)
- [x] Dependencies upgraded (wgpu 29.0.1, uuid 1.23.0)
- [x] All subcrate README.md and Cargo.toml metadata completed

## 1.0 Release Goals

- [ ] Finalize public API for 1.0 release
- [ ] Complete documentation for all public interfaces
- [ ] Add comprehensive examples for all major features
- [ ] Add contribution guidelines

## Infrastructure

- [ ] Implement automated testing in CI pipeline
- [ ] Publish Python package to PyPI

## SymEngine Integration (quantrs2-symengine-pure — 5,739 LoC, 333 public APIs)

- [x] Complete SymEngine pure-Rust implementation (parser, eval, diff, simplify, quantum, matrix)
- [x] Complete the patching of the `symengine` crate → replaced with `symengine-pure`
- [x] Fix type and function reference issues → pure-Rust rewrite, no C/Fortran deps
- [x] Test compatibility with the D-Wave system

## Module-Specific Roadmaps

- [quantrs2-tytan](tytan/TODO.md)
- [quantrs2-py](py/TODO.md)
