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

## 1.0 Release Goals

- [ ] Finalize public API for 1.0 release
- [ ] Complete documentation for all public interfaces
- [ ] Add comprehensive examples for all major features
- [ ] Add contribution guidelines

## Infrastructure

- [ ] Implement automated testing in CI pipeline
- [ ] Publish Python package to PyPI

## SymEngine Integration

- [ ] Complete SymEngine compatibility updates for enhanced D-Wave support
- [ ] Complete the patching of the `symengine` crate
- [ ] Fix type and function reference issues in `symengine` crate
- [ ] Test compatibility with the D-Wave system

## Module-Specific Roadmaps

- [quantrs2-tytan](tytan/TODO.md)
- [quantrs2-py](py/TODO.md)
