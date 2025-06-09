# QuantRS2-Circuit Development Roadmap

This document outlines the development plans and future tasks for the QuantRS2-Circuit module.

## Current Status

### Completed Features

- ✅ Fluent builder API for quantum circuits
- ✅ Type-safe circuit operations with const generics
- ✅ Support for all standard quantum gates
- ✅ Basic macros for circuit construction
- ✅ Integration with simulator backends
- ✅ Circuit depth and gate count analysis
- ✅ Support for multi-qubit gates
- ✅ Circuit validation and error checking
- ✅ Circuit optimization passes using gate properties
- ✅ Modular optimization framework with multiple passes
- ✅ Hardware-aware cost models and optimization
- ✅ Circuit analysis and metrics calculation

### In Progress

- 🔄 SciRS2-powered circuit optimization
- 🔄 Graph-based circuit representation
- 🔄 Quantum circuit synthesis algorithms

## Planned Enhancements

### Near-term (v0.1.x)

- [x] Implement circuit DAG representation using SciRS2 graphs ✅
- [x] Add commutation analysis for gate reordering ✅
- [x] Create QASM 2.0/3.0 import/export functionality ✅
- [x] Implement circuit slicing for parallel execution ✅
- [x] Add topological sorting for dependency analysis ✅
- [x] Create circuit equivalence checking algorithms ✅
- [x] Implement peephole optimization passes ✅
- [x] Add support for classical control flow ✅
- [x] Implement template matching using SciRS2 pattern recognition ✅
- [x] Add routing algorithms (SABRE, lookahead) with SciRS2 graphs ✅
- [x] Create noise-aware circuit optimization ✅
- [x] Implement unitary synthesis from circuit description ✅
- [x] Add support for mid-circuit measurements and feed-forward ✅
- [x] Create circuit compression using tensor networks ✅
- [x] Implement cross-talk aware scheduling ✅
- [x] Add support for pulse-level control ✅

### Long-term (Future Versions)

- [x] Implement ZX-calculus optimization using SciRS2 graph algorithms ✅
- [x] Add support for photonic quantum circuits ✅
- [x] Create ML-based circuit optimization with SciRS2 ML integration ✅
- [x] Implement fault-tolerant circuit compilation ✅
- [x] Add support for topological quantum circuits ✅
- [ ] Create distributed circuit execution framework
- [ ] Implement quantum-classical co-optimization
- [ ] Add support for variational quantum eigensolver circuits

## Implementation Notes

### Architecture Decisions
- Use SciRS2 directed graphs for circuit DAG representation
- Implement lazy evaluation for circuit transformations
- Store gates as indices into a gate library for efficiency
- Use bit-packed representations for qubit connectivity
- Implement copy-on-write for circuit modifications

### Performance Considerations
- Cache commutation relations between gates
- Use SIMD for parallel gate property calculations
- Implement incremental circuit analysis
- Use memory pools for gate allocation
- Optimize for common circuit patterns

## Known Issues

- The current DSL macros are incomplete and marked as `ignore` in doctests
- Circuit cloning has inefficiencies due to trait object limitations
- Need better error messages for invalid circuit operations
- Large circuits may have memory fragmentation issues

## Integration Tasks

### SciRS2 Integration
- [x] Use SciRS2 graph algorithms for circuit analysis ✅
- [ ] Leverage SciRS2 sparse matrices for gate representations
- [ ] Integrate SciRS2 optimization for parameter tuning
- [ ] Use SciRS2 statistical tools for circuit benchmarking
- [ ] Implement circuit similarity metrics using SciRS2

### Module Integration
- [x] Create efficient circuit-to-simulator interfaces ✅
- [ ] Implement device-specific transpiler passes
- [ ] Add hardware noise model integration
- [ ] Create circuit validation for each backend
- [ ] Implement circuit caching for repeated execution