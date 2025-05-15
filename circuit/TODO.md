# QuantRS2-Circuit Development Roadmap

This document outlines the development plans and future tasks for the QuantRS2-Circuit module.

## Current Status

### Completed Features

- ✅ Fluent builder API for quantum circuits
- ✅ Type-safe circuit operations with const generics
- ✅ Support for all standard quantum gates
- ✅ Basic macros for circuit construction
- ✅ Integration with simulator backends

### In Progress

- 🔄 Enhanced DSL macros for circuit construction
- 🔄 Advanced circuit visualization capabilities
- 🔄 Circuit optimization algorithms

## Planned Enhancements

### Near-term (v0.1.x)

- [ ] Complete quantum DSL macro implementation
- [ ] Add circuit composition operations
- [ ] Implement circuit serialization/deserialization
- [ ] Support for conditional operations
- [ ] Add circuit inspection and analysis tools

### Medium-term (v0.2.x)

- [ ] Implement automatic circuit optimization
- [ ] Add circuit decomposition algorithms
- [ ] Support for parametric circuits (with runtime parameters)
- [ ] Add measurement and post-processing capabilities
- [ ] Implement circuit transpilation for hardware constraints

### Long-term (Future Versions)

- [ ] Advanced circuit visualization with interactive diagrams
- [ ] Hybrid classical-quantum circuit support
- [ ] Automated qubit routing algorithms
- [ ] Machine learning integration for circuit optimization
- [ ] Support for continuous-variable quantum computing

## Implementation Notes

- The `quantum!` macro needs to be rewritten as a proper procedural macro
- Consider using trait objects for gate storage to allow heterogeneous collections
- Need to balance flexibility and performance in the circuit representation

## Known Issues

- The current DSL macros are incomplete and marked as `ignore` in doctests
- Circuit cloning has inefficiencies due to trait object limitations

## Integration Tasks

- [ ] Enhance interoperability with the simulator module
- [ ] Implement device-specific circuit transformations
- [ ] Create adapters for different quantum hardware architectures