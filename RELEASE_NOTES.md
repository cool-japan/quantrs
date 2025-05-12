# Release Notes for Quantrs v0.1.0-alpha.1

We're excited to announce the first alpha release of Quantrs, a high-performance quantum computing framework written in Rust!

## Highlights

- **Type-Safe Quantum Circuits**: Build quantum circuits with compile-time safety using Rust's const generics
- **High-Performance Simulation**: Efficiently simulate 30+ qubits with optimized state vector and tensor network backends
- **GPU Acceleration**: Run simulations on GPU using WGPU for significant speedups on large qubit counts
- **Quantum Hardware Integration**: Connect to IBM Quantum and D-Wave quantum hardware
- **Advanced Noise Models**: Simulate realistic quantum hardware with configurable noise channels
- **Quantum Error Correction**: Protect quantum information with various error correction codes
- **Tensor Network Optimization**: Specialized contraction path optimization for common circuit structures

## System Requirements

- Rust 1.86.0 or newer
- Optional: WGPU-compatible GPU for GPU acceleration
- Optional: IBM Quantum account for hardware connectivity
- Optional: D-Wave account for quantum annealing hardware

## Installation

Add Quantrs to your Cargo project:

```toml
[dependencies]
quantrs-core = "0.1.0-alpha.1"
quantrs-circuit = "0.1.0-alpha.1"
quantrs-sim = "0.1.0-alpha.1"
```

For optional features:

```toml
# GPU acceleration
quantrs-sim = { version = "0.1.0-alpha.1", features = ["gpu"] }

# IBM Quantum integration
quantrs-device = { version = "0.1.0-alpha.1", features = ["ibm"] }

# D-Wave quantum annealing
quantrs-anneal = { version = "0.1.0-alpha.1", features = ["dwave"] }
```

## Known Issues

- The D-Wave integration requires additional setup for SymEngine on macOS. See the README.md for details.
- GPU acceleration is currently in beta and may have performance variability across different GPU models.

## Acknowledgments

We would like to thank all contributors to the Quantrs project, as well as the broader quantum computing and Rust communities for their support and feedback.

## Future Plans

This is an alpha release, and we're actively working on:

- Automated testing in CI pipeline
- Improved documentation
- Additional quantum hardware integrations
- Enhanced tensor network contraction algorithms
- Python package distribution via PyPI

For more details, see our [Roadmap](https://github.com/quantrs/quantrs/blob/master/TODO.md).