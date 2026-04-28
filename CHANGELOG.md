# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.2.0

### Added

- **Contribution Governance**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` — project contribution guidelines, community conduct policy, and security-disclosure workflow (responsible disclosure via `kitahata@gmail.com`, embargo/coordinated-disclosure policy, supported-version table).

- **Quantum Error Correction — Surface Codes**: Production-grade QEC stack in `core/src/error_correction/`:
  - `RotatedSurfaceCode` — `d × d` rotated planar surface code with data/X-ancilla/Z-ancilla qubit layout, stabilizer schedule, logical operator extraction; supports d=3, 5, 7.
  - `MwpmSurfaceDecoder` — bitmask-DP minimum-weight perfect matching (O(n²·2^n), optimal for ≤24 defects, i.e. d≤7 surface code); Dijkstra over rotated lattice for syndrome-graph edge weights.
  - `UnionFindDecoder` — Delfosse-Nickerson weighted union-find decoder with peeling.
  - `PauliFrame` — Clifford-conjugation Pauli-frame tracker (H/S/CNOT propagation rules).
  - Python bindings via `py::qec` (`PyRotatedSurfaceCode`, `PyMwpmSurfaceDecoder`, `PyUnionFindDecoder`, `PyPauliFrame`).

- **3D Quantum State Visualization** (`core/src/state_visualization_3d/`): five Plotly-JSON renderers:
  - Multi-qubit Bloch-sphere array (per-qubit reduced density matrix via `partial_trace`).
  - Q-sphere (Qiskit-style, latitude ∝ Hamming weight, phase-colored markers).
  - Discrete Wigner function (Wootters 1987 displacement operators for n=1,2; explicit error for n≥3).
  - Husimi-Q distribution (spin-coherent state projection on 64×64 grid).
  - Density-matrix 3D bar plot (Re/Im side-by-side, basis labels).
  - Python bindings via `PyQuantumState3DVisualizer` with `{bloch_array,qsphere,wigner,husimi,density_bars}_html()`.

- **Tytan Advanced Samplers** (`tytan/src/sampler/`): three new native-Rust QUBO/PUBO samplers:
  - `TabuSampler` — FIFO-ring tabu search with O(n) incremental ΔE, aspiration criterion, restart-from-best strategy.
  - `SBSampler` — Toshiba Simulated Bifurcation (Goto-Tatsumura-Dixon 2019) in two variants: Ballistic (bSB) and Discrete (dSB); symplectic Euler dynamics.
  - `PopulationAnnealingSampler` — Hukushima-Iba population annealing with importance-weighted resampling and log-sum-exp ESS threshold.
  - All three implement the canonical `Sampler` trait (`run_qubo`, `run_hobo`).

- **Tytan Sampler Test Coverage** (`tytan/tests/sampler_tests.rs`): expanded from 278 to 1429 lines:
  - Canonical problem suite: K4 Max-Cut, number partitioning, 3-SAT-as-QUBO.
  - Cross-sampler agreement: SA, GA, PT, Tabu, SB (bSB+dSB), PA — all must find the same minimum on shared instances.
  - Determinism tests: same seed → identical results.
  - HOBO smoke tests: 3-body PUBO instances for all new samplers.
  - Random-QUBO property tests: 20-seed sweep over n=4 instances with brute-force optimal verification.

---

## [0.1.3] - 2026-03-27

### Further Enhancements (2026-03-27)

#### Policy Compliance — Files Split Below 2000-Line Limit
- `quantrs2/src/lib.rs` (2073→1680): removed duplicate inline test block, use external `feature_gate_tests.rs`
- `py/src/lib.rs` (2066→300): extracted `CircuitOp`/`PyCircuit` → `circuit_core.rs`, `PySimulationResult` → `simulation_result.rs`, `PyRealisticNoiseModel` → `noise_model.rs`
- `circuit/src/builder.rs` (2050): converted to `builder/` directory with `mod.rs` + `tests.rs`
- `core/src/quantum_walk.rs` (2004): converted to `quantum_walk/` directory with 8 files (graph, discrete, continuous, multi, search, eigensolvers, tests)

#### Circuit ML Optimization (circuit crate)
- Implemented Q-learning circuit optimizer (`optimize_with_rl`): Q-table, ε-greedy exploration, depth/gate-count reward
- Implemented genetic algorithm optimizer (`optimize_with_ga`): tournament selection, OX crossover, mutation, elitism
- Implemented neural network optimizer (`optimize_with_nn`): feedforward forward pass with learned action selection

#### Tensor Network (circuit crate)
- Implemented `TensorNetwork::compress` using SVD truncation with bond dimension and tolerance controls
- Implemented `MatrixProductState::from_circuit`: |0…0⟩ initialization + per-gate unitary contraction + SVD split
- Implemented `MatrixProductState::compress`: left-to-right SVD sweep with truncation

#### VQE Enhancement (circuit crate)
- Fixed `set_parameters` to actually rebuild parameterized gates with updated rotation angles
- Added `ParameterizedGateRecord` tracking for Ry/Rz/Rx gates linked to parameter indices

#### Quantum Supremacy Simulation (sim crate)
- Implemented `apply_gate_to_state`: inline statevector gate application for H, X, Y, Z, S, T, SX, SqrtY, SqrtW, RZ, RX, RY, CNOT, CZ
- Implemented `sample_from_amplitudes`: inverse-CDF bitstring sampling from |amplitude|² probabilities
- Replaced zero-filled state vector placeholders with real simulation results

#### Distributed Job Tracking (circuit crate)
- Implemented `JobRecord` struct + `job_registry: HashMap<String, JobRecord>` in `DistributedExecutor`
- Implemented `submit_job`, `get_job_status`, `cancel_job`, `get_results` with proper state transitions
- Fixed `expect()` calls in backend selection with graceful `.unwrap_or()` fallbacks

#### Tytan Enhancements
- Implemented `constraint_impact` in sensitivity analysis using real constraint violation counting
- Implemented `get_nbit_value` in auto_array via SymEngine expression evaluation
- Replaced random GPU HOBO solver with proper simulated annealing (Metropolis criterion, geometric cooling)
- Implemented mean-field Hamiltonian evaluation for hybrid quantum-classical algorithms

#### Bug Fixes
- Fixed `error_mitigation.rs` production `unwrap()` → `ok_or_else` with descriptive error message
- Fixed `const fn` qualifier on non-const functions in quantum supremacy and auto_array modules
- Fixed `const fn has_value` in py/gates.rs calling non-const method

---

### Further Enhancements (2026-03-23 continuation 2)

#### Code Quality — No Unwrap Policy
- Eliminated approximately 210 `unwrap()` calls across production code; all replaced with proper error propagation via the `?` operator or `ok_or`/`ok_or_else` combinators
- Test functions across the workspace converted to `-> std::result::Result<(), Box<dyn std::error::Error>>` signatures to support `?`-based assertion propagation
- All production paths now return typed errors instead of panicking on unexpected `None`/`Err` values

#### Algorithm Implementations (core crate)
- **ZYZ Decomposition** (`core/src/synthesis.rs`): Corrected theta formulas to `theta1 = -arg(a) - arg(c)` and `theta2 = arg(c) - arg(a)`; removed `#[ignore]` from the corresponding test
- **Holonomic Gate Synthesis** (`core/src/holonomic.rs`): Removed `#[ignore]`; test now runs with graceful convergence handling instead of hard-panicking on non-convergence
- **Cartan (KAK) Decomposition** (`core/src/cartan.rs`): Implemented real QR iteration eigensolver (Householder tridiagonalization + Givens rotations) for interaction coefficient extraction, replacing the previous placeholder
- **Adiabatic Eigenvalue Solver** (`core/src/adiabatic.rs`): Replaced diagonal placeholder with inverse power iteration + deflation + Rayleigh quotient refinement for accurate ground-state energy estimation
- **Amplitude Encoding** (`core/src/qml/encoding.rs`): Implemented Mottonen-style amplitude encoding with recursive binary-tree multiplexor for arbitrary state preparation
- **IQP ZZ Interaction** (`core/src/qml/encoding.rs`): Implemented correct `e^{-iθ/2 Z⊗Z}` via CNOT·(I⊗RZ(θ))·CNOT decomposition
- **Rotation Layer Gradients** (`core/src/qml/layers.rs`): Implemented parameter-shift rule for exact variational layer gradients
- **QPE Bug Fix** (`core/src/quantum_counting.rs`): Fixed controlled-U power application that was applying `U^(N·2^target)` instead of `U^(2^target)` per control qubit
- **QML NLP Parameters** (`core/src/qml/nlp.rs`): Implemented `parameters()` and `parameters_mut()` accessors for `QuantumWordEmbedding` and `QuantumAttention`
- **Symbolic Evaluation** (`core/src/symbolic.rs`): Wired `evaluate()`, `variables()`, and `substitute()` to the `quantrs2-symengine-pure` backend

#### Circuit Optimizations (circuit crate)
- **SABRE Routing** (`circuit/src/routing/sabre.rs`): Replaced uniform swap scoring with real coupling-map distance-based scoring for better routing quality
- **Noise-Aware Optimization** (`circuit/src/optimization/noise.rs`): Implemented ASAP scheduling via Kahn's topological-sort algorithm; added greedy noise-aware qubit remapping; added XY4, CPMG, and XY8 dynamical decoupling insertion
- **Template Matching** (`circuit/src/optimizer.rs`): Added 2-gate peephole patterns (H-H cancellation, X-X cancellation) and a 3-gate pattern (H-X-H → Z)
- **VQE Gradients** (`circuit/src/vqe.rs`): Implemented full parameter-shift rule for exact analytical gradients
- **RL/GA/NN Circuit Optimization** (`circuit/src/ml_optimization.rs`): Implemented Q-learning optimizer (ε-greedy, depth/gate-count reward), genetic algorithm optimizer (tournament selection, OX crossover, elitism), and feedforward neural network policy for gate-sequence optimization

#### File Refactoring (2000-line policy)
- `tytan/src/advanced_visualization/types.rs` (1938 lines) split into `types/` module directory
- `device/src/hybrid_quantum_classical/types.rs` (1932 lines) split into `types/` module directory
- `device/src/cloud/cost_estimation.rs` (1903 lines) split into `cost_estimation/` module directory

#### Python Bindings (py crate)
- Migrated from pyo3 0.22 to pyo3 0.26 API: `Python::attach` (replaces `with_gil`), `Py<PyAny>` (replaces `PyObject`)
- Split `py/src/lib.rs` into `circuit_core.rs` (`CircuitOp`, `PyCircuit`), `simulation_result.rs` (`PySimulationResult`), and `noise_model.rs` (`PyRealisticNoiseModel`)
- Implemented error mitigation bindings: quasi-probability decomposition (PEC), virtual distillation via SWAP-test circuit, and symmetry verification (Z2/parity, U(1)/particle-number, time-reversal)

---

### Further Enhancements (2026-03-21 continuation 3)

#### Symbolic Expression Engine (core crate)
- Implemented SymEngine `evaluate()` wiring to `quantrs2-symengine-pure::eval()`
- Implemented SymEngine `evaluate_complex()` using complex-valued variable maps
- Implemented `variables()` / `free_symbols()` traversal on SymEngine expressions
- Implemented `is_constant()` using `free_symbols().is_empty()`
- Implemented `substitute()` iterating over variable→expression map
- Added `free_symbols()` method to `quantrs2-symengine-pure::Expression`
- Added `to_symengine_expr()` and `from_symengine_str()` bridge helpers

#### ZX-Calculus Optimization (core crate)
- Implemented real Clifford spider rewrite rules in `decompose_clifford_component`:
  - Spider fusion: same-color spiders on regular edge → merged with summed phase
  - Hadamard cancellation: zero-phase degree-2 spider with two H-edges → Regular wire
  - Identity removal: zero-phase degree-2 spider → pass-through wire
- Implemented `apply_tableau_reduction` with convergence loop (was `const fn` returning 0)

#### Quantum Walk Eigenvalue Solver (core crate)
- Replaced degree-sequence approximation with Golub-Reinsch QR iteration
- Implemented Householder tridiagonalization for real symmetric matrices
- Replaced broken implicit QR bulge-chase with Sturm-sequence bisection method
- Implemented Wilkinson-shift implicit QR iteration for tridiagonal eigenproblems
- Implemented Rayleigh-quotient Fiedler value estimation via power iteration
- Verified: P4 eigenvalues = {0, 2-√2, 2, 2+√2}, K4 Fiedler = 4.0

#### Python Mitigation (py crate)
- Implemented PEC `quasi_probability_decomposition`: returns (1+3n) quasi-probability terms
- Implemented Virtual Distillation SWAP-test circuit for M=2 copies
- Implemented `verify_symmetry` for Z2/parity, U(1)/particle-number, time-reversal

#### Compilation Fixes (2026-03-21)
- Resolved all E0761 duplicate-module errors by removing stale `.rs` files that conflicted with split module directories across `sim`, `device`, `anneal`, `ml`, `tytan`, and `circuit` crates
- Added `Circuit::from_gates()` constructor with `BoxGateWrapper` to support optimization pass pipeline
- Fixed `scirs2_core::Complex64` method names (`norm_sqr()` instead of `norm_squared()`, `norm()` instead of `abs()`) throughout `anneal` crate
- Removed custom `complex::Complex64` and `utils::Complex` shims in `anneal`, replaced with `scirs2_core::Complex64`
- Made private struct fields public in split modules (`DecoherenceModel`, `QuantumPositionalEncoder`, `QuantumAugmenter`, `QuantumMixtureOfExperts`)
- Replaced broken implicit QR bulge-chase (non-similarity-preserving) with unconditionally correct Sturm-sequence bisection for Laplacian eigenvalues

#### Refactoring (policy compliance)
- Split `ml/src/quantum_mixture_of_experts/types.rs` (1978 lines)
- Split `core/src/realtime_monitoring.rs` (1977 lines) into module directory
- Split `ml/src/quantum_implicit_neural_representations.rs` (1972 lines)
- Split `ml/src/quantum_self_supervised_learning.rs` (1945 lines)
- Split `anneal/src/qaoa.rs` (1945 lines) into module directory

---

## [0.1.2] - 2026-01-23

### Changed
- Bugfix (Python bindings)
- Update docs

### Fixed
- Dependency version updates for better compatibility

---

## [0.1.1] - 2026-01-21

### Fixed
- **Device crate**: Added missing `#[cfg(feature = "photonic")]` guard on photonic module re-exports
- **Cross-platform benchmarking**: Fixed conditional compilation for `aws`, `azure`, and `ibm` client imports and struct fields
- **Feature gating**: Improved conditional compilation to avoid compilation errors when cloud provider features are disabled

### Changed
- All workspace crate versions bumped from 0.1.0 to 0.1.1
- Updated workspace dependencies to use version 0.1.1

---

## [0.1.0] - 2026-01-20

### Added

#### Core Framework
- **QuantRS2 Quantum Computing Framework**: Complete modular quantum computing toolkit
- **quantrs2-core**: Core types, traits, and abstractions for quantum computing
- **quantrs2-circuit**: Quantum circuit representation with DSL and gate library
- **quantrs2-sim**: High-performance quantum simulators (state-vector, tensor-network, stabilizer)
- **quantrs2-device**: Remote quantum hardware integration (IBM Quantum, Azure Quantum, AWS Braket)
- **quantrs2-ml**: Quantum machine learning with QNNs, QGANs, and HEP classifiers
- **quantrs2-anneal**: Quantum annealing support with D-Wave integration
- **quantrs2-tytan**: High-level quantum annealing library
- **quantrs2-symengine-pure**: Pure Rust symbolic mathematics engine (100% Rust, no C++ dependencies)
- **quantrs2-py**: Python bindings via PyO3 for seamless Python integration

#### SciRS2 Integration
- Full integration with SciRS2 ecosystem (v0.1.2) for scientific computing
- Unified array operations via `scirs2-core::ndarray`
- Unified random number generation via `scirs2-core::random`
- Complex number support via `scirs2_core::{Complex64, Complex32}`
- SIMD-accelerated quantum operations via `scirs2-core::simd_ops`
- Parallel quantum circuit execution via `scirs2-core::parallel_ops`
- GPU acceleration support via `scirs2-core::gpu`

#### Quantum Algorithms
- **Grover's Algorithm**: Quantum search with amplitude amplification
- **Quantum Fourier Transform (QFT)**: Foundation for quantum algorithms
- **Variational Quantum Eigensolver (VQE)**: Quantum chemistry and optimization
- **Quantum Approximate Optimization Algorithm (QAOA)**: Combinatorial optimization
- **Shor's Algorithm**: Integer factorization (simulation)
- **Quantum Phase Estimation (QPE)**: Eigenvalue estimation
- **Quantum Machine Learning**: QNN, QGAN, quantum reservoirs, HEP classifiers

#### Quantum Simulators
- **State Vector Simulator**: Up to 30+ qubits with optimized complex arithmetic
- **Stabilizer Simulator**: Up to 50+ qubits using stabilizer formalism
- **Tensor Network Simulator**: Efficient simulation for circuits with limited entanglement
- **Density Matrix Simulator**: Mixed state and open quantum system support
- **Quantum Reservoir Computing**: Novel ML approach with quantum dynamics

#### Hardware Integration
- IBM Quantum platform integration with Qiskit compatibility
- Azure Quantum integration
- AWS Braket integration
- D-Wave quantum annealer support
- Error mitigation and measurement optimization

#### Performance Features
- SIMD vectorization for quantum gate operations
- Multi-threaded parallel execution for independent operations
- GPU acceleration for large-scale quantum simulation
- Sparse matrix representations for memory efficiency
- Adaptive chunking for tensor network contractions

#### Documentation
- Comprehensive API documentation with rustdoc
- Quantum algorithm examples and tutorials
- Integration guides for SciRS2 ecosystem
- Python binding examples
- Performance benchmarking suite

### Changed
- Migration from SymEngine C++ bindings to pure Rust implementation (`quantrs2-symengine-pure`)
- Unified dependency management via workspace inheritance
- Optimized memory layout for quantum states
- Enhanced error handling with detailed quantum-specific error types

### Fixed
- Rustdoc HTML tag warnings in symbolic mathematics module
- Clippy warnings across all workspace crates
- Feature flag dependencies for optional GPU and CUDA support
- Documentation generation for docs.rs

### Compatibility

#### Target Frameworks (99%+ Compatibility)
- **Stim**: Stabilizer circuit simulation (99%+ compatibility)
- **cuQuantum**: NVIDIA GPU quantum simulation (95%+ compatibility)
- **TorchQuantum**: PyTorch quantum ML integration (99%+ compatibility)
- **IBM Qiskit**: IBM quantum platform (90%+ compatibility)
- **Google Cirq**: Google quantum platform (90%+ compatibility)
- **PennyLane**: Quantum ML framework (85%+ compatibility)

#### Pure Rust Policy
- **100% Pure Rust** default features (no C/C++/Fortran dependencies)
- Optional C/C++ dependencies feature-gated (CUDA, MKL)
- Full compliance with COOLJAPAN Pure Rust Policy

#### SciRS2 Ecosystem
- **SciRS2**: v0.1.2 (scientific computing core)
- **NumRS2**: v0.1.2 (numerical computing)
- **OptiRS**: v0.1.0 (ML optimization algorithms)
- **OxiBLAS**: Pure Rust BLAS implementation
- **Oxicode**: Pure Rust binary encoding

### Security
- Memory-safe quantum state management via Rust ownership
- No unsafe code in default features
- Dependency audit passing
- Secure random number generation for quantum measurements

### Performance
- State vector simulation: 30+ qubits
- Stabilizer simulation: 50+ qubits
- Tensor network simulation: 50+ qubits (circuit-dependent)
- SIMD acceleration: 2-4x speedup on supported platforms
- GPU acceleration: 10-100x speedup for large circuits (optional)

### Platform Support
- Linux (x86_64, aarch64)
- macOS (x86_64, Apple Silicon)
- Windows (x86_64)
- WebAssembly (wasm32)

### License
- Dual licensed under Apache-2.0

### Authors
- COOLJAPAN OU (Team Kitasan)

### Repository
- <https://github.com/cool-japan/quantrs>

### Documentation
- API Docs: <https://docs.rs/quantrs2>
- Examples: See `examples/` directory
- Integration Guide: See `SCIRS2_INTEGRATION_POLICY.md`

---

## [Unreleased]

No unreleased changes yet.

---

[0.1.3]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.0
