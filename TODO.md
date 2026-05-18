# QuantRS2 Roadmap

## Current Version: 0.2.0 (in progress)

## v0.2.0 (in progress, branch: 0.2.0)

- [x] **Build fix** — removed broken `serialization` feature from scirs2-core in workspace `Cargo.toml`; this feature transitively pulled `oxiarc-lz4/zstd 0.2.8` which referenced `oxiarc_core::cancel`/`progress` paths that don't exist in `oxiarc-core 0.3.0`, blocking the entire workspace from compiling.
- [x] **HOBO energy library** (`tytan/src/sampler/energy.rs`) — unified 3-body and 4-body PUBO energy evaluation with early-out pruning; wired into `PopulationAnnealing`, `SimulatedAnnealing`, and `GASampler` replacing per-sampler incomplete implementations. See [tytan/TODO.md](tytan/TODO.md) for full design notes.
- [x] **Clippy fixes** — `manual_checked_ops` in `sim/src/cuda/kernels.rs`, `map_unwrap_or` in `quantrs2/src/diagnostics.rs`.
- [x] **CIM stochastic noise** (`tytan/src/coherent_ising_machine.rs`) — replaced hardcoded zero noise with Euler-Maruyama Gaussian increments via `scirs2_core::random::RandNormal`. CIM now converges stochastically as designed.
- [x] **HOBO 3/4-body parallelization** (`tytan/src/sampler/energy.rs`) — outer-loop `into_par_iter` for n≥32 (3-body) / n≥16 (4-body) with scalar fallback for small tensors.
- [x] **ML k-means clustering** (`ml/src/clustering/core.rs`) — replaced placeholder `fit`/`predict` with Lloyd's algorithm and k-means++ greedy initialization. Real inertia = sum of squared within-cluster distances. DBSCAN path uses iterative union-find.
- [x] **Quantum Differential Evolution + Quantum Harmony Search** (`sim/src/quantum_inspired_classical/types.rs`) — replaced `NotImplemented` stubs with DE/rand/1/bin + quantum tunneling, and HS harmony memory + quantum pitch adjustment respectively. 73 new tests added workspace-wide; total 5267 tests pass.
- [x] **Classical QAOA + Classical VQE + Quantum Ant Colony** (`sim/src/quantum_inspired_classical/framework.rs`) — replaced `NotImplemented` stubs with sinusoidal gamma/beta schedule (QAOA), coordinate-descent parameter-shift (VQE), and 10-level pheromone roulette with quantum interference (ACO). Signatures updated from `&self` to `&mut self`.
- [x] **Quantum-inspired framework completions** (`sim/src/quantum_inspired_classical/framework.rs`) — `train_ml_model` (linear W·x+b SGD + tunneling perturbations), `sample` (Metropolis-Hastings MCMC with quantum jump proposals + full `SampleStatistics`), `solve_linear_algebra` (Gauss-Seidel for complex A·x=b), `solve_graph_problem` (greedy graph coloring + max-cut with quantum tunneling + BFS `GraphMetrics`).
- [x] **Device noise characterization stubs** (`device/src/scirs2_noise_characterization_enhanced/impls.rs`) — `calculate_error_bars` (Wald 95% CI), `calculate_fit_confidence_interval` (residual SE on decay-constant fit). Crosstalk mitigation (`device/src/advanced_crosstalk_mitigation/prediction.rs`) — bootstrap fallbacks for Bayesian/ensemble/dropout uncertainty; Adams-MacKay BOCPD changepoint detection with NIG prior; exponential-smoothing + ARIMA fallbacks for deep learning forecasting stubs.
- [x] **`quantum_inspired_classical/types.rs` refactoring** — split 2529-line file into `types.rs` (734 lines, data structures), `framework.rs` (1711 lines, `impl QuantumInspiredFramework`), and `extra_types.rs` (103 lines, `QuantumInspiredUtils`). Total 5270 tests pass.
- [x] **VQA objective evaluator stubs** (`device/src/vqa_support/objectives.rs`) — implemented TSP/MIS/Portfolio/custom cost functions (graph-aware), `simulate_circuit_exact` (RY product statevector), finite/central/forward/natural/automatic gradient methods (self-referential via `self.evaluate()`), and all evaluation types: cost (Z expectation), classification (cross-entropy), regression (MSE), fidelity (state overlap), expectation value (parity). Gradient methods changed from static `fn` to `&self` for function-evaluation access.
- [x] **JIT compiler multi-qubit stubs** (`sim/src/jit_compilation/compiler.rs`) — `apply_multi_qubit_gate`: controlled-gate expansion for 3+ qubits via 4×4 matrix on first two targets; `extract_gates_from_instructions`: graceful handling of `FusedOperation`/`Prefetch`/`Barrier` instructions; `TensorContraction`/`SparseOperation` in `execute_matrix_operations`: matrix application via existing single/two-qubit primitives.
- [x] **Hardware HOBO quadratization** (`tytan/src/sampler/hardware/{fpga,nec,hitachi,fujitsu}.rs`, `tytan/src/sampler/energy.rs`) — `hobo_to_qubo`: Rosenberg quadratization reducing 3-body PUBO to QUBO with auxiliary variables and penalty terms (P = (1 + max|T|) × n). Hardware samplers now call `hobo_to_qubo` + `run_qubo`, then strip `_aux_` entries from result assignments.
- [x] **execute_vqa gradient-descent loop** (`device/src/vqa_support/mod.rs`) — replaced async placeholder with synchronous gradient-descent loop: `evaluate()` → track best → `compute_gradient()` → `params -= grad * 0.01`; `analyze_convergence` statistics; `VQAResult` with full history. Step size 0.01 with configurable max_iterations and convergence_tolerance.
- [x] **Error mitigation Lagrange extrapolation** (`sim/src/error_mitigation.rs`) — replaced Richardson extrapolation fallback for degree > 2 with full Lagrange polynomial interpolation at x=0: `L(0) = Σ_i y_i * Π_{j≠i}(-x_j) / (x_i - x_j)`. Handles arbitrary noise scaling points beyond linear/quadratic Richardson.
- [x] **STIM REPEAT block parser** (`sim/src/stim_parser/types.rs`, `functions.rs`) — rewrote `StimCircuit::from_str` to handle multi-line `REPEAT N { … }` blocks via peekable line iterator with brace-terminated body accumulation. Recursive `from_str` call on the body; `parse_repeat` now returns `Repeat { count, instructions: body }` for single-line forms instead of `NotImplemented`.
- [x] **Cache-optimized two-qubit gate** (`sim/src/cache_optimized_layouts.rs`) — implemented `apply_two_qubit_gate_cache_optimized`: iterates over |00⟩ corner states, applies logical↔physical qubit permutation, extracts and updates 4 amplitudes `[|00⟩, |01⟩, |10⟩, |11⟩]` using the 4×4 gate matrix. Cache-friendly access pattern exploiting layout metadata.
- [x] **BBM92 and SARG04 QKD protocols** (`ml/src/crypto.rs`) — implemented `bbm92_protocol` (entanglement-based, ~50% key retention via basis sifting on maximally entangled Bell pairs) and `sarg04_protocol` (BB84 variant with unambiguous state discrimination, ~25% key retention via USD conclusive-measurement filter). Dispatch wired in `run_qkd`.

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

- [x] Finalize public API for 1.0 release (completed 2026-04-28)
  - **Goal:** Concrete 1.0 API surface hardening: (a) add `#[non_exhaustive]` to all public enums/structs likely to gain variants; (b) fix naming-convention violations across the workspace; (c) audit and annotate `quantrs2/src/lib.rs` re-exports with doc comments and `#[doc(hidden)]` where appropriate; (d) add crate-level `//!` rustdoc to every `lib.rs` that lacks one; (e) ensure all `pub use` re-exports in the top-level aggregator crate have accompanying documentation.
  - **Design:** Error enum sweep across core/sim/anneal/tytan/circuit/ml/device — add `#[non_exhaustive]` to extensible enums. Naming audit via clippy. Crate `//!` sweep for core/sim/anneal/circuit/ml/device. `quantrs2/src/lib.rs` audit with `#[doc(hidden)]` on internal-but-pub items.
  - **Files:** `quantrs2/src/lib.rs`, `core/src/lib.rs`, `sim/src/lib.rs`, `anneal/src/lib.rs`, `tytan/src/lib.rs`, `circuit/src/lib.rs`, `ml/src/lib.rs`, `device/src/lib.rs` — targeted attribute + doc additions. No new files.
  - **Prerequisites:** None.
  - **Tests:** `cargo clippy --all-features --all-targets -- -D warnings` green. `cargo doc --no-deps -p quantrs2` zero warnings.
  - **Risk:** `#[non_exhaustive]` may break exhaustive `match` in tests — add wildcard arms where needed.
- [x] Complete documentation for all public interfaces (completed 2026-04-28)
  - **Goal:** Zero rustdoc warnings workspace-wide. Every `pub fn`, `pub struct`, `pub enum`, and `pub trait` across all 10 crates must have at least one `///` doc line. Top 10 most-used public APIs per crate get runnable `# Examples` blocks.
  - **Design:** Per-crate grep for bare `pub` items without preceding `///`. Add missing doc comments and `//!` module-level blocks. Add compilable `# Examples` for the 10 most-important types/functions per crate.
  - **Files:** Doc-comment additions across all module source files. No new files.
  - **Prerequisites:** Item 1 (API finalization) — complete first to stabilize the pub surface.
  - **Tests:** `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` zero warnings. `cargo test --doc --workspace --all-features` all examples pass.
  - **Risk:** Large mechanical scope — hundreds of pub items; mitigated by systematic per-crate approach.
- [x] Add comprehensive examples for all major features (completed 2026-04-28)
  - **Goal:** 15 runnable `examples/*.rs` binaries across 6 crates (core, sim, anneal, tytan, circuit, ml), each 100–200 lines demonstrating a key capability. Runnable via `cargo run --example <name> -p quantrs2-<crate>`.
  - **Design:** core (bell_pair, grover_search, quantum_phase_estimation), sim (state_vector_demo, mps_ghz, noisy_circuit), anneal (ising_ground_state, max_cut_annealing), tytan (number_partition, max_cut_qubo, knapsack_pubo), circuit (compile_and_route, zx_optimize), ml (qnn_xor, amplitude_encoding).
  - **Files:** 15 new `examples/*.rs` across 6 crates (~2500 LoC). Modify each crate's `Cargo.toml` for `[[example]]` stanzas as needed.
  - **Prerequisites:** None — uses stable existing APIs only.
  - **Tests:** All 15 examples run exit-0 via `cargo run --example <name> -p quantrs2-<crate> --all-features`.
  - **Risk:** Examples may expose latent API bugs — fix any found rather than working around them.
- [x] Add contribution guidelines

## SymEngine Integration (quantrs2-symengine-pure — 5,739 LoC, 333 public APIs)

- [x] Complete SymEngine pure-Rust implementation (parser, eval, diff, simplify, quantum, matrix)
- [x] Complete the patching of the `symengine` crate → replaced with `symengine-pure`
- [x] Fix type and function reference issues → pure-Rust rewrite, no C/Fortran deps
- [x] Test compatibility with the D-Wave system

## Module-Specific Roadmaps

- [quantrs2-tytan](tytan/TODO.md)
- [quantrs2-py](py/TODO.md)
