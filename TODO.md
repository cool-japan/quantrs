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

- [x] Finalize public API for 1.0 release (completed 2026-04-28)
  - **Goal:** Concrete 1.0 API surface hardening: (a) add `#[non_exhaustive]` to all public enums/structs likely to gain variants; (b) fix naming-convention violations across the workspace; (c) audit and annotate `quantrs2/src/lib.rs` re-exports with doc comments and `#[doc(hidden)]` where appropriate; (d) add crate-level `//!` rustdoc to every `lib.rs` that lacks one; (e) ensure all `pub use` re-exports in the top-level aggregator crate have accompanying documentation.
  - **Design:** Error enum sweep across core/sim/anneal/tytan/circuit/ml/device — add `#[non_exhaustive]` to extensible enums. Naming audit via clippy. Crate `//!` sweep for core/sim/anneal/circuit/ml/device. `quantrs2/src/lib.rs` audit with `#[doc(hidden)]` on internal-but-pub items.
  - **Files:** `quantrs2/src/lib.rs`, `core/src/lib.rs`, `sim/src/lib.rs`, `anneal/src/lib.rs`, `tytan/src/lib.rs`, `circuit/src/lib.rs`, `ml/src/lib.rs`, `device/src/lib.rs` — targeted attribute + doc additions. No new files.
  - **Prerequisites:** None.
  - **Tests:** `cargo clippy --all-features --all-targets -- -D warnings` green. `cargo doc --no-deps -p quantrs2` zero warnings.
  - **Risk:** `#[non_exhaustive]` may break exhaustive `match` in tests — add wildcard arms where needed.
- [~] Complete documentation for all public interfaces (planned 2026-04-27)
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
