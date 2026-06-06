# quantrs2-tytan Roadmap

## v0.2.0 (released 2026-06-06)

- [x] **Advanced samplers** (`src/sampler/`): `TabuSampler` (FIFO-ring tabu search, O(n) incremental ΔE, aspiration + restart-from-best); `SBSampler` (Toshiba Simulated Bifurcation, Ballistic + Discrete variants, symplectic Euler); `PopulationAnnealingSampler` (Hukushima–Iba population annealing with importance-weighted resampling).
- [x] **Tytan Energy Engine** (`src/sampler/energy.rs`): shared QUBO/PUBO energy kernels with autovectorized `_simd` companions, specialized 3-/4-body HOBO fast paths (`hobo_energy_full_dispatch` picks `ArrayView3`/`ArrayView4` for ndim∈{3,4} with ≥90% early-out pruning; generic `indexed_iter` for ndim≥5), delta/influence API (`hobo_energy_delta_{3,4}body`, `hobo_compute/update/recompute_influence`), and `hobo_to_qubo` Rosenberg quadratization. `PopulationAnnealing`/`SimulatedAnnealing`/`GASampler` routed through the shared kernel.
- [x] **HOBO 3/4-body parallelization** — rayon `into_par_iter` outer loop in `hobo_energy_full_3body` (n≥32) / `_4body` (n≥16) with scalar fallback below threshold.
- [x] **Hardware / CIM / photonic HOBO support via Rosenberg quadratization** — `run_hobo` (previously `NotImplemented`) now quadratizes via `hobo_to_qubo` (y_{ij}=x_i·x_j, penalty P=(1+max|T|)·n, `_aux_i_j` vars), solves the QUBO, and strips auxiliary keys; wired for CIM, photonic, and the four hardware samplers (FPGA/NEC/Hitachi/Fujitsu).
- [x] **CIM stochastic noise fix** (`src/coherent_ising_machine.rs`) — replaced hardcoded `Complex64::new(0.0, 0.0)` noise with seeded Euler–Maruyama Gaussian Wiener increments (`scirs2_core::random::RandNormal`); 4 tests in `tytan/tests/cim_noise_tests.rs`.
- [x] **VQF multilevel factorization** (`src/variational_quantum_factoring.rs`) — opt-in `with_multilevel` recursive full prime factorization.
- [x] **Build fix** — removed broken `serialization` feature from scirs2-core (was pulling `oxiarc-lz4/zstd 0.2.8` with unresolved import paths against the `oxiarc-core 0.3.0` API).
- [x] **Expanded test coverage** — `tests/sampler_tests.rs` (cross-sampler agreement, determinism, HOBO smoke, random-QUBO property tests) and `tests/energy_correctness.rs` (HOBO energy/delta correctness for n∈{4,16,32,64,128}).

## v0.1.3 (2026-03-27)

- [x] Advanced optimization algorithms — TabuSampler, SBSampler (bSB/dSB), PopulationAnnealingSampler implemented
- [x] Large file refactoring — all modules split into directory structures; all files under 2000 lines
- [x] TODO/FIXME resolution — 0 `todo!/unimplemented!` stubs remain; 1,874 public items fully implemented
- [x] Performance optimizations (2026-04-27)
  - **Completed:** SIMD-accelerated incremental QUBO energy-delta computation as shared inner loop.
  - **Integrated:** TabuSampler, SBSampler, PASampler use energy_full_simd / energy_delta_simd / compute_influence_simd / update_influence_simd from tytan/src/sampler/energy.rs.
  - **SASampler:** API mismatch — delegates to `quantrs2_anneal::ClassicalAnnealingSimulator`, which owns its own energy bookkeeping. SIMD integration would require rewriting SA from scratch; documented in sampler source with comment.
  - **New files:** tytan/src/sampler/energy.rs (~500 lines), tytan/benches/energy_eval.rs, tytan/tests/energy_correctness.rs (21 correctness tests).
  - **Tests:** energy_delta correctness for n ∈ {4,16,32,64,128}; FP tolerance relaxed to 1e-9 for n=128 (expected SIMD accumulation-order difference); all existing sampler tests pass.
- [x] Documentation enhancements (planned 2026-04-27)
  - **Goal:** Comprehensive //! module-level rustdoc for all 6 sampler files: algorithm description, mathematical formulation, citation (DOI/ISBN), parameter reference, when-to-use, runnable doc-examples. Comparison table in mod.rs. `cargo doc -p quantrs2-tytan` zero warnings.
  - **Design:** Extend top-of-file //! in: mod.rs (overview table + decision tree), simulated_annealing.rs (Kirkpatrick 1983), genetic_algorithm.rs (Holland 1975 ISBN), parallel_tempering.rs (Hukushima-Nemoto 1996, doi:10.1143/JPSJ.65.1604), tabu_search.rs (Glover 1989/1990), simulated_bifurcation.rs (Goto 2019, doi:10.1126/sciadv.aav2372), population_annealing.rs (Hukushima-Iba 2003, doi:10.1063/1.1632130).
  - **Files:** modify 7 existing files; no new files.
  - **Tests:** `cargo test --doc -p quantrs2-tytan` + `cargo doc --no-deps -p quantrs2-tytan` zero warnings.
  - **Risk:** Doc-test runtime — keep each example ≤ 10 lines, ≤ 20 shots.
- [x] Testing improvements — sampler property + integration tests (2026-04-27)

## Proposed follow-ups

- **GPU-only SB kernel via wgpu/ocl** — port SBSampler inner loop to GPU compute shader for large dense instances (n ≥ 1000).
- **Tensor-core HOBO contraction** — BLAS-level contraction for high-order Boltzmann machine terms.
- **MIKAS/Armin GPU HOBO** — depends on GPU SB kernel above.
