# quantrs2-tytan Roadmap

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

## v0.2.0 (2026-05-17)

- [x] HOBO energy library (`tytan/src/sampler/energy.rs`) — 3-body and 4-body specialized fast paths plus generic ndim dispatch; all CPU samplers routed through single shared kernel.
  - **Design:** `hobo_energy_full_dispatch` picks specialized `ArrayView3`/`ArrayView4` paths for ndim ∈ {3,4} (early-out pruning ≥ 90% on 50% dense state), generic `indexed_iter` for ndim ≥ 5, existing scalar QUBO path for ndim == 2.
  - **Delta / influence API:** `hobo_energy_delta_{3,4}body`, `hobo_compute_influence`, `hobo_update_influence`, `hobo_recompute_influence`.
  - **Key correctness insight:** ΔE = (1 − 2x[k]) · g[k] where g[k] counts each tensor entry containing k ONCE regardless of k-multiplicity (binary identity x[k]^m = x[k]).
  - **Samplers updated:** `PopulationAnnealing`, `SimulatedAnnealing`, `GASampler` — all ndim warnings / silent-zero returns eliminated.
  - **Tests added:** 8 HOBO correctness tests in `tytan/tests/energy_correctness.rs`; all 507 tytan tests pass.
- [x] Build fix — removed broken `serialization` feature from scirs2-core (was pulling oxiarc-lz4/zstd 0.2.8 with unresolved import paths against oxiarc-core 0.3.0 API).
- [x] CIM noise injection (2026-05-18) — replaced hardcoded `Complex64::new(0.0, 0.0)` noise at `coherent_ising_machine.rs:172` with Euler-Maruyama Gaussian increments via existing `scirs2_core::random::RandNormal`. Seeded from `self.seed`; deterministic replay guaranteed. Added 4 tests in `tytan/tests/cim_noise_tests.rs`.
- [x] HOBO 3/4-body outer-loop parallelization (2026-05-18) — added `into_par_iter` outer loop to `hobo_energy_full_3body` (threshold n≥32) and `hobo_energy_full_4body` (threshold n≥16) in `energy.rs`. Scalar fallback for small n avoids rayon spawn overhead. FP accumulation tolerance 1e-7 (3body n=64) / 1e-9 (4body n=16). 2 new tests added.
- [x] Hardware HOBO support via Rosenberg quadratization (2026-05-18) — added `hobo_to_qubo` to `energy.rs`: Rosenberg substitution y_{ij}=x_i*x_j with penalty P=(1+max|T|)*n; builds extended var_map with `_aux_i_j` entries. All four hardware samplers (FPGA, NEC, Hitachi, Fujitsu) now call `hobo_to_qubo` + `run_qubo` and strip auxiliary variable keys from results.

## Proposed follow-ups

- **GPU-only SB kernel via wgpu/ocl** — port SBSampler inner loop to GPU compute shader for large dense instances (n ≥ 1000).
- **Tensor-core HOBO contraction** — BLAS-level contraction for high-order Boltzmann machine terms.
- **MIKAS/Armin GPU HOBO** — depends on GPU SB kernel above.
