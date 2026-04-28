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

## Proposed follow-ups

- **SIMD QUBO energy eval for HOBO (higher-order terms)** — extend energy_delta_simd to handle 3-body and 4-body PUBO interactions efficiently.
- **GPU-only SB kernel via wgpu/ocl** — port SBSampler inner loop to GPU compute shader for large dense instances (n ≥ 1000).
- **Tensor-core HOBO contraction** — BLAS-level contraction for high-order Boltzmann machine terms.
