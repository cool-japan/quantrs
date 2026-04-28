# quantrs2-tytan Roadmap

## v0.1.3 (2026-03-27)

- [x] Advanced optimization algorithms — TabuSampler, SBSampler (bSB/dSB), PopulationAnnealingSampler implemented
- [x] Large file refactoring — all modules split into directory structures; all files under 2000 lines
- [x] TODO/FIXME resolution — 0 `todo!/unimplemented!` stubs remain; 1,874 public items fully implemented
- [~] Performance optimizations (planned 2026-04-27)
  - **Goal:** SIMD-accelerated incremental QUBO energy-delta computation as shared inner loop of all samplers (Tabu, SB, PA, SA, GA, PT). Plus parallel replica init for PA via rayon. Target ≥ 2× speedup on energy-delta for n ≥ 64 variable QUBOs.
  - **Design:** tytan/src/sampler/energy.rs (new ~400 lines: energy_full_simd, energy_delta_simd, build_dense_q), integrate into tabu_search.rs + simulated_bifurcation.rs + population_annealing.rs + simulated_annealing.rs, tytan/benches/energy_eval.rs (criterion), tytan/tests/energy_correctness.rs (scalar-vs-SIMD tolerance 1e-12).
  - **Files:** 3 new files; modify 4 sampler files + sampler/mod.rs.
  - **Tests:** energy_delta correctness for n ∈ {4,16,32,64,128}; all 34 existing sampler tests still pass.
  - **Risk:** Sampler regression from sign flip in energy eval — mitigated by correctness suite.
- [~] Documentation enhancements (planned 2026-04-27)
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
