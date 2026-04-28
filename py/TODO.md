# quantrs2-py Roadmap

## v0.1.3 (2026-03-27)
- 108 public API items; aligned with QuantRS2 v0.1.3 workspace

## Future Enhancements

- [ ] Quantum hardware integration: Direct integration with more providers
- [x] Documentation expansion: Enhanced tutorials and examples (implemented 2026-04-27)
  - **Goal:** New `py/python/quantrs2/tutorials/` package with 8 standalone Python tutorial scripts covering every major QuantRS2 feature surface. Each tutorial: 150–300 lines, extensive in-line commentary explaining quantum-computing concepts, runnable as `python -m quantrs2.tutorials.NN_topic`. Tutorials double as integration tests via a single pytest harness. Plus README.md tutorials index.
  - **Design:** tutorials/__init__.py + 01_bell_state.py + 02_vqe_h2.py + 03_qaoa_maxcut.py + 04_qec_surface_code.py + 05_qubo_sampling.py + 06_3d_state_visualization.py + 07_parameterized_circuits.py + 08_error_mitigation.py. Each guards optional imports (matplotlib, scipy) with try/except.
  - **Files:** new `py/python/quantrs2/tutorials/` (9 files). New `py/tests/test_tutorials.py`. Modify `README.md`.
  - **Tests:** subprocess run each tutorial script, assert exit code 0. `@pytest.mark.slow` on heavy tutorials.
  - **Risk:** Tutorial fragility if bindings missing — each does `try: import; except: sys.exit(0)`.
- [x] Quantum error correction: Advanced QEC with surface codes (implemented 2026-04-27)
- [ ] Quantum networking: Extended communication protocols
- [ ] Hybrid algorithms: More quantum-classical hybrid approaches
- [x] Advanced visualization: 3D quantum state visualization
- [ ] Quantum compilers: Advanced circuit compilation
- [ ] Enterprise security: Enhanced security features
- [~] Scalability testing: Large-scale simulation validation (planned 2026-04-27)
  - **Goal:** Scalability test + bench suite for state-vector sim, MPS sim, and QUBO sampling at real-world problem sizes. Smoke tests < 60s; heavy benches behind `--ignored`.
  - **Design:** sim/tests/scalability_smoke.rs (5 smoke tests: 15q SV, 18q QFT, 20q MPS GHZ, 25q MPS random, 30q stabilizer), sim/benches/large_scale_simulation.rs (criterion), tytan/tests/sampler_scalability_smoke.rs (50/100/200-var QUBO smokes), tytan/benches/sampler_scalability.rs (criterion), py/tests/test_scalability.py.
  - **Files:** 6 new files across sim/, tytan/, py/tests/.
  - **Tests:** nextest smoke tests + bench compile-check.
  - **Risk:** MPS bond-dim explosion on deep random circuits — fallback to depth 5.
- [ ] Integration testing: Comprehensive external system testing
- [~] Performance optimization: Further SIMD and GPU optimizations (planned 2026-04-27)
  - **Goal:** SIMD-accelerated single-qubit-gate kernels in `sim/` crate. Target ≥ 2× over scalar for H/X/Y/Z/S/T/RX/RY/RZ on q=0. Pure-Rust via `wide` crate (no nightly std::simd).
  - **Design:** sim/src/state_vector_simd.rs (~600 lines, 9 gate kernels), dispatch in sim/src/state_vector.rs, sim/benches/simd_state_vector.rs (criterion scalar vs SIMD), sim/tests/state_vector_simd_correctness.rs (L2 < 1e-12 correctness suite).
  - **Files:** 3 new files in sim/; modify sim/src/state_vector.rs, sim/Cargo.toml (add `wide`).
  - **Tests:** correctness suite for every gate at n ∈ {3,5,7}, each target qubit.
  - **Risk:** Floating-point non-determinism across SIMD lanes — tolerance 1e-12 in tests.
- [ ] Ecosystem integration: Enhanced quantum software stack compatibility

## Proposed follow-ups

- **Quantum hardware integration** — blocked: requires user decision on provider(s) (IBM Q / Rigetti / AWS Braket / IonQ / Quantinuum) plus credentials. Surface as per-provider sub-tickets once provider is chosen.
- **Quantum networking** — vague: surface as concrete-protocol sub-tickets (BB84, E91, teleportation chains, entanglement swapping).
- **Hybrid algorithms** — vague: surface as algorithm sub-tickets (QAOA-warm-start, VQE-natural-gradient, hybrid kernel methods).
- **Quantum compilers** — vague: surface gap items vs existing SABRE/ZX/noise-aware coverage (cross-compiler IR bridge, pulse-level compilation).
- **Enterprise security** — vague: surface as concrete sub-tickets (credential storage abstraction, audit logging, multi-tenant sandboxing).
- **Integration testing** — blocked: external cloud/hardware backends require credentials (same as hardware integration).
- **Ecosystem integration** — vague: surface as Qiskit-OpenQASM-roundtrip, Cirq-import, PennyLane-bridge sub-tickets.
