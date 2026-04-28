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
- [x] Quantum networking: Extended communication protocols (implemented 2026-04-28)
  - **Goal:** Simulate 3 canonical quantum communication protocols in `core/src/networking/`: BB84 QKD, E91 (Ekert 1991) QKD, and quantum teleportation with entanglement swapping. Full Python bindings. Each protocol: stateful simulation, noise model, concrete metrics (QBER, CHSH value, fidelity).
  - **Design:** `core/src/networking/channel.rs` (~150 lines) — QuantumChannel trait + Depolarizing/Dephasing/AmplitudeDamping impls. `core/src/networking/bb84.rs` (~400 lines) — Bb84Protocol with eavesdropping simulation, QBER estimation, privacy amplification. `core/src/networking/e91.rs` (~350 lines) — E91Protocol with CHSH computation (Alice bases {0°,45°,90°}, Bob {22.5°,67.5°,112.5°}), Bell test. `core/src/networking/teleportation.rs` (~350 lines) — TeleportationProtocol (Bell measurement + correction, fidelity) and EntanglementSwapping (n-hop chain). `py/src/networking.rs` (~220 lines) — PyBb84Protocol, PyE91Protocol, PyTeleportationProtocol Python classes; submodule `quantrs2.networking`.
  - **Files:** New `core/src/networking/` (5 files, ~1280 LoC). New `py/src/networking.rs` (~220 lines). Modify `core/src/lib.rs`, `py/src/lib.rs`.
  - **Prerequisites:** `ndarray::Array2<Complex64>`, `rand` — both already in workspace.
  - **Tests:** `core/tests/networking_tests.rs` (new, ~300 lines): BB84 QBER≈0 at p=0; QBER≈0.25 at 100% eavesdrop; E91 CHSH≈2√2 at noise=0; teleportation fidelity=1 at noise=0; n-hop fidelity monotone-decreasing. `py/tests/test_networking.py` (new, ~80 lines) smoke tests.
  - **Risk:** CHSH estimators need large n — use n=1000 with tolerance ±0.1 in unit tests; n=10000 for accuracy demos.
- [ ] Hybrid algorithms: More quantum-classical hybrid approaches
- [x] Advanced visualization: 3D quantum state visualization
- [ ] Quantum compilers: Advanced circuit compilation
- [ ] Enterprise security: Enhanced security features
- [x] Scalability testing: Large-scale simulation validation (implemented 2026-04-27)
  - **Goal:** Scalability test + bench suite for state-vector sim, MPS sim, and QUBO sampling at real-world problem sizes. Smoke tests < 60s; heavy benches behind `--ignored`.
  - **Design:** sim/tests/scalability_smoke.rs (5 smoke tests: 15q SV, 18q QFT, 20q MPS GHZ, 25q MPS random, 30q stabilizer), sim/benches/large_scale_simulation.rs (criterion), tytan/tests/sampler_scalability_smoke.rs (50/100/200-var QUBO smokes), tytan/benches/sampler_scalability.rs (criterion), py/tests/test_scalability.py.
  - **Files:** 6 new files across sim/, tytan/, py/tests/.
  - **Tests:** nextest smoke tests + bench compile-check.
  - **Risk:** MPS bond-dim explosion on deep random circuits — fallback to depth 5.
- [ ] Integration testing: Comprehensive external system testing
- [x] Performance optimization: Further SIMD and GPU optimizations (planned 2026-04-27)
  - **Goal:** SIMD-accelerated single-qubit-gate kernels in `sim/` crate. Target ≥ 2× over scalar for H/X/Y/Z/S/T/RX/RY/RZ on q=0. Pure-Rust via `wide` crate (no nightly std::simd).
  - **Design:** sim/src/state_vector_simd.rs (~600 lines, 9 gate kernels), dispatch in sim/src/state_vector.rs, sim/benches/simd_state_vector.rs (criterion scalar vs SIMD), sim/tests/state_vector_simd_correctness.rs (L2 < 1e-12 correctness suite).
  - **Files:** 3 new files in sim/; modify sim/src/state_vector.rs, sim/Cargo.toml (add `wide`).
  - **Tests:** correctness suite for every gate at n ∈ {3,5,7}, each target qubit.
  - **Risk:** Floating-point non-determinism across SIMD lanes — tolerance 1e-12 in tests.
- [~] Ecosystem integration: Enhanced quantum software stack compatibility (planned 2026-04-27)
  - **Goal:** OpenQASM 2.0 bidirectional round-trip (QuantRS2 Circuit ↔ QASM string) and a PennyLane-compatible execution backend. Both exposed via Python bindings.
  - **Design:** `circuit/src/qasm/error.rs` (~50 lines) — QasmError enum. `circuit/src/qasm/export.rs` (~400 lines) — `circuit_to_qasm` with full gate mapping table (H/X/Y/Z/S/T/CNOT/CZ/SWAP/CCX/RX/RY/RZ/U3). `circuit/src/qasm/import.rs` (~600 lines) — hand-written recursive-descent parser for OPENQASM 2.0 (no external parser crate); handles qreg/creg/standard gates/measurements/barriers; `Err(UnsupportedFeature)` for `if`/custom gate defs. `circuit/src/pennylane/wire.rs` (~250 lines) — PennyLane JSON wire format serde. `circuit/src/pennylane/device.rs` (~350 lines) — QuantrsDevice: `capabilities()`, `execute(circuit_json)`, `batch_execute`. `py/src/ecosystem.rs` (~220 lines) — `circuit_to_qasm`, `qasm_to_circuit` Python functions; `PyQuantrsDevice` class; submodule `quantrs2.ecosystem`.
  - **Files:** New `circuit/src/qasm/` (4 files, ~1080 LoC). New `circuit/src/pennylane/` (3 files, ~630 LoC). New `py/src/ecosystem.rs` (~220 lines). Modify `circuit/src/lib.rs`, `circuit/Cargo.toml`, `py/src/lib.rs`.
  - **Prerequisites:** `serde`/`serde_json` at workspace level (verify; add if missing). Circuit gate-iterator API — add if not present.
  - **Tests:** `circuit/tests/qasm_roundtrip_tests.rs` (new, ~300 lines): Bell/QFT/parameterized round-trips; unknown gate → Err; malformed → Err(Parse). `circuit/tests/pennylane_device_tests.rs` (new, ~200 lines): capabilities JSON valid; Bell ⟨Z⟩=0; probabilities sum to 1. `py/tests/test_ecosystem.py` (new, ~80 lines) smoke.
  - **Risk:** QASM angle arithmetic (π/2, negation) — support π literal + negation; return UnsupportedFeature for complex expressions. PennyLane JSON targets >= 0.39 wire format.

## Proposed follow-ups

- **Quantum hardware integration** — blocked: requires user decision on provider(s) (IBM Q / Rigetti / AWS Braket / IonQ / Quantinuum) plus credentials. Surface as per-provider sub-tickets once provider is chosen.
- **Quantum networking** — vague: surface as concrete-protocol sub-tickets (BB84, E91, teleportation chains, entanglement swapping).
- **Hybrid algorithms** — vague: surface as algorithm sub-tickets (QAOA-warm-start, VQE-natural-gradient, hybrid kernel methods).
- **Quantum compilers** — vague: surface gap items vs existing SABRE/ZX/noise-aware coverage (cross-compiler IR bridge, pulse-level compilation).
- **Enterprise security** — vague: surface as concrete sub-tickets (credential storage abstraction, audit logging, multi-tenant sandboxing).
- **Integration testing** — blocked: external cloud/hardware backends require credentials (same as hardware integration).
- **Ecosystem integration** — vague: surface as Qiskit-OpenQASM-roundtrip, Cirq-import, PennyLane-bridge sub-tickets.
