# quantrs2-py Roadmap

## v0.1.3 (2026-03-27)
- 108 public API items; aligned with QuantRS2 v0.1.3 workspace

## Future Enhancements

- [x] Quantum hardware integration: Direct integration with more providers (implemented 2026-04-29)
  - **Goal:** Unified job tracking API across IBM/AWS/Azure providers; provider capability discovery module; mock backend for full-pipeline integration tests without real credentials. IBM/AWS/Azure REST clients already exist (feature-gated).
  - **Design:** `device/src/mock_backend.rs` — MockQuantumBackend (configurable latency/error/fail rates, deterministic result generation). `device/tests/integration_tests.rs` — 20+ tests of circuit→compile→submit→result pipeline without real cloud connections. Provider capability discovery: query supported gates/qubits per backend.
  - **Files:** New `device/src/mock_backend.rs`. New `device/tests/integration_tests.rs`. Modify `device/src/lib.rs`.
  - **Prerequisites:** None — mock-based, no real credentials needed.
  - **Tests:** All integration tests pass without real cloud credentials.
  - **Risk:** Format adapters may expose gaps in IBM/AWS/Azure conversion — fix any found.
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
- [x] Hybrid algorithms: More quantum-classical hybrid approaches (implemented 2026-04-29)
  - **Goal:** (a) QAOA warm-start using spectral relaxation for MaxCut initialization. (b) VQE with quantum natural gradient via proper QFIM — fix stubs in autodiff.rs (compute_fisher_element returns random values; solve_linear_system just returns rhs). (c) Hybrid kernel SVM enhancement.
  - **Design:** `ml/src/qaoa_warm_start.rs` — WarmStartQAOAOptimizer with graph Laplacian Fiedler-vector initialization. `ml/src/vqe_natural_gradient.rs` — QFIMEstimator using 4-point parameter-shift rule (F_ij = (E(++)+E(--)-E(+-)-E(-+))/4). Gaussian elimination solve for natural gradient step. Fix autodiff.rs stubs; wire QuantumNaturalGradient branch in optimization.rs.
  - **Files:** New `ml/src/qaoa_warm_start.rs`, new `ml/src/vqe_natural_gradient.rs`. Modify `ml/src/autodiff.rs`, `ml/src/optimization.rs`, `ml/src/lib.rs`.
  - **Prerequisites:** scirs2-linalg (already in workspace).
  - **Tests:** QAOA warm-start: 4-vertex MaxCut, warm angles give lower initial energy than random. VQE natural gradient: 2-qubit Hamiltonian converges in ≤ 20 steps.
  - **Risk:** QFIM needs O(n²) evaluations — cap n=20 in tests.
- [x] Advanced visualization: 3D quantum state visualization
- [x] Quantum compilers: Advanced circuit compilation (implemented 2026-04-29)
  - **Goal:** (a) Solovay-Kitaev algorithm — approximate arbitrary SU(2) with {H,T,T†} sequences. (b) Template matching pass — reduce gate count via precomputed equivalence patterns (HH=I, XX=I, TT=S, CNOT·CNOT=I, etc.).
  - **Design:** `circuit/src/solovay_kitaev.rs` (~500 lines) — SU2 struct (2x2 complex), BasicApproximation table (depth-9 sequences), SOKDecomposer::decompose(u, ε) using balanced commutator recursion per Dawson-Nielsen 2005. `circuit/src/template_matching.rs` (~400 lines) — TemplateLibrary with 20+ patterns, DAG-based matching, TemplateMatchingPass::run(circuit) → Circuit.
  - **Files:** New `circuit/src/solovay_kitaev.rs`, new `circuit/src/template_matching.rs`. Modify `circuit/src/lib.rs`.
  - **Prerequisites:** scirs2-linalg (SU2 operations).
  - **Tests:** SK: decompose RZ(0.3) to ε=0.01, verify ‖approx - exact‖ ≤ 0.01. Template: Bell circuit with redundant H reduced by ≥ 2 gates.
  - **Risk:** SK precomputed table (~7000 elements) built at first use via once_cell. Depth capped at 5 recursion levels for tests.
- [x] Enterprise security: Enhanced security features (implemented 2026-04-29)
  - **Goal:** Complete `device/src/security/` (directory exists but is empty). Implement: (a) structured JSON audit logging. (b) credential vault trait + env-var + file-based implementations. (c) token bucket rate limiter for cloud API calls.
  - **Design:** `device/src/security/audit.rs` — AuditEvent {id, timestamp, operation: OperationType, backend_id, circuit_hash, user_id, success, duration_ms, error}; AuditLogger trait; FileAuditLogger + InMemoryAuditLogger. `device/src/security/credentials.rs` — SecretString (zeroed on drop); CredentialProvider trait; EnvVarCredentialProvider + FileCredentialProvider (checks 0o600 mode). `device/src/security/rate_limit.rs` — TokenBucket + RateLimiter (per-provider buckets). `device/src/security/mod.rs` — re-exports.
  - **Files:** New `device/src/security/audit.rs`, `credentials.rs`, `rate_limit.rs`, `mod.rs`.
  - **Prerequisites:** Check if `zeroize` is in workspace; add if missing.
  - **Tests:** AuditLogger: write 100 events, verify JSON roundtrip. CredentialProvider: env-var round-trip. RateLimiter: exhaust tokens, verify refill after delay.
  - **Risk:** File permission checks are Unix-only (#[cfg(unix)]). Sandboxing deferred.
- [x] Scalability testing: Large-scale simulation validation (implemented 2026-04-27)
  - **Goal:** Scalability test + bench suite for state-vector sim, MPS sim, and QUBO sampling at real-world problem sizes. Smoke tests < 60s; heavy benches behind `--ignored`.
  - **Design:** sim/tests/scalability_smoke.rs (5 smoke tests: 15q SV, 18q QFT, 20q MPS GHZ, 25q MPS random, 30q stabilizer), sim/benches/large_scale_simulation.rs (criterion), tytan/tests/sampler_scalability_smoke.rs (50/100/200-var QUBO smokes), tytan/benches/sampler_scalability.rs (criterion), py/tests/test_scalability.py.
  - **Files:** 6 new files across sim/, tytan/, py/tests/.
  - **Tests:** nextest smoke tests + bench compile-check.
  - **Risk:** MPS bond-dim explosion on deep random circuits — fallback to depth 5.
- [x] Integration testing: Comprehensive external system testing (implemented 2026-04-29)
  - **Goal:** Mock backend for credential-free full-pipeline integration tests. Exercises IBM/AWS/Azure format adapters, error handling, and job tracking without real cloud.
  - **Design:** `device/src/mock_backend.rs` — MockQuantumBackend {config: MockBackendConfig {latency_ms, error_rate, fail_rate, max_qubits, gate_set, connectivity}, job_records: Arc<Mutex<Vec<MockJobRecord>>>, rng_seed}. Generates deterministic measurement results from circuit structure. `device/tests/integration_tests.rs` — 20+ tests: pipeline, format adapters, timeout, failure injection.
  - **Files:** New `device/src/mock_backend.rs`. New `device/tests/integration_tests.rs`. Modify `device/src/lib.rs`.
  - **Prerequisites:** None.
  - **Tests:** All 20+ integration tests pass without real credentials.
  - **Risk:** IBM/AWS/Azure format adapter gaps may surface — fix in-place.
- [x] Performance optimization: Further SIMD and GPU optimizations (planned 2026-04-27)
  - **Goal:** SIMD-accelerated single-qubit-gate kernels in `sim/` crate. Target ≥ 2× over scalar for H/X/Y/Z/S/T/RX/RY/RZ on q=0. Pure-Rust via `wide` crate (no nightly std::simd).
  - **Design:** sim/src/state_vector_simd.rs (~600 lines, 9 gate kernels), dispatch in sim/src/state_vector.rs, sim/benches/simd_state_vector.rs (criterion scalar vs SIMD), sim/tests/state_vector_simd_correctness.rs (L2 < 1e-12 correctness suite).
  - **Files:** 3 new files in sim/; modify sim/src/state_vector.rs, sim/Cargo.toml (add `wide`).
  - **Tests:** correctness suite for every gate at n ∈ {3,5,7}, each target qubit.
  - **Risk:** Floating-point non-determinism across SIMD lanes — tolerance 1e-12 in tests.
- [x] Ecosystem integration: Enhanced quantum software stack compatibility (implemented 2026-04-28)
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
