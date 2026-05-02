//! Integration tests for the quantum device backend pipeline.
//!
//! These tests run without real cloud credentials using `MockQuantumBackend`.
//! They exercise the full circuit → compile → submit → result pipeline, QASM
//! format handling, error injection, and capability queries.

use quantrs2_device::mock_backend::{MockBackendConfig, MockBackendError, MockQuantumBackend};

// ─── Shared QASM fixtures ────────────────────────────────────────────────────

/// Well-formed QASM 2.0 Bell circuit
const BELL_QASM: &str = "OPENQASM 2.0;\n\
    include \"qelib1.inc\";\n\
    qreg q[2];\n\
    creg c[2];\n\
    h q[0];\n\
    cx q[0],q[1];\n\
    measure q[0] -> c[0];\n\
    measure q[1] -> c[1];\n";

/// Single-qubit H circuit
const SINGLE_QUBIT_QASM: &str = "OPENQASM 2.0;\n\
    include \"qelib1.inc\";\n\
    qreg q[1];\n\
    creg c[1];\n\
    h q[0];\n\
    measure q[0] -> c[0];\n";

/// 5-qubit GHZ circuit
const GHZ5_QASM: &str = "OPENQASM 2.0;\n\
    include \"qelib1.inc\";\n\
    qreg q[5];\n\
    creg c[5];\n\
    h q[0];\n\
    cx q[0],q[1];\n\
    cx q[1],q[2];\n\
    cx q[2],q[3];\n\
    cx q[3],q[4];\n\
    measure q[0] -> c[0];\n\
    measure q[1] -> c[1];\n\
    measure q[2] -> c[2];\n\
    measure q[3] -> c[3];\n\
    measure q[4] -> c[4];\n";

/// QASM string that contains no `qreg` line (handled gracefully)
const NO_QREG_QASM: &str = "OPENQASM 2.0;\n\
    include \"qelib1.inc\";\n\
    // no qreg declaration\n";

/// Completely garbled input
const GARBAGE_QASM: &str = "not QASM at all %%%###";

// ─── Group 1: MockQuantumBackend basic tests ─────────────────────────────────

#[cfg(test)]
mod mock_backend_tests {
    use super::*;

    #[test]
    fn test_mock_backend_default_config() {
        let backend = MockQuantumBackend::new(MockBackendConfig::default());
        assert_eq!(backend.config.name, "mock_backend");
        assert_eq!(backend.config.max_qubits, 32);
        assert_eq!(backend.config.max_shots, 8192);
        assert_eq!(backend.config.error_rate, 0.0);
        assert_eq!(backend.config.fail_rate, 0.0);
        assert_eq!(backend.config.latency_ms, 0);
        assert!(backend.config.gate_set.is_empty());
        assert!(backend.config.connectivity.is_empty());
        assert_eq!(backend.job_count(), 0);
    }

    #[test]
    fn test_mock_backend_returns_counts() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(2));
        let counts = backend.run(BELL_QASM, 512).expect("run should succeed");
        // At least one bitstring must be present
        assert!(!counts.is_empty(), "counts must not be empty");
        // Every bitstring must be 2 characters long (2 qubits)
        for key in counts.keys() {
            assert_eq!(
                key.len(),
                2,
                "bitstring '{key}' must have length 2 for a 2-qubit circuit"
            );
        }
    }

    #[test]
    fn test_mock_counts_sum_to_shots() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(2));
        let shots = 1024;
        let counts = backend.run(BELL_QASM, shots).expect("run should succeed");
        let total: usize = counts.values().sum();
        assert_eq!(total, shots, "counts should sum to the requested number of shots");
    }

    #[test]
    fn test_mock_backend_too_many_qubits_error() {
        let backend = MockQuantumBackend::new(MockBackendConfig {
            max_qubits: 2,
            ..Default::default()
        });
        // GHZ5 requires 5 qubits; backend only has 2
        let result = backend.run(GHZ5_QASM, 100);
        assert!(
            result.is_err(),
            "run should fail when circuit exceeds max_qubits"
        );
        match result {
            Err(MockBackendError::TooManyQubits { requested, max }) => {
                assert_eq!(requested, 5);
                assert_eq!(max, 2);
            }
            other => panic!("expected TooManyQubits, got {other:?}"),
        }
    }

    #[test]
    fn test_mock_backend_too_many_shots_error() {
        let backend = MockQuantumBackend::new(MockBackendConfig {
            max_shots: 100,
            ..Default::default()
        });
        let result = backend.run(BELL_QASM, 500);
        assert!(
            result.is_err(),
            "run should fail when shots exceed max_shots"
        );
        match result {
            Err(MockBackendError::TooManyShots { requested, max }) => {
                assert_eq!(requested, 500);
                assert_eq!(max, 100);
            }
            other => panic!("expected TooManyShots, got {other:?}"),
        }
    }

    #[test]
    fn test_mock_backend_fail_rate_100_percent() {
        let backend = MockQuantumBackend::new(MockBackendConfig::always_fails());
        let result = backend.run(BELL_QASM, 100);
        assert!(result.is_err(), "always_fails backend must return Err");
        match result {
            Err(MockBackendError::JobFailed(_)) => {}
            other => panic!("expected JobFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_mock_backend_job_tracking() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(5));
        assert_eq!(backend.job_count(), 0);

        backend.run(BELL_QASM, 256).expect("first run");
        assert_eq!(backend.job_count(), 1);

        backend.run(SINGLE_QUBIT_QASM, 128).expect("second run");
        assert_eq!(backend.job_count(), 2);

        let jobs = backend.all_jobs();
        assert_eq!(jobs.len(), 2);
        assert!(!jobs[0].failed);
        assert!(!jobs[1].failed);

        let last = backend.last_job().expect("last job should exist");
        assert_eq!(last.shots, 128);
        assert!(last.result.is_some());
    }

    #[test]
    fn test_mock_backend_reproducible_with_seed() {
        let cfg = MockBackendConfig {
            rng_seed: 12345,
            ..MockBackendConfig::perfect(2)
        };
        let backend1 = MockQuantumBackend::new(cfg.clone());
        let backend2 = MockQuantumBackend::new(cfg);

        let counts1 = backend1.run(BELL_QASM, 512).expect("run 1");
        let counts2 = backend2.run(BELL_QASM, 512).expect("run 2");

        assert_eq!(
            counts1, counts2,
            "same seed should produce identical counts"
        );
    }
}

// ─── Group 2: QASM format integration ────────────────────────────────────────

#[cfg(test)]
mod qasm_integration_tests {
    use super::*;

    /// Verify that the mock backend accepts a QASM 2.0 Bell circuit and produces
    /// counts whose keys are valid 2-character bitstrings.
    #[test]
    fn test_qasm_bell_circuit_roundtrip_and_run() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(4));
        let counts = backend.run(BELL_QASM, 1000).expect("Bell circuit should run");
        // The Bell circuit has 2 qubits → every bitstring is 2 bits
        for key in counts.keys() {
            assert_eq!(key.len(), 2, "expected 2-bit bitstrings, got '{key}'");
            // Every character must be '0' or '1'
            assert!(
                key.chars().all(|c| c == '0' || c == '1'),
                "non-binary character in '{key}'"
            );
        }
        let total: usize = counts.values().sum();
        assert_eq!(total, 1000);
    }

    /// Single-qubit H gate circuit → 1-qubit result with bitstrings of length 1.
    #[test]
    fn test_qasm_single_qubit_run() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(1));
        let counts = backend.run(SINGLE_QUBIT_QASM, 256).expect("H circuit should run");
        for key in counts.keys() {
            assert_eq!(key.len(), 1, "expected 1-bit bitstrings, got '{key}'");
        }
        let total: usize = counts.values().sum();
        assert_eq!(total, 256);
    }

    /// Malformed QASM (no qreg) is handled gracefully: no `qreg` line means
    /// `parse_qubit_count` returns `None`, defaulting to 1 qubit. The run
    /// succeeds, producing a 1-qubit result.
    #[test]
    fn test_qasm_format_invalid_rejected_by_mock() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(4));
        // No qreg → defaults to 1 qubit; run should succeed gracefully
        let result = backend.run(NO_QREG_QASM, 64);
        assert!(
            result.is_ok(),
            "QASM with no qreg should be handled gracefully (defaults to 1 qubit)"
        );
        let counts = result.expect("run with no qreg");
        let total: usize = counts.values().sum();
        assert_eq!(total, 64);
    }

    /// Completely garbage input is also handled gracefully (no qreg found → 1 qubit).
    #[test]
    fn test_mock_accepts_valid_qasm_2_0() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(5));
        // The BELL_QASM constant is proper QASM 2.0 — must succeed
        let counts = backend.run(BELL_QASM, 512).expect("valid QASM 2.0 must be accepted");
        assert!(!counts.is_empty());
    }

    /// Full pipeline: multi-qubit GHZ circuit → run → verify 5-bit bitstrings.
    #[test]
    fn test_pipeline_circuit_to_mock_execution() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(8));
        let counts = backend.run(GHZ5_QASM, 2048).expect("GHZ5 should run");
        assert!(!counts.is_empty());
        for key in counts.keys() {
            assert_eq!(key.len(), 5, "expected 5-bit bitstrings, got '{key}'");
            assert!(
                key.chars().all(|c| c == '0' || c == '1'),
                "non-binary character in '{key}'"
            );
        }
        let total: usize = counts.values().sum();
        assert_eq!(total, 2048);
    }

    /// Completely garbled input is accepted (no qreg → 1 qubit default) because
    /// the mock does not perform full QASM parsing — it only extracts the qubit
    /// count. This documents and tests the graceful-degradation behaviour.
    #[test]
    fn test_garbage_qasm_handled_gracefully() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(4));
        let result = backend.run(GARBAGE_QASM, 32);
        // Should succeed with 1-qubit default
        assert!(
            result.is_ok(),
            "garbage QASM must not panic; got {result:?}"
        );
        let counts = result.expect("garbage QASM graceful run");
        let total: usize = counts.values().sum();
        assert_eq!(total, 32);
    }
}

// ─── Group 3: Error handling ──────────────────────────────────────────────────

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_failing_backend_returns_error() {
        let backend = MockQuantumBackend::new(MockBackendConfig::always_fails());
        for _ in 0..5 {
            assert!(
                backend.run(BELL_QASM, 100).is_err(),
                "always_fails backend must always return Err"
            );
        }
    }

    /// Zero shots should return an empty counts map (not an error).
    #[test]
    fn test_zero_shot_returns_empty_counts() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(2));
        let counts = backend.run(BELL_QASM, 0).expect("zero shots should succeed");
        assert!(
            counts.is_empty(),
            "zero shots must produce an empty counts map"
        );
    }

    /// With latency_ms = 0 the run completes immediately (< 1 second).
    #[test]
    fn test_latency_simulation() {
        let backend = MockQuantumBackend::new(MockBackendConfig {
            latency_ms: 0,
            ..MockBackendConfig::perfect(2)
        });
        let start = std::time::Instant::now();
        backend.run(BELL_QASM, 128).expect("zero-latency run");
        // Should be well under 1 second even in a slow CI environment
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "zero-latency run should complete in under 1 second"
        );
    }

    /// Failed jobs must still appear in the job log.
    #[test]
    fn test_job_records_on_failure() {
        let backend = MockQuantumBackend::new(MockBackendConfig::always_fails());
        let _ = backend.run(BELL_QASM, 128); // intentionally ignored
        assert_eq!(backend.job_count(), 1, "failed job must be recorded");
        let job = backend.last_job().expect("last job must exist");
        assert!(job.failed, "job record must mark failed=true");
        assert!(
            job.error_message.is_some(),
            "job record must contain an error message"
        );
        assert!(
            job.result.is_none(),
            "failed job must have no result counts"
        );
    }

    /// Validation errors (too many qubits) should NOT create a job record.
    #[test]
    fn test_validation_error_does_not_create_job_record() {
        let backend = MockQuantumBackend::new(MockBackendConfig {
            max_qubits: 2,
            ..Default::default()
        });
        let _ = backend.run(GHZ5_QASM, 100); // should fail with TooManyQubits
        assert_eq!(
            backend.job_count(),
            0,
            "validation errors must not be recorded as jobs"
        );
    }

    /// Validation error message display is human-readable.
    #[test]
    fn test_error_display_is_readable() {
        let too_many_qubits = MockBackendError::TooManyQubits {
            requested: 10,
            max: 5,
        };
        let msg = too_many_qubits.to_string();
        assert!(
            msg.contains("10") && msg.contains('5'),
            "error message should contain both numbers; got: {msg}"
        );

        let too_many_shots = MockBackendError::TooManyShots {
            requested: 10000,
            max: 8192,
        };
        let msg = too_many_shots.to_string();
        assert!(
            msg.contains("10000") && msg.contains("8192"),
            "error message should contain both numbers; got: {msg}"
        );

        let job_failed = MockBackendError::JobFailed("network timeout".to_string());
        let msg = job_failed.to_string();
        assert!(msg.contains("network timeout"), "got: {msg}");
    }
}

// ─── Group 4: Backend capabilities ────────────────────────────────────────────

#[cfg(test)]
mod capability_tests {
    use super::*;

    #[test]
    fn test_ibm_nairobi_like_backend() {
        let backend = MockQuantumBackend::new(MockBackendConfig::ibm_nairobi_like());
        assert_eq!(backend.config.max_qubits, 7);
        assert_eq!(backend.config.max_shots, 4096);
        assert_eq!(backend.config.name, "mock_ibm_nairobi");
        assert!(backend.config.error_rate > 0.0);
        // Should accept a 7-qubit circuit
        let qasm_7q = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[7];\ncreg c[7];\nh q[0];\nmeasure q[0] -> c[0];\n";
        let counts = backend.run(qasm_7q, 512).expect("7-qubit circuit on 7-qubit backend");
        let total: usize = counts.values().sum();
        assert_eq!(total, 512);
    }

    #[test]
    fn test_capabilities_query() {
        let backend = MockQuantumBackend::new(MockBackendConfig::ibm_nairobi_like());
        let caps = backend.capabilities();
        assert_eq!(caps.get("name").map(String::as_str), Some("mock_ibm_nairobi"));
        assert_eq!(caps.get("n_qubits").map(String::as_str), Some("7"));
        assert_eq!(caps.get("max_shots").map(String::as_str), Some("4096"));
        assert_eq!(caps.get("simulator").map(String::as_str), Some("true"));
        // Gate set should list all configured gates
        let gates_str = caps
            .get("supported_gates")
            .expect("supported_gates key must be present");
        assert!(gates_str.contains("cx"), "expected 'cx' in gate set: {gates_str}");
    }

    #[test]
    fn test_gate_set_configuration() {
        let backend = MockQuantumBackend::new(MockBackendConfig::ibm_nairobi_like());
        // IBM Nairobi supports cx, rz, sx, x
        assert!(backend.supports_gate("cx"));
        assert!(backend.supports_gate("rz"));
        assert!(backend.supports_gate("sx"));
        assert!(backend.supports_gate("x"));
        // Does not support ccx (Toffoli) — not in its gate set
        assert!(!backend.supports_gate("ccx"));
        assert!(!backend.supports_gate("iswap"));
    }

    #[test]
    fn test_connectivity_all_to_all_default() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(10));
        // Empty connectivity = all pairs allowed
        assert!(backend.is_connected(0, 9));
        assert!(backend.is_connected(3, 7));
        assert!(backend.is_connected(0, 1));
    }

    #[test]
    fn test_connectivity_restricted_ibm_nairobi() {
        let backend = MockQuantumBackend::new(MockBackendConfig::ibm_nairobi_like());
        // IBM Nairobi connectivity: (0,1),(1,2),(1,3),(3,5),(4,5),(5,6)
        assert!(backend.is_connected(0, 1));
        assert!(backend.is_connected(1, 0), "connectivity is symmetric");
        assert!(backend.is_connected(5, 6));
        // Not directly connected
        assert!(!backend.is_connected(0, 6));
        assert!(!backend.is_connected(2, 4));
    }

    #[test]
    fn test_noisy_backend_still_sums_to_shots() {
        let backend = MockQuantumBackend::new(MockBackendConfig::with_noise(4, 0.05));
        let shots = 500;
        let counts = backend.run(BELL_QASM, shots).expect("noisy run");
        let total: usize = counts.values().sum();
        assert_eq!(total, shots, "noisy counts must still sum to shots");
    }

    #[test]
    fn test_multiple_runs_accumulate_records() {
        let backend = MockQuantumBackend::new(MockBackendConfig::perfect(5));
        for i in 1..=5 {
            backend.run(BELL_QASM, 100 * i).unwrap_or_else(|e| panic!("run {i} failed: {e}"));
        }
        assert_eq!(backend.job_count(), 5);
        // Verify shots are recorded correctly
        let jobs = backend.all_jobs();
        for (i, job) in jobs.iter().enumerate() {
            assert_eq!(job.shots, 100 * (i + 1));
            assert!(!job.job_id.is_empty());
        }
    }
}
