//! Benchmarks for stabilizer (Clifford) simulator
//!
//! This benchmark suite demonstrates the efficiency of the stabilizer formalism
//! for simulating large Clifford circuits that would be intractable for
//! state-vector simulators.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator};

/// Benchmark single-qubit Clifford gates
fn bench_single_qubit_clifford_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_single_qubit");

    for num_qubits in [10, 100, 1000, 10000].iter() {
        let target = 0;

        // Hadamard gate
        group.bench_with_input(
            BenchmarkId::new("hadamard", num_qubits),
            num_qubits,
            |b, &n| {
                let mut sim = StabilizerSimulator::new(n);
                b.iter(|| {
                    sim.apply_gate(StabilizerGate::H(black_box(target)))
                        .unwrap();
                });
            },
        );

        // S gate
        group.bench_with_input(BenchmarkId::new("s", num_qubits), num_qubits, |b, &n| {
            let mut sim = StabilizerSimulator::new(n);
            b.iter(|| {
                sim.apply_gate(StabilizerGate::S(black_box(target)))
                    .unwrap();
            });
        });

        // S† gate
        group.bench_with_input(
            BenchmarkId::new("s_dag", num_qubits),
            num_qubits,
            |b, &n| {
                let mut sim = StabilizerSimulator::new(n);
                b.iter(|| {
                    sim.apply_gate(StabilizerGate::SDag(black_box(target)))
                        .unwrap();
                });
            },
        );

        // √X gate
        group.bench_with_input(
            BenchmarkId::new("sqrt_x", num_qubits),
            num_qubits,
            |b, &n| {
                let mut sim = StabilizerSimulator::new(n);
                b.iter(|| {
                    sim.apply_gate(StabilizerGate::SqrtX(black_box(target)))
                        .unwrap();
                });
            },
        );

        // √Y gate
        group.bench_with_input(
            BenchmarkId::new("sqrt_y", num_qubits),
            num_qubits,
            |b, &n| {
                let mut sim = StabilizerSimulator::new(n);
                b.iter(|| {
                    sim.apply_gate(StabilizerGate::SqrtY(black_box(target)))
                        .unwrap();
                });
            },
        );

        // Pauli-X gate
        group.bench_with_input(
            BenchmarkId::new("pauli_x", num_qubits),
            num_qubits,
            |b, &n| {
                let mut sim = StabilizerSimulator::new(n);
                b.iter(|| {
                    sim.apply_gate(StabilizerGate::X(black_box(target)))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark two-qubit Clifford gates
fn bench_two_qubit_clifford_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_two_qubit");

    for num_qubits in [10, 100, 1000, 10000].iter() {
        let control = 0;
        let target = 1;

        // CNOT gate
        group.bench_with_input(BenchmarkId::new("cnot", num_qubits), num_qubits, |b, &n| {
            let mut sim = StabilizerSimulator::new(n);
            b.iter(|| {
                sim.apply_gate(StabilizerGate::CNOT(black_box(control), black_box(target)))
                    .unwrap();
            });
        });

        // CZ gate
        group.bench_with_input(BenchmarkId::new("cz", num_qubits), num_qubits, |b, &n| {
            let mut sim = StabilizerSimulator::new(n);
            b.iter(|| {
                sim.apply_gate(StabilizerGate::CZ(black_box(control), black_box(target)))
                    .unwrap();
            });
        });

        // CY gate
        group.bench_with_input(BenchmarkId::new("cy", num_qubits), num_qubits, |b, &n| {
            let mut sim = StabilizerSimulator::new(n);
            b.iter(|| {
                sim.apply_gate(StabilizerGate::CY(black_box(control), black_box(target)))
                    .unwrap();
            });
        });

        // SWAP gate
        group.bench_with_input(BenchmarkId::new("swap", num_qubits), num_qubits, |b, &n| {
            let mut sim = StabilizerSimulator::new(n);
            b.iter(|| {
                sim.apply_gate(StabilizerGate::SWAP(black_box(control), black_box(target)))
                    .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark stabilizer state creation and initialization
fn bench_stabilizer_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_creation");

    for num_qubits in [10, 100, 1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let _sim = StabilizerSimulator::new(black_box(n));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark measurement operations
fn bench_stabilizer_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_measurement");

    for num_qubits in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                let mut sim = StabilizerSimulator::new(n);
                // Apply Hadamard to create superposition
                sim.apply_gate(StabilizerGate::H(0)).unwrap();

                b.iter(|| {
                    let mut test_sim = sim.clone();
                    black_box(test_sim.measure(black_box(0)).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GHZ state preparation (scaling test)
fn bench_ghz_state_preparation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz_state_preparation");

    for num_qubits in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let mut sim = StabilizerSimulator::new(n);
                    // Apply Hadamard to first qubit
                    sim.apply_gate(StabilizerGate::H(0)).unwrap();

                    // Apply CNOT chain to create GHZ state
                    for i in 0..n - 1 {
                        sim.apply_gate(StabilizerGate::CNOT(i, i + 1)).unwrap();
                    }

                    black_box(sim);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark surface code preparation (realistic workload)
fn bench_surface_code_preparation(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_code");

    // Surface code d=3, 5, 7, 9 (9, 25, 49, 81 qubits)
    for distance in [3, 5, 7, 9].iter() {
        let num_qubits = distance * distance;

        group.bench_with_input(
            BenchmarkId::new("distance", distance),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    let mut sim = StabilizerSimulator::new(n);

                    // Initialize data qubits in |+⟩ state
                    for i in 0..n {
                        sim.apply_gate(StabilizerGate::H(i)).unwrap();
                    }

                    // Apply stabilizer measurements (simplified)
                    // In a real surface code, this would be more complex
                    for i in 0..n - 1 {
                        if i % 2 == 0 {
                            sim.apply_gate(StabilizerGate::CZ(i, i + 1)).unwrap();
                        }
                    }

                    black_box(sim);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark deep Clifford circuits
fn bench_deep_clifford_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_clifford_circuit");

    // Fixed number of qubits, varying depth
    let num_qubits = 100;

    for depth in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("depth", depth), depth, |b, &d| {
            b.iter(|| {
                let mut sim = StabilizerSimulator::new(num_qubits);

                for layer in 0..d {
                    // Alternate between single-qubit and two-qubit layers
                    if layer % 2 == 0 {
                        // Single-qubit layer
                        for q in 0..num_qubits {
                            let gate = match layer % 3 {
                                0 => StabilizerGate::H(q),
                                1 => StabilizerGate::S(q),
                                _ => StabilizerGate::SqrtX(q),
                            };
                            sim.apply_gate(gate).unwrap();
                        }
                    } else {
                        // Two-qubit layer (nearest neighbor)
                        for q in (layer % 2..(num_qubits - 1)).step_by(2) {
                            sim.apply_gate(StabilizerGate::CNOT(q, q + 1)).unwrap();
                        }
                    }
                }

                black_box(sim);
            });
        });
    }

    group.finish();
}

/// Benchmark random Clifford circuits
fn bench_random_clifford_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_clifford");

    for num_qubits in [10, 100, 1000].iter() {
        let depth = 100; // Fixed depth

        group.bench_with_input(
            BenchmarkId::new("qubits", num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let mut sim = StabilizerSimulator::new(n);

                    for layer in 0..depth {
                        // Random single-qubit gates
                        for q in 0..n {
                            let gate_type = (q + layer) % 6;
                            let gate = match gate_type {
                                0 => StabilizerGate::H(q),
                                1 => StabilizerGate::S(q),
                                2 => StabilizerGate::SDag(q),
                                3 => StabilizerGate::SqrtX(q),
                                4 => StabilizerGate::SqrtY(q),
                                _ => StabilizerGate::X(q),
                            };
                            sim.apply_gate(gate).unwrap();
                        }

                        // Random two-qubit gates
                        for q in (layer % 2..(n - 1)).step_by(2) {
                            let gate_type = (q + layer) % 3;
                            let gate = match gate_type {
                                0 => StabilizerGate::CNOT(q, q + 1),
                                1 => StabilizerGate::CZ(q, q + 1),
                                _ => StabilizerGate::CY(q, q + 1),
                            };
                            sim.apply_gate(gate).unwrap();
                        }
                    }

                    black_box(sim);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark massive qubit counts (Stim advantage)
fn bench_massive_qubit_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("massive_qubits");
    group.sample_size(10); // Reduce sample size for very large circuits

    for num_qubits in [10000, 100000, 1000000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let mut sim = StabilizerSimulator::new(n);

                    // Apply Hadamard to every 100th qubit
                    for q in (0..n).step_by(100) {
                        sim.apply_gate(StabilizerGate::H(q)).unwrap();
                    }

                    // Apply CNOT to adjacent pairs
                    for q in (0..n - 1).step_by(1000) {
                        sim.apply_gate(StabilizerGate::CNOT(q, q + 1)).unwrap();
                    }

                    black_box(sim);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_qubit_clifford_gates,
    bench_two_qubit_clifford_gates,
    bench_stabilizer_creation,
    bench_stabilizer_measurement,
    bench_ghz_state_preparation,
    bench_surface_code_preparation,
    bench_deep_clifford_circuit,
    bench_random_clifford_circuit,
    bench_massive_qubit_counts,
);

criterion_main!(benches);
