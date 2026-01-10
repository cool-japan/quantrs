//! Benchmarks for Qulacs-inspired quantum backend
//!
//! This benchmark suite measures the performance of the Qulacs backend
//! for various quantum operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_sim::prelude::*;

/// Benchmark single-qubit gates
fn bench_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");

    for num_qubits in [5, 10, 15, 20].iter() {
        let target = 0; // Always apply to qubit 0

        // Hadamard gate
        group.bench_with_input(
            BenchmarkId::new("hadamard", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    qulacs_gates::hadamard(black_box(&mut state), black_box(target)).unwrap();
                });
            },
        );

        // Pauli-X gate
        group.bench_with_input(
            BenchmarkId::new("pauli_x", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    qulacs_gates::pauli_x(black_box(&mut state), black_box(target)).unwrap();
                });
            },
        );

        // Pauli-Y gate
        group.bench_with_input(
            BenchmarkId::new("pauli_y", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    qulacs_gates::pauli_y(black_box(&mut state), black_box(target)).unwrap();
                });
            },
        );

        // Pauli-Z gate
        group.bench_with_input(
            BenchmarkId::new("pauli_z", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    qulacs_gates::pauli_z(black_box(&mut state), black_box(target)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark rotation gates
fn bench_rotation_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_gates");

    for num_qubits in [5, 10, 15, 20].iter() {
        let target = 0;
        let angle = std::f64::consts::PI / 4.0;

        // RX gate
        group.bench_with_input(BenchmarkId::new("rx", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            b.iter(|| {
                qulacs_gates::rx(black_box(&mut state), black_box(target), black_box(angle))
                    .unwrap();
            });
        });

        // RY gate
        group.bench_with_input(BenchmarkId::new("ry", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            b.iter(|| {
                qulacs_gates::ry(black_box(&mut state), black_box(target), black_box(angle))
                    .unwrap();
            });
        });

        // RZ gate
        group.bench_with_input(BenchmarkId::new("rz", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            b.iter(|| {
                qulacs_gates::rz(black_box(&mut state), black_box(target), black_box(angle))
                    .unwrap();
            });
        });

        // U3 gate (universal single-qubit)
        group.bench_with_input(BenchmarkId::new("u3", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            let theta = std::f64::consts::PI / 3.0;
            let phi = std::f64::consts::PI / 6.0;
            let lambda = std::f64::consts::PI / 4.0;
            b.iter(|| {
                qulacs_gates::u3(
                    black_box(&mut state),
                    black_box(target),
                    black_box(theta),
                    black_box(phi),
                    black_box(lambda),
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark two-qubit gates
fn bench_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gates");

    for num_qubits in [5, 10, 15, 20].iter() {
        let control = 0;
        let target = 1;

        // CNOT gate
        group.bench_with_input(BenchmarkId::new("cnot", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            b.iter(|| {
                qulacs_gates::cnot(black_box(&mut state), black_box(control), black_box(target))
                    .unwrap();
            });
        });

        // CZ gate
        group.bench_with_input(BenchmarkId::new("cz", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            b.iter(|| {
                qulacs_gates::cz(black_box(&mut state), black_box(control), black_box(target))
                    .unwrap();
            });
        });

        // SWAP gate
        group.bench_with_input(BenchmarkId::new("swap", num_qubits), num_qubits, |b, &n| {
            let mut state = QulacsStateVector::new(n).unwrap();
            b.iter(|| {
                qulacs_gates::swap(black_box(&mut state), black_box(control), black_box(target))
                    .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark three-qubit gates
fn bench_three_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("three_qubit_gates");

    for num_qubits in [5, 10, 15].iter() {
        let control1 = 0;
        let control2 = 1;
        let target = 2;

        // Toffoli gate
        group.bench_with_input(
            BenchmarkId::new("toffoli", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    qulacs_gates::toffoli(
                        black_box(&mut state),
                        black_box(control1),
                        black_box(control2),
                        black_box(target),
                    )
                    .unwrap();
                });
            },
        );

        // Fredkin gate
        group.bench_with_input(
            BenchmarkId::new("fredkin", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    qulacs_gates::fredkin(
                        black_box(&mut state),
                        black_box(control1),
                        black_box(target),
                        black_box(control2),
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark state vector operations
fn bench_state_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_operations");

    for num_qubits in [5, 10, 15, 20].iter() {
        // State creation
        group.bench_with_input(
            BenchmarkId::new("creation", num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let _state = QulacsStateVector::new(black_box(n)).unwrap();
                });
            },
        );

        // Norm squared calculation
        group.bench_with_input(
            BenchmarkId::new("norm_squared", num_qubits),
            num_qubits,
            |b, &n| {
                let state = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    black_box(state.norm_squared());
                });
            },
        );

        // Inner product
        group.bench_with_input(
            BenchmarkId::new("inner_product", num_qubits),
            num_qubits,
            |b, &n| {
                let state1 = QulacsStateVector::new(n).unwrap();
                let state2 = QulacsStateVector::new(n).unwrap();
                b.iter(|| {
                    black_box(state1.inner_product(&state2).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark measurement operations
fn bench_measurement_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurement_operations");

    for num_qubits in [5, 10, 15].iter() {
        let target = 0;

        // Probability calculation
        group.bench_with_input(
            BenchmarkId::new("probability", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                qulacs_gates::hadamard(&mut state, target).unwrap();
                b.iter(|| {
                    black_box(state.probability_one(black_box(target)).unwrap());
                });
            },
        );

        // Single measurement (with collapse)
        group.bench_with_input(
            BenchmarkId::new("measure", num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let mut state = QulacsStateVector::new(n).unwrap();
                    qulacs_gates::hadamard(&mut state, target).unwrap();
                    black_box(state.measure(black_box(target)).unwrap());
                });
            },
        );

        // Sampling (100 shots)
        group.bench_with_input(
            BenchmarkId::new("sample_100", num_qubits),
            num_qubits,
            |b, &n| {
                let mut state = QulacsStateVector::new(n).unwrap();
                qulacs_gates::hadamard(&mut state, target).unwrap();
                b.iter(|| {
                    black_box(state.sample(black_box(100)).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Bell state preparation
fn bench_bell_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("bell_state");

    for num_qubits in [2, 5, 10, 15].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &n| {
                b.iter(|| {
                    let mut state = QulacsStateVector::new(n).unwrap();
                    qulacs_gates::hadamard(&mut state, 0).unwrap();
                    qulacs_gates::cnot(&mut state, 0, 1).unwrap();
                    black_box(state);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark quantum circuit execution (realistic workload)
fn bench_circuit_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_execution");

    // Small circuit: 5 qubits, 10 gates
    group.bench_function("5q_10gates", |b| {
        b.iter(|| {
            let mut state = QulacsStateVector::new(5).unwrap();
            qulacs_gates::hadamard(&mut state, 0).unwrap();
            qulacs_gates::cnot(&mut state, 0, 1).unwrap();
            qulacs_gates::ry(&mut state, 2, std::f64::consts::PI / 4.0).unwrap();
            qulacs_gates::cnot(&mut state, 1, 2).unwrap();
            qulacs_gates::rz(&mut state, 3, std::f64::consts::PI / 3.0).unwrap();
            qulacs_gates::cnot(&mut state, 2, 3).unwrap();
            qulacs_gates::rx(&mut state, 4, std::f64::consts::PI / 6.0).unwrap();
            qulacs_gates::cnot(&mut state, 3, 4).unwrap();
            qulacs_gates::hadamard(&mut state, 4).unwrap();
            qulacs_gates::cnot(&mut state, 4, 0).unwrap();
            black_box(state);
        });
    });

    // Medium circuit: 10 qubits, 20 gates
    group.bench_function("10q_20gates", |b| {
        b.iter(|| {
            let mut state = QulacsStateVector::new(10).unwrap();
            for i in 0..10 {
                qulacs_gates::hadamard(&mut state, i).unwrap();
            }
            for i in 0..9 {
                qulacs_gates::cnot(&mut state, i, i + 1).unwrap();
            }
            for i in 0..10 {
                qulacs_gates::rz(&mut state, i, std::f64::consts::PI / (i as f64 + 1.0)).unwrap();
            }
            black_box(state);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_qubit_gates,
    bench_rotation_gates,
    bench_two_qubit_gates,
    bench_three_qubit_gates,
    bench_state_operations,
    bench_measurement_operations,
    bench_bell_state,
    bench_circuit_execution,
);

criterion_main!(benches);
