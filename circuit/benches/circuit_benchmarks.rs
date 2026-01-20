#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
// Circuit Benchmarks for QuantRS2-Circuit
//
// This benchmark suite measures performance of core circuit operations
// to help identify optimization opportunities and track performance regressions.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantrs2_circuit::builder::Circuit;
use quantrs2_circuit::optimization::CircuitOptimizer2;

/// Benchmark circuit construction with varying qubit counts
fn benchmark_circuit_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_construction");

    group.bench_function("2_qubits", |b| {
        b.iter(|| {
            let circuit = Circuit::<2>::new();
            black_box(circuit)
        });
    });

    group.bench_function("5_qubits", |b| {
        b.iter(|| {
            let circuit = Circuit::<5>::new();
            black_box(circuit)
        });
    });

    group.bench_function("10_qubits", |b| {
        b.iter(|| {
            let circuit = Circuit::<10>::new();
            black_box(circuit)
        });
    });

    group.bench_function("20_qubits", |b| {
        b.iter(|| {
            let circuit = Circuit::<20>::new();
            black_box(circuit)
        });
    });

    group.finish();
}

/// Benchmark single-qubit gate application
fn benchmark_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");

    group.bench_function("hadamard", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            for i in 0..5 {
                circuit.h(black_box(i));
            }
            black_box(circuit)
        });
    });

    group.bench_function("pauli_x", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            for i in 0..5 {
                circuit.x(black_box(i));
            }
            black_box(circuit)
        });
    });

    group.bench_function("rotation_rx", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            for i in 0..5 {
                circuit.rx(black_box(i), black_box(std::f64::consts::PI / 4.0));
            }
            black_box(circuit)
        });
    });

    group.finish();
}

/// Benchmark two-qubit gate application
fn benchmark_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gates");

    group.bench_function("cnot", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            for i in 0..4 {
                circuit.cnot(black_box(i), black_box(i + 1));
            }
            black_box(circuit)
        });
    });

    group.bench_function("cz", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            for i in 0..4 {
                circuit.cz(black_box(i), black_box(i + 1));
            }
            black_box(circuit)
        });
    });

    group.finish();
}

/// Benchmark circuit optimization
fn benchmark_circuit_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_optimization");

    group.bench_function("simple_optimization", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            // Create a circuit with some redundancy
            circuit.h(0);
            circuit.h(0); // Double Hadamard = Identity
            circuit.x(1);
            circuit.x(1); // Double X = Identity
            circuit.cnot(0, 1);
            circuit.cnot(0, 1); // Double CNOT = Identity

            let optimizer: CircuitOptimizer2<5> = CircuitOptimizer2::new();
            black_box((circuit, optimizer))
        });
    });

    group.finish();
}

/// Benchmark complex circuit patterns
fn benchmark_circuit_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_patterns");

    group.bench_function("bell_state", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<2>::new();
            circuit.h(0);
            circuit.cnot(0, 1);
            black_box(circuit)
        });
    });

    group.bench_function("ghz_state_5q", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            circuit.h(0);
            for i in 0..4 {
                circuit.cnot(black_box(i), black_box(i + 1));
            }
            black_box(circuit)
        });
    });

    group.bench_function("qft_5q", |b| {
        b.iter(|| {
            let mut circuit = Circuit::<5>::new();
            // Simplified QFT pattern
            for i in 0..5 {
                circuit.h(i);
                for j in (i + 1)..5 {
                    let angle = std::f64::consts::PI / 2_f64.powi(j - i);
                    circuit.cp(i, j, angle);
                }
            }
            black_box(circuit)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_circuit_construction,
    benchmark_single_qubit_gates,
    benchmark_two_qubit_gates,
    benchmark_circuit_optimization,
    benchmark_circuit_patterns
);

criterion_main!(benches);
