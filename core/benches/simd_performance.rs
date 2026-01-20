//! SIMD Performance Benchmarks
//!
//! This benchmark suite measures the performance of SIMD-accelerated
//! quantum gate operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_core::simd_enhanced::{RotationAxis, SimdGateEngine};
use scirs2_core::Complex64;

fn benchmark_rx_gates(c: &mut Criterion) {
    let engine = SimdGateEngine::new();

    let mut group = c.benchmark_group("rx_gate");

    for num_qubits in &[4, 6, 8, 10, 12] {
        let size = 1 << num_qubits;
        let mut state: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                b.iter(|| {
                    let mut state_copy = state.clone();
                    engine
                        .apply_rotation_gate(
                            black_box(&mut state_copy),
                            black_box(0),
                            black_box(RotationAxis::X),
                            black_box(std::f64::consts::PI / 4.0),
                        )
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cnot_gates(c: &mut Criterion) {
    let engine = SimdGateEngine::new();

    let mut group = c.benchmark_group("cnot_gate");

    for num_qubits in &[4, 6, 8, 10, 12] {
        let size = 1 << num_qubits;
        let mut state: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, &num_qubits| {
                b.iter(|| {
                    let mut state_copy = state.clone();
                    engine
                        .apply_cnot(black_box(&mut state_copy), black_box(0), black_box(1))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_batch_gates(c: &mut Criterion) {
    let engine = SimdGateEngine::new();

    let mut group = c.benchmark_group("batch_gates");

    for num_qubits in &[4, 6, 8] {
        let size = 1 << num_qubits;
        let mut state: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0);

        // Create a batch of gates
        let gates: Vec<_> = (0..*num_qubits)
            .map(|i| (i, RotationAxis::X, std::f64::consts::PI / 4.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    let mut state_copy = state.clone();
                    engine
                        .batch_apply_single_qubit(black_box(&mut state_copy), black_box(&gates))
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_fidelity(c: &mut Criterion) {
    let engine = SimdGateEngine::new();

    let mut group = c.benchmark_group("fidelity");

    for num_qubits in &[4, 6, 8, 10, 12] {
        let size = 1 << num_qubits;
        let state1: Vec<Complex64> = vec![Complex64::new(1.0 / (size as f64).sqrt(), 0.0); size];
        let state2: Vec<Complex64> = vec![Complex64::new(1.0 / (size as f64).sqrt(), 0.0); size];

        group.bench_with_input(
            BenchmarkId::from_parameter(num_qubits),
            num_qubits,
            |b, _| {
                b.iter(|| {
                    engine
                        .fidelity(black_box(&state1), black_box(&state2))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_rx_gates,
    benchmark_cnot_gates,
    benchmark_batch_gates,
    benchmark_fidelity
);
criterion_main!(benches);
