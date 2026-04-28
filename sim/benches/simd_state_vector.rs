//! Benchmarks for SIMD-accelerated single-qubit gate kernels.
//!
//! Compares `apply_h_simd` (SIMD path) against the scalar reference
//! `apply_gate_2x2_scalar` for state vectors of various sizes.
//!
//! Run with: `cargo bench --bench simd_state_vector -p quantrs2-sim`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_sim::state_vector_simd::{apply_gate_2x2_scalar, apply_h_simd};
use scirs2_core::Complex64;
use std::time::Duration;

// ============================================================================
// State helpers
// ============================================================================

fn make_state(n_qubits: u32) -> Vec<Complex64> {
    let size = 1usize << n_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(1.0, 0.0);
    state
}

// ============================================================================
// Benchmark: H gate SIMD vs scalar
// ============================================================================

fn bench_h_gate_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_h_gate");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);

    for &n in &[16u32, 18, 20] {
        // SIMD path
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &n| {
            let mut state = make_state(n);
            b.iter(|| {
                apply_h_simd(black_box(&mut state), 0, n as usize);
                black_box(&state);
            });
        });

        // Scalar reference
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            use std::f64::consts::FRAC_1_SQRT_2;
            let h_mat = [
                [
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                ],
                [
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                    Complex64::new(-FRAC_1_SQRT_2, 0.0),
                ],
            ];
            let mut state = make_state(n);
            b.iter(|| {
                apply_gate_2x2_scalar(black_box(&mut state), h_mat, 0, n as usize);
                black_box(&state);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark: H gate across different target qubits
// ============================================================================

fn bench_h_gate_target_qubit(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_h_gate_targets");
    group
        .measurement_time(Duration::from_secs(3))
        .sample_size(20);

    let n = 18u32;

    for &target in &[0usize, 5, 10, 17] {
        // SIMD
        group.bench_with_input(
            BenchmarkId::new("simd_target", target),
            &target,
            |b, &target| {
                let mut state = make_state(n);
                b.iter(|| {
                    apply_h_simd(black_box(&mut state), target, n as usize);
                    black_box(&state);
                });
            },
        );

        // Scalar
        group.bench_with_input(
            BenchmarkId::new("scalar_target", target),
            &target,
            |b, &target| {
                use std::f64::consts::FRAC_1_SQRT_2;
                let h_mat = [
                    [
                        Complex64::new(FRAC_1_SQRT_2, 0.0),
                        Complex64::new(FRAC_1_SQRT_2, 0.0),
                    ],
                    [
                        Complex64::new(FRAC_1_SQRT_2, 0.0),
                        Complex64::new(-FRAC_1_SQRT_2, 0.0),
                    ],
                ];
                let mut state = make_state(n);
                b.iter(|| {
                    apply_gate_2x2_scalar(black_box(&mut state), h_mat, target, n as usize);
                    black_box(&state);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion entry-point
// ============================================================================

criterion_group!(benches, bench_h_gate_simd, bench_h_gate_target_qubit);
criterion_main!(benches);
