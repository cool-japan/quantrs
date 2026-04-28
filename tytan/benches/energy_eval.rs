#![allow(
    clippy::pedantic,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::doc_lazy_continuation,
)]
//! Criterion benchmarks for SIMD-accelerated QUBO energy evaluation.
//!
//! Measures throughput of scalar vs SIMD implementations of:
//! - `energy_full` / `energy_full_simd`: full O(n²) energy computation
//! - `energy_delta` / `energy_delta_simd`: O(n) incremental flip delta
//! - `compute_influence` / `compute_influence_simd`: O(n²) influence vector init
//! - `update_influence` / `update_influence_simd`: O(n) influence vector update
//!
//! Run with:
//!   cargo bench -p quantrs2-tytan --bench energy_eval --all-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_tytan::sampler::energy::{
    compute_influence, compute_influence_simd, energy_delta, energy_delta_simd, energy_full,
    energy_full_simd, update_influence, update_influence_simd,
};
use std::time::Duration;

// ─── QUBO generator (deterministic LCG, no external deps) ────────────────────

fn make_qubo(n: usize, seed: u64) -> Vec<f64> {
    let mut q = vec![0.0f64; n * n];
    let mut s = seed;
    let mut lcg = || -> f64 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as f64 / u32::MAX as f64 * 4.0 - 2.0
    };
    for i in 0..n {
        q[i * n + i] = lcg();
        for j in (i + 1)..n {
            let v = lcg();
            q[i * n + j] = v;
            q[j * n + i] = v;
        }
    }
    q
}

fn make_state(n: usize, seed: u64) -> Vec<bool> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 63) == 1
        })
        .collect()
}

// ─── energy_full benchmarks ───────────────────────────────────────────────────

fn bench_energy_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_full");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);

    for &n in &[16usize, 64, 256, 1024] {
        let q = make_qubo(n, 42);
        let state = make_state(n, 99);

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
            b.iter(|| energy_full(black_box(&state), black_box(&q), black_box(n)));
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &_n| {
            b.iter(|| energy_full_simd(black_box(&state), black_box(&q), black_box(n)));
        });
    }

    group.finish();
}

// ─── energy_delta benchmarks ──────────────────────────────────────────────────

fn bench_energy_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_delta");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);

    for &n in &[16usize, 64, 256, 1024] {
        let q = make_qubo(n, 123);
        let state = make_state(n, 456);
        let k = n / 3; // flip index in middle of vector

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
            b.iter(|| energy_delta(black_box(&state), black_box(&q), black_box(n), black_box(k)));
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &_n| {
            b.iter(|| {
                energy_delta_simd(black_box(&state), black_box(&q), black_box(n), black_box(k))
            });
        });
    }

    group.finish();
}

// ─── compute_influence benchmarks ─────────────────────────────────────────────

fn bench_compute_influence(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_influence");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);

    for &n in &[16usize, 64, 256, 1024] {
        let q = make_qubo(n, 789);
        let state = make_state(n, 321);

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
            b.iter(|| compute_influence(black_box(&state), black_box(&q), black_box(n)));
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &_n| {
            b.iter(|| compute_influence_simd(black_box(&state), black_box(&q), black_box(n)));
        });
    }

    group.finish();
}

// ─── update_influence benchmarks ──────────────────────────────────────────────

fn bench_update_influence(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_influence");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);

    for &n in &[16usize, 64, 256, 1024] {
        let q = make_qubo(n, 999);
        let state = make_state(n, 111);
        let k = n / 4;

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
            let g_init = {
                let mut g = vec![0.0f64; n];
                for i in 0..n {
                    g[i] = q[i * n + i];
                }
                g
            };
            b.iter(|| {
                let mut g = g_init.clone();
                update_influence(
                    black_box(&mut g),
                    black_box(&q),
                    black_box(n),
                    black_box(k),
                    black_box(true),
                );
                black_box(g)
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &_n| {
            let g_init = {
                let mut g = vec![0.0f64; n];
                for i in 0..n {
                    g[i] = q[i * n + i];
                }
                g
            };
            b.iter(|| {
                let mut g = g_init.clone();
                update_influence_simd(
                    black_box(&mut g),
                    black_box(&q),
                    black_box(n),
                    black_box(k),
                    black_box(true),
                );
                black_box(g)
            });
        });
    }

    group.finish();
}

// ─── sampler hot-loop simulation ─────────────────────────────────────────────

/// Simulate the hot inner loop of a tabu/SA sampler:
/// - compute_influence once, then update_influence per flip.
/// Measures throughput of the incremental pattern used by real samplers.
fn bench_sampler_inner_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler_inner_loop");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[64usize, 256, 1024] {
        let q = make_qubo(n, 7777);
        let state = make_state(n, 8888);

        group.bench_with_input(BenchmarkId::new("scalar_100_flips", n), &n, |b, &_n| {
            b.iter(|| {
                let mut s = state.clone();
                let mut g = compute_influence(&s, &q, n);
                for step in 0..100 {
                    let k = step % n;
                    let new_val = !s[k];
                    update_influence(&mut g, &q, n, k, new_val);
                    s[k] = new_val;
                }
                black_box((s, g))
            });
        });

        group.bench_with_input(BenchmarkId::new("simd_100_flips", n), &n, |b, &_n| {
            b.iter(|| {
                let mut s = state.clone();
                let mut g = compute_influence_simd(&s, &q, n);
                for step in 0..100 {
                    let k = step % n;
                    let new_val = !s[k];
                    update_influence_simd(&mut g, &q, n, k, new_val);
                    s[k] = new_val;
                }
                black_box((s, g))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_energy_full,
    bench_energy_delta,
    bench_compute_influence,
    bench_update_influence,
    bench_sampler_inner_loop,
);
criterion_main!(benches);
