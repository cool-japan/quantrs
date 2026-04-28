#![allow(clippy::pedantic)]
//! Sampler scalability benchmarks for QuantRS2 v0.2.0
//!
//! Measures the wall-time scaling of SA, Tabu, and SB samplers over
//! random QUBO problems at n ∈ {10, 30, 50} variables.
//!
//! Run with:
//!   cargo bench -p quantrs2-tytan --bench sampler_scalability --all-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_tytan::sampler::{SASampler, SBSampler, SBVariant, Sampler, TabuSampler};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helper: reproducible random QUBO generator (LCG, no external deps)
// ---------------------------------------------------------------------------
fn generate_random_qubo(n: usize, seed: u64) -> scirs2_core::ndarray::Array2<f64> {
    const A: u64 = 1_664_525;
    const C: u64 = 1_013_904_223;
    let mut lcg_state = seed;

    let mut lcg_next = || -> f64 {
        lcg_state = lcg_state.wrapping_mul(A).wrapping_add(C);
        (lcg_state as f64 / u64::MAX as f64) * 4.0 - 2.0
    };

    let mut q = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        q[[i, i]] = lcg_next();
        for j in (i + 1)..n {
            q[[i, j]] = lcg_next();
        }
    }
    q
}

/// Build a variable name map for `n` variables: x0..x(n-1).
fn build_var_map(n: usize) -> HashMap<String, usize> {
    (0..n).map(|i| (format!("x{i}"), i)).collect()
}

// ---------------------------------------------------------------------------
// Benchmark 1: SASampler — shots=1, n ∈ {10, 30, 50}
// ---------------------------------------------------------------------------
fn bench_sa_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sa_sampler_scaling");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[10usize, 30, 50] {
        let q = generate_random_qubo(n, 42 * n as u64);
        let var_map = build_var_map(n);
        let qubo = (q, var_map);

        group.bench_with_input(
            BenchmarkId::new("sa_1shot", format!("{n}var")),
            &n,
            |b, _| {
                let sampler = SASampler::new(Some(0));
                b.iter(|| {
                    let results = sampler
                        .run_qubo(black_box(&qubo), 1)
                        .expect("SA failed");
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: TabuSampler — shots=1, n ∈ {10, 30, 50}
// ---------------------------------------------------------------------------
fn bench_tabu_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("tabu_sampler_scaling");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[10usize, 30, 50] {
        let q = generate_random_qubo(n, 99 * n as u64);
        let var_map = build_var_map(n);
        let qubo = (q, var_map);

        group.bench_with_input(
            BenchmarkId::new("tabu_1shot", format!("{n}var")),
            &n,
            |b, _| {
                let sampler = TabuSampler::new().with_seed(0);
                b.iter(|| {
                    let results = sampler
                        .run_qubo(black_box(&qubo), 1)
                        .expect("Tabu failed");
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: SBSampler (Discrete) — shots=1, n ∈ {10, 30, 50}
//
// Discrete SB is typically faster than Ballistic, so both CI and local
// benchmarks use it without needing `#[ignore]`.
// ---------------------------------------------------------------------------
fn bench_sb_discrete_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sb_discrete_sampler_scaling");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[10usize, 30, 50] {
        let q = generate_random_qubo(n, 77 * n as u64);
        let var_map = build_var_map(n);
        let qubo = (q, var_map);

        group.bench_with_input(
            BenchmarkId::new("sb_discrete_1shot", format!("{n}var")),
            &n,
            |b, _| {
                let sampler = SBSampler::new()
                    .with_seed(0)
                    .with_variant(SBVariant::Discrete);
                b.iter(|| {
                    let results = sampler
                        .run_qubo(black_box(&qubo), 1)
                        .expect("SB(Discrete) failed");
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: SBSampler (Ballistic) — shots=1, n ∈ {10, 30, 50}
// ---------------------------------------------------------------------------
fn bench_sb_ballistic_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("sb_ballistic_sampler_scaling");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[10usize, 30, 50] {
        let q = generate_random_qubo(n, 55 * n as u64);
        let var_map = build_var_map(n);
        let qubo = (q, var_map);

        group.bench_with_input(
            BenchmarkId::new("sb_ballistic_1shot", format!("{n}var")),
            &n,
            |b, _| {
                let sampler = SBSampler::new()
                    .with_seed(0)
                    .with_variant(SBVariant::Ballistic);
                b.iter(|| {
                    let results = sampler
                        .run_qubo(black_box(&qubo), 1)
                        .expect("SB(Ballistic) failed");
                    black_box(results.len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sa_sampler,
    bench_tabu_sampler,
    bench_sb_discrete_sampler,
    bench_sb_ballistic_sampler,
);
criterion_main!(benches);
