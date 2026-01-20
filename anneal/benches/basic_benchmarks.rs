//! Basic benchmarks for quantrs2-anneal
//!
//! Simplified benchmark suite that works with the current API

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quantrs2_anneal::ising::IsingModel;
use quantrs2_anneal::scirs2_integration::SciRS2QuboModel;

/// Create a test Ising problem (ring graph)
fn create_ring_ising(size: usize) -> IsingModel {
    let mut ising = IsingModel::new(size);
    for i in 0..size {
        let j = (i + 1) % size;
        let _ = ising.set_coupling(i, j, -1.0);
    }
    ising
}

/// Benchmark Ising model creation
fn bench_ising_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ising_creation");

    for size in &[10, 50, 100, 500] {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut ising = IsingModel::new(black_box(size));
                // Add some structure
                for i in 0..size.min(10) {
                    let _ = ising.set_coupling(i, (i + 1) % size, -1.0);
                }
                black_box(ising)
            });
        });
    }

    group.finish();
}

/// Benchmark Ising energy calculation
fn bench_ising_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("ising_energy");

    for size in &[50, 100, 200, 500] {
        let ising = create_ring_ising(*size);
        let spins: Vec<i8> = (0..*size)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(ising.energy(black_box(&spins)).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark SciRS2 QUBO model creation
fn bench_scirs2_qubo_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("scirs2_qubo_creation");

    for size in &[100, 500, 1000] {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| black_box(SciRS2QuboModel::new(black_box(size)).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark SciRS2 sparse QUBO operations
fn bench_scirs2_sparse_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("scirs2_sparse_eval");

    for size in &[100, 500, 1000] {
        let mut qubo = SciRS2QuboModel::new(*size).unwrap();

        // Add sparse structure
        for i in 0..(*size / 10) {
            let row = i;
            let col = (i + 1) % size;
            if row != col {
                qubo.set_quadratic(row, col, 1.0).unwrap();
            }
        }

        let solution: Vec<i8> = (0..*size).map(|i| i8::from(i % 2 == 0)).collect();

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(qubo.evaluate(black_box(&solution)).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark SciRS2 QUBO statistics
fn bench_scirs2_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("scirs2_stats");

    for size in &[500, 1000, 2000] {
        let mut qubo = SciRS2QuboModel::new(*size).unwrap();

        // Add moderate structure
        for i in 0..(*size / 5) {
            let row = (i * 7) % size;
            let col = (i * 13) % size;
            if row != col {
                qubo.set_quadratic(row, col, 1.0).unwrap();
            }
        }

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(qubo.get_statistics()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ising_creation,
    bench_ising_energy,
    bench_scirs2_qubo_creation,
    bench_scirs2_sparse_eval,
    bench_scirs2_stats
);

criterion_main!(benches);
