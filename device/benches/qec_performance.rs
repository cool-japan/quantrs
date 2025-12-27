//! QEC Performance Benchmarks using Criterion
//!
//! This benchmark suite measures the performance of quantum error correction
//! codes and algorithms using the Criterion benchmarking framework.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use quantrs2_device::qec::{
    benchmarking::{QECBenchmarkConfig, QECBenchmarkSuite},
    QuantumErrorCode, ShorCode, SteaneCode, SurfaceCode, ToricCode,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

/// Benchmark QEC code encoding performance
fn bench_qec_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_encoding");
    group.measurement_time(Duration::from_secs(10));

    // Prepare logical state
    let logical_state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

    // Benchmark Steane Code [[7,1,3]]
    let steane_code = SteaneCode::new();
    group.bench_function("steane_7_1_3", |b| {
        b.iter(|| {
            black_box(steane_code.encode_logical_state(&logical_state).unwrap());
        });
    });

    // Benchmark Shor Code [[9,1,3]]
    let shor_code = ShorCode::new();
    group.bench_function("shor_9_1_3", |b| {
        b.iter(|| {
            black_box(shor_code.encode_logical_state(&logical_state).unwrap());
        });
    });

    // Benchmark Surface Code [[13,1,3]]
    let surface_code = SurfaceCode::new(3);
    group.bench_function("surface_13_1_3", |b| {
        b.iter(|| {
            black_box(surface_code.encode_logical_state(&logical_state).unwrap());
        });
    });

    // Benchmark Toric Code 2x2
    let toric_code = ToricCode::new((2, 2));
    group.bench_function("toric_2x2", |b| {
        b.iter(|| {
            black_box(toric_code.encode_logical_state(&logical_state).unwrap());
        });
    });

    group.finish();
}

/// Benchmark QEC stabilizer generation
fn bench_stabilizer_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_generation");

    let steane_code = SteaneCode::new();
    group.bench_function("steane", |b| {
        b.iter(|| {
            black_box(steane_code.get_stabilizers());
        });
    });

    let shor_code = ShorCode::new();
    group.bench_function("shor", |b| {
        b.iter(|| {
            black_box(shor_code.get_stabilizers());
        });
    });

    let surface_code = SurfaceCode::new(3);
    group.bench_function("surface", |b| {
        b.iter(|| {
            black_box(surface_code.get_stabilizers());
        });
    });

    let toric_code = ToricCode::new((2, 2));
    group.bench_function("toric", |b| {
        b.iter(|| {
            black_box(toric_code.get_stabilizers());
        });
    });

    group.finish();
}

/// Benchmark logical operator generation
fn bench_logical_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("logical_operators");

    let steane_code = SteaneCode::new();
    group.bench_function("steane", |b| {
        b.iter(|| {
            black_box(steane_code.get_logical_operators());
        });
    });

    let shor_code = ShorCode::new();
    group.bench_function("shor", |b| {
        b.iter(|| {
            black_box(shor_code.get_logical_operators());
        });
    });

    let surface_code = SurfaceCode::new(3);
    group.bench_function("surface", |b| {
        b.iter(|| {
            black_box(surface_code.get_logical_operators());
        });
    });

    let toric_code = ToricCode::new((2, 2));
    group.bench_function("toric", |b| {
        b.iter(|| {
            black_box(toric_code.get_logical_operators());
        });
    });

    group.finish();
}

/// Benchmark comprehensive QEC suite (lightweight version for CI)
fn bench_comprehensive_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_suite");
    group.sample_size(10); // Reduce sample size for faster benchmarks
    group.measurement_time(Duration::from_secs(30));

    // Create lightweight config for benchmarking
    let config = QECBenchmarkConfig {
        iterations: 10, // Reduced for benchmarking
        shots_per_measurement: 100,
        error_rates: vec![0.001, 0.01],
        circuit_depths: vec![10, 20],
        enable_detailed_stats: true,
        enable_profiling: false,
        max_duration: Duration::from_secs(60),
        confidence_level: 0.95,
    };

    let suite = QECBenchmarkSuite::new(config);

    group.bench_function("full_suite", |b| {
        b.iter(|| {
            black_box(suite.run_comprehensive_benchmark().unwrap());
        });
    });

    group.finish();
}

/// Benchmark QEC code scaling with different parameters
fn bench_code_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("code_scaling");
    group.measurement_time(Duration::from_secs(10));

    let logical_state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

    // Benchmark Surface Code with different distances
    for distance in &[3, 5] {
        let surface_code = SurfaceCode::new(*distance);
        let num_qubits = surface_code.num_data_qubits() + surface_code.num_ancilla_qubits();

        group.throughput(Throughput::Elements(num_qubits as u64));
        group.bench_with_input(
            BenchmarkId::new("surface_code", distance),
            distance,
            |b, _| {
                b.iter(|| {
                    black_box(surface_code.encode_logical_state(&logical_state).unwrap());
                });
            },
        );
    }

    // Benchmark Toric Code with different lattice sizes
    for size in &[2, 3] {
        let toric_code = ToricCode::new((*size, *size));
        let num_qubits = toric_code.num_data_qubits() + toric_code.num_ancilla_qubits();

        group.throughput(Throughput::Elements(num_qubits as u64));
        group.bench_with_input(BenchmarkId::new("toric_code", size), size, |b, _| {
            b.iter(|| {
                black_box(toric_code.encode_logical_state(&logical_state).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark QEC code properties
fn bench_code_properties(c: &mut Criterion) {
    let mut group = c.benchmark_group("code_properties");

    let steane_code = SteaneCode::new();
    group.bench_function("steane_distance", |b| {
        b.iter(|| {
            black_box(steane_code.distance());
        });
    });

    group.bench_function("steane_num_qubits", |b| {
        b.iter(|| {
            black_box(steane_code.num_data_qubits());
            black_box(steane_code.num_ancilla_qubits());
        });
    });

    let surface_code = SurfaceCode::new(3);
    group.bench_function("surface_distance", |b| {
        b.iter(|| {
            black_box(surface_code.distance());
        });
    });

    group.bench_function("surface_num_qubits", |b| {
        b.iter(|| {
            black_box(surface_code.num_data_qubits());
            black_box(surface_code.num_ancilla_qubits());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_qec_encoding,
    bench_stabilizer_generation,
    bench_logical_operators,
    bench_code_properties,
    bench_code_scaling,
    bench_comprehensive_suite,
);

criterion_main!(benches);
