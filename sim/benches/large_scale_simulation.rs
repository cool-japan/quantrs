#![allow(clippy::pedantic)]
//! Large-scale simulation benchmarks for QuantRS2 v0.2.0
//!
//! Measures wall-time scaling of state-vector simulation at n ∈ {10, 12, 14}
//! qubits for two circuit families:
//!
//! 1. H-layer (all qubits) — embarrassingly parallel single-qubit workload
//! 2. Entangling layer (alternating CNOT chain) — two-qubit workload
//!
//! Run with:
//!   cargo bench -p quantrs2-sim --bench large_scale_simulation --all-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helper: build H-layer circuit (type-erased via macro)
// ---------------------------------------------------------------------------

/// Build and run an N-qubit H-layer circuit, returning the norm of the state.
///
/// We use a macro instead of a generic function here because the qubit count
/// must be a compile-time constant (`const N: usize`) for `Circuit<N>`.
macro_rules! bench_h_layer {
    ($n:expr, $sim:expr) => {{
        let mut circuit = Circuit::<{ $n }>::new();
        for q in 0..{ $n } {
            circuit.h(q).expect("H gate failed");
        }
        let register = $sim.run(black_box(&circuit)).expect("simulation failed");
        let norm: f64 = register.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        norm
    }};
}

/// Build and run an N-qubit GHZ-state circuit.
macro_rules! bench_ghz {
    ($n:expr, $sim:expr) => {{
        let mut circuit = Circuit::<{ $n }>::new();
        circuit.h(0).expect("H gate failed");
        for k in 1..{ $n } {
            circuit.cnot(0, k).expect("CNOT gate failed");
        }
        let register = $sim.run(black_box(&circuit)).expect("simulation failed");
        let norm: f64 = register.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        norm
    }};
}

/// Build and run an N-qubit nearest-neighbour CNOT chain (after H prep).
macro_rules! bench_cnot_chain {
    ($n:expr, $sim:expr) => {{
        let mut circuit = Circuit::<{ $n }>::new();
        // Prepare superposition
        for q in 0..{ $n } {
            circuit.h(q).expect("H gate failed");
        }
        // Nearest-neighbour CNOT chain
        for k in 0..({ $n } - 1) {
            circuit.cnot(k, k + 1).expect("CNOT gate failed");
        }
        let register = $sim.run(black_box(&circuit)).expect("simulation failed");
        let norm: f64 = register.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        norm
    }};
}

// ---------------------------------------------------------------------------
// Benchmark 1: State-vector H-layer scaling
// ---------------------------------------------------------------------------
fn bench_state_vector_h_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_h_layer");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    group.bench_with_input(BenchmarkId::new("h_layer", "10q"), &10u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_h_layer!(10, sim));
    });

    group.bench_with_input(BenchmarkId::new("h_layer", "12q"), &12u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_h_layer!(12, sim));
    });

    group.bench_with_input(BenchmarkId::new("h_layer", "14q"), &14u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_h_layer!(14, sim));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: State-vector GHZ-state scaling
// ---------------------------------------------------------------------------
fn bench_state_vector_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_ghz");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    group.bench_with_input(BenchmarkId::new("ghz", "10q"), &10u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_ghz!(10, sim));
    });

    group.bench_with_input(BenchmarkId::new("ghz", "12q"), &12u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_ghz!(12, sim));
    });

    group.bench_with_input(BenchmarkId::new("ghz", "14q"), &14u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_ghz!(14, sim));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: State-vector nearest-neighbour CNOT chain
// ---------------------------------------------------------------------------
fn bench_state_vector_cnot_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_cnot_chain");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    group.bench_with_input(BenchmarkId::new("cnot_chain", "10q"), &10u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_cnot_chain!(10, sim));
    });

    group.bench_with_input(BenchmarkId::new("cnot_chain", "12q"), &12u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_cnot_chain!(12, sim));
    });

    group.bench_with_input(BenchmarkId::new("cnot_chain", "14q"), &14u32, |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| bench_cnot_chain!(14, sim));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: Stabilizer — Bell-chain scaling (much larger qubits possible)
// ---------------------------------------------------------------------------
fn bench_stabilizer_bell_chain(c: &mut Criterion) {
    use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator};

    let mut group = c.benchmark_group("stabilizer_bell_chain");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[20usize, 50, 100] {
        let num_pairs = n / 2;
        group.bench_with_input(BenchmarkId::new("bell_chain", format!("{n}q")), &n, |b, &n| {
            b.iter(|| {
                let mut sim = StabilizerSimulator::new(n);
                for k in 0..num_pairs {
                    sim.apply_gate(StabilizerGate::H(2 * k)).expect("H failed");
                    sim.apply_gate(StabilizerGate::CNOT(2 * k, 2 * k + 1))
                        .expect("CNOT failed");
                }
                black_box(sim.get_stabilizers().len())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_state_vector_h_layer,
    bench_state_vector_ghz,
    bench_state_vector_cnot_chain,
    bench_stabilizer_bell_chain,
);
criterion_main!(benches);
