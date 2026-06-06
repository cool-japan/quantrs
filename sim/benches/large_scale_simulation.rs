#![allow(clippy::pedantic)]
//! Large-scale simulation benchmarks for QuantRS2 v0.2.0
//!
//! Measures wall-time scaling of state-vector simulation at n ∈ {10, 12, 14}
//! qubits for two circuit families:
//!
//! 1. H-layer (all qubits) — embarrassingly parallel single-qubit workload
//! 2. GHZ state (H + CNOT chain) — two-qubit entanglement workload
//! 3. Stabilizer Bell chain (scalable to large n with O(n²) cost)
//!
//! Run with:
//!   cargo bench -p quantrs2-sim --bench large_scale_simulation --all-features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Benchmark 1: State-vector H-layer scaling
// ---------------------------------------------------------------------------

fn run_h_layer_10(sim: &StateVectorSimulator) -> f64 {
    const N: usize = 10;
    let mut circuit = Circuit::<N>::new();
    for q in 0..N {
        circuit.h(q).expect("H gate failed");
    }
    let register = sim.run(&circuit).expect("simulation failed");
    register.amplitudes().iter().map(|a| a.norm_sqr()).sum()
}

fn run_h_layer_12(sim: &StateVectorSimulator) -> f64 {
    const N: usize = 12;
    let mut circuit = Circuit::<N>::new();
    for q in 0..N {
        circuit.h(q).expect("H gate failed");
    }
    let register = sim.run(&circuit).expect("simulation failed");
    register.amplitudes().iter().map(|a| a.norm_sqr()).sum()
}

fn run_h_layer_14(sim: &StateVectorSimulator) -> f64 {
    const N: usize = 14;
    let mut circuit = Circuit::<N>::new();
    for q in 0..N {
        circuit.h(q).expect("H gate failed");
    }
    let register = sim.run(&circuit).expect("simulation failed");
    register.amplitudes().iter().map(|a| a.norm_sqr()).sum()
}

fn bench_state_vector_h_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_h_layer");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    group.bench_with_input(BenchmarkId::new("h_layer", "10q"), &(), |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| black_box(run_h_layer_10(&sim)));
    });

    group.bench_with_input(BenchmarkId::new("h_layer", "12q"), &(), |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| black_box(run_h_layer_12(&sim)));
    });

    group.bench_with_input(BenchmarkId::new("h_layer", "14q"), &(), |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| black_box(run_h_layer_14(&sim)));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: State-vector GHZ-state scaling
// ---------------------------------------------------------------------------

fn run_ghz_10(sim: &StateVectorSimulator) -> f64 {
    const N: usize = 10;
    let mut circuit = Circuit::<N>::new();
    circuit.h(0).expect("H gate failed");
    for k in 1..N {
        circuit.cnot(0, k).expect("CNOT gate failed");
    }
    let register = sim.run(&circuit).expect("simulation failed");
    register.amplitudes().iter().map(|a| a.norm_sqr()).sum()
}

fn run_ghz_12(sim: &StateVectorSimulator) -> f64 {
    const N: usize = 12;
    let mut circuit = Circuit::<N>::new();
    circuit.h(0).expect("H gate failed");
    for k in 1..N {
        circuit.cnot(0, k).expect("CNOT gate failed");
    }
    let register = sim.run(&circuit).expect("simulation failed");
    register.amplitudes().iter().map(|a| a.norm_sqr()).sum()
}

fn run_ghz_14(sim: &StateVectorSimulator) -> f64 {
    const N: usize = 14;
    let mut circuit = Circuit::<N>::new();
    circuit.h(0).expect("H gate failed");
    for k in 1..N {
        circuit.cnot(0, k).expect("CNOT gate failed");
    }
    let register = sim.run(&circuit).expect("simulation failed");
    register.amplitudes().iter().map(|a| a.norm_sqr()).sum()
}

fn bench_state_vector_ghz(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector_ghz");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    group.bench_with_input(BenchmarkId::new("ghz", "10q"), &(), |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| black_box(run_ghz_10(&sim)));
    });

    group.bench_with_input(BenchmarkId::new("ghz", "12q"), &(), |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| black_box(run_ghz_12(&sim)));
    });

    group.bench_with_input(BenchmarkId::new("ghz", "14q"), &(), |b, _| {
        let sim = StateVectorSimulator::new();
        b.iter(|| black_box(run_ghz_14(&sim)));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: Stabilizer — Bell-chain scaling (large n, O(n²) cost)
// ---------------------------------------------------------------------------
fn bench_stabilizer_bell_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_bell_chain");
    group
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);

    for &n in &[20usize, 50, 100] {
        let num_pairs = n / 2;
        group.bench_with_input(
            BenchmarkId::new("bell_chain", format!("{n}q")),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut sim = StabilizerSimulator::new(n);
                    for k in 0..num_pairs {
                        sim.apply_gate(StabilizerGate::H(2 * k))
                            .expect("H gate failed");
                        sim.apply_gate(StabilizerGate::CNOT(2 * k, 2 * k + 1))
                            .expect("CNOT gate failed");
                    }
                    black_box(sim.get_stabilizers().len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_state_vector_h_layer,
    bench_state_vector_ghz,
    bench_stabilizer_bell_chain,
);
criterion_main!(benches);
