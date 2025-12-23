//! Gate Performance Benchmarks
//!
//! Benchmarks for quantum gate operations performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantrs2_core::{
    gate::{multi, single, GateOp},
    qubit::QubitId,
};

fn bench_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");

    // Hadamard gate
    group.bench_function("hadamard_matrix", |b| {
        let gate = single::Hadamard { target: QubitId(0) };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    // Pauli-X gate
    group.bench_function("pauli_x_matrix", |b| {
        let gate = single::PauliX { target: QubitId(0) };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    // Rotation gates
    group.bench_function("rotation_x_matrix", |b| {
        let gate = single::RotationX {
            target: QubitId(0),
            theta: std::f64::consts::PI / 4.0,
        };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    group.finish();
}

fn bench_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gates");

    // CNOT gate
    group.bench_function("cnot_matrix", |b| {
        let gate = multi::CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    // CZ gate
    group.bench_function("cz_matrix", |b| {
        let gate = multi::CZ {
            control: QubitId(0),
            target: QubitId(1),
        };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    // SWAP gate
    group.bench_function("swap_matrix", |b| {
        let gate = multi::SWAP {
            qubit1: QubitId(0),
            qubit2: QubitId(1),
        };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    group.finish();
}

fn bench_three_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("three_qubit_gates");

    // Toffoli gate
    group.bench_function("toffoli_matrix", |b| {
        let gate = multi::Toffoli {
            control1: QubitId(0),
            control2: QubitId(1),
            target: QubitId(2),
        };
        b.iter(|| {
            black_box(gate.matrix().unwrap());
        });
    });

    group.finish();
}

fn bench_gate_properties(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_properties");

    let h_gate = single::Hadamard { target: QubitId(0) };

    group.bench_function("gate_name", |b| {
        b.iter(|| {
            black_box(h_gate.name());
        });
    });

    group.bench_function("gate_qubits", |b| {
        b.iter(|| {
            black_box(h_gate.qubits());
        });
    });

    group.bench_function("gate_num_qubits", |b| {
        b.iter(|| {
            black_box(h_gate.num_qubits());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_qubit_gates,
    bench_two_qubit_gates,
    bench_three_qubit_gates,
    bench_gate_properties
);
criterion_main!(benches);
