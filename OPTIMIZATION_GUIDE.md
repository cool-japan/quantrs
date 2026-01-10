# QuantRS2 Optimization Guide

This guide provides practical recommendations for optimizing quantum circuit simulation performance in QuantRS2.

## Quick Decision Tree

```
Does your circuit use only Clifford gates (H, S, CNOT, etc.)?
│
├─ YES → Use StabilizerSimulator
│   │
│   ├─ >20 qubits? → Stabilizer REQUIRED
│   └─ <20 qubits? → Stabilizer recommended (faster)
│
└─ NO (has T, RZ, or arbitrary rotations)
    │
    ├─ <25 qubits? → Use QulacsStateVector
    └─ ≥25 qubits? → Consider tensor network or GPU acceleration
```

## Simulator Selection

### QulacsStateVector (State-Vector Backend)

**Best For:**
- Universal quantum circuits
- VQE, QAOA, quantum chemistry
- Small systems (<25 qubits)
- Amplitude inspection needed

**Performance Characteristics:**
- Memory: O(2^n) = 16 bytes × 2^n
- Time per gate: O(2^n)
- Max practical: ~30 qubits

**Example:**
```rust
use quantrs2_sim::prelude::*;

let mut state = QulacsStateVector::new(20)?;
qulacs_gates::hadamard(&mut state, 0)?;
qulacs_gates::ry(&mut state, 0, angle)?;  // Non-Clifford
qulacs_gates::cnot(&mut state, 0, 1)?;
```

### StabilizerSimulator (Clifford Backend)

**Best For:**
- Error correction simulations
- Randomized benchmarking
- Large Clifford circuits (100+ qubits)
- Deep circuits (1000+ layers)

**Performance Characteristics:**
- Memory: O(n²) ≈ 2n² bytes
- Time per gate: O(n²)
- Max practical: 1,000,000+ qubits

**Example:**
```rust
use quantrs2_sim::stabilizer::*;

let mut sim = StabilizerSimulator::new(1000);
sim.apply_gate(StabilizerGate::H(0))?;
sim.apply_gate(StabilizerGate::CNOT(0, 1))?;
sim.apply_gate(StabilizerGate::CZ(1, 2))?;
```

## Performance Optimization Techniques

### 1. Gate Ordering Optimization

**Bad:**
```rust
// Jumping between qubits - poor cache locality
qulacs_gates::hadamard(&mut state, 0)?;
qulacs_gates::hadamard(&mut state, 10)?;
qulacs_gates::rx(&mut state, 0, angle)?;
qulacs_gates::rx(&mut state, 10, angle)?;
```

**Good:**
```rust
// Apply consecutive gates to same qubit
qulacs_gates::hadamard(&mut state, 0)?;
qulacs_gates::rx(&mut state, 0, angle)?;
qulacs_gates::hadamard(&mut state, 10)?;
qulacs_gates::rx(&mut state, 10, angle)?;
```

**Impact:** 10-20% speedup for state-vector

### 2. Batch Measurements

**Bad:**
```rust
// Measuring 1000 times individually
for _ in 0..1000 {
    let outcome = state.measure(0)?;
    // Process outcome
}
```

**Good:**
```rust
// Single batch measurement
let samples = state.sample(1000)?;
for sample in samples {
    // Process sample
}
```

**Impact:** 50-100x speedup

### 3. Prefer Qubit 0 for Single-Qubit Gates

**Context:** Qulacs backend has optimized path for qubit 0

```rust
// ✅ Fastest path
qulacs_gates::hadamard(&mut state, 0)?;

// ⚠️ Slower but still optimized
qulacs_gates::hadamard(&mut state, 5)?;
```

**Impact:** 2x speedup for single-qubit gates on qubit 0

### 4. Use Builder Pattern for Stabilizer

**Bad:**
```rust
let mut sim = StabilizerSimulator::new(10);
sim.apply_gate(StabilizerGate::H(0))?;
sim.apply_gate(StabilizerGate::CNOT(0, 1))?;
```

**Good:**
```rust
let sim = CliffordCircuitBuilder::new(10)
    .h(0)
    .cnot(0, 1)
    .run()?;
```

**Impact:** Cleaner code, potential for compiler optimizations

### 5. Reuse Simulators

**Bad:**
```rust
for trial in 0..1000 {
    let mut sim = StabilizerSimulator::new(100);  // Allocates each time
    // Run circuit
}
```

**Good:**
```rust
let mut sim = StabilizerSimulator::new(100);
for trial in 0..1000 {
    sim.reset();  // Resets to |0⟩ state
    // Run circuit
}
```

**Impact:** 10-100x speedup by avoiding allocations

### 6. Profile Before Optimizing

```rust
use std::time::Instant;

let start = Instant::now();
// Your circuit here
let duration = start.elapsed();
println!("Circuit took: {:?}", duration);
```

**Tools:**
- `cargo build --release` for benchmarks
- `cargo bench` for criterion benchmarks
- Profile with `perf` on Linux

## Memory Optimization

### Calculate Memory Requirements

**State-Vector:**
```rust
fn state_vector_memory(num_qubits: usize) -> usize {
    (1 << num_qubits) * 16  // 16 bytes per complex number
}

// Example: 25 qubits = 536 MB
```

**Stabilizer:**
```rust
fn stabilizer_memory(num_qubits: usize) -> usize {
    2 * num_qubits * num_qubits  // 2 bits per entry
}

// Example: 1000 qubits = 250 KB
```

### Memory-Constrained Scenarios

If you have limited memory:

1. **Use stabilizer when possible** (polynomial memory)
2. **Reduce qubit count** in state-vector
3. **Consider tensor networks** for sparse circuits
4. **Stream measurements** instead of storing all

## Circuit Complexity Analysis

### Clifford Gate Detection

Check if your circuit can use stabilizer simulator:

```rust
use quantrs2_sim::stabilizer::is_clifford_circuit;

let circuit = /* your circuit */;
if is_clifford_circuit(&circuit) {
    // Use StabilizerSimulator
    println!("Circuit is Clifford - use stabilizer");
} else {
    // Use QulacsStateVector
    println!("Circuit has non-Clifford gates - use state-vector");
}
```

### Gate Count Estimation

For stabilizer simulation:
- **Time ~ O(n² × depth)**
- **Memory ~ O(n²)**

Example:
```rust
// 100 qubits, 1000 layers
// Time: ~100² × 1000 = 10M operations
// Memory: ~100² × 2 bytes = 20 KB
```

For state-vector:
- **Time ~ O(2^n × depth)**
- **Memory ~ O(2^n)**

Example:
```rust
// 20 qubits, 100 layers
// Time: ~2^20 × 100 = 104M operations
// Memory: ~2^20 × 16 = 16 MB
```

## Scaling Recommendations

### Small Circuits (< 10 qubits)

**Recommendation:** Either simulator works fine

```rust
// State-vector is simpler for small circuits
let mut state = QulacsStateVector::new(8)?;
```

### Medium Circuits (10-20 qubits)

**Recommendation:**
- Stabilizer if Clifford-only (10x faster)
- State-vector if non-Clifford needed

```rust
// Check gate types first
if all_clifford {
    StabilizerSimulator::new(15)
} else {
    QulacsStateVector::new(15)
}
```

### Large Circuits (20-30 qubits)

**Recommendation:**
- Stabilizer strongly recommended if possible
- State-vector at practical limit

```rust
// Stabilizer handles this easily
let mut sim = StabilizerSimulator::new(25);

// State-vector needs significant memory
let mut state = QulacsStateVector::new(25)?;  // 512 MB!
```

### Massive Circuits (> 30 qubits)

**Recommendation:**
- **Stabilizer is only option** for Clifford circuits
- Non-Clifford requires specialized techniques (GPU, distributed)

```rust
// This works!
let mut sim = StabilizerSimulator::new(1000);

// This is impossible
// let mut state = QulacsStateVector::new(1000)?;  // Would need 10^300 bytes
```

## Benchmarking Best Practices

### Using Criterion

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_bell_state(c: &mut Criterion) {
    c.bench_function("bell_state", |b| {
        b.iter(|| {
            let mut state = QulacsStateVector::new(2).unwrap();
            qulacs_gates::hadamard(&mut state, black_box(0)).unwrap();
            qulacs_gates::cnot(&mut state, black_box(0), black_box(1)).unwrap();
            black_box(state);
        });
    });
}

criterion_group!(benches, bench_bell_state);
criterion_main!(benches);
```

### Profiling Tips

1. **Always use `--release` for benchmarks**
   ```bash
   cargo bench
   cargo run --release --example my_circuit
   ```

2. **Use `black_box` to prevent optimizer from removing code**
   ```rust
   black_box(state.measure(0)?);
   ```

3. **Warm up before measurement**
   ```rust
   // Run once to warm up
   run_circuit(&mut state)?;

   // Now measure
   let start = Instant::now();
   run_circuit(&mut state)?;
   let duration = start.elapsed();
   ```

## Common Performance Pitfalls

### ❌ Pitfall 1: Using Wrong Simulator

```rust
// BAD: Using state-vector for large Clifford circuit
let mut state = QulacsStateVector::new(50)?;  // Will fail or be very slow
```

**Fix:** Use stabilizer for Clifford circuits
```rust
let mut sim = StabilizerSimulator::new(50);  // Fast!
```

### ❌ Pitfall 2: Unnecessary Cloning

```rust
// BAD: Cloning state repeatedly
for _ in 0..1000 {
    let mut new_state = state.clone();  // Expensive!
    measure(&mut new_state)?;
}
```

**Fix:** Use sampling instead
```rust
let samples = state.sample(1000)?;  // One operation
```

### ❌ Pitfall 3: Inefficient Measurements

```rust
// BAD: Measuring and reconstructing state
for _ in 0..shots {
    let mut temp = state.clone();
    let outcome = temp.measure(0)?;
}
```

**Fix:** Use non-destructive sampling
```rust
let samples = state.sample(shots)?;  // Doesn't modify state
```

### ❌ Pitfall 4: Not Using Builder Pattern

```rust
// BAD: Verbose and error-prone
let mut sim = StabilizerSimulator::new(10);
sim.apply_gate(StabilizerGate::H(0)).unwrap();
sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
```

**Fix:** Use builder
```rust
let sim = CliffordCircuitBuilder::new(10)
    .h(0)
    .cnot(0, 1)
    .run()?;
```

## Advanced Optimizations (Future)

### SIMD Acceleration

When available in SciRS2:
```rust
use scirs2_core::simd_ops::PlatformCapabilities;

if PlatformCapabilities::current().has_avx2() {
    // Will automatically use SIMD
    qulacs_gates::hadamard(&mut state, 0)?;
}
```

### GPU Acceleration

When implemented:
```rust
use scirs2_core::gpu::*;

let gpu_state = GPUStateVector::from(state);
gpu_state.apply_circuit(&circuit)?;
```

### Distributed Simulation

For truly massive Clifford circuits:
```rust
// Future: MPI support for stabilizer
let sim = DistributedStabilizer::new(10_000_000, MPI_COMM_WORLD)?;
```

## Summary

| Scenario | Recommended Simulator | Expected Performance |
|----------|----------------------|---------------------|
| <10 qubits, any gates | QulacsStateVector | µs per circuit |
| 10-20 qubits, Clifford | StabilizerSimulator | µs per circuit, 10x faster |
| 10-20 qubits, non-Clifford | QulacsStateVector | ms per circuit |
| 20-30 qubits, Clifford | StabilizerSimulator | µs per circuit |
| 20-30 qubits, non-Clifford | QulacsStateVector | seconds per circuit |
| >30 qubits, Clifford | StabilizerSimulator | ms-seconds per circuit |
| >30 qubits, non-Clifford | GPU or distributed | specialized solutions |

**Golden Rule:** Use stabilizer whenever your circuit is Clifford-only. Use state-vector when you need non-Clifford gates or amplitude access.
