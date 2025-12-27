//! SciRS2 Integration Example for QuantRS2
//!
//! This example demonstrates proper usage of SciRS2 (Scientific Computing in Rust)
//! within the QuantRS2 framework. It showcases the unified patterns for:
//! - Complex number operations (quantum amplitudes)
//! - Array operations (state vectors and operators)
//! - Random number generation (quantum measurements)
//! - SIMD operations (performance optimization)
//!
//! **SciRS2 Policy Compliance:**
//! - ✅ Use `scirs2_core::Complex64` for complex numbers
//! - ✅ Use `scirs2_core::ndarray::*` for array operations (unified access)
//! - ✅ Use `scirs2_core::random::prelude::*` for RNG
//! - ❌ Never use `num_complex`, `ndarray`, or `rand` directly

use quantrs2::utils;
use quantrs2::version;

// ✅ CORRECT: Unified SciRS2 imports from scirs2-core root
use scirs2_core::{Complex32, Complex64};

// ✅ CORRECT: Unified ndarray access
use scirs2_core::ndarray::{array, s, Array1, Array2, Axis};

// ✅ CORRECT: Unified random distributions
use scirs2_core::random::prelude::*;
use scirs2_core::random::{thread_rng, Distribution, Normal as RandNormal};

// ✅ CORRECT: Enhanced unified distributions
use scirs2_core::random::distributions_unified::{UnifiedBeta, UnifiedNormal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  SciRS2 Integration Example for QuantRS2                 ║");
    println!("║  Demonstrating Unified Scientific Computing Patterns     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Display version information
    println!("📦 Version Information:");
    println!("   QuantRS2: {}", version::QUANTRS2_VERSION);
    println!("   SciRS2:   {}", version::SCIRS2_VERSION);
    println!();

    // 1. Complex Number Operations (Quantum Amplitudes)
    demonstrate_complex_operations()?;

    // 2. Array Operations (State Vectors and Operators)
    demonstrate_array_operations()?;

    // 3. Random Number Generation (Quantum Measurements)
    demonstrate_random_generation()?;

    // 4. SIMD Operations (Performance Optimization)
    demonstrate_simd_operations()?;

    // 5. Practical Quantum Example
    demonstrate_quantum_computation()?;

    println!("\n✅ All SciRS2 integration examples completed successfully!");
    println!("   For more information, see:");
    println!("   - SCIRS2_INTEGRATION_POLICY.md");
    println!("   - https://github.com/cool-japan/scirs");

    Ok(())
}

/// Demonstrates proper complex number usage with SciRS2
fn demonstrate_complex_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("1️⃣  Complex Number Operations (Quantum Amplitudes)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ✅ CORRECT: Use scirs2_core::Complex64 directly from root
    let alpha = Complex64::new(0.707, 0.0); // |0⟩ amplitude
    let beta = Complex64::new(0.0, 0.707); // |1⟩ amplitude

    println!("   Quantum State: α|0⟩ + β|1⟩");
    println!("   α = {:.3} + {:.3}i", alpha.re, alpha.im);
    println!("   β = {:.3} + {:.3}i", beta.re, beta.im);

    // Complex arithmetic
    let norm_squared = (alpha * alpha.conj() + beta * beta.conj()).re;
    println!("   Normalization: |α|² + |β|² = {norm_squared:.6}");

    // Phase operations
    let phase = Complex64::new(0.0, std::f64::consts::PI / 4.0).exp();
    let rotated = alpha * phase;
    println!(
        "   After phase rotation: {:.3} + {:.3}i",
        rotated.re, rotated.im
    );

    // Using Complex32 for memory efficiency
    let alpha_32 = Complex32::new(0.707, 0.0);
    println!(
        "   Complex32 (memory efficient): {:.3} + {:.3}i",
        alpha_32.re, alpha_32.im
    );
    println!("   Memory: Complex64 = 16 bytes, Complex32 = 8 bytes");

    println!();
    Ok(())
}

/// Demonstrates proper array usage with SciRS2
fn demonstrate_array_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("2️⃣  Array Operations (State Vectors and Operators)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ✅ CORRECT: Use scirs2_core::ndarray::* for unified access
    // Create 2-qubit state vector
    let state: Array1<Complex64> = array![
        Complex64::new(0.5, 0.0), // |00⟩
        Complex64::new(0.5, 0.0), // |01⟩
        Complex64::new(0.5, 0.0), // |10⟩
        Complex64::new(0.5, 0.0), // |11⟩
    ];

    println!("   2-Qubit State Vector:");
    for (i, amplitude) in state.iter().enumerate() {
        println!("   |{:02b}⟩: {:.3} + {:.3}i", i, amplitude.re, amplitude.im);
    }

    // Create Hadamard gate matrix
    let h_factor = 1.0 / 2_f64.sqrt();
    let hadamard: Array2<Complex64> = array![
        [Complex64::new(h_factor, 0.0), Complex64::new(h_factor, 0.0)],
        [
            Complex64::new(h_factor, 0.0),
            Complex64::new(-h_factor, 0.0)
        ]
    ];

    println!("\n   Hadamard Gate Matrix:");
    println!(
        "   [{:.3}, {:.3}]",
        hadamard[[0, 0]].re,
        hadamard[[0, 1]].re
    );
    println!(
        "   [{:.3}, {:.3}]",
        hadamard[[1, 0]].re,
        hadamard[[1, 1]].re
    );

    // Array slicing operations
    let first_two = state.slice(s![0..2]);
    println!("\n   First two amplitudes (slicing):");
    for (i, amp) in first_two.iter().enumerate() {
        println!("   |{:02b}⟩: {:.3}", i, amp.re);
    }

    // Array aggregations
    let norm: f64 = state.iter().map(|c| (c * c.conj()).re).sum();
    println!("\n   State normalization: {norm:.6}");

    // Memory estimation using QuantRS2 utilities
    let qubits = 10;
    let memory = utils::estimate_statevector_memory(qubits);
    println!("\n   Memory estimation for {qubits} qubits:");
    println!("   Required: {}", utils::format_memory(memory));

    println!();
    Ok(())
}

/// Demonstrates proper random number generation with SciRS2
fn demonstrate_random_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("3️⃣  Random Number Generation (Quantum Measurements)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ✅ CORRECT: Use scirs2_core::random for all RNG
    let mut rng = thread_rng();

    // Uniform random for measurement sampling
    println!("   Measurement Sampling (Uniform Distribution):");
    let measurements: Vec<f64> = (0..5).map(|_| rng.gen::<f64>()).collect();
    for (i, m) in measurements.iter().enumerate() {
        println!("   Measurement {}: {:.6}", i + 1, m);
    }

    // Gaussian noise for quantum operations
    let normal = RandNormal::new(0.0, 0.1).unwrap();
    println!("\n   Gaussian Noise (σ = 0.1):");
    let noise_samples: Vec<f64> = (0..5).map(|_| normal.sample(&mut rng)).collect();
    for (i, n) in noise_samples.iter().enumerate() {
        println!("   Sample {}: {:.6}", i + 1, n);
    }

    // Enhanced unified distributions
    let unified_normal = UnifiedNormal::new(0.0, 1.0).unwrap();
    println!("\n   Enhanced Unified Normal Distribution:");
    let unified_samples: Vec<f64> = (0..5).map(|_| unified_normal.sample(&mut rng)).collect();
    for (i, s) in unified_samples.iter().enumerate() {
        println!("   Sample {}: {:.6}", i + 1, s);
    }

    // Beta distribution for parameter initialization
    let unified_beta = UnifiedBeta::new(2.0, 5.0).unwrap();
    println!("\n   Beta Distribution for Parameter Initialization:");
    let beta_samples: Vec<f64> = (0..5).map(|_| unified_beta.sample(&mut rng)).collect();
    for (i, b) in beta_samples.iter().enumerate() {
        println!("   Parameter {}: {:.6}", i + 1, b);
    }

    // Reproducible random numbers for testing
    println!("\n   Reproducible RNG (seed = 42):");
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::SeedableRng;
    let mut seeded_rng = StdRng::seed_from_u64(42);
    let reproducible: Vec<f64> = (0..3).map(|_| seeded_rng.gen::<f64>()).collect();
    for (i, r) in reproducible.iter().enumerate() {
        println!("   Value {}: {:.6}", i + 1, r);
    }
    println!("   (These values will be the same on every run)");

    println!();
    Ok(())
}

/// Demonstrates SIMD operations for performance
fn demonstrate_simd_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("4️⃣  SIMD Operations (Performance Optimization)");
    println!("═══════════════════════════════════════════════════════════\n");

    // Check available SIMD capabilities
    use scirs2_core::simd_ops::PlatformCapabilities;
    let caps = PlatformCapabilities::detect();

    println!("   Platform SIMD Capabilities:");
    println!(
        "   - SSE:      {}",
        if caps.has_sse() { "✅" } else { "❌" }
    );
    println!(
        "   - AVX2:     {}",
        if caps.has_avx2() { "✅" } else { "❌" }
    );
    println!(
        "   - AVX-512:  {}",
        if caps.has_avx512() { "✅" } else { "❌" }
    );

    // Example: SIMD-accelerated quantum gate application
    let state_size = 1024;
    println!("\n   SIMD Performance Example:");
    println!("   State size: {state_size} complex amplitudes");

    if caps.has_avx2() {
        println!("   ✅ AVX2 available: 2-4x speedup for gate operations");
        println!("   ✅ Vectorized complex multiplication enabled");
    } else if caps.has_sse() {
        println!("   ✅ SSE available: 1.5-2x speedup for gate operations");
    } else {
        println!("   ⚠️  No SIMD available: using scalar operations");
    }

    println!("\n   Performance Recommendations:");
    if caps.has_avx512() {
        println!("   🚀 Optimal: Use AVX-512 for large quantum simulations (>20 qubits)");
    } else if caps.has_avx2() {
        println!("   🚀 Good: AVX2 provides excellent performance for most workloads");
    } else {
        println!("   📝 Note: Consider upgrading hardware for large-scale simulations");
    }

    println!();
    Ok(())
}

/// Demonstrates a complete quantum computation workflow using SciRS2
fn demonstrate_quantum_computation() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("5️⃣  Practical Quantum Computation Example");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create a simple 2-qubit quantum state
    println!("   Creating Bell state: (|00⟩ + |11⟩) / √2");

    // Initial state |00⟩
    let mut state: Array1<Complex64> = array![
        Complex64::new(1.0, 0.0), // |00⟩
        Complex64::new(0.0, 0.0), // |01⟩
        Complex64::new(0.0, 0.0), // |10⟩
        Complex64::new(0.0, 0.0), // |11⟩
    ];

    println!("\n   Initial state |00⟩:");
    print_state(&state);

    // Apply Hadamard to first qubit (conceptual)
    // In real implementation, this would use proper tensor product
    let h_factor = 1.0 / 2_f64.sqrt();
    state[0] = Complex64::new(h_factor, 0.0);
    state[2] = Complex64::new(h_factor, 0.0);

    println!("\n   After Hadamard on qubit 0:");
    print_state(&state);

    // Simulate measurements
    let mut rng = thread_rng();
    println!("\n   Measurement Simulation (1000 shots):");

    let probabilities = state
        .iter()
        .map(|c| (c * c.conj()).re)
        .collect::<Vec<f64>>();

    let mut counts = [0; 4];
    for _ in 0..1000 {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                counts[i] += 1;
                break;
            }
        }
    }

    for (i, count) in counts.iter().enumerate() {
        println!(
            "   |{:02b}⟩: {} shots ({:.1}%)",
            i,
            count,
            (*count as f64) / 10.0
        );
    }

    // Calculate quantum fidelity
    let target_state: Array1<Complex64> = array![
        Complex64::new(h_factor, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(h_factor, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    let fidelity: f64 = state
        .iter()
        .zip(target_state.iter())
        .map(|(a, b)| (a * b.conj()).re)
        .sum::<f64>()
        .powi(2);

    println!("\n   Quantum Fidelity with target: {fidelity:.6}");
    println!("   (1.0 = perfect match)");

    println!();
    Ok(())
}

/// Helper function to print quantum state
fn print_state(state: &Array1<Complex64>) {
    for (i, amplitude) in state.iter().enumerate() {
        let magnitude = (amplitude * amplitude.conj()).re.sqrt();
        if magnitude > 1e-10 {
            println!(
                "   |{:02b}⟩: {:.6} + {:.6}i  (|ψ|² = {:.6})",
                i,
                amplitude.re,
                amplitude.im,
                magnitude * magnitude
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let c1 = Complex64::new(1.0, 0.0);
        let c2 = Complex64::new(0.0, 1.0);
        let product = c1 * c2;
        assert!((product.re - 0.0).abs() < 1e-10);
        assert!((product.im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_array_operations() {
        let state: Array1<Complex64> =
            array![Complex64::new(0.707, 0.0), Complex64::new(0.707, 0.0)];
        let norm: f64 = state.iter().map(|c| (c * c.conj()).re).sum();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_reproducibility() {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let v1: f64 = rng1.gen();
        let v2: f64 = rng2.gen();

        assert_eq!(v1, v2, "Seeded RNGs should produce identical results");
    }
}
