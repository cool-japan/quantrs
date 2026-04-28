//! MPS (Matrix Product State) Simulation of a 20-qubit GHZ State
//!
//! GHZ state: (|00...0⟩ + |11...1⟩) / √2
//!
//! The MPS representation is exponentially more memory-efficient than a full
//! state-vector for states with limited entanglement, since it stores tensors of
//! bond dimension χ rather than 2^N complex amplitudes.
//!
//! GHZ has bond dimension χ = 2 — the minimum non-trivial case — making it an
//! ideal benchmark for MPS simulation efficiency.
//!
//! Run with:
//!   cargo run --example mps_ghz -p quantrs2-sim --all-features --features mps

#[cfg(feature = "mps")]
use quantrs2_core::gate::{multi::CNOT, single::Hadamard};
#[cfg(feature = "mps")]
use quantrs2_core::qubit::QubitId;
#[cfg(feature = "mps")]
use quantrs2_sim::mps_enhanced::{EnhancedMPS, MPSConfig};
#[cfg(feature = "mps")]
use std::time::Instant;

/// Build a GHZ state via the public `apply_gate` interface.
/// N is known at compile time; the MPS is built dynamically for `n` sites.
#[cfg(feature = "mps")]
fn build_ghz_mps(n: usize, config: MPSConfig) -> Result<EnhancedMPS, Box<dyn std::error::Error>> {
    let mut mps = EnhancedMPS::new(n, config);

    // H on qubit 0
    mps.apply_gate(&Hadamard { target: QubitId(0) })?;

    // CNOT cascade: (0→1), (1→2), ..., (n-2→n-1)
    for i in 0..(n - 1) {
        mps.apply_gate(&CNOT {
            control: QubitId(i as u32),
            target: QubitId((i + 1) as u32),
        })?;
    }

    Ok(mps)
}

#[cfg(feature = "mps")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 20-Qubit GHZ State via MPS Simulator ===\n");

    let n = 20usize;
    let config = MPSConfig {
        max_bond_dim: 4, // χ=2 is exact for GHZ; 4 allows some margin
        svd_threshold: 1e-12,
        use_randomized_svd: false, // deterministic SVD for correctness
        auto_canonicalize: false,  // skip canonicalization to avoid bond mismatches
        seed: None,
    };

    let start = Instant::now();
    let mut mps = build_ghz_mps(n, config.clone())?;
    let elapsed = start.elapsed();

    println!("Qubits : {n}");
    println!("Runtime: {elapsed:.2?}");

    // Memory comparison
    let mps_approx_bytes = n * 2 * 2 * 16; // rough: N sites × χ × d × 16 bytes/complex
    let sv_bytes = (1usize << n) * 16; // 2^20 complex<f64>
    println!("MPS memory  : ≈ {mps_approx_bytes} bytes");
    println!(
        "SV  memory  : ≈ {} MB  (infeasible on most machines)",
        sv_bytes / (1024 * 1024)
    );

    // ---- Verify amplitudes ----
    let all_zeros: Vec<bool> = vec![false; n];
    let all_ones: Vec<bool> = vec![true; n];

    let amp0 = mps.get_amplitude(&all_zeros)?;
    let amp1 = mps.get_amplitude(&all_ones)?;
    let p0 = amp0.norm_sqr();
    let p1 = amp1.norm_sqr();

    println!("\nAmplitudes:");
    println!(
        "  |{:0>20}⟩ : |amp|² = {p0:.6}  (expected 0.5)",
        0
    );
    println!(
        "  |{:1>20}⟩ : |amp|² = {p1:.6}  (expected 0.5)",
        1
    );

    // ---- Entanglement entropy at the middle cut ----
    // For GHZ: S(n/2) = ln(2) ≈ 0.6931
    let entropy = mps.entanglement_entropy(n / 2)?;
    println!(
        "\nEntanglement entropy (cut at qubit {}): {entropy:.6}  (expected ln2 ≈ 0.6931)",
        n / 2
    );

    // ---- Bond dimensions ----
    let bonds = mps.bond_dimensions();
    let max_bond = mps.max_bond_dimension();
    println!("\nBond dimensions: {bonds:?}");
    println!("Max bond dim   : {max_bond}  (χ=2 is exact for GHZ)");

    // ---- Scaling: smaller GHZ states ----
    println!("\n--- GHZ scaling ---");
    println!(
        "  {:>4}  {:>12}  {:>12}  {:>14}",
        "N", "runtime", "MPS mem (B)", "SV mem (MB)"
    );
    for &n_small in &[5usize, 8, 10, 12] {
        let t = Instant::now();
        let _m = build_ghz_mps(n_small, config.clone())?;
        let dur = t.elapsed();
        let mps_b = n_small * 2 * 2 * 16;
        let sv_mb = (1usize << n_small) * 16 / (1024 * 1024);
        println!("  {n_small:>4}  {dur:>12.2?}  {mps_b:>12}  {sv_mb:>14}");
    }

    // ---- Assertions ----
    assert!(
        (p0 - 0.5).abs() < 1e-5,
        "GHZ |0..0⟩ probability should be 0.5; got {p0}"
    );
    assert!(
        (p1 - 0.5).abs() < 1e-5,
        "GHZ |1..1⟩ probability should be 0.5; got {p1}"
    );
    assert!(
        (entropy - std::f64::consts::LN_2).abs() < 0.05,
        "Entanglement entropy should be ln(2) ≈ 0.6931; got {entropy}"
    );

    println!("\nOK — 20-qubit GHZ state verified via MPS simulator.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mps")]
    run()?;

    #[cfg(not(feature = "mps"))]
    {
        println!("This example requires the 'mps' feature.");
        println!("Run with: cargo run --example mps_ghz -p quantrs2-sim --features mps");
    }

    Ok(())
}
