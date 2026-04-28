//! BB84 Quantum Key Distribution protocol simulation.
//!
//! Implements the Bennett-Brassard 1984 (BB84) QKD protocol with:
//! - Alice's qubit preparation in rectilinear (Z) and diagonal (X) bases
//! - Optional eavesdropping by Eve (intercept-resend attack)
//! - Depolarizing channel noise
//! - Bob's measurement in random basis
//! - Sifting: retain bits where Alice and Bob chose the same basis
//! - Quantum bit error rate (QBER) estimation from a sample
//! - Privacy amplification via XOR hashing to half the sifted key length
//! - Eavesdrop detection: `qber > 0.10`

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::networking::channel::{
    ket_minus, ket_one, ket_plus, ket_zero, measure_computational, pure_state_density,
    DepolarizingChannel, NoiseChannel,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha20Rng;
use scirs2_core::Complex64;
use std::f64::consts::SQRT_2;

// ---------------------------------------------------------------------------
// Helper: convert u64 seed → 32-byte array for ChaCha20
// ---------------------------------------------------------------------------
fn seed_from_u64(seed: u64) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    let s = seed.to_le_bytes();
    bytes[..8].copy_from_slice(&s);
    bytes[8..16].copy_from_slice(&s);
    bytes[16..24].copy_from_slice(&s);
    bytes[24..32].copy_from_slice(&s);
    bytes
}

// ---------------------------------------------------------------------------
// BB84 measurement bases
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bb84Basis {
    /// {|0⟩, |1⟩}
    Rectilinear,
    /// {|+⟩, |−⟩}
    Diagonal,
}

/// Prepare qubit state for given bit and basis.
fn prepare_qubit(bit: bool, basis: Bb84Basis) -> [Complex64; 2] {
    match (bit, basis) {
        (false, Bb84Basis::Rectilinear) => ket_zero(),
        (true, Bb84Basis::Rectilinear) => ket_one(),
        (false, Bb84Basis::Diagonal) => ket_plus(),
        (true, Bb84Basis::Diagonal) => ket_minus(),
    }
}

/// Measure qubit density matrix in given basis; returns the bit.
fn measure_in_basis(rho: &Array2<Complex64>, basis: Bb84Basis, rand_val: f64) -> bool {
    match basis {
        Bb84Basis::Rectilinear => {
            let (outcome, _) = measure_computational(rho, rand_val);
            outcome
        }
        Bb84Basis::Diagonal => {
            // Rotate to Z basis by applying H, then measure.
            // H ρ H†: uses (H ρ H)_ij formula.
            let h = 1.0 / SQRT_2;
            let rho00 = rho[[0, 0]];
            let rho01 = rho[[0, 1]];
            let rho10 = rho[[1, 0]];
            let rho11 = rho[[1, 1]];
            let mut rho_rot = Array2::<Complex64>::zeros((2, 2));
            rho_rot[[0, 0]] = Complex64::new(
                h * h * (rho00 + rho01 + rho10 + rho11).re,
                0.0,
            );
            rho_rot[[1, 1]] = Complex64::new(
                h * h * (rho00 - rho01 - rho10 + rho11).re,
                0.0,
            );
            let (outcome, _) = measure_computational(&rho_rot, rand_val);
            outcome
        }
    }
}

// ---------------------------------------------------------------------------
// BB84 Protocol
// ---------------------------------------------------------------------------

/// BB84 Quantum Key Distribution protocol.
#[derive(Debug, Clone)]
pub struct Bb84Protocol {
    /// Number of raw qubits Alice sends.
    pub n_bits: usize,
    /// Channel depolarizing error probability per qubit in [0, 1].
    pub error_rate: f64,
    /// Fraction of qubits Eve intercepts (intercept-resend attack) in [0, 1].
    pub eavesdrop_rate: f64,
    /// Seed for the random number generator.
    pub rng_seed: u64,
}

/// Result of running the BB84 protocol.
#[derive(Debug, Clone)]
pub struct Bb84Result {
    /// Number of raw qubits sent.
    pub raw_bits: usize,
    /// Sifted key (after basis reconciliation, excluding QBER sample).
    pub sifted_key: Vec<bool>,
    /// Quantum bit error rate estimated from sample.
    pub qber: f64,
    /// Final secret key after privacy amplification.
    pub secret_key: Vec<bool>,
    /// Whether eavesdropping was detected (`qber > 0.10`).
    pub detected_eavesdrop: bool,
}

impl Bb84Protocol {
    /// Create a new BB84 protocol instance.
    pub fn new(n_bits: usize, error_rate: f64, eavesdrop_rate: f64, rng_seed: u64) -> Self {
        Self {
            n_bits,
            error_rate: error_rate.clamp(0.0, 1.0),
            eavesdrop_rate: eavesdrop_rate.clamp(0.0, 1.0),
            rng_seed,
        }
    }

    /// Execute the BB84 protocol and return the result.
    pub fn run(&self) -> QuantRS2Result<Bb84Result> {
        if self.n_bits == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "n_bits must be > 0".to_string(),
            ));
        }

        let mut rng = ChaCha20Rng::from_seed(seed_from_u64(self.rng_seed));
        let depolarizing = DepolarizingChannel::new(self.error_rate);

        // Alice: random bits and bases
        let alice_bits: Vec<bool> = (0..self.n_bits).map(|_| rng.random::<bool>()).collect();
        let alice_bases: Vec<Bb84Basis> = (0..self.n_bits)
            .map(|_| {
                if rng.random::<bool>() {
                    Bb84Basis::Rectilinear
                } else {
                    Bb84Basis::Diagonal
                }
            })
            .collect();

        // Bob: random measurement bases
        let bob_bases: Vec<Bb84Basis> = (0..self.n_bits)
            .map(|_| {
                if rng.random::<bool>() {
                    Bb84Basis::Rectilinear
                } else {
                    Bb84Basis::Diagonal
                }
            })
            .collect();

        // Per-qubit transmission
        let mut bob_bits: Vec<bool> = Vec::with_capacity(self.n_bits);

        for i in 0..self.n_bits {
            let alice_state = prepare_qubit(alice_bits[i], alice_bases[i]);

            // Eve intercept-resend
            let eve_threshold: f64 = rng.random();
            let state_after_eve = if eve_threshold < self.eavesdrop_rate {
                let eve_basis = if rng.random::<bool>() {
                    Bb84Basis::Rectilinear
                } else {
                    Bb84Basis::Diagonal
                };
                let eve_rho = pure_state_density(&alice_state);
                let eve_rand: f64 = rng.random();
                let eve_bit = measure_in_basis(&eve_rho, eve_basis, eve_rand);
                prepare_qubit(eve_bit, eve_basis)
            } else {
                alice_state
            };

            // Channel noise
            let mut rho = pure_state_density(&state_after_eve);
            depolarizing.apply(&mut rho);

            // Bob measures
            let bob_rand: f64 = rng.random();
            bob_bits.push(measure_in_basis(&rho, bob_bases[i], bob_rand));
        }

        // Sifting
        let mut alice_sifted: Vec<bool> = Vec::new();
        let mut bob_sifted: Vec<bool> = Vec::new();
        for i in 0..self.n_bits {
            if alice_bases[i] == bob_bases[i] {
                alice_sifted.push(alice_bits[i]);
                bob_sifted.push(bob_bits[i]);
            }
        }

        if alice_sifted.is_empty() {
            return Ok(Bb84Result {
                raw_bits: self.n_bits,
                sifted_key: vec![],
                qber: 0.0,
                secret_key: vec![],
                detected_eavesdrop: false,
            });
        }

        // QBER estimation on ~20% sample
        let sample_size = (alice_sifted.len() / 5).max(1);
        let errors: usize = (0..sample_size)
            .filter(|&k| alice_sifted[k] != bob_sifted[k])
            .count();
        let qber = errors as f64 / sample_size as f64;

        // Keep remaining sifted bits (exclude sample)
        let sifted_key: Vec<bool> = bob_sifted[sample_size..].to_vec();

        // Privacy amplification
        let secret_key = privacy_amplification(&sifted_key);
        let detected_eavesdrop = qber > 0.10;

        Ok(Bb84Result {
            raw_bits: self.n_bits,
            sifted_key,
            qber,
            secret_key,
            detected_eavesdrop,
        })
    }
}

/// Privacy amplification: XOR neighbouring pairs → half-length key.
fn privacy_amplification(key: &[bool]) -> Vec<bool> {
    let n = key.len() & !1;
    (0..n / 2).map(|i| key[2 * i] ^ key[2 * i + 1]).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bb84_no_noise_no_eve_low_qber() {
        let proto = Bb84Protocol::new(2000, 0.0, 0.0, 7);
        let result = proto.run().expect("bb84 run");
        assert!(
            result.qber < 0.05,
            "expected low QBER without noise, got {}",
            result.qber
        );
        assert!(!result.detected_eavesdrop);
    }

    #[test]
    fn bb84_full_eavesdrop_qber_near_quarter() {
        let proto = Bb84Protocol::new(4000, 0.0, 1.0, 13);
        let result = proto.run().expect("bb84 run");
        // Full intercept-resend attack → QBER ≈ 25%
        assert!(
            result.qber > 0.15,
            "expected QBER ≈ 0.25 with full eavesdropping, got {}",
            result.qber
        );
        assert!(result.detected_eavesdrop);
    }

    #[test]
    fn bb84_secret_key_half_sifted_length() {
        let proto = Bb84Protocol::new(2000, 0.0, 0.0, 55);
        let result = proto.run().expect("bb84 run");
        if !result.sifted_key.is_empty() {
            let expected = result.sifted_key.len() / 2;
            assert_eq!(result.secret_key.len(), expected);
        }
    }
}
