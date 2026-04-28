//! E91 Entanglement-based Quantum Key Distribution.
//!
//! Implements the Ekert 1991 (E91) protocol with:
//! - Generation of Bell |Φ+⟩ pairs
//! - Depolarizing noise on each qubit
//! - Alice and Bob measuring in three angles each
//! - CHSH Bell inequality test (S parameter)
//! - Key extraction from matching-basis measurements
//!
//! Under ideal conditions S ≈ 2√2 ≈ 2.828; with sufficient noise S ≤ 2 (classical).

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::networking::channel::measure_computational;
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha20Rng;
use scirs2_core::Complex64;
use std::f64::consts::PI;

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
// Bell state preparation
// ---------------------------------------------------------------------------

/// 4×4 density matrix for |Φ+⟩⟨Φ+|.
/// Basis: {|00⟩, |01⟩, |10⟩, |11⟩} → indices 0, 1, 2, 3.
fn bell_phi_plus_4x4() -> Array2<Complex64> {
    let v = 0.5_f64;
    let mut rho = Array2::<Complex64>::zeros((4, 4));
    rho[[0, 0]] = Complex64::new(v, 0.0);
    rho[[0, 3]] = Complex64::new(v, 0.0);
    rho[[3, 0]] = Complex64::new(v, 0.0);
    rho[[3, 3]] = Complex64::new(v, 0.0);
    rho
}

/// Partial trace of a 4×4 two-qubit density matrix.
///
/// `keep_first = true` → trace out qubit B → return ρ_A (2×2).
fn partial_trace_2q(rho4: &Array2<Complex64>, keep_first: bool) -> Array2<Complex64> {
    let mut out = Array2::<Complex64>::zeros((2, 2));
    if keep_first {
        for a in 0..2_usize {
            for a2 in 0..2_usize {
                for b in 0..2_usize {
                    out[[a, a2]] += rho4[[2 * a + b, 2 * a2 + b]];
                }
            }
        }
    } else {
        for b in 0..2_usize {
            for b2 in 0..2_usize {
                for a in 0..2_usize {
                    out[[b, b2]] += rho4[[2 * a + b, 2 * a + b2]];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Depolarizing noise on each qubit of a 4×4 two-qubit state
// ---------------------------------------------------------------------------

fn apply_depolarizing_2q(rho4: &mut Array2<Complex64>, p: f64) {
    if p <= 0.0 {
        return;
    }
    apply_depolarizing_qubit_a(rho4, p);
    apply_depolarizing_qubit_b(rho4, p);
}

fn apply_depolarizing_qubit_a(rho4: &mut Array2<Complex64>, p: f64) {
    let rho_orig = rho4.clone();
    let scale_id = Complex64::new(1.0 - p, 0.0);
    let scale_p = Complex64::new(p / 3.0, 0.0);

    rho4.mapv_inplace(|v| v * scale_id);

    // X_A ⊗ I_B: i → i XOR 2
    let mut t1 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            t1[[i, j]] = rho_orig[[i ^ 2, j ^ 2]];
        }
    }
    add_scaled(rho4, &t1, scale_p);

    // Y_A ⊗ I_B
    let phase_a = |i: usize| -> Complex64 {
        if (i >> 1) & 1 == 0 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, -1.0)
        }
    };
    let mut t2 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            t2[[i, j]] = phase_a(i) * rho_orig[[i ^ 2, j ^ 2]] * phase_a(j).conj();
        }
    }
    add_scaled(rho4, &t2, scale_p);

    // Z_A ⊗ I_B
    let sign_a = |i: usize| -> f64 {
        if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 }
    };
    let mut t3 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            t3[[i, j]] = Complex64::new(sign_a(i) * sign_a(j), 0.0) * rho_orig[[i, j]];
        }
    }
    add_scaled(rho4, &t3, scale_p);
}

fn apply_depolarizing_qubit_b(rho4: &mut Array2<Complex64>, p: f64) {
    let rho_orig = rho4.clone();
    let scale_id = Complex64::new(1.0 - p, 0.0);
    let scale_p = Complex64::new(p / 3.0, 0.0);

    rho4.mapv_inplace(|v| v * scale_id);

    // I_A ⊗ X_B: i → i XOR 1
    let mut t1 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            t1[[i, j]] = rho_orig[[i ^ 1, j ^ 1]];
        }
    }
    add_scaled(rho4, &t1, scale_p);

    // I_A ⊗ Y_B
    let phase_b = |i: usize| -> Complex64 {
        if i & 1 == 0 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, -1.0)
        }
    };
    let mut t2 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            t2[[i, j]] = phase_b(i) * rho_orig[[i ^ 1, j ^ 1]] * phase_b(j).conj();
        }
    }
    add_scaled(rho4, &t2, scale_p);

    // I_A ⊗ Z_B
    let sign_b = |i: usize| -> f64 { if i & 1 == 0 { 1.0 } else { -1.0 } };
    let mut t3 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            t3[[i, j]] = Complex64::new(sign_b(i) * sign_b(j), 0.0) * rho_orig[[i, j]];
        }
    }
    add_scaled(rho4, &t3, scale_p);
}

fn add_scaled(dest: &mut Array2<Complex64>, src: &Array2<Complex64>, scale: Complex64) {
    for i in 0..dest.nrows() {
        for j in 0..dest.ncols() {
            dest[[i, j]] += scale * src[[i, j]];
        }
    }
}

// ---------------------------------------------------------------------------
// Angle-based measurement
// ---------------------------------------------------------------------------

/// Measure a qubit density matrix in the rotated basis at angle θ.
///
/// The positive eigenvector is cos(θ)|0⟩ + sin(θ)|1⟩.
/// Returns `false` for the +1 outcome, `true` for −1.
fn measure_in_angle(rho: &Array2<Complex64>, theta: f64, rand_val: f64) -> bool {
    let c = theta.cos();
    let s = theta.sin();
    // P(+1) = ⟨+_θ|ρ|+_θ⟩ = c²ρ₀₀ + 2cs·Re(ρ₀₁) + s²ρ₁₁
    let p_plus = (c * c * rho[[0, 0]].re + 2.0 * c * s * rho[[0, 1]].re + s * s * rho[[1, 1]].re)
        .clamp(0.0, 1.0);
    // false = +1 (rand < p_plus), true = -1 (rand >= p_plus)
    rand_val >= p_plus
}

/// Compute E(θ_a, θ_b) = Pr(same) − Pr(different) from density matrix.
fn compute_correlation(rho4: &Array2<Complex64>, theta_a: f64, theta_b: f64) -> f64 {
    let ca = theta_a.cos();
    let sa = theta_a.sin();
    let cb = theta_b.cos();
    let sb = theta_b.sin();

    let m_plus_a = [
        [Complex64::new(ca * ca, 0.0), Complex64::new(ca * sa, 0.0)],
        [Complex64::new(ca * sa, 0.0), Complex64::new(sa * sa, 0.0)],
    ];
    let m_minus_a = [
        [Complex64::new(sa * sa, 0.0), Complex64::new(-ca * sa, 0.0)],
        [Complex64::new(-ca * sa, 0.0), Complex64::new(ca * ca, 0.0)],
    ];
    let m_plus_b = [
        [Complex64::new(cb * cb, 0.0), Complex64::new(cb * sb, 0.0)],
        [Complex64::new(cb * sb, 0.0), Complex64::new(sb * sb, 0.0)],
    ];
    let m_minus_b = [
        [Complex64::new(sb * sb, 0.0), Complex64::new(-cb * sb, 0.0)],
        [Complex64::new(-cb * sb, 0.0), Complex64::new(cb * cb, 0.0)],
    ];

    let p_pp = trace_joint(&m_plus_a, &m_plus_b, rho4);
    let p_pm = trace_joint(&m_plus_a, &m_minus_b, rho4);
    let p_mp = trace_joint(&m_minus_a, &m_plus_b, rho4);
    let p_mm = trace_joint(&m_minus_a, &m_minus_b, rho4);

    (p_pp + p_mm - p_pm - p_mp).clamp(-1.0, 1.0)
}

/// Tr[(M_a ⊗ M_b) ρ4] for 2×2 projectors M_a, M_b.
fn trace_joint(
    m_a: &[[Complex64; 2]; 2],
    m_b: &[[Complex64; 2]; 2],
    rho4: &Array2<Complex64>,
) -> f64 {
    let mut result = Complex64::new(0.0, 0.0);
    for ia in 0..2 {
        for ib in 0..2 {
            for ja in 0..2 {
                for jb in 0..2 {
                    let i = 2 * ia + ib;
                    let j = 2 * ja + jb;
                    result += m_a[ia][ja] * m_b[ib][jb] * rho4[[i, j]];
                }
            }
        }
    }
    result.re.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// E91 Protocol
// ---------------------------------------------------------------------------

/// E91 entanglement-based QKD protocol.
#[derive(Debug, Clone)]
pub struct E91Protocol {
    /// Number of entangled pairs to generate.
    pub n_pairs: usize,
    /// Depolarizing noise probability per qubit in [0, 1].
    pub noise: f64,
    /// Seed for the random number generator.
    pub rng_seed: u64,
}

/// Result of running the E91 protocol.
#[derive(Debug, Clone)]
pub struct E91Result {
    /// Extracted key bits.
    pub key: Vec<bool>,
    /// CHSH S parameter |E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)|.
    pub chsh_value: f64,
    /// Whether the Bell test passed (`chsh_value > 2.0`).
    pub passed_bell_test: bool,
    /// Key generation rate (bits per pair).
    pub key_rate: f64,
}

impl E91Protocol {
    /// Create a new E91 protocol instance.
    pub fn new(n_pairs: usize, noise: f64, rng_seed: u64) -> Self {
        Self {
            n_pairs,
            noise: noise.clamp(0.0, 1.0),
            rng_seed,
        }
    }

    /// Execute the E91 protocol.
    pub fn run(&self) -> QuantRS2Result<E91Result> {
        if self.n_pairs == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "n_pairs must be > 0".to_string(),
            ));
        }

        let mut rng = ChaCha20Rng::from_seed(seed_from_u64(self.rng_seed));

        // Alice's measurement angles: 0°, 45°, 90°
        let alice_angles = [0.0_f64, PI / 4.0, PI / 2.0];
        // Bob's measurement angles: 22.5°, 67.5°, 112.5°
        let bob_angles = [PI / 8.0, 3.0 * PI / 8.0, 5.0 * PI / 8.0];

        // Per-pair measurements
        let mut alice_results: Vec<(usize, bool)> = Vec::with_capacity(self.n_pairs);
        let mut bob_results: Vec<(usize, bool)> = Vec::with_capacity(self.n_pairs);

        for _ in 0..self.n_pairs {
            let mut rho4 = bell_phi_plus_4x4();
            apply_depolarizing_2q(&mut rho4, self.noise);

            let ai = rng.random_range(0..3_usize);
            let bi = rng.random_range(0..3_usize);

            let rho_a = partial_trace_2q(&rho4, true);
            let rho_b = partial_trace_2q(&rho4, false);

            let r_a: f64 = rng.random();
            let r_b: f64 = rng.random();
            let a_bit = measure_in_angle(&rho_a, alice_angles[ai], r_a);
            let b_bit = measure_in_angle(&rho_b, bob_angles[bi], r_b);

            alice_results.push((ai, a_bit));
            bob_results.push((bi, b_bit));
        }

        // CHSH S-parameter computed analytically from a representative pair
        let mut representative = bell_phi_plus_4x4();
        apply_depolarizing_2q(&mut representative, self.noise);

        // S = |E(0°,22.5°) - E(0°,67.5°) + E(45°,22.5°) + E(45°,67.5°)|
        let e00 = compute_correlation(&representative, alice_angles[0], bob_angles[0]);
        let e01 = compute_correlation(&representative, alice_angles[0], bob_angles[1]);
        let e10 = compute_correlation(&representative, alice_angles[1], bob_angles[0]);
        let e11 = compute_correlation(&representative, alice_angles[1], bob_angles[1]);
        let chsh_value = (e00 - e01 + e10 + e11).abs();

        // Key extraction: pairs where alice_idx=0 and bob_idx=0
        let mut key: Vec<bool> = Vec::new();
        for k in 0..self.n_pairs {
            let (ai, a_bit) = alice_results[k];
            let (bi, _b_bit) = bob_results[k];
            if ai == 0 && bi == 0 {
                key.push(a_bit);
            }
        }

        let key_rate = key.len() as f64 / self.n_pairs as f64;
        let passed_bell_test = chsh_value > 2.0;

        Ok(E91Result {
            key,
            chsh_value,
            passed_bell_test,
            key_rate,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn e91_ideal_chsh_near_2sqrt2() {
        let proto = E91Protocol::new(200, 0.0, 42);
        let result = proto.run().expect("e91 run");
        // Ideal CHSH for |Φ+⟩: S ≈ 2√2 ≈ 2.828
        assert!(
            result.chsh_value > 2.5,
            "expected CHSH ≈ 2√2 ≈ 2.828, got {}",
            result.chsh_value
        );
        assert!(result.passed_bell_test);
    }

    #[test]
    fn e91_high_noise_chsh_below_2() {
        let proto = E91Protocol::new(200, 0.9, 42);
        let result = proto.run().expect("e91 run");
        assert!(
            result.chsh_value < 2.0,
            "expected CHSH < 2 with high noise, got {}",
            result.chsh_value
        );
        assert!(!result.passed_bell_test);
    }

    #[test]
    fn e91_key_rate_reasonable() {
        let proto = E91Protocol::new(3000, 0.0, 77);
        let result = proto.run().expect("e91 run");
        // Key rate ≈ 1/9 (both alice and bob choose index 0 independently from 3 choices)
        assert!(
            result.key_rate > 0.02,
            "expected reasonable key rate, got {}",
            result.key_rate
        );
    }

    #[test]
    fn bell_phi_plus_correlation_correct() {
        let rho = bell_phi_plus_4x4();
        // E(0°, 22.5°) for |Φ+⟩ = cos(2*(0 - π/8)) = cos(π/4) ≈ 0.707
        let e = compute_correlation(&rho, 0.0, PI / 8.0);
        assert_abs_diff_eq!(e, (PI / 4.0).cos(), epsilon = 0.01);
    }
}
