//! Quantum channel models for networking protocols.
//!
//! Provides Kraus-operator-based single-qubit channel primitives
//! (depolarizing, dephasing, amplitude damping) used by BB84, E91, and
//! quantum teleportation simulations.

use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::f64::consts::SQRT_2;

/// A single-qubit quantum channel acting on a density matrix ρ.
///
/// Channels are applied in-place via Kraus decomposition:
/// ρ → Σ_k  K_k ρ K_k†
pub trait NoiseChannel: Send + Sync {
    /// Apply the channel in-place on a 2×2 density matrix.
    fn apply(&self, rho: &mut Array2<Complex64>);
}

// ---------------------------------------------------------------------------
// Helper: compute  K rho K†  and accumulate into `out`
// ---------------------------------------------------------------------------
fn kraus_act(k: &[[Complex64; 2]; 2], rho: &Array2<Complex64>, out: &mut Array2<Complex64>) {
    // k_rho[i,j] = Σ_l k[i][l] * rho[l,j]
    let mut k_rho = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for l in 0..2 {
                k_rho[i][j] += k[i][l] * rho[[l, j]];
            }
        }
    }
    // out += k_rho * k†  ↔  out[i,j] += Σ_l k_rho[i][l] * conj(k[j][l])
    for i in 0..2 {
        for j in 0..2 {
            for l in 0..2 {
                out[[i, j]] += k_rho[i][l] * k[j][l].conj();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pauli matrices as [[Complex64; 2]; 2]
// ---------------------------------------------------------------------------
fn pauli_x() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_y() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]
}

fn pauli_z() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]
}

// ---------------------------------------------------------------------------
// Depolarizing channel:  ρ → (1−p)ρ + (p/3)(XρX + YρY + ZρZ)
// ---------------------------------------------------------------------------

/// Depolarizing channel with probability `p`.
///
/// `p = 0`: identity; `p = 3/4`: fully mixed output for any input.
#[derive(Debug, Clone)]
pub struct DepolarizingChannel {
    /// Error probability in [0, 1].
    pub p: f64,
}

impl DepolarizingChannel {
    /// Create a new depolarizing channel.
    pub fn new(p: f64) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }
}

impl NoiseChannel for DepolarizingChannel {
    fn apply(&self, rho: &mut Array2<Complex64>) {
        if self.p == 0.0 {
            return;
        }
        let p = self.p;
        let identity_weight = Complex64::new(1.0 - p, 0.0);
        let pauli_weight = Complex64::new(p / 3.0, 0.0);

        let rho_orig = rho.clone();

        // Start with scaled identity term
        rho.mapv_inplace(|v| v * identity_weight);

        let paulis = [pauli_x(), pauli_y(), pauli_z()];
        for pauli in &paulis {
            let mut term = Array2::<Complex64>::zeros((2, 2));
            kraus_act(pauli, &rho_orig, &mut term);
            for i in 0..2 {
                for j in 0..2 {
                    rho[[i, j]] += pauli_weight * term[[i, j]];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dephasing channel:  ρ → (1−p)ρ + p·ZρZ
// ---------------------------------------------------------------------------

/// Dephasing (phase-flip) channel with probability `p`.
#[derive(Debug, Clone)]
pub struct DephazingChannel {
    /// Dephasing probability in [0, 1].
    pub p: f64,
}

impl DephazingChannel {
    /// Create a new dephasing channel.
    pub fn new(p: f64) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }
}

impl NoiseChannel for DephazingChannel {
    fn apply(&self, rho: &mut Array2<Complex64>) {
        if self.p == 0.0 {
            return;
        }
        let p = self.p;
        let z = pauli_z();

        let rho_orig = rho.clone();
        rho.mapv_inplace(|v| v * Complex64::new(1.0 - p, 0.0));

        let mut z_term = Array2::<Complex64>::zeros((2, 2));
        kraus_act(&z, &rho_orig, &mut z_term);
        for i in 0..2 {
            for j in 0..2 {
                rho[[i, j]] += Complex64::new(p, 0.0) * z_term[[i, j]];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Amplitude damping channel:  K0 = [[1,0],[0,√(1-γ)]],  K1 = [[0,√γ],[0,0]]
// ---------------------------------------------------------------------------

/// Amplitude damping channel (spontaneous emission) with decay rate `gamma`.
#[derive(Debug, Clone)]
pub struct AmplitudeDampingChannel {
    /// Decay probability γ in [0, 1].
    pub gamma: f64,
}

impl AmplitudeDampingChannel {
    /// Create a new amplitude damping channel.
    pub fn new(gamma: f64) -> Self {
        Self { gamma: gamma.clamp(0.0, 1.0) }
    }
}

impl NoiseChannel for AmplitudeDampingChannel {
    fn apply(&self, rho: &mut Array2<Complex64>) {
        if self.gamma == 0.0 {
            return;
        }
        let gamma = self.gamma;
        let sqrt_1mg = (1.0 - gamma).sqrt();
        let sqrt_g = gamma.sqrt();

        let k0: [[Complex64; 2]; 2] = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(sqrt_1mg, 0.0)],
        ];
        let k1: [[Complex64; 2]; 2] = [
            [Complex64::new(0.0, 0.0), Complex64::new(sqrt_g, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        ];

        let rho_orig = rho.clone();
        rho.mapv_inplace(|_| Complex64::new(0.0, 0.0));
        kraus_act(&k0, &rho_orig, rho);
        kraus_act(&k1, &rho_orig, rho);
    }
}

// ---------------------------------------------------------------------------
// Pure-state helpers
// ---------------------------------------------------------------------------

/// Build the density matrix ρ = |ψ⟩⟨ψ| from a normalised 2-component state.
pub fn pure_state_density(psi: &[Complex64; 2]) -> Array2<Complex64> {
    let mut rho = Array2::<Complex64>::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            rho[[i, j]] = psi[i] * psi[j].conj();
        }
    }
    rho
}

/// Compute the fidelity F = ⟨ψ|ρ|ψ⟩ for a pure target state |ψ⟩.
pub fn fidelity_pure(psi: &[Complex64; 2], rho: &Array2<Complex64>) -> f64 {
    // F = Σ_{i,j} conj(psi[i]) * rho[i,j] * psi[j]
    let mut f = Complex64::new(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            f += psi[i].conj() * rho[[i, j]] * psi[j];
        }
    }
    f.re.clamp(0.0, 1.0)
}

/// |0⟩ state
pub fn ket_zero() -> [Complex64; 2] {
    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
}

/// |1⟩ state
pub fn ket_one() -> [Complex64; 2] {
    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
}

/// |+⟩ = (|0⟩+|1⟩)/√2 state
pub fn ket_plus() -> [Complex64; 2] {
    let v = 1.0 / SQRT_2;
    [Complex64::new(v, 0.0), Complex64::new(v, 0.0)]
}

/// |−⟩ = (|0⟩−|1⟩)/√2 state
pub fn ket_minus() -> [Complex64; 2] {
    let v = 1.0 / SQRT_2;
    [Complex64::new(v, 0.0), Complex64::new(-v, 0.0)]
}

/// Measure a single qubit in the computational basis; returns (outcome, collapsed density matrix).
///
/// `outcome = false` → |0⟩, `outcome = true` → |1⟩.
/// The random choice is driven by the Born rule probability of each outcome.
pub fn measure_computational(rho: &Array2<Complex64>, r: f64) -> (bool, Array2<Complex64>) {
    // Probability of |0⟩
    let p0 = rho[[0, 0]].re.clamp(0.0, 1.0);
    let outcome = r >= p0;
    let mut collapsed = Array2::<Complex64>::zeros((2, 2));
    if !outcome {
        // Project onto |0⟩: Π_0 ρ Π_0 / p0
        if p0 > 1e-15 {
            collapsed[[0, 0]] = Complex64::new(1.0, 0.0);
        }
    } else {
        let p1 = (1.0 - p0).max(0.0);
        if p1 > 1e-15 {
            collapsed[[1, 1]] = Complex64::new(1.0, 0.0);
        }
    }
    (outcome, collapsed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn identity_rho() -> Array2<Complex64> {
        let mut rho = Array2::<Complex64>::zeros((2, 2));
        rho[[0, 0]] = Complex64::new(0.5, 0.0);
        rho[[1, 1]] = Complex64::new(0.5, 0.0);
        rho
    }

    #[test]
    fn depolarizing_zero_is_identity() {
        let ch = DepolarizingChannel::new(0.0);
        let mut rho = pure_state_density(&ket_zero());
        let orig = rho.clone();
        ch.apply(&mut rho);
        assert_abs_diff_eq!(rho[[0, 0]].re, orig[[0, 0]].re, epsilon = 1e-12);
        assert_abs_diff_eq!(rho[[1, 1]].re, orig[[1, 1]].re, epsilon = 1e-12);
    }

    #[test]
    fn depolarizing_fully_mixed_stays_mixed() {
        let ch = DepolarizingChannel::new(0.75);
        let mut rho = identity_rho();
        ch.apply(&mut rho);
        // Fully mixed state is a fixed point of depolarizing channel
        assert_abs_diff_eq!(rho[[0, 0]].re, 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(rho[[1, 1]].re, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn amplitude_damping_relaxes_to_ground() {
        let ch = AmplitudeDampingChannel::new(1.0);
        let mut rho = pure_state_density(&ket_one());
        ch.apply(&mut rho);
        assert_abs_diff_eq!(rho[[0, 0]].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rho[[1, 1]].re, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn dephasing_zero_is_identity() {
        let ch = DephazingChannel::new(0.0);
        let mut rho = pure_state_density(&ket_plus());
        let orig = rho.clone();
        ch.apply(&mut rho);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(rho[[i, j]].re, orig[[i, j]].re, epsilon = 1e-12);
            }
        }
    }
}
