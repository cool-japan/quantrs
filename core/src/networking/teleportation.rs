//! Quantum teleportation and entanglement swapping.
//!
//! Implements single-hop quantum teleportation and an n-hop entanglement
//! swapping chain using density-matrix representation with depolarizing noise.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::networking::channel::{
    fidelity_pure, measure_computational, pure_state_density, DepolarizingChannel, NoiseChannel,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha20Rng;
use scirs2_core::Complex64;
use std::f64::consts::SQRT_2;

// ---------------------------------------------------------------------------
// Helper: convert u64 seed into 32-byte seed array for ChaCha20
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
// Two-qubit depolarizing channel helpers
// ---------------------------------------------------------------------------

/// Apply depolarizing noise independently to each qubit of a 4×4 two-qubit state.
fn apply_noise_2q(rho4: &mut Array2<Complex64>, p: f64) {
    if p <= 0.0 {
        return;
    }
    apply_depolarizing_qubit_top(rho4, p);
    apply_depolarizing_qubit_bot(rho4, p);
}

/// Apply depolarizing(p) to the top qubit (bit 1) of a 4-dim two-qubit density matrix.
fn apply_depolarizing_qubit_top(rho4: &mut Array2<Complex64>, p: f64) {
    let rho_orig = rho4.clone();
    let scale_id = Complex64::new(1.0 - p, 0.0);
    let scale_p = Complex64::new(p / 3.0, 0.0);

    // (1-p) * ρ
    rho4.mapv_inplace(|v| v * scale_id);

    // X_A ⊗ I_B: index i → i XOR 2
    let mut term = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            term[[i, j]] = rho_orig[[i ^ 2, j ^ 2]];
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] += scale_p * term[[i, j]];
        }
    }

    // Y_A ⊗ I_B: phase = i if top bit = 0, else -i
    let phase_top = |i: usize| -> Complex64 {
        if (i >> 1) & 1 == 0 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, -1.0)
        }
    };
    let mut term2 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            term2[[i, j]] = phase_top(i) * rho_orig[[i ^ 2, j ^ 2]] * phase_top(j).conj();
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] += scale_p * term2[[i, j]];
        }
    }

    // Z_A ⊗ I_B: sign = +1 if top bit = 0, else -1
    let sign_top = |i: usize| -> f64 {
        if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 }
    };
    let mut term3 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            term3[[i, j]] = Complex64::new(sign_top(i) * sign_top(j), 0.0) * rho_orig[[i, j]];
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] += scale_p * term3[[i, j]];
        }
    }
}

/// Apply depolarizing(p) to the bottom qubit (bit 0) of a 4-dim two-qubit density matrix.
fn apply_depolarizing_qubit_bot(rho4: &mut Array2<Complex64>, p: f64) {
    let rho_orig = rho4.clone();
    let scale_id = Complex64::new(1.0 - p, 0.0);
    let scale_p = Complex64::new(p / 3.0, 0.0);

    rho4.mapv_inplace(|v| v * scale_id);

    // I_A ⊗ X_B: index i → i XOR 1
    let mut term = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            term[[i, j]] = rho_orig[[i ^ 1, j ^ 1]];
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] += scale_p * term[[i, j]];
        }
    }

    // I_A ⊗ Y_B: phase = i if bottom bit = 0, else -i
    let phase_bot = |i: usize| -> Complex64 {
        if i & 1 == 0 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, -1.0)
        }
    };
    let mut term2 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            term2[[i, j]] = phase_bot(i) * rho_orig[[i ^ 1, j ^ 1]] * phase_bot(j).conj();
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] += scale_p * term2[[i, j]];
        }
    }

    // I_A ⊗ Z_B: sign = +1 if bottom bit = 0, else -1
    let sign_bot = |i: usize| -> f64 { if i & 1 == 0 { 1.0 } else { -1.0 } };
    let mut term3 = Array2::<Complex64>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            term3[[i, j]] = Complex64::new(sign_bot(i) * sign_bot(j), 0.0) * rho_orig[[i, j]];
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] += scale_p * term3[[i, j]];
        }
    }
}

// ---------------------------------------------------------------------------
// Bell state: |Φ+⟩ = (|00⟩+|11⟩)/√2
// ---------------------------------------------------------------------------

/// 4×4 density matrix for the Bell state |Φ+⟩⟨Φ+|.
/// Basis order: 00→0, 01→1, 10→2, 11→3.
fn bell_phi_plus() -> Array2<Complex64> {
    let v = 0.5_f64;
    let mut rho = Array2::<Complex64>::zeros((4, 4));
    rho[[0, 0]] = Complex64::new(v, 0.0);
    rho[[0, 3]] = Complex64::new(v, 0.0);
    rho[[3, 0]] = Complex64::new(v, 0.0);
    rho[[3, 3]] = Complex64::new(v, 0.0);
    rho
}

// ---------------------------------------------------------------------------
// Single-qubit corrections
// ---------------------------------------------------------------------------

/// Apply Pauli-X on a single-qubit 2×2 density matrix.
fn apply_x(rho: &mut Array2<Complex64>) {
    let r00 = rho[[0, 0]];
    let r01 = rho[[0, 1]];
    let r10 = rho[[1, 0]];
    let r11 = rho[[1, 1]];
    rho[[0, 0]] = r11;
    rho[[0, 1]] = r10;
    rho[[1, 0]] = r01;
    rho[[1, 1]] = r00;
}

/// Apply Pauli-Z on a single-qubit 2×2 density matrix.
fn apply_z(rho: &mut Array2<Complex64>) {
    rho[[0, 1]] = -rho[[0, 1]];
    rho[[1, 0]] = -rho[[1, 0]];
}

// ---------------------------------------------------------------------------
// Bell measurement on (ψ, A) within a 3-qubit 8×8 system
// ---------------------------------------------------------------------------

/// Build 8×8 ρ = ρ_ψ ⊗ ρ_AB.
/// Bit ordering: ψ = bit 2 (MSB), A = bit 1, B = bit 0 (LSB).
/// Index i = 4*ψ + 2*A + B.
fn tensor3_density(rho_psi: &Array2<Complex64>, rho_ab: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::<Complex64>::zeros((8, 8));
    for p in 0..2_usize {
        for p2 in 0..2_usize {
            for a in 0..2_usize {
                for a2 in 0..2_usize {
                    for b in 0..2_usize {
                        for b2 in 0..2_usize {
                            let i = 4 * p + 2 * a + b;
                            let j = 4 * p2 + 2 * a2 + b2;
                            out[[i, j]] = rho_psi[[p, p2]] * rho_ab[[2 * a + b, 2 * a2 + b2]];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Apply the Bell-measurement circuit CNOT(ψ→A) then H(ψ) on an 8×8 density matrix.
fn apply_bell_circuit_8x8(rho8: Array2<Complex64>) -> Array2<Complex64> {
    // CNOT: control = ψ (bit 2), target = A (bit 1).
    //   |ψ,a,b⟩ → |ψ, ψ XOR a, b⟩
    //   New index: i' where bit2 unchanged, bit1 flipped iff bit2=1.
    let cnot_perm = |i: usize| -> usize {
        let psi_bit = (i >> 2) & 1;
        if psi_bit == 1 { i ^ 2 } else { i }
    };
    let mut after_cnot = Array2::<Complex64>::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            after_cnot[[cnot_perm(i), cnot_perm(j)]] = rho8[[i, j]];
        }
    }

    // H on ψ (bit 2):
    //   |psi,a,b⟩ → (1/√2) Σ_{p'} (-1)^{psi*p'} |p',a,b⟩
    let h_val = 1.0 / SQRT_2;
    let mut after_h = Array2::<Complex64>::zeros((8, 8));
    for i in 0..8 {
        let pi = (i >> 2) & 1;
        for j in 0..8 {
            let pj = (j >> 2) & 1;
            for pi2 in 0..2_usize {
                for pj2 in 0..2_usize {
                    let phase_i: f64 = if pi == 1 && pi2 == 1 { -1.0 } else { 1.0 };
                    let phase_j: f64 = if pj == 1 && pj2 == 1 { -1.0 } else { 1.0 };
                    let i2 = (i & 3) | (pi2 << 2);
                    let j2 = (j & 3) | (pj2 << 2);
                    after_h[[i2, j2]] +=
                        Complex64::new(h_val * h_val * phase_i * phase_j, 0.0)
                        * after_cnot[[i, j]];
                }
            }
        }
    }
    after_h
}

/// Probability of qubit ψ (bit 2) being |0⟩.
fn prob_qubit_0_in_8x8(rho8: &Array2<Complex64>) -> f64 {
    (0..4_usize).map(|i| rho8[[i, i]].re).sum::<f64>().clamp(0.0, 1.0)
}

/// Project qubit ψ (bit 2) onto |m⟩ and renormalise.
fn project_qubit_2_in_8x8(rho8: &Array2<Complex64>, m: bool) -> Array2<Complex64> {
    let start = if m { 4 } else { 0 };
    let end = start + 4;
    let prob: f64 = (start..end).map(|i| rho8[[i, i]].re).sum::<f64>().max(1e-15);
    let mut out = Array2::<Complex64>::zeros((8, 8));
    for i in start..end {
        for j in start..end {
            out[[i, j]] = rho8[[i, j]] / Complex64::new(prob, 0.0);
        }
    }
    out
}

/// Probability of qubit A (bit 1) being |0⟩.
fn prob_qubit_1_in_8x8(rho8: &Array2<Complex64>) -> f64 {
    (0..8_usize)
        .filter(|&i| (i & 2) == 0)
        .map(|i| rho8[[i, i]].re)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Project qubit A (bit 1) onto |m⟩ and renormalise.
fn project_qubit_1_in_8x8(rho8: &Array2<Complex64>, m: bool) -> Array2<Complex64> {
    let bit_val = if m { 2 } else { 0 };
    let prob: f64 = (0..8_usize)
        .filter(|&i| (i & 2) == bit_val)
        .map(|i| rho8[[i, i]].re)
        .sum::<f64>()
        .max(1e-15);
    let mut out = Array2::<Complex64>::zeros((8, 8));
    for i in 0..8 {
        if (i & 2) == bit_val {
            for j in 0..8 {
                if (j & 2) == bit_val {
                    out[[i, j]] = rho8[[i, j]] / Complex64::new(prob, 0.0);
                }
            }
        }
    }
    out
}

/// Trace out qubits ψ (bit 2) and A (bit 1); return reduced density matrix of B (bit 0).
fn partial_trace_to_qubit_b(rho8: &Array2<Complex64>) -> Array2<Complex64> {
    let mut rho2 = Array2::<Complex64>::zeros((2, 2));
    for b in 0..2_usize {
        for b2 in 0..2_usize {
            for psi in 0..2_usize {
                for a in 0..2_usize {
                    let i = 4 * psi + 2 * a + b;
                    let j = 4 * psi + 2 * a + b2;
                    rho2[[b, b2]] += rho8[[i, j]];
                }
            }
        }
    }
    rho2
}

/// Perform a Bell-basis measurement on (ψ, A) and return classical bits + Bob's state.
fn bell_measure_and_correct(
    psi: &[Complex64; 2],
    rho_ab: &Array2<Complex64>,
    rng: &mut ChaCha20Rng,
) -> (bool, bool, Array2<Complex64>) {
    let rho_in = pure_state_density(psi);
    let rho8 = tensor3_density(&rho_in, rho_ab);
    let rho8_after = apply_bell_circuit_8x8(rho8);

    // Measure qubit ψ
    let p0_psi = prob_qubit_0_in_8x8(&rho8_after);
    let r0: f64 = rng.random();
    let m0 = r0 >= p0_psi;
    let rho8_m0 = project_qubit_2_in_8x8(&rho8_after, m0);

    // Measure qubit A
    let p0_a = prob_qubit_1_in_8x8(&rho8_m0);
    let r1: f64 = rng.random();
    let m1 = r1 >= p0_a;
    let rho8_m1 = project_qubit_1_in_8x8(&rho8_m0, m1);

    let rho_b = partial_trace_to_qubit_b(&rho8_m1);
    (m0, m1, rho_b)
}

// ---------------------------------------------------------------------------
// TeleportationProtocol
// ---------------------------------------------------------------------------

/// Single-hop quantum teleportation protocol.
#[derive(Debug, Clone)]
pub struct TeleportationProtocol {
    /// Depolarizing noise probability applied to each qubit of the resource Bell pair.
    pub noise: f64,
    /// Seed for the random number generator.
    pub rng_seed: u64,
}

/// Result of a single teleportation run.
#[derive(Debug, Clone)]
pub struct TeleportationResult {
    /// Fidelity between the teleported state and the original input state.
    pub fidelity: f64,
    /// Classical correction bits sent from Alice to Bob: (m0, m1).
    pub correction_bits: (bool, bool),
}

impl TeleportationProtocol {
    /// Create a new teleportation protocol.
    pub fn new(noise: f64, rng_seed: u64) -> Self {
        Self {
            noise: noise.clamp(0.0, 1.0),
            rng_seed,
        }
    }

    /// Run the teleportation protocol for the given input state.
    ///
    /// `state` must be a normalised 2-component complex vector.
    pub fn teleport(&self, state: [Complex64; 2]) -> QuantRS2Result<TeleportationResult> {
        let norm_sq = state[0].norm_sqr() + state[1].norm_sqr();
        if (norm_sq - 1.0).abs() > 0.05 {
            return Err(QuantRS2Error::InvalidInput(
                "Input state must be normalised".to_string(),
            ));
        }

        let mut rng = ChaCha20Rng::from_seed(seed_from_u64(self.rng_seed));

        // Prepare resource Bell pair |Φ+⟩ with noise
        let mut rho_ab = bell_phi_plus();
        apply_noise_2q(&mut rho_ab, self.noise);

        // Bell measurement and classical corrections
        let (m0, m1, mut rho_b) = bell_measure_and_correct(&state, &rho_ab, &mut rng);

        if m1 {
            apply_x(&mut rho_b);
        }
        if m0 {
            apply_z(&mut rho_b);
        }

        let fidelity = fidelity_pure(&state, &rho_b);

        Ok(TeleportationResult {
            fidelity,
            correction_bits: (m0, m1),
        })
    }
}

// ---------------------------------------------------------------------------
// EntanglementSwapping
// ---------------------------------------------------------------------------

/// Multi-hop entanglement swapping chain.
///
/// Topology: A — B₁ — B₂ — ... — B_{n-1} — C
/// Each link is a Bell pair with independent depolarizing noise.
#[derive(Debug, Clone)]
pub struct EntanglementSwapping {
    /// Number of hops (links). 1 = direct Bell pair; 2 = one relay; etc.
    pub n_hops: usize,
    /// Depolarizing noise probability per qubit per hop.
    pub noise_per_link: f64,
    /// Seed for RNG.
    pub rng_seed: u64,
}

/// Result of the entanglement swapping protocol.
#[derive(Debug, Clone)]
pub struct SwappingResult {
    /// End-to-end fidelity of the final A-C Bell pair.
    pub end_to_end_fidelity: f64,
}

impl EntanglementSwapping {
    /// Create a new entanglement swapping chain.
    pub fn new(n_hops: usize, noise_per_link: f64, rng_seed: u64) -> QuantRS2Result<Self> {
        if n_hops == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "n_hops must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            n_hops,
            noise_per_link: noise_per_link.clamp(0.0, 1.0),
            rng_seed,
        })
    }

    /// Run the entanglement swapping protocol.
    pub fn run(&self) -> QuantRS2Result<SwappingResult> {
        let mut rng = ChaCha20Rng::from_seed(seed_from_u64(self.rng_seed));

        // Prepare all n_hops Bell pairs with noise
        let mut links: Vec<Array2<Complex64>> = (0..self.n_hops)
            .map(|_| {
                let mut rho = bell_phi_plus();
                apply_noise_2q(&mut rho, self.noise_per_link);
                rho
            })
            .collect();

        // Merge links one by one via entanglement swapping
        while links.len() > 1 {
            let rho_left = links.remove(0);
            let rho_right = links.remove(0);
            let rho_ac = bell_swap_4qubit(&rho_left, &rho_right, &mut rng);
            links.insert(0, rho_ac);
        }

        let rho_ac = &links[0];
        let fidelity = bell_state_fidelity(rho_ac);

        Ok(SwappingResult {
            end_to_end_fidelity: fidelity,
        })
    }
}

// ---------------------------------------------------------------------------
// Bell measurement for entanglement swapping (4-qubit → 2-qubit)
// ---------------------------------------------------------------------------

/// Tensor product of two 4×4 density matrices → 16×16.
/// Index ordering: left-pair (bits 3,2), right-pair (bits 1,0).
fn tensor_product_4qubit(
    rho_left: &Array2<Complex64>,
    rho_right: &Array2<Complex64>,
) -> Array2<Complex64> {
    let mut out = Array2::<Complex64>::zeros((16, 16));
    for ab in 0..4 {
        for ab2 in 0..4 {
            for cd in 0..4 {
                for cd2 in 0..4 {
                    let i = 4 * ab + cd;
                    let j = 4 * ab2 + cd2;
                    out[[i, j]] = rho_left[[ab, ab2]] * rho_right[[cd, cd2]];
                }
            }
        }
    }
    out
}

/// Apply CNOT(qubit1→qubit2) then H(qubit1) on a 4-qubit 16×16 density matrix.
/// Qubit indexing (from MSB): qubit0=bit3, qubit1=bit2, qubit2=bit1, qubit3=bit0.
fn apply_bell_circuit_on_middle_qubits(rho: Array2<Complex64>) -> Array2<Complex64> {
    // CNOT: control = qubit1 (bit2), target = qubit2 (bit1)
    let cnot = |i: usize| -> usize {
        let bit2 = (i >> 2) & 1;
        if bit2 == 1 { i ^ 2 } else { i }
    };
    let mut after_cnot = Array2::<Complex64>::zeros((16, 16));
    for i in 0..16 {
        for j in 0..16 {
            after_cnot[[cnot(i), cnot(j)]] = rho[[i, j]];
        }
    }

    // H on qubit1 (bit2)
    let h = 1.0 / SQRT_2;
    let mut after_h = Array2::<Complex64>::zeros((16, 16));
    for i in 0..16 {
        let pi = (i >> 2) & 1;
        for j in 0..16 {
            let pj = (j >> 2) & 1;
            for pi2 in 0..2_usize {
                for pj2 in 0..2_usize {
                    let phi: f64 = if pi == 1 && pi2 == 1 { -1.0 } else { 1.0 };
                    let phj: f64 = if pj == 1 && pj2 == 1 { -1.0 } else { 1.0 };
                    let i2 = (i & !4) | (pi2 << 2);
                    let j2 = (j & !4) | (pj2 << 2);
                    after_h[[i2, j2]] +=
                        Complex64::new(h * h * phi * phj, 0.0) * after_cnot[[i, j]];
                }
            }
        }
    }
    after_h
}

/// Probability of qubit k (bit k from LSB) being |0⟩ in a 16-dim system.
fn prob_qubit_k_16(rho16: &Array2<Complex64>, k: usize) -> f64 {
    (0..16_usize)
        .filter(|&i| (i >> k) & 1 == 0)
        .map(|i| rho16[[i, i]].re)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Project qubit k in a 16-dim system onto |m⟩ and renormalise.
fn project_qubit_k_16(rho16: &Array2<Complex64>, k: usize, m: bool) -> Array2<Complex64> {
    let bit_val = if m { 1 << k } else { 0 };
    let mask = 1 << k;
    let prob: f64 = (0..16_usize)
        .filter(|&i| (i & mask) == bit_val)
        .map(|i| rho16[[i, i]].re)
        .sum::<f64>()
        .max(1e-15);
    let mut out = Array2::<Complex64>::zeros((16, 16));
    for i in 0..16 {
        if (i & mask) == bit_val {
            for j in 0..16 {
                if (j & mask) == bit_val {
                    out[[i, j]] = rho16[[i, j]] / Complex64::new(prob, 0.0);
                }
            }
        }
    }
    out
}

/// Trace out qubits 1 and 2 (bits 2 and 1) from a 16-dim system → 4-dim ρ_{A,C}.
/// Remaining: qubit0 (bit3) and qubit3 (bit0) → index 2*a_bit + c_bit.
fn partial_trace_middle_qubits(rho16: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::<Complex64>::zeros((4, 4));
    for a in 0..2_usize {
        for c in 0..2_usize {
            for a2 in 0..2_usize {
                for c2 in 0..2_usize {
                    let out_i = 2 * a + c;
                    let out_j = 2 * a2 + c2;
                    for b in 0..2_usize {
                        for d in 0..2_usize {
                            let i = (a << 3) | (b << 2) | (d << 1) | c;
                            let j = (a2 << 3) | (b << 2) | (d << 1) | c2;
                            out[[out_i, out_j]] += rho16[[i, j]];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Apply X on the right (bottom) qubit of a 4-dim two-qubit density matrix.
fn apply_x_bot(rho4: &mut Array2<Complex64>) {
    let orig = rho4.clone();
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] = orig[[i ^ 1, j ^ 1]];
        }
    }
}

/// Apply Z on the right (bottom) qubit of a 4-dim two-qubit density matrix.
fn apply_z_bot(rho4: &mut Array2<Complex64>) {
    let sign = |i: usize| -> f64 { if i & 1 == 0 { 1.0 } else { -1.0 } };
    for i in 0..4 {
        for j in 0..4 {
            rho4[[i, j]] *= Complex64::new(sign(i) * sign(j), 0.0);
        }
    }
}

/// Bell-measure the two middle qubits and return ρ_{A,C}.
fn bell_swap_4qubit(
    rho_left: &Array2<Complex64>,
    rho_right: &Array2<Complex64>,
    rng: &mut ChaCha20Rng,
) -> Array2<Complex64> {
    let rho16 = tensor_product_4qubit(rho_left, rho_right);
    let rho16_bell = apply_bell_circuit_on_middle_qubits(rho16);

    // Measure qubit1 (bit2 = Bl)
    let p0_bl = prob_qubit_k_16(&rho16_bell, 2);
    let r_bl: f64 = rng.random();
    let m_bl = r_bl >= p0_bl;
    let rho16_mbl = project_qubit_k_16(&rho16_bell, 2, m_bl);

    // Measure qubit2 (bit1 = Br)
    let p0_br = prob_qubit_k_16(&rho16_mbl, 1);
    let r_br: f64 = rng.random();
    let m_br = r_br >= p0_br;
    let rho16_mbr = project_qubit_k_16(&rho16_mbl, 1, m_br);

    let mut rho_ac = partial_trace_middle_qubits(&rho16_mbr);

    // Apply corrections to C (right qubit): if m_br: X; if m_bl: Z
    if m_br {
        apply_x_bot(&mut rho_ac);
    }
    if m_bl {
        apply_z_bot(&mut rho_ac);
    }

    rho_ac
}

/// Fidelity of a 4×4 density matrix with |Φ+⟩.
/// F = ⟨Φ+|ρ|Φ+⟩ = (ρ\[0,0\] + ρ\[0,3\] + ρ\[3,0\] + ρ\[3,3\]) / 2.
pub fn bell_state_fidelity(rho4: &Array2<Complex64>) -> f64 {
    let f = 0.5 * (rho4[[0, 0]].re + rho4[[0, 3]].re + rho4[[3, 0]].re + rho4[[3, 3]].re);
    f.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn normalised_state(alpha: Complex64, beta: Complex64) -> [Complex64; 2] {
        let n = (alpha.norm_sqr() + beta.norm_sqr()).sqrt();
        [alpha / n, beta / n]
    }

    #[test]
    fn teleportation_no_noise_fidelity_one() {
        let psi = normalised_state(
            Complex64::new(1.0 / SQRT_2, 0.0),
            Complex64::new(0.0, 1.0 / SQRT_2),
        );
        let proto = TeleportationProtocol::new(0.0, 42);
        let result = proto.teleport(psi).expect("teleport");
        assert!(
            result.fidelity > 0.98,
            "expected fidelity ≈ 1, got {}",
            result.fidelity
        );
    }

    #[test]
    fn teleportation_noisy_fidelity_decreases() {
        let psi = normalised_state(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let proto_clean = TeleportationProtocol::new(0.0, 42);
        let proto_noisy = TeleportationProtocol::new(0.3, 42);
        let f_clean = proto_clean.teleport(psi).expect("teleport").fidelity;
        let f_noisy = proto_noisy.teleport(psi).expect("teleport").fidelity;
        assert!(
            f_clean > f_noisy,
            "clean fidelity ({}) should exceed noisy ({})",
            f_clean,
            f_noisy
        );
    }

    #[test]
    fn entanglement_swapping_single_hop_no_noise() {
        let swap = EntanglementSwapping::new(1, 0.0, 42).expect("create");
        let result = swap.run().expect("run");
        assert!(
            result.end_to_end_fidelity > 0.95,
            "expected high fidelity for 1 hop noiseless, got {}",
            result.end_to_end_fidelity
        );
    }

    #[test]
    fn entanglement_swapping_n_hops_degrades() {
        let swap1 = EntanglementSwapping::new(1, 0.05, 42).expect("create");
        let swap3 = EntanglementSwapping::new(3, 0.05, 42).expect("create");
        let f1 = swap1.run().expect("run").end_to_end_fidelity;
        let f3 = swap3.run().expect("run").end_to_end_fidelity;
        assert!(
            f1 > f3,
            "fidelity should degrade with more hops: 1-hop={}, 3-hop={}",
            f1,
            f3
        );
    }
}
