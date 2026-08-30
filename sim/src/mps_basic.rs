//! Basic MPS simulator implementation without external linear algebra dependencies
//!
//! This provides a simplified MPS implementation that doesn't require ndarray-linalg

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    register::Register,
};
use scirs2_core::ndarray::{array, s, Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{thread_rng, Rng};
use scirs2_core::Complex64;
use std::f64::consts::SQRT_2;

/// Configuration for basic MPS simulator
#[derive(Debug, Clone)]
pub struct BasicMPSConfig {
    /// Maximum allowed bond dimension
    pub max_bond_dim: usize,
    /// SVD truncation threshold
    pub svd_threshold: f64,
}

impl Default for BasicMPSConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            svd_threshold: 1e-10,
        }
    }
}

/// MPS tensor for a single qubit
#[derive(Debug, Clone)]
struct MPSTensor {
    /// The tensor data: `left_bond` x physical x `right_bond`
    data: Array3<Complex64>,
}

impl MPSTensor {
    /// Create initial tensor for |0> state
    fn zero_state(position: usize, num_qubits: usize) -> Self {
        let is_first = position == 0;
        let is_last = position == num_qubits - 1;

        let data = if is_first && is_last {
            // Single qubit: 1x2x1 tensor
            let mut tensor = Array3::zeros((1, 2, 1));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        } else if is_first {
            // First qubit: 1x2x2 tensor
            let mut tensor = Array3::zeros((1, 2, 2));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        } else if is_last {
            // Last qubit: 2x2x1 tensor
            let mut tensor = Array3::zeros((2, 2, 1));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        } else {
            // Middle qubit: 2x2x2 tensor
            let mut tensor = Array3::zeros((2, 2, 2));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        };
        Self { data }
    }
}

/// Basic Matrix Product State representation
pub struct BasicMPS {
    /// MPS tensors for each qubit
    tensors: Vec<MPSTensor>,
    /// Number of qubits
    num_qubits: usize,
    /// Configuration
    config: BasicMPSConfig,
}

impl BasicMPS {
    /// Create a new MPS in the |0...0> state
    #[must_use]
    pub fn new(num_qubits: usize, config: BasicMPSConfig) -> Self {
        let tensors = (0..num_qubits)
            .map(|i| MPSTensor::zero_state(i, num_qubits))
            .collect();

        Self {
            tensors,
            num_qubits,
            config,
        }
    }

    /// Apply a single-qubit gate
    pub fn apply_single_qubit_gate(
        &mut self,
        gate_matrix: &Array2<Complex64>,
        qubit: usize,
    ) -> QuantRS2Result<()> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        let tensor = &mut self.tensors[qubit];
        let shape = tensor.data.shape();
        let (left_dim, _, right_dim) = (shape[0], shape[1], shape[2]);

        let mut new_data = Array3::zeros((left_dim, 2, right_dim));

        // Apply gate to physical index
        for l in 0..left_dim {
            for r in 0..right_dim {
                for new_phys in 0..2 {
                    for old_phys in 0..2 {
                        new_data[[l, new_phys, r]] +=
                            gate_matrix[[new_phys, old_phys]] * tensor.data[[l, old_phys, r]];
                    }
                }
            }
        }

        tensor.data = new_data;
        Ok(())
    }

    /// Apply a two-qubit gate to adjacent qubits
    pub fn apply_two_qubit_gate(
        &mut self,
        gate_matrix: &Array2<Complex64>,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<()> {
        if (qubit1 as i32 - qubit2 as i32).abs() != 1 {
            return Err(QuantRS2Error::InvalidInput(
                "MPS requires adjacent qubits for two-qubit gates".to_string(),
            ));
        }

        let (left_q, right_q) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        // Simple implementation: contract and re-decompose
        // This is not optimal but works for demonstration

        let left_shape = self.tensors[left_q].data.shape().to_vec();
        let right_shape = self.tensors[right_q].data.shape().to_vec();

        // Contract the two tensors
        let mut combined = Array3::<Complex64>::zeros((left_shape[0], 4, right_shape[2]));

        for l in 0..left_shape[0] {
            for r in 0..right_shape[2] {
                for i in 0..2 {
                    for j in 0..2 {
                        for m in 0..left_shape[2] {
                            combined[[l, i * 2 + j, r]] += self.tensors[left_q].data[[l, i, m]]
                                * self.tensors[right_q].data[[m, j, r]];
                        }
                    }
                }
            }
        }

        // Apply gate
        let mut result = Array3::<Complex64>::zeros((left_shape[0], 4, right_shape[2]));
        for l in 0..left_shape[0] {
            for r in 0..right_shape[2] {
                for out_idx in 0..4 {
                    for in_idx in 0..4 {
                        result[[l, out_idx, r]] +=
                            gate_matrix[[out_idx, in_idx]] * combined[[l, in_idx, r]];
                    }
                }
            }
        }

        // Split the gated two-site tensor back into two MPS tensors using a genuine SVD.
        // Reshape the (left, 4, right) tensor into a matrix M[(left, s_left), (s_right, right)]
        // where the physical indices are split between the two sites, then truncate to the
        // largest singular values. Anything less than a real SVD silently destroys the
        // entanglement the gate just created.
        let left_dim = left_shape[0];
        let right_dim = right_shape[2];
        let mut matrix = Array2::<Complex64>::zeros((left_dim * 2, 2 * right_dim));
        for l in 0..left_dim {
            for r in 0..right_dim {
                for i in 0..2 {
                    for j in 0..2 {
                        // combined physical index i*2 + j -> row (l, i), col (j, r)
                        matrix[[l * 2 + i, j * right_dim + r]] = result[[l, i * 2 + j, r]];
                    }
                }
            }
        }

        let (u, singular_values, vt) =
            truncated_svd(&matrix, self.config.max_bond_dim, self.config.svd_threshold)?;
        let new_bond = singular_values.len();

        // Left tensor receives U reshaped to (left_dim, 2, new_bond).
        let mut left_new = Array3::<Complex64>::zeros((left_dim, 2, new_bond));
        for l in 0..left_dim {
            for i in 0..2 {
                for b in 0..new_bond {
                    left_new[[l, i, b]] = u[[l * 2 + i, b]];
                }
            }
        }

        // Right tensor receives diag(S) · Vt reshaped to (new_bond, 2, right_dim), folding
        // the singular values into the right tensor so the product reconstructs the state.
        let mut right_new = Array3::<Complex64>::zeros((new_bond, 2, right_dim));
        for b in 0..new_bond {
            let scale = Complex64::new(singular_values[b], 0.0);
            for j in 0..2 {
                for r in 0..right_dim {
                    right_new[[b, j, r]] = scale * vt[[b, j * right_dim + r]];
                }
            }
        }

        self.tensors[left_q].data = left_new;
        self.tensors[right_q].data = right_new;

        Ok(())
    }

    /// Get amplitude of a computational basis state
    pub fn get_amplitude(&self, bitstring: &[bool]) -> QuantRS2Result<Complex64> {
        if bitstring.len() != self.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Bitstring length {} doesn't match qubit count {}",
                bitstring.len(),
                self.num_qubits
            )));
        }

        // Contract MPS from left to right
        let mut result = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));

        for (i, &bit) in bitstring.iter().enumerate() {
            let tensor = &self.tensors[i];
            let physical_idx = i32::from(bit);

            // Extract matrix for this physical index
            let matrix = tensor.data.slice(s![.., physical_idx, ..]);

            // Contract with accumulated result
            result = result.dot(&matrix);
        }

        Ok(result[[0, 0]])
    }

    /// Sample a measurement outcome
    #[must_use]
    pub fn sample(&self) -> Vec<bool> {
        let mut rng = thread_rng();
        let mut result = vec![false; self.num_qubits];
        let mut accumulated = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));

        for (i, tensor) in self.tensors.iter().enumerate() {
            // Compute probabilities for this qubit
            let matrix0 = tensor.data.slice(s![.., 0, ..]);
            let matrix1 = tensor.data.slice(s![.., 1, ..]);

            let branch0: Array2<Complex64> = accumulated.dot(&matrix0);
            let branch1: Array2<Complex64> = accumulated.dot(&matrix1);

            // Compute norms (simplified - doesn't contract remaining qubits)
            let norm0_sq: f64 = branch0.iter().map(scirs2_core::Complex::norm_sqr).sum();
            let norm1_sq: f64 = branch1.iter().map(scirs2_core::Complex::norm_sqr).sum();

            let total = norm0_sq + norm1_sq;
            let prob0 = norm0_sq / total;

            if rng.random::<f64>() < prob0 {
                result[i] = false;
                accumulated = branch0;
            } else {
                result[i] = true;
                accumulated = branch1;
            }

            // Renormalize
            let norm_sq: f64 = accumulated.iter().map(scirs2_core::Complex::norm_sqr).sum();
            if norm_sq > 0.0 {
                accumulated /= Complex64::new(norm_sq.sqrt(), 0.0);
            }
        }

        result
    }

    /// Contract the MPS into a dense state vector of `2^n` complex amplitudes.
    ///
    /// Amplitudes follow the little-endian convention `amplitude[index]`, where bit `q` of
    /// `index` holds the computational-basis value of qubit `q`. Each amplitude is computed
    /// as the genuine chain of matrix products selected by those bits, so an entangled MPS
    /// (e.g. a Bell pair) yields the correct entangled amplitudes.
    fn to_statevector(&self) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1usize << self.num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];

        for (index, amplitude) in amplitudes.iter_mut().enumerate() {
            let mut accumulated = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
            for qubit in 0..self.num_qubits {
                let bit = (index >> qubit) & 1;
                let slice = self.tensors[qubit].data.slice(s![.., bit as i32, ..]);
                accumulated = accumulated.dot(&slice);
            }
            *amplitude = accumulated[[0, 0]];
        }

        Ok(amplitudes)
    }
}

/// Truncated complex singular value decomposition `A = U · diag(S) · Vt`.
///
/// Delegates the full factorization to the robust one-sided Jacobi SVD in
/// [`crate::mps_simulator::complex_jacobi_svd`], then keeps the largest singular values
/// (those above `threshold`, capped at `max_bond`), returning `(U, S, Vt)` with the bond
/// dimension truncated to the number of retained singular values. At least one singular
/// value is always retained so the resulting tensors remain well-formed.
fn truncated_svd(
    matrix: &Array2<Complex64>,
    max_bond: usize,
    threshold: f64,
) -> QuantRS2Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>)> {
    let (full_u, full_s, full_vt) = crate::mps_simulator::complex_jacobi_svd(matrix)?;
    let full_rank = full_s.len();

    let mut kept = full_s.iter().filter(|&&value| value > threshold).count();
    kept = kept.min(max_bond).min(full_rank);
    if kept == 0 {
        kept = full_rank.max(1);
    }

    let u: Array2<Complex64> = full_u.slice(s![.., ..kept]).to_owned();
    let truncated_s: Array1<f64> = full_s.slice(s![..kept]).to_owned();
    let vt: Array2<Complex64> = full_vt.slice(s![..kept, ..]).to_owned();

    Ok((u, truncated_s, vt))
}

/// Basic MPS quantum simulator
pub struct BasicMPSSimulator {
    config: BasicMPSConfig,
}

impl BasicMPSSimulator {
    /// Create a new basic MPS simulator
    #[must_use]
    pub const fn new(config: BasicMPSConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(BasicMPSConfig::default())
    }
}

impl<const N: usize> Simulator<N> for BasicMPSSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        // Create initial MPS state
        let mut mps = BasicMPS::new(N, self.config.clone());

        // Apply gates from circuit
        for gate in circuit.gates() {
            match gate.name() {
                "H" => {
                    let h_matrix = {
                        let h = 1.0 / SQRT_2;
                        array![
                            [Complex64::new(h, 0.), Complex64::new(h, 0.)],
                            [Complex64::new(h, 0.), Complex64::new(-h, 0.)]
                        ]
                    };
                    if let Some(&qubit) = gate.qubits().first() {
                        mps.apply_single_qubit_gate(&h_matrix, qubit.id() as usize)?;
                    }
                }
                "X" => {
                    let x_matrix = array![
                        [Complex64::new(0., 0.), Complex64::new(1., 0.)],
                        [Complex64::new(1., 0.), Complex64::new(0., 0.)]
                    ];
                    if let Some(&qubit) = gate.qubits().first() {
                        mps.apply_single_qubit_gate(&x_matrix, qubit.id() as usize)?;
                    }
                }
                "CNOT" | "CX" => {
                    let cnot_matrix = array![
                        [
                            Complex64::new(1., 0.),
                            Complex64::new(0., 0.),
                            Complex64::new(0., 0.),
                            Complex64::new(0., 0.)
                        ],
                        [
                            Complex64::new(0., 0.),
                            Complex64::new(1., 0.),
                            Complex64::new(0., 0.),
                            Complex64::new(0., 0.)
                        ],
                        [
                            Complex64::new(0., 0.),
                            Complex64::new(0., 0.),
                            Complex64::new(0., 0.),
                            Complex64::new(1., 0.)
                        ],
                        [
                            Complex64::new(0., 0.),
                            Complex64::new(0., 0.),
                            Complex64::new(1., 0.),
                            Complex64::new(0., 0.)
                        ],
                    ];
                    let qubits = gate.qubits();
                    if qubits.len() == 2 {
                        mps.apply_two_qubit_gate(
                            &cnot_matrix,
                            qubits[0].id() as usize,
                            qubits[1].id() as usize,
                        )?;
                    }
                }
                _ => {
                    // Gate not supported in basic implementation
                }
            }
        }

        // Contract the MPS into a dense state vector and build the register from it.
        let amplitudes = mps.to_statevector()?;
        Register::<N>::with_amplitudes(amplitudes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_mps_initialization() {
        let mps = BasicMPS::new(4, BasicMPSConfig::default());

        // Check |0000> state
        let amp = mps
            .get_amplitude(&[false, false, false, false])
            .expect("Failed to get amplitude for |0000>");
        assert!((amp.norm() - 1.0).abs() < 1e-10);

        let amp = mps
            .get_amplitude(&[true, false, false, false])
            .expect("Failed to get amplitude for |1000>");
        assert!(amp.norm() < 1e-10);
    }

    #[test]
    fn test_single_qubit_gate() {
        let mut mps = BasicMPS::new(3, BasicMPSConfig::default());

        // Apply X to first qubit
        let x_matrix = array![
            [Complex64::new(0., 0.), Complex64::new(1., 0.)],
            [Complex64::new(1., 0.), Complex64::new(0., 0.)]
        ];
        mps.apply_single_qubit_gate(&x_matrix, 0)
            .expect("Failed to apply X gate");

        // Check |100> state
        let amp = mps
            .get_amplitude(&[true, false, false])
            .expect("Failed to get amplitude for |100>");
        assert!((amp.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_truncated_svd_reconstructs_matrix() {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.0, 2.0),
            ],
        )
        .expect("failed to build test matrix");

        let (u, s, vt) =
            truncated_svd(&matrix, 16, 1e-14).expect("real SVD decomposition should succeed");

        let k = s.len();
        let mut max_err = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = Complex64::new(0.0, 0.0);
                for r in 0..k {
                    acc += u[[i, r]] * Complex64::new(s[r], 0.0) * vt[[r, j]];
                }
                max_err = max_err.max((acc - matrix[[i, j]]).norm());
            }
        }
        assert!(
            max_err < 1e-8,
            "SVD reconstruction error too large: {max_err}"
        );
        // Not the fabricated identity-like decomposition.
        assert!(
            s.iter().any(|&value| (value - 1.0).abs() > 1e-6),
            "singular values are all ~1 (identity fabrication)"
        );
    }

    #[test]
    fn test_two_qubit_gate_bell_contraction() {
        // Prepare a Bell state via H on qubit 0 followed by CNOT(0, 1) using real SVD.
        let mut mps = BasicMPS::new(2, BasicMPSConfig::default());

        let h = 1.0 / SQRT_2;
        let h_matrix = array![
            [Complex64::new(h, 0.), Complex64::new(h, 0.)],
            [Complex64::new(h, 0.), Complex64::new(-h, 0.)]
        ];
        mps.apply_single_qubit_gate(&h_matrix, 0)
            .expect("failed to apply H");

        let cnot = array![
            [
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.)
            ],
            [
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.)
            ],
            [
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.)
            ],
            [
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.)
            ],
        ];
        mps.apply_two_qubit_gate(&cnot, 0, 1)
            .expect("failed to apply CNOT");

        let state = mps.to_statevector().expect("contraction should succeed");

        let inv_sqrt2 = 1.0 / SQRT_2;
        assert!((state[0].re - inv_sqrt2).abs() < 1e-10, "amp(|00>) wrong");
        assert!(state[1].norm() < 1e-10, "amp(|01>) should vanish");
        assert!(state[2].norm() < 1e-10, "amp(|10>) should vanish");
        assert!((state[3].re - inv_sqrt2).abs() < 1e-10, "amp(|11>) wrong");

        // Honest check: genuinely entangled, not the empty/|00> placeholder register.
        assert!(
            state[3].norm() > 1e-3,
            "two-qubit gate fabricated a non-entangled state"
        );
    }
}
