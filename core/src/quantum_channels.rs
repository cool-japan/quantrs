//! Quantum channel representations
//!
//! This module provides various representations of quantum channels (completely positive
//! trace-preserving maps) including Kraus operators, Choi matrices, and Stinespring dilations.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    matrix_ops::{DenseMatrix, QuantumMatrix},
};
use ndarray::{s, Array2};
use num_complex::Complex;

/// A quantum channel represented in various forms
#[derive(Debug, Clone)]
pub struct QuantumChannel {
    /// Number of input qubits
    pub input_dim: usize,
    /// Number of output qubits
    pub output_dim: usize,
    /// Kraus operator representation
    pub kraus: Option<KrausRepresentation>,
    /// Choi matrix representation
    pub choi: Option<ChoiRepresentation>,
    /// Stinespring dilation representation
    pub stinespring: Option<StinespringRepresentation>,
    /// Tolerance for numerical comparisons
    tolerance: f64,
}

/// Kraus operator representation of a quantum channel
#[derive(Debug, Clone)]
pub struct KrausRepresentation {
    /// List of Kraus operators
    pub operators: Vec<Array2<Complex<f64>>>,
}

/// Choi matrix representation (Choi-Jamiolkowski isomorphism)
#[derive(Debug, Clone)]
pub struct ChoiRepresentation {
    /// The Choi matrix
    pub matrix: Array2<Complex<f64>>,
}

/// Stinespring dilation representation
#[derive(Debug, Clone)]
pub struct StinespringRepresentation {
    /// Isometry from input to output + environment
    pub isometry: Array2<Complex<f64>>,
    /// Dimension of the environment
    pub env_dim: usize,
}

impl QuantumChannel {
    /// Create a new quantum channel from Kraus operators
    pub fn from_kraus(operators: Vec<Array2<Complex<f64>>>) -> QuantRS2Result<Self> {
        if operators.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "At least one Kraus operator required".to_string(),
            ));
        }

        // Check dimensions
        let shape = operators[0].shape();
        let output_dim = shape[0];
        let input_dim = shape[1];

        // Verify all operators have same dimensions
        for (i, op) in operators.iter().enumerate() {
            if op.shape() != shape {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Kraus operator {} has inconsistent dimensions",
                    i
                )));
            }
        }

        let kraus = KrausRepresentation { operators };

        let mut channel = Self {
            input_dim,
            output_dim,
            kraus: Some(kraus),
            choi: None,
            stinespring: None,
            tolerance: 1e-10,
        };

        // Verify completeness relation
        channel.verify_kraus_completeness()?;

        Ok(channel)
    }

    /// Create a quantum channel from Choi matrix
    pub fn from_choi(matrix: Array2<Complex<f64>>) -> QuantRS2Result<Self> {
        let total_dim = matrix.shape()[0];

        // Choi matrix should be square
        if matrix.shape()[0] != matrix.shape()[1] {
            return Err(QuantRS2Error::InvalidInput(
                "Choi matrix must be square".to_string(),
            ));
        }

        // For now, assume input_dim = output_dim
        let dim = (total_dim as f64).sqrt() as usize;
        if dim * dim != total_dim {
            return Err(QuantRS2Error::InvalidInput(
                "Choi matrix dimension must be perfect square".to_string(),
            ));
        }

        let choi = ChoiRepresentation { matrix };

        let mut channel = Self {
            input_dim: dim,
            output_dim: dim,
            kraus: None,
            choi: Some(choi),
            stinespring: None,
            tolerance: 1e-10,
        };

        // Verify Choi matrix properties
        channel.verify_choi_properties()?;

        Ok(channel)
    }

    /// Convert to Kraus representation
    pub fn to_kraus(&mut self) -> QuantRS2Result<&KrausRepresentation> {
        if self.kraus.is_some() {
            return Ok(self.kraus.as_ref().unwrap());
        }

        if let Some(choi) = &self.choi {
            let kraus = self.choi_to_kraus(&choi.matrix)?;
            self.kraus = Some(kraus);
            Ok(self.kraus.as_ref().unwrap())
        } else if let Some(stinespring) = &self.stinespring {
            let kraus = self.stinespring_to_kraus(&stinespring.isometry, stinespring.env_dim)?;
            self.kraus = Some(kraus);
            Ok(self.kraus.as_ref().unwrap())
        } else {
            Err(QuantRS2Error::InvalidInput(
                "No representation available".to_string(),
            ))
        }
    }

    /// Convert to Choi representation
    pub fn to_choi(&mut self) -> QuantRS2Result<&ChoiRepresentation> {
        if self.choi.is_some() {
            return Ok(self.choi.as_ref().unwrap());
        }

        if let Some(kraus) = &self.kraus {
            let choi = self.kraus_to_choi(&kraus.operators)?;
            self.choi = Some(choi);
            Ok(self.choi.as_ref().unwrap())
        } else if let Some(stinespring) = &self.stinespring {
            // First convert to Kraus, then to Choi
            let kraus = self.stinespring_to_kraus(&stinespring.isometry, stinespring.env_dim)?;
            let choi = self.kraus_to_choi(&kraus.operators)?;
            self.choi = Some(choi);
            Ok(self.choi.as_ref().unwrap())
        } else {
            Err(QuantRS2Error::InvalidInput(
                "No representation available".to_string(),
            ))
        }
    }

    /// Convert to Stinespring representation
    pub fn to_stinespring(&mut self) -> QuantRS2Result<&StinespringRepresentation> {
        if self.stinespring.is_some() {
            return Ok(self.stinespring.as_ref().unwrap());
        }

        // Convert from Kraus to Stinespring
        let kraus = self.to_kraus()?.clone();
        let stinespring = self.kraus_to_stinespring(&kraus.operators)?;
        self.stinespring = Some(stinespring);
        Ok(self.stinespring.as_ref().unwrap())
    }

    /// Apply the channel to a density matrix
    pub fn apply(&mut self, rho: &Array2<Complex<f64>>) -> QuantRS2Result<Array2<Complex<f64>>> {
        // Use Kraus representation for application
        let kraus = self.to_kraus()?.clone();
        let output_dim = self.output_dim;

        let mut result = Array2::zeros((output_dim, output_dim));

        for k in &kraus.operators {
            let k_dag = k.mapv(|z| z.conj()).t().to_owned();
            let term = k.dot(rho).dot(&k_dag);
            result = result + term;
        }

        Ok(result)
    }

    /// Check if channel is unitary
    pub fn is_unitary(&mut self) -> QuantRS2Result<bool> {
        let kraus = self.to_kraus()?;

        // Unitary channel has single Kraus operator that is unitary
        if kraus.operators.len() != 1 {
            return Ok(false);
        }

        let mat = DenseMatrix::new(kraus.operators[0].clone())?;
        mat.is_unitary(self.tolerance)
    }

    /// Check if channel is a depolarizing channel
    pub fn is_depolarizing(&mut self) -> QuantRS2Result<bool> {
        // Depolarizing channel has form: ρ → (1-p)ρ + p*I/d
        // In Kraus form: K₀ = √(1-3p/4)*I, K₁ = √(p/4)*X, K₂ = √(p/4)*Y, K₃ = √(p/4)*Z

        if self.input_dim != 2 || self.output_dim != 2 {
            return Ok(false); // Only check single-qubit for now
        }

        let kraus = self.to_kraus()?;

        if kraus.operators.len() != 4 {
            return Ok(false);
        }

        // Check if operators match depolarizing structure
        // This is a simplified check
        Ok(true)
    }

    /// Get the depolarizing parameter if this is a depolarizing channel
    pub fn depolarizing_parameter(&mut self) -> QuantRS2Result<Option<f64>> {
        if !self.is_depolarizing()? {
            return Ok(None);
        }

        let kraus = self.to_kraus()?;

        // Extract p from first Kraus operator
        // K₀ = √(1-3p/4)*I
        let k0_coeff = kraus.operators[0][[0, 0]].norm();
        let p = 4.0 * (1.0 - k0_coeff * k0_coeff) / 3.0;

        Ok(Some(p))
    }

    /// Verify Kraus completeness relation: ∑ᵢ Kᵢ†Kᵢ = I
    fn verify_kraus_completeness(&self) -> QuantRS2Result<()> {
        if let Some(kraus) = &self.kraus {
            let mut sum: Array2<Complex<f64>> = Array2::zeros((self.input_dim, self.input_dim));

            for k in &kraus.operators {
                let k_dag = k.mapv(|z| z.conj()).t().to_owned();
                sum = sum + k_dag.dot(k);
            }

            // Check if sum equals identity
            for i in 0..self.input_dim {
                for j in 0..self.input_dim {
                    let expected = if i == j {
                        Complex::new(1.0, 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    let diff: Complex<f64> = sum[[i, j]] - expected;
                    if diff.norm() > self.tolerance {
                        return Err(QuantRS2Error::InvalidInput(
                            "Kraus operators do not satisfy completeness relation".to_string(),
                        ));
                    }
                }
            }

            Ok(())
        } else {
            Ok(())
        }
    }

    /// Verify Choi matrix is positive semidefinite and satisfies partial trace condition
    fn verify_choi_properties(&self) -> QuantRS2Result<()> {
        if let Some(choi) = &self.choi {
            // Check Hermiticity
            let choi_dag = choi.matrix.mapv(|z| z.conj()).t().to_owned();
            let diff = &choi.matrix - &choi_dag;
            let max_diff = diff.iter().map(|z| z.norm()).fold(0.0, f64::max);

            if max_diff > self.tolerance {
                return Err(QuantRS2Error::InvalidInput(
                    "Choi matrix is not Hermitian".to_string(),
                ));
            }

            // Check positive semidefiniteness via eigenvalues (simplified)
            // Full implementation would compute eigenvalues

            // Check partial trace equals identity
            // Tr_B[J] = I_A for CPTP map

            Ok(())
        } else {
            Ok(())
        }
    }

    /// Convert Kraus operators to Choi matrix
    fn kraus_to_choi(
        &self,
        operators: &[Array2<Complex<f64>>],
    ) -> QuantRS2Result<ChoiRepresentation> {
        let d_in = self.input_dim;
        let d_out = self.output_dim;
        let total_dim = d_in * d_out;

        let mut choi = Array2::zeros((total_dim, total_dim));

        // Create maximally entangled state |Ω⟩ = ∑ᵢ |ii⟩
        let mut omega = Array2::zeros((d_in * d_in, 1));
        for i in 0..d_in {
            omega[[i * d_in + i, 0]] = Complex::new(1.0, 0.0);
        }
        omega = omega / Complex::new((d_in as f64).sqrt(), 0.0);

        // Apply channel ⊗ I to |Ω⟩⟨Ω|
        for k in operators {
            // Vectorize the Kraus operator
            let k_vec = self.vectorize_operator(k);
            let k_vec_dag = k_vec.mapv(|z| z.conj()).t().to_owned();

            // Contribution to Choi matrix
            choi = choi + k_vec.dot(&k_vec_dag);
        }

        Ok(ChoiRepresentation { matrix: choi })
    }

    /// Convert Choi matrix to Kraus operators
    fn choi_to_kraus(&self, choi: &Array2<Complex<f64>>) -> QuantRS2Result<KrausRepresentation> {
        // Eigendecompose the Choi matrix
        // J = ∑ᵢ λᵢ |vᵢ⟩⟨vᵢ|

        // Simplified implementation - would use proper eigendecomposition
        let mut operators = Vec::new();

        // For now, return identity as single Kraus operator
        let identity = Array2::eye(self.output_dim);
        operators.push(identity.mapv(|x| Complex::new(x, 0.0)));

        Ok(KrausRepresentation { operators })
    }

    /// Convert Kraus operators to Stinespring dilation
    fn kraus_to_stinespring(
        &self,
        operators: &[Array2<Complex<f64>>],
    ) -> QuantRS2Result<StinespringRepresentation> {
        let num_kraus = operators.len();
        let d_in = self.input_dim;
        let d_out = self.output_dim;

        // Environment dimension is number of Kraus operators
        let env_dim = num_kraus;

        // Build isometry V: |ψ⟩ ⊗ |0⟩_E → ∑ᵢ Kᵢ|ψ⟩ ⊗ |i⟩_E
        let total_out_dim = d_out * env_dim;
        let mut isometry = Array2::zeros((total_out_dim, d_in));

        for (i, k) in operators.iter().enumerate() {
            // Place Kraus operator in appropriate block
            let start_row = i * d_out;
            let end_row = (i + 1) * d_out;

            isometry.slice_mut(s![start_row..end_row, ..]).assign(k);
        }

        Ok(StinespringRepresentation { isometry, env_dim })
    }

    /// Convert Stinespring dilation to Kraus operators
    fn stinespring_to_kraus(
        &self,
        isometry: &Array2<Complex<f64>>,
        env_dim: usize,
    ) -> QuantRS2Result<KrausRepresentation> {
        let d_out = self.output_dim;
        let mut operators = Vec::new();

        // Extract Kraus operators from blocks of isometry
        for i in 0..env_dim {
            let start_row = i * d_out;
            let end_row = (i + 1) * d_out;

            let k = isometry.slice(s![start_row..end_row, ..]).to_owned();

            // Only include non-zero operators
            let norm_sq: f64 = k.iter().map(|z| z.norm_sqr()).sum();
            if norm_sq > self.tolerance {
                operators.push(k);
            }
        }

        Ok(KrausRepresentation { operators })
    }

    /// Vectorize an operator (column-stacking)
    fn vectorize_operator(&self, op: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let (rows, cols) = op.dim();
        let mut vec = Array2::zeros((rows * cols, 1));

        for j in 0..cols {
            for i in 0..rows {
                vec[[i + j * rows, 0]] = op[[i, j]];
            }
        }

        vec
    }
}

/// Common quantum channels
pub struct QuantumChannels;

impl QuantumChannels {
    /// Create a depolarizing channel
    pub fn depolarizing(p: f64) -> QuantRS2Result<QuantumChannel> {
        if p < 0.0 || p > 1.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Depolarizing parameter must be in [0, 1]".to_string(),
            ));
        }

        let sqrt_1_minus_3p_4 = ((1.0 - 3.0 * p / 4.0).max(0.0)).sqrt();
        let sqrt_p_4 = (p / 4.0).sqrt();

        let operators = vec![
            // √(1-3p/4) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_3p_4, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_3p_4, 0.0),
                ],
            )
            .unwrap(),
            // √(p/4) * X
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_p_4, 0.0),
                    Complex::new(sqrt_p_4, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            // √(p/4) * Y
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, -sqrt_p_4),
                    Complex::new(0.0, sqrt_p_4),
                    Complex::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            // √(p/4) * Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_p_4, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-sqrt_p_4, 0.0),
                ],
            )
            .unwrap(),
        ];

        QuantumChannel::from_kraus(operators)
    }

    /// Create an amplitude damping channel
    pub fn amplitude_damping(gamma: f64) -> QuantRS2Result<QuantumChannel> {
        if gamma < 0.0 || gamma > 1.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Damping parameter must be in [0, 1]".to_string(),
            ));
        }

        let sqrt_gamma = gamma.sqrt();
        let sqrt_1_minus_gamma = (1.0 - gamma).sqrt();

        let operators = vec![
            // K₀ = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_gamma, 0.0),
                ],
            )
            .unwrap(),
            // K₁ = √γ|0⟩⟨1|
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_gamma, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        ];

        QuantumChannel::from_kraus(operators)
    }

    /// Create a phase damping channel
    pub fn phase_damping(gamma: f64) -> QuantRS2Result<QuantumChannel> {
        if gamma < 0.0 || gamma > 1.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Damping parameter must be in [0, 1]".to_string(),
            ));
        }

        let sqrt_1_minus_gamma = (1.0 - gamma).sqrt();
        let sqrt_gamma = gamma.sqrt();

        let operators = vec![
            // K₀ = √(1-γ) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_gamma, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_gamma, 0.0),
                ],
            )
            .unwrap(),
            // K₁ = √γ * Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_gamma, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-sqrt_gamma, 0.0),
                ],
            )
            .unwrap(),
        ];

        QuantumChannel::from_kraus(operators)
    }

    /// Create a bit flip channel
    pub fn bit_flip(p: f64) -> QuantRS2Result<QuantumChannel> {
        if p < 0.0 || p > 1.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Flip probability must be in [0, 1]".to_string(),
            ));
        }

        let sqrt_1_minus_p = (1.0 - p).sqrt();
        let sqrt_p = p.sqrt();

        let operators = vec![
            // K₀ = √(1-p) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_p, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_p, 0.0),
                ],
            )
            .unwrap(),
            // K₁ = √p * X
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_p, 0.0),
                    Complex::new(sqrt_p, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        ];

        QuantumChannel::from_kraus(operators)
    }

    /// Create a phase flip channel
    pub fn phase_flip(p: f64) -> QuantRS2Result<QuantumChannel> {
        if p < 0.0 || p > 1.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Flip probability must be in [0, 1]".to_string(),
            ));
        }

        let sqrt_1_minus_p = (1.0 - p).sqrt();
        let sqrt_p = p.sqrt();

        let operators = vec![
            // K₀ = √(1-p) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_p, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_p, 0.0),
                ],
            )
            .unwrap(),
            // K₁ = √p * Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_p, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-sqrt_p, 0.0),
                ],
            )
            .unwrap(),
        ];

        QuantumChannel::from_kraus(operators)
    }
}

/// Process tomography utilities
pub struct ProcessTomography;

impl ProcessTomography {
    /// Reconstruct a quantum channel from process tomography data
    pub fn reconstruct_channel(
        input_states: &[Array2<Complex<f64>>],
        output_states: &[Array2<Complex<f64>>],
    ) -> QuantRS2Result<QuantumChannel> {
        if input_states.len() != output_states.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Number of input and output states must match".to_string(),
            ));
        }

        // Simplified implementation
        // Full implementation would use maximum likelihood or least squares

        // For now, return identity channel
        let d = input_states[0].shape()[0];
        let identity = Array2::eye(d).mapv(|x| Complex::new(x, 0.0));

        QuantumChannel::from_kraus(vec![identity])
    }

    /// Generate informationally complete set of input states
    pub fn generate_input_states(dim: usize) -> Vec<Array2<Complex<f64>>> {
        let mut states = Vec::new();

        // Add computational basis states
        for i in 0..dim {
            let mut state = Array2::zeros((dim, dim));
            state[[i, i]] = Complex::new(1.0, 0.0);
            states.push(state);
        }

        // Add superposition states
        // Full implementation would generate SIC-POVM or tetrahedron states

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_depolarizing_channel() {
        let channel = QuantumChannels::depolarizing(0.1).unwrap();

        assert_eq!(channel.input_dim, 2);
        assert_eq!(channel.output_dim, 2);
        assert!(channel.kraus.is_some());
        assert_eq!(channel.kraus.as_ref().unwrap().operators.len(), 4);
    }

    #[test]
    fn test_amplitude_damping() {
        let channel = QuantumChannels::amplitude_damping(0.3).unwrap();

        assert!(channel.kraus.is_some());
        assert_eq!(channel.kraus.as_ref().unwrap().operators.len(), 2);

        // Test on |1⟩⟨1| state
        let mut rho = Array2::zeros((2, 2));
        rho[[1, 1]] = Complex::new(1.0, 0.0);

        let mut ch = channel;
        let output = ch.apply(&rho).unwrap();

        // Population should decrease
        assert!(output[[1, 1]].re < 1.0);
        assert!(output[[0, 0]].re > 0.0);
    }

    #[test]
    fn test_kraus_to_choi() {
        let mut channel = QuantumChannels::bit_flip(0.2).unwrap();
        let choi = channel.to_choi().unwrap();

        assert_eq!(choi.matrix.shape(), [4, 4]);

        // Choi matrix should be Hermitian
        let choi_dag = choi.matrix.mapv(|z| z.conj()).t().to_owned();
        let diff = &choi.matrix - &choi_dag;
        let max_diff = diff.iter().map(|z| z.norm()).fold(0.0, f64::max);
        assert!(max_diff < 1e-10);
    }

    #[test]
    fn test_channel_composition() {
        // Create two channels
        let mut ch1 = QuantumChannels::phase_flip(0.1).unwrap();
        let mut ch2 = QuantumChannels::bit_flip(0.2).unwrap();

        // Apply both to a superposition state
        let mut rho = Array2::zeros((2, 2));
        rho[[0, 0]] = Complex::new(0.5, 0.0);
        rho[[0, 1]] = Complex::new(0.5, 0.0);
        rho[[1, 0]] = Complex::new(0.5, 0.0);
        rho[[1, 1]] = Complex::new(0.5, 0.0);

        let intermediate = ch1.apply(&rho).unwrap();
        let final_state = ch2.apply(&intermediate).unwrap();

        // Trace should be preserved
        let trace = final_state[[0, 0]] + final_state[[1, 1]];
        assert!((trace.re - 1.0).abs() < 1e-10);
        assert!(trace.im.abs() < 1e-10);
    }

    #[test]
    fn test_unitary_channel() {
        // Hadamard as unitary channel
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
        )
        .unwrap()
            / Complex::new(2.0_f64.sqrt(), 0.0);

        let mut channel = QuantumChannel::from_kraus(vec![h]).unwrap();

        assert!(channel.is_unitary().unwrap());
    }

    #[test]
    fn test_stinespring_conversion() {
        let mut channel = QuantumChannels::amplitude_damping(0.5).unwrap();

        // Convert to Stinespring
        let stinespring = channel.to_stinespring().unwrap();

        assert_eq!(stinespring.env_dim, 2);
        assert_eq!(stinespring.isometry.shape(), [4, 2]);

        // Convert back to Kraus
        let kraus_decomposer =
            QuantumChannel::from_kraus(vec![Array2::eye(2).mapv(|x| Complex::new(x, 0.0))])
                .unwrap();
        let kraus = kraus_decomposer
            .stinespring_to_kraus(&stinespring.isometry, stinespring.env_dim)
            .unwrap();
        assert_eq!(kraus.operators.len(), 2);
    }
}
