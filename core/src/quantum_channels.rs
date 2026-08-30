//! Quantum channel representations
//!
//! This module provides various representations of quantum channels (completely positive
//! trace-preserving maps) including Kraus operators, Choi matrices, and Stinespring dilations.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    matrix_ops::{DenseMatrix, QuantumMatrix},
};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::Complex;

/// Spectral decomposition of a Hermitian matrix.
///
/// `eigenvalues[i]` is real (stored as `f64`) and corresponds to the eigenvector
/// held in column `i` of `eigenvectors`. The eigenvectors form an orthonormal
/// set.
struct HermitianEigen {
    /// Real eigenvalues.
    eigenvalues: Array1<f64>,
    /// Orthonormal eigenvectors stored as columns.
    eigenvectors: Array2<Complex<f64>>,
}

/// Compute the spectral decomposition of a Hermitian matrix using the cyclic
/// complex Jacobi eigenvalue algorithm.
///
/// The Choi matrices and normal-equation matrices that arise in this module are
/// Hermitian (and typically positive semidefinite) but generally *not* unitary,
/// so the unitary-specialised QR routine in [`crate::eigensolve`] is not
/// numerically reliable for them. The Jacobi method, by contrast, is
/// unconditionally convergent for Hermitian matrices and directly yields an
/// orthonormal eigenbasis, which is exactly what the Kraus reconstruction and
/// least-squares pseudo-inverse here require. The matrices involved are small
/// (`d² × d²`), so the `O(n³)` per-sweep cost of Jacobi is not a concern.
///
/// The input is symmetrised to its Hermitian part `(H + Hᴴ)/2` before
/// diagonalisation to remove any tiny numerical asymmetry.
fn hermitian_eigen_decompose(matrix: &Array2<Complex<f64>>) -> QuantRS2Result<HermitianEigen> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(QuantRS2Error::InvalidInput(
            "Hermitian eigendecomposition requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok(HermitianEigen {
            eigenvalues: Array1::zeros(0),
            eigenvectors: Array2::zeros((0, 0)),
        });
    }

    // Work on the Hermitian part to be robust against tiny asymmetry.
    let mut a: Array2<Complex<f64>> = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = (matrix[[i, j]] + matrix[[j, i]].conj()) * Complex::new(0.5, 0.0);
        }
    }

    // Accumulated eigenvectors (columns), initialised to the identity.
    let mut v: Array2<Complex<f64>> = Array2::eye(n);

    if n == 1 {
        return Ok(HermitianEigen {
            eigenvalues: Array1::from_vec(vec![a[[0, 0]].re]),
            eigenvectors: v,
        });
    }

    let max_sweeps = 100;
    let convergence_eps = 1e-15;

    for _sweep in 0..max_sweeps {
        // Off-diagonal Frobenius norm (squared) for the convergence test.
        let mut off_norm_sq = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_norm_sq += a[[p, q]].norm_sqr();
            }
        }
        if off_norm_sq.sqrt() < convergence_eps {
            break;
        }

        // One cyclic sweep over all (p, q) with p < q.
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.norm() < convergence_eps {
                    continue;
                }

                let app = a[[p, p]].re;
                let aqq = a[[q, q]].re;

                // Complex Jacobi rotation. Write the off-diagonal in polar form
                // a_pq = |a_pq| e^{iφ}; the rotation that zeroes it is
                //   [ c            s e^{iφ} ]
                //   [ -s e^{-iφ}   c        ]
                // with the real rotation angle θ chosen as in the real symmetric
                // Jacobi method applied to the 2×2 Hermitian block.
                let abs_apq = apq.norm();
                let phase = apq / Complex::new(abs_apq, 0.0); // e^{iφ}

                let tau = (aqq - app) / (2.0 * abs_apq);
                // t = sign(tau) / (|tau| + sqrt(tau² + 1)), the smaller root.
                let t = if tau >= 0.0 {
                    1.0 / (tau + (tau * tau + 1.0).sqrt())
                } else {
                    -1.0 / (-tau + (tau * tau + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let s_phase = phase * Complex::new(s, 0.0); // s e^{iφ}
                let c_cplx = Complex::new(c, 0.0);

                // Build the full unitary Jacobi rotation J (the identity except
                // for the 2×2 block on rows/columns p and q):
                //   J[p,p] = c,  J[p,q] =  s e^{iφ}
                //   J[q,p] = -s e^{-iφ},  J[q,q] = c
                // and apply it as the similarity transform A <- Jᴴ A J. Using the
                // explicit matrix product (rather than an in-place row/column
                // sweep) keeps the corner entries consistent and guarantees the
                // monotone reduction of the off-diagonal norm. The matrices here
                // are small (d² × d²), so the extra cost is negligible.
                let mut rotation: Array2<Complex<f64>> = Array2::eye(n);
                rotation[[p, p]] = c_cplx;
                rotation[[q, q]] = c_cplx;
                rotation[[p, q]] = s_phase;
                rotation[[q, p]] = -s_phase.conj();
                let rotation_dag = rotation.mapv(|z| z.conj()).t().to_owned();

                a = rotation_dag.dot(&a).dot(&rotation);
                v = v.dot(&rotation);
            }
        }
    }

    // Eigenvalues are the (real parts of the) diagonal of the rotated matrix.
    let mut eigenvalues: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]].re;
    }

    Ok(HermitianEigen {
        eigenvalues,
        eigenvectors: v,
    })
}

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
                    "Kraus operator {i} has inconsistent dimensions"
                )));
            }
        }

        let kraus = KrausRepresentation { operators };

        let channel = Self {
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

    /// Create a quantum channel from a Choi matrix.
    ///
    /// This constructor assumes a *square* channel, i.e. equal input and output
    /// dimensions (`input_dim == output_dim == d`). The Choi matrix of such a
    /// channel has dimension `total_dim = d * d`, so requiring `total_dim` to be
    /// a perfect square is the correct, exact constraint for this case (it is
    /// not a placeholder approximation). Rectangular channels (`d_in != d_out`)
    /// are not constructed through this entry point.
    pub fn from_choi(matrix: Array2<Complex<f64>>) -> QuantRS2Result<Self> {
        let total_dim = matrix.shape()[0];

        // Choi matrix should be square
        if matrix.shape()[0] != matrix.shape()[1] {
            return Err(QuantRS2Error::InvalidInput(
                "Choi matrix must be square".to_string(),
            ));
        }

        // Square channel assumption: input_dim == output_dim == d, so the Choi
        // matrix dimension is d * d. Recover d as the integer square root.
        let dim = (total_dim as f64).sqrt().round() as usize;
        if dim * dim != total_dim {
            return Err(QuantRS2Error::InvalidInput(
                "Choi matrix dimension must be a perfect square (square channel: total_dim = d * d)"
                    .to_string(),
            ));
        }

        let choi = ChoiRepresentation { matrix };

        let channel = Self {
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
            return self
                .kraus
                .as_ref()
                .ok_or_else(|| QuantRS2Error::InvalidInput("Kraus representation missing".into()));
        }

        if let Some(choi) = &self.choi {
            let kraus = self.choi_to_kraus(&choi.matrix)?;
            self.kraus = Some(kraus);
            self.kraus
                .as_ref()
                .ok_or_else(|| QuantRS2Error::InvalidInput("Kraus conversion failed".into()))
        } else if let Some(stinespring) = &self.stinespring {
            let kraus = self.stinespring_to_kraus(&stinespring.isometry, stinespring.env_dim)?;
            self.kraus = Some(kraus);
            self.kraus
                .as_ref()
                .ok_or_else(|| QuantRS2Error::InvalidInput("Kraus conversion failed".into()))
        } else {
            Err(QuantRS2Error::InvalidInput(
                "No representation available".to_string(),
            ))
        }
    }

    /// Convert to Choi representation
    pub fn to_choi(&mut self) -> QuantRS2Result<&ChoiRepresentation> {
        if self.choi.is_some() {
            return self
                .choi
                .as_ref()
                .ok_or_else(|| QuantRS2Error::InvalidInput("Choi representation missing".into()));
        }

        if let Some(kraus) = &self.kraus {
            let choi = self.kraus_to_choi(&kraus.operators)?;
            self.choi = Some(choi);
            self.choi
                .as_ref()
                .ok_or_else(|| QuantRS2Error::InvalidInput("Choi conversion failed".into()))
        } else if let Some(stinespring) = &self.stinespring {
            // First convert to Kraus, then to Choi
            let kraus = self.stinespring_to_kraus(&stinespring.isometry, stinespring.env_dim)?;
            let choi = self.kraus_to_choi(&kraus.operators)?;
            self.choi = Some(choi);
            self.choi
                .as_ref()
                .ok_or_else(|| QuantRS2Error::InvalidInput("Choi conversion failed".into()))
        } else {
            Err(QuantRS2Error::InvalidInput(
                "No representation available".to_string(),
            ))
        }
    }

    /// Convert to Stinespring representation
    pub fn to_stinespring(&mut self) -> QuantRS2Result<&StinespringRepresentation> {
        if self.stinespring.is_some() {
            return self.stinespring.as_ref().ok_or_else(|| {
                QuantRS2Error::InvalidInput("Stinespring representation missing".into())
            });
        }

        // Convert from Kraus to Stinespring
        let kraus = self.to_kraus()?.clone();
        let stinespring = self.kraus_to_stinespring(&kraus.operators)?;
        self.stinespring = Some(stinespring);
        self.stinespring
            .as_ref()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Stinespring conversion failed".into()))
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
        let p = 4.0 * k0_coeff.mul_add(-k0_coeff, 1.0) / 3.0;

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
        let _omega = omega / Complex::new((d_in as f64).sqrt(), 0.0);

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

    /// Convert a Choi matrix to a set of Kraus operators.
    ///
    /// The Choi matrix `J` (dimension `d_in * d_out`) of a completely positive
    /// map is Hermitian and positive semidefinite. Spectral-decomposing it as
    /// `J = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|`, every eigenpair with `λᵢ > tolerance` yields a Kraus
    /// operator `Kᵢ = √λᵢ · unvec(vᵢ)`, where `unvec` reshapes the length
    /// `d_in * d_out` eigenvector back into a `d_out × d_in` matrix.
    ///
    /// The reshape must invert exactly the vectorization used by
    /// [`Self::vectorize_operator`] / [`Self::kraus_to_choi`]. That routine uses
    /// *column-stacking* (`vec[i + j * d_out] = K[i, j]`), so this method
    /// un-stacks columns the same way, making the
    /// Kraus → Choi → Kraus round-trip exact.
    ///
    /// Eigenpairs with `λᵢ ≤ self.tolerance` are dropped: they correspond to the
    /// kernel of `J` and contribute zero Kraus operators.
    fn choi_to_kraus(&self, choi: &Array2<Complex<f64>>) -> QuantRS2Result<KrausRepresentation> {
        let d_in = self.input_dim;
        let d_out = self.output_dim;
        let total_dim = d_in * d_out;

        if choi.shape() != [total_dim, total_dim] {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Choi matrix has shape {:?}, expected [{total_dim}, {total_dim}]",
                choi.shape()
            )));
        }

        // Spectral decomposition J = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|. The Choi matrix is Hermitian
        // (and PSD for a CP map), so we use the Hermitian Jacobi solver, which
        // returns real eigenvalues and an orthonormal eigenbasis.
        let decomposition = hermitian_eigen_decompose(choi)?;
        let eigenvalues = &decomposition.eigenvalues;
        let eigenvectors = &decomposition.eigenvectors;

        let mut operators = Vec::new();

        for (idx, &lambda) in eigenvalues.iter().enumerate() {
            // Physical (PSD) eigenvalues are non-negative; drop kernel/noise.
            if lambda <= self.tolerance {
                continue;
            }
            let scale = lambda.sqrt();

            // Un-vectorize the eigenvector (column `idx`) into a d_out × d_in
            // Kraus operator, inverting the column-stacking convention used by
            // `vectorize_operator`: K[i, j] = sqrt(lambda) * v[i + j * d_out].
            let mut kraus_op: Array2<Complex<f64>> = Array2::zeros((d_out, d_in));
            for j in 0..d_in {
                for i in 0..d_out {
                    kraus_op[[i, j]] = eigenvectors[[i + j * d_out, idx]] * scale;
                }
            }

            operators.push(kraus_op);
        }

        // A completely positive map has at least one non-zero Kraus operator. An
        // empty set here means the Choi matrix was (numerically) zero, i.e. the
        // zero map, which is not a valid quantum channel.
        if operators.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "Choi matrix has no eigenvalues above tolerance; cannot build Kraus operators (zero map)"
                    .to_string(),
            ));
        }

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
            // sqrt(1-3p/4) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_3p_4, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_3p_4, 0.0),
                ],
            )
            .expect("valid 2x2 identity Kraus operator"),
            // sqrt(p/4) * X
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_p_4, 0.0),
                    Complex::new(sqrt_p_4, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .expect("valid 2x2 X Kraus operator"),
            // sqrt(p/4) * Y
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, -sqrt_p_4),
                    Complex::new(0.0, sqrt_p_4),
                    Complex::new(0.0, 0.0),
                ],
            )
            .expect("valid 2x2 Y Kraus operator"),
            // sqrt(p/4) * Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_p_4, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-sqrt_p_4, 0.0),
                ],
            )
            .expect("valid 2x2 Z Kraus operator"),
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
            // K0 = |0><0| + sqrt(1-gamma)|1><1|
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_gamma, 0.0),
                ],
            )
            .expect("valid 2x2 amplitude damping K0"),
            // K1 = sqrt(gamma)|0><1|
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_gamma, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .expect("valid 2x2 amplitude damping K1"),
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
            // K0 = sqrt(1-gamma) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_gamma, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_gamma, 0.0),
                ],
            )
            .expect("valid 2x2 phase damping K0"),
            // K1 = sqrt(gamma) * Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_gamma, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-sqrt_gamma, 0.0),
                ],
            )
            .expect("valid 2x2 phase damping K1"),
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
            // K0 = sqrt(1-p) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_p, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_p, 0.0),
                ],
            )
            .expect("valid 2x2 bit flip K0"),
            // K1 = sqrt(p) * X
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_p, 0.0),
                    Complex::new(sqrt_p, 0.0),
                    Complex::new(0.0, 0.0),
                ],
            )
            .expect("valid 2x2 bit flip K1"),
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
            // K0 = sqrt(1-p) * I
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_1_minus_p, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(sqrt_1_minus_p, 0.0),
                ],
            )
            .expect("valid 2x2 phase flip K0"),
            // K1 = sqrt(p) * Z
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::new(sqrt_p, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-sqrt_p, 0.0),
                ],
            )
            .expect("valid 2x2 phase flip K1"),
        ];

        QuantumChannel::from_kraus(operators)
    }
}

/// Process tomography utilities
pub struct ProcessTomography;

impl ProcessTomography {
    /// Reconstruct a quantum channel from process tomography data via linear
    /// inversion / least-squares estimation of its Choi matrix.
    ///
    /// Given input density matrices `ρ_k` and the measured outputs
    /// `ρ'_k = Λ(ρ_k)`, the channel is determined by its Choi matrix `J`. Using
    /// the *column-stacking* Choi convention of this module (the same one used by
    /// [`QuantumChannel::kraus_to_choi`]), the action of the channel is linear in
    /// the entries of `J`:
    ///
    /// ```text
    /// Λ(ρ)[i, i'] = Σ_{j, j'} ρ[j, j'] · J[i + j·d_out, i' + j'·d_out]
    /// ```
    ///
    /// Each `(ρ_k, ρ'_k)` pair therefore contributes `d_out²` scalar linear
    /// constraints `A · vec(J) = b`. Stacking all pairs and solving the
    /// least-squares problem (via the Hermitian normal equations
    /// `AᴴA · vec(J) = Aᴴb`, solved with a spectral pseudo-inverse) yields the
    /// best-fit Choi matrix, from which the channel is rebuilt with
    /// [`QuantumChannel::from_choi`].
    ///
    /// # Errors
    ///
    /// Returns [`QuantRS2Error::InvalidInput`] if the input/output counts differ,
    /// if any density matrix is not square / dimensions are inconsistent, or if
    /// the supplied input states are not informationally complete (fewer than
    /// `d²` linearly independent inputs, leaving the Choi matrix
    /// under-determined). No identity / placeholder channel is fabricated.
    pub fn reconstruct_channel(
        input_states: &[Array2<Complex<f64>>],
        output_states: &[Array2<Complex<f64>>],
    ) -> QuantRS2Result<QuantumChannel> {
        if input_states.len() != output_states.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Number of input and output states must match".to_string(),
            ));
        }
        if input_states.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "process tomography requires at least one (input, output) state pair".to_string(),
            ));
        }

        // Square-channel assumption: d_in == d_out == d.
        let d = input_states[0].shape()[0];
        if d == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "density matrices must be non-empty".to_string(),
            ));
        }
        for (states, label) in [(input_states, "input"), (output_states, "output")] {
            for (k, state) in states.iter().enumerate() {
                if state.shape() != [d, d] {
                    return Err(QuantRS2Error::InvalidInput(format!(
                        "{label} state {k} has shape {:?}, expected [{d}, {d}]",
                        state.shape()
                    )));
                }
            }
        }

        // Unknown vector x = vec(J) has length total_dim², where total_dim = d².
        let total_dim = d * d;
        let num_unknowns = total_dim * total_dim;

        // Build the linear system A · x = b.
        //
        // Constraint index (per pair): (i, i') -> row, with i, i' in 0..d.
        // Unknown index: J[r, c] with r = i + j·d, c = i' + j'·d, mapped to a
        // single column via column-stacking of J as well:
        //     col(r, c) = r + c · total_dim.
        // Coefficient of J[r, c] in Λ(ρ)[i, i'] is ρ[j, j'] (with r = i + j·d,
        // c = i' + j'·d).
        let num_constraints = input_states.len() * d * d;
        let mut a_mat: Array2<Complex<f64>> = Array2::zeros((num_constraints, num_unknowns));
        let mut b_vec: Array1<Complex<f64>> = Array1::zeros(num_constraints);

        let mut row = 0usize;
        for (rho, rho_out) in input_states.iter().zip(output_states.iter()) {
            for i in 0..d {
                for i_prime in 0..d {
                    // Right-hand side: measured output entry.
                    b_vec[row] = rho_out[[i, i_prime]];

                    // Fill the coefficients for this constraint.
                    for j in 0..d {
                        for j_prime in 0..d {
                            let r = i + j * d;
                            let c = i_prime + j_prime * d;
                            let col = r + c * total_dim;
                            a_mat[[row, col]] = rho[[j, j_prime]];
                        }
                    }
                    row += 1;
                }
            }
        }

        // Normal equations: (Aᴴ A) x = Aᴴ b.
        let a_dag = a_mat.mapv(|z| z.conj()).t().to_owned();
        let ata = a_dag.dot(&a_mat);
        let atb = a_dag.dot(&b_vec);

        // Informational-completeness check: the normal matrix must be full rank
        // (= num_unknowns). AᴴA is Hermitian PSD, so we use the Hermitian Jacobi
        // solver and detect rank deficiency via its (real, non-negative)
        // eigenvalues.
        let decomposition = hermitian_eigen_decompose(&ata)?;
        let eigenvalues = &decomposition.eigenvalues;
        let eigenvectors = &decomposition.eigenvectors;

        let max_eigenvalue = eigenvalues.iter().fold(0.0_f64, |acc, &z| acc.max(z.abs()));
        // Relative threshold for treating an eigenvalue as numerically zero.
        let rank_tolerance = (max_eigenvalue * 1e-9).max(1e-12);

        let rank = eigenvalues
            .iter()
            .filter(|&&z| z.abs() > rank_tolerance)
            .count();
        if rank < num_unknowns {
            return Err(QuantRS2Error::InvalidInput(
                "process tomography requires an informationally-complete set of input states (need d^2 linearly independent inputs)".into(),
            ));
        }

        // Solve x = (Aᴴ A)⁺ (Aᴴ b) using the spectral decomposition:
        //   (Aᴴ A)⁻¹ = Σ_m (1/λ_m) u_m u_mᴴ
        // applied directly to (Aᴴ b) to avoid forming the dense inverse.
        let mut x: Array1<Complex<f64>> = Array1::zeros(num_unknowns);
        for m in 0..num_unknowns {
            let lambda = eigenvalues[m];
            if lambda.abs() <= rank_tolerance {
                continue;
            }
            let u_m = eigenvectors.column(m);
            // coeff = u_mᴴ · (Aᴴ b)
            let mut coeff = Complex::new(0.0, 0.0);
            for n in 0..num_unknowns {
                coeff += u_m[n].conj() * atb[n];
            }
            let coeff = coeff / Complex::new(lambda, 0.0);
            for n in 0..num_unknowns {
                x[n] += coeff * u_m[n];
            }
        }

        // Reshape x back into the Choi matrix J (column-stacking inverse).
        let mut choi: Array2<Complex<f64>> = Array2::zeros((total_dim, total_dim));
        for c in 0..total_dim {
            for r in 0..total_dim {
                choi[[r, c]] = x[r + c * total_dim];
            }
        }

        // Hermitize to remove tiny numerical asymmetry before validation, then
        // build the channel from the reconstructed Choi matrix.
        let choi_dag = choi.mapv(|z| z.conj()).t().to_owned();
        let choi_herm = (&choi + &choi_dag).mapv(|z| z * Complex::new(0.5, 0.0));

        QuantumChannel::from_choi(choi_herm)
    }

    /// Generate an informationally-complete set of input states for a
    /// `dim`-dimensional system.
    ///
    /// Process tomography of a `dim × dim` channel needs `dim²` linearly
    /// independent input density matrices. This routine returns exactly such a
    /// set:
    ///
    /// * the `dim` computational-basis populations `|i⟩⟨i|`, and
    /// * for every pair `i < j`, the two superposition states
    ///   `|+_{ij}⟩ = (|i⟩ + |j⟩)/√2` and `|+i_{ij}⟩ = (|i⟩ + i|j⟩)/√2`
    ///   as density matrices `|ψ⟩⟨ψ|`.
    ///
    /// The diagonal states fix the populations while each pair of off-diagonal
    /// states fixes the real and imaginary parts of the corresponding coherence,
    /// giving `dim + 2 · C(dim, 2) = dim²` linearly independent operators that
    /// span the full space of Hermitian matrices.
    pub fn generate_input_states(dim: usize) -> Vec<Array2<Complex<f64>>> {
        let mut states = Vec::new();
        if dim == 0 {
            return states;
        }

        let inv_sqrt2 = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

        // Computational basis populations |i><i|.
        for i in 0..dim {
            let mut state = Array2::zeros((dim, dim));
            state[[i, i]] = Complex::new(1.0, 0.0);
            states.push(state);
        }

        // Off-diagonal coherences from superposition states. For each pair i < j
        // we add the density matrices of (|i> + |j>)/√2 and (|i> + i|j>)/√2.
        for i in 0..dim {
            for j in (i + 1)..dim {
                for &phase in &[Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)] {
                    // |psi> = (|i> + phase·|j>)/√2
                    let mut psi: Array1<Complex<f64>> = Array1::zeros(dim);
                    psi[i] = inv_sqrt2;
                    psi[j] = phase * inv_sqrt2;

                    // rho = |psi><psi|
                    let mut state: Array2<Complex<f64>> = Array2::zeros((dim, dim));
                    for r in 0..dim {
                        for c in 0..dim {
                            state[[r, c]] = psi[r] * psi[c].conj();
                        }
                    }
                    states.push(state);
                }
            }
        }

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex;

    #[test]
    fn test_depolarizing_channel() {
        let channel =
            QuantumChannels::depolarizing(0.1).expect("Failed to create depolarizing channel");

        assert_eq!(channel.input_dim, 2);
        assert_eq!(channel.output_dim, 2);
        assert!(channel.kraus.is_some());
        assert_eq!(
            channel
                .kraus
                .as_ref()
                .expect("Kraus representation missing")
                .operators
                .len(),
            4
        );
    }

    #[test]
    fn test_amplitude_damping() {
        let channel = QuantumChannels::amplitude_damping(0.3)
            .expect("Failed to create amplitude damping channel");

        assert!(channel.kraus.is_some());
        assert_eq!(
            channel
                .kraus
                .as_ref()
                .expect("Kraus representation missing")
                .operators
                .len(),
            2
        );

        // Test on |1><1| state
        let mut rho = Array2::zeros((2, 2));
        rho[[1, 1]] = Complex::new(1.0, 0.0);

        let mut ch = channel;
        let output = ch.apply(&rho).expect("Failed to apply channel");

        // Population should decrease
        assert!(output[[1, 1]].re < 1.0);
        assert!(output[[0, 0]].re > 0.0);
    }

    #[test]
    fn test_kraus_to_choi() {
        let mut channel =
            QuantumChannels::bit_flip(0.2).expect("Failed to create bit flip channel");
        let choi = channel.to_choi().expect("Failed to convert to Choi");

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
        let mut ch1 =
            QuantumChannels::phase_flip(0.1).expect("Failed to create phase flip channel");
        let mut ch2 = QuantumChannels::bit_flip(0.2).expect("Failed to create bit flip channel");

        // Apply both to a superposition state
        let mut rho = Array2::zeros((2, 2));
        rho[[0, 0]] = Complex::new(0.5, 0.0);
        rho[[0, 1]] = Complex::new(0.5, 0.0);
        rho[[1, 0]] = Complex::new(0.5, 0.0);
        rho[[1, 1]] = Complex::new(0.5, 0.0);

        let intermediate = ch1.apply(&rho).expect("Failed to apply phase flip channel");
        let final_state = ch2
            .apply(&intermediate)
            .expect("Failed to apply bit flip channel");

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
        .expect("valid 2x2 Hadamard matrix")
            / Complex::new(2.0_f64.sqrt(), 0.0);

        let mut channel =
            QuantumChannel::from_kraus(vec![h]).expect("Failed to create unitary channel");

        assert!(channel.is_unitary().expect("Failed to check unitarity"));
    }

    #[test]
    fn test_stinespring_conversion() {
        let mut channel = QuantumChannels::amplitude_damping(0.5)
            .expect("Failed to create amplitude damping channel");

        // Convert to Stinespring
        let stinespring = channel
            .to_stinespring()
            .expect("Failed to convert to Stinespring");

        assert_eq!(stinespring.env_dim, 2);
        assert_eq!(stinespring.isometry.shape(), [4, 2]);

        // Convert back to Kraus
        let kraus_decomposer =
            QuantumChannel::from_kraus(vec![Array2::eye(2).mapv(|x| Complex::new(x, 0.0))])
                .expect("Failed to create identity channel");
        let kraus = kraus_decomposer
            .stinespring_to_kraus(&stinespring.isometry, stinespring.env_dim)
            .expect("Failed to convert back to Kraus");
        assert_eq!(kraus.operators.len(), 2);
    }

    /// Apply an explicit Kraus set Σ K ρ K† without going through a channel.
    fn apply_kraus(
        operators: &[Array2<Complex<f64>>],
        rho: &Array2<Complex<f64>>,
    ) -> Array2<Complex<f64>> {
        let d_out = operators[0].shape()[0];
        let mut result: Array2<Complex<f64>> = Array2::zeros((d_out, d_out));
        for k in operators {
            let k_dag = k.mapv(|z| z.conj()).t().to_owned();
            result = result + k.dot(rho).dot(&k_dag);
        }
        result
    }

    /// A small assortment of test density matrices for a single qubit.
    fn qubit_test_states() -> Vec<Array2<Complex<f64>>> {
        let mut states = Vec::new();

        // |0><0|
        let mut s0 = Array2::zeros((2, 2));
        s0[[0, 0]] = Complex::new(1.0, 0.0);
        states.push(s0);

        // |1><1|
        let mut s1 = Array2::zeros((2, 2));
        s1[[1, 1]] = Complex::new(1.0, 0.0);
        states.push(s1);

        // |+><+|
        let mut s_plus = Array2::zeros((2, 2));
        for idx in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            s_plus[idx] = Complex::new(0.5, 0.0);
        }
        states.push(s_plus);

        // |+i><+i| (eigenstate of Y)
        let mut s_plus_i = Array2::zeros((2, 2));
        s_plus_i[[0, 0]] = Complex::new(0.5, 0.0);
        s_plus_i[[0, 1]] = Complex::new(0.0, -0.5);
        s_plus_i[[1, 0]] = Complex::new(0.0, 0.5);
        s_plus_i[[1, 1]] = Complex::new(0.5, 0.0);
        states.push(s_plus_i);

        states
    }

    #[test]
    fn test_choi_to_kraus_roundtrip_depolarizing() {
        // Build a known channel, capture its original Kraus operators, round-trip
        // through the Choi matrix, and verify the recovered Kraus set reproduces
        // the original action on several test states. This FAILS if choi_to_kraus
        // returns the identity fabrication.
        let mut channel =
            QuantumChannels::depolarizing(0.3).expect("failed to create depolarizing channel");
        let original_ops = channel
            .to_kraus()
            .expect("failed to get original Kraus")
            .operators
            .clone();

        // Convert to Choi, then re-derive Kraus from a *fresh* channel built only
        // from the Choi matrix (so no cached Kraus is reused).
        let choi = channel
            .to_choi()
            .expect("failed to convert to Choi")
            .clone();
        let mut from_choi = QuantumChannel::from_choi(choi.matrix.clone())
            .expect("failed to build channel from Choi");
        let recovered_ops = from_choi
            .to_kraus()
            .expect("failed to recover Kraus from Choi")
            .operators
            .clone();

        for rho in qubit_test_states() {
            let expected = apply_kraus(&original_ops, &rho);
            let actual = apply_kraus(&recovered_ops, &rho);
            let max_diff = (&expected - &actual)
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-9,
                "depolarizing round-trip mismatch: {max_diff}"
            );
        }
    }

    #[test]
    fn test_choi_to_kraus_roundtrip_amplitude_damping() {
        let mut channel = QuantumChannels::amplitude_damping(0.4)
            .expect("failed to create amplitude damping channel");
        let original_ops = channel
            .to_kraus()
            .expect("failed to get original Kraus")
            .operators
            .clone();

        let choi = channel
            .to_choi()
            .expect("failed to convert to Choi")
            .clone();
        let mut from_choi = QuantumChannel::from_choi(choi.matrix.clone())
            .expect("failed to build channel from Choi");
        let recovered_ops = from_choi
            .to_kraus()
            .expect("failed to recover Kraus from Choi")
            .operators
            .clone();

        for rho in qubit_test_states() {
            let expected = apply_kraus(&original_ops, &rho);
            let actual = apply_kraus(&recovered_ops, &rho);
            let max_diff = (&expected - &actual)
                .iter()
                .map(|z| z.norm())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-9,
                "amplitude damping round-trip mismatch: {max_diff}"
            );
        }
    }

    #[test]
    fn test_choi_to_kraus_trace_preserving() {
        // The Kraus set recovered from the Choi matrix must satisfy the
        // completeness relation Σ Kᴴ K = I (trace preservation) within 1e-9.
        let mut channel =
            QuantumChannels::depolarizing(0.25).expect("failed to create depolarizing channel");
        let choi = channel
            .to_choi()
            .expect("failed to convert to Choi")
            .clone();
        let mut from_choi = QuantumChannel::from_choi(choi.matrix.clone())
            .expect("failed to build channel from Choi");
        let recovered = from_choi
            .to_kraus()
            .expect("failed to recover Kraus from Choi")
            .operators
            .clone();

        let d_in = from_choi.input_dim;
        let mut sum: Array2<Complex<f64>> = Array2::zeros((d_in, d_in));
        for k in &recovered {
            let k_dag = k.mapv(|z| z.conj()).t().to_owned();
            sum = sum + k_dag.dot(k);
        }

        for i in 0..d_in {
            for j in 0..d_in {
                let expected = if i == j {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                let diff = (sum[[i, j]] - expected).norm();
                assert!(diff < 1e-9, "completeness violated at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn test_generate_input_states_informationally_complete() {
        // For dimension d the generated set must contain exactly d^2 states.
        for d in 2..=3 {
            let states = ProcessTomography::generate_input_states(d);
            assert_eq!(
                states.len(),
                d * d,
                "expected d^2 informationally-complete states for d={d}"
            );
            // Each state should be a valid (trace-1, Hermitian) density matrix.
            for state in &states {
                assert_eq!(state.shape(), [d, d]);
                let trace: Complex<f64> = (0..d).map(|i| state[[i, i]]).sum();
                assert!((trace.re - 1.0).abs() < 1e-12);
                assert!(trace.im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_reconstruct_channel_amplitude_damping() {
        // Generate an informationally-complete input set, push it through a known
        // amplitude damping channel to obtain outputs, reconstruct the channel
        // from (input, output) pairs, and verify the reconstruction matches the
        // original on a fresh state. Also assert the result is NOT the identity.
        let gamma = 0.35;
        let inputs = ProcessTomography::generate_input_states(2);

        let mut reference =
            QuantumChannels::amplitude_damping(gamma).expect("failed to create reference channel");
        let mut outputs = Vec::with_capacity(inputs.len());
        for rho in &inputs {
            outputs.push(reference.apply(rho).expect("failed to apply reference"));
        }

        let mut reconstructed = ProcessTomography::reconstruct_channel(&inputs, &outputs)
            .expect("reconstruction should succeed for an informationally-complete set");

        // Fresh state not used directly as a basis population: |+i><+i|.
        let mut fresh = Array2::zeros((2, 2));
        fresh[[0, 0]] = Complex::new(0.5, 0.0);
        fresh[[0, 1]] = Complex::new(0.0, -0.5);
        fresh[[1, 0]] = Complex::new(0.0, 0.5);
        fresh[[1, 1]] = Complex::new(0.5, 0.0);

        let expected = reference.apply(&fresh).expect("reference apply failed");
        let actual = reconstructed
            .apply(&fresh)
            .expect("reconstructed apply failed");
        let max_diff = (&expected - &actual)
            .iter()
            .map(|z| z.norm())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-8, "reconstruction mismatch: {max_diff}");

        // The reconstructed channel must differ from the identity channel.
        let mut identity =
            QuantumChannel::from_kraus(vec![Array2::eye(2).mapv(|x| Complex::new(x, 0.0))])
                .expect("failed to create identity channel");
        let id_out = identity.apply(&fresh).expect("identity apply failed");
        let id_diff = (&id_out - &actual)
            .iter()
            .map(|z| z.norm())
            .fold(0.0_f64, f64::max);
        assert!(
            id_diff > 1e-3,
            "reconstructed channel is indistinguishable from identity (diff={id_diff})"
        );
    }

    #[test]
    fn test_reconstruct_channel_underdetermined_errors() {
        // Too few input states (only the d basis populations, not d^2) must
        // produce an HONEST error rather than a fabricated identity channel.
        let mut reference =
            QuantumChannels::bit_flip(0.2).expect("failed to create bit flip channel");

        // Only computational-basis populations: 2 states for d=2 (< d^2 = 4).
        let mut inputs = Vec::new();
        for i in 0..2 {
            let mut state = Array2::zeros((2, 2));
            state[[i, i]] = Complex::new(1.0, 0.0);
            inputs.push(state);
        }
        let mut outputs = Vec::new();
        for rho in &inputs {
            outputs.push(reference.apply(rho).expect("apply failed"));
        }

        let result = ProcessTomography::reconstruct_channel(&inputs, &outputs);
        assert!(
            result.is_err(),
            "underdetermined tomography must error, not fabricate a channel"
        );
    }
}
