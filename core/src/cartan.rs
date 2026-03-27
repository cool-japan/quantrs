//! Cartan (KAK) decomposition for two-qubit unitaries
//!
//! This module implements the Cartan decomposition, which decomposes any
//! two-qubit unitary into a canonical form with at most 3 CNOT gates.
//! The decomposition has the form:
//!
//! U = (A₁ ⊗ B₁) · exp(i(aXX + bYY + cZZ)) · (A₂ ⊗ B₂)
//!
//! where A₁, B₁, A₂, B₂ are single-qubit unitaries and a, b, c are real.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::*, single::*, GateOp},
    matrix_ops::{DenseMatrix, QuantumMatrix},
    qubit::QubitId,
    synthesis::{decompose_single_qubit_zyz, SingleQubitDecomposition},
};
use rustc_hash::FxHashMap;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::Complex;
use std::f64::consts::PI;

/// Result of Cartan decomposition for a two-qubit unitary
#[derive(Debug, Clone)]
pub struct CartanDecomposition {
    /// Left single-qubit gates (A₁, B₁)
    pub left_gates: (SingleQubitDecomposition, SingleQubitDecomposition),
    /// Right single-qubit gates (A₂, B₂)
    pub right_gates: (SingleQubitDecomposition, SingleQubitDecomposition),
    /// Interaction coefficients (a, b, c) for exp(i(aXX + bYY + cZZ))
    pub interaction: CartanCoefficients,
    /// Global phase
    pub global_phase: f64,
}

/// Cartan interaction coefficients
#[derive(Debug, Clone, Copy)]
pub struct CartanCoefficients {
    /// Coefficient for XX interaction
    pub xx: f64,
    /// Coefficient for YY interaction
    pub yy: f64,
    /// Coefficient for ZZ interaction
    pub zz: f64,
}

impl CartanCoefficients {
    /// Create new coefficients
    pub const fn new(xx: f64, yy: f64, zz: f64) -> Self {
        Self { xx, yy, zz }
    }

    /// Check if this is equivalent to identity (all coefficients near zero)
    pub fn is_identity(&self, tolerance: f64) -> bool {
        self.xx.abs() < tolerance && self.yy.abs() < tolerance && self.zz.abs() < tolerance
    }

    /// Get the number of CNOTs required
    pub fn cnot_count(&self, tolerance: f64) -> usize {
        let eps = tolerance;

        // Special cases based on coefficients
        if self.is_identity(eps) {
            0
        } else if (self.xx - self.yy).abs() < eps && self.zz.abs() < eps {
            // a = b, c = 0: Can be done with 2 CNOTs
            2
        } else if (self.xx - PI / 4.0).abs() < eps
            && (self.yy - PI / 4.0).abs() < eps
            && (self.zz - PI / 4.0).abs() < eps
        {
            // Maximally entangling: exactly 3 CNOTs
            3
        } else if self.xx.abs() < eps || self.yy.abs() < eps || self.zz.abs() < eps {
            // One coefficient is zero: 2 CNOTs
            2
        } else {
            // General case: 3 CNOTs
            3
        }
    }

    /// Convert to canonical form with ordered coefficients
    pub fn canonicalize(&mut self) {
        // Ensure |xx| >= |yy| >= |zz| by permutation
        let mut vals = [
            (self.xx.abs(), self.xx, 0),
            (self.yy.abs(), self.yy, 1),
            (self.zz.abs(), self.zz, 2),
        ];
        vals.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .expect("Failed to compare Cartan coefficients in CartanCoefficients::canonicalize")
        });

        self.xx = vals[0].1;
        self.yy = vals[1].1;
        self.zz = vals[2].1;
    }
}

/// Cartan decomposer for two-qubit gates
pub struct CartanDecomposer {
    /// Tolerance for numerical comparisons
    tolerance: f64,
    /// Cache for common gates
    #[allow(dead_code)]
    cache: FxHashMap<u64, CartanDecomposition>,
}

impl CartanDecomposer {
    /// Create a new Cartan decomposer
    pub fn new() -> Self {
        Self {
            tolerance: 1e-10,
            cache: FxHashMap::default(),
        }
    }

    /// Create with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            cache: FxHashMap::default(),
        }
    }

    /// Decompose a two-qubit unitary using Cartan decomposition
    pub fn decompose(
        &mut self,
        unitary: &Array2<Complex<f64>>,
    ) -> QuantRS2Result<CartanDecomposition> {
        // Validate input
        if unitary.shape() != [4, 4] {
            return Err(QuantRS2Error::InvalidInput(
                "Cartan decomposition requires 4x4 unitary".to_string(),
            ));
        }

        // Check unitarity
        let mat = DenseMatrix::new(unitary.clone())?;
        if !mat.is_unitary(self.tolerance)? {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix is not unitary".to_string(),
            ));
        }

        // Transform to magic basis
        let magic_basis = Self::get_magic_basis();
        let u_magic = Self::to_magic_basis(unitary, &magic_basis);

        // Compute M = U_magic^T · U_magic
        let u_magic_t = u_magic.t().to_owned();
        let m = u_magic_t.dot(&u_magic);

        // Diagonalize M to find the canonical form
        let (d, p) = Self::diagonalize_symmetric(&m)?;

        // Extract interaction coefficients from eigenvalues
        let coeffs = Self::extract_coefficients(&d);

        // Compute single-qubit gates
        let (left_gates, right_gates) = self.compute_local_gates(unitary, &u_magic, &p, &coeffs)?;

        // Compute global phase
        let global_phase = Self::compute_global_phase(unitary, &left_gates, &right_gates, &coeffs)?;

        Ok(CartanDecomposition {
            left_gates,
            right_gates,
            interaction: coeffs,
            global_phase,
        })
    }

    /// Get the magic basis transformation matrix
    fn get_magic_basis() -> Array2<Complex<f64>> {
        let sqrt2 = 2.0_f64.sqrt();
        Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
        )
        .expect("Failed to create magic basis matrix in CartanDecomposer::get_magic_basis")
            / Complex::new(sqrt2, 0.0)
    }

    /// Transform matrix to magic basis
    fn to_magic_basis(
        u: &Array2<Complex<f64>>,
        magic: &Array2<Complex<f64>>,
    ) -> Array2<Complex<f64>> {
        let magic_dag = magic.mapv(|z| z.conj()).t().to_owned();
        magic_dag.dot(u).dot(magic)
    }

    /// Diagonalize a symmetric complex matrix via QR iteration
    ///
    /// For the Cartan decomposition, M = U^T U is complex symmetric.
    /// Its eigenvalues are complex numbers on the unit circle: exp(2i*phi_k).
    /// Returns (eigenvalues as Complex, approximate eigenvectors).
    fn diagonalize_symmetric(
        m: &Array2<Complex<f64>>,
    ) -> QuantRS2Result<(Array1<Complex<f64>>, Array2<Complex<f64>>)> {
        let n = m.nrows();
        // QR iteration with Francis shifts to find all eigenvalues of the complex matrix M.
        // We work with a Hessenberg reduction first, then apply shifted QR steps.
        let mut h = m.to_owned();
        let mut q = Array2::<Complex<f64>>::eye(n);

        // Reduce to upper Hessenberg form using Householder reflections
        for k in 0..n.saturating_sub(2) {
            // Build Householder vector from column k, rows k+1..n
            let col: Vec<Complex<f64>> = (k + 1..n).map(|i| h[[i, k]]).collect();
            let sigma_sq: f64 = col.iter().map(|z| z.norm_sqr()).sum();
            let sigma = sigma_sq.sqrt();
            if sigma < 1e-14 {
                continue;
            }
            // Choose Householder sign to maximise numerical stability
            let phase = if col[0].norm() > 1e-14 {
                col[0] / col[0].norm()
            } else {
                Complex::new(1.0, 0.0)
            };
            let mut v = col.clone();
            v[0] = v[0] + phase * sigma;
            let v_norm_sq: f64 = v.iter().map(|z| z.norm_sqr()).sum();
            if v_norm_sq < 1e-28 {
                continue;
            }
            let m_len = v.len(); // = n - (k+1)

            // Apply H from the left: h[k+1.., ..] -= (2/v^†v) v (v^† h[k+1.., ..])
            for j in 0..n {
                let dot: Complex<f64> = (0..m_len).map(|i| v[i].conj() * h[[k + 1 + i, j]]).sum();
                let scale = dot * Complex::new(2.0 / v_norm_sq, 0.0);
                for i in 0..m_len {
                    h[[k + 1 + i, j]] = h[[k + 1 + i, j]] - v[i] * scale;
                }
            }
            // Apply H from the right: h[.., k+1..] -= (2/v^†v) (h[.., k+1..] v) v^†
            for i in 0..n {
                let dot: Complex<f64> = (0..m_len).map(|j| h[[i, k + 1 + j]] * v[j]).sum();
                let scale = dot * Complex::new(2.0 / v_norm_sq, 0.0);
                for j in 0..m_len {
                    h[[i, k + 1 + j]] = h[[i, k + 1 + j]] - scale * v[j].conj();
                }
            }
            // Accumulate Q
            for i in 0..n {
                let dot: Complex<f64> = (0..m_len).map(|j| q[[i, k + 1 + j]] * v[j]).sum();
                let scale = dot * Complex::new(2.0 / v_norm_sq, 0.0);
                for j in 0..m_len {
                    q[[i, k + 1 + j]] = q[[i, k + 1 + j]] - scale * v[j].conj();
                }
            }
        }

        // Francis double-shift QR iteration on the Hessenberg matrix
        let max_iter = 300 * n;
        let mut active = n;
        for _iter in 0..max_iter {
            if active <= 1 {
                break;
            }
            // Deflate converged eigenvalues at the bottom
            while active > 1 {
                let off = h[[active - 1, active - 2]].norm();
                let d1 = h[[active - 1, active - 1]].norm();
                let d0 = h[[active - 2, active - 2]].norm();
                if off < 1e-12 * (d1 + d0) {
                    active -= 1;
                } else {
                    break;
                }
            }
            if active <= 1 {
                break;
            }

            // Wilkinson (single complex) shift: eigenvalue of bottom 2x2 closest to h[a-1,a-1]
            let a = active;
            let s = h[[a - 1, a - 1]];

            // Single-shift QR step: compute Givens rotations to push shift through
            // Apply shift: h' = h - s*I, QR decompose, then h'' = RQ + s*I
            for k in 0..a - 1 {
                // Compute Givens rotation to zero h[k+1, k]
                let x = h[[k, k]] - s;
                let y = h[[k + 1, k]];
                let r = (x.norm_sqr() + y.norm_sqr()).sqrt();
                if r < 1e-14 {
                    continue;
                }
                let c_val = x / r;
                let s_val = -y / r;

                // Apply Givens rotation from left: rows k and k+1
                for j in 0..n {
                    let tmp0 = c_val * h[[k, j]] - s_val.conj() * h[[k + 1, j]];
                    let tmp1 = s_val * h[[k, j]] + c_val.conj() * h[[k + 1, j]];
                    h[[k, j]] = tmp0;
                    h[[k + 1, j]] = tmp1;
                }
                // Apply Givens rotation from right: cols k and k+1
                for i in 0..n {
                    let tmp0 = c_val.conj() * h[[i, k]] - s_val.conj() * h[[i, k + 1]];
                    let tmp1 = s_val * h[[i, k]] + c_val * h[[i, k + 1]];
                    h[[i, k]] = tmp0;
                    h[[i, k + 1]] = tmp1;
                }
                // Accumulate in Q
                for i in 0..n {
                    let tmp0 = c_val.conj() * q[[i, k]] - s_val.conj() * q[[i, k + 1]];
                    let tmp1 = s_val * q[[i, k]] + c_val * q[[i, k + 1]];
                    q[[i, k]] = tmp0;
                    q[[i, k + 1]] = tmp1;
                }
            }
        }

        // Extract eigenvalues from the diagonal of h
        let mut eigenvalues = Array1::zeros(n);
        for i in 0..n {
            eigenvalues[i] = h[[i, i]];
        }

        Ok((eigenvalues, q))
    }

    /// Extract Cartan coefficients from complex eigenvalues of M = U^T U
    ///
    /// The eigenvalues are exp(2i·phi_k). For U = exp(i(aXX + bYY + cZZ)),
    /// the phases come in pairs: {+(a+b+c), +(a-b-c), +(-a+b-c), +(-a-b+c)}.
    /// Sorting and averaging the phases gives:
    ///   a = (phi_0 + phi_1 - phi_2 - phi_3) / 4   (after appropriate ordering)
    /// More robustly: solve the 4×4 linear system.
    fn extract_coefficients(eigenvalues: &Array1<Complex<f64>>) -> CartanCoefficients {
        // Extract phases phi_k from eigenvalues exp(2i*phi_k)
        // The arg() gives 2*phi, so phi = arg/2
        let mut phases: Vec<f64> = eigenvalues.iter().map(|z| z.arg() / 2.0).collect();
        // Sort phases for stable extraction
        phases.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // For the four Cartan phases {a+b+c, a-b-c, -a+b-c, -a-b+c}:
        // Sum = 0, so use differences.
        // Ordered phases p0 <= p1 <= p2 <= p3 with p0+p3 ≈ 0, p1+p2 ≈ 0
        // a = (p3 - p2 + p1 - p0) / 4 ... but sign ordering depends on values.
        // Use the symmetric formula: after sorting ascending,
        //   a+b+c corresponds to the largest magnitude phase
        //   We identify:
        //     c = (p3 - p0) / 4   (half the spread of extreme phases)
        //     b = (p2 - p1) / 4   (half the spread of middle phases)
        //     a ≈ (p3 + p2 - p1 - p0) / 4
        let p0 = phases.first().copied().unwrap_or(0.0);
        let p1 = phases.get(1).copied().unwrap_or(0.0);
        let p2 = phases.get(2).copied().unwrap_or(0.0);
        let p3 = phases.get(3).copied().unwrap_or(0.0);

        // Solve: a+b+c=p3, a-b-c=p0, -a+b-c=p1, -a-b+c=p2  (one consistent assignment)
        // Adding all: 0 = p0+p1+p2+p3 (true up to 2π ambiguity)
        // From (p3+p0)/2 = a and (p3-p0)/2 = b+c
        // From (p2+p1)/2 = -a and (p2-p1)/2 = c-b
        let a = (p3 + p0) / 2.0;
        let b_plus_c = (p3 - p0) / 2.0;
        let c_minus_b = (p2 - p1) / 2.0;
        let b = (b_plus_c - c_minus_b) / 2.0;
        let c = (b_plus_c + c_minus_b) / 2.0;

        let mut coeffs = CartanCoefficients::new(a, b, c);
        coeffs.canonicalize();
        coeffs
    }

    /// Compute single-qubit gates from decomposition
    fn compute_local_gates(
        &self,
        u: &Array2<Complex<f64>>,
        _u_magic: &Array2<Complex<f64>>,
        _p: &Array2<Complex<f64>>,
        coeffs: &CartanCoefficients,
    ) -> QuantRS2Result<(
        (SingleQubitDecomposition, SingleQubitDecomposition),
        (SingleQubitDecomposition, SingleQubitDecomposition),
    )> {
        // Build the canonical gate
        let _canonical = Self::build_canonical_gate(coeffs);

        // The local gates satisfy:
        // U = (A₁ ⊗ B₁) · canonical · (A₂ ⊗ B₂)

        // Extract 2x2 blocks to find single-qubit gates
        // This is simplified - proper implementation uses the full KAK theorem

        let a1 = u.slice(s![..2, ..2]).to_owned();
        let b1 = u.slice(s![2..4, 2..4]).to_owned();

        let left_a = decompose_single_qubit_zyz(&a1.view())?;
        let left_b = decompose_single_qubit_zyz(&b1.view())?;

        // For right gates, we'd compute from the decomposition
        // For now, use identity
        let ident = Array2::eye(2);
        let right_a = decompose_single_qubit_zyz(&ident.view())?;
        let right_b = decompose_single_qubit_zyz(&ident.view())?;

        Ok(((left_a, left_b), (right_a, right_b)))
    }

    /// Build the canonical gate from coefficients
    fn build_canonical_gate(coeffs: &CartanCoefficients) -> Array2<Complex<f64>> {
        // exp(i(aXX + bYY + cZZ))
        let a = coeffs.xx;
        let b = coeffs.yy;
        let c = coeffs.zz;

        // Direct computation of matrix exponential for this special form
        let cos_a = a.cos();
        let sin_a = a.sin();
        let cos_b = b.cos();
        let sin_b = b.sin();
        let cos_c = c.cos();
        let sin_c = c.sin();

        // Build the 4x4 matrix
        let mut result = Array2::zeros((4, 4));

        // This is the explicit form of exp(i(aXX + bYY + cZZ))
        result[[0, 0]] = Complex::new(cos_a * cos_b * cos_c, sin_c);
        result[[0, 3]] = Complex::new(0.0, sin_a * cos_b * cos_c);
        result[[1, 1]] = Complex::new(cos_a * cos_c, -sin_a * sin_b * sin_c);
        result[[1, 2]] = Complex::new(0.0, cos_a.mul_add(sin_c, sin_a * sin_b * cos_c));
        result[[2, 1]] = Complex::new(0.0, cos_a.mul_add(sin_c, -(sin_a * sin_b * cos_c)));
        result[[2, 2]] = Complex::new(cos_a * cos_c, sin_a * sin_b * sin_c);
        result[[3, 0]] = Complex::new(0.0, sin_a * cos_b * cos_c);
        result[[3, 3]] = Complex::new(cos_a * cos_b * cos_c, -sin_c);

        result
    }

    /// Compute global phase
    const fn compute_global_phase(
        _u: &Array2<Complex<f64>>,
        _left: &(SingleQubitDecomposition, SingleQubitDecomposition),
        _right: &(SingleQubitDecomposition, SingleQubitDecomposition),
        _coeffs: &CartanCoefficients,
    ) -> QuantRS2Result<f64> {
        // Global phase is the phase difference between U and the reconstructed gate
        // For now, return 0
        Ok(0.0)
    }

    /// Convert Cartan decomposition to gate sequence
    pub fn to_gates(
        &self,
        decomp: &CartanDecomposition,
        qubit_ids: &[QubitId],
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if qubit_ids.len() != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Cartan decomposition requires exactly 2 qubits".to_string(),
            ));
        }

        let q0 = qubit_ids[0];
        let q1 = qubit_ids[1];
        let mut gates: Vec<Box<dyn GateOp>> = Vec::new();

        // Left single-qubit gates
        gates.extend(self.single_qubit_to_gates(&decomp.left_gates.0, q0));
        gates.extend(self.single_qubit_to_gates(&decomp.left_gates.1, q1));

        // Canonical two-qubit gate
        gates.extend(self.canonical_to_gates(&decomp.interaction, q0, q1)?);

        // Right single-qubit gates
        gates.extend(self.single_qubit_to_gates(&decomp.right_gates.0, q0));
        gates.extend(self.single_qubit_to_gates(&decomp.right_gates.1, q1));

        Ok(gates)
    }

    /// Convert single-qubit decomposition to gates
    fn single_qubit_to_gates(
        &self,
        decomp: &SingleQubitDecomposition,
        qubit: QubitId,
    ) -> Vec<Box<dyn GateOp>> {
        let mut gates = Vec::new();

        if decomp.theta1.abs() > self.tolerance {
            gates.push(Box::new(RotationZ {
                target: qubit,
                theta: decomp.theta1,
            }) as Box<dyn GateOp>);
        }

        if decomp.phi.abs() > self.tolerance {
            gates.push(Box::new(RotationY {
                target: qubit,
                theta: decomp.phi,
            }) as Box<dyn GateOp>);
        }

        if decomp.theta2.abs() > self.tolerance {
            gates.push(Box::new(RotationZ {
                target: qubit,
                theta: decomp.theta2,
            }) as Box<dyn GateOp>);
        }

        gates
    }

    /// Convert canonical coefficients to gate sequence
    fn canonical_to_gates(
        &self,
        coeffs: &CartanCoefficients,
        q0: QubitId,
        q1: QubitId,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = Vec::new();
        let cnots = coeffs.cnot_count(self.tolerance);

        match cnots {
            0 => {
                // Identity - no gates needed
            }
            1 => {
                // Special case: can be done with 1 CNOT
                gates.push(Box::new(CNOT {
                    control: q0,
                    target: q1,
                }));
            }
            2 => {
                // Can be done with 2 CNOTs
                // Add rotations
                if coeffs.xx.abs() > self.tolerance {
                    gates.push(Box::new(RotationX {
                        target: q0,
                        theta: coeffs.xx * 2.0,
                    }));
                }

                gates.push(Box::new(CNOT {
                    control: q0,
                    target: q1,
                }));

                if coeffs.zz.abs() > self.tolerance {
                    gates.push(Box::new(RotationZ {
                        target: q1,
                        theta: coeffs.zz * 2.0,
                    }));
                }

                gates.push(Box::new(CNOT {
                    control: q0,
                    target: q1,
                }));
            }
            3 => {
                // General case: 3 CNOTs with intermediate rotations
                gates.push(Box::new(CNOT {
                    control: q0,
                    target: q1,
                }));

                gates.push(Box::new(RotationZ {
                    target: q0,
                    theta: coeffs.xx * 2.0,
                }));
                gates.push(Box::new(RotationZ {
                    target: q1,
                    theta: coeffs.yy * 2.0,
                }));

                gates.push(Box::new(CNOT {
                    control: q1,
                    target: q0,
                }));

                gates.push(Box::new(RotationZ {
                    target: q0,
                    theta: coeffs.zz * 2.0,
                }));

                gates.push(Box::new(CNOT {
                    control: q0,
                    target: q1,
                }));
            }
            _ => unreachable!("CNOT count should be 0-3"),
        }

        Ok(gates)
    }
}

/// Optimized Cartan decomposer with special case handling
pub struct OptimizedCartanDecomposer {
    pub base: CartanDecomposer,
    /// Enable special case optimizations
    optimize_special_cases: bool,
    /// Enable phase optimization
    optimize_phase: bool,
}

impl OptimizedCartanDecomposer {
    /// Create new optimized decomposer
    pub fn new() -> Self {
        Self {
            base: CartanDecomposer::new(),
            optimize_special_cases: true,
            optimize_phase: true,
        }
    }

    /// Decompose with optimizations
    pub fn decompose(
        &mut self,
        unitary: &Array2<Complex<f64>>,
    ) -> QuantRS2Result<CartanDecomposition> {
        // Check for special cases first
        if self.optimize_special_cases {
            if let Some(special) = self.check_special_cases(unitary)? {
                return Ok(special);
            }
        }

        // Use base decomposition
        let mut decomp = self.base.decompose(unitary)?;

        // Optimize phase if enabled
        if self.optimize_phase {
            self.optimize_global_phase(&mut decomp);
        }

        Ok(decomp)
    }

    /// Check for special gate cases
    fn check_special_cases(
        &self,
        unitary: &Array2<Complex<f64>>,
    ) -> QuantRS2Result<Option<CartanDecomposition>> {
        // Check for CNOT
        if self.is_cnot(unitary) {
            return Ok(Some(Self::cnot_decomposition()));
        }

        // Check for controlled-Z
        if self.is_cz(unitary) {
            return Ok(Some(Self::cz_decomposition()));
        }

        // Check for SWAP
        if self.is_swap(unitary) {
            return Ok(Some(Self::swap_decomposition()));
        }

        Ok(None)
    }

    /// Check if matrix is CNOT
    fn is_cnot(&self, u: &Array2<Complex<f64>>) -> bool {
        let cnot = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .expect("Failed to create CNOT matrix in OptimizedCartanDecomposer::is_cnot");

        self.matrices_equal(u, &cnot)
    }

    /// Check if matrix is CZ
    fn is_cz(&self, u: &Array2<Complex<f64>>) -> bool {
        let cz = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
        )
        .expect("Failed to create CZ matrix in OptimizedCartanDecomposer::is_cz");

        self.matrices_equal(u, &cz)
    }

    /// Check if matrix is SWAP
    fn is_swap(&self, u: &Array2<Complex<f64>>) -> bool {
        let swap = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
        )
        .expect("Failed to create SWAP matrix in OptimizedCartanDecomposer::is_swap");

        self.matrices_equal(u, &swap)
    }

    /// Check matrix equality up to global phase
    fn matrices_equal(&self, a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>) -> bool {
        // Find first non-zero element
        let mut phase = Complex::new(1.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                if b[[i, j]].norm() > self.base.tolerance {
                    phase = a[[i, j]] / b[[i, j]];
                    break;
                }
            }
        }

        // Check all elements match up to phase
        for i in 0..4 {
            for j in 0..4 {
                if (a[[i, j]] - phase * b[[i, j]]).norm() > self.base.tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Decomposition for CNOT
    fn cnot_decomposition() -> CartanDecomposition {
        let ident = Array2::eye(2);
        let ident_decomp = decompose_single_qubit_zyz(&ident.view()).expect(
            "Failed to decompose identity in OptimizedCartanDecomposer::cnot_decomposition",
        );

        CartanDecomposition {
            left_gates: (ident_decomp.clone(), ident_decomp.clone()),
            right_gates: (ident_decomp.clone(), ident_decomp),
            interaction: CartanCoefficients::new(PI / 4.0, PI / 4.0, 0.0),
            global_phase: 0.0,
        }
    }

    /// Decomposition for CZ
    fn cz_decomposition() -> CartanDecomposition {
        let ident = Array2::eye(2);
        let ident_decomp = decompose_single_qubit_zyz(&ident.view())
            .expect("Failed to decompose identity in OptimizedCartanDecomposer::cz_decomposition");

        CartanDecomposition {
            left_gates: (ident_decomp.clone(), ident_decomp.clone()),
            right_gates: (ident_decomp.clone(), ident_decomp),
            interaction: CartanCoefficients::new(0.0, 0.0, PI / 4.0),
            global_phase: 0.0,
        }
    }

    /// Decomposition for SWAP
    fn swap_decomposition() -> CartanDecomposition {
        let ident = Array2::eye(2);
        let ident_decomp = decompose_single_qubit_zyz(&ident.view()).expect(
            "Failed to decompose identity in OptimizedCartanDecomposer::swap_decomposition",
        );

        CartanDecomposition {
            left_gates: (ident_decomp.clone(), ident_decomp.clone()),
            right_gates: (ident_decomp.clone(), ident_decomp),
            interaction: CartanCoefficients::new(PI / 4.0, PI / 4.0, PI / 4.0),
            global_phase: 0.0,
        }
    }

    /// Optimize global phase
    fn optimize_global_phase(&self, decomp: &mut CartanDecomposition) {
        // Absorb global phase into one of the single-qubit gates
        if decomp.global_phase.abs() > self.base.tolerance {
            decomp.left_gates.0.global_phase += decomp.global_phase;
            decomp.global_phase = 0.0;
        }
    }
}

/// Utility function for quick Cartan decomposition
pub fn cartan_decompose(unitary: &Array2<Complex<f64>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    let mut decomposer = CartanDecomposer::new();
    let decomp = decomposer.decompose(unitary)?;
    let qubit_ids = vec![QubitId(0), QubitId(1)];
    decomposer.to_gates(&decomp, &qubit_ids)
}

impl Default for OptimizedCartanDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CartanDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex;

    #[test]
    fn test_cartan_coefficients() {
        let coeffs = CartanCoefficients::new(0.1, 0.2, 0.3);
        assert!(!coeffs.is_identity(1e-10));
        assert_eq!(coeffs.cnot_count(1e-10), 3);

        let zero_coeffs = CartanCoefficients::new(0.0, 0.0, 0.0);
        assert!(zero_coeffs.is_identity(1e-10));
        assert_eq!(zero_coeffs.cnot_count(1e-10), 0);
    }

    #[test]
    fn test_cartan_cnot() {
        let mut decomposer = CartanDecomposer::new();

        // CNOT matrix
        let cnot = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .expect("Failed to create CNOT matrix in test_cartan_cnot");

        let decomp = decomposer
            .decompose(&cnot)
            .expect("Failed to decompose CNOT in test_cartan_cnot");

        // CNOT should have specific interaction coefficients
        assert!(decomp.interaction.cnot_count(1e-10) <= 1);
    }

    #[test]
    fn test_optimized_special_cases() {
        let mut opt_decomposer = OptimizedCartanDecomposer::new();

        // Test SWAP gate
        let swap = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
        )
        .expect("Failed to create SWAP matrix in test_optimized_special_cases");

        let decomp = opt_decomposer
            .decompose(&swap)
            .expect("Failed to decompose SWAP in test_optimized_special_cases");

        // SWAP requires exactly 3 CNOTs
        assert_eq!(decomp.interaction.cnot_count(1e-10), 3);
    }

    #[test]
    fn test_cartan_identity() {
        let mut decomposer = CartanDecomposer::new();

        // Identity matrix
        let identity = Array2::eye(4);
        let identity_complex = identity.mapv(|x| Complex::new(x, 0.0));

        let decomp = decomposer
            .decompose(&identity_complex)
            .expect("Failed to decompose identity in test_cartan_identity");

        // Identity should have zero interaction
        assert!(decomp.interaction.is_identity(1e-10));
        assert_eq!(decomp.interaction.cnot_count(1e-10), 0);
    }
}
