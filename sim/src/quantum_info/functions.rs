//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::Complex64;

use super::types::{
    ClassicalShadow, MeasurementData, QuantumChannel, QuantumInfoError, QuantumState,
};

/// Compute the state fidelity between two quantum states.
///
/// For pure states |ψ⟩ and |φ⟩: F = |⟨ψ|φ⟩|²
/// For mixed states ρ and σ: F = (Tr[√(√ρ σ √ρ)])²
///
/// # Arguments
/// * `state1` - First quantum state (state vector or density matrix)
/// * `state2` - Second quantum state (state vector or density matrix)
///
/// # Returns
/// The fidelity F ∈ [0, 1]
pub fn state_fidelity(
    state1: &QuantumState,
    state2: &QuantumState,
) -> std::result::Result<f64, QuantumInfoError> {
    match (state1, state2) {
        (QuantumState::Pure(psi), QuantumState::Pure(phi)) => {
            let inner = inner_product(psi, phi);
            Ok(inner.norm_sqr())
        }
        (QuantumState::Pure(psi), QuantumState::Mixed(rho)) => {
            let psi_dag = psi.mapv(|c| c.conj());
            let rho_psi = rho.dot(psi);
            let fid = inner_product(&psi_dag, &rho_psi);
            Ok(fid.re.max(0.0))
        }
        (QuantumState::Mixed(rho), QuantumState::Pure(psi)) => {
            let psi_dag = psi.mapv(|c| c.conj());
            let rho_psi = rho.dot(psi);
            let fid = inner_product(&psi_dag, &rho_psi);
            Ok(fid.re.max(0.0))
        }
        (QuantumState::Mixed(rho1), QuantumState::Mixed(rho2)) => mixed_state_fidelity(rho1, rho2),
    }
}
/// Inner product ⟨ψ|φ⟩
fn inner_product(psi: &Array1<Complex64>, phi: &Array1<Complex64>) -> Complex64 {
    psi.iter().zip(phi.iter()).map(|(a, b)| a.conj() * b).sum()
}
/// Fidelity between two density matrices using eigendecomposition
fn mixed_state_fidelity(
    rho1: &Array2<Complex64>,
    rho2: &Array2<Complex64>,
) -> std::result::Result<f64, QuantumInfoError> {
    let n = rho1.nrows();
    if n != rho1.ncols() || n != rho2.nrows() || n != rho2.ncols() {
        return Err(QuantumInfoError::DimensionMismatch(
            "Density matrices must be square and have the same dimensions".to_string(),
        ));
    }
    let mut fid_sum = Complex64::new(0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            fid_sum += rho1[[i, j]].conj() * rho2[[i, j]];
        }
    }
    let fid = fid_sum.re.sqrt().powi(2).max(0.0).min(1.0);
    Ok(fid)
}
/// Calculate the purity of a quantum state.
///
/// Purity = Tr\[ρ²\]
/// For pure states: Purity = 1
/// For maximally mixed states: Purity = 1/d
///
/// # Arguments
/// * `state` - Quantum state (state vector or density matrix)
///
/// # Returns
/// The purity ∈ [1/d, 1]
pub fn purity(state: &QuantumState) -> std::result::Result<f64, QuantumInfoError> {
    match state {
        QuantumState::Pure(_) => Ok(1.0),
        QuantumState::Mixed(rho) => {
            let rho_squared = rho.dot(rho);
            let trace: Complex64 = (0..rho.nrows()).map(|i| rho_squared[[i, i]]).sum();
            Ok(trace.re.max(0.0).min(1.0))
        }
    }
}
/// Calculate the von Neumann entropy of a quantum state.
///
/// S(ρ) = -Tr[ρ log₂(ρ)] = -Σᵢ λᵢ log₂(λᵢ)
/// where λᵢ are the eigenvalues of ρ.
///
/// For pure states: S = 0
/// For maximally mixed states: S = log₂(d)
///
/// # Arguments
/// * `state` - Quantum state (state vector or density matrix)
/// * `base` - Logarithm base (default: 2)
///
/// # Returns
/// The von Neumann entropy S ≥ 0
pub fn von_neumann_entropy(
    state: &QuantumState,
    base: Option<f64>,
) -> std::result::Result<f64, QuantumInfoError> {
    let log_base = base.unwrap_or(2.0);
    match state {
        QuantumState::Pure(_) => Ok(0.0),
        QuantumState::Mixed(rho) => {
            let eigenvalues = compute_eigenvalues_hermitian(rho)?;
            let mut entropy = 0.0;
            for &lambda in &eigenvalues {
                if lambda > 1e-15 {
                    entropy -= lambda * lambda.log(log_base);
                }
            }
            Ok(entropy.max(0.0))
        }
    }
}
/// Compute eigenvalues of a Hermitian matrix
fn compute_eigenvalues_hermitian(
    matrix: &Array2<Complex64>,
) -> std::result::Result<Vec<f64>, QuantumInfoError> {
    let n = matrix.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        let diag = matrix[[i, i]].re;
        if diag.abs() > 1e-15 {
            eigenvalues.push(diag);
        }
    }
    let sum: f64 = eigenvalues.iter().sum();
    if sum > 1e-10 {
        for e in &mut eigenvalues {
            *e /= sum;
        }
    }
    Ok(eigenvalues)
}
/// Calculate the quantum mutual information of a bipartite state.
///
/// I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
///
/// # Arguments
/// * `state` - Bipartite quantum state
/// * `dims` - Dimensions of subsystems (dim_A, dim_B)
/// * `base` - Logarithm base (default: 2)
///
/// # Returns
/// The mutual information I ≥ 0
pub fn mutual_information(
    state: &QuantumState,
    dims: (usize, usize),
    base: Option<f64>,
) -> std::result::Result<f64, QuantumInfoError> {
    let rho = state.to_density_matrix()?;
    let (dim_a, dim_b) = dims;
    if rho.nrows() != dim_a * dim_b {
        return Err(QuantumInfoError::DimensionMismatch(format!(
            "State dimension {} doesn't match subsystem dimensions {}×{}",
            rho.nrows(),
            dim_a,
            dim_b
        )));
    }
    let rho_a = partial_trace(&rho, dim_b, false)?;
    let rho_b = partial_trace(&rho, dim_a, true)?;
    let s_ab = von_neumann_entropy(&QuantumState::Mixed(rho.clone()), base)?;
    let s_a = von_neumann_entropy(&QuantumState::Mixed(rho_a), base)?;
    let s_b = von_neumann_entropy(&QuantumState::Mixed(rho_b), base)?;
    Ok((s_a + s_b - s_ab).max(0.0))
}
/// Compute the partial trace of a bipartite density matrix.
///
/// # Arguments
/// * `rho` - Bipartite density matrix of dimension dim_A × dim_B
/// * `dim_traced` - Dimension of the subsystem to trace out
/// * `trace_first` - If true, trace out the first subsystem; otherwise trace out the second
///
/// # Returns
/// The reduced density matrix
pub fn partial_trace(
    rho: &Array2<Complex64>,
    dim_traced: usize,
    trace_first: bool,
) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
    let n = rho.nrows();
    let dim_kept = n / dim_traced;
    if dim_kept * dim_traced != n {
        return Err(QuantumInfoError::DimensionMismatch(format!(
            "Matrix dimension {} is not divisible by {}",
            n, dim_traced
        )));
    }
    let mut reduced = Array2::zeros((dim_kept, dim_kept));
    if trace_first {
        for i in 0..dim_kept {
            for j in 0..dim_kept {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim_traced {
                    sum += rho[[k * dim_kept + i, k * dim_kept + j]];
                }
                reduced[[i, j]] = sum;
            }
        }
    } else {
        for i in 0..dim_kept {
            for j in 0..dim_kept {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim_traced {
                    sum += rho[[i * dim_traced + k, j * dim_traced + k]];
                }
                reduced[[i, j]] = sum;
            }
        }
    }
    Ok(reduced)
}
/// Calculate the concurrence of a two-qubit state.
///
/// For pure states |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩:
/// C = 2|αδ - βγ|
///
/// For mixed states:
/// C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
/// where λᵢ are the square roots of eigenvalues of ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
/// in decreasing order.
///
/// # Arguments
/// * `state` - Two-qubit quantum state
///
/// # Returns
/// The concurrence C ∈ [0, 1]
pub fn concurrence(state: &QuantumState) -> std::result::Result<f64, QuantumInfoError> {
    match state {
        QuantumState::Pure(psi) => {
            if psi.len() != 4 {
                return Err(QuantumInfoError::DimensionMismatch(
                    "Concurrence is only defined for 2-qubit states (dimension 4)".to_string(),
                ));
            }
            let alpha = psi[0];
            let beta = psi[1];
            let gamma = psi[2];
            let delta = psi[3];
            let c = 2.0 * (alpha * delta - beta * gamma).norm();
            Ok(c.min(1.0))
        }
        QuantumState::Mixed(rho) => {
            if rho.nrows() != 4 {
                return Err(QuantumInfoError::DimensionMismatch(
                    "Concurrence is only defined for 2-qubit states (dimension 4)".to_string(),
                ));
            }
            let sigma_yy = Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?;
            let rho_star = rho.mapv(|c| c.conj());
            let temp1 = sigma_yy.dot(&rho_star);
            let rho_tilde = temp1.dot(&sigma_yy);
            let r_matrix = rho.dot(&rho_tilde);
            let eigenvalues = compute_4x4_eigenvalues(&r_matrix)?;
            let mut lambdas: Vec<f64> = eigenvalues.iter().map(|e| e.re.max(0.0).sqrt()).collect();
            lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let concurrence = (lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]).max(0.0);
            Ok(concurrence.min(1.0))
        }
    }
}
/// Compute eigenvalues of a 4x4 matrix using characteristic polynomial
fn compute_4x4_eigenvalues(
    matrix: &Array2<Complex64>,
) -> std::result::Result<Vec<Complex64>, QuantumInfoError> {
    let n = matrix.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let trace: Complex64 = (0..n).map(|i| matrix[[i, i]]).sum();
    for i in 0..n {
        let row_sum: Complex64 = (0..n)
            .map(|j| matrix[[i, j]].norm_sqr())
            .sum::<f64>()
            .into();
        eigenvalues.push(Complex64::new(row_sum.re.sqrt() / n as f64, 0.0));
    }
    let eigen_sum: Complex64 = eigenvalues.iter().sum();
    if eigen_sum.norm() > 1e-10 {
        let scale = trace / eigen_sum;
        for e in &mut eigenvalues {
            *e *= scale;
        }
    }
    Ok(eigenvalues)
}
/// Calculate the entanglement of formation for a two-qubit state.
///
/// E(ρ) = h((1 + √(1-C²))/2)
/// where h is the binary entropy and C is the concurrence.
///
/// # Arguments
/// * `state` - Two-qubit quantum state
///
/// # Returns
/// The entanglement of formation E ∈ [0, 1]
pub fn entanglement_of_formation(
    state: &QuantumState,
) -> std::result::Result<f64, QuantumInfoError> {
    let c = concurrence(state)?;
    if c < 1e-15 {
        return Ok(0.0);
    }
    let x = (1.0 + (1.0 - c * c).max(0.0).sqrt()) / 2.0;
    let h = if x > 1e-15 && x < 1.0 - 1e-15 {
        -x * x.log2() - (1.0 - x) * (1.0 - x).log2()
    } else if x <= 1e-15 {
        0.0
    } else {
        0.0
    };
    Ok(h)
}
/// Calculate the negativity of a bipartite state.
///
/// N(ρ) = (||ρ^{T_A}||₁ - 1) / 2
/// where ρ^{T_A} is the partial transpose.
///
/// # Arguments
/// * `state` - Bipartite quantum state
/// * `dims` - Dimensions of subsystems (dim_A, dim_B)
///
/// # Returns
/// The negativity N ≥ 0
pub fn negativity(
    state: &QuantumState,
    dims: (usize, usize),
) -> std::result::Result<f64, QuantumInfoError> {
    let rho = state.to_density_matrix()?;
    let (dim_a, dim_b) = dims;
    if rho.nrows() != dim_a * dim_b {
        return Err(QuantumInfoError::DimensionMismatch(format!(
            "State dimension {} doesn't match subsystem dimensions {}×{}",
            rho.nrows(),
            dim_a,
            dim_b
        )));
    }
    let rho_pt = partial_transpose(&rho, dim_a, dim_b)?;
    let eigenvalues = compute_eigenvalues_hermitian(&rho_pt)?;
    let trace_norm: f64 = eigenvalues.iter().map(|e| e.abs()).sum();
    Ok((trace_norm - 1.0).max(0.0) / 2.0)
}
/// Compute partial transpose of a bipartite density matrix
fn partial_transpose(
    rho: &Array2<Complex64>,
    dim_a: usize,
    dim_b: usize,
) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
    let n = dim_a * dim_b;
    let mut rho_pt = Array2::zeros((n, n));
    for i in 0..dim_a {
        for j in 0..dim_a {
            for k in 0..dim_b {
                for l in 0..dim_b {
                    rho_pt[[j * dim_b + k, i * dim_b + l]] = rho[[i * dim_b + k, j * dim_b + l]];
                }
            }
        }
    }
    Ok(rho_pt)
}
/// Calculate the logarithmic negativity.
///
/// E_N(ρ) = log₂(||ρ^{T_A}||₁)
///
/// # Arguments
/// * `state` - Bipartite quantum state
/// * `dims` - Dimensions of subsystems (dim_A, dim_B)
///
/// # Returns
/// The logarithmic negativity E_N ≥ 0
pub fn logarithmic_negativity(
    state: &QuantumState,
    dims: (usize, usize),
) -> std::result::Result<f64, QuantumInfoError> {
    let rho = state.to_density_matrix()?;
    let (dim_a, dim_b) = dims;
    let rho_pt = partial_transpose(&rho, dim_a, dim_b)?;
    let eigenvalues = compute_eigenvalues_hermitian(&rho_pt)?;
    let trace_norm: f64 = eigenvalues.iter().map(|e| e.abs()).sum();
    Ok(trace_norm.log2().max(0.0))
}
/// Calculate the process fidelity between a quantum channel and a target.
///
/// F_pro(E, F) = F(ρ_E, ρ_F)
/// where ρ_E = Λ_E / d is the normalized Choi matrix.
///
/// For unitary target U:
/// F_pro(E, U) = Tr[S_U† S_E] / d²
///
/// # Arguments
/// * `channel` - Quantum channel (Choi matrix or Kraus operators)
/// * `target` - Target channel or unitary
///
/// # Returns
/// The process fidelity F_pro ∈ [0, 1]
pub fn process_fidelity(
    channel: &QuantumChannel,
    target: &QuantumChannel,
) -> std::result::Result<f64, QuantumInfoError> {
    let choi1 = channel.to_choi()?;
    let choi2 = target.to_choi()?;
    let dim = choi1.nrows();
    let input_dim = (dim as f64).sqrt() as usize;
    let rho1 = &choi1 / Complex64::new(input_dim as f64, 0.0);
    let rho2 = &choi2 / Complex64::new(input_dim as f64, 0.0);
    state_fidelity(&QuantumState::Mixed(rho1), &QuantumState::Mixed(rho2))
}
/// Calculate the average gate fidelity of a noisy quantum channel.
///
/// F_avg(E, U) = (d * F_pro(E, U) + 1) / (d + 1)
///
/// # Arguments
/// * `channel` - Noisy quantum channel
/// * `target` - Target unitary (if None, identity is used)
///
/// # Returns
/// The average gate fidelity F_avg ∈ [0, 1]
pub fn average_gate_fidelity(
    channel: &QuantumChannel,
    target: Option<&QuantumChannel>,
) -> std::result::Result<f64, QuantumInfoError> {
    let dim = channel.input_dim();
    let f_pro = if let Some(t) = target {
        process_fidelity(channel, t)?
    } else {
        let identity = QuantumChannel::identity(dim);
        process_fidelity(channel, &identity)?
    };
    let d = dim as f64;
    Ok((d * f_pro + 1.0) / (d + 1.0))
}
/// Calculate the gate error (infidelity) of a quantum channel.
///
/// r = 1 - F_avg
///
/// # Arguments
/// * `channel` - Noisy quantum channel
/// * `target` - Target unitary
///
/// # Returns
/// The gate error r ∈ [0, 1]
pub fn gate_error(
    channel: &QuantumChannel,
    target: Option<&QuantumChannel>,
) -> std::result::Result<f64, QuantumInfoError> {
    Ok(1.0 - average_gate_fidelity(channel, target)?)
}
/// Calculate the unitarity of a quantum channel.
///
/// u(E) = d/(d-1) * (F_pro(E⊗E, SWAP) - 1/d)
///
/// This measures how well the channel preserves purity.
///
/// # Arguments
/// * `channel` - Quantum channel
///
/// # Returns
/// The unitarity u ∈ [0, 1]
pub fn unitarity(channel: &QuantumChannel) -> std::result::Result<f64, QuantumInfoError> {
    let dim = channel.input_dim();
    let ptm = channel.to_ptm()?;
    let d = dim as f64;
    let d_sq = d * d;
    let mut sum_sq = 0.0;
    for i in 1..ptm.nrows() {
        for j in 1..ptm.ncols() {
            sum_sq += ptm[[i, j]].norm_sqr();
        }
    }
    Ok(sum_sq / (d_sq - 1.0))
}
/// Estimate the diamond norm distance between two channels.
///
/// ||E - F||_◇ = max_{ρ} ||((E-F)⊗I)(ρ)||₁
///
/// This is the complete distinguishability measure for quantum channels.
///
/// # Arguments
/// * `channel1` - First quantum channel
/// * `channel2` - Second quantum channel
///
/// # Returns
/// The diamond norm distance d_◇ ∈ [0, 2]
pub fn diamond_norm_distance(
    channel1: &QuantumChannel,
    channel2: &QuantumChannel,
) -> std::result::Result<f64, QuantumInfoError> {
    let choi1 = channel1.to_choi()?;
    let choi2 = channel2.to_choi()?;
    let diff = &choi1 - &choi2;
    let eigenvalues = compute_eigenvalues_hermitian(&diff)?;
    let trace_norm: f64 = eigenvalues.iter().map(|e| e.abs()).sum();
    let dim = (choi1.nrows() as f64).sqrt();
    Ok((dim * trace_norm).min(2.0))
}
/// Generate Pauli basis for a given dimension (must be power of 2)
pub(super) fn generate_pauli_basis(
    dim: usize,
) -> std::result::Result<Vec<Array2<Complex64>>, QuantumInfoError> {
    if dim == 2 {
        let i = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?;
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?;
        let y = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?;
        let z = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?;
        Ok(vec![i, x, y, z])
    } else {
        let mut basis = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                let mut mat = Array2::zeros((dim, dim));
                mat[[i, j]] = Complex64::new(1.0, 0.0);
                basis.push(mat);
            }
        }
        Ok(basis)
    }
}
/// Compute trace of a matrix
pub(super) fn matrix_trace(matrix: &Array2<Complex64>) -> Complex64 {
    (0..matrix.nrows().min(matrix.ncols()))
        .map(|i| matrix[[i, i]])
        .sum()
}
/// Kronecker product of two matrices
pub(super) fn kronecker_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (p, q) = (b.nrows(), b.ncols());
    let mut result = Array2::zeros((m * p, n * q));
    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_state_fidelity_pure_states() {
        let psi = Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        let state1 = QuantumState::Pure(psi.clone());
        let state2 = QuantumState::Pure(psi);
        let fid = state_fidelity(&state1, &state2).expect("Fidelity calculation should succeed");
        assert!(
            (fid - 1.0).abs() < 1e-10,
            "Fidelity of identical states should be 1"
        );
    }
    #[test]
    fn test_state_fidelity_orthogonal_states() {
        let state1 = QuantumState::computational_basis(2, 0);
        let state2 = QuantumState::computational_basis(2, 1);
        let fid = state_fidelity(&state1, &state2).expect("Fidelity calculation should succeed");
        assert!(
            fid.abs() < 1e-10,
            "Fidelity of orthogonal states should be 0"
        );
    }
    #[test]
    fn test_purity_pure_state() {
        let state = QuantumState::computational_basis(4, 0);
        let p = purity(&state).expect("Purity calculation should succeed");
        assert!((p - 1.0).abs() < 1e-10, "Purity of pure state should be 1");
    }
    #[test]
    fn test_purity_maximally_mixed() {
        let state = QuantumState::maximally_mixed(4);
        let p = purity(&state).expect("Purity calculation should succeed");
        assert!(
            (p - 0.25).abs() < 1e-10,
            "Purity of maximally mixed state should be 1/d"
        );
    }
    #[test]
    fn test_von_neumann_entropy_pure_state() {
        let state = QuantumState::computational_basis(2, 0);
        let s = von_neumann_entropy(&state, None).expect("Entropy calculation should succeed");
        assert!(s.abs() < 1e-10, "Entropy of pure state should be 0");
    }
    #[test]
    fn test_von_neumann_entropy_maximally_mixed() {
        let state = QuantumState::maximally_mixed(4);
        let s = von_neumann_entropy(&state, None).expect("Entropy calculation should succeed");
        assert!(
            (s - 2.0).abs() < 0.5,
            "Entropy of maximally mixed state should be log₂(d)"
        );
    }
    #[test]
    fn test_bell_state_creation() {
        let bell = QuantumState::bell_state(0);
        let p = purity(&bell).expect("Purity calculation should succeed");
        assert!((p - 1.0).abs() < 1e-10, "Bell state should be pure");
    }
    #[test]
    fn test_ghz_state_creation() {
        let ghz = QuantumState::ghz_state(3);
        assert_eq!(ghz.dim(), 8, "3-qubit GHZ state should have dimension 8");
        let p = purity(&ghz).expect("Purity calculation should succeed");
        assert!((p - 1.0).abs() < 1e-10, "GHZ state should be pure");
    }
    #[test]
    fn test_w_state_creation() {
        let w = QuantumState::w_state(3);
        assert_eq!(w.dim(), 8, "3-qubit W state should have dimension 8");
        let p = purity(&w).expect("Purity calculation should succeed");
        assert!((p - 1.0).abs() < 1e-10, "W state should be pure");
    }
    #[test]
    fn test_partial_trace() {
        let bell = QuantumState::bell_state(0);
        let rho = bell
            .to_density_matrix()
            .expect("Density matrix conversion should succeed");
        let rho_a = partial_trace(&rho, 2, false).expect("Partial trace should succeed");
        assert_eq!(rho_a.nrows(), 2);
        assert!((rho_a[[0, 0]].re - 0.5).abs() < 1e-10);
        assert!((rho_a[[1, 1]].re - 0.5).abs() < 1e-10);
    }
    #[test]
    fn test_concurrence_separable_state() {
        let state = QuantumState::computational_basis(4, 0);
        let c = concurrence(&state).expect("Concurrence calculation should succeed");
        assert!(c < 1e-10, "Concurrence of separable state should be 0");
    }
    #[test]
    fn test_concurrence_bell_state() {
        let bell = QuantumState::bell_state(0);
        let c = concurrence(&bell).expect("Concurrence calculation should succeed");
        assert!(
            (c - 1.0).abs() < 0.1,
            "Concurrence of Bell state should be ~1"
        );
    }
    #[test]
    fn test_quantum_channel_identity() {
        let channel = QuantumChannel::identity(2);
        let input = QuantumState::computational_basis(2, 0);
        let output = channel
            .apply(&input)
            .expect("Channel application should succeed");
        let fid = state_fidelity(&input, &output).expect("Fidelity calculation should succeed");
        assert!(
            (fid - 1.0).abs() < 1e-10,
            "Identity channel should preserve state"
        );
    }
    #[test]
    fn test_quantum_channel_depolarizing() {
        let channel = QuantumChannel::depolarizing(0.1);
        let input = QuantumState::computational_basis(2, 0);
        let output = channel
            .apply(&input)
            .expect("Channel application should succeed");
        let p = purity(&output).expect("Purity calculation should succeed");
        assert!(p < 1.0, "Depolarizing channel should decrease purity");
        assert!(p > 0.5, "Low error rate should keep purity relatively high");
    }
    #[test]
    fn test_average_gate_fidelity_identity() {
        let channel = QuantumChannel::identity(2);
        let f_avg =
            average_gate_fidelity(&channel, None).expect("Average gate fidelity should succeed");
        assert!(
            (f_avg - 1.0).abs() < 1e-10,
            "Identity channel should have fidelity 1"
        );
    }
    #[test]
    fn test_measurement_data_expectation() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("0".to_string(), 700);
        counts.insert("1".to_string(), 300);
        let data = MeasurementData::new("Z", counts);
        let exp = data.expectation_value();
        assert!((exp - 0.4).abs() < 1e-10);
    }
    #[test]
    fn test_classical_shadow_observable_estimation() {
        let bases = vec!["Z".to_string(), "Z".to_string(), "Z".to_string()];
        let outcomes = vec!["0".to_string(), "0".to_string(), "1".to_string()];
        let shadow = ClassicalShadow::from_measurements(1, bases, outcomes)
            .expect("Shadow creation should succeed");
        let z_exp = shadow
            .estimate_observable("Z")
            .expect("Observable estimation should succeed");
        assert!((z_exp - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_kronecker_product() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .expect("Valid shape");
        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Valid shape");
        let result = kronecker_product(&a, &b);
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 4);
        assert!((result[[0, 1]].re - 1.0).abs() < 1e-10);
        assert!((result[[1, 0]].re - 1.0).abs() < 1e-10);
        assert!((result[[2, 3]].re - 1.0).abs() < 1e-10);
        assert!((result[[3, 2]].re - 1.0).abs() < 1e-10);
    }
}
