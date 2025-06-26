//! Linear inversion reconstruction method

use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;

use super::super::core::SciRS2ProcessTomographer;
use super::super::results::{ExperimentalData, ReconstructionQuality};
use super::utils::calculate_reconstruction_quality;
use crate::DeviceResult;

// Conditional imports
#[cfg(feature = "scirs2")]
use scirs2_linalg::{inv, matrix_norm, svd};

#[cfg(not(feature = "scirs2"))]
use super::super::fallback::{inv, matrix_norm, svd};

/// Linear inversion reconstruction implementation
pub fn reconstruct_linear_inversion(
    tomographer: &SciRS2ProcessTomographer,
    experimental_data: &ExperimentalData,
) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
    // Build measurement matrix A such that A * chi = b
    let measurement_matrix = tomographer.build_measurement_matrix(experimental_data)?;
    let measurement_vector = Array1::from_vec(experimental_data.measurement_results.clone());

    // Solve the linear system A * chi = b using pseudoinverse
    let process_vector = solve_linear_system(&measurement_matrix, &measurement_vector)?;

    // Reshape the solution vector back to process matrix form
    let process_matrix = reshape_to_process_matrix(&process_vector)?;

    // Calculate log-likelihood (simplified for linear inversion)
    let log_likelihood = calculate_log_likelihood(&process_matrix, experimental_data, tomographer)?;

    // Calculate reconstruction quality
    let reconstruction_quality =
        calculate_reconstruction_quality(&process_matrix, experimental_data, log_likelihood);

    Ok((process_matrix, reconstruction_quality))
}

/// Solve linear system using pseudoinverse
fn solve_linear_system(
    measurement_matrix: &Array2<f64>,
    measurement_vector: &Array1<f64>,
) -> DeviceResult<Array1<Complex64>> {
    #[cfg(feature = "scirs2")]
    {
        // Use SVD for robust pseudoinverse calculation
        if let Ok((u, s, vt)) = svd(&measurement_matrix.view(), true, None) {
            // Calculate pseudoinverse using SVD: A+ = V * S+ * U^T
            let mut s_pinv = Array1::zeros(s.len());
            let tolerance = 1e-12;

            for (i, &singular_value) in s.iter().enumerate() {
                if singular_value > tolerance {
                    s_pinv[i] = 1.0 / singular_value;
                }
            }

            // Compute A+ * b
            let mut solution = Array1::zeros(measurement_matrix.ncols());
            for i in 0..measurement_matrix.ncols() {
                let mut sum = 0.0;
                for j in 0..measurement_matrix.nrows() {
                    let mut inner_sum = 0.0;
                    for k in 0..s.len() {
                        inner_sum += vt[[k, i]] * s_pinv[k] * u[[j, k]];
                    }
                    sum += inner_sum * measurement_vector[j];
                }
                solution[i] = sum;
            }

            // Convert to complex
            let complex_solution = solution.mapv(|x| Complex64::new(x, 0.0));
            Ok(complex_solution)
        } else {
            // Fallback to simple least squares
            simple_least_squares(measurement_matrix, measurement_vector)
        }
    }

    #[cfg(not(feature = "scirs2"))]
    {
        simple_least_squares(measurement_matrix, measurement_vector)
    }
}

/// Simple least squares solution (fallback)
fn simple_least_squares(
    measurement_matrix: &Array2<f64>,
    measurement_vector: &Array1<f64>,
) -> DeviceResult<Array1<Complex64>> {
    let n = measurement_matrix.ncols();
    let m = measurement_matrix.nrows();

    // Normal equations: (A^T * A) * x = A^T * b
    let mut ata = Array2::zeros((n, n));
    let mut atb = Array1::zeros(n);

    // Compute A^T * A
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += measurement_matrix[[k, i]] * measurement_matrix[[k, j]];
            }
            ata[[i, j]] = sum;
        }
    }

    // Compute A^T * b
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += measurement_matrix[[k, i]] * measurement_vector[k];
        }
        atb[i] = sum;
    }

    // Solve (A^T * A) * x = A^T * b using simple inversion
    #[cfg(feature = "scirs2")]
    {
        if let Ok(ata_inv) = inv(&ata.view(), None) {
            let solution = ata_inv.dot(&atb);
            Ok(solution.mapv(|x| Complex64::new(x, 0.0)))
        } else {
            // Diagonal regularization
            for i in 0..n {
                ata[[i, i]] += 1e-8;
            }
            if let Ok(ata_inv) = inv(&ata.view(), None) {
                let solution = ata_inv.dot(&atb);
                Ok(solution.mapv(|x| Complex64::new(x, 0.0)))
            } else {
                Ok(Array1::<Complex64>::zeros(n))
            }
        }
    }

    #[cfg(not(feature = "scirs2"))]
    {
        // Fallback: return zero solution
        Ok(Array1::zeros(n).mapv(|_| Complex64::new(0.0, 0.0)))
    }
}

/// Reshape solution vector to process matrix
fn reshape_to_process_matrix(
    process_vector: &Array1<Complex64>,
) -> DeviceResult<Array4<Complex64>> {
    let total_elements = process_vector.len();
    let dim = ((total_elements as f64).powf(0.25).round() as usize).max(2);

    let mut process_matrix = Array4::zeros((dim, dim, dim, dim));

    let mut idx = 0;
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                for l in 0..dim {
                    if idx < process_vector.len() {
                        process_matrix[[i, j, k, l]] = process_vector[idx];
                        idx += 1;
                    }
                }
            }
        }
    }

    Ok(process_matrix)
}

/// Calculate log-likelihood for linear inversion
fn calculate_log_likelihood(
    process_matrix: &Array4<Complex64>,
    experimental_data: &ExperimentalData,
    tomographer: &SciRS2ProcessTomographer,
) -> DeviceResult<f64> {
    let mut log_likelihood = 0.0;

    // Predict measurement outcomes using reconstructed process
    for (m_idx, (&observed, &uncertainty)) in experimental_data
        .measurement_results
        .iter()
        .zip(experimental_data.measurement_uncertainties.iter())
        .enumerate()
    {
        let input_idx = m_idx / experimental_data.measurement_operators.len();
        let meas_idx = m_idx % experimental_data.measurement_operators.len();

        if input_idx < experimental_data.input_states.len()
            && meas_idx < experimental_data.measurement_operators.len()
        {
            let predicted = predict_measurement_outcome(
                process_matrix,
                &experimental_data.input_states[input_idx],
                &experimental_data.measurement_operators[meas_idx],
            )?;

            // Gaussian likelihood
            let diff = observed - predicted;
            let variance = uncertainty * uncertainty;
            log_likelihood -=
                0.5 * (diff * diff / variance + (2.0 * std::f64::consts::PI * variance).ln());
        }
    }

    Ok(log_likelihood)
}

/// Predict measurement outcome from process matrix
fn predict_measurement_outcome(
    process_matrix: &Array4<Complex64>,
    input_state: &Array2<Complex64>,
    measurement: &Array2<Complex64>,
) -> DeviceResult<f64> {
    let dim = process_matrix.dim().0;
    let mut result = Complex64::new(0.0, 0.0);

    // Compute Tr(M * Λ(ρ)) where Λ is the quantum process
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                for l in 0..dim {
                    result += measurement[[i, j]]
                        * process_matrix[[i, j, k, l]]
                        * input_state[[k, l]].conj();
                }
            }
        }
    }

    Ok(result.re)
}
