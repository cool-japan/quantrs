//! Ensemble methods reconstruction

use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;

use super::super::core::SciRS2ProcessTomographer;
use super::super::results::{ExperimentalData, ReconstructionQuality};
use super::utils::calculate_reconstruction_quality;
use super::{bayesian, compressed_sensing, linear_inversion, maximum_likelihood};
use crate::DeviceResult;

/// Ensemble reconstruction combining multiple methods
pub fn reconstruct_ensemble(
    tomographer: &SciRS2ProcessTomographer,
    experimental_data: &ExperimentalData,
) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
    // Run multiple reconstruction methods
    let (li_process, li_quality) =
        linear_inversion::reconstruct_linear_inversion(tomographer, experimental_data)?;
    let (mle_process, mle_quality) =
        maximum_likelihood::reconstruct_maximum_likelihood(tomographer, experimental_data)?;
    let (cs_process, cs_quality) =
        compressed_sensing::reconstruct_compressed_sensing(tomographer, experimental_data)?;
    let (bayes_process, bayes_quality) =
        bayesian::reconstruct_bayesian(tomographer, experimental_data)?;

    // Calculate weights based on reconstruction quality
    let li_weight = calculate_weight(&li_quality);
    let mle_weight = calculate_weight(&mle_quality);
    let cs_weight = calculate_weight(&cs_quality);
    let bayes_weight = calculate_weight(&bayes_quality);

    let total_weight = li_weight + mle_weight + cs_weight + bayes_weight;

    // Weighted combination
    let dim = li_process.dim().0;
    let mut ensemble_process = Array4::zeros((dim, dim, dim, dim));

    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                for l in 0..dim {
                    ensemble_process[[i, j, k, l]] = (li_process[[i, j, k, l]] * li_weight
                        + mle_process[[i, j, k, l]] * mle_weight
                        + cs_process[[i, j, k, l]] * cs_weight
                        + bayes_process[[i, j, k, l]] * bayes_weight)
                        / total_weight;
                }
            }
        }
    }

    let log_likelihood = calculate_ensemble_log_likelihood(&ensemble_process, experimental_data)?;
    let reconstruction_quality =
        calculate_reconstruction_quality(&ensemble_process, experimental_data, log_likelihood);

    Ok((ensemble_process, reconstruction_quality))
}

/// Calculate weight for ensemble based on reconstruction quality
fn calculate_weight(quality: &ReconstructionQuality) -> f64 {
    // Higher log-likelihood and better physical validity get higher weights
    let ll_weight = (quality.log_likelihood + 100.0).max(0.1); // Offset to ensure positive
    let physical_weight = quality.physical_validity.positivity_measure
        * quality.physical_validity.trace_preservation_measure;
    let stability_weight = 1.0 / (1.0 + quality.condition_number / 100.0);

    ll_weight * physical_weight * stability_weight
}

/// Calculate ensemble log-likelihood
fn calculate_ensemble_log_likelihood(
    process_matrix: &Array4<Complex64>,
    experimental_data: &ExperimentalData,
) -> DeviceResult<f64> {
    let mut log_likelihood = 0.0;

    for (observed, &uncertainty) in experimental_data
        .measurement_results
        .iter()
        .zip(experimental_data.measurement_uncertainties.iter())
    {
        let predicted = 0.5; // Placeholder
        let diff = observed - predicted;
        let variance = uncertainty * uncertainty;
        log_likelihood -= 0.5 * (diff * diff / variance);
    }

    Ok(log_likelihood)
}
