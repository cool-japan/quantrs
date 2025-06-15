//! Comprehensive quantum process tomography module
//!
//! This module provides advanced quantum process tomography capabilities using SciRS2
//! for statistical analysis, optimization, and machine learning.

pub mod analysis;
pub mod config;
pub mod core;
pub mod fallback;
pub mod reconstruction;
pub mod results;
pub mod utils;
pub mod validation;

// Re-export main types and functions
pub use analysis::*;
pub use config::*;
pub use core::{ProcessTomographyExecutor, SciRS2ProcessTomographer};
pub use reconstruction::*;
pub use results::*;
pub use utils::*;
pub use validation::*;

// Conditional exports based on feature availability
#[cfg(feature = "scirs2")]
pub use fallback as scirs2_fallback;

#[cfg(not(feature = "scirs2"))]
pub use fallback::*;

use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;
use quantrs2_circuit::prelude::*;
use std::collections::HashMap;

use crate::{calibration::CalibrationManager, DeviceResult};

/// Create a new process tomographer with default configuration
pub fn create_process_tomographer(
    calibration_manager: CalibrationManager,
) -> SciRS2ProcessTomographer {
    let config = SciRS2ProcessTomographyConfig::default();
    SciRS2ProcessTomographer::new(config, calibration_manager)
}

/// Create a process tomographer with custom configuration
pub fn create_process_tomographer_with_config(
    config: SciRS2ProcessTomographyConfig,
    calibration_manager: CalibrationManager,
) -> SciRS2ProcessTomographer {
    SciRS2ProcessTomographer::new(config, calibration_manager)
}

/// Perform quick process tomography with minimal configuration
pub async fn quick_process_tomography<const N: usize, E: ProcessTomographyExecutor>(
    process_circuit: &Circuit<N>,
    executor: &E,
    num_qubits: usize,
) -> DeviceResult<ProcessMetrics> {
    let calibration_manager = CalibrationManager::new();
    let mut tomographer = create_process_tomographer(calibration_manager);

    // Generate input states and measurements
    tomographer.generate_input_states(num_qubits)?;
    tomographer.generate_measurement_operators(num_qubits)?;

    // Perform tomography
    let result = tomographer
        .perform_process_tomography("quick_tomography", process_circuit, executor)
        .await?;

    Ok(result.process_metrics)
}

/// Comprehensive process characterization with full analysis
pub async fn comprehensive_process_characterization<
    const N: usize,
    E: ProcessTomographyExecutor,
>(
    device_id: &str,
    process_circuit: &Circuit<N>,
    executor: &E,
    num_qubits: usize,
    config: Option<SciRS2ProcessTomographyConfig>,
) -> DeviceResult<SciRS2ProcessTomographyResult> {
    let calibration_manager = CalibrationManager::new();
    let config = config.unwrap_or_else(SciRS2ProcessTomographyConfig::default);
    let mut tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

    // Generate input states and measurements
    tomographer.generate_input_states(num_qubits)?;
    tomographer.generate_measurement_operators(num_qubits)?;

    // Perform comprehensive tomography
    tomographer
        .perform_process_tomography(device_id, process_circuit, executor)
        .await
}

/// Create process monitoring system for continuous characterization
pub fn create_process_monitoring_system(
    reference_metrics: ProcessMetrics,
    anomaly_threshold: f64,
    drift_sensitivity: f64,
) -> (ProcessAnomalyDetector, ProcessDriftDetector) {
    let anomaly_detector = ProcessAnomalyDetector::new(
        anomaly_threshold,
        AnomalyDetectionAlgorithm::StatisticalThreshold,
    );

    let drift_detector = ProcessDriftDetector::new(
        reference_metrics,
        drift_sensitivity,
        DriftDetectionMethod::StatisticalTest,
    );

    (anomaly_detector, drift_detector)
}

/// Benchmark process against standard quantum channels
pub async fn benchmark_process<const N: usize, E: ProcessTomographyExecutor>(
    process_circuit: &Circuit<N>,
    executor: &E,
    num_qubits: usize,
    benchmark_channels: &[String],
) -> DeviceResult<HashMap<String, f64>> {
    let calibration_manager = CalibrationManager::new();
    let mut config = SciRS2ProcessTomographyConfig::default();
    config.validation_config.enable_benchmarking = true;
    config.validation_config.benchmark_processes = benchmark_channels.to_vec();

    let mut tomographer = SciRS2ProcessTomographer::new(config, calibration_manager);

    // Generate input states and measurements
    tomographer.generate_input_states(num_qubits)?;
    tomographer.generate_measurement_operators(num_qubits)?;

    // Perform tomography
    let result = tomographer
        .perform_process_tomography("benchmark", process_circuit, executor)
        .await?;

    Ok(result.process_comparisons.standard_process_fidelities)
}

/// Compare two quantum processes
pub fn compare_processes(
    process1: &Array4<Complex64>,
    process2: &Array4<Complex64>,
) -> DeviceResult<ProcessComparisonResult> {
    // Calculate various distance measures
    let trace_distance = utils::process_utils::trace_distance(process1, process2)?;

    // Calculate fidelity (simplified)
    let mut fidelity = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    let dim = process1.dim();
    for i in 0..dim.0 {
        for j in 0..dim.1 {
            for k in 0..dim.2 {
                for l in 0..dim.3 {
                    let element1 = process1[[i, j, k, l]];
                    let element2 = process2[[i, j, k, l]];

                    fidelity += (element1.conj() * element2).re;
                    norm1 += element1.norm_sqr();
                    norm2 += element2.norm_sqr();
                }
            }
        }
    }

    let process_fidelity = if norm1 > 1e-12 && norm2 > 1e-12 {
        fidelity / (norm1 * norm2).sqrt()
    } else {
        0.0
    };

    Ok(ProcessComparisonResult {
        process_fidelity,
        trace_distance,
        diamond_norm_distance: 2.0 * (1.0 - process_fidelity).sqrt(),
    })
}

/// Result of process comparison
#[derive(Debug, Clone)]
pub struct ProcessComparisonResult {
    pub process_fidelity: f64,
    pub trace_distance: f64,
    pub diamond_norm_distance: f64,
}

/// Validate process tomography result
pub fn validate_process_result(
    result: &SciRS2ProcessTomographyResult,
    tolerance: f64,
) -> ProcessValidationReport {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();

    // Check physical validity
    let physical_validity = &result
        .statistical_analysis
        .reconstruction_quality
        .physical_validity;

    if !physical_validity.is_completely_positive {
        issues.push("Process is not completely positive".to_string());
    }

    if !physical_validity.is_trace_preserving {
        issues.push("Process is not trace preserving".to_string());
    }

    if physical_validity.positivity_measure < 0.9 {
        warnings.push(format!(
            "Low positivity measure: {:.3}",
            physical_validity.positivity_measure
        ));
    }

    if physical_validity.trace_preservation_measure < 0.95 {
        warnings.push(format!(
            "Poor trace preservation: {:.3}",
            physical_validity.trace_preservation_measure
        ));
    }

    // Check process metrics
    if result.process_metrics.process_fidelity < 0.5 {
        warnings.push("Low process fidelity detected".to_string());
    }

    if result.process_metrics.unitarity < 0.1 {
        warnings.push("Very low unitarity detected".to_string());
    }

    // Check reconstruction quality
    if result
        .statistical_analysis
        .reconstruction_quality
        .condition_number
        > 1e10
    {
        warnings.push("High condition number indicates numerical instability".to_string());
    }

    let is_valid = issues.is_empty();
    let quality_score = calculate_overall_quality_score(result);

    ProcessValidationReport {
        is_valid,
        quality_score,
        issues,
        warnings,
    }
}

/// Process validation report
#[derive(Debug, Clone)]
pub struct ProcessValidationReport {
    pub is_valid: bool,
    pub quality_score: f64,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

/// Calculate overall quality score for a process tomography result
fn calculate_overall_quality_score(result: &SciRS2ProcessTomographyResult) -> f64 {
    let physical_score = (result
        .statistical_analysis
        .reconstruction_quality
        .physical_validity
        .positivity_measure
        + result
            .statistical_analysis
            .reconstruction_quality
            .physical_validity
            .trace_preservation_measure)
        / 2.0;

    let fidelity_score = result.process_metrics.process_fidelity;
    let unitarity_score = result.process_metrics.unitarity;

    let numerical_score = 1.0
        / (1.0
            + result
                .statistical_analysis
                .reconstruction_quality
                .condition_number
                / 1e6);

    // Weighted average
    (physical_score * 0.3 + fidelity_score * 0.3 + unitarity_score * 0.2 + numerical_score * 0.2)
        .max(0.0)
        .min(1.0)
}

/// Export process tomography results in various formats
pub fn export_process_results(
    result: &SciRS2ProcessTomographyResult,
    format: ExportFormat,
) -> DeviceResult<String> {
    match format {
        ExportFormat::Json => export_as_json(result),
        ExportFormat::Csv => export_as_csv(result),
        ExportFormat::Hdf5 => export_as_hdf5(result),
        ExportFormat::Matlab => export_as_matlab(result),
    }
}

/// Export formats
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Hdf5,
    Matlab,
}

fn export_as_json(result: &SciRS2ProcessTomographyResult) -> DeviceResult<String> {
    // Simplified JSON export
    let json_data = format!(
        r#"{{
    "device_id": "{}",
    "process_fidelity": {:.6},
    "average_gate_fidelity": {:.6},
    "unitarity": {:.6},
    "entangling_power": {:.6},
    "reconstruction_method": "{:?}",
    "log_likelihood": {:.6}
}}"#,
        result.device_id,
        result.process_metrics.process_fidelity,
        result.process_metrics.average_gate_fidelity,
        result.process_metrics.unitarity,
        result.process_metrics.entangling_power,
        result.config.reconstruction_method,
        result
            .statistical_analysis
            .reconstruction_quality
            .log_likelihood
    );

    Ok(json_data)
}

fn export_as_csv(result: &SciRS2ProcessTomographyResult) -> DeviceResult<String> {
    let csv_data = format!(
        "device_id,process_fidelity,average_gate_fidelity,unitarity,entangling_power,log_likelihood\n{},{:.6},{:.6},{:.6},{:.6},{:.6}",
        result.device_id,
        result.process_metrics.process_fidelity,
        result.process_metrics.average_gate_fidelity,
        result.process_metrics.unitarity,
        result.process_metrics.entangling_power,
        result.statistical_analysis.reconstruction_quality.log_likelihood
    );

    Ok(csv_data)
}

fn export_as_hdf5(_result: &SciRS2ProcessTomographyResult) -> DeviceResult<String> {
    // Placeholder for HDF5 export
    Ok("HDF5 export not yet implemented".to_string())
}

fn export_as_matlab(_result: &SciRS2ProcessTomographyResult) -> DeviceResult<String> {
    // Placeholder for MATLAB export
    Ok("MATLAB export not yet implemented".to_string())
}
