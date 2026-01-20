//! Advanced noise modeling using SciRS2's statistical and machine learning capabilities
//!
//! This module provides sophisticated noise modeling techniques leveraging SciRS2's
//! comprehensive statistical analysis, signal processing, and machine learning tools.
//!
//! The module is organized into focused sub-modules for better maintainability:
//! - `config`: Configuration structures and enums
//! - `types`: Data type definitions and structures
//! - `statistical`: Statistical analysis and distribution modeling
//! - `spectral`: Spectral analysis and frequency domain methods
//! - `temporal`: Temporal correlation and time series analysis
//! - `spatial`: Spatial correlation and geographical analysis
//! - `ml_integration`: Machine learning model integration
//! - `validation`: Model validation and testing frameworks
//! - `utils`: Utility functions and helpers

pub mod config;
pub mod spectral;
pub mod statistical;
pub mod temporal;

// Re-export all types for backward compatibility
pub use config::*;
pub use spectral::*;
pub use statistical::*;
pub use temporal::*;

use crate::{
    calibration::DeviceCalibration,
    noise_model::{CalibrationNoiseModel, CrosstalkNoise},
    DeviceError, DeviceResult,
};
use quantrs2_core::{error::QuantRS2Result, qubit::QubitId};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Statistical noise analysis results
#[derive(Debug, Clone)]
pub struct StatisticalNoiseAnalysis {
    pub noise_statistics: HashMap<String, NoiseStatistics>,
    pub correlation_analysis: CorrelationAnalysis,
    pub temporal_analysis: Option<TemporalAnalysis>,
}

/// Individual noise source statistics
#[derive(Debug, Clone)]
pub struct NoiseStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub distribution_type: DistributionType,
    pub parameters: Vec<f64>,
}

/// Correlation analysis between noise sources
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub correlationmatrix: Array2<f64>,
    pub correlation_strength: f64,
}

/// Temporal analysis of noise (placeholder for future implementation)
#[derive(Debug, Clone)]
pub struct TemporalAnalysis {
    pub autocorrelation: Array1<f64>,
    pub power_spectrum: Array1<f64>,
}

/// Main SciRS2 noise modeling coordinator
///
/// This struct provides the primary interface for advanced noise modeling
/// using SciRS2's comprehensive statistical and machine learning capabilities.
#[derive(Debug, Clone)]
pub struct SciRS2NoiseModeler {
    config: SciRS2NoiseConfig,
    device_id: String,
}

impl SciRS2NoiseModeler {
    /// Create a new noise modeler with default configuration
    pub fn new(device_id: String) -> Self {
        Self {
            config: SciRS2NoiseConfig::default(),
            device_id,
        }
    }

    /// Create a new noise modeler with custom configuration
    pub const fn with_config(device_id: String, config: SciRS2NoiseConfig) -> Self {
        Self { config, device_id }
    }

    /// Perform comprehensive noise modeling
    pub fn model_noise(
        &self,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<CalibrationNoiseModel> {
        use crate::noise_model::{GateNoiseParams, QubitNoiseParams, ReadoutNoiseParams};
        use scirs2_core::ndarray::Array1;
        use std::collections::HashMap;

        // Step 1: Extract noise data from calibration
        let noise_data = self.extract_noise_data(calibration)?;

        // Step 2: Perform statistical analysis
        let statistical_model = if self.config.enable_ml_modeling {
            self.perform_statistical_analysis(&noise_data)?
        } else {
            self.simple_noise_analysis(&noise_data)?
        };

        // Step 3: Build CalibrationNoiseModel from analysis
        self.build_calibration_noise_model(calibration, &statistical_model)
    }

    /// Extract noise data from device calibration
    fn extract_noise_data(
        &self,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<HashMap<String, Array1<f64>>> {
        use scirs2_core::ndarray::Array1;
        let mut noise_data = HashMap::new();

        // Extract single-qubit gate noise
        for (gate_name, gate_cal) in &calibration.single_qubit_gates {
            for (qubit, qubit_data) in &gate_cal.qubit_data {
                let key = format!("single_qubit_{}_{}", gate_name, qubit.0);
                let error_data = vec![qubit_data.error_rate, 1.0 - qubit_data.fidelity];
                noise_data.insert(key, Array1::from_vec(error_data));
            }
        }

        // Extract two-qubit gate noise
        for (qubit_pair, gate_cal) in &calibration.two_qubit_gates {
            let key = format!("two_qubit_{}_{}", qubit_pair.0 .0, qubit_pair.1 .0);
            let error_data = vec![gate_cal.error_rate, 1.0 - gate_cal.fidelity];
            noise_data.insert(key, Array1::from_vec(error_data));
        }

        // Extract readout noise
        for (qubit, readout_cal) in &calibration.readout_calibration.qubit_readout {
            let key = format!("readout_{}", qubit.0);
            let error_data = vec![1.0 - readout_cal.p0_given_0, 1.0 - readout_cal.p1_given_1];
            noise_data.insert(key, Array1::from_vec(error_data));
        }

        Ok(noise_data)
    }

    /// Perform statistical analysis using SciRS2
    fn perform_statistical_analysis(
        &self,
        noise_data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<StatisticalNoiseAnalysis> {
        use scirs2_stats::{mean, std, var};

        let mut analysis_results = HashMap::new();

        for (noise_type, data) in noise_data {
            let data_view = data.view();

            let mean_val = mean(&data_view)
                .map_err(|e| DeviceError::APIError(format!("Statistical analysis error: {e:?}")))?;

            let std_val = std(&data_view, 1, None)
                .map_err(|e| DeviceError::APIError(format!("Statistical analysis error: {e:?}")))?;

            let var_val = var(&data_view, 1, None)
                .map_err(|e| DeviceError::APIError(format!("Statistical analysis error: {e:?}")))?;

            let noise_stats = NoiseStatistics {
                mean: mean_val,
                std_dev: std_val,
                variance: var_val,
                distribution_type: DistributionType::Normal, // Simplified
                parameters: vec![mean_val, std_val],
            };

            analysis_results.insert(noise_type.clone(), noise_stats);
        }

        Ok(StatisticalNoiseAnalysis {
            noise_statistics: analysis_results,
            correlation_analysis: self.perform_correlation_analysis(noise_data)?,
            temporal_analysis: None, // Would implement if temporal data available
        })
    }

    /// Simple noise analysis fallback
    fn simple_noise_analysis(
        &self,
        noise_data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<StatisticalNoiseAnalysis> {
        let mut analysis_results = HashMap::new();

        for (noise_type, data) in noise_data {
            let mean_val = data.mean().unwrap_or(0.01);
            let std_val = data.std(1.0).max(0.001);

            let noise_stats = NoiseStatistics {
                mean: mean_val,
                std_dev: std_val,
                variance: std_val * std_val,
                distribution_type: DistributionType::Normal,
                parameters: vec![mean_val, std_val],
            };

            analysis_results.insert(noise_type.clone(), noise_stats);
        }

        Ok(StatisticalNoiseAnalysis {
            noise_statistics: analysis_results,
            correlation_analysis: CorrelationAnalysis {
                correlationmatrix: scirs2_core::ndarray::Array2::eye(noise_data.len()),
                correlation_strength: 0.1,
            },
            temporal_analysis: None,
        })
    }

    /// Perform correlation analysis between noise sources
    fn perform_correlation_analysis(
        &self,
        noise_data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<CorrelationAnalysis> {
        use scirs2_core::ndarray::Array2;

        let n_sources = noise_data.len();
        let mut correlationmatrix = Array2::eye(n_sources);

        // Calculate pairwise correlations
        let sources: Vec<_> = noise_data.keys().collect();

        for (i, source1) in sources.iter().enumerate() {
            for (j, source2) in sources.iter().enumerate() {
                if i != j {
                    let data1 = &noise_data[*source1];
                    let data2 = &noise_data[*source2];

                    // Simple correlation calculation
                    let corr = if data1.len() == data2.len() && data1.len() > 1 {
                        let mean1 = data1.mean().unwrap_or(0.0);
                        let mean2 = data2.mean().unwrap_or(0.0);

                        let cov = data1
                            .iter()
                            .zip(data2.iter())
                            .map(|(x, y)| (x - mean1) * (y - mean2))
                            .sum::<f64>()
                            / (data1.len() - 1) as f64;

                        let std1 = data1.std(1.0).max(1e-10);
                        let std2 = data2.std(1.0).max(1e-10);

                        cov / (std1 * std2)
                    } else {
                        0.0
                    };

                    correlationmatrix[[i, j]] = corr;
                }
            }
        }

        let avg_correlation = correlationmatrix.iter()
            .filter(|&&x| x != 1.0)  // Exclude diagonal
            .map(|&x| x.abs())
            .sum::<f64>()
            / ((n_sources * n_sources - n_sources) as f64).max(1.0);

        Ok(CorrelationAnalysis {
            correlationmatrix,
            correlation_strength: avg_correlation,
        })
    }

    /// Build CalibrationNoiseModel from statistical analysis
    fn build_calibration_noise_model(
        &self,
        calibration: &DeviceCalibration,
        analysis: &StatisticalNoiseAnalysis,
    ) -> DeviceResult<CalibrationNoiseModel> {
        use crate::noise_model::{GateNoiseParams, QubitNoiseParams, ReadoutNoiseParams};
        use std::collections::HashMap;

        // Build qubit noise parameters
        let mut qubit_noise = HashMap::new();
        for i in 0..calibration.topology.num_qubits {
            let qubit_id = QubitId(i as u32);

            // Get noise statistics for this qubit
            let t1_key = format!("t1_{i}");
            let t2_key = format!("t2_{i}");

            let t1_stats = analysis.noise_statistics.get(&t1_key);
            let t2_stats = analysis.noise_statistics.get(&t2_key);

            let qubit_params = QubitNoiseParams {
                gamma_1: t1_stats.map_or(1.0 / 50000.0, |s| 1.0 / s.mean), // 1/T1
                gamma_phi: t2_stats.map_or(1.0 / 25000.0, |s| 1.0 / s.mean), // 1/T2
                thermal_population: 0.01,
                frequency_drift: 0.001,
                flicker_noise: 0.0005,
            };

            qubit_noise.insert(qubit_id, qubit_params);
        }

        // Build gate noise parameters
        let mut gate_noise = HashMap::new();
        for gate_name in calibration.single_qubit_gates.keys() {
            let noise_stats = analysis.noise_statistics.values().next();

            let gate_params = GateNoiseParams {
                depolarizing_rate: noise_stats.map_or(0.001, |s| s.mean),
                incoherent_error: noise_stats.map_or(0.0005, |s| s.std_dev),
                amplitude_noise: 0.0001,
                coherent_error: 0.0002,
                duration: 100.0, // 100ns default
                phase_noise: 0.0001,
            };

            gate_noise.insert(gate_name.clone(), gate_params);
        }

        // Build readout noise parameters
        let mut readout_noise = HashMap::new();
        for i in 0..calibration.topology.num_qubits {
            let qubit_id = QubitId(i as u32);
            let readout_key = format!("readout_{i}");

            let readout_stats = analysis.noise_statistics.get(&readout_key);

            let error_01 = readout_stats.map_or(0.02, |s| s.mean);
            let error_10 = readout_stats.map_or(0.02, |s| s.std_dev);
            let readout_params = ReadoutNoiseParams {
                assignment_matrix: [[1.0 - error_01, error_01], [error_10, 1.0 - error_10]],
                readout_excitation: 0.001,
                readout_relaxation: 0.001,
            };

            readout_noise.insert(qubit_id, readout_params);
        }

        Ok(CalibrationNoiseModel {
            device_id: self.device_id.clone(),
            qubit_noise,
            gate_noise,
            two_qubit_noise: HashMap::new(),
            readout_noise,
            crosstalk: CrosstalkNoise {
                crosstalk_matrix: {
                    let matrix = &analysis.correlation_analysis.correlationmatrix;
                    let rows = matrix.nrows();
                    let cols = matrix.ncols();
                    let mut vec_matrix = Vec::with_capacity(rows);
                    for i in 0..rows {
                        let mut row = Vec::with_capacity(cols);
                        for j in 0..cols {
                            row.push(matrix[[i, j]]);
                        }
                        vec_matrix.push(row);
                    }
                    vec_matrix
                },
                threshold: 0.1,
                single_qubit_crosstalk: 0.001,
                two_qubit_crosstalk: 0.005,
            },
            temperature: 15.0, // mK
        })
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SciRS2NoiseConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub const fn config(&self) -> &SciRS2NoiseConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::create_ideal_calibration;

    #[test]
    fn test_noise_modeler_creation() {
        let modeler = SciRS2NoiseModeler::new("test_device".to_string());
        assert_eq!(modeler.device_id, "test_device");
        assert!(modeler.config.enable_ml_modeling);
    }

    #[test]
    fn test_noise_data_extraction() {
        let calibration = create_ideal_calibration("test".to_string(), 4);
        let modeler = SciRS2NoiseModeler::new("test_device".to_string());

        let noise_data = modeler
            .extract_noise_data(&calibration)
            .expect("noise data extraction should succeed");
        assert!(!noise_data.is_empty());

        // Should have readout noise data for all qubits
        for i in 0..4 {
            let key = format!("readout_{}", i);
            assert!(noise_data.contains_key(&key));
        }
    }

    #[test]
    fn test_simple_noise_analysis() {
        let modeler = SciRS2NoiseModeler::new("test_device".to_string());
        let mut noise_data = HashMap::new();

        // Add test data
        noise_data.insert(
            "test_noise".to_string(),
            Array1::from_vec(vec![0.01, 0.02, 0.015]),
        );

        let analysis = modeler
            .simple_noise_analysis(&noise_data)
            .expect("simple noise analysis should succeed");
        assert!(analysis.noise_statistics.contains_key("test_noise"));
        assert!(analysis.correlation_analysis.correlation_strength >= 0.0);
    }

    #[test]
    fn test_full_noise_modeling() {
        let calibration = create_ideal_calibration("test".to_string(), 2);
        let modeler = SciRS2NoiseModeler::new("test_device".to_string());

        let noise_model = modeler
            .model_noise(&calibration)
            .expect("noise modeling should succeed");
        assert_eq!(noise_model.device_id, "test_device");
        assert_eq!(noise_model.qubit_noise.len(), 2);
        assert!(!noise_model.gate_noise.is_empty());
        assert_eq!(noise_model.readout_noise.len(), 2);
    }

    #[test]
    fn test_correlation_analysis() {
        let modeler = SciRS2NoiseModeler::new("test_device".to_string());
        let mut noise_data = HashMap::new();

        // Add correlated test data
        noise_data.insert(
            "noise1".to_string(),
            Array1::from_vec(vec![0.01, 0.02, 0.03]),
        );
        noise_data.insert(
            "noise2".to_string(),
            Array1::from_vec(vec![0.02, 0.04, 0.06]),
        );

        let correlation = modeler
            .perform_correlation_analysis(&noise_data)
            .expect("correlation analysis should succeed");
        assert_eq!(correlation.correlationmatrix.shape(), [2, 2]);
        assert_eq!(correlation.correlationmatrix[[0, 0]], 1.0);
        assert_eq!(correlation.correlationmatrix[[1, 1]], 1.0);

        // Off-diagonal should show correlation
        assert!(correlation.correlationmatrix[[0, 1]].abs() > 0.5);
    }
}
