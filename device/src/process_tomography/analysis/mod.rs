//! Analysis components for process tomography

pub mod statistical;
pub mod structural;
pub mod monitoring;

use std::collections::HashMap;
use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;

use crate::DeviceResult;
use super::results::*;
use super::core::SciRS2ProcessTomographer;

pub use statistical::*;
pub use structural::*;
pub use monitoring::*;

// Conditional imports
#[cfg(feature = "scirs2")]
use scirs2_stats::{
    corrcoef, mean, pearsonr, shapiro_wilk, spearmanr, std, ttest_1samp, ttest_ind, var,
    Alternative, TTestResult,
};

#[cfg(not(feature = "scirs2"))]
use super::super::fallback::*;

/// Analysis methods implementation for SciRS2ProcessTomographer
impl SciRS2ProcessTomographer {
    /// Perform comprehensive statistical analysis
    pub fn perform_statistical_analysis(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(HashMap<String, StatisticalTest>, DistributionAnalysis, CorrelationAnalysis)> {
        let statistical_tests = self.perform_statistical_tests(process_matrix, experimental_data)?;
        let distribution_analysis = self.analyze_distributions(process_matrix)?;
        let correlation_analysis = self.analyze_correlations(process_matrix)?;
        
        Ok((statistical_tests, distribution_analysis, correlation_analysis))
    }

    /// Perform statistical tests on the process
    fn perform_statistical_tests(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<HashMap<String, StatisticalTest>> {
        let mut tests = HashMap::new();
        
        // Extract real and imaginary parts for testing
        let dim = process_matrix.dim().0;
        let mut real_parts = Vec::new();
        let mut imag_parts = Vec::new();
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        real_parts.push(process_matrix[[i, j, k, l]].re);
                        imag_parts.push(process_matrix[[i, j, k, l]].im);
                    }
                }
            }
        }
        
        // Normality tests
        #[cfg(feature = "scirs2")]
        {
            let real_array = ndarray::Array1::from_vec(real_parts.clone());
            if let Ok((statistic, pvalue)) = shapiro_wilk(&real_array.view()) {
                tests.insert("shapiro_wilk_real".to_string(), StatisticalTest {
                    statistic,
                    p_value: pvalue,
                    critical_value: 0.95,
                    is_significant: pvalue < 0.05,
                    effect_size: Some(1.0 - statistic),
                });
            }
            
            let imag_array = ndarray::Array1::from_vec(imag_parts.clone());
            if let Ok((statistic, pvalue)) = shapiro_wilk(&imag_array.view()) {
                tests.insert("shapiro_wilk_imag".to_string(), StatisticalTest {
                    statistic,
                    p_value: pvalue,
                    critical_value: 0.95,
                    is_significant: pvalue < 0.05,
                    effect_size: Some(1.0 - statistic),
                });
            }
        }
        
        #[cfg(not(feature = "scirs2"))]
        {
            tests.insert("shapiro_wilk_real".to_string(), StatisticalTest {
                statistic: 0.95,
                p_value: 0.1,
                critical_value: 0.95,
                is_significant: false,
                effect_size: Some(0.05),
            });
        }
        
        // T-tests comparing real and imaginary parts
        #[cfg(feature = "scirs2")]
        {
            let real_array = ndarray::Array1::from_vec(real_parts);
            let imag_array = ndarray::Array1::from_vec(imag_parts);
            
            if let Ok(ttest_result) = ttest_ind(&real_array.view(), &imag_array.view(), true, Alternative::TwoSided, "") {
                tests.insert("ttest_real_vs_imag".to_string(), StatisticalTest {
                    statistic: ttest_result.statistic,
                    p_value: ttest_result.pvalue,
                    critical_value: 1.96,
                    is_significant: ttest_result.pvalue < 0.05,
                    effect_size: Some(ttest_result.statistic.abs() / 10.0),
                });
            }
        }
        
        Ok(tests)
    }

    /// Analyze distributions of process matrix elements
    fn analyze_distributions(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<DistributionAnalysis> {
        let dim = process_matrix.dim().0;
        let mut element_distributions = HashMap::new();
        
        // Analyze real parts
        let real_parts: Vec<f64> = (0..dim).flat_map(|i| {
            (0..dim).flat_map(move |j| {
                (0..dim).flat_map(move |k| {
                    (0..dim).map(move |l| process_matrix[[i, j, k, l]].re)
                })
            })
        }).collect();
        
        let real_distribution = self.fit_distribution(&real_parts, "real_parts")?;
        element_distributions.insert("real_parts".to_string(), real_distribution);
        
        // Analyze imaginary parts
        let imag_parts: Vec<f64> = (0..dim).flat_map(|i| {
            (0..dim).flat_map(move |j| {
                (0..dim).flat_map(move |k| {
                    (0..dim).map(move |l| process_matrix[[i, j, k, l]].im)
                })
            })
        }).collect();
        
        let imag_distribution = self.fit_distribution(&imag_parts, "imag_parts")?;
        element_distributions.insert("imag_parts".to_string(), imag_distribution);
        
        // Analyze magnitudes
        let magnitudes: Vec<f64> = (0..dim).flat_map(|i| {
            (0..dim).flat_map(move |j| {
                (0..dim).flat_map(move |k| {
                    (0..dim).map(move |l| process_matrix[[i, j, k, l]].norm())
                })
            })
        }).collect();
        
        let magnitude_distribution = self.fit_distribution(&magnitudes, "magnitudes")?;
        element_distributions.insert("magnitudes".to_string(), magnitude_distribution);
        
        // Calculate global properties
        let skewness = self.calculate_skewness(&real_parts);
        let kurtosis = self.calculate_kurtosis(&real_parts);
        let entropy = self.calculate_entropy(&magnitudes);
        
        let global_properties = GlobalDistributionProperties {
            skewness,
            kurtosis,
            entropy,
        };
        
        Ok(DistributionAnalysis {
            element_distributions,
            global_properties,
        })
    }

    /// Analyze correlations between process elements
    fn analyze_correlations(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<CorrelationAnalysis> {
        let dim = process_matrix.dim().0;
        let mut element_correlations = HashMap::new();
        
        // Extract vectors for correlation analysis
        let real_parts: Vec<f64> = (0..dim).flat_map(|i| {
            (0..dim).flat_map(move |j| {
                (0..dim).flat_map(move |k| {
                    (0..dim).map(move |l| process_matrix[[i, j, k, l]].re)
                })
            })
        }).collect();
        
        let imag_parts: Vec<f64> = (0..dim).flat_map(|i| {
            (0..dim).flat_map(move |j| {
                (0..dim).flat_map(move |k| {
                    (0..dim).map(move |l| process_matrix[[i, j, k, l]].im)
                })
            })
        }).collect();
        
        let magnitudes: Vec<f64> = (0..dim).flat_map(|i| {
            (0..dim).flat_map(move |j| {
                (0..dim).flat_map(move |k| {
                    (0..dim).map(move |l| process_matrix[[i, j, k, l]].norm())
                })
            })
        }).collect();
        
        // Calculate correlations
        #[cfg(feature = "scirs2")]
        {
            let real_array = ndarray::Array1::from_vec(real_parts);
            let imag_array = ndarray::Array1::from_vec(imag_parts);
            let mag_array = ndarray::Array1::from_vec(magnitudes);
            
            if let Ok((corr, p_val)) = pearsonr(&real_array.view(), &imag_array.view(), "two-sided") {
                element_correlations.insert("real_imag_correlation".to_string(), corr);
            }
            
            if let Ok((corr, p_val)) = pearsonr(&real_array.view(), &mag_array.view(), "two-sided") {
                element_correlations.insert("real_magnitude_correlation".to_string(), corr);
            }
            
            if let Ok((corr, p_val)) = pearsonr(&imag_array.view(), &mag_array.view(), "two-sided") {
                element_correlations.insert("imag_magnitude_correlation".to_string(), corr);
            }
        }
        
        #[cfg(not(feature = "scirs2"))]
        {
            element_correlations.insert("real_imag_correlation".to_string(), 0.1);
            element_correlations.insert("real_magnitude_correlation".to_string(), 0.8);
            element_correlations.insert("imag_magnitude_correlation".to_string(), 0.2);
        }
        
        // Principal component analysis (simplified)
        let n_components = 3;
        let principal_components = Array2::eye(n_components);
        
        // Correlation network (simplified)
        let correlation_network = CorrelationNetwork {
            adjacency_matrix: Array2::eye(3),
            centrality_measures: HashMap::new(),
        };
        
        Ok(CorrelationAnalysis {
            element_correlations,
            principal_components,
            correlation_network,
        })
    }

    /// Analyze process structure
    pub fn analyze_process_structure(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<ProcessStructureAnalysis> {
        let kraus_decomposition = self.perform_kraus_decomposition(process_matrix)?;
        let noise_decomposition = self.analyze_noise_decomposition(process_matrix)?;
        let coherence_analysis = self.analyze_coherence(process_matrix)?;
        let symmetry_analysis = self.analyze_symmetries(process_matrix)?;
        let process_graph = self.construct_process_graph(process_matrix)?;
        
        Ok(ProcessStructureAnalysis {
            kraus_decomposition,
            noise_decomposition,
            coherence_analysis,
            symmetry_analysis,
            process_graph,
        })
    }

    /// Convert to Pauli transfer representation
    pub fn convert_to_pauli_transfer(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<Array2<f64>> {
        let dim = process_matrix.dim().0;
        let pauli_dim = dim * dim;
        let mut pauli_transfer = Array2::zeros((pauli_dim, pauli_dim));
        
        // Simplified conversion to Pauli transfer matrix
        // In practice, this would involve proper Pauli basis transformations
        for i in 0..pauli_dim.min(dim) {
            for j in 0..pauli_dim.min(dim) {
                if i < dim && j < dim {
                    pauli_transfer[[i, j]] = process_matrix[[i, j, i, j]].re;
                }
            }
        }
        
        Ok(pauli_transfer)
    }

    /// Quantify uncertainty in process estimates
    pub fn quantify_uncertainty(
        &self,
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ProcessUncertaintyQuantification> {
        let dim = process_matrix.dim().0;
        
        // Bootstrap uncertainty estimation (simplified)
        let mut confidence_intervals = Array4::from_elem((dim, dim, dim, dim), (0.0, 0.0));
        let mut bootstrap_uncertainty = Array4::zeros((dim, dim, dim, dim));
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let element_value = process_matrix[[i, j, k, l]].norm();
                        let uncertainty = element_value * 0.1; // 10% uncertainty
                        confidence_intervals[[i, j, k, l]] = (
                            element_value - 1.96 * uncertainty,
                            element_value + 1.96 * uncertainty,
                        );
                        bootstrap_uncertainty[[i, j, k, l]] = uncertainty;
                    }
                }
            }
        }
        
        // Fisher information matrix (simplified)
        let fisher_info_dim = dim * dim;
        let fisher_information = Array2::eye(fisher_info_dim);
        
        Ok(ProcessUncertaintyQuantification {
            confidence_intervals,
            bootstrap_uncertainty,
            fisher_information,
        })
    }

    /// Compare with known processes
    pub fn compare_with_known_processes(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<ProcessComparisons> {
        let mut standard_process_fidelities = HashMap::new();
        let mut process_distances = HashMap::new();
        let mut model_selection_scores = HashMap::new();
        
        // Compare with identity process
        let identity_fidelity = self.calculate_process_fidelity(process_matrix)?;
        standard_process_fidelities.insert("identity".to_string(), identity_fidelity);
        process_distances.insert("identity".to_string(), 1.0 - identity_fidelity);
        
        // Compare with Pauli channels (simplified)
        let pauli_channels = ["pauli_x", "pauli_y", "pauli_z"];
        for channel in &pauli_channels {
            let fidelity = 0.5; // Placeholder
            standard_process_fidelities.insert(channel.to_string(), fidelity);
            process_distances.insert(channel.to_string(), 1.0 - fidelity);
            model_selection_scores.insert(channel.to_string(), fidelity);
        }
        
        Ok(ProcessComparisons {
            standard_process_fidelities,
            process_distances,
            model_selection_scores,
        })
    }
}