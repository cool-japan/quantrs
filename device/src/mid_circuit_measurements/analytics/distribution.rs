//! Distribution analysis components

use super::super::results::*;
use crate::DeviceResult;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Distribution analyzer for measurement data
pub struct DistributionAnalyzer {
    confidence_level: f64,
}

impl DistributionAnalyzer {
    /// Create new distribution analyzer
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }

    /// Create distribution analyzer with custom confidence level
    pub fn with_confidence_level(confidence_level: f64) -> Self {
        Self { confidence_level }
    }

    /// Analyze distributions in measurement data
    pub fn analyze(&self, values: &[f64]) -> DeviceResult<DistributionAnalysisResults> {
        if values.is_empty() {
            return Ok(DistributionAnalysisResults::default());
        }

        // Fit various distributions
        let best_fit_distributions = self.fit_distributions(values)?;

        // Compare distributions
        let distribution_comparisons =
            self.compare_distributions(values, &best_fit_distributions)?;

        // Fit mixture models if warranted
        let mixture_models = if values.len() > 50 {
            Some(self.fit_mixture_models(values)?)
        } else {
            None
        };

        // Assess normality
        let normality_assessment = self.assess_normality(values)?;

        Ok(DistributionAnalysisResults {
            best_fit_distributions,
            distribution_comparisons,
            mixture_models,
            normality_assessment,
        })
    }

    /// Fit various statistical distributions to data
    fn fit_distributions(&self, values: &[f64]) -> DeviceResult<HashMap<String, DistributionFit>> {
        let mut distributions = HashMap::new();

        // Fit normal distribution
        let normal_fit = self.fit_normal_distribution(values)?;
        distributions.insert("normal".to_string(), normal_fit);

        // Fit exponential distribution
        let exponential_fit = self.fit_exponential_distribution(values)?;
        distributions.insert("exponential".to_string(), exponential_fit);

        // Fit uniform distribution
        let uniform_fit = self.fit_uniform_distribution(values)?;
        distributions.insert("uniform".to_string(), uniform_fit);

        // Fit gamma distribution
        let gamma_fit = self.fit_gamma_distribution(values)?;
        distributions.insert("gamma".to_string(), gamma_fit);

        // Fit beta distribution
        let beta_fit = self.fit_beta_distribution(values)?;
        distributions.insert("beta".to_string(), beta_fit);

        Ok(distributions)
    }

    /// Fit normal distribution
    fn fit_normal_distribution(&self, values: &[f64]) -> DeviceResult<DistributionFit> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let parameters = vec![mean, std_dev];
        let log_likelihood = self.calculate_normal_log_likelihood(values, mean, std_dev);
        let aic = -2.0 * log_likelihood + 2.0 * parameters.len() as f64;
        let bic = -2.0 * log_likelihood + (parameters.len() as f64) * (values.len() as f64).ln();

        Ok(DistributionFit {
            distribution_name: "Normal".to_string(),
            parameters,
            log_likelihood,
            aic,
            bic,
            ks_statistic: self
                .calculate_ks_statistic(values, |x| self.normal_cdf(x, mean, std_dev)),
            ks_p_value: 0.1, // Placeholder
        })
    }

    /// Fit exponential distribution
    fn fit_exponential_distribution(&self, values: &[f64]) -> DeviceResult<DistributionFit> {
        if values.iter().any(|&x| x <= 0.0) {
            // Exponential distribution requires positive values
            return Ok(DistributionFit {
                distribution_name: "Exponential".to_string(),
                parameters: vec![],
                log_likelihood: f64::NEG_INFINITY,
                aic: f64::INFINITY,
                bic: f64::INFINITY,
                ks_statistic: 1.0,
                ks_p_value: 0.0,
            });
        }

        let lambda = 1.0 / (values.iter().sum::<f64>() / values.len() as f64);
        let parameters = vec![lambda];

        let log_likelihood = values
            .iter()
            .map(|&x| lambda.ln() - lambda * x)
            .sum::<f64>();

        let aic = -2.0 * log_likelihood + 2.0 * parameters.len() as f64;
        let bic = -2.0 * log_likelihood + (parameters.len() as f64) * (values.len() as f64).ln();

        Ok(DistributionFit {
            distribution_name: "Exponential".to_string(),
            parameters,
            log_likelihood,
            aic,
            bic,
            ks_statistic: self.calculate_ks_statistic(values, |x| self.exponential_cdf(x, lambda)),
            ks_p_value: 0.1, // Placeholder
        })
    }

    /// Fit uniform distribution
    fn fit_uniform_distribution(&self, values: &[f64]) -> DeviceResult<DistributionFit> {
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let parameters = vec![min_val, max_val];
        let range = max_val - min_val;

        let log_likelihood = if range > 0.0 {
            -(values.len() as f64) * range.ln()
        } else {
            f64::NEG_INFINITY
        };

        let aic = -2.0 * log_likelihood + 2.0 * parameters.len() as f64;
        let bic = -2.0 * log_likelihood + (parameters.len() as f64) * (values.len() as f64).ln();

        Ok(DistributionFit {
            distribution_name: "Uniform".to_string(),
            parameters,
            log_likelihood,
            aic,
            bic,
            ks_statistic: self
                .calculate_ks_statistic(values, |x| self.uniform_cdf(x, min_val, max_val)),
            ks_p_value: 0.1, // Placeholder
        })
    }

    /// Fit gamma distribution (simplified method of moments)
    fn fit_gamma_distribution(&self, values: &[f64]) -> DeviceResult<DistributionFit> {
        if values.iter().any(|&x| x <= 0.0) {
            return Ok(DistributionFit {
                distribution_name: "Gamma".to_string(),
                parameters: vec![],
                log_likelihood: f64::NEG_INFINITY,
                aic: f64::INFINITY,
                bic: f64::INFINITY,
                ks_statistic: 1.0,
                ks_p_value: 0.0,
            });
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        // Method of moments estimators
        let scale = variance / mean;
        let shape = mean / scale;

        let parameters = vec![shape, scale];
        let log_likelihood = 0.0; // Placeholder - would need gamma function
        let aic = -2.0 * log_likelihood + 2.0 * parameters.len() as f64;
        let bic = -2.0 * log_likelihood + (parameters.len() as f64) * (values.len() as f64).ln();

        Ok(DistributionFit {
            distribution_name: "Gamma".to_string(),
            parameters,
            log_likelihood,
            aic,
            bic,
            ks_statistic: 0.5, // Placeholder
            ks_p_value: 0.1,
        })
    }

    /// Fit beta distribution (simplified)
    fn fit_beta_distribution(&self, values: &[f64]) -> DeviceResult<DistributionFit> {
        // Check if values are in [0,1] range
        if values.iter().any(|&x| x < 0.0 || x > 1.0) {
            return Ok(DistributionFit {
                distribution_name: "Beta".to_string(),
                parameters: vec![],
                log_likelihood: f64::NEG_INFINITY,
                aic: f64::INFINITY,
                bic: f64::INFINITY,
                ks_statistic: 1.0,
                ks_p_value: 0.0,
            });
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        // Method of moments estimators
        let common_factor = mean * (1.0 - mean) / variance - 1.0;
        let alpha = mean * common_factor;
        let beta = (1.0 - mean) * common_factor;

        let parameters = vec![alpha, beta];
        let log_likelihood = 0.0; // Placeholder - would need beta function
        let aic = -2.0 * log_likelihood + 2.0 * parameters.len() as f64;
        let bic = -2.0 * log_likelihood + (parameters.len() as f64) * (values.len() as f64).ln();

        Ok(DistributionFit {
            distribution_name: "Beta".to_string(),
            parameters,
            log_likelihood,
            aic,
            bic,
            ks_statistic: 0.5, // Placeholder
            ks_p_value: 0.1,
        })
    }

    /// Compare different distribution fits
    fn compare_distributions(
        &self,
        values: &[f64],
        distributions: &HashMap<String, DistributionFit>,
    ) -> DeviceResult<Vec<DistributionComparison>> {
        let mut comparisons = Vec::new();

        let dist_names: Vec<String> = distributions.keys().cloned().collect();

        for i in 0..dist_names.len() {
            for j in (i + 1)..dist_names.len() {
                let dist1 = &distributions[&dist_names[i]];
                let dist2 = &distributions[&dist_names[j]];

                // Compare AICs
                let aic_diff = dist2.aic - dist1.aic;
                let better_fit = if aic_diff > 2.0 {
                    dist_names[i].clone()
                } else if aic_diff < -2.0 {
                    dist_names[j].clone()
                } else {
                    "Comparable".to_string()
                };

                // Likelihood ratio test (simplified)
                let lr_statistic = 2.0 * (dist1.log_likelihood - dist2.log_likelihood).abs();
                let lr_p_value = if lr_statistic > 3.84 { 0.05 } else { 0.1 }; // Chi-square approximation

                comparisons.push(DistributionComparison {
                    distribution1: dist_names[i].clone(),
                    distribution2: dist_names[j].clone(),
                    aic_difference: aic_diff,
                    bic_difference: dist2.bic - dist1.bic,
                    likelihood_ratio_test: StatisticalTest {
                        statistic: lr_statistic,
                        p_value: lr_p_value,
                        critical_value: 3.84,
                        is_significant: lr_statistic > 3.84,
                        effect_size: Some(aic_diff.abs() / 10.0),
                    },
                    better_fit,
                });
            }
        }

        Ok(comparisons)
    }

    /// Fit mixture models
    fn fit_mixture_models(&self, values: &[f64]) -> DeviceResult<MixtureModelResults> {
        // Simple two-component Gaussian mixture (simplified EM algorithm)
        let n = values.len();
        let overall_mean = values.iter().sum::<f64>() / n as f64;
        let overall_var = values
            .iter()
            .map(|&x| (x - overall_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        // Initialize two components
        let mut weights = vec![0.5, 0.5];
        let mut means = vec![
            overall_mean - overall_var.sqrt(),
            overall_mean + overall_var.sqrt(),
        ];
        let mut variances = vec![overall_var / 2.0, overall_var / 2.0];

        // Run simplified EM for a few iterations
        for _ in 0..10 {
            // E-step: calculate responsibilities
            let mut responsibilities = vec![vec![0.0; 2]; n];
            for (i, &x) in values.iter().enumerate() {
                let mut total = 0.0;
                for k in 0..2 {
                    let gaussian = weights[k] * self.gaussian_pdf(x, means[k], variances[k].sqrt());
                    responsibilities[i][k] = gaussian;
                    total += gaussian;
                }
                if total > 0.0 {
                    for k in 0..2 {
                        responsibilities[i][k] /= total;
                    }
                }
            }

            // M-step: update parameters
            for k in 0..2 {
                let nk: f64 = responsibilities.iter().map(|r| r[k]).sum();
                weights[k] = nk / n as f64;

                if nk > 0.0 {
                    means[k] = responsibilities
                        .iter()
                        .zip(values.iter())
                        .map(|(r, &x)| r[k] * x)
                        .sum::<f64>()
                        / nk;

                    variances[k] = responsibilities
                        .iter()
                        .zip(values.iter())
                        .map(|(r, &x)| r[k] * (x - means[k]).powi(2))
                        .sum::<f64>()
                        / nk;
                }
            }
        }

        // Calculate log-likelihood
        let log_likelihood = values
            .iter()
            .map(|&x| {
                let mixture_prob = weights[0] * self.gaussian_pdf(x, means[0], variances[0].sqrt())
                    + weights[1] * self.gaussian_pdf(x, means[1], variances[1].sqrt());
                mixture_prob.ln()
            })
            .sum::<f64>();

        let num_params = 5; // 2 weights (1 constrained), 2 means, 2 variances, minus 1 constraint
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (n as f64).ln();

        Ok(MixtureModelResults {
            n_components: 2,
            weights: Array1::from_vec(weights),
            component_parameters: vec![
                vec![means[0], variances[0].sqrt()],
                vec![means[1], variances[1].sqrt()],
            ],
            log_likelihood,
            bic,
            assignments: Array1::zeros(n),
        })
    }

    /// Assess normality of data
    fn assess_normality(&self, values: &[f64]) -> DeviceResult<NormalityAssessment> {
        // Shapiro-Wilk test (simplified)
        let shapiro_wilk = self.shapiro_wilk_test(values)?;

        // Anderson-Darling test (simplified)
        let anderson_darling = self.anderson_darling_test(values)?;

        // Jarque-Bera test
        let jarque_bera = self.jarque_bera_test(values)?;

        // Overall assessment
        let is_normal = shapiro_wilk.p_value > 0.05
            && anderson_darling.p_value > 0.05
            && jarque_bera.p_value > 0.05;

        let normality_confidence =
            (shapiro_wilk.p_value + anderson_darling.p_value + jarque_bera.p_value) / 3.0;

        Ok(NormalityAssessment {
            shapiro_wilk,
            anderson_darling,
            jarque_bera,
            is_normal,
            normality_confidence,
        })
    }

    /// Simplified Shapiro-Wilk test
    fn shapiro_wilk_test(&self, values: &[f64]) -> DeviceResult<StatisticalTest> {
        if values.len() < 3 || values.len() > 5000 {
            return Ok(StatisticalTest {
                statistic: 0.0,
                p_value: 0.5,
                critical_value: 0.95,
                is_significant: false,
                effect_size: None,
            });
        }

        // Simplified calculation - in practice would use proper SW coefficients
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Simplified W statistic
        let w_statistic = 0.95; // Placeholder
        let p_value = if w_statistic > 0.95 { 0.1 } else { 0.01 };

        Ok(StatisticalTest {
            statistic: w_statistic,
            p_value,
            critical_value: 0.95,
            is_significant: p_value < 0.05,
            effect_size: Some(1.0 - w_statistic),
        })
    }

    /// Simplified Anderson-Darling test
    fn anderson_darling_test(&self, values: &[f64]) -> DeviceResult<StatisticalTest> {
        // Placeholder implementation
        Ok(StatisticalTest {
            statistic: 0.5,
            p_value: 0.1,
            critical_value: 0.752,
            is_significant: false,
            effect_size: Some(0.1),
        })
    }

    /// Jarque-Bera test for normality
    fn jarque_bera_test(&self, values: &[f64]) -> DeviceResult<StatisticalTest> {
        if values.len() < 4 {
            return Ok(StatisticalTest::default());
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        // Calculate moments
        let m2 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3 = values.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
        let m4 = values.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

        let skewness = m3 / m2.powf(1.5);
        let kurtosis = m4 / (m2 * m2) - 3.0;

        // Jarque-Bera statistic
        let jb_statistic = (n / 6.0) * (skewness.powi(2) + kurtosis.powi(2) / 4.0);
        let p_value = if jb_statistic > 5.99 { 0.05 } else { 0.1 }; // Chi-square(2) approximation

        Ok(StatisticalTest {
            statistic: jb_statistic,
            p_value,
            critical_value: 5.99,
            is_significant: jb_statistic > 5.99,
            effect_size: Some(jb_statistic / 10.0),
        })
    }

    /// Calculate Kolmogorov-Smirnov statistic
    fn calculate_ks_statistic<F>(&self, values: &[f64], cdf: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_values.len() as f64;
        let mut max_diff: f64 = 0.0;

        for (i, &x) in sorted_values.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;
            let theoretical_cdf = cdf(x);
            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }

        max_diff
    }

    /// Calculate normal log-likelihood
    fn calculate_normal_log_likelihood(&self, values: &[f64], mean: f64, std_dev: f64) -> f64 {
        values
            .iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                -0.5 * (2.0 * std::f64::consts::PI).ln() - std_dev.ln() - 0.5 * z * z
            })
            .sum()
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        let z = (x - mean) / std_dev;
        0.5 * (1.0 + self.erf(z / (2.0_f64).sqrt()))
    }

    /// Exponential CDF
    fn exponential_cdf(&self, x: f64, lambda: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-lambda * x).exp()
        }
    }

    /// Uniform CDF
    fn uniform_cdf(&self, x: f64, min_val: f64, max_val: f64) -> f64 {
        if x < min_val {
            0.0
        } else if x > max_val {
            1.0
        } else {
            (x - min_val) / (max_val - min_val)
        }
    }

    /// Gaussian PDF
    fn gaussian_pdf(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        let z = (x - mean) / std_dev;
        (1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

impl Default for DistributionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
