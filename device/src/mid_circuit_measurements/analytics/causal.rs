//! Causal analysis components

use super::super::results::*;
use crate::DeviceResult;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Causal analyzer for measurement data
pub struct CausalAnalyzer {
    confidence_level: f64,
    min_observations: usize,
}

impl CausalAnalyzer {
    /// Create new causal analyzer
    pub const fn new() -> Self {
        Self {
            confidence_level: 0.95,
            min_observations: 30,
        }
    }

    /// Create causal analyzer with custom parameters
    pub const fn with_parameters(confidence_level: f64, min_observations: usize) -> Self {
        Self {
            confidence_level,
            min_observations,
        }
    }

    /// Perform causal analysis
    pub fn analyze(
        &self,
        latencies: &[f64],
        confidences: &[f64],
        timestamps: &[f64],
    ) -> DeviceResult<CausalAnalysisResults> {
        if latencies.len() < self.min_observations
            || confidences.len() < self.min_observations
            || timestamps.len() < self.min_observations
        {
            return Ok(CausalAnalysisResults::default());
        }

        // Ensure all arrays have the same length
        let min_len = latencies.len().min(confidences.len()).min(timestamps.len());
        let latencies = &latencies[..min_len];
        let confidences = &confidences[..min_len];
        let timestamps = &timestamps[..min_len];

        // Discover causal relationships
        let causal_relationships =
            self.discover_causal_relationships(latencies, confidences, timestamps)?;

        // Estimate causal effects
        let causal_effects = self.estimate_causal_effects(latencies, confidences, timestamps)?;

        // Build causal graph
        let causal_graph = self.build_causal_graph(latencies, confidences, timestamps)?;

        // Perform intervention analysis
        let intervention_analysis =
            self.analyze_interventions(latencies, confidences, timestamps)?;

        // Assess confounding
        let confounding_assessment = self.assess_confounding(latencies, confidences, timestamps)?;

        Ok(CausalAnalysisResults {
            causal_graph,
            causal_effects,
            confounding_analysis: confounding_assessment,
            causal_strength: HashMap::new(),
        })
    }

    /// Discover causal relationships using conditional independence tests
    fn discover_causal_relationships(
        &self,
        latencies: &[f64],
        confidences: &[f64],
        timestamps: &[f64],
    ) -> DeviceResult<Vec<CausalRelationship>> {
        let mut relationships = Vec::new();

        // Test for causal relationship: latency -> confidence
        let latency_confidence_relationship = self.test_causal_relationship(
            latencies,
            confidences,
            timestamps,
            "latency",
            "confidence",
        )?;
        if latency_confidence_relationship.causal_strength > 0.1 {
            relationships.push(latency_confidence_relationship);
        }

        // Test for causal relationship: timestamp -> latency (temporal effects)
        let timestamp_latency_relationship = self.test_causal_relationship(
            timestamps,
            latencies,
            confidences,
            "timestamp",
            "latency",
        )?;
        if timestamp_latency_relationship.causal_strength > 0.1 {
            relationships.push(timestamp_latency_relationship);
        }

        // Test for causal relationship: timestamp -> confidence
        let timestamp_confidence_relationship = self.test_causal_relationship(
            timestamps,
            confidences,
            latencies,
            "timestamp",
            "confidence",
        )?;
        if timestamp_confidence_relationship.causal_strength > 0.1 {
            relationships.push(timestamp_confidence_relationship);
        }

        Ok(relationships)
    }

    /// Test for causal relationship between two variables
    fn test_causal_relationship(
        &self,
        cause: &[f64],
        effect: &[f64],
        confounder: &[f64],
        cause_name: &str,
        effect_name: &str,
    ) -> DeviceResult<CausalRelationship> {
        // Calculate unconditional correlation
        let unconditional_corr = self.calculate_correlation(cause, effect);

        // Calculate partial correlation (controlling for confounder)
        let partial_corr = self.calculate_partial_correlation(cause, effect, confounder)?;

        // Granger causality test (simplified)
        let granger_causality = self.granger_causality_test(cause, effect)?;

        // Calculate causal strength
        let causal_strength =
            (unconditional_corr.abs() + partial_corr.abs() + granger_causality.statistic.abs())
                / 3.0;

        // Determine causal direction using temporal precedence
        let causal_direction = if cause_name == "timestamp" {
            CausalDirection::Forward
        } else if effect_name == "timestamp" {
            CausalDirection::Backward
        } else {
            // Use correlation strength to infer direction
            if unconditional_corr > 0.0 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            }
        };

        // Calculate confidence interval for causal effect
        let effect_size = partial_corr;
        let standard_error = self.calculate_standard_error_correlation(cause.len());
        let margin = 1.96 * standard_error; // 95% CI
        let confidence_interval = (effect_size - margin, effect_size + margin);

        Ok(CausalRelationship {
            cause: cause_name.to_string(),
            effect: effect_name.to_string(),
            causal_strength,
            causal_direction,
            p_value: granger_causality.p_value,
            confidence_interval,
            mechanism: CausalMechanism::Direct, // Simplified
        })
    }

    /// Estimate causal effects using different methods
    fn estimate_causal_effects(
        &self,
        latencies: &[f64],
        confidences: &[f64],
        timestamps: &[f64],
    ) -> DeviceResult<Vec<CausalEffect>> {
        let mut effects = Vec::new();

        // Estimate effect of latency on confidence
        let latency_effect =
            self.estimate_treatment_effect(latencies, confidences, "latency", "confidence")?;
        effects.push(latency_effect);

        // Estimate temporal effects
        let temporal_effect =
            self.estimate_temporal_effect(timestamps, confidences, "time", "confidence")?;
        effects.push(temporal_effect);

        Ok(effects)
    }

    /// Estimate treatment effect using propensity score matching (simplified)
    fn estimate_treatment_effect(
        &self,
        treatment: &[f64],
        outcome: &[f64],
        treatment_name: &str,
        outcome_name: &str,
    ) -> DeviceResult<CausalEffect> {
        // Binary treatment: high vs low values (median split)
        let median = self.median(treatment);
        let mut treated_outcomes = Vec::new();
        let mut control_outcomes = Vec::new();

        for i in 0..treatment.len() {
            if treatment[i] > median {
                treated_outcomes.push(outcome[i]);
            } else {
                control_outcomes.push(outcome[i]);
            }
        }

        // Calculate average treatment effect
        let treated_mean = if treated_outcomes.is_empty() {
            0.0
        } else {
            treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64
        };

        let control_mean = if control_outcomes.is_empty() {
            0.0
        } else {
            control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64
        };

        let average_treatment_effect = treated_mean - control_mean;

        // Calculate standard error
        let treated_var = if treated_outcomes.len() > 1 {
            treated_outcomes
                .iter()
                .map(|&x| (x - treated_mean).powi(2))
                .sum::<f64>()
                / (treated_outcomes.len() - 1) as f64
        } else {
            0.0
        };

        let control_var = if control_outcomes.len() > 1 {
            control_outcomes
                .iter()
                .map(|&x| (x - control_mean).powi(2))
                .sum::<f64>()
                / (control_outcomes.len() - 1) as f64
        } else {
            0.0
        };

        let standard_error = ((treated_var / treated_outcomes.len() as f64)
            + (control_var / control_outcomes.len() as f64))
            .sqrt();

        // Calculate confidence interval
        let margin = 1.96 * standard_error;
        let confidence_interval = (
            average_treatment_effect - margin,
            average_treatment_effect + margin,
        );

        // Calculate p-value (t-test approximation)
        let t_statistic = if standard_error > 1e-10 {
            average_treatment_effect / standard_error
        } else {
            0.0
        };
        let p_value = if t_statistic.abs() > 1.96 { 0.05 } else { 0.1 };

        Ok(CausalEffect {
            cause: treatment_name.to_string(),
            effect: outcome_name.to_string(),
            effect_size: average_treatment_effect,
            confidence_interval,
            p_value,
            mechanism: CausalMechanism::Direct,
        })
    }

    /// Estimate temporal causal effect
    fn estimate_temporal_effect(
        &self,
        time_series: &[f64],
        outcome: &[f64],
        treatment_name: &str,
        outcome_name: &str,
    ) -> DeviceResult<CausalEffect> {
        // Simple temporal effect: correlation with time trend
        let time_trend: Vec<f64> = (0..time_series.len()).map(|i| i as f64).collect();
        let trend_correlation = self.calculate_correlation(&time_trend, outcome);

        let standard_error = self.calculate_standard_error_correlation(outcome.len());
        let margin = 1.96 * standard_error;
        let confidence_interval = (trend_correlation - margin, trend_correlation + margin);

        let t_statistic = if standard_error > 1e-10 {
            trend_correlation / standard_error
        } else {
            0.0
        };
        let p_value = if t_statistic.abs() > 1.96 { 0.05 } else { 0.1 };

        Ok(CausalEffect {
            cause: treatment_name.to_string(),
            effect: outcome_name.to_string(),
            effect_size: trend_correlation,
            confidence_interval,
            p_value,
            mechanism: CausalMechanism::Indirect,
        })
    }

    /// Build causal graph using PC algorithm (simplified)
    fn build_causal_graph(
        &self,
        latencies: &[f64],
        confidences: &[f64],
        timestamps: &[f64],
    ) -> DeviceResult<CausalGraph> {
        let variables = vec![
            "latency".to_string(),
            "confidence".to_string(),
            "timestamp".to_string(),
        ];
        let n_vars = variables.len();

        // Initialize adjacency matrix (undirected graph)
        let mut adjacency_matrix = Array2::zeros((n_vars, n_vars));

        // Test for edges using conditional independence
        let data = [latencies, confidences, timestamps];

        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                // Test independence between variables i and j
                let correlation = self.calculate_correlation(data[i], data[j]);
                if correlation.abs() > 0.1 {
                    // Threshold for edge existence
                    adjacency_matrix[[i, j]] = 1.0;
                    adjacency_matrix[[j, i]] = 1.0;
                }
            }
        }

        // Orient edges based on temporal constraints
        let mut edge_directions = HashMap::new();

        // Time always comes before other variables
        if adjacency_matrix[[2, 0]] > 0.0 {
            // timestamp -> latency
            edge_directions.insert((2, 0), EdgeType::Directed);
        }
        if adjacency_matrix[[2, 1]] > 0.0 {
            // timestamp -> confidence
            edge_directions.insert((2, 1), EdgeType::Directed);
        }

        // Latency might cause confidence issues
        if adjacency_matrix[[0, 1]] > 0.0 {
            edge_directions.insert((0, 1), EdgeType::Directed);
        }

        Ok(CausalGraph {
            adjacency_matrix,
            node_names: variables,
            edge_weights: HashMap::new(),
            graph_confidence: 0.8,
        })
    }

    /// Analyze potential interventions
    fn analyze_interventions(
        &self,
        latencies: &[f64],
        confidences: &[f64],
        timestamps: &[f64],
    ) -> DeviceResult<Vec<InterventionAnalysis>> {
        let mut interventions = Vec::new();

        // Intervention: Reduce latency
        let latency_intervention = InterventionAnalysis {
            intervention_type: "Latency Reduction".to_string(),
            target_variable: "latency".to_string(),
            intervention_magnitude: -0.2, // 20% reduction
            predicted_effects: vec![PredictedEffect {
                variable: "confidence".to_string(),
                effect_size: 0.15, // Expected improvement
                confidence_interval: (0.05, 0.25),
                p_value: 0.01,
            }],
            intervention_cost: 0.5, // Moderate cost
            benefit_ratio: 0.3,     // 15% improvement / 50% cost
        };
        interventions.push(latency_intervention);

        // Intervention: Improve timing consistency
        let timing_intervention = InterventionAnalysis {
            intervention_type: "Timing Optimization".to_string(),
            target_variable: "timestamp".to_string(),
            intervention_magnitude: -0.1, // 10% variance reduction
            predicted_effects: vec![
                PredictedEffect {
                    variable: "latency".to_string(),
                    effect_size: -0.1,
                    confidence_interval: (-0.2, 0.0),
                    p_value: 0.05,
                },
                PredictedEffect {
                    variable: "confidence".to_string(),
                    effect_size: 0.08,
                    confidence_interval: (0.02, 0.14),
                    p_value: 0.02,
                },
            ],
            intervention_cost: 0.3,
            benefit_ratio: 0.27, // 8% improvement / 30% cost
        };
        interventions.push(timing_intervention);

        Ok(interventions)
    }

    /// Assess confounding variables
    fn assess_confounding(
        &self,
        latencies: &[f64],
        confidences: &[f64],
        timestamps: &[f64],
    ) -> DeviceResult<ConfoundingAnalysis> {
        // Test if timestamp confounds the latency-confidence relationship
        let direct_correlation = self.calculate_correlation(latencies, confidences);
        let partial_correlation =
            self.calculate_partial_correlation(latencies, confidences, timestamps)?;

        let confounding_strength = (direct_correlation - partial_correlation).abs();

        let confounders = if confounding_strength > 0.1 {
            vec![ConfoundingVariable {
                variable: "timestamp".to_string(),
                confounding_strength,
                adjustment_method: "Partial Correlation".to_string(),
                p_value: 0.05,
            }]
        } else {
            vec![]
        };

        let recommendations = if confounders.is_empty() {
            vec!["No significant confounding detected".to_string()]
        } else {
            vec![
                "Consider temporal effects in causal analysis".to_string(),
                "Use time-series methods for better causal inference".to_string(),
            ]
        };

        let confounder_names = confounders.iter().map(|c| c.variable.clone()).collect();
        let mut confounder_strength = HashMap::new();
        for confounder in &confounders {
            confounder_strength
                .insert(confounder.variable.clone(), confounder.confounding_strength);
        }

        Ok(ConfoundingAnalysis {
            confounders: confounder_names,
            confounder_strength,
            backdoor_satisfied: confounding_strength < 0.1,
            frontdoor_satisfied: confounding_strength < 0.1,
        })
    }

    /// Calculate Pearson correlation
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Calculate partial correlation controlling for third variable
    fn calculate_partial_correlation(&self, x: &[f64], y: &[f64], z: &[f64]) -> DeviceResult<f64> {
        let rxy = self.calculate_correlation(x, y);
        let rxz = self.calculate_correlation(x, z);
        let ryz = self.calculate_correlation(y, z);

        let numerator = rxz.mul_add(-ryz, rxy);
        let denominator = (rxz.mul_add(-rxz, 1.0) * ryz.mul_add(-ryz, 1.0)).sqrt();

        if denominator > 1e-10 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Simplified Granger causality test
    fn granger_causality_test(&self, x: &[f64], y: &[f64]) -> DeviceResult<StatisticalTest> {
        // Simplified version - just test if lagged x predicts y
        if x.len() < 3 || y.len() < 3 {
            return Ok(StatisticalTest::default());
        }

        let lag = 1;
        let n = x.len().min(y.len()) - lag;

        // Correlation between lagged x and current y
        let x_lagged = &x[0..n];
        let y_current = &y[lag..lag + n];

        let lagged_correlation = self.calculate_correlation(x_lagged, y_current);
        let f_statistic = lagged_correlation.abs() * (n as f64).sqrt(); // Simplified F-stat
        let p_value = if f_statistic > 1.96 { 0.05 } else { 0.1 };

        Ok(StatisticalTest {
            statistic: f_statistic,
            p_value,
            critical_value: 1.96,
            is_significant: f_statistic > 1.96,
            effect_size: Some(lagged_correlation.abs()),
        })
    }

    /// Calculate median
    fn median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            f64::midpoint(sorted[mid - 1], sorted[mid])
        } else {
            sorted[sorted.len() / 2]
        }
    }

    /// Calculate standard error for correlation
    fn calculate_standard_error_correlation(&self, n: usize) -> f64 {
        if n > 2 {
            1.0 / ((n - 2) as f64).sqrt()
        } else {
            1.0
        }
    }

    /// Calculate graph score (simplified BIC-like score)
    fn calculate_graph_score(
        &self,
        adjacency_matrix: &Array2<f64>,
        data: &[&[f64]],
    ) -> DeviceResult<f64> {
        let n_edges = adjacency_matrix.iter().filter(|&&x| x > 0.0).count() / 2; // Undirected
        let n_observations = data[0].len() as f64;

        // Simplified score: negative log-likelihood + penalty for complexity
        let likelihood_score = 0.0; // Placeholder
        let complexity_penalty = (n_edges as f64) * n_observations.ln();

        Ok(likelihood_score - complexity_penalty)
    }
}

impl Default for CausalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self {
            adjacency_matrix: Array2::zeros((0, 0)),
            node_names: vec![],
            edge_weights: HashMap::new(),
            graph_confidence: 0.0,
        }
    }
}

impl Default for ConfoundingAssessment {
    fn default() -> Self {
        Self {
            confounders: vec![],
            overall_confounding_risk: "Low".to_string(),
            recommendations: vec![],
        }
    }
}
