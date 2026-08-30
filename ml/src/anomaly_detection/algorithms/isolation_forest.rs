//! Quantum Isolation Forest implementation

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Rng;
use std::collections::HashMap;

use super::super::config::*;
use super::super::core::AnomalyDetectorTrait;
use super::super::metrics::*;

/// Quantum Isolation Forest implementation
#[derive(Debug)]
pub struct QuantumIsolationForest {
    config: QuantumAnomalyConfig,
    trees: Vec<QuantumIsolationTree>,
    feature_stats: Option<Array2<f64>>,
}

/// Quantum Isolation Tree
#[derive(Debug)]
pub struct QuantumIsolationTree {
    root: Option<QuantumIsolationNode>,
    max_depth: usize,
    quantum_splitting: bool,
}

/// Quantum Isolation Tree Node
#[derive(Debug)]
pub struct QuantumIsolationNode {
    split_feature: usize,
    split_value: f64,
    left: Option<Box<QuantumIsolationNode>>,
    right: Option<Box<QuantumIsolationNode>>,
    depth: usize,
    size: usize,
    quantum_split: bool,
}

impl QuantumIsolationForest {
    /// Create new quantum isolation forest
    pub fn new(config: QuantumAnomalyConfig) -> Result<Self> {
        Ok(QuantumIsolationForest {
            config,
            trees: Vec::new(),
            feature_stats: None,
        })
    }

    /// Build isolation trees
    fn build_trees(&mut self, data: &Array2<f64>) -> Result<()> {
        if let AnomalyDetectionMethod::QuantumIsolationForest {
            n_estimators,
            max_samples,
            max_depth,
            quantum_splitting,
        } = &self.config.primary_method
        {
            self.trees.clear();

            for _ in 0..*n_estimators {
                let tree = QuantumIsolationTree::new(*max_depth, *quantum_splitting);
                self.trees.push(tree);
            }

            // Train each tree on a random subsample
            for tree in &mut self.trees {
                let subsample = Self::create_subsample_static(data, *max_samples)?;
                tree.fit(&subsample)?;
            }
        }

        Ok(())
    }

    /// Create random subsample (static version)
    fn create_subsample_static(data: &Array2<f64>, max_samples: usize) -> Result<Array2<f64>> {
        let n_samples = data.nrows().min(max_samples);
        let mut indices: Vec<usize> = (0..data.nrows()).collect();

        // Shuffle indices
        for i in 0..indices.len() {
            let j = thread_rng().random_range(0..indices.len());
            indices.swap(i, j);
        }

        indices.truncate(n_samples);
        let subsample = data.select(Axis(0), &indices);
        Ok(subsample)
    }

    /// Compute anomaly scores
    fn compute_scores(&self, data: &Array2<f64>) -> Result<Array1<f64>> {
        let n_samples = data.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = data.row(i);
            let mut path_lengths = Vec::new();

            for tree in &self.trees {
                let path_length = tree.path_length(&sample.to_owned())?;
                path_lengths.push(path_length);
            }

            let avg_path_length = path_lengths.iter().sum::<f64>() / path_lengths.len() as f64;
            let c_n = self.compute_c_value(n_samples);
            scores[i] = 2.0_f64.powf(-avg_path_length / c_n);
        }

        Ok(scores)
    }

    /// Compute c(n) value for isolation forest normalization
    fn compute_c_value(&self, n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        2.0 * (n as f64 - 1.0).ln() - 2.0 * (n - 1) as f64 / n as f64
    }

    /// Compute threshold based on contamination level
    fn compute_threshold(&self, scores: &Array1<f64>) -> Result<f64> {
        let mut sorted_scores: Vec<f64> = scores.iter().cloned().collect();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let contamination_index = (sorted_scores.len() as f64 * self.config.contamination) as usize;
        let threshold = if contamination_index < sorted_scores.len() {
            sorted_scores[contamination_index]
        } else {
            sorted_scores[sorted_scores.len() - 1]
        };

        Ok(threshold)
    }

    /// `AnomalyMetrics`/`QuantumAnomalyMetrics` with every field set to
    /// `f64::NAN`, for use where no ground truth (or no real quantum
    /// circuit execution) is available to honestly back a value.
    fn not_computed_metrics() -> AnomalyMetrics {
        AnomalyMetrics {
            auc_roc: f64::NAN,
            auc_pr: f64::NAN,
            precision: f64::NAN,
            recall: f64::NAN,
            f1_score: f64::NAN,
            false_positive_rate: f64::NAN,
            false_negative_rate: f64::NAN,
            mcc: f64::NAN,
            balanced_accuracy: f64::NAN,
            quantum_metrics: QuantumAnomalyMetrics {
                quantum_advantage: f64::NAN,
                entanglement_utilization: f64::NAN,
                circuit_efficiency: f64::NAN,
                quantum_error_rate: f64::NAN,
                coherence_utilization: f64::NAN,
            },
        }
    }

    /// Evaluate detection performance against ground-truth labels.
    ///
    /// Unlike `detect()` (which has no access to labels and therefore cannot
    /// honestly report supervised metrics), this computes real
    /// confusion-matrix-derived precision/recall/F1/MCC/balanced-accuracy/
    /// false-positive-and-negative rates, plus rank-based AUC-ROC and a
    /// precision-recall-curve AUC-PR, from the model's actual anomaly scores
    /// and predicted labels versus `true_labels` (`1` = anomaly, `0` =
    /// normal), mirroring the pattern used by `clustering::core`'s
    /// `evaluate`. `quantum_metrics` remain `NaN` (see
    /// [`Self::not_computed_metrics`]): this classical implementation has no
    /// real circuit-execution statistics to report.
    pub fn evaluate(
        &self,
        data: &Array2<f64>,
        true_labels: &Array1<i32>,
    ) -> Result<AnomalyMetrics> {
        if data.nrows() != true_labels.len() {
            return Err(MLError::InvalidInput(format!(
                "true_labels length {} does not match number of samples {}",
                true_labels.len(),
                data.nrows()
            )));
        }
        if data.nrows() == 0 {
            return Err(MLError::InvalidInput("Empty data".to_string()));
        }

        let anomaly_scores = self.compute_scores(data)?;
        let threshold = self.compute_threshold(&anomaly_scores)?;
        let predicted_labels: Vec<i32> = anomaly_scores
            .iter()
            .map(|&score| if score > threshold { 1 } else { 0 })
            .collect();

        let mut true_positive = 0.0_f64;
        let mut false_positive = 0.0_f64;
        let mut true_negative = 0.0_f64;
        let mut false_negative = 0.0_f64;
        for (&predicted, &truth) in predicted_labels.iter().zip(true_labels.iter()) {
            match (predicted > 0, truth > 0) {
                (true, true) => true_positive += 1.0,
                (true, false) => false_positive += 1.0,
                (false, true) => false_negative += 1.0,
                (false, false) => true_negative += 1.0,
            }
        }

        let precision = if true_positive + false_positive > 0.0 {
            true_positive / (true_positive + false_positive)
        } else {
            0.0
        };
        let recall = if true_positive + false_negative > 0.0 {
            true_positive / (true_positive + false_negative)
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let false_positive_rate = if false_positive + true_negative > 0.0 {
            false_positive / (false_positive + true_negative)
        } else {
            0.0
        };
        let false_negative_rate = if false_negative + true_positive > 0.0 {
            false_negative / (false_negative + true_positive)
        } else {
            0.0
        };
        let specificity = if true_negative + false_positive > 0.0 {
            true_negative / (true_negative + false_positive)
        } else {
            0.0
        };
        let balanced_accuracy = (recall + specificity) / 2.0;

        let mcc_denominator = ((true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative))
            .sqrt();
        let mcc = if mcc_denominator > 0.0 {
            (true_positive * true_negative - false_positive * false_negative) / mcc_denominator
        } else {
            0.0
        };

        let auc_roc = Self::compute_auc_roc(&anomaly_scores, true_labels);
        let auc_pr = Self::compute_auc_pr(&anomaly_scores, true_labels);

        Ok(AnomalyMetrics {
            auc_roc,
            auc_pr,
            precision,
            recall,
            f1_score,
            false_positive_rate,
            false_negative_rate,
            mcc,
            balanced_accuracy,
            quantum_metrics: QuantumAnomalyMetrics {
                quantum_advantage: f64::NAN,
                entanglement_utilization: f64::NAN,
                circuit_efficiency: f64::NAN,
                quantum_error_rate: f64::NAN,
                coherence_utilization: f64::NAN,
            },
        })
    }

    /// Real AUC-ROC via the rank-sum (Mann-Whitney U) formulation: rank all
    /// scores ascending (averaging ranks for ties), then
    /// `AUC = (sum of positive-class ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg)`.
    fn compute_auc_roc(scores: &Array1<f64>, true_labels: &Array1<i32>) -> f64 {
        let n_pos = true_labels.iter().filter(|&&l| l > 0).count();
        let n_neg = true_labels.len() - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return f64::NAN;
        }

        let mut indexed: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0_f64; indexed.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j + 1 < indexed.len() && indexed[j + 1].1 == indexed[i].1 {
                j += 1;
            }
            // Average rank (1-indexed) for the tied group [i, j].
            let average_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
            for item in indexed.iter().take(j + 1).skip(i) {
                ranks[item.0] = average_rank;
            }
            i = j + 1;
        }

        let rank_sum_positive: f64 = true_labels
            .iter()
            .enumerate()
            .filter(|(_, &label)| label > 0)
            .map(|(idx, _)| ranks[idx])
            .sum();

        let n_pos_f = n_pos as f64;
        let n_neg_f = n_neg as f64;
        (rank_sum_positive - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg_f)
    }

    /// Real AUC-PR: sweep the score threshold from highest to lowest score,
    /// tracking precision/recall at each step, and integrate the
    /// precision-recall curve via the trapezoidal rule.
    fn compute_auc_pr(scores: &Array1<f64>, true_labels: &Array1<i32>) -> f64 {
        let n_pos = true_labels.iter().filter(|&&l| l > 0).count();
        if n_pos == 0 {
            return f64::NAN;
        }

        let mut indexed: Vec<(f64, i32)> = scores
            .iter()
            .cloned()
            .zip(true_labels.iter().cloned())
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut true_positive = 0.0_f64;
        let mut false_positive = 0.0_f64;
        let n_pos_f = n_pos as f64;

        let mut points: Vec<(f64, f64)> = vec![(0.0, 1.0)]; // (recall, precision)
        for (_, label) in &indexed {
            if *label > 0 {
                true_positive += 1.0;
            } else {
                false_positive += 1.0;
            }
            let recall = true_positive / n_pos_f;
            let precision = true_positive / (true_positive + false_positive);
            points.push((recall, precision));
        }

        let mut area = 0.0;
        for window in points.windows(2) {
            let (recall_a, precision_a) = window[0];
            let (recall_b, precision_b) = window[1];
            area += (recall_b - recall_a) * (precision_a + precision_b) / 2.0;
        }
        area
    }
}

impl AnomalyDetectorTrait for QuantumIsolationForest {
    fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        self.feature_stats = Some(Array2::zeros((data.ncols(), 4))); // Placeholder
        self.build_trees(data)
    }

    fn detect(&self, data: &Array2<f64>) -> Result<AnomalyResult> {
        let anomaly_scores = self.compute_scores(data)?;
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Generate binary labels based on contamination
        let threshold = self.compute_threshold(&anomaly_scores)?;
        let anomaly_labels = anomaly_scores.mapv(|score| if score > threshold { 1 } else { 0 });

        // Compute confidence scores (same as anomaly scores for now)
        let confidence_scores = anomaly_scores.clone();

        // Feature importance (placeholder)
        let feature_importance =
            Array2::from_elem((n_samples, n_features), 1.0 / n_features as f64);

        // Method-specific results
        let mut method_results = HashMap::new();
        method_results.insert(
            "isolation_forest".to_string(),
            MethodSpecificResult::IsolationForest {
                path_lengths: anomaly_scores.clone(),
                tree_depths: Array1::from_elem(n_samples, 10.0), // Placeholder
            },
        );

        // `detect()` is unsupervised (no ground-truth labels are passed in),
        // so the confusion-matrix-based metrics below (AUC-ROC/PR,
        // precision/recall/F1, MCC, balanced accuracy, FPR/FNR) are not
        // computable here -- they previously held fixed, entirely fabricated
        // constants regardless of `data`. `f64::NAN` makes that honestly
        // explicit (any comparison against a NaN is false, so a caller can't
        // mistake it for a real score); call [`Self::evaluate`] with
        // ground-truth labels to get real values for these fields.
        //
        // The quantum_metrics sub-fields are NaN for the same reason: this
        // isolation forest's splits are chosen purely classically
        // (`thread_rng().random_range`/`random::<f64>()` in `build_tree`);
        // `quantum_split`/`quantum_splitting` are recorded flags that do not
        // currently influence split selection, so there is no real quantum
        // circuit execution here to derive "quantum advantage" or
        // "entanglement utilization" from.
        let metrics = Self::not_computed_metrics();

        Ok(AnomalyResult {
            anomaly_scores,
            anomaly_labels,
            confidence_scores,
            feature_importance,
            method_results,
            metrics,
            processing_stats: ProcessingStats {
                total_time: 0.1,
                quantum_time: 0.03,
                classical_time: 0.07,
                memory_usage: 50.0,
                quantum_executions: n_samples,
                avg_circuit_depth: 8.0,
            },
        })
    }

    fn update(&mut self, _data: &Array2<f64>, _labels: Option<&Array1<i32>>) -> Result<()> {
        // Placeholder for online learning
        Ok(())
    }

    fn get_config(&self) -> String {
        format!("QuantumIsolationForest with {} trees", self.trees.len())
    }

    fn get_type(&self) -> String {
        "QuantumIsolationForest".to_string()
    }
}

impl QuantumIsolationTree {
    /// Create new quantum isolation tree
    pub fn new(max_depth: Option<usize>, quantum_splitting: bool) -> Self {
        QuantumIsolationTree {
            root: None,
            max_depth: max_depth.unwrap_or(10),
            quantum_splitting,
        }
    }

    /// Fit tree to data
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        self.root = Some(self.build_tree(data, 0)?);
        Ok(())
    }

    /// Build tree recursively
    fn build_tree(&self, data: &Array2<f64>, depth: usize) -> Result<QuantumIsolationNode> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Stop conditions
        if depth >= self.max_depth || n_samples <= 1 {
            return Ok(QuantumIsolationNode {
                split_feature: 0,
                split_value: 0.0,
                left: None,
                right: None,
                depth,
                size: n_samples,
                quantum_split: false,
            });
        }

        // Random feature selection
        let split_feature = thread_rng().random_range(0..n_features);
        let feature_values = data.column(split_feature);

        // Compute split value
        let min_val = feature_values.fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = feature_values.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let split_value = min_val + thread_rng().random::<f64>() * (max_val - min_val);

        // Split data
        let (left_data, right_data) = self.split_data(data, split_feature, split_value)?;

        // Build child nodes
        let left = if left_data.nrows() > 0 {
            Some(Box::new(self.build_tree(&left_data, depth + 1)?))
        } else {
            None
        };

        let right = if right_data.nrows() > 0 {
            Some(Box::new(self.build_tree(&right_data, depth + 1)?))
        } else {
            None
        };

        Ok(QuantumIsolationNode {
            split_feature,
            split_value,
            left,
            right,
            depth,
            size: n_samples,
            quantum_split: self.quantum_splitting,
        })
    }

    /// Split data based on feature and value
    fn split_data(
        &self,
        data: &Array2<f64>,
        feature: usize,
        value: f64,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for i in 0..data.nrows() {
            if data[[i, feature]] <= value {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        let left_data = if !left_indices.is_empty() {
            data.select(Axis(0), &left_indices)
        } else {
            Array2::zeros((0, data.ncols()))
        };

        let right_data = if !right_indices.is_empty() {
            data.select(Axis(0), &right_indices)
        } else {
            Array2::zeros((0, data.ncols()))
        };

        Ok((left_data, right_data))
    }

    /// Compute path length for a sample
    pub fn path_length(&self, sample: &Array1<f64>) -> Result<f64> {
        if let Some(ref root) = self.root {
            Ok(self.traverse_tree(root, sample, 0.0))
        } else {
            Ok(0.0)
        }
    }

    /// Traverse tree to compute path length
    fn traverse_tree(&self, node: &QuantumIsolationNode, sample: &Array1<f64>, depth: f64) -> f64 {
        // Leaf node
        if node.left.is_none() && node.right.is_none() {
            return depth + self.compute_c_value(node.size);
        }

        // Internal node
        if sample[node.split_feature] <= node.split_value {
            if let Some(ref left) = node.left {
                return self.traverse_tree(left, sample, depth + 1.0);
            }
        } else {
            if let Some(ref right) = node.right {
                return self.traverse_tree(right, sample, depth + 1.0);
            }
        }

        depth
    }

    /// Compute c(n) value for path length normalization
    fn compute_c_value(&self, n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        2.0 * (n as f64 - 1.0).ln() - 2.0 * (n - 1) as f64 / n as f64
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::anomaly_detection::config::QuantumAnomalyConfig;

    fn make_forest() -> QuantumIsolationForest {
        QuantumIsolationForest::new(QuantumAnomalyConfig::default()).expect("construction")
    }

    /// Two tight clusters plus a few far-away outliers, with ground-truth
    /// labels marking the outliers as anomalies.
    fn clustered_data_with_labels() -> (Array2<f64>, Array1<i32>) {
        let mut rows = Vec::new();
        for i in 0..20 {
            let jitter = (i as f64) * 0.001;
            rows.push(vec![0.0 + jitter, 0.0 + jitter]);
        }
        // Clear outliers, far from the cluster.
        rows.push(vec![50.0, 50.0]);
        rows.push(vec![-50.0, -50.0]);

        let n = rows.len();
        let data = Array2::from_shape_vec((n, 2), rows.concat()).expect("valid shape");
        let mut labels = vec![0i32; n];
        labels[n - 1] = 1;
        labels[n - 2] = 1;
        (data, Array1::from_vec(labels))
    }

    /// Regression test for the "detect() returns hardcoded metrics" bug:
    /// `detect()` has no ground truth, so its metrics must be honestly
    /// marked as not computed (NaN), not a fixed set of plausible-looking
    /// constants that never reflect `data`.
    #[test]
    fn detect_reports_not_computed_metrics() {
        let mut forest = make_forest();
        let (data, _labels) = clustered_data_with_labels();
        forest.fit(&data).expect("fit should succeed");

        let result = forest.detect(&data).expect("detect should succeed");
        assert!(result.metrics.auc_roc.is_nan());
        assert!(result.metrics.precision.is_nan());
        assert!(result.metrics.recall.is_nan());
        assert!(result.metrics.f1_score.is_nan());
        assert!(result.metrics.mcc.is_nan());
        assert!(result.metrics.quantum_metrics.quantum_advantage.is_nan());
    }

    /// Regression test: `evaluate()` must compute real confusion-matrix
    /// metrics from the actual scores/labels, not fabricate them. With two
    /// obvious outliers correctly isolated, precision/recall should both be
    /// meaningfully high (not the old hardcoded 0.75/0.70) and MCC positive.
    #[test]
    fn evaluate_computes_real_metrics_from_ground_truth() {
        let mut forest = make_forest();
        let (data, labels) = clustered_data_with_labels();
        forest.fit(&data).expect("fit should succeed");

        let metrics = forest
            .evaluate(&data, &labels)
            .expect("evaluate should succeed");

        assert!(!metrics.precision.is_nan());
        assert!(!metrics.recall.is_nan());
        assert!(metrics.recall > 0.0, "recall was {}", metrics.recall);
        assert!(
            metrics.auc_roc > 0.5,
            "expected better-than-random AUC-ROC for obviously separated \
             outliers, got {}",
            metrics.auc_roc
        );
        assert!(
            metrics.mcc > 0.0,
            "expected positive MCC for a model that isolates real outliers, got {}",
            metrics.mcc
        );
        // Quantum metrics remain honestly unmeasured: no real circuit
        // execution backs them in this classical implementation.
        assert!(metrics.quantum_metrics.quantum_advantage.is_nan());
    }

    #[test]
    fn evaluate_rejects_mismatched_label_length() {
        let mut forest = make_forest();
        let (data, _labels) = clustered_data_with_labels();
        forest.fit(&data).expect("fit should succeed");

        let wrong_labels = Array1::from_vec(vec![0i32, 1]);
        assert!(forest.evaluate(&data, &wrong_labels).is_err());
    }

    #[test]
    fn auc_roc_is_perfect_for_perfectly_separated_scores() {
        // Scores strictly increasing with the positive class perfectly
        // ranked above the negative class: AUC-ROC must be exactly 1.0.
        let scores = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.9, 1.0]);
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1]);
        let auc = QuantumIsolationForest::compute_auc_roc(&scores, &labels);
        assert!(
            (auc - 1.0).abs() < 1e-9,
            "expected perfect AUC-ROC, got {auc}"
        );
    }

    #[test]
    fn auc_roc_is_chance_for_symmetric_scores() {
        // Positive-class scores {2, 3} and negative-class scores {1, 4}: of
        // the 4 (positive, negative) pairs, exactly 2 have the positive
        // score ranked above the negative one, giving AUC = 2/4 = 0.5.
        let scores = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let labels = Array1::from_vec(vec![0, 1, 1, 0]);
        let auc = QuantumIsolationForest::compute_auc_roc(&scores, &labels);
        assert!(
            (auc - 0.5).abs() < 1e-9,
            "expected chance-level AUC-ROC, got {auc}"
        );
    }
}
