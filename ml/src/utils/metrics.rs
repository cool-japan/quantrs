//! Evaluation metrics for QML models

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

use super::*;
/// Calculate classification accuracy
pub fn accuracy(predictions: &Array1<usize>, labels: &Array1<usize>) -> f64 {
    if predictions.len() != labels.len() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(pred, label)| pred == label)
        .count();
    correct as f64 / labels.len() as f64
}
/// Calculate mean squared error
pub fn mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    if predictions.len() != targets.len() {
        return f64::NAN;
    }
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}
/// Calculate mean absolute error
pub fn mae(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    if predictions.len() != targets.len() {
        return f64::NAN;
    }
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).abs())
        .sum::<f64>()
        / predictions.len() as f64
}
/// Calculate root mean squared error
pub fn rmse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    mse(predictions, targets).sqrt()
}
/// Calculate R-squared (coefficient of determination)
pub fn r2_score(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return f64::NAN;
    }
    let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (target - pred).powi(2))
        .sum();
    let ss_tot: f64 = targets
        .iter()
        .map(|target| (target - mean_target).powi(2))
        .sum();
    if ss_tot < 1e-10 {
        return 0.0;
    }
    1.0 - (ss_res / ss_tot)
}
/// Compute confusion matrix for multi-class classification
pub fn confusion_matrix(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<Array2<usize>> {
    if predictions.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Predictions and labels must have same length".to_string(),
        ));
    }
    let mut matrix = Array2::zeros((num_classes, num_classes));
    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        if pred >= num_classes || label >= num_classes {
            return Err(MLError::InvalidInput(format!(
                "Class index out of bounds: pred={}, label={}, num_classes={}",
                pred, label, num_classes
            )));
        }
        matrix[(label, pred)] += 1;
    }
    Ok(matrix)
}
/// Calculate precision for each class
pub fn precision(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<Array1<f64>> {
    let cm = confusion_matrix(predictions, labels, num_classes)?;
    let mut precisions = Array1::zeros(num_classes);
    for class in 0..num_classes {
        let tp = cm[(class, class)] as f64;
        let predicted_positive: f64 = cm.column(class).iter().map(|&x| x as f64).sum();
        precisions[class] = if predicted_positive > 0.0 {
            tp / predicted_positive
        } else {
            0.0
        };
    }
    Ok(precisions)
}
/// Calculate recall for each class
pub fn recall(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<Array1<f64>> {
    let cm = confusion_matrix(predictions, labels, num_classes)?;
    let mut recalls = Array1::zeros(num_classes);
    for class in 0..num_classes {
        let tp = cm[(class, class)] as f64;
        let actual_positive: f64 = cm.row(class).iter().map(|&x| x as f64).sum();
        recalls[class] = if actual_positive > 0.0 {
            tp / actual_positive
        } else {
            0.0
        };
    }
    Ok(recalls)
}
/// Calculate F1 score for each class
pub fn f1_score(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<Array1<f64>> {
    let prec = precision(predictions, labels, num_classes)?;
    let rec = recall(predictions, labels, num_classes)?;
    let mut f1_scores = Array1::zeros(num_classes);
    for class in 0..num_classes {
        let p = prec[class];
        let r = rec[class];
        f1_scores[class] = if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        };
    }
    Ok(f1_scores)
}
/// Calculate macro-averaged F1 score
pub fn f1_macro(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<f64> {
    let f1_scores = f1_score(predictions, labels, num_classes)?;
    Ok(f1_scores.iter().sum::<f64>() / num_classes as f64)
}
/// Calculate weighted F1 score
pub fn f1_weighted(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<f64> {
    let f1_scores = f1_score(predictions, labels, num_classes)?;
    let mut class_counts = vec![0usize; num_classes];
    for &label in labels.iter() {
        if label < num_classes {
            class_counts[label] += 1;
        }
    }
    let total = labels.len() as f64;
    let mut weighted_sum = 0.0;
    for class in 0..num_classes {
        let weight = class_counts[class] as f64 / total;
        weighted_sum += f1_scores[class] * weight;
    }
    Ok(weighted_sum)
}
/// Calculate ROC curve points for binary classification
/// Returns (false_positive_rates, true_positive_rates, thresholds)
pub fn roc_curve(
    scores: &Array1<f64>,
    labels: &Array1<usize>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    if scores.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Scores and labels must have same length".to_string(),
        ));
    }
    let mut indexed: Vec<(f64, usize)> =
        scores.iter().cloned().zip(labels.iter().cloned()).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_positive = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_negative = labels.len() as f64 - n_positive;
    if n_positive == 0.0 || n_negative == 0.0 {
        return Err(MLError::InvalidInput(
            "Both positive and negative samples required for ROC".to_string(),
        ));
    }
    let mut fprs = vec![0.0];
    let mut tprs = vec![0.0];
    let mut thresholds = vec![indexed[0].0 + 1.0];
    let mut tp = 0.0;
    let mut fp = 0.0;
    for (score, label) in indexed {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        fprs.push(fp / n_negative);
        tprs.push(tp / n_positive);
        thresholds.push(score);
    }
    Ok((
        Array1::from_vec(fprs),
        Array1::from_vec(tprs),
        Array1::from_vec(thresholds),
    ))
}
/// Calculate Area Under ROC Curve (AUC-ROC)
pub fn auc_roc(scores: &Array1<f64>, labels: &Array1<usize>) -> Result<f64> {
    let (fprs, tprs, _) = roc_curve(scores, labels)?;
    let mut auc = 0.0;
    for i in 1..fprs.len() {
        let width = fprs[i] - fprs[i - 1];
        let height = (tprs[i] + tprs[i - 1]) / 2.0;
        auc += width * height;
    }
    Ok(auc)
}
/// Log loss (cross-entropy) for binary classification
pub fn log_loss(probabilities: &Array1<f64>, labels: &Array1<usize>) -> f64 {
    if probabilities.len() != labels.len() {
        return f64::NAN;
    }
    let eps = 1e-15;
    let n = labels.len() as f64;
    let loss: f64 = probabilities
        .iter()
        .zip(labels.iter())
        .map(|(&prob, &label)| {
            let p = prob.clamp(eps, 1.0 - eps);
            if label == 1 {
                -p.ln()
            } else {
                -(1.0 - p).ln()
            }
        })
        .sum();
    loss / n
}
/// Matthews Correlation Coefficient for binary classification
pub fn matthews_corrcoef(predictions: &Array1<usize>, labels: &Array1<usize>) -> f64 {
    if predictions.len() != labels.len() {
        return f64::NAN;
    }
    let mut tp = 0.0;
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        match (pred, label) {
            (1, 1) => tp += 1.0,
            (0, 0) => tn += 1.0,
            (1, 0) => fp += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }
    let numerator = tp * tn - fp * fn_;
    let product: f64 = (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_);
    let denominator = product.sqrt();
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}
/// Cohen's Kappa score
pub fn cohens_kappa(predictions: &Array1<usize>, labels: &Array1<usize>) -> f64 {
    if predictions.len() != labels.len() || predictions.is_empty() {
        return f64::NAN;
    }
    let n = predictions.len() as f64;
    let observed_agreement = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count() as f64
        / n;
    let max_class = predictions.iter().chain(labels.iter()).max().unwrap_or(&0) + 1;
    let mut expected_agreement = 0.0;
    for class in 0..max_class {
        let pred_freq = predictions.iter().filter(|&&p| p == class).count() as f64 / n;
        let label_freq = labels.iter().filter(|&&l| l == class).count() as f64 / n;
        expected_agreement += pred_freq * label_freq;
    }
    if (1.0 - expected_agreement).abs() < 1e-10 {
        1.0
    } else {
        (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    }
}
/// Calculate multi-class ROC AUC using One-vs-Rest (OvR) strategy
/// Returns AUC for each class
pub fn auc_roc_ovr(
    scores: &Array2<f64>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<Array1<f64>> {
    if scores.nrows() != labels.len() {
        return Err(MLError::InvalidInput(
            "Scores and labels must have same number of samples".to_string(),
        ));
    }
    if scores.ncols() != num_classes {
        return Err(MLError::InvalidInput(format!(
            "Scores must have {} columns for {} classes",
            num_classes, num_classes
        )));
    }
    let mut aucs = Array1::zeros(num_classes);
    for class in 0..num_classes {
        let class_scores: Array1<f64> = scores.column(class).to_owned();
        let binary_labels: Array1<usize> = labels
            .iter()
            .map(|&l| if l == class { 1 } else { 0 })
            .collect();
        let n_positive = binary_labels.iter().filter(|&&l| l == 1).count();
        let n_negative = binary_labels.len() - n_positive;
        if n_positive == 0 || n_negative == 0 {
            aucs[class] = f64::NAN;
            continue;
        }
        match auc_roc(&class_scores, &binary_labels) {
            Ok(auc) => aucs[class] = auc,
            Err(_) => aucs[class] = f64::NAN,
        }
    }
    Ok(aucs)
}
/// Calculate macro-averaged multi-class ROC AUC
pub fn auc_roc_macro(
    scores: &Array2<f64>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<f64> {
    let aucs = auc_roc_ovr(scores, labels, num_classes)?;
    let valid_aucs: Vec<f64> = aucs.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    if valid_aucs.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(valid_aucs.iter().sum::<f64>() / valid_aucs.len() as f64)
}
/// Calculate weighted multi-class ROC AUC
pub fn auc_roc_weighted(
    scores: &Array2<f64>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<f64> {
    let aucs = auc_roc_ovr(scores, labels, num_classes)?;
    let mut class_counts = vec![0usize; num_classes];
    for &label in labels.iter() {
        if label < num_classes {
            class_counts[label] += 1;
        }
    }
    let total = labels.len() as f64;
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for class in 0..num_classes {
        if !aucs[class].is_nan() {
            let weight = class_counts[class] as f64 / total;
            weighted_sum += aucs[class] * weight;
            total_weight += weight;
        }
    }
    if total_weight < 1e-10 {
        return Ok(f64::NAN);
    }
    Ok(weighted_sum / total_weight)
}
/// Calculate Brier score for probabilistic predictions
pub fn brier_score(probabilities: &Array1<f64>, labels: &Array1<usize>) -> f64 {
    if probabilities.len() != labels.len() {
        return f64::NAN;
    }
    let n = labels.len() as f64;
    probabilities
        .iter()
        .zip(labels.iter())
        .map(|(&prob, &label)| {
            let target = if label == 1 { 1.0 } else { 0.0 };
            (prob - target).powi(2)
        })
        .sum::<f64>()
        / n
}
/// Calculate balanced accuracy (average of recall for each class)
pub fn balanced_accuracy(
    predictions: &Array1<usize>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Result<f64> {
    let recalls = recall(predictions, labels, num_classes)?;
    Ok(recalls.iter().sum::<f64>() / num_classes as f64)
}
/// Calculate top-k accuracy for multi-class classification
pub fn top_k_accuracy(scores: &Array2<f64>, labels: &Array1<usize>, k: usize) -> Result<f64> {
    if scores.nrows() != labels.len() {
        return Err(MLError::InvalidInput(
            "Scores and labels must have same number of samples".to_string(),
        ));
    }
    if k == 0 || k > scores.ncols() {
        return Err(MLError::InvalidInput(format!(
            "k must be between 1 and {}",
            scores.ncols()
        )));
    }
    let mut correct = 0;
    for (i, &label) in labels.iter().enumerate() {
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .row(i)
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if indexed_scores[..k].iter().any(|(idx, _)| *idx == label) {
            correct += 1;
        }
    }
    Ok(correct as f64 / labels.len() as f64)
}

/// Expected Calibration Error (ECE) - measures calibration quality
/// Lower is better (0 = perfect calibration)
pub fn expected_calibration_error(
    probabilities: &Array1<f64>,
    labels: &Array1<usize>,
    n_bins: usize,
) -> Result<f64> {
    if probabilities.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    if n_bins < 2 {
        return Err(MLError::InvalidInput(
            "Number of bins must be at least 2".to_string(),
        ));
    }

    let n_samples = probabilities.len();
    let mut bins = vec![Vec::new(); n_bins];

    // Assign samples to bins
    for (i, &prob) in probabilities.iter().enumerate() {
        let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);
        bins[bin_idx].push((prob, labels[i]));
    }

    // Compute ECE
    let mut ece = 0.0;
    for bin in bins {
        if !bin.is_empty() {
            let bin_size = bin.len();
            let mean_prob: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / bin_size as f64;
            let frac_positive: f64 =
                bin.iter().map(|(_, l)| *l as f64).sum::<f64>() / bin_size as f64;

            let bin_weight = bin_size as f64 / n_samples as f64;
            ece += bin_weight * (mean_prob - frac_positive).abs();
        }
    }

    Ok(ece)
}

/// Maximum Calibration Error (MCE) - worst-case calibration error across bins
/// Lower is better (0 = perfect calibration)
pub fn maximum_calibration_error(
    probabilities: &Array1<f64>,
    labels: &Array1<usize>,
    n_bins: usize,
) -> Result<f64> {
    if probabilities.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    if n_bins < 2 {
        return Err(MLError::InvalidInput(
            "Number of bins must be at least 2".to_string(),
        ));
    }

    let mut bins = vec![Vec::new(); n_bins];

    // Assign samples to bins
    for (i, &prob) in probabilities.iter().enumerate() {
        let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);
        bins[bin_idx].push((prob, labels[i]));
    }

    // Compute MCE
    let mut mce: f64 = 0.0;
    for bin in bins {
        if !bin.is_empty() {
            let bin_size = bin.len();
            let mean_prob: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / bin_size as f64;
            let frac_positive: f64 =
                bin.iter().map(|(_, l)| *l as f64).sum::<f64>() / bin_size as f64;

            let bin_error = (mean_prob - frac_positive).abs();
            mce = mce.max(bin_error);
        }
    }

    Ok(mce)
}

/// Negative Log-Likelihood (NLL) for multi-class classification
/// Lower is better
pub fn negative_log_likelihood(probabilities: &Array2<f64>, labels: &Array1<usize>) -> Result<f64> {
    if probabilities.nrows() != labels.len() {
        return Err(MLError::InvalidInput(
            "Probabilities and labels must have same number of samples".to_string(),
        ));
    }

    let mut nll = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        if label >= probabilities.ncols() {
            return Err(MLError::InvalidInput(format!(
                "Label {} out of bounds for {} classes",
                label,
                probabilities.ncols()
            )));
        }

        let prob = probabilities[(i, label)].max(1e-10); // Avoid log(0)
        nll -= prob.ln();
    }

    Ok(nll / labels.len() as f64)
}

/// Brier score decomposition into reliability, resolution, and uncertainty
/// Returns (brier_score, reliability, resolution, uncertainty)
pub fn brier_score_decomposition(
    probabilities: &Array1<f64>,
    labels: &Array1<usize>,
    n_bins: usize,
) -> Result<(f64, f64, f64, f64)> {
    if probabilities.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    let n_samples = probabilities.len();
    let overall_positive_rate: f64 =
        labels.iter().map(|&l| l as f64).sum::<f64>() / n_samples as f64;

    let mut bins = vec![Vec::new(); n_bins];

    // Assign samples to bins
    for (i, &prob) in probabilities.iter().enumerate() {
        let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);
        bins[bin_idx].push((prob, labels[i]));
    }

    let mut reliability = 0.0;
    let mut resolution = 0.0;

    for bin in bins {
        if !bin.is_empty() {
            let bin_size = bin.len();
            let bin_weight = bin_size as f64 / n_samples as f64;

            let mean_prob: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / bin_size as f64;
            let frac_positive: f64 =
                bin.iter().map(|(_, l)| *l as f64).sum::<f64>() / bin_size as f64;

            // Reliability: weighted squared difference between mean predicted and observed
            reliability += bin_weight * (mean_prob - frac_positive).powi(2);

            // Resolution: weighted squared difference between bin accuracy and overall base rate
            resolution += bin_weight * (frac_positive - overall_positive_rate).powi(2);
        }
    }

    // Uncertainty: variance of the base rate
    let uncertainty = overall_positive_rate * (1.0 - overall_positive_rate);

    // Brier score = Reliability - Resolution + Uncertainty
    let brier_score = reliability - resolution + uncertainty;

    Ok((brier_score, reliability, resolution, uncertainty))
}

/// Confidence interval for calibration error using bootstrap
/// Returns (mean_ece, lower_bound, upper_bound)
pub fn calibration_error_confidence_interval(
    probabilities: &Array1<f64>,
    labels: &Array1<usize>,
    n_bins: usize,
    n_bootstrap: usize,
    confidence_level: f64,
) -> Result<(f64, f64, f64)> {
    if probabilities.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err(MLError::InvalidInput(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    let n_samples = probabilities.len();
    let mut bootstrap_eces = Vec::with_capacity(n_bootstrap);
    let mut rng = thread_rng();

    // Compute original ECE
    let original_ece = expected_calibration_error(probabilities, labels, n_bins)?;

    // Bootstrap resampling
    for _ in 0..n_bootstrap {
        let mut boot_probs = Array1::zeros(n_samples);
        let mut boot_labels = Array1::zeros(n_samples);

        // Resample with replacement
        for i in 0..n_samples {
            let idx = rng.gen_range(0..n_samples);
            boot_probs[i] = probabilities[idx];
            boot_labels[i] = labels[idx];
        }

        // Compute ECE for bootstrap sample
        if let Ok(boot_ece) = expected_calibration_error(&boot_probs, &boot_labels, n_bins) {
            bootstrap_eces.push(boot_ece);
        }
    }

    // Sort bootstrap ECEs
    bootstrap_eces.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute confidence interval
    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * bootstrap_eces.len() as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_eces.len() as f64) as usize;

    let lower_bound = bootstrap_eces[lower_idx.min(bootstrap_eces.len() - 1)];
    let upper_bound = bootstrap_eces[upper_idx.min(bootstrap_eces.len() - 1)];

    Ok((original_ece, lower_bound, upper_bound))
}
