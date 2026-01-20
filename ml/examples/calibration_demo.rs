#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Comprehensive Calibration Demo
//!
//! This example demonstrates all three calibration methods available in QuantRS2-ML:
//! 1. Platt Scaling (binary classification)
//! 2. Isotonic Regression (binary classification, non-parametric)
//! 3. Temperature Scaling (multi-class classification)
//!
//! Run with: cargo run --example calibration_demo

use quantrs2_ml::utils::calibration::*;
use quantrs2_ml::utils::metrics;
use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2-ML Calibration Demo ===\n");

    // Demo 1: Platt Scaling for Binary Classification
    println!("1. PLATT SCALING (Binary Classification)");
    println!("   Purpose: Parametric calibration using logistic regression");
    println!("   Best for: Well-separated binary classification\n");

    demo_platt_scaling()?;

    println!("\n{}\n", "=".repeat(60));

    // Demo 2: Isotonic Regression for Binary Classification
    println!("2. ISOTONIC REGRESSION (Binary Classification)");
    println!("   Purpose: Non-parametric monotonic calibration");
    println!("   Best for: Non-linearly separable binary data\n");

    demo_isotonic_regression()?;

    println!("\n{}\n", "=".repeat(60));

    // Demo 3: Temperature Scaling for Multi-class
    println!("3. TEMPERATURE SCALING (Multi-class Classification)");
    println!("   Purpose: Scale logits by single temperature parameter");
    println!("   Best for: Neural network outputs, multi-class problems\n");

    demo_temperature_scaling()?;

    println!("\n{}\n", "=".repeat(60));

    // Demo 4: Calibration Curve Visualization
    println!("4. CALIBRATION CURVE ANALYSIS");
    println!("   Purpose: Visualize calibration quality (reliability diagram)\n");

    demo_calibration_curve()?;

    println!("\n=== Demo Complete ===");
    println!("All calibration methods demonstrated successfully!");

    Ok(())
}

fn demo_platt_scaling() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic binary classification scores
    // Positive class: higher scores, Negative class: lower scores
    let scores = array![
        2.5, 2.0, 1.8, 1.5, 1.2, // Positive class (overconfident)
        -1.2, -1.5, -1.8, -2.0, -2.5 // Negative class (overconfident)
    ];
    let labels = array![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

    println!("   Input scores: {scores:?}");
    println!("   True labels:  {labels:?}\n");

    // Fit Platt scaler
    let mut scaler = PlattScaler::new();
    scaler.fit(&scores, &labels)?;

    // Get fitted parameters
    if let Some((a, b)) = scaler.parameters() {
        println!("   Fitted parameters:");
        println!("   - Slope (a):     {a:.4}");
        println!("   - Intercept (b): {b:.4}");
    }

    // Transform scores to calibrated probabilities
    let calibrated_probs = scaler.transform(&scores)?;
    println!("\n   Calibrated probabilities:");
    for (i, (&score, &prob)) in scores.iter().zip(calibrated_probs.iter()).enumerate() {
        println!("   Sample {i}: score={score:6.2} → P(class=1)={prob:.4}");
    }

    // Compute accuracy on predictions
    let predictions: Array1<usize> = calibrated_probs.mapv(|p| usize::from(p > 0.5));
    let accuracy = metrics::accuracy(&predictions, &labels);
    println!("\n   Calibrated accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn demo_isotonic_regression() -> Result<(), Box<dyn std::error::Error>> {
    // Generate non-linearly separable scores
    let scores = array![
        0.1, 0.25, 0.2, // Low scores
        0.4, 0.35, 0.55, // Mid-low scores
        0.6, 0.75, 0.7, // Mid-high scores
        0.85, 0.95, 0.9 // High scores
    ];
    // Non-linear relationship with labels
    let labels = array![0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1];

    println!("   Input scores: {scores:?}");
    println!("   True labels:  {labels:?}\n");

    // Fit isotonic regression
    let mut iso = IsotonicRegression::new();
    iso.fit(&scores, &labels)?;

    println!("   Fitted isotonic regression (maintains monotonicity)");

    // Transform scores
    let calibrated_probs = iso.transform(&scores)?;
    println!("\n   Calibrated probabilities:");
    for (i, (&score, &prob)) in scores.iter().zip(calibrated_probs.iter()).enumerate() {
        println!("   Sample {i}: score={score:.2} → P(class=1)={prob:.4}");
    }

    // Verify monotonicity
    let mut is_monotonic = true;
    for i in 0..calibrated_probs.len() - 1 {
        if calibrated_probs[i] > calibrated_probs[i + 1] + 1e-6 {
            is_monotonic = false;
            break;
        }
    }
    println!(
        "\n   Monotonicity preserved: {}",
        if is_monotonic { "✓" } else { "✗" }
    );

    Ok(())
}

fn demo_temperature_scaling() -> Result<(), Box<dyn std::error::Error>> {
    // Generate multi-class logits (4 classes, 8 samples)
    let logits = array![
        [5.0, 1.0, 0.5, 0.0], // Overconfident for class 0
        [1.0, 5.0, 0.5, 0.0], // Overconfident for class 1
        [0.5, 1.0, 5.0, 0.0], // Overconfident for class 2
        [0.0, 0.5, 1.0, 5.0], // Overconfident for class 3
        [3.0, 2.0, 1.0, 0.5], // Moderately confident for class 0
        [1.0, 3.0, 2.0, 0.5], // Moderately confident for class 1
        [0.5, 1.0, 3.0, 2.0], // Moderately confident for class 2
        [0.5, 0.5, 1.0, 3.0], // Moderately confident for class 3
    ];
    let labels = array![0, 1, 2, 3, 0, 1, 2, 3];

    println!("   Input: 4-class classification with 8 samples");
    println!("   Logits shape: {}×{}\n", logits.nrows(), logits.ncols());

    // Compute uncalibrated softmax for comparison
    let mut uncalibrated_probs = Array2::zeros((logits.nrows(), logits.ncols()));
    for i in 0..logits.nrows() {
        let max_logit = logits
            .row(i)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.row(i).iter().map(|&x| (x - max_logit).exp()).sum();
        for j in 0..logits.ncols() {
            uncalibrated_probs[(i, j)] = ((logits[(i, j)] - max_logit).exp()) / exp_sum;
        }
    }

    // Fit temperature scaler
    let mut scaler = TemperatureScaler::new();
    scaler.fit(&logits, &labels)?;

    // Get fitted temperature
    if let Some(temp) = scaler.temperature() {
        println!("   Fitted temperature: T = {temp:.4}");
        println!(
            "   Interpretation: {}",
            if temp > 1.0 {
                "Model is overconfident (T > 1 reduces confidence)"
            } else if temp < 1.0 {
                "Model is underconfident (T < 1 increases confidence)"
            } else {
                "Model is well-calibrated (T ≈ 1)"
            }
        );
    }

    // Transform to calibrated probabilities
    let calibrated_probs = scaler.transform(&logits)?;

    println!("\n   Comparison (first 4 samples):");
    println!(
        "   {:<8} | {:<20} | {:<20}",
        "Sample", "Uncalibrated Max P", "Calibrated Max P"
    );
    println!("   {}", "-".repeat(60));

    for i in 0..4 {
        let uncal_max = uncalibrated_probs
            .row(i)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let cal_max = calibrated_probs
            .row(i)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("   Sample {i:<2}  | {uncal_max:.4}               | {cal_max:.4}");
    }

    // Compute predictions
    let mut correct = 0;
    for i in 0..calibrated_probs.nrows() {
        let pred = calibrated_probs
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        if pred == labels[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / labels.len() as f64;
    println!("\n   Calibrated accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn demo_calibration_curve() -> Result<(), Box<dyn std::error::Error>> {
    // Generate predicted probabilities and true labels
    let probabilities = array![0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    let labels = array![0, 0, 0, 1, 0, 1, 1, 1, 1, 1];

    println!("   Probabilities: {probabilities:?}");
    println!("   True labels:   {labels:?}\n");

    // Compute calibration curve
    let (mean_predicted, fraction_positives) = calibration_curve(&probabilities, &labels, 5)?;

    println!("   Calibration Curve (5 bins):");
    println!(
        "   {:<5} | {:<18} | {:<20}",
        "Bin", "Mean Predicted P", "Fraction Positive"
    );
    println!("   {}", "-".repeat(60));

    for i in 0..mean_predicted.len() {
        println!(
            "   Bin {} | {:.4}              | {:.4}",
            i + 1,
            mean_predicted[i],
            fraction_positives[i]
        );
    }

    // Compute calibration error (Expected Calibration Error - ECE)
    let mut ece = 0.0;
    let mut total_samples = 0;

    // Count samples in each bin
    let n_bins = 5;
    for i in 0..probabilities.len() {
        let bin_idx = ((probabilities[i] * n_bins as f64).floor() as usize).min(n_bins - 1);
        if bin_idx < mean_predicted.len() {
            ece += (mean_predicted[bin_idx] - fraction_positives[bin_idx]).abs();
            total_samples += 1;
        }
    }

    if total_samples > 0 {
        ece /= total_samples as f64;
        println!("\n   Expected Calibration Error (ECE): {ece:.4}");
        println!(
            "   Interpretation: {}",
            if ece < 0.1 {
                "Well-calibrated (ECE < 0.1)"
            } else if ece < 0.2 {
                "Moderately calibrated"
            } else {
                "Poorly calibrated (ECE > 0.2)"
            }
        );
    }

    Ok(())
}
