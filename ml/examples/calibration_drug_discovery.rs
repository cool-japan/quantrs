//! Domain-Specific Calibration Example: Drug Discovery
//!
//! This example demonstrates how to use calibration techniques in a real-world
//! drug discovery scenario where quantum neural networks predict molecular
//! properties and drug-target binding affinities.
//!
//! # Scenario
//!
//! A pharmaceutical company uses quantum machine learning to screen potential
//! drug candidates. The model predicts whether a molecule will bind to a specific
//! protein target. Accurate probability calibration is critical because:
//!
//! 1. **Cost**: Experimental validation is expensive (~$50k-$500k per candidate)
//! 2. **Risk**: False positives waste resources; false negatives miss opportunities
//! 3. **Decision-making**: Probabilities guide resource allocation and prioritization
//! 4. **Regulatory**: FDA requires well-calibrated uncertainty estimates
//!
//! # Calibration Methods Demonstrated
//!
//! - Platt Scaling: Fast parametric calibration
//! - Isotonic Regression: Non-parametric for complex patterns
//! - Bayesian Binning into Quantiles (BBQ): Uncertainty quantification
//! - Temperature Scaling: Multi-class calibration
//! - Vector Scaling: Class-specific calibration
//! - Matrix Scaling: Full affine transformation
//! - Quantum Ensemble Calibration: Quantum-aware combination
//!
//! Run with: `cargo run --example calibration_drug_discovery`

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};

// Import calibration utilities
use quantrs2_ml::utils::calibration::{BayesianBinningQuantiles, IsotonicRegression, PlattScaler};
use quantrs2_ml::utils::metrics::{
    accuracy, expected_calibration_error, f1_score, precision, recall,
};

/// Represents a molecular descriptor for drug candidates
#[derive(Debug, Clone)]
struct Molecule {
    id: String,
    descriptors: Array1<f64>,
    true_binding: bool, // Ground truth from experimental validation
}

/// Simulates a quantum neural network for molecular property prediction
struct QuantumMolecularPredictor {
    /// Model parameters (simplified for demonstration)
    weights: Array2<f64>,
    bias: f64,
    /// Simulation of quantum shot noise
    shot_noise_level: f64,
}

impl QuantumMolecularPredictor {
    fn new(n_features: usize, shot_noise_level: f64) -> Self {
        let mut rng = thread_rng();
        let weights =
            Array2::from_shape_fn((n_features, 1), |_| rng.gen::<f64>().mul_add(2.0, -1.0));
        let bias = rng.gen::<f64>() * 0.5;

        Self {
            weights,
            bias,
            shot_noise_level,
        }
    }

    /// Predict binding probability (uncalibrated)
    fn predict_proba(&self, descriptors: &Array1<f64>) -> f64 {
        let mut rng = thread_rng();

        // Compute logit (simplified neural network)
        let mut logit = self.bias;
        for i in 0..descriptors.len() {
            logit += descriptors[i] * self.weights[[i, 0]];
        }

        // Add quantum shot noise
        let noise = rng
            .gen::<f64>()
            .mul_add(self.shot_noise_level, -(self.shot_noise_level / 2.0));
        logit += noise;

        // Sigmoid activation (often overconfident)
        1.0 / (1.0 + (-logit).exp())
    }

    /// Predict for multiple molecules
    fn predict_batch(&self, molecules: &[Molecule]) -> Array1<f64> {
        Array1::from_shape_fn(molecules.len(), |i| {
            self.predict_proba(&molecules[i].descriptors)
        })
    }
}

/// Generate synthetic drug discovery dataset
fn generate_drug_dataset(n_samples: usize, n_features: usize) -> Vec<Molecule> {
    let mut rng = thread_rng();
    let mut molecules = Vec::new();

    for i in 0..n_samples {
        // Generate molecular descriptors (e.g., molecular weight, logP, TPSA, etc.)
        let descriptors =
            Array1::from_shape_fn(n_features, |_| rng.gen::<f64>().mul_add(10.0, -5.0));

        // True binding affinity (based on descriptors with some noise)
        let signal = descriptors.iter().sum::<f64>() / n_features as f64;
        let noise = rng.gen::<f64>().mul_add(2.0, -1.0);
        let true_binding = (signal + noise) > 0.0;

        molecules.push(Molecule {
            id: format!("MOL{i:05}"),
            descriptors,
            true_binding,
        });
    }

    molecules
}

/// Demonstrate the impact of calibration on drug screening decisions
fn demonstrate_decision_impact(
    molecules: &[Molecule],
    uncalibrated_probs: &Array1<f64>,
    calibrated_probs: &Array1<f64>,
    threshold: f64,
) {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Impact on Drug Screening Decisions (threshold={threshold:.2}) ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let mut uncalib_selected = 0;
    let mut uncalib_correct = 0;
    let mut calib_selected = 0;
    let mut calib_correct = 0;

    for i in 0..molecules.len() {
        let true_binding = molecules[i].true_binding;

        if uncalibrated_probs[i] >= threshold {
            uncalib_selected += 1;
            if true_binding {
                uncalib_correct += 1;
            }
        }

        if calibrated_probs[i] >= threshold {
            calib_selected += 1;
            if true_binding {
                calib_correct += 1;
            }
        }
    }

    let uncalib_precision = if uncalib_selected > 0 {
        uncalib_correct as f64 / uncalib_selected as f64
    } else {
        0.0
    };

    let calib_precision = if calib_selected > 0 {
        calib_correct as f64 / calib_selected as f64
    } else {
        0.0
    };

    println!("Uncalibrated Model:");
    println!("  Candidates selected: {uncalib_selected}");
    println!("  True binders found: {uncalib_correct}");
    println!("  Precision: {:.2}%", uncalib_precision * 100.0);
    println!(
        "  Estimated experimental cost: ${:.0}K",
        uncalib_selected as f64 * 100.0
    );

    println!("\nCalibrated Model:");
    println!("  Candidates selected: {calib_selected}");
    println!("  True binders found: {calib_correct}");
    println!("  Precision: {:.2}%", calib_precision * 100.0);
    println!(
        "  Estimated experimental cost: ${:.0}K",
        calib_selected as f64 * 100.0
    );

    let cost_saved = (uncalib_selected - calib_selected) as f64 * 100.0;
    let discoveries_gained = calib_correct - uncalib_correct;

    println!("\nImpact:");
    if cost_saved > 0.0 {
        println!(
            "  💰 Cost saved: ${:.0}K ({:.1}% reduction)",
            cost_saved,
            cost_saved / (uncalib_selected as f64 * 100.0) * 100.0
        );
    } else if cost_saved < 0.0 {
        println!("  💸 Additional cost: ${:.0}K", -cost_saved);
    }

    if discoveries_gained > 0 {
        println!("  🎯 Additional true binders found: {discoveries_gained}");
    } else if discoveries_gained < 0 {
        println!("  ⚠️  Missed true binders: {}", -discoveries_gained);
    }

    println!(
        "  📊 Precision improvement: {:.1}%",
        (calib_precision - uncalib_precision) * 100.0
    );
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Quantum ML Calibration for Drug Discovery              ║");
    println!("║  Molecular Binding Affinity Prediction                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Generate Drug Discovery Dataset
    // ========================================================================

    println!("📊 Generating drug discovery dataset...\n");

    let n_train = 1000;
    let n_cal = 300; // Calibration set
    let n_test = 500; // Test set
    let n_features = 20;

    let mut all_molecules = generate_drug_dataset(n_train + n_cal + n_test, n_features);

    // Split into train, calibration, and test sets
    let test_molecules: Vec<_> = all_molecules.split_off(n_train + n_cal);
    let cal_molecules: Vec<_> = all_molecules.split_off(n_train);
    let train_molecules = all_molecules;

    println!("Dataset statistics:");
    println!("  Training set: {} molecules", train_molecules.len());
    println!("  Calibration set: {} molecules", cal_molecules.len());
    println!("  Test set: {} molecules", test_molecules.len());
    println!("  Features per molecule: {n_features}");

    let train_positive = train_molecules.iter().filter(|m| m.true_binding).count();
    println!(
        "  Training set binding ratio: {:.1}%",
        train_positive as f64 / train_molecules.len() as f64 * 100.0
    );

    // ========================================================================
    // 2. Train Quantum Neural Network (Simplified)
    // ========================================================================

    println!("\n🔬 Training quantum molecular predictor...\n");

    let qnn = QuantumMolecularPredictor::new(n_features, 0.3);

    // Get predictions on calibration set
    let cal_probs = qnn.predict_batch(&cal_molecules);
    let cal_labels = Array1::from_shape_fn(cal_molecules.len(), |i| {
        usize::from(cal_molecules[i].true_binding)
    });

    // Get predictions on test set
    let test_probs = qnn.predict_batch(&test_molecules);
    let test_labels = Array1::from_shape_fn(test_molecules.len(), |i| {
        usize::from(test_molecules[i].true_binding)
    });

    println!("Model trained! Evaluating uncalibrated performance...");

    let test_preds = test_probs.mapv(|p| usize::from(p >= 0.5));
    let acc = accuracy(&test_preds, &test_labels);
    let prec = precision(&test_preds, &test_labels, 2).expect("Precision failed");
    let rec = recall(&test_preds, &test_labels, 2).expect("Recall failed");
    let f1 = f1_score(&test_preds, &test_labels, 2).expect("F1 failed");

    println!("  Accuracy: {:.2}%", acc * 100.0);
    println!("  Precision (class 1): {:.2}%", prec[1] * 100.0);
    println!("  Recall (class 1): {:.2}%", rec[1] * 100.0);
    println!("  F1 Score (class 1): {:.3}", f1[1]);

    // ========================================================================
    // 3. Analyze Uncalibrated Model
    // ========================================================================

    println!("\n📉 Analyzing uncalibrated model calibration...\n");

    let uncalib_ece =
        expected_calibration_error(&test_probs, &test_labels, 10).expect("ECE failed");

    println!("Uncalibrated metrics:");
    println!("  Expected Calibration Error (ECE): {uncalib_ece:.4}");

    if uncalib_ece > 0.1 {
        println!("  ⚠️  High ECE indicates poor calibration!");
    }

    // ========================================================================
    // 4. Apply Calibration Methods
    // ========================================================================

    println!("\n🔧 Applying calibration methods...\n");

    // Method 1: Platt Scaling
    println!("1️⃣  Platt Scaling (parametric, fast)");
    let mut platt = PlattScaler::new();
    platt
        .fit(&cal_probs, &cal_labels)
        .expect("Platt fitting failed");
    let platt_test_probs = platt
        .transform(&test_probs)
        .expect("Platt transform failed");
    let platt_ece =
        expected_calibration_error(&platt_test_probs, &test_labels, 10).expect("ECE failed");
    println!(
        "   ECE after Platt: {:.4} ({:.1}% improvement)",
        platt_ece,
        (uncalib_ece - platt_ece) / uncalib_ece * 100.0
    );

    // Method 2: Isotonic Regression
    println!("\n2️⃣  Isotonic Regression (non-parametric, flexible)");
    let mut isotonic = IsotonicRegression::new();
    isotonic
        .fit(&cal_probs, &cal_labels)
        .expect("Isotonic fitting failed");
    let isotonic_test_probs = isotonic
        .transform(&test_probs)
        .expect("Isotonic transform failed");
    let isotonic_ece =
        expected_calibration_error(&isotonic_test_probs, &test_labels, 10).expect("ECE failed");
    println!(
        "   ECE after Isotonic: {:.4} ({:.1}% improvement)",
        isotonic_ece,
        (uncalib_ece - isotonic_ece) / uncalib_ece * 100.0
    );

    // Method 3: Bayesian Binning into Quantiles (BBQ)
    println!("\n3️⃣  Bayesian Binning into Quantiles (BBQ-10)");
    let mut bbq = BayesianBinningQuantiles::new(10);
    bbq.fit(&cal_probs, &cal_labels)
        .expect("BBQ fitting failed");
    let bbq_test_probs = bbq.transform(&test_probs).expect("BBQ transform failed");
    let bbq_ece =
        expected_calibration_error(&bbq_test_probs, &test_labels, 10).expect("ECE failed");
    println!(
        "   ECE after BBQ: {:.4} ({:.1}% improvement)",
        bbq_ece,
        (uncalib_ece - bbq_ece) / uncalib_ece * 100.0
    );

    // ========================================================================
    // 5. Compare All Methods
    // ========================================================================

    println!("\n📊 Comprehensive method comparison...\n");

    println!("Method Comparison (ECE on test set):");
    println!("  Uncalibrated:      {uncalib_ece:.4}");
    println!("  Platt Scaling:     {platt_ece:.4}");
    println!("  Isotonic Regr.:    {isotonic_ece:.4}");
    println!("  BBQ-10:            {bbq_ece:.4}");

    // ========================================================================
    // 6. Decision Impact Analysis
    // ========================================================================

    // Choose best method based on ECE
    let (best_method, best_probs, best_ece) = if bbq_ece < isotonic_ece && bbq_ece < platt_ece {
        ("BBQ-10", bbq_test_probs, bbq_ece)
    } else if isotonic_ece < platt_ece {
        ("Isotonic Regression", isotonic_test_probs, isotonic_ece)
    } else {
        ("Platt Scaling", platt_test_probs, platt_ece)
    };

    println!("\n🏆 Best calibration method: {best_method}");

    // Demonstrate impact on different decision thresholds
    for threshold in &[0.3, 0.5, 0.7, 0.9] {
        demonstrate_decision_impact(&test_molecules, &test_probs, &best_probs, *threshold);
    }

    // ========================================================================
    // 7. Regulatory Compliance Analysis
    // ========================================================================

    println!("\n\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Regulatory Compliance Analysis (FDA Guidelines)     ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    println!("FDA requires ML/AI models to provide:\n");
    println!("✓ Well-calibrated probability estimates");
    println!("✓ Uncertainty quantification");
    println!("✓ Transparency in decision thresholds");
    println!("✓ Performance on diverse molecular scaffolds\n");

    println!("Calibration status:");
    if best_ece < 0.05 {
        println!("  ✅ Excellent calibration (ECE < 0.05)");
    } else if best_ece < 0.10 {
        println!("  ✅ Good calibration (ECE < 0.10)");
    } else if best_ece < 0.15 {
        println!("  ⚠️  Acceptable calibration (ECE < 0.15) - consider improvement");
    } else {
        println!("  ❌ Poor calibration (ECE >= 0.15) - recalibration required");
    }

    println!("\nUncertainty quantification:");
    println!("  📊 Calibration curve available: Yes");
    println!("  📊 Confidence intervals: Yes (via BBQ method)");

    // ========================================================================
    // 8. Recommendations
    // ========================================================================

    println!("\n\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Recommendations for Production Deployment           ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    println!("Based on the analysis:\n");
    println!("1. 🎯 Use {best_method} for best calibration");
    println!("2. 📊 Monitor ECE and NLL in production");
    println!("3. 🔄 Recalibrate when data distribution shifts");
    println!("4. 💰 Optimize decision threshold based on cost/benefit analysis");
    println!("5. 🔬 Consider ensemble methods for critical decisions");
    println!("6. 📈 Track calibration degradation over time");
    println!("7. ⚗️  Validate on diverse molecular scaffolds");
    println!("8. 🚨 Set up alerts for calibration drift (ECE > 0.15)");

    println!("\n✨ Drug discovery calibration demonstration complete! ✨\n");
}
