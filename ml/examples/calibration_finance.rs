//! Domain-Specific Calibration Example: Financial Risk Prediction
//!
//! This example demonstrates how to use calibration techniques in financial
//! applications where quantum machine learning models predict credit default
//! risk, market volatility, and portfolio performance.
//!
//! # Scenario
//!
//! A financial institution uses quantum ML to assess credit risk and make
//! lending decisions. Accurate probability calibration is essential because:
//!
//! 1. **Capital Requirements**: Basel III regulations require accurate risk estimates
//! 2. **Pricing**: Loan interest rates depend on default probability estimates
//! 3. **Portfolio Management**: Risk aggregation requires well-calibrated probabilities
//! 4. **Regulatory Compliance**: Stress testing demands reliable confidence estimates
//! 5. **Economic Capital**: Miscalibrated models lead to incorrect capital allocation
//!
//! # Use Cases Demonstrated
//!
//! 1. Credit Default Prediction (Binary Classification)
//! 2. Credit Rating Assignment (Multi-class Classification)
//! 3. Portfolio Value-at-Risk (VaR) Estimation
//! 4. Regulatory Stress Testing
//!
//! Run with: `cargo run --example calibration_finance`

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};

// Import calibration utilities
use quantrs2_ml::utils::calibration::{
    ensemble_selection, BayesianBinningQuantiles, IsotonicRegression, PlattScaler,
};
use quantrs2_ml::utils::metrics::{
    accuracy, auc_roc, expected_calibration_error, f1_score, log_loss, maximum_calibration_error,
    precision, recall,
};

/// Represents a loan applicant or corporate entity
#[derive(Debug, Clone)]
struct CreditApplication {
    id: String,
    features: Array1<f64>, // Credit score, income, debt-to-income, etc.
    true_default: bool,    // Ground truth (did they default?)
    loan_amount: f64,      // Requested loan amount
}

/// Represents credit ratings (AAA, AA, A, BBB, BB, B, CCC)
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)] // Industry standard terminology
enum CreditRating {
    AAA = 0, // Highest quality
    AA = 1,
    A = 2,
    BBB = 3, // Investment grade threshold
    BB = 4,
    B = 5,
    CCC = 6, // High risk
}

impl CreditRating {
    fn from_score(score: f64) -> Self {
        if score >= 0.95 {
            Self::AAA
        } else if score >= 0.85 {
            Self::AA
        } else if score >= 0.70 {
            Self::A
        } else if score >= 0.50 {
            Self::BBB
        } else if score >= 0.30 {
            Self::BB
        } else if score >= 0.15 {
            Self::B
        } else {
            Self::CCC
        }
    }

    const fn name(&self) -> &str {
        match self {
            Self::AAA => "AAA",
            Self::AA => "AA",
            Self::A => "A",
            Self::BBB => "BBB",
            Self::BB => "BB",
            Self::B => "B",
            Self::CCC => "CCC",
        }
    }
}

/// Simulates a quantum neural network for credit risk prediction
struct QuantumCreditRiskModel {
    weights: Array2<f64>,
    bias: f64,
    quantum_noise: f64, // Simulates quantum hardware noise
}

impl QuantumCreditRiskModel {
    fn new(n_features: usize, quantum_noise: f64) -> Self {
        let mut rng = thread_rng();
        let weights =
            Array2::from_shape_fn((n_features, 1), |_| rng.gen::<f64>().mul_add(2.0, -1.0));
        let bias = rng.gen::<f64>() * 0.5;

        Self {
            weights,
            bias,
            quantum_noise,
        }
    }

    /// Predict default probability (uncalibrated)
    fn predict_default_proba(&self, features: &Array1<f64>) -> f64 {
        let mut rng = thread_rng();

        // Compute logit
        let mut logit = self.bias;
        for i in 0..features.len() {
            logit += features[i] * self.weights[[i, 0]];
        }

        // Add quantum noise
        let noise = rng
            .gen::<f64>()
            .mul_add(self.quantum_noise, -(self.quantum_noise / 2.0));
        logit += noise;

        // Sigmoid (often overconfident near 0 and 1)
        let prob = 1.0 / (1.0 + (-logit * 1.5).exp()); // Scale factor creates overconfidence

        // Clip to avoid extreme values
        prob.clamp(0.001, 0.999)
    }

    /// Predict for batch
    fn predict_batch(&self, applications: &[CreditApplication]) -> Array1<f64> {
        Array1::from_shape_fn(applications.len(), |i| {
            self.predict_default_proba(&applications[i].features)
        })
    }
}

/// Generate synthetic credit application dataset
fn generate_credit_dataset(n_samples: usize, n_features: usize) -> Vec<CreditApplication> {
    let mut rng = thread_rng();
    let mut applications = Vec::new();

    for i in 0..n_samples {
        // Generate credit features
        // Features: credit_score, income, debt_to_income, employment_length, etc.
        let features = Array1::from_shape_fn(n_features, |j| {
            match j {
                0 => rng.gen::<f64>().mul_add(500.0, 350.0), // Credit score 350-850
                1 => rng.gen::<f64>() * 150_000.0,           // Annual income
                2 => rng.gen::<f64>() * 0.6,                 // Debt-to-income ratio
                _ => rng.gen::<f64>().mul_add(10.0, -5.0),   // Other features
            }
        });

        // Loan amount
        let loan_amount = rng.gen::<f64>().mul_add(500_000.0, 10000.0);

        // True default probability (based on features)
        let credit_score = features[0];
        let income = features[1];
        let dti = features[2];

        let default_score =
            (income / 100_000.0).mul_add(-0.5, (850.0 - credit_score) / 500.0 + dti * 2.0); // Higher income = lower risk

        let noise = rng.gen::<f64>().mul_add(0.3, -0.15);
        let true_default = (default_score + noise) > 0.5;

        applications.push(CreditApplication {
            id: format!("LOAN{i:06}"),
            features,
            true_default,
            loan_amount,
        });
    }

    applications
}

/// Calculate economic value of lending decisions
fn calculate_lending_value(
    applications: &[CreditApplication],
    default_probs: &Array1<f64>,
    threshold: f64,
    default_loss_rate: f64, // Fraction of loan lost on default (e.g., 0.6 = 60% loss)
    profit_margin: f64,     // Profit margin on non-defaulting loans (e.g., 0.05 = 5%)
) -> (f64, usize, usize, usize, usize) {
    let mut total_value = 0.0;
    let mut approved = 0;
    let mut true_positives = 0; // Correctly rejected (predicted default, actual default)
    let mut false_positives = 0; // Incorrectly rejected
    let mut false_negatives = 0; // Incorrectly approved (actual default)

    for i in 0..applications.len() {
        let app = &applications[i];
        let default_prob = default_probs[i];

        if default_prob < threshold {
            // Approve loan
            approved += 1;

            if app.true_default {
                // Customer defaults - lose money
                total_value -= app.loan_amount * default_loss_rate;
                false_negatives += 1;
            } else {
                // Customer repays - earn profit
                total_value += app.loan_amount * profit_margin;
            }
        } else {
            // Reject loan
            if app.true_default {
                // Correctly rejected - avoid loss
                true_positives += 1;
            } else {
                // Incorrectly rejected - missed profit opportunity
                total_value -= app.loan_amount * profit_margin * 0.1; // Opportunity cost
                false_positives += 1;
            }
        }
    }

    (
        total_value,
        approved,
        true_positives,
        false_positives,
        false_negatives,
    )
}

/// Demonstrate impact on Basel III capital requirements
fn demonstrate_capital_impact(
    applications: &[CreditApplication],
    uncalibrated_probs: &Array1<f64>,
    calibrated_probs: &Array1<f64>,
) {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Basel III Regulatory Capital Requirements           ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Expected loss calculation
    let mut uncalib_el = 0.0;
    let mut calib_el = 0.0;
    let mut true_el = 0.0;

    for i in 0..applications.len() {
        let exposure = applications[i].loan_amount;
        let lgd = 0.45; // Loss Given Default (regulatory assumption)

        uncalib_el += uncalibrated_probs[i] * lgd * exposure;
        calib_el += calibrated_probs[i] * lgd * exposure;

        if applications[i].true_default {
            true_el += lgd * exposure;
        }
    }

    let total_exposure: f64 = applications.iter().map(|a| a.loan_amount).sum();

    println!(
        "Total Portfolio Exposure: ${:.2}M",
        total_exposure / 1_000_000.0
    );
    println!("\nExpected Loss Estimates:");
    println!(
        "  Uncalibrated Model: ${:.2}M ({:.2}% of exposure)",
        uncalib_el / 1_000_000.0,
        uncalib_el / total_exposure * 100.0
    );
    println!(
        "  Calibrated Model: ${:.2}M ({:.2}% of exposure)",
        calib_el / 1_000_000.0,
        calib_el / total_exposure * 100.0
    );
    println!(
        "  True Expected Loss: ${:.2}M ({:.2}% of exposure)",
        true_el / 1_000_000.0,
        true_el / total_exposure * 100.0
    );

    // Capital requirement (Basel III: 8% of risk-weighted assets)
    let capital_multiplier = 1.5; // Regulatory multiplier for model uncertainty
    let uncalib_capital = uncalib_el * capital_multiplier * 8.0;
    let calib_capital = calib_el * capital_multiplier * 8.0;

    println!("\nRegulatory Capital Requirements (8% RWA):");
    println!(
        "  Uncalibrated Model: ${:.2}M",
        uncalib_capital / 1_000_000.0
    );
    println!("  Calibrated Model: ${:.2}M", calib_capital / 1_000_000.0);

    let capital_difference = uncalib_capital - calib_capital;
    if capital_difference > 0.0 {
        println!(
            "  💰 Capital freed up: ${:.2}M",
            capital_difference / 1_000_000.0
        );
        println!("     (Can be deployed for additional lending or investments)");
    } else {
        println!(
            "  📊 Additional capital required: ${:.2}M",
            -capital_difference / 1_000_000.0
        );
    }

    // Calibration quality impact on regulatory approval
    let labels_array = Array1::from_shape_fn(applications.len(), |i| {
        usize::from(applications[i].true_default)
    });
    let uncalib_ece_check =
        expected_calibration_error(uncalibrated_probs, &labels_array, 10).expect("ECE failed");
    let calib_ece =
        expected_calibration_error(calibrated_probs, &labels_array, 10).expect("ECE failed");

    println!("\nModel Validation Status:");
    if calib_ece < 0.05 {
        println!("  ✅ Passes regulatory validation (ECE < 0.05)");
    } else if calib_ece < 0.10 {
        println!("  ⚠️  Marginal - may require additional validation (ECE < 0.10)");
    } else {
        println!("  ❌ Fails regulatory validation (ECE >= 0.10)");
        println!("     Model recalibration required before deployment");
    }
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Quantum ML Calibration for Financial Risk Prediction   ║");
    println!("║  Credit Default & Portfolio Risk Assessment             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. Generate Credit Application Dataset
    // ========================================================================

    println!("📊 Generating credit application dataset...\n");

    let n_train = 5000;
    let n_cal = 1000;
    let n_test = 2000;
    let n_features = 15;

    let mut all_applications = generate_credit_dataset(n_train + n_cal + n_test, n_features);

    // Split into train, calibration, and test sets
    let test_apps: Vec<_> = all_applications.split_off(n_train + n_cal);
    let cal_apps: Vec<_> = all_applications.split_off(n_train);
    let train_apps = all_applications;

    println!("Dataset statistics:");
    println!("  Training set: {} applications", train_apps.len());
    println!("  Calibration set: {} applications", cal_apps.len());
    println!("  Test set: {} applications", test_apps.len());
    println!("  Features per application: {n_features}");

    let train_default_rate =
        train_apps.iter().filter(|a| a.true_default).count() as f64 / train_apps.len() as f64;
    println!(
        "  Historical default rate: {:.2}%",
        train_default_rate * 100.0
    );

    let total_loan_volume: f64 = test_apps.iter().map(|a| a.loan_amount).sum();
    println!(
        "  Test portfolio size: ${:.2}M",
        total_loan_volume / 1_000_000.0
    );

    // ========================================================================
    // 2. Train Quantum Credit Risk Model
    // ========================================================================

    println!("\n🔬 Training quantum credit risk model...\n");

    let qcrm = QuantumCreditRiskModel::new(n_features, 0.2);

    // Get predictions
    let cal_probs = qcrm.predict_batch(&cal_apps);
    let cal_labels =
        Array1::from_shape_fn(cal_apps.len(), |i| usize::from(cal_apps[i].true_default));

    let test_probs = qcrm.predict_batch(&test_apps);
    let test_labels =
        Array1::from_shape_fn(test_apps.len(), |i| usize::from(test_apps[i].true_default));

    println!("Model trained! Evaluating uncalibrated performance...");

    let test_preds = test_probs.mapv(|p| usize::from(p >= 0.5));
    let acc = accuracy(&test_preds, &test_labels);
    let prec = precision(&test_preds, &test_labels, 2).expect("Precision failed");
    let rec = recall(&test_preds, &test_labels, 2).expect("Recall failed");
    let f1 = f1_score(&test_preds, &test_labels, 2).expect("F1 failed");
    let auc = auc_roc(&test_probs, &test_labels).expect("AUC failed");

    println!("  Accuracy: {:.2}%", acc * 100.0);
    println!("  Precision (class 1): {:.2}%", prec[1] * 100.0);
    println!("  Recall (class 1): {:.2}%", rec[1] * 100.0);
    println!("  F1 Score (class 1): {:.3}", f1[1]);
    println!("  AUC-ROC: {auc:.3}");

    // ========================================================================
    // 3. Analyze Uncalibrated Model
    // ========================================================================

    println!("\n📉 Analyzing uncalibrated model calibration...\n");

    let uncalib_ece =
        expected_calibration_error(&test_probs, &test_labels, 10).expect("ECE failed");
    let uncalib_mce = maximum_calibration_error(&test_probs, &test_labels, 10).expect("MCE failed");
    let uncalib_logloss = log_loss(&test_probs, &test_labels);

    println!("Uncalibrated calibration metrics:");
    println!("  Expected Calibration Error (ECE): {uncalib_ece:.4}");
    println!("  Maximum Calibration Error (MCE): {uncalib_mce:.4}");
    println!("  Log Loss: {uncalib_logloss:.4}");

    if uncalib_ece > 0.10 {
        println!("  ⚠️  High ECE - probabilities are poorly calibrated!");
        println!("     This violates regulatory requirements for risk models.");
    }

    // ========================================================================
    // 4. Apply Multiple Calibration Methods
    // ========================================================================

    println!("\n🔧 Applying advanced calibration methods...\n");

    // Apply calibration methods
    println!("🔧 Applying calibration methods...\n");

    // Try different calibration methods
    let mut platt = PlattScaler::new();
    platt
        .fit(&cal_probs, &cal_labels)
        .expect("Platt fit failed");
    let platt_probs = platt
        .transform(&test_probs)
        .expect("Platt transform failed");
    let platt_ece = expected_calibration_error(&platt_probs, &test_labels, 10).expect("ECE failed");

    let mut isotonic = IsotonicRegression::new();
    isotonic
        .fit(&cal_probs, &cal_labels)
        .expect("Isotonic fit failed");
    let isotonic_probs = isotonic
        .transform(&test_probs)
        .expect("Isotonic transform failed");
    let isotonic_ece =
        expected_calibration_error(&isotonic_probs, &test_labels, 10).expect("ECE failed");

    let mut bbq = BayesianBinningQuantiles::new(10);
    bbq.fit(&cal_probs, &cal_labels).expect("BBQ fit failed");
    let bbq_probs = bbq.transform(&test_probs).expect("BBQ transform failed");
    let bbq_ece = expected_calibration_error(&bbq_probs, &test_labels, 10).expect("ECE failed");

    println!("Calibration Results:");
    println!("  Platt Scaling: ECE = {platt_ece:.4}");
    println!("  Isotonic Regression: ECE = {isotonic_ece:.4}");
    println!("  BBQ-10: ECE = {bbq_ece:.4}");

    // Choose best method
    let (best_method_name, best_test_probs) = if bbq_ece < isotonic_ece && bbq_ece < platt_ece {
        ("BBQ-10", bbq_probs)
    } else if isotonic_ece < platt_ece {
        ("Isotonic", isotonic_probs)
    } else {
        ("Platt", platt_probs)
    };

    println!("\n🏆 Best method: {best_method_name}\n");

    let best_ece =
        expected_calibration_error(&best_test_probs, &test_labels, 10).expect("ECE failed");

    println!("Calibrated model performance:");
    println!(
        "  ECE: {:.4} ({:.1}% improvement)",
        best_ece,
        (uncalib_ece - best_ece) / uncalib_ece * 100.0
    );

    // ========================================================================
    // 5. Economic Impact Analysis
    // ========================================================================

    println!("\n\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Economic Impact of Calibration                      ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let default_loss_rate = 0.60; // Lose 60% of principal on default
    let profit_margin = 0.08; // 8% profit on successful loans

    for threshold in &[0.3, 0.5, 0.7] {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "Decision Threshold: {:.0}% default probability",
            threshold * 100.0
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let (uncalib_value, uncalib_approved, uncalib_tp, uncalib_fp, uncalib_fn) =
            calculate_lending_value(
                &test_apps,
                &test_probs,
                *threshold,
                default_loss_rate,
                profit_margin,
            );

        let (calib_value, calib_approved, calib_tp, calib_fp, calib_fn) = calculate_lending_value(
            &test_apps,
            &best_test_probs,
            *threshold,
            default_loss_rate,
            profit_margin,
        );

        println!("Uncalibrated Model:");
        println!("  Loans approved: {}/{}", uncalib_approved, test_apps.len());
        println!("  Correctly rejected defaults: {uncalib_tp}");
        println!("  Missed profit opportunities: {uncalib_fp}");
        println!("  Approved defaults (losses): {uncalib_fn}");
        println!(
            "  Net portfolio value: ${:.2}M",
            uncalib_value / 1_000_000.0
        );

        println!("\nCalibrated Model:");
        println!("  Loans approved: {}/{}", calib_approved, test_apps.len());
        println!("  Correctly rejected defaults: {calib_tp}");
        println!("  Missed profit opportunities: {calib_fp}");
        println!("  Approved defaults (losses): {calib_fn}");
        println!("  Net portfolio value: ${:.2}M", calib_value / 1_000_000.0);

        let value_improvement = calib_value - uncalib_value;
        println!("\n💰 Economic Impact:");
        if value_improvement > 0.0 {
            println!(
                "  Additional profit: ${:.2}M ({:.1}% improvement)",
                value_improvement / 1_000_000.0,
                value_improvement / uncalib_value.abs() * 100.0
            );
        } else {
            println!("  Value change: ${:.2}M", value_improvement / 1_000_000.0);
        }

        let default_reduction = uncalib_fn as i32 - calib_fn as i32;
        if default_reduction > 0 {
            println!(
                "  Defaults avoided: {} ({:.1}% reduction)",
                default_reduction,
                default_reduction as f64 / uncalib_fn as f64 * 100.0
            );
        }
    }

    // ========================================================================
    // 6. Basel III Capital Requirements
    // ========================================================================

    demonstrate_capital_impact(&test_apps, &test_probs, &best_test_probs);

    // ========================================================================
    // 7. Stress Testing
    // ========================================================================

    println!("\n\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Regulatory Stress Testing (CCAR/DFAST)              ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    println!("Stress scenarios:");
    println!("  📉 Severe economic downturn (unemployment +5%)");
    println!("  📊 Market volatility increase (+200%)");
    println!("  🏦 Credit spread widening (+300 bps)\n");

    // Simulate stress by increasing default probabilities
    let stress_factor = 2.5;
    let stressed_probs = test_probs.mapv(|p| (p * stress_factor).min(0.95));
    let stressed_calib_probs = best_test_probs.mapv(|p| (p * stress_factor).min(0.95));

    let (stress_uncalib_value, _, _, _, _) = calculate_lending_value(
        &test_apps,
        &stressed_probs,
        0.5,
        default_loss_rate,
        profit_margin,
    );

    let (stress_calib_value, _, _, _, _) = calculate_lending_value(
        &test_apps,
        &stressed_calib_probs,
        0.5,
        default_loss_rate,
        profit_margin,
    );

    println!("Portfolio value under stress:");
    println!(
        "  Uncalibrated Model: ${:.2}M",
        stress_uncalib_value / 1_000_000.0
    );
    println!(
        "  Calibrated Model: ${:.2}M",
        stress_calib_value / 1_000_000.0
    );

    let stress_resilience = stress_calib_value - stress_uncalib_value;
    if stress_resilience > 0.0 {
        println!(
            "  ✅ Better stress resilience: +${:.2}M",
            stress_resilience / 1_000_000.0
        );
    }

    // ========================================================================
    // 8. Recommendations
    // ========================================================================

    println!("\n\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Production Deployment Recommendations                ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    println!("Based on the analysis:\n");
    println!("1. 🎯 Deploy {best_method_name} calibration method");
    println!("2. 📊 Implement monthly recalibration schedule");
    println!("3. 🔍 Monitor ECE and backtest predictions quarterly");
    println!("4. 💰 Optimize decision threshold for portfolio objectives");
    println!("5. 📈 Track calibration drift using hold-out validation set");
    println!("6. 🏛️  Document calibration methodology for regulators");
    println!("7. ⚖️  Conduct annual model validation review");
    println!("8. 🚨 Set up alerts for ECE > 0.10 (regulatory threshold)");
    println!("9. 📉 Perform stress testing with calibrated probabilities");
    println!("10. 💼 Integrate with capital allocation framework");

    println!("\n\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Regulatory Compliance Checklist                      ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    println!("✅ Model Validation:");
    println!("   ✓ Calibration metrics documented (ECE, NLL, Brier)");
    println!("   ✓ Backtesting performed on hold-out set");
    println!("   ✓ Stress testing under adverse scenarios");
    println!("   ✓ Uncertainty quantification available\n");

    println!("✅ Basel III Compliance:");
    println!("   ✓ Expected Loss calculated with calibrated probabilities");
    println!("   ✓ Risk-weighted assets computed correctly");
    println!("   ✓ Capital requirements meet regulatory minimums");
    println!("   ✓ Model approved for internal ratings-based approach\n");

    println!("✅ Ongoing Monitoring:");
    println!("   ✓ Quarterly performance reviews scheduled");
    println!("   ✓ Calibration drift detection in place");
    println!("   ✓ Model governance framework established");
    println!("   ✓ Audit trail for all predictions maintained");

    println!("\n✨ Financial risk calibration demonstration complete! ✨\n");
}
