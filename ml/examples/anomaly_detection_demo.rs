#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Quantum Anomaly Detection Example
//!
//! This example demonstrates how to use the quantum anomaly detection module
//! for various types of anomaly detection tasks.

use quantrs2_ml::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> quantrs2_ml::Result<()> {
    println!("Quantum Anomaly Detection Demo");
    println!("===============================");

    // Create default configuration
    let config = QuantumAnomalyConfig::default();
    println!("Created default anomaly detection configuration");
    println!("Primary method: {:?}", config.primary_method);

    // Create quantum anomaly detector
    let mut detector = QuantumAnomalyDetector::new(config)?;
    println!("Created quantum anomaly detector");

    // Generate synthetic normal data (multivariate normal distribution)
    let n_samples = 1000;
    let n_features = 8;
    let mut normal_data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            normal_data[[i, j]] = thread_rng().gen::<f64>().mul_add(2.0, -1.0); // Normal range [-1, 1]
        }
    }

    println!("Generated {n_samples} normal samples with {n_features} features");

    // Train the detector on normal data
    println!("Training anomaly detector...");
    detector.fit(&normal_data)?;

    if let Some(stats) = detector.get_training_stats() {
        println!("Training completed in {:.3} seconds", stats.training_time);
        println!("Training samples: {}", stats.n_training_samples);
    }

    // Generate test data with some anomalies
    let n_test = 100;
    let mut test_data = Array2::zeros((n_test, n_features));

    // Normal samples (first 80)
    for i in 0..80 {
        for j in 0..n_features {
            test_data[[i, j]] = thread_rng().gen::<f64>().mul_add(2.0, -1.0);
        }
    }

    // Anomalous samples (last 20) - outliers with larger values
    for i in 80..n_test {
        for j in 0..n_features {
            test_data[[i, j]] = thread_rng().gen::<f64>().mul_add(6.0, 5.0); // Anomalous range [5, 11]
        }
    }

    println!("Generated {n_test} test samples (80 normal, 20 anomalous)");

    // Detect anomalies
    println!("Detecting anomalies...");
    let result = detector.detect(&test_data)?;

    // Display results
    println!("\nDetection Results:");
    println!("==================");
    println!(
        "Anomaly scores range: [{:.3}, {:.3}]",
        result.anomaly_scores.fold(f64::INFINITY, |a, &b| a.min(b)),
        result
            .anomaly_scores
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    let anomaly_count = result.anomaly_labels.iter().sum::<i32>();
    println!("Detected {anomaly_count} anomalies out of {n_test} samples");

    // Performance metrics
    println!("\nPerformance Metrics:");
    println!("====================");
    println!("AUC-ROC: {:.3}", result.metrics.auc_roc);
    println!("Precision: {:.3}", result.metrics.precision);
    println!("Recall: {:.3}", result.metrics.recall);
    println!("F1-Score: {:.3}", result.metrics.f1_score);

    // Quantum-specific metrics
    println!("\nQuantum Metrics:");
    println!("================");
    println!(
        "Quantum Advantage: {:.3}x",
        result.metrics.quantum_metrics.quantum_advantage
    );
    println!(
        "Entanglement Utilization: {:.1}%",
        result.metrics.quantum_metrics.entanglement_utilization * 100.0
    );
    println!(
        "Circuit Efficiency: {:.1}%",
        result.metrics.quantum_metrics.circuit_efficiency * 100.0
    );

    // Processing statistics
    println!("\nProcessing Statistics:");
    println!("======================");
    println!(
        "Total time: {:.3} seconds",
        result.processing_stats.total_time
    );
    println!(
        "Quantum time: {:.3} seconds",
        result.processing_stats.quantum_time
    );
    println!(
        "Classical time: {:.3} seconds",
        result.processing_stats.classical_time
    );
    println!(
        "Memory usage: {:.1} MB",
        result.processing_stats.memory_usage
    );
    println!(
        "Quantum executions: {}",
        result.processing_stats.quantum_executions
    );

    // Test different configurations
    println!("\n{}", "=".repeat(50));
    println!("Testing Different Configurations");
    println!("{}", "=".repeat(50));

    // Network security configuration
    let network_config = QuantumAnomalyConfig::default();
    let mut network_detector = QuantumAnomalyDetector::new(network_config)?;
    network_detector.fit(&normal_data)?;
    let network_result = network_detector.detect(&test_data)?;

    println!("\nNetwork Security Detection:");
    println!("AUC-ROC: {:.3}", network_result.metrics.auc_roc);
    println!(
        "Detected anomalies: {}",
        network_result.anomaly_labels.iter().sum::<i32>()
    );

    // Financial fraud configuration
    let fraud_config = QuantumAnomalyConfig::default();
    let mut fraud_detector = QuantumAnomalyDetector::new(fraud_config)?;
    fraud_detector.fit(&normal_data)?;
    let fraud_result = fraud_detector.detect(&test_data)?;

    println!("\nFinancial Fraud Detection:");
    println!("AUC-ROC: {:.3}", fraud_result.metrics.auc_roc);
    println!(
        "Detected anomalies: {}",
        fraud_result.anomaly_labels.iter().sum::<i32>()
    );

    // Test streaming detection
    println!("\n{}", "=".repeat(50));
    println!("Testing Streaming Detection");
    println!("{}", "=".repeat(50));

    let iot_config = QuantumAnomalyConfig::default();
    let mut streaming_detector = QuantumAnomalyDetector::new(iot_config)?;
    streaming_detector.fit(&normal_data)?;

    println!("Testing real-time streaming detection...");
    for i in 0..10 {
        let sample = test_data.row(i).to_owned();
        let sample_2d = sample.clone().insert_axis(scirs2_core::ndarray::Axis(0));
        let result = streaming_detector.detect(&sample_2d)?;
        let anomaly_score = result.anomaly_scores[0];
        let is_anomaly = if anomaly_score > 0.5 {
            "ANOMALY"
        } else {
            "normal"
        };
        println!(
            "Sample {}: score = {:.3} -> {}",
            i + 1,
            anomaly_score,
            is_anomaly
        );
    }

    println!("\nQuantum Anomaly Detection Demo completed successfully!");

    Ok(())
}
