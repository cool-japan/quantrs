//! Unit tests for quantum anomaly detection module

use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_ml::anomaly_detection::*;
use quantrs2_ml::error::Result;
use scirs2_core::random::prelude::*;

/// Create default anomaly detection configuration
fn create_default_anomaly_config() -> QuantumAnomalyConfig {
    QuantumAnomalyConfig::default()
}

/// Create comprehensive anomaly detection configuration
fn create_comprehensive_anomaly_config(domain: &str) -> Result<QuantumAnomalyConfig> {
    let mut config = QuantumAnomalyConfig::default();
    config.primary_method = match domain {
        "network_security" => AnomalyDetectionMethod::QuantumIsolationForest {
            n_estimators: 100,
            max_samples: 256,
            max_depth: None,
            quantum_splitting: true,
        },
        "financial_fraud" => AnomalyDetectionMethod::QuantumOneClassSVM {
            kernel_type: QuantumKernelType::RBF,
            nu: 0.05,
            gamma: 0.1,
        },
        "iot_monitoring" => AnomalyDetectionMethod::QuantumAutoencoder {
            encoder_layers: vec![16, 8],
            latent_dim: 4,
            decoder_layers: vec![8, 16],
            reconstruction_threshold: 0.1,
        },
        _ => AnomalyDetectionMethod::QuantumIsolationForest {
            n_estimators: 100,
            max_samples: 256,
            max_depth: None,
            quantum_splitting: true,
        },
    };
    Ok(config)
}

/// Test basic detector creation
#[test]
fn test_detector_creation() {
    let config = create_default_anomaly_config();
    let detector = QuantumAnomalyDetector::new(config);
    assert!(detector.is_ok());
}

/// Test different method configurations
#[test]
fn test_method_configurations() {
    // Test Isolation Forest
    let mut config = create_default_anomaly_config();
    config.primary_method = AnomalyDetectionMethod::QuantumIsolationForest {
        n_estimators: 50,
        max_samples: 100,
        max_depth: Some(8),
        quantum_splitting: true,
    };
    assert!(QuantumAnomalyDetector::new(config).is_ok());

    // Test Autoencoder
    let mut config = create_default_anomaly_config();
    config.primary_method = AnomalyDetectionMethod::QuantumAutoencoder {
        encoder_layers: vec![4, 2],
        latent_dim: 1,
        decoder_layers: vec![2, 4],
        reconstruction_threshold: 0.5,
    };
    assert!(QuantumAnomalyDetector::new(config).is_ok());

    // Test One-Class SVM
    let mut config = create_default_anomaly_config();
    config.primary_method = AnomalyDetectionMethod::QuantumOneClassSVM {
        kernel_type: QuantumKernelType::RBF,
        nu: 0.1,
        gamma: 0.1,
    };
    assert!(QuantumAnomalyDetector::new(config).is_ok());

    // Test LOF
    let mut config = create_default_anomaly_config();
    config.primary_method = AnomalyDetectionMethod::QuantumLOF {
        n_neighbors: 10,
        contamination: 0.1,
        quantum_distance: true,
    };
    assert!(QuantumAnomalyDetector::new(config).is_ok());
}

/// Test comprehensive configurations
#[test]
fn test_comprehensive_configs() -> Result<()> {
    let network_config = create_comprehensive_anomaly_config("network_security")?;
    assert!(QuantumAnomalyDetector::new(network_config).is_ok());

    let fraud_config = create_comprehensive_anomaly_config("financial_fraud")?;
    assert!(QuantumAnomalyDetector::new(fraud_config).is_ok());

    let iot_config = create_comprehensive_anomaly_config("iot_monitoring")?;
    assert!(QuantumAnomalyDetector::new(iot_config).is_ok());

    let default_config = create_comprehensive_anomaly_config("unknown")?;
    assert!(QuantumAnomalyDetector::new(default_config).is_ok());

    Ok(())
}

/// Create synthetic data for testing
fn create_synthetic_data(n_samples: usize, n_features: usize, noise_level: f64) -> Array2<f64> {
    let mut data = Array2::zeros((n_samples, n_features));
    let mut rng = thread_rng();

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.gen::<f64>() * noise_level;
        }
    }

    data
}

/// Test training and detection pipeline
#[test]
fn test_training_detection_pipeline() -> Result<()> {
    let config = create_default_anomaly_config();
    let mut detector = QuantumAnomalyDetector::new(config)?;

    // Create training data
    let training_data = create_synthetic_data(100, 4, 1.0);

    // Train detector
    detector.fit(&training_data)?;

    // Verify training stats
    assert!(detector.get_training_stats().is_some());
    let stats = detector.get_training_stats().unwrap();
    assert_eq!(stats.n_training_samples, 100);
    assert!(stats.training_time > 0.0);

    // Create test data
    let test_data = create_synthetic_data(20, 4, 1.0);

    // Detect anomalies
    let result = detector.detect(&test_data)?;

    // Verify results
    assert_eq!(result.anomaly_scores.len(), 20);
    assert_eq!(result.anomaly_labels.len(), 20);
    assert_eq!(result.confidence_scores.len(), 20);
    assert_eq!(result.feature_importance.nrows(), 20);
    assert_eq!(result.feature_importance.ncols(), 4);

    // Verify metrics are reasonable
    assert!(result.metrics.auc_roc >= 0.0 && result.metrics.auc_roc <= 1.0);
    assert!(result.metrics.precision >= 0.0 && result.metrics.precision <= 1.0);
    assert!(result.metrics.recall >= 0.0 && result.metrics.recall <= 1.0);
    assert!(result.metrics.f1_score >= 0.0 && result.metrics.f1_score <= 1.0);

    // Verify quantum metrics
    assert!(result.metrics.quantum_metrics.quantum_advantage >= 1.0);
    assert!(result.metrics.quantum_metrics.entanglement_utilization >= 0.0);
    assert!(result.metrics.quantum_metrics.entanglement_utilization <= 1.0);

    // Verify processing stats
    assert!(result.processing_stats.total_time > 0.0);
    assert!(result.processing_stats.quantum_executions > 0);
    assert!(result.processing_stats.memory_usage > 0.0);

    Ok(())
}

/// Test streaming detection
#[test]
fn test_streaming_detection() -> Result<()> {
    let mut config = create_default_anomaly_config();
    config.realtime_config = Some(RealtimeConfig {
        buffer_size: 50,
        update_frequency: 10,
        drift_detection: true,
        online_learning: true,
        max_latency_ms: 100,
    });

    let mut detector = QuantumAnomalyDetector::new(config)?;

    // Train detector
    let training_data = create_synthetic_data(100, 4, 1.0);
    detector.fit(&training_data)?;

    // Test streaming samples
    for i in 0..10 {
        let sample = Array1::from_vec(vec![0.1 * i as f64; 4]);
        let sample_2d = sample.insert_axis(scirs2_core::ndarray::Axis(0));
        let result = detector.detect(&sample_2d)?;
        let score = result.anomaly_scores[0]; // Extract first score from batch
        assert!(score >= 0.0);
    }

    Ok(())
}

/// Test data preprocessor
#[test]
fn test_data_preprocessor() -> Result<()> {
    let config = PreprocessingConfig {
        normalization: NormalizationType::ZScore,
        dimensionality_reduction: Some(DimensionalityReduction::PCA),
        feature_selection: Some(FeatureSelection::Variance),
        noise_filtering: Some(NoiseFiltering::GaussianFilter),
        missing_value_strategy: MissingValueStrategy::Mean,
    };

    let mut preprocessor = DataPreprocessor::new(config);

    // Create test data
    let data = create_synthetic_data(50, 6, 2.0);

    // Fit and transform
    let processed = preprocessor.fit_transform(&data)?;

    // Verify output dimensions (may be reduced due to feature selection)
    assert!(processed.nrows() == data.nrows());
    assert!(processed.ncols() <= data.ncols());

    // Test transform on new data
    let new_data = create_synthetic_data(10, 6, 2.0);
    let new_processed = preprocessor.transform(&new_data)?;
    assert_eq!(new_processed.ncols(), processed.ncols());

    Ok(())
}

/// Test isolation forest specifically
#[test]
fn test_isolation_forest() -> Result<()> {
    let config = create_default_anomaly_config();
    let mut forest = QuantumIsolationForest::new(config)?;

    // Create test data
    let data = create_synthetic_data(100, 4, 1.0);

    // Train forest
    forest.fit(&data)?;

    // Test detection
    let test_data = create_synthetic_data(20, 4, 1.0);
    let result = forest.detect(&test_data)?;

    // Verify forest-specific results
    assert!(result.method_results.contains_key("isolation_forest"));
    if let Some(MethodSpecificResult::IsolationForest {
        path_lengths,
        tree_depths,
    }) = result.method_results.get("isolation_forest")
    {
        assert_eq!(path_lengths.len(), 20);
        assert_eq!(tree_depths.len(), 20);
    }

    Ok(())
}

/// Test autoencoder specifically
#[test]
fn test_autoencoder() -> Result<()> {
    let mut config = create_default_anomaly_config();
    config.primary_method = AnomalyDetectionMethod::QuantumAutoencoder {
        encoder_layers: vec![4, 2],
        latent_dim: 1,
        decoder_layers: vec![2, 4],
        reconstruction_threshold: 0.5,
    };

    let mut autoencoder = QuantumAutoencoder::new(config)?;

    // Create test data
    let data = create_synthetic_data(50, 4, 1.0);

    // Train autoencoder
    autoencoder.fit(&data)?;

    // Test detection
    let test_data = create_synthetic_data(10, 4, 1.0);
    let result = autoencoder.detect(&test_data)?;

    // Verify autoencoder-specific results
    assert!(result.method_results.contains_key("autoencoder"));
    if let Some(MethodSpecificResult::Autoencoder {
        reconstruction_errors,
        latent_representations,
    }) = result.method_results.get("autoencoder")
    {
        assert_eq!(reconstruction_errors.len(), 10);
        assert_eq!(latent_representations.nrows(), 10);
        assert_eq!(latent_representations.ncols(), 1); // latent_dim
    }

    Ok(())
}

/// Test LOF specifically
#[test]
fn test_lof() -> Result<()> {
    let mut config = create_default_anomaly_config();
    config.primary_method = AnomalyDetectionMethod::QuantumLOF {
        n_neighbors: 5,
        contamination: 0.1,
        quantum_distance: true,
    };

    let mut lof = QuantumLOF::new(config)?;

    // Create test data
    let data = create_synthetic_data(50, 4, 1.0);

    // Train LOF
    lof.fit(&data)?;

    // Test detection
    let result = lof.detect(&data)?; // Use same data for simplicity

    // Verify LOF-specific results
    assert!(result.method_results.contains_key("lof"));

    Ok(())
}

/// Test invalid configurations
#[test]
fn test_invalid_configurations() {
    // Test zero qubits
    let mut config = create_default_anomaly_config();
    config.num_qubits = 0;
    assert!(QuantumAnomalyDetector::new(config).is_err());

    // Test invalid contamination
    let mut config = create_default_anomaly_config();
    config.contamination = 1.5; // > 1.0
    assert!(QuantumAnomalyDetector::new(config).is_err());

    let mut config = create_default_anomaly_config();
    config.contamination = -0.1; // < 0.0
    assert!(QuantumAnomalyDetector::new(config).is_err());
}

/// Test performance monitor
#[test]
fn test_performance_monitor() {
    let mut monitor = PerformanceMonitor::new();

    // Record some metrics
    monitor.record_latency(0.1);
    monitor.record_latency(0.2);
    monitor.record_latency(0.15);

    monitor.record_memory_usage(100.0);
    monitor.record_memory_usage(150.0);
    monitor.record_memory_usage(120.0);

    // Test calculations
    assert!((monitor.get_average_latency() - 0.15).abs() < 1e-10);
    assert_eq!(monitor.get_peak_memory_usage(), 150.0);
}

/// Test ensemble methods
#[test]
fn test_ensemble_methods() -> Result<()> {
    let mut config = create_default_anomaly_config();
    config.ensemble_methods = vec![
        AnomalyDetectionMethod::QuantumLOF {
            n_neighbors: 10,
            contamination: 0.1,
            quantum_distance: true,
        },
        AnomalyDetectionMethod::QuantumAutoencoder {
            encoder_layers: vec![4, 2],
            latent_dim: 1,
            decoder_layers: vec![2, 4],
            reconstruction_threshold: 0.5,
        },
    ];

    let mut detector = QuantumAnomalyDetector::new(config)?;

    // Create test data
    let training_data = create_synthetic_data(100, 4, 1.0);
    let test_data = create_synthetic_data(20, 4, 1.0);

    // Train and detect
    detector.fit(&training_data)?;
    let result = detector.detect(&test_data)?;

    // Should have results from multiple methods
    assert!(result.method_results.len() > 1);

    Ok(())
}

/// Test update functionality
#[test]
fn test_online_learning() -> Result<()> {
    let config = create_default_anomaly_config();
    let mut detector = QuantumAnomalyDetector::new(config)?;

    // Initial training
    let training_data = create_synthetic_data(100, 4, 1.0);
    detector.fit(&training_data)?;

    // Update with new data
    let new_data = create_synthetic_data(20, 4, 1.0);
    let labels = Some(Array1::zeros(20)); // All normal
    detector.update(&new_data, labels.as_ref())?;

    Ok(())
}
