//! Comprehensive Process Tomography Example
//!
//! This example demonstrates the advanced process tomography capabilities
//! of the quantrs device library with SciRS2 integration.
//!
//! NOTE: This example is currently commented out because it references types and
//! functionality that haven't been fully implemented in the process_tomography module.

fn main() {
    println!("This example is temporarily disabled due to missing types in the process_tomography module.");
}

/*
use quantrs2_device::{
    process_tomography::{
        SciRS2ProcessTomographer, SciRS2ProcessTomographyConfig, ReconstructionMethod,
        ProcessMonitoringResult, ProcessAnomalyDetector, ProcessDriftDetector,
        ExperimentalConditions, ProcessTomographyExecutor, ExperimentalData,
        AnomalyDetectionAlgorithm, DriftDetectionMethod
    },
    calibration::CalibrationManager,
    DeviceResult, DeviceError,
};
use quantrs2_circuit::prelude::*;
use scirs2_core::ndarray::{Array2, Array4};
use scirs2_core::Complex64;
use std::time::Duration;

/// Mock executor for demonstration purposes
struct MockProcessExecutor;

#[async_trait::async_trait]
impl ProcessTomographyExecutor for MockProcessExecutor {
    async fn execute_process_measurement<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        input_state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
        shots: usize,
    ) -> DeviceResult<f64> {
        // Simulate measurement with some noise
        let expectation_value = 0.5 + (rand::random::<f64>() - 0.5) * 0.1;
        let _uncertainty = 1.0 / (shots as f64).sqrt();

        Ok(expectation_value)
    }
}

#[tokio::main]
async fn main() -> DeviceResult<()> {
    println!("🔬 Comprehensive Quantum Process Tomography with SciRS2");
    println!("======================================================");

    // 1. Configure Process Tomography
    let mut config = SciRS2ProcessTomographyConfig::default();
    config.reconstruction_method = ReconstructionMethod::MaximumLikelihood;
    config.enable_compressed_sensing = true;
    config.enable_mle = true;
    config.enable_bayesian = true;
    config.enable_structure_analysis = true;

    println!("✅ Configured SciRS2 Process Tomography");
    println!("   - Reconstruction method: {:?}", config.reconstruction_method);
    println!("   - Input states: {}", config.num_input_states);
    println!("   - Shots per state: {}", config.shots_per_state);

    // 2. Initialize Tomographer
    let calibration_manager = CalibrationManager::new();
    let mut tomographer = SciRS2ProcessTomographer::new(config.clone(), calibration_manager);

    // Generate input states and measurement operators
    tomographer.generate_input_states(1)?; // Single qubit
    tomographer.generate_measurement_operators(1)?;

    println!("✅ Generated input states: {} states", tomographer.input_states.len());
    println!("✅ Generated measurement operators: {} operators", tomographer.measurement_operators.len());

    // 3. Create a simple test circuit (X gate)
    let circuit = Circuit::<1>::new();
    // In a real implementation, you would add gates to the circuit

    // 4. Perform Process Tomography
    let executor = MockProcessExecutor;
    println!("\n🧪 Performing Process Tomography...");

    let result = tomographer
        .perform_process_tomography("test_device", &circuit, &executor)
        .await?;

    println!("✅ Process tomography completed successfully!");

    // 5. Display Results
    println!("\n📊 Process Characterization Results:");
    println!("====================================");

    let metrics = &result.process_metrics;
    println!("Process Fidelity:        {:.4}", metrics.process_fidelity);
    println!("Average Gate Fidelity:   {:.4}", metrics.average_gate_fidelity);
    println!("Unitarity:               {:.4}", metrics.unitarity);
    println!("Entangling Power:        {:.4}", metrics.entangling_power);
    println!("Non-unitality:           {:.4}", metrics.non_unitality);
    println!("Channel Capacity:        {:.4}", metrics.channel_capacity);
    println!("Coherent Information:    {:.4}", metrics.coherent_information);
    println!("Diamond Norm Distance:   {:.4}", metrics.diamond_norm_distance);

    // 6. Statistical Analysis
    println!("\n📈 Statistical Analysis:");
    println!("========================");

    let stats = &result.statistical_analysis;
    println!("Reconstruction Quality:");
    println!("  - Likelihood:          {:.4}", stats.reconstruction_quality.likelihood);
    println!("  - R-squared:           {:.4}", stats.reconstruction_quality.r_squared);
    println!("  - Reconstruction Error: {:.4}", stats.reconstruction_quality.reconstruction_error);

    println!("Statistical Tests Performed: {}", stats.statistical_tests.len());
    for (test_name, test_result) in &stats.statistical_tests {
        println!("  - {}: p-value = {:.4}, significant = {}",
                 test_name, test_result.p_value, test_result.significant);
    }

    // 7. Real-time Process Monitoring Setup
    println!("\n🔍 Setting up Real-time Process Monitoring...");

    let monitoring_config = ProcessMonitoringConfig {
        monitoring_interval: 5.0, // 5 seconds
        history_length: 50,
        anomaly_threshold: 0.1,
        drift_sensitivity: 0.05,
        auto_recalibration: true,
        alert_thresholds: ProcessAlertThresholds {
            fidelity_warning: 0.9,
            fidelity_critical: 0.8,
            unitarity_warning: 0.9,
            unitarity_critical: 0.8,
            diamond_norm_warning: 0.1,
            diamond_norm_critical: 0.2,
        },
    };

    let mut monitor = ProcessMonitor::new(monitoring_config);
    println!("✅ Process monitor configured with real-time anomaly detection");

    // 8. Machine Learning Enhanced Reconstruction
    println!("\n🤖 Machine Learning Enhanced Reconstruction:");
    println!("===========================================");

    let mut ml_reconstructor = MLProcessReconstructor::new(MLModelType::NeuralNetwork);

    // Add some training data (in practice, this would be real experimental data)
    for i in 0..10 {
        let training_data = TrainingDataPoint {
            measurement_data: vec![0.1 * i as f64, 0.2, 0.3, 0.4],
            true_process_matrix: Array4::zeros((2, 2, 2, 2)),
            noise_level: 0.01,
            experimental_conditions: ExperimentalConditions {
                temperature: Some(20.0),
                noise_level: 0.01,
                calibration_age: Duration::from_secs(3600),
                gate_count: 10,
                circuit_depth: 5,
            },
        };
        ml_reconstructor.add_training_data(training_data);
    }

    // Train the model
    ml_reconstructor.train_model()?;
    println!("✅ ML model trained with {} data points", ml_reconstructor.training_data.len());

    // 9. Uncertainty Quantification
    println!("\n📏 Uncertainty Quantification:");
    println!("==============================");

    let uncertainty = &result.uncertainty_quantification;
    println!("Parameter covariance matrix shape: {:?}", uncertainty.parameter_covariance.dim());
    println!("Confidence intervals: {}", uncertainty.metric_confidence_intervals.len());
    println!("Uncertainty amplification: {:.4}", uncertainty.uncertainty_propagation.uncertainty_amplification);

    // 10. Process Comparisons
    println!("\n🔄 Process Comparisons:");
    println!("=======================");

    let comparisons = &result.process_comparisons;
    println!("Process Classification: {:?}", comparisons.classification.process_type);
    println!("Classification Confidence: {:.4}", comparisons.classification.classification_confidence);
    println!("Process distances: {}", comparisons.process_distances.len());

    // 11. Structure Analysis (if enabled)
    if let Some(structure) = &result.structure_analysis {
        println!("\n🏗️  Process Structure Analysis:");
        println!("==============================");

        let kraus = &structure.kraus_decomposition;
        println!("Kraus operators: {}", kraus.kraus_operators.len());
        println!("Minimal Kraus rank: {}", kraus.minimal_kraus_rank);
        println!("Decomposition error: {:.6}", kraus.decomposition_error);

        let noise = &structure.noise_decomposition;
        println!("Coherence ratio: {:.4}", noise.coherence_ratio);
        println!("Identified noise types: {}", noise.noise_types.len());
    }

    println!("\n🎉 Comprehensive Process Tomography Analysis Complete!");
    println!("=====================================================");

    println!("\n💡 Summary:");
    println!("   - Process successfully characterized with high fidelity");
    println!("   - Statistical analysis reveals robust reconstruction");
    println!("   - Real-time monitoring configured for continuous quality assurance");
    println!("   - ML-enhanced reconstruction provides additional validation");
    println!("   - Comprehensive uncertainty quantification ensures reliability");

    Ok(())
}
*/
