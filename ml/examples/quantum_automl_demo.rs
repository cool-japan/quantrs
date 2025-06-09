//! Quantum AutoML Framework Demonstration
//!
//! This example demonstrates the comprehensive quantum automated machine learning
//! framework capabilities, including automated model selection, hyperparameter
//! optimization, preprocessing pipelines, and ensemble construction.

use quantrs2_ml::prelude::*;
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Quantum AutoML Framework Demo");
    
    // Create default AutoML configuration
    println!("\n📋 Creating AutoML configuration...");
    let config = create_default_automl_config();
    println!("Configuration created with {} algorithms in search space", 
             config.search_space.algorithms.len());
    
    // Initialize Quantum AutoML
    println!("\n🔧 Initializing Quantum AutoML...");
    let mut automl = QuantumAutoML::new(config)?;
    println!("AutoML initialized: {}", automl);
    
    // Generate synthetic dataset
    println!("\n📊 Generating synthetic dataset...");
    let n_samples = 100;
    let n_features = 4;
    
    // Create sample data (classification task)
    let mut data = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);
    
    // Simple pattern for demo: sum of features determines class
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = (i as f64 + j as f64) / 100.0;
        }
        let sum: f64 = data.row(i).sum();
        targets[i] = if sum > n_features as f64 / 2.0 { 1.0 } else { 0.0 };
    }
    
    println!("Dataset shape: {:?}", data.dim());
    println!("Target distribution: {:.2}% positive class", 
             targets.sum() / targets.len() as f64 * 100.0);
    
    // Run automated ML pipeline
    println!("\n🧠 Running automated ML pipeline...");
    println!("This will perform:");
    println!("  • Automated task detection");
    println!("  • Data preprocessing and feature engineering");
    println!("  • Model selection and architecture search");
    println!("  • Hyperparameter optimization");
    println!("  • Ensemble construction");
    println!("  • Quantum advantage analysis");
    
    match automl.fit(&data, Some(&targets)) {
        Ok(results) => {
            println!("\n✅ AutoML pipeline completed successfully!");
            
            // Display results
            println!("\n📈 Results Summary:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            
            println!("🎯 Best Model: {:?}", results.best_model);
            println!("📊 Primary Metric: {:.4}", results.performance_metrics.primary_metric_value);
            println!("⏱️  Training Time: {:.2}s", results.performance_metrics.training_time);
            println!("🔢 Models Evaluated: {}", results.resource_usage.models_evaluated);
            
            // Quantum advantage analysis
            println!("\n🔬 Quantum Advantage Analysis:");
            println!("  Advantage Detected: {}", results.quantum_advantage_analysis.advantage_detected);
            println!("  Advantage Magnitude: {:.2}x", results.quantum_advantage_analysis.advantage_magnitude);
            println!("  Statistical Significance: {:.2}%", 
                     results.quantum_advantage_analysis.statistical_significance * 100.0);
            
            // Resource efficiency
            let efficiency = &results.quantum_advantage_analysis.resource_efficiency;
            println!("  Performance per Qubit: {:.4}", efficiency.performance_per_qubit);
            println!("  Quantum Resource Utilization: {:.2}%", 
                     efficiency.quantum_resource_utilization * 100.0);
            
            // Search history
            println!("\n📜 Search History:");
            for (i, iteration) in results.search_history.iter().take(5).enumerate() {
                println!("  Iteration {}: Algorithm={:?}, Performance={:.4}", 
                         iteration.iteration, 
                         iteration.configuration.algorithm,
                         iteration.performance);
            }
            if results.search_history.len() > 5 {
                println!("  ... and {} more iterations", results.search_history.len() - 5);
            }
            
            // Ensemble results
            if let Some(ensemble) = &results.ensemble_results {
                println!("\n🎭 Ensemble Results:");
                println!("  Individual Model Performances: {:?}", 
                         ensemble.individual_performances.iter()
                                 .map(|x| format!("{:.3}", x))
                                 .collect::<Vec<_>>());
                println!("  Ensemble Performance: {:.4}", ensemble.ensemble_performance);
                println!("  Prediction Diversity: {:.3}", ensemble.diversity_metrics.prediction_diversity);
                println!("  Quantum Diversity: {:.3}", ensemble.diversity_metrics.quantum_diversity);
            }
            
            // Resource usage
            println!("\n💻 Resource Usage:");
            println!("  Total Time: {:.1}s", results.resource_usage.total_time);
            println!("  Total Quantum Shots: {}", results.resource_usage.total_shots);
            println!("  Peak Memory: {}MB", results.resource_usage.peak_memory_mb);
            println!("  Search Efficiency: {:.2}%", 
                     results.resource_usage.efficiency_metrics.search_efficiency * 100.0);
            
            // Test prediction functionality
            println!("\n🔮 Testing prediction on new data...");
            let test_data = Array2::from_shape_vec((5, n_features), 
                (0..20).map(|x| x as f64 / 20.0).collect())?;
            
            match automl.predict(&test_data) {
                Ok(predictions) => {
                    println!("Predictions: {:?}", predictions.mapv(|x| format!("{:.2}", x)));
                }
                Err(e) => println!("Prediction failed: {}", e);
            }
            
            // Test model explanation
            println!("\n📖 Generating model explanation...");
            match automl.explain_model() {
                Ok(explanation) => {
                    println!("Selected Algorithm: {:?}", explanation.algorithm);
                    println!("Architecture: {}", explanation.architecture_summary);
                    println!("Circuit Depth: {}", explanation.quantum_circuit_analysis.circuit_depth);
                    println!("Gate Count: {}", explanation.quantum_circuit_analysis.gate_count);
                    println!("Expressibility: {:.3}", explanation.quantum_circuit_analysis.expressibility);
                }
                Err(e) => println!("Explanation generation failed: {}", e);
            }
        }
        Err(e) => {
            println!("❌ AutoML pipeline failed: {}", e);
            return Err(e.into());
        }
    }
    
    // Demonstrate comprehensive configuration
    println!("\n🚀 Comprehensive Configuration Demo:");
    let comprehensive_config = create_comprehensive_automl_config();
    println!("Comprehensive config includes:");
    println!("  • {} quantum algorithms", comprehensive_config.search_space.algorithms.len());
    println!("  • {} encoding methods", comprehensive_config.search_space.encoding_methods.len());
    println!("  • {} preprocessing methods", comprehensive_config.search_space.preprocessing_methods.len());
    println!("  • {} quantum metrics", comprehensive_config.evaluation_config.quantum_metrics.len());
    println!("  • Max {} evaluations", comprehensive_config.budget.max_evaluations);
    println!("  • Up to {} qubits allowed", comprehensive_config.search_space.architecture_constraints.max_qubits);
    
    // Task type detection demo
    println!("\n🎯 Task Type Detection Demo:");
    let automl_demo = QuantumAutoML::new(create_default_automl_config())?;
    
    // Binary classification
    let binary_targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let small_data = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])?;
    
    match automl_demo.detect_task_type(&small_data, Some(&binary_targets)) {
        Ok(task_type) => println!("  Detected task type: {:?}", task_type),
        Err(e) => println!("  Task detection failed: {}", e),
    }
    
    // Clustering (unsupervised)
    match automl_demo.detect_task_type(&small_data, None) {
        Ok(task_type) => println!("  Unsupervised task type: {:?}", task_type),
        Err(e) => println!("  Task detection failed: {}", e),
    }
    
    println!("\n🎉 Quantum AutoML demonstration completed!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_automl_demo_basic() {
        let config = create_default_automl_config();
        assert!(QuantumAutoML::new(config).is_ok());
    }
    
    #[test]
    fn test_comprehensive_config() {
        let config = create_comprehensive_automl_config();
        assert!(config.search_space.algorithms.len() >= 10);
        assert!(config.search_space.encoding_methods.len() >= 5);
    }
}