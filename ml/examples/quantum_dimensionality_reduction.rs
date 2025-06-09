//! Quantum Dimensionality Reduction Example
//!
//! This example demonstrates various quantum dimensionality reduction algorithms
//! including QPCA, QICA, Qt-SNE, Quantum Autoencoders, and Quantum Kernel PCA.

use quantrs2_ml::prelude::*;
use quantrs2_ml::dimensionality_reduction::*;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Quantum Dimensionality Reduction Examples ===\n");

    // Generate synthetic high-dimensional data
    let (data, labels) = generate_synthetic_data(100, 10)?;
    println!("Generated synthetic data: {} samples, {} features", data.nrows(), data.ncols());

    // Example 1: Quantum PCA
    demo_qpca(&data)?;

    // Example 2: Quantum ICA
    demo_qica(&data)?;

    // Example 3: Quantum t-SNE
    demo_qtsne(&data)?;

    // Example 4: Quantum Variational Autoencoder
    demo_qvae(&data)?;

    // Example 5: Quantum Kernel PCA
    demo_qkernel_pca(&data)?;

    // Example 6: Comparison of methods
    compare_methods(&data, &labels)?;

    // Example 7: Specialized configurations
    demo_specialized_configs(&data)?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Demonstrate Quantum Principal Component Analysis
fn demo_qpca(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Quantum PCA Demo ---");

    // Create QPCA configuration
    let config = QPCAConfig {
        n_components: 3,
        eigensolver: QuantumEigensolver::VQE,
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        whiten: false,
        random_state: Some(42),
        tolerance: 1e-6,
        max_iterations: 1000,
    };

    // Create and train QPCA reducer
    let mut qpca = QuantumDimensionalityReducer::qpca(config);
    
    println!("Training QPCA...");
    let result = qpca.fit(data)?;
    
    println!("Training completed in {:.3} seconds", result.training_time);
    println!("Transformation shape: {:?}", result.transformed_data.dim());
    println!("Reconstruction error: {:.6}", result.reconstruction_error);
    
    if let Some(explained_var) = &result.explained_variance_ratio {
        println!("Explained variance ratio: {:?}", explained_var);
        println!("Total explained variance: {:.4}", explained_var.sum());
    }

    // Test transform on new data
    println!("Testing transform on original data...");
    let transformed = qpca.transform(data)?;
    println!("Transform successful, output shape: {:?}", transformed.dim());

    // Test inverse transform
    println!("Testing inverse transform...");
    let reconstructed = qpca.inverse_transform(&transformed)?;
    println!("Inverse transform successful, output shape: {:?}", reconstructed.dim());

    // Print quantum metrics
    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", result.quantum_metrics.quantum_fidelity);
    println!("  Entanglement Entropy: {:.4}", result.quantum_metrics.entanglement_entropy);
    println!("  Gate Count: {}", result.quantum_metrics.gate_count);
    println!("  Circuit Depth: {}", result.quantum_metrics.circuit_depth);

    Ok(())
}

/// Demonstrate Quantum Independent Component Analysis
fn demo_qica(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Quantum ICA Demo ---");

    // Create QICA configuration
    let config = QICAConfig {
        n_components: 3,
        max_iterations: 200,
        tolerance: 1e-4,
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        learning_rate: 1.0,
        nonlinearity: "logcosh".to_string(),
        random_state: Some(42),
    };

    // Create and train QICA reducer
    let mut qica = QuantumDimensionalityReducer::qica(config);
    
    println!("Training QICA...");
    let result = qica.fit(data)?;
    
    println!("Training completed in {:.3} seconds", result.training_time);
    println!("Transformation shape: {:?}", result.transformed_data.dim());
    println!("Reconstruction error: {:.6}", result.reconstruction_error);

    // Test transform
    let transformed = qica.transform(data)?;
    println!("Transform output shape: {:?}", transformed.dim());

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", result.quantum_metrics.quantum_fidelity);
    println!("  Entanglement Entropy: {:.4}", result.quantum_metrics.entanglement_entropy);

    Ok(())
}

/// Demonstrate Quantum t-SNE
fn demo_qtsne(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Quantum t-SNE Demo ---");

    // Create Qt-SNE configuration
    let config = QtSNEConfig {
        n_components: 2,
        perplexity: 30.0,
        early_exaggeration: 12.0,
        learning_rate: 200.0,
        max_iterations: 500, // Reduced for demo
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        distance_metric: QuantumDistanceMetric::QuantumEuclidean,
        random_state: Some(42),
    };

    // Create and train Qt-SNE reducer
    let mut qtsne = QuantumDimensionalityReducer::qtsne(config);
    
    println!("Training Qt-SNE (this may take a while)...");
    let result = qtsne.fit(data)?;
    
    println!("Training completed in {:.3} seconds", result.training_time);
    println!("Embedding shape: {:?}", result.transformed_data.dim());

    // Note: t-SNE doesn't support out-of-sample transforms
    println!("Note: t-SNE doesn't support out-of-sample transforms");

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", result.quantum_metrics.quantum_fidelity);
    println!("  Circuit Depth: {}", result.quantum_metrics.circuit_depth);

    Ok(())
}

/// Demonstrate Quantum Variational Autoencoder
fn demo_qvae(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Quantum Variational Autoencoder Demo ---");

    // Create QVAE configuration
    let config = QAutoencoderConfig {
        encoder_layers: vec![8, 6, 4],
        decoder_layers: vec![4, 6, 8],
        latent_dim: 3,
        architecture: AutoencoderArchitecture::Standard,
        learning_rate: 0.001,
        epochs: 20, // Reduced for demo
        batch_size: 16,
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        beta: 1.0,
        noise_level: 0.1,
        sparsity_parameter: 0.01,
        random_state: Some(42),
    };

    // Create and train QVAE
    let mut qvae = QuantumDimensionalityReducer::qautoencoder(config);
    
    println!("Training QVAE...");
    let result = qvae.fit(data)?;
    
    println!("Training completed in {:.3} seconds", result.training_time);
    println!("Latent representation shape: {:?}", result.transformed_data.dim());
    println!("Reconstruction error: {:.6}", result.reconstruction_error);

    // Test encoding and decoding
    let encoded = qvae.transform(data)?;
    println!("Encoding output shape: {:?}", encoded.dim());

    let decoded = qvae.inverse_transform(&encoded)?;
    println!("Decoding output shape: {:?}", decoded.dim());

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", result.quantum_metrics.quantum_fidelity);
    println!("  Gate Count: {}", result.quantum_metrics.gate_count);

    Ok(())
}

/// Demonstrate Quantum Kernel PCA
fn demo_qkernel_pca(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Quantum Kernel PCA Demo ---");

    // Create kernel parameters
    let mut kernel_params = HashMap::new();
    kernel_params.insert("gamma".to_string(), 0.1);

    // Create Quantum Kernel PCA configuration
    let config = QKernelPCAConfig {
        n_components: 3,
        feature_map: QuantumFeatureMap::ZZFeatureMap,
        feature_map_reps: 2,
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        kernel_params,
        eigensolver: QuantumEigensolver::VQE,
        remove_zero_eig: true,
        eigenvalue_tolerance: 1e-6,
    };

    // Create and train Quantum Kernel PCA
    let mut qkpca = QuantumDimensionalityReducer::qkernel_pca(config);
    
    println!("Training Quantum Kernel PCA...");
    let result = qkpca.fit(data)?;
    
    println!("Training completed in {:.3} seconds", result.training_time);
    println!("Kernel space representation shape: {:?}", result.transformed_data.dim());
    
    if let Some(explained_var) = &result.explained_variance_ratio {
        println!("Explained variance ratio: {:?}", explained_var);
    }

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", result.quantum_metrics.quantum_fidelity);
    println!("  Quantum Volume: {:.4}", result.quantum_metrics.quantum_volume);

    Ok(())
}

/// Compare different dimensionality reduction methods
fn compare_methods(data: &Array2<f64>, labels: &Array1<i32>) -> Result<()> {
    println!("\n--- Method Comparison ---");

    let methods = vec![
        ("QPCA", create_comprehensive_qpca(3, QuantumEnhancementLevel::Moderate)),
        ("Qt-SNE", create_comprehensive_qtsne(2, 20.0, QuantumEnhancementLevel::Light)),
    ];

    for (name, mut method) in methods {
        println!("\nEvaluating {}...", name);
        
        let result = method.fit(data)?;
        let transformed = &result.transformed_data;
        
        // Evaluate the reduction
        let metrics = method.evaluate(data, transformed, Some(labels))?;
        
        println!("  Reconstruction Error: {:.6}", metrics.reconstruction_error);
        println!("  Explained Variance: {:.4}", metrics.explained_variance);
        println!("  Trustworthiness: {:.4}", metrics.trustworthiness);
        println!("  Continuity: {:.4}", metrics.continuity);
        
        if let Some(silhouette) = metrics.silhouette_score {
            println!("  Silhouette Score: {:.4}", silhouette);
        }
        
        if let Some(stress) = metrics.stress {
            println!("  Stress: {:.6}", stress);
        }
        
        if let Some(kl_div) = metrics.kl_divergence {
            println!("  KL Divergence: {:.6}", kl_div);
        }
    }

    Ok(())
}

/// Demonstrate specialized configurations
fn demo_specialized_configs(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Specialized Configurations Demo ---");

    // Time series dimensionality reduction configuration
    let time_series_config = QTimeSeriesConfig {
        window_size: 10,
        overlap: 5,
        temporal_regularization: 0.1,
        consider_seasonality: true,
        trend_removal: "linear".to_string(),
    };

    // Image/tensor dimensionality reduction configuration
    let image_config = QImageTensorConfig {
        patch_size: (4, 4),
        stride: (2, 2),
        spatial_regularization: 0.05,
        channel_handling: "concatenate".to_string(),
        preserve_spatial: true,
    };

    // Graph dimensionality reduction configuration
    let graph_config = QGraphConfig {
        graph_construction: "knn".to_string(),
        n_neighbors: 10,
        edge_weights: "distance".to_string(),
        graph_regularization: 0.1,
        preserve_structure: true,
    };

    // Streaming dimensionality reduction configuration
    let streaming_config = QStreamingConfig {
        batch_size: 32,
        forgetting_factor: 0.95,
        update_frequency: 10,
        memory_window: 1000,
        adaptation_rate: 0.01,
    };

    let specialized_config = QSpecializedConfig {
        time_series_config: Some(time_series_config),
        image_tensor_config: Some(image_config),
        graph_config: Some(graph_config),
        streaming_config: Some(streaming_config),
    };

    println!("Created specialized configurations:");
    println!("  - Time Series DR with window size 10");
    println!("  - Image/Tensor DR with 4x4 patches");
    println!("  - Graph DR with k-NN construction");
    println!("  - Streaming DR with adaptive learning");

    // Feature selection configuration
    let feature_selection_config = QFeatureSelectionConfig {
        n_features: 5,
        criterion: "mutual_info".to_string(),
        quantum_enhancement: QuantumEnhancementLevel::Light,
        num_qubits: 3,
        regularization: 0.01,
        cv_folds: 5,
        scoring: "accuracy".to_string(),
        step: 1,
        variance_threshold: 0.1,
    };

    println!("  - Feature Selection with mutual information criterion");

    // Demonstrate default configuration creators
    println!("\nDefault configurations available:");
    let _default_qpca = create_default_qpca_config();
    let _default_qica = create_default_qica_config();
    let _default_qtsne = create_default_qtsne_config();
    let _default_qautoencoder = create_default_qautoencoder_config();

    println!("  - Default QPCA config (2 components, VQE solver)");
    println!("  - Default QICA config (2 components, logcosh nonlinearity)");
    println!("  - Default Qt-SNE config (2 components, perplexity 30)");
    println!("  - Default QAutoencoder config (standard VAE)");

    Ok(())
}

/// Generate synthetic high-dimensional data with known structure
fn generate_synthetic_data(n_samples: usize, n_features: usize) -> Result<(Array2<f64>, Array1<i32>)> {
    use rand::distributions::{Distribution, Normal};
    
    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    
    // Create three clusters in high-dimensional space
    let clusters = [
        ([2.0, 2.0, 0.0], 0.5), // Cluster 0: center and std
        ([0.0, -2.0, 1.0], 0.7), // Cluster 1
        ([-2.0, 0.0, -1.0], 0.6), // Cluster 2
    ];
    
    for i in 0..n_samples {
        let cluster_idx = i % 3;
        let (center, std) = clusters[cluster_idx];
        labels[i] = cluster_idx as i32;
        
        // Generate data point
        for j in 0..n_features {
            let base_value = if j < 3 { center[j] } else { 0.0 };
            let noise = Normal::new(0.0, std).unwrap().sample(&mut rand::thread_rng());
            data[[i, j]] = base_value + noise;
            
            // Add some correlation structure
            if j > 2 {
                data[[i, j]] += 0.3 * data[[i, j % 3]];
            }
        }
    }
    
    Ok((data, labels))
}

/// Print algorithm information
fn print_algorithm_info() {
    println!("Available Quantum Dimensionality Reduction Algorithms:");
    println!("  1. QPCA - Quantum Principal Component Analysis");
    println!("  2. QICA - Quantum Independent Component Analysis");
    println!("  3. Qt-SNE - Quantum t-distributed Stochastic Neighbor Embedding");
    println!("  4. QUMAP - Quantum Uniform Manifold Approximation and Projection");
    println!("  5. QLDA - Quantum Linear Discriminant Analysis");
    println!("  6. QFactorAnalysis - Quantum Factor Analysis");
    println!("  7. QCCA - Quantum Canonical Correlation Analysis");
    println!("  8. QNMF - Quantum Non-negative Matrix Factorization");
    println!("  9. QVAE - Quantum Variational Autoencoder");
    println!(" 10. QDenoisingAE - Quantum Denoising Autoencoder");
    println!(" 11. QSparseAE - Quantum Sparse Autoencoder");
    println!(" 12. QManifoldLearning - Quantum Manifold Learning");
    println!(" 13. QKernelPCA - Quantum Kernel PCA");
    println!(" 14. QMDS - Quantum Multidimensional Scaling");
    println!(" 15. QIsomap - Quantum Isomap");
    println!(" 16. Feature Selection Methods (Mutual Info, RFE, LASSO, Ridge, Variance)");
    println!(" 17. Specialized Methods (Time Series, Image/Tensor, Graph, Streaming)");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let (data, labels) = generate_synthetic_data(50, 5).unwrap();
        assert_eq!(data.nrows(), 50);
        assert_eq!(data.ncols(), 5);
        assert_eq!(labels.len(), 50);
        
        // Check that we have three clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 3);
    }

    #[test]
    fn test_qpca_demo() {
        let (data, _) = generate_synthetic_data(30, 5).unwrap();
        assert!(demo_qpca(&data).is_ok());
    }

    #[test]
    fn test_qica_demo() {
        let (data, _) = generate_synthetic_data(30, 5).unwrap();
        assert!(demo_qica(&data).is_ok());
    }

    #[test]
    fn test_default_configs() {
        let _qpca_config = create_default_qpca_config();
        let _qica_config = create_default_qica_config();
        let _qtsne_config = create_default_qtsne_config();
        let _qautoencoder_config = create_default_qautoencoder_config();
        
        // If we get here without panicking, the configs are valid
        assert!(true);
    }
}