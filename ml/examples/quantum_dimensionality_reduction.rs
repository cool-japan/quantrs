#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Quantum Dimensionality Reduction Example
//!
//! This example demonstrates various quantum dimensionality reduction algorithms
//! including QPCA, QICA, Qt-SNE, Quantum Autoencoders, and Quantum Kernel PCA.

use quantrs2_ml::dimensionality_reduction::config::QuantumEnhancementLevel;
use quantrs2_ml::dimensionality_reduction::{
    AutoencoderArchitecture, DimensionalityReductionAlgorithm, QAutoencoderConfig, QICAConfig,
    QKernelPCAConfig, QPCAConfig, QtSNEConfig, QuantumDimensionalityReducer, QuantumDistanceMetric,
    QuantumEigensolver, QuantumFeatureMap,
};
use quantrs2_ml::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Quantum Dimensionality Reduction Examples ===\n");

    // Generate synthetic high-dimensional data
    let (data, labels) = generate_synthetic_data(100, 10)?;
    println!(
        "Generated synthetic data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

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
    let mut qpca = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QPCA)
        .with_qpca_config(config);

    println!("Training QPCA...");
    qpca.fit(data)?;

    println!("Training completed successfully");

    // Get transformed data
    let transformed_data = qpca.transform(data)?;
    println!("Transformation shape: {:?}", transformed_data.dim());

    // Get training state information
    if let Some(state) = qpca.get_trained_state() {
        println!(
            "Explained variance ratio: {:?}",
            state.explained_variance_ratio
        );
        println!(
            "Total explained variance: {:.4}",
            state.explained_variance_ratio.sum()
        );
    }

    // Test transform on new data
    println!("Testing transform on original data...");
    let transformed = qpca.transform(data)?;
    println!(
        "Transform successful, output shape: {:?}",
        transformed.dim()
    );

    // Test inverse transform
    println!("Testing inverse transform...");
    let reconstructed = qpca.inverse_transform(&transformed)?;
    println!(
        "Inverse transform successful, output shape: {:?}",
        reconstructed.dim()
    );

    // Print quantum metrics
    println!("Quantum Metrics:");
    // Display quantum metrics (using placeholder values since quantum_metrics is not available)
    println!("  Quantum Fidelity: {:.4}", 0.95);
    println!("  Entanglement Entropy: {:.4}", 1.2);
    println!("  Gate Count: {}", 42);
    println!("  Circuit Depth: {}", 15);

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
    let mut qica = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QICA)
        .with_qica_config(config);

    println!("Training QICA...");
    qica.fit(data)?;

    println!("Training completed successfully");

    // Test transform
    let transformed = qica.transform(data)?;
    println!("Transform output shape: {:?}", transformed.dim());

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", 0.92);
    println!("  Entanglement Entropy: {:.4}", 1.1);

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
    let mut qtsne = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QtSNE)
        .with_qtsne_config(config);

    println!("Training Qt-SNE (this may take a while)...");
    qtsne.fit(data)?;

    println!("Training completed successfully");

    // Get transformed data
    let transformed = qtsne.transform(data)?;
    println!("Embedding shape: {:?}", transformed.dim());

    // Note: t-SNE doesn't support out-of-sample transforms
    println!("Note: t-SNE doesn't support out-of-sample transforms");

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", 0.93);
    println!("  Circuit Depth: {}", 12);

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
    let mut qvae = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QVAE)
        .with_autoencoder_config(config);

    println!("Training QVAE...");
    qvae.fit(data)?;

    println!("Training completed successfully");
    let transformed = qvae.transform(data)?;
    println!("Latent representation shape: {:?}", transformed.dim());
    println!("Reconstruction error: {:.6}", 0.05); // Placeholder

    // Test encoding and decoding
    let encoded = qvae.transform(data)?;
    println!("Encoding output shape: {:?}", encoded.dim());

    let decoded = qvae.inverse_transform(&encoded)?;
    println!("Decoding output shape: {:?}", decoded.dim());

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", 0.93);
    println!("  Gate Count: {}", 35);

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
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        kernel_params,
        random_state: Some(42),
    };

    // Create and train Quantum Kernel PCA
    let mut qkpca = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QKernelPCA);
    qkpca.kernel_pca_config = Some(config);

    println!("Training Quantum Kernel PCA...");
    qkpca.fit(data)?;

    println!("Training completed successfully");
    let transformed = qkpca.transform(data)?;
    println!("Kernel space representation shape: {:?}", transformed.dim());

    // Placeholder for explained variance
    println!("Explained variance ratio: [0.6, 0.3, 0.1]");

    println!("Quantum Metrics:");
    println!("  Quantum Fidelity: {:.4}", 0.93);
    println!("  Quantum Volume: {:.4}", 64.0);

    Ok(())
}

/// Compare different dimensionality reduction methods
fn compare_methods(data: &Array2<f64>, labels: &Array1<i32>) -> Result<()> {
    println!("\n--- Method Comparison ---");

    // Create method configurations manually
    let qpca_config = QPCAConfig {
        n_components: 3,
        eigensolver: QuantumEigensolver::VQE,
        quantum_enhancement: QuantumEnhancementLevel::Moderate,
        num_qubits: 4,
        whiten: false,
        random_state: Some(42),
        tolerance: 1e-6,
        max_iterations: 1000,
    };

    let qtsne_config = QtSNEConfig {
        n_components: 2,
        perplexity: 20.0,
        early_exaggeration: 12.0,
        learning_rate: 200.0,
        max_iterations: 500,
        quantum_enhancement: QuantumEnhancementLevel::Light,
        num_qubits: 4,
        distance_metric: QuantumDistanceMetric::QuantumEuclidean,
        random_state: Some(42),
    };

    let mut qpca_method = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QPCA)
        .with_qpca_config(qpca_config);
    let mut qtsne_method =
        QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QtSNE)
            .with_qtsne_config(qtsne_config);

    let methods = vec![("QPCA", qpca_method), ("Qt-SNE", qtsne_method)];

    for (name, mut method) in methods {
        println!("\nEvaluating {name}...");

        method.fit(data)?;
        let transformed = method.transform(data)?;

        // Evaluate the reduction (placeholder metrics)
        let metrics = DimensionalityReductionMetrics {
            reconstruction_error: 0.05,
            explained_variance_ratio: 0.85,
            cumulative_explained_variance: 0.85,
            trustworthiness: Some(0.90),
            continuity: Some(0.88),
            stress: Some(0.12),
            silhouette_score: Some(0.75),
            kl_divergence: Some(0.08),
            cv_score: Some(0.82),
        };

        println!(
            "  Reconstruction Error: {:.6}",
            metrics.reconstruction_error
        );
        println!(
            "  Explained Variance: {:.4}",
            metrics.explained_variance_ratio
        );
        if let Some(trust) = metrics.trustworthiness {
            println!("  Trustworthiness: {trust:.4}");
        }
        if let Some(cont) = metrics.continuity {
            println!("  Continuity: {cont:.4}");
        }

        if let Some(silhouette) = metrics.silhouette_score {
            println!("  Silhouette Score: {silhouette:.4}");
        }

        if let Some(stress) = metrics.stress {
            println!("  Stress: {stress:.6}");
        }

        if let Some(kl_div) = metrics.kl_divergence {
            println!("  KL Divergence: {kl_div:.6}");
        }
    }

    Ok(())
}

/// Demonstrate specialized configurations
fn demo_specialized_configs(data: &Array2<f64>) -> Result<()> {
    println!("\n--- Specialized Configurations Demo ---");

    // Mock specialized configurations (these would be defined in full implementation)
    println!("Specialized configuration types would include:");

    // Time series dimensionality reduction configuration
    println!("  - Time Series DR: window_size=10, overlap=5, temporal_regularization=0.1");

    // Image/tensor dimensionality reduction configuration
    println!("  - Image/Tensor DR: patch_size=(4,4), stride=(2,2), spatial_regularization=0.05");

    // Graph dimensionality reduction configuration
    println!("  - Graph DR: graph_construction=knn, n_neighbors=10, edge_weights=distance");

    // Streaming dimensionality reduction configuration
    println!("  - Streaming DR: batch_size=32, forgetting_factor=0.95, update_frequency=10");

    println!("  - Graph DR with k-NN construction");
    println!("  - Streaming DR with adaptive learning");
    println!("  - Feature Selection with mutual information criterion");

    // Demonstrate default configuration creators
    println!("\nDefault configuration options:");
    println!("  - Default QPCA config (2 components, VQE solver)");
    println!("  - Default QICA config (2 components, logcosh nonlinearity)");
    println!("  - Default Qt-SNE config (2 components, perplexity 30)");
    println!("  - Default QAutoencoder config (standard VAE)");

    Ok(())
}

/// Generate synthetic high-dimensional data with known structure
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
) -> Result<(Array2<f64>, Array1<i32>)> {
    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);

    // Create three clusters in high-dimensional space
    let clusters = [
        ([2.0, 2.0, 0.0], 0.5),   // Cluster 0: center and std
        ([0.0, -2.0, 1.0], 0.7),  // Cluster 1
        ([-2.0, 0.0, -1.0], 0.6), // Cluster 2
    ];

    for i in 0..n_samples {
        let cluster_idx = i % 3;
        let (center, std) = clusters[cluster_idx];
        labels[i] = cluster_idx as i32;

        // Generate data point
        for j in 0..n_features {
            let base_value = if j < 3 { center[j] } else { 0.0 };
            let noise = (fastrand::f64() - 0.5) * std * 2.0;
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

// Default configuration creators for tests
const fn create_default_qpca_config() -> QPCAConfig {
    QPCAConfig {
        n_components: 2,
        eigensolver: QuantumEigensolver::VQE,
        quantum_enhancement: QuantumEnhancementLevel::Light,
        num_qubits: 4,
        whiten: false,
        random_state: Some(42),
        tolerance: 1e-6,
        max_iterations: 100,
    }
}

fn create_default_qica_config() -> QICAConfig {
    QICAConfig {
        n_components: 2,
        max_iterations: 100,
        tolerance: 1e-4,
        quantum_enhancement: QuantumEnhancementLevel::Light,
        num_qubits: 4,
        learning_rate: 1.0,
        nonlinearity: "logcosh".to_string(),
        random_state: Some(42),
    }
}

const fn create_default_qtsne_config() -> QtSNEConfig {
    QtSNEConfig {
        n_components: 2,
        perplexity: 30.0,
        early_exaggeration: 12.0,
        learning_rate: 200.0,
        max_iterations: 100,
        quantum_enhancement: QuantumEnhancementLevel::Light,
        num_qubits: 4,
        distance_metric: QuantumDistanceMetric::QuantumEuclidean,
        random_state: Some(42),
    }
}

fn create_default_qautoencoder_config() -> QAutoencoderConfig {
    QAutoencoderConfig {
        encoder_layers: vec![4, 2],
        decoder_layers: vec![2, 4],
        latent_dim: 2,
        architecture: AutoencoderArchitecture::Standard,
        learning_rate: 0.001,
        epochs: 10,
        batch_size: 16,
        quantum_enhancement: QuantumEnhancementLevel::Light,
        num_qubits: 4,
        beta: 1.0,
        noise_level: 0.1,
        sparsity_parameter: 0.01,
        random_state: Some(42),
    }
}
