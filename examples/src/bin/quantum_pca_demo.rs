//! Quantum Principal Component Analysis demonstration

use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Principal Component Analysis Demo ===\n");

    // Generate synthetic data with clear principal components
    let data = generate_synthetic_data();
    println!(
        "Generated synthetic data: {} samples × {} features",
        data.nrows(),
        data.ncols()
    );

    // Perform quantum PCA
    quantum_pca_demo(&data)?;

    // Demonstrate density matrix PCA
    density_matrix_pca_demo(&data)?;

    // Show quantum advantage analysis
    quantum_advantage_analysis()?;

    Ok(())
}

/// Generate synthetic dataset with known structure
fn generate_synthetic_data() -> Array2<f64> {
    let n_samples = 100;
    let n_features = 6;

    let mut data = Array2::zeros((n_samples, n_features));

    // Create data with 2 dominant principal components
    for i in 0..n_samples {
        let t = i as f64 / n_samples as f64;

        // First principal component (strong)
        data[[i, 0]] = 5.0f64.mul_add(t, 0.1 * thread_rng().gen::<f64>());
        data[[i, 1]] = 5.0f64.mul_add(t, 0.1 * thread_rng().gen::<f64>());

        // Second principal component (medium)
        data[[i, 2]] = 3.0f64.mul_add(1.0 - t, 0.1 * thread_rng().gen::<f64>());
        data[[i, 3]] = 3.0f64.mul_add(1.0 - t, 0.1 * thread_rng().gen::<f64>());

        // Noise dimensions (weak)
        data[[i, 4]] = 0.5 * thread_rng().gen::<f64>();
        data[[i, 5]] = 0.5 * thread_rng().gen::<f64>();
    }

    data
}

/// Demonstrate basic quantum PCA
fn quantum_pca_demo(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Quantum PCA:");
    println!("---------------------");

    let params = QPCAParams {
        precision_qubits: 8,
        num_samples: 1000,
        eigenvalue_threshold: 0.01,
        max_iterations: 100,
    };

    let mut qpca = QuantumPCA::new(data.clone(), params);

    // Compute density matrix
    let density = qpca.compute_density_matrix()?;
    println!("✓ Computed quantum density matrix");
    println!("  Matrix shape: {:?}", density.shape());

    // Extract principal components
    qpca.extract_components()?;
    println!("✓ Extracted principal components using quantum phase estimation");

    // Get results
    if let Some(eigenvalues) = qpca.eigenvalues() {
        println!("\nEigenvalues (sorted by magnitude):");
        for (i, &val) in eigenvalues.iter().enumerate() {
            println!("  Component {}: {:.6}", i + 1, val);
        }
    }

    // Compute explained variance
    let variance_ratio = qpca.explained_variance_ratio()?;
    println!("\nExplained variance ratio:");
    let mut cumulative = 0.0;
    for (i, &var) in variance_ratio.iter().enumerate() {
        cumulative += var;
        println!(
            "  Component {}: {:.2}% (cumulative: {:.2}%)",
            i + 1,
            var * 100.0,
            cumulative * 100.0
        );
    }

    // Transform data
    let transformed = qpca.transform(data)?;
    println!("\n✓ Transformed data to principal component space");
    println!("  New shape: {:?}", transformed.shape());

    Ok(())
}

/// Demonstrate density matrix PCA with automatic component selection
fn density_matrix_pca_demo(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Density Matrix PCA (Quantum-Inspired):");
    println!("-----------------------------------------");

    let params = QPCAParams::default();
    let mut pca = DensityMatrixPCA::new(params);
    pca.trace_threshold = 0.95; // Capture 95% of variance

    let (transformed, variance) = pca.fit_transform(data)?;

    println!("✓ Performed density matrix PCA with automatic dimensionality reduction");
    println!("  Original dimensions: {}", data.ncols());
    println!("  Reduced dimensions: {}", transformed.ncols());
    println!("  Variance captured: {:.2}%", variance.sum() * 100.0);

    println!("\nRetained component variances:");
    for (i, &var) in variance.iter().enumerate() {
        println!("  Component {}: {:.4}", i + 1, var);
    }

    Ok(())
}

/// Analyze quantum advantage for PCA
fn quantum_advantage_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Quantum Advantage Analysis:");
    println!("------------------------------");

    println!("Classical PCA complexity: O(n³) for n×n covariance matrix");
    println!("Quantum PCA complexity: O(log(n)²) with quantum phase estimation");

    // Show scaling comparison
    println!("\nScaling comparison:");
    let dimensions = vec![10, 100, 1000, 10000];
    for &n in &dimensions {
        let classical_ops = n * n * n;
        let quantum_ops = ((n as f64).log2() as usize).pow(2);
        let speedup = classical_ops / quantum_ops.max(1);

        println!(
            "  n={n:5}: Classical={classical_ops:12} ops, Quantum={quantum_ops:6} ops, Speedup={speedup:6}x"
        );
    }

    println!("\nKey advantages of quantum PCA:");
    println!("✓ Exponential speedup for high-dimensional data");
    println!("✓ Efficient eigenvalue estimation via phase estimation");
    println!("✓ Direct sampling from principal component subspace");
    println!("✓ Memory-efficient density matrix representation");

    println!("\nApplications:");
    println!("• Quantum machine learning");
    println!("• High-dimensional data analysis");
    println!("• Quantum state tomography");
    println!("• Financial portfolio optimization");

    Ok(())
}

/// Bonus: Demonstrate qPCA on quantum state data
fn quantum_state_pca_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Quantum State PCA (Bonus):");
    println!("-----------------------------");

    // Generate quantum state measurement data
    let n_measurements = 1000;
    let n_qubits = 3;
    let dim = 1 << n_qubits;

    let mut measurements = Array2::zeros((n_measurements, dim));

    // Simulate measurements of a mixed quantum state
    for i in 0..n_measurements {
        // Create random quantum state measurement
        let mut state = Array1::zeros(dim);

        // Bias towards certain basis states
        if thread_rng().gen::<f64>() < 0.7 {
            state[0] = 0.2f64.mul_add(thread_rng().gen::<f64>(), 0.8);
            state[1] = 0.2 * thread_rng().gen::<f64>();
        } else {
            state[6] = 0.4f64.mul_add(thread_rng().gen::<f64>(), 0.6);
            state[7] = 0.4 * thread_rng().gen::<f64>();
        }

        // Normalize
        let norm: f64 = state.iter().map(|x| x * x).sum::<f64>().sqrt();
        state /= norm;

        measurements.row_mut(i).assign(&state);
    }

    // Apply quantum PCA
    let params = QPCAParams::default();
    let mut qpca = QuantumPCA::new(measurements, params);

    qpca.compute_density_matrix()?;
    qpca.extract_components()?;

    let variance_ratio = qpca.explained_variance_ratio()?;

    println!("Applied qPCA to quantum state measurements:");
    println!(
        "  Found {} principal components",
        qpca.n_components().unwrap_or(0)
    );
    println!(
        "  Top component explains {:.2}% of variance",
        variance_ratio.get(0).unwrap_or(&0.0) * 100.0
    );

    Ok(())
}
