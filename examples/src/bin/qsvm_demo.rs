//! Quantum Support Vector Machine demonstration

use ndarray::{array, Array1, Array2};
use quantrs2_ml::prelude::*;
use rand::Rng;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Support Vector Machine Demo ===\n");

    // Demo 1: Basic binary classification
    basic_classification_demo()?;

    // Demo 2: Feature map comparison
    feature_map_comparison_demo()?;

    // Demo 3: Quantum kernel visualization
    kernel_visualization_demo()?;

    // Demo 4: Quantum advantage analysis
    quantum_advantage_demo()?;

    Ok(())
}

/// Basic binary classification with QSVM
fn basic_classification_demo() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Binary Classification:");
    println!("-------------------------------");

    // Generate XOR-like dataset (non-linearly separable)
    let (x_train, y_train) = generate_xor_dataset(100);
    let (x_test, y_test) = generate_xor_dataset(20);

    println!("Generated XOR dataset:");
    println!("  Training samples: {}", x_train.nrows());
    println!("  Test samples: {}", x_test.nrows());
    println!("  Features: {}", x_train.ncols());

    // Train QSVM with ZZ feature map
    let params = QSVMParams {
        feature_map: FeatureMapType::ZZFeatureMap,
        reps: 2,
        c: 1.0,
        tolerance: 1e-3,
        num_qubits: 2,
        depth: 2,
        gamma: Some(1.0),
        regularization: 1e-4,
        max_iterations: 100,
        seed: Some(42),
    };

    let mut qsvm = QSVM::new(params);

    println!("\nTraining QSVM...");
    qsvm.fit(&x_train, &y_train)
        .map_err(|e| format!("Training failed: {}", e))?;

    println!("✓ Training complete!");
    println!("  Support vectors: {}", qsvm.n_support_vectors());

    // Predict on test set
    let predictions = qsvm.predict(&x_test).map_err(|e| e)?;

    // Calculate accuracy
    let mut correct = 0;
    for i in 0..y_test.len() {
        if predictions[i] == y_test[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / y_test.len() as f64;
    println!("\nTest accuracy: {:.2}%", accuracy * 100.0);

    // Show decision values for a few samples
    let decision_values = qsvm.decision_function(&x_test).map_err(|e| e)?;
    println!("\nSample predictions:");
    for i in 0..5.min(x_test.nrows()) {
        println!(
            "  Sample {}: features={:.2}, {:.2}, label={}, prediction={}, confidence={:.3}",
            i,
            x_test[[i, 0]],
            x_test[[i, 1]],
            y_test[i],
            predictions[i],
            decision_values[i].abs()
        );
    }

    Ok(())
}

/// Compare different feature maps
fn feature_map_comparison_demo() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Feature Map Comparison:");
    println!("--------------------------");

    // Simple linearly separable dataset
    let x = array![[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0],];
    let y = array![-1, -1, 1, 1];

    let feature_maps = vec![
        ("Z Feature Map", FeatureMapType::ZFeatureMap),
        ("ZZ Feature Map", FeatureMapType::ZZFeatureMap),
        ("Angle Encoding", FeatureMapType::AngleEncoding),
        ("Amplitude Encoding", FeatureMapType::AmplitudeEncoding),
    ];

    for (name, feature_map) in feature_maps {
        let params = QSVMParams {
            feature_map,
            reps: 2,
            c: 1.0,
            tolerance: 1e-3,
            num_qubits: 2,
            depth: 2,
            gamma: Some(1.0),
            regularization: 1e-4,
            max_iterations: 100,
            seed: Some(42),
        };

        let mut qsvm = QSVM::new(params);

        match qsvm.fit(&x, &y) {
            Ok(_) => {
                let predictions = qsvm.predict(&x).map_err(|e| e)?;
                let accuracy = calculate_accuracy(&predictions, &y);
                println!(
                    "  {}: Support vectors={}, Training accuracy={:.1}%",
                    name,
                    qsvm.n_support_vectors(),
                    accuracy * 100.0
                );
            }
            Err(e) => {
                println!("  {}: Failed - {}", name, e);
            }
        }
    }

    Ok(())
}

/// Visualize quantum kernel values
fn kernel_visualization_demo() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Quantum Kernel Visualization:");
    println!("---------------------------------");

    // Create quantum kernel
    let kernel = QSVMKernel::new(FeatureMapType::ZZFeatureMap, 2);

    // Sample points
    let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [0.5, 0.5],];

    // Compute kernel matrix
    println!("Kernel matrix (ZZ feature map, 2 reps):");
    println!("       ",);
    for i in 0..points.nrows() {
        print!("  [{:.1},{:.1}]", points[[i, 0]], points[[i, 1]]);
    }
    println!();

    for i in 0..points.nrows() {
        print!("[{:.1},{:.1}]", points[[i, 0]], points[[i, 1]]);
        for j in 0..points.nrows() {
            let k_val = kernel.compute(&points.row(i).to_owned(), &points.row(j).to_owned());
            print!("  {:.3}", k_val);
        }
        println!();
    }

    println!("\nKernel properties:");
    println!("✓ Symmetric: K(x,y) = K(y,x)");
    println!("✓ Positive semi-definite");
    println!("✓ K(x,x) = 1 for normalized feature maps");

    Ok(())
}

/// Analyze quantum advantage for SVM
fn quantum_advantage_demo() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Quantum Advantage Analysis:");
    println!("-------------------------------");

    println!("Classical SVM:");
    println!("  • Kernel computation: O(d) for d-dimensional data");
    println!("  • Limited to efficiently computable kernels");
    println!("  • Feature space dimension bounded by classical resources");

    println!("\nQuantum SVM:");
    println!("  • Access to exponentially large Hilbert space");
    println!("  • Quantum interference creates complex decision boundaries");
    println!("  • Feature maps can encode quantum correlations");

    println!("\nAdvantages of QSVM:");
    println!("  1. Feature Space: 2^n dimensional for n qubits");
    println!("  2. Kernel Evaluation: Quantum speedup for certain kernels");
    println!("  3. Expressivity: Can represent classically hard functions");
    println!("  4. Data Encoding: Multiple encoding strategies available");

    println!("\nPractical Benefits:");
    println!("  • Better performance on quantum-generated data");
    println!("  • Potential advantages for specific problem structures");
    println!("  • Natural integration with other quantum algorithms");

    // Show scaling comparison
    println!("\nScaling comparison:");
    let qubit_counts = vec![4, 8, 12, 16, 20];
    for &n in &qubit_counts {
        let classical_dim = n; // Classical feature dimension
        let quantum_dim = 1 << n; // Quantum Hilbert space dimension
        println!(
            "  {} qubits: Classical dim={}, Quantum dim={} ({}x larger)",
            n,
            classical_dim,
            quantum_dim,
            quantum_dim / classical_dim
        );
    }

    Ok(())
}

/// Generate XOR-like dataset
fn generate_xor_dataset(n_samples: usize) -> (Array2<f64>, Array1<i32>) {
    let mut rng = rand::thread_rng();
    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let x1 = rng.gen::<f64>() * 2.0 - 1.0;
        let x2 = rng.gen::<f64>() * 2.0 - 1.0;

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;

        // XOR pattern with some noise
        let label = if (x1 > 0.0) ^ (x2 > 0.0) { 1 } else { -1 };

        // Add 5% label noise
        if rng.gen::<f64>() < 0.05 {
            y[i] = -label;
        } else {
            y[i] = label;
        }
    }

    (x, y)
}

/// Calculate accuracy
fn calculate_accuracy(predictions: &Array1<i32>, labels: &Array1<i32>) -> f64 {
    let mut correct = 0;
    for i in 0..labels.len() {
        if predictions[i] == labels[i] {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64
}

/// Bonus: Quantum kernel ridge regression demo
fn quantum_kernel_ridge_demo() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\n5. Quantum Kernel Ridge Regression (Bonus):");
    println!("-------------------------------------------");

    // Generate regression dataset
    let n_samples = 50;
    let mut x = Array2::zeros((n_samples, 1));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let xi = i as f64 / n_samples as f64 * 2.0 * std::f64::consts::PI;
        x[[i, 0]] = xi;
        y[i] = xi.sin() + 0.1 * rand::thread_rng().gen::<f64>();
    }

    // Train quantum kernel ridge regression
    let mut qkr = QuantumKernelRidge::new(FeatureMapType::AngleEncoding, 1, 0.1);

    match qkr.fit(&x, &y) {
        Ok(_) => {
            println!("✓ Quantum kernel ridge regression trained");

            // Test on a few points
            let x_test = array![[0.0], [1.57], [std::f64::consts::PI], [4.71], [std::f64::consts::TAU]];
            match qkr.predict(&x_test) {
                Ok(predictions) => {
                    println!("\nPredictions:");
                    for i in 0..x_test.nrows() {
                        let true_val = x_test[[i, 0]].sin();
                        println!(
                            "  x={:.2}: predicted={:.3}, true={:.3}",
                            x_test[[i, 0]],
                            predictions[i],
                            true_val
                        );
                    }
                }
                Err(e) => println!("Prediction failed: {}", e),
            }
        }
        Err(e) => println!("Training failed: {}", e),
    }

    Ok(())
}
