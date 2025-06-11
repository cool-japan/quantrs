//! Scikit-learn Compatible Quantum ML Pipeline Example
//!
//! This example demonstrates the scikit-learn compatibility layer, showing how to use
//! quantum models with familiar sklearn APIs including pipelines, cross-validation, and grid search.

use ndarray::{s, Array1, Array2, Axis};
use quantrs2_ml::prelude::*;
use quantrs2_ml::sklearn_compatibility::{
    metrics, model_selection, Pipeline, QuantumFeatureEncoder, SelectKBest, SklearnFit,
    StandardScaler,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Scikit-learn Compatible Quantum ML Demo ===\n");

    // Step 1: Create sklearn-style dataset
    println!("1. Creating scikit-learn style dataset...");

    let (X, y) = create_sklearn_dataset()?;
    println!("   - Dataset shape: {:?}", X.dim());
    println!(
        "   - Labels: {} classes",
        y.iter()
            .map(|&x| x as i32)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!(
        "   - Feature range: [{:.3}, {:.3}]",
        X.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        X.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Step 2: Create sklearn-compatible quantum estimators
    println!("\n2. Creating sklearn-compatible quantum estimators...");

    // Quantum Support Vector Classifier
    let qsvc = QuantumSVC::new();

    // Quantum Multi-Layer Perceptron Classifier
    let qmlp = QuantumMLPClassifier::new();

    // Quantum K-Means Clustering
    let mut qkmeans = QuantumKMeans::new(2); // n_clusters

    println!("   - QuantumSVC: quantum kernel");
    println!("   - QuantumMLP: multi-layer perceptron");
    println!("   - QuantumKMeans: 2 clusters");

    // Step 3: Create sklearn-style preprocessing pipeline
    println!("\n3. Building sklearn-compatible preprocessing pipeline...");

    let preprocessing_pipeline = Pipeline::new(vec![
        ("scaler", Box::new(StandardScaler::new())),
        (
            "feature_selection",
            Box::new(SelectKBest::new(
                "quantum_mutual_info", // score_func
                3,                     // k
            )),
        ),
        (
            "quantum_encoder",
            Box::new(QuantumFeatureEncoder::new(
                "angle", // encoding_type
                "l2",    // normalization
            )),
        ),
    ])?;

    // Step 4: Create complete quantum ML pipeline
    println!("\n4. Creating complete quantum ML pipeline...");

    let quantum_pipeline = Pipeline::new(vec![
        ("preprocessing", Box::new(preprocessing_pipeline)),
        ("classifier", Box::new(qsvc.clone())),
    ])?;

    println!("   Pipeline steps:");
    for (i, step_name) in quantum_pipeline.named_steps().iter().enumerate() {
        println!("   {}. {}", i + 1, step_name);
    }

    // Step 5: Train-test split (sklearn style)
    println!("\n5. Performing train-test split...");

    let (X_train, X_test, y_train, y_test) = model_selection::train_test_split(
        &X,
        &y,
        0.3,      // test_size
        Some(42), // random_state
    )?;

    println!("   - Training set: {:?}", X_train.dim());
    println!("   - Test set: {:?}", X_test.dim());

    // Step 6: Cross-validation with quantum models
    println!("\n6. Performing cross-validation...");

    let mut pipeline_clone = quantum_pipeline.clone();
    let cv_scores = model_selection::cross_val_score(
        &mut pipeline_clone,
        &X_train,
        &y_train,
        5, // cv
    )?;

    println!("   Cross-validation scores: {:?}", cv_scores);
    println!(
        "   Mean CV accuracy: {:.3} (+/- {:.3})",
        cv_scores.mean().unwrap(),
        cv_scores.std(0.0) * 2.0
    );

    // Step 7: Hyperparameter grid search
    println!("\n7. Hyperparameter optimization with GridSearchCV...");

    let param_grid = HashMap::from([
        (
            "classifier__C".to_string(),
            vec!["0.1".to_string(), "1.0".to_string(), "10.0".to_string()],
        ),
        (
            "classifier__feature_map_depth".to_string(),
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
        ),
        (
            "preprocessing__feature_selection__k".to_string(),
            vec!["2".to_string(), "3".to_string(), "4".to_string()],
        ),
    ]);

    let mut grid_search = model_selection::GridSearchCV::new(
        quantum_pipeline.clone(), // estimator
        param_grid,
        3, // cv
    );

    grid_search.fit(&X_train, &y_train)?;

    println!("   Best parameters: {:?}", grid_search.best_params_);
    println!(
        "   Best cross-validation score: {:.3}",
        grid_search.best_score_
    );

    // Step 8: Train best model and evaluate
    println!("\n8. Training best model and evaluation...");

    let best_model = grid_search.best_estimator_;
    let y_pred = best_model.predict(&X_test)?;

    // Calculate metrics using sklearn-style functions
    let y_test_int = y_test.mapv(|x| x.round() as i32);
    let accuracy = metrics::accuracy_score(&y_test_int, &y_pred);
    let precision = metrics::precision_score(&y_test_int, &y_pred, "weighted"); // average
    let recall = metrics::recall_score(&y_test_int, &y_pred, "weighted"); // average
    let f1 = metrics::f1_score(&y_test_int, &y_pred, "weighted"); // average

    println!("   Test Results:");
    println!("   - Accuracy: {:.3}", accuracy);
    println!("   - Precision: {:.3}", precision);
    println!("   - Recall: {:.3}", recall);
    println!("   - F1-score: {:.3}", f1);

    // Step 9: Classification report
    println!("\n9. Detailed classification report...");

    let classification_report = metrics::classification_report(
        &y_test_int,
        &y_pred,
        vec!["Class 0", "Class 1"], // target_names
        3,                          // digits
    );
    println!("{}", classification_report);

    // Step 10: Feature importance analysis
    println!("\n10. Feature importance analysis...");

    if let Some(feature_importances) = best_model.feature_importances() {
        println!("    Quantum Feature Importances:");
        for (i, importance) in feature_importances.iter().enumerate() {
            println!("    - Feature {}: {:.4}", i, importance);
        }
    }

    // Step 11: Model comparison with classical sklearn models
    println!("\n11. Comparing with classical sklearn models...");

    let classical_models = vec![
        (
            "Logistic Regression",
            Box::new(LogisticRegression::new()) as Box<dyn SklearnClassifier>,
        ),
        (
            "Random Forest",
            Box::new(RandomForestClassifier::new()) as Box<dyn SklearnClassifier>,
        ),
        ("SVM", Box::new(SVC::new()) as Box<dyn SklearnClassifier>),
    ];

    let mut comparison_results = Vec::new();

    for (name, mut model) in classical_models {
        model.fit(&X_train, Some(&y_train))?;
        let y_pred_classical = model.predict(&X_test)?;
        let classical_accuracy = metrics::accuracy_score(&y_test_int, &y_pred_classical);
        comparison_results.push((name, classical_accuracy));
    }

    println!("    Model Comparison:");
    println!("    - Quantum Pipeline: {:.3}", accuracy);
    for (name, classical_accuracy) in comparison_results {
        println!("    - {}: {:.3}", name, classical_accuracy);
    }

    // Step 12: Clustering with quantum K-means
    println!("\n12. Quantum clustering analysis...");

    let cluster_labels = qkmeans.fit_predict(&X)?;
    let silhouette_score = metrics::silhouette_score(&X, &cluster_labels, "euclidean"); // metric
    let calinski_score = metrics::calinski_harabasz_score(&X, &cluster_labels);

    println!("    Clustering Results:");
    println!("    - Silhouette Score: {:.3}", silhouette_score);
    println!("    - Calinski-Harabasz Score: {:.3}", calinski_score);
    println!(
        "    - Unique clusters found: {}",
        cluster_labels
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    // Step 13: Model persistence (sklearn style)
    println!("\n13. Model persistence (sklearn joblib style)...");

    // Save model
    best_model.save("quantum_sklearn_model.joblib")?;
    println!("    - Model saved to: quantum_sklearn_model.joblib");

    // Load model
    let loaded_model = QuantumSVC::load("quantum_sklearn_model.joblib")?;
    let test_subset = X_test.slice(s![..5, ..]).to_owned();
    let y_pred_loaded = loaded_model.predict(&test_subset)?;
    println!("    - Model loaded and tested on 5 samples");

    // Step 14: Advanced sklearn utilities
    println!("\n14. Advanced sklearn utilities...");

    // Learning curves (commented out - function not available)
    // let (train_sizes, train_scores, val_scores) = model_selection::learning_curve(...)?;
    println!("    Learning Curve Analysis: (Mock results)");
    let train_sizes = vec![0.1, 0.33, 0.55, 0.78, 1.0];
    let train_scores = vec![0.65, 0.72, 0.78, 0.82, 0.85];
    let val_scores = vec![0.62, 0.70, 0.76, 0.79, 0.81];

    for (i, &size) in train_sizes.iter().enumerate() {
        println!(
            "    - {:.0}% data: train={:.3}, val={:.3}",
            size * 100.0,
            train_scores[i],
            val_scores[i]
        );
    }

    // Validation curves (commented out - function not available)
    // let (train_scores_val, test_scores_val) = model_selection::validation_curve(...)?;
    println!("    Validation Curve (C parameter): (Mock results)");
    let param_range = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let train_scores_val = vec![0.70, 0.75, 0.80, 0.78, 0.75];
    let test_scores_val = vec![0.68, 0.73, 0.78, 0.76, 0.72];

    for (i, &param_value) in param_range.iter().enumerate() {
        println!(
            "    - C={}: train={:.3}, test={:.3}",
            param_value, train_scores_val[i], test_scores_val[i]
        );
    }

    // Step 15: Quantum-specific sklearn extensions
    println!("\n15. Quantum-specific sklearn extensions...");

    // Quantum feature analysis
    let quantum_feature_analysis = analyze_quantum_features(&best_model, &X_test)?;
    println!("    Quantum Feature Analysis:");
    println!(
        "    - Quantum advantage score: {:.3}",
        quantum_feature_analysis.advantage_score
    );
    println!(
        "    - Feature entanglement: {:.3}",
        quantum_feature_analysis.entanglement_measure
    );
    println!(
        "    - Circuit depth efficiency: {:.3}",
        quantum_feature_analysis.circuit_efficiency
    );

    // Quantum model interpretation
    let sample_row = X_test.row(0).to_owned();
    let quantum_interpretation = interpret_quantum_model(&best_model, &sample_row)?;
    println!("    Quantum Model Interpretation (sample 0):");
    println!(
        "    - Quantum state fidelity: {:.3}",
        quantum_interpretation.state_fidelity
    );
    println!(
        "    - Feature contributions: {:?}",
        quantum_interpretation.feature_contributions
    );

    println!("\n=== Scikit-learn Integration Demo Complete ===");

    Ok(())
}

fn create_sklearn_dataset() -> Result<(Array2<f64>, Array1<f64>)> {
    let num_samples = 300;
    let num_features = 4;

    // Create a dataset similar to sklearn's make_classification
    let X = Array2::from_shape_fn((num_samples, num_features), |(i, j)| {
        let base = (i as f64 * 0.02) + (j as f64 * 0.5);
        let noise = fastrand::f64() * 0.3 - 0.15;
        base.sin() + noise
    });

    // Create separable classes
    let y = Array1::from_shape_fn(num_samples, |i| {
        let feature_sum = (0..num_features).map(|j| X[[i, j]]).sum::<f64>();
        if feature_sum > 0.0 {
            1.0
        } else {
            0.0
        }
    });

    Ok((X, y))
}

fn analyze_quantum_features(
    model: &dyn SklearnClassifier,
    X: &Array2<f64>,
) -> Result<QuantumFeatureAnalysis> {
    // Analyze quantum-specific properties
    let predictions_quantum = model.predict(X)?;

    // Create classical baseline for comparison
    let mut classical_model = LogisticRegression::new();
    SklearnFit::fit(
        &mut classical_model,
        X,
        &predictions_quantum.mapv(|x| x as f64),
    )?; // Use quantum predictions as targets
    let predictions_classical = classical_model.predict(X)?;

    // Calculate quantum advantage score
    let advantage_score = predictions_quantum
        .iter()
        .zip(predictions_classical.iter())
        .map(|(&q, &c)| (q as f64 - c as f64).abs())
        .sum::<f64>()
        / predictions_quantum.len() as f64;

    Ok(QuantumFeatureAnalysis {
        advantage_score,
        entanglement_measure: 0.75, // Mock value
        circuit_efficiency: 0.85,   // Mock value
    })
}

fn interpret_quantum_model(
    model: &dyn SklearnClassifier,
    sample: &Array1<f64>,
) -> Result<QuantumInterpretation> {
    // Quantum model interpretation
    let prediction = model.predict(&sample.clone().insert_axis(Axis(0)))?;

    Ok(QuantumInterpretation {
        state_fidelity: 0.92,                            // Mock value
        feature_contributions: vec![0.3, 0.2, 0.4, 0.1], // Mock values
        prediction: prediction[0] as f64,
    })
}

// Supporting structures and trait implementations

struct QuantumFeatureAnalysis {
    advantage_score: f64,
    entanglement_measure: f64,
    circuit_efficiency: f64,
}

struct QuantumInterpretation {
    state_fidelity: f64,
    feature_contributions: Vec<f64>,
    prediction: f64,
}

// Mock implementations for classical sklearn models
struct LogisticRegression {
    fitted: bool,
}

impl LogisticRegression {
    fn new() -> Self {
        Self { fitted: false }
    }
}

impl SklearnEstimator for LogisticRegression {
    fn fit(&mut self, _X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn set_params(&mut self, _params: HashMap<String, String>) -> Result<()> {
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for LogisticRegression {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::InvalidConfiguration(
                "Model not fitted".to_string(),
            ));
        }
        // Mock predictions
        Ok(Array1::from_shape_fn(X.nrows(), |i| {
            if i % 2 == 0 {
                1
            } else {
                0
            }
        }))
    }

    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(Array2::from_shape_fn((X.nrows(), 2), |(i, j)| {
            if j == 0 {
                0.3
            } else {
                0.7
            }
        }))
    }

    fn classes(&self) -> &[i32] {
        &[0, 1]
    }
}

impl SklearnFit for LogisticRegression {
    fn fit(&mut self, _X: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }
}

struct RandomForestClassifier {
    fitted: bool,
}

impl RandomForestClassifier {
    fn new() -> Self {
        Self { fitted: false }
    }
}

impl SklearnEstimator for RandomForestClassifier {
    fn fit(&mut self, _X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn set_params(&mut self, _params: HashMap<String, String>) -> Result<()> {
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for RandomForestClassifier {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::InvalidConfiguration(
                "Model not fitted".to_string(),
            ));
        }
        // Mock predictions with higher accuracy
        Ok(Array1::from_shape_fn(X.nrows(), |i| {
            if (i * 3) % 4 != 0 {
                1
            } else {
                0
            }
        }))
    }

    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(Array2::from_shape_fn((X.nrows(), 2), |(i, j)| {
            if j == 0 {
                0.4
            } else {
                0.6
            }
        }))
    }

    fn classes(&self) -> &[i32] {
        &[0, 1]
    }
}

impl SklearnFit for RandomForestClassifier {
    fn fit(&mut self, _X: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }
}

struct SVC {
    fitted: bool,
}

impl SVC {
    fn new() -> Self {
        Self { fitted: false }
    }
}

impl SklearnEstimator for SVC {
    fn fit(&mut self, _X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn set_params(&mut self, _params: HashMap<String, String>) -> Result<()> {
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClassifier for SVC {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::InvalidConfiguration(
                "Model not fitted".to_string(),
            ));
        }
        // Mock predictions
        Ok(Array1::from_shape_fn(X.nrows(), |i| {
            if i % 3 == 0 {
                0
            } else {
                1
            }
        }))
    }

    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(Array2::from_shape_fn((X.nrows(), 2), |(i, j)| {
            if j == 0 {
                0.35
            } else {
                0.65
            }
        }))
    }

    fn classes(&self) -> &[i32] {
        &[0, 1]
    }
}

impl SklearnFit for SVC {
    fn fit(&mut self, _X: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }
}
