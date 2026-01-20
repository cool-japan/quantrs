//! Sklearn-compatible model selection utilities

use super::{SklearnClassifier, SklearnEstimator};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

/// Cross-validation score
#[allow(non_snake_case)]
pub fn cross_val_score<E>(
    estimator: &mut E,
    X: &Array2<f64>,
    y: &Array1<f64>,
    cv: usize,
) -> Result<Array1<f64>>
where
    E: SklearnClassifier,
{
    let n_samples = X.nrows();
    let fold_size = n_samples / cv;
    let mut scores = Array1::zeros(cv);

    // Create fold indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut thread_rng());

    for fold in 0..cv {
        let start_test = fold * fold_size;
        let end_test = if fold == cv - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Create train/test splits
        let test_indices = &indices[start_test..end_test];
        let train_indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < start_test || *i >= end_test)
            .map(|(_, &idx)| idx)
            .collect();

        // Extract train/test data
        let X_train = X.select(Axis(0), &train_indices);
        let y_train = y.select(Axis(0), &train_indices);
        let X_test = X.select(Axis(0), test_indices);
        let y_test = y.select(Axis(0), test_indices);

        // Convert to i32 for classification
        let y_test_int = y_test.mapv(|x| x.round() as i32);

        // Train and evaluate
        estimator.fit(&X_train, Some(&y_train))?;
        scores[fold] = estimator.score(&X_test, &y_test_int)?;
    }

    Ok(scores)
}

/// Train-test split
#[allow(non_snake_case)]
pub fn train_test_split(
    X: &Array2<f64>,
    y: &Array1<f64>,
    test_size: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
    let n_samples = X.nrows();
    let n_test = (n_samples as f64 * test_size).round() as usize;

    // Create indices
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if let Some(seed) = random_state {
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    } else {
        indices.shuffle(&mut thread_rng());
    }

    let test_indices = &indices[..n_test];
    let train_indices = &indices[n_test..];

    let X_train = X.select(Axis(0), train_indices);
    let X_test = X.select(Axis(0), test_indices);
    let y_train = y.select(Axis(0), train_indices);
    let y_test = y.select(Axis(0), test_indices);

    Ok((X_train, X_test, y_train, y_test))
}

/// Grid search for hyperparameter tuning
pub struct GridSearchCV<E> {
    /// Base estimator
    estimator: E,
    /// Parameter grid
    param_grid: HashMap<String, Vec<String>>,
    /// Cross-validation folds
    cv: usize,
    /// Best parameters
    pub best_params_: HashMap<String, String>,
    /// Best score
    pub best_score_: f64,
    /// Best estimator
    pub best_estimator_: E,
    /// Fitted flag
    fitted: bool,
}

impl<E> GridSearchCV<E>
where
    E: SklearnClassifier + Clone,
{
    /// Create new grid search
    pub fn new(estimator: E, param_grid: HashMap<String, Vec<String>>, cv: usize) -> Self {
        Self {
            best_estimator_: estimator.clone(),
            estimator,
            param_grid,
            cv,
            best_params_: HashMap::new(),
            best_score_: f64::NEG_INFINITY,
            fitted: false,
        }
    }

    /// Fit grid search
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let param_combinations = self.generate_param_combinations();

        for params in param_combinations {
            let mut estimator = self.estimator.clone();
            estimator.set_params(params.clone())?;

            let scores = cross_val_score(&mut estimator, X, y, self.cv)?;
            let mean_score = scores.mean().unwrap_or(0.0);

            if mean_score > self.best_score_ {
                self.best_score_ = mean_score;
                self.best_params_ = params.clone();
                self.best_estimator_ = estimator;
            }
        }

        // Fit best estimator
        if !self.best_params_.is_empty() {
            self.best_estimator_.set_params(self.best_params_.clone())?;
            self.best_estimator_.fit(X, Some(y))?;
        }

        self.fitted = true;
        Ok(())
    }

    /// Generate all parameter combinations
    fn generate_param_combinations(&self) -> Vec<HashMap<String, String>> {
        let mut combinations = vec![HashMap::new()];

        for (param_name, param_values) in &self.param_grid {
            let mut new_combinations = Vec::new();

            for combination in &combinations {
                for value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), value.clone());
                    new_combinations.push(new_combination);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }

    /// Get best parameters
    pub fn best_params(&self) -> &HashMap<String, String> {
        &self.best_params_
    }

    /// Get best score
    pub fn best_score(&self) -> f64 {
        self.best_score_
    }

    /// Predict with best estimator
    #[allow(non_snake_case)]
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }
        self.best_estimator_.predict(X)
    }
}

/// K-Fold cross-validation
pub struct KFold {
    /// Number of folds
    n_splits: usize,
    /// Whether to shuffle
    shuffle: bool,
    /// Random state
    random_state: Option<u64>,
}

impl KFold {
    /// Create new KFold
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Set shuffle
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Split data into folds
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            if let Some(seed) = self.random_state {
                fastrand::seed(seed);
            }
            for i in (1..indices.len()).rev() {
                let j = fastrand::usize(0..=i);
                indices.swap(i, j);
            }
        }

        let fold_size = n_samples / self.n_splits;
        let mut folds = Vec::with_capacity(self.n_splits);

        for fold in 0..self.n_splits {
            let start = fold * fold_size;
            let end = if fold == self.n_splits - 1 {
                n_samples
            } else {
                start + fold_size
            };

            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            folds.push((train_indices, test_indices));
        }

        folds
    }
}

/// Stratified K-Fold cross-validation
pub struct StratifiedKFold {
    /// Number of folds
    n_splits: usize,
    /// Whether to shuffle
    shuffle: bool,
    /// Random state
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create new StratifiedKFold
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Set shuffle
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Split data into stratified folds
    pub fn split(&self, y: &Array1<f64>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n_samples = y.len();

        // Group indices by class
        let mut class_indices: std::collections::HashMap<i64, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, &val) in y.iter().enumerate() {
            let class = val as i64;
            class_indices.entry(class).or_insert_with(Vec::new).push(i);
        }

        // Shuffle within each class if requested
        if self.shuffle {
            if let Some(seed) = self.random_state {
                fastrand::seed(seed);
            }
            for indices in class_indices.values_mut() {
                for i in (1..indices.len()).rev() {
                    let j = fastrand::usize(0..=i);
                    indices.swap(i, j);
                }
            }
        }

        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_splits)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();

        // Distribute samples from each class across folds
        for indices in class_indices.values() {
            let fold_sizes: Vec<usize> = (0..self.n_splits)
                .map(|f| {
                    let base = indices.len() / self.n_splits;
                    if f < indices.len() % self.n_splits {
                        base + 1
                    } else {
                        base
                    }
                })
                .collect();

            let mut current = 0;
            for (fold, &size) in fold_sizes.iter().enumerate() {
                for &idx in &indices[current..current + size] {
                    folds[fold].1.push(idx); // Test indices for this fold
                }
                current += size;
            }
        }

        // Create train indices as complement of test indices
        for fold_idx in 0..self.n_splits {
            let test_set: std::collections::HashSet<usize> =
                folds[fold_idx].1.iter().copied().collect();
            folds[fold_idx].0 = (0..n_samples).filter(|i| !test_set.contains(i)).collect();
        }

        folds
    }
}
