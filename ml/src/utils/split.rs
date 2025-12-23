//! Data splitting utilities for cross-validation and train/test splits

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;

use super::*;
/// Split data into train and test sets
pub fn train_test_split(
    features: &Array2<f64>,
    labels: &Array1<usize>,
    test_ratio: f64,
    shuffle: bool,
) -> Result<(Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>)> {
    if features.nrows() != labels.len() {
        return Err(MLError::InvalidInput(
            "Features and labels must have same number of samples".to_string(),
        ));
    }
    if test_ratio <= 0.0 || test_ratio >= 1.0 {
        return Err(MLError::InvalidInput(
            "Test ratio must be between 0 and 1".to_string(),
        ));
    }
    let n_samples = features.nrows();
    let n_test = (n_samples as f64 * test_ratio) as usize;
    let n_train = n_samples - n_test;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        let mut rng = thread_rng();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }
    let mut train_features = Array2::zeros((n_train, features.ncols()));
    let mut train_labels = Array1::zeros(n_train);
    let mut test_features = Array2::zeros((n_test, features.ncols()));
    let mut test_labels = Array1::zeros(n_test);
    for (i, &idx) in indices[..n_train].iter().enumerate() {
        train_features.row_mut(i).assign(&features.row(idx));
        train_labels[i] = labels[idx];
    }
    for (i, &idx) in indices[n_train..].iter().enumerate() {
        test_features.row_mut(i).assign(&features.row(idx));
        test_labels[i] = labels[idx];
    }
    Ok((train_features, train_labels, test_features, test_labels))
}
/// Split regression data into train and test sets (with continuous labels)
pub fn train_test_split_regression(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    test_ratio: f64,
    shuffle: bool,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>)> {
    if features.nrows() != labels.len() {
        return Err(MLError::InvalidInput(
            "Features and labels must have same number of samples".to_string(),
        ));
    }
    if test_ratio <= 0.0 || test_ratio >= 1.0 {
        return Err(MLError::InvalidInput(
            "Test ratio must be between 0 and 1".to_string(),
        ));
    }
    let n_samples = features.nrows();
    let n_test = (n_samples as f64 * test_ratio) as usize;
    let n_train = n_samples - n_test;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        let mut rng = thread_rng();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }
    let mut train_features = Array2::zeros((n_train, features.ncols()));
    let mut train_labels = Array1::zeros(n_train);
    let mut test_features = Array2::zeros((n_test, features.ncols()));
    let mut test_labels = Array1::zeros(n_test);
    for (i, &idx) in indices[..n_train].iter().enumerate() {
        train_features.row_mut(i).assign(&features.row(idx));
        train_labels[i] = labels[idx];
    }
    for (i, &idx) in indices[n_train..].iter().enumerate() {
        test_features.row_mut(i).assign(&features.row(idx));
        test_labels[i] = labels[idx];
    }
    Ok((train_features, train_labels, test_features, test_labels))
}
/// K-Fold cross-validation split indices generator
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    indices: Vec<usize>,
}
impl KFold {
    /// Create a new K-Fold splitter
    pub fn new(n_samples: usize, n_splits: usize, shuffle: bool) -> Result<Self> {
        if n_splits < 2 {
            return Err(MLError::InvalidInput(
                "Number of splits must be at least 2".to_string(),
            ));
        }
        if n_samples < n_splits {
            return Err(MLError::InvalidInput(format!(
                "Cannot have {} splits for {} samples",
                n_splits, n_samples
            )));
        }
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            let mut rng = thread_rng();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }
        Ok(Self {
            n_splits,
            shuffle,
            indices,
        })
    }
    /// Get the number of splits
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }
    /// Get whether shuffling is enabled
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }
    /// Get train and test indices for a specific fold
    pub fn get_fold(&self, fold: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(MLError::InvalidInput(format!(
                "Fold {} out of range for {} splits",
                fold, self.n_splits
            )));
        }
        let n_samples = self.indices.len();
        let fold_size = n_samples / self.n_splits;
        let n_larger_folds = n_samples % self.n_splits;
        let start = if fold < n_larger_folds {
            fold * (fold_size + 1)
        } else {
            n_larger_folds * (fold_size + 1) + (fold - n_larger_folds) * fold_size
        };
        let end = if fold < n_larger_folds {
            start + fold_size + 1
        } else {
            start + fold_size
        };
        let test_indices: Vec<usize> = self.indices[start..end].to_vec();
        let train_indices: Vec<usize> = self.indices[..start]
            .iter()
            .chain(self.indices[end..].iter())
            .cloned()
            .collect();
        Ok((train_indices, test_indices))
    }
    /// Split features and labels for a specific fold
    pub fn split(
        &self,
        features: &Array2<f64>,
        labels: &Array1<usize>,
        fold: usize,
    ) -> Result<(Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>)> {
        let (train_idx, test_idx) = self.get_fold(fold)?;
        let n_train = train_idx.len();
        let n_test = test_idx.len();
        let n_features = features.ncols();
        let mut train_features = Array2::zeros((n_train, n_features));
        let mut train_labels = Array1::zeros(n_train);
        let mut test_features = Array2::zeros((n_test, n_features));
        let mut test_labels = Array1::zeros(n_test);
        for (i, &idx) in train_idx.iter().enumerate() {
            train_features.row_mut(i).assign(&features.row(idx));
            train_labels[i] = labels[idx];
        }
        for (i, &idx) in test_idx.iter().enumerate() {
            test_features.row_mut(i).assign(&features.row(idx));
            test_labels[i] = labels[idx];
        }
        Ok((train_features, train_labels, test_features, test_labels))
    }
}
/// Stratified K-Fold cross-validation split indices generator
/// Ensures each fold has approximately the same percentage of samples of each class
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    n_splits: usize,
    fold_indices: Vec<Vec<usize>>,
}
impl StratifiedKFold {
    /// Create a new Stratified K-Fold splitter
    pub fn new(labels: &Array1<usize>, n_splits: usize, shuffle: bool) -> Result<Self> {
        if n_splits < 2 {
            return Err(MLError::InvalidInput(
                "Number of splits must be at least 2".to_string(),
            ));
        }
        let n_samples = labels.len();
        if n_samples < n_splits {
            return Err(MLError::InvalidInput(format!(
                "Cannot have {} splits for {} samples",
                n_splits, n_samples
            )));
        }
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }
        if shuffle {
            let mut rng = thread_rng();
            for indices in class_indices.values_mut() {
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
            }
        }
        let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); n_splits];
        for indices in class_indices.values() {
            let n_class = indices.len();
            let fold_size = n_class / n_splits;
            let remainder = n_class % n_splits;
            let mut current_idx = 0;
            for fold in 0..n_splits {
                let size = if fold < remainder {
                    fold_size + 1
                } else {
                    fold_size
                };
                for &idx in &indices[current_idx..current_idx + size] {
                    fold_indices[fold].push(idx);
                }
                current_idx += size;
            }
        }
        Ok(Self {
            n_splits,
            fold_indices,
        })
    }
    /// Get the number of splits
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }
    /// Get train and test indices for a specific fold
    pub fn get_fold(&self, fold: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(MLError::InvalidInput(format!(
                "Fold {} out of range for {} splits",
                fold, self.n_splits
            )));
        }
        let test_indices = self.fold_indices[fold].clone();
        let train_indices: Vec<usize> = self
            .fold_indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != fold)
            .flat_map(|(_, indices)| indices.iter().cloned())
            .collect();
        Ok((train_indices, test_indices))
    }
    /// Split features and labels for a specific fold
    pub fn split(
        &self,
        features: &Array2<f64>,
        labels: &Array1<usize>,
        fold: usize,
    ) -> Result<(Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>)> {
        let (train_idx, test_idx) = self.get_fold(fold)?;
        let n_train = train_idx.len();
        let n_test = test_idx.len();
        let n_features = features.ncols();
        let mut train_features = Array2::zeros((n_train, n_features));
        let mut train_labels = Array1::zeros(n_train);
        let mut test_features = Array2::zeros((n_test, n_features));
        let mut test_labels = Array1::zeros(n_test);
        for (i, &idx) in train_idx.iter().enumerate() {
            train_features.row_mut(i).assign(&features.row(idx));
            train_labels[i] = labels[idx];
        }
        for (i, &idx) in test_idx.iter().enumerate() {
            test_features.row_mut(i).assign(&features.row(idx));
            test_labels[i] = labels[idx];
        }
        Ok((train_features, train_labels, test_features, test_labels))
    }
}
/// Leave-One-Out cross-validation
pub struct LeaveOneOut {
    n_samples: usize,
}
impl LeaveOneOut {
    /// Create a new Leave-One-Out splitter
    pub fn new(n_samples: usize) -> Self {
        Self { n_samples }
    }
    /// Get the number of splits (equal to number of samples)
    pub fn n_splits(&self) -> usize {
        self.n_samples
    }
    /// Get train and test indices for a specific fold
    pub fn get_fold(&self, fold: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_samples {
            return Err(MLError::InvalidInput(format!(
                "Fold {} out of range for {} samples",
                fold, self.n_samples
            )));
        }
        let test_indices = vec![fold];
        let train_indices: Vec<usize> = (0..self.n_samples).filter(|&i| i != fold).collect();
        Ok((train_indices, test_indices))
    }
}
/// Repeated K-Fold cross-validation
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    n_splits: usize,
    n_repeats: usize,
    n_samples: usize,
}
impl RepeatedKFold {
    /// Create a new Repeated K-Fold splitter
    pub fn new(n_samples: usize, n_splits: usize, n_repeats: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(MLError::InvalidInput(
                "Number of splits must be at least 2".to_string(),
            ));
        }
        if n_repeats < 1 {
            return Err(MLError::InvalidInput(
                "Number of repeats must be at least 1".to_string(),
            ));
        }
        if n_samples < n_splits {
            return Err(MLError::InvalidInput(format!(
                "Cannot have {} splits for {} samples",
                n_splits, n_samples
            )));
        }
        Ok(Self {
            n_splits,
            n_repeats,
            n_samples,
        })
    }
    /// Get total number of splits across all repeats
    pub fn total_splits(&self) -> usize {
        self.n_splits * self.n_repeats
    }
    /// Get train and test indices for a specific iteration
    /// The iteration is: repeat * n_splits + fold
    pub fn get_iteration(&self, iteration: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if iteration >= self.total_splits() {
            return Err(MLError::InvalidInput(format!(
                "Iteration {} out of range for {} total splits",
                iteration,
                self.total_splits()
            )));
        }
        let fold = iteration % self.n_splits;
        let kfold = KFold::new(self.n_samples, self.n_splits, true)?;
        kfold.get_fold(fold)
    }
}
/// Time Series Split for temporal data
/// Provides train/test indices to split time series data while preserving temporal order
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    n_samples: usize,
    max_train_size: Option<usize>,
    test_size: Option<usize>,
    gap: usize,
}
impl TimeSeriesSplit {
    /// Create a new Time Series Split
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `n_splits` - Number of splits (must be at least 2)
    /// * `max_train_size` - Maximum size of training set (None for no limit)
    /// * `test_size` - Fixed test set size (None for equal splits)
    /// * `gap` - Number of samples to exclude between train and test
    pub fn new(
        n_samples: usize,
        n_splits: usize,
        max_train_size: Option<usize>,
        test_size: Option<usize>,
        gap: usize,
    ) -> Result<Self> {
        if n_splits < 2 {
            return Err(MLError::InvalidInput(
                "Number of splits must be at least 2".to_string(),
            ));
        }
        if n_samples < n_splits + 1 {
            return Err(MLError::InvalidInput(format!(
                "Cannot have {} splits for {} samples",
                n_splits, n_samples
            )));
        }
        Ok(Self {
            n_splits,
            n_samples,
            max_train_size,
            test_size,
            gap,
        })
    }
    /// Get the number of splits
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }
    /// Get train and test indices for a specific fold
    pub fn get_fold(&self, fold: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(MLError::InvalidInput(format!(
                "Fold {} out of range for {} splits",
                fold, self.n_splits
            )));
        }
        let test_size = self
            .test_size
            .unwrap_or((self.n_samples - self.gap) / (self.n_splits + 1));
        let test_start = (fold + 1) * test_size + self.gap;
        let test_end = (test_start + test_size).min(self.n_samples);
        let train_end = test_start - self.gap;
        let train_start = if let Some(max_size) = self.max_train_size {
            train_end.saturating_sub(max_size)
        } else {
            0
        };
        let train_indices: Vec<usize> = (train_start..train_end).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();
        Ok((train_indices, test_indices))
    }
    /// Split features and labels for a specific fold
    pub fn split(
        &self,
        features: &Array2<f64>,
        labels: &Array1<usize>,
        fold: usize,
    ) -> Result<(Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>)> {
        let (train_idx, test_idx) = self.get_fold(fold)?;
        let n_train = train_idx.len();
        let n_test = test_idx.len();
        let n_features = features.ncols();
        let mut train_features = Array2::zeros((n_train, n_features));
        let mut train_labels = Array1::zeros(n_train);
        let mut test_features = Array2::zeros((n_test, n_features));
        let mut test_labels = Array1::zeros(n_test);
        for (i, &idx) in train_idx.iter().enumerate() {
            train_features.row_mut(i).assign(&features.row(idx));
            train_labels[i] = labels[idx];
        }
        for (i, &idx) in test_idx.iter().enumerate() {
            test_features.row_mut(i).assign(&features.row(idx));
            test_labels[i] = labels[idx];
        }
        Ok((train_features, train_labels, test_features, test_labels))
    }
    /// Split regression features and labels for a specific fold
    pub fn split_regression(
        &self,
        features: &Array2<f64>,
        labels: &Array1<f64>,
        fold: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>)> {
        let (train_idx, test_idx) = self.get_fold(fold)?;
        let n_train = train_idx.len();
        let n_test = test_idx.len();
        let n_features = features.ncols();
        let mut train_features = Array2::zeros((n_train, n_features));
        let mut train_labels = Array1::zeros(n_train);
        let mut test_features = Array2::zeros((n_test, n_features));
        let mut test_labels = Array1::zeros(n_test);
        for (i, &idx) in train_idx.iter().enumerate() {
            train_features.row_mut(i).assign(&features.row(idx));
            train_labels[i] = labels[idx];
        }
        for (i, &idx) in test_idx.iter().enumerate() {
            test_features.row_mut(i).assign(&features.row(idx));
            test_labels[i] = labels[idx];
        }
        Ok((train_features, train_labels, test_features, test_labels))
    }
}
/// Blocked Time Series Split (for grouped temporal data)
/// Splits data into train/test while respecting group boundaries
#[derive(Debug, Clone)]
pub struct BlockedTimeSeriesSplit {
    n_splits: usize,
    group_boundaries: Vec<usize>,
}
impl BlockedTimeSeriesSplit {
    /// Create a new Blocked Time Series Split
    ///
    /// # Arguments
    /// * `group_sizes` - Sizes of each temporal group/block
    /// * `n_splits` - Number of splits
    pub fn new(group_sizes: &[usize], n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(MLError::InvalidInput(
                "Number of splits must be at least 2".to_string(),
            ));
        }
        if group_sizes.len() < n_splits + 1 {
            return Err(MLError::InvalidInput(format!(
                "Need at least {} groups for {} splits",
                n_splits + 1,
                n_splits
            )));
        }
        let mut boundaries = vec![0];
        let mut cumsum = 0;
        for &size in group_sizes {
            cumsum += size;
            boundaries.push(cumsum);
        }
        Ok(Self {
            n_splits,
            group_boundaries: boundaries,
        })
    }
    /// Get the number of splits
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }
    /// Get train and test indices for a specific fold
    pub fn get_fold(&self, fold: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(MLError::InvalidInput(format!(
                "Fold {} out of range for {} splits",
                fold, self.n_splits
            )));
        }
        let n_groups = self.group_boundaries.len() - 1;
        let groups_per_fold = n_groups / (self.n_splits + 1);
        let train_end_group = (fold + 1) * groups_per_fold;
        let test_end_group = (train_end_group + groups_per_fold).min(n_groups);
        let train_start = self.group_boundaries[0];
        let train_end = self.group_boundaries[train_end_group];
        let test_start = train_end;
        let test_end = self.group_boundaries[test_end_group];
        let train_indices: Vec<usize> = (train_start..train_end).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();
        Ok((train_indices, test_indices))
    }
}
