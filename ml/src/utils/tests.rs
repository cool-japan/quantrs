//! Tests for utils modules

use super::*;
use scirs2_core::ndarray::array;
#[test]
fn test_standardize() {
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let normalized = preprocessing::standardize(&data).unwrap();
    let mean = normalized.mean_axis(Axis(0)).unwrap();
    assert!(mean[0].abs() < 1e-10);
    assert!(mean[1].abs() < 1e-10);
}
#[test]
fn test_min_max_normalize() {
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let normalized = preprocessing::min_max_normalize(&data).unwrap();
    for val in normalized.iter() {
        assert!(*val >= 0.0 && *val <= 1.0);
    }
    assert!((normalized[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((normalized[(2, 0)] - 1.0).abs() < 1e-10);
}
#[test]
fn test_accuracy() {
    let predictions = array![0, 1, 2, 0, 1];
    let labels = array![0, 1, 0, 0, 1];
    let acc = metrics::accuracy(&predictions, &labels);
    assert_eq!(acc, 0.8);
}
#[test]
fn test_mse_and_rmse() {
    let predictions = array![1.0, 2.0, 3.0, 4.0];
    let targets = array![1.0, 2.0, 3.0, 4.0];
    assert!((metrics::mse(&predictions, &targets) - 0.0).abs() < 1e-10);
    assert!((metrics::rmse(&predictions, &targets) - 0.0).abs() < 1e-10);
    let predictions2 = array![1.0, 2.0, 3.0, 5.0];
    let mse = metrics::mse(&predictions2, &targets);
    assert!((mse - 0.25).abs() < 1e-10);
}
#[test]
fn test_r2_score() {
    let predictions = array![1.0, 2.0, 3.0, 4.0];
    let targets = array![1.0, 2.0, 3.0, 4.0];
    let r2 = metrics::r2_score(&predictions, &targets);
    assert!((r2 - 1.0).abs() < 1e-10);
}
#[test]
fn test_confusion_matrix() {
    let predictions = array![0, 1, 1, 0, 1];
    let labels = array![0, 1, 0, 0, 1];
    let cm = metrics::confusion_matrix(&predictions, &labels, 2).unwrap();
    assert_eq!(cm[(0, 0)], 2);
    assert_eq!(cm[(0, 1)], 1);
    assert_eq!(cm[(1, 0)], 0);
    assert_eq!(cm[(1, 1)], 2);
}
#[test]
fn test_precision_recall_f1() {
    let predictions = array![0, 1, 1, 0, 1, 0];
    let labels = array![0, 1, 0, 0, 1, 1];
    let prec = metrics::precision(&predictions, &labels, 2).unwrap();
    let rec = metrics::recall(&predictions, &labels, 2).unwrap();
    let f1 = metrics::f1_score(&predictions, &labels, 2).unwrap();
    assert!((prec[1] - 2.0 / 3.0).abs() < 1e-10);
    assert!((rec[1] - 2.0 / 3.0).abs() < 1e-10);
    assert!((f1[1] - 2.0 / 3.0).abs() < 1e-10);
}
#[test]
fn test_auc_roc() {
    let scores = array![0.9, 0.8, 0.4, 0.3, 0.2];
    let labels = array![1, 1, 0, 0, 0];
    let auc = metrics::auc_roc(&scores, &labels).unwrap();
    assert!((auc - 1.0).abs() < 1e-10);
}
#[test]
fn test_matthews_corrcoef() {
    let predictions = array![1, 1, 0, 0];
    let labels = array![1, 1, 0, 0];
    let mcc = metrics::matthews_corrcoef(&predictions, &labels);
    assert!((mcc - 1.0).abs() < 1e-10);
}
#[test]
fn test_cohens_kappa() {
    let predictions = array![0, 1, 0, 1, 0, 1];
    let labels = array![0, 1, 0, 1, 0, 1];
    let kappa = metrics::cohens_kappa(&predictions, &labels);
    assert!((kappa - 1.0).abs() < 1e-10);
}
#[test]
fn test_train_test_split() {
    let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let labels = array![0, 1, 0, 1];
    let (train_features, train_labels, test_features, test_labels) =
        split::train_test_split(&features, &labels, 0.5, false).unwrap();
    assert_eq!(train_features.nrows(), 2);
    assert_eq!(test_features.nrows(), 2);
    assert_eq!(train_labels.len(), 2);
    assert_eq!(test_labels.len(), 2);
}
#[test]
fn test_train_test_split_regression() {
    let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let labels = array![1.5, 2.5, 3.5, 4.5];
    let (train_features, train_labels, test_features, test_labels) =
        split::train_test_split_regression(&features, &labels, 0.25, false).unwrap();
    assert_eq!(train_features.nrows(), 3);
    assert_eq!(test_features.nrows(), 1);
    assert_eq!(train_labels.len(), 3);
    assert_eq!(test_labels.len(), 1);
}
#[test]
fn test_kfold() {
    let n_samples = 10;
    let n_splits = 5;
    let kfold = split::KFold::new(n_samples, n_splits, false).unwrap();
    assert_eq!(kfold.n_splits(), 5);
    let mut test_counts = vec![0; n_samples];
    for fold in 0..n_splits {
        let (train_idx, test_idx) = kfold.get_fold(fold).unwrap();
        assert_eq!(train_idx.len() + test_idx.len(), n_samples);
        for &idx in &test_idx {
            test_counts[idx] += 1;
        }
    }
    for count in test_counts {
        assert_eq!(count, 1);
    }
}
#[test]
fn test_stratified_kfold() {
    let labels = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
    let n_splits = 2;
    let skf = split::StratifiedKFold::new(&labels, n_splits, false).unwrap();
    for fold in 0..n_splits {
        let (train_idx, test_idx) = skf.get_fold(fold).unwrap();
        let mut class_0_count = 0;
        let mut class_1_count = 0;
        for &idx in &test_idx {
            if labels[idx] == 0 {
                class_0_count += 1;
            } else {
                class_1_count += 1;
            }
        }
        assert!(class_0_count >= 2 && class_0_count <= 4);
        assert!(class_1_count >= 1 && class_1_count <= 3);
        assert_eq!(train_idx.len() + test_idx.len(), 10);
    }
}
#[test]
fn test_leave_one_out() {
    let n_samples = 5;
    let loo = split::LeaveOneOut::new(n_samples);
    assert_eq!(loo.n_splits(), 5);
    for fold in 0..n_samples {
        let (train_idx, test_idx) = loo.get_fold(fold).unwrap();
        assert_eq!(train_idx.len(), 4);
        assert_eq!(test_idx.len(), 1);
        assert_eq!(test_idx[0], fold);
    }
}
#[test]
fn test_basis_encode() {
    let data = array![0, 1, 2, 3];
    let encoded = encoding::basis_encode(&data, 2).unwrap();
    assert_eq!(encoded[(0, 0)], 0);
    assert_eq!(encoded[(0, 1)], 0);
    assert_eq!(encoded[(1, 0)], 0);
    assert_eq!(encoded[(1, 1)], 1);
    assert_eq!(encoded[(2, 0)], 1);
    assert_eq!(encoded[(2, 1)], 0);
    assert_eq!(encoded[(3, 0)], 1);
    assert_eq!(encoded[(3, 1)], 1);
}
#[test]
fn test_product_encode() {
    let data = array![1.0, 2.0, 3.0];
    let encoded = encoding::product_encode(&data).unwrap();
    for val in encoded.iter() {
        assert!(*val >= 0.0 && *val <= std::f64::consts::PI + 1e-10);
    }
}
#[test]
fn test_iqp_encode() {
    let data = array![1.0, 2.0];
    let encoded = encoding::iqp_encode(&data, 2).unwrap();
    assert_eq!(encoded.len(), 5);
    assert!((encoded[0] - 1.0).abs() < 1e-10);
    assert!((encoded[1] - 2.0).abs() < 1e-10);
    assert!((encoded[2] - 1.0).abs() < 1e-10);
    assert!((encoded[3] - 2.0).abs() < 1e-10);
    assert!((encoded[4] - 4.0).abs() < 1e-10);
}
#[test]
fn test_dense_angle_encode() {
    let data = array![0.0, 1.0, -1.0];
    let encoded = encoding::dense_angle_encode(&data).unwrap();
    for val in encoded.iter() {
        assert!(*val >= 0.0 && *val <= 2.0 * std::f64::consts::PI + 1e-10);
    }
}
#[test]
fn test_amplitude_encode() {
    let data = array![3.0, 4.0];
    let encoded = encoding::amplitude_encode(&data).unwrap();
    let norm: f64 = encoded.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}
#[test]
fn test_log_loss() {
    let probabilities = array![0.9, 0.1, 0.8, 0.3];
    let labels = array![1, 0, 1, 0];
    let loss = metrics::log_loss(&probabilities, &labels);
    assert!(loss < 0.5);
}
#[test]
fn test_repeated_kfold() {
    let n_samples = 10;
    let n_splits = 2;
    let n_repeats = 3;
    let rkf = split::RepeatedKFold::new(n_samples, n_splits, n_repeats).unwrap();
    assert_eq!(rkf.total_splits(), 6);
}
#[test]
fn test_time_series_split() {
    let n_samples = 20;
    let n_splits = 4;
    let tss = split::TimeSeriesSplit::new(n_samples, n_splits, None, None, 0).unwrap();
    assert_eq!(tss.n_splits(), 4);
    let mut prev_test_end = 0;
    for fold in 0..n_splits {
        let (train_idx, test_idx) = tss.get_fold(fold).unwrap();
        if !train_idx.is_empty() && !test_idx.is_empty() {
            assert!(*train_idx.last().unwrap() < *test_idx.first().unwrap());
        }
        if !test_idx.is_empty() {
            assert!(*test_idx.first().unwrap() >= prev_test_end);
            prev_test_end = *test_idx.last().unwrap() + 1;
        }
    }
}
#[test]
fn test_time_series_split_with_gap() {
    let n_samples = 30;
    let n_splits = 3;
    let gap = 2;
    let tss = split::TimeSeriesSplit::new(n_samples, n_splits, None, None, gap).unwrap();
    for fold in 0..n_splits {
        let (train_idx, test_idx) = tss.get_fold(fold).unwrap();
        if !train_idx.is_empty() && !test_idx.is_empty() {
            let train_end = *train_idx.last().unwrap();
            let test_start = *test_idx.first().unwrap();
            assert!(test_start >= train_end + gap);
        }
    }
}
#[test]
fn test_blocked_time_series_split() {
    let group_sizes = vec![5, 5, 5, 5, 5, 5];
    let n_splits = 2;
    let btss = split::BlockedTimeSeriesSplit::new(&group_sizes, n_splits).unwrap();
    assert_eq!(btss.n_splits(), 2);
    for fold in 0..n_splits {
        let (train_idx, test_idx) = btss.get_fold(fold).unwrap();
        assert!(!train_idx.is_empty());
        assert!(!test_idx.is_empty());
    }
}
#[test]
fn test_auc_roc_multiclass() {
    let scores = array![
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.2, 0.7]
    ];
    let labels = array![0, 0, 1, 1, 2, 2];
    let aucs = metrics::auc_roc_ovr(&scores, &labels, 3).unwrap();
    for &auc in aucs.iter() {
        assert!(auc > 0.8 || auc.is_nan());
    }
    let macro_auc = metrics::auc_roc_macro(&scores, &labels, 3).unwrap();
    assert!(macro_auc > 0.8);
    let weighted_auc = metrics::auc_roc_weighted(&scores, &labels, 3).unwrap();
    assert!(weighted_auc > 0.8);
}
#[test]
fn test_brier_score() {
    let probabilities = array![0.9, 0.1, 0.8, 0.2];
    let labels = array![1, 0, 1, 0];
    let brier = metrics::brier_score(&probabilities, &labels);
    assert!(brier < 0.05);
    let perfect_probs = array![1.0, 0.0, 1.0, 0.0];
    let perfect_brier = metrics::brier_score(&perfect_probs, &labels);
    assert!((perfect_brier - 0.0).abs() < 1e-10);
}
#[test]
fn test_balanced_accuracy() {
    let predictions = array![0, 0, 0, 1, 1, 1];
    let labels = array![0, 0, 0, 1, 1, 1];
    let ba = metrics::balanced_accuracy(&predictions, &labels, 2).unwrap();
    assert!((ba - 1.0).abs() < 1e-10);
}
#[test]
fn test_top_k_accuracy() {
    let scores = array![
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
        [0.3, 0.3, 0.4]
    ];
    let labels = array![1, 0, 2, 0];
    let top1 = metrics::top_k_accuracy(&scores, &labels, 1).unwrap();
    assert!((top1 - 0.75).abs() < 1e-10);
    let top2 = metrics::top_k_accuracy(&scores, &labels, 2).unwrap();
    assert!(top2 >= top1);
}
#[test]
fn test_robust_scale() {
    let data = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [100.0, 400.0]];
    let scaled = preprocessing::robust_scale(&data).unwrap();
    let mut col0: Vec<f64> = scaled.column(0).to_vec();
    let mut col1: Vec<f64> = scaled.column(1).to_vec();
    col0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    col1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median0 = (col0[1] + col0[2]) / 2.0;
    let median1 = (col1[1] + col1[2]) / 2.0;
    assert!(median0.abs() < 2.0);
    assert!(median1.abs() < 2.0);
}
#[test]
fn test_quantile_normalize() {
    let data = array![[1.0, 5.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
    let normalized = preprocessing::quantile_normalize(&data).unwrap();
    let mean0 = normalized.column(0).mean().unwrap();
    let mean1 = normalized.column(1).mean().unwrap();
    assert!((mean0 - mean1).abs() < 1.0);
}
#[test]
fn test_max_abs_scale() {
    let data = array![[-2.0, 1.0], [-1.0, 2.0], [0.0, 0.0], [1.0, 3.0]];
    let scaled = preprocessing::max_abs_scale(&data).unwrap();
    let max_abs0 = scaled
        .column(0)
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max);
    let max_abs1 = scaled
        .column(1)
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max);
    assert!((max_abs0 - 1.0).abs() < 1e-10);
    assert!((max_abs1 - 1.0).abs() < 1e-10);
}
#[test]
fn test_l1_normalize() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let normalized = preprocessing::l1_normalize(&data).unwrap();
    for i in 0..normalized.nrows() {
        let row_sum: f64 = normalized.row(i).iter().map(|x| x.abs()).sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }
}
#[test]
fn test_l2_normalize() {
    let data = array![[3.0, 4.0], [5.0, 12.0]];
    let normalized = preprocessing::l2_normalize(&data).unwrap();
    for i in 0..normalized.nrows() {
        let row_norm: f64 = normalized.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((row_norm - 1.0).abs() < 1e-10);
    }
}
#[test]
fn test_platt_scaler_basic() {
    let scores = array![2.0, 1.5, 1.0, -1.0, -1.5, -2.0];
    let labels = array![1, 1, 1, 0, 0, 0];
    let mut scaler = calibration::PlattScaler::new();
    scaler.fit(&scores, &labels).unwrap();
    let params = scaler.parameters();
    assert!(params.is_some());
    let probs = scaler.transform(&scores).unwrap();
    for &p in probs.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }
    assert!(probs[0] > probs[5]);
}
#[test]
fn test_platt_scaler_fit_transform() {
    let scores = array![3.0, 2.0, 1.0, -1.0, -2.0, -3.0];
    let labels = array![1, 1, 1, 0, 0, 0];
    let mut scaler = calibration::PlattScaler::new();
    let probs = scaler.fit_transform(&scores, &labels).unwrap();
    for &p in probs.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }
    assert!(probs[0] > 0.5);
    assert!(probs[5] < 0.5);
}
#[test]
fn test_isotonic_regression_basic() {
    let scores = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let labels = array![0, 0, 0, 0, 1, 1, 1, 1, 1];
    let mut iso = calibration::IsotonicRegression::new();
    iso.fit(&scores, &labels).unwrap();
    let probs = iso.transform(&scores).unwrap();
    for &p in probs.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }
    for i in 0..probs.len() - 1 {
        assert!(probs[i] <= probs[i + 1] + 1e-10);
    }
}
#[test]
fn test_isotonic_regression_fit_transform() {
    let scores = array![0.1, 0.3, 0.2, 0.6, 0.5, 0.8, 0.7, 0.9];
    let labels = array![0, 0, 0, 1, 0, 1, 1, 1];
    let mut iso = calibration::IsotonicRegression::new();
    let probs = iso.fit_transform(&scores, &labels).unwrap();
    for &p in probs.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }
}
#[test]
fn test_calibration_curve() {
    let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let labels = array![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let (mean_pred, frac_pos) = calibration::calibration_curve(&probabilities, &labels, 5).unwrap();
    assert!(mean_pred.len() > 0);
    assert_eq!(mean_pred.len(), frac_pos.len());
    for &p in mean_pred.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }
    for &f in frac_pos.iter() {
        assert!(f >= 0.0 && f <= 1.0);
    }
}
#[test]
fn test_platt_scaler_error_handling() {
    let mut scaler = calibration::PlattScaler::new();
    let scores = array![1.0, 2.0, 3.0];
    assert!(scaler.transform(&scores).is_err());
    let scores = array![1.0, 2.0, 3.0];
    let labels = array![0, 1];
    assert!(scaler.fit(&scores, &labels).is_err());
    let scores = array![1.0];
    let labels = array![0];
    assert!(scaler.fit(&scores, &labels).is_err());
}
#[test]
fn test_isotonic_regression_error_handling() {
    let mut iso = calibration::IsotonicRegression::new();
    let scores = array![1.0, 2.0, 3.0];
    assert!(iso.transform(&scores).is_err());
    let scores = array![1.0, 2.0, 3.0];
    let labels = array![0, 1];
    assert!(iso.fit(&scores, &labels).is_err());
    let scores = array![1.0];
    let labels = array![0];
    assert!(iso.fit(&scores, &labels).is_err());
}
#[test]
fn test_calibration_curve_error_handling() {
    let probs = array![0.1, 0.2, 0.3];
    let labels = array![0, 1];
    assert!(calibration::calibration_curve(&probs, &labels, 5).is_err());
    let probs = array![0.1, 0.2, 0.3];
    let labels = array![0, 1, 0];
    assert!(calibration::calibration_curve(&probs, &labels, 1).is_err());
}

#[test]
fn test_temperature_scaler_basic() {
    // Create 3-class classification logits
    let logits = array![
        [2.0, 1.0, 0.5],
        [0.5, 2.0, 1.0],
        [1.0, 0.5, 2.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
    ];
    let labels = array![0, 1, 2, 0, 1, 2];

    let mut scaler = calibration::TemperatureScaler::new();
    scaler.fit(&logits, &labels).unwrap();

    // Check that temperature was fitted
    let temp = scaler.temperature();
    assert!(temp.is_some());
    assert!(temp.unwrap() > 0.0);

    // Transform logits
    let probs = scaler.transform(&logits).unwrap();

    // Check probabilities sum to 1 for each sample
    for i in 0..probs.nrows() {
        let row_sum: f64 = probs.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }

    // Check all probabilities are in [0, 1]
    for &p in probs.iter() {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

#[test]
fn test_temperature_scaler_fit_transform() {
    let logits = array![
        [3.0, 1.0, 0.0],
        [1.0, 3.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 0.0, 0.5],
    ];
    let labels = array![0, 1, 2, 0];

    let mut scaler = calibration::TemperatureScaler::new();
    let probs = scaler.fit_transform(&logits, &labels).unwrap();

    // Verify output shape
    assert_eq!(probs.nrows(), logits.nrows());
    assert_eq!(probs.ncols(), logits.ncols());

    // Verify probabilities
    for i in 0..probs.nrows() {
        let row_sum: f64 = probs.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_temperature_scaler_calibration_effect() {
    // Create overconfident logits (high magnitude)
    let logits = array![[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0],];
    let labels = array![0, 1, 2];

    let mut scaler = calibration::TemperatureScaler::new();
    scaler.fit(&logits, &labels).unwrap();

    let calibrated_probs = scaler.transform(&logits).unwrap();

    // Temperature must be positive
    let temp = scaler.temperature().unwrap();
    assert!(temp > 0.0);

    // Verify probabilities are valid
    for i in 0..calibrated_probs.nrows() {
        let row_sum: f64 = calibrated_probs.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);

        for &p in calibrated_probs.row(i).iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }
}

#[test]
fn test_temperature_scaler_error_handling() {
    let mut scaler = calibration::TemperatureScaler::new();

    // Should error when transforming before fitting
    let logits = array![[1.0, 2.0, 3.0]];
    assert!(scaler.transform(&logits).is_err());

    // Should error with mismatched lengths
    let logits = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let labels = array![0, 1];
    assert!(scaler.fit(&logits, &labels).is_err());

    // Should error with too few samples
    let logits = array![[1.0, 2.0]];
    let labels = array![0];
    assert!(scaler.fit(&logits, &labels).is_err());
}

#[test]
fn test_temperature_scaler_multiclass() {
    // Test with 5-class problem
    let logits = array![
        [2.0, 1.0, 0.5, 0.0, -0.5],
        [1.0, 2.0, 1.0, 0.5, 0.0],
        [0.5, 1.0, 2.0, 1.0, 0.5],
        [0.0, 0.5, 1.0, 2.0, 1.0],
        [-0.5, 0.0, 0.5, 1.0, 2.0],
    ];
    let labels = array![0, 1, 2, 3, 4];

    let mut scaler = calibration::TemperatureScaler::new();
    let probs = scaler.fit_transform(&logits, &labels).unwrap();

    // Check dimensions
    assert_eq!(probs.nrows(), 5);
    assert_eq!(probs.ncols(), 5);

    // Check probability constraints
    for i in 0..probs.nrows() {
        let row_sum: f64 = probs.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);

        for &p in probs.row(i).iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }
}

#[test]
fn test_temperature_scaler_vs_uncalibrated() {
    // Test that temperature scaling produces valid probabilities
    let logits = array![[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0],];
    let labels = array![0, 1, 2];

    // Compute calibrated probabilities
    let mut scaler = calibration::TemperatureScaler::new();
    let calibrated = scaler.fit_transform(&logits, &labels).unwrap();

    // Verify calibrated probabilities are valid
    for i in 0..3 {
        let row_sum: f64 = calibrated.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);

        // All probabilities should be in [0, 1]
        for &p in calibrated.row(i).iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }

        // Check that predictions are still correct
        let pred_class = calibrated
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(pred_class, labels[i]);
    }
}

#[test]
fn test_expected_calibration_error() {
    // Well-calibrated probabilities
    let probs_calibrated = array![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 1];

    let ece = metrics::expected_calibration_error(&probs_calibrated, &labels, 5).unwrap();
    // Well-calibrated should have low ECE
    assert!(ece < 0.3);

    // Poorly calibrated probabilities (all high confidence)
    let probs_uncalibrated = array![0.9, 0.9, 0.9, 0.9, 0.9];
    let ece_bad = metrics::expected_calibration_error(&probs_uncalibrated, &labels, 5).unwrap();
    // Poorly calibrated should have higher ECE
    assert!(ece_bad > ece);
}

#[test]
fn test_maximum_calibration_error() {
    let probabilities = array![0.1, 0.2, 0.8, 0.9];
    let labels = array![0, 0, 1, 1];

    let mce = metrics::maximum_calibration_error(&probabilities, &labels, 4).unwrap();
    // MCE should be non-negative
    assert!(mce >= 0.0);
    assert!(mce <= 1.0);
}

#[test]
fn test_negative_log_likelihood() {
    // 3-class perfect predictions
    let probs = array![[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]];
    let labels = array![0, 1, 2];

    let nll = metrics::negative_log_likelihood(&probs, &labels).unwrap();
    // Good predictions should have low NLL
    assert!(nll < 0.5);

    // Poor predictions
    let probs_bad = array![[0.1, 0.45, 0.45], [0.45, 0.1, 0.45], [0.45, 0.45, 0.1]];
    let nll_bad = metrics::negative_log_likelihood(&probs_bad, &labels).unwrap();
    // Poor predictions should have higher NLL
    assert!(nll_bad > nll);
}

#[test]
fn test_brier_score_decomposition() {
    let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 1];

    let (brier, reliability, resolution, uncertainty) =
        metrics::brier_score_decomposition(&probabilities, &labels, 5).unwrap();

    // All components should be non-negative
    assert!(brier >= 0.0);
    assert!(reliability >= 0.0);
    assert!(resolution >= 0.0);
    assert!(uncertainty >= 0.0);

    // Brier score = Reliability - Resolution + Uncertainty
    assert!((brier - (reliability - resolution + uncertainty)).abs() < 1e-6);
}

#[test]
fn test_calibration_error_confidence_interval() {
    let probabilities = array![0.2, 0.4, 0.6, 0.8];
    let labels = array![0, 0, 1, 1];

    let (mean_ece, lower, upper) =
        metrics::calibration_error_confidence_interval(&probabilities, &labels, 4, 50, 0.95)
            .unwrap();

    // Mean should be between bounds
    assert!(mean_ece >= lower - 1e-6);
    assert!(mean_ece <= upper + 1e-6);

    // Bounds should be in valid range
    assert!(lower >= 0.0);
    assert!(upper <= 1.0);
}

#[test]
fn test_calibration_metrics_error_handling() {
    let probs = array![0.1, 0.2, 0.3];
    let labels = array![0, 1];

    // Mismatched lengths should error
    assert!(metrics::expected_calibration_error(&probs, &labels, 5).is_err());
    assert!(metrics::maximum_calibration_error(&probs, &labels, 5).is_err());

    // Invalid n_bins should error
    let probs_valid = array![0.1, 0.2];
    let labels_valid = array![0, 1];
    assert!(metrics::expected_calibration_error(&probs_valid, &labels_valid, 1).is_err());
}

// ==================== New Calibration Method Tests ====================

#[test]
fn test_vector_scaler_basic() {
    use crate::utils::calibration::VectorScaler;
    use scirs2_core::ndarray::Array2;

    // Create simple 3-class logits
    let logits = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.5, 0.5, 0.5],
    )
    .unwrap();
    let labels = array![0, 1, 2, 0];

    let mut scaler = VectorScaler::new();
    let result = scaler.fit(&logits, &labels);
    assert!(result.is_ok());

    let calibrated = scaler.transform(&logits).unwrap();
    assert_eq!(calibrated.shape(), logits.shape());

    // Check probabilities sum to 1
    for i in 0..calibrated.nrows() {
        let row_sum: f64 = calibrated.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }

    // Check parameters are available
    assert!(scaler.parameters().is_some());
}

#[test]
fn test_vector_scaler_fit_transform() {
    use crate::utils::calibration::VectorScaler;
    use scirs2_core::ndarray::Array2;

    let logits = Array2::from_shape_vec((3, 2), vec![2.0, 1.0, 1.0, 3.0, 2.5, 0.5]).unwrap();
    let labels = array![0, 1, 0];

    let mut scaler = VectorScaler::new();
    let calibrated = scaler.fit_transform(&logits, &labels).unwrap();

    assert_eq!(calibrated.nrows(), 3);
    assert_eq!(calibrated.ncols(), 2);

    // Probabilities should be in [0, 1]
    for &val in calibrated.iter() {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_vector_scaler_error_handling() {
    use crate::utils::calibration::VectorScaler;
    use scirs2_core::ndarray::Array2;

    let mut scaler = VectorScaler::new();

    // Mismatched dimensions
    let logits = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();
    let labels = array![0, 1]; // Wrong length
    assert!(scaler.fit(&logits, &labels).is_err());

    // Transform before fit
    let scaler2 = VectorScaler::new();
    assert!(scaler2.transform(&logits).is_err());
}

#[test]
fn test_bayesian_binning_quantiles_basic() {
    use crate::utils::calibration::BayesianBinningQuantiles;

    let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95];
    let labels = array![0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

    let mut bbq = BayesianBinningQuantiles::new(5);
    let result = bbq.fit(&probabilities, &labels);
    assert!(result.is_ok());

    let calibrated = bbq.transform(&probabilities).unwrap();
    assert_eq!(calibrated.len(), probabilities.len());

    // Calibrated probabilities should be in [0, 1]
    for &val in calibrated.iter() {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_bayesian_binning_quantiles_fit_transform() {
    use crate::utils::calibration::BayesianBinningQuantiles;

    let probabilities = array![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 0, 1, 1, 1];

    let mut bbq = BayesianBinningQuantiles::new(4);
    let calibrated = bbq.fit_transform(&probabilities, &labels).unwrap();

    assert_eq!(calibrated.len(), probabilities.len());

    // Check bin statistics are available
    assert!(bbq.bin_statistics().is_some());
}

#[test]
fn test_bayesian_binning_quantiles_with_uncertainty() {
    use crate::utils::calibration::BayesianBinningQuantiles;

    let probabilities = array![0.2, 0.5, 0.8, 0.9];
    let labels = array![0, 0, 1, 1];

    let mut bbq = BayesianBinningQuantiles::new(4);
    bbq.fit(&probabilities, &labels).unwrap();

    let results = bbq.predict_with_uncertainty(&probabilities, 0.95).unwrap();

    assert_eq!(results.len(), probabilities.len());

    // Check each result has (mean, lower, upper)
    for (mean, lower, upper) in results {
        assert!(mean >= 0.0 && mean <= 1.0);
        assert!(lower >= 0.0 && lower <= 1.0);
        assert!(upper >= 0.0 && upper <= 1.0);
        assert!(lower <= mean);
        assert!(mean <= upper);
    }
}

#[test]
fn test_bayesian_binning_quantiles_error_handling() {
    use crate::utils::calibration::BayesianBinningQuantiles;

    let mut bbq = BayesianBinningQuantiles::new(5);

    // Mismatched lengths
    let probs = array![0.1, 0.2, 0.3];
    let labels = array![0, 1];
    assert!(bbq.fit(&probs, &labels).is_err());

    // Too few samples for bins
    let probs2 = array![0.1, 0.2];
    let labels2 = array![0, 1];
    assert!(bbq.fit(&probs2, &labels2).is_err());

    // Transform before fit
    let bbq2 = BayesianBinningQuantiles::new(3);
    assert!(bbq2.transform(&probs).is_err());
}

#[test]
fn test_calibration_visualization_plot_data() {
    use crate::utils::calibration::visualization::generate_calibration_plot_data;

    let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8];
    let labels = array![0, 0, 1, 1, 1, 0, 0, 1, 1];

    let plot_data = generate_calibration_plot_data(&probabilities, &labels, 5).unwrap();

    assert_eq!(plot_data.mean_predicted.len(), 5);
    assert_eq!(plot_data.fraction_positives.len(), 5);
    assert_eq!(plot_data.bin_counts.len(), 5);
    assert_eq!(plot_data.bin_edges.len(), 6); // n_bins + 1

    // Check bin edges are monotonic
    for i in 0..plot_data.bin_edges.len() - 1 {
        assert!(plot_data.bin_edges[i] <= plot_data.bin_edges[i + 1]);
    }
}

#[test]
fn test_calibration_analysis() {
    use crate::utils::calibration::visualization::analyze_calibration;

    let probabilities = array![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 0, 1, 1, 1];

    let analysis = analyze_calibration(&probabilities, &labels, 4).unwrap();

    assert!(analysis.ece >= 0.0 && analysis.ece <= 1.0);
    assert!(analysis.mce >= 0.0 && analysis.mce <= 1.0);
    assert!(analysis.brier_score >= 0.0 && analysis.brier_score <= 1.0);
    assert!(analysis.nll >= 0.0);
    assert_eq!(analysis.n_bins, 4);
    assert!(!analysis.interpretation.is_empty());
}

#[test]
fn test_calibration_comparison() {
    use crate::utils::calibration::visualization::compare_calibration_methods;

    let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95];
    let labels = array![0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

    let comparisons = compare_calibration_methods(&probabilities, &labels, 5).unwrap();

    assert!(comparisons.len() >= 1); // At least uncalibrated baseline

    // Check first comparison (uncalibrated)
    assert_eq!(comparisons[0].method_name, "Uncalibrated");
    assert_eq!(comparisons[0].calibrated_probs.len(), probabilities.len());
}

#[test]
fn test_calibration_comparison_report() {
    use crate::utils::calibration::visualization::{
        compare_calibration_methods, generate_comparison_report,
    };

    let probabilities = array![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 0, 1, 1, 1];

    let comparisons = compare_calibration_methods(&probabilities, &labels, 4).unwrap();
    let report = generate_comparison_report(&comparisons);

    assert!(!report.is_empty());
    assert!(report.contains("Calibration Methods Comparison Report"));
    assert!(report.contains("Recommendations"));
}

#[test]
fn test_quantum_calibrator_binary() {
    use crate::utils::calibration::quantum_calibration::{
        CalibrationMethod, QuantumNeuralNetworkCalibrator,
    };
    use crate::utils::calibration::PlattScaler;

    let probabilities = array![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 0, 1, 1, 1];
    let shot_counts = array![1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000];

    let mut calibrator =
        QuantumNeuralNetworkCalibrator::with_method(CalibrationMethod::Platt(PlattScaler::new()));

    let result = calibrator.fit_binary(&probabilities, &labels, Some(&shot_counts));
    assert!(result.is_ok());

    let calibrated = calibrator.transform_binary(&probabilities).unwrap();
    assert_eq!(calibrated.len(), probabilities.len());

    // Check probabilities are in valid range
    for &val in calibrated.iter() {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_quantum_calibrator_multiclass() {
    use crate::utils::calibration::quantum_calibration::{
        CalibrationMethod, QuantumNeuralNetworkCalibrator,
    };
    use crate::utils::calibration::TemperatureScaler;
    use scirs2_core::ndarray::Array2;

    let logits = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.5, 0.5, 0.5],
    )
    .unwrap();
    let labels = array![0, 1, 2, 0];
    let shot_counts = array![1000, 1000, 1000, 1000];

    let mut calibrator = QuantumNeuralNetworkCalibrator::with_method(
        CalibrationMethod::Temperature(TemperatureScaler::new()),
    );

    let result = calibrator.fit_multiclass(&logits, &labels, Some(&shot_counts));
    assert!(result.is_ok());

    let calibrated = calibrator.transform_multiclass(&logits).unwrap();
    assert_eq!(calibrated.shape(), logits.shape());
}

#[test]
fn test_quantum_calibrator_with_uncertainty() {
    use crate::utils::calibration::quantum_calibration::{
        CalibrationMethod, QuantumNeuralNetworkCalibrator,
    };
    use crate::utils::calibration::BayesianBinningQuantiles;

    let probabilities = array![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.15, 0.85];
    let labels = array![0, 0, 1, 1, 0, 1, 1, 1, 0, 1];
    let shot_counts = array![1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000];

    let mut calibrator = QuantumNeuralNetworkCalibrator::with_method(
        CalibrationMethod::BayesianBinning(BayesianBinningQuantiles::new(5)),
    );

    calibrator
        .fit_binary(&probabilities, &labels, Some(&shot_counts))
        .unwrap();

    let results = calibrator
        .transform_with_uncertainty(&probabilities)
        .unwrap();
    assert_eq!(results.len(), probabilities.len());

    for (mean, lower, upper) in results {
        assert!(lower <= mean);
        assert!(mean <= upper);
    }
}

#[test]
fn test_quantum_calibrator_metrics() {
    use crate::utils::calibration::quantum_calibration::{
        CalibrationMethod, QuantumNeuralNetworkCalibrator,
    };
    use crate::utils::calibration::PlattScaler;

    let probabilities = array![0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9];
    let labels = array![0, 0, 1, 1, 0, 1, 1, 1];
    let shot_counts = array![1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000];

    let mut calibrator =
        QuantumNeuralNetworkCalibrator::with_method(CalibrationMethod::Platt(PlattScaler::new()));
    calibrator
        .fit_binary(&probabilities, &labels, Some(&shot_counts))
        .unwrap();

    let metrics = calibrator
        .evaluate_quantum_calibration(&probabilities, &labels)
        .unwrap();

    assert!(metrics.ece >= 0.0 && metrics.ece <= 1.0);
    assert!(metrics.mce >= 0.0 && metrics.mce <= 1.0);
    assert!(metrics.brier_score >= 0.0);
    assert!(metrics.nll >= 0.0);
    assert!(metrics.shot_noise_impact >= 0.0);
    assert!(!metrics.interpretation.is_empty());
}

#[test]
fn test_quantum_ensemble_calibration() {
    use crate::utils::calibration::quantum_calibration::quantum_ensemble_calibration;

    let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95];
    let labels = array![0, 0, 1, 1, 1, 0, 0, 1, 1, 1];
    let shot_counts = array![1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000];

    let (ensemble_probs, metrics) =
        quantum_ensemble_calibration(&probabilities, &labels, &shot_counts, 5).unwrap();

    assert_eq!(ensemble_probs.len(), probabilities.len());

    // Check calibrated probabilities are valid
    for &val in ensemble_probs.iter() {
        assert!(val >= 0.0 && val <= 1.0);
    }

    // Check metrics
    assert!(metrics.ece >= 0.0 && metrics.ece <= 1.0);
    assert!(metrics.shot_noise_impact >= 0.0);
}
