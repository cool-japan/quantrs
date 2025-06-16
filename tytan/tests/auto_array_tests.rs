//! Tests for the auto_array module.

use quantrs2_tytan::sampler::SampleResult;
use quantrs2_tytan::*;
use std::collections::HashMap;

#[cfg(feature = "dwave")]
use quantrs2_tytan::auto_array::Auto_array;

#[test]
#[cfg(feature = "dwave")]
fn test_auto_array_1d() {
    // Test 1D array conversion
    let mut assignments = HashMap::new();
    assignments.insert("x0".to_string(), true);
    assignments.insert("x1".to_string(), false);
    assignments.insert("x2".to_string(), true);

    let result = SampleResult {
        assignments,
        energy: -1.0,
        occurrences: 1,
    };

    // Convert to ndarray
    let (arr, indices) = Auto_array::new(&result).get_ndarray("x{}").unwrap();

    // Check array dimensions
    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.shape(), &[3]);

    // Check array values
    assert_eq!(arr[0], 1);
    assert_eq!(arr[1], 0);
    assert_eq!(arr[2], 1);

    // Check indices
    assert_eq!(indices.len(), 3);
    assert_eq!(indices[0], "0");
    assert_eq!(indices[1], "1");
    assert_eq!(indices[2], "2");
}

#[test]
#[cfg(feature = "dwave")]
fn test_auto_array_2d() {
    // Test 2D array conversion
    let mut assignments = HashMap::new();
    assignments.insert("q0_0".to_string(), true);
    assignments.insert("q0_1".to_string(), false);
    assignments.insert("q1_0".to_string(), false);
    assignments.insert("q1_1".to_string(), true);

    let result = SampleResult {
        assignments,
        energy: -1.0,
        occurrences: 1,
    };

    // Convert to ndarray
    let (arr, indices) = Auto_array::new(&result).get_ndarray("q{}_{}").unwrap();

    // Check array dimensions
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.shape(), &[2, 2]);

    // Check array values
    assert_eq!(arr[[0, 0]], 1);
    assert_eq!(arr[[0, 1]], 0);
    assert_eq!(arr[[1, 0]], 0);
    assert_eq!(arr[[1, 1]], 1);

    // Check indices
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0], ["0", "1"]);
    assert_eq!(indices[1], ["0", "1"]);
}

#[test]
#[cfg(feature = "dwave")]
fn test_auto_array_dframe() {
    // Test DataFrame conversion (2D)
    let mut assignments = HashMap::new();
    assignments.insert("q0_0".to_string(), true);
    assignments.insert("q0_1".to_string(), false);
    assignments.insert("q1_0".to_string(), false);
    assignments.insert("q1_1".to_string(), true);

    let result = SampleResult {
        assignments,
        energy: -1.0,
        occurrences: 1,
    };

    // Convert to DataFrame
    let (df, indices) = Auto_array::new(&result).get_dframe("q{}_{}").unwrap();

    // Check DataFrame dimensions
    assert_eq!(df.shape(), &[2, 2]);

    // Check DataFrame values
    assert_eq!(df[[0, 0]], 1);
    assert_eq!(df[[0, 1]], 0);
    assert_eq!(df[[1, 0]], 0);
    assert_eq!(df[[1, 1]], 1);

    // Check indices
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0], ["0", "1"]);
    assert_eq!(indices[1], ["0", "1"]);
}

#[test]
#[cfg(feature = "dwave")]
fn test_auto_array_image() {
    // Test image conversion (2D)
    let mut assignments = HashMap::new();
    assignments.insert("q0_0".to_string(), true);
    assignments.insert("q0_1".to_string(), false);
    assignments.insert("q1_0".to_string(), false);
    assignments.insert("q1_1".to_string(), true);

    let result = SampleResult {
        assignments,
        energy: -1.0,
        occurrences: 1,
    };

    // Convert to image
    let (img, indices) = Auto_array::new(&result).get_image("q{}_{}").unwrap();

    // Check image dimensions
    assert_eq!(img.shape(), &[2, 2]);

    // Check image values (0 or 255)
    assert_eq!(img[[0, 0]], 255);
    assert_eq!(img[[0, 1]], 0);
    assert_eq!(img[[1, 0]], 0);
    assert_eq!(img[[1, 1]], 255);

    // Check indices
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0], ["0", "1"]);
    assert_eq!(indices[1], ["0", "1"]);
}

#[test]
#[cfg(feature = "dwave")]
fn test_auto_array_missing_values() {
    // Test handling of missing values
    let mut assignments = HashMap::new();
    assignments.insert("q0_0".to_string(), true);
    // Missing q0_1
    assignments.insert("q1_0".to_string(), false);
    assignments.insert("q1_1".to_string(), true);

    let result = SampleResult {
        assignments,
        energy: -1.0,
        occurrences: 1,
    };

    // Convert to ndarray
    let (arr, _) = Auto_array::new(&result).get_ndarray("q{}_{}").unwrap();

    // Check array dimensions
    assert_eq!(arr.shape(), &[2, 2]);

    // Check array values (missing should be -1)
    assert_eq!(arr[[0, 0]], 1);
    assert_eq!(arr[[0, 1]], -1); // Missing value
    assert_eq!(arr[[1, 0]], 0);
    assert_eq!(arr[[1, 1]], 1);
}

#[test]
#[cfg(feature = "dwave")]
fn test_auto_array_natural_sort() {
    // Test natural sorting of indices
    let mut assignments = HashMap::new();
    assignments.insert("q1_1".to_string(), true);
    assignments.insert("q1_2".to_string(), false);
    assignments.insert("q1_10".to_string(), true);
    assignments.insert("q2_1".to_string(), false);
    assignments.insert("q10_1".to_string(), true);

    let result = SampleResult {
        assignments,
        energy: -1.0,
        occurrences: 1,
    };

    // Convert to ndarray
    let (_, indices) = Auto_array::new(&result).get_ndarray("q{}_{}").unwrap();

    // Check that indices are sorted naturally (1, 2, 10 not 1, 10, 2)
    assert_eq!(indices[0], ["1", "2", "10"]);
    assert_eq!(indices[1], ["1", "2", "10"]);
}
