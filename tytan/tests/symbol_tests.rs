//! Tests for the symbol module.

use quantrs_tytan::*;

#[test]
fn test_basic_symbols() {
    // Test creating a single symbol
    let x = symbols("x");
    assert_eq!(x.to_string(), "x");
    
    // Test creating multiple symbols
    let y = symbols("y");
    let z = symbols("z");
    
    assert_eq!(y.to_string(), "y");
    assert_eq!(z.to_string(), "z");
}

#[test]
fn test_symbols_list_1d() {
    // Test creating a 1D array of symbols
    let x = symbols_list(5, "x{}").unwrap();
    
    assert_eq!(x.ndim(), 1);
    assert_eq!(x.shape(), &[5]);
    
    assert_eq!(x[0].to_string(), "x0");
    assert_eq!(x[1].to_string(), "x1");
    assert_eq!(x[2].to_string(), "x2");
    assert_eq!(x[3].to_string(), "x3");
    assert_eq!(x[4].to_string(), "x4");
}

#[test]
fn test_symbols_list_2d() {
    // Test creating a 2D array of symbols
    let q = symbols_list([3, 3], "q{}_{}").unwrap();
    
    assert_eq!(q.ndim(), 2);
    assert_eq!(q.shape(), &[3, 3]);
    
    assert_eq!(q[[0, 0]].to_string(), "q0_0");
    assert_eq!(q[[0, 1]].to_string(), "q0_1");
    assert_eq!(q[[1, 0]].to_string(), "q1_0");
    assert_eq!(q[[2, 2]].to_string(), "q2_2");
}

#[test]
fn test_symbols_list_error() {
    // Test error when format string doesn't match dimensions
    let result = symbols_list([3, 3], "q{}");
    assert!(result.is_err());
    
    // Test error when placeholders are not separated
    let result = symbols_list([3, 3], "q{}{}");
    assert!(result.is_err());
    
    // Test error when dimensions exceed limit
    let result = symbols_list([2, 2, 2, 2, 2, 2], "q{}_{}_{}_{}_{}_{}");
    assert!(result.is_err());
}

#[test]
fn test_symbols_define() {
    // Test generating symbol definition commands
    let command = symbols_define([2, 2], "q{}_{}").unwrap();
    
    // Check that the command contains all expected definitions
    assert!(command.contains("q0_0 = symbols(\"q0_0\")"));
    assert!(command.contains("q0_1 = symbols(\"q0_1\")"));
    assert!(command.contains("q1_0 = symbols(\"q1_0\")"));
    assert!(command.contains("q1_1 = symbols(\"q1_1\")"));
}

#[test]
fn test_symbols_nbit() {
    // Test creating an n-bit encoded variable
    let x = symbols_nbit(0, 16, "x{}", 4).unwrap();
    
    // Basic validation - should have components with correct names
    let expr_str = x.to_string();
    assert!(expr_str.contains("x0"));
    assert!(expr_str.contains("x1"));
    assert!(expr_str.contains("x2"));
    assert!(expr_str.contains("x3"));
    
    // Check bounds - should include coefficients for binary weighting
    // (exact comparison is tricky due to floating point and string formatting)
    assert!(expr_str.contains("8.0*x0"));
    assert!(expr_str.contains("4.0*x1"));
    assert!(expr_str.contains("2.0*x2"));
    assert!(expr_str.contains("1.0*x3"));
}