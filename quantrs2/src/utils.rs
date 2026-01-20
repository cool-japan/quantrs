//! # Utility Functions for `QuantRS2`
//!
//! This module provides cross-cutting utility functions that are useful across
//! the `QuantRS2` framework.
//!
//! ## Module Organization
//!
//! - **Memory**: Memory estimation and management utilities
//! - **Performance**: Performance monitoring and benchmarking helpers
//! - **Validation**: Input validation and sanitization helpers
//! - **Formatting**: Output formatting utilities
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use quantrs2::utils::*;
//!
//! // Estimate memory requirements for a quantum circuit
//! let mem = estimate_statevector_memory(30);
//! println!("30-qubit simulation requires {} GB", mem / (1024 * 1024 * 1024));
//!
//! // Validate qubit count
//! if is_valid_qubit_count(50, 1024 * 1024 * 1024) {
//!     println!("Can simulate 50 qubits with 1GB memory");
//! }
//! ```

#![allow(clippy::missing_panics_doc)] // Panics are from expect() which are documented
#![allow(clippy::must_use_candidate)] // Most utility functions naturally return values

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::time::Duration;

// =============================================================================
// Quantum Computing Constants
// =============================================================================

/// Square root of 2, commonly used in quantum computing
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// 1/sqrt(2), the coefficient for superposition states
pub const INV_SQRT_2: f64 = FRAC_1_SQRT_2;

/// Pi constant for phase rotations
pub const PI_CONST: f64 = PI;

/// Pi/2, commonly used for rotation gates
pub const PI_OVER_2: f64 = PI / 2.0;

/// Pi/4, commonly used for T gates
pub const PI_OVER_4: f64 = PI / 4.0;

/// Pi/8, commonly used for precision rotations
pub const PI_OVER_8: f64 = PI / 8.0;

/// Estimate the memory required for state vector simulation in bytes
///
/// For `n` qubits, the state vector requires `2^n` complex numbers.
/// Each complex number is represented as `Complex64` (16 bytes).
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits in the quantum system
///
/// # Returns
///
/// Estimated memory in bytes
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::estimate_statevector_memory;
///
/// let mem_20_qubits = estimate_statevector_memory(20);
/// assert_eq!(mem_20_qubits, (1 << 20) * 16); // 2^20 * 16 bytes
/// ```
pub const fn estimate_statevector_memory(num_qubits: u32) -> usize {
    // 2^n complex numbers * 16 bytes per Complex64
    if num_qubits >= 32 {
        // Prevent overflow - return max value
        usize::MAX
    } else {
        (1usize << num_qubits) * 16
    }
}

/// Check if a qubit count is valid given available memory
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits
/// * `available_memory` - Available memory in bytes
///
/// # Returns
///
/// `true` if the simulation can fit in available memory
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::is_valid_qubit_count;
///
/// // Check if 20 qubits can fit in 32 MB
/// assert!(is_valid_qubit_count(20, 32 * 1024 * 1024));
///
/// // 40 qubits won't fit in 1 GB
/// assert!(!is_valid_qubit_count(40, 1024 * 1024 * 1024));
/// ```
pub const fn is_valid_qubit_count(num_qubits: u32, available_memory: usize) -> bool {
    let required = estimate_statevector_memory(num_qubits);
    required <= available_memory
}

/// Calculate the maximum number of qubits that can be simulated with given memory
///
/// # Arguments
///
/// * `available_memory` - Available memory in bytes
///
/// # Returns
///
/// Maximum number of qubits that can fit in the available memory
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::max_qubits_for_memory;
///
/// // With 1 GB of memory
/// let max_qubits = max_qubits_for_memory(1024 * 1024 * 1024);
/// assert!(max_qubits >= 26); // At least 26 qubits
/// assert!(max_qubits <= 30); // At most 30 qubits
/// ```
#[allow(clippy::manual_div_ceil)] // Binary search pattern is clearer this way
#[allow(clippy::missing_const_for_fn)] // Cannot be const due to while loop
pub fn max_qubits_for_memory(available_memory: usize) -> u32 {
    // Binary search for the maximum number of qubits
    let mut low = 0;
    let mut high = 64; // Practical upper limit

    while low < high {
        let mid = low + (high - low + 1) / 2;
        if is_valid_qubit_count(mid, available_memory) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

/// Format memory size in human-readable form
///
/// # Arguments
///
/// * `bytes` - Memory size in bytes
///
/// # Returns
///
/// Human-readable string (e.g., "1.5 GB", "256 MB")
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::format_memory;
///
/// assert_eq!(format_memory(1024), "1.00 KB");
/// assert_eq!(format_memory(1024 * 1024), "1.00 MB");
/// assert_eq!(format_memory(1536 * 1024 * 1024), "1.50 GB");
/// ```
pub fn format_memory(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    const TB: usize = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format duration in human-readable form
///
/// # Arguments
///
/// * `duration` - Duration to format
///
/// # Returns
///
/// Human-readable string (e.g., "1.5s", "250ms", "1m 30s")
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::format_duration;
/// use std::time::Duration;
///
/// assert!(format_duration(Duration::from_millis(1500)).contains("1.5"));
/// assert!(format_duration(Duration::from_millis(250)).contains("250"));
/// ```
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();
    let micros = duration.subsec_micros();

    if total_secs >= 60 {
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{mins}m {secs}s")
    } else if total_secs > 0 {
        if millis > 0 {
            format!("{total_secs}.{}s", millis / 100)
        } else {
            format!("{total_secs}s")
        }
    } else if millis > 0 {
        format!("{millis}ms")
    } else {
        format!("{micros}μs")
    }
}

/// Validate that a value is within a range
///
/// # Type Parameters
///
/// * `T` - Type that implements `PartialOrd`
///
/// # Arguments
///
/// * `value` - Value to validate
/// * `min` - Minimum allowed value (inclusive)
/// * `max` - Maximum allowed value (inclusive)
///
/// # Returns
///
/// `true` if value is in range [min, max]
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::is_in_range;
///
/// assert!(is_in_range(&5, &0, &10));
/// assert!(!is_in_range(&15, &0, &10));
/// ```
pub fn is_in_range<T: PartialOrd>(value: &T, min: &T, max: &T) -> bool {
    value >= min && value <= max
}

/// Calculate the binomial coefficient C(n, k)
///
/// # Arguments
///
/// * `n` - Total number of items
/// * `k` - Number of items to choose
///
/// # Returns
///
/// Binomial coefficient C(n, k)
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::binomial;
///
/// assert_eq!(binomial(5, 2), 10);
/// assert_eq!(binomial(10, 5), 252);
/// ```
pub fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }

    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Optimize using symmetry

    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }

    result
}

/// Calculate factorial of a number
///
/// # Arguments
///
/// * `n` - Input number
///
/// # Returns
///
/// n! (factorial of n), saturating at u64::MAX
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::factorial;
///
/// assert_eq!(factorial(5), 120);
/// assert_eq!(factorial(10), 3628800);
/// ```
pub fn factorial(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }

    let mut result = 1u64;
    for i in 2..=n {
        result = result.saturating_mul(i);
        if result == u64::MAX {
            break;
        }
    }

    result
}

// =============================================================================
// Quantum State Utilities
// =============================================================================

/// Check if a probability distribution is normalized (sums to 1)
///
/// # Arguments
///
/// * `probabilities` - Slice of probability values
/// * `tolerance` - Maximum allowed deviation from 1.0
///
/// # Returns
///
/// `true` if the sum is within tolerance of 1.0
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::is_normalized;
///
/// let probs = vec![0.5, 0.3, 0.2];
/// assert!(is_normalized(&probs, 1e-10));
///
/// let invalid = vec![0.5, 0.5, 0.5];
/// assert!(!is_normalized(&invalid, 1e-10));
/// ```
pub fn is_normalized(probabilities: &[f64], tolerance: f64) -> bool {
    let sum: f64 = probabilities.iter().sum();
    (sum - 1.0).abs() <= tolerance
}

/// Normalize a vector of values to sum to 1
///
/// # Arguments
///
/// * `values` - Mutable slice of values to normalize
///
/// # Returns
///
/// `true` if normalization was successful, `false` if sum was zero
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::normalize_probabilities;
///
/// let mut values = vec![1.0, 2.0, 3.0];
/// normalize_probabilities(&mut values);
/// assert!((values[0] - 1.0/6.0).abs() < 1e-10);
/// ```
pub fn normalize_probabilities(values: &mut [f64]) -> bool {
    let sum: f64 = values.iter().sum();
    if sum == 0.0 {
        return false;
    }
    for v in values.iter_mut() {
        *v /= sum;
    }
    true
}

/// Calculate classical fidelity between two probability distributions
///
/// F = sum_i sqrt(p_i * q_i)
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Fidelity value in range [0, 1], or None if distributions have different lengths
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::classical_fidelity;
///
/// let p = vec![0.5, 0.5];
/// let q = vec![0.5, 0.5];
/// let fid = classical_fidelity(&p, &q).unwrap();
/// assert!((fid - 1.0).abs() < 1e-10);
/// ```
pub fn classical_fidelity(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    let fid: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum();
    Some(fid)
}

/// Calculate trace distance between two probability distributions
///
/// D = 0.5 * sum_i |p_i - q_i|
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// Trace distance in range [0, 1], or None if distributions have different lengths
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::trace_distance;
///
/// let p = vec![1.0, 0.0];
/// let q = vec![0.0, 1.0];
/// let dist = trace_distance(&p, &q).unwrap();
/// assert!((dist - 1.0).abs() < 1e-10);
/// ```
pub fn trace_distance(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    let dist: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum::<f64>()
        / 2.0;
    Some(dist)
}

/// Calculate entropy of a probability distribution
///
/// H = -sum_i p_i * log2(p_i)
///
/// # Arguments
///
/// * `probabilities` - Probability distribution
///
/// # Returns
///
/// Shannon entropy in bits
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::entropy;
///
/// let uniform = vec![0.5, 0.5];
/// let h = entropy(&uniform);
/// assert!((h - 1.0).abs() < 1e-10); // 1 bit for uniform binary
/// ```
pub fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Convert angle from degrees to radians
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::deg_to_rad;
///
/// let rad = deg_to_rad(180.0);
/// assert!((rad - std::f64::consts::PI).abs() < 1e-10);
/// ```
#[allow(clippy::missing_const_for_fn)] // to_radians is not const stable
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees.to_radians()
}

/// Convert angle from radians to degrees
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::rad_to_deg;
///
/// let deg = rad_to_deg(std::f64::consts::PI);
/// assert!((deg - 180.0).abs() < 1e-10);
/// ```
#[allow(clippy::missing_const_for_fn)] // to_degrees is not const stable
pub fn rad_to_deg(radians: f64) -> f64 {
    radians.to_degrees()
}

/// Calculate the Hilbert space dimension for a given number of qubits
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits
///
/// # Returns
///
/// Dimension of the Hilbert space (2^n)
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::hilbert_dim;
///
/// assert_eq!(hilbert_dim(3), 8);
/// assert_eq!(hilbert_dim(10), 1024);
/// ```
pub const fn hilbert_dim(num_qubits: u32) -> usize {
    if num_qubits >= 64 {
        usize::MAX
    } else {
        1usize << num_qubits
    }
}

/// Calculate the number of qubits from Hilbert space dimension
///
/// # Arguments
///
/// * `dim` - Dimension of the Hilbert space (must be power of 2)
///
/// # Returns
///
/// Number of qubits, or None if dimension is not a power of 2
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::num_qubits_from_dim;
///
/// assert_eq!(num_qubits_from_dim(8), Some(3));
/// assert_eq!(num_qubits_from_dim(1024), Some(10));
/// assert_eq!(num_qubits_from_dim(100), None); // Not a power of 2
/// ```
pub const fn num_qubits_from_dim(dim: usize) -> Option<u32> {
    if dim == 0 || dim & (dim - 1) != 0 {
        return None; // Not a power of 2
    }
    Some(dim.trailing_zeros())
}

/// Check if a value is a valid probability (in range [0, 1])
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::is_valid_probability;
///
/// assert!(is_valid_probability(0.5));
/// assert!(is_valid_probability(0.0));
/// assert!(is_valid_probability(1.0));
/// assert!(!is_valid_probability(-0.1));
/// assert!(!is_valid_probability(1.1));
/// ```
pub fn is_valid_probability(p: f64) -> bool {
    (0.0..=1.0).contains(&p)
}

/// Clamp a value to valid probability range [0, 1]
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::clamp_probability;
///
/// assert_eq!(clamp_probability(0.5), 0.5);
/// assert_eq!(clamp_probability(-0.1), 0.0);
/// assert_eq!(clamp_probability(1.5), 1.0);
/// ```
#[allow(clippy::missing_const_for_fn)] // clamp is not const stable
pub fn clamp_probability(p: f64) -> f64 {
    p.clamp(0.0, 1.0)
}

/// Calculate the number of CNOT gates needed for full entanglement of n qubits
///
/// For a linear chain, this is n-1 CNOT gates.
///
/// # Example
///
/// ```rust
/// use quantrs2::utils::min_cnots_for_entanglement;
///
/// assert_eq!(min_cnots_for_entanglement(2), 1);
/// assert_eq!(min_cnots_for_entanglement(4), 3);
/// ```
pub const fn min_cnots_for_entanglement(num_qubits: u32) -> u32 {
    num_qubits.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_statevector_memory() {
        assert_eq!(estimate_statevector_memory(0), 16); // 2^0 = 1 state
        assert_eq!(estimate_statevector_memory(1), 32); // 2^1 = 2 states
        assert_eq!(estimate_statevector_memory(10), (1 << 10) * 16); // 2^10 states
        assert_eq!(estimate_statevector_memory(20), (1 << 20) * 16); // 2^20 states
    }

    #[test]
    fn test_is_valid_qubit_count() {
        // 20 qubits requires 16 MB, should fit in 32 MB
        assert!(is_valid_qubit_count(20, 32 * 1024 * 1024));

        // 30 qubits requires 16 GB, won't fit in 1 GB
        assert!(!is_valid_qubit_count(30, 1024 * 1024 * 1024));
    }

    #[test]
    fn test_max_qubits_for_memory() {
        // 1 GB should support about 26 qubits
        let max = max_qubits_for_memory(1024 * 1024 * 1024);
        assert!(max >= 26);
        assert!(max <= 27);
    }

    #[test]
    fn test_format_memory() {
        assert!(format_memory(1024).contains("KB"));
        assert!(format_memory(1024 * 1024).contains("MB"));
        assert!(format_memory(1024 * 1024 * 1024).contains("GB"));
    }

    #[test]
    fn test_format_duration() {
        let d = Duration::from_millis(1500);
        let formatted = format_duration(d);
        assert!(formatted.contains('1') && (formatted.contains('s') || formatted.contains("500")));

        let d = Duration::from_millis(250);
        let formatted = format_duration(d);
        assert!(formatted.contains("ms"));
    }

    #[test]
    fn test_is_in_range() {
        assert!(is_in_range(&5, &0, &10));
        assert!(is_in_range(&0, &0, &10));
        assert!(is_in_range(&10, &0, &10));
        assert!(!is_in_range(&11, &0, &10));
        assert!(!is_in_range(&-1, &0, &10));
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 5), 252);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3_628_800);
    }

    #[test]
    fn test_is_normalized() {
        let valid = vec![0.5, 0.3, 0.2];
        assert!(is_normalized(&valid, 1e-10));

        let invalid = vec![0.5, 0.5, 0.5];
        assert!(!is_normalized(&invalid, 1e-10));
    }

    #[test]
    fn test_normalize_probabilities() {
        let mut values = vec![1.0, 2.0, 3.0];
        assert!(normalize_probabilities(&mut values));
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!((values[0] - 1.0 / 6.0).abs() < 1e-10);

        let mut zeros = vec![0.0, 0.0];
        assert!(!normalize_probabilities(&mut zeros));
    }

    #[test]
    fn test_classical_fidelity() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let fid =
            classical_fidelity(&p, &q).expect("identical distributions should compute fidelity");
        assert!((fid - 1.0).abs() < 1e-10);

        let r = vec![1.0, 0.0];
        let s = vec![0.0, 1.0];
        let fid2 =
            classical_fidelity(&r, &s).expect("orthogonal distributions should compute fidelity");
        assert!((fid2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_distance() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let dist =
            trace_distance(&p, &q).expect("orthogonal distributions should compute trace distance");
        assert!((dist - 1.0).abs() < 1e-10);

        let same = trace_distance(&p, &p).expect("same distribution should compute trace distance");
        assert!((same - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy() {
        let uniform = vec![0.5, 0.5];
        let h = entropy(&uniform);
        assert!((h - 1.0).abs() < 1e-10);

        let certain = vec![1.0, 0.0];
        let h2 = entropy(&certain);
        assert!((h2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_deg_rad_conversion() {
        let rad = deg_to_rad(180.0);
        assert!((rad - std::f64::consts::PI).abs() < 1e-10);

        let deg = rad_to_deg(std::f64::consts::PI);
        assert!((deg - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_hilbert_dim() {
        assert_eq!(hilbert_dim(0), 1);
        assert_eq!(hilbert_dim(1), 2);
        assert_eq!(hilbert_dim(3), 8);
        assert_eq!(hilbert_dim(10), 1024);
    }

    #[test]
    fn test_num_qubits_from_dim() {
        assert_eq!(num_qubits_from_dim(1), Some(0));
        assert_eq!(num_qubits_from_dim(2), Some(1));
        assert_eq!(num_qubits_from_dim(8), Some(3));
        assert_eq!(num_qubits_from_dim(1024), Some(10));
        assert_eq!(num_qubits_from_dim(100), None);
        assert_eq!(num_qubits_from_dim(0), None);
    }

    #[test]
    fn test_is_valid_probability() {
        assert!(is_valid_probability(0.5));
        assert!(is_valid_probability(0.0));
        assert!(is_valid_probability(1.0));
        assert!(!is_valid_probability(-0.1));
        assert!(!is_valid_probability(1.1));
    }

    #[test]
    fn test_clamp_probability() {
        assert_eq!(clamp_probability(0.5), 0.5);
        assert_eq!(clamp_probability(-0.1), 0.0);
        assert_eq!(clamp_probability(1.5), 1.0);
    }

    #[test]
    fn test_min_cnots_for_entanglement() {
        assert_eq!(min_cnots_for_entanglement(1), 0);
        assert_eq!(min_cnots_for_entanglement(2), 1);
        assert_eq!(min_cnots_for_entanglement(4), 3);
        assert_eq!(min_cnots_for_entanglement(10), 9);
    }

    #[test]
    fn test_quantum_constants() {
        assert!((SQRT_2 - std::f64::consts::SQRT_2).abs() < 1e-15);
        assert!(INV_SQRT_2.mul_add(SQRT_2, -1.0).abs() < 1e-15);
        assert!(PI_OVER_2.mul_add(2.0, -PI_CONST).abs() < 1e-15);
        assert!(PI_OVER_4.mul_add(4.0, -PI_CONST).abs() < 1e-15);
    }
}
