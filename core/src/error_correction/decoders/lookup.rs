//! Lookup table decoder for stabilizer codes.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::error_correction::pauli::{Pauli, PauliString};
use crate::error_correction::stabilizer::StabilizerCode;
use crate::error_correction::SyndromeDecoder;
use std::collections::HashMap;

/// Lookup table decoder.
///
/// Pre-computes all correctable Pauli errors (up to weight `⌊(d-1)/2⌋`) and maps
/// each syndrome to the minimum-weight error that produces it.
pub struct LookupDecoder {
    /// Syndrome to error mapping
    syndrome_table: HashMap<Vec<bool>, PauliString>,
}

impl LookupDecoder {
    /// Create a lookup decoder for the given stabilizer code.
    pub fn new(code: &StabilizerCode) -> QuantRS2Result<Self> {
        let mut syndrome_table = HashMap::new();

        // Generate all correctable errors (up to weight floor(d/2))
        let max_weight = (code.d - 1) / 2;
        let all_errors = Self::generate_pauli_errors(code.n, max_weight);

        for error in all_errors {
            let syndrome = code.syndrome(&error)?;

            // Only keep lowest weight error for each syndrome
            syndrome_table
                .entry(syndrome)
                .and_modify(|e: &mut PauliString| {
                    if error.weight() < e.weight() {
                        *e = error.clone();
                    }
                })
                .or_insert(error);
        }

        Ok(Self { syndrome_table })
    }

    /// Generate all Pauli errors up to given weight.
    fn generate_pauli_errors(n: usize, max_weight: usize) -> Vec<PauliString> {
        let mut errors = vec![PauliString::identity(n)];

        for weight in 1..=max_weight {
            let weight_errors = Self::generate_weight_k_errors(n, weight);
            errors.extend(weight_errors);
        }

        errors
    }

    /// Generate all weight-k Pauli errors.
    fn generate_weight_k_errors(n: usize, k: usize) -> Vec<PauliString> {
        let mut errors = Vec::new();
        let paulis = [Pauli::X, Pauli::Y, Pauli::Z];

        // Generate all combinations of k positions
        let positions = Self::combinations(n, k);

        for pos_set in positions {
            // For each position set, try all Pauli combinations
            let pauli_combinations = Self::cartesian_power(&paulis, k);

            for pauli_combo in pauli_combinations {
                let mut error_paulis = vec![Pauli::I; n];
                for (i, &pos) in pos_set.iter().enumerate() {
                    error_paulis[pos] = pauli_combo[i];
                }
                errors.push(PauliString::new(error_paulis));
            }
        }

        errors
    }

    /// Generate all k-combinations from n elements.
    fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut combo = (0..k).collect::<Vec<_>>();

        loop {
            result.push(combo.clone());

            // Find rightmost element that can be incremented
            let mut i = k;
            while i > 0 && (i == k || combo[i] == n - k + i) {
                i -= 1;
            }

            if i == 0 && combo[0] == n - k {
                break;
            }

            // Increment and reset following elements
            combo[i] += 1;
            for j in i + 1..k {
                combo[j] = combo[j - 1] + 1;
            }
        }

        result
    }

    /// Generate Cartesian power of a set.
    fn cartesian_power<T: Clone>(set: &[T], k: usize) -> Vec<Vec<T>> {
        if k == 0 {
            return vec![vec![]];
        }

        let mut result = Vec::new();
        let smaller = Self::cartesian_power(set, k - 1);

        for item in set {
            for mut combo in smaller.clone() {
                combo.push(item.clone());
                result.push(combo);
            }
        }

        result
    }
}

impl SyndromeDecoder for LookupDecoder {
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString> {
        self.syndrome_table
            .get(syndrome)
            .cloned()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Unknown syndrome".to_string()))
    }
}
