//! Syndrome decoders for quantum error correction

use super::pauli::{Pauli, PauliString};
use super::stabilizer::StabilizerCode;
use super::surface_code::SurfaceCode;
use super::SyndromeDecoder;
use crate::error::{QuantRS2Error, QuantRS2Result};
use std::collections::HashMap;

/// Lookup table decoder
pub struct LookupDecoder {
    /// Syndrome to error mapping
    syndrome_table: HashMap<Vec<bool>, PauliString>,
}

impl LookupDecoder {
    /// Create decoder for a stabilizer code
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

    /// Generate all Pauli errors up to given weight
    fn generate_pauli_errors(n: usize, max_weight: usize) -> Vec<PauliString> {
        let mut errors = vec![PauliString::identity(n)];

        for weight in 1..=max_weight {
            let weight_errors = Self::generate_weight_k_errors(n, weight);
            errors.extend(weight_errors);
        }

        errors
    }

    /// Generate all weight-k Pauli errors
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

    /// Generate all k-combinations from n elements
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

    /// Generate Cartesian power of a set
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

/// Minimum Weight Perfect Matching decoder for surface codes
pub struct MWPMDecoder {
    surface_code: SurfaceCode,
}

impl MWPMDecoder {
    /// Create MWPM decoder for surface code
    pub const fn new(surface_code: SurfaceCode) -> Self {
        Self { surface_code }
    }

    /// Find minimum weight matching for syndrome
    pub fn decode_syndrome(
        &self,
        x_syndrome: &[bool],
        z_syndrome: &[bool],
    ) -> QuantRS2Result<PauliString> {
        let n = self.surface_code.qubit_map.len();
        let mut error_paulis = vec![Pauli::I; n];

        // Decode X errors using Z syndrome
        let z_defects = self.find_defects(z_syndrome, &self.surface_code.z_stabilizers);
        let x_corrections = self.minimum_weight_matching(&z_defects, Pauli::X)?;

        for (qubit, pauli) in x_corrections {
            error_paulis[qubit] = pauli;
        }

        // Decode Z errors using X syndrome
        let x_defects = self.find_defects(x_syndrome, &self.surface_code.x_stabilizers);
        let z_corrections = self.minimum_weight_matching(&x_defects, Pauli::Z)?;

        for (qubit, pauli) in z_corrections {
            if error_paulis[qubit] == Pauli::I {
                error_paulis[qubit] = pauli;
            } else {
                // Combine X and Z to get Y
                error_paulis[qubit] = Pauli::Y;
            }
        }

        Ok(PauliString::new(error_paulis))
    }

    /// Find stabilizer defects from syndrome
    fn find_defects(&self, syndrome: &[bool], _stabilizers: &[Vec<usize>]) -> Vec<usize> {
        syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i) } else { None })
            .collect()
    }

    /// Simple minimum weight matching (for demonstration)
    fn minimum_weight_matching(
        &self,
        defects: &[usize],
        error_type: Pauli,
    ) -> QuantRS2Result<Vec<(usize, Pauli)>> {
        // This is a simplified version - real implementation would use blossom algorithm
        let mut corrections = Vec::new();

        if defects.len() % 2 != 0 {
            return Err(QuantRS2Error::InvalidInput(
                "Odd number of defects".to_string(),
            ));
        }

        // Simple greedy pairing
        let mut paired = vec![false; defects.len()];

        for i in 0..defects.len() {
            if paired[i] {
                continue;
            }

            // Find nearest unpaired defect
            let mut min_dist = usize::MAX;
            let mut min_j = i;

            for j in i + 1..defects.len() {
                if !paired[j] {
                    let dist = self.defect_distance(defects[i], defects[j]);
                    if dist < min_dist {
                        min_dist = dist;
                        min_j = j;
                    }
                }
            }

            if min_j != i {
                paired[i] = true;
                paired[min_j] = true;

                // Add correction path
                let path = self.shortest_path(defects[i], defects[min_j])?;
                for qubit in path {
                    corrections.push((qubit, error_type));
                }
            }
        }

        Ok(corrections)
    }

    /// Manhattan distance between defects
    const fn defect_distance(&self, defect1: usize, defect2: usize) -> usize {
        // This is simplified - would need proper defect coordinates
        (defect1 as isize - defect2 as isize).unsigned_abs()
    }

    /// Find shortest path between defects
    fn shortest_path(&self, start: usize, end: usize) -> QuantRS2Result<Vec<usize>> {
        // Simplified path - in practice would use proper graph traversal
        let path = if start < end {
            (start..=end).collect()
        } else {
            (end..=start).rev().collect()
        };

        Ok(path)
    }
}
