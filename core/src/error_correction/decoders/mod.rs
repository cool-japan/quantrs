//! Syndrome decoders for quantum error correction.
//!
//! This module provides multiple decoding strategies:
//! - [`LookupDecoder`]: Pre-computed lookup table for small codes
//! - [`MwpmSurfaceDecoder`]: Minimum-Weight Perfect Matching for rotated surface codes
//! - [`UnionFindDecoder`]: Near-linear time Union-Find decoder
//! - [`min_weight_perfect_matching`]: Bitmask-DP perfect matching primitive

pub mod lookup;
pub mod min_matching;
pub mod mwpm;
pub mod union_find;

pub use lookup::LookupDecoder;
pub use min_matching::min_weight_perfect_matching;
pub use mwpm::MwpmSurfaceDecoder;
pub use union_find::UnionFindDecoder;

// Re-export legacy decoder for backward compatibility
use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::error_correction::pauli::{Pauli, PauliString};
use crate::error_correction::stabilizer::StabilizerCode;
use crate::error_correction::surface_code::SurfaceCode;
use crate::error_correction::SyndromeDecoder;

/// Minimum Weight Perfect Matching decoder for surface codes (legacy).
///
/// This is a simplified decoder kept for backward compatibility.
/// For production use with the rotated surface code, use [`MwpmSurfaceDecoder`].
pub struct MWPMDecoder {
    surface_code: SurfaceCode,
}

impl MWPMDecoder {
    /// Create MWPM decoder for surface code.
    pub const fn new(surface_code: SurfaceCode) -> Self {
        Self { surface_code }
    }

    /// Find minimum weight matching for syndrome.
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

    /// Find stabilizer defects from syndrome.
    fn find_defects(&self, syndrome: &[bool], _stabilizers: &[Vec<usize>]) -> Vec<usize> {
        syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i) } else { None })
            .collect()
    }

    /// Simple minimum weight matching.
    fn minimum_weight_matching(
        &self,
        defects: &[usize],
        error_type: Pauli,
    ) -> QuantRS2Result<Vec<(usize, Pauli)>> {
        let mut corrections = Vec::new();

        if defects.len() % 2 != 0 {
            return Err(QuantRS2Error::InvalidInput(
                "Odd number of defects".to_string(),
            ));
        }

        let mut paired = vec![false; defects.len()];

        for i in 0..defects.len() {
            if paired[i] {
                continue;
            }

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

                let path = self.shortest_path(defects[i], defects[min_j])?;
                for qubit in path {
                    corrections.push((qubit, error_type));
                }
            }
        }

        Ok(corrections)
    }

    /// Manhattan distance between defects.
    const fn defect_distance(&self, defect1: usize, defect2: usize) -> usize {
        (defect1 as isize - defect2 as isize).unsigned_abs()
    }

    /// Find shortest path between defects.
    fn shortest_path(&self, start: usize, end: usize) -> QuantRS2Result<Vec<usize>> {
        let path = if start < end {
            (start..=end).collect()
        } else {
            (end..=start).rev().collect()
        };

        Ok(path)
    }
}

impl SyndromeDecoder for MWPMDecoder {
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString> {
        let n = self.surface_code.qubit_map.len();
        let n_x = self.surface_code.x_stabilizers.len();
        let n_z = self.surface_code.z_stabilizers.len();

        if syndrome.len() != n_x + n_z {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Syndrome length {} != expected {}",
                syndrome.len(),
                n_x + n_z
            )));
        }

        let x_syndrome = &syndrome[..n_x];
        let z_syndrome = &syndrome[n_x..];
        self.decode_syndrome(x_syndrome, z_syndrome)
    }
}
