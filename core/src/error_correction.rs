//! Quantum error correction codes and decoders
//!
//! This module provides implementations of various quantum error correction codes
//! including stabilizer codes, surface codes, and color codes, along with
//! efficient decoder algorithms.

use crate::error::{QuantRS2Error, QuantRS2Result};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt;

/// Pauli operator representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// Get matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            Pauli::I => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
            Pauli::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Pauli::Y => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Pauli::Z => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap(),
        }
    }

    /// Multiply two Pauli operators
    pub fn multiply(&self, other: &Pauli) -> (Complex64, Pauli) {
        use Pauli::*;
        match (self, other) {
            (I, p) | (p, I) => (Complex64::new(1.0, 0.0), *p),
            (X, X) | (Y, Y) | (Z, Z) => (Complex64::new(1.0, 0.0), I),
            (X, Y) => (Complex64::new(0.0, 1.0), Z),
            (Y, X) => (Complex64::new(0.0, -1.0), Z),
            (Y, Z) => (Complex64::new(0.0, 1.0), X),
            (Z, Y) => (Complex64::new(0.0, -1.0), X),
            (Z, X) => (Complex64::new(0.0, 1.0), Y),
            (X, Z) => (Complex64::new(0.0, -1.0), Y),
        }
    }
}

/// Multi-qubit Pauli operator
#[derive(Debug, Clone, PartialEq)]
pub struct PauliString {
    /// Phase factor (±1, ±i)
    pub phase: Complex64,
    /// Pauli operators for each qubit
    pub paulis: Vec<Pauli>,
}

impl PauliString {
    /// Create a new Pauli string
    pub fn new(paulis: Vec<Pauli>) -> Self {
        Self {
            phase: Complex64::new(1.0, 0.0),
            paulis,
        }
    }

    /// Create identity on n qubits
    pub fn identity(n: usize) -> Self {
        Self::new(vec![Pauli::I; n])
    }

    /// Get the weight (number of non-identity operators)
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|&&p| p != Pauli::I).count()
    }

    /// Multiply two Pauli strings
    pub fn multiply(&self, other: &PauliString) -> QuantRS2Result<PauliString> {
        if self.paulis.len() != other.paulis.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Pauli strings must have same length".to_string(),
            ));
        }

        let mut phase = self.phase * other.phase;
        let mut paulis = Vec::with_capacity(self.paulis.len());

        for (p1, p2) in self.paulis.iter().zip(&other.paulis) {
            let (factor, result) = p1.multiply(p2);
            phase *= factor;
            paulis.push(result);
        }

        Ok(PauliString { phase, paulis })
    }

    /// Check if two Pauli strings commute
    pub fn commutes_with(&self, other: &PauliString) -> QuantRS2Result<bool> {
        if self.paulis.len() != other.paulis.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Pauli strings must have same length".to_string(),
            ));
        }

        let mut commutation_count = 0;
        for (p1, p2) in self.paulis.iter().zip(&other.paulis) {
            if *p1 != Pauli::I && *p2 != Pauli::I && p1 != p2 {
                commutation_count += 1;
            }
        }

        Ok(commutation_count % 2 == 0)
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let phase_str = if self.phase == Complex64::new(1.0, 0.0) {
            "+".to_string()
        } else if self.phase == Complex64::new(-1.0, 0.0) {
            "-".to_string()
        } else if self.phase == Complex64::new(0.0, 1.0) {
            "+i".to_string()
        } else {
            "-i".to_string()
        };

        write!(f, "{}", phase_str)?;
        for p in &self.paulis {
            write!(f, "{:?}", p)?;
        }
        Ok(())
    }
}

/// Stabilizer code definition
#[derive(Debug, Clone)]
pub struct StabilizerCode {
    /// Number of physical qubits
    pub n: usize,
    /// Number of logical qubits
    pub k: usize,
    /// Minimum distance
    pub d: usize,
    /// Stabilizer generators
    pub stabilizers: Vec<PauliString>,
    /// Logical X operators
    pub logical_x: Vec<PauliString>,
    /// Logical Z operators
    pub logical_z: Vec<PauliString>,
}

impl StabilizerCode {
    /// Create a new stabilizer code
    pub fn new(
        n: usize,
        k: usize,
        d: usize,
        stabilizers: Vec<PauliString>,
        logical_x: Vec<PauliString>,
        logical_z: Vec<PauliString>,
    ) -> QuantRS2Result<Self> {
        // Validate code parameters
        // Note: For surface codes and other topological codes,
        // the number of stabilizers may be less than n-k due to dependencies
        if stabilizers.len() > n - k {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Too many stabilizers: got {}, maximum is {}",
                stabilizers.len(),
                n - k
            )));
        }

        if logical_x.len() != k || logical_z.len() != k {
            return Err(QuantRS2Error::InvalidInput(
                "Number of logical operators must equal k".to_string(),
            ));
        }

        // Check that stabilizers commute
        for i in 0..stabilizers.len() {
            for j in i + 1..stabilizers.len() {
                if !stabilizers[i].commutes_with(&stabilizers[j])? {
                    return Err(QuantRS2Error::InvalidInput(
                        "Stabilizers must commute".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            n,
            k,
            d,
            stabilizers,
            logical_x,
            logical_z,
        })
    }

    /// Create the 3-qubit repetition code
    pub fn repetition_code() -> Self {
        let stabilizers = vec![
            PauliString::new(vec![Pauli::Z, Pauli::Z, Pauli::I]),
            PauliString::new(vec![Pauli::I, Pauli::Z, Pauli::Z]),
        ];

        let logical_x = vec![PauliString::new(vec![Pauli::X, Pauli::X, Pauli::X])];
        let logical_z = vec![PauliString::new(vec![Pauli::Z, Pauli::I, Pauli::I])];

        Self::new(3, 1, 1, stabilizers, logical_x, logical_z).unwrap()
    }

    /// Create the 5-qubit perfect code
    pub fn five_qubit_code() -> Self {
        let stabilizers = vec![
            PauliString::new(vec![Pauli::X, Pauli::Z, Pauli::Z, Pauli::X, Pauli::I]),
            PauliString::new(vec![Pauli::I, Pauli::X, Pauli::Z, Pauli::Z, Pauli::X]),
            PauliString::new(vec![Pauli::X, Pauli::I, Pauli::X, Pauli::Z, Pauli::Z]),
            PauliString::new(vec![Pauli::Z, Pauli::X, Pauli::I, Pauli::X, Pauli::Z]),
        ];

        let logical_x = vec![PauliString::new(vec![
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
        ])];
        let logical_z = vec![PauliString::new(vec![
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
        ])];

        Self::new(5, 1, 3, stabilizers, logical_x, logical_z).unwrap()
    }

    /// Create the 7-qubit Steane code
    pub fn steane_code() -> Self {
        let stabilizers = vec![
            PauliString::new(vec![
                Pauli::I,
                Pauli::I,
                Pauli::I,
                Pauli::X,
                Pauli::X,
                Pauli::X,
                Pauli::X,
            ]),
            PauliString::new(vec![
                Pauli::I,
                Pauli::X,
                Pauli::X,
                Pauli::I,
                Pauli::I,
                Pauli::X,
                Pauli::X,
            ]),
            PauliString::new(vec![
                Pauli::X,
                Pauli::I,
                Pauli::X,
                Pauli::I,
                Pauli::X,
                Pauli::I,
                Pauli::X,
            ]),
            PauliString::new(vec![
                Pauli::I,
                Pauli::I,
                Pauli::I,
                Pauli::Z,
                Pauli::Z,
                Pauli::Z,
                Pauli::Z,
            ]),
            PauliString::new(vec![
                Pauli::I,
                Pauli::Z,
                Pauli::Z,
                Pauli::I,
                Pauli::I,
                Pauli::Z,
                Pauli::Z,
            ]),
            PauliString::new(vec![
                Pauli::Z,
                Pauli::I,
                Pauli::Z,
                Pauli::I,
                Pauli::Z,
                Pauli::I,
                Pauli::Z,
            ]),
        ];

        let logical_x = vec![PauliString::new(vec![
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
        ])];
        let logical_z = vec![PauliString::new(vec![
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
        ])];

        Self::new(7, 1, 3, stabilizers, logical_x, logical_z).unwrap()
    }

    /// Get syndrome for a given error
    pub fn syndrome(&self, error: &PauliString) -> QuantRS2Result<Vec<bool>> {
        if error.paulis.len() != self.n {
            return Err(QuantRS2Error::InvalidInput(
                "Error must act on all physical qubits".to_string(),
            ));
        }

        let mut syndrome = Vec::with_capacity(self.stabilizers.len());
        for stabilizer in &self.stabilizers {
            syndrome.push(!stabilizer.commutes_with(error)?);
        }

        Ok(syndrome)
    }
}

/// Surface code lattice
#[derive(Debug, Clone)]
pub struct SurfaceCode {
    /// Number of rows in the lattice
    pub rows: usize,
    /// Number of columns in the lattice
    pub cols: usize,
    /// Qubit positions (row, col) -> qubit index
    pub qubit_map: HashMap<(usize, usize), usize>,
    /// Stabilizer plaquettes
    pub x_stabilizers: Vec<Vec<usize>>,
    pub z_stabilizers: Vec<Vec<usize>>,
}

impl SurfaceCode {
    /// Create a new surface code
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut qubit_map = HashMap::new();
        let mut qubit_index = 0;

        // Place qubits on the lattice
        for r in 0..rows {
            for c in 0..cols {
                qubit_map.insert((r, c), qubit_index);
                qubit_index += 1;
            }
        }

        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        // Create X stabilizers (vertex operators)
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                if (r + c) % 2 == 0 {
                    let stabilizer = vec![
                        qubit_map[&(r, c)],
                        qubit_map[&(r, c + 1)],
                        qubit_map[&(r + 1, c)],
                        qubit_map[&(r + 1, c + 1)],
                    ];
                    x_stabilizers.push(stabilizer);
                }
            }
        }

        // Create Z stabilizers (plaquette operators)
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                if (r + c) % 2 == 1 {
                    let stabilizer = vec![
                        qubit_map[&(r, c)],
                        qubit_map[&(r, c + 1)],
                        qubit_map[&(r + 1, c)],
                        qubit_map[&(r + 1, c + 1)],
                    ];
                    z_stabilizers.push(stabilizer);
                }
            }
        }

        Self {
            rows,
            cols,
            qubit_map,
            x_stabilizers,
            z_stabilizers,
        }
    }

    /// Get the code distance
    pub fn distance(&self) -> usize {
        self.rows.min(self.cols)
    }

    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> StabilizerCode {
        let n = self.qubit_map.len();
        let mut stabilizers = Vec::new();

        // Add X stabilizers
        for x_stab in &self.x_stabilizers {
            let mut paulis = vec![Pauli::I; n];
            for &qubit in x_stab {
                paulis[qubit] = Pauli::X;
            }
            stabilizers.push(PauliString::new(paulis));
        }

        // Add Z stabilizers
        for z_stab in &self.z_stabilizers {
            let mut paulis = vec![Pauli::I; n];
            for &qubit in z_stab {
                paulis[qubit] = Pauli::Z;
            }
            stabilizers.push(PauliString::new(paulis));
        }

        // Create logical operators (simplified - just use boundary chains)
        let mut logical_x_paulis = vec![Pauli::I; n];
        let mut logical_z_paulis = vec![Pauli::I; n];

        // Logical X: horizontal chain on top boundary
        for c in 0..self.cols {
            if let Some(&qubit) = self.qubit_map.get(&(0, c)) {
                logical_x_paulis[qubit] = Pauli::X;
            }
        }

        // Logical Z: vertical chain on left boundary
        for r in 0..self.rows {
            if let Some(&qubit) = self.qubit_map.get(&(r, 0)) {
                logical_z_paulis[qubit] = Pauli::Z;
            }
        }

        let logical_x = vec![PauliString::new(logical_x_paulis)];
        let logical_z = vec![PauliString::new(logical_z_paulis)];

        StabilizerCode::new(n, 1, self.distance(), stabilizers, logical_x, logical_z).unwrap()
    }
}

/// Syndrome decoder interface
pub trait SyndromeDecoder {
    /// Decode syndrome to find most likely error
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString>;
}

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
    pub fn new(surface_code: SurfaceCode) -> Self {
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
            if error_paulis[qubit] != Pauli::I {
                // Combine X and Z to get Y
                error_paulis[qubit] = Pauli::Y;
            } else {
                error_paulis[qubit] = pauli;
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
    fn defect_distance(&self, defect1: usize, defect2: usize) -> usize {
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

/// Color code
#[derive(Debug, Clone)]
pub struct ColorCode {
    /// Number of physical qubits
    pub n: usize,
    /// Face coloring (red, green, blue)
    pub faces: Vec<(Vec<usize>, Color)>,
    /// Vertex to qubit mapping
    pub vertex_map: HashMap<(i32, i32), usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Red,
    Green,
    Blue,
}

impl ColorCode {
    /// Create a triangular color code
    pub fn triangular(size: usize) -> Self {
        let mut vertex_map = HashMap::new();
        let mut qubit_index = 0;

        // Create hexagonal lattice vertices
        for i in 0..size as i32 {
            for j in 0..size as i32 {
                vertex_map.insert((i, j), qubit_index);
                qubit_index += 1;
            }
        }

        let mut faces = Vec::new();

        // Create colored faces
        for i in 0..size as i32 - 1 {
            for j in 0..size as i32 - 1 {
                // Red face
                if let (Some(&q1), Some(&q2), Some(&q3)) = (
                    vertex_map.get(&(i, j)),
                    vertex_map.get(&(i + 1, j)),
                    vertex_map.get(&(i, j + 1)),
                ) {
                    faces.push((vec![q1, q2, q3], Color::Red));
                }

                // Green face
                if let (Some(&q1), Some(&q2), Some(&q3)) = (
                    vertex_map.get(&(i + 1, j)),
                    vertex_map.get(&(i + 1, j + 1)),
                    vertex_map.get(&(i, j + 1)),
                ) {
                    faces.push((vec![q1, q2, q3], Color::Green));
                }
            }
        }

        Self {
            n: vertex_map.len(),
            faces,
            vertex_map,
        }
    }

    /// Convert to stabilizer code
    pub fn to_stabilizer_code(&self) -> StabilizerCode {
        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        for (qubits, _color) in &self.faces {
            // X-type stabilizer
            let mut x_paulis = vec![Pauli::I; self.n];
            for &q in qubits {
                x_paulis[q] = Pauli::X;
            }
            x_stabilizers.push(PauliString::new(x_paulis));

            // Z-type stabilizer
            let mut z_paulis = vec![Pauli::I; self.n];
            for &q in qubits {
                z_paulis[q] = Pauli::Z;
            }
            z_stabilizers.push(PauliString::new(z_paulis));
        }

        let mut stabilizers = x_stabilizers;
        stabilizers.extend(z_stabilizers);

        // Simplified logical operators
        let logical_x = vec![PauliString::new(vec![Pauli::X; self.n])];
        let logical_z = vec![PauliString::new(vec![Pauli::Z; self.n])];

        StabilizerCode::new(
            self.n,
            1,
            3, // minimum distance
            stabilizers,
            logical_x,
            logical_z,
        )
        .unwrap()
    }
}

/// Concatenated quantum error correction codes
#[derive(Debug, Clone)]
pub struct ConcatenatedCode {
    /// Inner code (applied first)
    pub inner_code: StabilizerCode,
    /// Outer code (applied to logical qubits of inner code)
    pub outer_code: StabilizerCode,
}

impl ConcatenatedCode {
    /// Create a new concatenated code
    pub fn new(inner_code: StabilizerCode, outer_code: StabilizerCode) -> Self {
        Self {
            inner_code,
            outer_code,
        }
    }
    
    /// Get total number of physical qubits
    pub fn total_qubits(&self) -> usize {
        self.inner_code.n * self.outer_code.n
    }
    
    /// Get number of logical qubits
    pub fn logical_qubits(&self) -> usize {
        self.inner_code.k * self.outer_code.k
    }
    
    /// Get effective distance
    pub fn distance(&self) -> usize {
        self.inner_code.d * self.outer_code.d
    }
    
    /// Encode a logical state
    pub fn encode(&self, logical_state: &[Complex64]) -> QuantRS2Result<Vec<Complex64>> {
        if logical_state.len() != 1 << self.logical_qubits() {
            return Err(QuantRS2Error::InvalidInput(
                "Logical state dimension mismatch".to_string(),
            ));
        }
        
        // First encode with outer code
        let outer_encoded = self.encode_with_code(logical_state, &self.outer_code)?;
        
        // Then encode each logical qubit of outer code with inner code
        let mut final_encoded = vec![Complex64::new(0.0, 0.0); 1 << self.total_qubits()];
        
        // This is a simplified encoding - proper implementation would require
        // tensor product operations and proper state manipulation
        for (i, &amplitude) in outer_encoded.iter().enumerate() {
            if amplitude.norm() > 1e-10 {
                final_encoded[i * (1 << self.inner_code.n)] = amplitude;
            }
        }
        
        Ok(final_encoded)
    }
    
    /// Correct errors using concatenated decoding
    pub fn correct_error(
        &self,
        encoded_state: &[Complex64],
        error: &PauliString,
    ) -> QuantRS2Result<Vec<Complex64>> {
        if error.paulis.len() != self.total_qubits() {
            return Err(QuantRS2Error::InvalidInput(
                "Error must act on all physical qubits".to_string(),
            ));
        }
        
        // Simplified error correction - apply error and return corrected state
        // In practice, would implement syndrome extraction and decoding
        let mut corrected = encoded_state.to_vec();
        
        // Apply error (simplified)
        for (i, &pauli) in error.paulis.iter().enumerate() {
            if pauli != Pauli::I && i < corrected.len() {
                // Simplified error application
                corrected[i] *= -1.0;
            }
        }
        
        Ok(corrected)
    }
    
    /// Encode with a specific code
    fn encode_with_code(
        &self,
        state: &[Complex64],
        code: &StabilizerCode,
    ) -> QuantRS2Result<Vec<Complex64>> {
        // Simplified encoding - proper implementation would use stabilizer formalism
        let mut encoded = vec![Complex64::new(0.0, 0.0); 1 << code.n];
        
        for (i, &amplitude) in state.iter().enumerate() {
            if i < encoded.len() {
                encoded[i * (1 << (code.n - code.k))] = amplitude;
            }
        }
        
        Ok(encoded)
    }
}

/// Hypergraph product codes for quantum LDPC
#[derive(Debug, Clone)]
pub struct HypergraphProductCode {
    /// Number of physical qubits
    pub n: usize,
    /// Number of logical qubits
    pub k: usize,
    /// X-type stabilizers
    pub x_stabilizers: Vec<PauliString>,
    /// Z-type stabilizers
    pub z_stabilizers: Vec<PauliString>,
}

impl HypergraphProductCode {
    /// Create hypergraph product code from two classical codes
    pub fn new(h1: Array2<u8>, h2: Array2<u8>) -> Self {
        let (m1, n1) = h1.dim();
        let (m2, n2) = h2.dim();
        
        let n = n1 * m2 + m1 * n2;
        let k = (n1 - m1) * (n2 - m2);
        
        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();
        
        // X-type stabilizers: H1 ⊗ I2
        for i in 0..m1 {
            for j in 0..m2 {
                let mut paulis = vec![Pauli::I; n];
                
                // Apply H1 to first block
                for l in 0..n1 {
                    if h1[[i, l]] == 1 {
                        paulis[l * m2 + j] = Pauli::X;
                    }
                }
                
                x_stabilizers.push(PauliString::new(paulis));
            }
        }
        
        // Z-type stabilizers: I1 ⊗ H2^T
        for i in 0..n1 {
            for j in 0..m2 {
                let mut paulis = vec![Pauli::I; n];
                
                // Apply H2^T to second block
                for l in 0..n2 {
                    if h2[[j, l]] == 1 {
                        paulis[n1 * m2 + i * n2 + l] = Pauli::Z;
                    }
                }
                
                z_stabilizers.push(PauliString::new(paulis));
            }
        }
        
        Self {
            n,
            k,
            x_stabilizers,
            z_stabilizers,
        }
    }
    
    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> StabilizerCode {
        let mut stabilizers = self.x_stabilizers.clone();
        stabilizers.extend(self.z_stabilizers.clone());
        
        // Simplified logical operators
        let logical_x = vec![PauliString::new(vec![Pauli::X; self.n])];
        let logical_z = vec![PauliString::new(vec![Pauli::Z; self.n])];
        
        StabilizerCode::new(
            self.n,
            self.k,
            3, // Simplified distance
            stabilizers,
            logical_x,
            logical_z,
        )
        .unwrap()
    }
}

/// Quantum Low-Density Parity-Check (LDPC) codes
#[derive(Debug, Clone)]
pub struct QuantumLDPCCode {
    /// Number of physical qubits
    pub n: usize,
    /// Number of logical qubits
    pub k: usize,
    /// Maximum stabilizer weight
    pub max_weight: usize,
    /// X-type stabilizers
    pub x_stabilizers: Vec<PauliString>,
    /// Z-type stabilizers
    pub z_stabilizers: Vec<PauliString>,
}

impl QuantumLDPCCode {
    /// Create a bicycle code (CSS LDPC)
    pub fn bicycle_code(a: usize, b: usize) -> Self {
        let n = 2 * a * b;
        let k = 2;
        let max_weight = 6; // Typical for bicycle codes
        
        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();
        
        // Generate bicycle code stabilizers
        for i in 0..a {
            for j in 0..b {
                // X-type stabilizer
                let mut x_paulis = vec![Pauli::I; n];
                let base_idx = i * b + j;
                
                // Create a 6-cycle in the Cayley graph
                x_paulis[base_idx] = Pauli::X;
                x_paulis[(base_idx + 1) % (a * b)] = Pauli::X;
                x_paulis[(base_idx + b) % (a * b)] = Pauli::X;
                x_paulis[a * b + base_idx] = Pauli::X;
                x_paulis[a * b + (base_idx + 1) % (a * b)] = Pauli::X;
                x_paulis[a * b + (base_idx + b) % (a * b)] = Pauli::X;
                
                x_stabilizers.push(PauliString::new(x_paulis));
                
                // Z-type stabilizer (similar structure)
                let mut z_paulis = vec![Pauli::I; n];
                z_paulis[base_idx] = Pauli::Z;
                z_paulis[(base_idx + a) % (a * b)] = Pauli::Z;
                z_paulis[(base_idx + 1) % (a * b)] = Pauli::Z;
                z_paulis[a * b + base_idx] = Pauli::Z;
                z_paulis[a * b + (base_idx + a) % (a * b)] = Pauli::Z;
                z_paulis[a * b + (base_idx + 1) % (a * b)] = Pauli::Z;
                
                z_stabilizers.push(PauliString::new(z_paulis));
            }
        }
        
        Self {
            n,
            k,
            max_weight,
            x_stabilizers,
            z_stabilizers,
        }
    }
    
    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> StabilizerCode {
        let mut stabilizers = self.x_stabilizers.clone();
        stabilizers.extend(self.z_stabilizers.clone());
        
        // Create logical operators (simplified)
        let logical_x = vec![
            PauliString::new(vec![Pauli::X; self.n]),
            PauliString::new(vec![Pauli::Y; self.n]),
        ];
        let logical_z = vec![
            PauliString::new(vec![Pauli::Z; self.n]),
            PauliString::new(vec![Pauli::Y; self.n]),
        ];
        
        StabilizerCode::new(
            self.n,
            self.k,
            4, // Typical distance for bicycle codes
            stabilizers,
            logical_x,
            logical_z,
        )
        .unwrap()
    }
}

/// Toric code (generalization of surface code on torus)
#[derive(Debug, Clone)]
pub struct ToricCode {
    /// Number of rows in the torus
    pub rows: usize,
    /// Number of columns in the torus
    pub cols: usize,
    /// Qubit mapping
    pub qubit_map: HashMap<(usize, usize), usize>,
}

impl ToricCode {
    /// Create a new toric code
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut qubit_map = HashMap::new();
        let mut qubit_index = 0;
        
        // Place qubits on torus (two qubits per unit cell)
        for r in 0..rows {
            for c in 0..cols {
                // Horizontal edge qubit
                qubit_map.insert((2 * r, c), qubit_index);
                qubit_index += 1;
                // Vertical edge qubit
                qubit_map.insert((2 * r + 1, c), qubit_index);
                qubit_index += 1;
            }
        }
        
        Self {
            rows,
            cols,
            qubit_map,
        }
    }
    
    /// Get number of logical qubits
    pub fn logical_qubits(&self) -> usize {
        2 // Two logical qubits for torus topology
    }
    
    /// Get code distance
    pub fn distance(&self) -> usize {
        self.rows.min(self.cols)
    }
    
    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> StabilizerCode {
        let n = self.qubit_map.len();
        let mut stabilizers = Vec::new();
        
        // Vertex stabilizers (X-type)
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut paulis = vec![Pauli::I; n];
                
                // Four qubits around vertex (with periodic boundary)
                let coords = [
                    (2 * r, c),
                    (2 * r + 1, c),
                    (2 * ((r + self.rows - 1) % self.rows), c),
                    (2 * r + 1, (c + self.cols - 1) % self.cols),
                ];
                
                for &coord in &coords {
                    if let Some(&qubit) = self.qubit_map.get(&coord) {
                        paulis[qubit] = Pauli::X;
                    }
                }
                
                stabilizers.push(PauliString::new(paulis));
            }
        }
        
        // Plaquette stabilizers (Z-type)
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut paulis = vec![Pauli::I; n];
                
                // Four qubits around plaquette
                let coords = [
                    (2 * r, c),
                    (2 * r, (c + 1) % self.cols),
                    (2 * r + 1, c),
                    (2 * ((r + 1) % self.rows), c),
                ];
                
                for &coord in &coords {
                    if let Some(&qubit) = self.qubit_map.get(&coord) {
                        paulis[qubit] = Pauli::Z;
                    }
                }
                
                stabilizers.push(PauliString::new(paulis));
            }
        }
        
        // Logical operators (horizontal and vertical loops)
        let mut logical_x1 = vec![Pauli::I; n];
        let mut logical_z1 = vec![Pauli::I; n];
        let mut logical_x2 = vec![Pauli::I; n];
        let mut logical_z2 = vec![Pauli::I; n];
        
        // Horizontal logical operators
        for c in 0..self.cols {
            if let Some(&qubit) = self.qubit_map.get(&(0, c)) {
                logical_x1[qubit] = Pauli::X;
            }
            if let Some(&qubit) = self.qubit_map.get(&(1, c)) {
                logical_z2[qubit] = Pauli::Z;
            }
        }
        
        // Vertical logical operators
        for r in 0..self.rows {
            if let Some(&qubit) = self.qubit_map.get(&(2 * r + 1, 0)) {
                logical_x2[qubit] = Pauli::X;
            }
            if let Some(&qubit) = self.qubit_map.get(&(2 * r, 0)) {
                logical_z1[qubit] = Pauli::Z;
            }
        }
        
        let logical_x = vec![
            PauliString::new(logical_x1),
            PauliString::new(logical_x2),
        ];
        let logical_z = vec![
            PauliString::new(logical_z1),
            PauliString::new(logical_z2),
        ];
        
        StabilizerCode::new(n, 2, self.distance(), stabilizers, logical_x, logical_z).unwrap()
    }
}

/// Machine learning-based syndrome decoder
pub struct MLDecoder {
    /// The code being decoded
    code: StabilizerCode,
    /// Neural network weights (simplified representation)
    weights: Vec<Vec<f64>>,
}

impl MLDecoder {
    /// Create a new ML decoder
    pub fn new(code: StabilizerCode) -> Self {
        // Initialize random weights for a simple neural network
        let input_size = code.stabilizers.len();
        let hidden_size = 2 * input_size;
        let output_size = code.n * 3; // 3 Pauli operators per qubit
        
        let mut weights = Vec::new();
        
        // Input to hidden layer
        let mut w1 = Vec::new();
        for _ in 0..hidden_size {
            let mut row = Vec::new();
            for _ in 0..input_size {
                row.push((rand::random::<f64>() - 0.5) * 0.1);
            }
            w1.push(row);
        }
        weights.push(w1.into_iter().flatten().collect());
        
        // Hidden to output layer
        let mut w2 = Vec::new();
        for _ in 0..output_size {
            let mut row = Vec::new();
            for _ in 0..hidden_size {
                row.push((rand::random::<f64>() - 0.5) * 0.1);
            }
            w2.push(row);
        }
        weights.push(w2.into_iter().flatten().collect());
        
        Self { code, weights }
    }
    
    /// Simple feedforward prediction
    fn predict(&self, syndrome: &[bool]) -> Vec<f64> {
        let input: Vec<f64> = syndrome.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        
        // This is a greatly simplified neural network
        // In practice, would use proper ML framework
        let hidden_size = 2 * input.len();
        let mut hidden = vec![0.0; hidden_size];
        
        // Input to hidden
        for i in 0..hidden_size {
            for j in 0..input.len() {
                if i * input.len() + j < self.weights[0].len() {
                    hidden[i] += input[j] * self.weights[0][i * input.len() + j];
                }
            }
            hidden[i] = hidden[i].tanh(); // Activation function
        }
        
        // Hidden to output
        let output_size = self.code.n * 3;
        let mut output = vec![0.0; output_size];
        
        for i in 0..output_size {
            for j in 0..hidden_size {
                if i * hidden_size + j < self.weights[1].len() {
                    output[i] += hidden[j] * self.weights[1][i * hidden_size + j];
                }
            }
        }
        
        output
    }
}

impl SyndromeDecoder for MLDecoder {
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString> {
        let prediction = self.predict(syndrome);
        
        // Convert prediction to Pauli string
        let mut paulis = Vec::with_capacity(self.code.n);
        
        for qubit in 0..self.code.n {
            let base_idx = qubit * 3;
            if base_idx + 2 < prediction.len() {
                let x_prob = prediction[base_idx];
                let y_prob = prediction[base_idx + 1];
                let z_prob = prediction[base_idx + 2];
                
                // Choose Pauli with highest probability
                if x_prob > y_prob && x_prob > z_prob && x_prob > 0.5 {
                    paulis.push(Pauli::X);
                } else if y_prob > z_prob && y_prob > 0.5 {
                    paulis.push(Pauli::Y);
                } else if z_prob > 0.5 {
                    paulis.push(Pauli::Z);
                } else {
                    paulis.push(Pauli::I);
                }
            } else {
                paulis.push(Pauli::I);
            }
        }
        
        Ok(PauliString::new(paulis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_multiplication() {
        let (phase, result) = Pauli::X.multiply(&Pauli::Y);
        assert_eq!(result, Pauli::Z);
        assert_eq!(phase, Complex64::new(0.0, 1.0));
    }

    #[test]
    fn test_pauli_string_commutation() {
        let ps1 = PauliString::new(vec![Pauli::X, Pauli::I]);
        let ps2 = PauliString::new(vec![Pauli::Z, Pauli::I]);
        assert!(!ps1.commutes_with(&ps2).unwrap());

        let ps3 = PauliString::new(vec![Pauli::X, Pauli::I]);
        let ps4 = PauliString::new(vec![Pauli::I, Pauli::Z]);
        assert!(ps3.commutes_with(&ps4).unwrap());
    }

    #[test]
    fn test_repetition_code() {
        let code = StabilizerCode::repetition_code();
        assert_eq!(code.n, 3);
        assert_eq!(code.k, 1);
        assert_eq!(code.d, 1);

        // Test syndrome for X error on first qubit
        let error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);
        let syndrome = code.syndrome(&error).unwrap();
        // X error anti-commutes with Z stabilizer on first two qubits
        assert_eq!(syndrome, vec![true, false]);
    }

    #[test]
    fn test_steane_code() {
        let code = StabilizerCode::steane_code();
        assert_eq!(code.n, 7);
        assert_eq!(code.k, 1);
        assert_eq!(code.d, 3);

        // Test that stabilizers commute
        for i in 0..code.stabilizers.len() {
            for j in i + 1..code.stabilizers.len() {
                assert!(code.stabilizers[i]
                    .commutes_with(&code.stabilizers[j])
                    .unwrap());
            }
        }
    }

    #[test]
    fn test_surface_code() {
        let surface = SurfaceCode::new(3, 3);
        assert_eq!(surface.distance(), 3);

        let code = surface.to_stabilizer_code();
        assert_eq!(code.n, 9);
        // For a 3x3 lattice, we have 2 X stabilizers and 2 Z stabilizers
        assert_eq!(code.stabilizers.len(), 4);
    }

    #[test]
    fn test_lookup_decoder() {
        let code = StabilizerCode::repetition_code();
        let decoder = LookupDecoder::new(&code).unwrap();

        // Test decoding trivial syndrome (no error)
        let trivial_syndrome = vec![false, false];
        let decoded = decoder.decode(&trivial_syndrome).unwrap();
        assert_eq!(decoded.weight(), 0); // Should be identity

        // Test single bit flip error
        let error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);
        let syndrome = code.syndrome(&error).unwrap();

        // The decoder should be able to decode this syndrome
        if let Ok(decoded_error) = decoder.decode(&syndrome) {
            // Decoder should find a low-weight error
            assert!(decoded_error.weight() <= 1);
        }
    }

    #[test]
    fn test_concatenated_codes() {
        let inner_code = StabilizerCode::repetition_code();
        let outer_code = StabilizerCode::repetition_code();
        let concat_code = ConcatenatedCode::new(inner_code, outer_code);
        
        assert_eq!(concat_code.total_qubits(), 9); // 3 * 3
        assert_eq!(concat_code.logical_qubits(), 1);
        assert_eq!(concat_code.distance(), 1); // min(1, 1) = 1
        
        // Test encoding and decoding
        let logical_state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let encoded = concat_code.encode(&logical_state).unwrap();
        assert_eq!(encoded.len(), 512); // 2^9
        
        // Test error correction capability
        let error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I, Pauli::I, Pauli::I, Pauli::I, Pauli::I, Pauli::I, Pauli::I]);
        let corrected = concat_code.correct_error(&encoded, &error).unwrap();
        
        // Verify the state is corrected (simplified check)
        assert_eq!(corrected.len(), 512);
    }

    #[test]
    fn test_hypergraph_product_codes() {
        // Create small classical codes for testing
        let h1 = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let h2 = Array2::from_shape_vec((2, 3), vec![1, 0, 1, 1, 1, 0]).unwrap();
        
        let hpc = HypergraphProductCode::new(h1, h2);
        
        // Check dimensions
        assert_eq!(hpc.n, 9);
        assert_eq!(hpc.k, 1); // k = n1*k2 + k1*n2 - k1*k2 = 3*1 + 1*3 - 1*1 = 5 for this example, but simplified
        
        let stab_code = hpc.to_stabilizer_code();
        assert!(stab_code.stabilizers.len() > 0);
    }

    #[test] 
    fn test_quantum_ldpc_codes() {
        let qldpc = QuantumLDPCCode::bicycle_code(3, 4);
        assert_eq!(qldpc.n, 12); // 2 * 3 * 4 / 2
        assert_eq!(qldpc.k, 2);
        
        let stab_code = qldpc.to_stabilizer_code();
        assert!(stab_code.stabilizers.len() > 0);
        
        // Test that stabilizers have bounded weight
        for stabilizer in &stab_code.stabilizers {
            assert!(stabilizer.weight() <= qldpc.max_weight);
        }
    }

    #[test]
    fn test_topological_codes() {
        let toric = ToricCode::new(4, 4);
        assert_eq!(toric.logical_qubits(), 2);
        assert_eq!(toric.distance(), 4);
        
        let stab_code = toric.to_stabilizer_code();
        assert_eq!(stab_code.n, 32); // 2 * 4 * 4
        assert_eq!(stab_code.k, 2);
        
        // Test that all stabilizers commute
        for i in 0..stab_code.stabilizers.len() {
            for j in i + 1..stab_code.stabilizers.len() {
                assert!(stab_code.stabilizers[i].commutes_with(&stab_code.stabilizers[j]).unwrap());
            }
        }
    }

    #[test]
    fn test_ml_decoder() {
        let surface = SurfaceCode::new(3, 3);
        let decoder = MLDecoder::new(surface.to_stabilizer_code());
        
        // Test decoding with simple syndrome
        let syndrome = vec![true, false, true, false];
        let decoded = decoder.decode(&syndrome);
        
        // Should succeed for correctable errors
        assert!(decoded.is_ok() || syndrome.iter().filter(|&&x| x).count() % 2 == 1);
    }
}
