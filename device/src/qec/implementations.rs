//! Quantum Error Correction Code Implementations
//!
//! This module provides implementations of various quantum error correction codes:
//! - **Steane Code**: [[7,1,3]] CSS code that can correct any single qubit error
//! - **Shor Code**: [[9,1,3]] code that protects against both bit and phase flips
//! - **Surface Code**: Topological code with planar lattice geometry
//! - **Toric Code**: Topological code defined on a toroidal lattice

use quantrs2_core::qubit::QubitId;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

use super::{
    LogicalOperator, LogicalOperatorType, PauliOperator, QECResult, QuantumErrorCode,
    StabilizerGroup, StabilizerType,
};

/// Steane \[\[7,1,3\]\] quantum error correction code
///
/// The Steane code is a CSS (Calderbank-Shor-Steane) code that encodes
/// one logical qubit into seven physical qubits and can correct any single
/// qubit error (X, Y, or Z).
pub struct SteaneCode;

impl SteaneCode {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SteaneCode {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumErrorCode for SteaneCode {
    fn get_stabilizers(&self) -> Vec<StabilizerGroup> {
        vec![
            // X-stabilizers for Steane [[7,1,3]] code
            StabilizerGroup {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            // Z-stabilizers for Steane [[7,1,3]] code
            StabilizerGroup {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
        ]
    }

    fn get_logical_operators(&self) -> Vec<LogicalOperator> {
        vec![
            // Logical X operator (acts on all 7 qubits)
            LogicalOperator {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                ],
                operator_type: LogicalOperatorType::LogicalX,
            },
            // Logical Z operator (acts on all 7 qubits)
            LogicalOperator {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                operator_type: LogicalOperatorType::LogicalZ,
            },
        ]
    }

    fn distance(&self) -> usize {
        3
    }

    fn num_data_qubits(&self) -> usize {
        7
    }

    fn num_ancilla_qubits(&self) -> usize {
        6
    }

    fn logical_qubit_count(&self) -> usize {
        1
    }

    fn encode_logical_state(
        &self,
        logical_state: &Array1<Complex64>,
    ) -> QECResult<Array1<Complex64>> {
        Ok(logical_state.clone())
    }
}

/// Shor \[\[9,1,3\]\] quantum error correction code
///
/// The Shor code protects one logical qubit using nine physical qubits
/// and can correct any single qubit error. It combines both bit-flip
/// and phase-flip protection.
pub struct ShorCode;

impl ShorCode {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ShorCode {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumErrorCode for ShorCode {
    fn get_stabilizers(&self) -> Vec<StabilizerGroup> {
        vec![
            // Z-stabilizers for bit-flip correction (6 generators)
            StabilizerGroup {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 2,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 2,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 2,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 2,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 2,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 2,
            },
            // X-stabilizers for phase-flip correction (2 generators)
            StabilizerGroup {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 6,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::X,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 6,
            },
        ]
    }

    fn get_logical_operators(&self) -> Vec<LogicalOperator> {
        vec![
            // Logical X operator (one qubit from each group)
            LogicalOperator {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                operator_type: LogicalOperatorType::LogicalX,
            },
            // Logical Z operator (all qubits)
            LogicalOperator {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                operator_type: LogicalOperatorType::LogicalZ,
            },
        ]
    }

    fn distance(&self) -> usize {
        3
    }

    fn num_data_qubits(&self) -> usize {
        9
    }

    fn num_ancilla_qubits(&self) -> usize {
        8
    }

    fn logical_qubit_count(&self) -> usize {
        1
    }

    fn encode_logical_state(
        &self,
        logical_state: &Array1<Complex64>,
    ) -> QECResult<Array1<Complex64>> {
        Ok(logical_state.clone())
    }
}

/// Surface code with parameterized distance
///
/// The surface code is a topological quantum error correction code
/// defined on a 2D planar lattice. The code distance determines the
/// number of errors that can be corrected.
pub struct SurfaceCode {
    distance: usize,
}

impl SurfaceCode {
    pub fn new(distance: usize) -> Self {
        Self { distance }
    }
}

impl QuantumErrorCode for SurfaceCode {
    fn get_stabilizers(&self) -> Vec<StabilizerGroup> {
        // For simplicity, implement stabilizers for distance-3 surface code
        // This is a basic implementation - full surface codes require more complex lattice handling
        if self.distance != 3 {
            // Return a minimal set for other distances - could be extended
            return vec![
                StabilizerGroup {
                    operators: vec![PauliOperator::X, PauliOperator::X],
                    qubits: vec![QubitId::new(0), QubitId::new(1)],
                    stabilizer_type: StabilizerType::XStabilizer,
                    weight: 2,
                },
                StabilizerGroup {
                    operators: vec![PauliOperator::Z, PauliOperator::Z],
                    qubits: vec![QubitId::new(0), QubitId::new(1)],
                    stabilizer_type: StabilizerType::ZStabilizer,
                    weight: 2,
                },
            ];
        }

        // Distance-3 surface code stabilizers (simplified square lattice)
        // Data qubits: 0-8 arranged as:
        // 0 1 2
        // 3 4 5
        // 6 7 8
        vec![
            // X-stabilizers (vertex type)
            StabilizerGroup {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            // Z-stabilizers (plaquette type)
            StabilizerGroup {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                    QubitId::new(8),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
        ]
    }

    fn get_logical_operators(&self) -> Vec<LogicalOperator> {
        if self.distance != 3 {
            // Basic logical operators for other distances
            return vec![
                LogicalOperator {
                    operators: vec![PauliOperator::X, PauliOperator::I],
                    operator_type: LogicalOperatorType::LogicalX,
                },
                LogicalOperator {
                    operators: vec![PauliOperator::Z, PauliOperator::I],
                    operator_type: LogicalOperatorType::LogicalZ,
                },
            ];
        }

        // Distance-3 surface code logical operators
        vec![
            // Logical X operator (horizontal string)
            LogicalOperator {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                ],
                operator_type: LogicalOperatorType::LogicalX,
            },
            // Logical Z operator (vertical string)
            LogicalOperator {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                operator_type: LogicalOperatorType::LogicalZ,
            },
        ]
    }

    fn distance(&self) -> usize {
        self.distance
    }

    fn num_data_qubits(&self) -> usize {
        self.distance * self.distance
    }

    fn num_ancilla_qubits(&self) -> usize {
        self.distance * self.distance - 1
    }

    fn logical_qubit_count(&self) -> usize {
        1
    }

    fn encode_logical_state(
        &self,
        logical_state: &Array1<Complex64>,
    ) -> QECResult<Array1<Complex64>> {
        Ok(logical_state.clone())
    }
}

/// Toric code with parameterized lattice dimensions
///
/// The toric code is a topological quantum error correction code
/// defined on a torus (periodic boundary conditions in both directions).
/// It encodes two logical qubits.
pub struct ToricCode {
    dimensions: (usize, usize),
}

impl ToricCode {
    pub fn new(dimensions: (usize, usize)) -> Self {
        Self { dimensions }
    }
}

impl QuantumErrorCode for ToricCode {
    fn get_stabilizers(&self) -> Vec<StabilizerGroup> {
        // Implement a basic 2x2 toric code for simplicity
        // For general dimensions, this would need more complex lattice handling
        if self.dimensions != (2, 2) {
            // Return minimal stabilizers for other dimensions
            return vec![
                StabilizerGroup {
                    operators: vec![PauliOperator::X, PauliOperator::X],
                    qubits: vec![QubitId::new(0), QubitId::new(1)],
                    stabilizer_type: StabilizerType::XStabilizer,
                    weight: 2,
                },
                StabilizerGroup {
                    operators: vec![PauliOperator::Z, PauliOperator::Z],
                    qubits: vec![QubitId::new(0), QubitId::new(1)],
                    stabilizer_type: StabilizerType::ZStabilizer,
                    weight: 2,
                },
            ];
        }

        // 2x2 toric code has 8 data qubits arranged on a torus
        // X-stabilizers (vertex type) and Z-stabilizers (plaquette type)
        vec![
            // X-stabilizers (vertex type) - 4 stabilizers for 2x2 torus
            StabilizerGroup {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::X,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::XStabilizer,
                weight: 4,
            },
            // Z-stabilizers (plaquette type) - 4 stabilizers for 2x2 torus
            StabilizerGroup {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
            StabilizerGroup {
                operators: vec![
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::Z,
                ],
                qubits: vec![
                    QubitId::new(0),
                    QubitId::new(1),
                    QubitId::new(2),
                    QubitId::new(3),
                    QubitId::new(4),
                    QubitId::new(5),
                    QubitId::new(6),
                    QubitId::new(7),
                ],
                stabilizer_type: StabilizerType::ZStabilizer,
                weight: 4,
            },
        ]
    }

    fn get_logical_operators(&self) -> Vec<LogicalOperator> {
        if self.dimensions != (2, 2) {
            // Basic logical operators for other dimensions
            return vec![
                LogicalOperator {
                    operators: vec![PauliOperator::X, PauliOperator::I],
                    operator_type: LogicalOperatorType::LogicalX,
                },
                LogicalOperator {
                    operators: vec![PauliOperator::Z, PauliOperator::I],
                    operator_type: LogicalOperatorType::LogicalZ,
                },
            ];
        }

        // 2x2 toric code logical operators (2 logical qubits due to torus topology)
        vec![
            // First logical X operator (horizontal winding)
            LogicalOperator {
                operators: vec![
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::X,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                operator_type: LogicalOperatorType::LogicalX,
            },
            // First logical Z operator (vertical winding)
            LogicalOperator {
                operators: vec![
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::Z,
                    PauliOperator::I,
                    PauliOperator::I,
                    PauliOperator::I,
                ],
                operator_type: LogicalOperatorType::LogicalZ,
            },
        ]
    }

    fn distance(&self) -> usize {
        self.dimensions.0.min(self.dimensions.1)
    }

    fn num_data_qubits(&self) -> usize {
        2 * self.dimensions.0 * self.dimensions.1
    }

    fn num_ancilla_qubits(&self) -> usize {
        self.dimensions.0 * self.dimensions.1
    }

    fn logical_qubit_count(&self) -> usize {
        2
    }

    fn encode_logical_state(
        &self,
        logical_state: &Array1<Complex64>,
    ) -> QECResult<Array1<Complex64>> {
        Ok(logical_state.clone())
    }
}
