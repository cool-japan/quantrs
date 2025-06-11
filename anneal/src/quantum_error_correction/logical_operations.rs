//! Logical Operations Configuration Types

/// Logical operations
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperation {
    /// Logical Pauli X
    LogicalX,
    /// Logical Pauli Y
    LogicalY,
    /// Logical Pauli Z
    LogicalZ,
    /// Logical Hadamard
    LogicalH,
    /// Logical CNOT
    LogicalCNOT,
    /// Logical T gate
    LogicalT,
    /// Logical measurement
    LogicalMeasurement,
    /// Logical preparation
    LogicalPreparation,
}