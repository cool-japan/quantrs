use std::f64::consts::PI;
use std::any::Any;
use std::fmt::Debug;
use num_complex::Complex64;

use crate::error::QuantrsResult;
use crate::qubit::QubitId;

/// Trait for quantum gate operations
pub trait GateOp: Debug {
    /// Returns the name of the gate
    fn name(&self) -> &'static str;
    
    /// Returns the qubits that this gate acts on
    fn qubits(&self) -> Vec<QubitId>;
    
    /// Returns the number of qubits this gate acts on
    fn num_qubits(&self) -> usize {
        self.qubits().len()
    }
    
    /// Returns true if this gate is parameterized
    fn is_parameterized(&self) -> bool {
        false
    }
    
    /// Returns the matrix representation of this gate
    fn matrix(&self) -> QuantrsResult<Vec<Complex64>>;
    
    /// Downcast to concrete gate type
    fn as_any(&self) -> &dyn Any;
}

/// Single-qubit gate operations
pub mod single {
    use super::*;
    use std::any::Any;

    /// Hadamard gate
    #[derive(Debug, Clone, Copy)]
    pub struct Hadamard {
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for Hadamard {
        fn name(&self) -> &'static str {
            "H"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
            Ok(vec![
                Complex64::new(sqrt2_inv, 0.0), Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0), Complex64::new(-sqrt2_inv, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Pauli-X gate
    #[derive(Debug, Clone, Copy)]
    pub struct PauliX {
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for PauliX {
        fn name(&self) -> &'static str {
            "X"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            Ok(vec![
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Pauli-Y gate
    #[derive(Debug, Clone, Copy)]
    pub struct PauliY {
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for PauliY {
        fn name(&self) -> &'static str {
            "Y"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            Ok(vec![
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Pauli-Z gate
    #[derive(Debug, Clone, Copy)]
    pub struct PauliZ {
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for PauliZ {
        fn name(&self) -> &'static str {
            "Z"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Rotation around X-axis
    #[derive(Debug, Clone, Copy)]
    pub struct RotationX {
        /// Target qubit
        pub target: QubitId,
        
        /// Rotation angle (in radians)
        pub theta: f64,
    }
    
    impl GateOp for RotationX {
        fn name(&self) -> &'static str {
            "RX"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn is_parameterized(&self) -> bool {
            true
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            let cos = (self.theta / 2.0).cos();
            let sin = (self.theta / 2.0).sin();
            Ok(vec![
                Complex64::new(cos, 0.0), Complex64::new(0.0, -sin),
                Complex64::new(0.0, -sin), Complex64::new(cos, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Rotation around Y-axis
    #[derive(Debug, Clone, Copy)]
    pub struct RotationY {
        /// Target qubit
        pub target: QubitId,
        
        /// Rotation angle (in radians)
        pub theta: f64,
    }
    
    impl GateOp for RotationY {
        fn name(&self) -> &'static str {
            "RY"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn is_parameterized(&self) -> bool {
            true
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            let cos = (self.theta / 2.0).cos();
            let sin = (self.theta / 2.0).sin();
            Ok(vec![
                Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0),
                Complex64::new(sin, 0.0), Complex64::new(cos, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Rotation around Z-axis
    #[derive(Debug, Clone, Copy)]
    pub struct RotationZ {
        /// Target qubit
        pub target: QubitId,
        
        /// Rotation angle (in radians)
        pub theta: f64,
    }
    
    impl GateOp for RotationZ {
        fn name(&self) -> &'static str {
            "RZ"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn is_parameterized(&self) -> bool {
            true
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            let phase = Complex64::new(0.0, -self.theta / 2.0).exp();
            let phase_conj = Complex64::new(0.0, self.theta / 2.0).exp();
            Ok(vec![
                phase_conj, Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), phase,
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Phase gate (S gate)
    #[derive(Debug, Clone, Copy)]
    pub struct Phase {
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for Phase {
        fn name(&self) -> &'static str {
            "S"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// T gate
    #[derive(Debug, Clone, Copy)]
    pub struct T {
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for T {
        fn name(&self) -> &'static str {
            "T"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            let phase = Complex64::new((PI / 4.0).cos(), (PI / 4.0).sin());
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), phase,
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}

/// Multi-qubit gate operations
pub mod multi {
    use super::*;
    use std::any::Any;

    /// Controlled-NOT gate
    #[derive(Debug, Clone, Copy)]
    pub struct CNOT {
        /// Control qubit
        pub control: QubitId,
        
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for CNOT {
        fn name(&self) -> &'static str {
            "CNOT"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.control, self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            // For a 2-qubit gate, we return a 4x4 matrix in row-major order
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Controlled-Z gate
    #[derive(Debug, Clone, Copy)]
    pub struct CZ {
        /// Control qubit
        pub control: QubitId,
        
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for CZ {
        fn name(&self) -> &'static str {
            "CZ"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.control, self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            // For a 2-qubit gate, we return a 4x4 matrix in row-major order
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// SWAP gate
    #[derive(Debug, Clone, Copy)]
    pub struct SWAP {
        /// First qubit
        pub qubit1: QubitId,
        
        /// Second qubit
        pub qubit2: QubitId,
    }
    
    impl GateOp for SWAP {
        fn name(&self) -> &'static str {
            "SWAP"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            // For a 2-qubit gate, we return a 4x4 matrix in row-major order
            Ok(vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            ])
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Toffoli (CCNOT) gate
    #[derive(Debug, Clone, Copy)]
    pub struct Toffoli {
        /// First control qubit
        pub control1: QubitId,
        
        /// Second control qubit
        pub control2: QubitId,
        
        /// Target qubit
        pub target: QubitId,
    }
    
    impl GateOp for Toffoli {
        fn name(&self) -> &'static str {
            "Toffoli"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.control1, self.control2, self.target]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            // For a 3-qubit gate, we return an 8x8 matrix
            // This would be very large to write out fully, so for now we return
            // an error indicating it should be expanded into its constituent gates
            Err(crate::error::QuantrsError::UnsupportedOperation(
                "Direct matrix representation of Toffoli gate not supported. \
                 Use gate decomposition.".into()
            ))
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    /// Fredkin (CSWAP) gate
    #[derive(Debug, Clone, Copy)]
    pub struct Fredkin {
        /// Control qubit
        pub control: QubitId,
        
        /// First target qubit
        pub target1: QubitId,
        
        /// Second target qubit
        pub target2: QubitId,
    }
    
    impl GateOp for Fredkin {
        fn name(&self) -> &'static str {
            "Fredkin"
        }
        
        fn qubits(&self) -> Vec<QubitId> {
            vec![self.control, self.target1, self.target2]
        }
        
        fn matrix(&self) -> QuantrsResult<Vec<Complex64>> {
            // For a 3-qubit gate, we return an 8x8 matrix
            // This would be very large to write out fully, so for now we return
            // an error indicating it should be expanded into its constituent gates
            Err(crate::error::QuantrsError::UnsupportedOperation(
                "Direct matrix representation of Fredkin gate not supported. \
                 Use gate decomposition.".into()
            ))
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}