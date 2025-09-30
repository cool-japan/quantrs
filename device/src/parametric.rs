//! Parametric circuit execution for variational quantum algorithms.
//!
//! This module provides support for parameterized quantum circuits,
//! enabling efficient execution of variational algorithms like VQE and QAOA.

use crate::{CircuitResult, DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::*, single::*, GateOp};
use std::collections::HashMap;
use std::sync::Arc;

/// Parameter type for circuits
#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
    /// Fixed value
    Fixed(f64),
    /// Named parameter
    Named(String),
    /// Expression of parameters
    Expression(Box<ParameterExpression>),
}

/// Parameter expression for complex relationships
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterExpression {
    /// Single parameter
    Param(String),
    /// Constant value
    Const(f64),
    /// Addition
    Add(Parameter, Parameter),
    /// Multiplication
    Mul(Parameter, Parameter),
    /// Division
    Div(Parameter, Parameter),
    /// Trigonometric functions
    Sin(Parameter),
    Cos(Parameter),
    /// Power
    Pow(Parameter, f64),
}

/// Parametric gate representation
#[derive(Debug, Clone)]
pub struct ParametricGate {
    /// Gate type
    pub gate_type: String,
    /// Target qubits
    pub qubits: Vec<usize>,
    /// Parameters (if any)
    pub parameters: Vec<Parameter>,
}

/// Parametric quantum circuit
#[derive(Debug, Clone)]
pub struct ParametricCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gates in the circuit
    pub gates: Vec<ParametricGate>,
    /// Parameter names
    pub parameter_names: Vec<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ParametricCircuit {
    /// Create a new parametric circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            parameter_names: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a parametric gate
    pub fn add_gate(&mut self, gate: ParametricGate) -> &mut Self {
        // Extract parameter names
        for param in &gate.parameters {
            self.extract_parameter_names(param);
        }

        self.gates.push(gate);
        self
    }

    /// Extract parameter names from a parameter
    fn extract_parameter_names(&mut self, param: &Parameter) {
        match param {
            Parameter::Named(name) => {
                if !self.parameter_names.contains(name) {
                    self.parameter_names.push(name.clone());
                }
            }
            Parameter::Expression(expr) => {
                self.extract_expr_parameter_names(expr);
            }
            _ => {}
        }
    }

    /// Extract parameter names from expression
    fn extract_expr_parameter_names(&mut self, expr: &ParameterExpression) {
        match expr {
            ParameterExpression::Param(name) => {
                if !self.parameter_names.contains(name) {
                    self.parameter_names.push(name.clone());
                }
            }
            ParameterExpression::Add(p1, p2)
            | ParameterExpression::Mul(p1, p2)
            | ParameterExpression::Div(p1, p2) => {
                self.extract_parameter_names(p1);
                self.extract_parameter_names(p2);
            }
            ParameterExpression::Sin(p)
            | ParameterExpression::Cos(p)
            | ParameterExpression::Pow(p, _) => {
                self.extract_parameter_names(p);
            }
            _ => {}
        }
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameter_names.len()
    }

    /// Bind parameters to create a concrete circuit
    pub fn bind_parameters<const N: usize>(
        &self,
        params: &HashMap<String, f64>,
    ) -> DeviceResult<Circuit<N>> {
        if N != self.num_qubits {
            return Err(DeviceError::APIError(
                "Circuit qubit count mismatch".to_string(),
            ));
        }

        let mut circuit = Circuit::<N>::new();

        for gate in &self.gates {
            use quantrs2_core::qubit::QubitId;

            match gate.gate_type.as_str() {
                "H" => {
                    circuit
                        .add_gate(Hadamard {
                            target: QubitId(gate.qubits[0] as u32),
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "X" => {
                    circuit
                        .add_gate(PauliX {
                            target: QubitId(gate.qubits[0] as u32),
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "Y" => {
                    circuit
                        .add_gate(PauliY {
                            target: QubitId(gate.qubits[0] as u32),
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "Z" => {
                    circuit
                        .add_gate(PauliZ {
                            target: QubitId(gate.qubits[0] as u32),
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "RX" => {
                    let angle = self.evaluate_parameter(&gate.parameters[0], params)?;
                    circuit
                        .add_gate(RotationX {
                            target: QubitId(gate.qubits[0] as u32),
                            theta: angle,
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "RY" => {
                    let angle = self.evaluate_parameter(&gate.parameters[0], params)?;
                    circuit
                        .add_gate(RotationY {
                            target: QubitId(gate.qubits[0] as u32),
                            theta: angle,
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "RZ" => {
                    let angle = self.evaluate_parameter(&gate.parameters[0], params)?;
                    circuit
                        .add_gate(RotationZ {
                            target: QubitId(gate.qubits[0] as u32),
                            theta: angle,
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "CNOT" => {
                    circuit
                        .add_gate(CNOT {
                            control: QubitId(gate.qubits[0] as u32),
                            target: QubitId(gate.qubits[1] as u32),
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "CRX" => {
                    let angle = self.evaluate_parameter(&gate.parameters[0], params)?;
                    circuit
                        .add_gate(CRX {
                            control: QubitId(gate.qubits[0] as u32),
                            target: QubitId(gate.qubits[1] as u32),
                            theta: angle,
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "CRY" => {
                    let angle = self.evaluate_parameter(&gate.parameters[0], params)?;
                    circuit
                        .add_gate(CRY {
                            control: QubitId(gate.qubits[0] as u32),
                            target: QubitId(gate.qubits[1] as u32),
                            theta: angle,
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                "CRZ" => {
                    let angle = self.evaluate_parameter(&gate.parameters[0], params)?;
                    circuit
                        .add_gate(CRZ {
                            control: QubitId(gate.qubits[0] as u32),
                            target: QubitId(gate.qubits[1] as u32),
                            theta: angle,
                        })
                        .map_err(|e| DeviceError::APIError(e.to_string()))?;
                }
                _ => {
                    return Err(DeviceError::UnsupportedOperation(format!(
                        "Gate type {} not supported",
                        gate.gate_type
                    )));
                }
            }
        }

        Ok(circuit)
    }

    /// Bind parameters using array
    pub fn bind_parameters_array<const N: usize>(
        &self,
        params: &[f64],
    ) -> DeviceResult<Circuit<N>> {
        if params.len() != self.parameter_names.len() {
            return Err(DeviceError::APIError(
                "Parameter count mismatch".to_string(),
            ));
        }

        let param_map: HashMap<String, f64> = self
            .parameter_names
            .iter()
            .zip(params.iter())
            .map(|(name, &value)| (name.clone(), value))
            .collect();

        self.bind_parameters(&param_map)
    }

    /// Evaluate parameter value
    fn evaluate_parameter(
        &self,
        param: &Parameter,
        values: &HashMap<String, f64>,
    ) -> DeviceResult<f64> {
        match param {
            Parameter::Fixed(val) => Ok(*val),
            Parameter::Named(name) => values
                .get(name)
                .copied()
                .ok_or_else(|| DeviceError::APIError(format!("Missing parameter: {}", name))),
            Parameter::Expression(expr) => self.evaluate_expression(expr, values),
        }
    }

    /// Evaluate parameter expression
    fn evaluate_expression(
        &self,
        expr: &ParameterExpression,
        values: &HashMap<String, f64>,
    ) -> DeviceResult<f64> {
        match expr {
            ParameterExpression::Param(name) => values
                .get(name)
                .copied()
                .ok_or_else(|| DeviceError::APIError(format!("Missing parameter: {}", name))),
            ParameterExpression::Const(val) => Ok(*val),
            ParameterExpression::Add(p1, p2) => {
                let v1 = self.evaluate_parameter(p1, values)?;
                let v2 = self.evaluate_parameter(p2, values)?;
                Ok(v1 + v2)
            }
            ParameterExpression::Mul(p1, p2) => {
                let v1 = self.evaluate_parameter(p1, values)?;
                let v2 = self.evaluate_parameter(p2, values)?;
                Ok(v1 * v2)
            }
            ParameterExpression::Div(p1, p2) => {
                let v1 = self.evaluate_parameter(p1, values)?;
                let v2 = self.evaluate_parameter(p2, values)?;
                if v2.abs() < f64::EPSILON {
                    Err(DeviceError::APIError("Division by zero".to_string()))
                } else {
                    Ok(v1 / v2)
                }
            }
            ParameterExpression::Sin(p) => {
                let v = self.evaluate_parameter(p, values)?;
                Ok(v.sin())
            }
            ParameterExpression::Cos(p) => {
                let v = self.evaluate_parameter(p, values)?;
                Ok(v.cos())
            }
            ParameterExpression::Pow(p, exp) => {
                let v = self.evaluate_parameter(p, values)?;
                Ok(v.powf(*exp))
            }
        }
    }

    /// Instantiate a gate with concrete values
    fn instantiate_gate(
        &self,
        gate: &ParametricGate,
        values: &HashMap<String, f64>,
    ) -> DeviceResult<Box<dyn GateOp>> {
        use quantrs2_core::qubit::QubitId;

        match gate.gate_type.as_str() {
            "H" => Ok(Box::new(Hadamard {
                target: QubitId(gate.qubits[0] as u32),
            })),
            "X" => Ok(Box::new(PauliX {
                target: QubitId(gate.qubits[0] as u32),
            })),
            "Y" => Ok(Box::new(PauliY {
                target: QubitId(gate.qubits[0] as u32),
            })),
            "Z" => Ok(Box::new(PauliZ {
                target: QubitId(gate.qubits[0] as u32),
            })),
            "RX" => {
                let angle = self.evaluate_parameter(&gate.parameters[0], values)?;
                Ok(Box::new(RotationX {
                    target: QubitId(gate.qubits[0] as u32),
                    theta: angle,
                }))
            }
            "RY" => {
                let angle = self.evaluate_parameter(&gate.parameters[0], values)?;
                Ok(Box::new(RotationY {
                    target: QubitId(gate.qubits[0] as u32),
                    theta: angle,
                }))
            }
            "RZ" => {
                let angle = self.evaluate_parameter(&gate.parameters[0], values)?;
                Ok(Box::new(RotationZ {
                    target: QubitId(gate.qubits[0] as u32),
                    theta: angle,
                }))
            }
            "CNOT" => Ok(Box::new(CNOT {
                control: QubitId(gate.qubits[0] as u32),
                target: QubitId(gate.qubits[1] as u32),
            })),
            "CRX" => {
                let angle = self.evaluate_parameter(&gate.parameters[0], values)?;
                Ok(Box::new(CRX {
                    control: QubitId(gate.qubits[0] as u32),
                    target: QubitId(gate.qubits[1] as u32),
                    theta: angle,
                }))
            }
            "CRY" => {
                let angle = self.evaluate_parameter(&gate.parameters[0], values)?;
                Ok(Box::new(CRY {
                    control: QubitId(gate.qubits[0] as u32),
                    target: QubitId(gate.qubits[1] as u32),
                    theta: angle,
                }))
            }
            "CRZ" => {
                let angle = self.evaluate_parameter(&gate.parameters[0], values)?;
                Ok(Box::new(CRZ {
                    control: QubitId(gate.qubits[0] as u32),
                    target: QubitId(gate.qubits[1] as u32),
                    theta: angle,
                }))
            }
            _ => Err(DeviceError::APIError(format!(
                "Unsupported gate type: {}",
                gate.gate_type
            ))),
        }
    }
}

/// Builder for parametric circuits
pub struct ParametricCircuitBuilder {
    circuit: ParametricCircuit,
}

impl ParametricCircuitBuilder {
    /// Create a new builder
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: ParametricCircuit::new(num_qubits),
        }
    }

    /// Add a Hadamard gate
    pub fn h(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(ParametricGate {
            gate_type: "H".to_string(),
            qubits: vec![qubit],
            parameters: vec![],
        });
        self
    }

    /// Add a parametric RX gate
    pub fn rx(mut self, qubit: usize, param: Parameter) -> Self {
        self.circuit.add_gate(ParametricGate {
            gate_type: "RX".to_string(),
            qubits: vec![qubit],
            parameters: vec![param],
        });
        self
    }

    /// Add a parametric RY gate
    pub fn ry(mut self, qubit: usize, param: Parameter) -> Self {
        self.circuit.add_gate(ParametricGate {
            gate_type: "RY".to_string(),
            qubits: vec![qubit],
            parameters: vec![param],
        });
        self
    }

    /// Add a parametric RZ gate
    pub fn rz(mut self, qubit: usize, param: Parameter) -> Self {
        self.circuit.add_gate(ParametricGate {
            gate_type: "RZ".to_string(),
            qubits: vec![qubit],
            parameters: vec![param],
        });
        self
    }

    /// Add a CNOT gate
    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(ParametricGate {
            gate_type: "CNOT".to_string(),
            qubits: vec![control, target],
            parameters: vec![],
        });
        self
    }

    /// Add a parametric controlled rotation
    pub fn crx(mut self, control: usize, target: usize, param: Parameter) -> Self {
        self.circuit.add_gate(ParametricGate {
            gate_type: "CRX".to_string(),
            qubits: vec![control, target],
            parameters: vec![param],
        });
        self
    }

    /// Build the circuit
    pub fn build(self) -> ParametricCircuit {
        self.circuit
    }
}

/// Batch executor for parametric circuits
pub struct ParametricExecutor<E> {
    /// Underlying executor
    executor: Arc<E>,
    /// Cache for compiled circuits
    cache: HashMap<String, Box<dyn GateOp>>,
}

impl<E> ParametricExecutor<E> {
    /// Create a new parametric executor
    pub fn new(executor: E) -> Self {
        Self {
            executor: Arc::new(executor),
            cache: HashMap::new(),
        }
    }
}

/// Batch execution request
#[derive(Debug, Clone)]
pub struct BatchExecutionRequest {
    /// Parametric circuit
    pub circuit: ParametricCircuit,
    /// Parameter sets to execute
    pub parameter_sets: Vec<Vec<f64>>,
    /// Number of shots per parameter set
    pub shots: usize,
    /// Observable to measure (optional)
    pub observable: Option<crate::zero_noise_extrapolation::Observable>,
}

/// Batch execution result
#[derive(Debug, Clone)]
pub struct BatchExecutionResult {
    /// Results for each parameter set
    pub results: Vec<CircuitResult>,
    /// Expectation values (if observable provided)
    pub expectation_values: Option<Vec<f64>>,
    /// Execution time (milliseconds)
    pub execution_time: u128,
}

/// Standard parametric circuit templates
pub struct ParametricTemplates;

impl ParametricTemplates {
    /// Hardware-efficient ansatz
    pub fn hardware_efficient_ansatz(num_qubits: usize, num_layers: usize) -> ParametricCircuit {
        let mut builder = ParametricCircuitBuilder::new(num_qubits);
        let mut param_idx = 0;

        for layer in 0..num_layers {
            // Single-qubit rotations
            for q in 0..num_qubits {
                builder = builder
                    .ry(q, Parameter::Named(format!("theta_{}_{}_y", layer, q)))
                    .rz(q, Parameter::Named(format!("theta_{}_{}_z", layer, q)));
                param_idx += 2;
            }

            // Entangling gates
            for q in 0..num_qubits - 1 {
                builder = builder.cnot(q, q + 1);
            }
        }

        // Final layer of rotations
        for q in 0..num_qubits {
            builder = builder.ry(q, Parameter::Named(format!("theta_final_{}", q)));
        }

        builder.build()
    }

    /// QAOA ansatz
    pub fn qaoa_ansatz(
        num_qubits: usize,
        num_layers: usize,
        problem_edges: Vec<(usize, usize)>,
    ) -> ParametricCircuit {
        let mut builder = ParametricCircuitBuilder::new(num_qubits);

        // Initial Hadamard layer
        for q in 0..num_qubits {
            builder = builder.h(q);
        }

        for p in 0..num_layers {
            // Problem Hamiltonian layer
            let gamma = Parameter::Named(format!("gamma_{}", p));
            for (u, v) in &problem_edges {
                builder = builder.cnot(*u, *v).rz(*v, gamma.clone()).cnot(*u, *v);
            }

            // Mixer Hamiltonian layer
            let beta = Parameter::Named(format!("beta_{}", p));
            for q in 0..num_qubits {
                builder = builder.rx(q, beta.clone());
            }
        }

        builder.build()
    }

    /// Strongly entangling layers
    pub fn strongly_entangling_layers(num_qubits: usize, num_layers: usize) -> ParametricCircuit {
        let mut builder = ParametricCircuitBuilder::new(num_qubits);

        for layer in 0..num_layers {
            // Rotation layer
            for q in 0..num_qubits {
                builder = builder
                    .rx(q, Parameter::Named(format!("r_{}_{}_x", layer, q)))
                    .ry(q, Parameter::Named(format!("r_{}_{}_y", layer, q)))
                    .rz(q, Parameter::Named(format!("r_{}_{}_z", layer, q)));
            }

            // Entangling layer with circular connectivity
            for q in 0..num_qubits {
                let target = (q + 1) % num_qubits;
                builder = builder.cnot(q, target);
            }
        }

        builder.build()
    }

    /// Excitation preserving ansatz (for chemistry)
    pub fn excitation_preserving(num_qubits: usize, num_electrons: usize) -> ParametricCircuit {
        let mut builder = ParametricCircuitBuilder::new(num_qubits);

        // Initialize with computational basis state for electrons
        // (This would need X gates in practice)

        // Single excitations
        for i in 0..num_electrons {
            for a in num_electrons..num_qubits {
                let theta = Parameter::Named(format!("t1_{}_{}", i, a));
                // Simplified - real implementation would use controlled rotations
                builder = builder.cnot(i, a).ry(a, theta).cnot(i, a);
            }
        }

        // Double excitations (simplified)
        for i in 0..num_electrons - 1 {
            for j in i + 1..num_electrons {
                for a in num_electrons..num_qubits - 1 {
                    for b in a + 1..num_qubits {
                        let theta = Parameter::Named(format!("t2_{}_{}_{}", i, j, a));
                        // Very simplified - real implementation is more complex
                        builder = builder
                            .cnot(i, a)
                            .cnot(j, b)
                            .rz(b, theta)
                            .cnot(j, b)
                            .cnot(i, a);
                    }
                }
            }
        }

        builder.build()
    }
}

/// Parameter optimization utilities
pub struct ParameterOptimizer;

impl ParameterOptimizer {
    /// Calculate parameter gradient using parameter shift rule
    pub fn parameter_shift_gradient(
        circuit: &ParametricCircuit,
        params: &[f64],
        observable: &crate::zero_noise_extrapolation::Observable,
        shift: f64,
    ) -> Vec<f64> {
        let mut gradients = vec![0.0; params.len()];

        // For each parameter
        for (i, _) in params.iter().enumerate() {
            // Shift parameter positively
            let mut params_plus = params.to_vec();
            params_plus[i] += shift;

            // Shift parameter negatively
            let mut params_minus = params.to_vec();
            params_minus[i] -= shift;

            // Calculate gradient (would need actual execution)
            // gradient[i] = (f(θ + s) - f(θ - s)) / (2 * sin(s))
            gradients[i] = 0.0; // Placeholder
        }

        gradients
    }

    /// Natural gradient using quantum Fisher information
    pub fn natural_gradient(
        circuit: &ParametricCircuit,
        params: &[f64],
        gradients: &[f64],
        regularization: f64,
    ) -> Vec<f64> {
        let n = params.len();

        // Calculate quantum Fisher information matrix (simplified)
        let mut fisher = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            fisher[[i, i]] = 1.0 + regularization; // Placeholder
        }

        // Solve F * nat_grad = grad
        // Simplified - would use proper linear algebra
        gradients.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parametric_circuit_builder() {
        let circuit = ParametricCircuitBuilder::new(2)
            .h(0)
            .ry(0, Parameter::Named("theta".to_string()))
            .cnot(0, 1)
            .rz(1, Parameter::Fixed(1.57))
            .build();

        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 4);
        assert_eq!(circuit.parameter_names, vec!["theta"]);
    }

    #[test]
    fn test_parameter_binding() {
        let circuit = ParametricCircuitBuilder::new(2)
            .ry(0, Parameter::Named("a".to_string()))
            .rz(0, Parameter::Named("b".to_string()))
            .build();

        let mut params = HashMap::new();
        params.insert("a".to_string(), 1.0);
        params.insert("b".to_string(), 2.0);

        let concrete = circuit.bind_parameters::<2>(&params).unwrap();
        assert_eq!(concrete.num_gates(), 2);
    }

    #[test]
    fn test_parameter_expressions() {
        use std::f64::consts::PI;

        let expr = Parameter::Expression(Box::new(ParameterExpression::Mul(
            Parameter::Named("theta".to_string()),
            Parameter::Fixed(2.0),
        )));

        let circuit = ParametricCircuitBuilder::new(1).rx(0, expr).build();

        let mut params = HashMap::new();
        params.insert("theta".to_string(), PI / 4.0);

        let concrete = circuit.bind_parameters::<1>(&params).unwrap();
        assert_eq!(concrete.num_gates(), 1);
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let ansatz = ParametricTemplates::hardware_efficient_ansatz(4, 2);

        assert_eq!(ansatz.num_qubits, 4);
        assert_eq!(ansatz.num_parameters(), 20); // 2 layers * 4 qubits * 2 params + 4 final
    }

    #[test]
    fn test_qaoa_ansatz() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let ansatz = ParametricTemplates::qaoa_ansatz(4, 3, edges);

        assert_eq!(ansatz.num_qubits, 4);
        assert_eq!(ansatz.num_parameters(), 6); // 3 layers * 2 params (gamma, beta)
    }
}
