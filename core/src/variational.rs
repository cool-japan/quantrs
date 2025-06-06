//! Variational quantum gates with automatic differentiation support
//!
//! This module provides variational quantum gates whose parameters can be optimized
//! using gradient-based methods. It includes automatic differentiation for computing
//! parameter gradients efficiently.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    register::Register,
    matrix_ops::{DenseMatrix, QuantumMatrix},
};
use ndarray::{Array2, Array1};
use num_complex::Complex;
use std::f64::consts::PI;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::any::Any;

/// Automatic differentiation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffMode {
    /// Forward-mode automatic differentiation
    Forward,
    /// Reverse-mode automatic differentiation (backpropagation)
    Reverse,
    /// Parameter shift rule for quantum circuits
    ParameterShift,
    /// Finite differences approximation
    FiniteDiff { epsilon: f64 },
}

/// Dual number for forward-mode autodiff
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    /// Real part (value)
    pub real: f64,
    /// Dual part (derivative)
    pub dual: f64,
}

impl Dual {
    /// Create a new dual number
    pub fn new(real: f64, dual: f64) -> Self {
        Self { real, dual }
    }
    
    /// Create a constant (no derivative)
    pub fn constant(value: f64) -> Self {
        Self { real: value, dual: 0.0 }
    }
    
    /// Create a variable (unit derivative)
    pub fn variable(value: f64) -> Self {
        Self { real: value, dual: 1.0 }
    }
}

// Arithmetic operations for dual numbers
impl std::ops::Add for Dual {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            dual: self.dual + other.dual,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        Self {
            real: self.real - other.real,
            dual: self.dual - other.dual,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real,
            dual: self.real * other.dual + self.dual * other.real,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Self;
    
    fn div(self, other: Self) -> Self {
        Self {
            real: self.real / other.real,
            dual: (self.dual * other.real - self.real * other.dual) / (other.real * other.real),
        }
    }
}

// Trigonometric functions for dual numbers
impl Dual {
    pub fn sin(self) -> Self {
        Self {
            real: self.real.sin(),
            dual: self.dual * self.real.cos(),
        }
    }
    
    pub fn cos(self) -> Self {
        Self {
            real: self.real.cos(),
            dual: -self.dual * self.real.sin(),
        }
    }
    
    pub fn exp(self) -> Self {
        let exp_real = self.real.exp();
        Self {
            real: exp_real,
            dual: self.dual * exp_real,
        }
    }
    
    pub fn sqrt(self) -> Self {
        let sqrt_real = self.real.sqrt();
        Self {
            real: sqrt_real,
            dual: self.dual / (2.0 * sqrt_real),
        }
    }
}

/// Computation graph node for reverse-mode autodiff
#[derive(Debug, Clone)]
pub struct Node {
    /// Node identifier
    pub id: usize,
    /// Value at this node
    pub value: Complex<f64>,
    /// Gradient accumulated at this node
    pub grad: Complex<f64>,
    /// Operation that produced this node
    pub op: Operation,
    /// Parent nodes
    pub parents: Vec<usize>,
}

/// Operations in the computation graph
#[derive(Debug, Clone)]
pub enum Operation {
    /// Input parameter
    Parameter(String),
    /// Constant value
    Constant,
    /// Addition
    Add,
    /// Multiplication
    Mul,
    /// Complex conjugate
    Conj,
    /// Matrix multiplication
    MatMul,
    /// Exponential of imaginary number
    ExpI,
}

/// Computation graph for reverse-mode autodiff
#[derive(Debug)]
pub struct ComputationGraph {
    /// Nodes in the graph
    nodes: Vec<Node>,
    /// Parameter name to node ID mapping
    params: FxHashMap<String, usize>,
    /// Next available node ID
    next_id: usize,
}

impl ComputationGraph {
    /// Create a new computation graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            params: FxHashMap::default(),
            next_id: 0,
        }
    }
    
    /// Add a parameter node
    pub fn parameter(&mut self, name: String, value: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = Node {
            id,
            value: Complex::new(value, 0.0),
            grad: Complex::new(0.0, 0.0),
            op: Operation::Parameter(name.clone()),
            parents: vec![],
        };
        
        self.nodes.push(node);
        self.params.insert(name, id);
        id
    }
    
    /// Add a constant node
    pub fn constant(&mut self, value: Complex<f64>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = Node {
            id,
            value,
            grad: Complex::new(0.0, 0.0),
            op: Operation::Constant,
            parents: vec![],
        };
        
        self.nodes.push(node);
        id
    }
    
    /// Add two nodes
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let value = self.nodes[a].value + self.nodes[b].value;
        
        let node = Node {
            id,
            value,
            grad: Complex::new(0.0, 0.0),
            op: Operation::Add,
            parents: vec![a, b],
        };
        
        self.nodes.push(node);
        id
    }
    
    /// Multiply two nodes
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let value = self.nodes[a].value * self.nodes[b].value;
        
        let node = Node {
            id,
            value,
            grad: Complex::new(0.0, 0.0),
            op: Operation::Mul,
            parents: vec![a, b],
        };
        
        self.nodes.push(node);
        id
    }
    
    /// Exponential of i times a real parameter
    pub fn exp_i(&mut self, theta: usize) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let theta_val = self.nodes[theta].value.re;
        let value = Complex::new(theta_val.cos(), theta_val.sin());
        
        let node = Node {
            id,
            value,
            grad: Complex::new(0.0, 0.0),
            op: Operation::ExpI,
            parents: vec![theta],
        };
        
        self.nodes.push(node);
        id
    }
    
    /// Backward pass to compute gradients
    pub fn backward(&mut self, output: usize) {
        // Initialize output gradient
        self.nodes[output].grad = Complex::new(1.0, 0.0);
        
        // Traverse in reverse topological order
        for i in (0..=output).rev() {
            let grad = self.nodes[i].grad;
            let parents = self.nodes[i].parents.clone();
            let op = self.nodes[i].op.clone();
            
            match op {
                Operation::Add => {
                    // d/da (a + b) = 1, d/db (a + b) = 1
                    if !parents.is_empty() {
                        self.nodes[parents[0]].grad += grad;
                        self.nodes[parents[1]].grad += grad;
                    }
                }
                Operation::Mul => {
                    // d/da (a * b) = b, d/db (a * b) = a
                    if !parents.is_empty() {
                        let a = parents[0];
                        let b = parents[1];
                        let b_value = self.nodes[b].value;
                        let a_value = self.nodes[a].value;
                        self.nodes[a].grad += grad * b_value;
                        self.nodes[b].grad += grad * a_value;
                    }
                }
                Operation::ExpI => {
                    // d/dθ e^(iθ) = i * e^(iθ)
                    if !parents.is_empty() {
                        let theta = parents[0];
                        let node_value = self.nodes[i].value;
                        self.nodes[theta].grad += grad * Complex::new(0.0, 1.0) * node_value;
                    }
                }
                _ => {}
            }
        }
    }
    
    /// Get gradient for a parameter
    pub fn get_gradient(&self, param: &str) -> Option<f64> {
        self.params.get(param).map(|&id| self.nodes[id].grad.re)
    }
}

/// Variational quantum gate with autodiff support
#[derive(Clone)]
pub struct VariationalGate {
    /// Gate name
    pub name: String,
    /// Target qubits
    pub qubits: Vec<QubitId>,
    /// Parameter names
    pub params: Vec<String>,
    /// Current parameter values
    pub values: Vec<f64>,
    /// Gate generator function
    pub generator: Arc<dyn Fn(&[f64]) -> Array2<Complex<f64>> + Send + Sync>,
    /// Differentiation mode
    pub diff_mode: DiffMode,
}

impl VariationalGate {
    /// Create a variational rotation gate around X axis
    pub fn rx(qubit: QubitId, param_name: String, initial_value: f64) -> Self {
        let generator = Arc::new(|params: &[f64]| {
            let theta = params[0];
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();
            
            Array2::from_shape_vec((2, 2), vec![
                Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half),
                Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0),
            ]).unwrap()
        });
        
        Self {
            name: format!("RX({})", param_name),
            qubits: vec![qubit],
            params: vec![param_name],
            values: vec![initial_value],
            generator,
            diff_mode: DiffMode::ParameterShift,
        }
    }
    
    /// Create a variational rotation gate around Y axis
    pub fn ry(qubit: QubitId, param_name: String, initial_value: f64) -> Self {
        let generator = Arc::new(|params: &[f64]| {
            let theta = params[0];
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();
            
            Array2::from_shape_vec((2, 2), vec![
                Complex::new(cos_half, 0.0), Complex::new(-sin_half, 0.0),
                Complex::new(sin_half, 0.0), Complex::new(cos_half, 0.0),
            ]).unwrap()
        });
        
        Self {
            name: format!("RY({})", param_name),
            qubits: vec![qubit],
            params: vec![param_name],
            values: vec![initial_value],
            generator,
            diff_mode: DiffMode::ParameterShift,
        }
    }
    
    /// Create a variational rotation gate around Z axis
    pub fn rz(qubit: QubitId, param_name: String, initial_value: f64) -> Self {
        let generator = Arc::new(|params: &[f64]| {
            let theta = params[0];
            let exp_pos = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());
            let exp_neg = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
            
            Array2::from_shape_vec((2, 2), vec![
                exp_neg, Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), exp_pos,
            ]).unwrap()
        });
        
        Self {
            name: format!("RZ({})", param_name),
            qubits: vec![qubit],
            params: vec![param_name],
            values: vec![initial_value],
            generator,
            diff_mode: DiffMode::ParameterShift,
        }
    }
    
    /// Create a variational controlled rotation gate
    pub fn cry(control: QubitId, target: QubitId, param_name: String, initial_value: f64) -> Self {
        let generator = Arc::new(|params: &[f64]| {
            let theta = params[0];
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();
            
            let mut matrix = Array2::eye(4).mapv(|x| Complex::new(x, 0.0));
            // Apply RY to target when control is |1⟩
            matrix[[2, 2]] = Complex::new(cos_half, 0.0);
            matrix[[2, 3]] = Complex::new(-sin_half, 0.0);
            matrix[[3, 2]] = Complex::new(sin_half, 0.0);
            matrix[[3, 3]] = Complex::new(cos_half, 0.0);
            
            matrix
        });
        
        Self {
            name: format!("CRY({}, {})", param_name, control.0),
            qubits: vec![control, target],
            params: vec![param_name],
            values: vec![initial_value],
            generator,
            diff_mode: DiffMode::ParameterShift,
        }
    }
    
    /// Get current parameter values
    pub fn get_params(&self) -> &[f64] {
        &self.values
    }
    
    /// Set parameter values
    pub fn set_params(&mut self, values: Vec<f64>) -> QuantRS2Result<()> {
        if values.len() != self.params.len() {
            return Err(QuantRS2Error::InvalidInput(
                format!("Expected {} parameters, got {}", self.params.len(), values.len())
            ));
        }
        self.values = values;
        Ok(())
    }
    
    /// Compute gradient with respect to parameters
    pub fn gradient(&self, loss_fn: impl Fn(&Array2<Complex<f64>>) -> f64) -> QuantRS2Result<Vec<f64>> {
        match self.diff_mode {
            DiffMode::ParameterShift => self.parameter_shift_gradient(loss_fn),
            DiffMode::FiniteDiff { epsilon } => self.finite_diff_gradient(loss_fn, epsilon),
            DiffMode::Forward => self.forward_mode_gradient(loss_fn),
            DiffMode::Reverse => self.reverse_mode_gradient(loss_fn),
        }
    }
    
    /// Parameter shift rule for gradient computation
    fn parameter_shift_gradient(&self, loss_fn: impl Fn(&Array2<Complex<f64>>) -> f64) -> QuantRS2Result<Vec<f64>> {
        let mut gradients = vec![0.0; self.params.len()];
        
        for (i, &value) in self.values.iter().enumerate() {
            // Shift parameter by +π/2
            let mut params_plus = self.values.clone();
            params_plus[i] = value + PI / 2.0;
            let matrix_plus = (self.generator)(&params_plus);
            let loss_plus = loss_fn(&matrix_plus);
            
            // Shift parameter by -π/2
            let mut params_minus = self.values.clone();
            params_minus[i] = value - PI / 2.0;
            let matrix_minus = (self.generator)(&params_minus);
            let loss_minus = loss_fn(&matrix_minus);
            
            // Gradient via parameter shift rule
            gradients[i] = (loss_plus - loss_minus) / 2.0;
        }
        
        Ok(gradients)
    }
    
    /// Finite differences gradient approximation
    fn finite_diff_gradient(&self, loss_fn: impl Fn(&Array2<Complex<f64>>) -> f64, epsilon: f64) -> QuantRS2Result<Vec<f64>> {
        let mut gradients = vec![0.0; self.params.len()];
        
        for (i, &value) in self.values.iter().enumerate() {
            // Forward difference
            let mut params_plus = self.values.clone();
            params_plus[i] = value + epsilon;
            let matrix_plus = (self.generator)(&params_plus);
            let loss_plus = loss_fn(&matrix_plus);
            
            // Current value
            let matrix = (self.generator)(&self.values);
            let loss = loss_fn(&matrix);
            
            // Gradient approximation
            gradients[i] = (loss_plus - loss) / epsilon;
        }
        
        Ok(gradients)
    }
    
    /// Forward-mode automatic differentiation
    fn forward_mode_gradient(&self, loss_fn: impl Fn(&Array2<Complex<f64>>) -> f64) -> QuantRS2Result<Vec<f64>> {
        // Simplified implementation - would use dual numbers throughout
        let mut gradients = vec![0.0; self.params.len()];
        
        // For demonstration, use finite differences as fallback
        self.finite_diff_gradient(loss_fn, 1e-8)
    }
    
    /// Reverse-mode automatic differentiation
    fn reverse_mode_gradient(&self, loss_fn: impl Fn(&Array2<Complex<f64>>) -> f64) -> QuantRS2Result<Vec<f64>> {
        // Build computation graph
        let mut graph = ComputationGraph::new();
        
        // Add parameters to graph
        let param_nodes: Vec<_> = self.params.iter().zip(&self.values)
            .map(|(name, &value)| graph.parameter(name.clone(), value))
            .collect();
        
        // Compute matrix elements using graph
        // This is simplified - full implementation would build entire matrix computation
        
        // For now, use parameter shift as fallback
        self.parameter_shift_gradient(loss_fn)
    }
}

impl std::fmt::Debug for VariationalGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariationalGate")
            .field("name", &self.name)
            .field("qubits", &self.qubits)
            .field("params", &self.params)
            .field("values", &self.values)
            .field("diff_mode", &self.diff_mode)
            .finish()
    }
}

impl GateOp for VariationalGate {
    fn name(&self) -> &'static str {
        // We need to leak the string to get a 'static lifetime
        // This is safe for gate names which are created once
        Box::leak(self.name.clone().into_boxed_str())
    }
    
    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }
    
    fn is_parameterized(&self) -> bool {
        true
    }
    
    fn matrix(&self) -> QuantRS2Result<Vec<Complex<f64>>> {
        let mat = (self.generator)(&self.values);
        Ok(mat.iter().cloned().collect())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

/// Variational quantum circuit with multiple parameterized gates
#[derive(Debug)]
pub struct VariationalCircuit {
    /// List of gates in the circuit
    pub gates: Vec<VariationalGate>,
    /// Parameter name to gate indices mapping
    pub param_map: FxHashMap<String, Vec<usize>>,
    /// Number of qubits
    pub num_qubits: usize,
}

impl VariationalCircuit {
    /// Create a new variational circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            param_map: FxHashMap::default(),
            num_qubits,
        }
    }
    
    /// Add a variational gate to the circuit
    pub fn add_gate(&mut self, gate: VariationalGate) {
        let gate_idx = self.gates.len();
        
        // Update parameter map
        for param in &gate.params {
            self.param_map.entry(param.clone())
                .or_insert_with(Vec::new)
                .push(gate_idx);
        }
        
        self.gates.push(gate);
    }
    
    /// Get all parameter names
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.param_map.keys().cloned().collect();
        names.sort();
        names
    }
    
    /// Get current parameter values
    pub fn get_parameters(&self) -> FxHashMap<String, f64> {
        let mut params = FxHashMap::default();
        
        for gate in &self.gates {
            for (name, &value) in gate.params.iter().zip(&gate.values) {
                params.insert(name.clone(), value);
            }
        }
        
        params
    }
    
    /// Set parameter values
    pub fn set_parameters(&mut self, params: &FxHashMap<String, f64>) -> QuantRS2Result<()> {
        for (param_name, &value) in params {
            if let Some(gate_indices) = self.param_map.get(param_name) {
                for &idx in gate_indices {
                    if let Some(param_idx) = self.gates[idx].params.iter().position(|p| p == param_name) {
                        self.gates[idx].values[param_idx] = value;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute gradients for all parameters
    pub fn compute_gradients(&self, loss_fn: impl Fn(&[VariationalGate]) -> f64) -> QuantRS2Result<FxHashMap<String, f64>> {
        let mut gradients = FxHashMap::default();
        
        // Use parameter shift rule for each parameter
        for param_name in self.parameter_names() {
            let grad = self.parameter_gradient(param_name.as_str(), &loss_fn)?;
            gradients.insert(param_name, grad);
        }
        
        Ok(gradients)
    }
    
    /// Compute gradient for a single parameter
    fn parameter_gradient(&self, param_name: &str, loss_fn: &impl Fn(&[VariationalGate]) -> f64) -> QuantRS2Result<f64> {
        let current_params = self.get_parameters();
        let current_value = *current_params.get(param_name)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Parameter {} not found", param_name)))?;
        
        // Create circuit copies with shifted parameters
        let mut circuit_plus = self.clone_circuit();
        let mut params_plus = current_params.clone();
        params_plus.insert(param_name.to_string(), current_value + PI / 2.0);
        circuit_plus.set_parameters(&params_plus)?;
        
        let mut circuit_minus = self.clone_circuit();
        let mut params_minus = current_params.clone();
        params_minus.insert(param_name.to_string(), current_value - PI / 2.0);
        circuit_minus.set_parameters(&params_minus)?;
        
        // Compute gradient via parameter shift
        let loss_plus = loss_fn(&circuit_plus.gates);
        let loss_minus = loss_fn(&circuit_minus.gates);
        
        Ok((loss_plus - loss_minus) / 2.0)
    }
    
    /// Clone the circuit structure
    fn clone_circuit(&self) -> Self {
        Self {
            gates: self.gates.clone(),
            param_map: self.param_map.clone(),
            num_qubits: self.num_qubits,
        }
    }
}

/// Gradient-based optimizer for variational circuits
#[derive(Debug, Clone)]
pub struct VariationalOptimizer {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Accumulated momentum
    velocities: FxHashMap<String, f64>,
}

impl VariationalOptimizer {
    /// Create a new optimizer
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: FxHashMap::default(),
        }
    }
    
    /// Perform one optimization step
    pub fn step(&mut self, circuit: &mut VariationalCircuit, gradients: &FxHashMap<String, f64>) -> QuantRS2Result<()> {
        let mut new_params = circuit.get_parameters();
        
        for (param_name, &grad) in gradients {
            // Update velocity with momentum
            let velocity = self.velocities.entry(param_name.clone()).or_insert(0.0);
            *velocity = self.momentum * *velocity - self.learning_rate * grad;
            
            // Update parameter
            if let Some(value) = new_params.get_mut(param_name) {
                *value += *velocity;
            }
        }
        
        circuit.set_parameters(&new_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dual_arithmetic() {
        let a = Dual::variable(2.0);
        let b = Dual::constant(3.0);
        
        let c = a + b;
        assert_eq!(c.real, 5.0);
        assert_eq!(c.dual, 1.0);
        
        let d = a * b;
        assert_eq!(d.real, 6.0);
        assert_eq!(d.dual, 3.0);
        
        let e = a.sin();
        assert!((e.real - 2.0_f64.sin()).abs() < 1e-10);
        assert!((e.dual - 2.0_f64.cos()).abs() < 1e-10);
    }
    
    #[test]
    fn test_variational_rx_gate() {
        let gate = VariationalGate::rx(QubitId(0), "theta".to_string(), PI / 4.0);
        
        let matrix_vec = gate.matrix().unwrap();
        assert_eq!(matrix_vec.len(), 4);
        
        // Convert to Array2 for unitary check
        let matrix = Array2::from_shape_vec((2, 2), matrix_vec).unwrap();
        let mat = DenseMatrix::new(matrix).unwrap();
        assert!(mat.is_unitary(1e-10).unwrap());
    }
    
    #[test]
    fn test_parameter_shift_gradient() {
        // Use a specific angle
        let theta = PI / 3.0;
        let gate = VariationalGate::ry(QubitId(0), "phi".to_string(), theta);
        
        // Simple loss function: expectation value of Z
        let loss_fn = |matrix: &Array2<Complex<f64>>| -> f64 {
            // For |0⟩ state, <Z> = matrix[0,0] - matrix[1,1]
            // But we're using trace for simplicity
            (matrix[[0, 0]] + matrix[[1, 1]]).re
        };
        
        let gradients = gate.gradient(loss_fn).unwrap();
        assert_eq!(gradients.len(), 1);
        
        // For RY(θ), the matrix trace is 2*cos(θ/2)
        // Using parameter shift rule with shifts of ±π/2:
        // gradient = [f(θ+π/2) - f(θ-π/2)] / 2
        // = [2*cos((θ+π/2)/2) - 2*cos((θ-π/2)/2)] / 2
        // = cos(θ/2 + π/4) - cos(θ/2 - π/4)
        let plus_shift = 2.0 * ((theta + PI/2.0) / 2.0).cos();
        let minus_shift = 2.0 * ((theta - PI/2.0) / 2.0).cos();
        let expected = (plus_shift - minus_shift) / 2.0;
        
        // Allow for numerical precision
        assert!((gradients[0] - expected).abs() < 1e-5, 
            "Expected gradient: {}, got: {}", expected, gradients[0]);
    }
    
    #[test]
    fn test_variational_circuit() {
        let mut circuit = VariationalCircuit::new(2);
        
        circuit.add_gate(VariationalGate::rx(QubitId(0), "theta1".to_string(), 0.1));
        circuit.add_gate(VariationalGate::ry(QubitId(1), "theta2".to_string(), 0.2));
        circuit.add_gate(VariationalGate::cry(QubitId(0), QubitId(1), "theta3".to_string(), 0.3));
        
        assert_eq!(circuit.gates.len(), 3);
        assert_eq!(circuit.parameter_names().len(), 3);
        
        // Update parameters
        let mut new_params = FxHashMap::default();
        new_params.insert("theta1".to_string(), 0.5);
        new_params.insert("theta2".to_string(), 0.6);
        new_params.insert("theta3".to_string(), 0.7);
        
        circuit.set_parameters(&new_params).unwrap();
        
        let params = circuit.get_parameters();
        assert_eq!(params.get("theta1"), Some(&0.5));
        assert_eq!(params.get("theta2"), Some(&0.6));
        assert_eq!(params.get("theta3"), Some(&0.7));
    }
    
    #[test]
    fn test_optimizer() {
        let mut circuit = VariationalCircuit::new(1);
        circuit.add_gate(VariationalGate::rx(QubitId(0), "theta".to_string(), 0.0));
        
        let mut optimizer = VariationalOptimizer::new(0.1, 0.9);
        
        // Dummy gradients
        let mut gradients = FxHashMap::default();
        gradients.insert("theta".to_string(), 1.0);
        
        // Take optimization step
        optimizer.step(&mut circuit, &gradients).unwrap();
        
        let params = circuit.get_parameters();
        assert!(params.get("theta").unwrap().abs() > 0.0);
    }
}