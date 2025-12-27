//! Automatic differentiation for quantum machine learning.
//!
//! This module provides SciRS2-style automatic differentiation capabilities
//! for computing gradients of quantum circuits and variational algorithms.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::error::{MLError, Result};
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::GateOp;

/// Differentiable parameter in a quantum circuit
#[derive(Debug, Clone)]
pub struct DifferentiableParam {
    /// Parameter name/ID
    pub name: String,
    /// Current value
    pub value: f64,
    /// Gradient accumulator
    pub gradient: f64,
    /// Whether this parameter requires gradient
    pub requires_grad: bool,
}

impl DifferentiableParam {
    /// Create a new differentiable parameter
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            gradient: 0.0,
            requires_grad: true,
        }
    }

    /// Create a constant (non-differentiable) parameter
    pub fn constant(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            gradient: 0.0,
            requires_grad: false,
        }
    }
}

/// Computation graph node for automatic differentiation
#[derive(Debug, Clone)]
pub enum ComputationNode {
    /// Input parameter
    Parameter(String),
    /// Constant value
    Constant(f64),
    /// Addition operation
    Add(Box<ComputationNode>, Box<ComputationNode>),
    /// Multiplication operation
    Mul(Box<ComputationNode>, Box<ComputationNode>),
    /// Sine function
    Sin(Box<ComputationNode>),
    /// Cosine function
    Cos(Box<ComputationNode>),
    /// Exponential function
    Exp(Box<ComputationNode>),
    /// Quantum expectation value
    Expectation {
        circuit_params: Vec<String>,
        observable: String,
    },
}

/// Automatic differentiation engine
pub struct AutoDiff {
    /// Parameters registry
    parameters: HashMap<String, DifferentiableParam>,
    /// Computation graph
    graph: Option<ComputationNode>,
    /// Cached forward values
    forward_cache: HashMap<String, f64>,
}

impl AutoDiff {
    /// Create a new AutoDiff engine
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            graph: None,
            forward_cache: HashMap::new(),
        }
    }

    /// Register a parameter
    pub fn register_parameter(&mut self, param: DifferentiableParam) {
        self.parameters.insert(param.name.clone(), param);
    }

    /// Set computation graph
    pub fn set_graph(&mut self, graph: ComputationNode) {
        self.graph = Some(graph);
    }

    /// Forward pass - compute value
    pub fn forward(&mut self) -> Result<f64> {
        self.forward_cache.clear();

        if let Some(graph) = self.graph.clone() {
            self.evaluate_node(&graph)
        } else {
            Err(MLError::InvalidConfiguration(
                "No computation graph set".to_string(),
            ))
        }
    }

    /// Backward pass - compute gradients
    pub fn backward(&mut self, loss_gradient: f64) -> Result<()> {
        // Reset gradients
        for param in self.parameters.values_mut() {
            param.gradient = 0.0;
        }

        if let Some(graph) = self.graph.clone() {
            self.backpropagate(&graph, loss_gradient)?;
        }

        Ok(())
    }

    /// Evaluate a computation node
    fn evaluate_node(&mut self, node: &ComputationNode) -> Result<f64> {
        match node {
            ComputationNode::Parameter(name) => {
                self.parameters.get(name).map(|p| p.value).ok_or_else(|| {
                    MLError::InvalidConfiguration(format!("Unknown parameter: {}", name))
                })
            }
            ComputationNode::Constant(value) => Ok(*value),
            ComputationNode::Add(left, right) => {
                let l = self.evaluate_node(left)?;
                let r = self.evaluate_node(right)?;
                Ok(l + r)
            }
            ComputationNode::Mul(left, right) => {
                let l = self.evaluate_node(left)?;
                let r = self.evaluate_node(right)?;
                Ok(l * r)
            }
            ComputationNode::Sin(inner) => {
                let x = self.evaluate_node(inner)?;
                Ok(x.sin())
            }
            ComputationNode::Cos(inner) => {
                let x = self.evaluate_node(inner)?;
                Ok(x.cos())
            }
            ComputationNode::Exp(inner) => {
                let x = self.evaluate_node(inner)?;
                Ok(x.exp())
            }
            ComputationNode::Expectation {
                circuit_params,
                observable,
            } => {
                // Simplified - would compute actual expectation value
                let mut sum = 0.0;
                for param_name in circuit_params {
                    if let Some(param) = self.parameters.get(param_name) {
                        sum += param.value;
                    }
                }
                Ok(sum.cos()) // Placeholder
            }
        }
    }

    /// Backpropagate gradients through the graph
    fn backpropagate(&mut self, node: &ComputationNode, grad: f64) -> Result<()> {
        match node {
            ComputationNode::Parameter(name) => {
                if let Some(param) = self.parameters.get_mut(name) {
                    if param.requires_grad {
                        param.gradient += grad;
                    }
                }
            }
            ComputationNode::Constant(_) => {
                // No gradient for constants
            }
            ComputationNode::Add(left, right) => {
                // Gradient distributes equally for addition
                self.backpropagate(left, grad)?;
                self.backpropagate(right, grad)?;
            }
            ComputationNode::Mul(left, right) => {
                // Product rule
                let l_val = self.evaluate_node(left)?;
                let r_val = self.evaluate_node(right)?;
                self.backpropagate(left, grad * r_val)?;
                self.backpropagate(right, grad * l_val)?;
            }
            ComputationNode::Sin(inner) => {
                // d/dx sin(x) = cos(x)
                let x = self.evaluate_node(inner)?;
                self.backpropagate(inner, grad * x.cos())?;
            }
            ComputationNode::Cos(inner) => {
                // d/dx cos(x) = -sin(x)
                let x = self.evaluate_node(inner)?;
                self.backpropagate(inner, grad * (-x.sin()))?;
            }
            ComputationNode::Exp(inner) => {
                // d/dx exp(x) = exp(x)
                let x = self.evaluate_node(inner)?;
                self.backpropagate(inner, grad * x.exp())?;
            }
            ComputationNode::Expectation { circuit_params, .. } => {
                // Use parameter shift rule for quantum gradients
                for param_name in circuit_params {
                    let shift_grad = self.parameter_shift_gradient(param_name, PI / 2.0)?;
                    if let Some(param) = self.parameters.get_mut(param_name) {
                        if param.requires_grad {
                            param.gradient += grad * shift_grad;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute gradient using parameter shift rule
    fn parameter_shift_gradient(&self, param_name: &str, shift: f64) -> Result<f64> {
        // Simplified parameter shift rule
        // In practice, would evaluate circuit with ±shift
        Ok(0.5) // Placeholder
    }

    /// Get all gradients
    pub fn gradients(&self) -> HashMap<String, f64> {
        self.parameters
            .iter()
            .filter(|(_, p)| p.requires_grad)
            .map(|(name, param)| (name.clone(), param.gradient))
            .collect()
    }

    /// Update parameters using gradients
    pub fn update_parameters(&mut self, learning_rate: f64) {
        for param in self.parameters.values_mut() {
            if param.requires_grad {
                param.value -= learning_rate * param.gradient;
            }
        }
    }
}

/// Quantum-aware automatic differentiation
pub struct QuantumAutoDiff {
    /// Base autodiff engine
    autodiff: AutoDiff,
    /// Circuit executor (placeholder)
    executor: Box<dyn Fn(&[f64]) -> f64>,
}

impl QuantumAutoDiff {
    /// Create a new quantum autodiff engine
    pub fn new<F>(executor: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + 'static,
    {
        Self {
            autodiff: AutoDiff::new(),
            executor: Box::new(executor),
        }
    }

    /// Compute gradients using parameter shift rule
    pub fn parameter_shift_gradients(&self, params: &[f64], shift: f64) -> Result<Vec<f64>> {
        let mut gradients = vec![0.0; params.len()];

        for (i, _) in params.iter().enumerate() {
            // Shift parameter positively
            let mut params_plus = params.to_vec();
            params_plus[i] += shift;
            let val_plus = (self.executor)(&params_plus);

            // Shift parameter negatively
            let mut params_minus = params.to_vec();
            params_minus[i] -= shift;
            let val_minus = (self.executor)(&params_minus);

            // Parameter shift rule gradient
            gradients[i] = (val_plus - val_minus) / (2.0 * shift.sin());
        }

        Ok(gradients)
    }

    /// Compute natural gradients using quantum Fisher information
    pub fn natural_gradients(
        &self,
        params: &[f64],
        gradients: &[f64],
        regularization: f64,
    ) -> Result<Vec<f64>> {
        let n = params.len();
        let mut fisher = Array2::<f64>::zeros((n, n));

        // Compute quantum Fisher information matrix
        for i in 0..n {
            for j in 0..n {
                fisher[[i, j]] = self.compute_fisher_element(params, i, j)?;
            }
        }

        // Add regularization
        for i in 0..n {
            fisher[[i, i]] += regularization;
        }

        // Solve F * nat_grad = grad
        self.solve_linear_system(&fisher, gradients)
    }

    /// Compute element of quantum Fisher information matrix
    fn compute_fisher_element(&self, params: &[f64], i: usize, j: usize) -> Result<f64> {
        // Simplified - would compute <∂ψ/∂θᵢ|∂ψ/∂θⱼ>
        if i == j {
            Ok(1.0 + 0.1 * fastrand::f64())
        } else {
            Ok(0.1 * fastrand::f64())
        }
    }

    /// Solve linear system (simplified)
    fn solve_linear_system(&self, matrix: &Array2<f64>, rhs: &[f64]) -> Result<Vec<f64>> {
        // Simplified - would use proper linear algebra
        Ok(rhs.to_vec())
    }
}

/// Gradient tape for recording operations
#[derive(Debug, Clone)]
pub struct GradientTape {
    /// Recorded operations
    operations: Vec<Operation>,
    /// Variable values
    variables: HashMap<String, f64>,
}

/// Recorded operation
#[derive(Debug, Clone)]
enum Operation {
    /// Variable assignment
    Assign { var: String, value: f64 },
    /// Addition
    Add {
        result: String,
        left: String,
        right: String,
    },
    /// Multiplication
    Mul {
        result: String,
        left: String,
        right: String,
    },
    /// Quantum operation
    Quantum { result: String, params: Vec<String> },
}

impl GradientTape {
    /// Create a new gradient tape
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            variables: HashMap::new(),
        }
    }

    /// Record a variable
    pub fn variable(&mut self, name: impl Into<String>, value: f64) -> String {
        let name = name.into();
        self.variables.insert(name.clone(), value);
        self.operations.push(Operation::Assign {
            var: name.clone(),
            value,
        });
        name
    }

    /// Record addition
    pub fn add(&mut self, left: &str, right: &str) -> String {
        let result = format!("tmp_{}", self.operations.len());
        let left_val = self.variables[left];
        let right_val = self.variables[right];
        self.variables.insert(result.clone(), left_val + right_val);
        self.operations.push(Operation::Add {
            result: result.clone(),
            left: left.to_string(),
            right: right.to_string(),
        });
        result
    }

    /// Record multiplication
    pub fn mul(&mut self, left: &str, right: &str) -> String {
        let result = format!("tmp_{}", self.operations.len());
        let left_val = self.variables[left];
        let right_val = self.variables[right];
        self.variables.insert(result.clone(), left_val * right_val);
        self.operations.push(Operation::Mul {
            result: result.clone(),
            left: left.to_string(),
            right: right.to_string(),
        });
        result
    }

    /// Compute gradients
    pub fn gradient(&self, output: &str, inputs: &[&str]) -> HashMap<String, f64> {
        let mut gradients: HashMap<String, f64> = HashMap::new();

        // Initialize output gradient
        gradients.insert(output.to_string(), 1.0);

        // Backward pass through operations
        for op in self.operations.iter().rev() {
            match op {
                Operation::Add {
                    result,
                    left,
                    right,
                } => {
                    if let Some(&grad) = gradients.get(result) {
                        *gradients.entry(left.clone()).or_insert(0.0) += grad;
                        *gradients.entry(right.clone()).or_insert(0.0) += grad;
                    }
                }
                Operation::Mul {
                    result,
                    left,
                    right,
                } => {
                    if let Some(&grad) = gradients.get(result) {
                        let left_val = self.variables[left];
                        let right_val = self.variables[right];
                        *gradients.entry(left.clone()).or_insert(0.0) += grad * right_val;
                        *gradients.entry(right.clone()).or_insert(0.0) += grad * left_val;
                    }
                }
                _ => {}
            }
        }

        // Extract gradients for requested inputs
        inputs
            .iter()
            .map(|&input| {
                (
                    input.to_string(),
                    gradients.get(input).copied().unwrap_or(0.0),
                )
            })
            .collect()
    }
}

/// Optimizers for gradient-based training
pub mod optimizers {
    use super::*;

    /// Base optimizer trait
    pub trait Optimizer {
        /// Update parameters given gradients
        fn step(&mut self, params: &mut HashMap<String, f64>, gradients: &HashMap<String, f64>);

        /// Reset optimizer state
        fn reset(&mut self);
    }

    /// Stochastic Gradient Descent
    pub struct SGD {
        learning_rate: f64,
        momentum: f64,
        velocities: HashMap<String, f64>,
    }

    impl SGD {
        pub fn new(learning_rate: f64, momentum: f64) -> Self {
            Self {
                learning_rate,
                momentum,
                velocities: HashMap::new(),
            }
        }
    }

    impl Optimizer for SGD {
        fn step(&mut self, params: &mut HashMap<String, f64>, gradients: &HashMap<String, f64>) {
            for (name, grad) in gradients {
                let velocity = self.velocities.entry(name.clone()).or_insert(0.0);
                *velocity = self.momentum * *velocity - self.learning_rate * grad;

                if let Some(param) = params.get_mut(name) {
                    *param += *velocity;
                }
            }
        }

        fn reset(&mut self) {
            self.velocities.clear();
        }
    }

    /// Adam optimizer
    pub struct Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        m: HashMap<String, f64>,
        v: HashMap<String, f64>,
    }

    impl Adam {
        pub fn new(learning_rate: f64) -> Self {
            Self {
                learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                t: 0,
                m: HashMap::new(),
                v: HashMap::new(),
            }
        }
    }

    impl Optimizer for Adam {
        fn step(&mut self, params: &mut HashMap<String, f64>, gradients: &HashMap<String, f64>) {
            self.t += 1;
            let t = self.t as f64;

            for (name, grad) in gradients {
                let m_t = self.m.entry(name.clone()).or_insert(0.0);
                let v_t = self.v.entry(name.clone()).or_insert(0.0);

                // Update biased moments
                *m_t = self.beta1 * *m_t + (1.0 - self.beta1) * grad;
                *v_t = self.beta2 * *v_t + (1.0 - self.beta2) * grad * grad;

                // Bias correction
                let m_hat = *m_t / (1.0 - self.beta1.powf(t));
                let v_hat = *v_t / (1.0 - self.beta2.powf(t));

                // Update parameters
                if let Some(param) = params.get_mut(name) {
                    *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }

        fn reset(&mut self) {
            self.t = 0;
            self.m.clear();
            self.v.clear();
        }
    }

    /// Quantum Natural Gradient
    pub struct QNG {
        learning_rate: f64,
        regularization: f64,
    }

    impl QNG {
        pub fn new(learning_rate: f64, regularization: f64) -> Self {
            Self {
                learning_rate,
                regularization,
            }
        }
    }

    impl Optimizer for QNG {
        fn step(&mut self, params: &mut HashMap<String, f64>, gradients: &HashMap<String, f64>) {
            // Simplified - would compute natural gradient
            for (name, grad) in gradients {
                if let Some(param) = params.get_mut(name) {
                    *param -= self.learning_rate * grad;
                }
            }
        }

        fn reset(&mut self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodiff_basic() {
        let mut autodiff = AutoDiff::new();

        // Register parameters
        autodiff.register_parameter(DifferentiableParam::new("x", 2.0));
        autodiff.register_parameter(DifferentiableParam::new("y", 3.0));

        // Build computation graph: z = x * y
        let graph = ComputationNode::Mul(
            Box::new(ComputationNode::Parameter("x".to_string())),
            Box::new(ComputationNode::Parameter("y".to_string())),
        );
        autodiff.set_graph(graph);

        // Forward pass
        let result = autodiff.forward().expect("forward pass should succeed");
        assert_eq!(result, 6.0);

        // Backward pass
        autodiff
            .backward(1.0)
            .expect("backward pass should succeed");
        let gradients = autodiff.gradients();

        assert_eq!(gradients["x"], 3.0); // dz/dx = y
        assert_eq!(gradients["y"], 2.0); // dz/dy = x
    }

    #[test]
    fn test_gradient_tape() {
        let mut tape = GradientTape::new();

        let x = tape.variable("x", 2.0);
        let y = tape.variable("y", 3.0);
        let z = tape.mul(&x, &y);

        let gradients = tape.gradient(&z, &[&x, &y]);

        assert_eq!(gradients[&x], 3.0);
        assert_eq!(gradients[&y], 2.0);
    }

    #[test]
    fn test_optimizers() {
        use optimizers::*;

        let mut params = HashMap::new();
        params.insert("x".to_string(), 5.0);

        let mut gradients = HashMap::new();
        gradients.insert("x".to_string(), 2.0);

        // Test SGD
        let mut sgd = SGD::new(0.1, 0.0);
        sgd.step(&mut params, &gradients);
        assert!((params["x"] - 4.8).abs() < 1e-6);

        // Test Adam
        params.insert("x".to_string(), 5.0);
        let mut adam = Adam::new(0.1);
        adam.step(&mut params, &gradients);
        assert!(params["x"] < 5.0); // Should decrease
    }

    #[test]
    fn test_parameter_shift() {
        let executor = |params: &[f64]| -> f64 { params[0].cos() + params[1].sin() };

        let qad = QuantumAutoDiff::new(executor);
        let params = vec![PI / 4.0, PI / 3.0];

        let gradients = qad
            .parameter_shift_gradients(&params, PI / 2.0)
            .expect("parameter shift gradients should succeed");
        assert_eq!(gradients.len(), 2);
    }
}
