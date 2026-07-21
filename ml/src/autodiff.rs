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
    /// Circuit executor backing `ComputationNode::Expectation` nodes: given
    /// the ordered values of `circuit_params` and the observable name,
    /// returns the real expectation value ⟨observable⟩(params) (e.g. by
    /// simulating a parameterised circuit). Evaluating or differentiating a
    /// graph containing an `Expectation` node without one configured
    /// returns an honest `MLError::NotSupported` rather than a fabricated
    /// placeholder value.
    executor: Option<Box<dyn Fn(&[f64], &str) -> f64>>,
}

impl AutoDiff {
    /// Create a new AutoDiff engine
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            graph: None,
            forward_cache: HashMap::new(),
            executor: None,
        }
    }

    /// Attach a circuit executor used to evaluate `ComputationNode::Expectation`
    /// nodes. `executor(param_values, observable)` must return the real
    /// expectation value of `observable` for the circuit parameterised by
    /// `param_values` (in the same order as that node's `circuit_params`).
    pub fn with_executor<F>(mut self, executor: F) -> Self
    where
        F: Fn(&[f64], &str) -> f64 + 'static,
    {
        self.executor = Some(Box::new(executor));
        self
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
                let values = self.circuit_param_values(circuit_params)?;
                let executor = self.executor.as_ref().ok_or_else(|| {
                    MLError::NotSupported(
                        "ComputationNode::Expectation requires a circuit executor; call \
                         AutoDiff::with_executor() before forward()/backward()"
                            .to_string(),
                    )
                })?;
                Ok(executor(&values, observable))
            }
        }
    }

    /// Look up the current values of a list of registered parameter names,
    /// in order, for use as circuit-executor input.
    fn circuit_param_values(&self, circuit_params: &[String]) -> Result<Vec<f64>> {
        circuit_params
            .iter()
            .map(|name| {
                self.parameters.get(name).map(|p| p.value).ok_or_else(|| {
                    MLError::InvalidConfiguration(format!("Unknown parameter: {}", name))
                })
            })
            .collect()
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
            ComputationNode::Expectation {
                circuit_params,
                observable,
            } => {
                // Use the exact parameter shift rule, evaluating the real
                // executor at θ±π/2 for each circuit parameter in turn.
                for (index, param_name) in circuit_params.iter().enumerate() {
                    let shift_grad =
                        self.parameter_shift_gradient(circuit_params, observable, index, PI / 2.0)?;
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

    /// Compute the gradient of `⟨observable⟩` with respect to the
    /// `index`-th entry of `circuit_params` using the exact two-point
    /// parameter-shift rule: `(E(θ+shift) - E(θ-shift)) / (2 sin(shift))`,
    /// evaluated by calling into the configured executor.
    fn parameter_shift_gradient(
        &self,
        circuit_params: &[String],
        observable: &str,
        index: usize,
        shift: f64,
    ) -> Result<f64> {
        let executor = self.executor.as_ref().ok_or_else(|| {
            MLError::NotSupported(
                "parameter_shift_gradient requires a circuit executor; call \
                 AutoDiff::with_executor() before forward()/backward()"
                    .to_string(),
            )
        })?;
        let mut values = self.circuit_param_values(circuit_params)?;
        if index >= values.len() {
            return Err(MLError::InvalidParameter(format!(
                "parameter index {index} out of range for {} circuit parameters",
                values.len()
            )));
        }

        let original = values[index];
        values[index] = original + shift;
        let plus = executor(&values, observable);
        values[index] = original - shift;
        let minus = executor(&values, observable);

        Ok((plus - minus) / (2.0 * shift.sin()))
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

    /// Compute element of quantum Fisher information matrix using 4-point parameter-shift QFIM formula.
    ///
    /// F_ij = (E(θ+π/2·e_i+π/2·e_j) - E(θ+π/2·e_i-π/2·e_j)
    ///        - E(θ-π/2·e_i+π/2·e_j) + E(θ-π/2·e_i-π/2·e_j)) / 4
    fn compute_fisher_element(&self, params: &[f64], i: usize, j: usize) -> Result<f64> {
        let shift = PI / 2.0;

        let mut p_pp = params.to_vec();
        let mut p_pm = params.to_vec();
        let mut p_mp = params.to_vec();
        let mut p_mm = params.to_vec();

        p_pp[i] += shift;
        p_pp[j] += shift;

        p_pm[i] += shift;
        p_pm[j] -= shift;

        p_mp[i] -= shift;
        p_mp[j] += shift;

        p_mm[i] -= shift;
        p_mm[j] -= shift;

        let e_pp = (self.executor)(&p_pp);
        let e_pm = (self.executor)(&p_pm);
        let e_mp = (self.executor)(&p_mp);
        let e_mm = (self.executor)(&p_mm);

        Ok((e_pp - e_pm - e_mp + e_mm) / 4.0)
    }

    /// Solve linear system A·x = b using Gaussian elimination with partial pivoting.
    ///
    /// Returns `Ok(x)` on success, `Err(NumericalError)` if the matrix is singular
    /// (i.e., |pivot| < 1e-12 at any elimination step).
    fn solve_linear_system(&self, matrix: &Array2<f64>, rhs: &[f64]) -> Result<Vec<f64>> {
        let n = rhs.len();
        if matrix.nrows() != n || matrix.ncols() != n {
            return Err(MLError::DimensionMismatch(format!(
                "Matrix ({} x {}) incompatible with rhs length {}",
                matrix.nrows(),
                matrix.ncols(),
                n
            )));
        }

        // Build augmented matrix [A | b]
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row: Vec<f64> = (0..n).map(|j| matrix[[i, j]]).collect();
                row.push(rhs[i]);
                row
            })
            .collect();

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot row: row with max |a[row][k]| for row >= k
            let mut max_val = a[k][k].abs();
            let mut max_idx = k;
            for row in (k + 1)..n {
                let val = a[row][k].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = row;
                }
            }

            if max_val < 1e-12 {
                return Err(MLError::NumericalError(format!(
                    "Singular matrix: |pivot| = {:.2e} < 1e-12 at column {}",
                    max_val, k
                )));
            }

            // Swap rows k and max_idx
            if max_idx != k {
                a.swap(k, max_idx);
            }

            let pivot = a[k][k];

            // Eliminate below pivot
            for i in (k + 1)..n {
                let factor = a[i][k] / pivot;
                for col in k..=n {
                    let sub = factor * a[k][col];
                    a[i][col] -= sub;
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut sum = a[i][n]; // rhs column
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }

        Ok(x)
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

    #[test]
    fn test_expectation_node_without_executor_errors_honestly() {
        // Regression test: an `Expectation` node evaluated/backpropagated
        // without a configured executor must return an honest
        // `MLError::NotSupported`, not a fabricated placeholder value.
        let mut autodiff = AutoDiff::new();
        autodiff.register_parameter(DifferentiableParam::new("theta", PI / 4.0));
        autodiff.set_graph(ComputationNode::Expectation {
            circuit_params: vec!["theta".to_string()],
            observable: "Z".to_string(),
        });

        let forward_result = autodiff.forward();
        assert!(forward_result.is_err());
        match forward_result {
            Err(MLError::NotSupported(_)) => {}
            other => panic!("expected MLError::NotSupported, got {other:?}"),
        }
    }

    #[test]
    fn test_expectation_node_real_parameter_shift_gradient() {
        // Regression test: with a real executor attached, forward() must
        // return the executor's actual value (not a hardcoded placeholder),
        // and backward() must compute the true parameter-shift derivative
        // instead of the previous hardcoded `0.5`.
        //
        // Observable: ⟨Z⟩(θ) = cos(θ), whose exact derivative is -sin(θ).
        let executor = |params: &[f64], _observable: &str| -> f64 { params[0].cos() };

        let mut autodiff = AutoDiff::new().with_executor(executor);
        let theta = PI / 3.0;
        autodiff.register_parameter(DifferentiableParam::new("theta", theta));
        autodiff.set_graph(ComputationNode::Expectation {
            circuit_params: vec!["theta".to_string()],
            observable: "Z".to_string(),
        });

        let forward_value = autodiff.forward().expect("forward should succeed");
        assert!((forward_value - theta.cos()).abs() < 1e-9);

        autodiff.backward(1.0).expect("backward should succeed");
        let gradients = autodiff.gradients();

        let expected_gradient = -theta.sin();
        assert!(
            (gradients["theta"] - expected_gradient).abs() < 1e-6,
            "expected d/dtheta cos(theta) = {expected_gradient}, got {}",
            gradients["theta"]
        );
        // Must not be the old hardcoded placeholder value.
        assert!((gradients["theta"] - 0.5).abs() > 1e-3);
    }
}
