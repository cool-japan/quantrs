use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::HashMap;
use std::fmt;

/// Spall's recommended step-size decay exponent for SPSA's gain sequence
/// `a_k = a / (k+1)^alpha`.
const SPSA_ALPHA: f64 = 0.602;
/// Spall's recommended perturbation decay exponent for SPSA's gain sequence
/// `c_k = c / (k+1)^gamma`.
const SPSA_GAMMA: f64 = 0.101;

/// Optimization method to use for training quantum machine learning models
#[derive(Debug, Clone, Copy)]
pub enum OptimizationMethod {
    /// Gradient descent
    GradientDescent,

    /// Adam optimizer
    Adam,

    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA,

    /// L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
    LBFGS,

    /// Quantum Natural Gradient
    QuantumNaturalGradient,

    /// SciRS2 Adam optimizer
    SciRS2Adam,

    /// SciRS2 L-BFGS optimizer
    SciRS2LBFGS,

    /// SciRS2 Conjugate Gradient
    SciRS2CG,
}

/// Optimizer for quantum machine learning models
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Gradient descent
    GradientDescent {
        /// Learning rate
        learning_rate: f64,
    },

    /// Adam optimizer
    Adam {
        /// Learning rate
        learning_rate: f64,

        /// Beta1 parameter
        beta1: f64,

        /// Beta2 parameter
        beta2: f64,

        /// Epsilon parameter
        epsilon: f64,

        /// First-moment (momentum) accumulator. Lazily (re)sized to match
        /// the parameter vector on the first `update_parameters` call.
        m: Array1<f64>,

        /// Second-moment (uncentered variance) accumulator. Lazily
        /// (re)sized to match the parameter vector on the first
        /// `update_parameters` call.
        v: Array1<f64>,
    },

    /// SPSA optimizer
    SPSA {
        /// Learning rate
        learning_rate: f64,

        /// Perturbation size
        perturbation: f64,
    },

    /// Quantum Natural Gradient optimizer.
    ///
    /// This variant stores the scalar hyper-parameters only.  Callers are expected
    /// to pre-condition gradients through `QuantumAutoDiff::natural_gradients()`
    /// (which requires a circuit executor closure) and then pass the resulting
    /// natural-gradient vector to `update_parameters`.  The `regularization` field
    /// is used as additive damping: `Δθ_i = −lr · g_i / (1 + reg)`.
    QuantumNaturalGradient {
        /// Learning rate
        learning_rate: f64,
        /// Tikhonov regularisation added to the QFIM diagonal before inversion
        regularization: f64,
    },

    /// SciRS2-based optimizers: Adam, L-BFGS (two-loop recursion), and
    /// nonlinear Conjugate Gradient (Fletcher-Reeves/Polak-Ribiere), each
    /// with real per-parameter state carried between `update_parameters`
    /// calls.
    SciRS2 {
        /// Optimizer method: "adam", "lbfgs", or "cg"
        method: String,
        /// Configuration parameters
        config: HashMap<String, f64>,
        /// Adam first-moment accumulator (method == "adam")
        adam_m: Array1<f64>,
        /// Adam second-moment accumulator (method == "adam")
        adam_v: Array1<f64>,
        /// Bounded curvature-pair history `(s_k, y_k)` for L-BFGS's
        /// two-loop recursion (method == "lbfgs"), newest last, truncated to
        /// `config["m"]` entries.
        lbfgs_history: Vec<(Array1<f64>, Array1<f64>)>,
        /// Previous call's parameter vector, used by both L-BFGS and CG to
        /// form the next curvature pair / conjugate direction.
        prev_params: Option<Array1<f64>>,
        /// Previous call's gradient vector.
        prev_gradient: Option<Array1<f64>>,
        /// Previous CG search direction (method == "cg").
        cg_direction: Option<Array1<f64>>,
    },
}

impl Optimizer {
    /// Creates a new optimizer with default parameters
    pub fn new(method: OptimizationMethod) -> Self {
        match method {
            OptimizationMethod::GradientDescent => Optimizer::GradientDescent {
                learning_rate: 0.01,
            },
            OptimizationMethod::Adam => Optimizer::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                m: Array1::zeros(0),
                v: Array1::zeros(0),
            },
            OptimizationMethod::SPSA => Optimizer::SPSA {
                learning_rate: 0.01,
                perturbation: 0.01,
            },
            OptimizationMethod::LBFGS => {
                // Default to Adam as LBFGS is not implemented yet
                Optimizer::Adam {
                    learning_rate: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    m: Array1::zeros(0),
                    v: Array1::zeros(0),
                }
            }
            OptimizationMethod::QuantumNaturalGradient => Optimizer::QuantumNaturalGradient {
                learning_rate: 0.01,
                regularization: 1e-3,
            },
            OptimizationMethod::SciRS2Adam => {
                let mut config = HashMap::new();
                config.insert("learning_rate".to_string(), 0.001);
                config.insert("beta1".to_string(), 0.9);
                config.insert("beta2".to_string(), 0.999);
                config.insert("epsilon".to_string(), 1e-8);
                Optimizer::SciRS2 {
                    method: "adam".to_string(),
                    config,
                    adam_m: Array1::zeros(0),
                    adam_v: Array1::zeros(0),
                    lbfgs_history: Vec::new(),
                    prev_params: None,
                    prev_gradient: None,
                    cg_direction: None,
                }
            }
            OptimizationMethod::SciRS2LBFGS => {
                let mut config = HashMap::new();
                config.insert("m".to_string(), 10.0); // Memory size
                config.insert("c1".to_string(), 1e-4);
                config.insert("c2".to_string(), 0.9);
                config.insert("learning_rate".to_string(), 0.1);
                Optimizer::SciRS2 {
                    method: "lbfgs".to_string(),
                    config,
                    adam_m: Array1::zeros(0),
                    adam_v: Array1::zeros(0),
                    lbfgs_history: Vec::new(),
                    prev_params: None,
                    prev_gradient: None,
                    cg_direction: None,
                }
            }
            OptimizationMethod::SciRS2CG => {
                let mut config = HashMap::new();
                config.insert("beta_method".to_string(), 0.0); // Fletcher-Reeves
                config.insert("restart_threshold".to_string(), 100.0);
                config.insert("learning_rate".to_string(), 0.01);
                Optimizer::SciRS2 {
                    method: "cg".to_string(),
                    config,
                    adam_m: Array1::zeros(0),
                    adam_v: Array1::zeros(0),
                    lbfgs_history: Vec::new(),
                    prev_params: None,
                    prev_gradient: None,
                    cg_direction: None,
                }
            }
        }
    }

    /// Updates parameters based on gradients.
    ///
    /// Each variant now carries and mutates its own real optimizer state
    /// (Adam's first/second moments, L-BFGS's curvature-pair history, CG's
    /// previous conjugate direction, ...), so this takes `&mut self`.
    pub fn update_parameters(
        &mut self,
        parameters: &mut Array1<f64>,
        gradients: &ArrayView1<f64>,
        iteration: usize,
    ) -> Result<()> {
        match self {
            Optimizer::GradientDescent { learning_rate } => {
                // Simple gradient descent update
                for i in 0..parameters.len() {
                    parameters[i] -= *learning_rate * gradients[i];
                }
                Ok(())
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                m,
                v,
            } => {
                Self::adam_update(
                    parameters,
                    gradients,
                    iteration,
                    *learning_rate,
                    *beta1,
                    *beta2,
                    *epsilon,
                    m,
                    v,
                );
                Ok(())
            }
            Optimizer::SPSA {
                learning_rate,
                perturbation,
            } => {
                // Spall's canonical two-gain-sequence SPSA schedule:
                // a_k = a / (k+1)^alpha decays the step size, while
                // c_k = c / (k+1)^gamma reflects the (decaying) magnitude of
                // the simultaneous perturbation used upstream to estimate
                // `gradients`; a larger effective c_k means a noisier
                // one-shot gradient estimate, so we damp the step by
                // (1 + c_k) to avoid overreacting to it.
                let k = iteration as f64 + 1.0;
                let a_k = *learning_rate / k.powf(SPSA_ALPHA);
                let c_k = *perturbation / k.powf(SPSA_GAMMA);
                let damping = 1.0 + c_k;
                let step = a_k / damping;
                for i in 0..parameters.len() {
                    parameters[i] -= step * gradients[i];
                }
                Ok(())
            }
            Optimizer::QuantumNaturalGradient {
                learning_rate,
                regularization,
            } => {
                // Gradients are expected to be pre-conditioned natural gradients
                // (computed via `QuantumAutoDiff::natural_gradients()`).
                // Apply Tikhonov-damped update: Δθ = -lr * g / (1 + reg).
                let damp = 1.0 + *regularization;
                for i in 0..parameters.len() {
                    parameters[i] -= *learning_rate * gradients[i] / damp;
                }
                Ok(())
            }
            Optimizer::SciRS2 {
                method,
                config,
                adam_m,
                adam_v,
                lbfgs_history,
                prev_params,
                prev_gradient,
                cg_direction,
            } => match method.as_str() {
                "adam" => {
                    let learning_rate = config.get("learning_rate").copied().unwrap_or(0.001);
                    let beta1 = config.get("beta1").copied().unwrap_or(0.9);
                    let beta2 = config.get("beta2").copied().unwrap_or(0.999);
                    let epsilon = config.get("epsilon").copied().unwrap_or(1e-8);
                    Self::adam_update(
                        parameters,
                        gradients,
                        iteration,
                        learning_rate,
                        beta1,
                        beta2,
                        epsilon,
                        adam_m,
                        adam_v,
                    );
                    Ok(())
                }
                "lbfgs" => {
                    Self::lbfgs_update(
                        parameters,
                        gradients,
                        config,
                        lbfgs_history,
                        prev_params,
                        prev_gradient,
                    );
                    Ok(())
                }
                "cg" => {
                    Self::cg_update(
                        parameters,
                        gradients,
                        iteration,
                        config,
                        prev_params,
                        prev_gradient,
                        cg_direction,
                    );
                    Ok(())
                }
                _ => Err(MLError::InvalidConfiguration(format!(
                    "Unknown SciRS2 optimizer method: {}",
                    method
                ))),
            },
        }
    }

    /// Real Adam update: tracks biased first/second moment estimates and
    /// applies bias-corrected parameter updates.
    #[allow(clippy::too_many_arguments)]
    fn adam_update(
        parameters: &mut Array1<f64>,
        gradients: &ArrayView1<f64>,
        iteration: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        m: &mut Array1<f64>,
        v: &mut Array1<f64>,
    ) {
        let n = parameters.len();
        if m.len() != n {
            *m = Array1::zeros(n);
            *v = Array1::zeros(n);
        }
        let t = iteration as f64 + 1.0;
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i] * gradients[i];
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            parameters[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    /// Real nonlinear Conjugate Gradient update (Fletcher-Reeves or
    /// Polak-Ribiere, selected by `config["beta_method"]`), with periodic
    /// restart to steepest descent every `config["restart_threshold"]`
    /// iterations (or whenever no prior direction is available).
    #[allow(clippy::too_many_arguments)]
    fn cg_update(
        parameters: &mut Array1<f64>,
        gradients: &ArrayView1<f64>,
        iteration: usize,
        config: &HashMap<String, f64>,
        prev_params: &mut Option<Array1<f64>>,
        prev_gradient: &mut Option<Array1<f64>>,
        cg_direction: &mut Option<Array1<f64>>,
    ) {
        let n = parameters.len();
        let learning_rate = config.get("learning_rate").copied().unwrap_or(0.01);
        let beta_method = config.get("beta_method").copied().unwrap_or(0.0);
        let restart_threshold = config
            .get("restart_threshold")
            .copied()
            .unwrap_or(100.0)
            .max(1.0) as usize;

        // Snapshot the pre-update iterate: this is theta_t, stored as
        // `prev_params` for the *next* call (which will need it to relate
        // to theta_{t+1}). Restart to steepest descent whenever we lack a
        // prior gradient/direction, or every `restart_threshold` iterations.
        let theta_t = parameters.clone();
        let restart =
            prev_gradient.is_none() || cg_direction.is_none() || iteration % restart_threshold == 0;

        let direction = if !restart {
            let prev_g = prev_gradient.as_ref().expect("checked Some above");
            let prev_d = cg_direction.as_ref().expect("checked Some above");
            let denom: f64 = prev_g.iter().map(|g| g * g).sum();
            let beta = if denom.abs() > 1e-15 {
                if beta_method < 0.5 {
                    // Fletcher-Reeves
                    let numer: f64 = gradients.iter().map(|g| g * g).sum();
                    (numer / denom).max(0.0)
                } else {
                    // Polak-Ribiere (clamped to be non-negative, i.e.
                    // automatic restart to steepest descent when negative)
                    let numer: f64 = gradients
                        .iter()
                        .zip(prev_g.iter())
                        .map(|(g, gp)| g * (g - gp))
                        .sum();
                    (numer / denom).max(0.0)
                }
            } else {
                0.0
            };
            let mut d = Array1::zeros(n);
            for i in 0..n {
                d[i] = -gradients[i] + beta * prev_d[i];
            }
            d
        } else {
            let mut d = Array1::zeros(n);
            for i in 0..n {
                d[i] = -gradients[i];
            }
            d
        };

        for i in 0..n {
            parameters[i] += learning_rate * direction[i];
        }

        *prev_params = Some(theta_t);
        *prev_gradient = Some(gradients.to_owned());
        *cg_direction = Some(direction);
    }

    /// Real L-BFGS update using the standard two-loop recursion over a
    /// bounded history of curvature pairs `(s_k = Δθ_k, y_k = Δg_k)`
    /// (`config["m"]` most recent pairs), with a fixed damped step in place
    /// of a Wolfe line search.
    fn lbfgs_update(
        parameters: &mut Array1<f64>,
        gradients: &ArrayView1<f64>,
        config: &HashMap<String, f64>,
        history: &mut Vec<(Array1<f64>, Array1<f64>)>,
        prev_params: &mut Option<Array1<f64>>,
        prev_gradient: &mut Option<Array1<f64>>,
    ) {
        let n = parameters.len();
        let memory = config.get("m").copied().unwrap_or(10.0).max(1.0) as usize;
        let learning_rate = config.get("learning_rate").copied().unwrap_or(0.1);

        // Snapshot the pre-update iterate theta_t; this (not the post-update
        // theta_{t+1}) is what the *next* call needs as its "previous"
        // iterate to form the following curvature pair.
        let theta_t = parameters.clone();

        // Form the newest curvature pair from the previous call's iterate.
        if let (Some(prev_p), Some(prev_g)) = (prev_params.as_ref(), prev_gradient.as_ref()) {
            let s = &theta_t - prev_p;
            let y = gradients.to_owned() - prev_g;
            let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            // Only accept curvature-condition-satisfying pairs to keep the
            // implied Hessian approximation positive definite.
            if sy > 1e-10 {
                history.push((s, y));
                while history.len() > memory {
                    history.remove(0);
                }
            }
        }

        // Two-loop recursion computing r ≈ H_k * g_k.
        let mut q = gradients.to_owned();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y) in history.iter().rev() {
            let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let rho = 1.0 / sy;
            let alpha = rho * s.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
            for i in 0..n {
                q[i] -= alpha * y[i];
            }
            alphas.push((rho, alpha));
        }

        let gamma = match history.last() {
            Some((s, y)) => {
                let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                let yy: f64 = y.iter().map(|v| v * v).sum();
                if yy.abs() > 1e-15 {
                    sy / yy
                } else {
                    1.0
                }
            }
            None => 1.0,
        };
        let mut r = q.mapv(|x| x * gamma);

        for (idx, (s, y)) in history.iter().enumerate() {
            let (rho, alpha) = alphas[history.len() - 1 - idx];
            let beta = rho * y.iter().zip(r.iter()).map(|(a, b)| a * b).sum::<f64>();
            for i in 0..n {
                r[i] += s[i] * (alpha - beta);
            }
        }

        // r approximates H_k * g_k; the descent direction is -r.
        for i in 0..n {
            parameters[i] -= learning_rate * r[i];
        }

        *prev_params = Some(theta_t);
        *prev_gradient = Some(gradients.to_owned());
    }
}

/// Objective function for optimization
pub trait ObjectiveFunction {
    /// Evaluates the objective function at the given parameters
    fn evaluate(&self, parameters: &ArrayView1<f64>) -> Result<f64>;

    /// Computes the gradient of the objective function
    fn gradient(&self, parameters: &ArrayView1<f64>) -> Result<Array1<f64>> {
        // Default implementation uses finite differences
        let epsilon = 1e-6;
        let n = parameters.len();
        let mut gradient = Array1::zeros(n);

        let f0 = self.evaluate(parameters)?;

        for i in 0..n {
            let mut params_plus = parameters.to_owned();
            params_plus[i] += epsilon;

            let f_plus = self.evaluate(&params_plus.view())?;

            gradient[i] = (f_plus - f0) / epsilon;
        }

        Ok(gradient)
    }
}

impl fmt::Display for OptimizationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationMethod::GradientDescent => write!(f, "Gradient Descent"),
            OptimizationMethod::Adam => write!(f, "Adam"),
            OptimizationMethod::SPSA => write!(f, "SPSA"),
            OptimizationMethod::LBFGS => write!(f, "L-BFGS"),
            OptimizationMethod::QuantumNaturalGradient => write!(f, "Quantum Natural Gradient"),
            OptimizationMethod::SciRS2Adam => write!(f, "SciRS2 Adam"),
            OptimizationMethod::SciRS2LBFGS => write!(f, "SciRS2 L-BFGS"),
            OptimizationMethod::SciRS2CG => write!(f, "SciRS2 Conjugate Gradient"),
        }
    }
}

impl fmt::Display for Optimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Optimizer::GradientDescent { learning_rate } => {
                write!(f, "Gradient Descent (learning_rate: {})", learning_rate)
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                ..
            } => {
                write!(
                    f,
                    "Adam (learning_rate: {}, beta1: {}, beta2: {}, epsilon: {})",
                    learning_rate, beta1, beta2, epsilon
                )
            }
            Optimizer::SPSA {
                learning_rate,
                perturbation,
            } => {
                write!(
                    f,
                    "SPSA (learning_rate: {}, perturbation: {})",
                    learning_rate, perturbation
                )
            }
            Optimizer::QuantumNaturalGradient {
                learning_rate,
                regularization,
            } => {
                write!(
                    f,
                    "Quantum Natural Gradient (learning_rate: {}, regularization: {})",
                    learning_rate, regularization
                )
            }
            Optimizer::SciRS2 { method, config, .. } => {
                write!(f, "SciRS2 {} with config: {:?}", method, config)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `Optimizer::Adam` must behave like real Adam (with
    /// persistent first/second moment accumulators), not degrade into plain
    /// gradient descent. On a constant gradient, Adam's first update step
    /// size is `learning_rate * sign(gradient)` (up to the bias-correction
    /// ratio, which is exactly 1 on the very first step), which is *not*
    /// equal to `learning_rate * gradient` unless `|gradient| == 1`.
    #[test]
    fn test_adam_is_not_plain_gradient_descent() {
        let mut optimizer = Optimizer::new(OptimizationMethod::Adam);
        let mut params = Array1::from_vec(vec![1.0, 1.0]);
        let gradients = Array1::from_vec(vec![10.0, 10.0]);

        optimizer
            .update_parameters(&mut params, &gradients.view(), 0)
            .expect("Adam update should succeed");

        // Plain gradient descent with lr=0.01 would move by exactly 0.1;
        // Adam's normalized step is close to +/- learning_rate instead.
        let plain_gd_step = 0.01 * 10.0;
        let actual_step = 1.0 - params[0];
        assert!(
            (actual_step - plain_gd_step).abs() > 1e-3,
            "Adam step ({actual_step}) should differ from plain GD step ({plain_gd_step})"
        );
        // The normalized Adam step on the very first iteration is
        // approximately learning_rate (moment ratio ~ sign(gradient)).
        assert!((actual_step - 0.01).abs() < 1e-3);
    }

    #[test]
    fn test_adam_moments_persist_across_calls() {
        // A gradient sign-flip (+1.0 then -1.0) exercises Adam's momentum
        // smoothing: the *same* second gradient produces a much smaller
        // step for an optimizer warmed up by the first call than for a
        // fresh optimizer seeing that gradient for the first time at the
        // same iteration index (so bias correction alone cannot explain the
        // difference) -- proof that `m`/`v` genuinely persist in the
        // optimizer's own state instead of being recomputed from scratch
        // (or ignored) on every call.
        let mut warmed = Optimizer::new(OptimizationMethod::Adam);
        let mut params_warm = Array1::from_vec(vec![0.0]);
        let g1 = Array1::from_vec(vec![1.0]);
        warmed
            .update_parameters(&mut params_warm, &g1.view(), 0)
            .expect("update 1");
        let g2 = Array1::from_vec(vec![-1.0]);
        let before_second = params_warm[0];
        warmed
            .update_parameters(&mut params_warm, &g2.view(), 1)
            .expect("update 2");
        let warmed_step = params_warm[0] - before_second;

        let mut fresh = Optimizer::new(OptimizationMethod::Adam);
        let mut params_fresh = Array1::from_vec(vec![0.0]);
        fresh
            .update_parameters(&mut params_fresh, &g2.view(), 1)
            .expect("fresh update");
        let fresh_step = params_fresh[0];

        assert!(
            (warmed_step - fresh_step).abs() > 1e-3,
            "step with accumulated momentum ({warmed_step}) should differ substantially from a \
             fresh optimizer's step ({fresh_step}) given the same gradient/iteration"
        );
    }

    #[test]
    fn test_spsa_uses_decaying_gain_sequence() {
        // SPSA's defining feature vs plain GD is a step size that decays
        // with the iteration count; verify the applied step actually
        // shrinks as `iteration` grows for an identical gradient.
        let mut optimizer = Optimizer::new(OptimizationMethod::SPSA);
        let gradients = Array1::from_vec(vec![1.0]);

        let mut params_early = Array1::from_vec(vec![0.0]);
        optimizer
            .update_parameters(&mut params_early, &gradients.view(), 0)
            .expect("update at iteration 0");

        let mut optimizer2 = Optimizer::new(OptimizationMethod::SPSA);
        let mut params_late = Array1::from_vec(vec![0.0]);
        optimizer2
            .update_parameters(&mut params_late, &gradients.view(), 1000)
            .expect("update at iteration 1000");

        assert!(
            params_early[0].abs() > params_late[0].abs(),
            "SPSA step at iteration 0 ({}) should be larger than at iteration 1000 ({})",
            params_early[0].abs(),
            params_late[0].abs()
        );
    }

    #[test]
    fn test_scirs2_cg_direction_differs_from_gradient_descent() {
        // Nonlinear CG's direction after the first restart-free step should
        // incorporate the Fletcher-Reeves beta term, differing from plain
        // steepest descent once a second gradient is supplied.
        let mut optimizer = Optimizer::new(OptimizationMethod::SciRS2CG);
        let mut params = Array1::from_vec(vec![1.0, 1.0]);

        // First call restarts to steepest descent (no history yet).
        let g1 = Array1::from_vec(vec![1.0, 0.0]);
        optimizer
            .update_parameters(&mut params, &g1.view(), 0)
            .expect("cg update 1");
        let after_first = params.clone();

        // Second call (iteration=1, below restart_threshold) should apply a
        // Fletcher-Reeves-conjugated direction, not steepest descent again.
        let g2 = Array1::from_vec(vec![0.5, 0.5]);
        optimizer
            .update_parameters(&mut params, &g2.view(), 1)
            .expect("cg update 2");

        let learning_rate = 0.01;
        let plain_steepest_descent_step = -learning_rate * g2[0];
        let actual_step = params[0] - after_first[0];
        assert!(
            (actual_step - plain_steepest_descent_step).abs() > 1e-8,
            "CG step ({actual_step}) should differ from plain steepest descent ({plain_steepest_descent_step})"
        );
    }

    #[test]
    fn test_scirs2_lbfgs_uses_curvature_history() {
        // After two calls, L-BFGS should have recorded a curvature pair and
        // used it (via the two-loop recursion) rather than falling back to
        // a fixed-scale gradient step every time.
        let mut optimizer = Optimizer::new(OptimizationMethod::SciRS2LBFGS);
        let mut params = Array1::from_vec(vec![2.0, -1.0]);

        let g1 = Array1::from_vec(vec![2.0, -1.0]);
        optimizer
            .update_parameters(&mut params, &g1.view(), 0)
            .expect("lbfgs update 1");

        let g2 = Array1::from_vec(vec![1.0, -0.5]);
        let before_second = params.clone();
        optimizer
            .update_parameters(&mut params, &g2.view(), 1)
            .expect("lbfgs update 2");

        // A fixed-scale (learning_rate-only) gradient step would move
        // exactly `-learning_rate * g2`; the two-loop-recursion direction
        // (scaled by accumulated curvature) should differ from that.
        let learning_rate = 0.1;
        let naive_step = -learning_rate * g2[0];
        let actual_step = params[0] - before_second[0];
        assert!(
            (actual_step - naive_step).abs() > 1e-8,
            "L-BFGS step ({actual_step}) should differ from a naive scaled-gradient step ({naive_step})"
        );
    }

    #[test]
    fn test_quantum_natural_gradient_damping_unchanged() {
        // Regression guard: this variant's Tikhonov damping behavior must
        // remain unaffected by the Adam/CG/L-BFGS rewrite.
        let mut optimizer = Optimizer::QuantumNaturalGradient {
            learning_rate: 0.1,
            regularization: 1.0,
        };
        let mut params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![2.0]);
        optimizer
            .update_parameters(&mut params, &gradients.view(), 0)
            .expect("QNG update");
        // Δθ = -lr * g / (1 + reg) = -0.1 * 2 / 2 = -0.1
        assert!((params[0] - 0.9).abs() < 1e-9);
    }
}
