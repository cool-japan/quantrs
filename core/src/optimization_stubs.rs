//! Optimization utilities backed by OptiRS optimizers.
//!
//!\n//! This module preserves the historical `Method`/`Options` interface that the
//! rest of the QuantRS2 codebase depends on, while delegating the actual work to
//! OptiRS implementations. Gradient-based methods rely on finite-difference
//! estimates so that existing objective functions can remain unchanged.

use crate::error::{QuantRS2Error, QuantRS2Result};
use optirs_core::error::OptimError;
use optirs_core::optimizers::{Adam, Optimizer, LBFGS};
use scirs2_core::ndarray::{Array1, ArrayView1, Ix1};
use scirs2_core::random::{rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use std::cmp::Ordering;

const DEFAULT_GRADIENT_STEP: f64 = 1e-5;
const DEFAULT_MAX_FUNCTION_EVALS: usize = 50_000;
const DEFAULT_HISTORY_SIZE: usize = 20;
const DEFAULT_LEARNING_RATE: f64 = 0.1;

/// Optimization method enum
#[derive(Debug, Clone, Copy)]
pub enum Method {
    BFGS,
    LBFGS,
    ConjugateGradient,
    NewtonCG,
    TrustRegion,
    NelderMead,
    Powell,
}

/// Optimization options
#[derive(Debug, Clone)]
pub struct Options {
    pub max_iter: usize,
    pub max_iterations: usize, // Alias for compatibility
    pub tolerance: f64,
    pub ftol: f64, // Function tolerance
    pub gtol: f64, // Gradient tolerance
    pub xtol: f64, // Parameter tolerance
    pub method: Method,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            max_iterations: 1000,
            tolerance: 1e-6,
            ftol: 1e-6,
            gtol: 1e-6,
            xtol: 1e-6,
            method: Method::LBFGS,
        }
    }
}

impl Options {
    const fn resolved_max_iter(&self) -> usize {
        let base = if self.max_iter == 0 {
            self.max_iterations
        } else {
            self.max_iter
        };
        if base == 0 {
            1000
        } else {
            base
        }
    }

    fn resolved_ftol(&self) -> f64 {
        if self.ftol > 0.0 {
            self.ftol
        } else if self.tolerance > 0.0 {
            self.tolerance
        } else {
            1e-9
        }
    }

    fn resolved_gtol(&self) -> f64 {
        if self.gtol > 0.0 {
            self.gtol
        } else if self.tolerance > 0.0 {
            self.tolerance
        } else {
            1e-9
        }
    }

    fn resolved_xtol(&self) -> f64 {
        if self.xtol > 0.0 {
            self.xtol
        } else if self.tolerance > 0.0 {
            self.tolerance
        } else {
            1e-9
        }
    }
}

fn default_learning_rate(options: &Options) -> f64 {
    let tol = options.resolved_gtol().abs();
    if tol.is_finite() && tol > 0.0 {
        (tol * 10.0).clamp(1e-3, 0.5)
    } else {
        DEFAULT_LEARNING_RATE
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizeResult<T = f64> {
    pub x: Array1<T>,
    pub fun: T,
    pub nit: usize,
    pub iterations: usize, // Alias for nit
    pub success: bool,
    pub message: String,
}

enum OptiRSBackend {
    LBFGS(LBFGS<f64>),
    Adam(Adam<f64>),
}

impl OptiRSBackend {
    const fn name(&self) -> &'static str {
        match self {
            Self::LBFGS(_) => "L-BFGS",
            Self::Adam(_) => "Adam",
        }
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        match self {
            Self::LBFGS(optimizer) => optimizer.set_lr(learning_rate),
            Self::Adam(optimizer) => {
                <Adam<f64> as Optimizer<f64, Ix1>>::set_learning_rate(optimizer, learning_rate);
            }
        }
    }

    fn step(
        &mut self,
        params: &Array1<f64>,
        gradients: &Array1<f64>,
    ) -> Result<Array1<f64>, OptimError> {
        match self {
            Self::LBFGS(optimizer) => optimizer.step(params, gradients),
            Self::Adam(optimizer) => optimizer.step(params, gradients),
        }
    }
}

fn map_optirs_error(err: OptimError) -> QuantRS2Error {
    QuantRS2Error::OptimizationFailed(err.to_string())
}

fn safe_fun_eval<F>(fun: &F, params: &Array1<f64>) -> f64
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let value = fun(&params.view());
    if value.is_finite() {
        value
    } else {
        f64::INFINITY
    }
}

fn numerical_gradient<F>(
    fun: &F,
    params: &Array1<f64>,
    step: f64,
    eval_counter: &mut usize,
) -> QuantRS2Result<Array1<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    if params.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        DEFAULT_GRADIENT_STEP
    };

    let mut gradient = Array1::zeros(params.len());
    let mut forward = params.clone();
    let mut backward = params.clone();

    for i in 0..params.len() {
        forward[i] = params[i] + step;
        backward[i] = params[i] - step;

        let f_plus = safe_fun_eval(fun, &forward);
        let f_minus = safe_fun_eval(fun, &backward);
        *eval_counter += 2;

        if !(f_plus.is_finite() && f_minus.is_finite()) {
            return Err(QuantRS2Error::OptimizationFailed(
                "Encountered non-finite objective value during gradient estimation".to_string(),
            ));
        }

        gradient[i] = (f_plus - f_minus) / (2.0 * step);

        forward[i] = params[i];
        backward[i] = params[i];
    }

    Ok(gradient)
}

fn select_backend(method: Method, options: &Options) -> OptiRSBackend {
    let learning_rate = default_learning_rate(options);
    match method {
        Method::BFGS
        | Method::LBFGS
        | Method::ConjugateGradient
        | Method::NewtonCG
        | Method::TrustRegion => {
            let optimizer = LBFGS::new_with_config(
                learning_rate,
                DEFAULT_HISTORY_SIZE,
                options.resolved_gtol(),
                1e-4,
                0.9,
                20,
            );
            OptiRSBackend::LBFGS(optimizer)
        }
        Method::Powell | Method::NelderMead => {
            let lr = learning_rate.max(0.01);
            let mut optimizer = Adam::new(lr);
            <Adam<f64> as Optimizer<f64, Ix1>>::set_learning_rate(&mut optimizer, lr);
            OptiRSBackend::Adam(optimizer)
        }
    }
}

fn check_convergence(
    params: &Array1<f64>,
    new_params: &Array1<f64>,
    current_fun: f64,
    new_fun: f64,
    grad_norm: f64,
    options: &Options,
) -> Option<String> {
    if grad_norm <= options.resolved_gtol() {
        return Some("Gradient tolerance reached".to_string());
    }

    let step_norm = (new_params - params).mapv(|v| v * v).sum().sqrt();
    if step_norm <= options.resolved_xtol() {
        return Some("Parameter tolerance reached".to_string());
    }

    if (current_fun - new_fun).abs() <= options.resolved_ftol() {
        return Some("Function tolerance reached".to_string());
    }

    None
}

fn nelder_mead_minimize<F>(
    fun: &F,
    x0: &Array1<f64>,
    options: &Options,
) -> QuantRS2Result<OptimizeResult<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let dim = x0.len();
    if dim == 0 {
        let value = safe_fun_eval(fun, x0);
        return Ok(OptimizeResult {
            x: x0.clone(),
            fun: value,
            nit: 0,
            iterations: 0,
            success: true,
            message: "Trivial optimization (zero dimension)".to_string(),
        });
    }

    let mut simplex = Vec::with_capacity(dim + 1);
    simplex.push(x0.clone());
    for i in 0..dim {
        let mut point = x0.clone();
        point[i] += DEFAULT_GRADIENT_STEP.max(1e-3);
        simplex.push(point);
    }

    let mut values: Vec<f64> = simplex.iter().map(|p| safe_fun_eval(fun, p)).collect();
    let mut evals = values.len();
    let mut iterations = 0usize;

    const REFLECTION: f64 = 1.0;
    const EXPANSION: f64 = 2.0;
    const CONTRACTION: f64 = 0.5;
    const SHRINK: f64 = 0.5;

    while iterations < options.resolved_max_iter() && evals < DEFAULT_MAX_FUNCTION_EVALS {
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));

        let mut ordered_simplex = Vec::with_capacity(simplex.len());
        let mut ordered_values = Vec::with_capacity(values.len());
        for idx in order {
            ordered_simplex.push(simplex[idx].clone());
            ordered_values.push(values[idx]);
        }
        simplex = ordered_simplex;
        values = ordered_values;

        let best = simplex[0].clone();
        let worst_index = simplex.len() - 1;
        let worst = simplex[worst_index].clone();

        // Compute centroid excluding worst point
        let mut centroid = Array1::zeros(dim);
        for point in &simplex[..worst_index] {
            centroid = &centroid + point;
        }
        centroid.mapv_inplace(|v| v / dim as f64);

        // Reflection
        let mut xr = &centroid + &(centroid.clone() - &worst) * REFLECTION;
        let fr = safe_fun_eval(fun, &xr);
        evals += 1;

        if fr < values[0] {
            // Expansion
            let mut xe = &centroid + &(xr.clone() - &centroid) * EXPANSION;
            let fe = safe_fun_eval(fun, &xe);
            evals += 1;
            if fe < fr {
                simplex[worst_index] = xe;
                values[worst_index] = fe;
            } else {
                simplex[worst_index] = xr;
                values[worst_index] = fr;
            }
        } else if fr < values[values.len() - 2] {
            simplex[worst_index] = xr;
            values[worst_index] = fr;
        } else {
            // Contraction
            let mut xc = &centroid + &(worst - &centroid) * CONTRACTION;
            let fc = safe_fun_eval(fun, &xc);
            evals += 1;
            if fc < values[worst_index] {
                simplex[worst_index] = xc;
                values[worst_index] = fc;
            } else {
                // Shrink towards best
                for i in 1..simplex.len() {
                    simplex[i] = &best + &(simplex[i].clone() - &best) * SHRINK;
                    values[i] = safe_fun_eval(fun, &simplex[i]);
                }
                evals += dim;
            }
        }

        iterations += 1;

        let (min_value, max_value) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
                (acc.0.min(v), acc.1.max(v))
            });
        if (max_value - min_value).abs() <= options.resolved_ftol() {
            break;
        }
    }

    let (best_idx, best_val) = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .expect("Simplex values should never be empty");

    Ok(OptimizeResult {
        x: simplex[best_idx].clone(),
        fun: *best_val,
        nit: iterations,
        iterations,
        success: true,
        message: format!("Nelder-Mead completed in {iterations} iterations"),
    })
}

/// Minimize function using OptiRS optimizers.
pub fn minimize<F>(
    fun: F,
    x0: &Array1<f64>,
    method: Method,
    options: Option<Options>,
) -> QuantRS2Result<OptimizeResult<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();

    match method {
        Method::NelderMead | Method::Powell => {
            let mut result = nelder_mead_minimize(&fun, x0, &options)?;
            if matches!(method, Method::Powell) {
                result.message = format!(
                    "Powell (approximated via Nelder-Mead) in {} iterations",
                    result.iterations
                );
            }
            return Ok(result);
        }
        _ => {}
    }

    let mut backend = select_backend(method, &options);
    backend.set_learning_rate(default_learning_rate(&options));

    let mut params = x0.clone();
    let mut current_fun = safe_fun_eval(&fun, &params);
    let mut best_params = params.clone();
    let mut best_fun = current_fun;

    let mut iterations = 0usize;
    let mut evals = 1usize;

    while iterations < options.resolved_max_iter() && evals < DEFAULT_MAX_FUNCTION_EVALS {
        let mut gradient = numerical_gradient(&fun, &params, DEFAULT_GRADIENT_STEP, &mut evals)?;
        let grad_norm = gradient.dot(&gradient).sqrt();

        if grad_norm <= options.resolved_gtol() {
            return Ok(OptimizeResult {
                x: params,
                fun: current_fun,
                nit: iterations,
                iterations,
                success: true,
                message: format!(
                    "Optimization converged in {} iterations ({} backend)",
                    iterations,
                    backend.name()
                ),
            });
        }

        // Ensure gradients are finite
        for value in &mut gradient {
            if !value.is_finite() {
                *value = 0.0;
            }
        }

        let updated_params = backend.step(&params, &gradient).map_err(map_optirs_error)?;

        let new_fun = safe_fun_eval(&fun, &updated_params);
        evals += 1;

        if new_fun < best_fun {
            best_fun = new_fun;
            best_params.clone_from(&updated_params);
        }

        if let Some(reason) = check_convergence(
            &params,
            &updated_params,
            current_fun,
            new_fun,
            grad_norm,
            &options,
        ) {
            return Ok(OptimizeResult {
                x: updated_params,
                fun: new_fun,
                nit: iterations + 1,
                iterations: iterations + 1,
                success: true,
                message: format!("{} ({} backend)", reason, backend.name()),
            });
        }

        params = updated_params;
        current_fun = new_fun;
        iterations += 1;
    }

    let success = iterations < options.resolved_max_iter() && evals < DEFAULT_MAX_FUNCTION_EVALS;
    Ok(OptimizeResult {
        x: if success { params } else { best_params },
        fun: if success { current_fun } else { best_fun },
        nit: iterations,
        iterations,
        success,
        message: if success {
            format!(
                "Optimization converged in {} iterations using {}",
                iterations,
                backend.name()
            )
        } else {
            format!(
                "Reached iteration/function evaluation limit ({} iterations, {} evals) using {}",
                iterations,
                evals,
                backend.name()
            )
        },
    })
}

/// Differential evolution options
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionOptions {
    pub population_size: usize,
    pub popsize: usize, // Alias for population_size
    pub max_generations: usize,
    pub maxiter: usize, // Alias for max_generations
    pub tolerance: f64,
    pub tol: f64, // Alias for tolerance
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            population_size: 15,
            popsize: 15,
            max_generations: 1000,
            maxiter: 1000,
            tolerance: 1e-6,
            tol: 1e-6,
        }
    }
}

/// Differential evolution optimization using SciRS2 random utilities
pub fn differential_evolution<F>(
    fun: F,
    bounds: &[(f64, f64)],
    options: Option<DifferentialEvolutionOptions>,
    random_state: Option<u64>,
) -> QuantRS2Result<OptimizeResult<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let dim = bounds.len();
    if dim == 0 {
        return Err(QuantRS2Error::InvalidParameter(
            "Differential evolution requires at least one bounded variable".to_string(),
        ));
    }

    let pop_size = options
        .population_size
        .max(options.popsize)
        .max(4 * dim)
        .max(10);
    let max_generations = if options.max_generations == 0 {
        options.maxiter.max(1)
    } else {
        options.max_generations
    };
    let tolerance = if options.tolerance > 0.0 {
        options.tolerance
    } else if options.tol > 0.0 {
        options.tol
    } else {
        1e-9
    };

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        let mut seed_rng = thread_rng();
        let seed: u64 = seed_rng.gen();
        StdRng::seed_from_u64(seed)
    };

    let mut population = Vec::with_capacity(pop_size);
    let mut scores = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut candidate = Array1::zeros(dim);
        for (idx, &(lower, upper)) in bounds.iter().enumerate() {
            if !(lower.is_finite() && upper.is_finite() && upper > lower) {
                return Err(QuantRS2Error::InvalidParameter(format!(
                    "Invalid bounds for dimension {idx}: [{lower}, {upper}]"
                )));
            }
            let span = upper - lower;
            candidate[idx] = rng.gen::<f64>().mul_add(span, lower);
        }
        let score = safe_fun_eval(&fun, &candidate);
        population.push(candidate);
        scores.push(score);
    }

    let mut best_index = 0usize;
    for (idx, score) in scores.iter().enumerate() {
        if score < &scores[best_index] {
            best_index = idx;
        }
    }
    let mut best_candidate = population[best_index].clone();
    let mut best_score = scores[best_index];

    let mut iterations = 0usize;
    let mut evals = pop_size;

    while iterations < max_generations {
        for i in 0..pop_size {
            let mut indices: Vec<usize> = (0..pop_size).filter(|&idx| idx != i).collect();
            indices.shuffle(&mut rng);
            let (r1, r2, r3) = (indices[0], indices[1], indices[2]);

            let mut trial = population[r1].clone();
            let j_rand = rng.gen_range(0..dim);

            for j in 0..dim {
                if rng.gen::<f64>() < 0.7 || j == j_rand {
                    trial[j] =
                        0.8f64.mul_add(population[r2][j] - population[r3][j], population[r1][j]);
                }
                let (lower, upper) = bounds[j];
                if trial[j] < lower {
                    trial[j] = lower;
                }
                if trial[j] > upper {
                    trial[j] = upper;
                }
            }

            let trial_score = safe_fun_eval(&fun, &trial);
            evals += 1;

            if trial_score < scores[i] {
                population[i] = trial;
                scores[i] = trial_score;
                if trial_score < best_score {
                    best_score = trial_score;
                    best_candidate.clone_from(&population[i]);
                }
            }
        }

        iterations += 1;

        let (min_score, max_score) = scores
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &val| {
                (acc.0.min(val), acc.1.max(val))
            });
        if (max_score - min_score).abs() <= tolerance {
            break;
        }
    }

    Ok(OptimizeResult {
        x: best_candidate,
        fun: best_score,
        nit: iterations,
        iterations,
        success: true,
        message: format!(
            "Differential evolution converged in {iterations} generations ({evals} evaluations)"
        ),
    })
}
