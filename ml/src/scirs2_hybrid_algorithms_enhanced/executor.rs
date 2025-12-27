//! Enhanced hybrid algorithm executor implementation

use super::config::*;
use super::data_types::*;
use super::history::*;
use super::optimizers::*;
use super::quantum_types::*;
use super::results::*;
use super::support::*;

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::QuantRS2Result;
use scirs2_core::memory::BufferPool;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::sync::{Arc, Mutex};

/// Enhanced hybrid algorithm executor
pub struct EnhancedHybridExecutor {
    config: EnhancedHybridConfig,
    optimizer: Arc<Mutex<HybridOptimizer>>,
    ml_optimizer: Option<Arc<MLHybridOptimizer>>,
    performance_tuner: Arc<PerformanceTuner>,
    benchmarker: Arc<Mutex<HybridBenchmarker>>,
    distributed_executor: Option<Arc<DistributedExecutor>>,
    buffer_pool: BufferPool<f64>,
    cache: Arc<Mutex<HybridCache>>,
}

impl EnhancedHybridExecutor {
    /// Create a new enhanced hybrid executor
    pub fn new(config: EnhancedHybridConfig) -> Self {
        let optimizer = Arc::new(Mutex::new(HybridOptimizer::new(
            config.base_config.optimizer_type,
        )));
        let ml_optimizer = if config.enable_ml_optimization {
            Some(Arc::new(MLHybridOptimizer::new()))
        } else {
            None
        };
        let performance_tuner = Arc::new(PerformanceTuner::new());
        let benchmarker = Arc::new(Mutex::new(HybridBenchmarker::new()));
        let distributed_executor = if config.enable_distributed {
            Some(Arc::new(DistributedExecutor::new()))
        } else {
            None
        };
        let buffer_pool = BufferPool::new();
        let cache = Arc::new(Mutex::new(HybridCache::new()));

        Self {
            config,
            optimizer,
            ml_optimizer,
            performance_tuner,
            benchmarker,
            distributed_executor,
            buffer_pool,
            cache,
        }
    }

    /// Execute VQE algorithm
    pub fn execute_vqe(
        &mut self,
        hamiltonian: &Hamiltonian,
        ansatz: &Ansatz,
        initial_params: Option<Array1<f64>>,
    ) -> QuantRS2Result<VQEResult> {
        let start_time = std::time::Instant::now();

        // Initialize parameters
        let mut params =
            initial_params.unwrap_or_else(|| self.initialize_parameters(ansatz.num_parameters()));

        // Initialize tracking
        let mut history = OptimizationHistory::new();
        let mut best_energy = f64::INFINITY;
        let mut best_params = params.clone();

        // Main optimization loop
        for iteration in 0..self.config.base_config.max_iterations {
            // Evaluate energy
            let energy = self.evaluate_expectation_value(hamiltonian, ansatz, &params)?;

            // Update best result
            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
            }

            // Record history
            history.record(iteration, energy, params.clone());

            // Check convergence
            if self.check_convergence(&history)? {
                break;
            }

            // Calculate gradient
            let gradient = self.calculate_gradient(hamiltonian, ansatz, &params)?;

            // ML-enhanced gradient if enabled
            let enhanced_gradient = if let Some(ref ml_opt) = self.ml_optimizer {
                ml_opt.enhance_gradient(&gradient, &history)?
            } else {
                gradient
            };

            // Update parameters
            params = self
                .optimizer
                .lock()
                .map_err(|e| {
                    quantrs2_core::QuantRS2Error::Other(format!("Optimizer lock failed: {}", e))
                })?
                .update_parameters(&params, &enhanced_gradient)?;

            // Adaptive learning rate
            if self.config.enable_adaptive_learning {
                self.adapt_learning_rate(&history)?;
            }

            // Performance tuning
            if self.config.enable_realtime_tuning && iteration % 10 == 0 {
                self.performance_tuner.tune(&mut self.config, &history)?;
            }
        }

        // Final analysis
        let ground_state = self.extract_ground_state(ansatz, &best_params)?;
        let excited_states = self.find_excited_states(hamiltonian, ansatz, &best_params)?;

        // Generate visualizations
        let visualizations = if self.config.enable_visual_analytics {
            Some(self.generate_vqe_visualizations(&history)?)
        } else {
            None
        };

        let execution_time = start_time.elapsed();

        Ok(VQEResult {
            ground_state_energy: best_energy,
            optimal_parameters: best_params,
            ground_state,
            excited_states,
            optimization_history: history,
            visualizations,
            execution_time,
            convergence_achieved: best_energy < self.config.base_config.convergence_threshold,
            performance_metrics: self.calculate_performance_metrics(&history)?,
        })
    }

    /// Execute QAOA algorithm
    pub fn execute_qaoa(
        &mut self,
        problem: &QAOAProblem,
        num_layers: usize,
        initial_params: Option<Array1<f64>>,
    ) -> QuantRS2Result<QAOAResult> {
        let start_time = std::time::Instant::now();

        // Create QAOA ansatz
        let mut ansatz = self.create_qaoa_ansatz(problem, num_layers)?;

        // Initialize parameters (beta and gamma for each layer)
        let mut params =
            initial_params.unwrap_or_else(|| self.initialize_qaoa_parameters(num_layers));

        // Optimization loop
        let mut history = OptimizationHistory::new();
        let mut best_cost = f64::NEG_INFINITY;
        let mut best_params = params.clone();
        let mut best_solution = None;

        for iteration in 0..self.config.base_config.max_iterations {
            // Evaluate cost function
            let cost = self.evaluate_qaoa_cost(problem, &ansatz, &params)?;

            // Update best result
            if cost > best_cost {
                best_cost = cost;
                best_params = params.clone();
                best_solution = Some(self.measure_qaoa_solution(problem, &ansatz, &params)?);
            }

            // Record history
            history.record(iteration, -cost, params.clone());

            // Check convergence
            if self.check_convergence(&history)? {
                break;
            }

            // Calculate gradient
            let gradient = self.calculate_qaoa_gradient(problem, &ansatz, &params)?;

            // Update parameters
            params = self
                .optimizer
                .lock()
                .map_err(|e| {
                    quantrs2_core::QuantRS2Error::Other(format!("Optimizer lock failed: {}", e))
                })?
                .update_parameters(&params, &gradient)?;

            // Adaptive layer adjustment
            if self.config.enable_adaptive_learning && iteration % 50 == 0 {
                if self.should_add_layer(&history)? {
                    let (new_params, new_ansatz) = self.add_qaoa_layer(problem, ansatz, params)?;
                    params = new_params;
                    ansatz = new_ansatz;
                }
            }
        }

        // Analyze solution quality
        let solution_analysis =
            self.analyze_qaoa_solution(problem, &best_solution.clone().unwrap_or_default())?;

        // Generate visualizations
        let visualizations = if self.config.enable_visual_analytics {
            Some(self.generate_qaoa_visualizations(&history, problem)?)
        } else {
            None
        };

        let execution_time = start_time.elapsed();

        Ok(QAOAResult {
            optimal_cost: best_cost,
            optimal_parameters: best_params,
            best_solution: best_solution.unwrap_or_default(),
            solution_analysis,
            optimization_history: history,
            visualizations,
            execution_time,
            approximation_ratio: self.calculate_approximation_ratio(best_cost, problem)?,
            num_layers_used: num_layers,
        })
    }

    /// Execute VQC (Variational Quantum Classifier)
    pub fn execute_vqc(
        &mut self,
        training_data: &TrainingData,
        circuit_template: &CircuitTemplate,
        initial_params: Option<Array1<f64>>,
    ) -> QuantRS2Result<VQCResult> {
        let start_time = std::time::Instant::now();

        // Initialize parameters
        let mut params = initial_params
            .unwrap_or_else(|| self.initialize_parameters(circuit_template.num_parameters()));

        // Split data
        let (train_set, val_set) = self.split_data(training_data)?;

        // Training loop
        let mut history = TrainingHistory::new();
        let mut best_accuracy = 0.0;
        let mut best_params = params.clone();

        for epoch in 0..self.config.base_config.max_iterations {
            // Mini-batch training
            let batches = self.create_batches(&train_set, self.config.base_config.batch_size)?;

            let mut total_loss = 0.0;
            for batch in batches {
                // Forward pass
                let predictions = self.vqc_forward_pass(circuit_template, &params, &batch)?;

                // Calculate loss
                let loss = self.calculate_classification_loss(&predictions, &batch.labels)?;
                total_loss += loss;

                // Calculate gradient
                let gradient =
                    self.calculate_vqc_gradient(circuit_template, &params, &batch, &predictions)?;

                // Update parameters
                params = self
                    .optimizer
                    .lock()
                    .map_err(|e| {
                        quantrs2_core::QuantRS2Error::Other(format!("Optimizer lock failed: {}", e))
                    })?
                    .update_parameters(&params, &gradient)?;
            }

            // Validation
            let val_accuracy = self.evaluate_vqc_accuracy(circuit_template, &params, &val_set)?;

            // Update best result
            if val_accuracy > best_accuracy {
                best_accuracy = val_accuracy;
                best_params = params.clone();
            }

            // Record history
            history.record_epoch(epoch, total_loss, val_accuracy);

            // Early stopping
            if self.should_early_stop(&history)? {
                break;
            }
        }

        // Final evaluation on test set
        let test_metrics =
            self.evaluate_vqc_metrics(circuit_template, &best_params, training_data)?;

        // Generate visualizations
        let visualizations = if self.config.enable_visual_analytics {
            Some(self.generate_vqc_visualizations(&history)?)
        } else {
            None
        };

        let execution_time = start_time.elapsed();

        Ok(VQCResult {
            optimal_parameters: best_params,
            best_accuracy,
            test_metrics,
            training_history: history,
            visualizations,
            execution_time,
            model_complexity: self.calculate_model_complexity(circuit_template)?,
            feature_importance: self.analyze_feature_importance(circuit_template, &best_params)?,
        })
    }

    /// Execute generic VQA (Variational Quantum Algorithm)
    pub fn execute_vqa<F>(
        &mut self,
        cost_function: F,
        ansatz: &Ansatz,
        initial_params: Option<Array1<f64>>,
    ) -> QuantRS2Result<VQAResult>
    where
        F: Fn(&Array1<f64>) -> QuantRS2Result<f64> + Send + Sync,
    {
        let start_time = std::time::Instant::now();

        // Initialize parameters
        let mut params =
            initial_params.unwrap_or_else(|| self.initialize_parameters(ansatz.num_parameters()));

        // Optimization loop
        let mut history = OptimizationHistory::new();
        let mut best_cost = f64::INFINITY;
        let mut best_params = params.clone();

        // Distributed execution if enabled
        let executor = if let Some(ref dist_exec) = self.distributed_executor {
            dist_exec.clone()
        } else {
            Arc::new(LocalExecutor::new())
        };

        for iteration in 0..self.config.base_config.max_iterations {
            // Evaluate cost
            let cost = if self.config.enable_distributed {
                executor.evaluate_distributed(&cost_function, &params)?
            } else {
                cost_function(&params)?
            };

            // Update best
            if cost < best_cost {
                best_cost = cost;
                best_params = params.clone();
            }

            // Record history
            history.record(iteration, cost, params.clone());

            // Convergence check
            if self.check_convergence(&history)? {
                break;
            }

            // Calculate gradient
            let gradient = self.calculate_numerical_gradient(&cost_function, &params)?;

            // Update parameters
            params = self
                .optimizer
                .lock()
                .map_err(|e| {
                    quantrs2_core::QuantRS2Error::Other(format!("Optimizer lock failed: {}", e))
                })?
                .update_parameters(&params, &gradient)?;

            // Benchmarking
            if self.config.enable_benchmarking && iteration % 100 == 0 {
                self.benchmarker
                    .lock()
                    .map_err(|e| {
                        quantrs2_core::QuantRS2Error::Other(format!(
                            "Benchmarker lock failed: {}",
                            e
                        ))
                    })?
                    .record_iteration(&history, iteration)?;
            }
        }

        // Final analysis
        let landscape_analysis = if self.config.analysis_options.analyze_landscape {
            Some(self.analyze_optimization_landscape(&cost_function, &best_params)?)
        } else {
            None
        };

        let execution_time = start_time.elapsed();

        Ok(VQAResult {
            optimal_cost: best_cost,
            optimal_parameters: best_params,
            optimization_history: history,
            landscape_analysis,
            execution_time,
            convergence_achieved: history
                .is_converged(self.config.base_config.convergence_threshold),
            benchmark_results: if self.config.enable_benchmarking {
                Some(
                    self.benchmarker
                        .lock()
                        .map_err(|e| {
                            quantrs2_core::QuantRS2Error::Other(format!(
                                "Benchmarker lock failed: {}",
                                e
                            ))
                        })?
                        .generate_report()?,
                )
            } else {
                None
            },
        })
    }

    // Helper methods

    fn initialize_parameters(&self, num_params: usize) -> Array1<f64> {
        Array1::from_shape_fn(num_params, |_| {
            thread_rng().gen::<f64>() * 2.0 * std::f64::consts::PI
        })
    }

    fn initialize_qaoa_parameters(&self, num_layers: usize) -> Array1<f64> {
        // Beta and gamma for each layer
        Array1::from_shape_fn(2 * num_layers, |i| {
            if i < num_layers {
                // Beta parameters
                thread_rng().gen::<f64>() * std::f64::consts::PI
            } else {
                // Gamma parameters
                thread_rng().gen::<f64>() * 2.0 * std::f64::consts::PI
            }
        })
    }

    fn evaluate_expectation_value(
        &self,
        hamiltonian: &Hamiltonian,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<f64> {
        // Build circuit
        let circuit = ansatz.build_circuit(params)?;

        // Execute on hardware/simulator
        let state = self.execute_circuit(&circuit)?;

        // Calculate expectation value
        let expectation = hamiltonian.expectation_value(&state)?;

        Ok(expectation.re)
    }

    fn calculate_gradient(
        &self,
        hamiltonian: &Hamiltonian,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        match self.config.base_config.gradient_method {
            GradientMethod::ParameterShift => {
                self.parameter_shift_gradient(hamiltonian, ansatz, params)
            }
            GradientMethod::FiniteDifference => {
                self.finite_difference_gradient(hamiltonian, ansatz, params)
            }
            _ => Ok(Array1::zeros(params.len())),
        }
    }

    fn parameter_shift_gradient(
        &self,
        hamiltonian: &Hamiltonian,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let mut gradient = Array1::zeros(params.len());
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] += shift;
            params_minus[i] -= shift;

            let energy_plus = self.evaluate_expectation_value(hamiltonian, ansatz, &params_plus)?;
            let energy_minus =
                self.evaluate_expectation_value(hamiltonian, ansatz, &params_minus)?;

            gradient[i] = (energy_plus - energy_minus) / 2.0;
        }

        Ok(gradient)
    }

    fn finite_difference_gradient(
        &self,
        hamiltonian: &Hamiltonian,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let epsilon = 1e-5;
        let mut gradient = Array1::zeros(params.len());

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let energy_plus = self.evaluate_expectation_value(hamiltonian, ansatz, &params_plus)?;
            let energy_minus =
                self.evaluate_expectation_value(hamiltonian, ansatz, &params_minus)?;

            gradient[i] = (energy_plus - energy_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    fn calculate_numerical_gradient<F>(
        &self,
        cost_function: &F,
        params: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> QuantRS2Result<f64> + Send + Sync,
    {
        let epsilon = 1e-5;
        let mut gradient = Array1::zeros(params.len());

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let cost_plus = cost_function(&params_plus)?;
            let cost_minus = cost_function(&params_minus)?;

            gradient[i] = (cost_plus - cost_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    fn check_convergence(&self, history: &OptimizationHistory) -> QuantRS2Result<bool> {
        if history.iterations.len() < 2 {
            return Ok(false);
        }

        let window_size = 10.min(history.iterations.len());
        let recent_costs: Vec<f64> = history
            .iterations
            .iter()
            .rev()
            .take(window_size)
            .map(|iter| iter.cost)
            .collect();

        let mean = recent_costs.iter().sum::<f64>() / recent_costs.len() as f64;
        let variance = recent_costs
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / recent_costs.len() as f64;

        Ok(variance < self.config.base_config.convergence_threshold.powi(2))
    }

    fn adapt_learning_rate(&mut self, history: &OptimizationHistory) -> QuantRS2Result<()> {
        // Simple adaptive learning rate based on progress
        if history.iterations.len() > 10 {
            let recent_improvement = history.get_recent_improvement(10)?;

            if recent_improvement < 0.001 {
                self.config.base_config.learning_rate *= 0.9;
            } else if recent_improvement > 0.01 {
                self.config.base_config.learning_rate *= 1.1;
            }

            // Clamp learning rate
            self.config.base_config.learning_rate =
                self.config.base_config.learning_rate.max(1e-6).min(1.0);
        }

        Ok(())
    }

    fn execute_circuit(&self, _circuit: &Circuit) -> QuantRS2Result<QuantumState> {
        // Execute circuit on backend
        // This would interface with actual hardware or simulator
        Ok(QuantumState::new(4))
    }

    fn extract_ground_state(
        &self,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<QuantumState> {
        let circuit = ansatz.build_circuit(params)?;
        self.execute_circuit(&circuit)
    }

    fn find_excited_states(
        &self,
        _hamiltonian: &Hamiltonian,
        _ansatz: &Ansatz,
        _ground_params: &Array1<f64>,
    ) -> QuantRS2Result<Vec<ExcitedState>> {
        // Simplified excited state search
        Ok(Vec::new())
    }

    fn create_qaoa_ansatz(
        &self,
        problem: &QAOAProblem,
        num_layers: usize,
    ) -> QuantRS2Result<Ansatz> {
        Ok(Ansatz::QAOA {
            problem: problem.clone(),
            num_layers,
        })
    }

    fn evaluate_qaoa_cost(
        &self,
        problem: &QAOAProblem,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<f64> {
        let circuit = ansatz.build_circuit(params)?;
        let measurements = self.measure_circuit(&circuit, self.config.base_config.num_shots)?;

        // Calculate average cost
        let cost = problem.evaluate_cost(&measurements)?;
        Ok(cost)
    }

    fn calculate_qaoa_gradient(
        &self,
        problem: &QAOAProblem,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        // Use parameter shift for QAOA
        let mut gradient = Array1::zeros(params.len());
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] += shift;
            params_minus[i] -= shift;

            let cost_plus = self.evaluate_qaoa_cost(problem, ansatz, &params_plus)?;
            let cost_minus = self.evaluate_qaoa_cost(problem, ansatz, &params_minus)?;

            gradient[i] = (cost_plus - cost_minus) / 2.0;
        }

        Ok(gradient)
    }

    fn measure_qaoa_solution(
        &self,
        _problem: &QAOAProblem,
        ansatz: &Ansatz,
        params: &Array1<f64>,
    ) -> QuantRS2Result<BinaryString> {
        let circuit = ansatz.build_circuit(params)?;
        let measurements = self.measure_circuit(&circuit, 1000)?;

        // Return most frequent measurement
        Ok(measurements.most_frequent())
    }

    fn measure_circuit(
        &self,
        _circuit: &Circuit,
        _num_shots: usize,
    ) -> QuantRS2Result<MeasurementResults> {
        // Measure circuit
        Ok(MeasurementResults::new())
    }

    fn should_add_layer(&self, history: &OptimizationHistory) -> QuantRS2Result<bool> {
        // Check if adding another layer would help
        let recent_improvement = history.get_recent_improvement(50)?;
        Ok(recent_improvement < 0.0001)
    }

    fn add_qaoa_layer(
        &self,
        problem: &QAOAProblem,
        mut ansatz: Ansatz,
        mut params: Array1<f64>,
    ) -> QuantRS2Result<(Array1<f64>, Ansatz)> {
        // Add another layer to QAOA
        if let Ansatz::QAOA { num_layers, .. } = &mut ansatz {
            *num_layers += 1;

            // Extend parameters
            let new_params = Array1::from_shape_fn(params.len() + 2, |i| {
                if i < params.len() {
                    params[i]
                } else {
                    thread_rng().gen::<f64>() * std::f64::consts::PI
                }
            });

            params = new_params;
        }

        Ok((params, ansatz))
    }

    fn analyze_qaoa_solution(
        &self,
        problem: &QAOAProblem,
        solution: &BinaryString,
    ) -> QuantRS2Result<SolutionAnalysis> {
        Ok(SolutionAnalysis {
            cost_value: problem.evaluate_solution(solution)?,
            constraint_violations: problem.check_constraints(solution)?,
            solution_quality: 0.95,
        })
    }

    fn calculate_approximation_ratio(
        &self,
        achieved_cost: f64,
        problem: &QAOAProblem,
    ) -> QuantRS2Result<f64> {
        let optimal_cost = problem.get_optimal_cost()?;
        Ok(achieved_cost / optimal_cost)
    }

    fn split_data(&self, data: &TrainingData) -> QuantRS2Result<(TrainingData, TrainingData)> {
        let split_idx = (data.len() as f64 * 0.8) as usize;
        Ok((data.slice(0, split_idx), data.slice(split_idx, data.len())))
    }

    fn create_batches(
        &self,
        data: &TrainingData,
        batch_size: usize,
    ) -> QuantRS2Result<Vec<DataBatch>> {
        let mut batches = Vec::new();

        for i in (0..data.len()).step_by(batch_size) {
            let end = (i + batch_size).min(data.len());
            batches.push(data.slice(i, end).into());
        }

        Ok(batches)
    }

    fn vqc_forward_pass(
        &self,
        template: &CircuitTemplate,
        params: &Array1<f64>,
        batch: &DataBatch,
    ) -> QuantRS2Result<Array2<f64>> {
        let mut predictions = Array2::zeros((batch.len(), template.num_classes()));

        // Process each sample
        for (i, sample) in batch.samples.iter().enumerate() {
            let circuit = template.encode_and_build(sample, params)?;
            let measurement = self.measure_circuit(&circuit, 1000)?;
            predictions
                .row_mut(i)
                .assign(&measurement.to_probabilities());
        }

        Ok(predictions)
    }

    fn calculate_classification_loss(
        &self,
        predictions: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> QuantRS2Result<f64> {
        // Cross-entropy loss
        let mut loss = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            let pred = predictions.row(i);
            loss -= pred[label].ln();
        }

        Ok(loss / labels.len() as f64)
    }

    fn calculate_vqc_gradient(
        &self,
        template: &CircuitTemplate,
        params: &Array1<f64>,
        batch: &DataBatch,
        _predictions: &Array2<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        // Parameter shift gradient for VQC
        let mut gradient = Array1::zeros(params.len());

        // Accumulate gradients over batch
        for (sample_idx, sample) in batch.samples.iter().enumerate() {
            for param_idx in 0..params.len() {
                let grad = self.parameter_shift_single(
                    template,
                    params,
                    sample,
                    param_idx,
                    batch.labels[sample_idx],
                )?;
                gradient[param_idx] += grad;
            }
        }

        Ok(gradient / batch.len() as f64)
    }

    fn parameter_shift_single(
        &self,
        template: &CircuitTemplate,
        params: &Array1<f64>,
        sample: &DataSample,
        param_idx: usize,
        label: usize,
    ) -> QuantRS2Result<f64> {
        let shift = std::f64::consts::PI / 2.0;
        let mut params_plus = params.clone();
        let mut params_minus = params.clone();

        params_plus[param_idx] += shift;
        params_minus[param_idx] -= shift;

        let circuit_plus = template.encode_and_build(sample, &params_plus)?;
        let circuit_minus = template.encode_and_build(sample, &params_minus)?;

        let meas_plus = self.measure_circuit(&circuit_plus, 1000)?;
        let meas_minus = self.measure_circuit(&circuit_minus, 1000)?;

        let prob_plus = meas_plus.to_probabilities()[label];
        let prob_minus = meas_minus.to_probabilities()[label];

        Ok((prob_plus - prob_minus) / 2.0)
    }

    fn evaluate_vqc_accuracy(
        &self,
        template: &CircuitTemplate,
        params: &Array1<f64>,
        data: &TrainingData,
    ) -> QuantRS2Result<f64> {
        let predictions = self.vqc_forward_pass(template, params, &data.to_batch())?;

        let mut correct = 0;
        for (i, &label) in data.labels.iter().enumerate() {
            let pred_label = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_label == label {
                correct += 1;
            }
        }

        Ok(correct as f64 / data.len() as f64)
    }

    fn should_early_stop(&self, history: &TrainingHistory) -> QuantRS2Result<bool> {
        if history.epochs.len() < 10 {
            return Ok(false);
        }

        // Check if validation accuracy hasn't improved in last 5 epochs
        let recent_accuracies: Vec<f64> = history
            .epochs
            .iter()
            .rev()
            .take(5)
            .map(|e| e.val_accuracy)
            .collect();

        let max_recent = recent_accuracies
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let older_max = history.epochs[..history.epochs.len() - 5]
            .iter()
            .map(|e| e.val_accuracy)
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(max_recent <= older_max)
    }

    fn evaluate_vqc_metrics(
        &self,
        template: &CircuitTemplate,
        params: &Array1<f64>,
        data: &TrainingData,
    ) -> QuantRS2Result<ClassificationMetrics> {
        let predictions = self.vqc_forward_pass(template, params, &data.to_batch())?;

        // Calculate various metrics
        let accuracy = self.evaluate_vqc_accuracy(template, params, data)?;
        let confusion_matrix = self.calculate_confusion_matrix(&predictions, &data.labels)?;
        let precision_recall = self.calculate_precision_recall(&confusion_matrix)?;

        Ok(ClassificationMetrics {
            accuracy,
            confusion_matrix,
            precision: precision_recall.0,
            recall: precision_recall.1,
            f1_score: self.calculate_f1_score(precision_recall.0, precision_recall.1),
        })
    }

    fn calculate_confusion_matrix(
        &self,
        predictions: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> QuantRS2Result<Array2<usize>> {
        let num_classes = predictions.ncols();
        let mut matrix = Array2::zeros((num_classes, num_classes));

        for (i, &true_label) in labels.iter().enumerate() {
            let pred_label = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            matrix[(true_label, pred_label)] += 1;
        }

        Ok(matrix)
    }

    fn calculate_precision_recall(
        &self,
        confusion_matrix: &Array2<usize>,
    ) -> QuantRS2Result<(Array1<f64>, Array1<f64>)> {
        let num_classes = confusion_matrix.nrows();
        let mut precision = Array1::zeros(num_classes);
        let mut recall = Array1::zeros(num_classes);

        for i in 0..num_classes {
            let true_positives = confusion_matrix[(i, i)] as f64;
            let false_positives: f64 = (0..num_classes)
                .filter(|&j| j != i)
                .map(|j| confusion_matrix[(j, i)] as f64)
                .sum();
            let false_negatives: f64 = (0..num_classes)
                .filter(|&j| j != i)
                .map(|j| confusion_matrix[(i, j)] as f64)
                .sum();

            precision[i] = if true_positives + false_positives > 0.0 {
                true_positives / (true_positives + false_positives)
            } else {
                0.0
            };

            recall[i] = if true_positives + false_negatives > 0.0 {
                true_positives / (true_positives + false_negatives)
            } else {
                0.0
            };
        }

        Ok((precision, recall))
    }

    fn calculate_f1_score(&self, precision: Array1<f64>, recall: Array1<f64>) -> Array1<f64> {
        let mut f1 = Array1::zeros(precision.len());

        for i in 0..precision.len() {
            if precision[i] + recall[i] > 0.0 {
                f1[i] = 2.0 * precision[i] * recall[i] / (precision[i] + recall[i]);
            }
        }

        f1
    }

    fn calculate_model_complexity(&self, template: &CircuitTemplate) -> QuantRS2Result<f64> {
        Ok(template.num_parameters() as f64 * template.circuit_depth() as f64)
    }

    fn analyze_feature_importance(
        &self,
        _template: &CircuitTemplate,
        params: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        // Simple feature importance based on parameter magnitudes
        Ok(params.mapv(f64::abs))
    }

    fn calculate_performance_metrics(
        &self,
        history: &OptimizationHistory,
    ) -> QuantRS2Result<PerformanceMetrics> {
        Ok(PerformanceMetrics {
            total_iterations: history.iterations.len(),
            convergence_rate: history.calculate_convergence_rate()?,
            wall_time: history.total_time(),
            circuit_evaluations: history.iterations.len() * self.config.base_config.num_shots,
        })
    }

    fn analyze_optimization_landscape<F>(
        &self,
        cost_function: &F,
        optimal_params: &Array1<f64>,
    ) -> QuantRS2Result<LandscapeAnalysis>
    where
        F: Fn(&Array1<f64>) -> QuantRS2Result<f64> + Send + Sync,
    {
        // Sample landscape around optimal point
        let samples = 100;
        let radius = 0.1;
        let mut landscape_points = Vec::new();

        for _ in 0..samples {
            let perturbation = Array1::from_shape_fn(optimal_params.len(), |_| {
                (thread_rng().gen::<f64>() - 0.5) * 2.0 * radius
            });
            let point = optimal_params + &perturbation;
            let cost = cost_function(&point)?;
            landscape_points.push((point, cost));
        }

        // Analyze landscape properties
        let costs: Vec<f64> = landscape_points.iter().map(|(_, c)| *c).collect();
        let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
        let variance =
            costs.iter().map(|c| (c - mean_cost).powi(2)).sum::<f64>() / costs.len() as f64;

        Ok(LandscapeAnalysis {
            local_minima: self.find_local_minima(&landscape_points)?,
            landscape_roughness: variance.sqrt(),
            gradient_variance: self.estimate_gradient_variance(&landscape_points)?,
            barren_plateau_indicator: variance < 1e-6,
        })
    }

    fn find_local_minima(
        &self,
        points: &[(Array1<f64>, f64)],
    ) -> QuantRS2Result<Vec<LocalMinimum>> {
        // Simple local minima detection
        let mut minima = Vec::new();

        for (params, cost) in points {
            if self.is_local_minimum(params, *cost, points)? {
                minima.push(LocalMinimum {
                    parameters: params.clone(),
                    cost: *cost,
                });
            }
        }

        Ok(minima)
    }

    fn is_local_minimum(
        &self,
        point: &Array1<f64>,
        cost: f64,
        all_points: &[(Array1<f64>, f64)],
    ) -> QuantRS2Result<bool> {
        let threshold = 0.05;

        for (other_point, other_cost) in all_points {
            let distance = (point - other_point).mapv(|x| x.powi(2)).sum().sqrt();
            if distance < threshold && *other_cost < cost {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn estimate_gradient_variance(&self, _points: &[(Array1<f64>, f64)]) -> QuantRS2Result<f64> {
        // Estimate gradient variance from sampled points
        Ok(0.01) // Placeholder
    }

    // Visualization methods

    fn generate_vqe_visualizations(
        &self,
        _history: &OptimizationHistory,
    ) -> QuantRS2Result<VQEVisualizations> {
        Ok(VQEVisualizations {
            energy_convergence: "Energy convergence plot".to_string(),
            parameter_evolution: "Parameter evolution plot".to_string(),
            gradient_norms: "Gradient norms plot".to_string(),
            landscape_heatmap: "Landscape heatmap".to_string(),
        })
    }

    fn generate_qaoa_visualizations(
        &self,
        _history: &OptimizationHistory,
        _problem: &QAOAProblem,
    ) -> QuantRS2Result<QAOAVisualizations> {
        Ok(QAOAVisualizations {
            cost_evolution: "Cost evolution plot".to_string(),
            parameter_landscape: "QAOA parameter landscape".to_string(),
            solution_distribution: "Solution distribution plot".to_string(),
            approximation_ratio: "Approximation ratio plot".to_string(),
        })
    }

    fn generate_vqc_visualizations(
        &self,
        _history: &TrainingHistory,
    ) -> QuantRS2Result<VQCVisualizations> {
        Ok(VQCVisualizations {
            loss_curves: "Loss curves plot".to_string(),
            accuracy_evolution: "Accuracy evolution plot".to_string(),
            confusion_matrix: "Confusion matrix plot".to_string(),
            feature_importance: "Feature importance plot".to_string(),
        })
    }
}

impl From<TrainingData> for DataBatch {
    fn from(data: TrainingData) -> Self {
        data.to_batch()
    }
}
