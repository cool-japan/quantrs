//! History tracking types for hybrid algorithm optimization

use quantrs2_core::QuantRS2Result;
use scirs2_core::ndarray::Array1;
use std::time::Instant;

/// Optimization history
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    pub iterations: Vec<OptimizationIteration>,
    start_time: Instant,
}

impl OptimizationHistory {
    pub fn new() -> Self {
        Self {
            iterations: Vec::new(),
            start_time: Instant::now(),
        }
    }

    pub fn record(&mut self, iteration: usize, cost: f64, params: Array1<f64>) {
        self.iterations.push(OptimizationIteration {
            iteration,
            cost,
            params,
            timestamp: Instant::now(),
        });
    }

    pub fn get_recent_improvement(&self, window: usize) -> QuantRS2Result<f64> {
        if self.iterations.len() < window + 1 {
            return Ok(0.0);
        }

        let recent_idx = self.iterations.len() - 1;
        let old_idx = recent_idx - window;

        let improvement = (self.iterations[old_idx].cost - self.iterations[recent_idx].cost).abs();
        Ok(improvement)
    }

    pub fn is_converged(&self, threshold: f64) -> bool {
        self.get_recent_improvement(10).unwrap_or(1.0) < threshold
    }

    pub fn calculate_convergence_rate(&self) -> QuantRS2Result<f64> {
        if self.iterations.len() < 2 {
            return Ok(0.0);
        }

        // Safe: we checked iterations.len() >= 2 above
        let first_cost = self.iterations.first().expect("guaranteed by len check").cost;
        let last_cost = self.iterations.last().expect("guaranteed by len check").cost;
        let iterations = self.iterations.len() as f64;

        Ok((first_cost - last_cost).abs() / iterations)
    }

    pub fn total_time(&self) -> std::time::Duration {
        Instant::now() - self.start_time
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationIteration {
    pub iteration: usize,
    pub cost: f64,
    pub params: Array1<f64>,
    pub timestamp: Instant,
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub epochs: Vec<TrainingEpoch>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self { epochs: Vec::new() }
    }

    pub fn record_epoch(&mut self, epoch: usize, loss: f64, val_accuracy: f64) {
        self.epochs.push(TrainingEpoch {
            epoch,
            loss,
            val_accuracy,
        });
    }
}

#[derive(Debug, Clone)]
pub struct TrainingEpoch {
    pub epoch: usize,
    pub loss: f64,
    pub val_accuracy: f64,
}
