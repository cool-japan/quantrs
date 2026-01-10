//! Learning rate schedulers for PyTorch-like API

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get current learning rate
    fn get_lr(&self) -> f64;
    /// Step the scheduler
    fn step(&mut self);
    /// Set the epoch
    fn set_epoch(&mut self, epoch: usize);
}

/// Step learning rate scheduler
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    current_epoch: usize,
}

impl StepLR {
    /// Create new step LR scheduler
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self) -> f64 {
        self.base_lr
            * self
                .gamma
                .powi((self.current_epoch / self.step_size) as i32)
    }

    fn step(&mut self) {
        self.current_epoch += 1;
    }

    fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
    current_epoch: usize,
}

impl ExponentialLR {
    /// Create new exponential LR scheduler
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f64 {
        self.base_lr * self.gamma.powi(self.current_epoch as i32)
    }

    fn step(&mut self) {
        self.current_epoch += 1;
    }

    fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    base_lr: f64,
    t_max: usize,
    eta_min: f64,
    current_epoch: usize,
}

impl CosineAnnealingLR {
    /// Create new cosine annealing LR scheduler
    pub fn new(base_lr: f64, t_max: usize) -> Self {
        Self {
            base_lr,
            t_max,
            eta_min: 0.0,
            current_epoch: 0,
        }
    }

    /// Set minimum learning rate
    pub fn eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f64 {
        self.eta_min
            + (self.base_lr - self.eta_min)
                * (1.0
                    + (std::f64::consts::PI * self.current_epoch as f64 / self.t_max as f64).cos())
                / 2.0
    }

    fn step(&mut self) {
        self.current_epoch += 1;
    }

    fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }
}

/// ReduceLROnPlateau scheduler
pub struct ReduceLROnPlateau {
    base_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    best_score: f64,
    num_bad_epochs: usize,
    current_lr: f64,
}

impl ReduceLROnPlateau {
    /// Create new ReduceLROnPlateau scheduler
    pub fn new(base_lr: f64) -> Self {
        Self {
            base_lr,
            factor: 0.1,
            patience: 10,
            min_lr: 1e-8,
            best_score: f64::INFINITY,
            num_bad_epochs: 0,
            current_lr: base_lr,
        }
    }

    /// Set reduction factor
    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set patience
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Step based on validation loss
    pub fn step_with_metric(&mut self, metric: f64) {
        if metric < self.best_score {
            self.best_score = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
            if self.num_bad_epochs >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.num_bad_epochs = 0;
            }
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn step(&mut self) {
        // No-op for ReduceLROnPlateau - use step_with_metric instead
    }

    fn set_epoch(&mut self, _epoch: usize) {
        // No-op for ReduceLROnPlateau
    }
}
