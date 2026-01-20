//! Learning rate schedules for Keras-like API

/// Learning rate schedule trait
pub trait LearningRateSchedule: Send + Sync {
    /// Get learning rate for given step
    fn get_lr(&self, step: usize) -> f64;
}

/// Exponential decay schedule
pub struct ExponentialDecay {
    /// Initial learning rate
    initial_lr: f64,
    /// Decay steps
    decay_steps: usize,
    /// Decay rate
    decay_rate: f64,
    /// Staircase (step-wise decay)
    staircase: bool,
}

impl ExponentialDecay {
    /// Create new exponential decay schedule
    pub fn new(initial_lr: f64, decay_steps: usize, decay_rate: f64) -> Self {
        Self {
            initial_lr,
            decay_steps,
            decay_rate,
            staircase: false,
        }
    }

    /// Set staircase mode
    pub fn staircase(mut self, staircase: bool) -> Self {
        self.staircase = staircase;
        self
    }
}

impl LearningRateSchedule for ExponentialDecay {
    fn get_lr(&self, step: usize) -> f64 {
        let progress = if self.staircase {
            (step / self.decay_steps) as f64
        } else {
            step as f64 / self.decay_steps as f64
        };
        self.initial_lr * self.decay_rate.powf(progress)
    }
}

/// Piecewise constant schedule
pub struct PiecewiseConstantDecay {
    /// Boundaries
    boundaries: Vec<usize>,
    /// Values
    values: Vec<f64>,
}

impl PiecewiseConstantDecay {
    /// Create new piecewise constant decay schedule
    pub fn new(boundaries: Vec<usize>, values: Vec<f64>) -> Self {
        Self { boundaries, values }
    }
}

impl LearningRateSchedule for PiecewiseConstantDecay {
    fn get_lr(&self, step: usize) -> f64 {
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if step < boundary {
                return self.values[i];
            }
        }
        *self.values.last().unwrap_or(&0.001)
    }
}

/// Polynomial decay schedule
pub struct PolynomialDecay {
    /// Initial learning rate
    initial_lr: f64,
    /// Decay steps
    decay_steps: usize,
    /// End learning rate
    end_lr: f64,
    /// Power
    power: f64,
}

impl PolynomialDecay {
    /// Create new polynomial decay schedule
    pub fn new(initial_lr: f64, decay_steps: usize, end_lr: f64, power: f64) -> Self {
        Self {
            initial_lr,
            decay_steps,
            end_lr,
            power,
        }
    }
}

impl LearningRateSchedule for PolynomialDecay {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.decay_steps);
        let decay = (1.0 - step as f64 / self.decay_steps as f64).powf(self.power);
        (self.initial_lr - self.end_lr) * decay + self.end_lr
    }
}

/// Cosine decay schedule
pub struct CosineDecay {
    /// Initial learning rate
    initial_lr: f64,
    /// Decay steps
    decay_steps: usize,
    /// Alpha (minimum LR factor)
    alpha: f64,
}

impl CosineDecay {
    /// Create new cosine decay schedule
    pub fn new(initial_lr: f64, decay_steps: usize) -> Self {
        Self {
            initial_lr,
            decay_steps,
            alpha: 0.0,
        }
    }

    /// Set alpha (minimum LR factor)
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl LearningRateSchedule for CosineDecay {
    fn get_lr(&self, step: usize) -> f64 {
        let step = step.min(self.decay_steps);
        let progress = step as f64 / self.decay_steps as f64;
        let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.initial_lr * (self.alpha + (1.0 - self.alpha) * cosine_decay)
    }
}
