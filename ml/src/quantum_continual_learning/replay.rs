//! Experience replay for quantum continual learning

use crate::error::{MLError, Result};
use ndarray::{Array1, Array2};

/// Experience replay buffer
#[derive(Debug)]
pub struct ExperienceReplayBuffer {
    capacity: usize,
    data: Vec<Array1<f64>>,
    labels: Vec<i32>,
    task_ids: Vec<usize>,
}

impl ExperienceReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: Vec::new(),
            labels: Vec::new(),
            task_ids: Vec::new(),
        }
    }

    pub fn add_experience(&mut self, sample: Array1<f64>, label: i32, task_id: usize) {
        if self.data.len() >= self.capacity {
            // Remove oldest experience
            self.data.remove(0);
            self.labels.remove(0);
            self.task_ids.remove(0);
        }

        self.data.push(sample);
        self.labels.push(label);
        self.task_ids.push(task_id);
    }

    pub fn sample_batch(&self, batch_size: usize) -> Result<(Array2<f64>, Array1<i32>, Array1<usize>)> {
        let n = batch_size.min(self.data.len());
        if n == 0 {
            return Ok((Array2::zeros((0, 0)), Array1::zeros(0), Array1::zeros(0)));
        }

        // Placeholder sampling
        let data = Array2::zeros((n, self.data[0].len()));
        let labels = Array1::zeros(n);
        let task_ids = Array1::zeros(n);

        Ok((data, labels, task_ids))
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}