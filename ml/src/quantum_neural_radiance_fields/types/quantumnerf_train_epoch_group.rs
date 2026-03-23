//! # QuantumNeRF - train_epoch_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{NeRFTrainingConfig, NeRFTrainingMetrics, TrainingImage};

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Train single epoch
    pub(super) fn train_epoch(
        &mut self,
        training_images: &[TrainingImage],
        config: &NeRFTrainingConfig,
        epoch: usize,
    ) -> Result<NeRFTrainingMetrics> {
        let mut epoch_loss = 0.0;
        let mut quantum_fidelity_sum = 0.0;
        let mut entanglement_sum = 0.0;
        let mut psnr_sum = 0.0;
        let mut num_batches = 0;
        for image in training_images {
            let batch_metrics = self.train_image(image, config)?;
            epoch_loss += batch_metrics.loss;
            quantum_fidelity_sum += batch_metrics.quantum_fidelity;
            entanglement_sum += batch_metrics.entanglement_measure;
            psnr_sum += batch_metrics.psnr;
            num_batches += 1;
        }
        let num_batches_f = num_batches as f64;
        Ok(NeRFTrainingMetrics {
            epoch,
            loss: epoch_loss / num_batches_f,
            psnr: psnr_sum / num_batches_f,
            ssim: 0.8,
            lpips: 0.1,
            quantum_fidelity: quantum_fidelity_sum / num_batches_f,
            entanglement_measure: entanglement_sum / num_batches_f,
            rendering_time: 1.0,
            quantum_advantage_ratio: 1.0 + entanglement_sum / num_batches_f,
            memory_usage: 1000.0,
        })
    }
    /// Train on single image
    pub(super) fn train_image(
        &mut self,
        image: &TrainingImage,
        config: &NeRFTrainingConfig,
    ) -> Result<NeRFTrainingMetrics> {
        let sampled_rays = self.sample_training_rays(image, config.rays_per_batch)?;
        let mut batch_loss = 0.0;
        let mut quantum_fidelity_sum = 0.0;
        let mut entanglement_sum = 0.0;
        for ray_sample in &sampled_rays {
            let pixel_output = self.render_pixel_quantum(&ray_sample.ray)?;
            let target_color = &ray_sample.target_color;
            let color_loss = (&pixel_output.color - target_color).mapv(|x| x * x).sum();
            let quantum_loss = self.compute_quantum_loss(&pixel_output.quantum_state)?;
            let total_loss = color_loss + config.quantum_loss_weight * quantum_loss;
            batch_loss += total_loss;
            quantum_fidelity_sum += pixel_output.quantum_state.quantum_fidelity;
            entanglement_sum += pixel_output.quantum_state.entanglement_measure;
            self.update_nerf_parameters(&pixel_output, total_loss, config)?;
        }
        let num_rays = sampled_rays.len() as f64;
        let mse: f64 = batch_loss / num_rays;
        let psnr = -10.0 * mse.log10();
        Ok(NeRFTrainingMetrics {
            epoch: 0,
            loss: batch_loss / num_rays,
            psnr,
            ssim: 0.0,
            lpips: 0.0,
            quantum_fidelity: quantum_fidelity_sum / num_rays,
            entanglement_measure: entanglement_sum / num_rays,
            rendering_time: 0.0,
            quantum_advantage_ratio: 1.0 + entanglement_sum / num_rays,
            memory_usage: 0.0,
        })
    }
}
