//! Enhanced Pulse Control with SciRS2 Signal Processing
//!
//! This module extends the basic pulse control with advanced signal processing
//! capabilities from SciRS2, including:
//! - Frequency domain analysis via FFT
//! - Pulse shape optimization
//! - Spectral filtering
//! - Noise characterization
//! - Signal quality metrics

use crate::{
    pulse::{ChannelType, PulseCalibration, PulseInstruction, PulseSchedule, PulseShape},
    DeviceError, DeviceResult,
};
use scirs2_core::{
    ndarray::{Array1, Array2},
    Complex64,
};
use scirs2_fft::{fft, ifft};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Signal processing configuration for pulse control
#[derive(Debug, Clone)]
pub struct SignalProcessingConfig {
    /// Enable FFT-based pulse optimization
    pub enable_fft_optimization: bool,
    /// Enable spectral filtering
    pub enable_filtering: bool,
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Filter cutoff frequency (Hz)
    pub filter_cutoff: f64,
    /// Filter order
    pub filter_order: usize,
    /// Window function for spectral analysis
    pub window_function: WindowType,
    /// FFT size (power of 2)
    pub fft_size: usize,
}

impl Default for SignalProcessingConfig {
    fn default() -> Self {
        Self {
            enable_fft_optimization: true,
            enable_filtering: true,
            sample_rate: 1e9,     // 1 GHz default
            filter_cutoff: 100e6, // 100 MHz
            filter_order: 4,
            window_function: WindowType::Hamming,
            fft_size: 1024,
        }
    }
}

/// Window function types for signal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser,
}

impl WindowType {
    /// Apply window function to signal
    pub fn apply(&self, signal: &mut Array1<Complex64>) {
        let n = signal.len();
        match self {
            WindowType::Rectangular => {
                // No windowing
            }
            WindowType::Hamming => {
                for i in 0..n {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
                    signal[i] *= w;
                }
            }
            WindowType::Hanning => {
                for i in 0..n {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
                    signal[i] *= w;
                }
            }
            WindowType::Blackman => {
                for i in 0..n {
                    let w = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
                        + 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos();
                    signal[i] *= w;
                }
            }
            WindowType::Kaiser => {
                // Simplified Kaiser window (beta=5)
                for i in 0..n {
                    let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                    let w = if x.abs() < 1.0 {
                        (1.0 - x * x).sqrt()
                    } else {
                        0.0
                    };
                    signal[i] *= w;
                }
            }
        }
    }
}

/// Pulse signal quality metrics
#[derive(Debug, Clone)]
pub struct PulseQualityMetrics {
    /// Signal-to-noise ratio (dB)
    pub snr: f64,
    /// Peak signal power
    pub peak_power: f64,
    /// Average signal power
    pub average_power: f64,
    /// Bandwidth (Hz)
    pub bandwidth: f64,
    /// Center frequency (Hz)
    pub center_frequency: f64,
    /// Spectral purity (fraction of power in main lobe)
    pub spectral_purity: f64,
    /// Total harmonic distortion (%)
    pub thd: f64,
}

/// Spectral analysis result
#[derive(Debug, Clone)]
pub struct SpectralAnalysisResult {
    /// Frequency bins (Hz)
    pub frequencies: Vec<f64>,
    /// Power spectral density
    pub psd: Vec<f64>,
    /// Dominant frequency components
    pub peaks: Vec<(f64, f64)>, // (frequency, power)
    /// Total signal power
    pub total_power: f64,
}

/// Enhanced pulse controller with SciRS2 signal processing
pub struct SciRS2PulseController {
    config: SignalProcessingConfig,
    calibration: Option<PulseCalibration>,
}

impl SciRS2PulseController {
    /// Create a new SciRS2 pulse controller
    pub fn new(config: SignalProcessingConfig) -> Self {
        Self {
            config,
            calibration: None,
        }
    }

    /// Set calibration data
    pub fn set_calibration(&mut self, calibration: PulseCalibration) {
        self.calibration = Some(calibration);
    }

    /// Optimize pulse shape using FFT analysis
    pub fn optimize_pulse_shape(
        &self,
        pulse: &PulseShape,
        sample_rate: f64,
    ) -> DeviceResult<PulseShape> {
        // Convert pulse to time-domain samples
        let samples = self.pulse_to_samples(pulse, sample_rate)?;

        // Apply FFT for frequency domain analysis
        let fft_result = self.compute_fft(&samples)?;

        // Analyze spectrum and filter unwanted components
        let filtered_fft = self.apply_spectral_filter(&fft_result)?;

        // Convert back to time domain
        let optimized_samples = self.compute_ifft(&filtered_fft)?;

        // Create optimized pulse
        Ok(PulseShape::Arbitrary {
            samples: optimized_samples,
            sample_rate,
        })
    }

    /// Convert pulse shape to time-domain samples
    fn pulse_to_samples(
        &self,
        pulse: &PulseShape,
        sample_rate: f64,
    ) -> DeviceResult<Vec<Complex64>> {
        match pulse {
            PulseShape::Gaussian {
                duration,
                sigma,
                amplitude,
            } => {
                let n_samples = (duration * sample_rate) as usize;
                let mut samples = Vec::with_capacity(n_samples);
                let t_center = duration / 2.0;

                for i in 0..n_samples {
                    let t = i as f64 / sample_rate;
                    let gaussian = (-(t - t_center).powi(2) / (2.0 * sigma.powi(2))).exp();
                    samples.push(*amplitude * gaussian);
                }

                Ok(samples)
            }
            PulseShape::GaussianDrag {
                duration,
                sigma,
                amplitude,
                beta,
            } => {
                let n_samples = (duration * sample_rate) as usize;
                let mut samples = Vec::with_capacity(n_samples);
                let t_center = duration / 2.0;

                for i in 0..n_samples {
                    let t = i as f64 / sample_rate;
                    let t_shifted = t - t_center;
                    let gaussian = (-(t_shifted).powi(2) / (2.0 * sigma.powi(2))).exp();
                    let derivative = -t_shifted / sigma.powi(2) * gaussian;
                    let drag = gaussian + Complex64::i() * beta * derivative;
                    samples.push(*amplitude * drag);
                }

                Ok(samples)
            }
            PulseShape::Square {
                duration,
                amplitude,
            } => {
                let n_samples = (duration * sample_rate) as usize;
                Ok(vec![*amplitude; n_samples])
            }
            PulseShape::CosineTapered {
                duration,
                amplitude,
                rise_time,
            } => {
                let n_samples = (duration * sample_rate) as usize;
                let n_rise = (rise_time * sample_rate) as usize;
                let mut samples = Vec::with_capacity(n_samples);

                for i in 0..n_samples {
                    let t = i as f64 / sample_rate;
                    let envelope = if t < *rise_time {
                        0.5 * (1.0 - (PI * (rise_time - t) / rise_time).cos())
                    } else if t > *duration - *rise_time {
                        0.5 * (1.0 - (PI * (t - (*duration - *rise_time)) / rise_time).cos())
                    } else {
                        1.0
                    };
                    samples.push(*amplitude * envelope);
                }

                Ok(samples)
            }
            PulseShape::Arbitrary {
                samples,
                sample_rate: _,
            } => Ok(samples.clone()),
        }
    }

    /// Compute FFT of signal
    fn compute_fft(&self, samples: &[Complex64]) -> DeviceResult<Array1<Complex64>> {
        let mut signal = Array1::from(samples.to_vec());

        // Apply window function
        self.config.window_function.apply(&mut signal);

        // Pad to FFT size
        let mut padded_vec = vec![Complex64::new(0.0, 0.0); self.config.fft_size];
        let copy_len = samples.len().min(self.config.fft_size);
        for i in 0..copy_len {
            padded_vec[i] = signal[i];
        }

        // Perform FFT using scirs2-fft high-level API
        let fft_result = fft(&padded_vec, None)
            .map_err(|e| DeviceError::InvalidInput(format!("FFT failed: {}", e)))?;

        Ok(Array1::from(fft_result))
    }

    /// Compute inverse FFT
    fn compute_ifft(&self, spectrum: &Array1<Complex64>) -> DeviceResult<Vec<Complex64>> {
        // Convert Array1 to Vec for scirs2-fft API
        let spectrum_vec = spectrum.to_vec();

        // Perform IFFT using scirs2-fft high-level API
        let ifft_result = ifft(&spectrum_vec, None)
            .map_err(|e| DeviceError::InvalidInput(format!("IFFT failed: {}", e)))?;

        Ok(ifft_result)
    }

    /// Apply spectral filter to remove unwanted frequency components
    fn apply_spectral_filter(
        &self,
        spectrum: &Array1<Complex64>,
    ) -> DeviceResult<Array1<Complex64>> {
        if !self.config.enable_filtering {
            return Ok(spectrum.clone());
        }

        let mut filtered = spectrum.clone();
        let cutoff_bin = (self.config.filter_cutoff * self.config.fft_size as f64
            / self.config.sample_rate) as usize;

        // Apply low-pass filter (simple frequency domain cutoff)
        for i in cutoff_bin..filtered.len() {
            filtered[i] = Complex64::new(0.0, 0.0);
        }

        Ok(filtered)
    }

    /// Analyze pulse spectrum
    pub fn analyze_spectrum(
        &self,
        pulse: &PulseShape,
        sample_rate: f64,
    ) -> DeviceResult<SpectralAnalysisResult> {
        let samples = self.pulse_to_samples(pulse, sample_rate)?;
        let fft_result = self.compute_fft(&samples)?;

        // Compute power spectral density
        let psd: Vec<f64> = fft_result.iter().map(|c| c.norm_sqr()).collect();

        // Frequency bins
        let df = sample_rate / self.config.fft_size as f64;
        let frequencies: Vec<f64> = (0..psd.len()).map(|i| i as f64 * df).collect();

        // Find peaks (local maxima)
        let mut peaks = Vec::new();
        let max_psd = psd.iter().cloned().fold(0.0f64, f64::max);
        let threshold = 0.001 * max_psd; // Lower threshold (0.1% of max)

        for i in 1..psd.len() - 1 {
            if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > threshold {
                peaks.push((frequencies[i], psd[i]));
            }
        }

        // If no peaks found, add the maximum value as a peak
        if peaks.is_empty() {
            if let Some(max_idx) = psd
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
            {
                peaks.push((frequencies[max_idx], psd[max_idx]));
            }
        }

        // Sort peaks by power
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(10); // Keep top 10 peaks

        // Total power
        let total_power: f64 = psd.iter().sum();

        Ok(SpectralAnalysisResult {
            frequencies,
            psd,
            peaks,
            total_power,
        })
    }

    /// Compute pulse quality metrics
    pub fn compute_quality_metrics(
        &self,
        pulse: &PulseShape,
        sample_rate: f64,
    ) -> DeviceResult<PulseQualityMetrics> {
        let samples = self.pulse_to_samples(pulse, sample_rate)?;
        let spectrum = self.analyze_spectrum(pulse, sample_rate)?;

        // Signal power
        let signal_power: f64 =
            samples.iter().map(|s| s.norm_sqr()).sum::<f64>() / samples.len() as f64;
        let peak_power = samples.iter().map(|s| s.norm_sqr()).fold(0.0f64, f64::max);

        // Estimate SNR (simplified)
        let noise_floor = spectrum.psd.iter().cloned().fold(f64::INFINITY, f64::min);
        let snr = 10.0 * (signal_power / noise_floor.max(1e-10)).log10();

        // Bandwidth (3dB bandwidth)
        let max_psd = spectrum.psd.iter().cloned().fold(0.0f64, f64::max);
        let threshold = max_psd / 2.0; // -3dB point
        let bandwidth = spectrum
            .frequencies
            .iter()
            .zip(&spectrum.psd)
            .filter(|(_, &p)| p > threshold)
            .count() as f64
            * (sample_rate / self.config.fft_size as f64);

        // Center frequency (weighted average)
        let center_frequency = if spectrum.total_power > 0.0 {
            spectrum
                .frequencies
                .iter()
                .zip(&spectrum.psd)
                .map(|(f, p)| f * p)
                .sum::<f64>()
                / spectrum.total_power
        } else {
            0.0
        };

        // Spectral purity (power in main peak vs total)
        let main_peak_power = spectrum.peaks.first().map(|(_, p)| *p).unwrap_or(0.0);
        let spectral_purity = if spectrum.total_power > 0.0 {
            main_peak_power / spectrum.total_power
        } else {
            0.0
        };

        // Total harmonic distortion (simplified)
        let fundamental = spectrum.peaks.first().map(|(_, p)| *p).unwrap_or(0.0);
        let harmonics: f64 = spectrum.peaks.iter().skip(1).take(5).map(|(_, p)| p).sum();
        let thd = if fundamental > 0.0 {
            100.0 * (harmonics / fundamental).sqrt()
        } else {
            0.0
        };

        Ok(PulseQualityMetrics {
            snr,
            peak_power,
            average_power: signal_power,
            bandwidth,
            center_frequency,
            spectral_purity,
            thd,
        })
    }

    /// Optimize pulse schedule using signal processing
    pub fn optimize_schedule(&self, schedule: &PulseSchedule) -> DeviceResult<PulseSchedule> {
        let mut optimized = schedule.clone();

        if self.config.enable_fft_optimization {
            // Optimize each pulse in the schedule
            for instruction in &mut optimized.instructions {
                let sample_rate = self
                    .calibration
                    .as_ref()
                    .map(|c| 1.0 / c.dt)
                    .unwrap_or(self.config.sample_rate);

                instruction.pulse = self.optimize_pulse_shape(&instruction.pulse, sample_rate)?;
            }
        }

        Ok(optimized)
    }

    /// Generate pulse quality report
    pub fn generate_quality_report(
        &self,
        pulse: &PulseShape,
        sample_rate: f64,
    ) -> DeviceResult<String> {
        let metrics = self.compute_quality_metrics(pulse, sample_rate)?;
        let spectrum = self.analyze_spectrum(pulse, sample_rate)?;

        let mut report = String::from("=== Pulse Quality Analysis Report ===\n\n");
        report.push_str("Signal Quality Metrics:\n");
        report.push_str(&format!("  SNR: {:.2} dB\n", metrics.snr));
        report.push_str(&format!("  Peak Power: {:.4}\n", metrics.peak_power));
        report.push_str(&format!("  Average Power: {:.4}\n", metrics.average_power));
        report.push_str(&format!(
            "  Bandwidth: {:.2} MHz\n",
            metrics.bandwidth / 1e6
        ));
        report.push_str(&format!(
            "  Center Frequency: {:.2} MHz\n",
            metrics.center_frequency / 1e6
        ));
        report.push_str(&format!(
            "  Spectral Purity: {:.1}%\n",
            metrics.spectral_purity * 100.0
        ));
        report.push_str(&format!("  THD: {:.2}%\n\n", metrics.thd));

        report.push_str("Spectral Analysis:\n");
        report.push_str(&format!("  Total Power: {:.4}\n", spectrum.total_power));
        report.push_str(&format!("  Number of Peaks: {}\n", spectrum.peaks.len()));
        report.push_str("  Top Frequency Components:\n");
        for (i, (freq, power)) in spectrum.peaks.iter().take(5).enumerate() {
            report.push_str(&format!(
                "    {}: {:.2} MHz ({:.2}% of total)\n",
                i + 1,
                freq / 1e6,
                100.0 * power / spectrum.total_power
            ));
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let config = SignalProcessingConfig::default();
        let controller = SciRS2PulseController::new(config);
        assert!(controller.calibration.is_none());
    }

    #[test]
    fn test_pulse_to_samples_gaussian() {
        let config = SignalProcessingConfig::default();
        let controller = SciRS2PulseController::new(config);

        let pulse = PulseShape::Gaussian {
            duration: 100e-9, // 100 ns
            sigma: 20e-9,     // 20 ns
            amplitude: Complex64::new(1.0, 0.0),
        };

        let samples = controller
            .pulse_to_samples(&pulse, 1e9)
            .expect("Failed to convert pulse to samples");

        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|s| s.norm() <= 1.0));
    }

    #[test]
    fn test_pulse_to_samples_square() {
        let config = SignalProcessingConfig::default();
        let controller = SciRS2PulseController::new(config);

        let pulse = PulseShape::Square {
            duration: 50e-9,
            amplitude: Complex64::new(0.5, 0.0),
        };

        let samples = controller
            .pulse_to_samples(&pulse, 1e9)
            .expect("Failed to convert pulse to samples");

        assert_eq!(samples.len(), 50);
        assert!(samples.iter().all(|s| (s.norm() - 0.5).abs() < 1e-10));
    }

    #[test]
    fn test_window_functions() {
        let mut signal = Array1::from(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);

        // Test Hamming window
        WindowType::Hamming.apply(&mut signal);
        assert!(signal[0].re < 1.0); // Windowed values should be < 1
        assert!(signal[signal.len() - 1].re < 1.0);

        // Test Hanning window
        let mut signal2 = Array1::from(vec![Complex64::new(1.0, 0.0); 4]);
        WindowType::Hanning.apply(&mut signal2);
        assert!(signal2[0].re < 1.0);
    }

    #[test]
    fn test_spectrum_analysis() {
        let config = SignalProcessingConfig {
            fft_size: 256,
            ..Default::default()
        };
        let controller = SciRS2PulseController::new(config);

        let pulse = PulseShape::Gaussian {
            duration: 100e-9,
            sigma: 20e-9,
            amplitude: Complex64::new(1.0, 0.0),
        };

        let spectrum = controller
            .analyze_spectrum(&pulse, 1e9)
            .expect("Failed to analyze spectrum");

        assert_eq!(spectrum.frequencies.len(), spectrum.psd.len());
        assert!(spectrum.total_power > 0.0);
        assert!(!spectrum.peaks.is_empty());
    }

    #[test]
    fn test_quality_metrics() {
        let config = SignalProcessingConfig {
            fft_size: 256,
            ..Default::default()
        };
        let controller = SciRS2PulseController::new(config);

        let pulse = PulseShape::Gaussian {
            duration: 100e-9,
            sigma: 20e-9,
            amplitude: Complex64::new(1.0, 0.0),
        };

        let metrics = controller
            .compute_quality_metrics(&pulse, 1e9)
            .expect("Failed to compute metrics");

        assert!(metrics.peak_power > 0.0);
        assert!(metrics.average_power > 0.0);
        assert!(metrics.bandwidth > 0.0);
        assert!(metrics.spectral_purity >= 0.0 && metrics.spectral_purity <= 1.0);
    }

    #[test]
    fn test_quality_report_generation() {
        let config = SignalProcessingConfig {
            fft_size: 256,
            ..Default::default()
        };
        let controller = SciRS2PulseController::new(config);

        let pulse = PulseShape::Gaussian {
            duration: 100e-9,
            sigma: 20e-9,
            amplitude: Complex64::new(1.0, 0.0),
        };

        let report = controller
            .generate_quality_report(&pulse, 1e9)
            .expect("Failed to generate report");

        assert!(report.contains("SNR"));
        assert!(report.contains("Bandwidth"));
        assert!(report.contains("Spectral Analysis"));
    }

    #[test]
    fn test_pulse_optimization() {
        let config = SignalProcessingConfig {
            fft_size: 128,
            enable_fft_optimization: true,
            ..Default::default()
        };
        let controller = SciRS2PulseController::new(config);

        let pulse = PulseShape::Square {
            duration: 50e-9,
            amplitude: Complex64::new(1.0, 0.0),
        };

        let optimized = controller
            .optimize_pulse_shape(&pulse, 1e9)
            .expect("Failed to optimize pulse");

        // Optimized pulse should be Arbitrary type
        match optimized {
            PulseShape::Arbitrary { samples, .. } => {
                assert!(!samples.is_empty());
            }
            _ => panic!("Expected Arbitrary pulse shape"),
        }
    }
}
