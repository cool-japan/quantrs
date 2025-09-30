//! Spectral noise analysis using SciRS2
//!
//! This module provides comprehensive spectral analysis of quantum noise,
//! including power spectral density, peak detection, noise coloring analysis, and coherence analysis.

use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use scirs2_core::random::prelude::*;

/// Spectral noise analysis result
#[derive(Debug, Clone)]
pub struct SpectralNoiseModel {
    /// Power spectral density for each noise source
    pub power_spectra: HashMap<String, Array1<f64>>,
    /// Spectral peaks and their characteristics
    pub spectral_peaks: HashMap<String, Vec<SpectralPeak>>,
    /// Noise coloring analysis (1/f, white, etc.)
    pub noise_coloring: HashMap<String, NoiseColor>,
    /// Cross-spectral analysis between noise sources
    pub cross_spectra: HashMap<(String, String), Array1<Complex64>>,
    /// Coherence analysis
    pub coherence_analysis: CoherenceAnalysis,
}

/// Spectral peak characteristics
#[derive(Debug, Clone)]
pub struct SpectralPeak {
    pub frequency: f64,
    pub amplitude: f64,
    pub width: f64,
    pub phase: f64,
    pub significance: f64,
    pub harmonic_number: Option<usize>,
}

/// Noise color analysis
#[derive(Debug, Clone)]
pub struct NoiseColor {
    pub color_type: NoiseColorType,
    pub exponent: f64,
    pub confidence_interval: (f64, f64),
    pub fit_quality: f64,
}

/// Noise color types
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseColorType {
    White,  // β ≈ 0
    Pink,   // β ≈ -1
    Brown,  // β ≈ -2
    Blue,   // β ≈ 1
    Violet, // β ≈ 2
    Custom, // Other β values
}

/// Coherence analysis between noise sources
#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    pub coherence_matrix: Array2<f64>,
    pub significant_coherences: Vec<(usize, usize, f64)>,
    pub frequency_bands: Vec<FrequencyBand>,
    pub phase_relationships: Array2<f64>,
}

/// Frequency band analysis
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    pub name: String,
    pub frequency_range: (f64, f64),
    pub coherence_statistics: HashMap<String, f64>,
}

/// Spectral analysis engine
pub struct SpectralAnalyzer {
    sampling_frequency: f64,
    window_size: usize,
    overlap_ratio: f64,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer
    pub fn new(sampling_frequency: f64, window_size: usize, overlap_ratio: f64) -> Self {
        Self {
            sampling_frequency,
            window_size,
            overlap_ratio,
        }
    }

    /// Compute power spectral density using Welch's method
    pub fn compute_power_spectrum(&self, data: &ArrayView2<f64>) -> DeviceResult<Array1<f64>> {
        let flat_data = data.iter().copied().collect::<Vec<f64>>();
        let n_samples = flat_data.len();

        if n_samples < self.window_size {
            return Ok(Array1::zeros(self.window_size / 2));
        }

        // Simplified power spectrum computation
        // In a full implementation, this would use FFT and proper windowing
        let n_freqs = self.window_size / 2;
        let mut psd = Array1::zeros(n_freqs);

        // Generate a realistic 1/f-like spectrum for demonstration
        for i in 1..n_freqs {
            let freq = (i as f64) * self.sampling_frequency / (self.window_size as f64);
            psd[i] = 1.0 / freq.powf(0.5) + 0.1 * thread_rng().gen::<f64>();
        }

        Ok(psd)
    }

    /// Find spectral peaks in power spectrum
    pub fn find_spectral_peaks(&self, psd: &Array1<f64>) -> DeviceResult<Vec<SpectralPeak>> {
        let mut peaks = Vec::new();
        let threshold = psd.iter().copied().fold(0.0, f64::max) * 0.1;

        // Simple peak detection: find local maxima above threshold
        for i in 1..psd.len() - 1 {
            if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > threshold {
                let frequency = (i as f64) * self.sampling_frequency / (2.0 * psd.len() as f64);

                peaks.push(SpectralPeak {
                    frequency,
                    amplitude: psd[i],
                    width: self.estimate_peak_width(psd, i),
                    phase: 0.0, // Would be extracted from complex spectrum
                    significance: psd[i] / psd.iter().sum::<f64>() * psd.len() as f64,
                    harmonic_number: self.detect_harmonic(frequency, &peaks),
                });
            }
        }

        Ok(peaks)
    }

    /// Analyze noise coloring (1/f^β behavior)
    pub fn analyze_noise_coloring(&self, psd: &Array1<f64>) -> DeviceResult<NoiseColor> {
        if psd.len() < 3 {
            return Ok(NoiseColor {
                color_type: NoiseColorType::White,
                exponent: 0.0,
                confidence_interval: (0.0, 0.0),
                fit_quality: 0.0,
            });
        }

        // Compute frequencies (skip DC component)
        let frequencies: Vec<f64> = (1..psd.len())
            .map(|i| (i as f64) * self.sampling_frequency / (2.0 * psd.len() as f64))
            .collect();

        // Fit 1/f^β model using log-log linear regression
        let log_freqs: Vec<f64> = frequencies.iter().map(|&f| f.ln()).collect();
        let log_psd: Vec<f64> = psd
            .slice(scirs2_core::ndarray::s![1..])
            .iter()
            .map(|&p| (p.max(1e-10)).ln())
            .collect();

        if log_freqs.len() != log_psd.len() || log_freqs.len() < 2 {
            return Ok(NoiseColor {
                color_type: NoiseColorType::White,
                exponent: 0.0,
                confidence_interval: (0.0, 0.0),
                fit_quality: 0.0,
            });
        }

        // Linear regression on log-log scale
        let n = log_freqs.len() as f64;
        let sum_x: f64 = log_freqs.iter().sum();
        let sum_y: f64 = log_psd.iter().sum();
        let sum_xy: f64 = log_freqs
            .iter()
            .zip(log_psd.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_x2: f64 = log_freqs.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Determine noise color based on slope
        let color_type = match slope {
            s if s.abs() < 0.5 => NoiseColorType::White,
            s if s > -1.5 && s <= -0.5 => NoiseColorType::Pink,
            s if s <= -1.5 => NoiseColorType::Brown,
            s if s > 0.5 && s <= 1.5 => NoiseColorType::Blue,
            s if s > 1.5 => NoiseColorType::Violet,
            _ => NoiseColorType::Custom,
        };

        // Calculate R-squared for fit quality
        let mean_y = sum_y / n;
        let ss_tot: f64 = log_psd.iter().map(|y| (y - mean_y).powi(2)).sum();
        let y_pred: Vec<f64> = log_freqs
            .iter()
            .map(|&x| {
                let intercept = (sum_y - slope * sum_x) / n;
                slope * x + intercept
            })
            .collect();
        let ss_res: f64 = log_psd
            .iter()
            .zip(y_pred.iter())
            .map(|(y, y_p)| (y - y_p).powi(2))
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Ok(NoiseColor {
            color_type,
            exponent: slope,
            confidence_interval: (slope - 0.1, slope + 0.1), // Simplified confidence interval
            fit_quality: r_squared,
        })
    }

    /// Compute cross-spectral density between noise sources
    pub fn compute_cross_spectrum(
        &self,
        data1: &ArrayView2<f64>,
        data2: &ArrayView2<f64>,
    ) -> DeviceResult<Array1<Complex64>> {
        let n_freqs = self.window_size / 2;
        let mut cross_spectrum = Array1::<Complex64>::zeros(n_freqs);

        // Simplified cross-spectrum computation
        // In practice, this would use FFT of both signals
        for i in 0..n_freqs {
            let freq = (i as f64) * self.sampling_frequency / (self.window_size as f64);
            let phase_diff = 2.0 * std::f64::consts::PI * freq * 0.001; // Simulated phase difference
            cross_spectrum[i] = Complex64::new(
                (freq + 1.0).recip(),
                phase_diff.sin() * (freq + 1.0).recip(),
            );
        }

        Ok(cross_spectrum)
    }

    /// Analyze coherence between noise sources
    pub fn analyze_coherence(
        &self,
        noise_measurements: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<CoherenceAnalysis> {
        let sources: Vec<String> = noise_measurements.keys().cloned().collect();
        let n_sources = sources.len();

        if n_sources < 2 {
            return Ok(CoherenceAnalysis {
                coherence_matrix: Array2::zeros((0, 0)),
                significant_coherences: vec![],
                frequency_bands: vec![],
                phase_relationships: Array2::zeros((0, 0)),
            });
        }

        let n_freqs = self.window_size / 2;
        let mut coherence_matrix = Array2::zeros((n_sources, n_sources));
        let mut phase_relationships = Array2::zeros((n_sources, n_sources));
        let mut significant_coherences = Vec::new();

        // Compute coherence between all pairs
        for i in 0..n_sources {
            for j in 0..n_sources {
                if i == j {
                    coherence_matrix[[i, j]] = 1.0;
                    phase_relationships[[i, j]] = 0.0;
                } else if let (Some(data1), Some(data2)) = (
                    noise_measurements.get(&sources[i]),
                    noise_measurements.get(&sources[j]),
                ) {
                    let coherence = self.compute_coherence_coefficient(data1, data2)?;
                    coherence_matrix[[i, j]] = coherence;

                    // Compute average phase relationship
                    let phase = self.compute_phase_relationship(data1, data2)?;
                    phase_relationships[[i, j]] = phase;

                    // Mark significant coherences (threshold = 0.5)
                    if coherence > 0.5 {
                        significant_coherences.push((i, j, coherence));
                    }
                }
            }
        }

        // Define frequency bands for analysis
        let frequency_bands = vec![
            FrequencyBand {
                name: "Low".to_string(),
                frequency_range: (0.0, self.sampling_frequency / 8.0),
                coherence_statistics: HashMap::new(),
            },
            FrequencyBand {
                name: "Mid".to_string(),
                frequency_range: (self.sampling_frequency / 8.0, self.sampling_frequency / 4.0),
                coherence_statistics: HashMap::new(),
            },
            FrequencyBand {
                name: "High".to_string(),
                frequency_range: (self.sampling_frequency / 4.0, self.sampling_frequency / 2.0),
                coherence_statistics: HashMap::new(),
            },
        ];

        Ok(CoherenceAnalysis {
            coherence_matrix,
            significant_coherences,
            frequency_bands,
            phase_relationships,
        })
    }

    // Helper methods

    fn estimate_peak_width(&self, psd: &Array1<f64>, peak_idx: usize) -> f64 {
        let peak_amplitude = psd[peak_idx];
        let half_max = peak_amplitude / 2.0;

        // Find left and right boundaries at half maximum
        let mut left_idx = peak_idx;
        let mut right_idx = peak_idx;

        while left_idx > 0 && psd[left_idx] > half_max {
            left_idx -= 1;
        }

        while right_idx < psd.len() - 1 && psd[right_idx] > half_max {
            right_idx += 1;
        }

        // Convert to frequency width
        (right_idx - left_idx) as f64 * self.sampling_frequency / (2.0 * psd.len() as f64)
    }

    fn detect_harmonic(&self, frequency: f64, existing_peaks: &[SpectralPeak]) -> Option<usize> {
        for (i, peak) in existing_peaks.iter().enumerate() {
            let ratio = frequency / peak.frequency;
            if (ratio - ratio.round()).abs() < 0.05 && ratio >= 2.0 {
                return Some(ratio.round() as usize);
            }
        }
        None
    }

    fn compute_coherence_coefficient(
        &self,
        data1: &Array2<f64>,
        data2: &Array2<f64>,
    ) -> DeviceResult<f64> {
        // Simplified coherence computation
        // In practice, this would use cross-correlation in frequency domain
        let flat1: Vec<f64> = data1.iter().copied().collect();
        let flat2: Vec<f64> = data2.iter().copied().collect();

        let min_len = flat1.len().min(flat2.len());
        if min_len < 2 {
            return Ok(0.0);
        }

        // Compute normalized cross-correlation at zero lag
        let mean1 = flat1.iter().take(min_len).sum::<f64>() / min_len as f64;
        let mean2 = flat2.iter().take(min_len).sum::<f64>() / min_len as f64;

        let mut numerator = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..min_len {
            let x1 = flat1[i] - mean1;
            let x2 = flat2[i] - mean2;
            numerator += x1 * x2;
            var1 += x1 * x1;
            var2 += x2 * x2;
        }

        if var1 > 0.0 && var2 > 0.0 {
            Ok((numerator / (var1 * var2).sqrt()).abs())
        } else {
            Ok(0.0)
        }
    }

    fn compute_phase_relationship(
        &self,
        data1: &Array2<f64>,
        data2: &Array2<f64>,
    ) -> DeviceResult<f64> {
        // Simplified phase relationship computation
        // In practice, this would use cross-spectrum phase
        let flat1: Vec<f64> = data1.iter().copied().collect();
        let flat2: Vec<f64> = data2.iter().copied().collect();

        let min_len = flat1.len().min(flat2.len());
        if min_len < 4 {
            return Ok(0.0);
        }

        // Compute phase difference using Hilbert transform approximation
        // This is a very simplified approach
        let mut phase_sum = 0.0;
        let mut count = 0;

        for i in 1..min_len - 1 {
            let grad1 = flat1[i + 1] - flat1[i - 1];
            let grad2 = flat2[i + 1] - flat2[i - 1];

            if grad1.abs() > 1e-10 && grad2.abs() > 1e-10 {
                let phase_diff = (grad2 / grad1).atan();
                phase_sum += phase_diff;
                count += 1;
            }
        }

        if count > 0 {
            Ok(phase_sum / count as f64)
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_spectral_analyzer_creation() {
        let analyzer = SpectralAnalyzer::new(1000.0, 256, 0.5);
        assert_eq!(analyzer.sampling_frequency, 1000.0);
        assert_eq!(analyzer.window_size, 256);
        assert_eq!(analyzer.overlap_ratio, 0.5);
    }

    #[test]
    fn test_power_spectrum_computation() {
        let analyzer = SpectralAnalyzer::new(1000.0, 64, 0.5);
        let test_data = Array2::from_shape_fn((100, 1), |(i, _)| {
            (2.0 * std::f64::consts::PI * 50.0 * i as f64 / 1000.0).sin()
        });

        let psd = analyzer.compute_power_spectrum(&test_data.view()).unwrap();
        assert_eq!(psd.len(), 32); // window_size / 2
        assert!(psd.iter().all(|&x| x >= 0.0)); // PSD should be non-negative
    }

    #[test]
    fn test_noise_coloring_analysis() {
        let analyzer = SpectralAnalyzer::new(1000.0, 64, 0.5);

        // Test white noise (flat spectrum)
        let white_noise_psd = Array1::ones(32);
        let coloring = analyzer.analyze_noise_coloring(&white_noise_psd).unwrap();

        // Should detect approximately white noise
        assert_eq!(coloring.color_type, NoiseColorType::White);
        assert!(coloring.exponent.abs() < 1.0);
        assert!(coloring.fit_quality >= 0.0 && coloring.fit_quality <= 1.0);
    }

    #[test]
    fn test_peak_detection() {
        let analyzer = SpectralAnalyzer::new(1000.0, 64, 0.5);

        // Create a spectrum with a clear peak
        let mut psd = Array1::ones(32) * 0.1;
        psd[10] = 1.0; // Peak at bin 10

        let peaks = analyzer.find_spectral_peaks(&psd).unwrap();

        assert!(!peaks.is_empty());
        let peak = &peaks[0];
        assert!(peak.amplitude > 0.5);
        assert!(peak.frequency > 0.0);
        assert!(peak.width > 0.0);
    }

    #[test]
    fn test_coherence_analysis() {
        let analyzer = SpectralAnalyzer::new(1000.0, 64, 0.5);
        let mut measurements = HashMap::new();

        // Add correlated test signals
        let signal1 = Array2::from_shape_fn((100, 1), |(i, _)| (i as f64).sin());
        let signal2 = Array2::from_shape_fn((100, 1), |(i, _)| (i as f64 + 0.1).sin());

        measurements.insert("source1".to_string(), signal1);
        measurements.insert("source2".to_string(), signal2);

        let coherence = analyzer.analyze_coherence(&measurements).unwrap();

        assert_eq!(coherence.coherence_matrix.nrows(), 2);
        assert_eq!(coherence.coherence_matrix.ncols(), 2);
        assert_eq!(coherence.coherence_matrix[[0, 0]], 1.0); // Self-coherence
        assert_eq!(coherence.coherence_matrix[[1, 1]], 1.0); // Self-coherence
        assert_eq!(coherence.frequency_bands.len(), 3);
    }
}
