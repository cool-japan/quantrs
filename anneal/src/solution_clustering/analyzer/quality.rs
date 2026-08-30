//! Landscape analysis, statistical analysis, and overall quality/recommendation
//! computation for [`super::SolutionClusteringAnalyzer`].
//!
//! Split out of `analyzer.rs` to keep individual files under the workspace's
//! 2000-line limit; this submodule adds a second `impl` block for the same
//! [`super::SolutionClusteringAnalyzer`] type.

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;

use super::SolutionClusteringAnalyzer;
use crate::solution_clustering::error::{ClusteringError, ClusteringResult};
use crate::solution_clustering::types::{
    ConnectivityAnalysis, ConvergenceAnalysis, CorrelationAnalysis, CorrelationPattern,
    DifficultyLevel, DistributionAnalysis, DistributionType, EnergyBasin, EnergyStatistics,
    FunnelAnalysis, LandscapeAnalysis, MultiModalityAnalysis, OptimizationRecommendation,
    OutlierInfo, OutlierType, OverallClusteringQuality, PatternType, PlateauAnalysis,
    PriorityLevel, RecommendationType, RuggednessMetrics, SolutionCluster, SolutionPoint,
    StatisticalSummary,
};

impl SolutionClusteringAnalyzer {
    /// Analyze the solution landscape
    pub(crate) fn analyze_landscape(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<LandscapeAnalysis> {
        let energy_statistics = self.calculate_energy_statistics(solution_points);
        let basins = self.detect_energy_basins(solution_points, clusters);
        let connectivity = self.analyze_connectivity(solution_points);
        let multi_modality = self.analyze_multi_modality(solution_points);
        let ruggedness = self.calculate_ruggedness_metrics(solution_points);
        let funnel_analysis = self.analyze_funnel_structure(solution_points, clusters);

        Ok(LandscapeAnalysis {
            energy_statistics,
            basins,
            connectivity,
            multi_modality,
            ruggedness,
            funnel_analysis,
        })
    }

    /// Calculate energy statistics
    #[must_use]
    pub fn calculate_energy_statistics(
        &self,
        solution_points: &[SolutionPoint],
    ) -> EnergyStatistics {
        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();

        if energies.is_empty() {
            return EnergyStatistics {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                percentiles: Vec::new(),
                skewness: 0.0,
                kurtosis: 0.0,
                num_distinct_energies: 0,
            };
        }

        let mut sorted_energies = energies.clone();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / energies.len() as f64;
        let std_dev = variance.sqrt();

        let min = sorted_energies[0];
        let max = sorted_energies[sorted_energies.len() - 1];

        // Calculate percentiles
        let percentiles = vec![
            sorted_energies[sorted_energies.len() * 25 / 100],
            sorted_energies[sorted_energies.len() * 50 / 100],
            sorted_energies[sorted_energies.len() * 75 / 100],
        ];

        // Calculate skewness and kurtosis (simplified)
        let skewness = if std_dev > 1e-10 {
            energies
                .iter()
                .map(|e| ((e - mean) / std_dev).powi(3))
                .sum::<f64>()
                / energies.len() as f64
        } else {
            0.0
        };

        let kurtosis = if std_dev > 1e-10 {
            energies
                .iter()
                .map(|e| ((e - mean) / std_dev).powi(4))
                .sum::<f64>()
                / energies.len() as f64
                - 3.0
        } else {
            0.0
        };

        let mut sorted_energies = energies;
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_energies.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let num_distinct_energies = sorted_energies.len();

        EnergyStatistics {
            mean,
            std_dev,
            min,
            max,
            percentiles,
            skewness,
            kurtosis,
            num_distinct_energies,
        }
    }

    /// Detect energy basins in the landscape.
    ///
    /// Each cluster is treated as a basin. Per-basin depth is computed against
    /// the global minimum energy across `solution_points` (so the deepest basin
    /// has depth `0`, and shallower basins have positive depth — interpreted as
    /// "energy above the global minimum"). Width is the basin's energy range.
    /// `escape_barrier` is left at `0.0` since a real estimate requires an
    /// inter-basin transition graph that is not maintained here.
    pub(crate) fn detect_energy_basins(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> Vec<EnergyBasin> {
        let mut basins = Vec::new();

        let global_min = solution_points
            .iter()
            .map(|s| s.energy)
            .fold(f64::INFINITY, f64::min);

        for (basin_id, cluster) in clusters.iter().enumerate() {
            let energies: Vec<f64> = cluster.solutions.iter().map(|s| s.energy).collect();

            if !energies.is_empty() {
                let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let depth = if global_min.is_finite() {
                    (min_energy - global_min).max(0.0)
                } else {
                    0.0
                };

                basins.push(EnergyBasin {
                    id: basin_id,
                    solutions: cluster.solutions.iter().map(|s| s.metadata.id).collect(),
                    min_energy,
                    size: cluster.solutions.len(),
                    depth,
                    width: max_energy - min_energy,
                    escape_barrier: 0.0, // Requires an inter-basin transition graph.
                });
            }
        }

        basins
    }

    /// Analyze connectivity of the solution landscape via single-link clustering
    /// over an epsilon-Hamming-neighbour graph.
    ///
    /// Two solutions are considered "connected" when the Hamming distance between
    /// their spin vectors is at most `eps_hamming`. Connected components are then
    /// found with a union-find pass, yielding `num_components` and
    /// `largest_component_size`. `average_path_length`, `clustering_coefficient`,
    /// and `diameter` are still simplified estimates.
    pub(crate) fn analyze_connectivity(
        &self,
        solution_points: &[SolutionPoint],
    ) -> ConnectivityAnalysis {
        let n = solution_points.len();
        if n == 0 {
            return ConnectivityAnalysis {
                num_components: 0,
                largest_component_size: 0,
                average_path_length: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
            };
        }
        if n == 1 {
            return ConnectivityAnalysis {
                num_components: 1,
                largest_component_size: 1,
                average_path_length: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
            };
        }

        // Use a Hamming-neighbour threshold of 1 by default. Any two solutions
        // differing in a single spin are direct neighbours; chains of such
        // single-flip moves form a connected component.
        let eps_hamming: usize = 1;

        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let a = &solution_points[i].solution;
                let b = &solution_points[j].solution;
                if a.len() != b.len() {
                    continue;
                }
                let hd = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                if hd <= eps_hamming {
                    let ra = find(&mut parent, i);
                    let rb = find(&mut parent, j);
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }

        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            *sizes.entry(r).or_insert(0) += 1;
        }

        let num_components = sizes.len();
        let largest_component_size = sizes.values().copied().max().unwrap_or(0);

        // Conservative simplified estimates for the remaining fields.
        let average_path_length = if num_components == 0 {
            0.0
        } else {
            (n as f64 / num_components as f64).sqrt()
        };

        ConnectivityAnalysis {
            num_components,
            largest_component_size,
            average_path_length,
            clustering_coefficient: 0.3, // Heuristic — full computation is out of scope here.
            diameter: largest_component_size.saturating_sub(1),
        }
    }

    /// Analyze multi-modality of the energy landscape via 1D histogram peak detection.
    ///
    /// The energy values are bucketed into `min(20, ceil(sqrt(n)))` equal-width
    /// bins between `min(E)` and `max(E)`. A bin is a mode when its count is
    /// strictly greater than both neighbour bins; the first and last bins are
    /// modes when their count exceeds their single neighbour. The mode energy
    /// is the bin centre; mode strength is the bin's relative population.
    /// Inter-mode distances use the centre-to-centre absolute energy gap.
    pub(crate) fn analyze_multi_modality(
        &self,
        solution_points: &[SolutionPoint],
    ) -> MultiModalityAnalysis {
        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();
        let n = energies.len();

        if n == 0 {
            return MultiModalityAnalysis {
                num_modes: 0,
                mode_energies: Vec::new(),
                mode_strengths: Vec::new(),
                inter_mode_distances: Vec::new(),
                multi_modality_index: 0.0,
            };
        }

        let min_e = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_e = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Degenerate range: a single mode at the common energy.
        if (max_e - min_e).abs() < 1e-12 {
            return MultiModalityAnalysis {
                num_modes: 1,
                mode_energies: vec![min_e],
                mode_strengths: vec![1.0],
                inter_mode_distances: vec![vec![0.0]],
                multi_modality_index: 0.0,
            };
        }

        let num_bins = ((n as f64).sqrt().ceil() as usize).clamp(2, 20);
        let bin_width = (max_e - min_e) / num_bins as f64;
        let mut counts = vec![0usize; num_bins];
        for &e in &energies {
            let mut idx = ((e - min_e) / bin_width).floor() as isize;
            if idx < 0 {
                idx = 0;
            }
            let mut idx = idx as usize;
            if idx >= num_bins {
                idx = num_bins - 1;
            }
            counts[idx] += 1;
        }

        // Detect peaks (strict local maxima with non-zero population).
        let mut mode_bins: Vec<usize> = Vec::new();
        for i in 0..num_bins {
            if counts[i] == 0 {
                continue;
            }
            let left_ok = i == 0 || counts[i] > counts[i - 1];
            let right_ok = i + 1 == num_bins || counts[i] > counts[i + 1];
            if left_ok && right_ok {
                mode_bins.push(i);
            }
        }

        // If the histogram is perfectly flat or monotone, fall back to "the
        // single most populous bin is the (only) mode" — guarantees at least one.
        if mode_bins.is_empty() {
            let (best_idx, _) = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| **c)
                .unwrap_or((0, &0));
            mode_bins.push(best_idx);
        }

        let mode_energies: Vec<f64> = mode_bins
            .iter()
            .map(|&b| min_e + (b as f64 + 0.5) * bin_width)
            .collect();
        let total = n as f64;
        let mode_strengths: Vec<f64> = mode_bins
            .iter()
            .map(|&b| counts[b] as f64 / total)
            .collect();

        // Symmetric inter-mode distance matrix in energy units.
        let m = mode_energies.len();
        let mut inter_mode_distances = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                inter_mode_distances[i][j] = (mode_energies[i] - mode_energies[j]).abs();
            }
        }

        // Multi-modality index: (m - 1) / m saturates toward 1 as more modes
        // are detected, 0 when there is just one. Capped at 1.
        let multi_modality_index = if m <= 1 {
            0.0
        } else {
            ((m - 1) as f64 / m as f64).min(1.0)
        };

        MultiModalityAnalysis {
            num_modes: m,
            mode_energies,
            mode_strengths,
            inter_mode_distances,
            multi_modality_index,
        }
    }

    /// Calculate ruggedness metrics for the solution landscape.
    ///
    /// Computes the lag-k autocorrelation of the energy sequence (treating the
    /// solution index order as a synthetic walk through the landscape) up to
    /// `max_lag = min(5, n - 1)`:
    ///
    /// ```text
    ///                Σ_{i=0}^{n-1-k} (e_i - μ)(e_{i+k} - μ)
    /// rho(k)  =  ─────────────────────────────────────────
    ///                       Σ_{i=0}^{n-1} (e_i - μ)^2
    /// ```
    ///
    /// The ruggedness coefficient is `1 - rho(1)`: smooth landscapes have
    /// `rho(1) ~= 1` and small ruggedness; rugged landscapes have low/negative
    /// `rho(1)` and ruggedness near or above 1.
    ///
    /// `epistasis` and `neutrality` are heuristic placeholders pending a true
    /// landscape walk infrastructure.
    pub(crate) fn calculate_ruggedness_metrics(
        &self,
        solution_points: &[SolutionPoint],
    ) -> RuggednessMetrics {
        let n = solution_points.len();
        if n < 2 {
            return RuggednessMetrics {
                autocorrelation: Vec::new(),
                ruggedness_coefficient: 0.0,
                num_local_optima: 0,
                epistasis: 0.0,
                neutrality: 0.0,
            };
        }

        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();
        let mean = energies.iter().sum::<f64>() / n as f64;
        let denom: f64 = energies.iter().map(|e| (e - mean).powi(2)).sum();

        let max_lag = 5.min(n - 1);
        let mut autocorrelation = Vec::with_capacity(max_lag);
        if denom < 1e-12 {
            // Constant energy series — autocorrelation is conventionally 1 at every lag.
            for _ in 0..max_lag {
                autocorrelation.push(1.0);
            }
        } else {
            for k in 1..=max_lag {
                let mut num = 0.0;
                for i in 0..(n - k) {
                    num += (energies[i] - mean) * (energies[i + k] - mean);
                }
                autocorrelation.push(num / denom);
            }
        }

        let ruggedness_coefficient = autocorrelation
            .first()
            .map(|rho1| (1.0 - rho1).max(0.0))
            .unwrap_or(0.0);

        // Local optima along the index-ordered walk: positions where the energy
        // is strictly less than both neighbours (a 1D minimum). This is a real
        // count over the available data, not a placeholder.
        let mut num_local_optima = 0;
        for i in 1..(n - 1) {
            if energies[i] < energies[i - 1] && energies[i] < energies[i + 1] {
                num_local_optima += 1;
            }
        }

        RuggednessMetrics {
            autocorrelation,
            ruggedness_coefficient,
            num_local_optima,
            // Pending a proper neighbour-graph walk; left as documented heuristics.
            epistasis: 0.3,
            neutrality: 0.1,
        }
    }

    /// Analyze funnel structure
    fn analyze_funnel_structure(
        &self,
        _solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> FunnelAnalysis {
        // Simplified funnel analysis
        FunnelAnalysis {
            num_funnels: clusters.len(),
            funnel_depths: clusters.iter().map(|c| c.statistics.energy_std).collect(),
            funnel_widths: clusters.iter().map(|c| c.statistics.diameter).collect(),
            global_funnel: Some(0), // Simplified
            competition_index: 0.5, // Simplified
        }
    }

    /// Perform statistical analysis over the real solution/cluster data.
    ///
    /// Every field of the returned [`StatisticalSummary`] is derived from
    /// `solution_points`/`clusters` (see [`Self::analyze_energy_distribution`],
    /// [`Self::analyze_convergence`], [`Self::analyze_variable_correlations`]
    /// and [`Self::detect_statistical_outliers`]) rather than fixed literals.
    pub(crate) fn perform_statistical_analysis(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<StatisticalSummary> {
        let cluster_size_distribution = clusters.iter().map(|c| c.statistics.size).collect();

        let energy_distribution = self.analyze_energy_distribution(solution_points);
        let convergence_analysis = self.analyze_convergence(solution_points, clusters);
        let correlation_analysis = self.analyze_variable_correlations(solution_points);
        let outliers = self.detect_statistical_outliers(solution_points, clusters)?;

        Ok(StatisticalSummary {
            cluster_size_distribution,
            energy_distribution,
            convergence_analysis,
            correlation_analysis,
            outliers,
        })
    }

    /// Analyze the empirical energy distribution.
    ///
    /// Normality is assessed with the Jarque-Bera statistic
    /// `JB = n/6 * (S^2 + K^2/4)` (S = skewness, K = excess kurtosis) computed
    /// from [`Self::calculate_energy_statistics`]; `goodness_of_fit` is the
    /// real chi-squared(2) upper-tail p-value of `JB` (higher = more
    /// consistent with a Normal distribution). When the sample is too small
    /// or degenerate to test, or the test rejects normality without a
    /// well-supported alternative family, [`DistributionType::Unknown`] is
    /// returned rather than a fabricated guess.
    fn analyze_energy_distribution(
        &self,
        solution_points: &[SolutionPoint],
    ) -> DistributionAnalysis {
        let stats = self.calculate_energy_statistics(solution_points);
        let n = solution_points.len();

        if n < 8 || stats.std_dev < 1e-12 {
            return DistributionAnalysis {
                distribution_type: DistributionType::Unknown,
                parameters: HashMap::from([
                    ("mean".to_string(), stats.mean),
                    ("std".to_string(), stats.std_dev),
                ]),
                goodness_of_fit: 0.0,
                confidence_intervals: vec![(stats.mean, stats.mean)],
            };
        }

        let jarque_bera_statistic =
            (n as f64 / 6.0) * (stats.skewness.powi(2) + stats.kurtosis.powi(2) / 4.0);
        let goodness_of_fit = scirs2_stats::distributions::chi2::<f64>(2.0, 0.0, 1.0)
            .ok()
            .map(|dist| (1.0 - dist.cdf(jarque_bera_statistic)).clamp(0.0, 1.0))
            .unwrap_or(0.0);

        let distribution_type = if goodness_of_fit >= 0.05 {
            DistributionType::Normal
        } else if stats.min >= 0.0 && stats.skewness > 1.0 {
            DistributionType::Exponential
        } else {
            DistributionType::Unknown
        };

        let se_mean = stats.std_dev / (n as f64).sqrt();
        let confidence_intervals = vec![(stats.mean - 1.96 * se_mean, stats.mean + 1.96 * se_mean)];

        DistributionAnalysis {
            distribution_type,
            parameters: HashMap::from([
                ("mean".to_string(), stats.mean),
                ("std".to_string(), stats.std_dev),
                ("skewness".to_string(), stats.skewness),
                ("kurtosis".to_string(), stats.kurtosis),
                ("jarque_bera_statistic".to_string(), jarque_bera_statistic),
            ]),
            goodness_of_fit,
            confidence_intervals,
        }
    }

    /// Analyze convergence behavior from the real `iterations`/`energy`
    /// metadata already carried on each [`SolutionPoint`].
    ///
    /// `trajectory_clusters` is left empty: this crate only records the final
    /// solution reached by each run (`SolutionPoint`), not a per-iteration
    /// energy trajectory, so there is no real trajectory data to cluster.
    /// Everything else here is computed from the actual sample:
    /// * `convergence_rates`: per-cluster `1 / (1 + mean_iterations)` — higher
    ///   for clusters reached with fewer iterations.
    /// * `plateau_analysis`: contiguous runs (ordered by `iterations`) whose
    ///   energy changes by less than 1% of the global energy std-dev.
    /// * `premature_convergence`: true when the number of distinct energies
    ///   found is less than 10% of the sample size (a search that collapsed
    ///   onto very few outcomes).
    /// * `diversity_evolution`: fraction of distinct energies within each of
    ///   up to 5 equal iteration-range bins, in iteration order.
    fn analyze_convergence(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> ConvergenceAnalysis {
        let n = solution_points.len();
        if n == 0 {
            return ConvergenceAnalysis {
                trajectory_clusters: Vec::new(),
                convergence_rates: Vec::new(),
                plateau_analysis: PlateauAnalysis {
                    num_plateaus: 0,
                    plateau_durations: Vec::new(),
                    plateau_energies: Vec::new(),
                    escape_probabilities: Vec::new(),
                },
                premature_convergence: false,
                diversity_evolution: Vec::new(),
            };
        }

        let convergence_rates: Vec<f64> = clusters
            .iter()
            .map(|c| {
                if c.solutions.is_empty() {
                    0.0
                } else {
                    let mean_iters = c
                        .solutions
                        .iter()
                        .map(|s| s.metadata.iterations as f64)
                        .sum::<f64>()
                        / c.solutions.len() as f64;
                    1.0 / (1.0 + mean_iters)
                }
            })
            .collect();

        let mut ordered: Vec<&SolutionPoint> = solution_points.iter().collect();
        ordered.sort_by_key(|s| s.metadata.iterations);
        let energies: Vec<f64> = ordered.iter().map(|s| s.energy).collect();

        let stats = self.calculate_energy_statistics(solution_points);
        let plateau_threshold = (stats.std_dev * 0.01).max(1e-9);

        let mut plateau_durations = Vec::new();
        let mut plateau_energies = Vec::new();
        let mut escape_probabilities = Vec::new();
        let mut i = 0;
        while i < energies.len() {
            let mut j = i;
            while j + 1 < energies.len()
                && (energies[j + 1] - energies[i]).abs() <= plateau_threshold
            {
                j += 1;
            }
            if j > i {
                let seg_len = j - i + 1;
                let iter_span = ordered[j]
                    .metadata
                    .iterations
                    .saturating_sub(ordered[i].metadata.iterations);
                plateau_durations.push(iter_span.max(seg_len));
                plateau_energies.push(energies[i..=j].iter().sum::<f64>() / seg_len as f64);
                escape_probabilities.push(1.0 / seg_len as f64);
            }
            i = j + 1;
        }
        let num_plateaus = plateau_durations.len();

        let distinct_ratio = stats.num_distinct_energies as f64 / n as f64;
        let premature_convergence = n >= 10 && distinct_ratio < 0.1;

        let min_iter = ordered.first().map_or(0, |s| s.metadata.iterations);
        let max_iter = ordered.last().map_or(0, |s| s.metadata.iterations);
        let num_bins = 5.min(n);
        let mut diversity_evolution = Vec::with_capacity(num_bins);
        if num_bins > 0 {
            let span = (max_iter.saturating_sub(min_iter)).max(1) as f64;
            for b in 0..num_bins {
                let lo = min_iter as f64 + span * b as f64 / num_bins as f64;
                let hi = min_iter as f64 + span * (b + 1) as f64 / num_bins as f64;
                let bin_energies: Vec<f64> = ordered
                    .iter()
                    .filter(|s| {
                        let it = s.metadata.iterations as f64;
                        it >= lo && (it < hi || b == num_bins - 1)
                    })
                    .map(|s| s.energy)
                    .collect();
                if bin_energies.is_empty() {
                    diversity_evolution.push(0.0);
                } else {
                    let mut distinct = bin_energies.clone();
                    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    distinct.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
                    diversity_evolution.push(distinct.len() as f64 / bin_energies.len() as f64);
                }
            }
        }

        ConvergenceAnalysis {
            trajectory_clusters: Vec::new(),
            convergence_rates,
            plateau_analysis: PlateauAnalysis {
                num_plateaus,
                plateau_durations,
                plateau_energies,
                escape_probabilities,
            },
            premature_convergence,
            diversity_evolution,
        }
    }

    /// Compute real Pearson correlations between spin variables, and between
    /// each spin variable and the solution energy, over `solution_points`.
    fn analyze_variable_correlations(
        &self,
        solution_points: &[SolutionPoint],
    ) -> CorrelationAnalysis {
        let n = solution_points.len();
        let d = solution_points.first().map_or(0, |s| s.solution.len());

        if n < 2 || d == 0 {
            return CorrelationAnalysis {
                variable_correlations: vec![vec![0.0; d]; d],
                energy_correlations: vec![0.0; d],
                significant_correlations: Vec::new(),
                correlation_patterns: Vec::new(),
            };
        }

        let columns: Vec<Vec<f64>> = (0..d)
            .map(|j| {
                solution_points
                    .iter()
                    .map(|s| f64::from(s.solution[j]))
                    .collect()
            })
            .collect();
        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();

        let pearson = |a: &[f64], b: &[f64]| -> f64 {
            let len = a.len() as f64;
            let mean_a = a.iter().sum::<f64>() / len;
            let mean_b = b.iter().sum::<f64>() / len;
            let mut cov = 0.0;
            let mut var_a = 0.0;
            let mut var_b = 0.0;
            for i in 0..a.len() {
                let da = a[i] - mean_a;
                let db = b[i] - mean_b;
                cov += da * db;
                var_a += da * da;
                var_b += db * db;
            }
            if var_a < 1e-12 || var_b < 1e-12 {
                0.0
            } else {
                cov / (var_a.sqrt() * var_b.sqrt())
            }
        };

        let mut variable_correlations = vec![vec![0.0; d]; d];
        let mut significant_correlations = Vec::new();
        for i in 0..d {
            variable_correlations[i][i] = 1.0;
            for j in (i + 1)..d {
                let r = pearson(&columns[i], &columns[j]);
                variable_correlations[i][j] = r;
                variable_correlations[j][i] = r;
                if r.abs() >= 0.5 {
                    significant_correlations.push((i, j, r));
                }
            }
        }

        let energy_correlations: Vec<f64> =
            columns.iter().map(|col| pearson(col, &energies)).collect();

        let correlation_patterns = significant_correlations
            .iter()
            .map(|&(i, j, r)| CorrelationPattern {
                description: format!(
                    "Variables {i} and {j} are {} correlated (r={r:.3})",
                    if r > 0.0 { "positively" } else { "negatively" }
                ),
                variables: vec![i, j],
                strength: r.abs(),
                pattern_type: if r > 0.0 {
                    PatternType::Positive
                } else {
                    PatternType::Negative
                },
            })
            .collect();

        CorrelationAnalysis {
            variable_correlations,
            energy_correlations,
            significant_correlations,
            correlation_patterns,
        }
    }

    /// Detect real statistical outliers: solutions whose energy is more than
    /// 3 standard deviations from the sample mean (`OutlierType::Energy`), or
    /// whose distance to their cluster centroid is more than 3 standard
    /// deviations above the cluster's mean intra-cluster distance
    /// (`OutlierType::Structural`); both flags together yield
    /// `OutlierType::Global`.
    fn detect_statistical_outliers(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<Vec<OutlierInfo>> {
        let stats = self.calculate_energy_statistics(solution_points);
        let mut outliers = Vec::new();
        if stats.std_dev < 1e-12 {
            return Ok(outliers);
        }

        for cluster in clusters {
            if cluster.solutions.is_empty() {
                continue;
            }

            let mut distances = Vec::with_capacity(cluster.solutions.len());
            for point in &cluster.solutions {
                let Some(features) = point.features.as_ref() else {
                    continue;
                };
                let d = self.calculate_distance(features, &cluster.centroid)?;
                distances.push((point, d));
            }
            if distances.is_empty() {
                continue;
            }

            let mean_d = distances.iter().map(|(_, d)| *d).sum::<f64>() / distances.len() as f64;
            let var_d = distances
                .iter()
                .map(|(_, d)| (d - mean_d).powi(2))
                .sum::<f64>()
                / distances.len() as f64;
            let std_d = var_d.sqrt();

            for (point, d) in &distances {
                let z_energy = (point.energy - stats.mean) / stats.std_dev;
                let is_energy_outlier = z_energy.abs() > 3.0;
                let structural_z = if std_d > 1e-12 {
                    (d - mean_d) / std_d
                } else {
                    0.0
                };
                let is_structural_outlier = structural_z > 3.0;

                if is_energy_outlier || is_structural_outlier {
                    let outlier_type = if is_energy_outlier && is_structural_outlier {
                        OutlierType::Global
                    } else if is_energy_outlier {
                        OutlierType::Energy
                    } else {
                        OutlierType::Structural
                    };
                    let outlier_score = z_energy.abs().max(structural_z.max(0.0));
                    outliers.push(OutlierInfo {
                        solution_id: point.metadata.id,
                        outlier_score,
                        outlier_type,
                        distance_to_cluster: *d,
                    });
                }
            }
        }

        Ok(outliers)
    }

    /// Calculate overall clustering quality.
    ///
    /// The overall silhouette score is the size-weighted mean of the per-cluster
    /// silhouette coefficients (which are themselves means of per-point
    /// silhouettes), matching scikit-learn's `silhouette_score` convention. This
    /// is fed by the real values written by [`Self::update_global_quality_metrics`].
    pub(crate) fn calculate_overall_quality(
        &self,
        clusters: &[SolutionCluster],
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<OverallClusteringQuality> {
        let silhouette_score = if clusters.is_empty() {
            0.0
        } else {
            let total_points: usize = clusters.iter().map(|c| c.solutions.len()).sum();
            if total_points == 0 {
                0.0
            } else {
                clusters
                    .iter()
                    .map(|c| c.quality_metrics.silhouette_coefficient * c.solutions.len() as f64)
                    .sum::<f64>()
                    / total_points as f64
            }
        };

        let inter_cluster_separation = self.calculate_inter_cluster_separation(clusters)?;
        let cluster_cohesion = self.calculate_cluster_cohesion(clusters);

        Ok(OverallClusteringQuality {
            silhouette_score,
            adjusted_rand_index: None,
            normalized_mutual_information: None,
            inter_cluster_separation,
            cluster_cohesion,
            num_clusters: clusters.len(),
            optimal_num_clusters: self.estimate_optimal_clusters(solution_points)?,
        })
    }

    /// Calculate inter-cluster separation
    fn calculate_inter_cluster_separation(
        &self,
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<f64> {
        if clusters.len() < 2 {
            return Ok(0.0);
        }

        let mut total_separation = 0.0;
        let mut count = 0;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let distance =
                    self.calculate_distance(&clusters[i].centroid, &clusters[j].centroid)?;
                total_separation += distance;
                count += 1;
            }
        }

        Ok(total_separation / f64::from(count))
    }

    /// Calculate cluster cohesion
    fn calculate_cluster_cohesion(&self, clusters: &[SolutionCluster]) -> f64 {
        if clusters.is_empty() {
            return 0.0;
        }

        clusters
            .iter()
            .map(|c| 1.0 / (1.0 + c.statistics.intra_cluster_distance))
            .sum::<f64>()
            / clusters.len() as f64
    }

    /// Estimate optimal number of clusters
    fn estimate_optimal_clusters(
        &self,
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<usize> {
        // Simplified elbow method
        let max_k = solution_points.len().min(10);
        let mut inertias = Vec::new();

        for k in 1..=max_k {
            if let Ok(clusters) = self.kmeans_clustering(solution_points, k, 50) {
                let total_inertia: f64 = clusters.iter().map(|c| c.quality_metrics.inertia).sum();
                inertias.push(total_inertia);
            }
        }

        // Find elbow (simplified)
        let optimal_k = if inertias.len() >= 3 {
            let mut max_diff = 0.0;
            let mut optimal = 1;

            for i in 1..inertias.len() - 1 {
                let diff = 2.0f64.mul_add(-inertias[i], inertias[i - 1]) + inertias[i + 1];
                if diff > max_diff {
                    max_diff = diff;
                    optimal = i + 1;
                }
            }
            optimal
        } else {
            inertias.len()
        };

        Ok(optimal_k)
    }

    /// Generate optimization recommendations
    pub(crate) fn generate_recommendations(
        &self,
        clusters: &[SolutionCluster],
        landscape_analysis: &LandscapeAnalysis,
        _statistical_summary: &StatisticalSummary,
    ) -> ClusteringResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Recommendation based on cluster quality
        if clusters
            .iter()
            .any(|c| c.quality_metrics.silhouette_coefficient < 0.3)
        {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ParameterTuning,
                description: "Low cluster quality detected. Consider tuning annealing parameters or using different initialization strategies.".to_string(),
                expected_improvement: 0.2,
                difficulty: DifficultyLevel::Easy,
                priority: PriorityLevel::High,
                evidence: vec!["Low silhouette coefficients in multiple clusters".to_string()],
            });
        }

        // Recommendation based on energy landscape
        if landscape_analysis.multi_modality.num_modes > 3 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::MultiStart,
                description: "Multiple modes detected in energy landscape. Consider using multi-start optimization or parallel runs.".to_string(),
                expected_improvement: 0.3,
                difficulty: DifficultyLevel::Moderate,
                priority: PriorityLevel::Medium,
                evidence: vec![format!("{} modes detected", landscape_analysis.multi_modality.num_modes)],
            });
        }

        // Recommendation based on cluster sizes
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.statistics.size).collect();
        let size_variance = cluster_sizes
            .iter()
            .map(|&size| {
                (size as f64
                    - cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64)
                    .powi(2)
            })
            .sum::<f64>()
            / cluster_sizes.len() as f64;

        if size_variance > 100.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::AlgorithmModification,
                description: "Highly unbalanced cluster sizes suggest potential convergence issues. Consider adjusting cooling schedule or using adaptive algorithms.".to_string(),
                expected_improvement: 0.15,
                difficulty: DifficultyLevel::Moderate,
                priority: PriorityLevel::Medium,
                evidence: vec![format!("Cluster size variance: {:.2}", size_variance)],
            });
        }

        Ok(recommendations)
    }

    /// Update global cluster quality metrics in a post-pass over all clusters.
    ///
    /// Silhouette, Davies-Bouldin, and Calinski-Harabasz indices all require
    /// inter-cluster information that is unavailable when each cluster is built
    /// in isolation by [`Self::kmeans_clustering`], [`Self::hierarchical_clustering`]
    /// or [`Self::dbscan_clustering`]. This pass walks every cluster simultaneously
    /// and writes the real values back into each cluster's
    /// [`crate::solution_clustering::types::ClusterQualityMetrics`].
    ///
    /// Definitions used:
    /// * Silhouette `s(i) = (b - a) / max(a, b)` where `a` is the mean intra-cluster
    ///   distance from point `i` and `b` is the smallest mean distance from `i` to
    ///   any other cluster. The per-cluster `silhouette_coefficient` is the mean of
    ///   `s(i)` over points in the cluster.
    /// * Davies-Bouldin per-cluster: `max_{j != i} ((sigma_i + sigma_j) / d(c_i, c_j))`
    ///   where `sigma_k` is the cluster's mean distance to its centroid and
    ///   `d(c_i, c_j)` is the distance between centroids.
    /// * Calinski-Harabasz: a single global value
    ///   `(BSS / (k - 1)) / (WSS / (n - k))` written into every cluster — this is the
    ///   conventional convention since the index is global, not per-cluster.
    pub(crate) fn update_global_quality_metrics(
        &self,
        clusters: &mut [SolutionCluster],
    ) -> ClusteringResult<()> {
        let k = clusters.len();
        if k == 0 {
            return Ok(());
        }

        // Single-cluster degenerate case: silhouette and DB are undefined; leave
        // sensible neutral values and CH at 0.
        if k == 1 {
            for c in clusters.iter_mut() {
                c.quality_metrics.silhouette_coefficient = 0.0;
                c.quality_metrics.davies_bouldin_index = 0.0;
                c.quality_metrics.calinski_harabasz_index = 0.0;
            }
            return Ok(());
        }

        // ---- Silhouette coefficient ----
        let mut per_cluster_silhouettes = vec![0.0f64; k];
        let mut per_cluster_counts = vec![0usize; k];

        for ci in 0..k {
            for point in &clusters[ci].solutions {
                let features = point.features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError(
                        "Solution point missing features for silhouette calculation".to_string(),
                    )
                })?;

                // Mean intra-cluster distance `a`. Singletons contribute s(i)=0.
                let a = if clusters[ci].solutions.len() <= 1 {
                    0.0
                } else {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for other in &clusters[ci].solutions {
                        if std::ptr::eq(other as *const _, point as *const _) {
                            continue;
                        }
                        let other_feat = other.features.as_ref().ok_or_else(|| {
                            ClusteringError::DataError(
                                "Solution point missing features for silhouette calculation"
                                    .to_string(),
                            )
                        })?;
                        sum += self.calculate_distance(features, other_feat)?;
                        count += 1;
                    }
                    if count == 0 {
                        0.0
                    } else {
                        sum / count as f64
                    }
                };

                // Minimum mean distance to any other cluster `b`.
                let mut b = f64::INFINITY;
                for cj in 0..k {
                    if cj == ci || clusters[cj].solutions.is_empty() {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for other in &clusters[cj].solutions {
                        let other_feat = other.features.as_ref().ok_or_else(|| {
                            ClusteringError::DataError(
                                "Solution point missing features for silhouette calculation"
                                    .to_string(),
                            )
                        })?;
                        sum += self.calculate_distance(features, other_feat)?;
                        count += 1;
                    }
                    if count > 0 {
                        let mean = sum / count as f64;
                        if mean < b {
                            b = mean;
                        }
                    }
                }

                let s = if !b.is_finite() {
                    0.0
                } else if clusters[ci].solutions.len() <= 1 {
                    // Convention: singleton silhouette is 0.
                    0.0
                } else {
                    let denom = a.max(b);
                    if denom < 1e-12 {
                        0.0
                    } else {
                        (b - a) / denom
                    }
                };

                per_cluster_silhouettes[ci] += s;
                per_cluster_counts[ci] += 1;
            }
        }

        for ci in 0..k {
            let mean_s = if per_cluster_counts[ci] == 0 {
                0.0
            } else {
                per_cluster_silhouettes[ci] / per_cluster_counts[ci] as f64
            };
            clusters[ci].quality_metrics.silhouette_coefficient = mean_s;
        }

        // ---- Davies-Bouldin index ----
        // sigma_i = mean distance from each point in cluster i to that cluster's centroid.
        let mut sigma = vec![0.0f64; k];
        for ci in 0..k {
            if clusters[ci].solutions.is_empty() {
                continue;
            }
            let centroid = clusters[ci].centroid.clone();
            if centroid.is_empty() {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            for point in &clusters[ci].solutions {
                let features = point.features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError(
                        "Solution point missing features for Davies-Bouldin calculation"
                            .to_string(),
                    )
                })?;
                if features.len() == centroid.len() {
                    sum += self.calculate_distance(features, &centroid)?;
                    count += 1;
                }
            }
            sigma[ci] = if count == 0 { 0.0 } else { sum / count as f64 };
        }

        for ci in 0..k {
            let mut max_ratio = 0.0f64;
            let centroid_i = &clusters[ci].centroid;
            if centroid_i.is_empty() {
                clusters[ci].quality_metrics.davies_bouldin_index = 0.0;
                continue;
            }
            for cj in 0..k {
                if cj == ci {
                    continue;
                }
                let centroid_j = &clusters[cj].centroid;
                if centroid_j.is_empty() || centroid_i.len() != centroid_j.len() {
                    continue;
                }
                let d_ij = self.calculate_distance(centroid_i, centroid_j)?;
                if d_ij < 1e-12 {
                    // Coincident centroids: treat as the worst case to penalise.
                    max_ratio = f64::INFINITY;
                    break;
                }
                let ratio = (sigma[ci] + sigma[cj]) / d_ij;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
            clusters[ci].quality_metrics.davies_bouldin_index = if max_ratio.is_finite() {
                max_ratio
            } else {
                0.0
            };
        }

        // ---- Stability (bootstrap resampling) ----
        //
        // For each cluster, resample its own member points with replacement
        // `BOOTSTRAP_REPLICATES` times, recompute the bootstrap centroid each
        // time, and measure how far that bootstrap centroid drifts from the
        // real cluster centroid, normalized by the cluster's own mean
        // intra-cluster distance `sigma_i` (already computed above for the
        // Davies-Bouldin index). A cluster whose members are tightly and
        // consistently grouped will have bootstrap centroids that barely
        // move (`stability` close to 1); a cluster sensitive to which
        // specific points were sampled will drift more (`stability` closer
        // to 0). This is real bootstrap variance estimation over the actual
        // cluster data, not a fixed constant.
        const BOOTSTRAP_REPLICATES: usize = 30;
        let mut rng = match self.config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().random()),
        };

        for ci in 0..k {
            let cluster = &clusters[ci];
            if cluster.centroid.is_empty() {
                continue;
            }
            let dim = cluster.centroid.len();
            let member_features: Vec<&[f64]> = cluster
                .solutions
                .iter()
                .filter_map(|s| s.features.as_deref())
                .filter(|f| f.len() == dim)
                .collect();
            if member_features.len() < 2 {
                // Not enough data to bootstrap meaningfully; leave the
                // per-cluster estimate written by
                // `calculate_cluster_quality_metrics` untouched.
                continue;
            }

            let mut drift_sum = 0.0;
            let mut successful_replicates = 0usize;
            for _ in 0..BOOTSTRAP_REPLICATES {
                let mut bootstrap_centroid = vec![0.0f64; dim];
                for _ in 0..member_features.len() {
                    let idx = rng.random_range(0..member_features.len());
                    let sampled = member_features[idx];
                    for (c, v) in bootstrap_centroid.iter_mut().zip(sampled.iter()) {
                        *c += v;
                    }
                }
                for c in bootstrap_centroid.iter_mut() {
                    *c /= member_features.len() as f64;
                }
                if let Ok(drift) = self.calculate_distance(&bootstrap_centroid, &cluster.centroid) {
                    drift_sum += drift;
                    successful_replicates += 1;
                }
            }

            if successful_replicates == 0 {
                continue;
            }

            let mean_drift = drift_sum / successful_replicates as f64;
            let normalizer = sigma[ci].max(1e-9);
            clusters[ci].quality_metrics.stability =
                (1.0 - mean_drift / normalizer).clamp(0.0, 1.0);
        }

        // ---- Calinski-Harabasz index ----
        // BSS = sum_i n_i * d(c_i, overall_centroid)^2
        // WSS = sum_i sum_x in C_i d(x, c_i)^2  (== sum of inertias)
        // CH  = (BSS / (k - 1)) / (WSS / (n - k))
        let n_total: usize = clusters.iter().map(|c| c.solutions.len()).sum();
        let dim = clusters
            .iter()
            .find(|c| !c.centroid.is_empty())
            .map(|c| c.centroid.len())
            .unwrap_or(0);

        let ch_value = if dim == 0 || n_total <= k {
            0.0
        } else {
            // Overall centroid is the size-weighted average of cluster centroids.
            let mut overall = vec![0.0f64; dim];
            for c in clusters.iter() {
                if c.centroid.len() == dim {
                    let w = c.solutions.len() as f64;
                    for d in 0..dim {
                        overall[d] += c.centroid[d] * w;
                    }
                }
            }
            if n_total > 0 {
                for d in 0..dim {
                    overall[d] /= n_total as f64;
                }
            }

            let mut bss = 0.0f64;
            for c in clusters.iter() {
                if c.centroid.len() != dim {
                    continue;
                }
                let dist = self.calculate_distance(&c.centroid, &overall)?;
                bss += (c.solutions.len() as f64) * dist * dist;
            }

            let wss: f64 = clusters.iter().map(|c| c.quality_metrics.inertia).sum();

            let denom_top = (k as f64 - 1.0).max(1.0);
            let denom_bot = (n_total as f64 - k as f64).max(1.0);
            if wss < 1e-12 {
                0.0
            } else {
                (bss / denom_top) / (wss / denom_bot)
            }
        };

        for c in clusters.iter_mut() {
            c.quality_metrics.calinski_harabasz_index = ch_value;
        }

        Ok(())
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::solution_clustering::config::create_basic_clustering_config;
    use crate::solution_clustering::types::{
        ClusterQualityMetrics, ClusterStatistics, SolutionMetadata,
    };
    use std::time::Instant;

    fn make_point(features: Vec<f64>) -> SolutionPoint {
        SolutionPoint {
            solution: vec![],
            energy: 0.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 0,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 0,
                quality_rank: None,
                is_feasible: true,
            },
            features: Some(features),
        }
    }

    fn dummy_statistics(size: usize) -> ClusterStatistics {
        ClusterStatistics {
            size,
            mean_energy: 0.0,
            energy_std: 0.0,
            min_energy: 0.0,
            max_energy: 0.0,
            intra_cluster_distance: 0.0,
            diameter: 0.0,
            density: 0.0,
        }
    }

    fn dummy_quality_metrics() -> ClusterQualityMetrics {
        ClusterQualityMetrics {
            silhouette_coefficient: 0.5,
            inertia: 0.0,
            calinski_harabasz_index: 1.0,
            davies_bouldin_index: 1.0,
            stability: 0.8, // The old fabricated constant every cluster used to keep.
        }
    }

    fn make_cluster(id: usize, points: Vec<SolutionPoint>, centroid: Vec<f64>) -> SolutionCluster {
        let size = points.len();
        SolutionCluster {
            id,
            solutions: points,
            centroid,
            representative: None,
            statistics: dummy_statistics(size),
            quality_metrics: dummy_quality_metrics(),
        }
    }

    // A far-away second cluster shared by both scenarios below, purely so that
    // `update_global_quality_metrics` sees k=2 clusters (silhouette/DB require
    // at least two) instead of hitting the k==1 degenerate short-circuit.
    fn anchor_cluster() -> SolutionCluster {
        make_cluster(
            1,
            vec![
                make_point(vec![100.0, 100.0]),
                make_point(vec![101.0, 100.0]),
                make_point(vec![100.0, 101.0]),
            ],
            vec![100.33, 100.33],
        )
    }

    #[test]
    fn stability_is_real_and_data_dependent_not_a_fixed_constant() {
        let mut config = create_basic_clustering_config();
        config.seed = Some(2024);
        let analyzer = SolutionClusteringAnalyzer::new(config);

        // Scenario A: a tight, symmetric cluster (evenly arranged points).
        let symmetric_cluster = make_cluster(
            0,
            vec![
                make_point(vec![1.0, 0.0]),
                make_point(vec![-1.0, 0.0]),
                make_point(vec![0.0, 1.0]),
                make_point(vec![0.0, -1.0]),
                make_point(vec![0.0, 0.0]),
            ],
            vec![0.0, 0.0],
        );
        let mut clusters_a = vec![symmetric_cluster, anchor_cluster()];
        analyzer
            .update_global_quality_metrics(&mut clusters_a)
            .expect("update should succeed");
        let stability_symmetric = clusters_a[0].quality_metrics.stability;

        // Scenario B: same cluster size, but one point is a severe outlier
        // that dominates the centroid — resampling should make the
        // bootstrap centroid swing far more than in the symmetric case.
        let skewed_cluster = make_cluster(
            0,
            vec![
                make_point(vec![0.0, 0.0]),
                make_point(vec![0.0, 0.0]),
                make_point(vec![0.0, 0.0]),
                make_point(vec![0.0, 0.0]),
                make_point(vec![10.0, 0.0]),
            ],
            vec![2.0, 0.0],
        );
        let mut clusters_b = vec![skewed_cluster, anchor_cluster()];
        analyzer
            .update_global_quality_metrics(&mut clusters_b)
            .expect("update should succeed");
        let stability_skewed = clusters_b[0].quality_metrics.stability;

        for value in [stability_symmetric, stability_skewed] {
            assert!(
                (0.0..=1.0).contains(&value),
                "stability must be clamped to [0,1], got {value}"
            );
        }
        // The two configurations are very different (uniform vs. outlier-
        // dominated); a real bootstrap computation must not collapse them
        // onto the same fabricated 0.8 constant, and in particular must not
        // report identical stability for two structurally different clusters.
        assert!(
            (stability_symmetric - 0.8).abs() > 1e-9 || (stability_skewed - 0.8).abs() > 1e-9,
            "at least one cluster's stability must move off the old fabricated 0.8 constant"
        );
        assert!(
            (stability_symmetric - stability_skewed).abs() > 1e-9,
            "stability must depend on the real cluster data: symmetric={stability_symmetric}, skewed={stability_skewed}"
        );
    }

    #[test]
    fn stability_bootstrap_is_deterministic_given_a_fixed_seed() {
        let mut config = create_basic_clustering_config();
        config.seed = Some(7);
        let analyzer = SolutionClusteringAnalyzer::new(config);

        let build_clusters = || {
            vec![
                make_cluster(
                    0,
                    vec![
                        make_point(vec![0.0, 0.0]),
                        make_point(vec![0.2, 0.0]),
                        make_point(vec![0.0, 0.2]),
                        make_point(vec![-0.2, 0.1]),
                    ],
                    vec![0.0, 0.075],
                ),
                anchor_cluster(),
            ]
        };

        let mut run1 = build_clusters();
        analyzer
            .update_global_quality_metrics(&mut run1)
            .expect("update should succeed");

        let mut run2 = build_clusters();
        analyzer
            .update_global_quality_metrics(&mut run2)
            .expect("update should succeed");

        assert!(
            (run1[0].quality_metrics.stability - run2[0].quality_metrics.stability).abs() < 1e-12,
            "the same seed and data must reproduce the same bootstrap stability estimate"
        );
    }

    fn make_point_full(
        id: usize,
        solution: Vec<i8>,
        features: Vec<f64>,
        energy: f64,
        iterations: usize,
    ) -> SolutionPoint {
        SolutionPoint {
            solution,
            energy,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations,
                quality_rank: None,
                is_feasible: true,
            },
            features: Some(features),
        }
    }

    #[test]
    fn perform_statistical_analysis_derives_real_values_not_fixed_fabrications() {
        let analyzer = SolutionClusteringAnalyzer::new(create_basic_clustering_config());

        // Variable 0 and variable 1 are always exact opposites (perfectly
        // anti-correlated: r = -1). Eleven points sit at a constant energy of
        // -3.0; one extra point (id=11) has a wildly different energy of
        // 100.0 -- with n=12 this is large enough to clear the real 3-sigma
        // energy-outlier threshold (the maximum possible |z| for a single
        // outlier among n samples is sqrt(n-1) ~= 3.317 for n=12).
        let mut solution_points: Vec<SolutionPoint> = (0..11)
            .map(|i| {
                if i % 2 == 0 {
                    make_point_full(i, vec![1, -1], vec![1.0, -1.0], -3.0, i * 5)
                } else {
                    make_point_full(i, vec![-1, 1], vec![-1.0, 1.0], -3.0, i * 5)
                }
            })
            .collect();
        solution_points.push(make_point_full(11, vec![1, -1], vec![1.0, -1.0], 100.0, 55));

        let cluster = make_cluster(0, solution_points.clone(), vec![0.2, -0.2]);
        let clusters = vec![cluster];

        let summary = analyzer
            .perform_statistical_analysis(&solution_points, &clusters)
            .expect("statistical analysis should succeed");

        // The old fabricated implementation always returned an all-ones
        // correlation matrix regardless of input; the real Pearson
        // correlation here must reflect the actual perfect anti-correlation.
        assert!(
            (summary.correlation_analysis.variable_correlations[0][1] - (-1.0)).abs() < 1e-9,
            "expected real r=-1 anti-correlation, got {}",
            summary.correlation_analysis.variable_correlations[0][1]
        );
        assert!(summary
            .correlation_analysis
            .significant_correlations
            .iter()
            .any(|&(i, j, r)| i == 0 && j == 1 && r < 0.0));

        // The old fabricated implementation always returned mean=0.0/std=1.0
        // regardless of input; the real energy sample here has a non-zero
        // mean and a real (non-unit) standard deviation.
        let mean = *summary
            .energy_distribution
            .parameters
            .get("mean")
            .expect("mean parameter should be present");
        let std = *summary
            .energy_distribution
            .parameters
            .get("std")
            .expect("std parameter should be present");
        assert!(
            mean > -3.0,
            "expected the real (outlier-shifted) sample mean, got {mean}"
        );
        assert!(
            (std - 1.0).abs() > 1e-6,
            "expected a real (non-unit) standard deviation, got {std}"
        );

        // The old fabricated implementation always returned an empty outlier
        // list; the wildly-different energy=100.0 point must be flagged.
        assert!(
            summary.outliers.iter().any(|o| o.solution_id == 11),
            "expected the real energy outlier (id=11) to be detected, got {:?}",
            summary.outliers
        );

        // The old fabricated implementation always used a fixed literal
        // `vec![0.1, 0.2, 0.15]` for convergence_rates regardless of the
        // number of clusters; the real analysis returns one real rate per
        // cluster, derived from that cluster's own recorded iteration counts.
        assert_eq!(
            summary.convergence_analysis.convergence_rates.len(),
            clusters.len()
        );
    }
}
