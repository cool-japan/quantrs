//! Energy analysis functionality for the solution debugger.

use ndarray::Array2;
use serde::Serialize;
use std::collections::HashMap;

/// Energy analyzer
pub struct EnergyAnalyzer {
    /// Energy breakdown cache
    energy_cache: HashMap<String, EnergyBreakdown>,
    /// Analysis depth
    analysis_depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnergyBreakdown {
    /// Total energy
    pub total_energy: f64,
    /// Linear terms contribution
    pub linear_terms: f64,
    /// Quadratic terms contribution
    pub quadratic_terms: f64,
    /// Constraint penalties
    pub constraint_penalties: f64,
    /// Per-variable contribution
    pub variable_contributions: HashMap<String, f64>,
    /// Per-interaction contribution
    pub interaction_contributions: HashMap<(String, String), f64>,
    /// Energy landscape
    pub energy_landscape: EnergyLandscape,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnergyLandscape {
    /// Local minima nearby
    pub local_minima: Vec<LocalMinimum>,
    /// Energy barriers
    pub barriers: Vec<EnergyBarrier>,
    /// Basin of attraction
    pub basin_size: usize,
    /// Ruggedness measure
    pub ruggedness: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalMinimum {
    /// Solution
    pub solution: HashMap<String, bool>,
    /// Energy
    pub energy: f64,
    /// Distance from current
    pub distance: usize,
    /// Escape barrier
    pub escape_barrier: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnergyBarrier {
    /// From solution
    pub from: HashMap<String, bool>,
    /// To solution
    pub to: HashMap<String, bool>,
    /// Barrier height
    pub height: f64,
    /// Transition path
    pub path: Vec<HashMap<String, bool>>,
}

impl EnergyAnalyzer {
    /// Create new energy analyzer
    pub fn new(analysis_depth: usize) -> Self {
        Self {
            energy_cache: HashMap::new(),
            analysis_depth,
        }
    }

    /// Analyze energy breakdown for a solution
    pub fn analyze(
        &mut self,
        qubo: &Array2<f64>,
        assignments: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
    ) -> EnergyBreakdown {
        let solution_key = self.solution_key(assignments);

        if let Some(cached) = self.energy_cache.get(&solution_key) {
            return cached.clone();
        }

        let breakdown = self.compute_energy_breakdown(qubo, assignments, var_map);
        self.energy_cache.insert(solution_key, breakdown.clone());
        breakdown
    }

    /// Compute energy breakdown
    fn compute_energy_breakdown(
        &self,
        qubo: &Array2<f64>,
        assignments: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
    ) -> EnergyBreakdown {
        let total_energy = 0.0;
        let linear_terms = 0.0;
        let quadratic_terms = 0.0;
        let variable_contributions = HashMap::new();
        let interaction_contributions = HashMap::new();

        // Calculate energy contributions
        for (var1, &idx1) in var_map {
            let val1 = if assignments.get(var1).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            };

            // Linear terms (diagonal)
            if idx1 < qubo.nrows() && idx1 < qubo.ncols() {
                let linear_contrib = qubo[[idx1, idx1]] * val1;
                linear_terms += linear_contrib;
                total_energy += linear_contrib;
                *variable_contributions.entry(var1.clone()).or_insert(0.0) += linear_contrib;
            }

            // Quadratic terms (off-diagonal)
            for (var2, &idx2) in var_map {
                if idx1 < idx2 && idx1 < qubo.nrows() && idx2 < qubo.ncols() {
                    let val2 = if assignments.get(var2).copied().unwrap_or(false) {
                        1.0
                    } else {
                        0.0
                    };
                    let quad_contrib = qubo[[idx1, idx2]] * val1 * val2;
                    quadratic_terms += quad_contrib;
                    total_energy += quad_contrib;

                    // Record interaction contribution
                    let key = if var1 < var2 {
                        (var1.clone(), var2.clone())
                    } else {
                        (var2.clone(), var1.clone())
                    };
                    interaction_contributions.insert(key, quad_contrib);

                    // Add to variable contributions
                    *variable_contributions.entry(var1.clone()).or_insert(0.0) +=
                        quad_contrib / 2.0;
                    *variable_contributions.entry(var2.clone()).or_insert(0.0) +=
                        quad_contrib / 2.0;
                }
            }
        }

        // Analyze energy landscape
        let energy_landscape = self.analyze_energy_landscape(qubo, assignments, var_map);

        EnergyBreakdown {
            total_energy,
            linear_terms,
            quadratic_terms,
            constraint_penalties: 0.0, // Would need constraint penalty information
            variable_contributions,
            interaction_contributions,
            energy_landscape,
        }
    }

    /// Analyze energy landscape around current solution
    fn analyze_energy_landscape(
        &self,
        qubo: &Array2<f64>,
        assignments: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
    ) -> EnergyLandscape {
        let local_minima = Vec::new();
        let current_energy = self.calculate_energy(qubo, assignments, var_map);

        // Explore 1-flip neighborhood
        for var in assignments.keys() {
            let neighbor = assignments.clone();
            neighbor.insert(var.clone(), !assignments[var]);

            let neighbor_energy = self.calculate_energy(qubo, &neighbor, var_map);
            if neighbor_energy < current_energy {
                local_minima.push(LocalMinimum {
                    solution: neighbor,
                    energy: neighbor_energy,
                    distance: 1,
                    escape_barrier: (current_energy - neighbor_energy).max(0.0),
                });
            }
        }

        // If analysis depth > 1, explore 2-flip neighborhood
        if self.analysis_depth > 1 {
            let vars: Vec<_> = assignments.keys().collect();
            for i in 0..vars.len() {
                for j in i + 1..vars.len() {
                    let neighbor = assignments.clone();
                    neighbor.insert(vars[i].clone(), !assignments[vars[i]]);
                    neighbor.insert(vars[j].clone(), !assignments[vars[j]]);

                    let neighbor_energy = self.calculate_energy(qubo, &neighbor, var_map);
                    if neighbor_energy < current_energy {
                        local_minima.push(LocalMinimum {
                            solution: neighbor,
                            energy: neighbor_energy,
                            distance: 2,
                            escape_barrier: (current_energy - neighbor_energy).max(0.0),
                        });
                    }
                }
            }
        }

        // Sort by energy
        local_minima.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        local_minima.truncate(10); // Keep top 10 local minima

        // Calculate ruggedness (simplified measure)
        let energy_variance = if local_minima.len() > 1 {
            let energies: Vec<f64> = local_minima.iter().map(|m| m.energy).collect();
            let mean = energies.iter().sum::<f64>() / energies.len() as f64;
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / energies.len() as f64
        } else {
            0.0
        };

        EnergyLandscape {
            local_minima,
            barriers: Vec::new(), // Would need more sophisticated analysis
            basin_size: 1,        // Simplified
            ruggedness: energy_variance.sqrt(),
        }
    }

    /// Calculate energy for a solution
    fn calculate_energy(
        &self,
        qubo: &Array2<f64>,
        assignments: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
    ) -> f64 {
        let energy = 0.0;

        for (var1, &idx1) in var_map {
            let val1 = if assignments.get(var1).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            };

            for (var2, &idx2) in var_map {
                if idx1 < qubo.nrows() && idx2 < qubo.ncols() {
                    let val2 = if assignments.get(var2).copied().unwrap_or(false) {
                        1.0
                    } else {
                        0.0
                    };
                    energy += qubo[[idx1, idx2]] * val1 * val2;
                }
            }
        }

        energy
    }

    /// Generate solution key for caching
    fn solution_key(&self, assignments: &HashMap<String, bool>) -> String {
        let mut vars: Vec<_> = assignments.iter().collect();
        vars.sort_by_key(|(name, _)| *name);
        vars.iter()
            .map(|(name, &val)| format!("{}:{}", name, if val { "1" } else { "0" }))
            .collect::<Vec<_>>()
            .join(",")
    }
}
