//! # QAOAOptimizer - evaluate_classical_cost_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::{QAOAProblemType, SolutionQuality};

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Evaluate classical cost for a bitstring
    pub(super) fn evaluate_classical_cost(&self, bitstring: &str) -> Result<f64> {
        let bits: Vec<bool> = bitstring.chars().map(|c| c == '1').collect();
        match self.problem_type {
            QAOAProblemType::MaxCut => self.evaluate_maxcut_cost(&bits),
            QAOAProblemType::MaxWeightIndependentSet => self.evaluate_mwis_cost(&bits),
            QAOAProblemType::TSP => self.evaluate_tsp_cost(&bits),
            QAOAProblemType::PortfolioOptimization => self.evaluate_portfolio_cost(&bits),
            QAOAProblemType::QUBO => self.evaluate_qubo_cost(&bits),
            _ => self.evaluate_generic_cost(&bits),
        }
    }
    /// Evaluate `MaxCut` cost
    pub(super) fn evaluate_maxcut_cost(&self, bits: &[bool]) -> Result<f64> {
        let mut cost = 0.0;
        for i in 0..self.graph.num_vertices {
            for j in i + 1..self.graph.num_vertices {
                let weight = self
                    .graph
                    .edge_weights
                    .get(&(i, j))
                    .or_else(|| self.graph.edge_weights.get(&(j, i)))
                    .unwrap_or(&self.graph.adjacency_matrix[[i, j]]);
                if weight.abs() > 1e-10 && bits[i] != bits[j] {
                    cost += weight;
                }
            }
        }
        Ok(cost)
    }
    /// Evaluate MWIS cost
    pub(super) fn evaluate_mwis_cost(&self, bits: &[bool]) -> Result<f64> {
        let mut cost = 0.0;
        let mut valid = true;
        for i in 0..self.graph.num_vertices {
            if bits[i] {
                for j in 0..self.graph.num_vertices {
                    if i != j && bits[j] && self.graph.adjacency_matrix[[i, j]] > 0.0 {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    cost += self.graph.vertex_weights.get(i).unwrap_or(&1.0);
                }
            }
        }
        if !valid {
            cost = -1000.0;
        }
        Ok(cost)
    }
    /// Evaluate TSP cost
    pub(super) fn evaluate_tsp_cost(&self, bits: &[bool]) -> Result<f64> {
        let num_cities = (self.graph.num_vertices as f64).sqrt() as usize;
        let mut cost = 0.0;
        let mut valid = true;
        let mut city_times = vec![-1i32; num_cities];
        let mut time_cities = vec![-1i32; num_cities];
        for city in 0..num_cities {
            for time in 0..num_cities {
                let qubit = city * num_cities + time;
                if qubit < bits.len() && bits[qubit] {
                    if city_times[city] != -1 || time_cities[time] != -1 {
                        valid = false;
                        break;
                    }
                    city_times[city] = time as i32;
                    time_cities[time] = city as i32;
                }
            }
            if !valid {
                break;
            }
        }
        if valid && city_times.iter().all(|&t| t != -1) {
            for t in 0..num_cities {
                let current_city = time_cities[t] as usize;
                let next_city = time_cities[(t + 1) % num_cities] as usize;
                cost += self.graph.adjacency_matrix[[current_city, next_city]];
            }
        } else {
            cost = 1000.0;
        }
        Ok(cost)
    }
    /// Evaluate portfolio cost
    pub(super) fn evaluate_portfolio_cost(&self, bits: &[bool]) -> Result<f64> {
        let mut expected_return = 0.0;
        let mut risk = 0.0;
        let lambda = 1.0;
        for i in 0..self.graph.num_vertices {
            if bits[i] {
                expected_return += self.graph.vertex_weights.get(i).unwrap_or(&0.1);
            }
        }
        for i in 0..self.graph.num_vertices {
            for j in 0..self.graph.num_vertices {
                if bits[i] && bits[j] {
                    risk += self.graph.adjacency_matrix[[i, j]];
                }
            }
        }
        Ok(expected_return - lambda * risk)
    }
    /// Evaluate QUBO cost
    pub(super) fn evaluate_qubo_cost(&self, bits: &[bool]) -> Result<f64> {
        let mut cost = 0.0;
        for i in 0..self.graph.num_vertices {
            if bits[i] {
                cost += self.graph.adjacency_matrix[[i, i]];
            }
        }
        for i in 0..self.graph.num_vertices {
            for j in i + 1..self.graph.num_vertices {
                if bits[i] && bits[j] {
                    cost += self.graph.adjacency_matrix[[i, j]];
                }
            }
        }
        Ok(cost)
    }
    /// Evaluate generic cost
    pub(super) fn evaluate_generic_cost(&self, bits: &[bool]) -> Result<f64> {
        self.evaluate_maxcut_cost(bits)
    }
    /// Solve `MaxCut` classically (greedy approximation)
    pub(super) fn solve_maxcut_classically(&self) -> Result<f64> {
        let mut best_cost = 0.0;
        let num_vertices = self.graph.num_vertices;
        let mut assignment = vec![false; num_vertices];
        for _ in 0..10 {
            for i in 0..num_vertices {
                assignment[i] = thread_rng().random::<bool>();
            }
            let mut improved = true;
            while improved {
                improved = false;
                for i in 0..num_vertices {
                    assignment[i] = !assignment[i];
                    let cost = self.evaluate_classical_cost(
                        &assignment
                            .iter()
                            .map(|&b| if b { '1' } else { '0' })
                            .collect::<String>(),
                    )?;
                    if cost > best_cost {
                        best_cost = cost;
                        improved = true;
                    } else {
                        assignment[i] = !assignment[i];
                    }
                }
            }
        }
        Ok(best_cost)
    }
    /// Brute force solver for small problems
    pub(super) fn solve_brute_force(&self) -> Result<f64> {
        if self.graph.num_vertices > 20 {
            return Ok(0.0);
        }
        let mut best_cost = f64::NEG_INFINITY;
        let num_states = 1 << self.graph.num_vertices;
        for state in 0..num_states {
            let bitstring = format!("{:0width$b}", state, width = self.graph.num_vertices);
            let cost = self.evaluate_classical_cost(&bitstring)?;
            if cost > best_cost {
                best_cost = cost;
            }
        }
        Ok(best_cost)
    }
    pub(super) fn evaluate_solution_quality(
        &self,
        solution: &str,
        _probabilities: &HashMap<String, f64>,
    ) -> Result<SolutionQuality> {
        let cost = self.evaluate_classical_cost(solution)?;
        let feasible = self.check_feasibility(solution)?;
        let optimality_gap = self
            .classical_optimum
            .map(|classical_opt| (classical_opt - cost) / classical_opt);
        Ok(SolutionQuality {
            feasible,
            optimality_gap,
            solution_variance: 0.0,
            confidence: 0.9,
            constraint_violations: usize::from(!feasible),
        })
    }
}
