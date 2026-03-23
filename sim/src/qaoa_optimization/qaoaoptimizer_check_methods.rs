//! # QAOAOptimizer - check_methods Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::types::{QAOAConstraint, QAOAProblemType};

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    pub(super) fn check_feasibility(&self, solution: &str) -> Result<bool> {
        let bits: Vec<bool> = solution.chars().map(|c| c == '1').collect();
        match self.problem_type {
            QAOAProblemType::MaxWeightIndependentSet => {
                for i in 0..self.graph.num_vertices {
                    if bits[i] {
                        for j in 0..self.graph.num_vertices {
                            if i != j && bits[j] && self.graph.adjacency_matrix[[i, j]] > 0.0 {
                                return Ok(false);
                            }
                        }
                    }
                }
            }
            QAOAProblemType::TSP => {
                let num_cities = (self.graph.num_vertices as f64).sqrt() as usize;
                let mut city_counts = vec![0; num_cities];
                let mut time_counts = vec![0; num_cities];
                for city in 0..num_cities {
                    for time in 0..num_cities {
                        let qubit = city * num_cities + time;
                        if qubit < bits.len() && bits[qubit] {
                            city_counts[city] += 1;
                            time_counts[time] += 1;
                        }
                    }
                }
                if !city_counts.iter().all(|&count| count == 1)
                    || !time_counts.iter().all(|&count| count == 1)
                {
                    return Ok(false);
                }
            }
            _ => {}
        }
        for constraint in &self.graph.constraints {
            match constraint {
                QAOAConstraint::Cardinality { target } => {
                    let count = bits.iter().filter(|&&b| b).count();
                    if count != *target {
                        return Ok(false);
                    }
                }
                QAOAConstraint::UpperBound { max_vertices } => {
                    let count = bits.iter().filter(|&&b| b).count();
                    if count > *max_vertices {
                        return Ok(false);
                    }
                }
                QAOAConstraint::LowerBound { min_vertices } => {
                    let count = bits.iter().filter(|&&b| b).count();
                    if count < *min_vertices {
                        return Ok(false);
                    }
                }
                QAOAConstraint::Parity { even } => {
                    let count = bits.iter().filter(|&&b| b).count();
                    if (count % 2 == 0) != *even {
                        return Ok(false);
                    }
                }
                QAOAConstraint::LinearConstraint {
                    coefficients,
                    bound,
                } => {
                    let mut sum = 0.0;
                    for (i, &coeff) in coefficients.iter().enumerate() {
                        if i < bits.len() && bits[i] {
                            sum += coeff;
                        }
                    }
                    if (sum - bound).abs() > 1e-10 {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }
}
