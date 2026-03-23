//! # QAOAOptimizer - solve_classically_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::types::QAOAProblemType;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Solve problem classically for comparison
    pub(super) fn solve_classically(&self) -> Result<f64> {
        match self.problem_type {
            QAOAProblemType::MaxCut => self.solve_maxcut_classically(),
            QAOAProblemType::MaxWeightIndependentSet => self.solve_mwis_classically(),
            _ => self.solve_brute_force(),
        }
    }
    /// Solve MWIS classically (greedy)
    pub(super) fn solve_mwis_classically(&self) -> Result<f64> {
        let mut vertices: Vec<usize> = (0..self.graph.num_vertices).collect();
        vertices.sort_by(|&a, &b| {
            let weight_a = self.graph.vertex_weights.get(a).unwrap_or(&1.0);
            let weight_b = self.graph.vertex_weights.get(b).unwrap_or(&1.0);
            weight_b
                .partial_cmp(weight_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected = vec![false; self.graph.num_vertices];
        let mut total_weight = 0.0;
        for &v in &vertices {
            let mut can_select = true;
            for &u in &vertices {
                if selected[u] && self.graph.adjacency_matrix[[u, v]] > 0.0 {
                    can_select = false;
                    break;
                }
            }
            if can_select {
                selected[v] = true;
                total_weight += self.graph.vertex_weights.get(v).unwrap_or(&1.0);
            }
        }
        Ok(total_weight)
    }
}
