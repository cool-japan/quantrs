//! # PerformanceProfiler - stop_profile_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use super::types::{
    CallEdge, CallGraph, CallNode, EventType, Percentiles, Profile, ProfileContext,
};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Stop profiling
    pub fn stop_profile(&mut self) -> Result<Profile, String> {
        if !self.config.enabled {
            return Err("Profiling not enabled".to_string());
        }
        let mut context = self
            .current_profile
            .take()
            .ok_or("No profile in progress")?;
        context.profile.end_time = Some(Instant::now());
        context.profile.metrics.time_metrics.total_time = context
            .profile
            .end_time
            .map(|end| end - context.profile.start_time)
            .unwrap_or_default();
        Self::process_metrics_buffer(&mut context)?;
        Self::build_call_graph(&mut context.profile)?;
        Self::calculate_percentiles(&mut context.profile)?;
        self.profiles.push(context.profile.clone());
        Ok(context.profile)
    }
    /// Process metrics buffer
    fn process_metrics_buffer(context: &mut ProfileContext) -> Result<(), String> {
        if !context.metrics_buffer.memory_samples.is_empty() {
            let total_memory: usize = context
                .metrics_buffer
                .memory_samples
                .iter()
                .map(|(_, mem)| mem)
                .sum();
            context.profile.metrics.memory_metrics.avg_memory =
                total_memory / context.metrics_buffer.memory_samples.len();
            context.profile.metrics.memory_metrics.peak_memory = context
                .metrics_buffer
                .memory_samples
                .iter()
                .map(|(_, mem)| *mem)
                .max()
                .unwrap_or(0);
            context.profile.metrics.memory_metrics.memory_timeline =
                context.metrics_buffer.memory_samples.clone();
        }
        if !context.metrics_buffer.cpu_samples.is_empty() {
            context.profile.resource_usage.cpu_usage = context.metrics_buffer.cpu_samples.clone();
        }
        Ok(())
    }
    /// Build call graph
    fn build_call_graph(profile: &mut Profile) -> Result<(), String> {
        let mut node_map: HashMap<String, usize> = HashMap::new();
        let mut nodes = Vec::new();
        let mut edges: HashMap<(usize, usize), CallEdge> = HashMap::new();
        for (func_name, &total_time) in &profile.metrics.time_metrics.function_times {
            let node_id = nodes.len();
            node_map.insert(func_name.clone(), node_id);
            nodes.push(CallNode {
                id: node_id,
                name: func_name.clone(),
                total_time,
                self_time: total_time,
                call_count: 0,
                avg_time: Duration::from_secs(0),
            });
        }
        let mut call_stack: Vec<usize> = Vec::new();
        for event in &profile.events {
            match event.event_type {
                EventType::FunctionCall => {
                    if let Some(&node_id) = node_map.get(&event.name) {
                        nodes[node_id].call_count += 1;
                        if let Some(&parent_id) = call_stack.last() {
                            let edge_key = (parent_id, node_id);
                            edges
                                .entry(edge_key)
                                .and_modify(|e| e.call_count += 1)
                                .or_insert(CallEdge {
                                    from: parent_id,
                                    to: node_id,
                                    call_count: 1,
                                    total_time: Duration::from_secs(0),
                                });
                        }
                        call_stack.push(node_id);
                    }
                }
                EventType::FunctionReturn => {
                    call_stack.pop();
                }
                _ => {}
            }
        }
        for node in &mut nodes {
            if node.call_count > 0 {
                node.avg_time = node.total_time / node.call_count as u32;
            }
        }
        let mut has_parent = vec![false; nodes.len()];
        for edge in edges.values() {
            has_parent[edge.to] = true;
        }
        let roots: Vec<usize> = (0..nodes.len()).filter(|&i| !has_parent[i]).collect();
        profile.call_graph = CallGraph {
            nodes,
            edges: edges.into_values().collect(),
            roots,
        };
        Ok(())
    }
    /// Calculate percentiles
    fn calculate_percentiles(profile: &mut Profile) -> Result<(), String> {
        let mut durations: Vec<Duration> =
            profile.events.iter().filter_map(|e| e.duration).collect();
        if durations.is_empty() {
            return Ok(());
        }
        durations.sort();
        let len = durations.len();
        profile.metrics.time_metrics.percentiles = Percentiles {
            p50: durations[len * 50 / 100],
            p90: durations[len * 90 / 100],
            p95: durations[len * 95 / 100],
            p99: durations[len * 99 / 100],
            p999: durations[len.saturating_sub(1)],
        };
        Ok(())
    }
}
