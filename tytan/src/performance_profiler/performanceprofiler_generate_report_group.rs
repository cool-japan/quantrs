//! # PerformanceProfiler - generate_report_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use super::types::{EventType, IOOperation, OutputFormat, Profile};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Generate report
    pub fn generate_report(
        &self,
        profile: &Profile,
        format: &OutputFormat,
    ) -> Result<String, String> {
        match format {
            OutputFormat::Json => Self::generate_json_report(profile),
            OutputFormat::Csv => Self::generate_csv_report(profile),
            OutputFormat::FlameGraph => Self::generate_flame_graph(profile),
            OutputFormat::ChromeTrace => Self::generate_chrome_trace(profile),
            OutputFormat::Binary => Err("Format not implemented".to_string()),
        }
    }
    /// Generate JSON report
    fn generate_json_report(profile: &Profile) -> Result<String, String> {
        use std::fmt::Write;
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str("  \"metadata\": {\n");
        writeln!(&mut json, "    \"id\": \"{}\",", profile.id).map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"start_time\": {},",
            profile.start_time.elapsed().as_millis()
        )
        .map_err(|e| e.to_string())?;
        if let Some(end_time) = profile.end_time {
            writeln!(
                &mut json,
                "    \"end_time\": {},",
                end_time.elapsed().as_millis()
            )
            .map_err(|e| e.to_string())?;
        }
        json.push_str("    \"duration_ms\": ");
        write!(
            &mut json,
            "{}",
            profile.metrics.time_metrics.total_time.as_millis()
        )
        .map_err(|e| e.to_string())?;
        json.push_str("\n  },\n");
        json.push_str("  \"time_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"total_time_ms\": {},",
            profile.metrics.time_metrics.total_time.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"qubo_generation_ms\": {},",
            profile
                .metrics
                .time_metrics
                .qubo_generation_time
                .as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"compilation_ms\": {},",
            profile.metrics.time_metrics.compilation_time.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"solving_ms\": {},",
            profile.metrics.time_metrics.solving_time.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"post_processing_ms\": {},",
            profile
                .metrics
                .time_metrics
                .post_processing_time
                .as_millis()
        )
        .map_err(|e| e.to_string())?;
        json.push_str("    \"function_times\": {\n");
        let func_entries: Vec<_> = profile.metrics.time_metrics.function_times.iter().collect();
        for (i, (func, time)) in func_entries.iter().enumerate() {
            write!(&mut json, "      \"{}\": {}", func, time.as_millis())
                .map_err(|e| e.to_string())?;
            if i < func_entries.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    },\n");
        json.push_str("    \"percentiles_ms\": {\n");
        writeln!(
            &mut json,
            "      \"p50\": {},",
            profile.metrics.time_metrics.percentiles.p50.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "      \"p90\": {},",
            profile.metrics.time_metrics.percentiles.p90.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "      \"p95\": {},",
            profile.metrics.time_metrics.percentiles.p95.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "      \"p99\": {},",
            profile.metrics.time_metrics.percentiles.p99.as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "      \"p999\": {}",
            profile.metrics.time_metrics.percentiles.p999.as_millis()
        )
        .map_err(|e| e.to_string())?;
        json.push_str("    }\n");
        json.push_str("  },\n");
        json.push_str("  \"memory_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"peak_memory_bytes\": {},",
            profile.metrics.memory_metrics.peak_memory
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"avg_memory_bytes\": {},",
            profile.metrics.memory_metrics.avg_memory
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"allocations\": {},",
            profile.metrics.memory_metrics.allocations
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"deallocations\": {},",
            profile.metrics.memory_metrics.deallocations
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"largest_allocation_bytes\": {}",
            profile.metrics.memory_metrics.largest_allocation
        )
        .map_err(|e| e.to_string())?;
        json.push_str("  },\n");
        json.push_str("  \"computation_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"flops\": {},",
            profile.metrics.computation_metrics.flops
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"memory_bandwidth_gbps\": {},",
            profile.metrics.computation_metrics.memory_bandwidth
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"cache_hit_rate\": {},",
            profile.metrics.computation_metrics.cache_hit_rate
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"branch_prediction_accuracy\": {},",
            profile
                .metrics
                .computation_metrics
                .branch_prediction_accuracy
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"vectorization_efficiency\": {}",
            profile.metrics.computation_metrics.vectorization_efficiency
        )
        .map_err(|e| e.to_string())?;
        json.push_str("  },\n");
        json.push_str("  \"quality_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"convergence_rate\": {},",
            profile.metrics.quality_metrics.convergence_rate
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"improvement_per_iteration\": {},",
            profile.metrics.quality_metrics.improvement_per_iteration
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"time_to_first_solution_ms\": {},",
            profile
                .metrics
                .quality_metrics
                .time_to_first_solution
                .as_millis()
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            &mut json,
            "    \"time_to_best_solution_ms\": {},",
            profile
                .metrics
                .quality_metrics
                .time_to_best_solution
                .as_millis()
        )
        .map_err(|e| e.to_string())?;
        json.push_str("    \"quality_timeline\": [\n");
        for (i, (time, quality)) in profile
            .metrics
            .quality_metrics
            .quality_timeline
            .iter()
            .enumerate()
        {
            json.push_str("      {\n");
            writeln!(&mut json, "        \"time_ms\": {},", time.as_millis())
                .map_err(|e| e.to_string())?;
            writeln!(&mut json, "        \"quality\": {quality}").map_err(|e| e.to_string())?;
            json.push_str("      }");
            if i < profile.metrics.quality_metrics.quality_timeline.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ]\n");
        json.push_str("  },\n");
        json.push_str("  \"call_graph\": {\n");
        json.push_str("    \"nodes\": [\n");
        for (i, node) in profile.call_graph.nodes.iter().enumerate() {
            json.push_str("      {\n");
            writeln!(&mut json, "        \"id\": {},", node.id).map_err(|e| e.to_string())?;
            writeln!(
                &mut json,
                "        \"name\": \"{}\",",
                node.name.replace('"', "\\\"")
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                &mut json,
                "        \"total_time_ms\": {},",
                node.total_time.as_millis()
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                &mut json,
                "        \"self_time_ms\": {},",
                node.self_time.as_millis()
            )
            .map_err(|e| e.to_string())?;
            writeln!(&mut json, "        \"call_count\": {},", node.call_count)
                .map_err(|e| e.to_string())?;
            writeln!(
                &mut json,
                "        \"avg_time_ms\": {}",
                node.avg_time.as_millis()
            )
            .map_err(|e| e.to_string())?;
            json.push_str("      }");
            if i < profile.call_graph.nodes.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ],\n");
        json.push_str("    \"edges\": [\n");
        for (i, edge) in profile.call_graph.edges.iter().enumerate() {
            json.push_str("      {\n");
            writeln!(&mut json, "        \"from\": {},", edge.from).map_err(|e| e.to_string())?;
            writeln!(&mut json, "        \"to\": {},", edge.to).map_err(|e| e.to_string())?;
            writeln!(&mut json, "        \"call_count\": {},", edge.call_count)
                .map_err(|e| e.to_string())?;
            writeln!(
                &mut json,
                "        \"total_time_ms\": {}",
                edge.total_time.as_millis()
            )
            .map_err(|e| e.to_string())?;
            json.push_str("      }");
            if i < profile.call_graph.edges.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ]\n");
        json.push_str("  },\n");
        json.push_str("  \"events_summary\": {\n");
        writeln!(&mut json, "    \"total_events\": {},", profile.events.len())
            .map_err(|e| e.to_string())?;
        let mut event_counts = std::collections::BTreeMap::new();
        for event in &profile.events {
            let type_name = match &event.event_type {
                EventType::FunctionCall => "function_call",
                EventType::FunctionReturn => "function_return",
                EventType::MemoryAlloc => "memory_alloc",
                EventType::MemoryFree => "memory_free",
                EventType::IOOperation => "io_operation",
                EventType::Synchronization => "synchronization",
                EventType::Custom(name) => name,
            };
            *event_counts.entry(type_name).or_insert(0) += 1;
        }
        json.push_str("    \"event_counts\": {\n");
        let count_entries: Vec<_> = event_counts.iter().collect();
        for (i, (event_type, count)) in count_entries.iter().enumerate() {
            write!(&mut json, "      \"{event_type}\": {count}").map_err(|e| e.to_string())?;
            if i < count_entries.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    }\n");
        json.push_str("  }\n");
        json.push_str("}\n");
        Ok(json)
    }
    /// Generate CSV report
    fn generate_csv_report(profile: &Profile) -> Result<String, String> {
        let mut csv = String::new();
        csv.push_str("function,total_time_ms,call_count,avg_time_ms\n");
        for node in &profile.call_graph.nodes {
            csv.push_str(&format!(
                "{},{},{},{}\n",
                node.name,
                node.total_time.as_millis(),
                node.call_count,
                node.avg_time.as_millis()
            ));
        }
        Ok(csv)
    }
    /// Generate flame graph
    fn generate_flame_graph(profile: &Profile) -> Result<String, String> {
        let mut stacks = Vec::new();
        for node in &profile.call_graph.nodes {
            let stack = vec![node.name.clone()];
            let value = node.self_time.as_micros() as usize;
            stacks.push((stack, value));
        }
        Ok(format!("Flame graph with {} stacks", stacks.len()))
    }
    /// Generate Chrome trace format
    fn generate_chrome_trace(profile: &Profile) -> Result<String, String> {
        #[derive(Serialize)]
        struct TraceEvent {
            name: String,
            cat: String,
            ph: String,
            ts: u64,
            dur: Option<u64>,
            pid: u32,
            tid: String,
        }
        let mut events = Vec::new();
        for event in &profile.events {
            let trace_event = TraceEvent {
                name: event.name.clone(),
                cat: "function".to_string(),
                ph: match event.event_type {
                    EventType::FunctionCall => "B".to_string(),
                    EventType::FunctionReturn => "E".to_string(),
                    _ => "i".to_string(),
                },
                ts: event.timestamp.elapsed().as_micros() as u64,
                dur: event.duration.map(|d| d.as_micros() as u64),
                pid: std::process::id(),
                tid: format!("{:?}", event.thread_id),
            };
            events.push(trace_event);
        }
        serde_json::to_string(&events).map_err(|e| format!("JSON serialization error: {e}"))
    }
}
