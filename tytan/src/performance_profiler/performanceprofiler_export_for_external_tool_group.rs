//! # PerformanceProfiler - export_for_external_tool_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::{EventType, ExternalTool, Profile};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Export profile for external analysis tools
    pub fn export_for_external_tool(
        &self,
        profile: &Profile,
        tool: ExternalTool,
    ) -> Result<String, String> {
        match tool {
            ExternalTool::Perf => Self::export_perf_script(profile),
            ExternalTool::Valgrind => Self::export_valgrind_format(profile),
            ExternalTool::FlameScope => Self::export_flamescope_format(profile),
            ExternalTool::SpeedScope => Self::export_speedscope_format(profile),
        }
    }
    /// Export in perf script format
    fn export_perf_script(profile: &Profile) -> Result<String, String> {
        let mut output = String::new();
        for event in &profile.events {
            if matches!(event.event_type, EventType::FunctionCall) {
                output.push_str(&format!(
                    "{} {} [{}] {}: {}\n",
                    "comm",
                    std::process::id(),
                    format!("{:?}", event.thread_id),
                    event.timestamp.elapsed().as_micros(),
                    event.name
                ));
            }
        }
        Ok(output)
    }
    /// Export in valgrind callgrind format
    fn export_valgrind_format(profile: &Profile) -> Result<String, String> {
        let mut output = String::new();
        output.push_str("events: Instructions\n");
        output.push_str("summary: 1000000\n\n");
        for node in &profile.call_graph.nodes {
            output.push_str(&format!(
                "fl={}\nfn={}\n1 {}\n\n",
                "unknown",
                node.name,
                node.total_time.as_micros()
            ));
        }
        Ok(output)
    }
    /// Export in FlameScope format
    fn export_flamescope_format(profile: &Profile) -> Result<String, String> {
        let mut stacks = Vec::new();
        for event in &profile.events {
            if matches!(event.event_type, EventType::FunctionCall) {
                if let Some(duration) = event.duration {
                    stacks.push(serde_json::json!(
                        { "name" : event.name, "value" : duration.as_micros(),
                        "start" : event.timestamp.elapsed().as_micros() }
                    ));
                }
            }
        }
        serde_json::to_string(&stacks).map_err(|e| format!("JSON error: {e}"))
    }
    /// Export in SpeedScope format
    fn export_speedscope_format(profile: &Profile) -> Result<String, String> {
        let speedscope_profile = serde_json::json!(
            { "$schema" : "https://www.speedscope.app/file-format-schema.json", "version"
            : "0.0.1", "shared" : { "frames" : profile.call_graph.nodes.iter().map(| node
            | { serde_json::json!({ "name" : node.name, "file" : "unknown", "line" : 0,
            "col" : 0 }) }).collect::< Vec < _ >> () }, "profiles" : [{ "type" :
            "evented", "name" : profile.id, "unit" : "microseconds", "startValue" : 0,
            "endValue" : profile.metrics.time_metrics.total_time.as_micros(), "events" :
            profile.events.iter().filter_map(| event | { match event.event_type {
            EventType::FunctionCall => Some(serde_json::json!({ "type" : "O", "at" :
            event.timestamp.elapsed().as_micros(), "frame" : event.name })),
            EventType::FunctionReturn => Some(serde_json::json!({ "type" : "C", "at" :
            event.timestamp.elapsed().as_micros(), "frame" : event.name })), _ => None }
            }).collect::< Vec < _ >> () }] }
        );
        serde_json::to_string(&speedscope_profile).map_err(|e| format!("JSON error: {e}"))
    }
}
