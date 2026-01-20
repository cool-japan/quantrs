//! Advanced Quantum Algorithm Profiling Utilities
//!
//! This module provides comprehensive profiling and performance analysis
//! for quantum algorithms, with detailed metrics and optimization recommendations.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::platform::PlatformCapabilities;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive quantum algorithm profiler
pub struct QuantumProfiler {
    /// Platform capabilities for context
    capabilities: PlatformCapabilities,
    /// Active profiling sessions
    sessions: HashMap<String, ProfilingSession>,
    /// Global statistics
    global_stats: GlobalStatistics,
}

impl Default for QuantumProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumProfiler {
    /// Create a new quantum profiler
    pub fn new() -> Self {
        Self {
            capabilities: PlatformCapabilities::detect(),
            sessions: HashMap::new(),
            global_stats: GlobalStatistics::default(),
        }
    }

    /// Start a new profiling session
    pub fn start_session(&mut self, name: impl Into<String>) -> ProfilingSessionHandle {
        let name = name.into();
        let session = ProfilingSession::new(name.clone());
        let handle = ProfilingSessionHandle {
            name: name.clone(),
            start_time: session.start_time,
        };
        self.sessions.insert(name, session);
        handle
    }

    /// End a profiling session and return results
    pub fn end_session(
        &mut self,
        handle: ProfilingSessionHandle,
    ) -> QuantRS2Result<ProfilingReport> {
        let session = self
            .sessions
            .remove(&handle.name)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Session not found".to_string()))?;

        let duration = handle.start_time.elapsed();

        // Update global statistics
        self.global_stats.total_sessions += 1;
        self.global_stats.total_duration += duration;

        // Generate recommendations before moving session data
        let recommendations = self.generate_recommendations(&session);

        Ok(ProfilingReport {
            session_name: handle.name,
            duration,
            metrics: session.metrics,
            gate_counts: session.gate_counts,
            memory_usage: session.memory_usage,
            recommendations,
        })
    }

    /// Record a gate operation
    pub fn record_gate(&mut self, session_name: &str, gate_type: &str) -> QuantRS2Result<()> {
        let session = self
            .sessions
            .get_mut(session_name)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Session not found".to_string()))?;

        *session
            .gate_counts
            .entry(gate_type.to_string())
            .or_insert(0) += 1;
        session.metrics.total_gates += 1;

        Ok(())
    }

    /// Record a measurement operation
    pub fn record_measurement(
        &mut self,
        session_name: &str,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let session = self
            .sessions
            .get_mut(session_name)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Session not found".to_string()))?;

        session.metrics.measurements += 1;
        session.metrics.qubits_measured = session.metrics.qubits_measured.max(num_qubits);

        Ok(())
    }

    /// Record memory usage
    pub fn record_memory(
        &mut self,
        session_name: &str,
        bytes: usize,
        phase: &str,
    ) -> QuantRS2Result<()> {
        let session = self
            .sessions
            .get_mut(session_name)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Session not found".to_string()))?;

        session.memory_usage.insert(phase.to_string(), bytes);
        session.metrics.peak_memory = session.metrics.peak_memory.max(bytes);

        Ok(())
    }

    /// Record circuit depth
    pub fn record_depth(&mut self, session_name: &str, depth: usize) -> QuantRS2Result<()> {
        let session = self
            .sessions
            .get_mut(session_name)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Session not found".to_string()))?;

        session.metrics.circuit_depth = session.metrics.circuit_depth.max(depth);

        Ok(())
    }

    /// Get global statistics
    pub const fn global_statistics(&self) -> &GlobalStatistics {
        &self.global_stats
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, session: &ProfilingSession) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check gate efficiency
        let two_qubit_gates = session
            .gate_counts
            .iter()
            .filter(|(name, _)| {
                name.contains("CNOT") || name.contains("CZ") || name.contains("SWAP")
            })
            .map(|(_, count)| count)
            .sum::<usize>();

        if two_qubit_gates > session.metrics.total_gates / 2 {
            recommendations.push(
                "High two-qubit gate count detected. Consider circuit optimization.".to_string(),
            );
        }

        // Check circuit depth
        if session.metrics.circuit_depth > 100 {
            recommendations.push(
                "Deep circuit detected. Consider gate fusion or compilation optimization."
                    .to_string(),
            );
        }

        // Check memory usage
        if session.metrics.peak_memory > 1024 * 1024 * 1024 {
            // > 1GB
            recommendations.push(
                "High memory usage detected. Consider tensor network or state vector compression."
                    .to_string(),
            );
        }

        // Platform-specific recommendations
        if !self.capabilities.has_simd() {
            recommendations.push(
                "SIMD not available. Performance may be limited for large state vectors."
                    .to_string(),
            );
        }

        if !self.capabilities.has_gpu() && session.metrics.total_gates > 1000 {
            recommendations.push(
                "GPU not available. Consider GPU acceleration for large circuits.".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Circuit is well-optimized for current platform.".to_string());
        }

        recommendations
    }
}

/// Handle for an active profiling session
#[derive(Debug, Clone)]
pub struct ProfilingSessionHandle {
    name: String,
    start_time: Instant,
}

/// Active profiling session
#[derive(Debug, Clone)]
struct ProfilingSession {
    name: String,
    start_time: Instant,
    metrics: SessionMetrics,
    gate_counts: HashMap<String, usize>,
    memory_usage: HashMap<String, usize>,
}

impl ProfilingSession {
    fn new(name: String) -> Self {
        Self {
            name,
            start_time: Instant::now(),
            metrics: SessionMetrics::default(),
            gate_counts: HashMap::new(),
            memory_usage: HashMap::new(),
        }
    }
}

/// Session-specific metrics
#[derive(Debug, Clone, Default)]
struct SessionMetrics {
    total_gates: usize,
    circuit_depth: usize,
    measurements: usize,
    qubits_measured: usize,
    peak_memory: usize,
}

/// Global profiling statistics
#[derive(Debug, Clone, Default)]
pub struct GlobalStatistics {
    /// Total number of sessions
    pub total_sessions: usize,
    /// Total cumulative duration
    pub total_duration: Duration,
}

/// Profiling report for a completed session
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Session name
    pub session_name: String,
    /// Session duration
    pub duration: Duration,
    /// Collected metrics
    pub metrics: SessionMetrics,
    /// Gate counts by type
    pub gate_counts: HashMap<String, usize>,
    /// Memory usage by phase
    pub memory_usage: HashMap<String, usize>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl ProfilingReport {
    /// Print a detailed report
    pub fn print_detailed(&self) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║          Quantum Algorithm Profiling Report                 ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!("Session: {}", self.session_name);
        println!("Duration: {:?}", self.duration);
        println!();
        println!("═══ Gate Statistics ═══");
        println!("  Total gates: {}", self.metrics.total_gates);
        println!("  Circuit depth: {}", self.metrics.circuit_depth);
        println!(
            "  Gates per layer: {:.2}",
            self.metrics.total_gates as f64 / self.metrics.circuit_depth.max(1) as f64
        );
        println!();

        if !self.gate_counts.is_empty() {
            println!("  Gate breakdown:");
            let mut sorted_gates: Vec<_> = self.gate_counts.iter().collect();
            sorted_gates.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
            for (gate, count) in sorted_gates.iter().take(10) {
                let percentage = (**count as f64 / self.metrics.total_gates as f64) * 100.0;
                println!("    {gate:<10} {count:>6}  ({percentage:>5.1}%)");
            }
            println!();
        }

        println!("═══ Measurement Statistics ═══");
        println!("  Total measurements: {}", self.metrics.measurements);
        println!("  Qubits measured: {}", self.metrics.qubits_measured);
        println!();

        println!("═══ Memory Statistics ═══");
        println!(
            "  Peak memory: {} MB",
            self.metrics.peak_memory / (1024 * 1024)
        );

        if !self.memory_usage.is_empty() {
            println!("  Memory by phase:");
            for (phase, bytes) in &self.memory_usage {
                println!("    {:<20} {:>8} MB", phase, bytes / (1024 * 1024));
            }
            println!();
        }

        println!("═══ Performance Metrics ═══");
        let gates_per_sec = self.metrics.total_gates as f64 / self.duration.as_secs_f64();
        println!("  Gates per second: {gates_per_sec:.2}");
        println!(
            "  Average gate time: {:.2} µs",
            self.duration.as_micros() as f64 / self.metrics.total_gates.max(1) as f64
        );
        println!();

        println!("═══ Recommendations ═══");
        for (i, rec) in self.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
        println!("╚══════════════════════════════════════════════════════════════╝");
    }

    /// Export to JSON format
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "session_name": "{}",
  "duration_ms": {},
  "metrics": {{
    "total_gates": {},
    "circuit_depth": {},
    "measurements": {},
    "qubits_measured": {},
    "peak_memory": {}
  }},
  "gate_counts": {:?},
  "memory_usage": {:?},
  "recommendations": {:?}
}}"#,
            self.session_name,
            self.duration.as_millis(),
            self.metrics.total_gates,
            self.metrics.circuit_depth,
            self.metrics.measurements,
            self.metrics.qubits_measured,
            self.metrics.peak_memory,
            self.gate_counts,
            self.memory_usage,
            self.recommendations
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = QuantumProfiler::new();
        assert_eq!(profiler.global_stats.total_sessions, 0);
    }

    #[test]
    fn test_session_lifecycle() {
        let mut profiler = QuantumProfiler::new();

        let handle = profiler.start_session("test_circuit");
        assert_eq!(handle.name, "test_circuit");

        let report = profiler.end_session(handle).expect("failed to end session");
        assert_eq!(report.session_name, "test_circuit");
        assert_eq!(profiler.global_stats.total_sessions, 1);
    }

    #[test]
    fn test_gate_recording() {
        let mut profiler = QuantumProfiler::new();
        let handle = profiler.start_session("test");

        profiler
            .record_gate("test", "H")
            .expect("failed to record H gate");
        profiler
            .record_gate("test", "CNOT")
            .expect("failed to record CNOT gate");
        profiler
            .record_gate("test", "H")
            .expect("failed to record H gate");

        let report = profiler.end_session(handle).expect("failed to end session");
        assert_eq!(report.metrics.total_gates, 3);
        assert_eq!(
            *report.gate_counts.get("H").expect("H gate count not found"),
            2
        );
        assert_eq!(
            *report
                .gate_counts
                .get("CNOT")
                .expect("CNOT gate count not found"),
            1
        );
    }

    #[test]
    fn test_measurement_recording() {
        let mut profiler = QuantumProfiler::new();
        let handle = profiler.start_session("test");

        profiler
            .record_measurement("test", 5)
            .expect("failed to record measurement");
        profiler
            .record_measurement("test", 3)
            .expect("failed to record measurement");

        let report = profiler.end_session(handle).expect("failed to end session");
        assert_eq!(report.metrics.measurements, 2);
        assert_eq!(report.metrics.qubits_measured, 5);
    }

    #[test]
    fn test_memory_recording() {
        let mut profiler = QuantumProfiler::new();
        let handle = profiler.start_session("test");

        profiler
            .record_memory("test", 1024 * 1024, "initialization")
            .expect("failed to record memory");
        profiler
            .record_memory("test", 2 * 1024 * 1024, "computation")
            .expect("failed to record memory");

        let report = profiler.end_session(handle).expect("failed to end session");
        assert_eq!(report.metrics.peak_memory, 2 * 1024 * 1024);
        assert_eq!(report.memory_usage.len(), 2);
    }

    #[test]
    fn test_recommendations() {
        let mut profiler = QuantumProfiler::new();
        let handle = profiler.start_session("deep_circuit");

        // Record many gates to trigger deep circuit warning
        for _ in 0..150 {
            profiler
                .record_gate("deep_circuit", "H")
                .expect("failed to record gate");
        }
        profiler
            .record_depth("deep_circuit", 150)
            .expect("failed to record depth");

        let report = profiler.end_session(handle).expect("failed to end session");
        assert!(!report.recommendations.is_empty());
        assert!(report
            .recommendations
            .iter()
            .any(|r| r.contains("deep circuit") || r.contains("Deep circuit")));
    }

    #[test]
    fn test_report_formatting() {
        let mut profiler = QuantumProfiler::new();
        let handle = profiler.start_session("test");

        profiler
            .record_gate("test", "H")
            .expect("failed to record gate");
        profiler
            .record_measurement("test", 1)
            .expect("failed to record measurement");

        let report = profiler.end_session(handle).expect("failed to end session");

        // Test that print_detailed doesn't panic
        report.print_detailed();

        // Test JSON export
        let json = report.to_json();
        assert!(json.contains("session_name"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_global_statistics() {
        let mut profiler = QuantumProfiler::new();

        for i in 0..5 {
            let handle = profiler.start_session(format!("session_{}", i));
            profiler.end_session(handle).expect("failed to end session");
        }

        let stats = profiler.global_statistics();
        assert_eq!(stats.total_sessions, 5);
        assert!(stats.total_duration.as_nanos() > 0);
    }
}
