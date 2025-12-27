//! Quantum Computing Profiling Integration with SciRS2 Beta.1
//!
//! This module provides comprehensive profiling capabilities for quantum
//! computations using the advanced profiling features in scirs2-core beta.1.

use crate::error::QuantRS2Result;
use scirs2_core::profiling::{MemoryTracker, Profiler, Timer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use std::fmt::Write;
/// Quantum operation profiling data
#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantumOperationProfile {
    pub operation_name: String,
    pub execution_count: u64,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub memory_usage: u64,
    pub gate_count: u64,
}

/// Comprehensive quantum profiler
pub struct QuantumProfiler {
    /// Operation profiles
    profiles: Arc<Mutex<HashMap<String, QuantumOperationProfile>>>,
    /// Active timers
    active_timers: Arc<Mutex<HashMap<String, Instant>>>,
    /// Global profiling enabled flag
    enabled: Arc<Mutex<bool>>,
    /// Memory tracking
    memory_tracker: Arc<Mutex<Option<MemoryTracker>>>,
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
            profiles: Arc::new(Mutex::new(HashMap::new())),
            active_timers: Arc::new(Mutex::new(HashMap::new())),
            enabled: Arc::new(Mutex::new(false)),
            memory_tracker: Arc::new(Mutex::new(None)),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        *self.enabled.lock().expect("Profiler enabled lock poisoned") = true;

        // Start global profiler
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.start();
        }
    }

    /// Disable profiling
    pub fn disable(&self) {
        *self.enabled.lock().expect("Profiler enabled lock poisoned") = false;

        // Stop global profiler
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.stop();
        }
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().expect("Profiler enabled lock poisoned")
    }

    /// Start profiling a quantum operation
    pub fn start_operation(&self, operation_name: &str) {
        if !self.is_enabled() {
            return;
        }

        let mut timers = self
            .active_timers
            .lock()
            .expect("Active timers lock poisoned");
        timers.insert(operation_name.to_string(), Instant::now());

        // Start memory tracking
        let mut tracker = self
            .memory_tracker
            .lock()
            .expect("Memory tracker lock poisoned");
        *tracker = Some(MemoryTracker::start(operation_name));
    }

    /// End profiling a quantum operation
    pub fn end_operation(&self, operation_name: &str, gate_count: u64) {
        if !self.is_enabled() {
            return;
        }

        let start_time = {
            let mut timers = self
                .active_timers
                .lock()
                .expect("Active timers lock poisoned");
            timers.remove(operation_name)
        };

        if let Some(start) = start_time {
            let execution_time = start.elapsed();

            // Stop memory tracking
            let memory_usage = {
                let mut tracker = self
                    .memory_tracker
                    .lock()
                    .expect("Memory tracker lock poisoned");
                if let Some(mem_tracker) = tracker.take() {
                    mem_tracker.stop();
                    // In a real implementation, this would return actual memory usage
                    0 // Placeholder
                } else {
                    0
                }
            };

            // Update profile
            let mut profiles = self.profiles.lock().expect("Profiles lock poisoned");
            let profile = profiles
                .entry(operation_name.to_string())
                .or_insert_with(|| QuantumOperationProfile {
                    operation_name: operation_name.to_string(),
                    execution_count: 0,
                    total_time: Duration::ZERO,
                    average_time: Duration::ZERO,
                    min_time: Duration::MAX,
                    max_time: Duration::ZERO,
                    memory_usage: 0,
                    gate_count: 0,
                });

            profile.execution_count += 1;
            profile.total_time += execution_time;
            profile.average_time = profile.total_time / profile.execution_count as u32;
            profile.min_time = profile.min_time.min(execution_time);
            profile.max_time = profile.max_time.max(execution_time);
            profile.memory_usage += memory_usage;
            profile.gate_count += gate_count;
        }
    }

    /// Profile a quantum operation with automatic timing
    pub fn profile_operation<F, R>(&self, operation_name: &str, gate_count: u64, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.is_enabled() {
            return operation();
        }

        self.start_operation(operation_name);
        let result = operation();
        self.end_operation(operation_name, gate_count);
        result
    }

    /// Get profiling results for all operations
    pub fn get_profiles(&self) -> HashMap<String, QuantumOperationProfile> {
        self.profiles
            .lock()
            .expect("Profiles lock poisoned")
            .clone()
    }

    /// Get profiling results for a specific operation
    pub fn get_operation_profile(&self, operation_name: &str) -> Option<QuantumOperationProfile> {
        self.profiles
            .lock()
            .expect("Profiles lock poisoned")
            .get(operation_name)
            .cloned()
    }

    /// Generate a comprehensive profiling report
    pub fn generate_report(&self) -> String {
        let profiles = self.get_profiles();
        let mut report = String::new();

        report.push_str("=== QuantRS2 Performance Profiling Report ===\n\n");

        if profiles.is_empty() {
            report.push_str("No profiling data available.\n");
            return report;
        }

        // Sort by total execution time
        let mut sorted_profiles: Vec<_> = profiles.values().collect();
        sorted_profiles.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        writeln!(
            report,
            "{:<30} {:<10} {:<12} {:<12} {:<12} {:<12} {:<10}",
            "Operation", "Count", "Total (ms)", "Avg (ms)", "Min (ms)", "Max (ms)", "Gates"
        )
        .expect("Writing to String cannot fail");
        report.push_str(&"-".repeat(110));
        report.push('\n');

        for profile in &sorted_profiles {
            writeln!(
                report,
                "{:<30} {:<10} {:<12.3} {:<12.3} {:<12.3} {:<12.3} {:<10}",
                profile.operation_name,
                profile.execution_count,
                profile.total_time.as_secs_f64() * 1000.0,
                profile.average_time.as_secs_f64() * 1000.0,
                profile.min_time.as_secs_f64() * 1000.0,
                profile.max_time.as_secs_f64() * 1000.0,
                profile.gate_count,
            )
            .expect("Writing to String cannot fail");
        }

        report.push_str("\n=== Performance Insights ===\n");

        // Find the most time-consuming operation
        if let Some(slowest) = sorted_profiles.first() {
            writeln!(
                report,
                "Most time-consuming operation: {} ({:.3}ms total)",
                slowest.operation_name,
                slowest.total_time.as_secs_f64() * 1000.0
            )
            .expect("Writing to String cannot fail");
        }

        // Find the most frequent operation
        let most_frequent = sorted_profiles.iter().max_by_key(|p| p.execution_count);
        if let Some(frequent) = most_frequent {
            writeln!(
                report,
                "Most frequent operation: {} ({} executions)",
                frequent.operation_name, frequent.execution_count
            )
            .expect("Writing to String cannot fail");
        }

        // Calculate total gate throughput
        let total_gates: u64 = profiles.values().map(|p| p.gate_count).sum();
        let total_time: Duration = profiles.values().map(|p| p.total_time).sum();
        if total_time.as_secs_f64() > 0.0 {
            let gate_throughput = total_gates as f64 / total_time.as_secs_f64();
            writeln!(
                report,
                "Total gate throughput: {gate_throughput:.0} gates/second"
            )
            .expect("Writing to String cannot fail");
        }

        report
    }

    /// Clear all profiling data
    pub fn clear(&self) {
        self.profiles
            .lock()
            .expect("Profiles lock poisoned")
            .clear();
        self.active_timers
            .lock()
            .expect("Active timers lock poisoned")
            .clear();
    }

    /// Export profiling data to JSON
    pub fn export_json(&self) -> QuantRS2Result<String> {
        let profiles = self.get_profiles();
        serde_json::to_string_pretty(&profiles).map_err(|e| e.into()) // Use the existing From<serde_json::Error> implementation
    }
}

/// Global quantum profiler instance
static GLOBAL_QUANTUM_PROFILER: OnceLock<QuantumProfiler> = OnceLock::new();

/// Get the global quantum profiler
pub fn global_quantum_profiler() -> &'static QuantumProfiler {
    GLOBAL_QUANTUM_PROFILER.get_or_init(QuantumProfiler::new)
}

/// Enable quantum profiling globally
pub fn enable_quantum_profiling() {
    global_quantum_profiler().enable();
}

/// Disable quantum profiling globally
pub fn disable_quantum_profiling() {
    global_quantum_profiler().disable();
}

/// Check if quantum profiling is active
pub fn is_profiling_active() -> bool {
    global_quantum_profiler().is_enabled()
}

/// Macro for easy profiling of quantum operations
#[macro_export]
macro_rules! profile_quantum_operation {
    ($operation_name:expr, $gate_count:expr, $operation:expr) => {{
        $crate::optimizations::profiling_integration::global_quantum_profiler().profile_operation(
            $operation_name,
            $gate_count,
            || $operation,
        )
    }};
}

/// Macro for easy profiling with automatic gate counting
#[macro_export]
macro_rules! profile_gate_operation {
    ($gate_name:expr, $operation:expr) => {{
        $crate::optimizations::profiling_integration::global_quantum_profiler().profile_operation(
            $gate_name,
            1,
            || $operation,
        )
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_profiling() {
        let profiler = QuantumProfiler::new();
        profiler.enable();

        // Profile a simple operation
        let result = profiler.profile_operation("test_gate", 1, || {
            thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        let profiles = profiler.get_profiles();
        assert!(profiles.contains_key("test_gate"));

        let test_profile = &profiles["test_gate"];
        assert_eq!(test_profile.execution_count, 1);
        assert_eq!(test_profile.gate_count, 1);
        assert!(test_profile.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_operations() {
        let profiler = QuantumProfiler::new();
        profiler.enable();

        // Profile multiple operations
        for i in 0..5 {
            profiler.profile_operation("hadamard", 1, || {
                thread::sleep(Duration::from_millis(1));
            });
        }

        for i in 0..3 {
            profiler.profile_operation("cnot", 2, || {
                thread::sleep(Duration::from_millis(2));
            });
        }

        let profiles = profiler.get_profiles();

        let hadamard_profile = &profiles["hadamard"];
        assert_eq!(hadamard_profile.execution_count, 5);
        assert_eq!(hadamard_profile.gate_count, 5);

        let cnot_profile = &profiles["cnot"];
        assert_eq!(cnot_profile.execution_count, 3);
        assert_eq!(cnot_profile.gate_count, 6); // 2 gates * 3 executions
    }

    #[test]
    fn test_profiling_disabled() {
        let profiler = QuantumProfiler::new();
        // Don't enable profiling

        let result = profiler.profile_operation("test_gate", 1, || 42);

        assert_eq!(result, 42);

        let profiles = profiler.get_profiles();
        assert!(profiles.is_empty()); // No profiling data should be collected
    }

    #[test]
    fn test_report_generation() {
        let profiler = QuantumProfiler::new();
        profiler.enable();

        profiler.profile_operation("fast_gate", 1, || {
            thread::sleep(Duration::from_millis(1));
        });

        profiler.profile_operation("slow_gate", 1, || {
            thread::sleep(Duration::from_millis(10));
        });

        let report = profiler.generate_report();
        assert!(report.contains("QuantRS2 Performance Profiling Report"));
        assert!(report.contains("fast_gate"));
        assert!(report.contains("slow_gate"));
        assert!(report.contains("Performance Insights"));
    }

    #[test]
    fn test_json_export() {
        let profiler = QuantumProfiler::new();
        profiler.enable();

        profiler.profile_operation("test_operation", 1, || {
            thread::sleep(Duration::from_millis(1));
        });

        let json_result = profiler.export_json();
        assert!(json_result.is_ok());

        let json = json_result.expect("Failed to export JSON");
        assert!(json.contains("test_operation"));
        assert!(json.contains("execution_count"));
    }
}
