//! Foundational scheduling primitives for `advanced_scheduling`.
//!
//! These are pure, side-effect-free building blocks used by the higher-level
//! scheduling logic in `crate::advanced_scheduling`:
//!
//! - **Priority computation** — Weighted Shortest Job First (WSJF)
//! - **Deadline arithmetic** — `time_to_deadline`, `is_overdue`, `slack_seconds`
//! - **Throughput estimation** — completed jobs per unit window
//! - **Sort keys** — Earliest Deadline First (EDF) ordering helper
//! - **Resource fit** — `available_qubits`, `device_compatible`
//!
//! They take primitive inputs (`Duration`, `SystemTime`, scalar counts) so they
//! can be tested in isolation and reused from the wider scheduling module
//! without touching internal scheduler state.

use std::time::{Duration, SystemTime};

use crate::job_scheduling::{JobPriority, ResourceRequirements};

/// Smallest divisor we accept when normalising by a runtime; prevents
/// division-by-zero blow-ups while keeping arithmetic finite.
const RUNTIME_EPSILON_SECS: f64 = 1.0e-3;

/// Weighted Shortest Job First priority.
///
/// `priority = weight / max(runtime_secs, ε)`. A higher returned value
/// indicates a more urgent job. Zero or near-zero runtimes are clamped to
/// `RUNTIME_EPSILON_SECS` to keep the division well-defined; negative
/// weights are passed through (callers may use them to *deprioritise* jobs).
pub fn wsjf_priority(weight: f64, estimated_runtime: Duration) -> f64 {
    let runtime_secs = estimated_runtime.as_secs_f64().max(RUNTIME_EPSILON_SECS);
    weight / runtime_secs
}

/// Time remaining until `deadline`, or `None` if the deadline has already
/// passed.
pub fn time_to_deadline(deadline: SystemTime, now: SystemTime) -> Option<Duration> {
    deadline.duration_since(now).ok()
}

/// `true` iff `now > deadline`.
pub fn is_overdue(deadline: SystemTime, now: SystemTime) -> bool {
    now > deadline
}

/// Signed slack in seconds: `deadline - now - estimated_runtime`.
///
/// Negative when the job cannot finish before its deadline. Computed via
/// `i64` differences so we can represent arbitrarily negative slack
/// (`Duration` is unsigned and would saturate).
pub fn slack_seconds(deadline: SystemTime, now: SystemTime, estimated_runtime: Duration) -> i64 {
    let to_deadline_secs: i64 = match deadline.duration_since(now) {
        Ok(d) => i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        Err(e) => -i64::try_from(e.duration().as_secs()).unwrap_or(i64::MAX),
    };
    let runtime_secs = i64::try_from(estimated_runtime.as_secs()).unwrap_or(i64::MAX);
    to_deadline_secs.saturating_sub(runtime_secs)
}

/// Estimate of throughput in completed jobs per second of `window`.
/// Returns `0.0` when the window is empty so callers don't need to special
/// case zero windows.
pub fn throughput(completed_jobs: usize, window: Duration) -> f64 {
    let secs = window.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        completed_jobs as f64 / secs
    }
}

/// Sort key for Earliest Deadline First scheduling.
///
/// Jobs without a deadline are pushed to the end (`Duration::MAX`); already
/// overdue jobs map to `Duration::ZERO` so they sort first. Pairing this
/// key with a stable sort yields EDF ordering that gracefully handles
/// missing deadlines.
pub fn edf_sort_key(deadline: Option<SystemTime>, now: SystemTime) -> Duration {
    match deadline {
        Some(d) => d.duration_since(now).unwrap_or(Duration::ZERO),
        None => Duration::MAX,
    }
}

/// Number of qubits a backend can still allocate without overcommitting.
/// Saturates at zero rather than wrapping when `used_qubits > capacity`.
pub fn available_qubits(capacity_qubits: usize, used_qubits: usize) -> usize {
    capacity_qubits.saturating_sub(used_qubits)
}

/// Returns `true` iff `capacity` can satisfy every constraint in `req`.
///
/// Checks (in order): qubit count, optional max circuit depth, optional
/// memory MB, optional CPU cores, and required-feature inclusion. `None`
/// limits on the capacity side are treated as "unlimited" — i.e. the
/// requirement is trivially met for that field.
pub fn device_compatible(
    req: &ResourceRequirements,
    capacity_qubits: usize,
    capacity_max_depth: Option<usize>,
    capacity_memory_mb: Option<u64>,
    capacity_cpu_cores: Option<u32>,
    capacity_features: &std::collections::HashSet<String>,
) -> bool {
    if capacity_qubits < req.min_qubits {
        return false;
    }
    if let Some(req_depth) = req.max_depth {
        if let Some(cap_depth) = capacity_max_depth {
            if req_depth > cap_depth {
                return false;
            }
        }
    }
    if let Some(req_mem) = req.memory_mb {
        if let Some(cap_mem) = capacity_memory_mb {
            if req_mem > cap_mem {
                return false;
            }
        }
    }
    if let Some(req_cpu) = req.cpu_cores {
        if let Some(cap_cpu) = capacity_cpu_cores {
            if req_cpu > cap_cpu {
                return false;
            }
        }
    }
    for feature in &req.required_features {
        if !capacity_features.contains(feature) {
            return false;
        }
    }
    true
}

/// Sort `items` in-place into Earliest Deadline First order.
///
/// `deadline_of` extracts the (optional) deadline for each item; missing
/// deadlines sort last. Stable so equal-deadline items retain their
/// submission order.
pub fn sort_by_edf<T, F>(items: &mut [T], now: SystemTime, deadline_of: F)
where
    F: Fn(&T) -> Option<SystemTime>,
{
    items.sort_by_key(|item| edf_sort_key(deadline_of(item), now));
}

/// Sort `items` in-place into Weighted Shortest Job First order — i.e.
/// descending priority, so the most urgent job is first.
pub fn sort_by_wsjf<T, FW, FR>(items: &mut [T], weight_of: FW, runtime_of: FR)
where
    FW: Fn(&T) -> f64,
    FR: Fn(&T) -> Duration,
{
    items.sort_by(|a, b| {
        let pa = wsjf_priority(weight_of(a), runtime_of(a));
        let pb = wsjf_priority(weight_of(b), runtime_of(b));
        // Descending: larger priority first.
        pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Numeric weight derived from a `JobPriority` for use with WSJF.
///
/// Higher numbers correspond to higher urgency:
/// `Critical=16, High=8, Normal=4, Low=2, BestEffort=1`. The exponential
/// spacing ensures a Critical job will outrank any Normal job regardless
/// of runtime within typical hardware time scales (sub-day).
pub fn priority_weight(p: JobPriority) -> f64 {
    match p {
        JobPriority::Critical => 16.0,
        JobPriority::High => 8.0,
        JobPriority::Normal => 4.0,
        JobPriority::Low => 2.0,
        JobPriority::BestEffort => 1.0,
    }
}
