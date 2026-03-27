//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    backend_traits::query_backend_capabilities, translation::HardwareBackend, CircuitResult,
    DeviceError, DeviceResult,
};
use quantrs2_circuit::{optimization::analysis::CircuitAnalyzer, prelude::Circuit};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
// SciRS2 dependencies for optimization algorithms
#[cfg(not(feature = "scirs2"))]
use super::fallback_scirs2::std as stats_std;
#[cfg(not(feature = "scirs2"))]
use super::fallback_scirs2::{mean, minimize, OptimizeResult};
use super::types::*;
#[cfg(feature = "scirs2")]
use scirs2_stats::{mean, std as stats_std};

impl QuantumJobScheduler {
    /// Create a new quantum job scheduler
    pub fn new(params: SchedulingParams) -> Self {
        let (event_sender, _) = mpsc::unbounded_channel();
        Self {
            params: Arc::new(RwLock::new(params)),
            job_queues: Arc::new(Mutex::new(BTreeMap::new())),
            jobs: Arc::new(RwLock::new(HashMap::new())),
            backend_performance: Arc::new(RwLock::new(HashMap::new())),
            backends: Arc::new(RwLock::new(HashSet::new())),
            running_jobs: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            user_shares: Arc::new(RwLock::new(HashMap::new())),
            scheduler_running: Arc::new(Mutex::new(false)),
            event_sender,
            performance_predictor: Arc::new(Mutex::new(PerformancePredictor::new())),
            resource_manager: Arc::new(Mutex::new(ResourceManager::new())),
            job_status_map: Arc::new(RwLock::new(HashMap::new())),
            job_config_map: Arc::new(RwLock::new(HashMap::new())),
            job_metrics_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    /// Register a backend
    pub async fn register_backend(&self, backend: HardwareBackend) -> DeviceResult<()> {
        let mut backends = self
            .backends
            .write()
            .expect("Failed to acquire write lock on backends in register_backend");
        backends.insert(backend);
        let mut performance = self
            .backend_performance
            .write()
            .expect("Failed to acquire write lock on backend_performance in register_backend");
        performance.insert(
            backend,
            BackendPerformance {
                backend,
                queue_length: 0,
                avg_queue_time: Duration::from_secs(0),
                avg_execution_time: Duration::from_secs(0),
                success_rate: 1.0,
                utilization: 0.0,
                avg_cost: None,
                last_updated: SystemTime::now(),
                history: VecDeque::new(),
            },
        );
        let capabilities = query_backend_capabilities(backend);
        let mut resource_manager = self
            .resource_manager
            .lock()
            .expect("Failed to acquire lock on resource_manager in register_backend");
        resource_manager.available_resources.insert(
            backend,
            ResourceCapacity {
                qubits: capabilities.features.max_qubits,
                max_circuit_depth: capabilities.features.max_depth,
                memory_mb: 8192,
                cpu_cores: 4,
                concurrent_jobs: 10,
                features: capabilities
                    .features
                    .supported_measurement_bases
                    .into_iter()
                    .collect(),
            },
        );
        Ok(())
    }
    /// Get list of available backends
    pub fn get_available_backends(&self) -> Vec<HardwareBackend> {
        let backends = self
            .backends
            .read()
            .expect("Failed to acquire read lock on backends in get_available_backends");
        backends.iter().copied().collect()
    }
    /// Submit a quantum job for execution
    pub async fn submit_job<const N: usize>(
        &self,
        circuit: Circuit<N>,
        shots: usize,
        config: JobConfig,
        user_id: String,
    ) -> DeviceResult<JobId> {
        let job_id = JobId::new();
        let now = SystemTime::now();
        self.validate_job_config(&config).await?;
        let estimated_duration = self
            .estimate_execution_time(&circuit, shots, &config)
            .await?;
        let estimated_cost = self.estimate_cost(&circuit, shots, &config).await?;
        let job = QuantumJob {
            id: job_id.clone(),
            config,
            circuit,
            shots,
            submitted_at: now,
            status: JobStatus::Pending,
            execution_history: vec![],
            metadata: HashMap::new(),
            user_id: user_id.clone(),
            group_id: None,
            estimated_duration: Some(estimated_duration),
            assigned_backend: None,
            estimated_cost: Some(estimated_cost),
            actual_cost: None,
        };
        let mut jobs = self
            .jobs
            .write()
            .expect("Failed to acquire write lock on jobs in submit_job");
        jobs.insert(job_id.clone(), Box::new(job.clone()));
        drop(jobs);
        {
            let mut config_map = self
                .job_config_map
                .write()
                .expect("Failed to acquire write lock on job_config_map in submit_job");
            config_map.insert(job_id.clone(), job.config.clone());
        }
        let mut queues = self
            .job_queues
            .lock()
            .expect("Failed to acquire lock on job_queues in submit_job");
        let queue = queues.entry(job.config.priority).or_default();
        queue.push_back(job_id.clone());
        drop(queues);
        self.update_user_share(&user_id, 1, 0).await;
        let _ = self
            .event_sender
            .send(SchedulerEvent::JobSubmitted(job_id.clone()));
        self.ensure_scheduler_running().await;
        Ok(job_id)
    }
    /// Cancel a queued or running job
    pub async fn cancel_job(&self, job_id: &JobId) -> DeviceResult<bool> {
        let mut queues = self
            .job_queues
            .lock()
            .expect("Failed to acquire lock on job_queues in cancel_job");
        for queue in queues.values_mut() {
            if let Some(pos) = queue.iter().position(|id| id == job_id) {
                queue.remove(pos);
                drop(queues);
                self.update_job_status(job_id, JobStatus::Cancelled).await?;
                let _ = self
                    .event_sender
                    .send(SchedulerEvent::JobCancelled(job_id.clone()));
                return Ok(true);
            }
        }
        drop(queues);
        let is_running = {
            let running_jobs = self.running_jobs.read().map_err(|_| {
                DeviceError::APIError("Lock poisoned on running_jobs in cancel_job".to_string())
            })?;
            running_jobs.contains_key(job_id)
        };
        if is_running {
            self.update_job_status(job_id, JobStatus::Cancelled).await?;
            let mut running_jobs = self.running_jobs.write().map_err(|_| {
                DeviceError::APIError(
                    "Lock poisoned on running_jobs write in cancel_job".to_string(),
                )
            })?;
            running_jobs.remove(job_id);
            let _ = self
                .event_sender
                .send(SchedulerEvent::JobCancelled(job_id.clone()));
            return Ok(true);
        }
        Ok(false)
    }
    /// Get job status and information
    pub async fn get_job_status<const N: usize>(
        &self,
        job_id: &JobId,
    ) -> DeviceResult<Option<QuantumJob<N>>> {
        let jobs = self
            .jobs
            .read()
            .expect("Failed to acquire read lock on jobs in get_job_status");
        if let Some(job_any) = jobs.get(job_id) {
            if let Some(job) = job_any.downcast_ref::<QuantumJob<N>>() {
                return Ok(Some(job.clone()));
            }
        }
        Ok(None)
    }
    /// Get queue analytics and predictions
    pub async fn get_queue_analytics(&self) -> DeviceResult<QueueAnalytics> {
        let queues = self
            .job_queues
            .lock()
            .expect("Failed to acquire lock on job_queues in get_queue_analytics");
        let backend_performance = self
            .backend_performance
            .read()
            .expect("Failed to acquire read lock on backend_performance in get_queue_analytics");
        let total_queue_length = queues.values().map(|q| q.len()).sum();
        let queue_by_priority = queues
            .iter()
            .map(|(priority, queue)| (*priority, queue.len()))
            .collect();
        let queue_by_backend = backend_performance
            .iter()
            .map(|(backend, perf)| (*backend, perf.queue_length))
            .collect();
        let predicted_queue_times = self.predict_queue_times(&backend_performance).await;
        let system_load = self.calculate_system_load(&backend_performance).await;
        let throughput = self.calculate_throughput().await;
        let avg_wait_time = self.calculate_average_wait_time().await;
        Ok(QueueAnalytics {
            total_queue_length,
            queue_by_priority,
            queue_by_backend,
            predicted_queue_times,
            system_load,
            throughput,
            avg_wait_time,
        })
    }
    /// Start the job scheduler
    pub async fn start_scheduler(&self) -> DeviceResult<()> {
        let mut running = self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in start_scheduler");
        if *running {
            return Err(DeviceError::APIError(
                "Scheduler already running".to_string(),
            ));
        }
        *running = true;
        drop(running);
        let scheduler = Arc::new(self.clone());
        tokio::spawn(async move {
            scheduler.scheduling_loop().await;
        });
        let scheduler = Arc::new(self.clone());
        tokio::spawn(async move {
            scheduler.performance_monitoring_loop().await;
        });
        let params = self
            .params
            .read()
            .expect("Failed to acquire read lock on params in start_scheduler");
        if params.scirs2_params.enabled {
            drop(params);
            let scheduler = Arc::new(self.clone());
            tokio::spawn(async move {
                scheduler.scirs2_optimization_loop().await;
            });
        }
        Ok(())
    }
    /// Stop the job scheduler
    pub async fn stop_scheduler(&self) -> DeviceResult<()> {
        let mut running = self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in stop_scheduler");
        *running = false;
        Ok(())
    }
    async fn validate_job_config(&self, config: &JobConfig) -> DeviceResult<()> {
        let backends = self
            .backends
            .read()
            .expect("Failed to acquire read lock on backends in validate_job_config");
        if backends.is_empty() {
            return Err(DeviceError::APIError("No backends available".to_string()));
        }
        let resource_manager = self
            .resource_manager
            .lock()
            .expect("Failed to acquire lock on resource_manager in validate_job_config");
        let mut can_satisfy = false;
        for (backend, capacity) in &resource_manager.available_resources {
            if capacity.qubits >= config.resource_requirements.min_qubits {
                if let Some(max_depth) = config.resource_requirements.max_depth {
                    if let Some(backend_max_depth) = capacity.max_circuit_depth {
                        if max_depth > backend_max_depth {
                            continue;
                        }
                    }
                }
                can_satisfy = true;
                break;
            }
        }
        if !can_satisfy {
            return Err(DeviceError::APIError(
                "No backend can satisfy resource requirements".to_string(),
            ));
        }
        Ok(())
    }
    async fn estimate_execution_time<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
        config: &JobConfig,
    ) -> DeviceResult<Duration> {
        let analyzer = CircuitAnalyzer::new();
        let metrics = analyzer
            .analyze(circuit)
            .map_err(|e| DeviceError::APIError(format!("Circuit analysis error: {e:?}")))?;
        let circuit_complexity = (metrics.gate_count as f64).mul_add(0.1, metrics.depth as f64);
        let shots_factor = (shots as f64).log10();
        let base_time = Duration::from_secs((circuit_complexity * shots_factor) as u64);
        let backend_performance = self.backend_performance.read().expect(
            "Failed to acquire read lock on backend_performance in estimate_execution_time",
        );
        let avg_execution_time = if backend_performance.is_empty() {
            Duration::from_secs(60)
        } else {
            let total_time: Duration = backend_performance
                .values()
                .map(|p| p.avg_execution_time)
                .sum();
            total_time / backend_performance.len() as u32
        };
        let estimated = Duration::from_millis(
            u128::midpoint(base_time.as_millis(), avg_execution_time.as_millis())
                .try_into()
                .expect(
                    "Failed to convert estimated execution time to u64 in estimate_execution_time",
                ),
        );
        Ok(estimated)
    }
    async fn estimate_cost<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
        config: &JobConfig,
    ) -> DeviceResult<f64> {
        let analyzer = CircuitAnalyzer::new();
        let metrics = analyzer
            .analyze(circuit)
            .map_err(|e| DeviceError::APIError(format!("Circuit analysis error: {e:?}")))?;
        let circuit_complexity = metrics.depth as f64 + metrics.gate_count as f64;
        let base_cost = circuit_complexity * shots as f64 * 0.001;
        let priority_multiplier = match config.priority {
            JobPriority::Critical => 3.0,
            JobPriority::High => 2.0,
            JobPriority::Normal => 1.0,
            JobPriority::Low => 0.7,
            JobPriority::BestEffort => 0.5,
        };
        Ok(base_cost * priority_multiplier)
    }
    async fn update_user_share(&self, user_id: &str, queued_delta: i32, running_delta: i32) {
        let mut user_shares = self
            .user_shares
            .write()
            .expect("Failed to acquire write lock on user_shares in update_user_share");
        let share = user_shares
            .entry(user_id.to_string())
            .or_insert_with(|| UserShare {
                user_id: user_id.to_string(),
                allocated_share: 1.0,
                used_share: 0.0,
                jobs_running: 0,
                jobs_queued: 0,
                last_updated: SystemTime::now(),
            });
        share.jobs_queued = (share.jobs_queued as i32 + queued_delta).max(0) as usize;
        share.jobs_running = (share.jobs_running as i32 + running_delta).max(0) as usize;
        share.last_updated = SystemTime::now();
    }
    async fn update_job_status(&self, job_id: &JobId, status: JobStatus) -> DeviceResult<()> {
        let mut status_map = self.job_status_map.write().map_err(|_| {
            DeviceError::APIError("Failed to acquire write lock on job_status_map".to_string())
        })?;
        status_map.insert(job_id.clone(), status);
        Ok(())
    }
    /// Query the current status of a job without needing the circuit type parameter.
    pub fn job_status(&self, job_id: &JobId) -> Option<JobStatus> {
        let status_map = self.job_status_map.read().ok()?;
        status_map.get(job_id).cloned()
    }
    /// Sort all pending jobs in all priority queues by estimated duration (shortest first)
    pub async fn sort_queues_by_duration(&self) -> DeviceResult<()> {
        let jobs_snapshot: HashMap<JobId, Option<std::time::Duration>> = {
            let jobs = self
                .jobs
                .read()
                .map_err(|_| DeviceError::APIError("Lock poisoned".to_string()))?;
            jobs.keys()
                .map(|id| (id.clone(), None::<std::time::Duration>))
                .collect()
        };
        let mut queues = self
            .job_queues
            .lock()
            .map_err(|_| DeviceError::APIError("Lock poisoned".to_string()))?;
        for queue in queues.values_mut() {
            queue.make_contiguous().sort_by(|a, b| {
                let da = jobs_snapshot
                    .get(a)
                    .and_then(|d| *d)
                    .unwrap_or(std::time::Duration::MAX);
                let db = jobs_snapshot
                    .get(b)
                    .and_then(|d| *d)
                    .unwrap_or(std::time::Duration::MAX);
                da.cmp(&db)
            });
        }
        Ok(())
    }
    /// Bin-pack jobs into backend slots by resource requirements.
    ///
    /// Returns a mapping from backend → list of job IDs assigned to it.
    pub async fn bin_pack_jobs(&self) -> DeviceResult<HashMap<HardwareBackend, Vec<JobId>>> {
        let resource_manager = self
            .resource_manager
            .lock()
            .map_err(|_| DeviceError::APIError("Lock poisoned".to_string()))?;
        let queues = self
            .job_queues
            .lock()
            .map_err(|_| DeviceError::APIError("Lock poisoned".to_string()))?;
        let jobs = self
            .jobs
            .read()
            .map_err(|_| DeviceError::APIError("Lock poisoned".to_string()))?;
        let mut remaining_slots: HashMap<HardwareBackend, usize> = resource_manager
            .available_resources
            .iter()
            .map(|(&b, cap)| (b, cap.concurrent_jobs))
            .collect();
        let mut assignment: HashMap<HardwareBackend, Vec<JobId>> = HashMap::new();
        for queue in queues.values() {
            for job_id in queue.iter() {
                let best_backend = remaining_slots
                    .iter()
                    .filter(|(_, &slots)| slots > 0)
                    .max_by_key(|(_, &slots)| slots)
                    .map(|(&b, _)| b);
                if let Some(backend) = best_backend {
                    assignment.entry(backend).or_default().push(job_id.clone());
                    if let Some(slots) = remaining_slots.get_mut(&backend) {
                        *slots = slots.saturating_sub(1);
                    }
                }
            }
        }
        Ok(assignment)
    }
    /// Route a single job to the backend with the lowest current load (queue_length).
    pub async fn route_to_least_loaded_backend(
        &self,
        job_id: &JobId,
    ) -> DeviceResult<Option<HardwareBackend>> {
        let backend_performance = self
            .backend_performance
            .read()
            .map_err(|_| DeviceError::APIError("Lock poisoned".to_string()))?;
        let chosen = backend_performance
            .iter()
            .filter(|_| true)
            .min_by(|(_, a), (_, b)| {
                a.queue_length.cmp(&b.queue_length).then_with(|| {
                    a.utilization
                        .partial_cmp(&b.utilization)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|(&backend, _)| backend);
        Ok(chosen)
    }
    async fn ensure_scheduler_running(&self) {
        let running = self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in ensure_scheduler_running");
        if !*running {
            drop(running);
            let _ = self.start_scheduler().await;
        }
    }
    async fn predict_queue_times(
        &self,
        backend_performance: &HashMap<HardwareBackend, BackendPerformance>,
    ) -> HashMap<HardwareBackend, Duration> {
        let mut predictions = HashMap::new();
        for (backend, perf) in backend_performance {
            let predicted_time = Duration::from_secs(
                (perf.queue_length as u64 * perf.avg_execution_time.as_secs())
                    / perf.success_rate.max(0.1) as u64,
            );
            predictions.insert(*backend, predicted_time);
        }
        predictions
    }
    async fn calculate_system_load(
        &self,
        backend_performance: &HashMap<HardwareBackend, BackendPerformance>,
    ) -> f64 {
        if backend_performance.is_empty() {
            return 0.0;
        }
        let total_utilization: f64 = backend_performance.values().map(|p| p.utilization).sum();
        total_utilization / backend_performance.len() as f64
    }
    async fn calculate_throughput(&self) -> f64 {
        let history = self
            .execution_history
            .read()
            .expect("Failed to acquire read lock on execution_history in calculate_throughput");
        if history.is_empty() {
            return 0.0;
        }
        let one_hour_ago = SystemTime::now() - Duration::from_secs(3600);
        let recent_completions = history
            .iter()
            .filter(|exec| exec.started_at > one_hour_ago)
            .count();
        recent_completions as f64
    }
    async fn calculate_average_wait_time(&self) -> Duration {
        let history = self.execution_history.read().expect(
            "Failed to acquire read lock on execution_history in calculate_average_wait_time",
        );
        if history.is_empty() {
            return Duration::from_secs(0);
        }
        let total_wait: Duration = history.iter().map(|exec| exec.metrics.queue_time).sum();
        total_wait / history.len() as u32
    }
    async fn scheduling_loop(&self) {
        while *self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in scheduling_loop")
        {
            if let Err(e) = self.schedule_next_jobs().await {
                eprintln!("Scheduling error: {e}");
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    async fn schedule_next_jobs(&self) -> DeviceResult<()> {
        let params = self
            .params
            .read()
            .expect("Failed to acquire read lock on params in schedule_next_jobs")
            .clone();
        match params.strategy {
            SchedulingStrategy::PriorityFIFO => self.schedule_priority_fifo().await,
            SchedulingStrategy::ShortestJobFirst => self.schedule_shortest_job_first().await,
            SchedulingStrategy::FairShare => self.schedule_fair_share().await,
            SchedulingStrategy::Backfill => self.schedule_backfill().await,
            SchedulingStrategy::MLOptimized => self.schedule_ml_optimized().await,
            _ => self.schedule_priority_fifo().await,
        }
    }
    async fn schedule_priority_fifo(&self) -> DeviceResult<()> {
        for priority in [
            JobPriority::Critical,
            JobPriority::High,
            JobPriority::Normal,
            JobPriority::Low,
            JobPriority::BestEffort,
        ] {
            let job_id = {
                let mut queues = self
                    .job_queues
                    .lock()
                    .expect("Failed to acquire lock on job_queues in schedule_priority_fifo");
                queues
                    .get_mut(&priority)
                    .and_then(|queue| queue.pop_front())
            };
            if let Some(job_id) = job_id {
                if let Some(backend) = self.find_best_backend(&job_id).await? {
                    self.assign_job_to_backend(&job_id, backend).await?;
                    break;
                } else {
                    let mut queues = self
                        .job_queues
                        .lock()
                        .expect(
                            "Failed to acquire lock on job_queues to requeue job in schedule_priority_fifo",
                        );
                    if let Some(queue) = queues.get_mut(&priority) {
                        queue.push_front(job_id);
                    }
                    break;
                }
            }
        }
        Ok(())
    }
    async fn schedule_shortest_job_first(&self) -> DeviceResult<()> {
        self.sort_queues_by_duration().await?;
        self.schedule_priority_fifo().await
    }
    async fn schedule_fair_share(&self) -> DeviceResult<()> {
        let user_shares_snapshot = {
            let user_shares = self
                .user_shares
                .read()
                .map_err(|_| DeviceError::APIError("Lock poisoned on user_shares".to_string()))?;
            user_shares.clone()
        };
        let fairness_score = |user_id: &str| -> f64 {
            user_shares_snapshot.get(user_id).map_or(0.0, |s| {
                if s.allocated_share < 1e-10 {
                    f64::MAX
                } else {
                    s.used_share / s.allocated_share
                }
            })
        };
        let mut candidate: Option<(JobPriority, usize, f64)> = None;
        {
            let queues = self
                .job_queues
                .lock()
                .map_err(|_| DeviceError::APIError("Lock poisoned on job_queues".to_string()))?;
            let jobs = self
                .jobs
                .read()
                .map_err(|_| DeviceError::APIError("Lock poisoned on jobs".to_string()))?;
            for (&priority, queue) in queues.iter() {
                if let Some((pos, job_id)) = queue.iter().enumerate().next() {
                    let score = {
                        macro_rules! try_downcast {
                            ($n:expr) => {
                                jobs.get(job_id)
                                    .and_then(|b| b.downcast_ref::<super::types::QuantumJob<$n>>())
                                    .map(|j| fairness_score(&j.user_id))
                            };
                        }
                        try_downcast!(1)
                            .or_else(|| try_downcast!(2))
                            .or_else(|| try_downcast!(4))
                            .or_else(|| try_downcast!(8))
                            .or_else(|| try_downcast!(16))
                            .or_else(|| try_downcast!(32))
                            .or_else(|| try_downcast!(64))
                            .unwrap_or(0.0)
                    };
                    let better = candidate.as_ref().map_or(true, |&(_, _, s)| score < s);
                    if better {
                        candidate = Some((priority, pos, score));
                    }
                }
            }
        }
        if let Some((priority, pos, _)) = candidate {
            let job_id =
                {
                    let mut queues = self.job_queues.lock().map_err(|_| {
                        DeviceError::APIError("Lock poisoned on job_queues".to_string())
                    })?;
                    queues.get_mut(&priority).and_then(|q| {
                        if pos < q.len() {
                            q.remove(pos)
                        } else {
                            None
                        }
                    })
                };
            if let Some(job_id) = job_id {
                if let Some(backend) = self.find_best_backend(&job_id).await? {
                    self.assign_job_to_backend(&job_id, backend).await?;
                }
            }
        }
        Ok(())
    }
    async fn schedule_backfill(&self) -> DeviceResult<()> {
        self.sort_queues_by_duration().await?;
        let assignment = self.bin_pack_jobs().await?;
        for (backend, job_ids) in assignment {
            for job_id in job_ids {
                let removed = {
                    let mut queues = self.job_queues.lock().map_err(|_| {
                        DeviceError::APIError("Lock poisoned on job_queues".to_string())
                    })?;
                    let mut found = false;
                    for queue in queues.values_mut() {
                        if let Some(pos) = queue.iter().position(|id| id == &job_id) {
                            queue.remove(pos);
                            found = true;
                            break;
                        }
                    }
                    found
                };
                if removed && self.is_backend_available(backend).await {
                    self.assign_job_to_backend(&job_id, backend).await?;
                }
            }
        }
        Ok(())
    }
    async fn schedule_ml_optimized(&self) -> DeviceResult<()> {
        #[cfg(feature = "scirs2")]
        {
            self.scirs2_optimize_schedule().await
        }
        #[cfg(not(feature = "scirs2"))]
        {
            self.schedule_priority_fifo().await
        }
    }
    #[cfg(feature = "scirs2")]
    async fn scirs2_optimize_schedule(&self) -> DeviceResult<()> {
        self.schedule_priority_fifo().await
    }
    async fn find_best_backend(&self, job_id: &JobId) -> DeviceResult<Option<HardwareBackend>> {
        {
            let jobs = self
                .jobs
                .read()
                .expect("Failed to acquire read lock on jobs in find_best_backend");
            let _job_any = jobs
                .get(job_id)
                .ok_or_else(|| DeviceError::APIError("Job not found".to_string()))?;
        }
        let job_resource_requirements: Option<ResourceRequirements> = {
            let config_map = self
                .job_config_map
                .read()
                .expect("Failed to acquire read lock on job_config_map in find_best_backend");
            config_map
                .get(job_id)
                .map(|cfg| cfg.resource_requirements.clone())
        };
        let backends: Vec<_> = {
            let backends = self
                .backends
                .read()
                .expect("Failed to acquire read lock on backends in find_best_backend");
            backends.iter().copied().collect()
        };
        let allocation_strategy = {
            let params = self
                .params
                .read()
                .expect("Failed to acquire read lock on params in find_best_backend");
            params.allocation_strategy.clone()
        };
        let backend_performance_snapshot = {
            let backend_performance = self
                .backend_performance
                .read()
                .expect("Failed to acquire read lock on backend_performance in find_best_backend");
            backend_performance.clone()
        };
        match allocation_strategy {
            AllocationStrategy::FirstFit => {
                for backend in backends {
                    if self.is_backend_available(backend).await {
                        return Ok(Some(backend));
                    }
                }
            }
            AllocationStrategy::BestFit => {
                let (capacity_snapshot, required_qubits) = {
                    let resource_manager = self.resource_manager.lock().expect(
                        "Failed to acquire lock on resource_manager in find_best_backend BestFit",
                    );
                    let snapshot: HashMap<HardwareBackend, usize> = resource_manager
                        .available_resources
                        .iter()
                        .map(|(&b, cap)| (b, cap.qubits))
                        .collect();
                    let req = job_resource_requirements
                        .as_ref()
                        .map_or(1, |r| r.min_qubits);
                    (snapshot, req)
                };
                let mut best_backend: Option<HardwareBackend> = None;
                let mut best_excess = usize::MAX;
                for &backend in &backends {
                    if !self.is_backend_available(backend).await {
                        continue;
                    }
                    if let Some(&cap_qubits) = capacity_snapshot.get(&backend) {
                        if cap_qubits >= required_qubits {
                            let excess = cap_qubits - required_qubits;
                            if excess < best_excess {
                                best_excess = excess;
                                best_backend = Some(backend);
                            }
                        }
                    }
                }
                return Ok(best_backend);
            }
            AllocationStrategy::LeastLoaded => {
                let mut best_backend = None;
                let mut lowest_utilization = f64::INFINITY;
                for (&backend, perf) in &backend_performance_snapshot {
                    if self.is_backend_available(backend).await
                        && perf.utilization < lowest_utilization
                    {
                        lowest_utilization = perf.utilization;
                        best_backend = Some(backend);
                    }
                }
                return Ok(best_backend);
            }
            _ => {
                for &backend in &backends {
                    if self.is_backend_available(backend).await {
                        return Ok(Some(backend));
                    }
                }
            }
        }
        Ok(None)
    }
    async fn is_backend_available(&self, backend: HardwareBackend) -> bool {
        let available = {
            let running_jobs = self
                .running_jobs
                .read()
                .expect("Failed to acquire read lock on running_jobs in is_backend_available");
            let backend_jobs = running_jobs.values().filter(|(b, _)| *b == backend).count();
            drop(running_jobs);
            let resource_manager = self
                .resource_manager
                .lock()
                .expect("Failed to acquire lock on resource_manager in is_backend_available");
            let result = resource_manager
                .available_resources
                .get(&backend)
                .is_some_and(|capacity| backend_jobs < capacity.concurrent_jobs);
            drop(resource_manager);
            result
        };
        available
    }
    async fn assign_job_to_backend(
        &self,
        job_id: &JobId,
        backend: HardwareBackend,
    ) -> DeviceResult<()> {
        {
            let mut running_jobs = self
                .running_jobs
                .write()
                .expect("Failed to acquire write lock on running_jobs in assign_job_to_backend");
            running_jobs.insert(job_id.clone(), (backend, SystemTime::now()));
        }
        self.update_job_status(job_id, JobStatus::Scheduled).await?;
        let _ = self
            .event_sender
            .send(SchedulerEvent::JobScheduled(job_id.clone(), backend));
        let job_id_clone = job_id.clone();
        let scheduler = Arc::new(self.clone());
        tokio::spawn(async move {
            let _ = scheduler.execute_job(&job_id_clone, backend).await;
        });
        Ok(())
    }
    async fn execute_job(&self, job_id: &JobId, backend: HardwareBackend) -> DeviceResult<()> {
        self.update_job_status(job_id, JobStatus::Running).await?;
        let _ = self
            .event_sender
            .send(SchedulerEvent::JobStarted(job_id.clone()));
        let execution_start = SystemTime::now();
        {
            let backends = self
                .backends
                .read()
                .expect("Failed to acquire read lock on backends in execute_job");
            if !backends.contains(&backend) {
                return Err(DeviceError::APIError("Backend not found".to_string()));
            }
        }
        let job_config = {
            let config_map = self
                .job_config_map
                .read()
                .expect("Failed to acquire read lock on job_config_map in execute_job");
            config_map.get(job_id).cloned()
        };
        let queue_time = {
            let jobs = self
                .jobs
                .read()
                .expect("Failed to acquire read lock on jobs in execute_job queue_time");
            macro_rules! try_submitted_at {
                ($n:expr) => {
                    jobs.get(job_id)
                        .and_then(|b| b.downcast_ref::<super::types::QuantumJob<$n>>())
                        .map(|j| {
                            execution_start
                                .duration_since(j.submitted_at)
                                .unwrap_or(Duration::from_secs(0))
                        })
                };
            }
            try_submitted_at!(1)
                .or_else(|| try_submitted_at!(2))
                .or_else(|| try_submitted_at!(4))
                .or_else(|| try_submitted_at!(8))
                .or_else(|| try_submitted_at!(16))
                .or_else(|| try_submitted_at!(32))
                .or_else(|| try_submitted_at!(64))
                .or_else(|| try_submitted_at!(128))
                .unwrap_or(Duration::from_secs(0))
        };
        {
            let status_map = self.job_status_map.read().expect(
                "Failed to acquire read lock on job_status_map in execute_job cancellation check",
            );
            if status_map.get(job_id) == Some(&JobStatus::Cancelled) {
                return Ok(());
            }
        }
        let simulated_execution_time = job_config
            .as_ref()
            .map(|_cfg| Duration::from_secs(1))
            .unwrap_or(Duration::from_secs(1));
        tokio::time::sleep(simulated_execution_time).await;
        {
            let mut running_jobs = self
                .running_jobs
                .write()
                .expect("Failed to acquire write lock on running_jobs in execute_job cleanup");
            running_jobs.remove(job_id);
        }
        self.update_job_status(job_id, JobStatus::Completed).await?;
        let execution_end = SystemTime::now();
        let execution_time = execution_end
            .duration_since(execution_start)
            .unwrap_or(Duration::from_secs(0));
        let metrics = ExecutionMetrics {
            queue_time,
            execution_time: Some(execution_time),
            resource_utilization: 1.0,
            cost: job_config.as_ref().and_then(|c| c.cost_limit),
            quality_metrics: {
                let mut m = HashMap::new();
                m.insert(
                    "execution_time_secs".to_string(),
                    execution_time.as_secs_f64(),
                );
                m.insert("queue_time_secs".to_string(), queue_time.as_secs_f64());
                m
            },
        };
        {
            let mut metrics_map = self
                .job_metrics_map
                .write()
                .expect("Failed to acquire write lock on job_metrics_map in execute_job");
            metrics_map.insert(job_id.clone(), metrics.clone());
        }
        {
            let mut perf_map = self
                .backend_performance
                .write()
                .expect("Failed to acquire write lock on backend_performance in execute_job");
            if let Some(perf) = perf_map.get_mut(&backend) {
                let alpha = 0.2_f64;
                let prev_exec_secs = perf.avg_execution_time.as_secs_f64();
                let new_exec_secs =
                    (1.0 - alpha) * prev_exec_secs + alpha * execution_time.as_secs_f64();
                perf.avg_execution_time = Duration::from_secs_f64(new_exec_secs.max(0.0));
                let prev_queue_secs = perf.avg_queue_time.as_secs_f64();
                let new_queue_secs =
                    (1.0 - alpha) * prev_queue_secs + alpha * queue_time.as_secs_f64();
                perf.avg_queue_time = Duration::from_secs_f64(new_queue_secs.max(0.0));
                perf.success_rate = (1.0 - alpha) * perf.success_rate + alpha * 1.0;
                perf.last_updated = execution_end;
            }
        }
        {
            let mut history = self
                .execution_history
                .write()
                .expect("Failed to acquire write lock on execution_history in execute_job");
            history.push(JobExecution {
                attempt: 1,
                backend,
                started_at: execution_start,
                ended_at: Some(execution_end),
                result: None,
                error: None,
                metrics,
            });
        }
        Ok(())
    }
    async fn performance_monitoring_loop(&self) {
        while *self
            .scheduler_running
            .lock()
            .expect("Failed to acquire lock on scheduler_running in performance_monitoring_loop")
        {
            self.update_backend_performance().await;
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    }
    async fn update_backend_performance(&self) {
        let mut backend_performance = self.backend_performance.write().expect(
            "Failed to acquire write lock on backend_performance in update_backend_performance",
        );
        let now = SystemTime::now();
        for (backend, perf) in backend_performance.iter_mut() {
            perf.last_updated = now;
            let snapshot = PerformanceSnapshot {
                timestamp: now,
                queue_length: perf.queue_length,
                utilization: perf.utilization,
                avg_queue_time_secs: perf.avg_queue_time.as_secs_f64(),
                success_rate: perf.success_rate,
            };
            perf.history.push_back(snapshot);
            let cutoff = now - Duration::from_secs(86400);
            while let Some(front) = perf.history.front() {
                if front.timestamp < cutoff {
                    perf.history.pop_front();
                } else {
                    break;
                }
            }
        }
    }
    async fn scirs2_optimization_loop(&self) {
        let frequency = {
            let params = self
                .params
                .read()
                .expect("Failed to acquire read lock on params in scirs2_optimization_loop");
            params.scirs2_params.optimization_frequency
        };
        loop {
            let should_continue = *self
                .scheduler_running
                .lock()
                .expect("Failed to acquire lock on scheduler_running in scirs2_optimization_loop");
            if !should_continue {
                break;
            }
            if let Err(e) = self.run_scirs2_optimization().await {
                eprintln!("SciRS2 optimization error: {e}");
            }
            tokio::time::sleep(frequency).await;
        }
    }
    async fn run_scirs2_optimization(&self) -> DeviceResult<()> {
        #[cfg(feature = "scirs2")]
        {
            let backend_snapshot: Vec<(HardwareBackend, f64)> = {
                let bp = self.backend_performance.read().expect(
                    "Failed to acquire read lock on backend_performance in run_scirs2_optimization",
                );
                bp.iter().map(|(&b, p)| (b, p.utilization)).collect()
            };
            let performance_data: Vec<f64> = backend_snapshot.iter().map(|(_, u)| *u).collect();
            if performance_data.len() > 1 {
                use scirs2_core::ndarray::Array1;
                let data_array = Array1::from_vec(performance_data);
                let avg_utilization: f64 = mean(&data_array.view()).unwrap_or(0.5);
                let utilization_std: f64 = stats_std(&data_array.view(), 1, None).unwrap_or(0.1);
                let overload_threshold = avg_utilization + utilization_std;
                let underload_threshold = (avg_utilization - utilization_std).max(0.0);
                let overloaded: Vec<HardwareBackend> = backend_snapshot
                    .iter()
                    .filter(|(_, u)| *u > overload_threshold)
                    .map(|(b, _)| *b)
                    .collect();
                let underloaded: Vec<(HardwareBackend, f64)> = backend_snapshot
                    .iter()
                    .filter(|(_, u)| *u < underload_threshold)
                    .copied()
                    .collect();
                if underloaded.is_empty() {
                    return Ok(());
                }
                for overloaded_backend in overloaded {
                    let target_backend = underloaded
                        .iter()
                        .min_by(|(_, ua), (_, ub)| {
                            ua.partial_cmp(ub).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(b, _)| *b)
                        .unwrap_or(underloaded[0].0);
                    let candidate_job_id: Option<JobId> = {
                        let queues = self
                            .job_queues
                            .lock()
                            .expect("Failed to acquire lock on job_queues in load balancing");
                        let config_map = self
                            .job_config_map
                            .read()
                            .expect("Failed to read job_config_map in load balancing");
                        let mut found = None;
                        'outer: for queue in queues.values() {
                            for job_id in queue.iter() {
                                if let Some(cfg) = config_map.get(job_id) {
                                    if cfg.preferred_backends.first() == Some(&overloaded_backend) {
                                        found = Some(job_id.clone());
                                        break 'outer;
                                    }
                                }
                            }
                        }
                        found
                    };
                    if let Some(job_id) = candidate_job_id {
                        let mut config_map = self
                            .job_config_map
                            .write()
                            .expect("Failed to write job_config_map in load balancing");
                        if let Some(cfg) = config_map.get_mut(&job_id) {
                            cfg.preferred_backends.insert(0, target_backend);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
