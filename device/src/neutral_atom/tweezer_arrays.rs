//! Optical tweezer array management for neutral atom quantum computing
//!
//! This module provides implementations for managing optical tweezer arrays,
//! including atom loading, positioning, manipulation, and optimization.

use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Optical tweezer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweezerArrayConfig {
    /// Array dimensions (x, y, z)
    pub array_dimensions: (usize, usize, usize),
    /// Tweezer spacing (μm)
    pub tweezer_spacing: (f64, f64, f64),
    /// Laser wavelength (nm)
    pub laser_wavelength: f64,
    /// Beam waist (μm)
    pub beam_waist: f64,
    /// Maximum laser power (mW)
    pub max_laser_power: f64,
    /// Trap depth (μK)
    pub trap_depth: f64,
    /// Loading efficiency
    pub loading_efficiency: f64,
    /// Movement precision (nm)
    pub movement_precision: f64,
}

/// Tweezer position in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TweezerPosition {
    /// X coordinate (μm)
    pub x: f64,
    /// Y coordinate (μm)
    pub y: f64,
    /// Z coordinate (μm)
    pub z: f64,
}

/// Atom state in a tweezer
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomState {
    /// No atom present
    Empty,
    /// Atom loaded successfully
    Loaded,
    /// Atom loading failed
    LoadingFailed,
    /// Atom lost during operation
    Lost,
    /// Atom in unknown state
    Unknown,
}

/// Individual tweezer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweezerInfo {
    /// Tweezer ID
    pub tweezer_id: usize,
    /// Current position
    pub position: TweezerPosition,
    /// Target position (for movement operations)
    pub target_position: Option<TweezerPosition>,
    /// Current laser power (mW)
    pub laser_power: f64,
    /// Atom state
    pub atom_state: AtomState,
    /// Loading attempt count
    pub loading_attempts: usize,
    /// Last update timestamp
    pub last_update: std::time::SystemTime,
}

/// Tweezer array state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweezerArrayState {
    /// Configuration
    pub config: TweezerArrayConfig,
    /// Individual tweezers
    pub tweezers: HashMap<usize, TweezerInfo>,
    /// Array loading statistics
    pub loading_stats: LoadingStatistics,
    /// Movement operations in progress
    pub active_movements: Vec<MovementOperation>,
}

/// Loading statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingStatistics {
    /// Total loading attempts
    pub total_attempts: usize,
    /// Successful loads
    pub successful_loads: usize,
    /// Failed loads
    pub failed_loads: usize,
    /// Current fill factor
    pub fill_factor: f64,
    /// Average loading time
    pub average_loading_time: Duration,
}

/// Movement operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementOperation {
    /// Operation ID
    pub operation_id: String,
    /// Tweezer ID
    pub tweezer_id: usize,
    /// Start position
    pub start_position: TweezerPosition,
    /// End position
    pub end_position: TweezerPosition,
    /// Movement parameters
    pub parameters: MovementParameters,
    /// Start time
    pub start_time: std::time::SystemTime,
    /// Expected completion time
    pub expected_completion: std::time::SystemTime,
    /// Current status
    pub status: MovementStatus,
}

/// Movement parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementParameters {
    /// Movement speed (μm/s)
    pub speed: f64,
    /// Acceleration (μm/s²)
    pub acceleration: f64,
    /// Movement trajectory
    pub trajectory: MovementTrajectory,
    /// Power ramping during movement
    pub power_ramping: PowerRampingConfig,
}

/// Movement trajectory types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MovementTrajectory {
    /// Direct linear path
    Linear,
    /// Smooth S-curve trajectory
    SCurve,
    /// Parabolic trajectory
    Parabolic,
    /// Custom waypoint-based trajectory
    Waypoints(Vec<TweezerPosition>),
}

/// Power ramping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerRampingConfig {
    /// Enable power ramping
    pub enabled: bool,
    /// Initial power factor
    pub initial_factor: f64,
    /// Final power factor
    pub final_factor: f64,
    /// Ramping duration
    pub ramping_duration: Duration,
}

/// Movement operation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MovementStatus {
    /// Movement is queued
    Queued,
    /// Movement is in progress
    InProgress,
    /// Movement completed successfully
    Completed,
    /// Movement failed
    Failed,
    /// Movement was cancelled
    Cancelled,
}

/// Tweezer array manager
pub struct TweezerArrayManager {
    state: TweezerArrayState,
    movement_queue: Vec<MovementOperation>,
    optimization_settings: OptimizationSettings,
}

/// Array optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// Enable automatic reloading
    pub auto_reload: bool,
    /// Maximum reloading attempts
    pub max_reload_attempts: usize,
    /// Enable position optimization
    pub position_optimization: bool,
    /// Enable power optimization
    pub power_optimization: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
}

impl TweezerArrayManager {
    /// Create a new tweezer array manager
    pub fn new(config: TweezerArrayConfig) -> Self {
        let total_tweezers =
            config.array_dimensions.0 * config.array_dimensions.1 * config.array_dimensions.2;
        let mut tweezers = HashMap::new();

        // Initialize tweezers
        for i in 0..total_tweezers {
            let (x_idx, y_idx, z_idx) = Self::index_to_coordinates(i, &config.array_dimensions);
            let position = TweezerPosition {
                x: x_idx as f64 * config.tweezer_spacing.0,
                y: y_idx as f64 * config.tweezer_spacing.1,
                z: z_idx as f64 * config.tweezer_spacing.2,
            };

            let tweezer_info = TweezerInfo {
                tweezer_id: i,
                position,
                target_position: None,
                laser_power: 0.0,
                atom_state: AtomState::Empty,
                loading_attempts: 0,
                last_update: std::time::SystemTime::now(),
            };

            tweezers.insert(i, tweezer_info);
        }

        let state = TweezerArrayState {
            config,
            tweezers,
            loading_stats: LoadingStatistics::default(),
            active_movements: Vec::new(),
        };

        Self {
            state,
            movement_queue: Vec::new(),
            optimization_settings: OptimizationSettings::default(),
        }
    }

    /// Convert linear index to 3D coordinates
    fn index_to_coordinates(
        index: usize,
        dimensions: &(usize, usize, usize),
    ) -> (usize, usize, usize) {
        let z_idx = index / (dimensions.0 * dimensions.1);
        let y_idx = (index % (dimensions.0 * dimensions.1)) / dimensions.0;
        let x_idx = index % dimensions.0;
        (x_idx, y_idx, z_idx)
    }

    /// Convert 3D coordinates to linear index
    fn coordinates_to_index(
        x: usize,
        y: usize,
        z: usize,
        dimensions: &(usize, usize, usize),
    ) -> usize {
        z * dimensions.0 * dimensions.1 + y * dimensions.0 + x
    }

    /// Load atoms into specified tweezers
    pub async fn load_atoms(&mut self, tweezer_ids: &[usize]) -> DeviceResult<LoadingResult> {
        let mut results = HashMap::new();
        let start_time = std::time::SystemTime::now();

        for &tweezer_id in tweezer_ids {
            let loading_result = self.load_single_atom(tweezer_id).await?;
            results.insert(tweezer_id, loading_result);
        }

        // Update loading statistics
        self.update_loading_statistics(&results, start_time);

        Ok(LoadingResult {
            attempted_tweezers: tweezer_ids.to_vec(),
            successful_loads: results.values().filter(|&&success| success).count(),
            failed_loads: results.values().filter(|&&success| !success).count(),
            individual_results: results,
            loading_time: start_time.elapsed().unwrap_or(Duration::ZERO),
        })
    }

    /// Load a single atom into a tweezer
    async fn load_single_atom(&mut self, tweezer_id: usize) -> DeviceResult<bool> {
        // First, get the tweezer info for loading calculation
        let (loading_attempts, base_success_rate) = {
            let tweezer = self.state.tweezers.get(&tweezer_id).ok_or_else(|| {
                DeviceError::InvalidInput(format!("Tweezer {} not found", tweezer_id))
            })?;
            (
                tweezer.loading_attempts,
                self.state.config.loading_efficiency,
            )
        };

        // Calculate loading success
        let loading_success =
            self.simulate_atom_loading_from_params(loading_attempts, base_success_rate);

        // Now update the tweezer
        let tweezer = self.state.tweezers.get_mut(&tweezer_id).ok_or_else(|| {
            DeviceError::InvalidInput(format!("Tweezer {} not found", tweezer_id))
        })?;

        tweezer.loading_attempts += 1;
        tweezer.last_update = std::time::SystemTime::now();
        tweezer.atom_state = if loading_success {
            AtomState::Loaded
        } else {
            AtomState::LoadingFailed
        };

        Ok(loading_success)
    }

    /// Simulate atom loading (mock implementation)
    fn simulate_atom_loading(&self, tweezer: &TweezerInfo) -> bool {
        self.simulate_atom_loading_from_params(
            tweezer.loading_attempts,
            self.state.config.loading_efficiency,
        )
    }

    /// Simulate atom loading from parameters
    fn simulate_atom_loading_from_params(
        &self,
        loading_attempts: usize,
        base_success_rate: f64,
    ) -> bool {
        // Simple mock based on loading efficiency and number of attempts
        let attempt_penalty = 0.05 * (loading_attempts as f64).max(1.0);
        let effective_success_rate = (base_success_rate - attempt_penalty).max(0.1);

        // Mock random decision
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        loading_attempts.hash(&mut hasher);
        let hash = hasher.finish();
        let random_value = (hash % 1000) as f64 / 1000.0;

        random_value < effective_success_rate
    }

    /// Move an atom from one position to another
    pub async fn move_atom(
        &mut self,
        tweezer_id: usize,
        target_position: TweezerPosition,
        parameters: MovementParameters,
    ) -> DeviceResult<String> {
        let tweezer = self.state.tweezers.get(&tweezer_id).ok_or_else(|| {
            DeviceError::InvalidInput(format!("Tweezer {} not found", tweezer_id))
        })?;

        if tweezer.atom_state != AtomState::Loaded {
            return Err(DeviceError::InvalidInput(
                "Cannot move atom: no atom loaded in tweezer".to_string(),
            ));
        }

        let movement_time =
            self.calculate_movement_time(&tweezer.position, &target_position, &parameters);
        let operation_id = format!("move_{}", uuid::Uuid::new_v4());

        let movement_op = MovementOperation {
            operation_id: operation_id.clone(),
            tweezer_id,
            start_position: tweezer.position,
            end_position: target_position,
            parameters,
            start_time: std::time::SystemTime::now(),
            expected_completion: std::time::SystemTime::now() + movement_time,
            status: MovementStatus::Queued,
        };

        self.movement_queue.push(movement_op);
        Ok(operation_id)
    }

    /// Calculate movement time based on distance and parameters
    fn calculate_movement_time(
        &self,
        start: &TweezerPosition,
        end: &TweezerPosition,
        parameters: &MovementParameters,
    ) -> Duration {
        let distance = Self::calculate_distance(start, end);
        let time_seconds = distance / parameters.speed;
        Duration::from_secs_f64(time_seconds)
    }

    /// Calculate distance between two positions
    fn calculate_distance(pos1: &TweezerPosition, pos2: &TweezerPosition) -> f64 {
        let dx = pos2.x - pos1.x;
        let dy = pos2.y - pos1.y;
        let dz = pos2.z - pos1.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Process movement operations
    pub async fn process_movements(&mut self) -> DeviceResult<()> {
        let mut operations_to_start = Vec::new();
        let mut operations_to_complete = Vec::new();
        let mut completed_indices = Vec::new();

        // First pass: identify operations that need processing
        for (i, movement) in self.movement_queue.iter_mut().enumerate() {
            match movement.status {
                MovementStatus::Queued => {
                    movement.status = MovementStatus::InProgress;
                    operations_to_start.push(movement.clone());
                }
                MovementStatus::InProgress => {
                    if std::time::SystemTime::now() >= movement.expected_completion {
                        movement.status = MovementStatus::Completed;
                        operations_to_complete.push(movement.clone());
                        completed_indices.push(i);
                    }
                }
                _ => {}
            }
        }

        // Second pass: execute the operations
        for operation in operations_to_start {
            self.start_movement_execution(&operation).await?;
        }

        for operation in operations_to_complete {
            self.complete_movement(&operation)?;
        }

        // Remove completed operations
        for &i in completed_indices.iter().rev() {
            self.movement_queue.remove(i);
        }

        Ok(())
    }

    /// Start executing a movement operation
    async fn start_movement_execution(&mut self, movement: &MovementOperation) -> DeviceResult<()> {
        // In real implementation, this would start the actual hardware movement
        // For now, just update the target position
        if let Some(tweezer) = self.state.tweezers.get_mut(&movement.tweezer_id) {
            tweezer.target_position = Some(movement.end_position);
        }
        Ok(())
    }

    /// Complete a movement operation
    fn complete_movement(&mut self, movement: &MovementOperation) -> DeviceResult<()> {
        if let Some(tweezer) = self.state.tweezers.get_mut(&movement.tweezer_id) {
            tweezer.position = movement.end_position;
            tweezer.target_position = None;
            tweezer.last_update = std::time::SystemTime::now();
        }
        Ok(())
    }

    /// Update loading statistics
    fn update_loading_statistics(
        &mut self,
        results: &HashMap<usize, bool>,
        start_time: std::time::SystemTime,
    ) {
        let successful = results.values().filter(|&&success| success).count();
        let failed = results.values().filter(|&&success| !success).count();
        let loading_time = start_time.elapsed().unwrap_or(Duration::ZERO);

        self.state.loading_stats.total_attempts += results.len();
        self.state.loading_stats.successful_loads += successful;
        self.state.loading_stats.failed_loads += failed;

        // Update average loading time
        let total_ops = self.state.loading_stats.total_attempts;
        let current_avg = self.state.loading_stats.average_loading_time;
        let new_avg = (current_avg * (total_ops - 1) as u32 + loading_time) / total_ops as u32;
        self.state.loading_stats.average_loading_time = new_avg;

        // Update fill factor
        let loaded_count = self
            .state
            .tweezers
            .values()
            .filter(|t| t.atom_state == AtomState::Loaded)
            .count();
        self.state.loading_stats.fill_factor =
            loaded_count as f64 / self.state.tweezers.len() as f64;
    }

    /// Get current array state
    pub fn get_array_state(&self) -> &TweezerArrayState {
        &self.state
    }

    /// Get loading statistics
    pub fn get_loading_statistics(&self) -> &LoadingStatistics {
        &self.state.loading_stats
    }

    /// Get atom positions
    pub fn get_atom_positions(&self) -> Vec<(usize, TweezerPosition)> {
        self.state
            .tweezers
            .iter()
            .filter(|(_, tweezer)| tweezer.atom_state == AtomState::Loaded)
            .map(|(&id, tweezer)| (id, tweezer.position))
            .collect()
    }

    /// Optimize array configuration
    pub async fn optimize_array(&mut self) -> DeviceResult<OptimizationResult> {
        let mut optimizations_applied = Vec::new();

        if self.optimization_settings.auto_reload {
            let reload_count = self.auto_reload_failed_tweezers().await?;
            if reload_count > 0 {
                optimizations_applied.push(format!("Reloaded {} failed tweezers", reload_count));
            }
        }

        if self.optimization_settings.position_optimization {
            let position_adjustments = self.optimize_positions().await?;
            if position_adjustments > 0 {
                optimizations_applied.push(format!("Adjusted {} positions", position_adjustments));
            }
        }

        if self.optimization_settings.power_optimization {
            let power_adjustments = self.optimize_power_levels().await?;
            if power_adjustments > 0 {
                optimizations_applied.push(format!("Optimized {} power levels", power_adjustments));
            }
        }

        Ok(OptimizationResult {
            optimizations_applied,
            fill_factor_improvement: 0.0, // Would calculate actual improvement
            loading_efficiency_improvement: 0.0,
        })
    }

    /// Auto-reload failed tweezers
    async fn auto_reload_failed_tweezers(&mut self) -> DeviceResult<usize> {
        let failed_tweezer_ids: Vec<usize> = self
            .state
            .tweezers
            .iter()
            .filter(|(_, tweezer)| {
                tweezer.atom_state == AtomState::LoadingFailed
                    && tweezer.loading_attempts < self.optimization_settings.max_reload_attempts
            })
            .map(|(&id, _)| id)
            .collect();

        if !failed_tweezer_ids.is_empty() {
            self.load_atoms(&failed_tweezer_ids).await?;
        }

        Ok(failed_tweezer_ids.len())
    }

    /// Optimize atom positions
    async fn optimize_positions(&mut self) -> DeviceResult<usize> {
        // Mock implementation - would implement actual position optimization
        Ok(0)
    }

    /// Optimize laser power levels
    async fn optimize_power_levels(&mut self) -> DeviceResult<usize> {
        // Mock implementation - would implement actual power optimization
        Ok(0)
    }
}

/// Loading operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingResult {
    /// Tweezers that were attempted
    pub attempted_tweezers: Vec<usize>,
    /// Number of successful loads
    pub successful_loads: usize,
    /// Number of failed loads
    pub failed_loads: usize,
    /// Individual results for each tweezer
    pub individual_results: HashMap<usize, bool>,
    /// Total loading time
    pub loading_time: Duration,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimizations that were applied
    pub optimizations_applied: Vec<String>,
    /// Fill factor improvement
    pub fill_factor_improvement: f64,
    /// Loading efficiency improvement
    pub loading_efficiency_improvement: f64,
}

impl Default for TweezerArrayConfig {
    fn default() -> Self {
        Self {
            array_dimensions: (10, 10, 1),
            tweezer_spacing: (5.0, 5.0, 0.0),
            laser_wavelength: 1064.0,
            beam_waist: 1.0,
            max_laser_power: 100.0,
            trap_depth: 1000.0,
            loading_efficiency: 0.8,
            movement_precision: 10.0,
        }
    }
}

impl Default for LoadingStatistics {
    fn default() -> Self {
        Self {
            total_attempts: 0,
            successful_loads: 0,
            failed_loads: 0,
            fill_factor: 0.0,
            average_loading_time: Duration::ZERO,
        }
    }
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            auto_reload: true,
            max_reload_attempts: 3,
            position_optimization: false,
            power_optimization: false,
            optimization_interval: Duration::from_secs(60),
        }
    }
}

impl Default for MovementParameters {
    fn default() -> Self {
        Self {
            speed: 10.0,       // μm/s
            acceleration: 5.0, // μm/s²
            trajectory: MovementTrajectory::Linear,
            power_ramping: PowerRampingConfig::default(),
        }
    }
}

impl Default for PowerRampingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_factor: 1.0,
            final_factor: 1.0,
            ramping_duration: Duration::from_millis(100),
        }
    }
}

/// Create a basic tweezer array configuration
pub fn create_basic_array_config(rows: usize, cols: usize, spacing: f64) -> TweezerArrayConfig {
    TweezerArrayConfig {
        array_dimensions: (rows, cols, 1),
        tweezer_spacing: (spacing, spacing, 0.0),
        ..Default::default()
    }
}

/// Create movement parameters for fast movement
pub fn create_fast_movement_params() -> MovementParameters {
    MovementParameters {
        speed: 50.0,
        acceleration: 20.0,
        trajectory: MovementTrajectory::SCurve,
        power_ramping: PowerRampingConfig {
            enabled: true,
            initial_factor: 1.2,
            final_factor: 1.0,
            ramping_duration: Duration::from_millis(50),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tweezer_array_creation() {
        let config = TweezerArrayConfig::default();
        let manager = TweezerArrayManager::new(config);

        assert_eq!(manager.state.tweezers.len(), 100); // 10x10x1 array
        assert!(manager
            .state
            .tweezers
            .values()
            .all(|t| t.atom_state == AtomState::Empty));
    }

    #[test]
    fn test_coordinate_conversion() {
        let dimensions = (3, 3, 2);

        // Test index to coordinates
        let (x, y, z) = TweezerArrayManager::index_to_coordinates(10, &dimensions);
        assert_eq!((x, y, z), (1, 0, 1));

        // Test coordinates to index
        let index = TweezerArrayManager::coordinates_to_index(1, 0, 1, &dimensions);
        assert_eq!(index, 10);
    }

    #[test]
    fn test_distance_calculation() {
        let pos1 = TweezerPosition {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let pos2 = TweezerPosition {
            x: 3.0,
            y: 4.0,
            z: 0.0,
        };

        let distance = TweezerArrayManager::calculate_distance(&pos1, &pos2);
        assert_eq!(distance, 5.0);
    }
}

// Mock UUID implementation
mod uuid {
    use std::fmt;

    pub struct Uuid([u8; 16]);

    impl Uuid {
        pub fn new_v4() -> Self {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::SystemTime;

            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            let hash = hasher.finish();

            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&hash.to_le_bytes());
            bytes[8..16].copy_from_slice(&hash.to_be_bytes());

            Self(bytes)
        }
    }

    impl fmt::Display for Uuid {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                self.0[0], self.0[1], self.0[2], self.0[3],
                self.0[4], self.0[5],
                self.0[6], self.0[7],
                self.0[8], self.0[9],
                self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15])
        }
    }
}
