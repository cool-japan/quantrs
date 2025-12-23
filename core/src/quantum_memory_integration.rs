//! Quantum Memory Integration
//!
//! Persistent quantum state storage with error correction and
//! advanced memory management for quantum computing systems.

use crate::error::QuantRS2Error;

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Quantum memory interface for persistent state storage
#[derive(Debug)]
pub struct QuantumMemory {
    pub memory_id: Uuid,
    pub storage_layers: Vec<Arc<dyn QuantumStorageLayer>>,
    pub cache: Arc<RwLock<QuantumCache>>,
    pub error_correction: QuantumMemoryErrorCorrection,
    pub coherence_manager: CoherenceManager,
    pub access_controller: MemoryAccessController,
}

/// Trait for quantum storage implementations
pub trait QuantumStorageLayer: Send + Sync + std::fmt::Debug {
    fn store_state(&self, state_id: Uuid, state: &QuantumState) -> Result<(), QuantRS2Error>;
    fn retrieve_state(&self, state_id: Uuid) -> Result<Option<QuantumState>, QuantRS2Error>;
    fn delete_state(&self, state_id: Uuid) -> Result<(), QuantRS2Error>;
    fn list_states(&self) -> Result<Vec<Uuid>, QuantRS2Error>;
    fn get_storage_info(&self) -> StorageLayerInfo;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub state_id: Uuid,
    pub amplitudes: Vec<Complex64>,
    pub qubit_count: usize,
    pub creation_time: SystemTime,
    pub last_access: SystemTime,
    pub coherence_time: Duration,
    pub fidelity: f64,
    pub metadata: StateMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub creator: Option<String>,
    pub access_count: u64,
    pub compression_ratio: f64,
    pub entanglement_entropy: f64,
}

#[derive(Debug, Clone)]
pub struct StorageLayerInfo {
    pub layer_type: StorageLayerType,
    pub capacity: usize, // Number of states
    pub latency: Duration,
    pub reliability: f64,
    pub energy_cost: f64, // Relative energy cost
}

#[derive(Debug, Clone)]
pub enum StorageLayerType {
    UltraFast,   // RAM-like
    Fast,        // SSD-like
    Persistent,  // HDD-like
    Archive,     // Tape-like
    Distributed, // Network storage
}

impl QuantumMemory {
    /// Create a new quantum memory system
    pub fn new() -> Self {
        let memory_id = Uuid::new_v4();
        let cache = Arc::new(RwLock::new(QuantumCache::new(1000))); // 1000 state cache

        Self {
            memory_id,
            storage_layers: Vec::new(),
            cache,
            error_correction: QuantumMemoryErrorCorrection::new(),
            coherence_manager: CoherenceManager::new(),
            access_controller: MemoryAccessController::new(),
        }
    }

    /// Add a storage layer to the memory hierarchy
    pub fn add_storage_layer(&mut self, layer: Arc<dyn QuantumStorageLayer>) {
        self.storage_layers.push(layer);
        // Sort layers by access speed (fastest first)
        self.storage_layers.sort_by(|a, b| {
            a.get_storage_info()
                .latency
                .cmp(&b.get_storage_info().latency)
        });
    }

    /// Store a quantum state with automatic tier selection
    pub async fn store_state(&self, state: QuantumState) -> Result<Uuid, QuantRS2Error> {
        // Check access permissions
        self.access_controller.check_write_permission(&state)?;

        // Apply error correction encoding
        let encoded_state = self.error_correction.encode_state(&state)?;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(state.state_id, encoded_state.clone());
        }

        // Store in appropriate layer based on access pattern prediction
        let target_layer = self.select_storage_layer(&encoded_state).await?;
        target_layer.store_state(encoded_state.state_id, &encoded_state)?;

        // Start coherence tracking
        self.coherence_manager
            .start_tracking(encoded_state.state_id, encoded_state.coherence_time);

        Ok(encoded_state.state_id)
    }

    /// Retrieve a quantum state with automatic error correction
    pub async fn retrieve_state(
        &self,
        state_id: Uuid,
    ) -> Result<Option<QuantumState>, QuantRS2Error> {
        // Check access permissions
        self.access_controller.check_read_permission(state_id)?;

        // Check cache first
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(cached_state) = cache.get(&state_id) {
                // Update access statistics
                self.update_access_stats(state_id).await;
                return Ok(Some(self.error_correction.decode_state(cached_state)?));
            }
        }

        // Search through storage layers
        for layer in &self.storage_layers {
            if let Some(encoded_state) = layer.retrieve_state(state_id)? {
                // Check coherence
                if self.coherence_manager.is_coherent(state_id) {
                    // Decode and cache
                    let decoded_state = self.error_correction.decode_state(&encoded_state)?;

                    {
                        let mut cache = self.cache.write().unwrap();
                        cache.insert(state_id, encoded_state);
                    }

                    self.update_access_stats(state_id).await;
                    return Ok(Some(decoded_state));
                }
                // State has decoherent, remove it
                layer.delete_state(state_id)?;
                return Err(QuantRS2Error::QuantumDecoherence(format!(
                    "State {} has decoherent",
                    state_id
                )));
            }
        }

        Ok(None)
    }

    /// Delete a quantum state from all storage layers
    pub async fn delete_state(&self, state_id: Uuid) -> Result<(), QuantRS2Error> {
        // Check permissions
        self.access_controller.check_delete_permission(state_id)?;

        // Remove from cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.remove(&state_id);
        }

        // Remove from all storage layers
        for layer in &self.storage_layers {
            let _ = layer.delete_state(state_id); // Ignore errors for layers that don't have the state
        }

        // Stop coherence tracking
        self.coherence_manager.stop_tracking(state_id);

        Ok(())
    }

    /// Select optimal storage layer for a state
    async fn select_storage_layer(
        &self,
        state: &QuantumState,
    ) -> Result<Arc<dyn QuantumStorageLayer>, QuantRS2Error> {
        // Analyze state characteristics
        let access_pattern = self.predict_access_pattern(state).await;
        let _importance_score = self.calculate_importance_score(state);

        // Select layer based on predicted usage
        for layer in &self.storage_layers {
            let info = layer.get_storage_info();

            match (&access_pattern, info.layer_type.clone()) {
                (
                    &AccessPattern::Frequent,
                    StorageLayerType::UltraFast | StorageLayerType::Fast,
                ) => {
                    return Ok(layer.clone());
                }
                (
                    &AccessPattern::Moderate,
                    StorageLayerType::Fast | StorageLayerType::Persistent,
                ) => {
                    return Ok(layer.clone());
                }
                (
                    &AccessPattern::Rare,
                    StorageLayerType::Persistent | StorageLayerType::Archive,
                ) => {
                    return Ok(layer.clone());
                }
                _ => {}
            }
        }

        // Fallback to first available layer
        self.storage_layers.first().cloned().ok_or_else(|| {
            QuantRS2Error::NoStorageAvailable("No storage layers configured".to_string())
        })
    }

    /// Predict access pattern for a state
    async fn predict_access_pattern(&self, state: &QuantumState) -> AccessPattern {
        // Simple heuristic-based prediction
        // In practice, this could use ML models

        let recency_factor = state
            .last_access
            .elapsed()
            .unwrap_or(Duration::ZERO)
            .as_secs() as f64
            / 3600.0; // Hours since last access

        let access_frequency = state.metadata.access_count as f64
            / state
                .creation_time
                .elapsed()
                .unwrap_or(Duration::from_secs(1))
                .as_secs() as f64;

        if access_frequency > 0.1 && recency_factor < 1.0 {
            AccessPattern::Frequent
        } else if access_frequency > 0.01 && recency_factor < 24.0 {
            AccessPattern::Moderate
        } else {
            AccessPattern::Rare
        }
    }

    /// Calculate importance score for a state
    fn calculate_importance_score(&self, state: &QuantumState) -> f64 {
        let mut score = 0.0;

        // Fidelity contributes to importance
        score += state.fidelity * 10.0;

        // Entanglement entropy (higher = more complex/important)
        score += state.metadata.entanglement_entropy * 5.0;

        // Recent access increases importance
        let hours_since_access = state
            .last_access
            .elapsed()
            .unwrap_or(Duration::ZERO)
            .as_secs() as f64
            / 3600.0;
        score += (24.0 - hours_since_access.min(24.0)) / 24.0 * 3.0;

        // Access frequency
        score += (state.metadata.access_count as f64).ln().max(0.0);

        score
    }

    /// Update access statistics for a state
    async fn update_access_stats(&self, _state_id: Uuid) {
        // This would update access patterns in a real implementation
        // For now, it's a placeholder
    }

    /// Perform garbage collection on expired/decoherent states
    pub async fn garbage_collect(&self) -> Result<GarbageCollectionResult, QuantRS2Error> {
        let mut collected_states = 0;
        let mut freed_space = 0;
        let start_time = Instant::now();

        // Get list of all states from all layers
        let mut all_states = Vec::new();
        for layer in &self.storage_layers {
            let layer_states = layer.list_states()?;
            all_states.extend(layer_states);
        }

        // Check each state for coherence and importance
        for state_id in all_states {
            if !self.coherence_manager.is_coherent(state_id) {
                // State has decoherent, remove it
                self.delete_state(state_id).await?;
                collected_states += 1;
                freed_space += 1; // Simplified space calculation
            }
        }

        // Compact cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.compact();
        }

        Ok(GarbageCollectionResult {
            collected_states,
            freed_space,
            execution_time: start_time.elapsed(),
        })
    }
}

/// Cache for frequently accessed quantum states
#[derive(Debug)]
pub struct QuantumCache {
    cache: HashMap<Uuid, QuantumState>,
    access_order: Vec<Uuid>,
    max_size: usize,
}

impl QuantumCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
        }
    }

    pub fn insert(&mut self, state_id: Uuid, state: QuantumState) {
        // Remove if already exists
        if self.cache.contains_key(&state_id) {
            self.remove(&state_id);
        }

        // Add to cache
        self.cache.insert(state_id, state);
        self.access_order.push(state_id);

        // Evict if necessary (LRU)
        while self.cache.len() > self.max_size {
            if let Some(oldest) = self.access_order.first().cloned() {
                self.remove(&oldest);
            }
        }
    }

    pub fn get(&mut self, state_id: &Uuid) -> Option<&QuantumState> {
        if self.cache.contains_key(state_id) {
            // Move to end (most recently used)
            self.access_order.retain(|&id| id != *state_id);
            self.access_order.push(*state_id);
            self.cache.get(state_id)
        } else {
            None
        }
    }

    pub fn remove(&mut self, state_id: &Uuid) {
        self.cache.remove(state_id);
        self.access_order.retain(|&id| id != *state_id);
    }

    pub fn compact(&mut self) {
        // Remove any inconsistencies
        self.access_order.retain(|id| self.cache.contains_key(id));
    }
}

/// Error correction for quantum memory
#[derive(Debug)]
pub struct QuantumMemoryErrorCorrection {
    pub code_type: QuantumErrorCode,
    pub syndrome_table: HashMap<Vec<bool>, Array1<Complex64>>,
    pub encoding_overhead: f64,
}

#[derive(Debug, Clone)]
pub enum QuantumErrorCode {
    ShorCode,
    SteaneCode,
    SurfaceCode {
        distance: usize,
    },
    ColorCode {
        distance: usize,
    },
    Custom {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

impl QuantumMemoryErrorCorrection {
    pub fn new() -> Self {
        Self {
            code_type: QuantumErrorCode::SteaneCode,
            syndrome_table: HashMap::new(),
            encoding_overhead: 7.0, // Steane code overhead
        }
    }

    /// Encode a quantum state with error correction
    pub fn encode_state(&self, state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        match &self.code_type {
            QuantumErrorCode::SteaneCode => self.encode_steane(state),
            QuantumErrorCode::ShorCode => self.encode_shor(state),
            QuantumErrorCode::SurfaceCode { distance } => self.encode_surface(state, *distance),
            QuantumErrorCode::ColorCode { distance } => self.encode_color(state, *distance),
            QuantumErrorCode::Custom { .. } => self.encode_custom(state),
        }
    }

    /// Decode a quantum state with error correction
    pub fn decode_state(
        &self,
        encoded_state: &QuantumState,
    ) -> Result<QuantumState, QuantRS2Error> {
        match &self.code_type {
            QuantumErrorCode::SteaneCode => self.decode_steane(encoded_state),
            QuantumErrorCode::ShorCode => self.decode_shor(encoded_state),
            QuantumErrorCode::SurfaceCode { distance } => {
                self.decode_surface(encoded_state, *distance)
            }
            QuantumErrorCode::ColorCode { distance } => self.decode_color(encoded_state, *distance),
            QuantumErrorCode::Custom { .. } => self.decode_custom(encoded_state),
        }
    }

    /// Steane code encoding
    fn encode_steane(&self, state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        // Simplified Steane code implementation
        let encoded_amplitudes = self.apply_steane_encoding(&state.amplitudes)?;

        Ok(QuantumState {
            state_id: state.state_id,
            amplitudes: encoded_amplitudes,
            qubit_count: state.qubit_count * 7, // Steane code uses 7 qubits per logical qubit
            creation_time: state.creation_time,
            last_access: SystemTime::now(),
            coherence_time: state.coherence_time,
            fidelity: state.fidelity * 0.99, // Small fidelity cost for encoding
            metadata: state.metadata.clone(),
        })
    }

    /// Apply Steane encoding matrix
    fn apply_steane_encoding(
        &self,
        amplitudes: &[Complex64],
    ) -> Result<Vec<Complex64>, QuantRS2Error> {
        // Simplified implementation - in practice would use proper Steane encoding
        let mut encoded = Vec::new();

        for amp in amplitudes {
            // Replicate amplitude across 7 qubits with redundancy
            for _ in 0..7 {
                encoded.push(*amp / (7.0_f64.sqrt()));
            }
        }

        Ok(encoded)
    }

    /// Steane code decoding
    fn decode_steane(&self, encoded_state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        // Simplified decoding - measure syndromes and correct
        let decoded_amplitudes = self.apply_steane_decoding(&encoded_state.amplitudes)?;

        Ok(QuantumState {
            state_id: encoded_state.state_id,
            amplitudes: decoded_amplitudes,
            qubit_count: encoded_state.qubit_count / 7,
            creation_time: encoded_state.creation_time,
            last_access: SystemTime::now(),
            coherence_time: encoded_state.coherence_time,
            fidelity: encoded_state.fidelity,
            metadata: encoded_state.metadata.clone(),
        })
    }

    /// Apply Steane decoding
    fn apply_steane_decoding(
        &self,
        encoded_amplitudes: &[Complex64],
    ) -> Result<Vec<Complex64>, QuantRS2Error> {
        // Simplified majority voting
        let mut decoded = Vec::new();

        for chunk in encoded_amplitudes.chunks(7) {
            if chunk.len() == 7 {
                // Simple majority decoding
                let recovered_amp = chunk.iter().sum::<Complex64>() / Complex64::new(7.0, 0.0)
                    * Complex64::new(7.0_f64.sqrt(), 0.0);
                decoded.push(recovered_amp);
            }
        }

        Ok(decoded)
    }

    /// Placeholder implementations for other codes
    fn encode_shor(&self, state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        // Similar to Steane but with 9 qubits per logical qubit
        let mut encoded_state = state.clone();
        encoded_state.qubit_count *= 9;
        Ok(encoded_state)
    }

    fn decode_shor(&self, encoded_state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        let mut decoded_state = encoded_state.clone();
        decoded_state.qubit_count /= 9;
        Ok(decoded_state)
    }

    fn encode_surface(
        &self,
        state: &QuantumState,
        distance: usize,
    ) -> Result<QuantumState, QuantRS2Error> {
        let mut encoded_state = state.clone();
        encoded_state.qubit_count *= distance * distance;
        Ok(encoded_state)
    }

    fn decode_surface(
        &self,
        encoded_state: &QuantumState,
        distance: usize,
    ) -> Result<QuantumState, QuantRS2Error> {
        let mut decoded_state = encoded_state.clone();
        decoded_state.qubit_count /= distance * distance;
        Ok(decoded_state)
    }

    fn encode_color(
        &self,
        state: &QuantumState,
        distance: usize,
    ) -> Result<QuantumState, QuantRS2Error> {
        let mut encoded_state = state.clone();
        encoded_state.qubit_count *= distance * distance * 2;
        Ok(encoded_state)
    }

    fn decode_color(
        &self,
        encoded_state: &QuantumState,
        distance: usize,
    ) -> Result<QuantumState, QuantRS2Error> {
        let mut decoded_state = encoded_state.clone();
        decoded_state.qubit_count /= distance * distance * 2;
        Ok(decoded_state)
    }

    fn encode_custom(&self, state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        Ok(state.clone())
    }

    fn decode_custom(&self, encoded_state: &QuantumState) -> Result<QuantumState, QuantRS2Error> {
        Ok(encoded_state.clone())
    }
}

/// Coherence management for quantum states
#[derive(Debug)]
pub struct CoherenceManager {
    coherence_tracking: Arc<Mutex<HashMap<Uuid, CoherenceInfo>>>,
}

#[derive(Debug, Clone)]
pub struct CoherenceInfo {
    pub creation_time: Instant,
    pub coherence_time: Duration,
    pub last_check: Instant,
    pub predicted_fidelity: f64,
}

impl CoherenceManager {
    pub fn new() -> Self {
        Self {
            coherence_tracking: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start tracking coherence for a quantum state
    pub fn start_tracking(&self, state_id: Uuid, coherence_time: Duration) {
        let info = CoherenceInfo {
            creation_time: Instant::now(),
            coherence_time,
            last_check: Instant::now(),
            predicted_fidelity: 1.0,
        };

        self.coherence_tracking
            .lock()
            .unwrap()
            .insert(state_id, info);
    }

    /// Stop tracking coherence for a quantum state
    pub fn stop_tracking(&self, state_id: Uuid) {
        self.coherence_tracking.lock().unwrap().remove(&state_id);
    }

    /// Check if a quantum state is still coherent
    pub fn is_coherent(&self, state_id: Uuid) -> bool {
        if let Some(info) = self.coherence_tracking.lock().unwrap().get(&state_id) {
            let elapsed = info.creation_time.elapsed();
            elapsed < info.coherence_time
        } else {
            false // Not tracked = assume decoherent
        }
    }

    /// Get predicted fidelity for a quantum state
    pub fn get_predicted_fidelity(&self, state_id: Uuid) -> f64 {
        if let Some(info) = self.coherence_tracking.lock().unwrap().get(&state_id) {
            let elapsed = info.creation_time.elapsed();
            let decay_factor = elapsed.as_secs_f64() / info.coherence_time.as_secs_f64();

            // Exponential decay model
            (1.0 - decay_factor).max(0.0)
        } else {
            0.0
        }
    }
}

/// Access control for quantum memory
#[derive(Debug)]
pub struct MemoryAccessController {
    permissions: Arc<RwLock<HashMap<Uuid, StatePermissions>>>,
    access_log: Arc<Mutex<Vec<AccessLogEntry>>>,
}

#[derive(Debug, Clone)]
pub struct StatePermissions {
    pub read: bool,
    pub write: bool,
    pub delete: bool,
    pub owner: Option<String>,
    pub authorized_users: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AccessLogEntry {
    pub timestamp: SystemTime,
    pub state_id: Uuid,
    pub operation: AccessOperation,
    pub user: Option<String>,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub enum AccessOperation {
    Read,
    Write,
    Delete,
}

impl MemoryAccessController {
    pub fn new() -> Self {
        Self {
            permissions: Arc::new(RwLock::new(HashMap::new())),
            access_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Check read permission for a state
    pub fn check_read_permission(&self, state_id: Uuid) -> Result<(), QuantRS2Error> {
        let permissions = self.permissions.read().unwrap();
        if let Some(perms) = permissions.get(&state_id) {
            if perms.read {
                self.log_access(state_id, AccessOperation::Read, true);
                Ok(())
            } else {
                self.log_access(state_id, AccessOperation::Read, false);
                Err(QuantRS2Error::AccessDenied(
                    "Read access denied".to_string(),
                ))
            }
        } else {
            // Default: allow read if no permissions set
            Ok(())
        }
    }

    /// Check write permission for a state
    pub fn check_write_permission(&self, state: &QuantumState) -> Result<(), QuantRS2Error> {
        let permissions = self.permissions.read().unwrap();
        if let Some(perms) = permissions.get(&state.state_id) {
            if perms.write {
                self.log_access(state.state_id, AccessOperation::Write, true);
                Ok(())
            } else {
                self.log_access(state.state_id, AccessOperation::Write, false);
                Err(QuantRS2Error::AccessDenied(
                    "Write access denied".to_string(),
                ))
            }
        } else {
            // Default: allow write if no permissions set
            Ok(())
        }
    }

    /// Check delete permission for a state
    pub fn check_delete_permission(&self, state_id: Uuid) -> Result<(), QuantRS2Error> {
        let permissions = self.permissions.read().unwrap();
        if let Some(perms) = permissions.get(&state_id) {
            if perms.delete {
                self.log_access(state_id, AccessOperation::Delete, true);
                Ok(())
            } else {
                self.log_access(state_id, AccessOperation::Delete, false);
                Err(QuantRS2Error::AccessDenied(
                    "Delete access denied".to_string(),
                ))
            }
        } else {
            // Default: allow delete if no permissions set
            Ok(())
        }
    }

    /// Log access attempt
    fn log_access(&self, state_id: Uuid, operation: AccessOperation, success: bool) {
        let entry = AccessLogEntry {
            timestamp: SystemTime::now(),
            state_id,
            operation,
            user: None, // Would get from current context
            success,
        };

        self.access_log.lock().unwrap().push(entry);
    }
}

/// Storage layer implementations
#[derive(Debug)]
pub struct InMemoryStorage {
    states: Arc<RwLock<HashMap<Uuid, QuantumState>>>,
    info: StorageLayerInfo,
}

impl InMemoryStorage {
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            info: StorageLayerInfo {
                layer_type: StorageLayerType::UltraFast,
                capacity,
                latency: Duration::from_nanos(100),
                reliability: 0.99999,
                energy_cost: 1.0,
            },
        }
    }
}

impl QuantumStorageLayer for InMemoryStorage {
    fn store_state(&self, state_id: Uuid, state: &QuantumState) -> Result<(), QuantRS2Error> {
        let mut states = self.states.write().unwrap();
        if states.len() >= self.info.capacity {
            return Err(QuantRS2Error::StorageCapacityExceeded(
                "Memory storage full".to_string(),
            ));
        }
        states.insert(state_id, state.clone());
        Ok(())
    }

    fn retrieve_state(&self, state_id: Uuid) -> Result<Option<QuantumState>, QuantRS2Error> {
        let states = self.states.read().unwrap();
        Ok(states.get(&state_id).cloned())
    }

    fn delete_state(&self, state_id: Uuid) -> Result<(), QuantRS2Error> {
        let mut states = self.states.write().unwrap();
        states.remove(&state_id);
        Ok(())
    }

    fn list_states(&self) -> Result<Vec<Uuid>, QuantRS2Error> {
        let states = self.states.read().unwrap();
        Ok(states.keys().cloned().collect())
    }

    fn get_storage_info(&self) -> StorageLayerInfo {
        self.info.clone()
    }
}

/// Utility types and enums
#[derive(Debug, Clone)]
pub enum AccessPattern {
    Frequent,
    Moderate,
    Rare,
}

#[derive(Debug)]
pub struct GarbageCollectionResult {
    pub collected_states: usize,
    pub freed_space: usize,
    pub execution_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_memory_creation() {
        let memory = QuantumMemory::new();
        assert_eq!(memory.storage_layers.len(), 0);
    }

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = InMemoryStorage::new(100);

        let state = QuantumState {
            state_id: Uuid::new_v4(),
            amplitudes: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            qubit_count: 1,
            creation_time: SystemTime::now(),
            last_access: SystemTime::now(),
            coherence_time: Duration::from_millis(100),
            fidelity: 0.99,
            metadata: StateMetadata {
                name: Some("test_state".to_string()),
                description: None,
                tags: vec!["test".to_string()],
                creator: None,
                access_count: 0,
                compression_ratio: 1.0,
                entanglement_entropy: 0.0,
            },
        };

        let state_id = state.state_id;
        storage.store_state(state_id, &state).unwrap();

        let retrieved = storage.retrieve_state(state_id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().state_id, state_id);
    }

    #[tokio::test]
    async fn test_quantum_cache() {
        let mut cache = QuantumCache::new(2);

        let state1 = QuantumState {
            state_id: Uuid::new_v4(),
            amplitudes: vec![Complex64::new(1.0, 0.0)],
            qubit_count: 1,
            creation_time: SystemTime::now(),
            last_access: SystemTime::now(),
            coherence_time: Duration::from_millis(100),
            fidelity: 0.99,
            metadata: StateMetadata {
                name: None,
                description: None,
                tags: Vec::new(),
                creator: None,
                access_count: 0,
                compression_ratio: 1.0,
                entanglement_entropy: 0.0,
            },
        };

        let state2 = QuantumState {
            state_id: Uuid::new_v4(),
            ..state1.clone()
        };

        let state3 = QuantumState {
            state_id: Uuid::new_v4(),
            ..state1.clone()
        };

        cache.insert(state1.state_id, state1.clone());
        cache.insert(state2.state_id, state2.clone());
        assert_eq!(cache.cache.len(), 2);

        // This should evict state1 (LRU)
        cache.insert(state3.state_id, state3.clone());
        assert_eq!(cache.cache.len(), 2);
        assert!(!cache.cache.contains_key(&state1.state_id));
    }

    #[tokio::test]
    async fn test_coherence_manager() {
        let manager = CoherenceManager::new();
        let state_id = Uuid::new_v4();

        manager.start_tracking(state_id, Duration::from_millis(100));
        assert!(manager.is_coherent(state_id));

        // Simulate time passage
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!manager.is_coherent(state_id));
    }
}
