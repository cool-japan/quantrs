//! Real-time performance monitoring

use super::*;

/// Performance monitor trait
pub trait PerformanceMonitor: Send + Sync + std::fmt::Debug {
    /// Start monitoring
    fn start_monitoring(&mut self) -> Result<(), AnalysisError>;
    
    /// Stop monitoring
    fn stop_monitoring(&mut self) -> Result<(), AnalysisError>;
    
    /// Get current metrics
    fn get_current_metrics(&self) -> Result<HashMap<String, f64>, AnalysisError>;
    
    /// Get monitor name
    fn get_monitor_name(&self) -> &str;
    
    /// Check if monitor is active
    fn is_active(&self) -> bool;
}

/// CPU monitor implementation
#[derive(Debug)]
pub struct CpuMonitor {
    active: bool,
}

impl CpuMonitor {
    pub fn new() -> Self {
        Self { active: false }
    }
}

impl PerformanceMonitor for CpuMonitor {
    fn start_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = true;
        Ok(())
    }
    
    fn stop_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = false;
        Ok(())
    }
    
    fn get_current_metrics(&self) -> Result<HashMap<String, f64>, AnalysisError> {
        if !self.active {
            return Err(AnalysisError::DataCollectionError("Monitor not active".to_string()));
        }
        
        let mut metrics = HashMap::new();
        metrics.insert("cpu_utilization".to_string(), 45.5); // Mock value
        Ok(metrics)
    }
    
    fn get_monitor_name(&self) -> &str {
        "CPU Monitor"
    }
    
    fn is_active(&self) -> bool {
        self.active
    }
}

/// Memory monitor implementation
#[derive(Debug)]
pub struct MemoryMonitor {
    active: bool,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self { active: false }
    }
}

impl PerformanceMonitor for MemoryMonitor {
    fn start_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = true;
        Ok(())
    }
    
    fn stop_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = false;
        Ok(())
    }
    
    fn get_current_metrics(&self) -> Result<HashMap<String, f64>, AnalysisError> {
        if !self.active {
            return Err(AnalysisError::DataCollectionError("Monitor not active".to_string()));
        }
        
        let mut metrics = HashMap::new();
        metrics.insert("memory_utilization".to_string(), 65.2); // Mock value
        Ok(metrics)
    }
    
    fn get_monitor_name(&self) -> &str {
        "Memory Monitor"
    }
    
    fn is_active(&self) -> bool {
        self.active
    }
}

/// I/O monitor implementation
#[derive(Debug)]
pub struct IoMonitor {
    active: bool,
}

impl IoMonitor {
    pub fn new() -> Self {
        Self { active: false }
    }
}

impl PerformanceMonitor for IoMonitor {
    fn start_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = true;
        Ok(())
    }
    
    fn stop_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = false;
        Ok(())
    }
    
    fn get_current_metrics(&self) -> Result<HashMap<String, f64>, AnalysisError> {
        if !self.active {
            return Err(AnalysisError::DataCollectionError("Monitor not active".to_string()));
        }
        
        let mut metrics = HashMap::new();
        metrics.insert("io_utilization".to_string(), 25.8); // Mock value
        Ok(metrics)
    }
    
    fn get_monitor_name(&self) -> &str {
        "I/O Monitor"
    }
    
    fn is_active(&self) -> bool {
        self.active
    }
}

/// Network monitor implementation
#[derive(Debug)]
pub struct NetworkMonitor {
    active: bool,
}

impl NetworkMonitor {
    pub fn new() -> Self {
        Self { active: false }
    }
}

impl PerformanceMonitor for NetworkMonitor {
    fn start_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = true;
        Ok(())
    }
    
    fn stop_monitoring(&mut self) -> Result<(), AnalysisError> {
        self.active = false;
        Ok(())
    }
    
    fn get_current_metrics(&self) -> Result<HashMap<String, f64>, AnalysisError> {
        if !self.active {
            return Err(AnalysisError::DataCollectionError("Monitor not active".to_string()));
        }
        
        let mut metrics = HashMap::new();
        metrics.insert("network_utilization".to_string(), 15.3); // Mock value
        Ok(metrics)
    }
    
    fn get_monitor_name(&self) -> &str {
        "Network Monitor"
    }
    
    fn is_active(&self) -> bool {
        self.active
    }
}
