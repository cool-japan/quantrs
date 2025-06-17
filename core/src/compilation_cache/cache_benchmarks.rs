//! Benchmarks for compilation cache performance
//!
//! This module provides comprehensive benchmarks to measure cache performance,
//! compilation time savings, and memory efficiency.

use super::*;
use crate::{
    gate::{single::*, multi::*},
    qubit::QubitId,
};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of trials per test
    pub num_trials: usize,
    /// Number of different gates to test
    pub num_gates: usize,
    /// Enable persistent storage tests
    pub test_persistence: bool,
    /// Warmup trials before measurement
    pub warmup_trials: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_trials: 100,
            num_gates: 50,
            test_persistence: true,
            warmup_trials: 10,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Average compilation time without cache (microseconds)
    pub avg_compile_time_us: f64,
    /// Average cache hit time (microseconds)
    pub avg_cache_hit_time_us: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Total time saved (microseconds)
    pub total_time_saved_us: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// Disk usage (bytes)
    pub disk_usage_bytes: usize,
    /// Cache efficiency (time saved / memory used)
    pub cache_efficiency: f64,
}

/// Cache benchmark suite
pub struct CacheBenchmarks {
    config: BenchmarkConfig,
    temp_dir: TempDir,
}

impl CacheBenchmarks {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> QuantRS2Result<Self> {
        let temp_dir = TempDir::new()?;
        Ok(Self { config, temp_dir })
    }

    /// Run all benchmarks
    pub fn run_all_benchmarks(&self) -> QuantRS2Result<BenchmarkResults> {
        println!("Running cache benchmarks...");
        
        // Benchmark cache hit performance
        let hit_times = self.benchmark_cache_hits()?;
        
        // Benchmark compilation time savings
        let compile_savings = self.benchmark_compilation_savings()?;
        
        // Benchmark memory efficiency
        let memory_stats = self.benchmark_memory_usage()?;
        
        // Benchmark persistence performance
        let persistence_stats = if self.config.test_persistence {
            Some(self.benchmark_persistence()?)
        } else {
            None
        };
        
        // Compile results
        let results = BenchmarkResults {
            avg_compile_time_us: compile_savings.0,
            avg_cache_hit_time_us: hit_times,
            cache_hit_rate: compile_savings.1,
            total_time_saved_us: compile_savings.2,
            memory_usage_bytes: memory_stats.0,
            disk_usage_bytes: persistence_stats.map(|s| s.0).unwrap_or(0),
            cache_efficiency: compile_savings.2 / (memory_stats.0 as f64),
        };
        
        self.print_results(&results);
        Ok(results)
    }

    /// Benchmark cache hit times
    fn benchmark_cache_hits(&self) -> QuantRS2Result<f64> {
        let cache_config = CacheConfig {
            cache_dir: self.temp_dir.path().to_path_buf(),
            enable_persistence: false,
            async_writes: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(cache_config)?;
        let gate = Hadamard { target: QubitId(0) };
        
        // Warm up cache
        for _ in 0..self.config.warmup_trials {
            let _ = cache.get_or_compile(&gate, compile_single_qubit_gate)?;
        }
        
        // Measure cache hit times
        let mut hit_times = Vec::new();
        
        for _ in 0..self.config.num_trials {
            let start = Instant::now();
            let _ = cache.get_or_compile(&gate, compile_single_qubit_gate)?;
            let hit_time = start.elapsed().as_micros() as f64;
            hit_times.push(hit_time);
        }
        
        Ok(hit_times.iter().sum::<f64>() / hit_times.len() as f64)
    }

    /// Benchmark compilation time savings
    fn benchmark_compilation_savings(&self) -> QuantRS2Result<(f64, f64, f64)> {
        let cache_config = CacheConfig {
            cache_dir: self.temp_dir.path().to_path_buf(),
            enable_persistence: false,
            async_writes: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(cache_config)?;
        
        // Generate test gates
        let gates = self.generate_test_gates();
        
        // Measure compilation times without cache
        let mut compile_times = Vec::new();
        for gate in &gates {
            let start = Instant::now();
            let _ = compile_single_qubit_gate(gate.as_ref())?;
            let compile_time = start.elapsed().as_micros() as f64;
            compile_times.push(compile_time);
        }
        
        let avg_compile_time = compile_times.iter().sum::<f64>() / compile_times.len() as f64;
        
        // Warm up cache with all gates
        for gate in &gates {
            let _ = cache.get_or_compile(gate.as_ref(), compile_single_qubit_gate)?;
        }
        
        // Measure cache access times
        let mut cache_times = Vec::new();
        let mut hits = 0;
        let total_accesses = self.config.num_trials;
        
        for _ in 0..total_accesses {
            let gate_idx = fastrand::usize(..gates.len());
            let gate = &gates[gate_idx];
            
            let start = Instant::now();
            let _ = cache.get_or_compile(gate.as_ref(), compile_single_qubit_gate)?;
            let access_time = start.elapsed().as_micros() as f64;
            cache_times.push(access_time);
            
            // Check if it was a hit (access time should be much lower)
            if access_time < avg_compile_time * 0.1 {
                hits += 1;
            }
        }
        
        let hit_rate = hits as f64 / total_accesses as f64;
        let time_saved = hits as f64 * (avg_compile_time - cache_times.iter().sum::<f64>() / cache_times.len() as f64);
        
        Ok((avg_compile_time, hit_rate, time_saved))
    }

    /// Benchmark memory usage
    fn benchmark_memory_usage(&self) -> QuantRS2Result<(usize, usize)> {
        let cache_config = CacheConfig {
            cache_dir: self.temp_dir.path().to_path_buf(),
            max_memory_entries: self.config.num_gates,
            enable_persistence: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(cache_config)?;
        let gates = self.generate_test_gates();
        
        let initial_stats = cache.statistics();
        
        // Fill cache with gates
        for gate in &gates {
            let _ = cache.get_or_compile(gate.as_ref(), compile_single_qubit_gate)?;
        }
        
        let final_stats = cache.statistics();
        let memory_used = final_stats.total_size_bytes - initial_stats.total_size_bytes;
        
        Ok((memory_used, final_stats.num_entries))
    }

    /// Benchmark persistence performance
    fn benchmark_persistence(&self) -> QuantRS2Result<(usize, f64, f64)> {
        let cache_config = CacheConfig {
            cache_dir: self.temp_dir.path().to_path_buf(),
            enable_persistence: true,
            async_writes: false,
            compression_level: 3,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(cache_config)?;
        let gates = self.generate_test_gates();
        
        // Measure write times
        let mut write_times = Vec::new();
        
        for gate in &gates {
            let start = Instant::now();
            let _ = cache.get_or_compile(gate.as_ref(), compile_single_qubit_gate)?;
            let write_time = start.elapsed().as_micros() as f64;
            write_times.push(write_time);
        }
        
        let avg_write_time = write_times.iter().sum::<f64>() / write_times.len() as f64;
        
        // Calculate disk usage
        let mut disk_usage = 0;
        for entry in std::fs::read_dir(&cache_config.cache_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                disk_usage += entry.metadata()?.len() as usize;
            }
        }
        
        // Create new cache instance and measure read times
        let new_cache = CompilationCache::new(cache_config)?;
        let mut read_times = Vec::new();
        
        for gate in &gates {
            let start = Instant::now();
            let _ = new_cache.get_or_compile(gate.as_ref(), compile_single_qubit_gate)?;
            let read_time = start.elapsed().as_micros() as f64;
            read_times.push(read_time);
        }
        
        let avg_read_time = read_times.iter().sum::<f64>() / read_times.len() as f64;
        
        Ok((disk_usage, avg_write_time, avg_read_time))
    }

    /// Generate test gates for benchmarking
    fn generate_test_gates(&self) -> Vec<Box<dyn GateOp>> {
        let mut gates: Vec<Box<dyn GateOp>> = Vec::new();
        
        // Single-qubit gates
        for i in 0..self.config.num_gates / 6 {
            let qubit = QubitId(i as u32 % 4);
            gates.push(Box::new(Hadamard { target: qubit }));
            gates.push(Box::new(PauliX { target: qubit }));
            gates.push(Box::new(PauliY { target: qubit }));
            gates.push(Box::new(PauliZ { target: qubit }));
            gates.push(Box::new(Phase { target: qubit }));
            gates.push(Box::new(TGate { target: qubit }));
        }
        
        gates
    }

    /// Print benchmark results
    fn print_results(&self, results: &BenchmarkResults) {
        println!("\n=== Cache Benchmark Results ===");
        println!("Average compilation time: {:.2} μs", results.avg_compile_time_us);
        println!("Average cache hit time: {:.2} μs", results.avg_cache_hit_time_us);
        println!("Cache hit rate: {:.1}%", results.cache_hit_rate * 100.0);
        println!("Total time saved: {:.2} μs", results.total_time_saved_us);
        println!("Memory usage: {} bytes", results.memory_usage_bytes);
        println!("Disk usage: {} bytes", results.disk_usage_bytes);
        println!("Cache efficiency: {:.2} μs/byte", results.cache_efficiency);
        
        let speedup = results.avg_compile_time_us / results.avg_cache_hit_time_us;
        println!("Cache speedup: {:.1}x", speedup);
        
        let compression_ratio = if results.disk_usage_bytes > 0 {
            results.memory_usage_bytes as f64 / results.disk_usage_bytes as f64
        } else {
            0.0
        };
        println!("Compression ratio: {:.1}:1", compression_ratio);
    }

    /// Run cache eviction benchmark
    pub fn benchmark_cache_eviction(&self) -> QuantRS2Result<(f64, f64)> {
        let cache_config = CacheConfig {
            cache_dir: self.temp_dir.path().to_path_buf(),
            max_memory_entries: 10, // Small cache to force eviction
            enable_persistence: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(cache_config)?;
        let gates = self.generate_test_gates();
        
        // Fill cache beyond capacity
        let mut access_times = Vec::new();
        
        for (i, gate) in gates.iter().enumerate() {
            let start = Instant::now();
            let _ = cache.get_or_compile(gate.as_ref(), compile_single_qubit_gate)?;
            let access_time = start.elapsed().as_micros() as f64;
            access_times.push(access_time);
            
            if i % 10 == 0 {
                println!("Processed {} gates, cache size: {} entries", 
                        i + 1, cache.statistics().num_entries);
            }
        }
        
        // Calculate eviction overhead
        let early_times: Vec<_> = access_times.iter().take(10).cloned().collect();
        let late_times: Vec<_> = access_times.iter().rev().take(10).cloned().collect();
        
        let avg_early = early_times.iter().sum::<f64>() / early_times.len() as f64;
        let avg_late = late_times.iter().sum::<f64>() / late_times.len() as f64;
        
        Ok((avg_early, avg_late))
    }

    /// Benchmark concurrent access patterns
    pub fn benchmark_concurrent_access(&self) -> QuantRS2Result<f64> {
        let cache_config = CacheConfig {
            cache_dir: self.temp_dir.path().to_path_buf(),
            enable_persistence: false,
            ..Default::default()
        };
        
        let cache = Arc::new(CompilationCache::new(cache_config)?);
        let gates = Arc::new(self.generate_test_gates());
        let num_threads = 4;
        let accesses_per_thread = self.config.num_trials / num_threads;
        
        let start = Instant::now();
        let mut handles = Vec::new();
        
        for _ in 0..num_threads {
            let cache_clone = Arc::clone(&cache);
            let gates_clone = Arc::clone(&gates);
            
            let handle = std::thread::spawn(move || {
                for _ in 0..accesses_per_thread {
                    let gate_idx = fastrand::usize(..gates_clone.len());
                    let gate = &gates_clone[gate_idx];
                    let _ = cache_clone.get_or_compile(gate.as_ref(), compile_single_qubit_gate);
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let total_time = start.elapsed().as_micros() as f64;
        let avg_time_per_access = total_time / self.config.num_trials as f64;
        
        Ok(avg_time_per_access)
    }
}

/// Run comprehensive cache benchmarks
pub fn run_cache_benchmarks() -> QuantRS2Result<BenchmarkResults> {
    let config = BenchmarkConfig {
        num_trials: 1000,
        num_gates: 100,
        test_persistence: true,
        warmup_trials: 50,
    };
    
    let benchmarks = CacheBenchmarks::new(config)?;
    benchmarks.run_all_benchmarks()
}

/// Run quick cache benchmarks for CI/testing
pub fn run_quick_cache_benchmarks() -> QuantRS2Result<BenchmarkResults> {
    let config = BenchmarkConfig {
        num_trials: 50,
        num_gates: 20,
        test_persistence: false,
        warmup_trials: 5,
    };
    
    let benchmarks = CacheBenchmarks::new(config)?;
    benchmarks.run_all_benchmarks()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_benchmark() {
        let config = BenchmarkConfig {
            num_trials: 10,
            num_gates: 5,
            test_persistence: false,
            warmup_trials: 2,
        };
        
        let benchmarks = CacheBenchmarks::new(config).unwrap();
        let hit_time = benchmarks.benchmark_cache_hits().unwrap();
        
        assert!(hit_time > 0.0);
        assert!(hit_time < 1000.0); // Should be under 1ms
    }

    #[test]
    fn test_compilation_savings_benchmark() {
        let config = BenchmarkConfig {
            num_trials: 20,
            num_gates: 10,
            test_persistence: false,
            warmup_trials: 2,
        };
        
        let benchmarks = CacheBenchmarks::new(config).unwrap();
        let (compile_time, hit_rate, time_saved) = benchmarks.benchmark_compilation_savings().unwrap();
        
        assert!(compile_time > 0.0);
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
        assert!(time_saved >= 0.0);
        
        println!("Compile time: {:.2}μs, Hit rate: {:.1}%, Time saved: {:.2}μs", 
                compile_time, hit_rate * 100.0, time_saved);
    }

    #[test]
    fn test_memory_usage_benchmark() {
        let config = BenchmarkConfig {
            num_trials: 10,
            num_gates: 15,
            test_persistence: false,
            warmup_trials: 1,
        };
        
        let benchmarks = CacheBenchmarks::new(config).unwrap();
        let (memory_used, num_entries) = benchmarks.benchmark_memory_usage().unwrap();
        
        assert!(memory_used > 0);
        assert!(num_entries > 0);
        assert!(num_entries <= config.num_gates);
        
        println!("Memory used: {} bytes, Entries: {}", memory_used, num_entries);
    }

    #[test]
    fn test_cache_eviction_benchmark() {
        let config = BenchmarkConfig {
            num_trials: 50,
            num_gates: 25,
            test_persistence: false,
            warmup_trials: 1,
        };
        
        let benchmarks = CacheBenchmarks::new(config).unwrap();
        let (early_time, late_time) = benchmarks.benchmark_cache_eviction().unwrap();
        
        assert!(early_time > 0.0);
        assert!(late_time > 0.0);
        
        println!("Early access time: {:.2}μs, Late access time: {:.2}μs", early_time, late_time);
    }

    #[test]
    fn test_concurrent_access_benchmark() {
        let config = BenchmarkConfig {
            num_trials: 100,
            num_gates: 20,
            test_persistence: false,
            warmup_trials: 1,
        };
        
        let benchmarks = CacheBenchmarks::new(config).unwrap();
        let avg_time = benchmarks.benchmark_concurrent_access().unwrap();
        
        assert!(avg_time > 0.0);
        
        println!("Average concurrent access time: {:.2}μs", avg_time);
    }

    #[test]
    fn test_quick_benchmarks() {
        let results = run_quick_cache_benchmarks().unwrap();
        
        assert!(results.avg_compile_time_us > results.avg_cache_hit_time_us);
        assert!(results.cache_hit_rate >= 0.0 && results.cache_hit_rate <= 1.0);
        assert!(results.memory_usage_bytes > 0);
    }
}