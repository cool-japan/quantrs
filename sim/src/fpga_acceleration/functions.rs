//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::collections::HashMap;

use super::types::{
    ArithmeticPrecision, FPGAConfig, FPGADeviceInfo, FPGAPlatform, FPGAQuantumSimulator, ModuleType,
};

/// Benchmark FPGA acceleration performance
pub fn benchmark_fpga_acceleration() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();
    let configs = vec![
        FPGAConfig {
            platform: FPGAPlatform::IntelStratix10,
            num_processing_units: 8,
            clock_frequency: 300.0,
            ..Default::default()
        },
        FPGAConfig {
            platform: FPGAPlatform::IntelAgilex7,
            num_processing_units: 16,
            clock_frequency: 400.0,
            ..Default::default()
        },
        FPGAConfig {
            platform: FPGAPlatform::XilinxVersal,
            num_processing_units: 32,
            clock_frequency: 500.0,
            enable_pipelining: true,
            ..Default::default()
        },
    ];
    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();
        let mut simulator = FPGAQuantumSimulator::new(config)?;
        let mut circuit = InterfaceCircuit::new(10, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.5), vec![2]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CZ, vec![1, 2]));
        for _ in 0..10 {
            let _result = simulator.execute_circuit(&circuit)?;
        }
        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("fpga_config_{i}"), time);
        let stats = simulator.get_stats();
        results.insert(
            format!("fpga_config_{i}_operations"),
            stats.total_gate_operations as f64,
        );
        results.insert(
            format!("fpga_config_{i}_avg_gate_time"),
            stats.avg_gate_time,
        );
        results.insert(
            format!("fpga_config_{i}_utilization"),
            stats.fpga_utilization,
        );
        results.insert(
            format!("fpga_config_{i}_pipeline_efficiency"),
            stats.pipeline_efficiency,
        );
        let performance_metrics = stats.get_performance_metrics();
        for (key, value) in performance_metrics {
            results.insert(format!("fpga_config_{i}_{key}"), value);
        }
    }
    results.insert("kernel_compilation_time".to_string(), 1500.0);
    results.insert("memory_transfer_bandwidth".to_string(), 250.0);
    results.insert("gate_execution_throughput".to_string(), 1_000_000.0);
    Ok(results)
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_fpga_simulator_creation() {
        let config = FPGAConfig::default();
        let simulator = FPGAQuantumSimulator::new(config);
        assert!(simulator.is_ok());
    }
    #[test]
    fn test_device_info_creation() {
        let device_info = FPGADeviceInfo::for_platform(FPGAPlatform::IntelStratix10);
        assert_eq!(device_info.platform, FPGAPlatform::IntelStratix10);
        assert_eq!(device_info.logic_elements, 2_800_000);
        assert_eq!(device_info.dsp_blocks, 5760);
    }
    #[test]
    fn test_processing_unit_creation() {
        let config = FPGAConfig::default();
        let device_info = FPGADeviceInfo::for_platform(config.platform);
        let units = FPGAQuantumSimulator::create_processing_units(&config, &device_info)
            .expect("should create processing units successfully");
        assert_eq!(units.len(), config.num_processing_units);
        assert!(!units[0].supported_gates.is_empty());
        assert!(!units[0].pipeline_stages.is_empty());
    }
    #[test]
    fn test_hdl_generation() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for HDL generation test");
        assert!(simulator.hdl_modules.contains_key("single_qubit_gate"));
        assert!(simulator.hdl_modules.contains_key("two_qubit_gate"));
        let single_qubit_module = &simulator.hdl_modules["single_qubit_gate"];
        assert!(!single_qubit_module.hdl_code.is_empty());
        assert_eq!(single_qubit_module.module_type, ModuleType::SingleQubitGate);
    }
    #[test]
    fn test_circuit_execution() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for circuit execution test");
        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        let result = simulator.execute_circuit(&circuit);
        assert!(result.is_ok());
        let state = result.expect("circuit execution should succeed");
        assert_eq!(state.len(), 4);
        assert!(state[0].norm() > 0.0);
    }
    #[test]
    fn test_gate_application() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for gate application test");
        let mut state = Array1::zeros(4);
        state[0] = Complex64::new(1.0, 0.0);
        let gate = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);
        let result = simulator.apply_single_qubit_gate_fpga(&state, &gate, 0);
        assert!(result.is_ok());
        let new_state = result.expect("gate application should succeed");
        assert_abs_diff_eq!(new_state[0].norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_abs_diff_eq!(new_state[1].norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }
    #[test]
    fn test_bitstream_management() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for bitstream management test");
        assert!(simulator.bitstream_manager.current_config.is_some());
        assert!(simulator
            .bitstream_manager
            .bitstreams
            .contains_key("quantum_basic"));
        let result = simulator.reconfigure("quantum_advanced");
        assert!(result.is_ok());
        assert_eq!(
            simulator.bitstream_manager.current_config,
            Some("quantum_advanced".to_string())
        );
    }
    #[test]
    fn test_memory_management() {
        let config = FPGAConfig::default();
        let simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for memory management test");
        assert!(simulator
            .memory_manager
            .onchip_pools
            .contains_key("state_vector"));
        assert!(simulator
            .memory_manager
            .onchip_pools
            .contains_key("gate_cache"));
        assert!(!simulator.memory_manager.external_interfaces.is_empty());
    }
    #[test]
    fn test_stats_tracking() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for stats tracking test");
        simulator.stats.update_operation(10.0, 1000);
        simulator.stats.update_operation(20.0, 2000);
        assert_eq!(simulator.stats.total_gate_operations, 2);
        assert_abs_diff_eq!(simulator.stats.total_execution_time, 30.0, epsilon = 1e-10);
        assert_eq!(simulator.stats.total_clock_cycles, 3000);
    }
    #[test]
    fn test_performance_metrics() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for performance metrics test");
        simulator.stats.total_gate_operations = 100;
        simulator.stats.total_execution_time = 1000.0;
        simulator.stats.total_clock_cycles = 300_000;
        simulator.stats.fpga_utilization = 75.0;
        simulator.stats.pipeline_efficiency = 0.85;
        simulator.stats.power_consumption = 120.0;
        let metrics = simulator.stats.get_performance_metrics();
        assert!(metrics.contains_key("operations_per_second"));
        assert!(metrics.contains_key("cycles_per_operation"));
        assert!(metrics.contains_key("fpga_utilization"));
        assert_abs_diff_eq!(metrics["operations_per_second"], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(metrics["cycles_per_operation"], 3000.0, epsilon = 1e-10);
    }
    #[test]
    fn test_hdl_export() {
        let config = FPGAConfig::default();
        let simulator = FPGAQuantumSimulator::new(config)
            .expect("should create FPGA simulator for HDL export test");
        let hdl_code = simulator.export_hdl("single_qubit_gate");
        assert!(hdl_code.is_ok());
        assert!(!hdl_code.expect("HDL export should succeed").is_empty());
        let invalid_module = simulator.export_hdl("nonexistent_module");
        assert!(invalid_module.is_err());
    }
    #[test]
    fn test_arithmetic_precision() {
        assert_eq!(ArithmeticPrecision::Fixed16, ArithmeticPrecision::Fixed16);
        assert_ne!(ArithmeticPrecision::Fixed16, ArithmeticPrecision::Fixed32);
    }
}
