# Device-Specific Gate Calibration Data Structures

## Overview

This document describes the comprehensive device-specific gate calibration data structures implemented for QuantRS2. These structures enable accurate modeling of quantum hardware characteristics, noise modeling, and circuit optimization based on real device parameters.

## Architecture

### Core Components

1. **DeviceCalibration** (`device/src/calibration.rs`)
   - Complete calibration data for a quantum device
   - Includes qubit parameters, gate fidelities, topology, and crosstalk
   - Timestamped with validity duration

2. **CalibrationManager**
   - Manages multiple device calibrations
   - Tracks calibration history
   - Load/save calibrations from/to files
   - Validity checking

3. **CalibrationNoiseModel** (`device/src/noise_model.rs`)
   - Derives realistic noise models from calibration data
   - Includes T1/T2 decoherence, gate errors, crosstalk
   - Customizable scaling factors

4. **CalibrationOptimizer** (`device/src/optimization.rs`)
   - Circuit optimization using calibration data
   - Fidelity and duration optimization
   - Gate substitution strategies
   - Crosstalk minimization

## Data Structures

### Qubit Calibration
```rust
pub struct QubitCalibration {
    pub qubit_id: QubitId,
    pub frequency: f64,          // Hz
    pub anharmonicity: f64,      // Hz
    pub t1: f64,                 // μs
    pub t2: f64,                 // μs
    pub t2_star: Option<f64>,    // μs
    pub readout_error: f64,
    pub thermal_population: f64,
    pub temperature: Option<f64>, // mK
    pub parameters: HashMap<String, f64>,
}
```

### Gate Calibration

#### Single-Qubit Gates
```rust
pub struct SingleQubitGateData {
    pub error_rate: f64,
    pub fidelity: f64,
    pub duration: f64,           // ns
    pub amplitude: f64,
    pub frequency: f64,          // Hz
    pub phase: f64,
    pub pulse_shape: PulseShape,
    pub calibrated_matrix: Option<Vec<Complex64>>,
    pub parameter_calibrations: Option<ParameterCalibration>,
}
```

#### Two-Qubit Gates
```rust
pub struct TwoQubitGateCalibration {
    pub gate_name: String,
    pub control: QubitId,
    pub target: QubitId,
    pub error_rate: f64,
    pub fidelity: f64,
    pub duration: f64,           // ns
    pub coupling_strength: f64,  // MHz
    pub cross_resonance: Option<CrossResonanceParameters>,
    pub calibrated_matrix: Option<Vec<Complex64>>,
    pub directional: bool,
}
```

### Readout Calibration
```rust
pub struct ReadoutCalibration {
    pub qubit_readout: HashMap<QubitId, QubitReadoutData>,
    pub mitigation_matrix: Option<Vec<Vec<f64>>>,
    pub duration: f64,           // ns
    pub integration_time: f64,   // ns
}
```

### Device Topology
```rust
pub struct DeviceTopology {
    pub num_qubits: usize,
    pub coupling_map: Vec<(QubitId, QubitId)>,
    pub layout_type: String,     // "linear", "grid", "heavy-hex"
    pub qubit_coordinates: Option<HashMap<QubitId, (f64, f64)>>,
}
```

## Features

### 1. Calibration Management

- **Storage**: Save/load calibrations to/from JSON files
- **Versioning**: Track calibration history with timestamps
- **Validity**: Automatic expiration checking
- **Updates**: Seamless calibration updates

### 2. Noise Modeling

Generate realistic noise models from calibration:
- **Decoherence**: T1/T2 effects during gates
- **Gate Errors**: Coherent and incoherent errors
- **Crosstalk**: Spectator qubit effects
- **Thermal Noise**: Temperature-dependent errors
- **Readout Errors**: Assignment matrix

### 3. Circuit Optimization

Optimize circuits based on device characteristics:
- **Fidelity Optimization**: Use highest quality qubits
- **Duration Minimization**: Parallelize compatible gates
- **Gate Substitution**: Replace with higher-fidelity alternatives
- **Crosstalk Avoidance**: Minimize simultaneous operations

### 4. Fidelity Estimation

Estimate circuit performance:
- **Process Fidelity**: Gate error accumulation
- **State Fidelity**: Including decoherence
- **Readout Fidelity**: Measurement errors

## Usage Examples

### Creating Calibration

```rust
use quantrs2_device::calibration::*;

let calibration = CalibrationBuilder::new("my_device".to_string())
    .valid_duration(Duration::from_secs(24 * 3600))
    .add_qubit_calibration(QubitCalibration {
        qubit_id: QubitId(0),
        frequency: 5e9,
        anharmonicity: -300e6,
        t1: 50_000.0,
        t2: 40_000.0,
        // ...
    })
    .add_single_qubit_gate("X".to_string(), x_gate_cal)
    .add_two_qubit_gate(QubitId(0), QubitId(1), cnot_cal)
    .readout_calibration(readout_cal)
    .topology(topology)
    .build()?;
```

### Building Noise Model

```rust
use quantrs2_device::noise_model::*;

// Standard noise model
let noise_model = CalibrationNoiseModel::from_calibration(&calibration);

// Custom scaling
let custom_noise = NoiseModelBuilder::from_calibration(calibration)
    .coherent_factor(0.5)    // Reduce coherent errors
    .thermal_factor(2.0)     // Increase thermal noise
    .build();
```

### Optimizing Circuits

```rust
use quantrs2_device::optimization::*;

let optimizer = CalibrationOptimizer::new(manager, config);
let result = optimizer.optimize_circuit(&circuit, "device_id")?;

println!("Optimized fidelity: {:.4}", result.estimated_fidelity);
println!("Duration: {:.1} ns", result.estimated_duration);
```

## Pulse-Level Details

### Pulse Shapes

Supported pulse shapes for gate implementation:
- **Gaussian**: Standard Gaussian envelope
- **GaussianDRAG**: With DRAG correction
- **Square**: Rectangular pulse
- **Cosine**: Raised cosine shape
- **Custom**: User-defined shapes

### Cross-Resonance Parameters

For CNOT implementation:
```rust
pub struct CrossResonanceParameters {
    pub drive_frequency: f64,    // Hz
    pub drive_amplitude: f64,
    pub pulse_duration: f64,     // ns
    pub echo_amplitude: f64,
    pub echo_duration: f64,      // ns
    pub zx_interaction_rate: f64, // MHz
}
```

## Performance Characteristics

### Memory Efficiency
- Compact representation of calibration data
- Optional fields for device-specific parameters
- Efficient serialization to JSON

### Scalability
- Supports devices with 100+ qubits
- Hierarchical organization of gate data
- Fast lookup for optimization decisions

### Integration
- Compatible with all QuantRS2 modules
- Works with circuit builders and simulators
- Extensible for new gate types

## Testing

Comprehensive test coverage includes:
- Calibration creation and validation
- Noise model generation
- Circuit optimization strategies
- Fidelity estimation accuracy

Run tests:
```bash
cargo test -p quantrs2-device calibration
```

## Future Enhancements

1. **Machine Learning Integration**
   - Learn calibration drift patterns
   - Predictive maintenance alerts
   - Optimal recalibration scheduling

2. **Advanced Noise Models**
   - Non-Markovian noise
   - Correlated errors
   - Time-dependent parameters

3. **Real-Time Updates**
   - Live calibration streaming
   - Dynamic optimization
   - Adaptive error mitigation

4. **Benchmarking Suite**
   - Standardized calibration benchmarks
   - Cross-device comparisons
   - Performance tracking