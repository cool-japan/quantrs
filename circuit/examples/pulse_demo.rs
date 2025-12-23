//! Demonstration of pulse-level control for quantum circuits
//!
//! This example shows how to use low-level pulse control for fine-grained
//! optimization and hardware-specific calibration.

use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::multi::CNOT;
use quantrs2_core::gate::single::{Hadamard, PauliX, RotationZ};
use quantrs2_core::qubit::QubitId;
use scirs2_core::Complex;
use std::f64::consts::PI;

type C64 = Complex<f64>;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Pulse-Level Control Demo ===\n");

    demo_pulse_waveforms()?;
    demo_pulse_schedules()?;
    demo_pulse_compilation()?;
    demo_pulse_calibration()?;
    demo_pulse_optimization()?;

    Ok(())
}

fn demo_pulse_waveforms() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Pulse Waveform Types ---");

    let sample_rate = 1.0; // 1 GS/s

    // Gaussian pulse
    let gaussian = Waveform::gaussian(0.5, 10.0, 40.0, sample_rate);
    println!("Gaussian pulse:");
    println!("  Amplitude: 0.5");
    println!("  Sigma: 10.0 ns");
    println!("  Duration: {} ns", gaussian.duration);
    println!("  Samples: {}", gaussian.samples.len());
    println!("  Max amplitude: {:.4}", gaussian.max_amplitude());

    // DRAG pulse
    let drag = Waveform::drag(0.5, 10.0, 0.1, 40.0, sample_rate);
    println!("\nDRAG pulse:");
    println!("  Beta: 0.1");
    println!(
        "  Has imaginary component: {}",
        drag.samples.iter().any(|s| s.im.abs() > 1e-10)
    );

    // Square pulse
    let square = Waveform::square(0.3, 20.0, sample_rate);
    println!("\nSquare pulse:");
    println!("  Amplitude: 0.3");
    println!("  Duration: {} ns", square.duration);

    // Modulated pulse
    let mut modulated = Waveform::gaussian(0.5, 10.0, 40.0, sample_rate);
    modulated.modulate(0.1, PI / 4.0); // 100 MHz modulation
    println!("\nModulated Gaussian:");
    println!("  Modulation frequency: 100 MHz");
    println!("  Phase: π/4");

    println!();
    Ok(())
}

fn demo_pulse_schedules() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Pulse Schedule Construction ---");

    let sample_rate = 1.0;
    let mut schedule = PulseSchedule::new();

    // Create waveforms
    let pi_pulse = Waveform::gaussian(0.5, 10.0, 40.0, sample_rate);
    let pi2_pulse = Waveform::gaussian(0.25, 10.0, 40.0, sample_rate);
    let cr_pulse = Waveform::square(0.1, 200.0, sample_rate);

    // Build schedule for a simple circuit
    let mut time = 0.0;

    // X gate on qubit 0
    schedule.add_instruction(
        time,
        PulseInstruction::Play {
            waveform: pi_pulse.clone(),
            channel: Channel::Drive(0),
            phase: 0.0,
        },
    );
    time += 40.0;

    // Delay
    schedule.add_instruction(
        time,
        PulseInstruction::Delay {
            duration: 20.0,
            channels: vec![Channel::Drive(0), Channel::Drive(1)],
        },
    );
    time += 20.0;

    // CNOT using cross-resonance
    schedule.add_instruction(
        time,
        PulseInstruction::Play {
            waveform: cr_pulse,
            channel: Channel::Control(0, 1),
            phase: 0.0,
        },
    );

    // Echo pulse on target
    schedule.add_instruction(
        time + 100.0,
        PulseInstruction::Play {
            waveform: pi_pulse,
            channel: Channel::Drive(1),
            phase: PI,
        },
    );
    time += 200.0;

    // Measurement
    schedule.add_instruction(
        time,
        PulseInstruction::Acquire {
            duration: 1000.0,
            channel: Channel::Measure(0),
            memory_slot: 0,
        },
    );

    println!("Built pulse schedule:");
    println!("  Total duration: {} ns", schedule.duration);
    println!("  Instructions: {}", schedule.instructions.len());
    println!("  Channels used: {}", schedule.channels.len());

    println!("\nSchedule timeline:");
    for (time, instruction) in &schedule.instructions {
        match instruction {
            PulseInstruction::Play { channel, .. } => {
                println!("  {time:6.1} ns: Play on {channel:?}");
            }
            PulseInstruction::Delay { duration, .. } => {
                println!("  {time:6.1} ns: Delay {duration} ns");
            }
            PulseInstruction::Acquire { channel, .. } => {
                println!("  {time:6.1} ns: Acquire on {channel:?}");
            }
            _ => {}
        }
    }

    println!();
    Ok(())
}

fn demo_pulse_compilation() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Circuit to Pulse Compilation ---");

    // Create device configuration
    let device_config = DeviceConfig::default_config(4);
    println!("Device configuration:");
    println!("  Qubits: 4");
    println!("  Qubit frequencies: ~5 GHz");
    println!("  Sample rate: {} GS/s", device_config.sample_rate);

    // Create compiler
    let compiler = PulseCompiler::new(device_config);

    // Create a simple circuit
    let mut circuit = Circuit::<4>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(RotationZ {
        target: QubitId(1),
        theta: PI / 4.0,
    })?;

    println!("\nCompiling circuit:");
    for (i, gate) in circuit.gates().iter().enumerate() {
        println!("  {}: {}", i, gate.name());
    }

    // Compile to pulses
    let pulse_schedule = compiler.compile(&circuit)?;

    println!("\nCompiled pulse schedule:");
    println!("  Duration: {} ns", pulse_schedule.duration);
    println!("  Instructions: {}", pulse_schedule.instructions.len());

    // Show channel usage
    let mut channel_usage = std::collections::HashMap::new();
    for (_, instruction) in &pulse_schedule.instructions {
        if let PulseInstruction::Play { channel, .. } = instruction {
            *channel_usage.entry(format!("{channel:?}")).or_insert(0) += 1;
        }
    }

    println!("\nChannel usage:");
    for (channel, count) in channel_usage {
        println!("  {channel}: {count} pulses");
    }

    println!();
    Ok(())
}

fn demo_pulse_calibration() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Pulse Calibration ---");

    let sample_rate = 1.0;

    // Create calibrations for different rotation angles
    let angles = vec![PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];
    let mut calibrations = Vec::new();

    for &angle in &angles {
        // Calculate pulse amplitude for rotation
        let amplitude = angle / (2.0 * PI) * 0.5; // Simplified
        let waveform = Waveform::gaussian(amplitude, 10.0, 40.0, sample_rate);

        let mut schedule = PulseSchedule::new();
        schedule.add_instruction(
            0.0,
            PulseInstruction::Play {
                waveform,
                channel: Channel::Drive(0),
                phase: 0.0,
            },
        );

        let mut parameters = std::collections::HashMap::new();
        parameters.insert("theta".to_string(), angle);

        calibrations.push(PulseCalibration {
            gate_name: "RZ".to_string(),
            qubits: vec![QubitId(0)],
            parameters,
            schedule,
        });
    }

    println!("Created {} calibrations for RZ gate", calibrations.len());

    for calib in &calibrations {
        let theta = calib.parameters.get("theta").unwrap();
        println!(
            "  RZ(θ={:.3}): duration = {} ns",
            theta, calib.schedule.duration
        );
    }

    // Demonstrate calibration interpolation
    println!("\nCalibration interpolation:");
    let target_angle = PI / 3.0;
    println!("  Target angle: π/3");
    println!("  Would interpolate between π/4 and π/2 calibrations");

    println!();
    Ok(())
}

fn demo_pulse_optimization() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Pulse Optimization ---");

    let sample_rate = 1.0;
    let optimizer = PulseOptimizer::new();

    // Create initial pulse
    let mut pulse = Waveform::gaussian(0.5, 15.0, 60.0, sample_rate);
    println!("Initial pulse:");
    println!("  Type: Gaussian");
    println!("  Duration: {} ns", pulse.duration);
    println!("  Max amplitude: {:.4}", pulse.max_amplitude());

    // Apply DRAG correction
    optimizer.apply_drag_correction(&mut pulse, 0.1)?;
    println!("\nAfter DRAG correction:");
    println!(
        "  Has imaginary component: {}",
        pulse.samples.iter().any(|s| s.im.abs() > 1e-10)
    );

    // Create a pulse schedule to optimize
    let mut schedule = PulseSchedule::new();

    // Add multiple pulses
    for i in 0..3 {
        let waveform = Waveform::gaussian(0.4, 12.0, 48.0, sample_rate);
        schedule.add_instruction(
            i as f64 * 60.0,
            PulseInstruction::Play {
                waveform,
                channel: Channel::Drive(i),
                phase: 0.0,
            },
        );
    }

    println!("\nSchedule optimization:");
    println!("  Original duration: {} ns", schedule.duration);

    // Optimize (placeholder - would do actual optimization)
    optimizer.optimize(&mut schedule)?;

    println!(
        "  Optimized duration: {} ns (placeholder)",
        schedule.duration
    );
    println!("  Target fidelity: 0.999"); // Placeholder since field is private

    // Demonstrate pulse shaping techniques
    println!("\nPulse shaping techniques:");
    println!("  - Gaussian: Smooth envelope, reduced frequency spread");
    println!("  - DRAG: Reduces leakage to higher levels");
    println!("  - Cosine: Smooth turn-on/off");
    println!("  - GRAPE: Optimal control for specific unitaries");

    println!();
    Ok(())
}

fn demo_advanced_pulses() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Advanced Pulse Techniques ---");

    let sample_rate = 2.0; // 2 GS/s for higher resolution

    // Composite pulse for robust gates
    println!("Composite pulse sequence (BB1):");
    let mut bb1_schedule = PulseSchedule::new();

    // BB1 sequence: Ry(π/2) - Rx(π) - Ry(3π/2) - Rx(π)
    let pi2_y = Waveform::gaussian(0.25, 8.0, 32.0, sample_rate);
    let pi_x = Waveform::gaussian(0.5, 8.0, 32.0, sample_rate);
    let three_pi2_y = Waveform::gaussian(0.75, 8.0, 32.0, sample_rate);

    let mut time = 0.0;

    bb1_schedule.add_instruction(
        time,
        PulseInstruction::Play {
            waveform: pi2_y,
            channel: Channel::Drive(0),
            phase: PI / 2.0, // Y rotation
        },
    );
    time += 32.0;

    bb1_schedule.add_instruction(
        time,
        PulseInstruction::Play {
            waveform: pi_x.clone(),
            channel: Channel::Drive(0),
            phase: 0.0, // X rotation
        },
    );
    time += 32.0;

    bb1_schedule.add_instruction(
        time,
        PulseInstruction::Play {
            waveform: three_pi2_y,
            channel: Channel::Drive(0),
            phase: PI / 2.0, // Y rotation
        },
    );
    time += 32.0;

    bb1_schedule.add_instruction(
        time,
        PulseInstruction::Play {
            waveform: pi_x,
            channel: Channel::Drive(0),
            phase: 0.0, // X rotation
        },
    );

    println!("  Total duration: {} ns", bb1_schedule.duration);
    println!("  Robustness: High (compensates for amplitude errors)");

    // Adiabatic pulses
    println!("\nAdiabatic pulse:");
    let mut adiabatic_samples = Vec::new();
    let duration = 100.0;
    let n_samples = (duration * sample_rate) as usize;

    for i in 0..n_samples {
        let t = i as f64 / sample_rate;
        let normalized_t = t / duration;

        // Tanh profile for adiabatic following
        let amplitude = 0.5 * ((10.0 * (normalized_t - 0.5)).tanh() + 1.0) / 2.0;
        adiabatic_samples.push(C64::new(amplitude, 0.0));
    }

    let adiabatic = Waveform::new(adiabatic_samples, sample_rate);
    println!("  Duration: {} ns", adiabatic.duration);
    println!("  Profile: Smooth tanh ramp");
    println!("  Use case: State transfer with high fidelity");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_demo() {
        assert!(main().is_ok());
    }
}
