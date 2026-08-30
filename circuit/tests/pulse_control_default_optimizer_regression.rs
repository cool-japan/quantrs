//! Regression tests for `DefaultPulseOptimizer::optimize` in
//! `scirs2_pulse_control_enhanced`.
//!
//! These exercise the real Z-X-Z Euler-angle pulse-shaping algorithm that
//! replaced the previous no-op `Ok(pulse.clone())` passthrough, verifying it
//! actually derives a physically meaningful drive phase/amplitude and
//! virtual-Z frame correction from the target unitary rather than silently
//! returning the input unchanged.

use quantrs2_circuit::scirs2_pulse_control_enhanced::{
    EnhancedPulseConfig, EnhancedPulseController, GateAnalysis, PulseChannel, PulseConstraints,
    PulseMetadata, PulseSequence, Waveform,
};
use scirs2_core::Complex64;

fn flat_waveform(amplitude: f64, samples: usize, sample_rate: f64) -> Waveform {
    Waveform {
        samples: vec![Complex64::new(amplitude, 0.0); samples],
        sample_rate,
    }
}

fn single_channel_pulse(waveform: Waveform, channel_id: usize) -> PulseSequence {
    PulseSequence {
        channels: vec![PulseChannel {
            channel_id,
            waveform,
            frequency: 5.0e9,
            phase: 0.0,
            frame_change: None,
        }],
        duration: 1.0,
        metadata: PulseMetadata {
            gate_name: "TEST".to_string(),
            target_qubits: vec![channel_id],
            fidelity_estimate: None,
            optimization_history: vec![],
        },
    }
}

fn integrated_area(channel: &PulseChannel) -> Complex64 {
    let dt = 1.0 / channel.waveform.sample_rate;
    channel.waveform.samples.iter().copied().sum::<Complex64>() * Complex64::new(dt, 0.0)
}

fn controller() -> EnhancedPulseController {
    EnhancedPulseController::new(EnhancedPulseConfig::default())
}

#[test]
fn default_optimizer_is_wired_in_by_default() {
    let controller = controller();
    assert!(
        controller.ml_optimizer.is_some(),
        "EnhancedPulseController::new must wire in a default pulse optimizer"
    );
}

#[test]
fn default_optimizer_realizes_pauli_x_as_a_pi_pulse() {
    let controller = controller();
    let optimizer = controller.ml_optimizer.as_ref().unwrap();

    // Pauli-X, up to the (physically unobservable) global phase, is a pi
    // rotation about the X axis: theta = pi, alpha = beta = 0.
    let target = GateAnalysis {
        target_unitary: vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
        qubit_indices: vec![0],
    };

    let pulse = single_channel_pulse(flat_waveform(1.0, 10, 10.0), 0);
    let constraints = PulseConstraints {
        max_amplitude: Some(1000.0),
        ..PulseConstraints::default()
    };

    let optimized = optimizer.optimize(&pulse, &target, &constraints).unwrap();

    assert_eq!(optimized.channels.len(), 1);
    let channel = &optimized.channels[0];

    let area = integrated_area(channel);
    assert!(
        (area.norm() - std::f64::consts::PI).abs() < 1e-6,
        "pi-pulse area magnitude should equal pi, got {}",
        area.norm()
    );
    assert!(
        area.arg().abs() < 1e-6,
        "X-gate drive phase should be ~0, got {}",
        area.arg()
    );
    let frame_change = channel.frame_change.unwrap_or(0.0);
    assert!(
        frame_change.abs() < 1e-6,
        "X-gate should require no net virtual-Z correction, got {frame_change}"
    );

    // The optimizer must have actually changed the pulse, not passed it through.
    assert_ne!(
        optimized.channels[0].waveform.samples, pulse.channels[0].waveform.samples,
        "optimize() must reshape the envelope, not return it unchanged"
    );
}

#[test]
fn default_optimizer_realizes_pauli_y_with_quarter_turn_drive_phase() {
    let controller = controller();
    let optimizer = controller.ml_optimizer.as_ref().unwrap();

    // Pauli-Y corresponds to theta = pi, alpha = pi/2 (a Y-axis drive).
    let target = GateAnalysis {
        target_unitary: vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            vec![Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ],
        qubit_indices: vec![0],
    };

    let pulse = single_channel_pulse(flat_waveform(1.0, 20, 20.0), 0);
    let constraints = PulseConstraints {
        max_amplitude: Some(1000.0),
        ..PulseConstraints::default()
    };

    let optimized = optimizer.optimize(&pulse, &target, &constraints).unwrap();
    let area = integrated_area(&optimized.channels[0]);

    assert!(
        (area.norm() - std::f64::consts::PI).abs() < 1e-6,
        "pi-pulse area magnitude should equal pi, got {}",
        area.norm()
    );
    assert!(
        (area.arg() - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
        "Y-gate drive phase should be ~pi/2, got {}",
        area.arg()
    );
}

#[test]
fn default_optimizer_respects_max_amplitude_constraint() {
    let controller = controller();
    let optimizer = controller.ml_optimizer.as_ref().unwrap();

    let target = GateAnalysis {
        target_unitary: vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
        qubit_indices: vec![0],
    };

    // A single, very short sample forces a huge required instantaneous
    // amplitude to hit the pi-pulse area, which must be clamped.
    let pulse = single_channel_pulse(flat_waveform(1.0, 1, 1.0e9), 0);
    let constraints = PulseConstraints {
        max_amplitude: Some(0.05),
        ..PulseConstraints::default()
    };

    let optimized = optimizer.optimize(&pulse, &target, &constraints).unwrap();
    let peak = optimized.channels[0]
        .waveform
        .samples
        .iter()
        .map(|sample| sample.norm())
        .fold(0.0_f64, f64::max);

    assert!(
        peak <= 0.05 + 1e-9,
        "peak amplitude {peak} must respect the configured max_amplitude constraint"
    );
}

#[test]
fn default_optimizer_leaves_multi_qubit_targets_unchanged() {
    let controller = controller();
    let optimizer = controller.ml_optimizer.as_ref().unwrap();

    // A 4x4 target (two-qubit gate) is out of scope for this closed-form
    // single-qubit optimizer; it must be an honest no-op, not a fabricated
    // multi-qubit solution.
    let mut target_unitary = vec![vec![Complex64::new(0.0, 0.0); 4]; 4];
    for (i, row) in target_unitary.iter_mut().enumerate() {
        row[i] = Complex64::new(1.0, 0.0);
    }
    let target = GateAnalysis {
        target_unitary,
        qubit_indices: vec![0, 1],
    };

    let pulse = single_channel_pulse(flat_waveform(0.3, 5, 5.0), 0);
    let constraints = PulseConstraints::default();

    let optimized = optimizer.optimize(&pulse, &target, &constraints).unwrap();
    assert_eq!(
        optimized.channels[0].waveform.samples, pulse.channels[0].waveform.samples,
        "multi-qubit targets must be passed through unchanged by this optimizer"
    );
    assert_eq!(
        optimized.channels[0].frame_change,
        pulse.channels[0].frame_change
    );
}

#[test]
fn default_optimizer_handles_empty_waveform_without_panicking() {
    let controller = controller();
    let optimizer = controller.ml_optimizer.as_ref().unwrap();

    let target = GateAnalysis {
        target_unitary: vec![
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
        qubit_indices: vec![0],
    };

    let pulse = single_channel_pulse(
        Waveform {
            samples: vec![],
            sample_rate: 10.0,
        },
        0,
    );
    let constraints = PulseConstraints::default();

    let optimized = optimizer.optimize(&pulse, &target, &constraints).unwrap();
    assert!(!optimized.channels[0].waveform.samples.is_empty());
    let area = integrated_area(&optimized.channels[0]);
    assert!((area.norm() - std::f64::consts::PI).abs() < 1e-6);
}
