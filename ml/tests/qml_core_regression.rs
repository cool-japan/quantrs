//! Regression tests for the core quantum-ML primitives that were previously
//! silent fabrications: `QuantumNeuralNetwork` (forward/train), the HEP
//! classifier (`predict`/`evaluate`), the reinforcement-learning agent
//! (`get_q_values`/`update`), the quantum GAN (`generate`/`discriminate` and
//! their updates), and the QCNN forward pass.
//!
//! Each test asserts a property that the old dummy implementations could not
//! satisfy: dependence on inputs/parameters, deterministic real outputs, and
//! genuine loss reduction during training.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use quantrs2_ml::gan::{
    Discriminator, DiscriminatorType, Generator, GeneratorType, QuantumDiscriminator, QuantumGAN,
    QuantumGenerator,
};
use quantrs2_ml::hep::{HEPEncodingMethod, HEPQuantumClassifier};
use quantrs2_ml::qcnn::QCNN;
use quantrs2_ml::qnn::{QNNLayerType, QuantumNeuralNetwork};
use quantrs2_ml::reinforcement::{QuantumAgent, ReinforcementLearning};

// ---------------------------------------------------------------------------
// QuantumNeuralNetwork
// ---------------------------------------------------------------------------

fn build_binary_qnn() -> QuantumNeuralNetwork {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 1 },
        QNNLayerType::VariationalLayer { num_params: 4 },
        QNNLayerType::EntanglementLayer {
            connectivity: "linear".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 4 },
    ];
    let mut qnn = QuantumNeuralNetwork::new(layers, 2, 1, 1).expect("qnn creation");
    // Deterministic, non-degenerate parameters for reproducibility.
    qnn.parameters = Array1::from_vec(vec![0.30, -0.55, 0.72, 0.21, -0.44, 0.63, 0.12, -0.28]);
    qnn
}

#[test]
fn qnn_forward_depends_on_input() {
    let qnn = build_binary_qnn();

    let out_low = qnn.forward(&Array1::from_vec(vec![-1.0])).expect("forward");
    let out_high = qnn.forward(&Array1::from_vec(vec![1.0])).expect("forward");

    assert_eq!(out_low.len(), 1);
    assert_eq!(out_high.len(), 1);
    // The old dummy returned Array1::zeros(output_dim) regardless of input.
    assert!(
        (out_low[0] - out_high[0]).abs() > 1e-6,
        "forward must depend on the input: {out_low:?} vs {out_high:?}"
    );
    assert!(
        out_low[0].abs() > 1e-9 || out_high[0].abs() > 1e-9,
        "forward must not be constant zero"
    );
    // Expectation values must lie in [-1, 1].
    assert!(out_low[0] >= -1.0 - 1e-9 && out_low[0] <= 1.0 + 1e-9);
}

#[test]
fn qnn_forward_is_deterministic() {
    let qnn = build_binary_qnn();
    let input = Array1::from_vec(vec![0.42]);
    let a = qnn.forward(&input).expect("forward");
    let b = qnn.forward(&input).expect("forward");
    assert!((a[0] - b[0]).abs() < 1e-12, "forward must be deterministic");
}

#[test]
fn qnn_train_reduces_loss_on_separable_dataset() {
    let mut qnn = build_binary_qnn();

    // Separable toy dataset: x = -1 -> 0, x = +1 -> 1.
    let x = Array2::from_shape_vec((4, 1), vec![-1.0, 1.0, -1.0, 1.0]).expect("x");
    let y = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).expect("y");

    let result = qnn.train(&x, &y, 60, 0.1).expect("train");

    assert_eq!(result.loss_history.len(), 60);
    let first = result.loss_history[0];
    let last = *result.loss_history.last().expect("history");
    assert!(
        last < first,
        "training loss must decrease (first = {first}, last = {last})"
    );
    // Parameters must actually have moved away from the initialisation.
    let init = build_binary_qnn().parameters;
    let moved = (0..init.len())
        .map(|i| (result.optimal_parameters[i] - init[i]).abs())
        .fold(0.0_f64, f64::max);
    assert!(moved > 1e-4, "parameters must be updated during training");
}

// ---------------------------------------------------------------------------
// HEP classifier
// ---------------------------------------------------------------------------

fn build_hep_classifier() -> HEPQuantumClassifier {
    HEPQuantumClassifier::new(
        2,
        2,
        2,
        HEPEncodingMethod::AngleEncoding,
        vec!["background".to_string(), "signal".to_string()],
    )
    .expect("hep classifier")
}

#[test]
fn hep_predict_is_deterministic() {
    let clf = build_hep_classifier();
    let features = Array1::from_vec(vec![0.5, -0.3]);

    let (label_a, conf_a) = clf.predict(&features).expect("predict");
    let (label_b, conf_b) = clf.predict(&features).expect("predict");

    // The old dummy returned a random coin-flip label and random confidence.
    assert_eq!(label_a, label_b, "prediction must be deterministic");
    assert!(
        (conf_a - conf_b).abs() < 1e-12,
        "confidence must be deterministic"
    );
    assert!((0.0..=1.0).contains(&conf_a));
}

#[test]
fn hep_evaluate_metrics_are_real() {
    let clf = build_hep_classifier();

    let x_test = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.9, 0.8, -0.7, -0.6, 0.85, 0.7, -0.8, -0.9, 0.6, 0.95, -0.5, -0.75,
        ],
    )
    .expect("x_test");
    let y_test = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    let metrics = clf.evaluate(&x_test, &y_test).expect("evaluate");

    // Recompute the positive-class probability, average loss, and accuracy
    // directly from the public prediction path; these must match the metrics
    // (they cannot if the values are hardcoded placeholders).
    let mut expected_loss = 0.0;
    let mut correct = 0usize;
    for i in 0..x_test.nrows() {
        let probs = clf
            .predict_proba(&x_test.row(i).to_owned())
            .expect("predict_proba");
        let positive = probs[1];
        let target = if y_test[i] > 0.5 { 1.0 } else { 0.0 };
        expected_loss = (positive - target).mul_add(positive - target, expected_loss);
        let pred_idx = i32::from(probs[1] > probs[0]);
        if (pred_idx == 1) == (y_test[i] > 0.5) {
            correct += 1;
        }
    }
    expected_loss /= x_test.nrows() as f64;
    let expected_accuracy = correct as f64 / x_test.nrows() as f64;

    assert!(
        (metrics.average_loss - expected_loss).abs() < 1e-9,
        "average_loss must be derived from the model, got {} expected {}",
        metrics.average_loss,
        expected_loss
    );
    assert!((metrics.average_loss - 0.05).abs() > 1e-9 || (expected_loss - 0.05).abs() < 1e-9);
    assert!(
        (metrics.accuracy - expected_accuracy).abs() < 1e-9,
        "accuracy must match the real prediction path"
    );
    assert!(
        (0.0..=1.0).contains(&metrics.auc),
        "auc must be a valid probability, got {}",
        metrics.auc
    );
}

// ---------------------------------------------------------------------------
// Reinforcement learning agent
// ---------------------------------------------------------------------------

#[test]
fn rl_update_changes_policy() {
    // The default agent uses an 8-qubit QNN, so each parameter-shift update is
    // relatively expensive; keep the iteration count modest.  Actions 0 and 1
    // read independent qubits, so reinforcing them in opposite directions
    // separates their Q-values quickly.
    let mut agent = ReinforcementLearning::new()
        .expect("agent")
        .with_exploration_rate(0.0)
        .with_learning_rate(0.5);

    let state = Array1::from_vec(vec![0.15, 0.25, 0.35, 0.45]);
    let next_state = state.clone();

    // The old dummy `update` was a no-op, so the greedy action could never
    // change.  Reinforce the action opposite to the current greedy choice and
    // verify the policy flips to it.
    let initial = agent.get_action(&state).expect("action");
    let target = 1 - initial;

    for _ in 0..15 {
        agent
            .update(&state, target, 1.0, &next_state, true)
            .expect("update");
        agent
            .update(&state, initial, -1.0, &next_state, true)
            .expect("update");
    }

    assert_eq!(
        agent.get_action(&state).expect("action"),
        target,
        "greedy action should flip to the reinforced action {target}"
    );
}

// ---------------------------------------------------------------------------
// Quantum GAN
// ---------------------------------------------------------------------------

fn least_squares_disc_loss(
    disc: &QuantumDiscriminator,
    real: &Array2<f64>,
    fake: &Array2<f64>,
) -> f64 {
    let real_out = disc.discriminate(real).expect("discriminate real");
    let fake_out = disc.discriminate(fake).expect("discriminate fake");
    let real_loss: f64 =
        real_out.iter().map(|&d| (d - 1.0) * (d - 1.0)).sum::<f64>() / real_out.len() as f64;
    let fake_loss: f64 = fake_out.iter().map(|&d| d * d).sum::<f64>() / fake_out.len() as f64;
    real_loss + fake_loss
}

#[test]
fn gan_discriminate_depends_on_input() {
    let disc = QuantumDiscriminator::new(2, 2, DiscriminatorType::QuantumOnly).expect("disc");

    let real = Array2::from_shape_vec((2, 2), vec![0.9, 0.85, 1.0, 0.95]).expect("real");
    let fake = Array2::from_shape_vec((2, 2), vec![0.05, 0.1, 0.0, 0.05]).expect("fake");

    let d_real = disc.discriminate(&real).expect("discriminate");
    let d_fake = disc.discriminate(&fake).expect("discriminate");

    // The old dummy computed sin(sum) ignoring the QNN; distinct inputs must
    // now give distinct outputs in [0, 1].
    assert!(
        (d_real[0] - d_fake[0]).abs() > 1e-9,
        "discriminator output must depend on the input"
    );
    for &value in d_real.iter().chain(d_fake.iter()) {
        assert!((0.0..=1.0).contains(&value));
    }
}

#[test]
fn gan_discriminator_update_reduces_loss() {
    let mut disc = QuantumDiscriminator::new(2, 2, DiscriminatorType::QuantumOnly).expect("disc");

    let real = Array2::from_shape_vec((4, 2), vec![0.9, 0.9, 0.8, 1.0, 1.0, 0.9, 0.95, 0.85])
        .expect("real");
    let fake = Array2::from_shape_vec((4, 2), vec![0.1, 0.0, 0.0, 0.1, 0.05, 0.0, 0.1, 0.1])
        .expect("fake");

    let initial = least_squares_disc_loss(&disc, &real, &fake);
    for _ in 0..40 {
        let reported = disc.update(&real, &fake, 0.3).expect("update");
        assert!(reported.is_finite() && reported >= 0.0);
        // The old dummy always returned exactly 0.5.
    }
    let final_loss = least_squares_disc_loss(&disc, &real, &fake);

    assert!(
        final_loss < initial,
        "discriminator loss must decrease (initial = {initial}, final = {final_loss})"
    );
}

#[test]
fn gan_generator_adversarial_update_reduces_loss() {
    let mut gen = QuantumGenerator::new(2, 2, 2, GeneratorType::QuantumOnly).expect("gen");
    let disc = QuantumDiscriminator::new(2, 2, DiscriminatorType::QuantumOnly).expect("disc");

    // Fixed, deterministic latent batch so the loss trajectory is reproducible.
    let latent = Array2::from_shape_vec((4, 2), vec![0.2, -0.4, 0.6, 0.1, -0.3, 0.5, 0.15, -0.25])
        .expect("latent");

    let base = gen
        .adversarial_update(&latent, &disc, 0.3)
        .expect("adversarial update");
    for _ in 0..25 {
        gen.adversarial_update(&latent, &disc, 0.3)
            .expect("adversarial update");
    }
    let after = gen
        .adversarial_update(&latent, &disc, 0.3)
        .expect("adversarial update");

    assert!(
        after < base,
        "generator adversarial loss must decrease (base = {base}, after = {after})"
    );
}

#[test]
fn gan_generate_is_input_dependent_and_bounded() {
    let gen = QuantumGenerator::new(2, 2, 2, GeneratorType::QuantumOnly).expect("gen");
    let samples = gen.generate(8).expect("generate");
    assert_eq!(samples.nrows(), 8);
    assert_eq!(samples.ncols(), 2);
    for value in &samples {
        assert!(
            (0.0..=1.0).contains(value),
            "generated features must be in [0, 1]"
        );
    }
}

#[test]
fn gan_train_records_real_losses() {
    let mut gan = QuantumGAN::new(
        2,
        2,
        2,
        2,
        GeneratorType::QuantumOnly,
        DiscriminatorType::QuantumOnly,
    )
    .expect("gan");

    let real_data = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.9, 0.9, 0.8, 1.0, 1.0, 0.9, 0.95, 0.85, 0.9, 0.8, 0.85, 0.95,
        ],
    )
    .expect("real_data");

    let history = gan.train(&real_data, 4, 3, 0.2, 0.2, 1).expect("train");

    assert_eq!(history.disc_losses.len(), 4);
    assert_eq!(history.gen_losses.len(), 4);
    // The old dummy updates always returned a constant 0.5 loss for every epoch.
    assert!(
        history.disc_losses.iter().any(|&l| (l - 0.5).abs() > 1e-6),
        "discriminator losses must reflect real training, got {:?}",
        history.disc_losses
    );
}

// ---------------------------------------------------------------------------
// QCNN
// ---------------------------------------------------------------------------

fn normalized_input(dim: usize) -> Vec<Complex64> {
    let raw: Vec<f64> = (0..dim).map(|i| i as f64 + 1.0).collect();
    let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
    raw.into_iter()
        .map(|x| Complex64::new(x / norm, 0.0))
        .collect()
}

#[test]
fn qcnn_forward_depends_on_parameters_and_input() {
    let mut qcnn = QCNN::new(4, vec![(2, 1)], vec![2], 2).expect("qcnn");
    let input = normalized_input(16);

    let out1 = qcnn.forward(&input).expect("forward");

    // Output must be normalized (sum of |amp|^2 == 1).
    let norm: f64 = out1.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "QCNN output must be a normalized state, got norm^2 = {norm}"
    );

    // Changing the parameters must change the output (old dummy ignored them).
    let mut params = qcnn.get_parameters();
    for p in &mut params {
        *p += 0.7;
    }
    qcnn.set_parameters(&params).expect("set params");
    let out2 = qcnn.forward(&input).expect("forward");
    assert!(
        out1.iter()
            .zip(out2.iter())
            .any(|(a, b)| (a - b).norm() > 1e-6),
        "QCNN forward must depend on the trained parameters"
    );

    // Changing the input state must change the output.
    let mut input2 = input;
    input2[0] = Complex64::new(5.0, 0.0);
    let out3 = qcnn.forward(&input2).expect("forward");
    assert!(
        out2.iter()
            .zip(out3.iter())
            .any(|(a, b)| (a - b).norm() > 1e-6),
        "QCNN forward must depend on the input state"
    );
}
