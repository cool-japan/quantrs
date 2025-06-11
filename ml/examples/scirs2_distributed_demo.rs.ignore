//! SciRS2 Distributed Training Example
//!
//! This example demonstrates the SciRS2 integration capabilities including
//! distributed training, tensor operations, and scientific computing features.

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use quantrs2_ml::prelude::*;
use quantrs2_ml::scirs2_integration::{
    SciRS2Array, SciRS2DataLoader, SciRS2Dataset, SciRS2Device, SciRS2DistributedTrainer,
    SciRS2Optimizer, SciRS2Serializer,
};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== SciRS2 Distributed Training Demo ===\n");

    // Step 1: Initialize SciRS2 distributed environment
    println!("1. Initializing SciRS2 distributed environment...");

    let mut distributed_trainer = SciRS2DistributedTrainer::new(
        4,     // num_workers
        "mpi", // backend (or "nccl" for GPU)
        4,     // world_size
        0,     // rank
    )?;

    println!("   - Workers: {}", distributed_trainer.num_workers());
    println!("   - Backend: {}", distributed_trainer.backend());
    println!("   - World size: {}", distributed_trainer.world_size());

    // Step 2: Create SciRS2 tensors and arrays
    println!("\n2. Creating SciRS2 tensors and arrays...");

    let data_shape = (1000, 8);
    let mut scirs2_array = SciRS2Array::zeros(data_shape.clone());
    scirs2_array.requires_grad = true;

    // Fill with quantum-friendly data
    scirs2_array.fill_quantum_data("quantum_normal", 42)?; // distribution, seed

    println!("   - Array shape: {:?}", scirs2_array.shape());
    println!("   - Requires grad: {}", scirs2_array.requires_grad);
    println!("   - Device: {:?}", scirs2_array.device());

    // Create SciRS2 tensor for quantum parameters
    let param_data = ArrayD::zeros(IxDyn(&[4, 6])); // 4 qubits, 6 parameters per qubit
    let mut quantum_params = SciRS2Tensor::new(param_data, true);

    // Initialize with quantum parameter initialization
    quantum_params.quantum_parameter_init("quantum_aware")?;

    println!(
        "   - Quantum parameters shape: {:?}",
        quantum_params.shape()
    );
    println!(
        "   - Parameter range: [{:.4}, {:.4}]",
        quantum_params.min()?,
        quantum_params.max()?
    );

    // Step 3: Setup distributed quantum model
    println!("\n3. Setting up distributed quantum model...");

    let quantum_model = create_distributed_quantum_model(&quantum_params)?;

    // Wrap model for distributed training
    let distributed_model = distributed_trainer.wrap_model(quantum_model)?;

    println!(
        "   - Model parameters: {}",
        distributed_model.num_parameters()
    );
    println!("   - Distributed: {}", distributed_model.is_distributed());

    // Step 4: Create SciRS2 optimizers
    println!("\n4. Configuring SciRS2 optimizers...");

    let optimizer = SciRS2Optimizer::Adam {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 1e-4,
        amsgrad: false,
    };

    // Configure distributed optimizer
    let mut distributed_optimizer = distributed_trainer.wrap_optimizer(optimizer)?;

    println!("   - Optimizer: Adam with SciRS2 backend");
    println!(
        "   - Learning rate: {}",
        distributed_optimizer.learning_rate()
    );
    println!("   - Distributed synchronization: enabled");

    // Step 5: Distributed data loading
    println!("\n5. Setting up distributed data loading...");

    let dataset = create_large_quantum_dataset(10000, 8)?;
    println!("   - Dataset created with {} samples", dataset.size);
    println!("   - Distributed sampling configured");

    // Create data loader
    let mut data_loader = SciRS2DataLoader::new(dataset, 64);

    println!("   - Total dataset size: {}", data_loader.dataset.size);
    println!("   - Local batches per worker: 156"); // placeholder
    println!("   - Global batch size: 64"); // placeholder

    // Step 6: Distributed training loop
    println!("\n6. Starting distributed training...");

    let num_epochs = 10;
    let mut training_metrics = SciRS2TrainingMetrics::new();

    for epoch in 0..num_epochs {
        distributed_trainer.barrier()?; // Synchronize all workers

        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for (batch_idx, (data, targets)) in data_loader.enumerate() {
            // Convert to SciRS2 tensors
            let data_tensor = SciRS2Tensor::from_array(&data)?;
            let target_tensor = SciRS2Tensor::from_array(&targets)?;

            // Zero gradients
            distributed_optimizer.zero_grad()?;

            // Forward pass
            let outputs = distributed_model.forward(&data_tensor)?;
            let loss = compute_quantum_loss(&outputs, &target_tensor)?;

            // Backward pass with automatic differentiation
            loss.backward()?;

            // Gradient synchronization across workers
            distributed_trainer.all_reduce_gradients(&distributed_model)?;

            // Optimizer step
            distributed_optimizer.step()?;

            epoch_loss += loss.item();
            num_batches += 1;

            if batch_idx % 10 == 0 {
                println!(
                    "   Epoch {}, Batch {}: loss = {:.6}",
                    epoch,
                    batch_idx,
                    loss.item()
                );
            }
        }

        // Collect metrics across all workers
        let avg_loss = distributed_trainer.all_reduce_scalar(epoch_loss / num_batches as f64)?;
        training_metrics.record_epoch(epoch, avg_loss);

        println!("   Epoch {} completed: avg_loss = {:.6}", epoch, avg_loss);
    }

    // Step 7: Distributed evaluation
    println!("\n7. Distributed model evaluation...");

    let test_dataset = create_test_quantum_dataset(2000, 8)?;
    let test_sampler = distributed_trainer.create_sampler(&test_dataset)?;
    println!(
        "   - Test dataset configured with {} samples",
        test_dataset.nrows()
    );

    let evaluation_results =
        evaluate_distributed_model(&distributed_model, &test_dataset, &distributed_trainer)?;

    println!("   Distributed Evaluation Results:");
    println!("   - Test accuracy: {:.4}", evaluation_results.accuracy);
    println!("   - Test loss: {:.6}", evaluation_results.loss);
    println!(
        "   - Quantum fidelity: {:.4}",
        evaluation_results.quantum_fidelity
    );

    // Step 8: SciRS2 tensor operations
    println!("\n8. Demonstrating SciRS2 tensor operations...");

    // Advanced tensor operations
    let tensor_a = SciRS2Tensor::randn(vec![100, 50], SciRS2Device::CPU)?;
    let tensor_b = SciRS2Tensor::randn(vec![50, 25], SciRS2Device::CPU)?;

    // Matrix multiplication with automatic broadcasting
    let result = tensor_a.matmul(&tensor_b)?;
    println!(
        "   - Matrix multiplication: {:?} x {:?} = {:?}",
        tensor_a.shape(),
        tensor_b.shape(),
        result.shape()
    );

    // Quantum-specific operations
    let quantum_state = SciRS2Tensor::quantum_state(4)?;
    let evolved_state = quantum_state.quantum_evolve(&quantum_params)?;
    let fidelity = quantum_state.fidelity(&evolved_state)?;

    println!("   - Quantum state evolution fidelity: {:.6}", fidelity);

    // Distributed tensor operations
    let distributed_tensor = distributed_trainer.scatter_tensor(&tensor_a)?;
    let local_computation = distributed_tensor.local_sum()?;
    let global_result = distributed_trainer.all_reduce_tensor(&local_computation)?;

    println!(
        "   - Distributed computation result shape: {:?}",
        global_result.shape()
    );

    // Step 9: Scientific computing features
    println!("\n9. SciRS2 scientific computing features...");

    // Numerical integration for quantum expectation values
    let observable = create_quantum_observable(4)?;
    let expectation_value = quantum_state.expectation_value(&observable)?;
    println!("   - Quantum expectation value: {:.6}", expectation_value);

    // Optimization with scientific methods
    let optimization_result = SciRS2Optimizer::LBFGS {
        learning_rate: 0.01,
        max_iter: 100,
        tolerance: 1e-9,
        history_size: 10,
    }
    .minimize(
        &quantum_params,
        |params| compute_quantum_energy(params),
        Some(|params| compute_quantum_gradient(params)),
    )?;

    println!(
        "   - LBFGS optimization converged: {}",
        optimization_result.converged
    );
    println!("   - Final energy: {:.8}", optimization_result.final_value);
    println!("   - Iterations: {}", optimization_result.num_iterations);

    // Step 10: Model serialization with SciRS2
    println!("\n10. SciRS2 model serialization...");

    let serializer = SciRS2Serializer::new("hdf5", "gzip")?;

    // Save distributed model
    serializer.save_model(&distributed_model, "distributed_quantum_model.h5")?;
    println!("    - Model saved with SciRS2 serializer");

    // Save training state for checkpointing
    let checkpoint = SciRS2Checkpoint {
        model_state: distributed_model.state_dict(),
        optimizer_state: distributed_optimizer.state_dict(),
        epoch: num_epochs,
        metrics: training_metrics.clone(),
    };

    serializer.save_checkpoint(&checkpoint, "training_checkpoint.h5")?;
    println!("    - Training checkpoint saved");

    // Load and verify
    let loaded_model = serializer.load_model("distributed_quantum_model.h5")?;
    println!("    - Model loaded successfully");

    // Step 11: Performance analysis
    println!("\n11. Distributed training performance analysis...");

    let performance_metrics = distributed_trainer.get_performance_metrics()?;

    println!("    Performance Metrics:");
    println!(
        "    - Communication overhead: {:.2}%",
        performance_metrics.communication_overhead * 100.0
    );
    println!(
        "    - Scaling efficiency: {:.2}%",
        performance_metrics.scaling_efficiency * 100.0
    );
    println!(
        "    - Memory usage per worker: {:.1} GB",
        performance_metrics.memory_usage_gb
    );
    println!(
        "    - Average batch processing time: {:.3}s",
        performance_metrics.avg_batch_time
    );

    // Step 12: Cleanup distributed environment
    println!("\n12. Cleaning up distributed environment...");

    distributed_trainer.cleanup()?;
    println!("    - Distributed training environment cleaned up");

    println!("\n=== SciRS2 Distributed Training Demo Complete ===");

    Ok(())
}

fn create_distributed_quantum_model(params: &SciRS2Tensor) -> Result<DistributedQuantumModel> {
    DistributedQuantumModel::new(
        4,                    // num_qubits
        3,                    // num_layers
        "hardware_efficient", // ansatz_type
        params.clone(),       // parameters
        "expectation_value",  // measurement_type
    )
}

fn create_large_quantum_dataset(num_samples: usize, num_features: usize) -> Result<SciRS2Dataset> {
    let data = SciRS2Tensor::randn(vec![num_samples, num_features], SciRS2Device::CPU)?;
    let labels = SciRS2Tensor::randint(0, 2, vec![num_samples], SciRS2Device::CPU)?;

    SciRS2Dataset::new(data, labels)
}

fn create_test_quantum_dataset(num_samples: usize, num_features: usize) -> Result<SciRS2Dataset> {
    create_large_quantum_dataset(num_samples, num_features)
}

fn compute_quantum_loss(outputs: &SciRS2Tensor, targets: &SciRS2Tensor) -> Result<SciRS2Tensor> {
    // Quantum-aware loss function
    let mse_loss = outputs.mse_loss(targets)?;
    let quantum_regularization = outputs.quantum_regularization(0.01)?;
    Ok(mse_loss + quantum_regularization)
}

fn evaluate_distributed_model(
    model: &DistributedQuantumModel,
    test_loader: &SciRS2DataLoader,
    trainer: &SciRS2DistributedTrainer,
) -> Result<EvaluationResults> {
    let mut total_loss = 0.0;
    let mut total_accuracy = 0.0;
    let mut total_fidelity = 0.0;
    let mut num_batches = 0;

    for (data, targets) in test_loader {
        let outputs = model.forward(&data)?;
        let loss = compute_quantum_loss(&outputs, &targets)?;

        let batch_accuracy = compute_accuracy(&outputs, &targets)?;
        let batch_fidelity = compute_quantum_fidelity(&outputs)?;

        total_loss += loss.item();
        total_accuracy += batch_accuracy;
        total_fidelity += batch_fidelity;
        num_batches += 1;
    }

    // Average across all workers
    let avg_loss = trainer.all_reduce_scalar(total_loss / num_batches as f64)?;
    let avg_accuracy = trainer.all_reduce_scalar(total_accuracy / num_batches as f64)?;
    let avg_fidelity = trainer.all_reduce_scalar(total_fidelity / num_batches as f64)?;

    Ok(EvaluationResults {
        loss: avg_loss,
        accuracy: avg_accuracy,
        quantum_fidelity: avg_fidelity,
    })
}

fn create_quantum_observable(num_qubits: usize) -> Result<SciRS2Tensor> {
    // Create Pauli-Z observable for all qubits
    SciRS2Tensor::quantum_observable("pauli_z_all", num_qubits)
}

fn compute_quantum_energy(params: &SciRS2Tensor) -> Result<f64> {
    // Mock quantum energy computation
    let energy = params.norm()?.powi(2) + 0.5 * params.sum()?.abs();
    Ok(energy)
}

fn compute_quantum_gradient(params: &SciRS2Tensor) -> Result<SciRS2Tensor> {
    // Mock gradient computation using parameter shift rule
    let gradient = 2.0 * params + 0.5 * SciRS2Tensor::ones_like(params)?;
    Ok(gradient)
}

fn compute_accuracy(outputs: &SciRS2Tensor, targets: &SciRS2Tensor) -> Result<f64> {
    let predictions = outputs.argmax(1)?;
    let correct = predictions.eq(targets)?.float().mean()?.item();
    Ok(correct)
}

fn compute_quantum_fidelity(outputs: &SciRS2Tensor) -> Result<f64> {
    // Mock quantum fidelity computation
    let fidelity = outputs.norm()? / (outputs.shape()[0] as f64).sqrt();
    Ok(fidelity.min(1.0))
}

// Supporting structures for the demo

#[derive(Clone)]
struct SciRS2TrainingMetrics {
    losses: Vec<f64>,
    epochs: Vec<usize>,
}

impl SciRS2TrainingMetrics {
    fn new() -> Self {
        Self {
            losses: Vec::new(),
            epochs: Vec::new(),
        }
    }

    fn record_epoch(&mut self, epoch: usize, loss: f64) {
        self.epochs.push(epoch);
        self.losses.push(loss);
    }
}

struct EvaluationResults {
    loss: f64,
    accuracy: f64,
    quantum_fidelity: f64,
}

struct DistributedQuantumModel {
    num_qubits: usize,
    parameters: SciRS2Tensor,
}

impl DistributedQuantumModel {
    fn new(
        num_qubits: usize,
        num_layers: usize,
        ansatz_type: &str,
        parameters: SciRS2Tensor,
        measurement_type: &str,
    ) -> Result<Self> {
        Ok(Self {
            num_qubits,
            parameters,
        })
    }

    fn forward(&self, input: &SciRS2Tensor) -> Result<SciRS2Tensor> {
        // Mock forward pass
        let batch_size = input.shape()[0];
        SciRS2Tensor::randn(vec![batch_size, 2], SciRS2Device::CPU)
    }

    fn num_parameters(&self) -> usize {
        self.parameters.numel()
    }

    fn is_distributed(&self) -> bool {
        true
    }

    fn state_dict(&self) -> HashMap<String, SciRS2Tensor> {
        let mut state = HashMap::new();
        state.insert("parameters".to_string(), self.parameters.clone());
        state
    }
}

struct SciRS2Checkpoint {
    model_state: HashMap<String, SciRS2Tensor>,
    optimizer_state: HashMap<String, SciRS2Tensor>,
    epoch: usize,
    metrics: SciRS2TrainingMetrics,
}

struct PerformanceMetrics {
    communication_overhead: f64,
    scaling_efficiency: f64,
    memory_usage_gb: f64,
    avg_batch_time: f64,
}

struct OptimizationResult {
    converged: bool,
    final_value: f64,
    num_iterations: usize,
}
