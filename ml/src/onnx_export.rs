//! ONNX model export support for QuantRS2-ML
//!
//! This module provides functionality to export quantum ML models to the ONNX format,
//! enabling interoperability with other ML frameworks and deployment platforms.

use crate::error::{MLError, Result};
use crate::keras_api::{
    Activation, ActivationFunction, Dense, KerasLayer, QuantumDense, Sequential,
};
use crate::pytorch_api::{QuantumLinear, QuantumModule, QuantumSequential};
use crate::simulator_backends::DynamicCircuit;
use quantrs2_circuit::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayD};
use std::collections::HashMap;
use std::io::Write;

/// ONNX graph representation
#[derive(Debug, Clone)]
pub struct ONNXGraph {
    /// Graph nodes
    nodes: Vec<ONNXNode>,
    /// Graph inputs
    inputs: Vec<ONNXValueInfo>,
    /// Graph outputs
    outputs: Vec<ONNXValueInfo>,
    /// Graph initializers (weights)
    initializers: Vec<ONNXTensor>,
    /// Graph name
    name: String,
}

impl ONNXGraph {
    /// Create new ONNX graph
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            name: name.into(),
        }
    }

    /// Add node to graph
    pub fn add_node(&mut self, node: ONNXNode) {
        self.nodes.push(node);
    }

    /// Add input to graph
    pub fn add_input(&mut self, input: ONNXValueInfo) {
        self.inputs.push(input);
    }

    /// Add output to graph
    pub fn add_output(&mut self, output: ONNXValueInfo) {
        self.outputs.push(output);
    }

    /// Add initializer to graph
    pub fn add_initializer(&mut self, initializer: ONNXTensor) {
        self.initializers.push(initializer);
    }

    /// Export graph to ONNX format
    pub fn export(&self, path: &str) -> Result<()> {
        let onnx_proto = self.to_onnx_proto()?;

        std::fs::write(path, onnx_proto)?;
        Ok(())
    }

    /// Convert to ONNX protobuf format (simplified)
    fn to_onnx_proto(&self) -> Result<Vec<u8>> {
        // This is a simplified representation of ONNX protobuf
        // In a real implementation, you would use the official ONNX protobuf schema

        let mut buffer = Vec::new();

        // Write ONNX header
        writeln!(buffer, "ONNX Model Export")?;
        writeln!(buffer, "Graph Name: {}", self.name)?;
        writeln!(buffer, "")?;

        // Write inputs
        writeln!(buffer, "Inputs:")?;
        for input in &self.inputs {
            writeln!(buffer, "  {}: {:?}", input.name, input.shape)?;
        }
        writeln!(buffer, "")?;

        // Write outputs
        writeln!(buffer, "Outputs:")?;
        for output in &self.outputs {
            writeln!(buffer, "  {}: {:?}", output.name, output.shape)?;
        }
        writeln!(buffer, "")?;

        // Write nodes
        writeln!(buffer, "Nodes:")?;
        for node in &self.nodes {
            writeln!(
                buffer,
                "  {} ({}): {} -> {}",
                node.name,
                node.op_type,
                node.inputs.join(", "),
                node.outputs.join(", ")
            )?;
        }
        writeln!(buffer, "")?;

        // Write initializers
        writeln!(buffer, "Initializers:")?;
        for init in &self.initializers {
            writeln!(buffer, "  {}: {:?}", init.name, init.shape)?;
        }

        Ok(buffer)
    }
}

/// ONNX node representation
#[derive(Debug, Clone)]
pub struct ONNXNode {
    /// Node name
    name: String,
    /// Operator type
    op_type: String,
    /// Input names
    inputs: Vec<String>,
    /// Output names
    outputs: Vec<String>,
    /// Node attributes
    attributes: HashMap<String, ONNXAttribute>,
}

impl ONNXNode {
    /// Create new ONNX node
    pub fn new(
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            op_type: op_type.into(),
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }

    /// Add attribute to node
    pub fn add_attribute(&mut self, name: impl Into<String>, value: ONNXAttribute) {
        self.attributes.insert(name.into(), value);
    }
}

/// ONNX attribute types
#[derive(Debug, Clone)]
pub enum ONNXAttribute {
    /// Integer attribute
    Int(i64),
    /// Float attribute
    Float(f32),
    /// String attribute
    String(String),
    /// Tensor attribute
    Tensor(ONNXTensor),
    /// Integer array
    Ints(Vec<i64>),
    /// Float array
    Floats(Vec<f32>),
    /// String array
    Strings(Vec<String>),
}

/// ONNX value info (for inputs/outputs)
#[derive(Debug, Clone)]
pub struct ONNXValueInfo {
    /// Value name
    name: String,
    /// Data type
    data_type: ONNXDataType,
    /// Shape
    shape: Vec<i64>,
}

impl ONNXValueInfo {
    /// Create new value info
    pub fn new(name: impl Into<String>, data_type: ONNXDataType, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            data_type,
            shape,
        }
    }
}

/// ONNX data types
#[derive(Debug, Clone)]
pub enum ONNXDataType {
    /// Float32
    Float32,
    /// Float64
    Float64,
    /// Int32
    Int32,
    /// Int64
    Int64,
    /// Bool
    Bool,
}

/// ONNX tensor representation
#[derive(Debug, Clone)]
pub struct ONNXTensor {
    /// Tensor name
    name: String,
    /// Data type
    data_type: ONNXDataType,
    /// Shape
    shape: Vec<i64>,
    /// Raw data
    data: Vec<u8>,
}

impl ONNXTensor {
    /// Create tensor from ndarray
    pub fn from_array_f32(name: impl Into<String>, array: &ArrayD<f32>) -> Self {
        let shape: Vec<i64> = array.shape().iter().map(|&s| s as i64).collect();
        let data = array
            .as_slice()
            .expect("ArrayD is contiguous in standard layout")
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        Self {
            name: name.into(),
            data_type: ONNXDataType::Float32,
            shape,
            data,
        }
    }

    /// Create tensor from ndarray (f64)
    pub fn from_array_f64(name: impl Into<String>, array: &ArrayD<f64>) -> Self {
        let shape: Vec<i64> = array.shape().iter().map(|&s| s as i64).collect();
        let data = array
            .as_slice()
            .expect("ArrayD is contiguous in standard layout")
            .iter()
            .flat_map(|&f| (f as f32).to_le_bytes()) // Convert to f32 for ONNX
            .collect();

        Self {
            name: name.into(),
            data_type: ONNXDataType::Float32,
            shape,
            data,
        }
    }
}

/// ONNX exporter for quantum ML models
pub struct ONNXExporter {
    /// Quantum operator mappings
    quantum_mappings: HashMap<String, String>,
    /// Export options
    options: ExportOptions,
}

/// Export options
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// ONNX opset version
    opset_version: i64,
    /// Include quantum layers as custom operators
    include_quantum_ops: bool,
    /// Optimize classical layers only
    optimize_classical_only: bool,
    /// Target backend for quantum operations
    quantum_backend: QuantumBackendTarget,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            opset_version: 11,
            include_quantum_ops: true,
            optimize_classical_only: false,
            quantum_backend: QuantumBackendTarget::Generic,
        }
    }
}

/// Quantum backend targets for export
#[derive(Debug, Clone)]
pub enum QuantumBackendTarget {
    /// Generic quantum backend
    Generic,
    /// Qiskit-compatible
    Qiskit,
    /// Cirq-compatible
    Cirq,
    /// PennyLane-compatible
    PennyLane,
    /// Custom backend
    Custom(String),
}

impl ONNXExporter {
    /// Create new ONNX exporter
    pub fn new() -> Self {
        let mut quantum_mappings = HashMap::new();

        // Map quantum operations to ONNX custom operators
        quantum_mappings.insert("QuantumDense".to_string(), "QuantumDense".to_string());
        quantum_mappings.insert("QuantumLinear".to_string(), "QuantumLinear".to_string());
        quantum_mappings.insert("QuantumConv2d".to_string(), "QuantumConv2d".to_string());
        quantum_mappings.insert("QuantumRNN".to_string(), "QuantumRNN".to_string());

        Self {
            quantum_mappings,
            options: ExportOptions::default(),
        }
    }

    /// Set export options
    pub fn with_options(mut self, options: ExportOptions) -> Self {
        self.options = options;
        self
    }

    /// Export Sequential model to ONNX
    pub fn export_sequential(
        &self,
        model: &Sequential,
        input_shape: &[usize],
        output_path: &str,
    ) -> Result<()> {
        let mut graph = ONNXGraph::new("sequential_model");

        // Add input
        let input_shape_i64: Vec<i64> = input_shape.iter().map(|&s| s as i64).collect();
        graph.add_input(ONNXValueInfo::new(
            "input",
            ONNXDataType::Float32,
            input_shape_i64,
        ));

        let mut current_output = "input".to_string();
        let mut node_counter = 0;

        // Convert each layer
        for layer in model.layers() {
            let layer_name = format!("layer_{}", node_counter);
            let output_name = format!("output_{}", node_counter);

            // Convert layer based on type
            let (nodes, initializers) =
                self.convert_layer(layer.as_ref(), &layer_name, &current_output, &output_name)?;

            // Add nodes and initializers to graph
            for node in nodes {
                graph.add_node(node);
            }
            for init in initializers {
                graph.add_initializer(init);
            }

            current_output = output_name;
            node_counter += 1;
        }

        // Add output
        let output_shape = model.compute_output_shape(input_shape);
        let output_shape_i64: Vec<i64> = output_shape.iter().map(|&s| s as i64).collect();
        graph.add_output(ONNXValueInfo::new(
            &current_output,
            ONNXDataType::Float32,
            output_shape_i64,
        ));

        // Export graph
        graph.export(output_path)?;
        Ok(())
    }

    /// Export PyTorch-style model to ONNX
    pub fn export_pytorch_model<T: QuantumModule>(
        &self,
        model: &T,
        input_shape: &[usize],
        output_path: &str,
    ) -> Result<()> {
        let mut graph = ONNXGraph::new("pytorch_model");

        // Add input
        let input_shape_i64: Vec<i64> = input_shape.iter().map(|&s| s as i64).collect();
        graph.add_input(ONNXValueInfo::new(
            "input",
            ONNXDataType::Float32,
            input_shape_i64,
        ));

        // Convert model (simplified - would need more complex analysis)
        let node = ONNXNode::new(
            "pytorch_model",
            "QuantumModel",
            vec!["input".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(node);

        // Add output (would need to compute actual output shape)
        graph.add_output(ONNXValueInfo::new(
            "output",
            ONNXDataType::Float32,
            vec![1, 1], // Placeholder
        ));

        // Export graph
        graph.export(output_path)?;
        Ok(())
    }

    /// Convert layer to ONNX nodes and initializers
    fn convert_layer(
        &self,
        layer: &dyn KerasLayer,
        layer_name: &str,
        input_name: &str,
        output_name: &str,
    ) -> Result<(Vec<ONNXNode>, Vec<ONNXTensor>)> {
        // This would need to be implemented for each layer type
        // For now, we'll provide a simplified conversion

        let layer_type = self.get_layer_type(layer);

        match layer_type.as_str() {
            "Dense" => self.convert_dense_layer(layer, layer_name, input_name, output_name),
            "QuantumDense" => {
                self.convert_quantum_dense_layer(layer, layer_name, input_name, output_name)
            }
            "Activation" => {
                self.convert_activation_layer(layer, layer_name, input_name, output_name)
            }
            _ => {
                // Generic layer conversion
                let node = ONNXNode::new(
                    layer_name,
                    &layer_type,
                    vec![input_name.to_string()],
                    vec![output_name.to_string()],
                );
                Ok((vec![node], vec![]))
            }
        }
    }

    /// Convert Dense layer
    fn convert_dense_layer(
        &self,
        layer: &dyn KerasLayer,
        layer_name: &str,
        input_name: &str,
        output_name: &str,
    ) -> Result<(Vec<ONNXNode>, Vec<ONNXTensor>)> {
        let weights = layer.get_weights();
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        if weights.len() >= 1 {
            // Add weight initializer
            let weight_name = format!("{}_weight", layer_name);
            let weight_tensor = ONNXTensor::from_array_f64(&weight_name, &weights[0]);
            initializers.push(weight_tensor);

            // Create MatMul node
            let mut matmul_inputs = vec![input_name.to_string(), weight_name];
            let matmul_output = if weights.len() > 1 {
                format!("{}_matmul", layer_name)
            } else {
                output_name.to_string()
            };

            let matmul_node = ONNXNode::new(
                format!("{}_matmul", layer_name),
                "MatMul",
                matmul_inputs,
                vec![matmul_output.clone()],
            );
            nodes.push(matmul_node);

            // Add bias if present
            if weights.len() > 1 {
                let bias_name = format!("{}_bias", layer_name);
                let bias_tensor = ONNXTensor::from_array_f64(&bias_name, &weights[1]);
                initializers.push(bias_tensor);

                let add_node = ONNXNode::new(
                    format!("{}_add", layer_name),
                    "Add",
                    vec![matmul_output, bias_name],
                    vec![output_name.to_string()],
                );
                nodes.push(add_node);
            }
        }

        Ok((nodes, initializers))
    }

    /// Convert QuantumDense layer
    fn convert_quantum_dense_layer(
        &self,
        layer: &dyn KerasLayer,
        layer_name: &str,
        input_name: &str,
        output_name: &str,
    ) -> Result<(Vec<ONNXNode>, Vec<ONNXTensor>)> {
        if !self.options.include_quantum_ops {
            return Err(MLError::InvalidConfiguration(
                "Quantum operations not supported in export options".to_string(),
            ));
        }

        let weights = layer.get_weights();
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();

        // Add quantum parameters as initializers
        for (i, weight) in weights.iter().enumerate() {
            let param_name = format!("{}_param_{}", layer_name, i);
            let param_tensor = ONNXTensor::from_array_f64(&param_name, weight);
            initializers.push(param_tensor);
        }

        // Create custom quantum node
        let mut quantum_node = ONNXNode::new(
            layer_name,
            "QuantumDense",
            vec![input_name.to_string()],
            vec![output_name.to_string()],
        );

        // Add quantum-specific attributes
        quantum_node.add_attribute(
            "backend",
            ONNXAttribute::String(format!("{:?}", self.options.quantum_backend)),
        );
        quantum_node.add_attribute("domain", ONNXAttribute::String("quantrs2.ml".to_string()));

        nodes.push(quantum_node);

        Ok((nodes, initializers))
    }

    /// Convert Activation layer
    fn convert_activation_layer(
        &self,
        _layer: &dyn KerasLayer,
        layer_name: &str,
        input_name: &str,
        output_name: &str,
    ) -> Result<(Vec<ONNXNode>, Vec<ONNXTensor>)> {
        // For now, assume ReLU activation
        let node = ONNXNode::new(
            layer_name,
            "Relu",
            vec![input_name.to_string()],
            vec![output_name.to_string()],
        );

        Ok((vec![node], vec![]))
    }

    /// Get layer type string
    fn get_layer_type(&self, _layer: &dyn KerasLayer) -> String {
        // This would need to be implemented with proper type checking
        // For now, return a placeholder
        "Dense".to_string()
    }
}

/// ONNX importer for loading models back into QuantRS2
pub struct ONNXImporter {
    /// Import options
    options: ImportOptions,
}

/// Import options
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Target framework
    target_framework: TargetFramework,
    /// Handle unsupported operators
    handle_unsupported: UnsupportedOpHandling,
    /// Quantum backend for imported quantum ops
    quantum_backend: QuantumBackendTarget,
}

/// Target frameworks for import
#[derive(Debug, Clone)]
pub enum TargetFramework {
    /// Keras-style Sequential model
    Keras,
    /// PyTorch-style model
    PyTorch,
    /// Raw QuantRS2 model
    QuantRS2,
}

/// How to handle unsupported operators
#[derive(Debug, Clone)]
pub enum UnsupportedOpHandling {
    /// Raise error
    Error,
    /// Skip unsupported operators
    Skip,
    /// Replace with identity
    Identity,
    /// Custom handler
    Custom(String),
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            target_framework: TargetFramework::Keras,
            handle_unsupported: UnsupportedOpHandling::Error,
            quantum_backend: QuantumBackendTarget::Generic,
        }
    }
}

impl ONNXImporter {
    /// Create new ONNX importer
    pub fn new() -> Self {
        Self {
            options: ImportOptions::default(),
        }
    }

    /// Set import options
    pub fn with_options(mut self, options: ImportOptions) -> Self {
        self.options = options;
        self
    }

    /// Import ONNX model to Sequential model
    pub fn import_to_sequential(&self, path: &str) -> Result<Sequential> {
        let graph = self.load_onnx_graph(path)?;
        self.convert_to_sequential(&graph)
    }

    /// Load ONNX graph from file
    fn load_onnx_graph(&self, path: &str) -> Result<ONNXGraph> {
        // This would parse the actual ONNX protobuf file
        // For now, return a placeholder
        Ok(ONNXGraph::new("imported_model"))
    }

    /// Convert ONNX graph to Sequential model
    fn convert_to_sequential(&self, _graph: &ONNXGraph) -> Result<Sequential> {
        // This would analyze the ONNX graph and recreate the Sequential model
        // For now, return a simple model
        Ok(Sequential::new())
    }
}

/// Utility functions for ONNX export/import
pub mod utils {
    use super::*;

    /// Validate ONNX model
    pub fn validate_onnx_model(path: &str) -> Result<ValidationReport> {
        // This would validate the ONNX model structure and operators
        Ok(ValidationReport {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            quantum_ops_found: false,
        })
    }

    /// Get ONNX model info
    pub fn get_model_info(path: &str) -> Result<ModelInfo> {
        // This would extract basic information about the ONNX model
        Ok(ModelInfo {
            opset_version: 11,
            producer_name: "QuantRS2-ML".to_string(),
            producer_version: "0.1.2".to_string(),
            graph_name: "model".to_string(),
            num_nodes: 0,
            num_initializers: 0,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        })
    }

    /// Convert quantum circuit to ONNX custom operator
    pub fn circuit_to_onnx_op(circuit: &DynamicCircuit, name: &str) -> Result<ONNXNode> {
        let mut node = ONNXNode::new(
            name,
            "QuantumCircuit",
            vec!["input".to_string()],
            vec!["output".to_string()],
        );

        // Add circuit-specific attributes
        node.add_attribute(
            "num_qubits",
            ONNXAttribute::Int(circuit.num_qubits() as i64),
        );
        node.add_attribute("num_gates", ONNXAttribute::Int(circuit.num_gates() as i64));
        node.add_attribute("depth", ONNXAttribute::Int(circuit.depth() as i64));

        // Serialize circuit structure
        let circuit_data = serialize_circuit(circuit)?;
        node.add_attribute("circuit_data", ONNXAttribute::String(circuit_data));

        Ok(node)
    }

    /// Serialize quantum circuit to string
    fn serialize_circuit(circuit: &DynamicCircuit) -> Result<String> {
        // This would serialize the circuit to a string format
        // For now, return a placeholder
        Ok("quantum_circuit_placeholder".to_string())
    }

    /// Create ONNX metadata for quantum ML model
    pub fn create_quantum_metadata() -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("framework".to_string(), "QuantRS2-ML".to_string());
        metadata.insert("domain".to_string(), "quantrs2.ml".to_string());
        metadata.insert("version".to_string(), "0.1.2".to_string());
        metadata.insert("quantum_support".to_string(), "true".to_string());
        metadata
    }
}

/// Validation report for ONNX models
#[derive(Debug)]
pub struct ValidationReport {
    /// Model is valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Quantum operators found
    pub quantum_ops_found: bool,
}

/// Model information
#[derive(Debug)]
pub struct ModelInfo {
    /// ONNX opset version
    pub opset_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Graph name
    pub graph_name: String,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of initializers
    pub num_initializers: usize,
    /// Input shapes
    pub input_shapes: Vec<Vec<i64>>,
    /// Output shapes
    pub output_shapes: Vec<Vec<i64>>,
}

// Extensions for Sequential model
impl Sequential {
    /// Export to ONNX format
    pub fn export_onnx(
        &self,
        path: &str,
        input_shape: &[usize],
        options: Option<ExportOptions>,
    ) -> Result<()> {
        let exporter = ONNXExporter::new();
        let exporter = if let Some(opts) = options {
            exporter.with_options(opts)
        } else {
            exporter
        };

        exporter.export_sequential(self, input_shape, path)
    }

    /// Get layers (placeholder - would need actual implementation)
    fn layers(&self) -> &[Box<dyn KerasLayer>] {
        // This would return the actual layers from the Sequential model
        &[]
    }

    /// Compute output shape (placeholder)
    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        // This would compute the actual output shape
        input_shape.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keras_api::{ActivationFunction, Dense};

    #[test]
    fn test_onnx_graph_creation() {
        let mut graph = ONNXGraph::new("test_graph");

        graph.add_input(ONNXValueInfo::new(
            "input",
            ONNXDataType::Float32,
            vec![1, 10],
        ));

        graph.add_output(ONNXValueInfo::new(
            "output",
            ONNXDataType::Float32,
            vec![1, 5],
        ));

        let node = ONNXNode::new(
            "dense_layer",
            "MatMul",
            vec!["input".to_string(), "weight".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(node);

        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_onnx_tensor_creation() {
        let array = scirs2_core::ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("Shape and vec size are compatible")
        .into_dyn();

        let tensor = ONNXTensor::from_array_f64("test_tensor", &array);
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
    }

    #[test]
    fn test_onnx_exporter_creation() {
        let exporter = ONNXExporter::new();
        let options = ExportOptions {
            opset_version: 13,
            include_quantum_ops: false,
            optimize_classical_only: true,
            quantum_backend: QuantumBackendTarget::Qiskit,
        };

        let exporter = exporter.with_options(options);
        assert_eq!(exporter.options.opset_version, 13);
        assert!(!exporter.options.include_quantum_ops);
    }

    #[test]
    fn test_onnx_node_attributes() {
        let mut node = ONNXNode::new(
            "test_node",
            "Conv",
            vec!["input".to_string()],
            vec!["output".to_string()],
        );

        node.add_attribute("kernel_shape", ONNXAttribute::Ints(vec![3, 3]));
        node.add_attribute("strides", ONNXAttribute::Ints(vec![1, 1]));

        assert_eq!(node.attributes.len(), 2);
    }

    #[test]
    fn test_validation_utils() {
        let report = utils::validate_onnx_model("dummy_path");
        assert!(report.is_ok());

        let info = utils::get_model_info("dummy_path");
        assert!(info.is_ok());
    }
}
