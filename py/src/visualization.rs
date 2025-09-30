use scirs2_core::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use std::collections::{HashMap, VecDeque};

/// Internal representation of a circuit for visualization
#[derive(Default, Clone)]
pub struct CircuitVisualization {
    /// The number of qubits in the circuit
    n_qubits: usize,

    /// Sequence of gates in the circuit, including position information
    operations: Vec<GateOperation>,

    /// Maximum depth of the circuit
    depth: usize,
}

/// Represents a gate operation in the circuit
#[derive(Clone)]
struct GateOperation {
    /// The type of gate
    gate_type: GateType,

    /// The qubits this gate acts on
    qubits: Vec<usize>,

    /// Column in the circuit visualization (time step)
    column: usize,

    /// Display parameters for the gate (angles, etc.)
    params: Option<String>,
}

/// Enum for gate types
#[derive(Clone, PartialEq)]
enum GateType {
    H,
    X,
    Y,
    Z,
    S,
    SDG,
    T,
    TDG,
    SX,
    SXDG,
    RX,
    RY,
    RZ,
    CNOT,
    CY,
    CZ,
    CH,
    CS,
    SWAP,
    CRX,
    CRY,
    CRZ,
    Toffoli,
    Fredkin,
    Custom(String),
}

impl GateType {
    /// Get the symbol for this gate type
    fn symbol(&self) -> &str {
        match self {
            GateType::H => "H",
            GateType::X => "X",
            GateType::Y => "Y",
            GateType::Z => "Z",
            GateType::S => "S",
            GateType::SDG => "S†",
            GateType::T => "T",
            GateType::TDG => "T†",
            GateType::SX => "√X",
            GateType::SXDG => "√X†",
            GateType::RX => "Rx",
            GateType::RY => "Ry",
            GateType::RZ => "Rz",
            GateType::CNOT => "●─┼─X",
            GateType::CY => "●─┼─Y",
            GateType::CZ => "●─┼─Z",
            GateType::CH => "●─┼─H",
            GateType::CS => "●─┼─S",
            GateType::SWAP => "⨯―⨯",
            GateType::CRX => "●─┼─Rx",
            GateType::CRY => "●─┼─Ry",
            GateType::CRZ => "●─┼─Rz",
            GateType::Toffoli => "●─●─X",
            GateType::Fredkin => "●─⨯─⨯",
            GateType::Custom(s) => s,
        }
    }
}

impl CircuitVisualization {
    /// Create a new circuit visualization
    pub fn new(n_qubits: usize) -> Self {
        CircuitVisualization {
            n_qubits,
            operations: Vec::new(),
            depth: 0,
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate_type: GateType, qubits: Vec<usize>, params: Option<String>) {
        let mut column = 0;

        // Find the first column where all qubits are free
        'col_search: loop {
            let mut available = true;
            for qubit in &qubits {
                if self.is_qubit_occupied_at(*qubit, column) {
                    available = false;
                    column += 1;
                    continue 'col_search;
                }
            }

            if available {
                break;
            }
        }

        // Add the gate to the circuit
        self.operations.push(GateOperation {
            gate_type,
            qubits,
            column,
            params,
        });

        // Update circuit depth
        self.depth = std::cmp::max(self.depth, column + 1);
    }

    /// Check if a qubit is occupied at a given column
    fn is_qubit_occupied_at(&self, qubit: usize, column: usize) -> bool {
        for op in &self.operations {
            if op.column == column && op.qubits.contains(&qubit) {
                return true;
            }
        }
        false
    }

    /// Generate a text representation of the circuit
    pub fn to_text(&self) -> String {
        if self.n_qubits == 0 || self.operations.is_empty() {
            return "Empty circuit".to_string();
        }

        // Create a 2D grid for the circuit
        let mut grid = vec![vec![' '; self.depth * 5]; self.n_qubits * 2];

        // Draw qubit lines
        for q in 0..self.n_qubits {
            let row = q * 2;
            for col in 0..self.depth * 5 {
                grid[row][col] = '─';
            }
        }

        // Add qubit labels
        let mut result = String::new();
        for q in 0..self.n_qubits {
            result.push_str(&format!("q{}:", q));
            // Padding to align all circuits
            for _ in 0..3 {
                result.push(' ');
            }

            // Draw the qubit line
            for col in 0..self.depth {
                let base_col = col * 5;
                let line = self.render_column_for_qubit(q, col, &grid);
                result.push_str(&line);
            }
            result.push('\n');
        }

        result
    }

    /// Generate a text representation of a column for a specific qubit
    fn render_column_for_qubit(&self, qubit: usize, column: usize, grid: &[Vec<char>]) -> String {
        // Find operations in this column that affect this qubit
        let mut has_gate = false;
        let mut gate_symbol = String::new();
        let mut control_type = None;

        for op in &self.operations {
            if op.column != column {
                continue;
            }

            if op.qubits.contains(&qubit) {
                has_gate = true;

                if op.qubits.len() == 1 {
                    // Single-qubit gate
                    gate_symbol = op.gate_type.symbol().to_string();
                    if let Some(params) = &op.params {
                        gate_symbol.push_str(&format!("({})", params));
                    }
                } else {
                    // Multi-qubit gate
                    match op.gate_type {
                        GateType::CNOT
                        | GateType::CY
                        | GateType::CZ
                        | GateType::CH
                        | GateType::CS
                        | GateType::CRX
                        | GateType::CRY
                        | GateType::CRZ => {
                            // Control or target?
                            if op.qubits[0] == qubit {
                                gate_symbol = "●".to_string();
                                control_type = Some("control");
                            } else {
                                match op.gate_type {
                                    GateType::CNOT => gate_symbol = "X".to_string(),
                                    GateType::CY => gate_symbol = "Y".to_string(),
                                    GateType::CZ => gate_symbol = "Z".to_string(),
                                    GateType::CH => gate_symbol = "H".to_string(),
                                    GateType::CS => gate_symbol = "S".to_string(),
                                    GateType::CRX => {
                                        gate_symbol = "Rx".to_string();
                                        if let Some(params) = &op.params {
                                            gate_symbol.push_str(&format!("({})", params));
                                        }
                                    }
                                    GateType::CRY => {
                                        gate_symbol = "Ry".to_string();
                                        if let Some(params) = &op.params {
                                            gate_symbol.push_str(&format!("({})", params));
                                        }
                                    }
                                    GateType::CRZ => {
                                        gate_symbol = "Rz".to_string();
                                        if let Some(params) = &op.params {
                                            gate_symbol.push_str(&format!("({})", params));
                                        }
                                    }
                                    _ => unreachable!(),
                                }
                                control_type = Some("target");
                            }
                        }
                        GateType::SWAP => {
                            gate_symbol = "⨯".to_string();
                            control_type = Some("swap");
                        }
                        GateType::Toffoli => {
                            if op.qubits[0] == qubit || op.qubits[1] == qubit {
                                gate_symbol = "●".to_string();
                                control_type = Some("control");
                            } else {
                                gate_symbol = "X".to_string();
                                control_type = Some("target");
                            }
                        }
                        GateType::Fredkin => {
                            if op.qubits[0] == qubit {
                                gate_symbol = "●".to_string();
                                control_type = Some("control");
                            } else {
                                gate_symbol = "⨯".to_string();
                                control_type = Some("swap");
                            }
                        }
                        _ => {
                            gate_symbol = op.gate_type.symbol().to_string();
                        }
                    }
                }
                break;
            }
        }

        if has_gate {
            if gate_symbol.len() > 5 {
                // Truncate long gate symbols
                format!("{:<5}", &gate_symbol[0..5])
            } else {
                format!("{:<5}", gate_symbol)
            }
        } else {
            "─────".to_string()
        }
    }

    /// Convert to a Python-friendly dictionary format for visualization
    pub fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        let operations = PyList::empty(py);
        for op in &self.operations {
            let op_dict = PyDict::new(py);
            op_dict.set_item("type", op.gate_type.symbol())?;

            let qubits = PyList::empty(py);
            for q in &op.qubits {
                qubits.append(*q)?;
            }
            op_dict.set_item("qubits", qubits)?;
            op_dict.set_item("column", op.column)?;

            if let Some(params) = &op.params {
                op_dict.set_item("params", params)?;
            }

            operations.append(op_dict)?;
        }

        dict.set_item("n_qubits", self.n_qubits)?;
        dict.set_item("operations", operations)?;
        dict.set_item("depth", self.depth)?;

        Ok(dict.into())
    }

    /// Generate an HTML representation of the circuit for display in Jupyter notebooks
    pub fn to_html(&self) -> String {
        if self.n_qubits == 0 || self.operations.is_empty() {
            return "<div class=\"qc-empty\">Empty circuit</div>".to_string();
        }

        let mut html = String::from(
            r#"
        <style>
            .qc-container {
                font-family: monospace;
                margin: 10px 0;
                display: grid;
            }
            .qc-qubit-labels {
                grid-column: 1;
                grid-row: 1 / span var(--n-qubits);
                display: grid;
                grid-template-rows: repeat(var(--n-qubits), 40px);
                align-items: center;
                margin-right: 10px;
            }
            .qc-qubit-label {
                text-align: right;
                padding-right: 5px;
            }
            .qc-circuit {
                grid-column: 2;
                grid-row: 1;
                display: grid;
                grid-template-rows: repeat(var(--n-qubits), 40px);
                grid-template-columns: repeat(var(--depth), 40px);
                grid-auto-flow: column;
                align-items: center;
                column-gap: 5px;
            }
            .qc-wire {
                height: 2px;
                background-color: #000;
                width: 100%;
                position: relative;
                z-index: 0;
            }
            .qc-gate {
                position: relative;
                z-index: 1;
                width: 30px;
                height: 30px;
                display: flex;
                justify-content: center;
                align-items: center;
                border: 2px solid #000;
                border-radius: 4px;
                background-color: white;
                font-weight: bold;
            }
            .qc-control {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #000;
            }
            .qc-target {
                width: 30px;
                height: 30px;
                display: flex;
                justify-content: center;
                align-items: center;
                border: 2px solid #000;
                border-radius: 4px;
                background-color: white;
                font-weight: bold;
            }
            .qc-swap {
                width: 20px;
                height: 20px;
                position: relative;
            }
            .qc-swap::before, .qc-swap::after {
                content: '';
                position: absolute;
                width: 20px;
                height: 2px;
                background-color: #000;
                top: 50%;
            }
            .qc-swap::before {
                transform: rotate(45deg);
            }
            .qc-swap::after {
                transform: rotate(-45deg);
            }
            .qc-connection {
                position: absolute;
                width: 2px;
                background-color: #000;
                z-index: 0;
            }
            .qc-empty {
                font-family: monospace;
                color: #666;
                margin: 10px 0;
            }
        </style>
        "#,
        );

        // Create the container with CSS variables for qubit count and depth
        html.push_str(&format!(
            "<div class=\"qc-container\" style=\"--n-qubits:{}; --depth:{}\">\n",
            self.n_qubits, self.depth
        ));

        // Add qubit labels
        html.push_str("  <div class=\"qc-qubit-labels\">\n");
        for q in 0..self.n_qubits {
            html.push_str(&format!(
                "    <div class=\"qc-qubit-label\">q{}:</div>\n",
                q
            ));
        }
        html.push_str("  </div>\n");

        // Create the circuit grid
        html.push_str("  <div class=\"qc-circuit\">\n");

        // Add qubit lines
        for q in 0..self.n_qubits {
            for col in 0..self.depth {
                html.push_str(&format!(
                    "    <div style=\"grid-row:{}; grid-column:{}\">\n",
                    q + 1,
                    col + 1
                ));

                // Check if there's a gate at this position
                let mut gate_at_pos = false;
                for op in &self.operations {
                    if op.column == col && op.qubits.contains(&q) {
                        gate_at_pos = true;

                        // Render appropriate gate
                        if op.qubits.len() == 1 {
                            // Single-qubit gate
                            html.push_str(&format!(
                                "      <div class=\"qc-gate\">{}</div>\n",
                                op.gate_type.symbol()
                            ));
                            if let Some(params) = &op.params {
                                // Add tooltip with parameters
                                html.push_str(&format!(
                                    "      <div title=\"{}\">{}</div>\n",
                                    params,
                                    op.gate_type.symbol()
                                ));
                            }
                        } else if op.qubits.len() >= 2 {
                            // Multi-qubit gate
                            match op.gate_type {
                                GateType::CNOT
                                | GateType::CY
                                | GateType::CZ
                                | GateType::CH
                                | GateType::CS
                                | GateType::CRX
                                | GateType::CRY
                                | GateType::CRZ => {
                                    if op.qubits[0] == q {
                                        // Control qubit
                                        html.push_str("      <div class=\"qc-control\"></div>\n");
                                    } else {
                                        // Target qubit
                                        html.push_str(&format!(
                                            "      <div class=\"qc-target\">{}</div>\n",
                                            match op.gate_type {
                                                GateType::CNOT => "X",
                                                GateType::CY => "Y",
                                                GateType::CZ => "Z",
                                                GateType::CH => "H",
                                                GateType::CS => "S",
                                                GateType::CRX => "Rx",
                                                GateType::CRY => "Ry",
                                                GateType::CRZ => "Rz",
                                                _ => unreachable!(),
                                            }
                                        ));
                                    }
                                }
                                GateType::SWAP => {
                                    html.push_str("      <div class=\"qc-swap\"></div>\n");
                                }
                                GateType::Toffoli => {
                                    if op.qubits[0] == q || op.qubits[1] == q {
                                        html.push_str("      <div class=\"qc-control\"></div>\n");
                                    } else {
                                        html.push_str("      <div class=\"qc-target\">X</div>\n");
                                    }
                                }
                                GateType::Fredkin => {
                                    if op.qubits[0] == q {
                                        html.push_str("      <div class=\"qc-control\"></div>\n");
                                    } else {
                                        html.push_str("      <div class=\"qc-swap\"></div>\n");
                                    }
                                }
                                _ => {
                                    html.push_str(&format!(
                                        "      <div class=\"qc-gate\">{}</div>\n",
                                        op.gate_type.symbol()
                                    ));
                                }
                            }

                            // Add connections between control and target qubits
                            let min_q = *op.qubits.iter().min().unwrap();
                            let max_q = *op.qubits.iter().max().unwrap();

                            if op.qubits.len() == 2 && min_q < max_q && (q == min_q || q == max_q) {
                                let top = if q == min_q { "50%" } else { "0" };
                                let height = if q == min_q {
                                    format!("{}px", (max_q - min_q) * 40 / 2)
                                } else {
                                    format!("{}px", (max_q - min_q) * 40 / 2)
                                };

                                html.push_str(&format!(
                                    "      <div class=\"qc-connection\" style=\"top:{}; height:{};\"></div>\n",
                                    top, height
                                ));
                            }
                        }

                        break;
                    }
                }

                if !gate_at_pos {
                    // Draw the wire if no gate at this position
                    html.push_str("      <div class=\"qc-wire\"></div>\n");
                }

                html.push_str("    </div>\n");
            }
        }

        html.push_str("  </div>\n");
        html.push_str("</div>\n");

        html
    }
}

/// Python circuit visualization helper
#[pyclass]
pub struct PyCircuitVisualizer {
    /// The internal circuit visualization
    visualization: CircuitVisualization,
}

#[pymethods]
impl PyCircuitVisualizer {
    /// Create a new circuit visualizer for a circuit with the given number of qubits
    #[new]
    pub fn new(n_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            visualization: CircuitVisualization::new(n_qubits),
        })
    }

    /// Add a gate to the circuit
    pub fn add_gate(
        &mut self,
        gate_type: &str,
        qubits: Vec<usize>,
        params: Option<String>,
    ) -> PyResult<()> {
        let gate_type = match gate_type {
            "h" | "H" => GateType::H,
            "x" | "X" => GateType::X,
            "y" | "Y" => GateType::Y,
            "z" | "Z" => GateType::Z,
            "s" | "S" => GateType::S,
            "sdg" | "S†" => GateType::SDG,
            "t" | "T" => GateType::T,
            "tdg" | "T†" => GateType::TDG,
            "sx" | "√X" => GateType::SX,
            "sxdg" | "√X†" => GateType::SXDG,
            "rx" | "Rx" => GateType::RX,
            "ry" | "Ry" => GateType::RY,
            "rz" | "Rz" => GateType::RZ,
            "cnot" | "CNOT" => GateType::CNOT,
            "cy" | "CY" => GateType::CY,
            "cz" | "CZ" => GateType::CZ,
            "ch" | "CH" => GateType::CH,
            "cs" | "CS" => GateType::CS,
            "swap" | "SWAP" => GateType::SWAP,
            "crx" | "CRX" => GateType::CRX,
            "cry" | "CRY" => GateType::CRY,
            "crz" | "CRZ" => GateType::CRZ,
            "toffoli" | "ccnot" | "CCNOT" => GateType::Toffoli,
            "fredkin" | "cswap" | "CSWAP" => GateType::Fredkin,
            _ => GateType::Custom(gate_type.to_string()),
        };

        self.visualization.add_gate(gate_type, qubits, params);
        Ok(())
    }

    /// Get a text representation of the circuit
    pub fn to_text(&self) -> String {
        self.visualization.to_text()
    }

    /// Get the circuit as a dictionary for customizable visualization
    pub fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        self.visualization.to_dict(py)
    }

    /// Get an HTML representation for display in Jupyter notebooks
    pub fn to_html(&self) -> String {
        self.visualization.to_html()
    }

    /// Display the circuit in a Jupyter notebook (implement _repr_html_)
    pub fn _repr_html_(&self) -> String {
        self.visualization.to_html()
    }
}

/// Visualizer factory for PyCircuit
pub fn create_visualizer_from_operations(
    py: Python,
    n_qubits: usize,
    operations: Vec<(String, Vec<usize>, Option<String>)>,
) -> PyResult<Py<PyCircuitVisualizer>> {
    let mut visualizer = PyCircuitVisualizer::new(n_qubits)?;

    for (gate_type, qubits, params) in operations {
        visualizer.add_gate(&gate_type, qubits, params)?;
    }

    Py::new(py, visualizer)
}
