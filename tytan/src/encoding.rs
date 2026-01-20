//! Variable encoding schemes for optimization problems.
//!
//! This module provides various encoding schemes to represent different types
//! of variables and constraints as binary optimization problems.

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[cfg(feature = "dwave")]
use crate::symbol::Expression;

/// Variable encoding scheme
#[derive(Debug, Clone)]
pub enum EncodingScheme {
    /// One-hot encoding: exactly one bit is 1
    OneHot { num_values: usize },
    /// Binary encoding: log2(n) bits for n values
    Binary { num_values: usize },
    /// Gray code encoding
    GrayCode { num_values: usize },
    /// Domain wall encoding: string of 1s followed by 0s
    DomainWall { num_values: usize },
    /// Unary/thermometer encoding: first k bits are 1
    Unary { num_values: usize },
    /// Order encoding: bit i is 1 if value >= i
    OrderEncoding { min_value: i32, max_value: i32 },
    /// Direct binary (for binary variables)
    Direct,
}

/// Encoded variable representation
#[derive(Debug, Clone)]
pub struct EncodedVariable {
    /// Original variable name
    pub name: String,
    /// Encoding scheme used
    pub scheme: EncodingScheme,
    /// Binary variable names
    pub binary_vars: Vec<String>,
    /// Encoding constraints (as penalty terms)
    #[cfg(feature = "dwave")]
    pub constraints: Option<Expression>,
}

impl EncodedVariable {
    /// Create new encoded variable
    pub fn new(name: &str, scheme: EncodingScheme) -> Self {
        let binary_vars = Self::generate_binary_vars(name, &scheme);
        Self {
            name: name.to_string(),
            scheme,
            binary_vars,
            #[cfg(feature = "dwave")]
            constraints: None,
        }
    }

    /// Generate binary variable names based on encoding
    fn generate_binary_vars(name: &str, scheme: &EncodingScheme) -> Vec<String> {
        match scheme {
            EncodingScheme::OneHot { num_values } => {
                (0..*num_values).map(|i| format!("{name}_{i}")).collect()
            }
            EncodingScheme::Binary { num_values } => {
                let num_bits = (*num_values as f64).log2().ceil() as usize;
                (0..num_bits).map(|i| format!("{name}_bit{i}")).collect()
            }
            EncodingScheme::GrayCode { num_values } => {
                let num_bits = (*num_values as f64).log2().ceil() as usize;
                (0..num_bits).map(|i| format!("{name}_gray{i}")).collect()
            }
            EncodingScheme::DomainWall { num_values } => (0..*num_values - 1)
                .map(|i| format!("{name}_dw{i}"))
                .collect(),
            EncodingScheme::Unary { num_values } => (0..*num_values - 1)
                .map(|i| format!("{name}_u{i}"))
                .collect(),
            EncodingScheme::OrderEncoding {
                min_value,
                max_value,
            } => {
                let range = max_value - min_value;
                (0..range).map(|i| format!("{name}_ord{i}")).collect()
            }
            EncodingScheme::Direct => vec![name.to_string()],
        }
    }

    /// Decode binary values to original value
    pub fn decode(&self, binary_values: &HashMap<String, bool>) -> Option<i32> {
        match &self.scheme {
            EncodingScheme::OneHot { .. } => {
                for (i, var) in self.binary_vars.iter().enumerate() {
                    if binary_values.get(var).copied().unwrap_or(false) {
                        return Some(i as i32);
                    }
                }
                None // Invalid: no bit set
            }
            EncodingScheme::Binary { .. } => {
                let mut value = 0;
                for (i, var) in self.binary_vars.iter().enumerate() {
                    if binary_values.get(var).copied().unwrap_or(false) {
                        value |= 1 << i;
                    }
                }
                Some(value)
            }
            EncodingScheme::GrayCode { .. } => {
                let mut gray = 0;
                for (i, var) in self.binary_vars.iter().enumerate() {
                    if binary_values.get(var).copied().unwrap_or(false) {
                        gray |= 1 << i;
                    }
                }
                // Convert Gray code to binary
                let mut binary = gray;
                binary ^= binary >> 16;
                binary ^= binary >> 8;
                binary ^= binary >> 4;
                binary ^= binary >> 2;
                binary ^= binary >> 1;
                Some(binary)
            }
            EncodingScheme::DomainWall { num_values } => {
                let mut value = *num_values as i32 - 1;
                for (i, var) in self.binary_vars.iter().enumerate() {
                    if !binary_values.get(var).copied().unwrap_or(false) {
                        value = i as i32;
                        break;
                    }
                }
                Some(value)
            }
            EncodingScheme::Unary { .. } => {
                let mut value = 0;
                for var in &self.binary_vars {
                    if binary_values.get(var).copied().unwrap_or(false) {
                        value += 1;
                    } else {
                        break;
                    }
                }
                Some(value)
            }
            EncodingScheme::OrderEncoding { min_value, .. } => {
                let mut value = *min_value;
                for var in &self.binary_vars {
                    if binary_values.get(var).copied().unwrap_or(false) {
                        value += 1;
                    }
                }
                Some(value - 1)
            }
            EncodingScheme::Direct => binary_values.get(&self.name).map(|&b| i32::from(b)),
        }
    }

    /// Encode value to binary representation
    pub fn encode(&self, value: i32) -> HashMap<String, bool> {
        let mut binary_values = HashMap::new();

        match &self.scheme {
            EncodingScheme::OneHot { num_values: _ } => {
                for (i, var) in self.binary_vars.iter().enumerate() {
                    binary_values.insert(var.clone(), i == value as usize);
                }
            }
            EncodingScheme::Binary { .. } => {
                for (i, var) in self.binary_vars.iter().enumerate() {
                    binary_values.insert(var.clone(), (value & (1 << i)) != 0);
                }
            }
            EncodingScheme::GrayCode { .. } => {
                // Convert to Gray code
                let gray = value ^ (value >> 1);
                for (i, var) in self.binary_vars.iter().enumerate() {
                    binary_values.insert(var.clone(), (gray & (1 << i)) != 0);
                }
            }
            EncodingScheme::DomainWall { num_values: _ } => {
                for (i, var) in self.binary_vars.iter().enumerate() {
                    binary_values.insert(var.clone(), i < value as usize);
                }
            }
            EncodingScheme::Unary { .. } => {
                for (i, var) in self.binary_vars.iter().enumerate() {
                    binary_values.insert(var.clone(), i < value as usize);
                }
            }
            EncodingScheme::OrderEncoding { min_value, .. } => {
                let adjusted = value - min_value + 1;
                for (i, var) in self.binary_vars.iter().enumerate() {
                    binary_values.insert(var.clone(), i < adjusted as usize);
                }
            }
            EncodingScheme::Direct => {
                binary_values.insert(self.name.clone(), value != 0);
            }
        }

        binary_values
    }

    /// Get penalty matrix for encoding constraints
    pub fn get_penalty_matrix(&self, var_indices: &HashMap<String, usize>) -> Array2<f64> {
        let n = var_indices.len();
        let mut penalty = Array2::zeros((n, n));

        match &self.scheme {
            EncodingScheme::OneHot { .. } => {
                // Exactly one bit must be 1
                // Penalty: (sum(xi) - 1)^2 = sum(xi)^2 - 2*sum(xi) + 1

                // Get indices of our binary variables
                let indices: Vec<usize> = self
                    .binary_vars
                    .iter()
                    .filter_map(|var| var_indices.get(var).copied())
                    .collect();

                // Quadratic terms: xi * xj for i != j
                for &i in &indices {
                    for &j in &indices {
                        if i != j {
                            penalty[[i, j]] += 1.0;
                        }
                    }
                }

                // Linear terms: -2 * xi
                for &i in &indices {
                    penalty[[i, i]] -= 2.0;
                }
            }
            EncodingScheme::DomainWall { .. } => {
                // Domain wall constraint: xi >= xi+1
                // Penalty for violation: xi+1 * (1 - xi)

                let indices: Vec<usize> = self
                    .binary_vars
                    .iter()
                    .filter_map(|var| var_indices.get(var).copied())
                    .collect();

                for i in 0..indices.len() - 1 {
                    let idx1 = indices[i];
                    let idx2 = indices[i + 1];

                    // Penalty term: x_{i+1} - x_i * x_{i+1}
                    penalty[[idx2, idx2]] += 1.0;
                    penalty[[idx1, idx2]] -= 1.0;
                    penalty[[idx2, idx1]] -= 1.0;
                }
            }
            EncodingScheme::Unary { .. } => {
                // Unary constraint: xi >= xi+1
                // Same as domain wall
                let indices: Vec<usize> = self
                    .binary_vars
                    .iter()
                    .filter_map(|var| var_indices.get(var).copied())
                    .collect();

                for i in 0..indices.len() - 1 {
                    let idx1 = indices[i];
                    let idx2 = indices[i + 1];

                    penalty[[idx2, idx2]] += 1.0;
                    penalty[[idx1, idx2]] -= 1.0;
                    penalty[[idx2, idx1]] -= 1.0;
                }
            }
            _ => {
                // No encoding constraints for binary, gray code, or direct
            }
        }

        penalty
    }
}

/// Encoding optimizer that selects best encoding for variables
pub struct EncodingOptimizer {
    /// Variable domains
    domains: HashMap<String, (i32, i32)>,
    /// Constraint information
    constraint_graph: HashMap<String, Vec<String>>,
}

impl Default for EncodingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingOptimizer {
    /// Create new encoding optimizer
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
            constraint_graph: HashMap::new(),
        }
    }

    /// Add variable with domain
    pub fn add_variable(&mut self, name: &str, min_value: i32, max_value: i32) {
        self.domains
            .insert(name.to_string(), (min_value, max_value));
    }

    /// Add constraint between variables
    pub fn add_constraint(&mut self, var1: &str, var2: &str) {
        self.constraint_graph
            .entry(var1.to_string())
            .or_default()
            .push(var2.to_string());
        self.constraint_graph
            .entry(var2.to_string())
            .or_default()
            .push(var1.to_string());
    }

    /// Select optimal encoding for each variable
    pub fn optimize_encodings(&self) -> HashMap<String, EncodingScheme> {
        let mut encodings = HashMap::new();

        for (var, &(min_val, max_val)) in &self.domains {
            let domain_size = (max_val - min_val + 1) as usize;
            let neighbors = self.constraint_graph.get(var).map_or(0, |v| v.len());

            // Heuristics for encoding selection
            let encoding = if domain_size == 2 {
                // Binary variable
                EncodingScheme::Direct
            } else if domain_size <= 4 && neighbors > 3 {
                // Small domain with many constraints: one-hot
                EncodingScheme::OneHot {
                    num_values: domain_size,
                }
            } else if domain_size <= 8 {
                // Medium domain: binary or gray code
                if self.has_ordering_constraints(var) {
                    EncodingScheme::GrayCode {
                        num_values: domain_size,
                    }
                } else {
                    EncodingScheme::Binary {
                        num_values: domain_size,
                    }
                }
            } else if self.has_ordering_constraints(var) {
                // Large ordered domain: order encoding or domain wall
                if domain_size <= 32 {
                    EncodingScheme::OrderEncoding {
                        min_value: min_val,
                        max_value: max_val,
                    }
                } else {
                    EncodingScheme::DomainWall {
                        num_values: domain_size,
                    }
                }
            } else {
                // Large unordered domain: binary
                EncodingScheme::Binary {
                    num_values: domain_size,
                }
            };

            encodings.insert(var.clone(), encoding);
        }

        encodings
    }

    /// Check if variable has ordering constraints
    const fn has_ordering_constraints(&self, _var: &str) -> bool {
        // Simplified: would check actual constraint types
        false
    }
}

/// Auxiliary variable generator for complex encodings
pub struct AuxiliaryVariableGenerator {
    /// Counter for generating unique names
    counter: usize,
    /// Prefix for auxiliary variables
    prefix: String,
}

impl AuxiliaryVariableGenerator {
    /// Create new generator
    pub fn new(prefix: &str) -> Self {
        Self {
            counter: 0,
            prefix: prefix.to_string(),
        }
    }

    /// Generate new auxiliary variable name
    pub fn next(&mut self) -> String {
        let name = format!("{}_{}", self.prefix, self.counter);
        self.counter += 1;
        name
    }

    /// Generate auxiliary variables for product encoding
    pub fn product_encoding(
        &mut self,
        _var1: &str,
        _var2: &str,
        enc1: &EncodedVariable,
        enc2: &EncodedVariable,
    ) -> Vec<(String, Vec<String>)> {
        let mut auxiliaries = Vec::new();

        // Generate auxiliary for each pair of binary variables
        for bin1 in &enc1.binary_vars {
            for bin2 in &enc2.binary_vars {
                let aux = self.next();
                auxiliaries.push((aux.clone(), vec![bin1.clone(), bin2.clone()]));
            }
        }

        auxiliaries
    }
}

/// Convert integer program to QUBO using encodings
pub struct EncodingConverter {
    /// Variable encodings
    encodings: HashMap<String, EncodedVariable>,
    /// Auxiliary variable generator
    aux_gen: AuxiliaryVariableGenerator,
}

impl Default for EncodingConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingConverter {
    /// Create new converter
    pub fn new() -> Self {
        Self {
            encodings: HashMap::new(),
            aux_gen: AuxiliaryVariableGenerator::new("aux"),
        }
    }

    /// Add encoded variable
    pub fn add_variable(&mut self, encoded: EncodedVariable) {
        self.encodings.insert(encoded.name.clone(), encoded);
    }

    /// Get all binary variables
    pub fn get_binary_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for encoded in self.encodings.values() {
            vars.extend(encoded.binary_vars.clone());
        }
        vars
    }

    /// Build QUBO matrix with encoding penalties
    pub fn build_qubo_matrix(&self, _base_matrix: Array2<f64>) -> Array2<f64> {
        let binary_vars = self.get_binary_variables();
        let var_indices: HashMap<String, usize> = binary_vars
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        let n = binary_vars.len();
        let mut qubo = Array2::zeros((n, n));

        // Add encoding penalties
        for encoded in self.encodings.values() {
            let penalty = encoded.get_penalty_matrix(&var_indices);
            qubo = qubo + penalty;
        }

        // Add base problem matrix (would need proper mapping)
        // This is simplified - real implementation would map original to binary vars

        qubo
    }
}

/// Compare different encodings
pub fn compare_encodings(
    domain_size: usize,
    constraint_density: f64,
) -> HashMap<String, EncodingMetrics> {
    let mut results = HashMap::new();

    // One-hot encoding
    let onehot_bits = domain_size;
    let onehot_constraints = domain_size * (domain_size - 1) / 2; // Quadratic
    results.insert(
        "one-hot".to_string(),
        EncodingMetrics {
            num_bits: onehot_bits,
            num_constraints: onehot_constraints,
            avg_connectivity: domain_size as f64 - 1.0,
            space_efficiency: 1.0 / domain_size as f64,
        },
    );

    // Binary encoding
    let binary_bits = (domain_size as f64).log2().ceil() as usize;
    results.insert(
        "binary".to_string(),
        EncodingMetrics {
            num_bits: binary_bits,
            num_constraints: 0,
            avg_connectivity: constraint_density * binary_bits as f64,
            space_efficiency: (domain_size as f64).log2() / domain_size as f64,
        },
    );

    // Domain wall encoding
    let dw_bits = domain_size - 1;
    let dw_constraints = domain_size - 1;
    results.insert(
        "domain-wall".to_string(),
        EncodingMetrics {
            num_bits: dw_bits,
            num_constraints: dw_constraints,
            avg_connectivity: 2.0,
            space_efficiency: 1.0 / domain_size as f64,
        },
    );

    results
}

#[derive(Debug, Clone)]
pub struct EncodingMetrics {
    pub num_bits: usize,
    pub num_constraints: usize,
    pub avg_connectivity: f64,
    pub space_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encoding() {
        let encoded = EncodedVariable::new("x", EncodingScheme::OneHot { num_values: 4 });
        assert_eq!(encoded.binary_vars.len(), 4);

        // Encode value 2
        let mut binary = encoded.encode(2);
        assert!(!binary[&"x_0".to_string()]);
        assert!(!binary[&"x_1".to_string()]);
        assert!(binary[&"x_2".to_string()]);
        assert!(!binary[&"x_3".to_string()]);

        // Decode back
        let value = encoded
            .decode(&binary)
            .expect("Failed to decode one-hot value");
        assert_eq!(value, 2);
    }

    #[test]
    fn test_binary_encoding() {
        let encoded = EncodedVariable::new("y", EncodingScheme::Binary { num_values: 8 });
        assert_eq!(encoded.binary_vars.len(), 3); // log2(8) = 3

        // Encode value 5 (binary: 101)
        let mut binary = encoded.encode(5);
        assert!(binary[&"y_bit0".to_string()]);
        assert!(!binary[&"y_bit1".to_string()]);
        assert!(binary[&"y_bit2".to_string()]);

        let value = encoded
            .decode(&binary)
            .expect("Failed to decode binary value");
        assert_eq!(value, 5);
    }

    #[test]
    fn test_domain_wall_encoding() {
        let encoded = EncodedVariable::new("z", EncodingScheme::DomainWall { num_values: 5 });
        assert_eq!(encoded.binary_vars.len(), 4);

        // Encode value 2 (domain wall: 1100)
        let mut binary = encoded.encode(2);
        assert!(binary[&"z_dw0".to_string()]);
        assert!(binary[&"z_dw1".to_string()]);
        assert!(!binary[&"z_dw2".to_string()]);
        assert!(!binary[&"z_dw3".to_string()]);

        let value = encoded
            .decode(&binary)
            .expect("Failed to decode domain wall value");
        assert_eq!(value, 2);
    }

    #[test]
    fn test_encoding_optimizer() {
        let mut optimizer = EncodingOptimizer::new();
        optimizer.add_variable("small", 0, 3);
        optimizer.add_variable("large", 0, 100);
        optimizer.add_variable("binary", 0, 1);

        let encodings = optimizer.optimize_encodings();

        // Binary variable should use direct encoding
        match &encodings["binary"] {
            EncodingScheme::Direct => {}
            _ => panic!("Expected direct encoding for binary variable"),
        }
    }
}
