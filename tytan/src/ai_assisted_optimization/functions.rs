//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};

use super::types::StructurePattern;

pub trait StructureDetector {
    fn detect_structure(&self, qubo: &Array2<f64>) -> Vec<StructurePattern>;
    fn confidence_score(&self, pattern: &StructurePattern) -> f64;
    fn detector_name(&self) -> &str;
}
