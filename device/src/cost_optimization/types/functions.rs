//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use super::types::{
    ActiveAlert, AlgorithmInfo, CrossValidationResult, NotificationHandlerInfo,
    OptimizationProblem, OptimizationResult, PredictionResult, TrainingData, TrainingResult,
};
use crate::DeviceResult;

/// Machine learning cost model trait
pub trait MLCostModel {
    fn predict(&self, features: &Array1<f64>) -> DeviceResult<f64>;
    fn train(&mut self, features: &Array2<f64>, targets: &Array1<f64>) -> DeviceResult<()>;
    fn get_feature_importance(&self) -> DeviceResult<Array1<f64>>;
}
/// Predictive model trait
pub trait PredictiveModel {
    fn predict(&self, features: &HashMap<String, f64>) -> DeviceResult<PredictionResult>;
    fn train(&mut self, training_data: &TrainingData) -> DeviceResult<TrainingResult>;
    fn get_feature_importance(&self) -> DeviceResult<HashMap<String, f64>>;
    fn cross_validate(
        &self,
        data: &TrainingData,
        folds: usize,
    ) -> DeviceResult<CrossValidationResult>;
}
/// Optimization algorithm trait
pub trait OptimizationAlgorithm {
    fn optimize(&self, problem: &OptimizationProblem) -> DeviceResult<OptimizationResult>;
    fn get_algorithm_info(&self) -> AlgorithmInfo;
    fn set_parameters(&mut self, parameters: HashMap<String, f64>) -> DeviceResult<()>;
}
/// Notification handler trait
pub trait NotificationHandler {
    fn send_notification(&self, alert: &ActiveAlert, message: &str) -> DeviceResult<()>;
    fn get_handler_info(&self) -> NotificationHandlerInfo;
}
