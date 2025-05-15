use crate::error::{MLError, Result};
use ndarray::{Array1, Array2};

/// Metrics for evaluating classification performance
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Accuracy (ratio of correctly classified samples)
    pub accuracy: f64,

    /// Precision (ratio of correctly predicted positive observations to the total predicted positives)
    pub precision: f64,

    /// Recall (ratio of correctly predicted positive observations to all actual positives)
    pub recall: f64,

    /// F1 score (harmonic mean of precision and recall)
    pub f1_score: f64,

    /// Area under the ROC curve
    pub auc: f64,

    /// Confusion matrix
    pub confusion_matrix: Array2<f64>,

    /// Per-class accuracy values
    pub class_accuracies: Vec<f64>,

    /// Class labels
    pub class_labels: Vec<String>,

    /// Average loss value
    pub average_loss: f64,
}

/// Trait for classification models
pub trait Classifier {
    /// Trains the classifier on a dataset
    fn train(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<()>;

    /// Predicts the class for a sample
    fn predict(&self, x: &Array1<f64>) -> Result<usize>;

    /// Predicts classes for a batch of samples
    fn predict_batch(&self, x: &Array2<f64>) -> Result<Array1<usize>> {
        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            predictions[i] = self.predict(&x.row(i).to_owned())?;
        }

        Ok(predictions)
    }

    /// Computes prediction probabilities for a sample
    fn predict_proba(&self, x: &Array1<f64>) -> Result<Array1<f64>>;

    /// Evaluates the classifier on a dataset
    fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array1<f64>)
        -> Result<ClassificationMetrics>;
}
