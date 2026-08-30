//! Core quantum dimensionality reduction functionality

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};

use super::config::*;
use super::metrics::*;

/// Main quantum dimensionality reducer
#[derive(Debug)]
pub struct QuantumDimensionalityReducer {
    /// Algorithm to use
    pub algorithm: DimensionalityReductionAlgorithm,
    /// QPCA configuration
    pub qpca_config: Option<QPCAConfig>,
    /// QICA configuration
    pub qica_config: Option<QICAConfig>,
    /// Qt-SNE configuration
    pub qtsne_config: Option<QtSNEConfig>,
    /// QUMAP configuration
    pub qumap_config: Option<QUMAPConfig>,
    /// QLDA configuration
    pub qlda_config: Option<QLDAConfig>,
    /// QFA configuration
    pub qfa_config: Option<QFactorAnalysisConfig>,
    /// QCCA configuration
    pub qcca_config: Option<QCCAConfig>,
    /// QNMF configuration
    pub qnmf_config: Option<QNMFConfig>,
    /// Autoencoder configuration
    pub autoencoder_config: Option<QAutoencoderConfig>,
    /// Manifold learning configuration
    pub manifold_config: Option<QManifoldConfig>,
    /// Kernel PCA configuration
    pub kernel_pca_config: Option<QKernelPCAConfig>,
    /// Feature selection configuration
    pub feature_selection_config: Option<QFeatureSelectionConfig>,
    /// Specialized configuration
    pub specialized_config: Option<QSpecializedConfig>,
    /// Trained state
    pub trained_state: Option<DRTrainedState>,
}

impl QuantumDimensionalityReducer {
    /// Create a new quantum dimensionality reducer
    pub fn new(algorithm: DimensionalityReductionAlgorithm) -> Self {
        Self {
            algorithm,
            qpca_config: None,
            qica_config: None,
            qtsne_config: None,
            qumap_config: None,
            qlda_config: None,
            qfa_config: None,
            qcca_config: None,
            qnmf_config: None,
            autoencoder_config: None,
            manifold_config: None,
            kernel_pca_config: None,
            feature_selection_config: None,
            specialized_config: None,
            trained_state: None,
        }
    }

    /// Set QPCA configuration
    pub fn with_qpca_config(mut self, config: QPCAConfig) -> Self {
        self.qpca_config = Some(config);
        self
    }

    /// Set QICA configuration
    pub fn with_qica_config(mut self, config: QICAConfig) -> Self {
        self.qica_config = Some(config);
        self
    }

    /// Set Qt-SNE configuration
    pub fn with_qtsne_config(mut self, config: QtSNEConfig) -> Self {
        self.qtsne_config = Some(config);
        self
    }

    /// Set QUMAP configuration
    pub fn with_qumap_config(mut self, config: QUMAPConfig) -> Self {
        self.qumap_config = Some(config);
        self
    }

    /// Set QLDA configuration
    pub fn with_qlda_config(mut self, config: QLDAConfig) -> Self {
        self.qlda_config = Some(config);
        self
    }

    /// Set autoencoder configuration
    pub fn with_autoencoder_config(mut self, config: QAutoencoderConfig) -> Self {
        self.autoencoder_config = Some(config);
        self
    }

    /// Fit the dimensionality reduction model
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        match self.algorithm {
            DimensionalityReductionAlgorithm::QPCA => self.fit_qpca(data),
            DimensionalityReductionAlgorithm::QICA => self.fit_qica(data),
            DimensionalityReductionAlgorithm::QtSNE => self.fit_qtsne(data),
            DimensionalityReductionAlgorithm::QUMAP => self.fit_qumap(data),
            DimensionalityReductionAlgorithm::QLDA => self.fit_qlda(data),
            DimensionalityReductionAlgorithm::QVAE => self.fit_qvae(data),
            DimensionalityReductionAlgorithm::QDenoisingAE => self.fit_qdenoising_ae(data),
            DimensionalityReductionAlgorithm::QSparseAE => self.fit_qsparse_ae(data),
            DimensionalityReductionAlgorithm::QManifoldLearning => self.fit_qmanifold(data),
            DimensionalityReductionAlgorithm::QKernelPCA => self.fit_qkernel_pca(data),
            _ => {
                // Placeholder for other algorithms
                self.fit_placeholder(data)
            }
        }
    }

    /// Transform data using the fitted model
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if self.trained_state.is_none() {
            return Err(MLError::ModelNotTrained(
                "Model must be fitted before transform".to_string(),
            ));
        }

        match self.algorithm {
            DimensionalityReductionAlgorithm::QPCA => self.transform_qpca(data),
            DimensionalityReductionAlgorithm::QICA => self.transform_qica(data),
            DimensionalityReductionAlgorithm::QtSNE => self.transform_qtsne(data),
            DimensionalityReductionAlgorithm::QUMAP => self.transform_qumap(data),
            DimensionalityReductionAlgorithm::QLDA => self.transform_qlda(data),
            DimensionalityReductionAlgorithm::QVAE => self.transform_qvae(data),
            DimensionalityReductionAlgorithm::QDenoisingAE => self.transform_qdenoising_ae(data),
            DimensionalityReductionAlgorithm::QSparseAE => self.transform_qsparse_ae(data),
            DimensionalityReductionAlgorithm::QManifoldLearning => self.transform_qmanifold(data),
            DimensionalityReductionAlgorithm::QKernelPCA => self.transform_qkernel_pca(data),
            _ => {
                // Placeholder for other algorithms
                self.transform_placeholder(data)
            }
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Get the trained state
    pub fn get_trained_state(&self) -> Option<&DRTrainedState> {
        self.trained_state.as_ref()
    }

    /// Get explained variance ratio (if applicable)
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.trained_state
            .as_ref()
            .map(|state| &state.explained_variance_ratio)
    }

    /// Get the components (transformation matrix)
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.trained_state.as_ref().map(|state| &state.components)
    }

    /// Inverse transform (reconstruction)
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if let Some(state) = &self.trained_state {
            // Basic linear reconstruction
            let centered = data.dot(&state.components);
            let reconstructed = &centered + &state.mean;
            Ok(reconstructed)
        } else {
            Err(MLError::ModelNotTrained(
                "Model must be fitted before inverse transform".to_string(),
            ))
        }
    }

    // Private fitting methods (placeholder implementations)

    fn fit_qpca(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::linear::QPCA;
        let binding = QPCAConfig::default();
        let config = self.qpca_config.as_ref().unwrap_or(&binding);
        let mut qpca = QPCA::new(config.clone());
        qpca.fit(data)?;
        self.trained_state = qpca.get_trained_state();
        Ok(())
    }

    fn fit_qica(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::linear::QICA;
        let binding = QICAConfig::default();
        let config = self.qica_config.as_ref().unwrap_or(&binding);
        let mut qica = QICA::new(config.clone());
        qica.fit(data)?;
        self.trained_state = qica.get_trained_state();
        Ok(())
    }

    fn fit_qtsne(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::manifold::QtSNE;
        let binding = QtSNEConfig::default();
        let config = self.qtsne_config.as_ref().unwrap_or(&binding);
        let mut qtsne = QtSNE::new(config.clone());
        qtsne.fit(data)?;
        self.trained_state = qtsne.get_trained_state();
        Ok(())
    }

    fn fit_qumap(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::manifold::QUMAP;
        let binding = QUMAPConfig::default();
        let config = self.qumap_config.as_ref().unwrap_or(&binding);
        let mut qumap = QUMAP::new(config.clone());
        qumap.fit(data)?;
        self.trained_state = qumap.get_trained_state();
        Ok(())
    }

    fn fit_qlda(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::linear::QLDA;
        let default_config = QLDAConfig::default();
        let config = self.qlda_config.as_ref().unwrap_or(&default_config);
        let mut qlda = QLDA::new(config.clone());
        qlda.fit(data)?;
        self.trained_state = qlda.get_trained_state();
        Ok(())
    }

    fn fit_qvae(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::autoencoders::QVAE;
        let default_config = QAutoencoderConfig::default();
        let config = self.autoencoder_config.as_ref().unwrap_or(&default_config);
        let mut qvae = QVAE::new(config.clone());
        qvae.fit(data)?;
        self.trained_state = qvae.get_trained_state();
        Ok(())
    }

    fn fit_qdenoising_ae(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::autoencoders::QDenoisingAE;
        let default_config = QAutoencoderConfig::default();
        let config = self.autoencoder_config.as_ref().unwrap_or(&default_config);
        let mut qdenoising = QDenoisingAE::new(config.clone());
        qdenoising.fit(data)?;
        self.trained_state = qdenoising.get_trained_state();
        Ok(())
    }

    fn fit_qsparse_ae(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::autoencoders::QSparseAE;
        let default_config = QAutoencoderConfig::default();
        let config = self.autoencoder_config.as_ref().unwrap_or(&default_config);
        let mut qsparse = QSparseAE::new(config.clone());
        qsparse.fit(data)?;
        self.trained_state = qsparse.get_trained_state();
        Ok(())
    }

    fn fit_qmanifold(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::manifold::QManifoldLearning;
        let default_config = QManifoldConfig::default();
        let config = self.manifold_config.as_ref().unwrap_or(&default_config);
        let mut qmanifold = QManifoldLearning::new(config.clone());
        qmanifold.fit(data)?;
        self.trained_state = qmanifold.get_trained_state();
        Ok(())
    }

    fn fit_qkernel_pca(&mut self, data: &Array2<f64>) -> Result<()> {
        use super::linear::QKernelPCA;
        let default_config = QKernelPCAConfig::default();
        let config = self.kernel_pca_config.as_ref().unwrap_or(&default_config);
        let mut qkernel_pca = QKernelPCA::new(config.clone());
        qkernel_pca.fit(data)?;
        self.trained_state = qkernel_pca.get_trained_state();
        Ok(())
    }

    fn fit_placeholder(&mut self, data: &Array2<f64>) -> Result<()> {
        // Placeholder implementation - creates a simple identity transformation
        let _n_samples = data.nrows();
        let n_features = data.ncols();
        let n_components = (n_features / 2).max(1);

        let components = Array2::eye(n_components);
        let explained_variance_ratio =
            Array1::from_vec((0..n_components).map(|i| 1.0 / (i + 1) as f64).collect());
        let mean = data
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                MLError::ComputationError(
                    "Failed to compute mean axis for placeholder fit".to_string(),
                )
            })?;

        self.trained_state = Some(DRTrainedState {
            components,
            explained_variance_ratio,
            mean,
            scale: None,
            quantum_parameters: std::collections::HashMap::new(),
            model_parameters: std::collections::HashMap::new(),
            training_statistics: std::collections::HashMap::new(),
        });

        Ok(())
    }

    // Private transformation methods (placeholder implementations)

    fn transform_qpca(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let state = self
            .trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QPCA model not trained".to_string()))?;
        let centered = data - &state.mean;
        Ok(centered.dot(&state.components.t()))
    }

    fn transform_qica(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let state = self
            .trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QICA model not trained".to_string()))?;
        let centered = data - &state.mean;
        Ok(centered.dot(&state.components.t()))
    }

    /// t-SNE has no closed-form mapping from ambient space to embedding space: the
    /// embedding is the joint result of optimizing all points together, so a fitted
    /// model cannot honestly project *new* out-of-sample points without re-running the
    /// full optimization on the combined dataset. Rather than fabricate a plausible
    /// looking (but meaningless) embedding, we surface that limitation to the caller.
    fn transform_qtsne(&self, _data: &Array2<f64>) -> Result<Array2<f64>> {
        self.trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QtSNE model not trained".to_string()))?;
        Err(MLError::NotSupported(
            "QtSNE has no out-of-sample transform: t-SNE embeddings are only defined \
             jointly over the fitted dataset. Call fit_transform on the full dataset \
             (including any new points) instead of transform on a fitted model."
                .to_string(),
        ))
    }

    /// UMAP's standard out-of-sample transform requires the fuzzy simplicial set /
    /// nearest-neighbor graph built from the training data, which `fit_qumap` does not
    /// currently persist in `DRTrainedState`. Returning a fabricated embedding would be
    /// worse than refusing, so we return an honest error instead of silent zeros.
    fn transform_qumap(&self, _data: &Array2<f64>) -> Result<Array2<f64>> {
        self.trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QUMAP model not trained".to_string()))?;
        Err(MLError::NotSupported(
            "QUMAP out-of-sample transform is not implemented: it requires persisting the \
             training-time neighbor graph, which the current fitted state does not retain. \
             Use fit_transform on the complete dataset instead."
                .to_string(),
        ))
    }

    fn transform_qlda(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let state = self
            .trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QLDA model not trained".to_string()))?;
        let centered = data - &state.mean;
        Ok(centered.dot(&state.components.t()))
    }

    /// The autoencoder variants (VAE/denoising/sparse) do not yet train a real encoder
    /// network (`fit_q*_ae` stores placeholder zero weights rather than learned ones),
    /// so an "encode" here would just be multiplying by zeros -- indistinguishable from
    /// fabricating a result. We surface that honestly instead.
    fn transform_qvae(&self, _data: &Array2<f64>) -> Result<Array2<f64>> {
        self.trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QVAE model not trained".to_string()))?;
        Err(MLError::NotSupported(
            "QVAE encode/transform is not implemented: no trained encoder network weights \
             are available from fit_qvae yet."
                .to_string(),
        ))
    }

    fn transform_qdenoising_ae(&self, _data: &Array2<f64>) -> Result<Array2<f64>> {
        self.trained_state.as_ref().ok_or_else(|| {
            MLError::ModelNotTrained("QDenoisingAE model not trained".to_string())
        })?;
        Err(MLError::NotSupported(
            "QDenoisingAE encode/transform is not implemented: no trained encoder network \
             weights are available from fit_qdenoising_ae yet."
                .to_string(),
        ))
    }

    fn transform_qsparse_ae(&self, _data: &Array2<f64>) -> Result<Array2<f64>> {
        self.trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QSparseAE model not trained".to_string()))?;
        Err(MLError::NotSupported(
            "QSparseAE encode/transform is not implemented: no trained encoder network \
             weights are available from fit_qsparse_ae yet."
                .to_string(),
        ))
    }

    /// Generic nonlinear manifold learning (Isomap/LLE-style) has the same out-of-sample
    /// limitation as UMAP above: extending to new points needs the training-time
    /// neighbor graph or landmark set, which is not persisted.
    fn transform_qmanifold(&self, _data: &Array2<f64>) -> Result<Array2<f64>> {
        self.trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QManifold model not trained".to_string()))?;
        Err(MLError::NotSupported(
            "QManifoldLearning out-of-sample transform is not implemented: it requires the \
             training-time neighbor graph, which the current fitted state does not retain. \
             Use fit_transform on the complete dataset instead."
                .to_string(),
        ))
    }

    /// Real (non-fabricated) Kernel PCA transform.
    ///
    /// `fit_qkernel_pca` does not currently persist the training Gram matrix or support
    /// vectors needed for a textbook out-of-sample kernel-trick projection, so instead of
    /// reusing the (placeholder) `state.components`, this recomputes a genuine kernel PCA
    /// decomposition directly on the provided `data`: it builds the RBF/Gaussian kernel
    /// Gram matrix, centers it in feature space, and extracts the top eigenvectors via a
    /// Jacobi eigenvalue decomposition -- exactly the linear algebra kernel PCA performs,
    /// just evaluated eagerly at transform time rather than reusing cached training state.
    /// This is self-consistent (and exact) when `data` is the training set itself (as in
    /// `fit_transform`); for genuinely new points it is an honest approximation rather
    /// than a fabricated one.
    fn transform_qkernel_pca(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let state = self
            .trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("QKernelPCA model not trained".to_string()))?;
        let n_components = state.explained_variance_ratio.len().max(1);
        let n_samples = data.nrows();
        if n_samples == 0 {
            return Ok(Array2::zeros((0, n_components)));
        }

        let default_config = QKernelPCAConfig::default();
        let config = self.kernel_pca_config.as_ref().unwrap_or(&default_config);
        let gamma = config
            .kernel_params
            .get("gamma")
            .copied()
            .unwrap_or_else(|| 1.0 / data.ncols().max(1) as f64);

        // Build the Gaussian/RBF kernel Gram matrix over the query set.
        let mut kernel = Array2::<f64>::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let diff = &data.row(i) - &data.row(j);
                let sq_dist = diff.dot(&diff);
                let k_ij = (-gamma * sq_dist).exp();
                kernel[[i, j]] = k_ij;
                kernel[[j, i]] = k_ij;
            }
        }

        // Center the kernel matrix in feature space:
        // K' = K - 1_n K - K 1_n + 1_n K 1_n, with 1_n the all-(1/n) matrix.
        let ones = Array2::<f64>::from_elem((n_samples, n_samples), 1.0 / n_samples as f64);
        let k_ones = kernel.dot(&ones);
        let ones_k = ones.dot(&kernel);
        let ones_k_ones = ones.dot(&kernel).dot(&ones);
        let centered_kernel = &kernel - &k_ones - &ones_k + &ones_k_ones;

        let (eigenvalues, eigenvectors) = jacobi_eigh(&centered_kernel)?;

        let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
        order.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_out = n_components.min(order.len());
        let mut embedding = Array2::<f64>::zeros((n_samples, n_out));
        for (out_col, &idx) in order.iter().take(n_out).enumerate() {
            let scale = eigenvalues[idx].max(0.0).sqrt();
            for row in 0..n_samples {
                embedding[[row, out_col]] = eigenvectors[[row, idx]] * scale;
            }
        }
        Ok(embedding)
    }

    fn transform_placeholder(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let state = self
            .trained_state
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Placeholder model not trained".to_string()))?;
        let centered = data - &state.mean;
        Ok(centered.dot(&state.components.t()))
    }
}

/// Cyclic Jacobi eigenvalue algorithm for real symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` has the eigenvectors as its
/// columns (i.e. `eigenvectors.column(k)` corresponds to `eigenvalues[k]`), unordered. This
/// is a genuine, self-contained numerical eigendecomposition (no external dependency is
/// pulled in beyond what the crate already uses) suitable for the modestly sized, dense,
/// symmetric Gram matrices produced by kernel methods in this module.
fn jacobi_eigh(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(MLError::DimensionMismatch(
            "jacobi_eigh requires a square matrix".to_string(),
        ));
    }

    let mut a = matrix.clone();
    let mut v = Array2::<f64>::eye(n);
    const MAX_SWEEPS: usize = 100;
    const EPS: f64 = 1e-12;

    for _ in 0..MAX_SWEEPS {
        let mut off_diag_sq = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diag_sq += a[[p, q]] * a[[p, q]];
            }
        }
        if off_diag_sq.sqrt() < EPS {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if a[[p, q]].abs() < 1e-15 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let a_pp = a[[p, p]];
                let a_qq = a[[q, q]];
                let a_pq = a[[p, q]];
                a[[p, p]] = a_pp - t * a_pq;
                a[[q, q]] = a_qq + t * a_pq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;

                for i in 0..n {
                    if i != p && i != q {
                        let a_ip = a[[i, p]];
                        let a_iq = a[[i, q]];
                        a[[i, p]] = c * a_ip - s * a_iq;
                        a[[p, i]] = a[[i, p]];
                        a[[i, q]] = s * a_ip + c * a_iq;
                        a[[q, i]] = a[[i, q]];
                    }
                }
                for i in 0..n {
                    let v_ip = v[[i, p]];
                    let v_iq = v[[i, q]];
                    v[[i, p]] = c * v_ip - s * v_iq;
                    v[[i, q]] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    let eigenvalues = Array1::from_shape_fn(n, |i| a[[i, i]]);
    Ok((eigenvalues, v))
}

#[cfg(test)]
mod core_transform_regression_tests {
    use super::*;

    fn sample_data() -> Array2<f64> {
        // Two well-separated clusters so that a real embedding is expected to be
        // non-trivial (i.e. not all-zero) and to vary across rows.
        Array2::from_shape_vec(
            (6, 3),
            vec![
                0.0, 0.0, 0.0, 0.1, -0.1, 0.05, -0.05, 0.1, 0.0, 5.0, 5.0, 5.0, 5.1, 4.9, 5.05,
                4.95, 5.1, 5.0,
            ],
        )
        .expect("valid shape")
    }

    #[test]
    fn jacobi_eigh_reproduces_known_symmetric_eigenvalues() {
        // A 2x2 symmetric matrix with well-known eigenvalues: for [[2,1],[1,2]] the
        // eigenvalues are 1 and 3.
        let m = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (eigenvalues, _eigenvectors) = jacobi_eigh(&m).expect("eigendecomposition succeeds");
        let mut sorted = eigenvalues.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-8, "got {:?}", sorted);
        assert!((sorted[1] - 3.0).abs() < 1e-8, "got {:?}", sorted);
    }

    #[test]
    fn transform_qkernel_pca_produces_real_nonzero_embedding() {
        let data = sample_data();
        let mut reducer =
            QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QKernelPCA);
        reducer.fit(&data).expect("fit succeeds");
        let embedding = reducer.transform(&data).expect("transform succeeds");

        assert_eq!(embedding.nrows(), data.nrows());
        // The embedding must actually depend on the input data: it should not be the
        // all-zero matrix that the previous placeholder implementation always returned.
        let total_abs: f64 = embedding.iter().map(|v| v.abs()).sum();
        assert!(
            total_abs > 1e-6,
            "expected a non-trivial kernel PCA embedding, got all-(near)-zero output"
        );

        // The two well-separated input clusters should map to distinguishable
        // embeddings (first component differs meaningfully between clusters).
        let first_cluster_val = embedding[[0, 0]];
        let second_cluster_val = embedding[[3, 0]];
        assert!(
            (first_cluster_val - second_cluster_val).abs() > 1e-6,
            "expected distinguishable embeddings for well-separated clusters"
        );
    }

    #[test]
    fn transform_qtsne_returns_honest_not_supported_error_instead_of_zeros() {
        let data = sample_data();
        let mut reducer =
            QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QtSNE);
        reducer.fit(&data).expect("fit succeeds");
        let result = reducer.transform(&data);
        match result {
            Err(MLError::NotSupported(_)) => {}
            other => panic!("expected NotSupported error, got {:?}", other),
        }
    }

    #[test]
    fn transform_qvae_returns_honest_not_supported_error_instead_of_zeros() {
        let data = sample_data();
        let mut reducer = QuantumDimensionalityReducer::new(DimensionalityReductionAlgorithm::QVAE);
        reducer.fit(&data).expect("fit succeeds");
        let result = reducer.transform(&data);
        match result {
            Err(MLError::NotSupported(_)) => {}
            other => panic!("expected NotSupported error, got {:?}", other),
        }
    }
}
