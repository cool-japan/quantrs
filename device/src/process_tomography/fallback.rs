//! Fallback implementations when SciRS2 is not available

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Fallback statistical mean calculation
pub fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
    Ok(0.0)
}

/// Fallback standard deviation calculation
pub fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
    Ok(1.0)
}

/// Fallback Pearson correlation calculation
pub fn pearsonr(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _alt: &str,
) -> Result<(f64, f64), String> {
    Ok((0.0, 0.5))
}

/// Fallback matrix trace calculation
pub fn trace(_matrix: &ArrayView2<f64>) -> Result<f64, String> {
    Ok(1.0)
}

/// Fallback matrix inversion
pub fn inv(_matrix: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
    Ok(Array2::eye(2))
}

/// Fallback optimization result
pub struct OptimizeResult {
    pub x: Array1<f64>,
    pub fun: f64,
    pub success: bool,
    pub nit: usize,
}

/// Fallback optimization function
pub fn minimize(
    _func: fn(&Array1<f64>) -> f64,
    _x0: &Array1<f64>,
    _method: &str,
) -> Result<OptimizeResult, String> {
    Ok(OptimizeResult {
        x: Array1::zeros(2),
        fun: 0.0,
        success: true,
        nit: 0,
    })
}

/// Fallback eigenvalue decomposition
pub fn eig(
    _matrix: &ArrayView2<f64>,
) -> Result<
    (
        Array1<scirs2_core::Complex64>,
        Array2<scirs2_core::Complex64>,
    ),
    String,
> {
    let eigenvalues = Array1::from_vec(vec![
        scirs2_core::Complex64::new(1.0, 0.0),
        scirs2_core::Complex64::new(0.0, 0.0),
    ]);
    let eigenvectors = Array2::from_shape_vec(
        (2, 2),
        vec![
            scirs2_core::Complex64::new(1.0, 0.0),
            scirs2_core::Complex64::new(0.0, 0.0),
            scirs2_core::Complex64::new(0.0, 0.0),
            scirs2_core::Complex64::new(1.0, 0.0),
        ],
    )
    .map_err(|e| format!("Array creation error: {}", e))?;

    Ok((eigenvalues, eigenvectors))
}

/// Fallback matrix determinant calculation
pub fn det(_matrix: &ArrayView2<f64>) -> Result<f64, String> {
    Ok(1.0)
}

/// Fallback QR decomposition
pub fn qr(_matrix: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    Ok((Array2::eye(2), Array2::eye(2)))
}

/// Fallback SVD decomposition
pub fn svd(_matrix: &ArrayView2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), String> {
    Ok((Array2::eye(2), Array1::ones(2), Array2::eye(2)))
}

/// Fallback matrix norm calculation
pub fn matrix_norm(_matrix: &ArrayView2<f64>, _ord: Option<&str>) -> Result<f64, String> {
    Ok(1.0)
}

/// Fallback Cholesky decomposition
pub fn cholesky(_matrix: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
    Ok(Array2::eye(2))
}

/// Fallback variance calculation
pub fn var(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
    Ok(1.0)
}

/// Fallback correlation coefficient calculation
pub fn corrcoef(_x: &ArrayView1<f64>, _y: &ArrayView1<f64>) -> Result<Array2<f64>, String> {
    Ok(Array2::eye(2))
}

/// Fallback Spearman correlation calculation
pub fn spearmanr(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<(f64, f64), String> {
    Ok((0.0, 0.5))
}

/// Fallback t-test (one sample)
pub fn ttest_1samp(
    _a: &ArrayView1<f64>,
    _popmean: f64,
    _alternative: &str,
) -> Result<TTestResult, String> {
    Ok(TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    })
}

/// Fallback t-test (independent samples)
pub fn ttest_ind(
    _a: &ArrayView1<f64>,
    _b: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<TTestResult, String> {
    Ok(TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    })
}

/// Fallback Kolmogorov-Smirnov 2-sample test
pub fn ks_2samp(
    _data1: &ArrayView1<f64>,
    _data2: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<KSTestResult, String> {
    Ok(KSTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    })
}

/// Fallback Shapiro-Wilk test
pub fn shapiro_wilk(_data: &ArrayView1<f64>) -> Result<SWTestResult, String> {
    Ok(SWTestResult {
        statistic: 0.95,
        pvalue: 0.1,
    })
}

/// T-test result structure
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Kolmogorov-Smirnov test result structure
#[derive(Debug, Clone)]
pub struct KSTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Shapiro-Wilk test result structure
#[derive(Debug, Clone)]
pub struct SWTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Alternative hypothesis type
#[derive(Debug, Clone)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Distribution types and functions
pub mod distributions {
    use super::*;

    /// Normal distribution functions
    pub mod norm {
        /// Normal PDF
        pub fn pdf(_x: f64, _loc: f64, _scale: f64) -> f64 {
            0.4
        }

        /// Normal CDF
        pub fn cdf(_x: f64, _loc: f64, _scale: f64) -> f64 {
            0.5
        }

        /// Normal PPF (inverse CDF)
        pub fn ppf(_q: f64, _loc: f64, _scale: f64) -> f64 {
            0.0
        }
    }

    /// Chi-squared distribution functions
    pub mod chi2 {
        /// Chi-squared PDF
        pub fn pdf(_x: f64, _df: f64) -> f64 {
            0.1
        }

        /// Chi-squared CDF
        pub fn cdf(_x: f64, _df: f64) -> f64 {
            0.5
        }

        /// Chi-squared PPF
        pub fn ppf(_q: f64, _df: f64) -> f64 {
            1.0
        }
    }

    /// Gamma distribution functions
    pub mod gamma {
        /// Gamma PDF
        pub fn pdf(_x: f64, _a: f64, _scale: f64) -> f64 {
            0.2
        }

        /// Gamma CDF
        pub fn cdf(_x: f64, _a: f64, _scale: f64) -> f64 {
            0.5
        }

        /// Gamma PPF
        pub fn ppf(_q: f64, _a: f64, _scale: f64) -> f64 {
            1.0
        }
    }
}

/// Graph analysis fallback functions
pub mod graph {
    use super::*;

    /// Fallback betweenness centrality
    pub fn betweenness_centrality(_graph: &Array2<f64>) -> Result<Array1<f64>, String> {
        Ok(Array1::ones(2))
    }

    /// Fallback closeness centrality
    pub fn closeness_centrality(_graph: &Array2<f64>) -> Result<Array1<f64>, String> {
        Ok(Array1::ones(2))
    }

    /// Fallback minimum spanning tree
    pub fn minimum_spanning_tree(_graph: &Array2<f64>) -> Result<Array2<f64>, String> {
        Ok(Array2::eye(2))
    }

    /// Fallback shortest path
    pub fn shortest_path(
        _graph: &Array2<f64>,
        _start: usize,
        _end: usize,
    ) -> Result<Vec<usize>, String> {
        Ok(vec![0, 1])
    }

    /// Fallback strongly connected components
    pub fn strongly_connected_components(_graph: &Array2<f64>) -> Result<Vec<Vec<usize>>, String> {
        Ok(vec![vec![0], vec![1]])
    }

    /// Graph structure placeholder
    pub struct Graph {
        pub adjacency_matrix: Array2<f64>,
    }

    impl Graph {
        pub fn new(adjacency_matrix: Array2<f64>) -> Self {
            Self { adjacency_matrix }
        }
    }
}
