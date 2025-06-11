//! Linear dimensionality reduction methods

pub mod qica;
pub mod qkernel_pca;
pub mod qlda;
pub mod qpca;

// Re-export all linear methods
pub use qica::*;
pub use qkernel_pca::*;
pub use qlda::*;
pub use qpca::*;
