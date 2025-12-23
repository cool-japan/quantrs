//! Model calibration utilities for probability calibration

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};

pub mod bayesianbinningquantiles_traits;
pub mod functions;
pub mod isotonicregression_traits;
pub mod matrixscaler_traits;
pub mod plattscaler_traits;
pub mod temperaturescaler_traits;
pub mod types;
pub mod vectorscaler_traits;

// Re-export all types
pub use bayesianbinningquantiles_traits::*;
pub use functions::*;
pub use isotonicregression_traits::*;
pub use matrixscaler_traits::*;
pub use plattscaler_traits::*;
pub use temperaturescaler_traits::*;
pub use types::*;
pub use vectorscaler_traits::*;
