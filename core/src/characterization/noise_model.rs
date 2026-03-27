//! Noise model types and Kraus operator implementations

use crate::error::QuantRS2Result;
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64 as Complex;

/// Noise model types for quantum systems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseModel {
    /// Depolarizing channel: ρ → (1-p)ρ + p(I/d)
    Depolarizing { probability: f64 },
    /// Amplitude damping: models energy dissipation
    AmplitudeDamping { gamma: f64 },
    /// Phase damping: models loss of quantum coherence
    PhaseDamping { lambda: f64 },
    /// Bit flip channel: X error with probability p
    BitFlip { probability: f64 },
    /// Phase flip channel: Z error with probability p
    PhaseFlip { probability: f64 },
    /// Bit-phase flip channel: Y error with probability p
    BitPhaseFlip { probability: f64 },
    /// Pauli channel: general combination of X, Y, Z errors
    Pauli { p_x: f64, p_y: f64, p_z: f64 },
    /// Thermal relaxation (T1 and T2 processes)
    ThermalRelaxation { t1: f64, t2: f64, time: f64 },
}

impl NoiseModel {
    /// Get Kraus operators for this noise model
    pub fn kraus_operators(&self) -> Vec<Array2<Complex>> {
        match self {
            Self::Depolarizing { probability } => {
                let p = *probability;
                let sqrt_p = p.sqrt();
                let sqrt_1_p = (1.0 - p).sqrt();

                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(sqrt_1_p, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(sqrt_1_p, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(sqrt_p / 3.0_f64.sqrt(), 0.0),
                            Complex::new(sqrt_p / 3.0_f64.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, -sqrt_p / 3.0_f64.sqrt()),
                            Complex::new(0.0, sqrt_p / 3.0_f64.sqrt()),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(sqrt_p / 3.0_f64.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(-sqrt_p / 3.0_f64.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::AmplitudeDamping { gamma } => {
                let g = *gamma;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(1.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - g).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(g.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::PhaseDamping { lambda } => {
                let l = *lambda;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(1.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - l).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(l.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::BitFlip { probability } => {
                let p = *probability;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new((1.0 - p).sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - p).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(p.sqrt(), 0.0),
                            Complex::new(p.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::PhaseFlip { probability } => {
                let p = *probability;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new((1.0 - p).sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - p).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(p.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(-p.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::BitPhaseFlip { probability } => {
                let p = *probability;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new((1.0 - p).sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - p).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, -p.sqrt()),
                            Complex::new(0.0, p.sqrt()),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::Pauli { p_x, p_y, p_z } => {
                let p_i = 1.0 - p_x - p_y - p_z;
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(p_i.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(p_i.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(p_x.sqrt(), 0.0),
                            Complex::new(p_x.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, -p_y.sqrt()),
                            Complex::new(0.0, p_y.sqrt()),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(p_z.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(-p_z.sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
            Self::ThermalRelaxation { t1, t2, time } => {
                let p1 = 1.0 - (-time / t1).exp();
                let p2 = 1.0 - (-time / t2).exp();

                let gamma = p1;
                let lambda = (p2 - p1 / 2.0).max(0.0);

                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(1.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new((1.0 - gamma) * (1.0 - lambda).sqrt(), 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex::new(0.0, 0.0),
                            Complex::new(gamma.sqrt(), 0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.0),
                        ],
                    )
                    .expect("2x2 matrix creation"),
                ]
            }
        }
    }

    /// Apply noise model to a density matrix
    pub fn apply_to_density_matrix(
        &self,
        rho: &Array2<Complex>,
    ) -> QuantRS2Result<Array2<Complex>> {
        let kraus_ops = self.kraus_operators();
        let dim = rho.nrows();
        let mut result = Array2::<Complex>::zeros((dim, dim));

        for k in &kraus_ops {
            let k_rho = k.dot(rho);
            let k_dag = k.t().mapv(|x| x.conj());
            let k_rho_k_dag = k_rho.dot(&k_dag);
            result = result + k_rho_k_dag;
        }

        Ok(result)
    }
}
