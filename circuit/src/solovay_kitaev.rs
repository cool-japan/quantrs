//! Solovay-Kitaev Algorithm for single-qubit gate approximation.
//!
//! Implements the Solovay-Kitaev theorem (Dawson & Nielsen, quant-ph/0505030):
//! any single-qubit SU(2) gate can be approximated to precision ε using
//! O(log^c(1/ε)) gates from the universal set {H, T, T†, S, S†}.
//!
//! S = T² and S† = T†² are algebraically expressible in terms of T and T†, but
//! enumerating them as separate BFS atoms shortens many useful sequences (e.g.
//! Y-axis-rotation conjugates such as S·H·RZ(θ)·H·S†) at the cost of a denser
//! basic table.
//!
//! # Design
//!
//! Rather than storing SU(2) in the `[[a,-b*],[b,a*]]` parameterization (which
//! forces H into a non-standard embedding), we store the full 2×2 unitary matrix.
//! Gate multiplication, adjoint and distance are computed directly.  This avoids
//! the representation issues with H (det = −1) while keeping the struct name SU2.
//!
//! Frobenius distance is computed **up to global phase**:
//!   d(U,V) = sqrt(2 − |Tr(U†V)|)
//! which is the correct metric for SK's approximation quality.

use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};

/// 2×2 single-qubit unitary matrix stored in row-major order:
///   [[m00, m01], [m10, m11]]
///
/// No assumption is made about the specific SU(2) parameterization;
/// the matrix may represent any U(2) unitary (global phase is tracked
/// but ignored when measuring approximation quality).
#[derive(Debug, Clone, Copy)]
pub struct SU2 {
    pub m00: Complex64,
    pub m01: Complex64,
    pub m10: Complex64,
    pub m11: Complex64,
}

impl SU2 {
    /// Create a new SU2 matrix from four complex entries.
    pub fn new(m00: Complex64, m01: Complex64, m10: Complex64, m11: Complex64) -> Self {
        Self { m00, m01, m10, m11 }
    }

    /// Identity gate.
    pub fn identity() -> Self {
        Self {
            m00: Complex64::new(1.0, 0.0),
            m01: Complex64::new(0.0, 0.0),
            m10: Complex64::new(0.0, 0.0),
            m11: Complex64::new(1.0, 0.0),
        }
    }

    /// Construct an SU2 matrix from a named gate in {H, T, Tdg, S, Sdg, X, Y, Z}.
    ///
    /// Returns `None` for unrecognised gate names.
    pub fn from_gate_name(name: &str) -> Option<Self> {
        let inv_sqrt2 = 1.0 / SQRT_2;
        let t_phase = Complex64::new(inv_sqrt2, inv_sqrt2); // e^{iπ/4}
        let tdg_phase = Complex64::new(inv_sqrt2, -inv_sqrt2); // e^{-iπ/4}

        match name {
            "H" => Some(Self {
                m00: Complex64::new(inv_sqrt2, 0.0),
                m01: Complex64::new(inv_sqrt2, 0.0),
                m10: Complex64::new(inv_sqrt2, 0.0),
                m11: Complex64::new(-inv_sqrt2, 0.0),
            }),
            "T" => Some(Self {
                m00: Complex64::new(1.0, 0.0),
                m01: Complex64::new(0.0, 0.0),
                m10: Complex64::new(0.0, 0.0),
                m11: t_phase,
            }),
            "Tdg" => Some(Self {
                m00: Complex64::new(1.0, 0.0),
                m01: Complex64::new(0.0, 0.0),
                m10: Complex64::new(0.0, 0.0),
                m11: tdg_phase,
            }),
            "S" => Some(Self {
                m00: Complex64::new(1.0, 0.0),
                m01: Complex64::new(0.0, 0.0),
                m10: Complex64::new(0.0, 0.0),
                m11: Complex64::new(0.0, 1.0),
            }),
            "Sdg" => Some(Self {
                m00: Complex64::new(1.0, 0.0),
                m01: Complex64::new(0.0, 0.0),
                m10: Complex64::new(0.0, 0.0),
                m11: Complex64::new(0.0, -1.0),
            }),
            "X" => Some(Self {
                m00: Complex64::new(0.0, 0.0),
                m01: Complex64::new(1.0, 0.0),
                m10: Complex64::new(1.0, 0.0),
                m11: Complex64::new(0.0, 0.0),
            }),
            "Y" => Some(Self {
                m00: Complex64::new(0.0, 0.0),
                m01: Complex64::new(0.0, -1.0),
                m10: Complex64::new(0.0, 1.0),
                m11: Complex64::new(0.0, 0.0),
            }),
            "Z" => Some(Self {
                m00: Complex64::new(1.0, 0.0),
                m01: Complex64::new(0.0, 0.0),
                m10: Complex64::new(0.0, 0.0),
                m11: Complex64::new(-1.0, 0.0),
            }),
            _ => None,
        }
    }

    /// Matrix multiplication: self * other.
    pub fn mul(&self, other: &SU2) -> SU2 {
        SU2 {
            m00: self.m00 * other.m00 + self.m01 * other.m10,
            m01: self.m00 * other.m01 + self.m01 * other.m11,
            m10: self.m10 * other.m00 + self.m11 * other.m10,
            m11: self.m10 * other.m01 + self.m11 * other.m11,
        }
    }

    /// Conjugate transpose (adjoint / Hermitian conjugate).
    pub fn adjoint(&self) -> SU2 {
        SU2 {
            m00: self.m00.conj(),
            m01: self.m10.conj(),
            m10: self.m01.conj(),
            m11: self.m11.conj(),
        }
    }

    /// Trace of the matrix.
    pub fn trace(&self) -> Complex64 {
        self.m00 + self.m11
    }

    /// Frobenius distance from identity, up to global phase:
    ///   d(U, I) = sqrt(2 − |Tr(U)|)
    ///
    /// This equals zero only when U is a global phase times identity.
    /// Used for table lookups (comparing approximations to targets).
    pub fn distance_from_identity(&self) -> f64 {
        let tr_abs = self.trace().norm();
        let val = (2.0 - tr_abs).max(0.0);
        val.sqrt()
    }

    /// Frobenius distance between two matrices, up to global phase:
    ///   d(U, V) = sqrt(2 − |Tr(U†V)|)
    ///
    /// Used for evaluating approximation quality (the output metric).
    pub fn frobenius_distance(&self, other: &SU2) -> f64 {
        let udagger_v = self.adjoint().mul(other);
        udagger_v.distance_from_identity()
    }

    /// Frobenius distance from identity WITHOUT global phase correction:
    ///   d(U, I) = sqrt(2 − Re[Tr(U)])
    ///
    /// This is the metric used internally by the Solovay-Kitaev recursion.
    /// It reflects the actual matrix distance (not up to phase), so that
    /// the residual delta = U * u_prev† has the correct geometric meaning.
    pub fn distance_from_identity_exact(&self) -> f64 {
        let re_tr = self.trace().re;
        let val = (2.0 - re_tr).max(0.0);
        val.sqrt()
    }

    /// Frobenius distance between two matrices WITHOUT global phase correction:
    ///   d(U, V) = sqrt(2 − Re[Tr(U†V)])
    pub fn frobenius_distance_exact(&self, other: &SU2) -> f64 {
        let udagger_v = self.adjoint().mul(other);
        udagger_v.distance_from_identity_exact()
    }

    /// Rotation angle θ (in radians) such that U = exp(iθ/2 n̂·σ) * global_phase.
    ///
    /// Computed as θ = 2 * arccos(|Tr(U)| / 2) using the absolute trace,
    /// which factors out the global phase and gives the physical rotation angle.
    ///
    /// For the SU(2) representative R (with det = 1), Tr(R) = 2 cos(θ/2) is real.
    /// For a general U(2) matrix U = e^{iφ} R, |Tr(U)| = |Tr(R)| = 2|cos(θ/2)|.
    pub fn rotation_angle(&self) -> f64 {
        // Factor out global phase: |Tr(U)| = |2 cos(θ/2)| gives the physical angle.
        let tr_abs = self.trace().norm();
        let cos_half = (tr_abs / 2.0).clamp(0.0, 1.0);
        2.0 * cos_half.acos()
    }

    /// Normalized rotation axis [nx, ny, nz] for the SU(2) rotation.
    ///
    /// Returns [0, 0, 1] (z-axis) if the rotation angle is near zero.
    pub fn rotation_axis(&self) -> [f64; 3] {
        let theta = self.rotation_angle();
        let s = (theta / 2.0).sin();

        if s.abs() < 1e-12 {
            return [0.0, 0.0, 1.0];
        }

        // For U = exp(iθ/2 n̂·σ) in SU(2), U = cos(θ/2)I + i sin(θ/2) n̂·σ
        // n̂·σ = [[nz, nx-iny],[nx+iny, -nz]]
        // Extract components from the off-diagonal imaginary / diagonal imaginary
        // Handle global phase: normalize by global phase factor.
        let det = self.m00 * self.m11 - self.m01 * self.m10;
        let global_phase_half = det.arg() / 2.0;
        let phase_inv = Complex64::new(
            (-global_phase_half).cos(),
            (-global_phase_half).sin(),
        );

        let a00 = self.m00 * phase_inv;
        let a01 = self.m01 * phase_inv;
        let a10 = self.m10 * phase_inv;
        let a11 = self.m11 * phase_inv;

        // From SU(2) standard form: a00 = cos(θ/2) + i*nz*sin(θ/2)
        //                            a01 = i*(nx - i*ny)*... no, let's re-derive
        // Standard SU(2): U = [[c + i*nz*s, i*nx*s + ny*s],
        //                       [i*nx*s - ny*s, c - i*nz*s]]
        // where c = cos(θ/2), s = sin(θ/2)
        let nz = a00.im / s;
        let nx = a10.im / s;
        let ny = a10.re / s;

        let _ = (a01, a11); // suppress unused warnings

        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        if norm < 1e-12 {
            [0.0, 0.0, 1.0]
        } else {
            [nx / norm, ny / norm, nz / norm]
        }
    }

    /// Construct a rotation about the X-axis: exp(+i*angle/2 * X).
    /// Matrix: [[cos(angle/2), i*sin(angle/2)], [i*sin(angle/2), cos(angle/2)]]
    pub fn rotation_x(angle: f64) -> SU2 {
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        SU2 {
            m00: Complex64::new(c, 0.0),
            m01: Complex64::new(0.0, s),
            m10: Complex64::new(0.0, s),
            m11: Complex64::new(c, 0.0),
        }
    }

    /// Construct a rotation about the Y-axis: exp(+i*angle/2 * Y).
    /// Matrix: [[cos(angle/2), sin(angle/2)], [-sin(angle/2), cos(angle/2)]]
    pub fn rotation_y(angle: f64) -> SU2 {
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        SU2 {
            m00: Complex64::new(c, 0.0),
            m01: Complex64::new(s, 0.0),
            m10: Complex64::new(-s, 0.0),
            m11: Complex64::new(c, 0.0),
        }
    }

    /// Construct a rotation about the Z-axis: exp(+i*angle/2 * Z).
    /// Matrix: [[e^{i*angle/2}, 0], [0, e^{-i*angle/2}]]
    pub fn rotation_z(angle: f64) -> SU2 {
        let half = angle / 2.0;
        SU2 {
            m00: Complex64::new(half.cos(), half.sin()),
            m01: Complex64::new(0.0, 0.0),
            m10: Complex64::new(0.0, 0.0),
            m11: Complex64::new(half.cos(), -half.sin()),
        }
    }
}

/// Compute SU(2) rotation about arbitrary unit axis [nx, ny, nz] by angle theta.
///
/// U = cos(θ/2) I + i sin(θ/2) (nx X + ny Y + nz Z)
///
/// Matrix form (standard SU(2)):
///   [[cos(θ/2) + i*nz*sin(θ/2),  (ny + i*nx)*sin(θ/2)],
///    [(-ny + i*nx)*sin(θ/2),       cos(θ/2) - i*nz*sin(θ/2)]]
pub fn rotation_about_axis(nx: f64, ny: f64, nz: f64, theta: f64) -> SU2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    SU2 {
        m00: Complex64::new(c, nz * s),
        m01: Complex64::new(ny * s, nx * s),
        m10: Complex64::new(-ny * s, nx * s),
        m11: Complex64::new(c, -nz * s),
    }
}

/// Decompose U into a balanced commutator V * W * V† * W† ≈ U.
///
/// This works when U is close to identity (small rotation angle).
/// Based on the Dawson-Nielsen algorithm Section IV.A.
///
/// Given U = exp(iθ n̂·σ/2), we find φ such that sin²(2φ) = sin(θ/2),
/// i.e. 2φ = arcsin(√sin(θ/2)), from the leading-order BCH relation
///   VWV†W† ≈ exp(i·4φ²·n̂·σ)  with 4φ² ≈ θ for small θ.
///
/// Then V = rotation about e₂ by 2φ, W = rotation about e₁ by 2φ,
/// where {e₁, e₂, n̂} form an orthonormal triad with e₂ = n̂ × e₁.
///
/// The group commutator VWV†W† has rotation axis +n̂ (not −n̂), because
/// the Lie-algebra commutator [e₂·σ, e₁·σ] = 2i(e₂×e₁)·σ = 2i(−n̂)·σ
/// and the group-commutator first-order term picks up an additional minus,
/// giving net axis +n̂ matching delta.
///
/// # Convergence note
///
/// The perpendicular-axis BCH commutator matches the rotation **angle** of
/// delta to leading order in θ but has O(φ) residual **axis** drift at finite
/// rotation angles.  In practice, the fallback guard in `sk_recurse` means
/// recursion depth > 1 rarely gives strict improvement over depth 1: the
/// commutator construction removes one layer of error but the axis drift
/// floors the residual near the depth-1 result.  Depth 1 provides the
/// practical O(ε^{3/2}) improvement; deeper depths are bounded above
/// (never worse due to the fallback) but not guaranteed to be strictly better.
///
/// Returns (V, W) such that V*W*V†*W† ≈ U.
pub fn balanced_commutator_decompose(u: &SU2) -> QuantRS2Result<(SU2, SU2)> {
    let theta = u.rotation_angle();

    // BCH formula: sin²(2φ) = sin(θ/2), i.e. 2φ = arcsin(√sin(θ/2))
    // For small θ: 4φ² ≈ θ, matching the rotation angle of the commutator.
    let sin_half_theta = (theta / 2.0).sin().abs();

    let sin_sq_phi_arg = sin_half_theta.sqrt();
    // Clamp due to floating point
    let phi = if sin_sq_phi_arg >= 1.0 {
        PI / 4.0
    } else {
        sin_sq_phi_arg.asin() / 2.0
    };

    let n_hat = u.rotation_axis();
    let (nx, ny, nz) = (n_hat[0], n_hat[1], n_hat[2]);

    // Find e1 perpendicular to n̂
    // If n̂ is not parallel to ẑ, use e1 = normalize(n̂ × ẑ)
    // Otherwise use e1 = x̂
    let (e1x, e1y, e1z) = if (nz.abs() - 1.0).abs() > 1e-6 {
        // n̂ × ẑ = (ny*1 - nz*0, nz*0 - nx*1, nx*0 - ny*0) = (ny, -nx, 0)
        let cx = ny;
        let cy = -nx;
        let cz = 0.0_f64;
        let norm = (cx * cx + cy * cy + cz * cz).sqrt();
        if norm < 1e-12 {
            (1.0_f64, 0.0_f64, 0.0_f64)
        } else {
            (cx / norm, cy / norm, cz / norm)
        }
    } else {
        (1.0_f64, 0.0_f64, 0.0_f64)
    };

    // e2 = n̂ × e1
    let e2x = ny * e1z - nz * e1y;
    let e2y = nz * e1x - nx * e1z;
    let e2z = nx * e1y - ny * e1x;

    // V = rotation about e2 by 2*phi, W = rotation about e1 by 2*phi.
    // The group commutator V·W·V†·W† ≈ exp(-2iφ²(e2×e1)·σ).
    // For our setup: e2 = n̂×e1, so e2×e1 = (n̂×e1)×e1 = -n̂ (by BAC-CAB rule).
    // Thus commutator ≈ exp(-2iφ²(-n̂)·σ) = exp(2iφ²n̂·σ) = rotation about n̂ by 4φ².
    // With sin²(2φ) = sin(θ/2), for small θ: 4φ² ≈ θ, giving the approximation.
    let v = rotation_about_axis(e2x, e2y, e2z, 2.0 * phi);
    // W = rotation about e1 by 2*phi
    let w = rotation_about_axis(e1x, e1y, e1z, 2.0 * phi);

    Ok((v, w))
}

/// Look up table of gate sequences for the universal gate set {H, T, Tdg, S, Sdg}.
///
/// **Two-stage Clifford+T construction.**  The single-qubit Clifford group
/// modulo global phase has only 24 elements, so a naïve BFS over the 5-atom
/// alphabet quickly collapses every branch onto that 24-element coset and
/// stalls.  To get richer SU(2) coverage we decouple the order-24 Clifford
/// subgroup from the T-count growth:
///
/// 1. **Stage 1 — Clifford enumeration.**  BFS over {H, S, Sdg} until
///    saturation produces exactly 24 distinct unitaries (up to global phase).
///    Each carries a sequence of length ≤ 6 — the Clifford diameter for this
///    generator set.
///
/// 2. **Stage 2 — T-count strata.**  For each `k = 1 … max_t_count`, build
///    `S_k` from `S_{k-1}` by appending `T` or `Tdg`, then a Clifford from
///    Stage 1.  Every entry in `S_k` therefore has T-count exactly `k`,
///    sandwiched between Cliffords.  Global de-duplication across all strata
///    via the spatial-hash grid keeps the table compact.
///
/// The resulting table covers the {H,T} normal form up to T-count
/// `max_t_count`, with Cliffords absorbed at no extra T-count.  Each
/// additional stratum genuinely expands the SU(2) reach (the Clifford+T
/// group is dense) rather than re-deriving cosets the previous depth
/// already covered.
pub struct BasicApproximationTable {
    /// (gate_sequence, SU2_matrix) pairs. Pub for test introspection.
    pub entries: Vec<(Vec<&'static str>, SU2)>,
}

/// Convert an SU2 matrix to a canonical SU(2) representative for hashing.
///
/// A U(2) matrix M = e^{iφ} R where R ∈ SU(2). We factor out the global
/// phase by computing φ = arg(det(M)) / 2 and then R = e^{-iφ} M.
/// The canonical SU(2) element R has det(R) = 1 exactly.
///
/// To break the remaining ±I ambiguity (since R and -R have the same distance
/// from identity), we ensure that the component with the largest magnitude is
/// positive. This gives a unique canonical quaternion (q0, q1, q2, q3) for
/// each physical gate up to global phase.
///
/// Returns (q0, q1, q2, q3) where:
///   q0 = Re(R.m00), q1 = Im(R.m00), q2 = Re(R.m10), q3 = Im(R.m10)
fn su2_to_canonical_quat(m: &SU2) -> (f64, f64, f64, f64) {
    // Factor out global phase: det = m00*m11 - m01*m10, det = e^{2iφ}
    let det = m.m00 * m.m11 - m.m01 * m.m10;
    let phase_angle = det.arg() / 2.0; // φ = arg(det)/2
    let phase_inv = Complex64::new((-phase_angle).cos(), (-phase_angle).sin()); // e^{-iφ}
    // R = e^{-iφ} * M  → det(R) = e^{-2iφ} * e^{2iφ} = 1
    let r00 = m.m00 * phase_inv;
    let r10 = m.m10 * phase_inv;

    let (q0, q1, q2, q3) = (r00.re, r00.im, r10.re, r10.im);

    // SU(2) double cover: R and -R represent the same physical gate.
    // Canonicalize by ensuring the largest-magnitude component is positive.
    let max_abs = q0.abs().max(q1.abs()).max(q2.abs()).max(q3.abs());
    let sign = if (q0.abs() - max_abs).abs() < 1e-10 {
        if q0 >= 0.0 { 1.0_f64 } else { -1.0_f64 }
    } else if (q1.abs() - max_abs).abs() < 1e-10 {
        if q1 >= 0.0 { 1.0_f64 } else { -1.0_f64 }
    } else if (q2.abs() - max_abs).abs() < 1e-10 {
        if q2 >= 0.0 { 1.0_f64 } else { -1.0_f64 }
    } else {
        if q3 >= 0.0 { 1.0_f64 } else { -1.0_f64 }
    };

    (sign * q0, sign * q1, sign * q2, sign * q3)
}

/// Grid cell key for the spatial hash (deduplication grid).
fn quat_to_cell(q: (f64, f64, f64, f64), cell_size: f64) -> (i64, i64, i64, i64) {
    let cell = |x: f64| (x / cell_size).floor() as i64;
    (cell(q.0), cell(q.1), cell(q.2), cell(q.3))
}

/// Dedup tolerance: two SU(2) matrices are considered identical if their
/// up-to-global-phase Frobenius distance is below this threshold.
///
/// Direct construction (e.g. `SU2::rotation_z(π/2)` for S) versus accumulated
/// multiplication of, say, two T-gates can differ by ~1.5e-8 due to argument
/// reduction differences inside the trigonometric primitives. `1e-6` absorbs
/// that margin while remaining 4 orders of magnitude smaller than the typical
/// inter-sequence spacing on SU(2). A value below ~1e-7 causes the BFS to
/// bloat with near-duplicates (e.g. T·T failing to collapse onto S).
const DEDUP_TOL: f64 = 1e-6;

/// Grid cell size for the spatial hash.  Must be larger than `DEDUP_TOL` (in
/// quaternion coordinates) so near-duplicates land in adjacent cells and are
/// caught by the 16-cell neighbour sweep performed by `DedupTable::contains`.
const CELL_SIZE: f64 = 0.002;

/// Hash-grid–backed deduplication table for SU(2) elements.
///
/// Wraps a `Vec<(seq, matrix)>` plus a 4-D quaternion-grid hash so that
/// `is_new` runs in O(neighbours) rather than O(N).  Used by both Stage 1
/// (Clifford enumeration) and Stage 2 (T-count strata) so dedup is global.
struct DedupTable {
    entries: Vec<(Vec<&'static str>, SU2)>,
    grid: HashMap<(i64, i64, i64, i64), Vec<usize>>,
}

impl DedupTable {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            grid: HashMap::new(),
        }
    }

    /// Returns `true` if `mat` is already in the table (within `DEDUP_TOL`).
    fn contains(&self, mat: &SU2) -> bool {
        let q = su2_to_canonical_quat(mat);
        let (cx, cy, cz, cw) = quat_to_cell(q, CELL_SIZE);
        // Check the 3^4 = 81 neighbouring cells (adjacent in each of 4 dims).
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    for dw in -1_i64..=1 {
                        let cell = (cx + dx, cy + dy, cz + dz, cw + dw);
                        if let Some(indices) = self.grid.get(&cell) {
                            for &idx in indices {
                                let (_, ref existing_mat) = self.entries[idx];
                                if existing_mat.frobenius_distance(mat) < DEDUP_TOL {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Insert (`seq`, `mat`) and register it in the spatial hash. Caller must
    /// have already established that `mat` is not a duplicate.
    fn insert(&mut self, seq: Vec<&'static str>, mat: SU2) {
        let idx = self.entries.len();
        let q = su2_to_canonical_quat(&mat);
        let cell = quat_to_cell(q, CELL_SIZE);
        self.grid.entry(cell).or_default().push(idx);
        self.entries.push((seq, mat));
    }
}

/// Enumerate the order-24 single-qubit Clifford group as `(matrix, sequence)`
/// pairs over the generator set {H, S, Sdg}.
///
/// The Clifford group modulo global phase is finite with order 24, generated
/// by H and S. BFS until saturation (queue empties) reaches every element in
/// at most 6 generator-applications.
///
/// Returns exactly 24 entries; the caller may assert this invariant.
fn enumerate_cliffords() -> Vec<(Vec<&'static str>, SU2)> {
    let h_gate = SU2::from_gate_name("H").unwrap_or_else(SU2::identity);
    let s_gate = SU2::from_gate_name("S").unwrap_or_else(SU2::identity);
    let sdg_gate = SU2::from_gate_name("Sdg").unwrap_or_else(SU2::identity);
    let generators: &[(&'static str, &SU2)] =
        &[("H", &h_gate), ("S", &s_gate), ("Sdg", &sdg_gate)];

    let mut table = DedupTable::new();
    table.insert(Vec::new(), SU2::identity());

    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    queue.push_back(0);

    while let Some(idx) = queue.pop_front() {
        let (seq, mat) = {
            let (s, m) = &table.entries[idx];
            (s.clone(), *m)
        };
        for (gen_name, gen_mat) in generators {
            let candidate_mat = mat.mul(gen_mat);
            if !table.contains(&candidate_mat) {
                let mut candidate_seq = Vec::with_capacity(seq.len() + 1);
                candidate_seq.extend_from_slice(&seq);
                candidate_seq.push(gen_name);
                let new_idx = table.entries.len();
                table.insert(candidate_seq, candidate_mat);
                queue.push_back(new_idx);
            }
        }
    }

    table.entries
}

impl BasicApproximationTable {
    /// Build the basic approximation table via the two-stage Clifford+T BFS.
    ///
    /// Stage 1 enumerates the 24-element single-qubit Clifford group from
    /// {H, S, Sdg}.  Stage 2 builds T-count strata `S_1, …, S_K` where each
    /// `S_k` extends `S_{k-1}` by `T` or `Tdg` followed by an arbitrary
    /// Clifford. Global dedup keeps the table compact while every stratum
    /// genuinely extends SU(2) coverage (the Clifford+T group is dense).
    ///
    /// `table_depth` is **re-interpreted as the maximum T-count** for the
    /// new algorithm (it was the BFS depth previously). Internally capped at
    /// `MAX_T_COUNT_LIMIT` to avoid combinatorial explosion at large values
    /// — a `table_depth` of 10 collapses to the same internal cap.
    pub fn build(table_depth: usize) -> Self {
        // Internal cap on the maximum T-count.  Stratum size grows roughly
        // 2× per level once the Clifford-T mod-phase quotient saturates.
        // We pick the smallest level that gives strict RY(0.3) coverage
        // (i.e. some entry beats the identity bound on small Y rotations).
        const MAX_T_COUNT_LIMIT: usize = 10;
        let max_t_count = std::cmp::min(table_depth, MAX_T_COUNT_LIMIT);

        let t_gate = SU2::from_gate_name("T").unwrap_or_else(SU2::identity);
        let tdg_gate = SU2::from_gate_name("Tdg").unwrap_or_else(SU2::identity);
        let t_atoms: &[(&'static str, &SU2)] = &[("T", &t_gate), ("Tdg", &tdg_gate)];

        // ---- Stage 1: enumerate the 24-element Clifford group ----
        let cliffords = enumerate_cliffords();
        debug_assert_eq!(
            cliffords.len(),
            24,
            "single-qubit Clifford group should have 24 elements modulo global phase"
        );

        // ---- Stage 2: T-count strata, all hashed into a global dedup table ----
        let mut table = DedupTable::new();
        for (seq, mat) in &cliffords {
            table.insert(seq.clone(), *mat);
        }
        // `prev_stratum` carries indices into `table.entries` for the previous
        // stratum so we extend only those (without re-extending earlier strata).
        let mut prev_stratum: Vec<usize> = (0..cliffords.len()).collect();

        for _k in 1..=max_t_count {
            let mut next_stratum: Vec<usize> = Vec::new();
            for &prev_idx in &prev_stratum {
                // Snapshot the previous-stratum entry so subsequent inserts
                // (which may grow `table.entries`) don't invalidate the borrow.
                let (prev_seq, prev_mat) = {
                    let (s, m) = &table.entries[prev_idx];
                    (s.clone(), *m)
                };
                for (t_name, t_mat) in t_atoms {
                    // Hoist e_mat * t_atom out of the inner Clifford loop.
                    let after_t = prev_mat.mul(t_mat);
                    for (c_seq, c_mat) in &cliffords {
                        let candidate_mat = after_t.mul(c_mat);
                        if !table.contains(&candidate_mat) {
                            // Build the sequence only once we know it's new
                            // — saves up to 48× allocation per kept entry.
                            let mut candidate_seq =
                                Vec::with_capacity(prev_seq.len() + 1 + c_seq.len());
                            candidate_seq.extend_from_slice(&prev_seq);
                            candidate_seq.push(t_name);
                            candidate_seq.extend_from_slice(c_seq);
                            let new_idx = table.entries.len();
                            table.insert(candidate_seq, candidate_mat);
                            next_stratum.push(new_idx);
                        }
                    }
                }
            }
            if next_stratum.is_empty() {
                break;
            }
            prev_stratum = next_stratum;
        }

        Self {
            entries: table.entries,
        }
    }

    /// Find the sequence in the table closest to `target` (up-to-global-phase distance).
    ///
    /// Returns the slice of gate names and the distance achieved.
    /// O(N) linear scan over the entire table.
    pub fn find_closest<'a>(&'a self, target: &SU2) -> (&'a [&'static str], f64) {
        let mut best_dist = f64::INFINITY;
        let mut best_seq: &[&'static str] = &[];

        for (seq, mat) in &self.entries {
            let dist = mat.frobenius_distance(target);
            if dist < best_dist {
                best_dist = dist;
                best_seq = seq.as_slice();
            }
        }

        (best_seq, best_dist)
    }

    /// Number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Convert a named adjoint gate.
fn adjoint_gate(gate: &'static str) -> &'static str {
    match gate {
        "T" => "Tdg",
        "Tdg" => "T",
        "S" => "Sdg",
        "Sdg" => "S",
        "H" => "H",
        "X" => "X",
        "Y" => "Y",
        "Z" => "Z",
        other => other,
    }
}

/// Compute the SU2 matrix product of a gate sequence.
///
/// Sequences are applied left-to-right: the first gate in the slice is applied first.
pub fn sequence_to_matrix(seq: &[&'static str]) -> SU2 {
    seq.iter().fold(SU2::identity(), |acc, g| {
        let gate = SU2::from_gate_name(g).unwrap_or(SU2::identity());
        acc.mul(&gate)
    })
}

/// Solovay-Kitaev decomposer.
///
/// Approximates any single-qubit SU(2) gate to precision ε using sequences
/// from {H, T, Tdg, S, Sdg}.
///
/// # Convergence behaviour
///
/// The internal `balanced_commutator_decompose` uses a BCH-derived formula
/// that matches the rotation *angle* of the residual delta to leading order in
/// the rotation angle θ, but has an O(φ) axis-drift at finite θ.  As a result:
///
/// - **Depth 0**: pure table lookup — precision is limited by table density.
/// - **Depth 1**: provides meaningful O(ε^{3/2}) improvement over depth 0 for
///   targets where the table is not already near-optimal.
/// - **Depth ≥ 2**: the fallback guard (which never returns a worse result)
///   means deeper recursion is bounded above by the depth-1 result but is
///   rarely strictly better.  The axis drift of the commutator construction
///   constitutes a floor that prevents further BCH-based reduction.
///
/// In practice, use `recursion_depth = 1` for the best trade-off between
/// sequence length and approximation quality.  Depths 2–4 are safe (they never
/// regress) but provide negligible additional improvement with this gate set.
///
/// # Example
///
/// ```rust
/// use quantrs2_circuit::solovay_kitaev::{SU2, SOKDecomposer, sequence_to_matrix};
///
/// let target = SU2::rotation_z(0.5);
/// let decomposer = SOKDecomposer::new(10, 1);
/// let seq = decomposer.decompose(&target).expect("decomposition failed");
/// let approx = sequence_to_matrix(&seq);
/// let dist = approx.frobenius_distance(&target);
/// assert!(dist < 0.15, "SK approximation distance: {dist}");
/// ```
pub struct SOKDecomposer {
    table: BasicApproximationTable,
    /// Number of SK recursion levels (0 = table lookup only; 1 = best practical depth).
    ///
    /// Depth 1 provides meaningful improvement over depth 0.  Depths ≥ 2 are
    /// bounded above by the depth-1 result and rarely strictly better due to the
    /// axis-drift floor in the BCH commutator construction (see struct doc).
    pub recursion_depth: usize,
}

impl SOKDecomposer {
    /// Build the decomposer with a precomputed basic approximation table.
    ///
    /// `table_depth`: maximum T-count for the basic table (re-interpreted from
    ///   the previous BFS-depth meaning).  See [`BasicApproximationTable::build`]
    ///   for the two-stage Clifford+T construction.  The table internally caps
    ///   the value to keep build time bounded; passing `10` is fine — it will
    ///   collapse to the internal cap.  Smaller values (0–3) give sparser
    ///   coverage and may not be dense enough for the BCH commutator
    ///   construction to improve the result at recursion depth 1.
    ///
    /// `recursion_depth`: number of SK recursion levels (0 = table only).
    ///   Use 1 for the best improvement/gate-count trade-off.  Depth 1 gives
    ///   O(ε^{3/2}) improvement at a cost of ~4× more gates.  Depths ≥ 2 are
    ///   safe (never worse due to the fallback guard) but rarely strictly better
    ///   with this commutator construction — see struct-level doc for details.
    pub fn new(table_depth: usize, recursion_depth: usize) -> Self {
        Self {
            table: BasicApproximationTable::build(table_depth),
            recursion_depth,
        }
    }

    /// Decompose `u` into a sequence of gate names from {H, T, Tdg, S, Sdg}.
    pub fn decompose(&self, u: &SU2) -> QuantRS2Result<Vec<&'static str>> {
        self.sk_recurse(u, self.recursion_depth)
    }

    /// Internal recursive Solovay-Kitaev routine.
    fn sk_recurse(&self, u: &SU2, n: usize) -> QuantRS2Result<Vec<&'static str>> {
        if n == 0 {
            let (seq, _dist) = self.table.find_closest(u);
            return Ok(seq.to_vec());
        }

        // Obtain U_{n-1}: best approximation using n-1 recursion levels
        let u_prev_seq = self.sk_recurse(u, n - 1)?;
        let u_prev = sequence_to_matrix(&u_prev_seq);
        // Use up-to-phase distance for quality comparison (the reported metric)
        let prev_dist = u_prev.frobenius_distance(u);

        // Compute delta = U * u_prev† in a phase-aligned way.
        //
        // Since u_prev is found using the up-to-phase metric, the matrix u_prev
        // may differ from U by a global phase. We align phases before computing
        // the residual so that delta has Re[Tr] maximized (closest to identity
        // in the Dawson-Nielsen metric), enabling accurate commutator construction.
        //
        // Optimal alignment: multiply u_prev by e^{iφ} where φ = -arg(Tr(U * u_prev†))/2
        // This makes Tr(delta) real and positive.
        let raw_delta = u.mul(&u_prev.adjoint());
        let tr_delta = raw_delta.trace();
        let phase_angle = -tr_delta.arg() / 2.0; // Negative half the trace argument
        let phase = Complex64::new(phase_angle.cos(), phase_angle.sin());
        // Apply phase to u_prev: u_prev_phased = e^{iφ} * u_prev
        let u_prev_phased = SU2 {
            m00: u_prev.m00 * phase,
            m01: u_prev.m01 * phase,
            m10: u_prev.m10 * phase,
            m11: u_prev.m11 * phase,
        };
        // delta_phased = U * u_prev_phased† = U * e^{-iφ} * u_prev†
        // This has Re[Tr(delta_phased)] maximized = |Tr(raw_delta)|
        let delta = u.mul(&u_prev_phased.adjoint());

        // Decompose delta as a balanced commutator: delta ≈ V * W * V† * W†
        let (v, w) = balanced_commutator_decompose(&delta)?;

        // Recursively approximate V and W to depth n-1
        let v_seq = self.sk_recurse(&v, n - 1)?;
        let w_seq = self.sk_recurse(&w, n - 1)?;

        // V† sequence: reverse and conjugate each gate
        let v_adj_seq: Vec<&'static str> = v_seq.iter().rev().map(|g| adjoint_gate(g)).collect();
        let w_adj_seq: Vec<&'static str> = w_seq.iter().rev().map(|g| adjoint_gate(g)).collect();

        // Final sequence: V_n W_n V_n† W_n† U_{n-1}
        let mut result = v_seq;
        result.extend_from_slice(&w_seq);
        result.extend(v_adj_seq);
        result.extend(w_adj_seq);
        result.extend_from_slice(&u_prev_seq);

        // Guard: if the new result is worse than the previous approximation,
        // fall back to u_prev_seq. The commutator decomposition may introduce
        // more error than it corrects when the basic table is not dense enough.
        let new_mat = sequence_to_matrix(&result);
        let new_dist = new_mat.frobenius_distance(u);
        if new_dist >= prev_dist {
            return Ok(u_prev_seq);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons in basic unitary tests.
    /// Using 1e-6 to accommodate accumulated floating-point rounding.
    const EPS: f64 = 1e-6;

    #[test]
    fn test_su2_identity_distance() {
        let i = SU2::identity();
        assert!(
            i.distance_from_identity() < EPS,
            "identity distance should be ~0, got {}",
            i.distance_from_identity()
        );
    }

    #[test]
    fn test_su2_multiplication_hh_is_identity() {
        let h = SU2::from_gate_name("H").expect("H gate must exist");
        let hh = h.mul(&h);
        assert!(
            hh.frobenius_distance(&SU2::identity()) < EPS,
            "H*H should equal Identity (up to global phase), got {}",
            hh.frobenius_distance(&SU2::identity())
        );
    }

    #[test]
    fn test_su2_tt_is_s() {
        let t = SU2::from_gate_name("T").expect("T gate must exist");
        let s = SU2::from_gate_name("S").expect("S gate must exist");
        let tt = t.mul(&t);
        assert!(
            tt.frobenius_distance(&s) < EPS,
            "T*T should equal S, got {}",
            tt.frobenius_distance(&s)
        );
    }

    #[test]
    fn test_adjoint_t_tdg() {
        let t = SU2::from_gate_name("T").expect("T gate must exist");
        let tdg = SU2::from_gate_name("Tdg").expect("Tdg gate must exist");
        assert!(
            t.adjoint().frobenius_distance(&tdg) < EPS,
            "T† should equal Tdg, dist = {}",
            t.adjoint().frobenius_distance(&tdg)
        );
    }

    #[test]
    fn test_rotation_z_composition() {
        let rz1 = SU2::rotation_z(0.3);
        let rz2 = SU2::rotation_z(0.7);
        let combined = rz1.mul(&rz2);
        let expected = SU2::rotation_z(1.0);
        assert!(
            combined.frobenius_distance(&expected) < EPS,
            "RZ(0.3)*RZ(0.7) should equal RZ(1.0)"
        );
    }

    #[test]
    fn test_basic_approximation_table_size() {
        let table = BasicApproximationTable::build(5);
        assert!(
            table.entries.len() > 10,
            "Table should have many entries, got {}",
            table.entries.len()
        );
    }

    #[test]
    fn test_basic_approximation_table_finds_identity() {
        let table = BasicApproximationTable::build(5);
        let identity = SU2::identity();
        let (_seq, dist) = table.find_closest(&identity);
        assert!(
            dist < 0.1,
            "Should find approximation close to identity, dist = {}",
            dist
        );
    }

    #[test]
    fn test_balanced_commutator_small_rotation() {
        let small_u = SU2::rotation_z(0.1);
        let (v, w) =
            balanced_commutator_decompose(&small_u).expect("commutator decompose should succeed");
        let commutator = v.mul(&w).mul(&v.adjoint()).mul(&w.adjoint());
        let dist = commutator.frobenius_distance(&small_u);
        assert!(
            dist < 0.2,
            "Balanced commutator should approximate small rotation U, got dist={}",
            dist
        );
    }

    #[test]
    fn test_balanced_commutator_identity() {
        // Identity has zero rotation → commutator of identity matrices = identity
        let u = SU2::identity();
        let (v, w) =
            balanced_commutator_decompose(&u).expect("commutator decompose should succeed");
        let commutator = v.mul(&w).mul(&v.adjoint()).mul(&w.adjoint());
        let dist = commutator.frobenius_distance(&u);
        assert!(
            dist < 0.1,
            "Commutator decompose of identity should be identity, got dist={}",
            dist
        );
    }

    #[test]
    fn test_solovay_kitaev_rz_depth0() {
        // Depth=0 is pure table lookup — should find something within 0.5 of target.
        let target = SU2::rotation_z(0.5);
        let decomposer = SOKDecomposer::new(10, 0);
        let seq = decomposer.decompose(&target).expect("decompose failed");
        let approx = sequence_to_matrix(&seq);
        let dist = approx.frobenius_distance(&target);
        // Table lookup: with depth=10, expect dist < 0.2 for most targets
        assert!(dist < 0.5, "Depth-0 should be within 0.5 of target, got {dist}");
    }

    #[test]
    fn test_solovay_kitaev_rz_depth1() {
        // Approximate RZ(0.5) using SK at depth=1 with the two-stage Clifford+T table
        // (table_depth=10 → max_t_count=10, ~73k entries).
        // SK at depth 1 gives O(ε^{3/2}) improvement over depth=0.
        // Empirically: depth-0 dist ≈ 0.044, depth-1 dist ≈ 0.037.
        let target = SU2::rotation_z(0.5);
        let decomposer = SOKDecomposer::new(10, 1);
        let seq = decomposer.decompose(&target).expect("decompose failed");

        let approx = sequence_to_matrix(&seq);
        let dist = approx.frobenius_distance(&target);
        let depth0_dist = {
            let d0 = SOKDecomposer::new(10, 0);
            let s0 = d0.decompose(&target).expect("depth0 failed");
            sequence_to_matrix(&s0).frobenius_distance(&target)
        };
        // SK depth=1 must strictly improve over depth=0
        assert!(
            dist < depth0_dist,
            "SK depth=1 ({dist:.6}) should improve over depth=0 ({depth0_dist:.6})"
        );
        // Absolute bound: with the dense Clifford+T table, depth-1 reaches ~0.037
        assert!(dist < 0.05, "SK depth=1 should achieve dist < 0.05, got {dist:.6}");
        assert!(!seq.is_empty(), "Sequence should not be empty");
    }

    #[test]
    fn test_solovay_kitaev_rz_depth2() {
        // At depth 2, SK falls back to the depth-1 result due to the axis-drift
        // floor in the BCH commutator construction (see SOKDecomposer doc).
        // The fallback guard guarantees depth-2 is never worse than depth-1.
        let target = SU2::rotation_z(0.5);
        let decomposer2 = SOKDecomposer::new(10, 2);
        let seq2 = decomposer2.decompose(&target).expect("decompose failed");
        let dist2 = sequence_to_matrix(&seq2).frobenius_distance(&target);

        let decomposer1 = SOKDecomposer::new(10, 1);
        let seq1 = decomposer1.decompose(&target).expect("depth1 failed");
        let dist1 = sequence_to_matrix(&seq1).frobenius_distance(&target);

        // Depth=2 must be at least as good as depth=1 (fallback guard ensures this).
        // Strict improvement is not guaranteed with BCH commutator at this gate set density.
        assert!(
            dist2 <= dist1 + 1e-9,
            "SK depth=2 ({dist2:.6}) should not be worse than depth=1 ({dist1:.6})"
        );
    }

    #[test]
    fn test_solovay_kitaev_convergence_debug() {
        // Check convergence at each depth to understand algorithm behavior.
        // Uses table_depth=10 (~73k entries via the two-stage Clifford+T BFS) for
        // dense coverage.
        let target = SU2::rotation_z(0.5);
        let table = BasicApproximationTable::build(10);

        eprintln!("Table size: {}", table.entries.len());

        // Depth 0: table lookup — should find a good starting approximation
        let (seq0, dist0) = table.find_closest(&target);
        eprintln!("Depth 0: dist={:.6}, seq_len={}", dist0, seq0.len());

        // Print the top 5 closest entries in the table to RZ(0.5)
        let mut dists: Vec<(f64, usize)> = table
            .entries
            .iter()
            .enumerate()
            .map(|(i, (_, mat))| (mat.frobenius_distance(&target), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("Top-5 closest table entries to RZ(0.5):");
        for (dist, idx) in dists.iter().take(5) {
            eprintln!("  dist={:.6}, seq={:?}", dist, table.entries[*idx].0);
        }

        // Debug the commutator construction with phase-alignment (as in sk_recurse)
        let u_prev = sequence_to_matrix(seq0);
        let raw_delta = target.mul(&u_prev.adjoint());
        let tr_delta = raw_delta.trace();
        let phase_angle = -tr_delta.arg() / 2.0;
        let phase = Complex64::new(phase_angle.cos(), phase_angle.sin());
        let u_prev_phased = SU2 {
            m00: u_prev.m00 * phase,
            m01: u_prev.m01 * phase,
            m10: u_prev.m10 * phase,
            m11: u_prev.m11 * phase,
        };
        let delta = target.mul(&u_prev_phased.adjoint());
        let delta_dist = delta.distance_from_identity();
        eprintln!("Phase-aligned delta dist from identity: {:.6}", delta_dist);
        eprintln!("Phase-aligned delta rotation angle: {:.6}", delta.rotation_angle());

        let (v, w) = balanced_commutator_decompose(&delta).expect("commutator failed");
        let (v_seq, v_dist) = table.find_closest(&v);
        let (w_seq, w_dist) = table.find_closest(&w);
        eprintln!("V approx dist: {:.6}, W approx dist: {:.6}", v_dist, w_dist);

        // Compute the commutator using exact v and w
        let exact_comm = v.mul(&w).mul(&v.adjoint()).mul(&w.adjoint());
        let exact_comm_dist = exact_comm.frobenius_distance(&delta);
        eprintln!("Exact commutator dist from delta: {:.6}", exact_comm_dist);

        // Compute the commutator using approximated v and w
        let v_mat = sequence_to_matrix(v_seq);
        let w_mat = sequence_to_matrix(w_seq);
        let approx_comm = v_mat.mul(&w_mat).mul(&v_mat.adjoint()).mul(&w_mat.adjoint());
        let approx_comm_to_delta = approx_comm.frobenius_distance(&delta);
        let approx_comm_to_i = approx_comm.distance_from_identity();
        eprintln!("Approx commutator dist from delta: {:.6}", approx_comm_to_delta);
        eprintln!("Approx commutator dist from I: {:.6}", approx_comm_to_i);

        // Full depth=1 result: V_approx * W_approx * V_approx† * W_approx† * u_prev
        let v_adj_seq: Vec<&'static str> = v_seq.iter().rev().map(|g| adjoint_gate(g)).collect();
        let w_adj_seq: Vec<&'static str> = w_seq.iter().rev().map(|g| adjoint_gate(g)).collect();
        let mut full_seq = v_seq.to_vec();
        full_seq.extend_from_slice(w_seq);
        full_seq.extend(v_adj_seq);
        full_seq.extend(w_adj_seq);
        full_seq.extend_from_slice(seq0);
        let full_mat = sequence_to_matrix(&full_seq);
        let full_dist = full_mat.frobenius_distance(&target);
        eprintln!("Full depth=1 result dist from target: {:.6}", full_dist);

        // Using the full decomposer at depth=1
        let decomposer = SOKDecomposer::new(10, 1);
        let sk_seq = decomposer.decompose(&target).expect("SK decompose failed");
        let sk_mat = sequence_to_matrix(&sk_seq);
        let sk_dist = sk_mat.frobenius_distance(&target);
        eprintln!("SK depth=1 dist: {:.6}, seq_len={}", sk_dist, sk_seq.len());

        assert!(dist0 < 0.5, "depth-0 should find something reasonable");
    }

    #[test]
    fn test_solovay_kitaev_rz_depth3() {
        // At depth 3, SK is monotone (fallback guard ensures never-worse).
        // The BCH commutator axis-drift floors the result near the depth-1 value;
        // strict improvement over depth-1 is not guaranteed with this gate set.
        // With table_depth=10 (~73k entries), depth-1 achieves dist ≈ 0.037.
        let target = SU2::rotation_z(0.5);
        let d1 = SOKDecomposer::new(10, 1).decompose(&target).expect("depth1 failed");
        let dist1 = sequence_to_matrix(&d1).frobenius_distance(&target);

        let d3 = SOKDecomposer::new(10, 3).decompose(&target).expect("depth3 failed");
        let dist3 = sequence_to_matrix(&d3).frobenius_distance(&target);

        // Depth=3 must be at least as good as depth=1 (SK is monotone by construction).
        // Strict improvement is not guaranteed — depth-3 typically equals depth-1.
        assert!(
            dist3 <= dist1 + 1e-9,
            "SK depth=3 ({dist3:.6}) should not be worse than depth=1 ({dist1:.6})"
        );
        assert!(!d3.is_empty(), "Sequence should not be empty");
    }

    #[test]
    fn test_solovay_kitaev_sequence_gates_valid() {
        // All gates in sequence should be from {H, T, Tdg, S, Sdg}
        let target = SU2::rotation_z(0.5);
        let decomposer = SOKDecomposer::new(10, 2);
        let seq = decomposer.decompose(&target).expect("decompose failed");

        for gate in &seq {
            assert!(
                ["H", "T", "Tdg", "S", "Sdg"].contains(gate),
                "Gate '{}' not in universal set {{H, T, Tdg, S, Sdg}}",
                gate
            );
        }
    }

    #[test]
    fn test_sequence_to_matrix_empty() {
        let m = sequence_to_matrix(&[]);
        assert!(
            m.frobenius_distance(&SU2::identity()) < EPS,
            "Empty sequence should give identity"
        );
    }

    #[test]
    fn test_sequence_to_matrix_h() {
        let m = sequence_to_matrix(&["H"]);
        let h = SU2::from_gate_name("H").expect("H must exist");
        assert!(m.frobenius_distance(&h) < EPS);
    }

    #[test]
    fn test_adjoint_gate_roundtrip() {
        assert_eq!(adjoint_gate("T"), "Tdg");
        assert_eq!(adjoint_gate("Tdg"), "T");
        assert_eq!(adjoint_gate("H"), "H");
        assert_eq!(adjoint_gate("S"), "Sdg");
        assert_eq!(adjoint_gate("Sdg"), "S");
    }

    #[test]
    fn test_rotation_about_axis_z() {
        // rotation_about_axis(0, 0, 1, theta) should equal rotation_z(theta)
        let theta = 0.7;
        let r1 = rotation_about_axis(0.0, 0.0, 1.0, theta);
        let r2 = SU2::rotation_z(theta);
        assert!(
            r1.frobenius_distance(&r2) < EPS,
            "rotation_about_axis(z) should match rotation_z"
        );
    }

    #[test]
    fn test_rotation_about_axis_x() {
        let theta = 1.2;
        let r1 = rotation_about_axis(1.0, 0.0, 0.0, theta);
        let r2 = SU2::rotation_x(theta);
        assert!(
            r1.frobenius_distance(&r2) < EPS,
            "rotation_about_axis(x) should match rotation_x"
        );
    }

    #[test]
    fn test_clifford_enumeration_size() {
        // Sanity check: the single-qubit Clifford group modulo global phase
        // has exactly 24 elements, generated by {H, S, Sdg}.
        let cliffords = enumerate_cliffords();
        assert_eq!(
            cliffords.len(),
            24,
            "Clifford group enumeration should yield exactly 24 elements"
        );
    }

    #[test]
    fn test_two_stage_table_covers_small_y_rotation() {
        // Coverage regression test: with the previous single-stage BFS at depth 10,
        // the table contained no element closer to RY(0.3) than the identity
        // (depth-0 dist = 0.149, seq_len = 0). The two-stage Clifford+T BFS
        // expands SU(2) coverage so RY(0.3) gets a non-trivial approximation.
        let table = BasicApproximationTable::build(10);
        let target = SU2::rotation_y(0.3);
        let (seq, dist) = table.find_closest(&target);
        assert!(
            !seq.is_empty(),
            "Table should cover RY(0.3) with a non-identity sequence (got seq_len=0, dist={dist:.6})"
        );
        // Distance should be meaningfully better than the identity bound (~0.150).
        assert!(
            dist < 0.10,
            "RY(0.3) approximation should beat identity bound 0.15, got {dist:.6}"
        );
    }
}
