//! Minimum-weight perfect matching via bitmask dynamic programming.
//!
//! For n defects (n must be even), this computes the minimum-weight perfect matching
//! using the recurrence:
//!
//! ```text
//! dp[mask] = min over v of (dist(u, v) + dp[mask ^ (1<<u) ^ (1<<v)])
//! ```
//!
//! where `u` is the lowest set bit of `mask`.
//!
//! Time complexity: O(n² · 2^n). Adequate for d ≤ 7 surface codes (≤ 24 typical defects).

use crate::error::{QuantRS2Error, QuantRS2Result};

/// Compute a minimum-weight perfect matching on `n` labeled vertices using bitmask DP.
///
/// # Arguments
///
/// * `n` - Number of vertices (must be even for a perfect matching to exist).
/// * `edges` - Slice of `(u, v, weight)` edge descriptors. Missing edges are treated
///   as having weight `f64::INFINITY`. Parallel edges: only the minimum weight is kept.
///
/// # Returns
///
/// `Ok(pairs)` where `pairs` is a vector of `(u, v)` matched pairs with `u < v`,
/// or `None` if `n` is odd (no perfect matching is possible).
///
/// Returns `Err` if `n > 24` (bitmask DP table would exceed 16 million entries)
/// or if any vertex index in `edges` is out of range `[0, n)`.
pub fn min_weight_perfect_matching(
    n: usize,
    edges: &[(usize, usize, f64)],
) -> QuantRS2Result<Option<Vec<(usize, usize)>>> {
    if n == 0 {
        return Ok(Some(Vec::new()));
    }
    if n % 2 != 0 {
        return Ok(None);
    }
    if n > 24 {
        return Err(QuantRS2Error::InvalidInput(
            "Too many defects for bitmask-DP matching: maximum supported is 24".to_string(),
        ));
    }

    // Validate edge vertex indices
    for &(u, v, _) in edges {
        if u >= n || v >= n {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Edge ({u}, {v}) references out-of-range vertex for n={n}"
            )));
        }
    }

    // Build dense distance matrix — start with infinity everywhere
    let mut dist = vec![f64::INFINITY; n * n];
    // Self-loops are zero
    for i in 0..n {
        dist[i * n + i] = 0.0;
    }
    // Fill in edge weights (keep minimum for parallel edges)
    for &(u, v, w) in edges {
        if w < dist[u * n + v] {
            dist[u * n + v] = w;
            dist[v * n + u] = w;
        }
    }

    let full_mask = (1usize << n) - 1;

    // dp[mask] = minimum weight perfect matching on vertices in `mask`
    let mut dp = vec![f64::INFINITY; 1 << n];
    // parent[mask] = (u, v) pair chosen for this mask
    let mut parent: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); 1 << n];
    dp[0] = 0.0;

    // Iterate over all even-popcount masks in order of increasing popcount
    for mask in 1..=full_mask {
        let popcount = mask.count_ones() as usize;
        if popcount % 2 != 0 {
            continue;
        }

        // u = lowest set bit index
        let u = mask.trailing_zeros() as usize;

        // Try pairing u with every other set bit v
        let remaining = mask ^ (1 << u);
        let mut iter_mask = remaining;
        while iter_mask != 0 {
            let v = iter_mask.trailing_zeros() as usize;
            iter_mask &= iter_mask - 1; // clear lowest set bit

            let sub_mask = mask ^ (1 << u) ^ (1 << v);
            let d_uv = dist[u * n + v];
            if d_uv.is_finite() {
                let candidate = d_uv + dp[sub_mask];
                if candidate < dp[mask] {
                    dp[mask] = candidate;
                    parent[mask] = (u, v);
                }
            }
        }
    }

    if dp[full_mask].is_infinite() {
        // No perfect matching exists with given edges
        return Ok(None);
    }

    // Reconstruct pairs from parent table
    let mut pairs = Vec::with_capacity(n / 2);
    let mut cur_mask = full_mask;
    while cur_mask != 0 {
        let (u, v) = parent[cur_mask];
        if u == usize::MAX {
            break;
        }
        pairs.push((u.min(v), u.max(v)));
        cur_mask ^= (1 << u) ^ (1 << v);
    }
    pairs.sort_unstable();

    Ok(Some(pairs))
}

/// Compute the total weight of a matching given an edge weight function.
pub fn matching_weight(pairs: &[(usize, usize)], n: usize, edges: &[(usize, usize, f64)]) -> f64 {
    let mut dist = vec![f64::INFINITY; n * n];
    for i in 0..n {
        dist[i * n + i] = 0.0;
    }
    for &(u, v, w) in edges {
        if w < dist[u * n + v] {
            dist[u * n + v] = w;
            dist[v * n + u] = w;
        }
    }
    pairs.iter().map(|&(u, v)| dist[u * n + v]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let result = min_weight_perfect_matching(0, &[]).expect("should succeed");
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_odd_n_returns_none() {
        let result = min_weight_perfect_matching(3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)])
            .expect("should succeed (not error)");
        assert_eq!(result, None, "Odd n must return None");
    }

    #[test]
    fn test_triangle_3v_picks_minimum() {
        // n=2 with edges weight 1.0 and 2.0 — only one possible pair (0,1)
        let result = min_weight_perfect_matching(2, &[(0, 1, 1.0)])
            .expect("should succeed")
            .expect("should have a matching");
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_n2_single_edge() {
        let edges = vec![(0, 1, 5.0)];
        let result = min_weight_perfect_matching(2, &edges)
            .expect("should succeed")
            .expect("matching exists");
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_n4_optimal() {
        // 4 vertices with edges: 0-1: 1, 0-2: 10, 0-3: 10, 1-2: 10, 1-3: 10, 2-3: 1
        // Optimal matching: (0,1) + (2,3) = cost 2, vs (0,2)+(1,3) = 20 or (0,3)+(1,2) = 20
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 10.0),
            (0, 3, 10.0),
            (1, 2, 10.0),
            (1, 3, 10.0),
            (2, 3, 1.0),
        ];
        let result = min_weight_perfect_matching(4, &edges)
            .expect("should succeed")
            .expect("matching exists");
        assert_eq!(result.len(), 2);
        let total_weight: f64 = result
            .iter()
            .map(|&(u, v)| {
                edges
                    .iter()
                    .find(|&&(a, b, _)| (a == u && b == v) || (a == v && b == u))
                    .map(|&(_, _, w)| w)
                    .unwrap_or(f64::INFINITY)
            })
            .sum();
        assert!(
            (total_weight - 2.0).abs() < 1e-9,
            "Optimal cost should be 2.0, got {total_weight}"
        );
    }

    #[test]
    fn test_determinism() {
        let edges = vec![
            (0, 1, 3.0),
            (0, 2, 1.0),
            (0, 3, 5.0),
            (1, 2, 4.0),
            (1, 3, 2.0),
            (2, 3, 6.0),
        ];
        let r1 = min_weight_perfect_matching(4, &edges)
            .expect("ok")
            .expect("match");
        let r2 = min_weight_perfect_matching(4, &edges)
            .expect("ok")
            .expect("match");
        assert_eq!(r1, r2, "Same input must produce same output");
    }

    #[test]
    fn test_n6_brute_force() {
        // n=6 complete graph — compare dp result to brute-force enumeration of all 15 matchings
        let weights = [
            [0.0, 1.0, 7.0, 3.0, 5.0, 2.0],
            [1.0, 0.0, 4.0, 8.0, 6.0, 9.0],
            [7.0, 4.0, 0.0, 2.0, 1.0, 3.0],
            [3.0, 8.0, 2.0, 0.0, 7.0, 5.0],
            [5.0, 6.0, 1.0, 7.0, 0.0, 4.0],
            [2.0, 9.0, 3.0, 5.0, 4.0, 0.0],
        ];
        let n = 6;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                edges.push((i, j, weights[i][j]));
            }
        }

        let dp_result = min_weight_perfect_matching(n, &edges)
            .expect("should succeed")
            .expect("matching exists");
        let dp_cost: f64 = dp_result.iter().map(|&(u, v)| weights[u][v]).sum();

        // Brute-force: enumerate all 15 perfect matchings of 6 vertices
        // A perfect matching of {0,1,2,3,4,5}: fix vertex 0, pair with each of 1..5,
        // then recurse on remaining 4 vertices.
        let bf_cost = brute_force_min_matching(n, &weights);
        assert!(
            (dp_cost - bf_cost).abs() < 1e-9,
            "DP cost {dp_cost} != brute-force cost {bf_cost}"
        );
    }

    #[test]
    fn test_too_large_returns_error() {
        let result = min_weight_perfect_matching(26, &[]);
        assert!(result.is_err(), "n=26 should return Err");
    }

    #[test]
    fn test_no_matching_disconnected() {
        // n=4 with only one edge 0-1: vertices 2,3 are isolated → no perfect matching
        let edges = vec![(0, 1, 1.0)];
        let result = min_weight_perfect_matching(4, &edges).expect("should succeed");
        assert_eq!(
            result, None,
            "Disconnected graph should have no perfect matching"
        );
    }

    fn brute_force_min_matching(n: usize, weights: &[[f64; 6]]) -> f64 {
        fn recurse(remaining: &[usize], weights: &[[f64; 6]]) -> f64 {
            if remaining.is_empty() {
                return 0.0;
            }
            let u = remaining[0];
            let rest: Vec<usize> = remaining[1..].to_vec();
            let mut best = f64::INFINITY;
            for i in 0..rest.len() {
                let v = rest[i];
                let mut next: Vec<usize> = rest.clone();
                next.remove(i);
                let cost = weights[u][v] + recurse(&next, weights);
                if cost < best {
                    best = cost;
                }
            }
            best
        }
        let all: Vec<usize> = (0..n).collect();
        recurse(&all, weights)
    }
}
