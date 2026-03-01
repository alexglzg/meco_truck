#pragma once
// ============================================================================
// sdmn2d.hpp — Small-Dimensional Minimum-Norm solver for 2D
//
// Reference: Wang et al., "Fast Iterative Region Inflation for Computing
//            Large 2-D/3-D Convex Regions of Obstacle-Free Space"
//            IEEE Trans. Robotics, Vol. 41, 2025
//            Section IV, Algorithm 2
//
// Solves:  min ||y||^2   s.t.  e_i^T y <= f_i,  i = 1..d
//
// Expected O(d) complexity for n=2 via randomized incremental construction.
// No ROS dependency — pure Eigen + STL.
// ============================================================================

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace firi {

class SDMN2D {
public:
    struct Result {
        Eigen::Vector2d y;      ///< Optimal point (minimum-norm solution)
        bool feasible;          ///< True if the constraint set is feasible
    };

    /// Construct with a fixed random seed for reproducibility.
    /// The randomized pivot order gives expected O(d) but the seed
    /// ensures identical results across runs (useful for debugging).
    explicit SDMN2D(unsigned int seed = 42) : rng_(seed) {}

    /// Solve min ||y||^2  s.t.  e[i]^T y <= f[i]  for all i.
    ///
    /// @param e  Constraint normal vectors (one per halfplane)
    /// @param f  Constraint offsets (one per halfplane)
    /// @return   Result with optimal y and feasibility flag
    Result solve(const std::vector<Eigen::Vector2d>& e,
                 const std::vector<double>& f)
    {
        const size_t d = e.size();
        if (d == 0) return {Eigen::Vector2d::Zero(), true};

        // Random permutation of constraint indices.
        // This is what gives expected O(d) complexity — worst-case
        // constraint orderings become exponentially unlikely.
        std::vector<size_t> perm(d);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        // Start at the unconstrained optimum (origin)
        Eigen::Vector2d y = Eigen::Vector2d::Zero();

        for (size_t ii = 0; ii < d; ++ii) {
            const size_t idx = perm[ii];

            // Check if current y already satisfies constraint idx
            if (e[idx].dot(y) <= f[idx] + 1e-12) continue;

            // ── Constraint idx is violated: project onto its boundary ──
            //
            // The new optimum must lie on the hyperplane e[idx]^T y = f[idx].
            // We parameterize this as y = v + t * m, where:
            //   v = projection of origin onto the hyperplane
            //   m = direction along the hyperplane (null space of e[idx])

            const Eigen::Vector2d& eh = e[idx];
            const double fh = f[idx];
            const double eTe = eh.squaredNorm();
            if (eTe < 1e-15) return {Eigen::Vector2d::Zero(), false};

            // v = closest point on hyperplane to origin
            const Eigen::Vector2d v = (fh / eTe) * eh;

            // ── Householder reflection to find m ──
            // m is a unit vector in the null space of eh (perpendicular to eh
            // but lying in the hyperplane). For 2D this is just rotating eh
            // by 90°, but the Householder approach generalizes to higher dims.
            //
            // We pick the Householder pivot axis j as the component of v with
            // largest magnitude for numerical stability.
            const int j = (std::abs(v(0)) >= std::abs(v(1))) ? 0 : 1;
            const int k = 1 - j;

            Eigen::Vector2d m_col;
            const double v_norm = v.norm();

            if (v_norm < 1e-15) {
                // v ≈ 0: hyperplane passes through origin, just rotate eh by 90°
                m_col = Eigen::Vector2d(-eh(1), eh(0));
                const double mn = m_col.norm();
                if (mn > 1e-15) m_col /= mn;
                else return {Eigen::Vector2d::Zero(), false};
            } else {
                const double sign_vj = (v(j) >= 0) ? 1.0 : -1.0;
                const Eigen::Vector2d u_ref = v + sign_vj * v_norm * Eigen::Vector2d::Unit(j);
                const double uTu = u_ref.squaredNorm();
                if (uTu < 1e-15) {
                    m_col = Eigen::Vector2d(-eh(1), eh(0)).normalized();
                } else {
                    // Householder reflection: m = e_k - 2(u_k/||u||^2) * u
                    m_col = Eigen::Vector2d::Unit(k) - (2.0 * u_ref(k) / uTu) * u_ref;
                }
            }

            // ── 1D subproblem: find t in [lo, hi] minimizing ||v + t*m||^2 ──
            // Subject to all previously-processed constraints (indices perm[0..ii-1]).
            double lo = -1e18, hi = 1e18;
            bool feasible = true;

            for (size_t pp = 0; pp < ii; ++pp) {
                const size_t pidx = perm[pp];
                const double a1d = e[pidx].dot(m_col);
                const double b1d = f[pidx] - e[pidx].dot(v);

                if (std::abs(a1d) < 1e-15) {
                    // Constraint is parallel to search direction
                    if (b1d < -1e-10) { feasible = false; break; }
                    continue;
                }

                const double bound = b1d / a1d;
                if (a1d > 0) hi = std::min(hi, bound);
                else         lo = std::max(lo, bound);
            }

            if (!feasible || lo > hi + 1e-10)
                return {Eigen::Vector2d::Zero(), false};

            // Optimal t: project onto [lo, hi], closest to 0
            double t;
            if (lo <= 0.0 && 0.0 <= hi) t = 0.0;
            else if (lo > 0.0)           t = lo;
            else                          t = hi;

            y = m_col * t + v;
        }

        return {y, true};
    }

private:
    mutable std::mt19937 rng_;
};

}  // namespace firi