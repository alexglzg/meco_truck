#pragma once
// ============================================================================
// mvie2d.hpp — Maximum Volume Inscribed Ellipsoid for 2D polytopes
//
// Reference: Wang et al., "Fast Iterative Region Inflation..."
//            Section V concept, simplified analytical approach for 2D.
//
// Finds the largest-area ellipsoid {L*u + d : ||u|| <= 1} contained in
// the polytope {x : A*x <= b}.
//
// Method: log-barrier interior-point with 5 decision variables:
//   x = [L11, L21, L22, d1, d2]
//   where L is lower-triangular (Cholesky factor of the ellipsoid shape).
//
// The ellipsoid volume is proportional to det(L) = L11 * L22.
// We maximize log(det(L)) subject to ||A_i * L||_2 + A_i * d <= b_i.
//
// No ROS dependency — pure Eigen.
// ============================================================================

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace firi {

class MVIE2D {
public:
    struct Ellipsoid {
        Eigen::Matrix2d L;      ///< Lower-triangular Cholesky factor
        Eigen::Vector2d d;      ///< Center
        double volume() const {
            return M_PI * std::abs(L(0, 0) * L(1, 1));
        }
    };

    /// Compute the MVIE for polytope {x : A*x <= b}.
    ///
    /// @param A             Halfplane normals (m × 2), one row per constraint
    /// @param b             Halfplane offsets (m × 1)
    /// @param center_hint   Initial guess for the ellipsoid center (e.g. robot position
    ///                      or centroid of the seed polytope). Doesn't need to be exact,
    ///                      but must be strictly inside the polytope.
    /// @return              Ellipsoid with shape L and center d
    Ellipsoid solve(const Eigen::MatrixXd& A,
                    const Eigen::VectorXd& b,
                    const Eigen::Vector2d& center_hint)
    {
        const int m = A.rows();

        // ── Initialize: largest isotropic ellipsoid centered at hint ──
        // Find the tightest constraint (smallest slack) to set initial radius.
        Eigen::Vector2d c = center_hint;
        double r = 1e18;
        for (int i = 0; i < m; ++i) {
            const double norm_ai = A.row(i).norm();
            if (norm_ai > 1e-10) {
                const double gap = b(i) - A.row(i).dot(c);
                r = std::min(r, gap / norm_ai);
            }
        }
        if (r <= 0) r = 1e-4;       // hint is outside or on boundary — use tiny radius
        r *= 0.9;                     // shrink slightly to stay strictly interior
        r = std::max(r, 1e-6);

        // Decision vector: [L11, L21, L22, d1, d2]
        // Start with isotropic (circular) ellipsoid
        Eigen::VectorXd x(5);
        x << r, 0.0, r, c(0), c(1);

        // ── Barrier method: outer iterations increase barrier weight t ──
        // At each t, we (approximately) minimize:
        //   -t * log(det(L)) + sum_i -log(b_i - a_i^T d - ||L^T a_i||)
        //
        // As t → ∞, the barrier terms vanish and we approach the true MVIE.
        double t = 1.0;
        const double mu = 4.0;      // barrier parameter growth rate

        for (int outer = 0; outer < 20; ++outer) {
            for (int inner = 0; inner < 40; ++inner) {
                // Newton step: solve H * dx = -g
                Eigen::VectorXd grad = gradient(A, b, x, t);
                Eigen::MatrixXd H = hessian(A, b, x, t);
                H += 1e-8 * Eigen::MatrixXd::Identity(5, 5);  // regularization

                Eigen::VectorXd dx = H.ldlt().solve(-grad);

                // Newton decrement: lambda^2 = -g^T dx
                // When lambda^2 < threshold, we're near the optimum for this t.
                const double lambda_sq = -grad.dot(dx);
                if (lambda_sq < 1e-6) break;

                // ── Backtracking line search ──
                // Ensure: (1) L11 > 0, L22 > 0  (ellipsoid stays valid)
                //         (2) sufficient decrease in objective
                double alpha = 1.0;
                const double f0 = objective(A, b, x, t);

                for (int ls = 0; ls < 32; ++ls) {
                    Eigen::VectorXd xn = x + alpha * dx;
                    if (xn(0) > 1e-10 && xn(2) > 1e-10) {
                        const double fn = objective(A, b, xn, t);
                        if (std::isfinite(fn) && fn < f0 + 0.3 * alpha * grad.dot(dx)) {
                            x = xn;
                            break;
                        }
                    }
                    alpha *= 0.5;
                    if (alpha < 1e-12) break;
                }
            }

            // Check duality gap: m/t < tolerance → converged
            if (static_cast<double>(m) / t < 1e-3) break;
            t *= mu;
        }

        // ── Extract result ──
        Ellipsoid E;
        E.L << x(0), 0.0,
               x(1), x(2);
        E.d << x(3), x(4);
        return E;
    }

private:
    // ── Barrier objective ──
    // f(x) = -t * log(det(L)) + sum_i -log(slack_i)
    // where slack_i = b_i - a_i^T d - ||L^T a_i||_2
    double objective(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                     const Eigen::VectorXd& x, double t)
    {
        const double L11 = x(0), L21 = x(1), L22 = x(2);
        const double d1 = x(3), d2 = x(4);

        if (L11 <= 0 || L22 <= 0) return 1e18;

        // -t * log(det(L)) = -t * (log(L11) + log(L22))
        double val = -t * (std::log(L11) + std::log(L22));

        for (int i = 0; i < A.rows(); ++i) {
            const double a1 = A(i, 0), a2 = A(i, 1);

            // L^T * a_i, where L is lower-triangular:
            //   L^T = [L11, L21]    a = [a1]
            //          [0,   L22]        [a2]
            const double r1 = L11 * a1 + L21 * a2;
            const double r2 = L22 * a2;

            // slack = b_i - a_i^T d - ||L^T a_i||
            const double gap = b(i) - a1 * d1 - a2 * d2
                             - std::sqrt(r1 * r1 + r2 * r2);
            if (gap <= 0) return 1e18;

            val -= std::log(gap);
        }
        return val;
    }

    // ── Analytical gradient (5-vector) ──
    Eigen::VectorXd gradient(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                             const Eigen::VectorXd& x, double t)
    {
        const double L11 = x(0), L21 = x(1), L22 = x(2);
        const double d1 = x(3), d2 = x(4);

        Eigen::VectorXd g = Eigen::VectorXd::Zero(5);

        // Gradient of -t * log(det(L))
        g(0) = -t / L11;
        g(2) = -t / L22;

        for (int i = 0; i < A.rows(); ++i) {
            const double a1 = A(i, 0), a2 = A(i, 1);
            const double r1 = L11 * a1 + L21 * a2;
            const double r2 = L22 * a2;
            const double nr = std::sqrt(r1 * r1 + r2 * r2);

            double gap = b(i) - a1 * d1 - a2 * d2 - nr;
            if (gap < 1e-15) gap = 1e-15;
            const double ig = 1.0 / gap;

            // Chain rule through ||L^T a||: d/dL_jk = (L^T a)_j * a_k / ||L^T a||
            if (nr > 1e-15) {
                const double inr = 1.0 / nr;
                g(0) += ig * r1 * a1 * inr;    // d/dL11
                g(1) += ig * r1 * a2 * inr;    // d/dL21
                g(2) += ig * r2 * a2 * inr;    // d/dL22
            }
            g(3) += ig * a1;                    // d/dd1
            g(4) += ig * a2;                    // d/dd2
        }
        return g;
    }

    // ── Hessian via finite differences (5×5) ──
    // Analytical Hessian is possible but messy for 5 variables.
    // At 5×5, finite-diff cost is negligible compared to the
    // O(m) gradient evaluations, so this is fine for real-time.
    Eigen::MatrixXd hessian(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& x, double t)
    {
        const double eps = 1e-6;
        Eigen::MatrixXd H(5, 5);

        for (int j = 0; j < 5; ++j) {
            Eigen::VectorXd xp = x, xm = x;
            xp(j) += eps;
            xm(j) -= eps;
            // Central difference: H[:,j] ≈ (g(x+eps*e_j) - g(x-eps*e_j)) / (2*eps)
            H.col(j) = (gradient(A, b, xp, t) - gradient(A, b, xm, t)) / (2.0 * eps);
        }

        // Symmetrize to remove numerical asymmetry from finite differences
        return 0.5 * (H + H.transpose());
    }
};

}  // namespace firi