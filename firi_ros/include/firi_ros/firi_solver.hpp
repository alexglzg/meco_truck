#pragma once
// ============================================================================
// firi_solver.hpp — Fast Iterative Region Inflation (2D)
//
// Reference: Wang et al., "Fast Iterative Region Inflation for Computing
//            Large 2-D/3-D Convex Regions of Obstacle-Free Space"
//            IEEE Trans. Robotics, Vol. 41, 2025
//            Algorithm 1
//
// Given:  - a set of 2D obstacle points
//         - a seed polytope (robot footprint) that must be contained
//         - a bounding box (heading-aligned search region)
// Finds:  the largest convex obstacle-free polytope containing the seed.
//
// The solver iterates:
//   1. RsI (Region-seed Inflation): for each obstacle, find the tightest
//      separating halfplane that still contains the seed. This uses SDMN
//      to solve a minimum-norm QP per obstacle, then greedily selects
//      halfplanes that cover the most obstacles.
//   2. MVIE: find the maximum-volume inscribed ellipsoid in the current
//      polytope. Its shape matrix L rescales the space for the next RsI
//      iteration, so the solver preferentially expands in directions with
//      more room.
//   3. Repeat until volume growth < rho (convergence threshold).
//
// No ROS dependency — pure Eigen + STL.
// ============================================================================

#include "firi_ros/sdmn2d.hpp"
#include "firi_ros/mvie2d.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace firi {

// ── Output type: a single halfplane n^T x <= b ──
struct HalfPlane {
    Eigen::Vector2d normal;     ///< Outward normal (unit length after normalization)
    double offset;              ///< Scalar b in n^T x <= b
};

class FIRISolver {
public:
    struct Result {
        std::vector<HalfPlane> planes;  ///< Final polytope (bbox + obstacle halfplanes)
        int iterations;                  ///< Number of FIRI outer iterations
        double solve_time_ms;            ///< Wall-clock time
    };

    FIRISolver() = default;

    /// Run FIRI.
    ///
    /// @param obstacles      2D obstacle points (boundary-extracted from grid)
    /// @param seed_vertices  Convex hull of robot footprint in world frame
    /// @param bbox_planes    Heading-aligned bounding box as halfplanes
    /// @param max_iter       Maximum outer iterations (typically 5–10)
    /// @param rho            Convergence threshold on relative volume growth
    /// @return               Result with final halfplanes, iteration count, timing
    Result compute(const std::vector<Eigen::Vector2d>& obstacles,
                   const std::vector<Eigen::Vector2d>& seed_vertices,
                   const std::vector<HalfPlane>& bbox_planes,
                   int max_iter = 10,
                   double rho = 0.02)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        if (obstacles.empty() || seed_vertices.empty()) {
            return {bbox_planes, 0, 0.0};
        }

        // ── Initialize ellipsoid from seed polytope ──
        // Center: centroid of the seed vertices
        Eigen::Vector2d d = Eigen::Vector2d::Zero();
        for (const auto& v : seed_vertices) d += v;
        d /= static_cast<double>(seed_vertices.size());

        // Shape: isotropic with radius = inscribed circle of seed
        double r_init = inscribed_radius(seed_vertices, d);
        r_init = std::max(r_init * 0.8, 1e-4);
        Eigen::Matrix2d L = r_init * Eigen::Matrix2d::Identity();

        double prev_vol = r_init * r_init * M_PI;
        std::vector<HalfPlane> best_planes = bbox_planes;
        int iters = 0;

        for (int k = 0; k < max_iter; ++k) {
            iters = k + 1;

            // Step 1: RsI — generate separating halfplanes
            auto planes = run_rsi(obstacles, seed_vertices, L, d, bbox_planes);
            best_planes = planes;

            // Step 2: MVIE — find largest ellipsoid in current polytope
            const int m = static_cast<int>(planes.size());
            Eigen::MatrixXd A(m, 2);
            Eigen::VectorXd b(m);
            for (int i = 0; i < m; ++i) {
                A.row(i) = planes[i].normal.transpose();
                b(i) = planes[i].offset;
            }

            auto mvie = mvie_solver_.solve(A, b, d);

            // Step 3: Check convergence
            const double new_vol = mvie.volume();
            if (k > 0 && (new_vol - prev_vol) / (prev_vol + 1e-15) < rho) {
                break;
            }

            prev_vol = new_vol;
            L = mvie.L;
            d = mvie.d;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        return {best_planes, iters, ms};
    }

private:
    SDMN2D sdmn_;
    MVIE2D mvie_solver_;

    // ── Inscribed radius of a convex polygon at a given center ──
    // Returns the distance from the center to the nearest edge.
    // Used to initialize the ellipsoid radius from the seed footprint.
    double inscribed_radius(const std::vector<Eigen::Vector2d>& verts,
                            const Eigen::Vector2d& c)
    {
        double r = 1e18;
        const int n = static_cast<int>(verts.size());

        for (int i = 0; i < n; ++i) {
            const Eigen::Vector2d& a = verts[i];
            const Eigen::Vector2d& b = verts[(i + 1) % n];
            const Eigen::Vector2d edge = b - a;
            const double len = edge.norm();
            if (len < 1e-15) continue;

            // Outward normal of edge (unnormalized, then normalize)
            Eigen::Vector2d normal(-edge.y(), edge.x());
            normal /= len;

            // Distance from center to the line containing this edge
            const double dist = std::abs(normal.dot(c - a));
            r = std::min(r, dist);
        }
        return r;
    }

    // ================================================================
    // RsI: Region-seed Inflation
    // ================================================================
    // For each obstacle point, solve a minimum-norm QP (via SDMN) to find
    // the tightest separating halfplane that still contains the seed.
    // Then greedily select halfplanes that cover the most obstacles.
    //
    // All computation happens in "bar" (ellipsoid-normalized) space:
    //   x_bar = L^{-1} (x - d)
    // where L, d are the current ellipsoid parameters. In this space
    // the ellipsoid is the unit ball, so distance comparisons are fair
    // across all directions.
    //
    // The seed manageability constraint ensures the robot footprint
    // is always contained: each seed vertex v_bar must satisfy
    //   v_bar^T y <= 1
    // which in original space means the halfplane doesn't cut into the seed.
    //
    // The obstacle separation constraint for obstacle u_bar is:
    //   -u_bar^T y <= -1
    // i.e. the obstacle is on the outside of the halfplane.
    //
    // SDMN finds the minimum-norm y satisfying both, and y/||y||^2
    // gives the separating halfplane normal in bar space.
    std::vector<HalfPlane> run_rsi(
        const std::vector<Eigen::Vector2d>& obstacles,
        const std::vector<Eigen::Vector2d>& seed_vertices,
        const Eigen::Matrix2d& L,
        const Eigen::Vector2d& d,
        const std::vector<HalfPlane>& bbox_planes)
    {
        // ── Transform to normalized space ──
        const Eigen::Matrix2d L_inv = L.inverse();
        const Eigen::Matrix2d L_inv_T = L_inv.transpose();

        std::vector<Eigen::Vector2d> seed_bar;
        seed_bar.reserve(seed_vertices.size());
        for (const auto& v : seed_vertices) {
            seed_bar.push_back(L_inv * (v - d));
        }

        std::vector<Eigen::Vector2d> obs_bar;
        obs_bar.reserve(obstacles.size());
        for (const auto& u : obstacles) {
            obs_bar.push_back(L_inv * (u - d));
        }

        // ── Build base constraints (seed manageability) ──
        // n_seed constraints: v_bar[i]^T y <= 1
        // Plus one slot for the obstacle constraint (reused per obstacle)
        const int n_seed = static_cast<int>(seed_bar.size());
        const int n_cons = n_seed + 1;

        std::vector<Eigen::Vector2d> base_normals(n_cons);
        std::vector<double> base_bounds(n_cons);
        for (int i = 0; i < n_seed; ++i) {
            base_normals[i] = seed_bar[i];
            base_bounds[i] = 1.0;
        }

        // ── Per-obstacle SDMN solve ──
        struct ObsHalfPlane {
            Eigen::Vector2d b_sol;  // SDMN solution y (in bar space)
            Eigen::Vector2d a;      // halfplane normal in bar space = y / ||y||^2
            double a_norm;          // ||a|| — used for greedy sorting
            int obs_idx;            // which obstacle generated this
        };

        std::vector<ObsHalfPlane> candidates;
        candidates.reserve(obs_bar.size());

        for (size_t i = 0; i < obs_bar.size(); ++i) {
            // Add obstacle constraint: -u_bar^T y <= -1
            base_normals[n_seed] = -obs_bar[i];
            base_bounds[n_seed] = -1.0;

            auto result = sdmn_.solve(base_normals, base_bounds);

            if (result.feasible) {
                const double b_sq = result.y.squaredNorm();
                if (b_sq > 1e-10) {
                    Eigen::Vector2d a = result.y / b_sq;
                    candidates.push_back({result.y, a, a.norm(), static_cast<int>(i)});
                }
            }
        }

        // ── Greedy halfplane selection ──
        // Sort by ||a|| ascending: tighter halfplanes first (they separate
        // more obstacles per plane, giving a compact representation).
        std::sort(candidates.begin(), candidates.end(),
                  [](const ObsHalfPlane& a, const ObsHalfPlane& b) {
                      return a.a_norm < b.a_norm;
                  });

        // Track which obstacles are already separated
        std::vector<bool> separated(obs_bar.size(), false);

        // Start with the bounding box planes
        std::vector<HalfPlane> result_planes = bbox_planes;

        for (const auto& hp : candidates) {
            if (separated[hp.obs_idx]) continue;

            // ── Transform halfplane back to original space ──
            // In bar space: a^T x_bar <= a^T b_sol = ||a||^2 ... wait, let's be precise.
            // The halfplane in bar space is: a^T x_bar <= 1  (by construction of SDMN)
            // Substituting x_bar = L^{-1}(x - d):
            //   a^T L^{-1} (x - d) <= 1
            //   (L^{-T} a)^T x <= 1 + (L^{-T} a)^T d
            //   n^T x <= ||a||^2 + n^T d     where n = L^{-T} a
            Eigen::Vector2d n_orig = L_inv_T * hp.a;
            double d_orig = hp.a.squaredNorm() + n_orig.dot(d);

            // Normalize to unit normal
            const double n_len = n_orig.norm();
            if (n_len < 1e-15) continue;

            result_planes.push_back({n_orig / n_len, d_orig / n_len});

            // Mark all obstacles separated by this halfplane
            // In bar space: obstacle j is separated if b_sol^T obs_bar[j] >= 1
            for (size_t j = 0; j < obs_bar.size(); ++j) {
                if (!separated[j] && hp.b_sol.dot(obs_bar[j]) >= 1.0 - 1e-8) {
                    separated[j] = true;
                }
            }

            // Safety limit on total halfplane count
            if (result_planes.size() > 50) break;
        }

        return result_planes;
    }
};

}  // namespace firi