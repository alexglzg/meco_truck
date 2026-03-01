#pragma once
// ============================================================================
// voxel_filter.hpp — Spatial hash voxel filter for 2D points
//
// Downsamples a point cloud by keeping one point per grid cell.
// No ROS dependency — pure Eigen + STL.
// ============================================================================

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <unordered_set>

namespace firi {

namespace detail {

/// Hash functor for Eigen::Vector2i (grid cell keys)
struct Vector2iHash {
    size_t operator()(const Eigen::Vector2i& k) const {
        // Combine two integer hashes with a bit shift
        return std::hash<int>()(k.x()) ^ (std::hash<int>()(k.y()) << 1);
    }
};

}  // namespace detail

/// Downsample 2D points: keep first point that falls in each grid cell.
/// @param pts      Input points in world coordinates
/// @param res      Grid cell size (meters). Points within the same cell are merged.
/// @return         Filtered points (one per cell, in insertion order)
inline std::vector<Eigen::Vector2d> voxel_filter(
    const std::vector<Eigen::Vector2d>& pts,
    double res)
{
    if (pts.empty()) return {};

    std::unordered_set<Eigen::Vector2i, detail::Vector2iHash> occupied;
    std::vector<Eigen::Vector2d> filtered;
    filtered.reserve(pts.size());

    const double inv_res = 1.0 / res;

    for (const auto& p : pts) {
        // Map continuous coordinate to integer grid cell
        Eigen::Vector2i key(
            static_cast<int>(std::floor(p.x() * inv_res)),
            static_cast<int>(std::floor(p.y() * inv_res))
        );

        // insert() returns {iterator, bool}. The bool is true if insertion happened.
        if (occupied.insert(key).second) {
            filtered.push_back(p);
        }
    }

    return filtered;
}

}  // namespace firi