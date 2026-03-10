// ============================================================================
// firi_grid_node.cpp — FIRI for 2D with OccupancyGrid input (ROS2 Foxy)
//
// Subscribes to an OccupancyGrid (from nav2_map_server) and robot state
// (from the truck simulator), computes the largest obstacle-free convex
// polytope containing the robot footprint using FIRI, and publishes it
// as a Polytope2DStamped message + RViz visualization markers.
//
// v2: Path-guided seeding
//   Optionally subscribes to /plan (global A* path) and /mpc/trajectory
//   (MPC predicted path). The first N poses from the closest point onward
//   are added to the FIRI seed, biasing the polytope inflation toward the
//   robot's intended travel direction. This prevents FIRI from expanding
//   into the wrong corridor at junctions.
//
//   Seed priority: MPC trajectory > global plan > footprint only
//   The solver itself is unchanged — we just feed it more seed points.
// ============================================================================

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// Custom messages
#include <firi_msgs/msg/polytope2_d_stamped.hpp>
#include <firi_msgs/msg/half_plane2_d.hpp>

// FIRI solver headers (pure Eigen, no ROS)
#include "firi_ros/firi_solver.hpp"
#include "firi_ros/voxel_filter.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mutex>


class FIRIGridNode : public rclcpp::Node
{
public:
    FIRIGridNode() : Node("firi_grid_node")
    {
        // ── Declare and read parameters ──

        // Robot footprint
        this->declare_parameter<double>("robot_length", 0.555);
        this->declare_parameter<double>("robot_width", 0.35);
        this->declare_parameter<double>("footprint_offset_x", 0.1975);

        // FIRI solver
        this->declare_parameter<double>("voxel_size", 0.05);
        this->declare_parameter<int>("max_firi_iter", 10);
        this->declare_parameter<double>("convergence_rho", 0.02);

        // Bounding box
        this->declare_parameter<double>("bbox_ahead", 3.0);
        this->declare_parameter<double>("bbox_behind", 1.0);
        this->declare_parameter<double>("bbox_side", 2.0);

        // Grid processing
        this->declare_parameter<int>("occupancy_threshold", 50);
        this->declare_parameter<bool>("use_boundary_extraction", true);

        // Path-guided seeding
        this->declare_parameter<int>("seed_n_samples", 4);
        this->declare_parameter<double>("seed_lookahead", 1.5);
        this->declare_parameter<double>("seed_path_timeout", 5.0);
        this->declare_parameter<bool>("seed_use_path", true);

        // Topics
        this->declare_parameter<std::string>("map_topic", "/map");
        this->declare_parameter<std::string>("state_topic", "/truck/state");

        robot_length_           = this->get_parameter("robot_length").as_double();
        robot_width_            = this->get_parameter("robot_width").as_double();
        footprint_offset_x_    = this->get_parameter("footprint_offset_x").as_double();
        voxel_size_             = this->get_parameter("voxel_size").as_double();
        max_firi_iter_          = this->get_parameter("max_firi_iter").as_int();
        convergence_rho_        = this->get_parameter("convergence_rho").as_double();
        bbox_ahead_             = this->get_parameter("bbox_ahead").as_double();
        bbox_behind_            = this->get_parameter("bbox_behind").as_double();
        bbox_side_              = this->get_parameter("bbox_side").as_double();
        occupancy_threshold_    = this->get_parameter("occupancy_threshold").as_int();
        use_boundary_extraction_= this->get_parameter("use_boundary_extraction").as_bool();
        seed_n_samples_         = this->get_parameter("seed_n_samples").as_int();
        seed_lookahead_         = this->get_parameter("seed_lookahead").as_double();
        seed_path_timeout_      = this->get_parameter("seed_path_timeout").as_double();
        seed_use_path_          = this->get_parameter("seed_use_path").as_bool();

        std::string map_topic   = this->get_parameter("map_topic").as_string();
        std::string state_topic = this->get_parameter("state_topic").as_string();

        // ── Publishers ──
        poly_pub_ = this->create_publisher<firi_msgs::msg::Polytope2DStamped>(
            "/firi/polytope", 10);

        viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/firi/polytope_mesh", 10);

        bound_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/firi/polytope_bound", 10);

        seed_viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/firi/seed_points", 10);

        // ── Subscriptions ──

        // Robot state: Float64MultiArray [x, y, theta, ...]
        state_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            state_topic, 10,
            std::bind(&FIRIGridNode::state_callback, this, std::placeholders::_1));

        // Map: transient local QoS to match nav2_map_server
        rclcpp::QoS map_qos(1);
        map_qos.reliable();
        map_qos.transient_local();

        grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            map_topic, map_qos,
            std::bind(&FIRIGridNode::grid_callback, this, std::placeholders::_1));

        // Global plan from NavFn (via bridge node)
        plan_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/plan", 10,
            std::bind(&FIRIGridNode::plan_callback, this, std::placeholders::_1));

        // MPC predicted trajectory
        mpc_traj_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/mpc/trajectory", 10,
            std::bind(&FIRIGridNode::mpc_traj_callback, this, std::placeholders::_1));

        // ── Timer: 10 Hz main loop ──
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&FIRIGridNode::timer_callback, this));

        // ── Startup logging ──
        RCLCPP_INFO(this->get_logger(),
            "FIRI Grid Node started (path-guided seeding)");
        RCLCPP_INFO(this->get_logger(),
            "  Robot: %.3f x %.3f m, offset_x: %.4f m",
            robot_length_, robot_width_, footprint_offset_x_);
        RCLCPP_INFO(this->get_logger(),
            "  Solver: max_iter=%d, rho=%.3f, voxel=%.3f m",
            max_firi_iter_, convergence_rho_, voxel_size_);
        RCLCPP_INFO(this->get_logger(),
            "  BBox: ahead=%.1f, behind=%.1f, side=%.1f m",
            bbox_ahead_, bbox_behind_, bbox_side_);
        RCLCPP_INFO(this->get_logger(),
            "  Path seed: n_samples=%d, lookahead=%.2f m, timeout=%.1f s, enabled=%s",
            seed_n_samples_, seed_lookahead_, seed_path_timeout_,
            seed_use_path_ ? "ON" : "OFF");
    }

private:
    // ================================================================
    // CALLBACKS
    // ================================================================

    void grid_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        frame_id_ = msg->header.frame_id;
        auto t_start = std::chrono::high_resolution_clock::now();

        if (use_boundary_extraction_) {
            extract_boundary_obstacles(msg);
        } else {
            extract_all_obstacles(msg);
        }

        grid_cached_ = true;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        RCLCPP_INFO(this->get_logger(),
            "Grid %dx%d (%.3fm) -> %zu boundary obstacles in %.1f ms",
            msg->info.width, msg->info.height, msg->info.resolution,
            all_boundary_obs_.size(), ms);
    }

    void state_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 3) return;
        robot_pos_.x() = msg->data[0];
        robot_pos_.y() = msg->data[1];
        robot_yaw_ = msg->data[2];
        state_received_ = true;
    }

    void plan_callback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.empty()) return;
        global_plan_ = *msg;
        global_plan_time_ = this->now();
    }

    void mpc_traj_callback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.empty()) return;
        mpc_traj_ = *msg;
        mpc_traj_time_ = this->now();
    }

    // ================================================================
    // TIMER CALLBACK (main loop)
    // ================================================================
    void timer_callback()
    {
        if (!grid_cached_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for occupancy grid...");
            return;
        }
        if (!state_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for robot state...");
            return;
        }

        // 1. Filter cached obstacles to local bounding box
        std::vector<Eigen::Vector2d> local_obs = filter_local_obstacles();

        if (local_obs.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "No obstacles in local region");
            return;
        }

        // 2. Optional voxel downsampling
        local_obs = firi::voxel_filter(local_obs, voxel_size_);

        // 3. Build robot footprint seed
        const double hl = robot_length_ / 2.0;
        const double hw = robot_width_ / 2.0;
        const double cos_yaw = std::cos(robot_yaw_);
        const double sin_yaw = std::sin(robot_yaw_);

        const Eigen::Vector2d body_center(
            robot_pos_.x() + footprint_offset_x_ * cos_yaw,
            robot_pos_.y() + footprint_offset_x_ * sin_yaw);

        Eigen::Matrix2d R;
        R << cos_yaw, -sin_yaw,
             sin_yaw,  cos_yaw;

        std::vector<Eigen::Vector2d> seed = {
            body_center + R * Eigen::Vector2d( hl,  hw),
            body_center + R * Eigen::Vector2d( hl, -hw),
            body_center + R * Eigen::Vector2d(-hl, -hw),
            body_center + R * Eigen::Vector2d(-hl,  hw)
        };

        // 4. Add path-guided seed points (if available)
        std::string seed_source = "footprint";
        if (seed_use_path_) {
            auto path_seeds = get_path_seed_points(seed_source);
            for (const auto& pt : path_seeds) {
                seed.push_back(pt);
            }
        }

        // 5. Heading-aligned bounding box as 4 halfplanes
        const Eigen::Vector2d fwd = R.col(0);
        const Eigen::Vector2d lft = R.col(1);

        std::vector<firi::HalfPlane> bbox_planes = {
            { fwd,  fwd.dot(robot_pos_) + bbox_ahead_},
            {-fwd, -fwd.dot(robot_pos_) + bbox_behind_},
            { lft,  lft.dot(robot_pos_) + bbox_side_},
            {-lft, -lft.dot(robot_pos_) + bbox_side_}
        };

        // 6. Run FIRI
        auto result = solver_.compute(
            local_obs, seed, bbox_planes, max_firi_iter_, convergence_rho_);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "%zu obs | %d iters | %zu planes | %.2f ms | seed: %s (%zu pts)",
            local_obs.size(), result.iterations, result.planes.size(),
            result.solve_time_ms, seed_source.c_str(), seed.size());

        // 7. Publish
        publish_polytope(result.planes);
        publish_visualization(result.planes);
        publish_seed_viz(seed);
    }

    // ================================================================
    // PATH-GUIDED SEED EXTRACTION
    // ================================================================
    // Returns the first N centerline points from the active path,
    // starting from the closest pose to the robot.
    //
    // Priority: MPC trajectory > global plan > empty
    //
    // "First N" means: find the closest pose, then take the next
    // seed_n_samples_ poses sequentially from the path. These are
    // the immediate next waypoints, not evenly spaced along a
    // lookahead distance. The seed_lookahead_ parameter is used
    // only to reject stale/far-away paths — if the closest point
    // is further than seed_lookahead_, we skip the path entirely.
    //
    // Note: we add centerline points only (not full footprint corners
    // at each pose). The MPC-CBF handles full-body safety. The seed
    // just biases FIRI's inflation direction.

    std::vector<Eigen::Vector2d> get_path_seed_points(std::string& source)
    {
        std::vector<Eigen::Vector2d> pts;
        const auto now = this->now();

        // Try MPC trajectory first (best source: dynamically feasible)
        if (!mpc_traj_.poses.empty()) {
            double age = (now - mpc_traj_time_).seconds();
            if (age < seed_path_timeout_) {
                pts = sample_first_n_from_path(mpc_traj_);
                if (!pts.empty()) {
                    source = "mpc";
                    return pts;
                }
            }
        }

        // Fall back to global plan (coarser, but gives direction)
        if (!global_plan_.poses.empty()) {
            double age = (now - global_plan_time_).seconds();
            if (age < seed_path_timeout_) {
                pts = sample_first_n_from_path(global_plan_);
                if (!pts.empty()) {
                    source = "plan";
                    return pts;
                }
            }
        }

        // No path available — footprint-only seeding
        source = "footprint";
        return pts;
    }

    // Extract the first N poses from the path starting at the closest point
    std::vector<Eigen::Vector2d> sample_first_n_from_path(
        const nav_msgs::msg::Path& path)
    {
        std::vector<Eigen::Vector2d> pts;
        if (path.poses.empty() || seed_n_samples_ <= 0) return pts;

        const int n_poses = static_cast<int>(path.poses.size());

        // Find closest pose to robot
        double min_dist_sq = 1e18;
        int closest_idx = 0;

        for (int i = 0; i < n_poses; ++i) {
            double dx = path.poses[i].pose.position.x - robot_pos_.x();
            double dy = path.poses[i].pose.position.y - robot_pos_.y();
            double d2 = dx * dx + dy * dy;
            if (d2 < min_dist_sq) {
                min_dist_sq = d2;
                closest_idx = i;
            }
        }

        // If closest point is too far, path is stale/irrelevant
        if (std::sqrt(min_dist_sq) > seed_lookahead_) {
            return pts;
        }

        // Take the next N poses after closest, but skip any that are
        // still inside the footprint rectangle. The front edge of the
        // truck is at front_edge_dist ahead of the rear axle. Any path
        // point within that distance is already contained in the convex
        // hull of the 4 footprint corners, so it adds nothing to the seed.
        const double front_edge_dist = footprint_offset_x_ + robot_length_ / 2.0;
        const double skip_dist_sq = front_edge_dist * front_edge_dist;

        int start_idx = closest_idx + 1;
        int count = 0;

        for (int i = start_idx; i < n_poses && count < seed_n_samples_; ++i) {
            double px = path.poses[i].pose.position.x;
            double py = path.poses[i].pose.position.y;

            // Skip points inside the footprint rectangle
            double dx = px - robot_pos_.x();
            double dy = py - robot_pos_.y();
            if (dx * dx + dy * dy < skip_dist_sq) continue;

            pts.emplace_back(px, py);
            count++;
        }

        return pts;
    }

    // ================================================================
    // BOUNDARY EXTRACTION
    // ================================================================
    // Extract obstacle boundary points at CELL FACES, not cell centers.
    //
    // For each occupied cell adjacent to a free cell, we emit a point
    // at the face between them. This places the obstacle point at the
    // actual wall surface, not 0.5*resolution inside the wall.
    //
    // Without this, at 0.05m resolution FIRI's halfplanes separate the
    // cell center but cut 0.025m into the occupied area.
    //
    // A boundary cell with 2 free neighbors produces 2 face points.
    // This slightly increases point count but gives FIRI accurate
    // wall geometry to work with.
    void extract_boundary_obstacles(const nav_msgs::msg::OccupancyGrid::SharedPtr& msg)
    {
        all_boundary_obs_.clear();
        const auto& info = msg->info;
        const int w = static_cast<int>(info.width);
        const int h = static_cast<int>(info.height);

        tf2::Quaternion q_map(
            info.origin.orientation.x, info.origin.orientation.y,
            info.origin.orientation.z, info.origin.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_map).getRPY(roll, pitch, yaw);
        const double cos_yaw = std::cos(yaw);
        const double sin_yaw = std::sin(yaw);
        const double ox = info.origin.position.x;
        const double oy = info.origin.position.y;
        const double res = info.resolution;

        // 4-connected neighbor offsets: right, left, up, down
        const int dc[] = {1, -1, 0, 0};
        const int dr[] = {0, 0, 1, -1};

        // Face offsets from cell center (in grid units, i.e. multiply by res)
        // For neighbor (dc, dr), the shared face center is at cell_center + 0.5*(dc, dr)
        const double face_dx[] = { 0.5, -0.5,  0.0,  0.0};
        const double face_dy[] = { 0.0,  0.0,  0.5, -0.5};

        all_boundary_obs_.reserve(w * h / 8);

        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const int idx = row * w + col;
                const int8_t val = static_cast<int8_t>(msg->data[idx]);

                if (val < occupancy_threshold_) continue;

                // Check each 4-connected neighbor
                for (int n = 0; n < 4; ++n) {
                    const int nr = row + dr[n];
                    const int nc = col + dc[n];

                    bool neighbor_is_free = false;

                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) {
                        // Edge of map: treat as boundary
                        neighbor_is_free = true;
                    } else {
                        const int8_t nval = static_cast<int8_t>(msg->data[nr * w + nc]);
                        if (nval >= 0 && nval < occupancy_threshold_) {
                            neighbor_is_free = true;
                        }
                    }

                    if (!neighbor_is_free) continue;

                    // Emit point at the face center between this cell and the free neighbor
                    const double x_grid = (col + 0.5 + face_dx[n]) * res;
                    const double y_grid = (row + 0.5 + face_dy[n]) * res;
                    const double x = ox + x_grid * cos_yaw - y_grid * sin_yaw;
                    const double y = oy + x_grid * sin_yaw + y_grid * cos_yaw;

                    all_boundary_obs_.emplace_back(x, y);
                }
            }
        }
    }

    void extract_all_obstacles(const nav_msgs::msg::OccupancyGrid::SharedPtr& msg)
    {
        all_boundary_obs_.clear();
        const auto& info = msg->info;
        const int w = static_cast<int>(info.width);
        const int h = static_cast<int>(info.height);

        tf2::Quaternion q_map(
            info.origin.orientation.x, info.origin.orientation.y,
            info.origin.orientation.z, info.origin.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_map).getRPY(roll, pitch, yaw);
        const double cos_yaw = std::cos(yaw);
        const double sin_yaw = std::sin(yaw);
        const double ox = info.origin.position.x;
        const double oy = info.origin.position.y;
        const double res = info.resolution;

        all_boundary_obs_.reserve(w * h / 4);

        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const int idx = row * w + col;
                if (static_cast<int8_t>(msg->data[idx]) >= occupancy_threshold_) {
                    const double x_grid = (col + 0.5) * res;
                    const double y_grid = (row + 0.5) * res;
                    const double x = ox + x_grid * cos_yaw - y_grid * sin_yaw;
                    const double y = oy + x_grid * sin_yaw + y_grid * cos_yaw;
                    all_boundary_obs_.emplace_back(x, y);
                }
            }
        }
    }

    // ================================================================
    // LOCAL OBSTACLE FILTERING
    // ================================================================
    std::vector<Eigen::Vector2d> filter_local_obstacles()
    {
        const double search_radius =
            std::max({bbox_ahead_, bbox_behind_, bbox_side_}) * 1.2;
        const double search_radius_sq = search_radius * search_radius;

        std::vector<Eigen::Vector2d> local;
        local.reserve(all_boundary_obs_.size() / 4);

        for (const auto& obs : all_boundary_obs_) {
            const double dx = obs.x() - robot_pos_.x();
            const double dy = obs.y() - robot_pos_.y();
            if (dx * dx + dy * dy <= search_radius_sq) {
                local.push_back(obs);
            }
        }
        return local;
    }

    // ================================================================
    // PUBLISH POLYTOPE MESSAGE
    // ================================================================
    void publish_polytope(const std::vector<firi::HalfPlane>& planes)
    {
        firi_msgs::msg::Polytope2DStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;

        msg.polytope.planes.resize(planes.size());
        for (size_t i = 0; i < planes.size(); ++i) {
            msg.polytope.planes[i].normal[0] = planes[i].normal.x();
            msg.polytope.planes[i].normal[1] = planes[i].normal.y();
            msg.polytope.planes[i].offset    = planes[i].offset;
        }

        poly_pub_->publish(msg);
    }

    // ================================================================
    // PUBLISH VISUALIZATION MARKERS (mesh + boundary)
    // ================================================================
    void publish_visualization(const std::vector<firi::HalfPlane>& planes)
    {
        if (planes.size() < 3) return;

        auto vertices = compute_polytope_vertices(planes);
        if (vertices.size() < 3) return;

        const auto stamp = this->now();

        // ── Marker 1: Filled mesh (TRIANGLE_LIST) ──
        visualization_msgs::msg::Marker mesh;
        mesh.header.stamp = stamp;
        mesh.header.frame_id = frame_id_;
        mesh.ns = "firi_mesh";
        mesh.id = 0;
        mesh.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        mesh.action = visualization_msgs::msg::Marker::ADD;

        mesh.scale.x = 1.0;
        mesh.scale.y = 1.0;
        mesh.scale.z = 1.0;

        mesh.color.r = 0.0f;
        mesh.color.g = 0.67f;
        mesh.color.b = 1.0f;
        mesh.color.a = 0.3f;

        mesh.pose.orientation.w = 1.0;

        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& v : vertices) centroid += v;
        centroid /= static_cast<double>(vertices.size());

        const float mesh_z = 0.02f;

        for (size_t i = 0; i < vertices.size(); ++i) {
            size_t j = (i + 1) % vertices.size();

            geometry_msgs::msg::Point p0, p1, p2;
            p0.x = centroid.x(); p0.y = centroid.y(); p0.z = mesh_z;
            p1.x = vertices[i].x(); p1.y = vertices[i].y(); p1.z = mesh_z;
            p2.x = vertices[j].x(); p2.y = vertices[j].y(); p2.z = mesh_z;

            mesh.points.push_back(p0);
            mesh.points.push_back(p1);
            mesh.points.push_back(p2);
        }

        viz_pub_->publish(mesh);

        // ── Marker 2: Boundary outline (LINE_STRIP) ──
        visualization_msgs::msg::Marker bound;
        bound.header.stamp = stamp;
        bound.header.frame_id = frame_id_;
        bound.ns = "firi_bound";
        bound.id = 0;
        bound.type = visualization_msgs::msg::Marker::LINE_STRIP;
        bound.action = visualization_msgs::msg::Marker::ADD;

        bound.scale.x = 0.02;

        bound.color.r = 1.0f;
        bound.color.g = 0.0f;
        bound.color.b = 0.0f;
        bound.color.a = 1.0f;

        bound.pose.orientation.w = 1.0;

        const float bound_z = 0.03f;

        for (const auto& v : vertices) {
            geometry_msgs::msg::Point p;
            p.x = v.x(); p.y = v.y(); p.z = bound_z;
            bound.points.push_back(p);
        }
        if (!vertices.empty()) {
            geometry_msgs::msg::Point p;
            p.x = vertices.front().x();
            p.y = vertices.front().y();
            p.z = bound_z;
            bound.points.push_back(p);
        }

        bound_pub_->publish(bound);
    }

    // ================================================================
    // PUBLISH SEED POINT VISUALIZATION
    // ================================================================
    // Shows all seed points as spheres in RViz2. Footprint corners
    // in green, path seed points in magenta.
    void publish_seed_viz(const std::vector<Eigen::Vector2d>& seed)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = frame_id_;
        marker.ns = "firi_seed";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.scale.x = 0.04;
        marker.scale.y = 0.04;
        marker.scale.z = 0.04;

        marker.pose.orientation.w = 1.0;

        for (size_t i = 0; i < seed.size(); ++i) {
            geometry_msgs::msg::Point p;
            p.x = seed[i].x();
            p.y = seed[i].y();
            p.z = 0.05;
            marker.points.push_back(p);

            std_msgs::msg::ColorRGBA c;
            if (i < 4) {
                // Footprint corners: green
                c.r = 0.2f; c.g = 1.0f; c.b = 0.2f; c.a = 1.0f;
            } else {
                // Path seed points: magenta
                c.r = 1.0f; c.g = 0.0f; c.b = 1.0f; c.a = 1.0f;
            }
            marker.colors.push_back(c);
        }

        seed_viz_pub_->publish(marker);
    }

    // ── Helper: compute sorted convex vertices from halfplanes ──
    std::vector<Eigen::Vector2d> compute_polytope_vertices(
        const std::vector<firi::HalfPlane>& planes)
    {
        std::vector<Eigen::Vector2d> vertices;
        const int n = static_cast<int>(planes.size());

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Eigen::Matrix2d M;
                M.row(0) = planes[i].normal.transpose();
                M.row(1) = planes[j].normal.transpose();

                const double det = M.determinant();
                if (std::abs(det) < 1e-10) continue;

                Eigen::Vector2d rhs(planes[i].offset, planes[j].offset);
                Eigen::Vector2d vertex = M.inverse() * rhs;

                bool inside = true;
                for (int k = 0; k < n; ++k) {
                    if (planes[k].normal.dot(vertex) > planes[k].offset + 1e-6) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    vertices.push_back(vertex);
                }
            }
        }

        if (vertices.size() < 3) return vertices;

        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& v : vertices) centroid += v;
        centroid /= static_cast<double>(vertices.size());

        std::sort(vertices.begin(), vertices.end(),
            [&centroid](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return std::atan2(a.y() - centroid.y(), a.x() - centroid.x())
                     < std::atan2(b.y() - centroid.y(), b.x() - centroid.x());
            });

        return vertices;
    }

    // ================================================================
    // MEMBER VARIABLES
    // ================================================================

    // Subscriptions
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr state_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr plan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr mpc_traj_sub_;

    // Publishers
    rclcpp::Publisher<firi_msgs::msg::Polytope2DStamped>::SharedPtr poly_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr bound_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr seed_viz_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // Solver
    firi::FIRISolver solver_;

    // Robot state
    Eigen::Vector2d robot_pos_ = Eigen::Vector2d::Zero();
    double robot_yaw_ = 0.0;
    bool state_received_ = false;

    // Cached obstacle data
    std::vector<Eigen::Vector2d> all_boundary_obs_;
    bool grid_cached_ = false;
    std::string frame_id_ = "map";

    // Cached paths
    nav_msgs::msg::Path global_plan_;
    nav_msgs::msg::Path mpc_traj_;
    rclcpp::Time global_plan_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time mpc_traj_time_{0, 0, RCL_ROS_TIME};

    // Parameters
    double robot_length_;
    double robot_width_;
    double footprint_offset_x_;
    double voxel_size_;
    int max_firi_iter_;
    double convergence_rho_;
    double bbox_ahead_;
    double bbox_behind_;
    double bbox_side_;
    int occupancy_threshold_;
    bool use_boundary_extraction_;
    int seed_n_samples_;
    double seed_lookahead_;
    double seed_path_timeout_;
    bool seed_use_path_;
};


// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FIRIGridNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}