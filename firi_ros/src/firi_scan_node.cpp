// ============================================================================
// firi_scan_node.cpp — FIRI for 2D with LaserScan input (ROS2 Foxy)
//
// Event-driven: each LaserScan triggers a FIRI solve.
// Converts scan rays to 2D obstacle points in map frame,
// builds footprint seed + optional path-guided seed, runs FIRI,
// publishes polytope + visualization.
//
// Port of firi_node_sdmn.cpp (ROS1) with path-guided seeding.
// Shares solver headers with firi_grid_node.
// ============================================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

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


class FIRIScanNode : public rclcpp::Node
{
public:
    FIRIScanNode() : Node("firi_scan_node")
    {
        // ── Parameters ──

        // Robot footprint
        this->declare_parameter<double>("robot_length", 0.55);
        this->declare_parameter<double>("robot_width", 0.35);
        this->declare_parameter<double>("footprint_offset_x", 0.215);

        // FIRI solver
        this->declare_parameter<double>("voxel_size", 0.05);
        this->declare_parameter<int>("max_firi_iter", 10);
        this->declare_parameter<double>("convergence_rho", 0.02);

        // Bounding box
        this->declare_parameter<double>("bbox_ahead", 3.0);
        this->declare_parameter<double>("bbox_behind", 1.0);
        this->declare_parameter<double>("bbox_side", 2.0);

        // Path-guided seeding
        this->declare_parameter<int>("seed_n_samples", 4);
        this->declare_parameter<double>("seed_lookahead", 1.5);
        this->declare_parameter<double>("seed_path_timeout", 5.0);
        this->declare_parameter<bool>("seed_use_path", true);

        // Topics
        this->declare_parameter<std::string>("scan_topic", "/filtered_scan");
        this->declare_parameter<std::string>("state_topic", "/truck/state");

        robot_length_        = this->get_parameter("robot_length").as_double();
        robot_width_         = this->get_parameter("robot_width").as_double();
        footprint_offset_x_  = this->get_parameter("footprint_offset_x").as_double();
        voxel_size_          = this->get_parameter("voxel_size").as_double();
        max_firi_iter_       = this->get_parameter("max_firi_iter").as_int();
        convergence_rho_     = this->get_parameter("convergence_rho").as_double();
        bbox_ahead_          = this->get_parameter("bbox_ahead").as_double();
        bbox_behind_         = this->get_parameter("bbox_behind").as_double();
        bbox_side_           = this->get_parameter("bbox_side").as_double();
        seed_n_samples_      = this->get_parameter("seed_n_samples").as_int();
        seed_lookahead_      = this->get_parameter("seed_lookahead").as_double();
        seed_path_timeout_   = this->get_parameter("seed_path_timeout").as_double();
        seed_use_path_       = this->get_parameter("seed_use_path").as_bool();

        std::string scan_topic  = this->get_parameter("scan_topic").as_string();
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
        state_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            state_topic, 10,
            std::bind(&FIRIScanNode::state_callback, this, std::placeholders::_1));

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic, 10,
            std::bind(&FIRIScanNode::scan_callback, this, std::placeholders::_1));

        plan_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/plan", 10,
            std::bind(&FIRIScanNode::plan_callback, this, std::placeholders::_1));

        mpc_traj_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/mpc/trajectory", 10,
            std::bind(&FIRIScanNode::mpc_traj_callback, this, std::placeholders::_1));

        // ── Startup logging ──
        RCLCPP_INFO(this->get_logger(),
            "FIRI Scan Node started (path-guided seeding)");
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
            "  Scan topic: '%s', State topic: '%s'",
            scan_topic.c_str(), state_topic.c_str());
    }

private:
    // ================================================================
    // CALLBACKS
    // ================================================================

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
    // SCAN CALLBACK — event-driven FIRI solve
    // ================================================================
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!state_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for robot state...");
            return;
        }

        // 1. Convert LaserScan rays to 2D obstacle points in map frame
        //    The scan is in base_link frame. We transform each ray endpoint
        //    using the robot pose (from /truck/state).
        std::vector<Eigen::Vector2d> raw_obs;
        raw_obs.reserve(msg->ranges.size());

        double angle = msg->angle_min;
        for (size_t i = 0; i < msg->ranges.size(); ++i, angle += msg->angle_increment) {
            const float r = msg->ranges[i];
            if (r < msg->range_min || r > msg->range_max || !std::isfinite(r))
                continue;

            // Ray endpoint in map frame
            const double map_angle = angle + robot_yaw_;
            raw_obs.emplace_back(
                robot_pos_.x() + r * std::cos(map_angle),
                robot_pos_.y() + r * std::sin(map_angle));
        }

        if (raw_obs.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "No valid scan points");
            return;
        }

        // 2. Downsample
        auto obstacles = firi::voxel_filter(raw_obs, voxel_size_);

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

        // 4. Add path-guided seed points
        std::string seed_source = "footprint";
        if (seed_use_path_) {
            auto path_seeds = get_path_seed_points(seed_source);
            for (const auto& pt : path_seeds) {
                seed.push_back(pt);
            }
        }

        // 5. Heading-aligned bounding box
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
            obstacles, seed, bbox_planes, max_firi_iter_, convergence_rho_);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "%zu rays -> %zu obs | %d iters | %zu planes | %.2f ms | seed: %s (%zu pts)",
            raw_obs.size(), obstacles.size(),
            result.iterations, result.planes.size(),
            result.solve_time_ms, seed_source.c_str(), seed.size());

        // 7. Publish
        frame_id_ = "map";
        publish_polytope(result.planes);
        publish_visualization(result.planes);
        publish_seed_viz(seed);
    }

    // ================================================================
    // PATH-GUIDED SEED EXTRACTION (same as grid node)
    // ================================================================
    std::vector<Eigen::Vector2d> get_path_seed_points(std::string& source)
    {
        std::vector<Eigen::Vector2d> pts;
        const auto now = this->now();

        if (!mpc_traj_.poses.empty()) {
            double age = (now - mpc_traj_time_).seconds();
            if (age < seed_path_timeout_) {
                pts = sample_first_n_from_path(mpc_traj_);
                if (!pts.empty()) { source = "mpc"; return pts; }
            }
        }

        if (!global_plan_.poses.empty()) {
            double age = (now - global_plan_time_).seconds();
            if (age < seed_path_timeout_) {
                pts = sample_first_n_from_path(global_plan_);
                if (!pts.empty()) { source = "plan"; return pts; }
            }
        }

        source = "footprint";
        return pts;
    }

    std::vector<Eigen::Vector2d> sample_first_n_from_path(
        const nav_msgs::msg::Path& path)
    {
        std::vector<Eigen::Vector2d> pts;
        if (path.poses.empty() || seed_n_samples_ <= 0) return pts;

        const int n_poses = static_cast<int>(path.poses.size());

        double min_dist_sq = 1e18;
        int closest_idx = 0;
        for (int i = 0; i < n_poses; ++i) {
            double dx = path.poses[i].pose.position.x - robot_pos_.x();
            double dy = path.poses[i].pose.position.y - robot_pos_.y();
            double d2 = dx * dx + dy * dy;
            if (d2 < min_dist_sq) { min_dist_sq = d2; closest_idx = i; }
        }

        if (std::sqrt(min_dist_sq) > seed_lookahead_) return pts;

        const double front_edge_dist = footprint_offset_x_ + robot_length_ / 2.0;
        const double skip_dist_sq = front_edge_dist * front_edge_dist;

        int count = 0;
        for (int i = closest_idx + 1; i < n_poses && count < seed_n_samples_; ++i) {
            double dx = path.poses[i].pose.position.x - robot_pos_.x();
            double dy = path.poses[i].pose.position.y - robot_pos_.y();
            if (dx * dx + dy * dy < skip_dist_sq) continue;

            pts.emplace_back(path.poses[i].pose.position.x,
                             path.poses[i].pose.position.y);
            count++;
        }
        return pts;
    }

    // ================================================================
    // PUBLISHERS (same as grid node)
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

    void publish_visualization(const std::vector<firi::HalfPlane>& planes)
    {
        if (planes.size() < 3) return;
        auto vertices = compute_polytope_vertices(planes);
        if (vertices.size() < 3) return;

        const auto stamp = this->now();

        // Filled mesh
        visualization_msgs::msg::Marker mesh;
        mesh.header.stamp = stamp;
        mesh.header.frame_id = frame_id_;
        mesh.ns = "firi_mesh";
        mesh.id = 0;
        mesh.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        mesh.action = visualization_msgs::msg::Marker::ADD;
        mesh.scale.x = 1.0; mesh.scale.y = 1.0; mesh.scale.z = 1.0;
        mesh.color.r = 0.0f; mesh.color.g = 0.67f; mesh.color.b = 1.0f; mesh.color.a = 0.3f;
        mesh.pose.orientation.w = 1.0;

        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& v : vertices) centroid += v;
        centroid /= static_cast<double>(vertices.size());

        for (size_t i = 0; i < vertices.size(); ++i) {
            size_t j = (i + 1) % vertices.size();
            geometry_msgs::msg::Point p0, p1, p2;
            p0.x = centroid.x(); p0.y = centroid.y(); p0.z = 0.02;
            p1.x = vertices[i].x(); p1.y = vertices[i].y(); p1.z = 0.02;
            p2.x = vertices[j].x(); p2.y = vertices[j].y(); p2.z = 0.02;
            mesh.points.push_back(p0);
            mesh.points.push_back(p1);
            mesh.points.push_back(p2);
        }
        viz_pub_->publish(mesh);

        // Boundary outline
        visualization_msgs::msg::Marker bound;
        bound.header.stamp = stamp;
        bound.header.frame_id = frame_id_;
        bound.ns = "firi_bound";
        bound.id = 0;
        bound.type = visualization_msgs::msg::Marker::LINE_STRIP;
        bound.action = visualization_msgs::msg::Marker::ADD;
        bound.scale.x = 0.02;
        bound.color.r = 1.0f; bound.color.g = 0.0f; bound.color.b = 0.0f; bound.color.a = 1.0f;
        bound.pose.orientation.w = 1.0;

        for (const auto& v : vertices) {
            geometry_msgs::msg::Point p;
            p.x = v.x(); p.y = v.y(); p.z = 0.03;
            bound.points.push_back(p);
        }
        if (!vertices.empty()) {
            geometry_msgs::msg::Point p;
            p.x = vertices.front().x(); p.y = vertices.front().y(); p.z = 0.03;
            bound.points.push_back(p);
        }
        bound_pub_->publish(bound);
    }

    void publish_seed_viz(const std::vector<Eigen::Vector2d>& seed)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = frame_id_;
        marker.ns = "firi_seed";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.04; marker.scale.y = 0.04; marker.scale.z = 0.04;
        marker.pose.orientation.w = 1.0;

        for (size_t i = 0; i < seed.size(); ++i) {
            geometry_msgs::msg::Point p;
            p.x = seed[i].x(); p.y = seed[i].y(); p.z = 0.05;
            marker.points.push_back(p);

            std_msgs::msg::ColorRGBA c;
            if (i < 4) { c.r = 0.2f; c.g = 1.0f; c.b = 0.2f; c.a = 1.0f; }
            else        { c.r = 1.0f; c.g = 0.0f; c.b = 1.0f; c.a = 1.0f; }
            marker.colors.push_back(c);
        }
        seed_viz_pub_->publish(marker);
    }

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
                        inside = false; break;
                    }
                }
                if (inside) vertices.push_back(vertex);
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
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr state_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr plan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr mpc_traj_sub_;

    // Publishers
    rclcpp::Publisher<firi_msgs::msg::Polytope2DStamped>::SharedPtr poly_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viz_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr bound_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr seed_viz_pub_;

    // Solver
    firi::FIRISolver solver_;

    // Robot state
    Eigen::Vector2d robot_pos_ = Eigen::Vector2d::Zero();
    double robot_yaw_ = 0.0;
    bool state_received_ = false;
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
    int seed_n_samples_;
    double seed_lookahead_;
    double seed_path_timeout_;
    bool seed_use_path_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FIRIScanNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}