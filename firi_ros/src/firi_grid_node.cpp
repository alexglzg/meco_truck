// ============================================================================
// firi_grid_node.cpp — FIRI for 2D with OccupancyGrid input (ROS2 Foxy)
//
// Subscribes to an OccupancyGrid (from nav2_map_server) and odometry
// (from the truck simulator), computes the largest obstacle-free convex
// polytope containing the robot footprint using FIRI, and publishes it
// as a Polytope2DStamped message + RViz visualization marker.
//
// Port of the ROS1 firi_grid_node.cpp to ROS2 ament_cmake.
// ============================================================================

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
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

class FIRIGridNode : public rclcpp::Node
{
public:
    FIRIGridNode() : Node("firi_grid_node")
    {
        // ── Declare and read parameters ──
        // In ROS2 Foxy, every parameter must be declared before use.
        // declare_parameter<T>(name, default_value) both registers the
        // parameter and sets its default. get_parameter() retrieves the
        // current value (which may have been overridden by a YAML file
        // or launch argument).

        this->declare_parameter<double>("robot_length", 0.555);
        this->declare_parameter<double>("robot_width", 0.35);
        this->declare_parameter<double>("footprint_offset_x", 0.1975);
        this->declare_parameter<double>("voxel_size", 0.05);
        this->declare_parameter<int>("max_firi_iter", 10);
        this->declare_parameter<double>("convergence_rho", 0.02);
        this->declare_parameter<double>("bbox_ahead", 3.0);
        this->declare_parameter<double>("bbox_behind", 1.0);
        this->declare_parameter<double>("bbox_side", 2.0);
        this->declare_parameter<int>("occupancy_threshold", 50);
        this->declare_parameter<bool>("use_boundary_extraction", true);
        this->declare_parameter<std::string>("map_topic", "/map");
        this->declare_parameter<std::string>("odom_topic", "/odom");

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

        std::string map_topic  = this->get_parameter("map_topic").as_string();
        std::string odom_topic = this->get_parameter("odom_topic").as_string();

        // ── Publishers ──
        // Polytope message for the MPC-CBF node
        poly_pub_ = this->create_publisher<firi_msgs::msg::Polytope2DStamped>(
            "/firi/polytope", 10);

        // Visualization for RViz2 (line strip showing polytope boundary)
        viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/firi/polytope_viz", 10);

        // ── Subscriptions ──
        // Odometry: default QoS is fine (the sim publishes at default QoS)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10,
            std::bind(&FIRIGridNode::odom_callback, this, std::placeholders::_1));

        // Map: nav2_map_server publishes with TRANSIENT_LOCAL durability.
        // This means it publishes once and "latches" the message for late
        // subscribers. If we use the default QoS (VOLATILE), we'll never
        // receive the map if we start after the map server. We must match
        // the publisher's QoS profile.
        rclcpp::QoS map_qos(1);
        map_qos.reliable();
        map_qos.transient_local();

        grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            map_topic, map_qos,
            std::bind(&FIRIGridNode::grid_callback, this, std::placeholders::_1));

        // ── Timer: 10 Hz main loop ──
        // create_wall_timer takes a duration and a callback. This replaces
        // the ROS1 pattern of while(ros::ok()) { spinOnce(); run(); rate.sleep(); }
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&FIRIGridNode::timer_callback, this));

        // ── Startup logging ──
        RCLCPP_INFO(this->get_logger(),
            "FIRI Grid Node started");
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
            "  Topics: map='%s', odom='%s'",
            map_topic.c_str(), odom_topic.c_str());
    }

private:
    // ================================================================
    // GRID CALLBACK
    // ================================================================
    // Called once (or rarely) when the map server publishes.
    // Extracts obstacle boundary points and caches them. The timer
    // callback then filters a local subset each cycle.
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

    // ================================================================
    // ODOMETRY CALLBACK
    // ================================================================
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        robot_pos_.x() = msg->pose.pose.position.x;
        robot_pos_.y() = msg->pose.pose.position.y;

        // Extract yaw from quaternion using tf2
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch;
        m.getRPY(roll, pitch, robot_yaw_);

        odom_received_ = true;
    }

    // ================================================================
    // TIMER CALLBACK (main loop, replaces ROS1 run())
    // ================================================================
    void timer_callback()
    {
        if (!grid_cached_) {
            // RCLCPP_WARN_THROTTLE requires a clock as first argument.
            // get_clock() returns this node's clock.
            // The second argument is the throttle period in milliseconds.
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for occupancy grid...");
            return;
        }
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for odometry...");
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

        // 3. Build robot footprint seed (rectangle around body center)
        //
        //    base_link is at the rear axle. The body center is
        //    footprint_offset_x_ ahead of base_link. We compute the
        //    body center in world frame, then place the seed rectangle
        //    around it.
        //
        //    URDF geometry:
        //      rear edge  = -0.080 m from rear axle
        //      front edge = +0.475 m from rear axle
        //      body center = +0.1975 m from rear axle
        //      half-length = 0.2775 m
        //      half-width  = 0.175 m

        const double hl = robot_length_ / 2.0;
        const double hw = robot_width_ / 2.0;
        const double cos_yaw = std::cos(robot_yaw_);
        const double sin_yaw = std::sin(robot_yaw_);

        // Body center in world frame
        const Eigen::Vector2d body_center(
            robot_pos_.x() + footprint_offset_x_ * cos_yaw,
            robot_pos_.y() + footprint_offset_x_ * sin_yaw);

        // Rotation matrix (heading)
        Eigen::Matrix2d R;
        R << cos_yaw, -sin_yaw,
             sin_yaw,  cos_yaw;

        // Seed vertices: rectangle centered on body_center
        std::vector<Eigen::Vector2d> seed = {
            body_center + R * Eigen::Vector2d( hl,  hw),   // front-left
            body_center + R * Eigen::Vector2d( hl, -hw),   // front-right
            body_center + R * Eigen::Vector2d(-hl, -hw),   // rear-right
            body_center + R * Eigen::Vector2d(-hl,  hw)    // rear-left
        };

        // 4. Heading-aligned bounding box as 4 halfplanes
        //    These define the maximum search region. The solver won't
        //    look for obstacles outside this box.
        const Eigen::Vector2d fwd = R.col(0);  // forward direction
        const Eigen::Vector2d lft = R.col(1);  // left direction

        // The bbox is centered on the robot position (rear axle),
        // not the body center. This way bbox_ahead controls how far
        // ahead of the rear axle we look, which is more intuitive
        // for tuning (you think "how far ahead can I see").
        std::vector<firi::HalfPlane> bbox_planes = {
            { fwd,  fwd.dot(robot_pos_) + bbox_ahead_},
            {-fwd, -fwd.dot(robot_pos_) + bbox_behind_},
            { lft,  lft.dot(robot_pos_) + bbox_side_},
            {-lft, -lft.dot(robot_pos_) + bbox_side_}
        };

        // 5. Run FIRI
        auto result = solver_.compute(
            local_obs, seed, bbox_planes, max_firi_iter_, convergence_rho_);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "%zu local obs | %d iters | %zu planes | %.2f ms",
            local_obs.size(), result.iterations, result.planes.size(),
            result.solve_time_ms);

        // 6. Publish
        publish_polytope(result.planes);
        publish_visualization(result.planes);
    }

    // ================================================================
    // BOUNDARY EXTRACTION
    // ================================================================
    // Only extract occupied cells adjacent to at least one free cell
    // (4-connected). A solid wall produces ~2 rows of points instead
    // of the full wall depth. Typically 3-10x reduction.
    void extract_boundary_obstacles(const nav_msgs::msg::OccupancyGrid::SharedPtr& msg)
    {
        all_boundary_obs_.clear();
        const auto& info = msg->info;
        const int w = static_cast<int>(info.width);
        const int h = static_cast<int>(info.height);

        // Map origin transform
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

        // 4-connected neighbor offsets
        const int dx[] = {1, -1, 0, 0};
        const int dy[] = {0, 0, 1, -1};

        all_boundary_obs_.reserve(w * h / 10);

        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const int idx = row * w + col;
                const int8_t val = static_cast<int8_t>(msg->data[idx]);

                // Must be occupied
                if (val < occupancy_threshold_) continue;

                // Check if any 4-connected neighbor is free
                bool is_boundary = false;
                for (int n = 0; n < 4; ++n) {
                    const int nr = row + dy[n];
                    const int nc = col + dx[n];

                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) {
                        is_boundary = true;
                        break;
                    }

                    const int8_t nval = static_cast<int8_t>(msg->data[nr * w + nc]);
                    if (nval >= 0 && nval < occupancy_threshold_) {
                        is_boundary = true;
                        break;
                    }
                }

                if (!is_boundary) continue;

                // Grid cell center -> world coordinates
                const double x_grid = (col + 0.5) * res;
                const double y_grid = (row + 0.5) * res;
                const double x = ox + x_grid * cos_yaw - y_grid * sin_yaw;
                const double y = oy + x_grid * sin_yaw + y_grid * cos_yaw;

                all_boundary_obs_.emplace_back(x, y);
            }
        }
    }

    // Fallback: extract ALL occupied cells (no boundary filtering)
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
    // Radius filter around robot position. Only obstacles within the
    // bounding box range are passed to the solver.
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
    // PUBLISH VISUALIZATION MARKER
    // ================================================================
    // Computes the polytope boundary as a set of vertices by intersecting
    // adjacent halfplanes, then publishes a LINE_STRIP marker for RViz2.
    //
    // This replaces the decomp_rviz_utils plugin from ROS1.
    void publish_visualization(const std::vector<firi::HalfPlane>& planes)
    {
        if (planes.size() < 3) return;

        // ── Compute polytope vertices by halfplane intersection ──
        // For each pair of adjacent halfplanes (i, j), solve:
        //   n_i^T x = b_i
        //   n_j^T x = b_j
        // Keep only vertices that satisfy ALL halfplanes (convex hull).
        std::vector<Eigen::Vector2d> vertices;
        const int n = static_cast<int>(planes.size());

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                // Solve 2x2 system: [n_i; n_j] * x = [b_i; b_j]
                Eigen::Matrix2d M;
                M.row(0) = planes[i].normal.transpose();
                M.row(1) = planes[j].normal.transpose();

                const double det = M.determinant();
                if (std::abs(det) < 1e-10) continue;  // parallel planes

                Eigen::Vector2d rhs(planes[i].offset, planes[j].offset);
                Eigen::Vector2d vertex = M.inverse() * rhs;

                // Check if vertex satisfies all halfplanes
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

        if (vertices.size() < 3) return;

        // ── Sort vertices by angle around centroid (convex hull ordering) ──
        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const auto& v : vertices) centroid += v;
        centroid /= static_cast<double>(vertices.size());

        std::sort(vertices.begin(), vertices.end(),
            [&centroid](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return std::atan2(a.y() - centroid.y(), a.x() - centroid.x())
                     < std::atan2(b.y() - centroid.y(), b.x() - centroid.x());
            });

        // ── Build LINE_STRIP marker ──
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = frame_id_;
        marker.ns = "firi_polytope";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.scale.x = 0.03;  // line width in meters

        // Green with some transparency
        marker.color.r = 0.2f;
        marker.color.g = 0.9f;
        marker.color.b = 0.2f;
        marker.color.a = 0.8f;

        marker.pose.orientation.w = 1.0;

        // Add vertices + close the loop
        for (const auto& v : vertices) {
            geometry_msgs::msg::Point p;
            p.x = v.x();
            p.y = v.y();
            p.z = 0.05;  // slightly above ground for visibility
            marker.points.push_back(p);
        }
        // Close the polygon
        if (!vertices.empty()) {
            geometry_msgs::msg::Point p;
            p.x = vertices.front().x();
            p.y = vertices.front().y();
            p.z = 0.05;
            marker.points.push_back(p);
        }

        viz_pub_->publish(marker);
    }

    // ================================================================
    // MEMBER VARIABLES
    // ================================================================

    // Subscriptions
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Publishers
    rclcpp::Publisher<firi_msgs::msg::Polytope2DStamped>::SharedPtr poly_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viz_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // Solver
    firi::FIRISolver solver_;

    // Robot state
    Eigen::Vector2d robot_pos_ = Eigen::Vector2d::Zero();
    double robot_yaw_ = 0.0;
    bool odom_received_ = false;

    // Cached obstacle data
    std::vector<Eigen::Vector2d> all_boundary_obs_;
    bool grid_cached_ = false;
    std::string frame_id_ = "map";

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
};

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    // Create the node and spin. In ROS2, spin() processes all callbacks
    // (subscriptions + timers) on the current thread. This is equivalent
    // to ros::spin() in ROS1, but the timer_callback replaces the
    // explicit while-loop + rate.sleep() pattern.
    auto node = std::make_shared<FIRIGridNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}