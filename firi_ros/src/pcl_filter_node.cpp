// ============================================================================
// pcl_filter_node.cpp — PointCloud2 filter + LaserScan projection (ROS2 Foxy)
//
// Subscribes to a 3D point cloud (e.g. /ouster/points), filters by height
// and range, downsamples with a voxel grid, and publishes:
//   - /filtered_cloud  (PointCloud2, map frame)
//   - /filtered_scan   (LaserScan, base_link frame)
//
// Uses two TF lookups:
//   - map → sensor_frame:  to transform the cloud into map frame
//   - map → base_link:     for robot pose (filtering center, scan projection)
//
// Filters:
//   1. Height (relative to base_link z)
//   2. Range (relative to base_link xy)
//   3. Footprint exclusion (rectangular zone in body frame)
//   4. Voxel downsample
//
// No pcl_ros dependency — transforms are done via tf2 + Eigen.
// ============================================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

using PointT = pcl::PointXYZ;


class PCLFilterNode : public rclcpp::Node
{
public:
    PCLFilterNode() : Node("pcl_filter_node")
    {
        // ── Parameters ──
        this->declare_parameter<double>("voxel_size", 0.1);
        this->declare_parameter<double>("height_min", -0.5);
        this->declare_parameter<double>("height_max", 2.0);
        this->declare_parameter<double>("range_min", 0.1);
        this->declare_parameter<double>("range_max", 10.0);
        this->declare_parameter<int>("num_beams", 720);
        this->declare_parameter<std::string>("scan_frame", "base_link");
        this->declare_parameter<std::string>("cloud_topic", "/ouster/points");

        // Robot footprint exclusion zone
        this->declare_parameter<double>("robot_length", 0.55);
        this->declare_parameter<double>("robot_width", 0.35);
        this->declare_parameter<double>("footprint_offset_x", 0.215);
        this->declare_parameter<double>("footprint_margin", 0.00);

        voxel_size_ = this->get_parameter("voxel_size").as_double();
        height_min_ = this->get_parameter("height_min").as_double();
        height_max_ = this->get_parameter("height_max").as_double();
        range_min_  = this->get_parameter("range_min").as_double();
        range_max_  = this->get_parameter("range_max").as_double();
        num_beams_  = this->get_parameter("num_beams").as_int();
        scan_frame_ = this->get_parameter("scan_frame").as_string();
        robot_length_        = this->get_parameter("robot_length").as_double();
        robot_width_         = this->get_parameter("robot_width").as_double();
        footprint_offset_x_  = this->get_parameter("footprint_offset_x").as_double();
        footprint_margin_    = this->get_parameter("footprint_margin").as_double();

        std::string cloud_topic = this->get_parameter("cloud_topic").as_string();

        // ── TF2 ──
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ── Publishers ──
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/filtered_cloud", 10);
        scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
            "/filtered_scan", 10);

        // ── Subscriber ──
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic, rclcpp::SensorDataQoS(),
            std::bind(&PCLFilterNode::cloud_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "PCL Filter Node started");
        RCLCPP_INFO(this->get_logger(),
            "  Voxel: %.2f m, Height: [%.2f, %.2f] m, Range: [%.2f, %.2f] m",
            voxel_size_, height_min_, height_max_, range_min_, range_max_);
        RCLCPP_INFO(this->get_logger(),
            "  Footprint exclusion: %.3f x %.3f m, offset_x: %.4f m, margin: %.3f m",
            robot_length_, robot_width_, footprint_offset_x_, footprint_margin_);
        RCLCPP_INFO(this->get_logger(),
            "  LaserScan: %d beams, frame: '%s'",
            num_beams_, scan_frame_.c_str());
        RCLCPP_INFO(this->get_logger(),
            "  Cloud topic: '%s'", cloud_topic.c_str());
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // ── Get transform: sensor frame → map (for cloud transform) ──
        geometry_msgs::msg::TransformStamped tf_sensor;
        try {
            tf_sensor = tf_buffer_->lookupTransform(
                "map", msg->header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(0.1));
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "TF map->%s failed: %s", msg->header.frame_id.c_str(), ex.what());
            return;
        }

        // ── Get transform: base_link → map (for filtering + scan projection) ──
        geometry_msgs::msg::TransformStamped tf_base;
        try {
            tf_base = tf_buffer_->lookupTransform(
                "map", "base_link",
                tf2::TimePointZero,
                tf2::durationFromSec(0.1));
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "TF map->base_link failed: %s", ex.what());
            return;
        }

        // ── Build Eigen transform for cloud (sensor → map) ──
        Eigen::Affine3f sensor_transform = Eigen::Affine3f::Identity();
        {
            const auto& t = tf_sensor.transform.translation;
            const auto& r = tf_sensor.transform.rotation;
            Eigen::Quaternionf q(r.w, r.x, r.y, r.z);
            sensor_transform.translation() = Eigen::Vector3f(t.x, t.y, t.z);
            sensor_transform.linear() = q.toRotationMatrix();
        }

        // ── Robot (base_link) pose in map frame ──
        const float robot_x = tf_base.transform.translation.x;
        const float robot_y = tf_base.transform.translation.y;
        const float robot_z = tf_base.transform.translation.z;

        const auto& r_base = tf_base.transform.rotation;
        const float robot_yaw = std::atan2(
            2.0 * (r_base.w * r_base.z + r_base.x * r_base.y),
            1.0 - 2.0 * (r_base.y * r_base.y + r_base.z * r_base.z));

        // ── Convert ROS msg → PCL ──
        pcl::PointCloud<PointT>::Ptr cloud_in(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*msg, *cloud_in);

        if (cloud_in->empty()) return;

        // ── Transform to map frame ──
        pcl::PointCloud<PointT>::Ptr cloud_map(new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*cloud_in, *cloud_map, sensor_transform);

        // ── Filter by height (relative to base_link z) ──
        pcl::PointCloud<PointT>::Ptr cloud_height(new pcl::PointCloud<PointT>());
        pcl::PassThrough<PointT> pass_z;
        pass_z.setInputCloud(cloud_map);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(robot_z + height_min_, robot_z + height_max_);
        pass_z.filter(*cloud_height);

        // ── Filter by range + footprint exclusion ──
        //
        // For each point that survived the height filter:
        //   1. Reject if outside [range_min, range_max] from base_link
        //   2. Transform delta into body frame (centered on body_center)
        //   3. Reject if inside footprint rectangle + margin
        //
        // The footprint rectangle is centered on body_center, which is
        // footprint_offset_x ahead of base_link (rear axle). This matches
        // the FIRI seed construction.

        const float cos_yaw = std::cos(robot_yaw);
        const float sin_yaw = std::sin(robot_yaw);

        // Body center in map frame
        const float body_cx = robot_x + footprint_offset_x_ * cos_yaw;
        const float body_cy = robot_y + footprint_offset_x_ * sin_yaw;

        // Exclusion half-extents (with margin)
        const float excl_hl = robot_length_ / 2.0f + footprint_margin_;
        const float excl_hw = robot_width_  / 2.0f + footprint_margin_;

        const float range_min_sq = range_min_ * range_min_;
        const float range_max_sq = range_max_ * range_max_;

        pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
        cloud_filtered->reserve(cloud_height->size());

        for (const auto& pt : cloud_height->points) {
            // Range check from base_link
            const float dx = pt.x - robot_x;
            const float dy = pt.y - robot_y;
            const float r2 = dx * dx + dy * dy;
            if (r2 < range_min_sq || r2 > range_max_sq) continue;

            // Footprint exclusion: transform into body frame
            const float dx_bc = pt.x - body_cx;
            const float dy_bc = pt.y - body_cy;
            const float bx =  dx_bc * cos_yaw + dy_bc * sin_yaw;
            const float by = -dx_bc * sin_yaw + dy_bc * cos_yaw;

            if (std::abs(bx) < excl_hl && std::abs(by) < excl_hw) continue;

            cloud_filtered->push_back(pt);
        }

        // ── Voxel downsample ──
        pcl::PointCloud<PointT>::Ptr cloud_ds(new pcl::PointCloud<PointT>());
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud_filtered);
        voxel.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        voxel.filter(*cloud_ds);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Cloud: %zu raw -> %zu height -> %zu range+footprint -> %zu voxel",
            cloud_in->size(), cloud_height->size(),
            cloud_filtered->size(), cloud_ds->size());

        // ── Publish filtered PointCloud2 ──
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_ds, output);
        output.header.stamp = msg->header.stamp;
        output.header.frame_id = "map";
        cloud_pub_->publish(output);

        // ── Publish LaserScan ──
        publish_laser_scan(cloud_ds, robot_x, robot_y, robot_yaw,
                           msg->header.stamp);
    }

    // ================================================================
    // PROJECT 3D CLOUD TO 2D LASERSCAN
    // ================================================================
    void publish_laser_scan(
        const pcl::PointCloud<PointT>::Ptr& cloud,
        float robot_x, float robot_y, float robot_yaw,
        const builtin_interfaces::msg::Time& stamp)
    {
        sensor_msgs::msg::LaserScan scan;
        scan.header.stamp = stamp;
        scan.header.frame_id = scan_frame_;
        scan.angle_min = -M_PI;
        scan.angle_max =  M_PI;
        scan.angle_increment = 2.0 * M_PI / num_beams_;
        scan.range_min = range_min_;
        scan.range_max = range_max_;
        scan.time_increment = 0.0;
        scan.scan_time = 0.0;
        scan.ranges.assign(num_beams_, std::numeric_limits<float>::infinity());

        for (const auto& pt : cloud->points) {
            const float dx = pt.x - robot_x;
            const float dy = pt.y - robot_y;
            const float range = std::sqrt(dx * dx + dy * dy);

            if (range < range_min_ || range > range_max_) continue;

            float angle = std::atan2(dy, dx) - robot_yaw;

            while (angle >= M_PI)  angle -= 2.0f * M_PI;
            while (angle < -M_PI) angle += 2.0f * M_PI;

            int idx = static_cast<int>((angle - scan.angle_min) / scan.angle_increment);
            if (idx < 0 || idx >= num_beams_) continue;

            if (range < scan.ranges[idx]) {
                scan.ranges[idx] = range;
            }
        }

        scan_pub_->publish(scan);
    }

    // ================================================================
    // MEMBER VARIABLES
    // ================================================================

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;

    // Parameters
    double voxel_size_;
    double height_min_;
    double height_max_;
    double range_min_;
    double range_max_;
    int num_beams_;
    std::string scan_frame_;
    double robot_length_;
    double robot_width_;
    double footprint_offset_x_;
    double footprint_margin_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLFilterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}