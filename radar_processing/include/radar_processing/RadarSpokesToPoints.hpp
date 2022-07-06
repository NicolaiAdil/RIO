#ifndef RADAR_SPOKES_TO_POINTS_H
#define RADAR_SPOKES_TO_POINTS_H

//////////////////////////
// FROM AUTOSEA PROJECT //
//////////////////////////

#include <ros/ros.h>
#include <string>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <vector>

#include <custom_msgs/RadarSpoke.h>
#include <geometry_msgs/Point32.h>
#include <pcl_ros/point_cloud.h>

class RadarPointBuffer {
public:
  RadarPointBuffer();
  void add_point(const geometry_msgs::Point32 &point, const uint8_t &intensity);
  void add_buffer(const RadarPointBuffer &buffer);

  void clear() {
    point_vector.clear();
    intensity_vector.clear();
  };
  const std::vector<geometry_msgs::Point32> &get_points() const {
    return point_vector;
  };
  const std::vector<uint8_t> &get_intensities() const {
    return intensity_vector;
  };

private:
  std::vector<geometry_msgs::Point32> point_vector;
  std::vector<uint8_t> intensity_vector;
};

class SpokesToPoints {
public:
  SpokesToPoints(ros::NodeHandle &nh_in);

private:
  void spoke_callback(const custom_msgs::RadarSpoke::ConstPtr &spoke_ptr);
  void project_points(const custom_msgs::RadarSpoke::ConstPtr &spoke_ptr,
                      RadarPointBuffer *point_buffer);
  void radar_to_body(const float &r, const float &cos_az, const float &sin_az,
                     geometry_msgs::Point32 *pt_out);
  void transform_point(const tf2::Stamped<tf2::Transform> &transform,
                       geometry_msgs::Point32 *pt);
  void rgb_point(const geometry_msgs::Point32 &pt_in, const uint8_t &intensity,
                 pcl::PointXYZRGB *pt_out);

  // Radar specific config parameters
  double config_min_range;
  double config_max_range;
  double current_min_range;
  double current_max_range;
  double start_azimuth;
  double last_azimuth;
  double bearing_offset;
  int min_intensity;
  int max_intensity;
  int scan_intensity_threshold;
  bool transform_data;

  // Helper variables to keep track of radar revolution
  bool full_rev;
  bool zero_crossed;

  std::string projection_frame_id;
  std::string output_frame_id;

  ros::NodeHandle nh;
  ros::Publisher pcl_pub;
  ros::Subscriber spoke_sub;
  tf2_ros::Buffer buffer;
  tf2_ros::TransformListener tf_listener;
  RadarPointBuffer scan_buffer;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud;
};

#endif // RADAR_SPOKES_TO_POINTS_H
