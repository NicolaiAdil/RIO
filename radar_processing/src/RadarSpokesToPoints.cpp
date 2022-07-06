//////////////////////////
// FROM AUTOSEA PROJECT //
//////////////////////////

#include <radar_processing/RadarSpokesToPoints.hpp>

#include <cmath>
#include <ros/ros.h>
#include <string>
#include <vector>

#include <custom_msgs/RadarPoints.h>
#include <custom_msgs/RadarSpoke.h>
#include <geometry_msgs/Point32.h>
#include <pcl_ros/point_cloud.h>

RadarPointBuffer::RadarPointBuffer(){};

void RadarPointBuffer::add_point(const geometry_msgs::Point32 &point,
                                 const uint8_t &intensity) {
  point_vector.push_back(point);
  intensity_vector.push_back(intensity);
}

void RadarPointBuffer::add_buffer(const RadarPointBuffer &buffer) {

  const auto &other_points = buffer.get_points();
  const auto &other_intensities = buffer.get_intensities();

  point_vector.insert(point_vector.end(), other_points.begin(),
                      other_points.end());
  intensity_vector.insert(intensity_vector.end(), other_intensities.begin(),
                          other_intensities.end());
}

SpokesToPoints::SpokesToPoints(ros::NodeHandle &nh_in)
    : nh(nh_in), rgb_cloud{new pcl::PointCloud<pcl::PointXYZRGB>}, tf_listener(buffer) {
  
  if (!nh.getParam("/sensors/radar/frames/input", projection_frame_id)) {
    projection_frame_id = "radar";
    ROS_WARN_STREAM("No input frame specified for radar, using "
                    << projection_frame_id);
  } else {
    ROS_INFO_STREAM("Input frame set to: " << projection_frame_id);
  }

  if (!nh.getParam("/sensors/radar/frames/output", output_frame_id)) {
    ROS_WARN_STREAM(
        "No output frame specified, points output in: " << projection_frame_id);
    output_frame_id = projection_frame_id;
  } else {
    ROS_INFO_STREAM("Output frame set to: " << output_frame_id);
  }
  transform_data = (projection_frame_id !=
                    output_frame_id); // Only transform when the input and
                                      // output frames are different

  nh.param<double>("/tracking/radar/parameters/min_range", config_min_range,
                   0.0);
  nh.param<double>("/tracking/radar/parameters/max_range", config_max_range,
                   1000.0);
  nh.param<double>("/tracking/radar/parameters/start_azimuth", start_azimuth,
                   0.0);
  nh.param<double>("/tracking/radar/parameters/bearing_offset", bearing_offset,
                   0.0);

  nh.param<int>("/tracking/radar/parameters/min_intensity", min_intensity, 0);
  nh.param<int>("/tracking/radar/parameters/max_intensity", max_intensity, 15);
  nh.param<int>("/tracking/radar/parameters/scan_intensity_threshold",
                scan_intensity_threshold, -1);

  std::string in_topic, out_pcl_topic;
  nh.param<std::string>("/topics/hardware/furuno_radar/spoke_data", in_topic,
                        "radar_spokes");
  nh.param<std::string>("/topics/hardware/furuno_radar/point_cloud",
                        out_pcl_topic, "radar_pcl");

  spoke_sub =
      nh.subscribe(in_topic, 1000, &SpokesToPoints::spoke_callback, this);
  pcl_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(out_pcl_topic, 1);

  // Values used to check if we have made a full revolution or nit
  full_rev = false;
  zero_crossed = false;
}

void SpokesToPoints::spoke_callback(
    const custom_msgs::RadarSpoke::ConstPtr &spoke_ptr) {
  // Project points
  RadarPointBuffer point_buffer;
  project_points(spoke_ptr, &point_buffer);

  // Add points to full scan buffer
  scan_buffer.add_buffer(point_buffer);

  // For a full rotation to have occured, we need to have crossed the zero-point
  // (i.e. 2pi -> 0) AND to end up with an azimuth greater than or equal to
  // where we started (since angle counts up from zero)
  float azimuth = spoke_ptr->azimuth;
  float total_angle_difference = azimuth - start_azimuth;
  float current_angle_difference = azimuth - last_azimuth;

  if (!zero_crossed) {
    zero_crossed = current_angle_difference < 0;
  }
  full_rev = zero_crossed && (total_angle_difference >= 0.0);

  if (full_rev) {
    // Publish points and clear buffers
    rgb_cloud->header.frame_id = output_frame_id;
    pcl_pub.publish(*rgb_cloud);

    scan_buffer.clear();
    rgb_cloud->clear();

    full_rev = false;
    zero_crossed = false;
  }

  last_azimuth = azimuth; // Store current azimuth for next time
}

void SpokesToPoints::project_points(
    const custom_msgs::RadarSpoke::ConstPtr &spoke_ptr,
    RadarPointBuffer *point_buffer) {
  geometry_msgs::TransformStamped transform;
  tf2::Stamped<tf2::Transform> transf2;
  if (transform_data) {
    try {
      transform = buffer.lookupTransform(output_frame_id, projection_frame_id,
                                spoke_ptr->header.stamp, ros::Duration(0.05));
    } catch (tf2::TransformException e) {
      ROS_WARN_STREAM_THROTTLE(
          5.0, "Failed to lookup transform, discarding points. Exception: "
                   << e.what());
      return;
    }
    tf2::convert(transform, transf2);
  }

  float azimuth = spoke_ptr->azimuth - bearing_offset;
  float dr = spoke_ptr->range_increment;
  float cos_az = cos(azimuth);
  float sin_az = sin(azimuth);
  for (int i = 0; i < spoke_ptr->num_samples; i++) {

    float r = i * dr + spoke_ptr->range_start;
    uint8_t point_intensity = spoke_ptr->intensity[i];

    if (point_intensity > scan_intensity_threshold && r > config_min_range &&
        r < config_max_range) {
      geometry_msgs::Point32 point;
      radar_to_body(r, cos_az, sin_az, &point);
      if (transform_data) {
        transform_point(transf2, &point);
      }
      point_buffer->add_point(point, point_intensity);

      pcl::PointXYZRGB pcl_point;
      rgb_point(point, point_intensity, &pcl_point);
      rgb_cloud->points.push_back(pcl_point);
    }
  }
}

void SpokesToPoints::radar_to_body(const float &r, const float &cos_az,
                                   const float &sin_az,
                                   geometry_msgs::Point32 *pt_out) {
  pt_out->x = r * cos_az;
  pt_out->y = r * sin_az;
  pt_out->z = 0.0;
}

void SpokesToPoints::transform_point(const tf2::Stamped<tf2::Transform> &transform,
                                     geometry_msgs::Point32 *pt) {
  tf2::Vector3 tf_pt = transform(tf2::Vector3(pt->x, pt->y, pt->z));
  pt->x = tf_pt.getX();
  pt->y = tf_pt.getY();
  pt->z = tf_pt.getZ();
}

void SpokesToPoints::rgb_point(const geometry_msgs::Point32 &pt_in,
                               const uint8_t &intensity,
                               pcl::PointXYZRGB *pt_out) {
  // Temporary testing values for colors: three intensity levels
  int R, G, B;

  if (intensity > 0.8 * max_intensity) {
    R = 255;
    G = 0;
    B = 0;
  } else if (intensity > 0.6 * max_intensity) {
    R = 255;
    G = 255;
    B = 0;
  } else if (intensity > 0.4 * max_intensity) {
    R = 0;
    G = 255;
    B = 0;
  } else {
    R = 0;
    G = 0;
    B = 0;
  }

  pt_out->x = pt_in.x;
  pt_out->y = pt_in.y;
  pt_out->z = pt_in.z;
  pt_out->r = R;
  pt_out->g = G;
  pt_out->b = B;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "spoke_to_points");
  ros::NodeHandle nh;
  SpokesToPoints radarToPoints(nh);

  ros::Duration(2.0).sleep(); // allow tf to buffer up some transforms
  ros::spin();

  return 0;
}
