#include "autosea_msgs/OwnShip.h"
#include "cmath"
#include "custom_msgs/ImuSensor.h"
#include "custom_msgs/gnssGGA.h"
#include "custom_msgs/gnssHDT.h"
#include "geometry_msgs/TransformStamped.h"
#include "ros/ros.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/convert.h"
#include "tf2_ros/transform_broadcaster.h"

#define DEG2RAD M_PI / 180.0

// Dorabassenget
// const double refLat = 63.4408 * DEG2RAD;
// const double refLon = 10.4226 * DEG2RAD;

// Pirbadet
// const double refLat = 63.43944 * DEG2RAD;
// const double refLon = 10.39973 * DEG2RAD;

// Transformtest
const double refLat = 63.440575 * DEG2RAD;
const double refLon = 10.413617 * DEG2RAD;

double avg_lat = 0.0;
double avg_lon = 0.0;
double sum_lat = 0.0;
double sum_lon = 0.0;
double lat_counter = 0.0;
double lon_counter = 0.0;

// Rad values
double roll = -10;
double pitch = -10;
double yaw = -10;
double heading =
    -10; // Initial value set low to avoid using heading when not defined.

ros::Publisher ownship_position;

void vectorFix_callback(const custom_msgs::gnssGGA &input) {
  static tf2_ros::TransformBroadcaster br;

  // TODO Fix timestamp from gps // Seems to be done?? summer 2022
  ros::Time ros_time = ros::Time::now();
  ros::Time fixStamp = input.sat_time;

  double lat = input.latitude * DEG2RAD;
  double lon = input.longitude * DEG2RAD;
  double h = input.altitude;

  // Use IMU-value if heading from Hemisphere is not present.
  if (heading < -5 || std::isnan(heading)) {
    heading = yaw;
  }

  double n = (lat - refLat) * 6378137.0 * (1.0 - pow(0.08182, 2)) /
             pow(1.0 - pow(0.08182 * sin(lat), 2), 1.5);
  double e = (lon - refLon) * cos(lat) * 6378137.0 /
             sqrt(1.0 - pow(0.08182 * sin(lat), 2));

  tf2::Quaternion body_bned_q;
  body_bned_q.setRPY(roll, pitch, yaw);

  tf2::Transform body_bned_tf;
  body_bned_tf.setRotation(body_bned_q);
  tf2::Vector3 vec(0, 0, 0);
  body_bned_tf.setOrigin(vec);
  tf2::Transform ned2body = body_bned_tf.inverse();

  // Lever arm compen sation
  tf2::Vector3 cg_gps_body;

  double lever_arm_x = 0.72;
  double lever_arm_y = 0;
  double lever_arm_z = -0.2;

  cg_gps_body.setValue(lever_arm_x, lever_arm_y,
                       lever_arm_z); // from Vegards master

  tf2::Vector3 cg_gps_bned;
  cg_gps_bned = body_bned_tf(cg_gps_body);

  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = fixStamp;
  transformStamped.header.frame_id = "ned";
  transformStamped.child_frame_id = "body";
  transformStamped.transform.translation.x = n - cg_gps_bned.getX();
  transformStamped.transform.translation.y = e - cg_gps_bned.getY();
  transformStamped.transform.translation.z = -h - cg_gps_bned.getZ();

  tf2::Quaternion q; // = ned2body.getRotation();

  q.setRPY(roll, pitch, heading);
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();
  transformStamped.transform.rotation.w = q.w();

  br.sendTransform(transformStamped);

  autosea_msgs::OwnShip ownship_msg;
  ownship_msg.header.frame_id = "ned";
  ownship_msg.header.stamp = fixStamp;
  ownship_msg.point.x = transformStamped.transform.translation.x;
  ownship_msg.point.y = transformStamped.transform.translation.y;
  ownship_msg.point.z = transformStamped.transform.translation.z;

  ownship_position.publish(ownship_msg);
}

void vectorHeading_callback(const custom_msgs::gnssHDT &input) {
  heading = (input.heading) * DEG2RAD;
}

void xsensImu_callback(const custom_msgs::ImuSensor &input) {
  roll = (input.orientation.roll) * DEG2RAD;
  pitch = (input.orientation.pitch) *
          DEG2RAD; // Add +4 to compensate for wrong body
  yaw = (input.orientation.yaw) * DEG2RAD; // In case of heading not working
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "FramePublisher");
  ros::NodeHandle nh;
  ros::Subscriber fix_sub =
      nh.subscribe("vectorVS330/fix", 10, vectorFix_callback);
  ros::Subscriber heading_sub =
      nh.subscribe("vectorVS330/heading", 10, vectorHeading_callback);
  ros::Subscriber imu_sub = nh.subscribe("xsens/imu", 10, xsensImu_callback);
  ownship_position =
      nh.advertise<autosea_msgs::OwnShip>("ownship_position_in_NED", 1);
  ros::spin();
}
