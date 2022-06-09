#include "autosea_msgs/RadarCluster.h"
#include "autosea_msgs/RadarScan.h"
#include "ros/ros.h"

#include "geometry_msgs/TransformStamped.h"
#include "pcl/point_types.h"
#include "pcl_ros/point_cloud.h"
#include "pcl_ros/transforms.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl_conversions/pcl_conversions.h>
#include "tf/transform_listener.h"
#include <cmath>

#include "dbscan.hpp"
#include "landmap.h"
#include "nanoflann.hpp"

tf::TransformListener *pListener = NULL;

ros::Publisher pub;
ros::Publisher pubtest;
ros::Publisher pubinit;
ros::Publisher pubtrans;
ros::Publisher pubtrans2;
ros::Publisher pubfiltered;
ros::Publisher pubcentroids;
Landmap *landmap = NULL;
pcl::PointCloud<pcl::PointXYZ> *land_cloud = NULL;

// Publisher for autosea radarscan/lidarscan
ros::Publisher scanpub;

double filtered_points = 0.0;
double total_points = 0.0;
double iterator = 0.0;
double avg_centroid = 0.0;

void velodyne_points_cb(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input) {

  // Change Timestamp to gps-time during normal run
  ros::Time pc2Stamp = ros::Time::now();
  geometry_msgs::TransformStamped transformStamped;
  tf::StampedTransform transform;
  tf::StampedTransform transform2;
  tf::StampedTransform fixed_ned_transform;
  tf::StampedTransform test_transform;
  pcl::PointCloud<pcl::PointXYZI> fixed_cloud;
  pcl::PointCloud<pcl::PointXYZI> ned_cloud;
  pcl::PointCloud<pcl::PointXYZI> init_cloud;
  pcl::PointCloud<pcl::PointXYZI> trans_cloud;
  pcl::PointCloud<pcl::PointXYZI> trans_cloud2;
  pcl::PointCloud<pcl::PointXYZI> fixed_filtered_cloud;
  pcl::PointCloud<pcl::PointXYZ> lidar_ned_pt;
  pcl::PointCloud<pcl::PointXYZ> centroids_fixed;
  pcl::PointCloud<pcl::PointXYZ> test_2loud;
  double pc2Stamp2 = ros::Time::now().toSec();
  double tnow;

  try {
    pListener->waitForTransform("fixed", "lidar", pc2Stamp, ros::Duration(1.0));
    tnow = ros::Time::now().toSec();
    pListener->lookupTransform("fixed", "lidar", pc2Stamp, transform);
    pListener->lookupTransform("ned", "lidar", pc2Stamp, transform2);
    pListener->lookupTransform("fixed", "ned", pc2Stamp, fixed_ned_transform);
    pListener->lookupTransform("body", "body", pc2Stamp, test_transform);

    double tlookup = ros::Time::now().toSec();
    pcl_ros::transformPointCloud(*input, fixed_cloud, transform);
    fixed_cloud.header.frame_id = "fixed";
    fixed_cloud.header.stamp = input->header.stamp;
    fixed_cloud.header.seq = input->header.seq;

    double ttrans = ros::Time::now().toSec();

    pcl::PointCloud<pcl::PointXYZ> lidar_pt;
    lidar_pt.header.frame_id = "ned";
    lidar_pt.header.stamp = input->header.stamp;
    lidar_pt.header.seq = input->header.seq;
    lidar_pt.height = 1;
    lidar_pt.width = 1;
    lidar_pt.points.emplace_back(pcl::PointXYZ(0.0, 0.0, 0.0));
    pcl_ros::transformPointCloud(lidar_pt, lidar_ned_pt, transform2);
    float lidar_n = lidar_ned_pt.points[0].x;
    float lidar_e = lidar_ned_pt.points[0].y;

    pcl_ros::transformPointCloud(*input, ned_cloud, transform2);
    ned_cloud.header.frame_id = "ned";
    ned_cloud.header.stamp = input->header.stamp;
    ned_cloud.header.seq = input->header.seq;

    int counter = 0;
    double fstart = ros::Time::now().toSec();
    PtCloud kd_cloud;

    for (auto i = 0; i < ned_cloud.size(); i++) {
      if (!landmap->isLand(ned_cloud.points[i].x, ned_cloud.points[i].y)) {
        float nrel = ned_cloud.points[i].x - lidar_n;
        float erel = ned_cloud.points[i].y - lidar_e;
        kd_cloud.pts.emplace_back(ned_cloud.points[i].x, ned_cloud.points[i].y,
                                  ned_cloud.points[i].z,
                                  0.5 * log(sqrt(pow(nrel, 2) + pow(erel, 2))));
        counter += 1;
      }
    }
    double fstop = ros::Time::now().toSec();
    double nstart = ros::Time::now().toSec();
    pcl::PointCloud<pcl::PointXYZ> centroids = DBScan::cluster(kd_cloud);
    double nstop = ros::Time::now().toSec();
    pub.publish(fixed_cloud);
    filtered_points += counter;
    total_points += ned_cloud.size();
    avg_centroid += centroids.size();
    iterator += 1;

    // Message for PDAF-tracker(autosea)
    autosea_msgs::RadarScan scanMsg;
    ros::Time rosTimeStamp;
    for (auto i = 0; i < centroids.size(); i++) {
      autosea_msgs::RadarCluster msg;
      msg.header.frame_id = "ned";
      pcl_conversions::fromPCL(input->header.stamp, rosTimeStamp);
      msg.header.stamp = rosTimeStamp;
      msg.type = 0; // point

      msg.centroid.x = centroids.points[i].x;
      msg.centroid.y = centroids.points[i].y;
      // z is set to zero in DBScan::cluster
      msg.centroid.z = centroids.points[i].z;

      scanMsg.radar_scan.push_back(msg);
    }
    scanpub.publish(scanMsg);

    pcl_ros::transformPointCloud(centroids, centroids_fixed,
                                 fixed_ned_transform);
    centroids_fixed.header.frame_id = "fixed";
    centroids_fixed.header.stamp = input->header.stamp;
    centroids_fixed.header.seq = input->header.seq;
    pubcentroids.publish(centroids_fixed);

    ros::Time endtime = ros::Time::now();

    double total_clustering_time = endtime.toSec() - pc2Stamp.toSec();

    pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
    filtered_cloud.header.frame_id = "fixed";
    filtered_cloud.height = 1;
    for (auto i = 0; i < ned_cloud.size(); i++) {
      if (!landmap->isLand(ned_cloud.points[i].x, ned_cloud.points[i].y)) {
        filtered_cloud.points.push_back(ned_cloud.points[i]);
        filtered_cloud.width += 1;
      }
    }
    pcl_ros::transformPointCloud(filtered_cloud, fixed_filtered_cloud,
                                 fixed_ned_transform);
    pubfiltered.publish(fixed_filtered_cloud);

  } catch (tf::TransformException &ex) {
    ROS_WARN("%s", ex.what());
  }
  pubtest.publish(*land_cloud);
  double tstop = ros::Time::now().toSec();
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "LidarClustering");
  ros::NodeHandle nh;
  landmap = new Landmap();

  double refLat, refLon;
  std::string location, frame_type;
  nh.getParam("/world/location", location);
  nh.getParam("/world/frame", frame_type);

  std::string world_frame = "/" + location + "_" + frame_type;
  nh.getParam(world_frame + "/lat0", refLat);
  nh.getParam(world_frame + "/lon0", refLon);

  landmap->initialize(refLat, refLon);

  land_cloud = new pcl::PointCloud<pcl::PointXYZ>();
  land_cloud->header.frame_id = "fixed";
  land_cloud->height = 1;
  for (int n_int = -1000; n_int < 1000; n_int++) {
    for (int e_int = -1000; e_int < 1000; e_int++) {
      double n = double(n_int) * 0.5;
      double e = double(e_int) * 0.5;
      if (landmap->isLand(n, e)) {
        land_cloud->points.push_back(pcl::PointXYZ(e, n, 0.0));
        land_cloud->width += 1;
      }
    }
  }
  ROS_INFO_STREAM("landmap done");

  scanpub = nh.advertise<autosea_msgs::RadarScan>("lidar_centroids", 1);
  pub = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("transformed_cloud", 1);
  pubtest = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("land_points", 1);
  pubinit = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("init_cloud", 1);
  pubtrans = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("trans_cloud", 1);
  pubfiltered =
      nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("filtered_cloud", 1);
  pubtrans2 = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("trans_cloud2", 1);
  pubcentroids =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ>>("centroid_cloud", 1);
  pListener = new tf::TransformListener(ros::Duration(3.0));
  ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZI>>(
      "velodyne_points", 10, velodyne_points_cb);
  ros::spin();
}
