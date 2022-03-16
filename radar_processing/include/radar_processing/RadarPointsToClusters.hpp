#ifndef RADAR_POINTS_TO_CLUSTERS_H
#define RADAR_POINTS_TO_CLUSTERS_H

#include "ros/ros.h"

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_ros/point_cloud.h>

#include "custom_msgs/RadarCluster.h"
#include "custom_msgs/RadarScan.h"
#include "geometry_msgs/PolygonStamped.h"
#include "jsk_recognition_msgs/PolygonArray.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::search::KdTree<pcl::PointXYZ> KdTree;

class PclClustering {
public:
  PclClustering(ros::NodeHandle nh_in);

private:
  ros::NodeHandle nh;
  ros::Subscriber pcl_sub;
  ros::Publisher hulls_pub;
  ros::Publisher scan_pub;

  int pub_seq;
  std::string output_frame;

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclidean_clustering;

  void pcl_cb(const PointCloud::ConstPtr &cloud);
  void cloud_to_hull(const PointCloud::Ptr input_cloud,
                     PointCloud &output_hull);
  void cloud_to_centroid(const PointCloud::Ptr input_cloud,
                         geometry_msgs::Point &output_point);
  void pcl_hull_to_polygon(const PointCloud &input_hull,
                           geometry_msgs::PolygonStamped &output_polygon);
};

#endif // RADAR_POINTS_TO_CLUSTERS_H