#include "radar_processing/RadarPointsToClusters.hpp"

PclClustering::PclClustering(ros::NodeHandle nh_in) : nh{nh_in}, pub_seq{0} {
  std::string point_cloud_topic, hulls_topic, scan_topic;
  nh.param<std::string>("/topics/hardware/furuno_radar/point_cloud",
                        point_cloud_topic, "/radar/points");
  nh.param<std::string>("/topics/hardware/furuno_radar/hulls", hulls_topic,
                        "/radar/hulls");
  nh.param<std::string>("/topics/hardware/furuno_radar/scan", scan_topic,
                        "/radar/scan");

  pcl_sub = nh.subscribe<PointCloud>(point_cloud_topic, 1,
                                     &PclClustering::pcl_cb, this);
  hulls_pub = nh.advertise<jsk_recognition_msgs::PolygonArray>(hulls_topic, 10);
  scan_pub = nh.advertise<custom_msgs::RadarScan>(scan_topic, 10);

  nh.param<std::string>("/sensors/radar/frames/output", output_frame, "body");

  double tolerance;
  int min_cluster_size, max_cluster_size;
  nh.param<double>("/tracking/radar/clustering/tolerance", tolerance, 1.0);
  nh.param<int>("/tracking/radar/clustering/min_size", min_cluster_size, 10);
  nh.param<int>("/tracking/radar/clustering/max_size", max_cluster_size, 1000);

  euclidean_clustering.setClusterTolerance(tolerance);
  euclidean_clustering.setMinClusterSize(min_cluster_size);
  euclidean_clustering.setMaxClusterSize(max_cluster_size);
}

void PclClustering::pcl_cb(const PointCloud::ConstPtr &cloud) {
  KdTree::Ptr tree{new KdTree};
  tree->setInputCloud(cloud);
  euclidean_clustering.setSearchMethod(tree);
  euclidean_clustering.setInputCloud(cloud);

  // Extract clusters from the cloud and save their indices. cluster_indices[i]
  // contains an array of each point in the original cloud that correspond to
  // the i'th cluster.
  std::vector<pcl::PointIndices> cluster_indices;
  euclidean_clustering.extract(cluster_indices);

  std_msgs::Header common_header;
  common_header.seq = pub_seq;
  common_header.stamp = ros::Time::now();
  common_header.frame_id = output_frame;

  jsk_recognition_msgs::PolygonArray hulls;
  custom_msgs::RadarScan scan;
  for (auto cit = cluster_indices.begin(); cit != cluster_indices.end();
       ++cit) {

    PointCloud::Ptr cluster{new PointCloud};
    for (const auto &idx :
         cit->indices) { // Fetch the points from the original cloud that
                         // correspond to the i'th cluster
      cluster->push_back((*cloud)[idx]);
    }

    PointCloud hull;
    cloud_to_hull(cluster, hull);
    geometry_msgs::PolygonStamped single_polygon;
    pcl_hull_to_polygon(hull, single_polygon);
    single_polygon.header = common_header;
    hulls.polygons.push_back(single_polygon);

    custom_msgs::RadarCluster cl;
    cl.header = common_header;
    // cl.type = custom_msgs::RadarCluster::EXTENDED;
    cl.type = custom_msgs::RadarCluster::POINT;
    geometry_msgs::Point centroid;
    cloud_to_centroid(cluster, centroid);
    cl.centroid = centroid;
    cl.n_points = 1;
    // cl.hull = single_polygon.polygon;
    scan.radar_scan.push_back(cl);
  }

  hulls.header = common_header;
  hulls_pub.publish(hulls);

  scan_pub.publish(scan);

  pub_seq++;
}

void PclClustering::cloud_to_hull(const PointCloud::Ptr input_cloud,
                                  PointCloud &output_cloud) {
  pcl::ConvexHull<pcl::PointXYZ> convex_hull;
  convex_hull.setInputCloud(input_cloud);
  convex_hull.reconstruct(output_cloud);
}

void PclClustering::cloud_to_centroid(const PointCloud::Ptr input_cloud,
                                      geometry_msgs::Point &output_point) {
  // Average x and y to find centroid
  double center_x = 0;
  double center_y = 0;
  int num_points = 0;
  for (const auto &point : *input_cloud) {
    center_x += point.x;
    center_y += point.y;
    num_points++;
  }
  center_x /= num_points;
  center_y /= num_points;

  output_point.x = center_x;
  output_point.y = center_y;
  output_point.z = 0;
}

void PclClustering::pcl_hull_to_polygon(
    const PointCloud &input_hull,
    geometry_msgs::PolygonStamped &output_polygon) {
  for (const auto point : input_hull) {
    geometry_msgs::Point32 pt;
    pt.x = point.x;
    pt.y = point.y;
    pt.z = 0;
    output_polygon.polygon.points.push_back(pt);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "points_to_cluster");
  ros::NodeHandle nh;
  PclClustering clustering_node(nh);
  landmap = new Landmap();

  double refLat, refLon;
  std::string location, frame_type;
  nh.getParam("/world/location", location);
  nh.getParam("/world/frame", frame_type);

  std::string world_frame = "/" + location + "_" + frame_type;
  nh.getParam(world_frame + "/lat0", refLat);
  nh.getParam(world_frame + "/lon0", refLon);

  landmap->initialize(refLat, refLon);
  // The following is only used to publish the land_cloud once made.
  // Should already be published when lidar_clustering is running.
  // Consider commenting it out.
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
  
  ros::spin();
  return 0;
}
