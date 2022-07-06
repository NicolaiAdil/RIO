#include "radar_processing/RadarPointsToClusters.hpp"

tf2_ros::Buffer buffer; 
tf2_ros::TransformListener *pListener = NULL;
Landmap *landmap = NULL;
pcl::PointCloud<pcl::PointXYZ> *land_cloud = NULL;

PclClustering::PclClustering(ros::NodeHandle nh_in) : nh{nh_in}, pub_seq{0} {
  // Change Timestamp to gps-time during normal run // Not documented how to :( 
  ros::Time pc2Stamp = ros::Time::now();

  std::string point_cloud_topic, hulls_topic, scan_topic;
  nh.param<std::string>("/topics/hardware/furuno_radar/point_cloud",
                        point_cloud_topic, "/radar/points");
  nh.param<std::string>("/topics/hardware/furuno_radar/hulls", hulls_topic,
                        "/radar/hulls");
  nh.param<std::string>("/topics/hardware/furuno_radar/scan", scan_topic,
                        "/radar/scan");

  if (!nh.getParam("/sensors/radar/frames/output", input_frame_id)) {
    input_frame_id = "radar"; // NOTE: This is the OUTPUT from the 
                              // spokes_to_points node, which is our INPUT
    ROS_WARN_STREAM(
        "No input frame specified, points input in: " << input_frame_id);
    input_frame_id = input_frame_id;
  } else {
    ROS_INFO_STREAM("Input frame set to: " << input_frame_id);
  }
  
  if (!nh.getParam("/sensors/radar/frames/output", output_frame_id)) {
    output_frame_id = "body";
    ROS_WARN_STREAM(
        "No cluster output frame specified, points input in: " << output_frame_id);
  } else {
    ROS_INFO_STREAM("Cluster output frame set to: " << output_frame_id);
  }


  pcl_sub = nh.subscribe<PointCloud>(point_cloud_topic, 1,
                                     &PclClustering::pcl_cb, this);
  hulls_pub = nh.advertise<jsk_recognition_msgs::PolygonArray>(hulls_topic, 10);
  scan_pub = nh.advertise<autosea_msgs::RadarScan>(scan_topic, 10);

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
  ros::Time pc2Stamp = ros::Time::now();
  pcl::PointCloud<pcl::PointXYZ> ned_cloud;
  pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
  pcl::PointCloud<pcl::PointXYZ> output_cloud;


  try{

    buffer.canTransform("fixed", input_frame_id, pc2Stamp, ros::Duration(1.0));

    const std::string target_frame = "ned";
    pcl_ros::transformPointCloud(target_frame, *cloud, ned_cloud, buffer);
    for (const auto& point: ned_cloud.points){
      if (landmap->isLand(point.x, point.y)){
        filtered_cloud.push_back(point);
      }
    }

    // Transform the land-filtered cloud to our output frame.  
    pcl_ros::transformPointCloud(output_frame_id, filtered_cloud, output_cloud, buffer);

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr temp_cloud (&output_cloud);

    KdTree::Ptr tree{new KdTree};
    tree->setInputCloud(temp_cloud);
    euclidean_clustering.setSearchMethod(tree);
    euclidean_clustering.setInputCloud(temp_cloud);

    // Extract clusters from the filtered cloud in output frame and save their indices. 
    // cluster_indices[i] contains an array of each point in the original cloud that 
    // correspond to the i'th cluster.

    std::vector<pcl::PointIndices> cluster_indices;
    euclidean_clustering.extract(cluster_indices);

    std_msgs::Header common_header;
    common_header.seq = pub_seq;
    common_header.stamp = ros::Time::now();
    common_header.frame_id = output_frame_id;

    jsk_recognition_msgs::PolygonArray hulls;
    autosea_msgs::RadarScan scan;
    for (auto cit = cluster_indices.begin(); cit != cluster_indices.end();
        ++cit) {

      PointCloud::Ptr cluster{new PointCloud};
      for (const auto &idx :
          cit->indices) { // Fetch the points from the output cloud that
                          // correspond to the i'th cluster
        cluster->push_back((*temp_cloud)[idx]);
      }

      PointCloud hull;
      cloud_to_hull(cluster, hull);
      geometry_msgs::PolygonStamped single_polygon;
      pcl_hull_to_polygon(hull, single_polygon);
      single_polygon.header = common_header;
      hulls.polygons.push_back(single_polygon);

      autosea_msgs::RadarCluster cl;
      cl.header = common_header;
      // cl.type = autosea_msgs::RadarCluster::EXTENDED;
      cl.type = autosea_msgs::RadarCluster::POINT;
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
  catch (tf2::TransformException &ex){
    ROS_WARN("%s", ex.what());
  }
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

  pListener = new tf2_ros::TransformListener(buffer);
  landmap->initialize(refLat, refLon);
  ros::spin();
  return 0;
}
