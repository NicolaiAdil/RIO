#include "pcl_ros/point_cloud.h"
#include "utils.hpp"

class DBScan {
public:
  static pcl::PointCloud<pcl::PointXYZ> cluster(PtCloud &kd_cloud);
};