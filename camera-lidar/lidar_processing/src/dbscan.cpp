#include "dbscan.hpp"
#include "nanoflann.hpp"
#include "pcl/point_types.h"
#include "ros/ros.h"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <queue>

using namespace boost::geometry;

typedef boost::geometry::model::d2::point_xy<float> xypoint;
typedef boost::geometry::model::polygon<xypoint> polygon;

pcl::PointCloud<pcl::PointXYZ> DBScan::cluster(PtCloud &kd_cloud) {
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PtCloud>, PtCloud, 2>
      my_kd_tree_t;
  my_kd_tree_t index(2, kd_cloud, nanoflann::KDTreeSingleIndexAdaptorParams());
  index.buildIndex();

  std::vector<std::pair<std::size_t, float>> ret_matches;
  std::vector<polygon> clusters;
  float queryPoint[2];
  nanoflann::SearchParams params;
  params.sorted = false;

  std::queue<int> neigh;
  int counter = 0;
  int reserveSize = 50;

  for (auto i = 0; i < kd_cloud.pts.size(); i++) {
    if (!kd_cloud.pts[i].visited) {
      ret_matches.clear();
      ret_matches.reserve(reserveSize);
      queryPoint[0] = kd_cloud.pts[i].x;
      queryPoint[1] = kd_cloud.pts[i].y;

      const std::size_t nMatches = index.radiusSearch(
          queryPoint, pow(kd_cloud.pts[i].distance, 2), ret_matches, params);
      if (nMatches == 1) {
        kd_cloud.pts[i].visited = true;
        counter += 1;
      } else if (nMatches > 3) {
        kd_cloud.pts[i].visited = true;
        clusters.emplace_back(polygon());
        append(clusters.back(), xypoint(queryPoint[0], queryPoint[1]));
        for (int j = 0; j < nMatches; j++) {
          std::size_t idx = ret_matches[j].first;
          if (!kd_cloud.pts[idx].visited) {
            neigh.push(idx); // insert neighbors in the neighbors list
          }
        }
        while (!neigh.empty()) {
          std::size_t id = neigh.front();
          neigh.pop();

          if (!kd_cloud.pts[id].visited) {
            queryPoint[0] = kd_cloud.pts[id].x;
            queryPoint[1] = kd_cloud.pts[id].y;
            kd_cloud.pts[id].visited = true;
            append(clusters.back(), xypoint(queryPoint[0], queryPoint[1]));

            ret_matches.clear();
            ret_matches.reserve(reserveSize);

            // find the number of neighbors of current processed neighbor point
            const std::size_t nMatches = index.radiusSearch(
                queryPoint, pow(kd_cloud.pts[id].distance, 2), ret_matches,
                params);
            if (nMatches > 3) {
              for (int j = 0; j < nMatches; j++) {
                size_t idx = ret_matches[j].first;
                if (!kd_cloud.pts[idx].visited) {
                  neigh.push(idx); // insert neighbors in the neighbors list
                }
              }
            }
          }
        }
      }
    }
  }
  pcl::PointCloud<pcl::PointXYZ> centroids;
  centroids.header.frame_id = "ned";
  centroids.height = 1;
  centroids.width = 0;
  for (auto i = 0; i < clusters.size(); i++) {
    polygon hull;
    convex_hull(clusters[i], hull);
    xypoint p;
    centroid(hull, p);
    centroids.points.emplace_back(pcl::PointXYZ(p.x(), p.y(), 0.0));
    centroids.width += 1;
  }
  return centroids;
}