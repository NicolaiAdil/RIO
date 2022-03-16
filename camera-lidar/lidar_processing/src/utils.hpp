#include <vector>

struct PtCloud {
  struct Point {
    float x, y, z, distance;
    bool visited;

    Point(float x, float y, float z, float distance)
        : x(x), y(y), z(z), distance(distance), visited(false) {}
  };

  std::vector<Point> pts;

  inline std::size_t kdtree_get_point_count() const { return pts.size(); }

  inline double kdtree_get_pt(const std::size_t idx,
                              const std::size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};