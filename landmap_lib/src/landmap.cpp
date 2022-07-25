#include "landmap_lib/landmap.h"

#define Deg2Rad M_PI / 180.0

// const double refLat = 63.4405479431 ;
// const double refLon = 10.4229335785 ;

double calculate_north(const double &latitude, const double &refLat) {
  double lat = latitude * Deg2Rad;

  return (lat - refLat) * 6378137.0 * (1.0 - pow(0.08182, 2)) /
         pow(1.0 - pow(0.08182 * sin(lat), 2), 1.5);
}

double calculate_east(const double &latitude, const double &longitude,
                      const double &refLon) {
  double lat = latitude * Deg2Rad;
  double lon = longitude * Deg2Rad;

  return (lon - refLon) * cos(lat) * 6378137.0 /
         sqrt(1.0 - pow(0.08182 * sin(lat), 2));
}

struct indexedPolygonPoint_t {
  int id;
  double n;
  double e;
};

struct polygonPoint_t {
  double n;
  double e;
};

struct polygon_t {
  std::vector<polygonPoint_t> points;
  double max_n;
  double max_e;
  double min_n;
  double min_e;
};

std::vector<std::string> readLine(const std::string &row) {
  std::vector<std::string> fields{""};
  int index = 0;
  for (char c : row) {
    if (c == ',') {
      fields.push_back("");
      index++;
    } else if (c == '"') {
    } else {
      fields[index].push_back(c);
    }
  }
  return fields;
}

std::vector<std::vector<std::string>> readCSV(std::istream &in) {
  std::vector<std::vector<std::string>> table;
  std::string row;
  while (!in.eof()) {
    std::getline(in, row);
    if (row.length() > 5) {
      std::vector<std::string> fields = readLine(row);
      table.push_back(fields);
    }
  }
  return table;
}

std::vector<indexedPolygonPoint_t> getPolygonPoints(const double &refLatDeg,
                                                    const double &refLonDeg) {
  std::ifstream myFile;
  std::string pkg_name = "landmap_lib";
  std::string pkg_path = ros::package::getPath(pkg_name);
  std::string csv_path = pkg_path + "/src/tempNodes.csv";
  myFile.open(csv_path.c_str());
  std::vector<std::vector<std::string>> stringTable = readCSV(myFile);

  double refLat = refLatDeg * Deg2Rad;
  double refLon = refLonDeg * Deg2Rad;
  std::vector<indexedPolygonPoint_t> pointTable;
  for (std::vector<int>::size_type i = 1; i < stringTable.size();
       i++) { // skip header
    if (stringTable[i].size() == 3) {
      try {
        indexedPolygonPoint_t point;
        double id = std::stod(stringTable[i][0]);
        point.id = (int)round(id);
        double lat = std::stod(stringTable[i][2]);
        double lon = std::stod(stringTable[i][1]);
        point.n = calculate_north(lat, refLat);
        point.e = calculate_east(lat, lon, refLon);
        pointTable.push_back(point);
      } catch (...) {
        ROS_WARN("Something wrong with land areas file");
      }
    }
  }
  return pointTable;
}

void updatePolygonMaxValues(polygon_t &p, double n, double e) {
  if (n > p.max_n) {
    p.max_n = n;
  }
  if (n < p.min_n) {
    p.min_n = n;
  }
  if (e > p.max_e) {
    p.max_e = e;
  }
  if (n < p.min_e) {
    p.min_e = e;
  }
}

std::vector<polygon_t> getRelevantPolygons(const double &lat,
                                           const double &lon) {
  std::vector<indexedPolygonPoint_t> polygonPoints = getPolygonPoints(lat, lon);
  std::vector<polygon_t> relevantPolygons;
  std::vector<int>::size_type i = 0;
  while (i < polygonPoints.size()) {
    polygon_t polygon;
    int curIndex = polygonPoints[i].id;
    polygonPoint_t initPoint;
    initPoint.n = polygonPoints[i].n;
    initPoint.e = polygonPoints[i].e;
    i++;
    polygon.points.push_back(initPoint);
    polygon.max_n = initPoint.n;
    polygon.min_n = initPoint.n;
    polygon.max_e = initPoint.e;
    polygon.min_e = initPoint.e;

    while (i < polygonPoints.size() && polygonPoints[i].id == curIndex) {
      polygonPoint_t pPoint;
      pPoint.n = polygonPoints[i].n;
      pPoint.e = polygonPoints[i].e;
      i++;
      polygon.points.push_back(pPoint);
      updatePolygonMaxValues(polygon, pPoint.n, pPoint.e);
    }
    if (polygon.max_n > -MAXRANGE && polygon.min_n < MAXRANGE &&
        polygon.max_e > -MAXRANGE && polygon.min_e < MAXRANGE) {
      ROS_INFO_STREAM("Relevant polygon index: " << curIndex);
      relevantPolygons.push_back(polygon);
    }
  }
  for (std::vector<int>::size_type i = 0; i < relevantPolygons.size(); i++) {
    int count = relevantPolygons[i].points.size();
    double max_n = relevantPolygons[i].max_n;
    double min_n = relevantPolygons[i].min_n;
    double max_e = relevantPolygons[i].max_e;
    double min_e = relevantPolygons[i].min_e;
    ROS_INFO_STREAM("Count: " << count << " max_n: " << max_n
                              << " min_n: " << min_n << " max_e: " << max_e
                              << " min_e: " << min_e);
  }
  return relevantPolygons;
}

int getBitPos(int n_int, int e_int) {
  if (n_int < 0 || n_int > int(2*MAXRANGE*DPM-1)) {
    return -1;
  }
  if (e_int < 0 || e_int > int(2*MAXRANGE*DPM-1)) {
    return -1;
  }
  if (n_int * int(2*MAXRANGE*DPM) + e_int > 16000000){
    ROS_INFO_STREAM(n_int * int(2*MAXRANGE*DPM) + e_int);
  }
  return n_int * int(2*MAXRANGE*DPM) + e_int;
}

bool checkPointsinArea(const polygonPoint_t &point1,
                       const polygonPoint_t &point2) {
  if (point1.n < -MAXRANGE && point2.n < -MAXRANGE)
    return false;
  if (point1.n > MAXRANGE && point2.n > MAXRANGE)
    return false;
  if (point1.e < -MAXRANGE && point2.e < -MAXRANGE)
    return false;
  if (point1.e > MAXRANGE && point2.e > MAXRANGE)
    return false;
  return true;
}

bool checkPointinArea(const polygonPoint_t &point) {
  if (point.n > -MAXRANGE && point.n < MAXRANGE && point.e > -MAXRANGE &&
      point.e < MAXRANGE)
    return true;
  return false;
}

void increasePolygonSize(std::bitset<MAPSIZE> &map, int n_int, int e_int,
                         int radius) {

  for (int n = n_int - radius; n < n_int + radius + 1; n++) {
    for (int e = e_int - radius; e < e_int + radius + 1; e++) {
      int pos = getBitPos(n, e);
      if (pos > -1) {
        map.set(pos);
      }
    }
  }
}

void traversePolygon(std::vector<polygonPoint_t> &points,
                     std::bitset<MAPSIZE> &map,
                     std::bitset<MAPSIZE> &checkMap) {
  polygonPoint_t point1 = points[0];
  polygonPoint_t point2;
  polygonPoint_t curPoint;
  for (std::vector<int>::size_type i = 1; i < points.size(); i++) {
    point2 = points[i];
    if (checkPointsinArea(point1, point2)) {
      double edgeLength =
          sqrt(pow(point2.n - point1.n, 2) + pow(point2.e - point1.e, 2));
      double n_unit = (point2.n - point1.n) / edgeLength;
      double e_unit = (point2.e - point1.e) / edgeLength;
      curPoint.n = point1.n;
      curPoint.e = point1.e;
      double abs_n = abs(point2.n - point1.n);
      double abs_e = abs(point2.e - point1.e);
      while (edgeLength > 0.001 && abs(curPoint.n - point1.n) <= abs_n &&
             abs(curPoint.e - point1.e) <= abs_e) {
        if (checkPointinArea(curPoint)) {
          int n_int = (int)(floor(MAXRANGE*DPM + DPM * curPoint.n));
          int e_int = (int)(floor(MAXRANGE*DPM + DPM * curPoint.e));
          map.set(getBitPos(n_int, e_int));
          increasePolygonSize(map, n_int, e_int, 6);
          checkMap.set(getBitPos(n_int, e_int));
        }
        curPoint.n += n_unit * 1.0/float(DPM);
        curPoint.e += e_unit * 1.0/float(DPM);
      }
      curPoint.n = point2.n;
      curPoint.e = point2.e;
      if (checkPointinArea(curPoint)) {
        int n_int = (int)(floor(MAXRANGE*DPM + DPM * curPoint.n));
        int e_int = (int)(floor(MAXRANGE*DPM + DPM * curPoint.e));
        map.set(getBitPos(n_int, e_int));
        increasePolygonSize(map, n_int, e_int, 6);
        checkMap.set(getBitPos(n_int, e_int));
      }
    }
    point1 = point2;
  }
}

void scanLinePolygon(std::vector<polygonPoint_t> &points,
                     std::bitset<MAPSIZE> &map, int n_min, int n_max) {
  double n =
      -MAXRANGE + n_min / DPM - 1/(2*DPM); // -1/(2DPM) because 1/dpm will be added in loop.
  for (int n_int = n_min; n_int <= n_max; n_int++) {
    n += 1/DPM;

    int i, j, nodes = 0;
    int nodeX[100], e_int;
    for (i = 0, j = points.size() - 1; i < points.size(); j = i++) {
      if ((points[i].n > n) != (points[j].n > n)) {
        nodeX[nodes++] = (int)(floor(
            MAXRANGE*DPM + DPM * ((points[j].e - points[i].e) * (n - points[i].n) /
                              (points[j].n - points[i].n) +
                          points[i].e)));
      }
    }
    if (nodes > 0) {
      std::sort(nodeX, nodeX + nodes);
      int mapN = n_int * 2 * MAXRANGE * DPM;
      for (i = 0; i < nodes; i += 2) {
        if (nodeX[i] > int(2 * MAXRANGE * DPM - 1))
          break;
        if (nodeX[i + 1] > 0) {
          if (nodeX[i] < 0) {
            nodeX[i] = 0;
          }
          if (nodeX[i + 1] > int(2 * MAXRANGE * DPM - 1)){
            nodeX[i + 1] = int(2 * MAXRANGE * DPM - 1);
          }
          for (int e_int = nodeX[i]; e_int < nodeX[i + 1] + 1; e_int++) {
            map.set(mapN + e_int);
          }
        }
      }
    }
  }
}

void scanLinePolygons(std::vector<polygon_t> &polygons,
                      std::bitset<MAPSIZE> &map) {
  for (auto i = 0; i < polygons.size(); i++) {
    int n_min = (int)(floor(MAXRANGE*DPM + DPM * polygons[i].min_n));
    int n_max = (int)(floor(MAXRANGE*DPM + DPM * polygons[i].max_n));
    if (n_min < 0)
      n_min = 0;
    if (n_max > 2*MAXRANGE*DPM)
      n_max = int(2*MAXRANGE*DPM - 1);
    scanLinePolygon(polygons[i].points, map, n_min, n_max);
  }
}

void checkPolygons(std::vector<polygon_t> &polygons,
                   std::bitset<MAPSIZE> &map) {
  std::bitset<MAPSIZE> checkMap;
  for (std::vector<int>::size_type i = 0; i < polygons.size(); i++) {
    traversePolygon(polygons[i].points, map, checkMap);
  }
  scanLinePolygons(polygons, map);
}

void Landmap::initialize(double lat, double lon) {
  double start = ros::Time::now().toSec();
  std::vector<polygon_t> polygons = getRelevantPolygons(lat, lon);
  checkPolygons(polygons, map);
  double stop = ros::Time::now().toSec();
  ROS_INFO_STREAM("Time spent: " << stop - start);
}

bool Landmap::isLand(double n, double e) {
  int n_int = (int)(floor(MAXRANGE*DPM + DPM * n));
  int e_int = (int)(floor(MAXRANGE*DPM + DPM * e));
  int bitPos = getBitPos(n_int, e_int);
  if (bitPos < 0) {
    return false;
  }
  return map.test(bitPos);
}