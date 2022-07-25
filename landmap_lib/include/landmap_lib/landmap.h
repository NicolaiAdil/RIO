#ifndef LANDMAP_H
#define LANDMAP_H
#define MAXRANGE 1000.0
#define DPM 2
#define MAPSIZE (int(2*2*MAXRANGE*MAXRANGE*DPM*DPM))
#include <bitset>
#include "ros/ros.h"
#include "ros/package.h"
#include <algorithm>
#include <cmath>
#include <fstream>

// The following block uses global variables in header file which is ugly
// May be useful if the #define macros fucks something up
// const double MAXRANGE = 1000.0; // meters
// const int DPM = 2; // dots per meter resolution of the map
// const int MAPSIZE = int(pow(2*MAXRANGE*DPM, 2.0)); // 


class Landmap {
    public:
        void initialize(double lat, double lon);
        bool isLand(double n, double e);
    private:
        std::bitset<MAPSIZE> map;
};
#endif LANDMAP_H