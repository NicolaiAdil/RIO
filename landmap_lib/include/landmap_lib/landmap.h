#include <bitset>
#include "ros/ros.h"
#include "ros/package.h"
#include <algorithm>
#include <cmath>
#include <fstream>

class Landmap {
    public:
        void initialize(double lat, double lon);
        bool isLand(double n, double e);
    private:
        std::bitset<16000000> map;
};
