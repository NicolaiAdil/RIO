#include <bitset>

class Landmap {
    public:
        void initialize(double lat, double lon);
        bool isLand(double n, double e);
    private:
        std::bitset<16000000> map;
};
