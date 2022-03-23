#include <ros/ros.h>
#include "PracticalSocket.h"
#include <signal.h>

#include "custom_msgs/gnssGGA.h"
#include "custom_msgs/gnssHDT.h"
#include "custom_msgs/gnssRMC.h"
#include "custom_msgs/bowControl.h"
#include "custom_msgs/podAngle.h"
#include "custom_msgs/SternThrusterSetpoints.h"
#include "custom_msgs/ExtCommand.h"
#include "custom_msgs/GuidanceLawData.h"
#include "custom_msgs/AisDynamic.h"
#include "std_msgs/UInt8.h"
#include "std_msgs/Float64.h"
#include <string.h>

#define SENDBUFSIZE 128 // size of send buffer
#define RECVBUFSIZE 256


class TCPDatatransceiver{
  public:
    explicit TCPDatatransceiver(int portNum, ros::NodeHandle nh);
    void spin();
  private:
    TCPSocket *serverSocket;
    TCPServerSocket *sock;
    void init_connection(int portNum);
    void fillDatabuffer(char *buffer);
    int sockDesc;

    // Callback functions
    void controlModeCallback(const std_msgs::UInt8& msg);
    void geodeticPositionCallback(const custom_msgs::gnssGGA &geoPos);
    void headingCallback(const custom_msgs::gnssHDT &headingMsg);
    void courseAndSpeedCallback(const custom_msgs::gnssRMC &courseAndSpeedMsg);
    void batteryVoltageCallback(const std_msgs::Float64 &battvolMsg);
    void bowSpeedAndDirectionCallback(const custom_msgs::bowControl &bowMsg);
    void sternSpeedCallback(const custom_msgs::SternThrusterSetpoints &sternMsg);
    void sternAngleCallback(const custom_msgs::podAngle &stepperMsg);
    void guidanceLawDataCallback(const custom_msgs::GuidanceLawData &guidanceLawDataMsg);
    void obstacleCallback(const custom_msgs::AisDynamic &msg);

    // Subscribers
    ros::Subscriber geodeticPosition_sub;
    ros::Subscriber heading_sub;
    ros::Subscriber courseAndSpeed_sub;
    ros::Subscriber batteryVoltage_sub;
    ros::Subscriber bowSpeedAndDirection_sub;
    ros::Subscriber sternSpeed_sub;
    ros::Subscriber sternAngle_sub_CME;
    ros::Subscriber sternAngle_sub_TA;
    ros::Subscriber control_mode_sub;
    ros::Subscriber guidance_law_data_sub;
    ros::Subscriber obstacle_sub;

    // Publishers
    ros::Publisher external_command_pub;


    //Storage variables
    double latitude;
    double longitude;
    double vessel_speed;
    double heading;
    double course;
    double battery_voltage;
    double star_rpm;
    double port_rpm;
    double bow_rpm;
    double star_deg;
    double port_deg;
    double bow_deg;

    double north_pos;
    double east_pos;
    double crosstrack_error;
    double alongtrack_distance;
    int current_wp_number;

    int control_mode;
    int op_override;
    int stop;
    double rc_heading_ref;
    double rc_speed_ref;

    bool guidance_finished;

    // Position and ID for obstacle
    double obs_latitude;
    double obs_longitude;
    double obs_mmsi;

    ros::Time begin;

    // TCP Variables
    char recvBuffer[RECVBUFSIZE];
    void appendNewObstacle(ostringstream &oss);
    void decodeExternalControlSignals(char *recvBuffer);
    void processWaypoints(std::vector<double> array);

};
