#include <ros/ros.h>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "autosea_msgs/AisDynamic.h"
#include "udp_receiver/PracticalSocket.h"

class UDP_ais{
public:
	explicit UDP_ais(ros::NodeHandle nh);
	void spin();

private:
    ros::NodeHandle nh;
	ros::Publisher ais_pub;

    int buffer_size;
	void decode(char buf[]);
    int rate;

	UDPSocket *sock;
	string sourceAddress;
	unsigned short sourcePort;
};