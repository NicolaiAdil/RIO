#include "udp_receiver/UDP_receiver.h"

#include "udp_receiver/portable.h"
#include "udp_receiver/nmea.h"
#include "udp_receiver/sixbit.h"
#include "udp_receiver/vdm_parse.h"
#include <ctime>

UDP_ais::UDP_ais(ros::NodeHandle nh) : nh{nh} {
    int portNum;
    nh.getParam("/udp/ais/port", portNum);
    nh.getParam("/udp/ais/buffer_size", buffer_size);
    nh.getParam("/udp/ais/rate", rate);

	sock = new UDPSocket(portNum);

    std::string ais_topic;
    nh.getParam("/topics/communication/udp/ais", ais_topic);
	ais_pub = nh.advertise<autosea_msgs::AisDynamic>(ais_topic, 1);
}

void UDP_ais::spin(){
    
    char buffer[buffer_size];
    ros::Rate loop_rate(rate);
	while(ros::ok()){
		int recvMsgSize = sock->recvFrom(buffer, buffer_size, sourceAddress, sourcePort);
		// ROS_INFO("Recieved data length %d\n", recvMsgSize);
		//ROS_INFO("D: %s", buffer);

		if(recvMsgSize != 0){
			decode(buffer);
		}
		
		ros::spinOnce();
		loop_rate.sleep();
	}
}

void UDP_ais::decode(char buf[]){
	ais_state ais;
	aismsg_1 msg_1;
	autosea_msgs::AisDynamic ais_dynamic_msg;
	double rate_ais = 1.0;
  	clock_t tick, tock;
 	double t = 0;
	double lat_dd   = 0;
	double long_ddd = 0;
	long userid     = 0;
	double course   = 0;
	memset(&ais, 0, sizeof(ais_state));
	assemble_vdm(&ais,buf);

	ais.msgid = (unsigned char) get_6bit(&ais.six_state, 6);
	parse_ais_1(&ais, &msg_1);
	userid = msg_1.userid;
	pos2ddd(msg_1.latitude, msg_1.longitude, &lat_dd, &long_ddd);

	//std::cout << "long = " << long_ddd << std::endl;
	//std::cout << "lat = " << lat_dd << std::endl;

	//printf("MSG ID: %d\n", ais.msgid);
	//printf("USER ID: %ld\n", userid );
	//printf("POS: %0.6f %0.6f \n",lat_dd,long_ddd);
	//printf("SOG: %f\n", (msg_1.sog*0.1)*0.5144444);
	//printf("COG: %f\n", msg_1.cog/10.0);
	//printf("rate_of_turn: %f\n", msg_1.rot);
	//if (t > 5){
      //  tick = clock();
	course = msg_1.cog/10.0;
	if(course > 180){
		course = course - 360;
	}
		//This is NOT the correct way of setting timestamp, but since 
		// the simulator does not have utc time stamp, a workaround must be done.
		// Normally the time stamp is set by the sensor (AIS in this case)

		ais_dynamic_msg.timestamp          = (int)ros::Time::now().toSec(); 
		
		ais_dynamic_msg.header.stamp       = ros::Time::now();
		ais_dynamic_msg.latitude           = lat_dd;
		ais_dynamic_msg.longitude          = long_ddd;
		ais_dynamic_msg.course_over_ground = course;//msg_1.cog/10.0;
		ais_dynamic_msg.speed_over_ground  = (msg_1.sog*0.1)*0.5144444;
		ais_dynamic_msg.yaw                = msg_1.trueh;
		ais_dynamic_msg.rate_of_turn       = pow((msg_1.rot/4.733),2);
		ais_dynamic_msg.mmsi               = userid;//ais.msgid;
		ais_pub.publish(ais_dynamic_msg);

		//tock = clock() - tick;
	    //t = 0;
   // }
    //t += 1/rate_ais;

}

void my_handler(int s){
	ROS_INFO("AIS signal received %d\n", s);
	exit(1);
}
void mySignalHandler(){
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
}


int main(int argc, char * argv[]){
	ros::init(argc, argv, "UDPAIS");
	ros::NodeHandle nh;
	mySignalHandler();
	UDP_ais ais(nh);

    ais.spin();
	return 0;
}