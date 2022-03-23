#include "tcp_datatransceiver/TCPDatatransceiver.h"

using namespace std;

TCPDatatransceiver::TCPDatatransceiver(int portNum, ros::NodeHandle nh){
  // Init control mode subscription here
  control_mode = 1;
  
  std::string control_mode_topic;
  nh.getParam("/topics/control/controllers/dp/control_mode", control_mode_topic);
  control_mode_sub = nh.subscribe(control_mode_topic, 10, &TCPDatatransceiver::controlModeCallback, this);

  // Subscribers initialization
  std::string fix_topic, heading_topic, cogsog_topic;
  nh.getParam("/topics/hardware/vs330_navsat/fix", fix_topic);
  nh.getParam("/topics/hardware/vs330_navsat/heading", heading_topic);
  nh.getParam("/topics/hardware/vs330_navsat/velocity", cogsog_topic);
  geodeticPosition_sub     = nh.subscribe(fix_topic, 10, &TCPDatatransceiver::geodeticPositionCallback, this);
  heading_sub              = nh.subscribe(heading_topic, 10, &TCPDatatransceiver::headingCallback, this);
  courseAndSpeed_sub       = nh.subscribe(cogsog_topic, 10, &TCPDatatransceiver::courseAndSpeedCallback, this);

  std::string V_batt_topic, bow_control_topic, stern_speed_topic, stern_angle_cme_topic, stern_angle_ta_topic;
  nh.getParam("/topics/hardware/actuators/stern/battery_voltage", V_batt_topic);
  nh.getParam("/topics/cme/bow_control", bow_control_topic);
  nh.getParam("/topics/control/allocators/stern_setpoints", stern_speed_topic);
  nh.getParam("/topics/cme/pod_angle", stern_angle_cme_topic);
  nh.getParam("/topics/control/allocators/stern_angles", stern_angle_ta_topic);
  batteryVoltage_sub       = nh.subscribe(V_batt_topic, 10, &TCPDatatransceiver::batteryVoltageCallback, this);
  bowSpeedAndDirection_sub = nh.subscribe(bow_control_topic, 10, &TCPDatatransceiver::bowSpeedAndDirectionCallback, this);
  sternSpeed_sub           = nh.subscribe(stern_speed_topic, 10, &TCPDatatransceiver::sternSpeedCallback, this);
  sternAngle_sub_CME       = nh.subscribe(stern_angle_cme_topic, 10, &TCPDatatransceiver::sternAngleCallback, this);
  sternAngle_sub_TA        = nh.subscribe(stern_angle_ta_topic, 10, &TCPDatatransceiver::sternAngleCallback, this);

  std::string guidance_law_topic, obstacle_topic;
  nh.getParam("/topics/guidance/guidance_law/data", guidance_law_topic);
  nh.getParam("/topics/communication/udp/ais", obstacle_topic);
  guidance_law_data_sub    = nh.subscribe(guidance_law_topic, 10, &TCPDatatransceiver::guidanceLawDataCallback, this);
  obstacle_sub             = nh.subscribe(obstacle_topic, 1, &TCPDatatransceiver::obstacleCallback, this);

  // Publisher initialization
  std::string command_topic;
  nh.getParam("/topics/communication/tcp/command", command_topic);
  external_command_pub = nh.advertise<custom_msgs::ExtCommand>(command_topic, 10);

  // Initialization of storage variables
  latitude        = 0.0;
  longitude       = 0.0;
  vessel_speed    = 0.0;
  heading         = 0.0;
  course          = 0.0;
  battery_voltage = 0.0;
  star_rpm        = 0.0;
  port_rpm        = 0.0;
  bow_rpm         = 0.0;
  star_deg        = 0.0;
  port_deg        = 0.0;
  bow_deg         = 0.0;

  north_pos           = 1.10;
  east_pos            = 3.2;
  crosstrack_error    = 10.0;
  alongtrack_distance = 23.0;
  current_wp_number   = 2;

  op_override    = 0;
  rc_heading_ref = 0;
  rc_speed_ref   = 0;

  stop = 0;
  begin = ros::Time::now();

  // Must be set in the end, else other things comming after is not initialized.
  init_connection(portNum);
}
void TCPDatatransceiver::init_connection(int portNum){
  try{
      sock = new TCPServerSocket(portNum);
      ROS_INFO("TCP: Server for external control started at port %d", portNum);
      serverSocket = sock->accept();
      ROS_INFO("TCP: Client at %s:%d connected", serverSocket->getForeignAddress().c_str(), serverSocket->getForeignPort());
  }
  catch(SocketException se){
    cerr << se.what() << endl;
    exit(0);
  }
}

void TCPDatatransceiver::spin(){

  char sendBuffer[SENDBUFSIZE];

  int bytesReceived;
  ros::Rate rate(10);
  do {
    try{
      // Fill and send buffer to socket
      fillDatabuffer(sendBuffer);
      serverSocket->send(sendBuffer, SENDBUFSIZE);

      // Receive control signals
      bytesReceived = serverSocket->recv(recvBuffer, RECVBUFSIZE);
      recvBuffer[bytesReceived] = '\0';
      decodeExternalControlSignals(recvBuffer);
    }
    catch(SocketException se){

    }
    ros::spinOnce();
    rate.sleep();
  } while(ros::ok() && stop != 1);
  ROS_INFO("TCP: Client at %s:%d is disconnecting...", serverSocket->getForeignAddress().c_str(), serverSocket->getForeignPort());
  ROS_INFO("TCP: Socket successfully closed");
  delete sock;
  delete serverSocket;
}

void TCPDatatransceiver::decodeExternalControlSignals(char *recvBuffer){
  ostringstream oss;
  oss << recvBuffer;
  string recvString = oss.str();
  string mode       = recvString.substr(0,2);
  replace(recvString.begin()+3, recvString.end(), ':', ' ');
  recvString        = recvString.substr(3, recvString.size()-1);
  stringstream ss(recvString);
  vector<double> array;
  double temp;
  while(ss >> temp){
    array.push_back(temp);
  }
  // if(mode.compare("HA") == 0){
  //   control_mode = 1;
  // }
  if(mode.compare("DP") == 0){
    control_mode = 4;
  }
  else if(mode.compare("GC") == 0){
    control_mode = 2;
    // Not implemented yet
    processWaypoints(array);
  }
  else if(mode.compare("CA") == 0){
    control_mode = 7;
    // Not implemented yet
    processWaypoints(array);
  }
  else if(mode.compare("EX") == 0){
    // signal to disable external commands
    control_mode = 1;
  }
  else if(mode.compare("ST") == 0)
  {
    // Stops the node, respawns after ~2 seconds
    stop = 1;
    return;
  }
  else if(mode.compare("OB") == 0){
  	control_mode = 8;
  }
  custom_msgs::ExtCommand external_command_msg;
  external_command_msg.mode = control_mode;
  
  for(int i = 0; i < array.size(); i++)
    external_command_msg.data.push_back(array[i]);

  external_command_pub.publish(external_command_msg);

}

void TCPDatatransceiver::processWaypoints(std::vector<double> array){

}

/**
* Creates the output stream to send to the RMC Station
* Fields in array:
* 0  - Latitude
* 1  - Longitude
* 2  - Heading
* 3  - Course
* 4  - Speed
* 5  - Battery Voltage
* 6  - Bow RPM
* 7  - Starboard RPM
* 8  - Port RPM
* 9  - Bow Angle Degree
* 10 - Starboard Angle Degree
* 11 - Port Angle Degree
* 12 - Control Mode
* 13 - Number of completed waypoints
* 14 - North Position
* 15 - East Position
* 16 - Cross-track Error
* 17 - Along-track Error
* 18 - Operator Override
* 19 - RC Remote Heading Ref
* 20 - RC Remote Speed Ref
**/
void TCPDatatransceiver::fillDatabuffer(char *buffer){

  /*
  PS! I noticed that the socket being sent with this data would occasionally miss the last elements.
  Therefore, I have changed the order of which index data is stored, with the more important elements first.
  */

  ostringstream oss;
  // Create the store all data in the line
  oss.precision(9);
  // Vessel Data
  oss
  << latitude             << ":" << longitude         << ":" << heading           << ":" << course   << ":"
  << vessel_speed         << ":" << battery_voltage   << ":" << obs_latitude             << ":" << obs_longitude << ":"
  << obs_mmsi             << ":" << bow_deg           << ":" << star_deg          << ":" << port_deg << ":"
  << control_mode         << ":" << current_wp_number << ":" << north_pos         << ":" << east_pos << ":" 
  << crosstrack_error     << ":" << alongtrack_distance  << ":" << op_override    << ":" << rc_heading_ref << ":" 
  << rc_speed_ref         << ":"; //<< obs_latitude             << ":" << obs_longitude << ":";

  //Bytter ut bow_rpm med obs_latitude og star_rpm med obs_longitude. Change port_rpm with mmsi

  // Collision Avoidance Data (Currently Max two obstacles tracked simultaneously)
  // set true to simulate a collision
  // << "obs1_id" << ":" << "obs1_N" << ":" << "obs1_E" << ":" << "obs1_vel" << ":" "obs1_course" << ":" << "radius" <<":";
  // Quick solution for simulating obstacles
  if(/*ros::Time::now().toSec() - begin.toSec() > 20*/false){
    oss << 1 << ":" << 1.4 << ":" << 3.2 << ":" << 0.0 << ":" << 0.0 << ":" << 5.0 << ":";
  }
  if(false){
    oss
    << "obs2_id" << ":" << "obs2_N" << ":" << "obs2_E" << ":" << "obs2_vel" << ":" << "obs2_course" << ":";
  }
  
  // appropriate conversions
  string var = oss.str();
  const char *var2 = var.c_str();
  // copy string to buffer
  strcpy(buffer, var2);
}

void TCPDatatransceiver::controlModeCallback(const std_msgs::UInt8& msg){
  control_mode = msg.data;
  ROS_INFO("Control mode callback in TCPDatatransceiver changed to %d", control_mode);
}

void TCPDatatransceiver::geodeticPositionCallback(const custom_msgs::gnssGGA &geoPosMsg){
  latitude = geoPosMsg.latitude;
  longitude = geoPosMsg.longitude;
  // ROS_INFO("In geodeticPositionCallback()");
}
void TCPDatatransceiver::headingCallback(const custom_msgs::gnssHDT &headingMsg){
  heading = headingMsg.heading;
  // ROS_INFO("In headingCallback()");
}
void TCPDatatransceiver::courseAndSpeedCallback(const custom_msgs::gnssRMC &courseAndSpeedMsg){
  course = courseAndSpeedMsg.course;
  vessel_speed = courseAndSpeedMsg.speed_mps;
  // ROS_INFO("In courseAndSpeedCallback()");
}
void TCPDatatransceiver::batteryVoltageCallback(const std_msgs::Float64 &battvolMsg){
  battery_voltage = battvolMsg.data;
  // ROS_INFO("In batteryVoltageCallback()");
}
void TCPDatatransceiver::bowSpeedAndDirectionCallback(const custom_msgs::bowControl &bowMsg){
  bow_rpm = bowMsg.throttle_bow;
  bow_deg = bowMsg.position_bow;
  // ROS_INFO("In bowSpeedAndDirectionCallback()");
}
void TCPDatatransceiver::sternSpeedCallback(const custom_msgs::SternThrusterSetpoints &sternMsg){
  star_rpm = sternMsg.star_effort;
  port_rpm = sternMsg.port_effort;
  // ROS_INFO("In sternSpeedCallback()");
}
void TCPDatatransceiver::sternAngleCallback(const custom_msgs::podAngle &stepperMsg){
  star_deg = stepperMsg.star;
  port_deg = stepperMsg.port;
  // ROS_INFO("In sternAngleCallback()");
}

void TCPDatatransceiver::guidanceLawDataCallback(const custom_msgs::GuidanceLawData &guidanceLawDataMsg) {
  current_wp_number = guidanceLawDataMsg.completed_waypoints;
}

void TCPDatatransceiver::obstacleCallback(const custom_msgs::AisDynamic &msg){
  obs_latitude = msg.latitude;
  obs_longitude = msg.longitude;
  obs_mmsi = msg.mmsi;
}

/**
* Due to accept() function, OS needs to escalate to SIGTERM
* to kill node. Signal handler in Node now reacts to SIGINT (ctrl+c)
*/
void my_handler(int s){
 ROS_INFO("Caught signal %d\n",s);
 exit(1);
}

void mySignalHandler()
{

   struct sigaction sigIntHandler;

   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;

   sigaction(SIGINT, &sigIntHandler, NULL);
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "TCPDatatransceiver");
  ros::NodeHandle nh;
  mySignalHandler();
  TCPDatatransceiver dt(2345, nh);
  dt.spin();
  return 0;
}
