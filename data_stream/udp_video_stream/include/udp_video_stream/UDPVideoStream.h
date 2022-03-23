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
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "udp_video_stream/PracticalSocket.h"
#include "udp_video_stream/config.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

class UDP_Video_Stream{

public:
  explicit UDP_Video_Stream(ros::NodeHandle nh);
  void spin();

private:
  // Subscriber
  ros::Subscriber frame_sub;

  // Callback
  void imageCallback(const sensor_msgs::ImageConstPtr &imgMsg);

  // init function
  void init_connection();

  // variables
  cv::VideoCapture cap;
  cv::Mat frame; // Frame from converted from sensor_msgs/image
  cv::Mat send; // frame resized and color converted
  std::vector < uchar > encoded; // encoded image send over network

  UDPSocket *sock;

  string sourceAddress;         // Address of datagram source
  unsigned short sourcePort;    // Port of datagram source
};

