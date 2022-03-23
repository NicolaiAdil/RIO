#include "udp_video_stream/UDPVideoStream.h"
#include "udp_video_stream/PracticalSocket.h"

// Constructor
UDP_Video_Stream::UDP_Video_Stream(ros::NodeHandle nh){
  // Blocks until client has connected
  init_connection();

  // Init image subscriber
  frame_sub = nh.subscribe("usb_cam/image_raw", 1, &UDP_Video_Stream::imageCallback, this);
}

void UDP_Video_Stream::imageCallback(const sensor_msgs::ImageConstPtr &imgMsg){
  // Perform conversion from imgMsg to OpenCv Matrix frame
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::RGB8);
  frame = cv_ptr->image;

  cv::resize(frame, send, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, CV_INTER_LINEAR);
  std::vector < int > compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(ENCODE_QUALITY);
  cv::imencode(".jpg", send, encoded, compression_params);
  int total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;

  // sending expected packet size
  int ibuf[1];
  ibuf[0] = total_pack;
  sock->sendTo(ibuf, sizeof(int), sourceAddress, sourcePort);

  // sending ecoded image
  for (int i = 0; i < total_pack; i++)
      sock->sendTo( & encoded[i * PACK_SIZE], PACK_SIZE, sourceAddress, sourcePort);
  ROS_INFO("UDP: Transmitting");
  char buf[4];
  // if(sock->recvFrom(buf, BUF_LEN, sourceAddress, sourcePort)<2){
  //   exit(0);
  // }
}

void UDP_Video_Stream::init_connection(){
  int portNum = 2222;           // Port at which socket is reached
  sock = new UDPSocket(portNum);// Open output socket
  char buffer[BUF_LEN];         // Buffer for echo string
  int recvMsgSize;              // Size of received message

  ROS_INFO("UDP: Video Stream Server started at port: %d", portNum);
  recvMsgSize = sock->recvFrom(buffer, BUF_LEN, sourceAddress, sourcePort);
  ROS_INFO("UDP: RMC station with IP:%s has connected for video stream!", sourceAddress.c_str());
}

/**
* Due to accept() function, OS needs to escalate to SIGTERM
* to kill node. This prevents that.
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

   // pause();
}

int main(int argc, char * argv[]){
  ros::init(argc, argv, "UDPVideoStream");
  ros::NodeHandle nh;
  mySignalHandler();
  UDP_Video_Stream rss(nh);
  ros::spin();
  return 0;
}