cmake_minimum_required(VERSION 2.8.3)
project(udp_receiver)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
	custom_msgs
	roscpp
	roslint
	rospy
	std_msgs
  autosea_msgs
)

catkin_package(
	INCLUDE_DIRS include
  LIBRARIES udp_receiver
	CATKIN_DEPENDS
  custom_msgs
  roscpp
  roslint
  rospy
  std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(UDP_receiver
  src/UDP_receiver.cpp
  src/PracticalSocket.cpp
  src/sixbit.cpp
  src/nmea.cpp
  src/vdm_parse.cpp
 )

add_dependencies(UDP_receiver custom_msgs_generate_messages_cpp)


 target_link_libraries(UDP_receiver
  ${catkin_LIBRARIES}
 )