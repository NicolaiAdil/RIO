cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  custom_msgs
  roscpp
  rospy
  std_msgs
  darknet_ros_msgs
  tf2
)

generate_messages(
   DEPENDENCIES
   custom_msgs
   std_msgs
   darknet_ros_msgs
 )

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)