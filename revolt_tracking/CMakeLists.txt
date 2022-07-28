cmake_minimum_required(VERSION 2.8.3)
project(revolt_tracking)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
  laser_geometry
  tf
  tf2_ros
  autosea_msgs
  asv_msgs
  nmea_msgs
  nmea_comms
)

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

add_definitions(-std=c++11)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs message_runtime autosea_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)