cmake_minimum_required(VERSION 2.8.3)
project(frame_transformer)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf2
  tf2_ros
  geometry_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)



## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#INCLUDE_DIRS include

 CATKIN_DEPENDS 
  roscpp 
  rospy 
  tf2 
  tf2_ros

)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
