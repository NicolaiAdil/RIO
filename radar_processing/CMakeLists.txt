cmake_minimum_required(VERSION 3.0.2)
project(radar_processing)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  custom_msgs
  diagnostic_msgs
  geometry_msgs
  roscpp
  roslint
  rospy
  sensor_msgs
  autosea_msgs
  std_msgs
  tf
  pcl_ros
  landmap_lib
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS custom_msgs diagnostic_msgs geometry_msgs roscpp 
    roslint rospy sensor_msgs std_msgs pcl_ros autosea_msgs landmap_lib
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_LIBRARIES}
)

add_executable(RadarSpokesToPointsNode src/RadarSpokesToPoints.cpp)
add_executable(RadarPointsToClustersNode src/RadarPointsToClusters.cpp)

target_link_libraries(RadarSpokesToPointsNode ${catkin_LIBRARIES} ${PCL_LIBRARIES})  
target_link_libraries(RadarPointsToClustersNode ${catkin_LIBRARIES} ${PCL_LIBRARIES}) 

add_dependencies(RadarSpokesToPointsNode custom_msgs_generate_messages_cpp)
add_dependencies(RadarPointsToClustersNode custom_msgs_generate_messages_cpp)