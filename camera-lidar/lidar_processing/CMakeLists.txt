cmake_minimum_required(VERSION 2.8.3)
project(lidar_processing)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  tf2_ros
  pcl_conversions
  pcl_ros
  tf2_geometry_msgs
  autosea_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(LidarProcessor src/lidar_clustering.cpp src/landmap.cpp src/dbscan.cpp)
add_dependencies(LidarProcessor sensor_msgs_generate_messages_cpp)
add_dependencies(LidarProcessor autosea_msgs_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(LidarProcessor ${catkin_LIBRARIES})