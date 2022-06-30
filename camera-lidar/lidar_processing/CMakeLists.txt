cmake_minimum_required(VERSION 2.8.3)
project(lidar_processing)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  tf2_ros
  pcl_conversions
  pcl_ros
  tf2_geometry_msgs
  autosea_msgs
  landmap_lib
)


catkin_package(
  INCLUDE_DIRS include  
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs tf2_ros pcl_conversions
    pcl_ros tf2_geometry_msgs autosea_msgs landmap_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_LIBRARIES}
)

add_executable(LidarProcessor src/lidar_clustering.cpp src/dbscan.cpp)

add_dependencies(LidarProcessor sensor_msgs_generate_messages_cpp)
add_dependencies(LidarProcessor autosea_msgs_generate_messages_cpp)

target_link_libraries(LidarProcessor ${catkin_LIBRARIES})

install(
  TARGETS LidarProcessor
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
