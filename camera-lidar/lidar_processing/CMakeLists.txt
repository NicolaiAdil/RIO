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
# DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_LIBRARIES}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/landmap.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(LidarProcessor src/lidar_clustering.cpp src/dbscan.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(LidarProcessor sensor_msgs_generate_messages_cpp)
add_dependencies(LidarProcessor autosea_msgs_generate_messages_cpp)

## Specify libraries to link a library or executable target against
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
