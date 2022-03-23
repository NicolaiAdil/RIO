cmake_minimum_required(VERSION 2.8.3)
project(tcp_datatransceiver)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# Ignore some build warnings
set(GCC_COVERAGE_COMPILE_FLAGS "-Wno-dev -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  custom_msgs
  roscpp
  roslint
  rospy
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES tcp_datatransceiver
  CATKIN_DEPENDS custom_msgs roscpp roslint rospy std_msgs
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
)

#############
## Install ##
#############

add_executable(TCPDatatransceiver
  src/TCPDatatransceiver.cpp
  src/PracticalSocket.cpp
)

target_link_libraries(TCPDatatransceiver
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(TCPDatatransceiver custom_msgs_generate_messages_cpp)

