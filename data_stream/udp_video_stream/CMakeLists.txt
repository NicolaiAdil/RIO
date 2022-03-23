cmake_minimum_required(VERSION 2.8.3)
project(udp_video_stream)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Ignore some build warnings
set(GCC_COVERAGE_COMPILE_FLAGS "-Wno-dev -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

#set (OpenCV_DIR /usr/share/OpenCV) # OLD METHOD BEFORE ADDING WARNING ON NOT BEING SET FROM BEFORE. KEPT FOR LEGACY

# Check for defined environment variable.
IF(NOT DEFINED ENV{OpenCV_DIR})
  MESSAGE([WARNING] " Message from the ReVolt team: OpenCV_DIR has to be specified by 'export OpenCV_DIR=/path/to/OpenCV'. This is where the file OpenCVConfig.cmake should be placed. Tip: path should be set in .bashrc to automate this task once and for all.")
ENDIF(NOT DEFINED ENV{OpenCV_DIR})

set (OpenCV_DIR $ENV{OpenCV_DIR}) # Gets global environment variable. This variable has to be set manually

find_package(catkin REQUIRED COMPONENTS
  custom_msgs
  roscpp
  roslint
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  OpenCV
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES udp_video_stream
  CATKIN_DEPENDS custom_msgs roscpp roslint rospy std_msgs sensor_msgs cv_bridge
  DEPENDS OpenCV
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
  ${OpenCV_INCLUDE_DIRS}
)

#############
## Install ##
#############

add_executable(UDPVideoStream
  src/UDPVideoStream.cpp
  src/PracticalSocket.cpp
  )

target_link_libraries(UDPVideoStream
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_udp_video_stream.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
