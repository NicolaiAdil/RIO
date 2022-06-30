cmake_minimum_required(VERSION 2.8.3)
project(frame_publisher)
find_package(catkin REQUIRED COMPONENTS
  custom_msgs
  roscpp
  tf2
  tf2_ros
  std_msgs
  autosea_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(FramePublisher src/publisher.cpp)
add_dependencies(FramePublisher custom_msgs_generate_messages_cpp)

target_link_libraries(FramePublisher ${catkin_LIBRARIES})