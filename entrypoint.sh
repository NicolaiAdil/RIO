#!/bin/bash

# Input arguments
# $1 = launchfile name (default revolt)
# $2 = type (either vessel or simulator, default simulator)

launchfile="revolt"
main="simulator"

while getopts ":f:m" opt; do
  case $opt in
    f)
      launchfile=$OPTARG
      ;;
    m)
      main=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# very dirty temporary hack to get the correct internal IP for ROS to run correctly
# required when running a linux container with compose when windows containers
# are active in Docker Desktop, since "localhost" does not resolve in this case,
host_machine_ip=$(ifconfig eth0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)
if [$host_machine_ip == ""]
then
  host_machine_ip="localhost"
fi

export ROS_MASTER_URI=http://$host_machine_ip:11311
source /sensor_fusion_ws/devel/setup.bash

roslaunch sensor_fusion_launch $launchfile.launch main:=$main machine_ip:=$host_machine_ip