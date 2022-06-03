# Introduction 
This repository contains all the higher-level sensor fusion systems currently (partially) implemented on the ReVolt platform. The goal of the sensor fusion stack is to convert data from a lidar, radar and 360 degree camera into points and track in 2D, as well as to gather semantic data about the 3D world around the vessel.

A previous master thesis focused on fusing camera and lidar data, while the radar tracker has for the most part been developed separately, based on code from the autosea project. Much work remains to polish these systems and to seamlessly integrate them with both the real and digital twin.

### SensorFusion/camera-lidar

The subpackage [SensorFusion/camera-lidar](SensorFusion/camera-lidar) folder consists of the following packages

* **frame_publisher**
    * Contains a node that takes data from the xsense IMU and the VS330 GNSS compass, and outputs the measured ship position in NED.
* **frame_transformer**
    * Contains nodes that implement the frame transforms ned->body and body->sensor.
* **lidar_processing**
    * Contains a node that takes the incoming point cloud from the velodyne LIDAR and converts it to various other point clouds. 
    * Note that this package contains a lot of code for processing; look in lidar_clustering.cpp for the main functionality.
* **object_detection**
    * A node that takes the feed from the Ladybug camera as input, and produces a camera measurement based on the bounding boxes produced by the darknet ROS network.
    * Classes of objects that can be recognized:  
        * Boat: Produces a camera measurement with bearing of the boat and confidence that said boat actually *is* a boat.
* **realtime validation**
    * A package that enables calculation of time occurences for different events, from a common time reference.
    * NB! Appears to be incomplete as of 03.06.2021

### SensorFusion/darknet_ros

* **darknet**
    * Contains darknet. A collection of neural network implentations used for computer vision.
* **darknet_ros**
    * Contains nodes that implement the ROS interface to darknet. We use the yolov3 part of this.
* **darknet_ros_msgs**
    * Contains the message types BoundingBox and BoundingBoxes that is used to convey these to ROS nodes.

### SensorFusion/data_stream

* **tcp_datatransciever**
    * Used to forward ROS topics to the RMC (remote control center) via TCP.
* **udp_receiver**
    * Parses [NMEA standard](https://en.wikipedia.org/wiki/NMEA_0183) obstacle-data sent from ReVolt to a ROS topic.
* **udp_video_stream**
    * Subscribes to camera ROS topic and forwards it using UDP.

### SensorFusion/radar_processing

* **RadarPointsToClusters**
    * Converts radar points to clusters.
* **RadarSpokesToPoints**
    * Converts radar spokes to points.

### SensorFusion/revolt_tracking

* **autoseapy_tracking**
    * Contains autosea implementation of a tracking library. Contains a lot of different tools for track initiation, track management and tracking models.

### SensorFusion/sensor_fusion_launch

* **launch**
    * Contains launch files for launching the tracking parts, camera tracking and LiDAR tracking.

# Getting Started
Like most other ReVolt submodules, this repository has its own Dockerfile and will run in its own container. To get started, perform the following steps:

1.	Clone this repository and the RevoltMsgs repository into the same parent folder
2.  Build the Docker container by running `docker build -f ./SensorFusion/Dockerfile --tag revolt:sensor-fusion .`
3.  Deploy the container using `docker run --rm -it revolt:sensor-fusion` or by using the entry in the docker-compose file found in the ControlSystem repository


# Build and Test
TODO: Describe and show how to build your code and run the tests. 

