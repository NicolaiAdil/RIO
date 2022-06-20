#!/usr/bin/python

import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from sensor_msgs.msg import TimeReference
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import *
import tf2_ros
import math

# For publishing a line in rviz
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from autosea_msgs.msg import CameraMeasurement


class image_proscessing(object):
    def __init__(self):
        self.focal = 1350
        self.mtx = np.matrix(
            "1237.1390741979703 0.0 1009.7992410572147; 0.0 1238.820793472432 1237.01069688158; 0.0 0.0 1.0"
        )
        self.matrixDict = {
            "cam0": np.matrix(
                "1237.1390741979703 0.0 1009.7992410572147; 0.0 1238.820793472432 1237.01069688158; 0.0 0.0 1.0"
            ),
            "cam1": np.matrix(
                "1237.1390741979703 0.0 1013.561056; 0.0 1238.820793472432 1238.029600; 0.0 0.0 1.0"
            ),
            "cam4": np.matrix(
                "1237.1390741979703 0.0 1044.799200; 0.0 1238.820793472432 1258.300928; 0.0 0.0 1.0"
            ),
        }

        self.distort = np.array(
            [
                -0.2868464038626275,
                0.07503624674155485,
                -0.00028458296649612995,
                0.0005428461035833637,
            ]
        )
        self.crop_x, self.crop_y = 0, 0  # 216,600
        self.height, self.width = 0, 0

        self.image_pub = rospy.Publisher("/ladybug/rect_image", Image, queue_size=10)
        self.real_time_validator = rospy.Publisher(
            "/rec_timeref", TimeReference, queue_size=10
        )

        self.bridge = CvBridge()

        self.image_front = rospy.Subscriber(
            "/ladybug/camera0/image_raw", Image, self.rec_image_callback
        )
        self.image_right = rospy.Subscriber(
            "/ladybug/camera1/image_raw", Image, self.rec_image_callback
        )
        self.image_left = rospy.Subscriber(
            "/ladybug/camera4/image_raw", Image, self.rec_image_callback
        )

        # Solution for using old rosbags
        self.use_old_rosbag = rospy.get_param("~old_rosbag")
        self.camera_names = {"camera0": "cam0", "camera1": "cam1", "camera4": "cam4"}

    def rec_image_callback(self, ros_image_msg):

        image_rec_start = rospy.Time.now()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="rgb8")
        except CvBridgeError as e:
            print(e)

        if self.use_old_rosbag:
            ros_image_msg.header.frame_id = self.camera_names.get(
                ros_image_msg.header.frame_id
            )

        h, w = cv_image.shape[:2]

        newcamera_mtx, roi = cv.getOptimalNewCameraMatrix(
            self.matrixDict.get(ros_image_msg.header.frame_id),
            self.distort,
            (w, h),
            1,
            (w, h),
        )

        dst = cv.undistort(cv_image, self.mtx, self.distort, None, None)
        self.height, self.width = dst.shape[:2]
        undis_image = dst[
            self.crop_y : self.height - self.crop_y,
            self.crop_x : self.width - self.crop_x,
        ]

        # TODO
        # Fix image header and correct image publishing
        ros_rec_image = self.bridge.cv2_to_imgmsg(undis_image, encoding="rgb8")
        ros_rec_image.header = ros_image_msg.header

        try:
            # self.image_pub.publish(self.bridge.cv2_to_imgmsg(undis_image, "rgb8"))
            self.image_pub.publish(ros_rec_image)
        except CvBridgeError as e:
            print(e)

        image_rec_stop = rospy.Time.now()

        real_time_validation_msg = TimeReference()
        real_time_validation_msg.header.frame_id = ros_image_msg.header.frame_id
        real_time_validation_msg.header.stamp = image_rec_stop - image_rec_start
        real_time_validation_msg.time_ref = ros_image_msg.header.stamp
        real_time_validation_msg.source = "rec_image_callback"

        self.real_time_validator.publish(real_time_validation_msg)


class image_projecting(object):
    def __init__(self):
        self.image_projecting = rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.image_projecting_callback
        )

        self.pub_line = rospy.Publisher("/rviz/camera_angle", MarkerArray, queue_size=1)
        self.pub_camera_measurement = rospy.Publisher(
            "/camera_measurement", CameraMeasurement, queue_size=4
        )
        self.real_time_validator = rospy.Publisher(
            "/pro_timeref", TimeReference, queue_size=10
        )

        self.matrixDict = {
            "cam0": np.matrix(
                "1237.1390741979703 0.0 1009.7992410572147; 0.0 1238.820793472432 1237.01069688158; 0.0 0.0 1.0"
            ),
            "cam1": np.matrix(
                "1237.1390741979703 0.0 1013.561056; 0.0 1238.820793472432 1238.029600; 0.0 0.0 1.0"
            ),
            "cam4": np.matrix(
                "1237.1390741979703 0.0 1044.799200; 0.0 1238.820793472432 1258.300928; 0.0 0.0 1.0"
            ),
        }

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.use_old_rosbag = rospy.get_param("~old_rosbag")

    def project(self):
        pass

    def boat_position(self):
        trans = self.tfBuffer.lookup_transform("ned", "body", rospy.Time(0))
        current_position = np.array(
            [trans.transform.translation.x, trans.transform.translation.y]
        )
        return current_position

    def image_projecting_callback(self, bounding_boxes):

        # Real-time-performance of image-projecting is discared since it uses so little time.
        end_of_pipeline = rospy.Time.now()

        image_frame = bounding_boxes.image_header.frame_id
        image_stamp = bounding_boxes.image_header.stamp

        try:
            trans = self.tfBuffer.lookup_transform(
                "percieved_calibrated_" + image_frame + "_frame",
                "cameraframe",
                image_stamp,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            print(e)

        # Used for visualization in rviz
        markerArray = MarkerArray()

        # Construct camera measurement message from autosea-msg
        cam_msg = CameraMeasurement()
        cam_msg.header.frame_id = "percieved_calibrated_" + image_frame + "_frame"
        cam_msg.header.stamp = bounding_boxes.header.stamp  # end_of_pipeline
        cam_msg.image_header = bounding_boxes.image_header

        for box in bounding_boxes.bounding_boxes:
            if box.Class == "boat":

                # Normalized values, x_hat.item(2)=0 = normalized Z
                bb_center_x, bb_center_y = (
                    (box.xmin + box.xmax) / 2,
                    (box.ymin + box.ymax) / 2,
                )

                k_matrix = self.matrixDict.get(image_frame)
                x_c = np.array([[bb_center_x], [bb_center_y], [1]])
                x_hat = np.linalg.inv(k_matrix) * x_c

                # Depending on the FOV of the camera, one should chose either arctan or arcsin. arcsin for camera with high fov
                angle = np.arctan(x_hat.item(0))
                # angle   = np.arcsin(x_hat.item(0)/x_hat.item(2))

                # Append angle and confidence to measurment msg
                cam_msg.bearing.append(math.degrees(angle))
                cam_msg.confidence.append(box.probability)

                try:
                    markerArray.markers.append(
                        self.add_line(trans, x_hat, bounding_boxes.image_header)
                    )
                except (UnboundLocalError):
                    print("Undboundlocal error")

            """else:
                bb_center_x,bb_center_y = ((box.xmin + box.xmax)/2, (box.ymin + box.ymax)/2)
                k_matrix = self.matrixDict.get(image_frame)
                x_c = np.array([[bb_center_x], [bb_center_y], [1]])
                x_hat = np.linalg.inv(k_matrix)*x_c
                angle = np.arctan(x_hat.item(0)/x_hat.item(2))"""

            self.pub_camera_measurement.publish(cam_msg)

        id = 0
        for mark in markerArray.markers:
            mark.id = id
            id += 1

        self.pub_line.publish(markerArray)

        # Create time-validation message representing total time from image aquisiton to image projecting
        real_time_validation_msg = TimeReference()
        real_time_validation_msg.header.frame_id = bounding_boxes.image_header.frame_id
        real_time_validation_msg.header.stamp = end_of_pipeline
        real_time_validation_msg.time_ref = image_stamp
        real_time_validation_msg.source = "image_projecting"
        self.real_time_validator.publish(real_time_validation_msg)

    # TODO this does not work properly
    def add_line(self, trans, x_hat, image_header):
        marker = Marker()

        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD

        # marker scale
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        # marker color
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # marker orientaiton
        marker.pose.orientation.x = trans.transform.rotation.x
        marker.pose.orientation.y = trans.transform.rotation.y
        marker.pose.orientation.z = trans.transform.rotation.z
        marker.pose.orientation.w = trans.transform.rotation.w

        # marker position
        marker.pose.position.x = trans.transform.translation.x
        marker.pose.position.y = trans.transform.translation.y
        marker.pose.position.z = trans.transform.translation.z

        # marker line points
        marker.points = []
        # first point
        first_line_point = Point()
        first_line_point.x = 0.0
        first_line_point.y = 0.0
        first_line_point.z = 0.0
        marker.points.append(first_line_point)

        # second point
        second_line_point = Point()
        second_line_point.x = x_hat.item(0) * 200000
        second_line_point.y = 0
        second_line_point.z = x_hat.item(2) * 200000
        marker.points.append(second_line_point)

        # return marker
        marker.header.frame_id = image_header.frame_id
        marker.id = image_header.seq
        marker.lifetime = rospy.Duration(1)

        if self.use_old_rosbag:
            marker.header.stamp = rospy.Time.now()
        else:
            marker.header.stamp = image_header.stamp

        # Publish the Marker
        return marker


if __name__ == "__main__":
    rospy.init_node("imageProcessing")
    processing = image_proscessing()
    projecting = image_projecting()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()
