#!/usr/bin/env python
import rospy

import math
import tf2_ros
import geometry_msgs.msg

if __name__ == "__main__":
    rospy.init_node("transform_listener", anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans_NED2body = tfBuffer.lookup_transform("NED", "body", rospy.Time())
            # trans_body2sensor = tfBuffer.lookup_transform('body', 'sensor', rospy.Time())
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rate.sleep()
            continue
