#!/usr/bin/env python
import rospy

# Because of transformations
import tf_conversions

import tf2_ros
from geometry_msgs.msg import Twist, TransformStamped


def transform_ned2body(eta_body):

    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "NED"
    t.child_frame_id = "body"
    t.transform.translation.x = eta_body.linear.x
    t.transform.translation.y = eta_body.linear.y
    t.transform.translation.z = 0.0
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, eta_body.angular.z)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    br.sendTransform(t)


if __name__ == "__main__":
    try:
        rospy.init_node("ned2body_broadcaster", anonymous=True)
        print("IN BROADCAST")
        body_sub = rospy.Subscriber("observer/eta/body", Twist, transform_ned2body)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
