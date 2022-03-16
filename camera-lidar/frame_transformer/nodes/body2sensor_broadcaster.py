#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg

# THIS IS JUST TO TEST
class Body2Sensor:
    def __init__(self):
        # self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
        br = tf2_ros.TransformBroadcaster()
        while not rospy.is_shutdown():
            # Run this loop at about 10Hz
            rospy.sleep(0.1)

            t = geometry_msgs.msg.TransformStamped()
            t.header.frame_id = "body"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "sensor"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 2.0
            t.transform.translation.z = 0.0

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            br.sendTransform(t)
            # tfm = tf2_msgs.msg.TFMessage([t])
            # self.pub_tf.publish(tfm)


if __name__ == "__main__":
    rospy.init_node("body2sensor_broadcaster")
    b2s = Body2Sensor()

    rospy.spin()
