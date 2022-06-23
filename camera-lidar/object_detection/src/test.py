#!/usr/bin/python


from darknet_ros_msgs.msg import *
from random import randint, choice
import rospy


def createBoxes():
    listofBoxes = BoundingBoxes()

    if False:
        for i in range(20):
            msg = BoundingBox()
            msg.Class = "boat"
            msg.probability = 0.6
            msg.xmin = randint(0, 2048)
            msg.ymin = randint(0, 2464)
            msg.xmax = randint(0, 2048)
            msg.ymax = randint(0, 2464)
            listofBoxes.bounding_boxes.append(msg)

    msg = BoundingBox()
    msg.Class = "boat"
    msg.probability = 0.5
    msg.xmin = 1019
    msg.ymin = 1000
    msg.xmax = 1200
    msg.ymax = 1000
    listofBoxes.bounding_boxes.append(msg)

    msg = BoundingBox()
    msg.Class = "boat"
    msg.probability = 0.4
    msg.xmin = 2048
    msg.ymin = 2448
    msg.xmax = 2048
    msg.ymax = 2448
    listofBoxes.bounding_boxes.append(msg)

    msg = BoundingBox()
    msg.Class = "boat"
    msg.probability = 0.4
    msg.xmin = 0
    msg.ymin = 0
    msg.xmax = 0
    msg.ymax = 0
    listofBoxes.bounding_boxes.append(msg)

    return listofBoxes


def sendboxes():

    sendBoxesros = rospy.Publisher(
        "/darknet_ros/bounding_boxes", BoundingBoxes, queue_size=30
    )

    rospy.init_node("test")
    names = ["cam0", "cam1", "cam4"]
    i = 0
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        boxes = createBoxes()
        boxes.header.stamp = rospy.get_rostime()
        boxes.header.frame_id = "Darknet testboxes"
        cam_name = choice(names)
        boxes.image_header.frame_id = cam_name
        boxes.image_header.stamp = boxes.header.stamp
        sendBoxesros.publish(boxes)
        rate.sleep()


if __name__ == "__main__":

    try:
        sendboxes()
    except rospy.ROSInterruptException:
        print("Shutting down")
