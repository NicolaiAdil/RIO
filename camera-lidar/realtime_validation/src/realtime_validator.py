#!/usr/bin/python
import rospy
from sensor_msgs.msg import TimeReference


# /ladybug/time

global seq


def time_reference_callback(msg, args):
    ladybug_dict, rec_image_dict, pro_image_dict = args
    source = msg.source

    # Using time_ref in each timereference-msg as the common point for all calculations.

    if time_source in ladybug_dict:
        # append the new number to the existing array at this slot

        ladybug_dict[time_source].append(msg)

    else:
        # create a new array in this slot

        ladybug_dict[time_source] = msg

    if source == "LadybugDriver":
        ladybug_dict[time_source] = msg
        print(
            "Image aquisition time : ",
            msg.header.stamp.to_sec() - msg.time_ref.to_sec(),
        )

    elif source == "rec_image_callback":
        ladybug_msg = ladybug_dict[time_source]

        print("Image rectification time :", msg.header.stamp.to_sec())

        rec_image_dict[time_source] = msg

    elif source == "image_projecting":
        ladybug_msg = ladybug_dict[time_source]

        if msg.header.frame_id == "cam0":
            print("Total time: ")
        elif msg.header.frame_id == "cam1":
            pass
        elif msg.header.frame_id == "cam4":
            pass


rospy.init_node("realtime_validation")


ref_ladybug_gps_times = dict()
rec_image_dict = dict()
pro_image_dict = dict()

rospy.Subscriber(
    "/ladybug/timeref",
    TimeReference,
    callback=time_reference_callback,
    callback_args=(ref_ladybug_gps_times, rec_image_dict, pro_image_dict),
    queue_size=30,
)
rospy.Subscriber(
    "/rec_timeref",
    TimeReference,
    callback=time_reference_callback,
    callback_args=(ref_ladybug_gps_times, rec_image_dict, pro_image_dict),
    queue_size=30,
)
rospy.Subscriber(
    "/pro_timeref",
    TimeReference,
    callback=time_reference_callback,
    callback_args=(ref_ladybug_gps_times, rec_image_dict, pro_image_dict),
    queue_size=30,
)


while not rospy.is_shutdown():
    rospy.spin()

# print("Print dict here")
# for keys,values in system_times.items():
#    print(keys)
#    print(values)
