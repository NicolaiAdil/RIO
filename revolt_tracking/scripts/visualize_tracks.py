#!/usr/bin/env python
import rospy
import numpy as np
import std_msgs.msg as stdmsg
import geometry_msgs.msg as geomsg
from visualization_msgs.msg import Marker
from autosea_msgs.msg import RadarEstimate
from collections import deque
import autoseapy_tracking.conversion as autoconv

identity_quat = geomsg.Quaternion(0, 0, 0, 1)
identity_pos = geomsg.Point(0, 0, 0)
identity_pose = geomsg.Pose(position=identity_pos, orientation=identity_quat)

# Set of default parameter values
def estimate_callback(msg, args):
    track_pub, estimates, marker_properties = args
    idx = msg.track_id
    if idx not in estimates.keys():
        estimates[idx] = deque(maxlen=marker_properties["max_line_length"])
    estimates[idx].append(msg)
    pub_tracks(track_pub, estimates[idx], marker_properties)


def estimate2point(estimate, marker_parameters):
    color = stdmsg.ColorRGBA(*marker_parameters["color"])
    scale = marker_parameters["end_width"]
    posterior_marker = Marker(
        ns="posterior_marker",
        id=estimate.track_id,
        type=Marker.SPHERE,
        action=Marker.ADD,
        pose=geomsg.Pose(
            geomsg.Point(estimate.posterior.pos_est.x, estimate.posterior.pos_est.y, 0),
            identity_quat,
        ),
        scale=geomsg.Vector3(scale, scale, scale),
        color=color,
        lifetime=rospy.Duration(15),
        frame_locked=True,
    )
    return posterior_marker


def estimate2text(estimate):
    vel = np.array([estimate.posterior.vel_est.x, estimate.posterior.vel_est.y])
    scale = 50
    text_marker = Marker(
        ns="posterior_text",
        id=estimate.track_id,
        type=Marker.TEXT_VIEW_FACING,
        text="Vel: %.2f m/s" % (np.linalg.norm(vel),),
        action=Marker.ADD,
        pose=geomsg.Pose(
            geomsg.Point(
                estimate.posterior.pos_est.x, estimate.posterior.pos_est.y, -20
            ),
            identity_quat,
        ),
        scale=geomsg.Vector3(1, 1, scale),
        color=stdmsg.ColorRGBA(1, 1, 1, 1),
        lifetime=rospy.Duration(15),
        frame_locked=True,
    )
    return text_marker


def estimate2cep(estimate, marker_parameters):
    color = stdmsg.ColorRGBA(*marker_parameters["cep_color"])
    point = geomsg.Point(estimate.posterior.pos_est.x, estimate.posterior.pos_est.y, 0)
    pose = geomsg.Pose(point, identity_quat)
    R_90 = autoconv.circular_error_probability_90(P)
    cep_marker = Marker(
        ns="posterior_cep",
        id=estimates[0].track_id,
        type=Marker.CYLINDER,
        action=Marker.ADD,
        pose=identity_pose,
        scale=geomsg.Vector3(R_90, R_90, 1),
        color=color,
        lifetime=rospy.Duration(5),
        frame_locked=True,
    )
    return cep_marker


def estimates2line(estimates, marker_parameters):
    color = stdmsg.ColorRGBA(*marker_parameters["color"])
    line_marker = Marker(
        ns="posterior_line",
        id=estimates[0].track_id,
        type=Marker.LINE_STRIP,
        action=Marker.ADD,
        pose=identity_pose,
        scale=geomsg.Vector3(marker_parameters["width"], 1, 1),
        color=color,
        lifetime=rospy.Duration(15),
        frame_locked=True,
        points=[
            geomsg.Point(e.posterior.pos_est.x, e.posterior.pos_est.y, 0)
            for e in estimates
        ],
        colors=[color for n in range(len(estimates))],
    )
    return line_marker


def pub_tracks(track_pub, estimates, marker_properties):
    marker_header = estimates[-1].header
    if len(estimates) > 1:
        lines = estimates2line(estimates, marker_properties)
        lines.header = marker_header
        track_pub.publish(lines)
    end_point = estimate2point(estimates[-1], marker_properties)
    end_point.header = marker_header
    cep = estimate2cep
    track_pub.publish(end_point)
    # Adds velocity text to marker:
    # vel_text = estimate2text(estimates[-1])
    # vel_text.header = marker_header
    # track_pub.publish(vel_text) # This line adds a text with velocity of the marker


if __name__ == "__main__":
    rospy.init_node("display_tracks")
    estimates = dict()
    marker_properties = rospy.get_param(rospy.get_name())
    marker_topic = rospy.get_param("~marker_topic", "rviz_marker")
    track_publisher = rospy.Publisher(marker_topic, Marker, queue_size=10)
    est_topic = rospy.get_param("~estimate_topic")
    rospy.Subscriber(
        est_topic,
        RadarEstimate,
        callback=estimate_callback,
        callback_args=(track_publisher, estimates, marker_properties),
        queue_size=10,
    )
    while not rospy.is_shutdown():
        rospy.spin()
