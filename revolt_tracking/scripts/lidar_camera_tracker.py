#!/usr/bin/env python
import rospy
import numpy as np
import autoseapy_tracking.tracking as autotrack
import autoseapy_tracking.track_initiation as autoinit
import autoseapy_tracking.tracking_common as tracking_common
import autoseapy_tracking.track_management as automanagers
import autosea_msgs.msg as automsg
import std_msgs.msg as stdmsg
import geometry_msgs.msg as geomsg
from autoseapy_tracking.definitions import NED_munkholmen, NED_seapath
from autosea_msgs.srv import GetTracks, GetTracksResponse
import tf2_ros
import tf
import matplotlib.pyplot as plt
from threading import Lock

from visualization_msgs.msg import Marker

identity_quat = geomsg.Quaternion(0, 0, 0, 1)
identity_pos = geomsg.Point(0, 0, 0)
identity_pose = geomsg.Pose(position=identity_pos, orientation=identity_quat)
mutex = Lock()

global node_start


def ownship_position(transformer):
    # timestamp = 0 represents the latest transform
    trans = tfBuffer.lookup_transform("ned", "body", rospy.Time(0))
    current_position = np.array(
        [trans.transform.translation.x, trans.transform.translation.y]
    )
    return current_position


def measurement2msg(measurement):
    header = stdmsg.Header(
        stamp=rospy.Time.from_sec(measurement.timestamp), frame_id="ned"
    )
    val = automsg.Vector2(measurement.value[0], measurement.value[1])
    cov = automsg.Covariance2(
        measurement.covariance[0, 0],
        measurement.covariance[1, 1],
        measurement.covariance[0, 1],
    )
    return automsg.RadarMeasurement(
        header, val, cov, automsg.RadarMeasurement.CARTESIAN
    )


def transform_cov(covariance):
    T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return T.dot(covariance).dot(T.T)


def est2kinematic(estimate, covariance):
    pos = automsg.Vector2(estimate[0], estimate[1])
    vel = automsg.Vector2(estimate[2], estimate[3])
    covariance = transform_cov(covariance)
    pos_cov = automsg.Covariance2(covariance[0, 0], covariance[1, 1], covariance[1, 0])
    vel_cov = automsg.Covariance2(covariance[2, 2], covariance[3, 3], covariance[3, 2])
    xcov = automsg.CrossCovariance2(
        covariance[2, 0], covariance[3, 0], covariance[2, 1], covariance[3, 1]
    )
    return automsg.KinematicEstimate(pos, vel, pos_cov, vel_cov, xcov)


def publish_estimate(estimate, publisher):

    first_published_estimate = rospy.Time.now()

    # print("Time until first publised estimate: ", first_published_estimate.to_sec() - node_start.to_sec())

    header = stdmsg.Header(
        stamp=rospy.Time.from_sec(estimate.timestamp), frame_id="ned"
    )
    prior_msg = est2kinematic(estimate.est_prior, estimate.cov_prior)
    posterior_msg = est2kinematic(estimate.est_posterior, estimate.cov_posterior)
    measurement_msgs = []
    estimate_msg = automsg.RadarEstimate(
        header, measurement_msgs, prior_msg, posterior_msg, estimate.track_index
    )
    publisher.publish(estimate_msg)


def lidar_scan_callback(msg, args):

    track_pub, manager, ownship_tf, cov_parameters, pdaf_tracker = args
    current_measurements = set()

    # ----- Start mutex ------
    mutex.acquire()

    pdaf_tracker.set_measurement_type("lidar")
    current_position = ownship_position(ownship_tf)
    manager.tracking_method.measurement_model.update_ownship(current_position)
    for cluster in msg.radar_scan:
        value = np.array([cluster.centroid.x, cluster.centroid.y])
        if np.linalg.norm(value) == np.inf:
            rospy.logwarn("Cluster centroid at infinity; discarding measurement")
            continue
        timestamp = cluster.header.stamp.to_sec()
        if cov_parameters["type"] == "cartesian":

            covariance = cov_parameters["cartesian"] * np.identity(2)

        elif cov_parameters["type"] == "polar":

            r = np.linalg.norm(value - current_position)
            bearing = np.arctan2(
                value[0] - current_position[0], value[1] - current_position[1]
            )
            covariance = autotrack.polar2cartesian_covariance(
                r, bearing, cov_parameters["range"], cov_parameters["bearing"]
            )

        z = tracking_common.Measurement(value, timestamp, covariance, logger=rospy)
        current_measurements.add(z)
    # process and update lists
    plt.show(block=True)

    estimates, new_tracks = manager.step(current_measurements, timestamp)

    # ----- End mutex ------
    mutex.release()

    # Publish estimates and status (for visualization)
    common_header = msg.radar_scan[0].header
    [publish_estimate(est, track_pub) for est in estimates]
    for new_track in new_tracks:
        [publish_estimate(estimate, track_pub) for estimate in new_track]


def camera_scan_callback(msg, args):

    track_pub, manager, ownship_tf, cov_parameters, pdaf_tracker = args
    current_measurements = set()

    # ----- Start mutex ------
    mutex.acquire()

    pdaf_tracker.set_measurement_type("camera")
    current_position = ownship_position(ownship_tf)
    manager.tracking_method.measurement_model.update_ownship(current_position)
    timestamp = msg.header.stamp
    for angle_cam2target, confidence in zip(msg.bearing, msg.confidence):

        camera_id = msg.image_header.frame_id
        covariance = cov_parameters["cartesian"]

        # _,q              =  tf.lookupTransform("body",camera_id, rospy.Time(0)) #Output is a quaternion
        # euler            = euler_from_quaternion(q)
        # angle_ned2cam    = euler[2] #Want the rotation about z-axis

        # The object detection output, angle_cam2target, gives angle to target relative to the camera
        # Need to convert this angle to ned-frame  -> ned2cam + cam2target
        # angle_ned2target = angle_ned2cam + np.deg2rad(angle_cam2target)
        measurement = angle_cam2target  # Redundant, but follows convention. Easier to follow the tracking pipeline when using the variable "measurement"

        z = tracking_common.CameraMeasurement(
            camera_id, measurement, timestamp, covariance, logger=rospy
        )
        current_measurements.add(z)

    # Get target estimates. New tracks are only initiated with lidar due to angle measurement
    estimates, _ = manager.step(current_measurements, timestamp)

    # ----- End mutex ---------
    mutex.release()

    # Publish estimates and covariance for visualization
    [publish_estimate(est, track_pub) for est in estimates]

def radar_scan_callback(msg, args):

    track_pub, manager, ownship_tf, cov_parameters, pdaf_tracker = args
    current_measurements = set()

    # ----- Start mutex ------
    mutex.acquire()

    pdaf_tracker.set_measurement_type("radar")
    current_position = ownship_position(ownship_tf)
    manager.tracking_method.measurement_model.update_ownship(current_position)
    for cluster in msg.radar_scan:
        value = np.array([cluster.centroid.x, cluster.centroid.y])
        if np.linalg.norm(value) == np.inf:
            rospy.logwarn("Cluster centroid at infinity; discarding measurement")
            continue
        timestamp = cluster.header.stamp.to_sec()
        if cov_parameters["type"] == "cartesian":

            covariance = cov_parameters["cartesian"] * np.identity(2)

        elif cov_parameters["type"] == "polar":

            r = np.linalg.norm(value - current_position)
            bearing = np.arctan2(
                value[0] - current_position[0], value[1] - current_position[1]
            )
            covariance = autotrack.polar2cartesian_covariance(
                r, bearing, cov_parameters["range"], cov_parameters["bearing"]
            )

        z = tracking_common.Measurement(value, timestamp, covariance, logger=rospy)
        current_measurements.add(z)
    # process and update lists
    plt.show(block=True)

    estimates, new_tracks = manager.step(current_measurements, timestamp)

    # ----- End mutex ------
    mutex.release()

    # Publish estimates and status (for visualization)
    common_header = msg.radar_scan[0].header
    [publish_estimate(est, track_pub) for est in estimates]
    for new_track in new_tracks:
        [publish_estimate(estimate, track_pub) for estimate in new_track]


def service_handler(tracking_requirements, track_manager, ownship_tf, N_smooth):
    t_start = rospy.Time.now()
    current_position = ownship_position(ownship_tf)
    tracks_out = track_manager.service_request(
        tracking_requirements, current_position, N_smooth
    )
    track_list = []

    for idx, est_list in tracks_out.items():
        # Construct a track
        kinematic_estimates = []
        for est in est_list:

            kin_msg = est2kinematic(est.est_posterior, est.cov_posterior)
            header = stdmsg.Header(
                stamp=rospy.Time.from_sec(est.timestamp), frame_id=NED_munkholmen
            )
            kinematic_estimates.append(
                automsg.KinematicEstimateStamped(header, kin_msg)
            )

        track_list.append(
            automsg.Track(
                header=stdmsg.Header(stamp=rospy.Time.now(), frame_id=NED_munkholmen),
                id=idx,
                status=tracking_requirements.status,
                trajectory=kinematic_estimates,
            )
        )

    t_end = rospy.Time.now()
    return GetTracksResponse(track_list=track_list)


if __name__ == "__main__":

    print("Init lidar camera tracker")
    # Update the autopy tracking parameters
    rospy.init_node("lidar_camera_tracker")
    track_parameters = rospy.get_param("~")
    measurement_covariance_parameters_lidar = track_parameters["measurement_covariance_lidar"]
    measurement_covariance_parameters_radar = track_parameters["measurement_covariance_radar"]
    measurement_covariance_parameters_camera = track_parameters["measurement_covariance_camera"]

    # Setup tracking for Revolt
    target_model = tracking_common.DWNAModelRevolt(
        track_parameters["target_process_noise_covariance"]
    )
    track_gate = tracking_common.TrackGate(
        track_parameters["gate_probability"], track_parameters["maximum_velocity"]
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    lidar_measurement_model = tracking_common.CartesianMeasurementModel(
        H,
        measurement_covariance_parameters_lidar["cartesian"] * np.identity(2),
        track_parameters["detection_probability"],
        clutter_model=tracking_common.NonparametricClutterModel(),
    )
    camera_measurement_model = tracking_common.NonlinearMeasurementModel(
        np.array([[1, 0, 0, 0]]),
        measurement_covariance_parameters_camera["cartesian"],
        track_parameters["detection_probability"],
        clutter_model=tracking_common.NonparametricClutterModel(),
    )

    my_measurement_models = []
    my_measurement_models.append(lidar_measurement_model)
    my_measurement_models.append(radar_measurement_model)
    my_measurement_models.append(camera_measurement_model)

    # Setup Telemetron ownship tracker
    tfBuffer = tf2_ros.Buffer()
    ReVolt_tf = tf2_ros.TransformListener(tfBuffer)
    tf = tf.TransformListener()

    if track_parameters["use_ipda"]:
        print("Using IPDA")
        tracker = autotrack.IPDAFTracker(
            target_model,
            my_measurement_models,
            track_gate,
            track_parameters["ipda_survival_probability"],
        )
        ipda_init = autoinit.IPDAInitiator(
            tracker,
            track_parameters["ipda_init_prob"],
            track_parameters["ipda_confirmation_threshold"],
            track_parameters["ipda_termination_threshold"],
        )
        ipda_term = autotrack.IPDATerminator(
            track_parameters["ipda_termination_threshold"]
        )
        track_manager = automanagers.Manager(tracker, ipda_init, ipda_term)
    else:
        print("Using PDAF")
        tracker = autotrack.PDAFTracker(target_model, my_measurement_models, track_gate)
        sprt_init = autoinit.MOfNInitiation(
            track_parameters["M_init"], track_parameters["N_init"], tracker, track_gate
        )
        mn_term = autotrack.TrackTerminatorMofN(track_parameters["N_terminate"])
        track_manager = automanagers.Manager(tracker, sprt_init, mn_term)

    # Setup publisher and service
    estimate_topic = rospy.get_param("~estimate_topic")
    track_publisher = rospy.Publisher(
        estimate_topic, automsg.RadarEstimate, queue_size=10
    )
    scan_topic = rospy.get_param("~scan_topic", "lidar_centroids")
    rospy.Subscriber(
        scan_topic,
        automsg.RadarScan,
        callback=lidar_scan_callback,
        callback_args=(
            track_publisher,
            track_manager,
            ReVolt_tf,
            measurement_covariance_parameters_lidar,
            tracker,
        ),
        queue_size=1,
    )
    radar_scan_topic = rospy.get_param("~radar_topic", "radar_centroids")
    rospy.Subscriber(
        scan_topic,
        automsg.RadarScan,
        callback=radar_scan_callback,
        callback_args=(
            track_publisher,
            track_manager,
            ReVolt_tf,
            measurement_covariance_parameters_radar,
            tracker,
        ),
        queue_size=1,
    )
    rospy.Subscriber(
        rospy.get_param("~camera_measurement_topic"),
        automsg.CameraMeasurement,
        callback=camera_scan_callback,
        callback_args=(
            track_publisher,
            track_manager,
            ReVolt_tf,
            measurement_covariance_parameters_camera,
            tracker,
        ),
        queue_size=5,
    )
    service_name = rospy.get_param("track_manager", "get_tracks")
    track_service = rospy.Service(
        service_name,
        GetTracks,
        lambda req: service_handler(
            req, track_manager, ReVolt_tf, track_parameters["N_smooth"]
        ),
    )

    node_start = rospy.Time.now()

    while not rospy.is_shutdown():
        rospy.spin()
