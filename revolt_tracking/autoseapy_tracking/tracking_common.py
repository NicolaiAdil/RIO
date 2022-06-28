from __future__ import division

import rospy
import tf
from tf.transformations import quaternion_matrix

import numpy as np
from scipy.stats import chi2
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

import conversion as autoconv


class DefaultLogger(object):
    """Print-style logger

    The reason for using this instead of simple print-statements is that rospy
    can be passed as a logger (see e.g. the Measurement class), and the errors
    will be logged using the ros logging functions.
    """

    def __init__(self):
        pass

    def print_log(self, logstring):
        print(logstring)

    def logerr(self, log):
        self.print_log("Error: " + log)

    def loginfo(self, log):
        self.print_log("Info: " + log)

    def logdebug(self, log):
        self.print_log("Debug: " + log)


def multivariate_normal_pdf(z, z_hat, S):
    S_inv = np.linalg.inv(S)
    nu = z - z_hat
    num = np.exp(-0.5 * nu.T.dot(S_inv.dot(nu)))
    denom = 2 * np.pi * np.sqrt(np.linalg.det(S))
    return num / denom


class Measurement(object):
    """Cartesian position measurements.

    The class methods include __eq__ and __hash__, which means the measurements
    can be collected in a python set. This is the logical collection for such
    objects, and is presumed in most other tracking methods.
    """

    def __init__(
        self, value, timestamp, covariance, frame=None, logger=DefaultLogger()
    ):
        self.frame = frame
        self.value = value
        self.timestamp = timestamp
        self.covariance = covariance
        self.logger = logger

        # Measurment-type is set to true for lidar-measurements and false for camera detections

    def is_zero_measurement(self):
        return np.any(np.isnan(self.value))

    def __repr__(self):
        if self.is_zero_measurement():
            meas_str = "Zero measurement"
        else:
            meas_str = "Measurement: {}".format(self.value)
        time_str = "Timestamp: %.2f" % (self.timestamp)
        return meas_str + ", " + time_str

    def __getitem__(self, key):
        if self.is_zero_measurement():
            return None
        else:
            return self.value[key]

    def __eq__(self, other):
        return (
            self.value[0] == other.value[0]
            and self.value[1] == other.value[1]
            and self.timestamp == other.timestamp
        )

    def __hash__(self):
        return hash((self.value[0], self.value[1], self.timestamp))

    @classmethod
    def zero_measurement(cls, timestamp):
        """Construct a zero measurement.

        Used as a pseudomeasurement in MHT-style methods, where the target may be
        undetected. May find use elsewhere as well.
        """
        return cls(np.array([np.nan, np.nan]), timestamp, None)


class CameraMeasurement(Measurement):
    """Angle measurement
    The class methods include __eq__ and __hash__, which means the measurements
    can be collected in a python set. This is the logical collection for such
    objects, and is presumed in most other tracking methods.
    """

    def __init__(
        self, camera_frame, value, timestamp, covariance, logger=DefaultLogger()
    ):
        self.frame = camera_frame
        self.value = value
        self.timestamp = timestamp
        self.covariance = covariance
        self.logger = logger

        # Measurment-type is set to true for lidar-measurements and false for camera detections

    def is_zero_measurement(self):
        return np.isnan(self.value)

    def __getitem__(self, key):
        if self.is_zero_measurement():
            return None
        else:
            return self.value

    def __eq__(self, other):
        return self.value == other.value and self.timestamp == other.timestamp

    def __hash__(self):
        return hash((self.value, self.timestamp))

    @classmethod
    def zero_measurement(cls, timestamp):
        """Construct a zero measurement.

        Used as a pseudomeasurement in MHT-style methods, where the target may be
        undetected. May find use elsewhere as well.
        """
        return cls(np.nan, timestamp, None)


class Estimate(object):
    """An estimate of the target state.

    In general used to represent a estimate of the target state, but may also
    be used to represent ground truth values, in which case the covariance does
    not matter.

    Arguments:
    timestamp
    estimate -- the state estimate
    covariance -- the corresponding covariance

    Keyword arguments:
    is_posterior -- the estimate and covariance are posterior values (default
    False)
    track_index -- the index of the estimate (default None)
    existence_probability -- used in e.g. IPDA
    """

    def __init__(
        self,
        t,
        mean,
        covariance,
        is_posterior=False,
        frame=None,
        track_index=None,
        existence_probability=1,
    ):
        self.timestamp = t
        self.measurements = set()
        self.est_prior = mean
        self.cov_prior = covariance
        self.existence_probability = existence_probability
        self.frame = frame
        if is_posterior:
            self.est_posterior = mean
            self.cov_posterior = covariance
        if track_index is not None:
            self.track_index = track_index
        else:
            self.track_index = -1

    def __repr__(self):
        ID_str = "Track ID: %d" % (self.track_index)
        timestamp_str = "Timestamp: %.2f" % self.timestamp
        return ID_str + ", " + timestamp_str

    def store_measurement(self, measurement):
        self.measurements.add(measurement)

    def get_position(self, posterior=True):

        if posterior:
            return np.array([self.est_posterior[0], self.est_posterior[1]])
        else:
            return np.array([self.est_prior[0], self.est_prior[1]])

    def get_existence_probability(self):
        if isinstance(self.existence_probability, float):
            return self.existence_probability
        else:
            return 1 - self.existence_probability[-1]

    @classmethod
    def from_ais_message(cls, message, origin):
        timestamp = message.header.stamp.to_sec()
        llh = np.array([message.latitude, message.longitude, 0])
        local_pos = autoconv.position_llh_to_local(llh, origin)
        local_vel = message.speed_over_ground * np.array(
            [np.cos(message.course_over_ground), np.sin(message.course_over_ground)]
        )
        local_state = np.array([local_pos[0], local_vel[0], local_pos[1], local_vel[1]])
        return cls(
            timestamp,
            local_state,
            np.identity(4),
            is_posterior=True,
            track_index=message.mmsi,
        )


class StateTransitionModel(object):
    """Generic state transition model."""

    def step(self, estimate, timestamp, is_posterior=False):
        dt = timestamp - estimate.timestamp
        F, Q = self.state_transition_model(dt)
        est_new = F.dot(estimate.est_posterior)
        cov_new = F.dot(estimate.cov_posterior).dot(F.T) + Q
        new_estimate = Estimate(
            timestamp,
            est_new,
            cov_new,
            track_index=estimate.track_index,
            existence_probability=estimate.existence_probability,
        )
        if is_posterior:
            new_estimate.est_posterior = new_estimate.est_prior
            new_estimate.cov_posterior = new_estimate.cov_prior
        return new_estimate

    def draw_transition(self, estimate, timestamp, is_posterior=False):
        dt = timestamp - estimate.timestamp
        F, Q = self.state_transition_model(dt)
        new_mean = F.dot(estimate.est_posterior)
        u, s, v = np.linalg.svd(Q)
        noise = u.dot(np.diag(np.sqrt(s))).dot(
            np.random.normal(size=estimate.est_posterior.shape)
        )
        return Estimate(
            timestamp,
            new_mean + noise,
            estimate.cov_posterior,
            track_index=estimate.track_index,
            is_posterior=is_posterior,
        )


class DWNAModel(StateTransitionModel):
    """Discrete white noise acceleration model."""

    def __init__(self, state_transition_covariance=0.05, dimension=2):
        self.state_transition_covariance = state_transition_covariance
        self.dimension = dimension

    def __repr__(self):
        return (
            "DWNA model with variance q=%.2f m^2/s^4" % self.state_transition_covariance
        )

    def state_transition_model(self, t):
        F_1D = np.array([[1, t], [0, 1]])
        Q_1D = self.state_transition_covariance * np.array(
            [[t ** 4 / 4, t ** 3 / 2], [t ** 3 / 2, t ** 2]]
        )

        if self.dimension == 1:
            return F_1D, Q_1D
        elif self.dimension == 2:
            return block_diag(*(F_1D, F_1D)), block_diag(*(Q_1D, Q_1D))

    def get_state_dimension(self):
        return 2 * self.dimension  # Since the state vector is position and velocity


class DWNAModelRevolt(StateTransitionModel):
    """Discrete white noise acceleration model."""

    def __init__(self, state_transition_covariance=0.05, dimension=2):
        self.state_transition_covariance = state_transition_covariance
        self.dimension = dimension

    def __repr__(self):
        return (
            "DWNA model with variance q=%.2f m^2/s^4" % self.state_transition_covariance
        )

    def state_transition_model(self, t):
        F_2D = np.array([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q_2D = self.state_transition_covariance * np.array(
            [
                [t ** 4 / 4, 0, t ** 3 / 2, 0],
                [0, t ** 4 / 4, 0, t ** 3 / 2],
                [t ** 3 / 2, 0, t ** 2, 0],
                [0, t ** 3 / 2, 0, t ** 2],
            ]
        )

        return F_2D, Q_2D

    def get_state_dimension(self):
        return 2 * self.dimension


class MeasurementModel(object):
    def predict_measurement(self, estimate):

        z_hat = self.measurement_mapping.dot(estimate.est_prior)
        S = (
            self.measurement_mapping.dot(estimate.cov_prior).dot(
                self.measurement_mapping.T
            )
            + self.measurement_covariance
        )

        return z_hat, S

    def generate_measurement(self, state):
        measurement = multivariate_normal(
            mean=self.measurement_mapping.dot(state.est_posterior),
            cov=self.measurement_covariance,
        ).rvs()
        return set(
            [Measurement(measurement, state.timestamp, self.measurement_covariance)]
        )

    def update_ownship(self, new_position):
        self.ownship_position = new_position

    def get_clutter_density(self, z):
        if self.clutter_model == None:
            return None
        else:
            return self.clutter_model.get_clutter_density(z)

    def generate_clutter(self, timestamp, n_clutter, limits):
        measurements = set()
        for n in range(n_clutter):
            val = np.random.uniform(limits[0], limits[1], self.n_z)
            measurements.add(Measurement(val, timestamp, self.measurement_covariance))
        return measurements

    def get_detection_probability(self, track_index):
        return self.detection_probability

    def update_detection_probability(self, estimates, gate_method):
        pass


class NonlinearMeasurementModel(MeasurementModel):
    def __init__(
        self,
        measurement_mapping=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
        measurement_covariance=47 * np.identity(2),
        detection_probability=0.9,
        clutter_model=None,
    ):  # Fix jacobian here as well

        self.measurement_mapping = measurement_mapping
        self.n_z = self.measurement_mapping.shape[0]
        self.measurement_covariance = measurement_covariance
        self.detection_probability = detection_probability
        self.clutter_model = clutter_model
        self.ownship_position = np.zeros(2)
        self.tf = tf.TransformListener()

    def get_measurement_mapping(self, estimate):
        # Define elements

        # pos_x        = self.ownship_position[0]
        # pos_y        = self.ownship_position[1]
        # target_pos_x = estimate.est_prior[0]
        # target_pos_y = estimate.est_prior[1]

        # #Notation: dh_1 -> dh/dx1 and so on
        # dh_1  = -( target_pos_y - pos_y)/( (target_pos_x - pos_x)**2 +(target_pos_y - pos_y)**2)
        # dh_2  = (target_pos_x - pos_x)/( (target_pos_x - pos_x)**2 +(target_pos_y - pos_y)**2)
        # H     = np.array([[dh_1, dh_2,0, 0]])
        # print(estimate.frame)
        t, q = self.tf.lookupTransform(
            "ned", "cam1", rospy.Time(0)
        )  # Output is a quaternion
        rot = quaternion_matrix(q)
        r21 = rot[1][0]
        r22 = rot[1][1]
        r31 = rot[2][0]
        r32 = rot[2][1]
        t2 = t[1]
        t3 = t[2]

        target_pos_x = estimate.est_prior[0]
        target_pos_y = estimate.est_prior[1]

        num = r21 * target_pos_x + r22 * target_pos_y + t2
        den = r31 * target_pos_x + r32 * target_pos_y + t3

        # Notation: dh_1 -> dh/dx1 and so on
        dh_1 = (r21 * den - r31 * num) / (den ** 2 + num ** 2)
        dh_2 = (r22 * den - r32 * num) / (den ** 2 + num ** 2)
        H = np.array([[dh_1, dh_2, 0, 0]])
        return H

    def predict_measurement(self, estimate):
        """
        Using old estimates, predict new measurements

          x     = [target_pos_x, target_pos_y, target_vel_x, target_vel_y]
          z_hat = H*x
          H     = Jacobian(h(x)) -> dh/dx |x = x_hat
          h(x)  = atan2(target_pos_y - ownship_pos_y/target_pos_x - ownship_pos_x)

         H            = self.get_measurement_mapping(estimate)
        target_pos_x = estimate.est_prior[0]
        target_pos_y = estimate.est_prior[1]
        pos_x        = self.ownship_position[0]
        pos_y        = self.ownship_position[1]

        z_hat = np.arctan2((target_pos_y - pos_y), (target_pos_x - pos_x))
        S     = H.dot(estimate.cov_prior).dot(H.T) + self.measurement_covariance

        """

        t, q = self.tf.lookupTransform(
            "ned", "cam1", rospy.Time(0)
        )  # Output is a quaternion
        rot = quaternion_matrix(q)
        r21 = rot[1][0]
        r22 = rot[1][1]
        r31 = rot[2][0]
        r32 = rot[2][1]
        t2 = t[1]
        t3 = t[2]

        target_pos_x = estimate.est_prior[0]
        target_pos_y = estimate.est_prior[1]

        num = r21 * target_pos_x + r22 * target_pos_y + t2
        den = r31 * target_pos_x + r32 * target_pos_y + t3

        H = self.get_measurement_mapping(estimate)
        z_hat = np.arctan2(num, den)
        S = H.dot(estimate.cov_prior).dot(H.T) + self.measurement_covariance

        return z_hat, S


class CartesianMeasurementModel(MeasurementModel):
    def __init__(
        self,
        measurement_mapping=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
        measurement_covariance=47 * np.identity(2),
        detection_probability=0.9,
        clutter_model=None,
    ):
        if len(measurement_mapping.shape) == 1:
            self.measurement_mapping = measurement_mapping.reshape(
                (1, len(measurement_mapping))
            )
        else:
            self.measurement_mapping = measurement_mapping

        self.n_z = self.measurement_mapping.shape[0]
        self.measurement_covariance = measurement_covariance
        self.detection_probability = detection_probability
        self.clutter_model = clutter_model

    def get_measurement_mapping(self, estimate):
        return self.measurement_mapping


class CartesianMeasurementModelMarkovDetectionProbability(CartesianMeasurementModel):
    def get_detection_probability(self, track_index):
        P_D = self.detection_probability.get_total_expectation(track_index)
        return P_D

    def update_detection_probability(self, estimates, gate_method):
        self.detection_probability.update_detection_probability(estimates, gate_method)


class ConvertedMeasurementModel(MeasurementModel):
    def __init__(
        self,
        measurement_mapping,
        range_covariance,
        bearing_covariance,
        detection_probability,
        min_pos_cov=0,
        clutter_model=None,
    ):
        self.measurement_mapping = measurement_mapping
        self.n_z = 2
        self.range_covariance = range_covariance
        self.bearing_covariance = bearing_covariance
        self.ownship_position = np.zeros(2)
        self.clutter_model = clutter_model
        self.min_pos_cov = min_pos_cov
        self.detection_probability = detection_probability

    def predict_measurement(self, estimate):
        estimate_position = self.measurement_mapping.dot(estimate.est_prior)
        current_range = np.linalg.norm(estimate_position - self.ownship_position)
        current_bearing = np.arctan2(
            estimate_position[0] - self.ownship_position[0],
            estimate_position[1] - self.ownship_position[1],
        )

        measurement_covariance = self.polar2cartesian_covariance(
            current_range, current_bearing
        )
        measurement_covariance[0, 0] = np.max(
            [self.min_pos_cov, measurement_covariance[0, 0]]
        )
        measurement_covariance[1, 1] = np.max(
            [self.min_pos_cov, measurement_covariance[1, 1]]
        )
        S = (
            self.measurement_mapping.dot(estimate.cov_prior).dot(
                self.measurement_mapping.T
            )
            + measurement_covariance
        )
        return estimate_position, S

    def polar2cartesian_covariance(self, r, theta):
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        R_11 = (
            r ** 2 * self.bearing_covariance * s_theta ** 2
            + self.range_covariance * c_theta ** 2
        )
        R_12 = (
            (self.range_covariance - r ** 2 * self.bearing_covariance)
            * s_theta
            * c_theta
        )
        R_22 = (
            r ** 2 * self.bearing_covariance * c_theta ** 2
            + self.range_covariance * s_theta ** 2
        )
        return np.array([[R_11, R_12], [R_12, R_22]])

    def get_measurement_mapping(self, estimate):
        return self.measurement_mapping


class ConvertedMeasurementModelMarkovDetectionProbability(ConvertedMeasurementModel):
    def get_detection_probability(self, track_index):
        P_D = self.detection_probability.get_total_expectation(track_index)
        return P_D

    def update_detection_probability(self, estimates, gate_method):
        self.detection_probability.update_detection_probability(estimates, gate_method)


class TrackGate(object):
    def __init__(self, gate_probability=0.99, v_max=25):
        self.gate_probability = gate_probability
        self.gamma = chi2(df=2).ppf(gate_probability)
        self.v_max = v_max
        self.received = 0
        self.accepted = 0

    def __repr__(self):
        return "Track gate with gate probability = %.2f, max. velocity %.1f m/2" % (
            self.gate_probability,
            self.v_max,
        )

    def gate_estimate(self, estimate, measurements, measurement_model):
        measurements_used = set()
        estimate.z_hat, estimate.S = measurement_model.predict_measurement(estimate)

        for measurement in measurements:
            estimate.frame = measurement.frame
            z = measurement.value
            nu = z - estimate.z_hat
            # camera measurement
            if type(estimate.z_hat) == np.float64:
                S_inv = 1 / estimate.S
                self.received += 1
                if nu * (S_inv) * nu < self.gamma:
                    self.accepted += 1
                    estimate.store_measurement(measurement)
                    measurements_used.add(measurement)
            else:
                # lidar measurement
                S_inv = np.linalg.inv(estimate.S)
                if nu.T.dot(S_inv).dot(nu) < self.gamma:
                    estimate.store_measurement(measurement)
                    measurements_used.add(measurement)
        return measurements_used

    def gate_measurement(self, center_measurement, test_measurement):
        dt = test_measurement.timestamp - center_measurement.timestamp
        d_plus = np.maximum(
            test_measurement.value - center_measurement.value - dt * self.v_max,
            np.zeros_like(center_measurement.value),
        )
        d_minus = np.maximum(
            -(test_measurement.value - center_measurement.value + dt * self.v_max),
            np.zeros_like(center_measurement.value),
        )
        d = d_plus + d_minus
        R_initiator = center_measurement.covariance
        R_measurement = test_measurement.covariance
        D = d.dot(np.linalg.inv(R_initiator + R_measurement)).dot(d)
        return D < self.gamma

    def filter_measurements(self, center, measurements):
        measurements_inside = []
        for measurement in measurements:
            if self.gate_measurement(center, measurement):
                measurements_inside.append(measurement)
        return measurements_inside

    def gate_area(self, innovation_covariance):
        return np.pi * np.sqrt(np.linalg.det(self.gamma * innovation_covariance))


class ClutterModel(object):
    def update_estimate(self, measurements):
        pass


class ConstantClutterModel(ClutterModel):
    def __init__(self, clutter_density):
        self.clutter_density = clutter_density

    def get_clutter_density(self, measurement):
        return self.clutter_density


class NonparametricClutterModel(ClutterModel):
    def __init__(self):
        self.clutter_density = "nonparametric"

    def get_clutter_density(self, measurement):
        pass


class Track(object):
    def __init__(self, track_index, initial_estimate=None):
        self.track_index = track_index
        self.track_list = []
        if initial_estimate is not None:
            self.track_list.append(initial_estimate)

    def add_estimate(self, estimate):
        """Adds the estimate to the track."""
        self.track_list.append(estimate)

    def is_equal(self, other_track):
        pass

    def merge_tracks(self, other_track):
        """Merge the information in other_track into this track."""
        pass
