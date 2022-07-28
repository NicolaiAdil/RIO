from __future__ import division
import numpy as np
from scipy.stats import chi2
from scipy.linalg import block_diag
import itertools
import copy

import tf2_ros
import rospy

import tracking_common


def polar2cartesian_covariance(r, theta, cov_r, cov_theta):
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    R_11 = r ** 2 * cov_theta * s_theta ** 2 + cov_r * c_theta ** 2
    R_12 = (cov_r - r ** 2 * cov_theta) * s_theta * c_theta
    R_22 = r ** 2 * cov_theta * c_theta ** 2 + cov_r * s_theta ** 2
    return np.array([[R_11, R_12], [R_12, R_22]])


def normalized_error_distance(
    first_estimate, second_estimate, first_is_true_state=False
):
    """The estimate covariance of first_estimate is only used if first_is_true_state is False, otherwise it is assumed to be correct"""
    delta = first_estimate.est_posterior - second_estimate.est_posterior
    T = second_estimate.cov_posterior
    if not first_is_true_state:
        T = (
            T
            + first_estimate.cov_posterior
            - 0.4 * (first_estimate.cov_posterior + second_estimate.cov_posterior)
        )  # The minus is due to common process noise

    delta_vec = delta.reshape((4, 1))
    D = delta_vec.T.dot(np.linalg.inv(T).dot(delta_vec)).squeeze()
    return D


class KFTracker(object):
    """Tracker based on nearest neighbor and standard kalman filter."""

    def __init__(self, target_model):
        self.target_model = target_model

    def __repr__(self):
        return "KF tracker"

    def step(self, old_estimates, measurements, timestamp):
        estimates = [
            self.target_model.step(old_est, timestamp) for old_est in old_estimates
        ]
        used_measurements = set()
        for estimate in estimates:
            gated_measurements = self.gate_method.gate_estimate(estimate, measurements)
            used_measurements = used_measurements | gated_measurements
            self.update_estimate(estimate)
        unused_measurements = measurements - used_measurements
        return estimates, unused_measurements


class PDAFTracker(object):
    def __init__(
        self, target_model, measurement_model, gate_method, survival_probability=1
    ):
        self.target_model = target_model
        self.lidar_measurement_model = measurement_model[0]
        self.radar_measurement_model = measurement_model[1]
        self.camera_measurement_model = measurement_model[2]
        self.measurement_model = measurement_model[0]
        self.gate_method = gate_method
        self.P_markov = np.array(
            [[survival_probability, 0], [1 - survival_probability, 1]]
        )

        self.tfBuffer = tf2_ros.Buffer()
        self.ReVolt_tf = tf2_ros.TransformListener(self.tfBuffer)

    def __repr__(self):
        return "PDAF tracker"

    def step(self, old_estimates, measurements, timestamp):

        if self.measurement_type == "lidar":
            self.measurement_model = self.lidar_measurement_model
            estimates, unused_measurements = self.step_lidar(
                old_estimates, measurements, timestamp
            )

        elif self.measurement_type == "radar":
            self.measurement_model = self.radar_measurement_model
            estimates, unused_measurements = self.step_radar(
                old_estimates, measurements, timestamp
            )

        elif self.measurement_type == "camera":

            self.measurement_model = self.camera_measurement_model
            estimates, unused_measurements = self.step_camera(
                old_estimates, measurements, timestamp
            )

        return estimates, unused_measurements

    def step_lidar(self, old_estimates, measurements, timestamp):
        estimates = [
            self.target_model.step(old_est, timestamp) for old_est in old_estimates
        ]
        if self.measurement_model.clutter_model is not None:
            self.measurement_model.clutter_model.update_estimate(measurements)
        used_measurements = set()
        for estimate in estimates:
            gated_measurements = self.gate_method.gate_estimate(
                estimate, measurements, self.measurement_model
            )
            used_measurements = used_measurements | gated_measurements
        # Update detection_probability here.
        self.measurement_model.update_detection_probability(estimates, self.gate_method)
        for estimate in estimates:
            if len(estimate.measurements) > 0:
                self.update_estimate(estimate)
            else:
                self.trivial_update(estimate)
        unused_measurements = measurements - used_measurements
        return estimates, unused_measurements

    def step_radar(self, old_estimates, measurements, timestamp):
        estimates = [
            self.target_model.step(old_est, timestamp) for old_est in old_estimates
        ]
        if self.measurement_model.clutter_model is not None:
            self.measurement_model.clutter_model.update_estimate(measurements)
        used_measurements = set()
        for estimate in estimates:
            gated_measurements = self.gate_method.gate_estimate(
                estimate, measurements, self.measurement_model
            )
            used_measurements = used_measurements | gated_measurements
        # Update detection_probability here.
        self.measurement_model.update_detection_probability(estimates, self.gate_method)
        for estimate in estimates:
            if len(estimate.measurements) > 0:
                self.update_estimate(estimate)
            else:
                self.trivial_update(estimate)
                # print "No association probability={}".format(1.0)
        unused_measurements = measurements - used_measurements
        return estimates, unused_measurements

    def step_camera(self, old_estimates, measurements, timestamp):  # TODO
        estimates = [
            self.target_model.step(old_est, timestamp) for old_est in old_estimates
        ]
        if self.measurement_model.clutter_model is not None:
            self.measurement_model.clutter_model.update_estimate(measurements)
        used_measurements = set()
        for estimate in estimates:
            gated_measurements = self.gate_method.gate_estimate(
                estimate, measurements, self.measurement_model
            )
            used_measurements = used_measurements | gated_measurements
        # Update detection_probability here.
        self.measurement_model.update_detection_probability(estimates, self.gate_method)
        for estimate in estimates:
            if len(estimate.measurements) > 0:
                self.update_camera_estimate(estimate)
            else:
                self.trivial_update(estimate)
        unused_measurements = measurements - used_measurements
        return estimates, unused_measurements

    def update_kinematic_estimate(
        self,
        estimate,
        innovations,
        innovation_covariance,
        association_probabilities,
        kalman_gain,
    ):
        total_innovation = np.zeros(
            2
        )
        for innovation, beta in zip(innovations, association_probabilities[:-1]):
            total_innovation += beta * innovation

        cov_terms = np.zeros((2, 2))

        for innovation, beta in zip(innovations, association_probabilities[:-1]):
            innovation_vec = innovation.reshape((2, 1))
            total_innovation_vec = total_innovation.reshape((2, 1))
            cov_terms += beta * innovation_vec.dot(innovation_vec.T)

        cov_terms -= total_innovation_vec.dot(total_innovation_vec.T)
        soi = np.dot(kalman_gain, np.dot(cov_terms, kalman_gain.T))
        P_c = estimate.cov_prior - np.dot(
            kalman_gain, np.dot(innovation_covariance, kalman_gain.T)
        )
        estimate.est_posterior = estimate.est_prior + kalman_gain.dot(total_innovation)
        estimate.cov_posterior = (
            association_probabilities[-1] * estimate.cov_prior
            + (1 - association_probabilities[-1]) * P_c
            + soi
        )

    def update_estimate(self, estimate):
        n_measurements = len(estimate.measurements)
        P_D = self.measurement_model.get_detection_probability(estimate.track_index)
        P_G = self.gate_method.gate_probability
        n_x = self.target_model.get_state_dimension()
        n_z = self.measurement_model.n_z
        H = self.measurement_model.get_measurement_mapping(estimate)

        z_hat, S = self.measurement_model.predict_measurement(estimate)
        z_hat = z_hat.reshape((n_z, 1))
        S = S.reshape((n_z, n_z))
        measurement_values = [
            measurement.value.reshape((n_z, 1)) for measurement in estimate.measurements
        ]
        innovation_all = [z - z_hat for z in measurement_values]
        S_inv = np.linalg.inv(S)

        if self.measurement_model.clutter_model.clutter_density == "nonparametric":
            clutter_density_inv = self.gate_method.gate_area(S) / len(innovation_all)
        else:
            clutter_density_inv = 1.0 / self.measurement_model.get_clutter_density(
                estimate.est_prior.take((0, 2))
            )

        betas = np.zeros(len(innovation_all) + 1)
        for i, innovation in enumerate(innovation_all):
            measurement_likelihood = np.exp(
                -0.5 * innovation.T.dot(S_inv).dot(innovation)
            ).squeeze() / (2 * np.pi * np.sqrt(np.linalg.det(S)))
            betas[i] = clutter_density_inv * measurement_likelihood * P_D

        betas[-1] = 1 - P_D * P_G
        betas = betas / np.sum(betas)
        kalman_gain = (
            estimate.cov_prior.dot(H.T).dot(S_inv).reshape((n_x, n_z))
        )  # Ensure the kalman gain is a proper matrix, even for 1D measurement
        total_innovation = np.zeros((n_z, 1))
        cov_terms = np.zeros((n_z, n_z))

        for innovation, beta in zip(innovation_all, betas[:-1]):
            total_innovation += beta * innovation
            innovation_vec = innovation.reshape((n_z, 1))
            cov_terms += beta * innovation_vec.dot(innovation_vec.T)

        cov_terms -= total_innovation.dot(total_innovation.T)
        soi = np.dot(kalman_gain, np.dot(cov_terms, kalman_gain.T))
        P_c = estimate.cov_prior - np.dot(kalman_gain, np.dot(S, kalman_gain.T))
        estimate.est_posterior = estimate.est_prior + kalman_gain.dot(
            total_innovation
        ).reshape(n_x)
        estimate.cov_posterior = (
            betas[-1] * estimate.cov_prior + (1 - betas[-1]) * P_c + soi
        )
        self.update_existence_probability(estimate, innovation_all, S)

    def update_camera_estimate(self, estimate):
        n_measurements = len(estimate.measurements)
        P_D = self.measurement_model.get_detection_probability(estimate.track_index)
        P_G = self.gate_method.gate_probability
        n_x = self.target_model.get_state_dimension()
        n_z = self.measurement_model.n_z
        H = self.measurement_model.get_measurement_mapping(estimate)

        z_hat, S = self.measurement_model.predict_measurement(estimate)
        z_hat = z_hat.reshape((1, 1))
        S = S.reshape((1, 1))
        measurement_values = [
            measurement.value.reshape((1, 1)) for measurement in estimate.measurements
        ]
        innovation_all = [z - z_hat for z in measurement_values]
        S_inv = 1 / S

        if self.measurement_model.clutter_model.clutter_density == "nonparametric":
            clutter_density_inv = self.gate_method.gate_area(S) / len(innovation_all)
        else:
            clutter_density_inv = 1.0 / self.measurement_model.get_clutter_density(
                estimate.est_prior.take((0, 2))
            )

        betas = np.zeros(len(innovation_all) + 1)
        for i, innovation in enumerate(innovation_all):
            measurement_likelihood = np.exp(
                -0.5 * innovation.T.dot(S_inv).dot(innovation)
            ).squeeze() / (2 * np.pi * np.sqrt(np.linalg.det(S)))
            betas[i] = clutter_density_inv * measurement_likelihood * P_D

        betas[-1] = 1 - P_D * P_G
        betas = betas / np.sum(betas)
        # TODO Write the handling of unlinear measurments from camera using current position and current estimate
        # H = @(x) [ -1*x(2)/(x(1)^2+x(2)^2),x(1)/((x(2)^2+x(1)^2)) zeros(1,2);zeros(1,4)];%[ 1*x(2)/(2*x(1)^2+x(2)^2),x(1)/(2*(x(2)^2+x(1)^2)) zeros(1,2);zeros(1,4)];
        # print("Position in x, y ", get_ownship_transformation(self.tfBuffer))
        # print "association probs={}".format(betas)
        kalman_gain = (
            estimate.cov_prior.dot(H.T).dot(S_inv).reshape((n_x, n_z))
        )  # Ensure the kalman gain is a proper matrix, even for 1D measurement
        total_innovation = np.zeros((n_z, 1))
        cov_terms = np.zeros((n_z, n_z))

        for innovation, beta in zip(innovation_all, betas[:-1]):
            total_innovation += beta * innovation
            innovation_vec = innovation.reshape((n_z, 1))
            cov_terms += beta * innovation_vec.dot(innovation_vec.T)

        cov_terms -= total_innovation.dot(total_innovation.T)
        soi = np.dot(kalman_gain, np.dot(cov_terms, kalman_gain.T))
        P_c = estimate.cov_prior - np.dot(kalman_gain, np.dot(S, kalman_gain.T))
        estimate.est_posterior = estimate.est_prior + kalman_gain.dot(
            total_innovation
        ).reshape(n_x)
        estimate.cov_posterior = (
            betas[-1] * estimate.cov_prior + (1 - betas[-1]) * P_c + soi
        )
        self.update_existence_probability(estimate, innovation_all, S)

    def trivial_kinematic_update(self, estimate):
        estimate.est_posterior = estimate.est_prior
        estimate.cov_posterior = estimate.cov_prior

    def trivial_update(self, estimate):
        self.trivial_kinematic_update(estimate)
        z_hat, S = self.measurement_model.predict_measurement(estimate)
        self.update_existence_probability(estimate, [], S)

    def update_existence_probability(
        self, estimate, innovations, innovation_covariance
    ):
        pass

    def reset(self):
        pass

    def set_measurement_type(self, measurement_type):
        self.measurement_type = measurement_type


class IPDAFTracker(PDAFTracker):
    def __repr__(self):
        return "MC1-IPDA tracker with P_D={}, p_s={}".format(
            self.measurement_model.detection_probability, self.P_markov[0, 0]
        )

    def update_existence_probability(
        self, estimate, innovations, innovation_covariance
    ):
        # Propagate in time
        estimate.existence_probability = self.P_markov[
            0, 0
        ] * estimate.existence_probability + self.P_markov[0, 1] * (
            1 - estimate.existence_probability
        )
        innovation_prob_sum = 0
        innovation_probs = []
        for innovation in innovations:
            if self.measurement_model.clutter_model.clutter_density == "nonparametric":
                clutter_density_inv = self.gate_method.gate_area(
                    innovation_covariance
                ) / (estimate.existence_probability * len(innovations))
            else:
                clutter_density_inv = 1.0 / self.measurement_model.get_clutter_density(
                    estimate.est_prior.take((0, 2))
                )

            innovation_prob_sum += (
                clutter_density_inv
                * tracking_common.multivariate_normal_pdf(
                    innovation.squeeze(), np.zeros(2), innovation_covariance
                )
            )
            innovation_probs.append(
                tracking_common.multivariate_normal_pdf(
                    innovation.squeeze(), np.zeros(2), innovation_covariance
                )
            )
        delta = (
            self.measurement_model.get_detection_probability(estimate.track_index)
            * self.gate_method.gate_probability
            * (1 - innovation_prob_sum)
        )
        estimate.existence_probability = (
            (1 - delta)
            / (1 - delta * estimate.existence_probability)
            * estimate.existence_probability
        )


class MC2IPDAFTracker(PDAFTracker):
    """Markov chain 2 IPDA tracker

    Is designed to work with a n-vector probability of existence, where the probabilities are explicitly represented (instead of as in the regular IPDA, where only the probability of existence is calculated and the probability of non-existence is 1-ext_prob). This is to be able to work with the general Markov transition probability matrix.

    This requires the estimates to be constructed with a n-vector probability of existence, and also the measurement model to have a list of detection probabilities.
    """

    def __init__(
        self,
        target_model,
        measurement_model,
        gate_method,
        transition_probability_matrix,
    ):
        self.target_model = target_model
        self.measurement_model = measurement_model
        self.gate_method = gate_method
        self.transition_probability_matrix = transition_probability_matrix

    def __repr__(self):
        return "MC2-IPDA tracker with P_D={}, p_s={}".format(
            self.measurement_model.detection_probability,
            1 - self.transition_probability_matrix[2, 0],
        )

    def update_estimate(self, estimate):
        innovations, innovation_covariance = self.calculate_innovations(estimate)
        innovation_probabilities = [
            tracking_common.multivariate_normal_pdf(
                innovation.squeeze(), np.zeros(2), innovation_covariance
            )
            for innovation in innovations
        ]
        self.update_existence_probability(
            estimate, innovation_probabilities, innovation_covariance
        )
        association_probabilities = self.calculate_association_probabilities(
            estimate, innovation_probabilities, innovation_covariance
        )
        kalman_gain = estimate.cov_prior.dot(
            self.measurement_model.measurement_mapping.T
        ).dot(np.linalg.inv(innovation_covariance))
        self.update_kinematic_estimate(
            estimate,
            innovations,
            innovation_covariance,
            association_probabilities,
            kalman_gain,
        )
        return estimate

    def trivial_update(self, estimate):
        innovations, innovation_covariance = self.calculate_innovations(estimate)
        self.update_existence_probability(estimate, [], innovation_covariance)
        self.trivial_kinematic_update(estimate)

    def calculate_innovations(self, estimate):
        z_hat, S = self.measurement_model.predict_measurement(estimate)
        innovations = [z.value - z_hat for z in estimate.measurements]
        return innovations, S

    def update_existence_probability(
        self, estimate, innovation_probabilities, innovation_covariance
    ):
        if isinstance(estimate.existence_probability, float):
            # Convert an existence probability (aka MC1-IPDA) to a vector
            # Assume that the scalar is the existence probability, and that all estimates are equally likely
            n_modes = len(self.measurement_model.detection_probability)
            partial_existence_prob = estimate.existence_probability / n_modes
            estimate.existence_probability = np.array(
                [partial_existence_prob for _ in range(n_modes)]
                + [1 - estimate.existence_probability]
            )

        old_existence_prob = estimate.existence_probability
        prior_existence_prob = self.transition_probability_matrix.dot(
            estimate.existence_probability
        )
        V = self.gate_method.gate_area(innovation_covariance)
        P_G = self.gate_method.gate_probability
        likelihoods = np.zeros_like(prior_existence_prob)

        if self.measurement_model.clutter_model.clutter_density == "nonparametric":
            if len(innovation_probabilities) > 0:
                clutter_density_inv = self.gate_method.gate_area(
                    innovation_covariance
                ) / len(innovation_probabilities)
            else:
                clutter_density_inv = 0.0
        else:
            clutter_density_inv = 1.0 / self.measurement_model.get_clutter_density(
                estimate.est_prior.take((0, 2))
            )
        for j, P_D in enumerate(self.measurement_model.detection_probability):
            likelihoods[j] = 1 - P_D * P_G
            if len(innovation_probabilities) > 0:
                m = len(innovation_probabilities)
                likelihoods[j] += (
                    P_D ** P_G * clutter_density_inv * np.sum(innovation_probabilities)
                )
        likelihoods[-1] = 1
        new_likelihoods = likelihoods * prior_existence_prob
        estimate.existence_probability = new_likelihoods / sum(new_likelihoods)

    def calculate_association_probabilities(
        self, estimate, innovation_probabilities, innovation_covariance
    ):
        mode_probabilities = estimate.existence_probability[
            : len(self.measurement_model.detection_probability)
        ]
        mode_probabilities = mode_probabilities / np.sum(mode_probabilities)
        num_measurements = len(innovation_probabilities)
        association_probabilities = np.zeros(num_measurements + 1)
        gate_area = self.gate_method.gate_area(innovation_covariance)
        gate_probability = self.gate_method.gate_probability
        total_existence_probability = 1 - estimate.existence_probability[-1]

        if self.measurement_model.clutter_model.clutter_density == "nonparametric":
            if len(innovation_probabilities) > 0:
                clutter_density_inv = self.gate_method.gate_area(
                    innovation_covariance
                ) / len(innovation_probabilities)
            else:
                clutter_density_inv = 0.0
        else:
            clutter_density_inv = 1.0 / self.measurement_model.get_clutter_density(
                estimate.est_prior.take((0, 2))
            )

        data_likelihood = np.zeros_like(association_probabilities)
        event_likelihood = np.zeros_like(association_probabilities)

        for i, innovation_probability in enumerate(innovation_probabilities):
            data_likelihood[i] = gate_area / gate_probability * innovation_probability
            event_likelihood_sum = 0
            for j, detection_probability in enumerate(
                self.measurement_model.detection_probability
            ):
                conditional_event_probability = detection_probability * gate_probability
                event_likelihood[i] += (
                    conditional_event_probability
                    * estimate.existence_probability[j]
                    / (total_existence_probability)
                )
        # No assoiation event
        data_likelihood[-1] = 1.0
        for j, detection_probability in enumerate(
            self.measurement_model.detection_probability
        ):
            conditional_event_probability = (
                (1 - detection_probability * gate_probability)
                * gate_area
                / clutter_density_inv
            )
            event_likelihood[-1] += (
                conditional_event_probability
                * estimate.existence_probability[j]
                / (total_existence_probability)
            )

        for i in range(len(association_probabilities)):
            association_probabilities[i] = data_likelihood[i] * event_likelihood[i]
        return association_probabilities / np.sum(association_probabilities)


class TrackTerminatorMofN(object):
    def __init__(self, N_terminate, fuse_prob=0.01):
        self.steps_wo_measurements = dict()
        self.N_terminate = N_terminate
        self.fuse_threshold = chi2(df=2).ppf(1 - fuse_prob)

    def reset(self):
        self.steps_wo_measurements = dict()

    def step(self, estimates):
        term_idx = set()
        # Check for track fusion
        for est_1, est_2 in itertools.combinations(estimates, 2):
            if self.fuse_test(est_1, est_2):
                if est_1.track_index <= est_2.track_index:
                    term_idx.add(est_2.track_index)
                else:
                    term_idx.add(est_1.track_index)
        # Check for number of estimates without measurements
        for estimate in estimates:
            t_idx = estimate.track_index
            if t_idx not in self.steps_wo_measurements.keys():
                self.steps_wo_measurements[t_idx] = 0
            if len(estimate.measurements) > 0:
                self.steps_wo_measurements[t_idx] = 0
            else:
                self.steps_wo_measurements[t_idx] += 1
            if self.steps_wo_measurements[t_idx] > self.N_terminate:
                term_idx.add(t_idx)
        return term_idx

    def fuse_test(self, est1, est2):
        D = normalized_error_distance(est1, est2)
        return D < self.fuse_threshold


class IPDATerminator(object):
    def __init__(self, term_threshold, fuse_prob=0.01):
        self.termination_threshold = term_threshold
        self.fuse_threshold = chi2(df=2).ppf(1 - fuse_prob)

    def step(self, estimates):
        term_idx = set()
        for est_1, est_2 in itertools.combinations(estimates, 2):
            if self.fuse_test(est_1, est_2):
                if est_1.track_index <= est_2.track_index:
                    term_idx.add(est_2.track_index)
                else:
                    term_idx.add(est_1.track_index)
        for estimate in estimates:
            if self.terminate_track_test(estimate) == True:
                term_idx.add(estimate.track_index)
        return term_idx

    def fuse_test(self, est1, est2):
        delta = est1.est_posterior - est2.est_posterior
        T = est1.cov_posterior + est2.cov_posterior - 2 * 0.4 * est1.cov_posterior
        delta_vec = delta.reshape((4, 1))
        D = delta_vec.T.dot(np.linalg.inv(T).dot(delta_vec)).squeeze()
        return D < self.fuse_threshold

    def terminate_track_test(self, estimate):
        return estimate.existence_probability < self.termination_threshold

    def reset(self):
        pass


class MC2IPDATerminator(IPDATerminator):
    def terminate_track_test(self, estimate):
        # The "no exist" state is always the last -> total existence probability = 1-p(no exist)
        return 1 - estimate.existence_probability[-1] < self.termination_threshold


class TrivialTrackInitiator(object):
    """Does not initiate any tracks."""

    def __init__(self):
        pass

    def step(self, measurements, timestamp):
        return []


class TrivialTrackTerminator(object):
    """Does not terminate any tracks."""

    def __init__(self):
        pass

    def step(self, current_estimates):
        return set()


def smooth_lsq(est_list):
    est_list = copy.deepcopy(est_list)
    new_est_list = []
    time = np.array([est.timestamp for est in est_list])
    time -= time[0]
    P_all = [est.cov_posterior for est in est_list]
    y_all = [est.est_posterior for est in est_list]
    F_all = []
    for t in time:
        F, _ = DWNAModel.model(t, 1)
        F_all.append(F)

    P = block_diag(*P_all)
    W = np.linalg.inv(P)
    F = np.vstack(F_all)
    y = np.hstack(y_all)
    pre_term = np.linalg.inv((F.T.dot(W).dot(F)))
    post_term = F.T.dot(W)
    total_term = pre_term.dot(post_term)
    x0 = total_term.dot(y)

    for est, F in zip(est_list, F_all):
        est.est_posterior = F.dot(x0)
        new_est_list.append(est)
    return new_est_list


def smooth_estimate_list(full_est_list, N_smooth):
    full_est_list = copy.deepcopy(full_est_list)
    out_list = []
    for idx in range(len(full_est_list)):
        if idx < N_smooth:
            est_list = full_est_list[: idx + 1]
        else:
            est_list = full_est_list[idx - N_smooth + 1 : idx + 1]

        smoothed_est_list = smooth_lsq(est_list)
        out_est = smoothed_est_list[-1]
        out_list.append(out_est)
    return out_list


def truncate_track_dict(track_dict, t_min, t_max):
    out_dict = {}
    for key, est_list in track_dict.iteritems():
        idx = key
        for est in est_list:
            if est.timestamp >= t_min and est.timestamp <= t_max:
                if idx not in out_dict.keys():
                    out_dict[idx] = []
                out_dict[idx].append(est)
    return out_dict


def sync_estimates_to_timestamp(timestamp, est_list, q):
    est_timestamps = np.array([est.timestamp for est in est_list])
    time_diff = np.abs(est_timestamps - timestamp)
    selected_index = np.argmin(time_diff)
    selected_est = est_list[selected_index]
    dt = timestamp - selected_est.timestamp
    dwna_model = tracking_common.DWNAModel(q)
    F, Q = dwna_model.state_transition_model(dt)
    new_mean = F.dot(selected_est.est_posterior)
    new_cov = F.dot(selected_est.cov_posterior).dot(F.T) + Q
    new_estimate = tracking_common.Estimate(
        timestamp, new_mean, new_cov, True, selected_est.track_index
    )
    return new_estimate


def sync_track_list(timestamps, old_estimate_list):
    sync_estimates = []
    for t in timestamps:
        sync_estimates.append(sync_estimates_to_timestamp(t, old_estimate_list, q=0.1))
    return sync_estimates


def get_speed_and_course(est_list):
    time = np.array([est.timestamp for est in est_list])
    n_vel = np.array([est.est_posterior[1] for est in est_list])
    e_vel = np.array([est.est_posterior[3] for est in est_list])
    speed = np.sqrt(n_vel ** 2 + e_vel ** 2)
    course = np.arctan2(e_vel, n_vel)
    return time, speed, course


def get_measurement_timestamps(measurements_list):
    timestamps = []
    for measurements in measurements_list:
        z = measurements.pop()
        timestamps.append(z.timestamp)
        measurements.add(z)
    return timestamps


def get_ownship_transformation(transformer):
    # timestamp = 0 represents the latest transform
    trans = transformer.lookup_transform("ned", "body", rospy.Time(0))
    return trans
