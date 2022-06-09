import numpy as np
from scipy.stats import poisson
from scipy.linalg import block_diag

import tracking_common


def poisson_clutter(density, volume):
    return lambda x: poisson.pmf(x, density * volume)


def single_point_track_init(measurement, v_max):
    pos = measurement.value
    R = measurement.covariance
    d = v_max ** 2 / 3
    x_pv = np.hstack((pos, np.zeros(2)))
    P_pv = block_diag(R, d * np.identity(2))
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    estimate = tracking_common.Estimate(
        measurement.timestamp, T.dot(x_pv), T.dot(P_pv).dot(T.T), True
    )
    estimate.measurements = set([measurement])
    return estimate


def two_point_track_init(measurement_1, measurement_2):
    pass


class Initiation(object):
    def __init__(
        self, tracking_method, track_gate, logger=tracking_common.DefaultLogger()
    ):
        self.tracking_method = tracking_method
        self.initiator_gate = track_gate
        self.initiators = []
        self.preliminary_tracks = dict()
        self.track_status = dict()
        self.current_index = 1
        self.logger = logger

    def reset(self):
        self.tracking_method.reset()
        self.preliminary_tracks = dict()
        self.track_status = dict()
        self.current_index = 1

    def step_unused_measurements(self, unused_measurements):
        # Use the unused measurements to form new tracks
        used_measurements = set()
        for measurement in unused_measurements:
            for initiator in self.initiators:
                if self.initiator_gate.gate_measurement(initiator, measurement):
                    used_measurements.add(measurement)
                    new_estimates = self.form_track(initiator, measurement)
                    self.preliminary_tracks[self.current_index] = new_estimates
                    self.track_status[self.current_index] = {"M": 0, "N": 0}
                    self.current_index += 1
        self.initiators = unused_measurements - used_measurements
        return

    def get_preliminary_estimates(self):
        return [
            estimates[-1]
            for track_id, estimates in self.preliminary_tracks.items()
            if track_id in self.track_status.keys()
        ]

    def get_preliminary_tracks(self):
        return self.preliminary_tracks

    def form_track(self, initiator, measurement):
        H = self.tracking_method.measurement_model.measurement_mapping
        R1 = initiator.covariance
        R2 = measurement.covariance
        t1 = initiator.timestamp
        t2 = measurement.timestamp
        dt = t2 - t1
        F, _ = self.tracking_method.target_model.state_transition_model(dt)
        H_s = np.vstack((H, np.dot(H, F)))
        z_s = np.hstack((initiator.value, measurement.value))
        R_s = block_diag(R1, R2)
        S_s = np.dot(H_s.T, np.dot(np.linalg.inv(R_s), H_s))
        S_s_inv = np.linalg.inv(S_s)
        est_x1 = np.dot(np.dot(S_s_inv, np.dot(H_s.T, np.linalg.inv(R_s))), z_s)
        est_x2 = np.dot(F, est_x1)
        cov_x1 = S_s_inv
        cov_x2 = np.dot(F, np.dot(S_s_inv, F.T))
        est_1 = tracking_common.Estimate(
            t1, est_x1, cov_x1, is_posterior=True, track_index=self.current_index
        )
        est_2 = tracking_common.Estimate(
            t2, est_x2, cov_x2, is_posterior=True, track_index=self.current_index
        )
        est_1.store_measurement(initiator)
        est_2.store_measurement(measurement)
        return [est_1, est_2]


class MOfNInitiation(Initiation):
    def __init__(
        self,
        M_required,
        N_test,
        tracking_method,
        track_gate,
        logger=tracking_common.DefaultLogger(),
    ):
        super(MOfNInitiation, self).__init__(tracking_method, track_gate)
        self.M_required = M_required
        self.N_test = N_test

    def step(self, measurements, timestamp):
        self.current_confirmed_tracks = 0
        self.current_terminated_tracks = 0
        preliminary_estimates = self.get_preliminary_estimates()
        estimates, unused_measurements = self.tracking_method.step(
            preliminary_estimates, measurements, timestamp
        )
        [
            self.preliminary_tracks[estimate.track_index].append(estimate)
            for estimate in estimates
        ]
        confirmed_estimates = self.update_tracks(estimates)
        self.step_unused_measurements(unused_measurements)
        return confirmed_estimates

    def update_tracks(self, estimates):
        confirmed_estimates = []
        for estimate in estimates:
            t_idx = estimate.track_index
            self.track_status[t_idx]["N"] += 1
            if len(estimate.measurements) > 0:
                self.track_status[t_idx]["M"] += 1
            M = self.track_status[t_idx]["M"]
            N = self.track_status[t_idx]["N"]
            if N <= self.N_test and M >= self.M_required:
                confirmed_estimates.append(self.preliminary_tracks[t_idx])
                del self.track_status[t_idx]
                self.current_confirmed_tracks += 1
            elif N >= self.N_test and M < self.M_required:
                del self.track_status[t_idx]
                self.current_terminated_tracks += 1
        return confirmed_estimates


# This and the following class are too similar to the IPDAManager, hysteresis should probably be added as a special case of that one
class IPDAInitiator(object):
    def __init__(self, tracker, initial_existence, conf_threshold, term_threshold):
        self.tracker = tracker
        self.initial_existence_probability = initial_existence
        self.confirmation_threshold = conf_threshold
        self.termination_threshold = term_threshold
        self.preliminary_tracks = dict()
        self.current_track_index = 1

    def step(self, measurements, timestamp):
        # Assign measurements to existing estimates (i.e. gating)
        used_measurements = set()
        previous_estimates = [
            est_list[-1] for est_list in self.preliminary_tracks.values()
        ]
        current_estimates, unused_measurements = self.tracker.step(
            previous_estimates, measurements, timestamp
        )
        confirmed_estimates = self.update_tracks(current_estimates)
        self.initiate_new_tracks(unused_measurements)
        return confirmed_estimates

    def update_tracks(self, estimates):
        confirmed_track_idx = set()
        terminated_track_idx = set()
        for estimate in estimates:
            self.preliminary_tracks[estimate.track_index].append(estimate)
            if self.check_track_confirmation(estimate) == True:
                confirmed_track_idx.add(estimate.track_index)
            elif self.check_track_termination(estimate) == True:
                terminated_track_idx.add(estimate.track_index)
        [self.preliminary_tracks.pop(term_idx) for term_idx in terminated_track_idx]
        confirmed_tracks = [
            self.preliminary_tracks.pop(idx) for idx in confirmed_track_idx
        ]
        return confirmed_tracks

    def initiate_new_tracks(self, measurements):
        for measurement in measurements:

            estimate = single_point_track_init(
                measurement, self.tracker.gate_method.v_max
            )
            estimate.track_index = self.current_track_index
            estimate.existence_probability = self.initial_existence_probability

            self.preliminary_tracks[estimate.track_index] = [estimate]
            self.current_track_index += 1

    def check_track_confirmation(self, estimate):
        return estimate.existence_probability > self.confirmation_threshold

    def check_track_termination(self, estimate):
        return estimate.existence_probability < self.termination_threshold

    def reset(self):
        self.tracker.reset()
        self.preliminary_tracks = dict()
        self.current_track_index = 1


class MC2IPDAInitiator(IPDAInitiator):
    def check_track_confirmation(self, estimate):
        return np.sum(estimate.existence_probability[:-1]) > self.confirmation_threshold

    def check_track_termination(self, estimate):
        return np.sum(estimate.existence_probability[:-1]) < self.termination_threshold


class SequentialRatioTest(object):
    def __init__(
        self,
        P0,
        P1,
        clutter_density,
        birth_density,
        gate,
        measurement_model,
        target_model,
        cluster_type,
        clutter_map=None,
    ):
        # P1: P(accept target | target)
        # P0: P(accept target | no target)
        self.P1 = P1
        self.P0 = P0
        self.measurement_model = measurement_model
        self.P_D = measurement_model.detection_probability
        self.lower_limit = np.log((1.0 - P1) / (1.0 - P0))
        self.upper_limit = np.log(P1 / P0)
        self.gate = gate
        self.clutter_density = clutter_density
        self.birth_density = birth_density
        self.clusters = []
        self.target_model = target_model
        self.current_track_index = 1
        self.cluster_type = cluster_type
        self.clutter_map = clutter_map
        self.latest_confirmed_clusters = []
        self.latest_terminated_clusters = []

    def reset(self):
        self.clusters = []
        self.current_track_index = 1

    def step(self, measurements, timestamp):
        self.latest_confirmed_clusters = []
        self.latest_terminated_clusters = []
        self.current_confirmed_tracks = 0
        self.current_terminated_tracks = 0
        remove_indices = []
        new_confirmed_tracks = []
        all_used_measurements = []
        for cluster_idx, cluster in enumerate(self.clusters):
            log_LR, unused_measurements = cluster.step(measurements, timestamp)
            used_measurements = measurements - unused_measurements
            all_used_measurements.append(used_measurements)
            if log_LR < self.lower_limit:
                self.current_terminated_tracks += 1
                remove_indices.append(cluster_idx)
                self.latest_terminated_clusters.append(cluster)
            elif log_LR > self.upper_limit:
                self.current_confirmed_tracks += 1
                self.latest_confirmed_clusters.append(cluster)
                estimates = cluster.get_track()
                leaf_nodes = cluster.root_node.find_leaf_nodes()
                for estimate in estimates:
                    estimate.track_index = self.current_track_index
                self.current_track_index += 1
                new_confirmed_tracks.append(estimates)
                remove_indices.append(cluster_idx)
            else:
                pass
        self.terminate_clusters(remove_indices)
        unused_measurements = measurements - frozenset().union(*all_used_measurements)
        self.add_new_clusters(unused_measurements)
        return new_confirmed_tracks

    def add_new_clusters(self, measurements):
        for measurement in measurements:
            if self.clutter_map == "nonparametric":
                clutter_density = self.calculate_initial_nonparametric_clutter_density(
                    measurements
                )
            else:
                clutter_density = self.clutter_density
            new_cluster = self.cluster_type(
                set([measurement]),
                clutter_density,
                self.birth_density,
                self.gate,
                self.target_model,
                self.measurement_model,
                self.clutter_map,
            )
            self.clusters.append(new_cluster)

    def terminate_clusters(self, indices):
        if len(indices) > 0:
            indices.sort(reverse=True)
        for cluster_idx in indices:
            self.clusters.pop(cluster_idx)

    def get_latest_confirmed_clusters(self):
        return self.latest_confirmed_clusters

    def get_latest_terminated_clusters(self):
        return self.latest_terminated_clusters

    def get_current_clusters(self):
        return self.clusters

    def calculate_initial_nonparametric_clutter_density(self, measurements):
        north_coordinates = [z.value[0] for z in measurements]
        east_coordinates = [z.value[1] for z in measurements]
        n_min, n_max = min(north_coordinates), max(north_coordinates)
        e_min, e_max = min(east_coordinates), max(east_coordinates)
        n_k = max(1, len(measurements) - 1)
        for z in measurements:
            r_G = 3 * np.sqrt(np.max(np.linalg.eig(z.covariance)[0]))
        V_k = (n_max - n_min + r_G) * (e_max - e_min + r_G)
        clutter_density = n_k / V_k
        return clutter_density


class Cluster(object):
    def __init__(
        self,
        measurements,
        clutter_density,
        birth_density,
        gate,
        target_model,
        measurement_model,
        clutter_map=None,
    ):
        self.measurement_model = measurement_model
        self.P_D = self.measurement_model.detection_probability
        z = measurements.pop()
        timestamp = z.timestamp
        self.default_measurement_covariance = z.covariance
        self.r_G = 3 * np.sqrt(np.max(np.linalg.eig(z.covariance)[0]))
        measurements.add(z)
        self.init_time = timestamp
        self.logLR_list = []
        self.root_node = None
        self.clutter_density = clutter_density
        self.birth_density = birth_density
        self.gate = gate
        self.clutter_map = clutter_map
        self.target_model = target_model
        self.measurements_all = set()
        self.root_measurements = set()
        [self.root_measurements.add(z) for z in measurements]
        self.step(measurements, timestamp)

    def step(self, measurements, timestamp):
        if (
            self.clutter_map == "nonparametric"
            and timestamp > self.init_time
            and len(measurements) > 0
        ):
            self.update_clutter_density(measurements, timestamp)
        (
            self.target_probability,
            unused_measurements,
        ) = self.calculate_target_probability(measurements, timestamp)
        current_logLR = self.update_logLR()
        return current_logLR, unused_measurements

    def calculate_area(self, timestamp):
        tree_height = len(self.logLR_list) + 1
        radius = 2 * self.r_G + (timestamp - self.init_time) * self.gate.v_max
        V = np.pi * radius ** 2
        return V

    def calculate_measurement_area(self, measurements):
        if len(measurements) == 1:
            timestamps = [z.timestamp for z in measurements]
            return self.calculate_area(timestamps[0])
        else:
            north_coordinates = [z.value[0] for z in measurements]
            east_coordinates = [z.value[1] for z in measurements]
            n_min, n_max = min(north_coordinates), max(north_coordinates)
            e_min, e_max = min(east_coordinates), max(east_coordinates)
            return (n_max - n_min + self.r_G) * (e_max - e_min + self.r_G)

    def update_logLR(self):
        self.logLR_list.append(np.log(self.target_probability))
        current_logLR = self.logLR_list[-1]
        return current_logLR

    def extend_tree(self, current_leaves, measurements, timestamp):
        zero_measurement = tracking_common.Measurement.zero_measurement(timestamp)
        zero_measurement.covariance = self.default_measurement_covariance
        new_leaves = []
        for leaf in current_leaves:
            new_prior = self.target_model.step(leaf.estimate, timestamp)
            gated_measurements = self.gate.gate_estimate(
                new_prior, measurements, self.measurement_model
            )
            [
                self.measurements_all.add(measurement)
                for measurement in gated_measurements
            ]
            for measurement in gated_measurements:
                new_node = Node(
                    measurement,
                    leaf,
                    self.gate.v_max,
                    self.target_model,
                    self.measurement_model,
                )
                leaf.add_child(new_node)
            leaf.add_child(
                Node(
                    zero_measurement,
                    leaf,
                    self.gate.v_max,
                    self.target_model,
                    self.measurement_model,
                )
            )
            new_leaves += leaf.get_children()
        return new_leaves, measurements - self.measurements_all

    def calculate_target_probability(self, measurements, timestamp):
        if self.root_node is not None:  # Extend tree, calculate, leaf hypotheses
            leaf_nodes = self.root_node.find_leaf_nodes()
            new_leaves, unused_measurements = self.extend_tree(
                leaf_nodes, measurements, timestamp
            )
            prob_altHyp = []
            for leaf in new_leaves:
                leaf_prob_altHyp = self.calculate_node_probability(leaf, measurements)
                prob_altHyp.append(leaf_prob_altHyp)
            prob_altHyp = np.sum(prob_altHyp)
        else:  # Construct tree and evaluate the likelihood
            # this should be fixed
            z = measurements.pop()
            self.root_node = Node(
                z, None, self.gate.v_max, self.target_model, self.measurement_model
            )
            self.measurements_all.add(z)
            measurements.add(z)
            unused_measurements = measurements
            prob_altHyp = self.calculate_initial_node_probability(measurements)
            self.root_node.hypothesis_probability = prob_altHyp
        return prob_altHyp, unused_measurements

    def get_track(self):
        measurements = dict()
        for measurement in self.measurements_all:
            if measurement.timestamp not in measurements.keys():
                measurements[measurement.timestamp] = set()
            measurements[measurement.timestamp].add(measurement)
        leaf_nodes = self.root_node.find_leaf_nodes()
        likelihoods = [node.hypothesis_probability for node in leaf_nodes]
        selected_node = leaf_nodes[np.argmax(likelihoods)]
        estimates = []
        while selected_node is not None:
            estimate = selected_node.estimate
            if estimate.timestamp in measurements.keys():
                estimate.measurements = measurements[estimate.timestamp]
            else:
                estimate.measurements = set()
            estimates.append(estimate)
            selected_node = selected_node.parent
        return estimates[::-1]

    def calculate_node_probability(self, node, measurements):
        if self.clutter_map is None or self.clutter_map == "nonparametric":
            if node.measurement.is_zero_measurement():
                hyp_likelihood = 1.0 - self.P_D
            else:
                measurement_likelihood = node.calculate_likelihood_from_prior()
                hyp_likelihood = (
                    self.P_D / self.clutter_density * measurement_likelihood
                )
        else:
            if node.measurement.is_zero_measurement():
                hyp_likelihood = 1.0 - self.P_D
            else:
                measurement_likelihood = node.calculate_likelihood_from_prior()
                current_density = self.clutter_map.get_density(node.measurement.value)
                hyp_likelihood = self.P_D / current_density * measurement_likelihood
        if node.parent is None:
            parent_hyp_likelihood = 1
        else:
            parent_hyp_likelihood = node.parent.hypothesis_probability
        node.hypothesis_probability = parent_hyp_likelihood * hyp_likelihood
        return node.hypothesis_probability

    def get_measurements(self):
        return self.measurements_all

    def update_clutter_density(self, measurements, timestamp):
        V_k = self.calculate_measurement_area(measurements)
        r_k = np.sqrt(V_k / np.pi)
        measurements_in_range = [
            measurement
            for measurement in measurements
            if np.linalg.norm(measurement.value - self.root_node.measurement.value)
            < r_k
        ]
        n_k = len(measurements_in_range)
        if n_k > 1:
            self.clutter_density = (n_k - 1) / V_k


class ReidCluster(Cluster):
    def __repr__(self):
        return "ReidCluster"

    def calculate_initial_node_probability(self, measurements):
        if self.clutter_map is None or self.clutter_map == "nonparametric":
            return self.birth_density / self.clutter_density
        else:
            chosen_density = self.clutter_map.get_density(
                self.root_node.measurement.value
            )
            return self.birth_density / chosen_density


class VanKeukCluster(Cluster):
    def __repr__(self):
        return "VanKeukCluster"

    def calculate_initial_node_probability(self, measurements):
        V_k = self.calculate_area(self.root_node.measurement.timestamp)
        if self.clutter_map is None or self.clutter_map == "nonparametric":
            n_k = len(measurements)
            return self.P_D / (self.clutter_density * V_k)
        else:
            current_density = self.clutter_map.get_density(
                self.root_node.measurement.value
            )
            return self.P_D / (current_density * V_k)


class Node(object):
    def __init__(self, measurement, parent, v_max, target_model, measurement_model):
        self.measurement = measurement
        self.parent = parent
        self.children = []
        self.measurement_model = measurement_model
        self.calculate_estimate(target_model, measurement.timestamp, v_max)

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def find_leaf_nodes(self):
        leaf_nodes = []
        current_nodes = [self]
        while len(current_nodes) > 0:
            current_node = current_nodes[0]
            children = current_node.children
            if len(children) > 0:  # The node has children
                current_nodes = current_nodes + children
            else:  # It is a leaf node
                leaf_nodes.append(current_node)
            current_nodes.pop(0)
        return leaf_nodes

    def get_children(self):
        return self.children

    def calculate_measurement_likelihood_target(self, surveilance_volume):
        if self.parent is None:
            measurement_likelihood_target = 1.0 / surveilance_volume
        else:
            measurement_likelihood_target = self.calculate_likelihood_from_prior()
        self.measurement_likelihood = measurement_likelihood_target
        return measurement_likelihood_target

    def calculate_likelihood_from_prior(self):
        z = self.measurement.value
        H = self.measurement_model.measurement_mapping
        z_hat = H.dot(self.estimate.est_prior)
        S = H.dot(self.estimate.cov_prior).dot(H.T) + self.measurement.covariance
        measurement_likelihood_target = tracking_common.multivariate_normal_pdf(
            z, z_hat, S
        )
        return measurement_likelihood_target

    def calculate_estimate(self, target_model, timestamp, v_max):
        if self.parent is None:  # root node
            estimate = single_point_track_init(self.measurement, v_max)
        else:
            estimate = target_model.step(self.parent.estimate, timestamp)
            if not self.measurement.is_zero_measurement():
                H = self.measurement_model.measurement_mapping
                S = H.dot(estimate.cov_prior).dot(H.T) + self.measurement.covariance
                K = estimate.cov_prior.dot(H.T).dot(np.linalg.inv(S))
                estimate.est_posterior = estimate.est_prior + K.dot(
                    self.measurement.value - H.dot(estimate.est_prior)
                )
                estimate.cov_posterior = (np.identity(4) - K.dot(H)).dot(
                    estimate.cov_prior
                )
            else:
                estimate.est_posterior = estimate.est_prior
                estimate.cov_posterior = estimate.cov_prior
        estimate.store_measurement(self.measurement)
        self.estimate = estimate

    def plot_hypothesis(self, ax, color="k"):
        pass
