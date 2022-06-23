import numpy as np

import tracking
import track_initiation
import tracking_common
from threading import Lock

mutex = Lock()

class Manager(object):
    """Generic track manager.

    This is an implementation of managers that can be structured with separate
    initiation and termination methods. This is opposed to e.g. an IPDA, where
    there are no explicit confirmation of tracks, except when a threshold is
    set.
    """

    def __init__(
        self,
        tracking_method,
        initiation_method,
        termination_method,
        logger=tracking_common.DefaultLogger(),
    ):
        self.tracking_method = tracking_method
        self.initiation_method = initiation_method
        self.termination_method = termination_method
        self.logger = logger
        self.track_file = dict()
        self.active_tracks = set()

    def __repr__(self):
        return "{} with {} init, {} term".format(
            repr(self.tracking_method),
            repr(self.initiation_method),
            repr(self.termination_method),
        )

    def step(self, measurements, timestamp):
        # Step active tracks
        debug_data = dict()
        latest_estimates = [self.track_file[idx][-1] for idx in self.active_tracks]
        estimates, unused_measurements = self.tracking_method.step(
            latest_estimates, measurements, timestamp
        )
        self.update_track_file(estimates)

        # Initiate new tracks on lidar only
        if self.tracking_method.measurement_type == "lidar":

            # Use unused measurements in initiation
            new_tracks = self.initiation_method.step(unused_measurements, timestamp)
            self.add_new_tracks(new_tracks)
            current_estimates = estimates + [est_list[-1] for est_list in new_tracks]

            # Check for termination - Why am i checking newly confirmed tracks?
            dead_track_indices = self.termination_method.step(estimates)
            self.terminate_tracks(dead_track_indices)

            # Remove this if SequentialRatioTest is set at track initatior
            debug_data["preliminary_tracks"] = self.initiation_method.preliminary_tracks
            return current_estimates, new_tracks
        else:
            # Camera estimates only on already existing targets
            return estimates, []

    def terminate_tracks(self, indices):
        [self.active_tracks.discard(idx) for idx in indices]

    def update_track_file(self, estimates):
        [self.track_file[est.track_index].append(est) for est in estimates]

    def add_new_tracks(self, new_estimates):
        # Assumes that new_estimates = [[est_11, est_12, ...], [est_21, est_22, ...], ...]
        # which means that the initiation method outputs a historic estimate list
        for estimates in new_estimates:
            t_idx = estimates[0].track_index
            self.track_file[t_idx] = estimates
            self.active_tracks.add(t_idx)

    def service_request(self, track_requirement, current_position, N_smooth=1):
        if track_requirement.radius > 0:
            max_radius = track_requirement.radius
        else:
            max_radius = np.inf
        # Pick out requested tracks
        selected_tracks = dict()
        if track_requirement.status.status == track_requirement.status.CONFIRMED:
            selected_tracks = {
                track_idx: self.track_file[track_idx]
                for track_idx in self.active_tracks
            }
        elif track_requirement.status.status == track_requirement.status.PRELIMINARY:
            selected_tracks = self.initiation_method.get_preliminary_tracks()
        output_tracks = dict()
        # Interpolate at given timesteps
        for track_id, track_list_in in selected_tracks.items():
            track_list = track_list_in[:]  # Deep copy to avoid
            if N_smooth > 1:
                track_list = tracking.smooth_estimate_list(track_list, N_smooth)
            new_tracks = []
            track_in_range = False
            for t_ros in track_requirement.time_list:
                t_requested = t_ros.to_sec()
                if t_requested < track_list[0].timestamp:
                    errstr = (
                        "Estimate did not exist at requested time. Requested timestamp=%.3f , Earliest timestamp=%.3f"
                        % (t_requested, track_list[-1].timestamp)
                    )
                    self.logger.logerr(errstr)
                else:
                    selected_estimate = track_list[-1]
                    for est in track_list:
                        if est.timestamp < t_requested:
                            selected_estimate = est
                new_est = self.tracking_method.target_model.step(
                    selected_estimate, t_requested
                )
                self.tracking_method.update_estimate(new_est)
                new_tracks.append(new_est)
                if (
                    np.linalg.norm(
                        current_position
                        - np.array([est.est_prior[0], est.est_prior[2]])
                    )
                    < max_radius
                ):
                    track_in_range = True
            if track_in_range:
                output_tracks[track_id] = new_tracks
        return output_tracks

    def get_preliminary_tracks(self):
        return self.initiation_method.preliminary_tracks

    def get_confirmed_tracks(self):
        return {idx: self.track_file[idx] for idx in self.active_tracks}

    @classmethod
    def MOfNWithPDAF(
        cls,
        M_init,
        N_init,
        N_term,
        gate=tracking_common.TrackGate(),
        target_model=tracking_common.DWNAModel(),
        measurement_model=tracking_common.CartesianMeasurementModel(),
    ):
        PDAF_tracker = tracking.PDAFTracker(target_model, measurement_model, gate)
        mn_init = track_initiation.MOfNInitiation(M_init, N_init, PDAF_tracker, gate)
        mn_term = tracking.TrackTerminatorMofN(N_term)
        return cls(PDAF_tracker, mn_init, mn_term)

    def reset(self):
        mutex.acquire()
        self.track_file = dict()
        self.active_tracks = set()
        self.tracking_method.reset()
        self.termination_method.reset()
        self.initiation_method.reset()
        mutex.release()

class MOfNManager(Manager):
    """Integrated M-of-N Manager. Checks last M of N scans, independent of status changes"""

    def __init__(self, tracking_method, M_init, N_init, M_term=None, N_term=None):
        self.tracking_method = tracking_method
        self.N_init = N_init
        self.M_init = M_init
        self.M_term = M_init if M_term is None else M_term
        self.N_term = N_init if N_term is None else N_term
        self.track_file = dict()
        self.confirmed_tracks = set()
        self.preliminary_tracks = set()
        self.current_track_index = 1

    def step(self, measurements, timestamp):
        previous_confirmed_estimates = [
            self.track_file[idx][-1] for idx in self.confirmed_tracks
        ]
        current_confirmed_estimates, unused_measurements = self.tracking_method.step(
            previous_confirmed_estimates, measurements, timestamp
        )
        previous_preliminary_estimates = [
            self.track_file[idx][-1] for idx in self.preliminary_tracks
        ]
        current_preliminary_estimates, unused_measurements = self.tracking_method.step(
            previous_preliminary_estimates, unused_measurements, timestamp
        )
        self.update_track_file(
            current_confirmed_estimates + current_preliminary_estimates
        )
        self.initiate_tracks(unused_measurements)
        self.update_track_status()

    def initiate_tracks(self, measurements):
        for measurement in measurements:
            estimate = track_initiation.single_point_track_init(
                measurement, self.tracking_method.gate_method.v_max
            )
            estimate.track_index = self.current_track_index
            self.preliminary_tracks.add(estimate.track_index)
            self.track_file[estimate.track_index] = [estimate]
            self.current_track_index += 1

    def update_track_status(self):
        # Check confirmed tracks
        terminated_confirmed_tracks = set()
        for track_index in self.confirmed_tracks:
            est_latest = self.track_file[track_index][-self.N_term :]
            N = len(est_latest)
            M = np.sum([np.min([1, len(est.measurements)]) for est in est_latest])
            if N >= self.N_term and M < self.M_term:
                terminated_confirmed_tracks.add(track_index)
        [
            self.confirmed_tracks.remove(track_index)
            for track_index in terminated_confirmed_tracks
        ]
        # Check preliminary tracks
        new_confirmed_tracks = set()
        terminated_preliminary_tracks = set()
        for track_index in self.preliminary_tracks:
            est_latest = self.track_file[track_index][-self.N_init :]
            N = len(est_latest)
            M = np.sum([np.min([1, len(est.measurements)]) for est in est_latest])
            if N <= self.N_init and M >= self.M_init:  # Confirm track
                new_confirmed_tracks.add(track_index)
            elif N >= self.N_init and M < self.M_init:  # Terminate track
                terminated_preliminary_tracks.add(track_index)
        [
            self.preliminary_tracks.remove(track_index)
            for track_index in new_confirmed_tracks.union(terminated_preliminary_tracks)
        ]
        [self.confirmed_tracks.add(track_index) for track_index in new_confirmed_tracks]

    def get_preliminary_tracks(self):
        return {
            track_index: self.track_file[track_index]
            for track_index in self.preliminary_tracks
        }

    def get_confirmed_tracks(self):
        return {
            track_index: self.track_file[track_index]
            for track_index in self.confirmed_tracks
        }


class IPDAManager(Manager):
    """IPDA Manager.

    In this manager, there are no explicit track confirmation, as in the Manager class. Instead, the existence probability is checked whenever preliminary and confirmed tracks are requested. Tracks are still terminated, in order to avoid maintaining false tracks, which will prevent new tracks being initiated.
    """

    def __init__(self, tracking_method, initial_probability, thres_conf, thres_term):
        self.tracking_method = tracking_method
        self.initial_existence_probability = initial_probability
        self.threshold_confirmation = thres_conf
        self.threshold_termination = thres_term
        self.track_file = dict()
        self.active_tracks = set()
        self.current_track_index = 1

    def step(self, measurements, timestamp):
        # Use measurements to step current estimates
        previous_estimates = [self.track_file[idx][-1] for idx in self.active_tracks]
        current_estimates, unused_measurements = self.tracking_method.step(
            previous_estimates, measurements, timestamp
        )
        self.update_track_file(current_estimates)
        self.terminate_tracks()
        new_tracks = self.initiate_tracks(unused_measurements)
        return current_estimates, new_tracks

    def initiate_tracks(self, measurements):
        new_tracks = []
        for measurement in measurements:

            estimate = track_initiation.single_point_track_init(
                measurement, self.tracking_method.gate_method.v_max
            )
            estimate.track_index = self.current_track_index
            estimate.existence_probability = self.initial_existence_probability
            self.track_file[estimate.track_index] = [estimate]
            self.active_tracks.add(estimate.track_index)
            self.current_track_index += 1
            new_tracks.append([estimate])
        return new_tracks

    def terminate_tracks(self):
        remove_indices = set()
        for track_index in self.active_tracks:
            if (
                self.track_file[track_index][-1].existence_probability
                < self.threshold_termination
            ):
                remove_indices.add(track_index)
        [self.active_tracks.remove(track_index) for track_index in remove_indices]

    def get_preliminary_tracks(self):
        preliminary_tracks = dict()
        for track_index, estimate_list in self.track_file.items():
            if (
                track_index in self.active_tracks
                and estimate_list[-1].existence_probability
                < self.threshold_confirmation
            ):
                preliminary_tracks[track_index] = estimate_list
        return preliminary_tracks

    def get_confirmed_tracks(self):
        confirmed_tracks = dict()
        for track_index, estimate_list in self.track_file.items():
            if (
                track_index in self.active_tracks
                and estimate_list[-1].existence_probability
                > self.threshold_confirmation
            ):
                confirmed_tracks[track_index] = estimate_list
        return confirmed_tracks


class ParticleFilterManager(Manager):
    """A very simple manager that (for now) only manages the single track from the particle filter tracker (i.e. splits the track when it is declared absent etc)"""

    def __init__(self, tracker, confirmation_threshold, termination_threshold):
        self.track_file = dict()
        self.current_track_index = 1
        self.tracker = tracker
        self.confirmation_threshold = confirmation_threshold
        self.termination_threshold = termination_threshold
        self.has_track = False

    def step(self, measurements, timestamp):
        current_estimate, debug_data = self.tracker.step(measurements, timestamp)
        current_estimate.track_index = self.current_track_index
        if (
            self.has_track
            and current_estimate.existence_probability > self.termination_threshold
        ):
            self.track_file[self.current_track_index].append(current_estimate)
        elif (
            self.has_track
            and current_estimate.existence_probability < self.termination_threshold
        ):
            self.current_track_index += 1
            self.has_track = False
        elif (
            not self.has_track
            and current_estimate.existence_probability > self.confirmation_threshold
        ):
            self.track_file[self.current_track_index] = [current_estimate]
            self.has_track = True
        return debug_data

    def reset(self):
        self.track_file = dict()
        self.current_track_index = 1
        self.has_track = False
        self.tracker.reset()
