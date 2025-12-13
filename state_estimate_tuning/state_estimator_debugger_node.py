#!/usr/bin/env python3
"""
EKF Debug Plotter (ROS 2 Jazzy)

Compares:
- Yaw: EKF (Odometry) vs LIO truth (/lio/pose)
- Roll/Pitch: EKF vs LIO truth
Also shows:
- Position (N–E, NED) tracks: EKF vs LIO truth
- Z position (D in NED) vs time: EKF vs LIO truth
- Velocity panel: EKF only (truth PoseStamped has no velocity)

NEW:
- Second window plotting radar extrinsics (position and attitude of radar in body frame)
  including ground truth extrinsics for comparison, with GT in same color (dashed).

UPDATE:
- Radar attitude is now shown in 3 separate plots (roll, pitch, yaw) to make convergence easier to assess.
"""

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf_transformations
import matplotlib.pyplot as plt
import time


def ssa(angle):
    """Wrap to (-pi, pi]."""
    a = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if np.isclose(a, -np.pi):
        a = np.pi
    return a


def unwrap_append(prev_unwrapped, new_wrapped):
    if prev_unwrapped is None:
        return float(new_wrapped)
    k = round((prev_unwrapped - new_wrapped) / (2 * np.pi))
    return float(new_wrapped + 2 * np.pi * k)


def enu_pose_to_ned_euler_and_ne(qx, qy, qz, qw, px, py, pz):
    """
    Convert an ENU pose (qx,qy,qz,qw, px,py,pz) to NED roll,pitch,yaw and (N,E).
    """
    R_enu_to_ned = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, -1]], dtype=float)
    R4 = np.eye(4, dtype=float)
    R4[:3, :3] = R_enu_to_ned

    # Orientation: ENU -> NED via similarity transform
    M_enu = tf_transformations.quaternion_matrix([qx, qy, qz, qw])  # 4x4
    M_ned = R4 @ M_enu @ R4.T
    r, p, y = tf_transformations.euler_from_matrix(M_ned)
    r, p, y = ssa(r), ssa(p), ssa(y)

    # Position: x=E, y=N, z=U  ->  N=y, E=x (D=-z not used here)
    N = float(py)
    E = float(px)
    return r, p, y, N, E


class EKFDebugPlotter(Node):
    def __init__(self):
        super().__init__('ekf_debug_plotter')

        # Parameters
        self.declare_parameter('topic_state', '/rio/pose')
        self.declare_parameter('topic_truth_pose', '/lio/pose')
        self.declare_parameter('topic_radar_extrinsics', '/radar/extrinsics')
        self.declare_parameter('truth_frame', 'NED')  # 'ENU' or 'NED'
        self.declare_parameter('window_secs', 300.0)
        self.declare_parameter('plot_rate_hz', 5.0)
        self.declare_parameter('flip_warn_thresh_deg', 150.0)

        self.topic_state = self.get_parameter('topic_state').value
        self.topic_truth_pose = self.get_parameter('topic_truth_pose').value
        self.topic_radar_extrinsics = self.get_parameter('topic_radar_extrinsics').value
        self.truth_frame = str(self.get_parameter('truth_frame').value).upper()
        self.window_secs = float(self.get_parameter('window_secs').value)
        self.plot_dt = 1.0 / float(self.get_parameter('plot_rate_hz').value)
        self.flip_warn_thresh = np.deg2rad(float(self.get_parameter('flip_warn_thresh_deg').value))

        # Time zero
        self.t0 = None

        # Per-series time/value histories (each with bounded length)
        self.t_yaw_est, self.yaw_est_hist = deque(), deque()
        self.t_yaw_truth, self.yaw_truth_hist = deque(), deque()

        self.t_roll_est, self.roll_est_hist = deque(), deque()
        self.t_pitch_est, self.pitch_est_hist = deque(), deque()
        self.t_roll_truth, self.roll_truth_hist = deque(), deque()
        self.t_pitch_truth, self.pitch_truth_hist = deque(), deque()

        # Z (D in NED) position histories
        self.t_z_est, self.z_est_hist = deque(), deque()
        self.t_z_truth, self.z_truth_hist = deque(), deque()

        # Velocities (EKF only here)
        self.t_vN_est, self.vN_est = deque(), deque()
        self.t_vE_est, self.vE_est = deque(), deque()

        # NE tracks
        self.ne_est = deque()    # (N,E)
        self.ne_truth = deque()  # (N,E)

        # Flip markers (times)
        self.flip_marks_t = deque()

        # Unwrap trackers
        self._last_yaw_est = None
        self._last_yaw_truth = None

        # Radar extrinsics histories (body -> radar)
        self.t_extr = deque()
        self.ex_px = deque()
        self.ex_py = deque()
        self.ex_pz = deque()
        self.ex_roll = deque()
        self.ex_pitch = deque()
        self.ex_yaw = deque()
        self._last_extr_yaw = None  # unwrap

        # Ground-truth radar extrinsics (body -> radar)
        # l_BR_B: [0.077, 0.016, -0.063]
        # q_R_B: [0.963, -0.021, -0.265, 0.021] (Radar->Body, xyzw)
        self.gt_pos = np.array([0.077, 0.016, -0.063], dtype=float)
        q_rb = [0.963, -0.021, -0.265, 0.021]  # radar->body
        q_br = [-q_rb[0], -q_rb[1], -q_rb[2], q_rb[3]]  # body->radar
        r_gt, p_gt, y_gt = tf_transformations.euler_from_quaternion(q_br)
        r_gt, p_gt, y_gt = ssa(r_gt), ssa(p_gt), ssa(y_gt)
        self.gt_att_deg = np.degrees([r_gt, p_gt, y_gt])

        # Subscribers
        self.create_subscription(Odometry, self.topic_state, self.cb_state, 10)
        self.create_subscription(PoseStamped, self.topic_truth_pose, self.cb_truth_pose, 10)
        self.create_subscription(PoseStamped, self.topic_radar_extrinsics, self.cb_radar_extrinsics, 10)

        # Figure + timer
        self._make_figure()
        self._plot_timer = self.create_timer(self.plot_dt, self._on_plot_timer)

        self.get_logger().info(
            "EKF Debug Plotter started.\n"
            f" Subscribed to:\n"
            f"  state:     {self.topic_state}\n"
            f"  truthPose: {self.topic_truth_pose}  (frame={self.truth_frame})\n"
            f"  extrinsics:{self.topic_radar_extrinsics}\n"
            f" Window = {self.window_secs}s, plot_rate = {1.0 / self.plot_dt:.1f} Hz"
        )

    # ----------------------- Callbacks -----------------------

    def _now_s(self):
        if self.t0 is None:
            self.t0 = time.time()
        return time.time() - self.t0

    def cb_state(self, msg: Odometry):
        t = self._now_s()

        # EKF orientation (RPY in NED)
        q = msg.pose.pose.orientation
        r, p, y = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        r, p, y = ssa(r), ssa(p), ssa(y)

        # Save yaw (unwrapped), roll, pitch
        un_yaw_est = unwrap_append(self._last_yaw_est, y)
        self._last_yaw_est = un_yaw_est
        self._append_tv(self.t_yaw_est, self.yaw_est_hist, t, un_yaw_est)
        self._append_tv(self.t_roll_est, self.roll_est_hist, t, r)
        self._append_tv(self.t_pitch_est, self.pitch_est_hist, t, p)

        # NE (NED) for map
        N = float(msg.pose.pose.position.x)
        E = float(msg.pose.pose.position.y)
        self._append_limited(self.ne_est, (N, E))

        # Z (D in NED) position
        D = float(msg.pose.pose.position.z)
        self._append_tv(self.t_z_est, self.z_est_hist, t, D)

        # Vel (NED)
        vN = float(msg.twist.twist.linear.x)
        vE = float(msg.twist.twist.linear.y)
        self._append_tv(self.t_vN_est, self.vN_est, t, vN)
        self._append_tv(self.t_vE_est, self.vE_est, t, vE)

    def cb_truth_pose(self, msg: PoseStamped):
        t = self._now_s()

        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w

        if self.truth_frame == 'ENU':
            r, p, y, N, E = enu_pose_to_ned_euler_and_ne(qx, qy, qz, qw, px, py, pz)
            D = -float(pz)  # ENU z is Up -> NED D = -z
        else:
            # Already NED
            r, p, y = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
            r, p, y = ssa(r), ssa(p), ssa(y)
            N, E = float(px), float(py)
            D = float(pz)

        # Append orientation
        un = unwrap_append(self._last_yaw_truth, y)
        self._last_yaw_truth = un
        self._append_tv(self.t_yaw_truth, self.yaw_truth_hist, t, un)
        self._append_tv(self.t_roll_truth, self.roll_truth_hist, t, r)
        self._append_tv(self.t_pitch_truth, self.pitch_truth_hist, t, p)

        # Append position (NE, Z)
        self._append_limited(self.ne_truth, (N, E))
        self._append_tv(self.t_z_truth, self.z_truth_hist, t, D)

        # Flip detector (vs latest EKF yaw if present)
        if len(self.yaw_est_hist) > 0:
            diff = ssa(self.yaw_truth_hist[-1] - self.yaw_est_hist[-1])
            if abs(abs(diff) - np.pi) < np.deg2rad(15) or abs(diff) > self.flip_warn_thresh:
                self._append_limited(self.flip_marks_t, t)
                self.get_logger().warn(f"Possible 180° flip: |Δyaw|={np.degrees(abs(diff)):.1f}° at t={t:.1f}s")

    def cb_radar_extrinsics(self, msg: PoseStamped):
        """
        Pose of radar in body frame (topic_radar_extrinsics, PoseStamped).
        We log position and attitude and plot them in a second window.
        """
        t = self._now_s()

        px = float(msg.pose.position.x)
        py = float(msg.pose.position.y)
        pz = float(msg.pose.position.z)

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        r, p, y = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
        r, p, y = ssa(r), ssa(p), ssa(y)

        # Unwrap yaw of extrinsics for nicer plot
        un_yaw_ex = unwrap_append(self._last_extr_yaw, y)
        self._last_extr_yaw = un_yaw_ex

        # Append keeping all arrays the same length
        self.t_extr.append(float(t))
        self.ex_px.append(px)
        self.ex_py.append(py)
        self.ex_pz.append(pz)
        self.ex_roll.append(r)
        self.ex_pitch.append(p)
        self.ex_yaw.append(un_yaw_ex)

        # Limit length
        maxlen = 5000
        if len(self.t_extr) > maxlen:
            self.t_extr.popleft()
            self.ex_px.popleft()
            self.ex_py.popleft()
            self.ex_pz.popleft()
            self.ex_roll.popleft()
            self.ex_pitch.popleft()
            self.ex_yaw.popleft()

    # ----------------------- Plotting -----------------------

    def _make_figure(self):
        plt.ion()
        # Main figure (existing)
        self.fig = plt.figure(figsize=(12, 14))
        gs = self.fig.add_gridspec(5, 1, height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0], hspace=0.35)

        # Yaw
        self.ax_head = self.fig.add_subplot(gs[0, 0])
        self.ax_head.set_title("Yaw vs Time")
        self.ax_head.set_ylabel("Yaw [deg]")
        self.ax_head.set_xlabel("Time [s]")
        self.l_est_head, = self.ax_head.plot([], [], label="EKF yaw", linewidth=1.2)
        self.l_truth_head, = self.ax_head.plot([], [], label="LIO truth yaw", linewidth=2.4)
        self.flip_scatter = self.ax_head.scatter([], [], marker='x', label="Flip?")
        self.ax_head.legend(loc='best')
        self.ax_head.grid(True)

        # Roll/Pitch
        self.ax_rp = self.fig.add_subplot(gs[1, 0])
        self.ax_rp.set_title("Roll & Pitch vs Time")
        self.ax_rp.set_ylabel("Angle [deg]")
        self.ax_rp.set_xlabel("Time [s]")
        self.l_roll_est, = self.ax_rp.plot([], [], label="EKF roll", linewidth=1.2)
        self.l_pitch_est, = self.ax_rp.plot([], [], label="EKF pitch", linewidth=1.2)
        self.l_roll_truth, = self.ax_rp.plot([], [], label="LIO roll", linewidth=2.4)
        self.l_pitch_truth, = self.ax_rp.plot([], [], label="LIO pitch", linewidth=2.4)
        self.ax_rp.legend(loc='best')
        self.ax_rp.grid(True)

        # Position NE
        self.ax_ne = self.fig.add_subplot(gs[2, 0])
        self.ax_ne.set_title("Position (N–E in NED)")
        self.ax_ne.set_xlabel("E [m]")
        self.ax_ne.set_ylabel("N [m]")
        self.l_est_ne, = self.ax_ne.plot([], [], label="EKF track")
        self.l_truth_ne, = self.ax_ne.plot([], [], linestyle='None', marker='.', label="LIO truth")
        self.ax_ne.axis('equal')
        self.ax_ne.grid(True)
        self.ax_ne.legend(loc='best')

        # Z (D in NED)
        self.ax_z = self.fig.add_subplot(gs[3, 0])
        self.ax_z.set_title("Vertical Position vs Time (D in NED)")
        self.ax_z.set_ylabel("D [m] (down positive)")
        self.ax_z.set_xlabel("Time [s]")
        self.l_z_est, = self.ax_z.plot([], [], label="EKF D")
        self.l_z_truth, = self.ax_z.plot([], [], label="LIO D", linewidth=2.0)
        self.ax_z.grid(True)
        self.ax_z.legend(loc='best')

        # Velocity (EKF only)
        self.ax_vel = self.fig.add_subplot(gs[4, 0])
        self.ax_vel.set_title("Velocity Components vs Time (NED)")
        self.ax_vel.set_ylabel("v [m/s]")
        self.ax_vel.set_xlabel("Time [s]")
        self.l_vN_est, = self.ax_vel.plot([], [], label="vN EKF")
        self.l_vE_est, = self.ax_vel.plot([], [], label="vE EKF")
        self.ax_vel.grid(True)
        self.ax_vel.legend(loc='best')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            plt.show(block=False)
        except Exception:
            pass

        # Second figure for radar extrinsics (position + attitude)
        self.fig_extr = plt.figure(figsize=(10, 10))
        gs2 = self.fig_extr.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.30)

        # --- Radar extrinsics: ATTITUDE as 3 separate plots (roll/pitch/yaw) ---
        att_gs = gs2[0, 0].subgridspec(3, 1, hspace=0.10)

        self.ax_extr_roll = self.fig_extr.add_subplot(att_gs[0, 0])
        self.ax_extr_pitch = self.fig_extr.add_subplot(att_gs[1, 0], sharex=self.ax_extr_roll)
        self.ax_extr_yaw = self.fig_extr.add_subplot(att_gs[2, 0], sharex=self.ax_extr_roll)

        self.ax_extr_roll.set_title("Radar Extrinsics: Attitude")
        self.ax_extr_roll.set_ylabel("Roll [deg]")
        self.ax_extr_pitch.set_ylabel("Pitch [deg]")
        self.ax_extr_yaw.set_ylabel("Yaw [deg]")
        self.ax_extr_yaw.set_xlabel("Time [s]")

        for ax in (self.ax_extr_roll, self.ax_extr_pitch, self.ax_extr_yaw):
            ax.grid(True)

        # Estimated attitude lines
        self.l_ex_roll, = self.ax_extr_roll.plot([], [], label="roll est")
        self.l_ex_pitch, = self.ax_extr_pitch.plot([], [], label="pitch est")
        self.l_ex_yaw, = self.ax_extr_yaw.plot([], [], label="yaw est")

        # Ground-truth attitude (same colors, dashed)
        self.l_ex_roll_gt, = self.ax_extr_roll.plot([], [], '--', label="roll GT",
                                                    color=self.l_ex_roll.get_color())
        self.l_ex_pitch_gt, = self.ax_extr_pitch.plot([], [], '--', label="pitch GT",
                                                      color=self.l_ex_pitch.get_color())
        self.l_ex_yaw_gt, = self.ax_extr_yaw.plot([], [], '--', label="yaw GT",
                                                  color=self.l_ex_yaw.get_color())

        self.ax_extr_roll.legend(loc='best')
        self.ax_extr_pitch.legend(loc='best')
        self.ax_extr_yaw.legend(loc='best')

        # Hide upper x tick labels (shared x)
        plt.setp(self.ax_extr_roll.get_xticklabels(), visible=False)
        plt.setp(self.ax_extr_pitch.get_xticklabels(), visible=False)

        # Position of radar wrt body
        self.ax_extr_pos = self.fig_extr.add_subplot(gs2[1, 0])
        self.ax_extr_pos.set_title("Radar Extrinsics: Position (body → radar)")
        self.ax_extr_pos.set_ylabel("Position [m]")
        self.ax_extr_pos.set_xlabel("Time [s]")
        self.ax_extr_pos.grid(True)

        # Estimated position lines
        self.l_ex_px, = self.ax_extr_pos.plot([], [], label="p_x est")
        self.l_ex_py, = self.ax_extr_pos.plot([], [], label="p_y est")
        self.l_ex_pz, = self.ax_extr_pos.plot([], [], label="p_z est")

        px_color = self.l_ex_px.get_color()
        py_color = self.l_ex_py.get_color()
        pz_color = self.l_ex_pz.get_color()

        # GT with same colors (dashed)
        self.l_ex_px_gt, = self.ax_extr_pos.plot([], [], '--', label="p_x GT", color=px_color)
        self.l_ex_py_gt, = self.ax_extr_pos.plot([], [], '--', label="p_y GT", color=py_color)
        self.l_ex_pz_gt, = self.ax_extr_pos.plot([], [], '--', label="p_z GT", color=pz_color)

        self.ax_extr_pos.legend(loc='best')

        self.fig_extr.canvas.draw()
        self.fig_extr.canvas.flush_events()
        try:
            plt.show(block=False)
        except Exception:
            pass

    def _on_plot_timer(self):
        try:
            self._refresh_plot()
        except Exception as e:
            self.get_logger().warn(f"Plot refresh error: {e}")

    @staticmethod
    def _finite_xy(tx, yy):
        if len(tx) == 0 or len(yy) == 0:
            return np.array([]), np.array([])
        tx = np.asarray(tx)
        yy = np.asarray(yy)
        m = np.isfinite(tx) & np.isfinite(yy)
        return tx[m], yy[m]

    def _refresh_plot(self):
        # Current visible window based on wall time
        now = self._now_s()
        tmin = max(0.0, now - self.window_secs)
        tmax = now

        # ---- Yaw (deg) ----
        te, ye = self._finite_xy(self.t_yaw_est, np.degrees(self.yaw_est_hist))
        tt, yt = self._finite_xy(self.t_yaw_truth, np.degrees(self.yaw_truth_hist))

        self.l_est_head.set_data(te, ye)
        self.l_truth_head.set_data(tt, yt)

        # Flip markers
        flips_t = np.asarray(self.flip_marks_t)
        if flips_t.size and te.size:
            flips_y = np.interp(flips_t, te, ye)
            self.flip_scatter.remove()
            self.flip_scatter = self.ax_head.scatter(flips_t, flips_y, marker='x')
        else:
            flips_y = np.array([])

        self.ax_head.set_xlim([tmin, tmax])
        self.ax_head.relim()
        if flips_t.size:
            self.ax_head.update_datalim(np.column_stack([flips_t, flips_y]))
        self.ax_head.autoscale_view(scalex=False, scaley=True)

        # ---- Roll/Pitch (deg) ----
        tr_e, rr_e = self._finite_xy(self.t_roll_est, np.degrees(self.roll_est_hist))
        tp_e, pp_e = self._finite_xy(self.t_pitch_est, np.degrees(self.pitch_est_hist))
        tr_t, rr_t = self._finite_xy(self.t_roll_truth, np.degrees(self.roll_truth_hist))
        tp_t, pp_t = self._finite_xy(self.t_pitch_truth, np.degrees(self.pitch_truth_hist))

        self.l_roll_est.set_data(tr_e, rr_e)
        self.l_pitch_est.set_data(tp_e, pp_e)
        self.l_roll_truth.set_data(tr_t, rr_t)
        self.l_pitch_truth.set_data(tp_t, pp_t)

        self.ax_rp.set_xlim([tmin, tmax])
        self.ax_rp.relim()
        self.ax_rp.autoscale_view(scalex=False, scaley=True)

        # ---- Position NE ----
        ne_est_arr = np.array(self.ne_est) if len(self.ne_est) else np.empty((0, 2))
        ne_tru_arr = np.array(self.ne_truth) if len(self.ne_truth) else np.empty((0, 2))
        if ne_est_arr.shape[0] > 0:
            self.l_est_ne.set_data(ne_est_arr[:, 1], ne_est_arr[:, 0])  # x=E, y=N
        else:
            self.l_est_ne.set_data([], [])
        if ne_tru_arr.shape[0] > 0:
            self.l_truth_ne.set_data(ne_tru_arr[:, 1], ne_tru_arr[:, 0])
        else:
            self.l_truth_ne.set_data([], [])

        if ne_est_arr.shape[0] + ne_tru_arr.shape[0] > 1:
            allE = []
            allN = []
            if ne_est_arr.shape[0] > 0:
                allE += ne_est_arr[:, 1].tolist()
                allN += ne_est_arr[:, 0].tolist()
            if ne_tru_arr.shape[0] > 0:
                allE += ne_tru_arr[:, 1].tolist()
                allN += ne_tru_arr[:, 0].tolist()
            padE = (max(allE) - min(allE)) * 0.1 + 1.0
            padN = (max(allN) - min(allN)) * 0.1 + 1.0
            self.ax_ne.set_xlim([min(allE) - padE, max(allE) + padE])
            self.ax_ne.set_ylim([min(allN) - padN, max(allN) + padN])

        # ---- Z (D in NED) ----
        t_ze, ze = self._finite_xy(self.t_z_est, self.z_est_hist)
        t_zt, zt = self._finite_xy(self.t_z_truth, self.z_truth_hist)

        self.l_z_est.set_data(t_ze, ze)
        self.l_z_truth.set_data(t_zt, zt)

        self.ax_z.set_xlim([tmin, tmax])
        self.ax_z.relim()
        self.ax_z.autoscale_view(scalex=False, scaley=True)

        # ---- Velocity (EKF only) ----
        t_vNe, vNe = self._finite_xy(self.t_vN_est, self.vN_est)
        t_vEe, vEe = self._finite_xy(self.t_vE_est, self.vE_est)

        self.l_vN_est.set_data(t_vNe, vNe)
        self.l_vE_est.set_data(t_vEe, vEe)

        self.ax_vel.set_xlim([tmin, tmax])
        self.ax_vel.relim()
        self.ax_vel.autoscale_view(scalex=False, scaley=True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # ---- Radar extrinsics (second figure) ----
        t_ex = np.asarray(self.t_extr, dtype=float)
        if t_ex.size > 0:
            mask = (t_ex >= tmin) & (t_ex <= tmax)
            t_ex_w = t_ex[mask]

            ex_roll = np.degrees(np.asarray(self.ex_roll)[mask])
            ex_pitch = np.degrees(np.asarray(self.ex_pitch)[mask])
            ex_yaw = np.degrees(np.asarray(self.ex_yaw)[mask])

            ex_px = np.asarray(self.ex_px)[mask]
            ex_py = np.asarray(self.ex_py)[mask]
            ex_pz = np.asarray(self.ex_pz)[mask]

            # Estimated
            self.l_ex_roll.set_data(t_ex_w, ex_roll)
            self.l_ex_pitch.set_data(t_ex_w, ex_pitch)
            self.l_ex_yaw.set_data(t_ex_w, ex_yaw)

            self.l_ex_px.set_data(t_ex_w, ex_px)
            self.l_ex_py.set_data(t_ex_w, ex_py)
            self.l_ex_pz.set_data(t_ex_w, ex_pz)

            # Ground truth (constant lines, same colors as estimate)
            roll_gt, pitch_gt, yaw_gt = self.gt_att_deg
            px_gt, py_gt, pz_gt = self.gt_pos

            self.l_ex_roll_gt.set_data(t_ex_w, np.full_like(t_ex_w, roll_gt))
            self.l_ex_pitch_gt.set_data(t_ex_w, np.full_like(t_ex_w, pitch_gt))
            self.l_ex_yaw_gt.set_data(t_ex_w, np.full_like(t_ex_w, yaw_gt))

            self.l_ex_px_gt.set_data(t_ex_w, np.full_like(t_ex_w, px_gt))
            self.l_ex_py_gt.set_data(t_ex_w, np.full_like(t_ex_w, py_gt))
            self.l_ex_pz_gt.set_data(t_ex_w, np.full_like(t_ex_w, pz_gt))

            # Attitude axes scaling (3 separate plots)
            for ax in (self.ax_extr_roll, self.ax_extr_pitch, self.ax_extr_yaw):
                ax.set_xlim([tmin, tmax])
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            # Position axis scaling
            self.ax_extr_pos.set_xlim([tmin, tmax])
            self.ax_extr_pos.relim()
            self.ax_extr_pos.autoscale_view(scalex=False, scaley=True)

        self.fig_extr.canvas.draw_idle()
        self.fig_extr.canvas.flush_events()

    # ----------------------- Helpers -----------------------

    def _append_limited(self, dq, value, maxlen=5000):
        dq.append(value)
        if len(dq) > maxlen:
            dq.popleft()

    def _append_tv(self, t_dq, v_dq, t, v, maxlen=5000):
        t_dq.append(float(t))
        v_dq.append(float(v))
        if len(t_dq) > maxlen:
            t_dq.popleft()
            v_dq.popleft()


def main():
    rclpy.init()
    node = EKFDebugPlotter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
