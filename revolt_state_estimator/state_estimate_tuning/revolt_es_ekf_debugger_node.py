#!/usr/bin/env python3
"""
EKF Debug Plotter (ROS 2 Jazzy)

Compares:
- Yaw: EKF (Odometry), GNSS heading (/heading), COG yaw (from /vel), IMU(AHRS)
- Roll/Pitch: IMU vs EKF
Also shows:
- Position (N–E, NED) and velocities (vN,vE).

Each series has its own timestamps to avoid compressing the visible time window.
"""

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import QuaternionStamped, TwistStamped
import tf_transformations
import pymap3d as pm
import matplotlib.pyplot as plt
import time

def ssa(angle):
    """Wrap to (-pi, pi]."""
    a = (angle + np.pi) % (2.0*np.pi) - np.pi
    if np.isclose(a, -np.pi):
        a = np.pi
    return a

def unwrap_append(prev_unwrapped, new_wrapped):
    if prev_unwrapped is None:
        return float(new_wrapped)
    k = round((prev_unwrapped - new_wrapped) / (2*np.pi))
    return float(new_wrapped + 2*np.pi*k)

class EKFDebugPlotter(Node):
    def __init__(self):
        super().__init__('ekf_debug_plotter')

        # Parameters
        self.declare_parameter('topic_state', '/state_estimate/revolt')
        self.declare_parameter('topic_imu', '/imu/data')
        self.declare_parameter('topic_fix', '/fix')
        self.declare_parameter('topic_head', '/heading')
        self.declare_parameter('topic_vel', '/vel')
        self.declare_parameter('window_secs', 300.0)
        self.declare_parameter('plot_rate_hz', 5.0)
        # self.declare_parameter('min_speed_for_cog', 0.5)
        self.declare_parameter('flip_warn_thresh_deg', 150.0)

        self.topic_state = self.get_parameter('topic_state').value
        self.topic_imu = self.get_parameter('topic_imu').value
        self.topic_fix = self.get_parameter('topic_fix').value
        self.topic_head = self.get_parameter('topic_head').value
        self.topic_vel = self.get_parameter('topic_vel').value
        self.window_secs = float(self.get_parameter('window_secs').value)
        self.plot_dt = 1.0 / float(self.get_parameter('plot_rate_hz').value)
        # self.min_speed_for_cog = float(self.get_parameter('min_speed_for_cog').value)
        self.flip_warn_thresh = np.deg2rad(float(self.get_parameter('flip_warn_thresh_deg').value))

        # Time zero
        self.t0 = None

        # NED reference
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None

        # Per-series time/value histories (each with bounded length)
        self.t_yaw_est, self.yaw_est_hist = deque(), deque()
        self.t_yaw_gps, self.yaw_gnss_hist = deque(), deque()
        # self.t_yaw_cog, self.yaw_cog_hist = deque(), deque()
        self.t_yaw_imu, self.yaw_imu_hist = deque(), deque()

        self.t_roll_imu, self.roll_imu_hist = deque(), deque()
        self.t_pitch_imu, self.pitch_imu_hist = deque(), deque()
        self.t_roll_est, self.roll_est_hist = deque(), deque()
        self.t_pitch_est, self.pitch_est_hist = deque(), deque()

        self.t_vN_est, self.vN_est = deque(), deque()
        self.t_vE_est, self.vE_est = deque(), deque()
        self.t_vN_gps, self.vN_gps = deque(), deque()
        self.t_vE_gps, self.vE_gps = deque(), deque()

        # NE tracks (no time needed for the map)
        self.ne_est = deque()  # (N,E)
        self.ne_gps = deque()

        # Flip markers (times)
        self.flip_marks_t = deque()

        # Unwrap trackers
        self._last_yaw_est = None
        self._last_yaw_gnss = None
        # self._last_yaw_cog = None
        self._last_yaw_imu = None

        # Subscribers
        self.create_subscription(Odometry, self.topic_state, self.cb_state, 10)
        self.create_subscription(Imu, self.topic_imu, self.cb_imu, 20)
        self.create_subscription(NavSatFix, self.topic_fix, self.cb_fix, 3)
        self.create_subscription(QuaternionStamped, self.topic_head, self.cb_heading, 3)
        self.create_subscription(TwistStamped, self.topic_vel, self.cb_vel, 10)

        # Figure + timer
        self._make_figure()
        self._plot_timer = self.create_timer(self.plot_dt, self._on_plot_timer)

        self.get_logger().info(
            f"EKF Debug Plotter started.\n"
            f" Subscribed to:\n"
            f"  state:   {self.topic_state}\n"
            f"  imu:     {self.topic_imu}\n"
            f"  fix:     {self.topic_fix}\n"
            f"  heading: {self.topic_head}\n"
            f"  vel:     {self.topic_vel}\n"
            f" Window = {self.window_secs}s, plot_rate = {1.0/self.plot_dt:.1f} Hz"
        )

    # ----------------------- Callbacks -----------------------

    def _now_s(self):
        if self.t0 is None:
            self.t0 = time.time()
        return time.time() - self.t0

    def cb_state(self, msg: Odometry):
        t = self._now_s()

        # EKF orientation (RPY)
        q = msg.pose.pose.orientation
        r, p, y = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        r, p, y = ssa(r), ssa(p), ssa(y)

        # Save yaw (unwrapped), roll, pitch
        un_yaw = unwrap_append(self._last_yaw_est, y); self._last_yaw_est = un_yaw
        self._append_tv(self.t_yaw_est, self.yaw_est_hist, t, un_yaw)
        self._append_tv(self.t_roll_est, self.roll_est_hist, t, r)
        self._append_tv(self.t_pitch_est, self.pitch_est_hist, t, p)

        # NE (NED) for map
        N = float(msg.pose.pose.position.x)
        E = float(msg.pose.pose.position.y)
        self._append_limited(self.ne_est, (N, E))
        # Vel (NED)
        vN = float(msg.twist.twist.linear.x)
        vE = float(msg.twist.twist.linear.y)
        self._append_tv(self.t_vN_est, self.vN_est, t, vN)
        self._append_tv(self.t_vE_est, self.vE_est, t, vE)

    def cb_imu(self, msg: Imu):
        t = self._now_s()
        q = msg.orientation
        r, p, y = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        r, p, y = ssa(r), ssa(p), ssa(y)

        self._append_tv(self.t_roll_imu, self.roll_imu_hist, t, r)
        self._append_tv(self.t_pitch_imu, self.pitch_imu_hist, t, p)

        un = unwrap_append(self._last_yaw_imu, y); self._last_yaw_imu = un
        self._append_tv(self.t_yaw_imu, self.yaw_imu_hist, t, un)

    def cb_fix(self, msg: NavSatFix):
        if np.isnan(msg.latitude) or np.isnan(msg.longitude):
            return
        if self.ref_lat is None:
            self.ref_lat, self.ref_lon, self.ref_alt = msg.latitude, msg.longitude, msg.altitude
            self.get_logger().info(f"Set NED reference lat={self.ref_lat:.8f}, lon={self.ref_lon:.8f}, alt={self.ref_alt:.2f}")
        n, e, d = pm.geodetic2ned(msg.latitude, msg.longitude, msg.altitude,
                                  self.ref_lat, self.ref_lon, self.ref_alt)
        self._append_limited(self.ne_gps, (float(n), float(e)))

    def cb_heading(self, msg: QuaternionStamped):
        t = self._now_s()
        q = msg.quaternion
        r, p, y = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        y = ssa(y)

        un = unwrap_append(self._last_yaw_gnss, y); self._last_yaw_gnss = un
        self._append_tv(self.t_yaw_gps, self.yaw_gnss_hist, t, un)

        # Flip detector (vs latest EKF yaw if present)
        if len(self.yaw_est_hist) > 0:
            diff = ssa(self.yaw_gnss_hist[-1] - self.yaw_est_hist[-1])
            if abs(abs(diff) - np.pi) < np.deg2rad(15) or abs(diff) > self.flip_warn_thresh:
                self._append_limited(self.flip_marks_t, t)
                self.get_logger().warn(f"Possible 180° flip: |Δyaw|={np.degrees(abs(diff)):.1f}° at t={t:.1f}s")

    def cb_vel(self, msg: TwistStamped):
        t = self._now_s()
        vx, vy, vz = msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
        v_enu = np.array([vx, vy, vz], dtype=float).reshape(3, 1)
        R_enu_to_ned = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0,-1]], dtype=float)
        v_ned = (R_enu_to_ned @ v_enu).ravel()
        vN, vE = float(v_ned[0]), float(v_ned[1])

        self._append_tv(self.t_vN_gps, self.vN_gps, t, vN)
        self._append_tv(self.t_vE_gps, self.vE_gps, t, vE)

        speed = float(np.hypot(vN, vE))
        # if speed > self.min_speed_for_cog:
        #     yaw_cog_wrapped = ssa(np.arctan2(vE, vN))
        #     un = unwrap_append(self._last_yaw_cog, yaw_cog_wrapped); self._last_yaw_cog = un
        #     self._append_tv(self.t_yaw_cog, self.yaw_cog_hist, t, un)
        # elif len(self.yaw_cog_hist) > 0:
        #     # hold last value (optionally append repeated point with current time)
        #     self._append_tv(self.t_yaw_cog, self.yaw_cog_hist, t, self.yaw_cog_hist[-1])

    # ----------------------- Plotting -----------------------

    def _make_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12))
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.0, 1.0], hspace=0.35)

        # Yaw
        self.ax_head = self.fig.add_subplot(gs[0, 0])
        self.ax_head.set_title("Yaw vs Time")
        self.ax_head.set_ylabel("Yaw [deg]")
        self.ax_head.set_xlabel("Time [s]")
        self.l_est_head, = self.ax_head.plot([], [], label="EKF yaw")
        self.l_gps_head, = self.ax_head.plot([], [], label="GNSS yaw")
        # self.l_cog_head, = self.ax_head.plot([], [], label="COG yaw (vel>thr)")
        self.l_imu_head, = self.ax_head.plot([], [], label="IMU(AHRS) yaw")
        self.flip_scatter = self.ax_head.scatter([], [], marker='x', label="Flip?")
        self.ax_head.legend(loc='best'); self.ax_head.grid(True)

        # Roll/Pitch
        self.ax_rp = self.fig.add_subplot(gs[1, 0])
        self.ax_rp.set_title("Roll & Pitch vs Time")
        self.ax_rp.set_ylabel("Angle [deg]"); self.ax_rp.set_xlabel("Time [s]")
        self.l_roll_imu, = self.ax_rp.plot([], [], label="IMU roll")
        self.l_pitch_imu, = self.ax_rp.plot([], [], label="IMU pitch")
        self.l_roll_est, = self.ax_rp.plot([], [], label="EKF roll")
        self.l_pitch_est, = self.ax_rp.plot([], [], label="EKF pitch")
        self.ax_rp.legend(loc='best'); self.ax_rp.grid(True)

        # Position NE
        self.ax_ne = self.fig.add_subplot(gs[2, 0])
        self.ax_ne.set_title("Position (N–E in NED)")
        self.ax_ne.set_xlabel("E [m]"); self.ax_ne.set_ylabel("N [m]")
        self.l_est_ne, = self.ax_ne.plot([], [], label="EKF track")
        self.l_gps_ne, = self.ax_ne.plot([], [], linestyle='None', marker='.', label="GNSS fix")
        self.ax_ne.axis('equal'); self.ax_ne.grid(True); self.ax_ne.legend(loc='best')

        # Velocity
        self.ax_vel = self.fig.add_subplot(gs[3, 0])
        self.ax_vel.set_title("Velocity Components vs Time (NED)")
        self.ax_vel.set_ylabel("v [m/s]"); self.ax_vel.set_xlabel("Time [s]")
        self.l_vN_est, = self.ax_vel.plot([], [], label="vN EKF")
        self.l_vE_est, = self.ax_vel.plot([], [], label="vE EKF")
        self.l_vN_gps, = self.ax_vel.plot([], [], label="vN GNSS")
        self.l_vE_gps, = self.ax_vel.plot([], [], label="vE GNSS")
        self.ax_vel.grid(True); self.ax_vel.legend(loc='best')

        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        try: plt.show(block=False)
        except Exception: pass

    def _on_plot_timer(self):
        try:
            self._refresh_plot()
        except Exception as e:
            self.get_logger().warn(f"Plot refresh error: {e}")

    @staticmethod
    def _finite_xy(tx, yy):
        if len(tx) == 0 or len(yy) == 0:
            return np.array([]), np.array([])
        tx = np.asarray(tx); yy = np.asarray(yy)
        m = np.isfinite(tx) & np.isfinite(yy)
        return tx[m], yy[m]

    def _refresh_plot(self):
        # Current visible window based on wall time
        now = self._now_s()
        tmin = max(0.0, now - self.window_secs)
        tmax = now

        # ---- Yaw (deg) ----
        te, ye = self._finite_xy(self.t_yaw_est, np.degrees(self.yaw_est_hist))
        tg, yg = self._finite_xy(self.t_yaw_gps, np.degrees(self.yaw_gnss_hist))
        # tc, yc = self._finite_xy(self.t_yaw_cog, np.degrees(self.yaw_cog_hist))
        ti, yi = self._finite_xy(self.t_yaw_imu, np.degrees(self.yaw_imu_hist))

        self.l_est_head.set_data(te, ye)
        self.l_gps_head.set_data(tg, yg)
        # self.l_cog_head.set_data(tc, yc)
        self.l_imu_head.set_data(ti, yi)

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
        tr_i, rr_i = self._finite_xy(self.t_roll_imu, np.degrees(self.roll_imu_hist))
        tp_i, pp_i = self._finite_xy(self.t_pitch_imu, np.degrees(self.pitch_imu_hist))
        tr_e, rr_e = self._finite_xy(self.t_roll_est, np.degrees(self.roll_est_hist))
        tp_e, pp_e = self._finite_xy(self.t_pitch_est, np.degrees(self.pitch_est_hist))

        self.l_roll_imu.set_data(tr_i, rr_i)
        self.l_pitch_imu.set_data(tp_i, pp_i)
        self.l_roll_est.set_data(tr_e, rr_e)
        self.l_pitch_est.set_data(tp_e, pp_e)

        self.ax_rp.set_xlim([tmin, tmax])
        self.ax_rp.relim(); self.ax_rp.autoscale_view(scalex=False, scaley=True)

        # ---- Position NE ----
        ne_est_arr = np.array(self.ne_est) if len(self.ne_est) else np.empty((0,2))
        ne_gps_arr = np.array(self.ne_gps) if len(self.ne_gps) else np.empty((0,2))
        if ne_est_arr.shape[0] > 0:
            self.l_est_ne.set_data(ne_est_arr[:,1], ne_est_arr[:,0])  # x=E, y=N
        else:
            self.l_est_ne.set_data([], [])
        if ne_gps_arr.shape[0] > 0:
            self.l_gps_ne.set_data(ne_gps_arr[:,1], ne_gps_arr[:,0])
        else:
            self.l_gps_ne.set_data([], [])

        if ne_est_arr.shape[0] + ne_gps_arr.shape[0] > 1:
            allE = []; allN = []
            if ne_est_arr.shape[0] > 0:
                allE += ne_est_arr[:,1].tolist(); allN += ne_est_arr[:,0].tolist()
            if ne_gps_arr.shape[0] > 0:
                allE += ne_gps_arr[:,1].tolist(); allN += ne_gps_arr[:,0].tolist()
            padE = (max(allE) - min(allE))*0.1 + 1.0
            padN = (max(allN) - min(allN))*0.1 + 1.0
            self.ax_ne.set_xlim([min(allE)-padE, max(allE)+padE])
            self.ax_ne.set_ylim([min(allN)-padN, max(allN)+padN])

        # ---- Velocity ----
        t_vNe, vNe = self._finite_xy(self.t_vN_est, self.vN_est)
        t_vEe, vEe = self._finite_xy(self.t_vE_est, self.vE_est)
        t_vNg, vNg = self._finite_xy(self.t_vN_gps, self.vN_gps)
        t_vEg, vEg = self._finite_xy(self.t_vE_gps, self.vE_gps)

        self.l_vN_est.set_data(t_vNe, vNe)
        self.l_vE_est.set_data(t_vEe, vEe)
        self.l_vN_gps.set_data(t_vNg, vNg)
        self.l_vE_gps.set_data(t_vEg, vEg)

        self.ax_vel.set_xlim([tmin, tmax])
        self.ax_vel.relim(); self.ax_vel.autoscale_view(scalex=False, scaley=True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ----------------------- Helpers -----------------------

    def _append_limited(self, dq, value, maxlen=5000):
        dq.append(value)
        if len(dq) > maxlen:
            dq.popleft()

    def _append_tv(self, t_dq, v_dq, t, v, maxlen=5000):
        t_dq.append(float(t)); v_dq.append(float(v))
        if len(t_dq) > maxlen:
            t_dq.popleft(); v_dq.popleft()

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
