import numpy as np

class ReVoltModel:
    def __init__(self, m, r, l, thruster_pos, v_lin_max, v_ang_max):
        """
        m            : mass [kg]
        r, l         : half-width and length of cuboid [m]
        thruster_pos : [x, y] position of thruster in body frame [m]
        v_lin_max    : for any clipping if you want
        v_ang_max    : likewise
        """
        self.m     = m
        self.r     = r
        self.l     = l
        self.thr_x, self.thr_y = thruster_pos
        self.Iz    = (1/12)*m*(l**2 + (2*r)**2)    # inertia about z
        # drag & thrust gains (tweak these)
        self.k_thr = 4.0       # N per unit effort
        self.cd    = 0.001     # linear drag coeff
        self.cyaw  = 0.2       # yaw drag coeff
        self.A     = (2*r)*l   # frontal area [m²]
        self.rho   = 1000.0    # water density

    def _calc_accel(self, v, omega, effort, thr_angle):
        """
        v, omega    : current forward speed [m/s] and yaw rate [rad/s]
        effort       : thruster effort (–100.0…100.0) scaled inside if you like
        thr_angle    : thruster pointing angle in body frame (-pi/2...pi/2) [rad]

        returns (a_forward, alpha_yaw)
        """
        # 1) thrust force in forward dir
        F_thr = self.k_thr * effort
        # 2) drag opposing motion: Fd = –½·ρ·Cd·A·v·|v|
        F_drag = -0.5 * self.rho * self.cd * self.A * v * abs(v)
        # total surge force
        F_total = F_thr + F_drag

        # 3) torque from thruster lever arm (2D cross: x⋅Fy – y⋅Fx)
        #    assume F_thr acts in thr_angle dir: split into Fx, Fy
        Fx = F_thr * np.cos(thr_angle)
        Fy = F_thr * np.sin(thr_angle)
        tau_thr =  self.thr_x * Fy - self.thr_y * Fx
        # yaw-drag
        tau_drag = -self.cyaw * omega
        tau_total = tau_thr + tau_drag

        # 4) accelerations
        a     = F_total / self.m           # forward accel
        alpha = tau_total / self.Iz        # yaw accel

        return a, alpha
    
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous‐time ODE:
          x = [x, y, yaw, v, omega]
          u = [effort, thruster_angle]
        Returns ẋ = [v cos yaw, v sin yaw, omega, a, alpha]
        """
        x_pos, y_pos, yaw, v, omega = x
        effort, angle = u

        # use internal accel calculation
        a, alpha = self._calc_accel(v, omega, effort, angle)

        return np.array([
            v * np.cos(yaw),
            v * np.sin(yaw),
            omega,
            a,
            alpha
        ])
