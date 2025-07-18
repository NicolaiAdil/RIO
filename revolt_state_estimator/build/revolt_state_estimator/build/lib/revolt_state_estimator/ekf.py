import numpy as np
import scipy

# =============================================================================
# ekf.py
# 
# Extended Kalman Filter implementation in Python, with numerical Jacobians,
# system discretization (ZOH), and predict/correct stages. 
# =============================================================================


def numerical_jacobian(f, x, epsilon=1e-4):
    """
    Numerically compute Jacobian J of f w.r.t. x at point x.
    f: R^n -> R^m, x: (n,), returns J: (m,n)
    """
    x = np.asarray(x)
    n = x.size
    f0 = np.asarray(f(x))
    m = f0.size
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = epsilon
        f_plus  = np.asarray(f(x + dx))
        f_minus = np.asarray(f(x - dx))
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
    return J


def discretize_ab_zoh(A, B, dt):
    """
    Discretize continuous-time system (A,B) via zero-order hold.
    Returns (Ad, Bd).
    """
    # build augmented matrix
    n, m = B.shape
    M = np.zeros((n+m, n+m))
    M[:n, :n] = A
    M[:n, n:] = B
    # matrix exponential
    Mexp = scipy.linalg.expm(M * dt)
    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
    return Ad, Bd

def euler_forward(f, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Perform one Euler forward step of the ODE x_dot = f(x, u).

    Parameters
    ----------
    f  : callable
        Function f(x, u) returning the time-derivative x_dot as a numpy array.
    x  : numpy.ndarray
        Current state vector.
    u  : numpy.ndarray
        Control input vector.
    dt : float
        Time increment for integration (seconds).

    Returns
    -------
    numpy.ndarray
        Next state vector x + f(x, u) * dt.
    """
    # compute time-derivative at the current point
    x_dot = f(x, u)
    # forward Euler step
    return x + x_dot * dt

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with Euler predict + ZOH discretization + numerical Jacobians.
    """
    def __init__(self, f, h, dim_x, dim_z, dt, Q, R):
        """
        f : function f(x,u)->x_dot
        h : function h(x)->z_pred
        dim_x, dim_z : state and measurement dims
        dt : timestep
        Q : process noise covariance (dim_x, dim_x)
        R : measurement noise covariance (dim_z, dim_z)
        """
        self.f     = f
        self.h     = h
        self.dt    = dt
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.Q     = Q
        self.R     = R
        # state estimates
        self.x_post = np.zeros(dim_x)
        self.P_post = np.eye(dim_x)
        # placeholders
        self.x_prior = np.zeros(dim_x)
        self.P_prior = np.eye(dim_x)

    def predict(self, u=0):
        """
        Predict step: discrete Euler + covariance update.
        u: control vector
        """
        # a) state propagation (Euler forward)
        self.x_prior = euler_forward(self.f, self.x_post, u, self.dt)

        # b) linearize
        A = numerical_jacobian(lambda x: self.f(x, u), self.x_post)
        B = numerical_jacobian(lambda uu: self.f(self.x_post, uu), u)
        # c) discretize
        Ad, Bd = discretize_ab_zoh(A, B, self.dt)

        # d) covariance predict
        self.P_prior = Ad @ self.P_post @ Ad.T + self.Q

        return self.x_prior, self.P_prior

    def update(self, z=0):
        """
        Update step: measurement z.
        """
        # linearize measurement
        H = numerical_jacobian(self.h, self.x_prior)
        # innovation
        z_pred = self.h(self.x_prior)
        y = z - z_pred
        # innovation covariance
        S = H @ self.P_prior @ H.T + self.R
        K = self.P_prior @ H.T @ np.linalg.inv(S)
        # state update
        self.x_post = self.x_prior + K @ y
        # covariance update
        I = np.eye(self.dim_x)
        self.P_post = (I - K @ H) @ self.P_prior

        return self.x_post, self.P_post
