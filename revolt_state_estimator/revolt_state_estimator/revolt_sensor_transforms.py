# measurement_models.py

import numpy as np

def h_fix(x: np.ndarray) -> np.ndarray:
    """
    GNSS /fix measurement model.

    State x = [x_e, y_n, yaw, v, w_yaw]
    returns z = [x_e, y_n]
    """
    # explicitly pick out east (x[0]) and north (x[1])
    return np.array([x[0], x[1]])

def h_head(x: np.ndarray) -> np.ndarray:
    """
    GNSS heading measurement model.

    State x = [x_e, y_n, yaw, v, w_yaw]
    returns z = [yaw]
    """
    # yaw is x[2]
    return np.array([x[2]])

def h_vel(x: np.ndarray) -> np.ndarray:
    """
    GNSS velocity measurement model in body frame.
    """
    v = x[3]

    return np.array([v])

def h_imu(x: np.ndarray) -> np.ndarray:
    """
    IMU measurement model for EKF (2D ship).

    State x = [x_e, y_n, yaw, v, w_yaw]

    Returns z = [yaw, v, w_yaw]
    """    
    # yaw from orientation (we assume quaternion→yaw done upstream)
    yaw   = x[2]
    # Linear velocity
    v     = x[3]
    # yaw rate directly
    w_yaw = x[4]

    return np.array([yaw, v, w_yaw])