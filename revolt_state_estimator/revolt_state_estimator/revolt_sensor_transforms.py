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
    GNSS velocity + yaw‐rate measurement model.

    State x = [x_e, y_n, yaw, v, w_yaw]
    returns z = [v_x_world, v_y_world, w_yaw]
      v_x_world = v * cos(yaw)
      v_y_world = v * sin(yaw)
      w_yaw     = x[4]
    """
    # unpack
    yaw   = x[2]
    v     = x[3]
    w_yaw = x[4]

    # project forward speed into world axes
    c = np.cos(yaw)
    s = np.sin(yaw)
    v_x = v * c
    v_y = v * s

    return np.array([v_x, v_y, w_yaw])

def h_imu(x: np.ndarray) -> np.ndarray:
    """
    IMU measurement model for EKF (2D ship).

    State x = [x_e, y_n, yaw, v, w_yaw]

    Returns z = [yaw, w_yaw]
    """    
    # yaw from orientation (we assume quaternion→yaw done upstream)
    yaw   = x[2]
    # yaw rate directly
    w_yaw = x[4]

    return np.array([yaw, w_yaw])
