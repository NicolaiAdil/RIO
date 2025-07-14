import numpy as np

def unwrap(prev_yaw, new_yaw):
    """
    prev_yaw: your last (unwrapped) heading
    new_yaw : the freshly measured yaw (wrapped to [−π,π])
    returns:  new continuous heading
    """
    delta = new_yaw - (prev_yaw % (2*np.pi))
    if delta > np.pi:
        delta -= 2*np.pi
    elif delta < -np.pi:
        delta += 2*np.pi
    return prev_yaw + delta
