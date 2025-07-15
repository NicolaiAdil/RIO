import numpy as np
import math

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

class HeadingAligner:
    """
    One-shot IMU↔GNSS yaw offset calibrator.

    - Call add_sample(imu, gnss) until it’s calibrated.
    - Then use align_imu(imu) to shift future IMU readings
      into the GNSS frame.
    """
    def __init__(self):
        self._offset = None

    def add_sample(self, imu_yaw: float, gnss_yaw: float) -> None:
        """
        On first call, compute offset = gnss_yaw − imu_yaw (smallest circle).
        Subsequent calls are no-ops.
        """
        if self._offset is None:
            # circular difference in [−π,π]
            diff = (gnss_yaw - imu_yaw + np.pi) % (2*np.pi) - np.pi
            self._offset = diff

    def is_calibrated(self) -> bool:
        return self._offset is not None

    def align_imu(self, imu_yaw: float) -> float:
        if self._offset is None:
            raise RuntimeError("HeadingAligner not yet calibrated")
        return imu_yaw + self._offset
