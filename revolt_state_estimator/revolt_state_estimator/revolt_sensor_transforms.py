import numpy as np

def Rzyx(phi, theta, psi):
    """
    Computes the Euler angle rotation matrix R in SO(3) using the zyx convention.

    Based on Fossen's Matlab implementation:
    * https://github.com/cybergalactic/MSS/blob/master/LIBRARY/kinematics/Rzyx.m

    Parameters
    ----------
    phi : float or array_like
        Rotation about the x-axis (roll).
    theta : float or array_like
        Rotation about the y-axis (pitch).
    psi : float or array_like
        Rotation about the z-axis (yaw).

    Returns
    -------
    R : ndarray, shape (3,3)
        Rotation matrix in SO(3) corresponding to the sequence
        R = R_z(psi) @ R_y(theta) @ R_x(phi).
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array(
        [
            [
                cpsi * cth,
                -spsi * cphi + cpsi * sth * sphi,
                spsi * sphi + cpsi * cphi * sth,
            ],
            [
                spsi * cth,
                cpsi * cphi + sphi * sth * spsi,
                -cpsi * sphi + sth * spsi * cphi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )
    return R


def Tzyx(phi, theta):
    """
    Computes the Euler angle transformation matrix T for attitude using the zyx convention.

    Based on Fossen's Matlab implementation:
    * https://github.com/cybergalactic/MSS/blob/master/LIBRARY/kinematics/Tzyx.m

    Parameters
    ----------
    phi : float or array_like
        Roll angle (rotation about the x-axis), in radians.
    theta : float or array_like
        Pitch angle (rotation about the y-axis), in radians.

    Returns
    -------
    T : ndarray, shape (3,3)
        The kinematic transformation matrix.

    Raises
    ------
    ValueError
        If cos(theta) is (close to) zero, since the matrix is singular at theta = ±90°.
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)

    if np.isclose(cth, 0.0):
        raise ValueError("Tzyx is singular for theta = ±90° (cos(theta) = 0)")

    T = np.array(
        [
            [1.0, sphi * sth / cth, cphi * sth / cth],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth],
        ]
    )
    return T