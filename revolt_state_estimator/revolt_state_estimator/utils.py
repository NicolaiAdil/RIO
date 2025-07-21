import numpy as np


def ssa(angle, unit="rad"):
    """
    SSA is the "Smallest-Signed Angle" or the smallest difference between two angles.

    For feedback control systems and state estimators used to control the
    attitude of a vehicle, it is often necessary to wrap angles to the
    mapped to [-pi pi) or [-180 180) to avoid step inputs/discontinuties.

    Based on Fossen's Matlab implementation:
    * https://github.com/cybergalactic/MSS/blob/master/LIBRARY/kinematics/ssa.m

    Parameters
    ----------
    angle : array_like or scalar
        Input angle(s), in radians or degrees.
    unit : {'rad', 'deg'}, optional
        Unit of the input and output angle. Defaults to 'rad'.

    Returns
    -------
    wrapped : ndarray or scalar
        Angle(s) wrapped to the smallest signed representation.
    """
    angle = np.asarray(angle)

    if unit == "rad":
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    elif unit == "deg":
        wrapped = (angle + 180.0) % 360.0 - 180.0
    else:
        raise ValueError(f"Invalid unit '{unit}'. Must be 'rad' or 'deg'.")

    # If input was a scalar, return a scalar
    if np.isscalar(angle):
        return wrapped.item()
    return wrapped


def gravity(lat):
    """
    g = gravity(latitude) computes the acceleration of gravity (m/s^2) as a function
    of latitude lat (rad) using the WGS-84 ellipsoid parameters

    Based on Fossen's Matlab implementation:
    * https://github.com/cybergalactic/MSS/blob/master/INS/functions/gravity.m

    Parameters
    ----------
    latitude : float
        Latitude in degrees.

    Returns
    -------
    g : float
        Gravitational acceleration in m/s² (NED)
    """
    # Convert latitude to radians
    lat_rad = np.radians(lat)

    # Gravitational constant at sea level (m/s²)
    g0 = 9.7803253359

    # Gravity formula based on latitude
    g = (
        g0
        * (1 + 0.001931850400 * np.sin(lat_rad) ** 2)
        / np.sqrt(1 - 0.006694384442 * np.sin(lat_rad) ** 2)
    )

    return g
