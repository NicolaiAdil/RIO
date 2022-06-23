import numpy as np
import tf

r_e = 6378137
# Equatorial radius
r_p = 6356752.3142
# Polar radius
ecc = np.sqrt(1 - (r_p / r_e) ** 2)  # Eccentricity


def position_llh_to_local(llh, llh0):
    C_e_l0 = dcm_llh_to_local(llh0[0], llh0[1])
    return np.dot(C_e_l0, position_llh_to_ecef(llh) - position_llh_to_ecef(llh0))


def position_local_to_llh(local_pos, llh0):
    C_l0_e = dcm_llh_to_local(llh0[0], llh0[1]).T
    r_eb_e = C_l0_e.dot(local_pos) + position_llh_to_ecef(llh0)
    return position_ecef_to_llh(r_eb_e)


def position_ecef_to_llh(r_ecef):
    lon = np.arctan2(r_ecef[1], r_ecef[0])
    z = r_ecef[2]
    p = np.linalg.norm(r_ecef[0:2])
    lat = 100
    lat_est = np.arctan(z / (p * (1 - ecc ** 2)))
    while not np.isclose(lat, lat_est):
        lat_est = lat
        N = normal_radius(lat_est)
        h = p / np.cos(lat_est) - N
        lat = np.arctan(z / (p * (1 - ecc ** 2 * N / (N + h))))
    N = normal_radius(lat_est)
    h = p / np.cos(lat_est) - N
    return np.array([lat, lon, h])


def euler_angles_to_quaternion(euler_angles):
    """
    Input matrix [roll, pitch, yaw] on each row and assumes an intrinsic rotation with yaw, pitch, roll order. This means a rotation about the z-axis, then a rotation about the new y-axis and finally a rotation about the new x-axis.
    """
    if euler_angles.ndim == 1:
        return tf.transformations.quaternion_from_euler(
            euler_angles[2], euler_angles[1], euler_angles[0], "rzyx"
        )
    else:
        return np.apply_along_axis(
            lambda eul: tf.transformations.quaternion_from_euler(
                eul[2], eul[1], eul[0], "rzyx"
            ),
            1,
            euler_angles,
        )


def quaternion_to_euler_angles(quaternion):
    """
    Returns [roll, pitch, yaw]
    """
    if quaternion.ndim == 1:
        return tf.transformations.euler_from_quaternion(quaternion, "rzyx")[::-1]
    else:
        return np.apply_along_axis(
            tf.transformations.euler_from_quaternion, 1, quaternion, "rzyx"
        )[
            :, ::-1
        ]  # The matrix from tf.euler_from_quaternion is returned as [zyx]


def dcm_llh_to_local(lat, lon):
    return np.array(
        [
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [-np.sin(lon), np.cos(lon), 0],
            [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)],
        ]
    )


def euler_angles_to_matrix(euler_angles):
    """
    Input matrix [roll, pitch, yaw] on each row and assumes an intrinsic rotation with yaw, pitch, roll order. This means a rotation about the z-axis, then a rotation about the new y-axis and finally a rotation about the new x-axis.
    """
    phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
    R = tf.transformations.euler_matrix(psi, theta, phi, "rzyx")
    return R[0:3, 0:3]


def heading_to_matrix_2D(heading):
    return euler_angles_to_matrix(np.array([0, 0, heading]))[0:2, 0:2]


def normal_radius(lat):
    return r_e ** 2 / np.sqrt((r_e * np.cos(lat)) ** 2 + (r_p * np.sin(lat)) ** 2)


def position_llh_to_ecef(llh):
    N = normal_radius(llh[0])
    x = (N + llh[2]) * np.cos(llh[0]) * np.cos(llh[1])
    y = (N + llh[2]) * np.cos(llh[0]) * np.sin(llh[1])
    z = (r_p ** 2 / r_e ** 2 * N + llh[2]) * np.sin(llh[0])
    return np.array([x, y, z])


def to_homo(mat):
    homo = np.zeros((4, 4))
    homo[3, 3] = 1
    homo[0:3, 0:3] = mat
    return homo


def from_homo(homo):
    mat = homo[0:3, 0:3]
    return mat


def rot_to_quat(rotmat):
    homo = to_homo(rotmat)
    return tf.transformations.quaternion_from_matrix(homo)


def quaternion(qx, qy, qz, qw):
    return np.array([qx, qy, qz, qw])


def quat_transform(vector, quat):
    q_vector = quat_mul(quat, quat_mul(np.hstack((vector, 0)), quat_conj(quat)))
    return q_vector[0:3]


def quat_inverse_transform(vector, quat):
    quat_inv = quat_conj(quat)
    vector_inv = -quat_transform(vector, quat_inv)
    return (vector_inv, quat_inv)


quat_to_rot = lambda quat: from_homo(tf.transformations.quaternion_matrix(quat))
quat_mul = tf.transformations.quaternion_multiply
quat_conj = tf.transformations.quaternion_conjugate


def skew_symmetric(qv):
    return np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])


circular_error_probability_coeffs = np.array(
    [
        #   50%     90%     95%      99%
        [0.6754, 1.6494, 1.9626, 2.5686],
        [-0.0208, -0.0588, 0.0100, 0.1479],
        [1.1009, 0.3996, 0.0700, -0.4285],
        [-0.5821, 0.1636, 0.4092, 0.7371],
    ]
)


def circular_error_probability_50(P):
    return circular_error_probability(P, circular_error_probability_coeffs[:, 0])


def circular_error_probability_90(P):
    return circular_error_probability(P, circular_error_probability_coeffs[:, 1])


def circular_error_probability_95(P):
    return circular_error_probability(P, circular_error_probability_coeffs[:, 2])


def circular_error_probability_99(P):
    return circular_error_probability(P, circular_error_probability_coeffs[:, 3])


def circular_error_probability(P, coeffs):
    """Calculate the circular error probability radius of a bivariate Gaussian distribution.
    Based on M. Ignagni 2010, Journal of Guidance, Control and Dynamics.
    """
    eig, _ = np.linalg.eig(P)
    eig1, eig2 = np.sort(eig)[::-1]  # sorted in reverse order, eig1 > eig2
    alpha = eig2 / eig1
    poly = np.array([1, alpha, alpha ** 2, alpha ** 3])
    R = eig1 * np.dot(coeffs, poly)
    return R


def inf2pipi(angle):
    """
    Map an angle to the domain [-pi, pi)
    """
    return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))


def R2DOF(angle):
    """
    Creates a 2DOF rotation matrix in SO(2)
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


def cart2pol(x, y):
    """
    Convert from cartesian to polar coordinates
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return (r, theta)


def pol2cart(r, theta):
    """
    Convert from polar to cartesian coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return (x, y)


def homo_2D_transform_matrix(ang, p):
    H = np.zeros((3, 3))
    H[0:2, 0:2] = R2DOF(ang)
    H[0:2, 2] = p.flatten()
    H[2, 2] = 1
    return H
