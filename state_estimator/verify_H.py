#!/usr/bin/env python3
"""
check_radar_jacobian_21.py

Numerically verify the analytical radar Jacobian H against a finite-difference
approximation with respect to the 21-dim error state

    δx = [δp(0:3),
          δv(3:6),
          δb_a(6:9),
          δθ_body(9:12),
          δb_g(12:15),
          δp_IR(15:18),
          δθ_IR(18:21)]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skew(v):
    v = np.asarray(v).reshape(3,)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=float)


def _exp_so3(phi):
    phi = np.asarray(phi).reshape(3,)
    a = np.linalg.norm(phi)
    if a < 1e-12:
        return np.eye(3) + _skew(phi)
    A = np.sin(a) / a
    B = (1.0 - np.cos(a)) / (a * a)
    K = _skew(phi)
    return np.eye(3) + A * K + B * (K @ K)


def Rzyx(roll, pitch, yaw):
    """
    R = Rz(yaw) * Ry(pitch) * Rx(roll), body->world (ZYX convention).
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    Ry = np.array([
        [ cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr],
    ])

    return Rz @ Ry @ Rx

# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------

def radar_h(mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR):
    """
    Scalar measurement model:

        v_R = R_RI [ R_IW v_WI + (w_imu - b_ars_ins) × p_IR ]
        h   = - mu_r^T v_R

    All vectors are 3x1 columns, rotations 3x3.
    """
    mu_r = mu_r.reshape(3, 1)
    v_WI = v_WI.reshape(3, 1)
    w_imu = w_imu.reshape(3, 1)
    b_ars_ins = b_ars_ins.reshape(3, 1)
    p_IR = p_IR.reshape(3, 1)

    R_IW = R_WI.T
    R_RI = R_IR.T

    v_I = R_IW @ v_WI
    w = w_imu - b_ars_ins
    spin = _skew(w) @ p_IR
    v_R = R_RI @ (v_I + spin)

    return float(-(mu_r.T @ v_R))


def calculate_radar_H(mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR):
    """
    Analytical Jacobian H (1x21) of h with respect to the 21-dim error state:

        δx = [δp, δv, δb_a, δθ_body, δb_g, δp_IR, δθ_IR]

    Mapping of δx into perturbations:
        v_WI'   = v_WI + δv
        R_WI'   = R_WI * exp([δθ_body]_x)
        b_g'    = b_ars_ins + δb_g
        p_IR'   = p_IR + δp_IR
        R_IR'   = R_IR * exp([δθ_IR]_x)
    """
    mu = mu_r.reshape(1, 3)
    R_IW = R_WI.T
    R_RI = R_IR.T

    v_WI = v_WI.reshape(3, 1)
    w_imu = w_imu.reshape(3, 1)
    b_ars_ins = b_ars_ins.reshape(3, 1)
    p_IR = p_IR.reshape(3, 1)

    v_I = R_IW @ v_WI                    # 3x1
    w = (w_imu - b_ars_ins).reshape(3,)  # 3,
    p = p_IR.reshape(3,)                 # 3,

    # v_R0 used for θ_IR derivative
    spin = _skew(w) @ p_IR               # 3x1
    v_I_plus = v_I + spin                # 3x1
    v_R0 = (R_RI @ v_I_plus).reshape(3,) # 3,

    H = np.zeros((1, 21), dtype=float)

    # d h / d (δv) = - μ^T R_RI R_IW
    H[0, 3:6] = -(mu @ (R_RI @ R_IW))

    # d h / d (δθ_body) = - μ^T R_RI [v_I]_x
    H[0, 9:12] = -(mu @ (R_RI @ _skew(v_I.reshape(3,))))

    # d h / d (δb_g) = - μ^T R_RI [p_IR]_x
    H[0, 12:15] = -(mu @ (R_RI @ _skew(p)))

    # d h / d (δp_IR) = - μ^T R_RI [w]_x
    H[0, 15:18] = -(mu @ (R_RI @ _skew(w)))

    # d h / d (δθ_IR) with R_IR' = R_IR exp([δθ_IR]_x) ⇒ R_RI' = (I - [δθ_IR]_x) R_RI
    # ⇒ δv_R = - [δθ_IR]_x v_R0, so δh = - μ^T δv_R = - μ^T ( - [δθ_IR]_x v_R0 )
    # ⇒ ∂h/∂δθ_IR = - μ^T [v_R0]_x
    H[0, 18:21] = -(mu @ _skew(v_R0))

    # δp and δb_a do not enter directly → zeros at 0:3 and 6:9, as expected.

    return H

# ---------------------------------------------------------------------------
# Numeric Jacobian with respect to δx (21-dim)
# ---------------------------------------------------------------------------

def numeric_H(mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR, eps=1e-6):
    """
    Finite-difference Jacobian of h w.r.t the 21-dim error state δx.

    We do forward differences:

        H[0,k] ≈ ( h(δx_k = eps) - h(δx = 0) ) / eps

    where each δx_k is applied via the **same perturbation mapping**
    assumed in calculate_radar_H.
    """
    # Nominal arguments
    v_WI = v_WI.reshape(3, 1)
    w_imu = w_imu.reshape(3, 1)
    b_ars_ins = b_ars_ins.reshape(3, 1)
    p_IR = p_IR.reshape(3, 1)

    h0 = radar_h(mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR)

    Hn = np.zeros((1, 21), dtype=float)

    for k in range(21):
        # Copy nominal
        vW_p = v_WI.copy()
        RWI_p = R_WI.copy()
        w_imu_p = w_imu.copy()
        b_ars_p = b_ars_ins.copy()
        pIR_p = p_IR.copy()
        RIR_p = R_IR.copy()

        d = np.zeros(21)
        d[k] = eps

        if 3 <= k <= 5:
            # δv
            vW_p += d[3:6].reshape(3, 1)

        elif 9 <= k <= 11:
            # δθ_body, right-multiplicative on R_WI
            dth = d[9:12]
            RWI_p = R_WI @ _exp_so3(dth)

        elif 12 <= k <= 14:
            # δb_g
            b_ars_p += d[12:15].reshape(3, 1)

        elif 15 <= k <= 17:
            # δp_IR
            pIR_p += d[15:18].reshape(3, 1)

        elif 18 <= k <= 20:
            # δθ_IR, right-multiplicative on R_IR
            dthIR = d[18:21]
            RIR_p = R_IR @ _exp_so3(dthIR)

        # δp and δb_a (0:3 and 6:9) do not affect the measurement directly.

        h_p = radar_h(mu_r, RWI_p, vW_p, w_imu_p, b_ars_p, pIR_p, RIR_p)
        Hn[0, k] = (h_p - h0) / eps

    return Hn

def numeric_H_central(mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR, eps=1e-6):
    """
    Central-difference Jacobian of h w.r.t the 21-dim error state δx.

        H[0,k] ≈ ( h(+eps) - h(-eps) ) / (2 eps)

    using the same perturbation mapping as calculate_radar_H.
    """
    v_WI = v_WI.reshape(3, 1)
    w_imu = w_imu.reshape(3, 1)
    b_ars_ins = b_ars_ins.reshape(3, 1)
    p_IR = p_IR.reshape(3, 1)

    Hn = np.zeros((1, 21), dtype=float)

    for k in range(21):
        # Create +/- perturbations
        def apply_perturb(sign):
            vW_p = v_WI.copy()
            RWI_p = R_WI.copy()
            b_ars_p = b_ars_ins.copy()
            pIR_p = p_IR.copy()
            RIR_p = R_IR.copy()

            d = np.zeros(21)
            d[k] = sign * eps

            if 3 <= k <= 5:
                vW_p += d[3:6].reshape(3, 1)

            elif 9 <= k <= 11:
                dth = d[9:12]
                RWI_p = R_WI @ _exp_so3(dth)

            elif 12 <= k <= 14:
                b_ars_p += d[12:15].reshape(3, 1)

            elif 15 <= k <= 17:
                pIR_p += d[15:18].reshape(3, 1)

            elif 18 <= k <= 20:
                dthIR = d[18:21]
                RIR_p = R_IR @ _exp_so3(dthIR)

            return RWI_p, vW_p, b_ars_p, pIR_p, RIR_p

        RWI_p, vW_p, b_ars_p, pIR_p, RIR_p = apply_perturb(+1.0)
        h_p = radar_h(mu_r, RWI_p, vW_p, w_imu, b_ars_p, pIR_p, RIR_p)

        RWI_m, vW_m, b_ars_m, pIR_m, RIR_m = apply_perturb(-1.0)
        h_m = radar_h(mu_r, RWI_m, vW_m, w_imu, b_ars_m, pIR_m, RIR_m)

        Hn[0, k] = (h_p - h_m) / (2.0 * eps)

    return Hn

def print_block_norms(diff, label, eps):
    print(f"\nBlock norm ({label}) for δx={eps}:")
    print("  d/d(δp)   :", np.linalg.norm(diff[0, 0:3]))
    print("  d/d(δv)   :", np.linalg.norm(diff[0, 3:6]))
    print("  d/d(δb_a) :", np.linalg.norm(diff[0, 6:9]))
    print("  d/d(δθ)   :", np.linalg.norm(diff[0, 9:12]))
    print("  d/d(δb_g) :", np.linalg.norm(diff[0, 12:15]))
    print("  d/d(δp_IR):", np.linalg.norm(diff[0, 15:18]))
    print("  d/d(δθ_IR):", np.linalg.norm(diff[0, 18:21]))


# ---------------------------------------------------------------------------
# Build nominal state and run check
# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # Nominal 21-dim "state" for building arguments
    x0 = np.zeros(21)

    # Position (unused in measurement)
    x0[0:3] = np.array([1.0, 2.0, -0.5])

    # Velocity in world
    x0[3:6] = np.array([2.0, -0.3, 0.1])

    # accel bias (unused)
    x0[6:9] = np.array([0.01, -0.02, 0.005])

    # body attitude [roll, pitch, yaw]
    x0[9:12] = np.array([0.1, -0.05, 0.3])

    # gyro bias b_g
    x0[12:15] = np.array([0.001, -0.0005, 0.0008])

    # radar position in inertial frame
    x0[15:18] = np.array([0.5, -0.2, 0.1])

    # radar attitude euler (extrinsics)
    x0[18:21] = np.array([0.02, -0.01, 0.03])

    # Build arguments from x0
    roll_b, pitch_b, yaw_b = x0[9:12]
    R_WI = Rzyx(roll_b, pitch_b, yaw_b)

    v_WI = x0[3:6].reshape(3, 1)
    b_ars_ins = x0[12:15].reshape(3, 1)
    p_IR = x0[15:18].reshape(3, 1)

    roll_ir, pitch_ir, yaw_ir = x0[18:21]
    R_IR = Rzyx(roll_ir, pitch_ir, yaw_ir)

    # Radar LOS unit vector
    mu_r = np.array([0.8, 0.2, 0.56], dtype=float)
    mu_r /= np.linalg.norm(mu_r)

    # Measured body angular rate (rad/s)
    w_meas = np.array([0.4, -0.7, 0.5], dtype=float).reshape(3, 1)

    # Compute analytic and numeric Jacobians
    epsilon = 1e-5
    H_analytic = calculate_radar_H(mu_r, R_WI, v_WI, w_meas, b_ars_ins, p_IR, R_IR)

    H_fd = numeric_H(mu_r, R_WI, v_WI, w_meas, b_ars_ins, p_IR, R_IR, eps=epsilon)
    H_cd = numeric_H_central(mu_r, R_WI, v_WI, w_meas, b_ars_ins, p_IR, R_IR, eps=epsilon)

    diff_fd = H_fd - H_analytic
    diff_cd = H_cd - H_analytic

    print("\nAnalytic H:\n", H_analytic)
    print("\nForward-diff H:\n", H_fd)
    print("\nCentral-diff H:\n", H_cd)

    print("\nForward - Analytic:\n", diff_fd)
    print("\nCentral - Analytic:\n", diff_cd)

    print("\nMax |error| (forward):", np.max(np.abs(diff_fd)))
    print("Max |error| (central):", np.max(np.abs(diff_cd)))

    print_block_norms(diff_fd, "forward", epsilon)
    print_block_norms(diff_cd, "central", epsilon)



if __name__ == "__main__":
    main()
