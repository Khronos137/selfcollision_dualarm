"""
Kinematics of the discrete chain points used by both collision models.

SHARED MODULE: this file must be identical in
    dual_manipulator_h/scripts/ and dual_manipulator_h_spheres/scripts/

Each arm is described by 11 chain points P[0..10] (P[10] is the end-effector).
Segment k (k = 1..10) joins P[k-1] and P[k]. Every chain point is a fixed linear
combination of the kinematic points p0, p1, p2, p4, p5, p6, so the linear-velocity
Jacobian of ANY point on a segment is the same linear combination of the exact
analytic Jacobians of those points (no numerical differentiation).
"""
import numpy as np

from libreria_cinematica_der import derJac6ManH
from libreria_cinematica_izq import derJac6IzqManH

N_PUNTOS = 11      # chain points per arm
K_SEG = 10         # segments (capsules) per arm

# Geometric parameters of the dual ROBOTIS Manipulator-H model [h, b, l1, l1b, l2, l3]
L_ROBOT = [0.70, 0.159 + 0.10, 0.264, 0.030, 0.258, 0.123]


# =====================================================================
# 1. CHAIN POINTS (forward kinematics)
# =====================================================================
def _puntos(L, q, lado):
    """Returns the 11 chain points of one arm as an array (11, 3).
    lado = 'left' or 'right'. Formulas identical to the original collision.py."""
    h, b, l1, l1b, l2, l3 = L
    l1a = (l1**2 + l1b**2)**0.5
    q1, q2, q3, q4, q5 = q[0], q[1], q[2] + 0.7854, q[3], q[4]
    sg = 1.0 if lado == 'left' else -1.0     # mirror sign in y (and in z terms of q1)

    c1, s1 = np.cos(q1), np.sin(q1)
    a2 = q2 + 0.113
    a3 = q3 + 0.67

    p0 = np.array([0.0, 0.0, h])
    p1 = np.array([0.0, sg * b, h])
    p2 = np.array([l1 * np.sin(q2) * c1,
                   sg * (b + l1 * np.cos(q2)),
                   h - sg * l1 * np.sin(q2) * s1])
    p3 = np.array([l1a * np.sin(a2) * c1,
                   sg * (b + l1a * np.cos(a2)),
                   h - sg * l1a * np.sin(a2) * s1])
    p4 = np.array([
        p3[0] - l1b * np.cos(a3) * np.cos(a2) * c1 + l1b * np.sin(a3) * np.sin(a2) * c1,
        p3[1] + sg * (l1b * np.cos(a3) * np.sin(a2) + l1b * np.cos(a2) * np.sin(a3)),
        p3[2] + sg * (l1b * np.cos(a3) * np.cos(a2) * s1 - l1b * np.sin(a3) * np.sin(a2) * s1)])
    p5 = np.array([
        p4[0] + l2 * (np.cos(a3) * np.sin(a2) * c1 + np.cos(a2) * np.sin(a3) * c1),
        p4[1] + sg * l2 * (np.cos(a3) * np.cos(a2) - np.sin(a3) * np.sin(a2)),
        p4[2] - sg * l2 * (np.cos(a3) * np.sin(a2) * s1 + np.cos(a2) * np.sin(a3) * s1)])
    dx6 = - l3 * (np.sin(q5) * (s1 * np.sin(q4) - np.cos(q4) * (np.cos(a3) * np.cos(a2) * c1 - np.sin(a3) * np.sin(a2) * c1))
                  - np.cos(q5) * (np.cos(a3) * np.sin(a2) * c1 + np.cos(a2) * np.sin(a3) * c1))
    dy6 = sg * l3 * (np.cos(q5) * (np.cos(a3) * np.cos(a2) - np.sin(a3) * np.sin(a2))
                     - np.cos(q4) * np.sin(q5) * (np.cos(a3) * np.sin(a2) + np.cos(a2) * np.sin(a3)))
    dz6 = - sg * l3 * (np.cos(q5) * (np.cos(a3) * np.sin(a2) * s1 + np.cos(a2) * np.sin(a3) * s1)
                       + np.sin(q5) * (np.cos(q4) * (np.cos(a3) * np.cos(a2) * s1 - np.sin(a3) * np.sin(a2) * s1) + c1 * np.sin(q4)))
    p6 = p5 + np.array([dx6, dy6, dz6])

    p1_in = p1 + 0.15 * (p0 - p1)
    p1_out = p1 + 0.20 * (p2 - p1)
    p4_out = p4 + 0.20 * (p5 - p4)
    p5_in = p5 - 0.10 * (p5 - p4)
    p5_out = p5 + 0.35 * (p6 - p5)
    return np.array([p0, p1_in, p1, p1_out, p2, p4, p4_out, p5_in, p5, p5_out, p6])


def puntos_cadena(q, lado, L=L_ROBOT):
    return _puntos(L, q, lado)


# =====================================================================
# 2. EXACT JACOBIANS OF THE CHAIN POINTS
# =====================================================================
def _jac_p2(L, q, lado):
    h, b, l1 = L[0], L[1], L[2]
    q1, q2 = q[0], q[1]
    sg = 1.0 if lado == 'left' else -1.0
    J = np.zeros((3, 6))
    J[:, 0] = [-l1 * np.sin(q2) * np.sin(q1), 0.0, -sg * l1 * np.sin(q2) * np.cos(q1)]
    J[:, 1] = [l1 * np.cos(q2) * np.cos(q1), -sg * l1 * np.sin(q2), -sg * l1 * np.cos(q2) * np.sin(q1)]
    return J


def jacobianos_cadena(q, lado, L=L_ROBOT):
    """Returns an array (11, 3, 6): linear-velocity Jacobian of every chain point.
    Columns of joints distal to a point are identically zero by construction."""
    h, b, l1, l1b, l2, l3 = L
    jac = derJac6IzqManH if lado == 'left' else derJac6ManH
    J2 = _jac_p2(L, q, lado)
    J4 = jac([h, b, l1, l1b, 0.0, 0.0], q)     # elbow point p4  (l2 = l3 = 0)
    J5 = jac([h, b, l1, l1b, l2, 0.0], q)      # wrist point p5  (l3 = 0)
    J6 = jac(L, q)                             # end-effector p6
    Z = np.zeros((3, 6))
    return np.array([
        Z,                      # p0
        Z,                      # p1_in
        Z,                      # p1
        0.20 * J2,              # p1_out = p1 + 0.2 (p2 - p1)
        J2,                     # p2
        J4,                     # p4
        0.80 * J4 + 0.20 * J5,  # p4_out
        0.10 * J4 + 0.90 * J5,  # p5_in
        J5,                     # p5
        0.65 * J5 + 0.35 * J6,  # p5_out
        J6,                     # p6 (end-effector)
    ])


def jacobiano_en_segmento(JC, k, s):
    """Jacobian of the point P[k-1] + s (P[k] - P[k-1]) on segment k (exact)."""
    return (1.0 - s) * JC[k - 1] + s * JC[k]
