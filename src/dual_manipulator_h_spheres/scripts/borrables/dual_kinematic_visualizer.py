#!/usr/bin/env python3
"""
Joint Points Visualizer — Dual Manipulator-H
Computes forward kinematics for each joint frame and publishes:
  - PointStamped per joint on /left/joints/<jN> and /right/joints/<jN>
  - PointStamped for torso reference points on /torso/base and /torso/top
  - Named MarkerArray (spheres + text labels) on /joints/named_points
  - Real-time 3D matplotlib plot in a background thread
"""

import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================
REFERENCE_FRAME = 'base_link'   # 'base_link' | 'arm'


# =============================================================================
# Robot geometry — Manipulator-H (matches dual_manipulator_h controller)
# =============================================================================
JOINTS = [
    ([0.000,  0.000,  0.126], 'z'),   # joint1
    ([0.000,  0.069,  0.033], 'y'),   # joint2
    ([0.030, -0.0115, 0.264], 'y'),   # joint3
    ([0.195, -0.0575, 0.030], 'x'),   # joint4
    ([0.063,  0.045,  0.000], 'y'),   # joint5
    ([0.123, -0.045,  0.000], 'x'),   # joint6
]

# Fixed Y offsets accumulated from URDF between selected joint frames
Y_OFF_J3 = JOINTS[1][0][1] + JOINTS[2][0][1]
Y_OFF_J5 = JOINTS[4][0][1]

# Torso dimensions from XACRO (torso_height=0.8, base_to_torso z=0.4)
TORSO_HEIGHT = 0.8


# =============================================================================
# Static transforms: arm link1 frame → base_link
# Derived from XACRO mount points:
#   left_link1  in torso_link: xyz=[0, -0.1, 0.3], rpy=[-pi/2, 0,  pi]
#   right_link1 in torso_link: xyz=[0,  0.1, 0.3], rpy=[ pi/2, 0, -pi]
#   torso_link  in base_link:  xyz=[0,  0,   0.4]
# =============================================================================
def _rpy_to_R(roll, pitch, yaw):
    """Rotation matrix from RPY angles: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    Rx = np.array([[1, 0,               0              ],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll),  math.cos(roll)]])
    Ry = np.array([[ math.cos(pitch), 0, math.sin(pitch)],
                   [0,                1, 0              ],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0,              0,              1]])
    return Rz @ Ry @ Rx


_pi = math.pi
_R_left  = _rpy_to_R(-_pi / 2, 0,  _pi)
_t_left  = np.array([0.0, -0.1, 0.7])

_R_right = _rpy_to_R( _pi / 2, 0, -_pi)
_t_right = np.array([0.0,  0.1, 0.7])


def to_base(pos: np.ndarray, arm: str) -> np.ndarray:
    """Transform a position from the arm's local link1 frame to base_link."""
    R, t = (_R_left, _t_left) if arm == 'left' else (_R_right, _t_right)
    return R @ pos + t


# =============================================================================
# Forward kinematics
# =============================================================================
def fk_all_joints(q: np.ndarray):
    """Return the origin of each joint frame (J1..J6) in link1 coordinates."""
    T = np.eye(4)
    positions = []
    for i, (xyz, axis) in enumerate(JOINTS):
        c, s = math.cos(q[i]), math.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, 3] = xyz
        if axis == 'z':
            Ti[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        elif axis == 'y':
            Ti[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis == 'x':
            Ti[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        T = T @ Ti
        positions.append(T[:3, 3].copy())
    return positions


# =============================================================================
# Color maps
# =============================================================================
COLORS_RVIZ = {
    'torso_base': (1.0, 0.5, 0.0),
    'torso_top':  (0.0, 0.5, 1.0),
    'j0': (1.0, 1.0, 1.0),
    'j1': (1.0, 0.2, 0.2),
    'j3': (1.0, 1.0, 0.0),
    'j4': (0.2, 0.9, 0.2),
    'j5': (0.0, 0.9, 1.0),
    'j6': (0.7, 0.0, 1.0),
}

COLORS_MPL = {
    'torso_base': 'orange',
    'torso_top':  'dodgerblue',
    'j0': 'white',
    'j1': 'red',
    'j3': 'yellow',
    'j4': 'lime',
    'j5': 'cyan',
    'j6': 'violet',
}


def _point_key(name: str) -> str:
    """Strip arm prefix to get the base key for color lookup."""
    return name.split('_', 1)[1] if name.startswith(('left_', 'right_')) else name


# =============================================================================
# ROS 2 Node
# =============================================================================
class JointPointsVisualizer(Node):

    def __init__(self):
        super().__init__('joint_points_visualizer')

        self.q_left  = np.zeros(6)
        self.q_right = np.zeros(6)

        self._shared_pts: dict = {}
        self._pts_lock = threading.Lock()

        self.create_subscription(JointState, '/joint_states', self._cb_joints, 10)

        jnames = ['j0', 'j1', 'j3', 'j4', 'j5', 'j6']
        self._pub_l = [self.create_publisher(PointStamped, f'/left/joints/{n}',  10) for n in jnames]
        self._pub_r = [self.create_publisher(PointStamped, f'/right/joints/{n}', 10) for n in jnames]

        self._pub_torso_base = self.create_publisher(PointStamped, '/torso/base', 10)
        self._pub_torso_top  = self.create_publisher(PointStamped, '/torso/top',  10)

        self._pub_l_skel = self.create_publisher(Marker,      '/left/joints/skeleton',  10)
        self._pub_r_skel = self.create_publisher(Marker,      '/right/joints/skeleton', 10)
        self._pub_named  = self.create_publisher(MarkerArray, '/joints/named_points',   10)

        self._ml_skel = self._make_skeleton_marker('left_link1',  'left_skeleton',  0)
        self._mr_skel = self._make_skeleton_marker('right_link1', 'right_skeleton', 1)

        self.create_timer(0.1, self._publish)

        threading.Thread(target=self._run_plot, daemon=True).start()

        self.get_logger().info(
            f'JointPointsVisualizer started | REFERENCE_FRAME = {REFERENCE_FRAME!r}'
        )

    # -------------------------------------------------------------------------
    # Marker factory helpers
    # -------------------------------------------------------------------------
    def _make_skeleton_marker(self, frame: str, ns: str, mid: int) -> Marker:
        m = Marker()
        m.header.frame_id = frame
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.004
        m.color.r = m.color.g = m.color.b = 0.8
        m.color.a = 1.0
        m.pose.orientation.w = 1.0
        return m

    def _sphere_marker(self, now, frame: str, name: str, mid: int, pos) -> Marker:
        r, g, b = COLORS_RVIZ.get(_point_key(name), (1.0, 1.0, 1.0))
        m = Marker()
        m.header.stamp = now
        m.header.frame_id = frame
        m.ns = name
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.025
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
        return m

    def _text_marker(self, now, frame: str, name: str, mid: int, pos) -> Marker:
        t = Marker()
        t.header.stamp = now
        t.header.frame_id = frame
        t.ns = name
        t.id = mid
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = float(pos[0])
        t.pose.position.y = float(pos[1])
        t.pose.position.z = float(pos[2]) + 0.04
        t.pose.orientation.w = 1.0
        t.scale.z = 0.022
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = name
        return t

    # -------------------------------------------------------------------------
    # Joint state callback
    # -------------------------------------------------------------------------
    def _cb_joints(self, msg: JointState):
        for i in range(6):
            nl, nr = f'left_joint{i+1}', f'right_joint{i+1}'
            if nl in msg.name:
                self.q_left[i]  = msg.position[msg.name.index(nl)]
            if nr in msg.name:
                self.q_right[i] = msg.position[msg.name.index(nr)]

    # -------------------------------------------------------------------------
    # Main publish cycle
    # -------------------------------------------------------------------------
    def _publish(self):
        now      = self.get_clock().now().to_msg()
        use_base = (REFERENCE_FRAME == 'base_link')

        # Compute joint positions in local arm frame
        arm_local: dict = {}
        for prefix, q, arm in [('left', self.q_left, 'left'), ('right', self.q_right, 'right')]:
            raw = fk_all_joints(q)
            arm_local[f'{prefix}_j0'] = (arm, np.zeros(3))
            arm_local[f'{prefix}_j1'] = (arm, raw[0].copy())
            arm_local[f'{prefix}_j3'] = (arm, np.array([raw[2][0], raw[2][1] - Y_OFF_J3, raw[2][2]]))
            arm_local[f'{prefix}_j4'] = (arm, raw[3].copy())
            arm_local[f'{prefix}_j5'] = (arm, np.array([raw[4][0], raw[4][1] - Y_OFF_J5, raw[4][2]]))
            arm_local[f'{prefix}_j6'] = (arm, raw[5].copy())

        def resolve(arm, lpos):
            return ('base_link', to_base(lpos, arm)) if use_base else (f'{arm}_link1', lpos)

        all_pts: dict = {name: resolve(arm, lpos) for name, (arm, lpos) in arm_local.items()}
        all_pts['torso_base'] = ('base_link', np.array([0.0, 0.0, 0.0]))
        all_pts['torso_top']  = ('base_link', np.array([0.0, 0.0, TORSO_HEIGHT]))

        # Update shared data for the matplotlib thread (always in base_link)
        with self._pts_lock:
            self._shared_pts = {
                name: to_base(lpos, arm) for name, (arm, lpos) in arm_local.items()
            }
            self._shared_pts['torso_base'] = np.array([0.0, 0.0, 0.0])
            self._shared_pts['torso_top']  = np.array([0.0, 0.0, TORSO_HEIGHT])

        # Publish individual PointStamped for arm joints
        keys_l = ['left_j0',  'left_j1',  'left_j3',  'left_j4',  'left_j5',  'left_j6']
        keys_r = ['right_j0', 'right_j1', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        for keys, pubs in [(keys_l, self._pub_l), (keys_r, self._pub_r)]:
            for key, pub in zip(keys, pubs):
                frame, pos = all_pts[key]
                ps = PointStamped()
                ps.header.stamp = now
                ps.header.frame_id = frame
                ps.point.x = float(pos[0]); ps.point.y = float(pos[1]); ps.point.z = float(pos[2])
                pub.publish(ps)

        # Publish torso reference points
        for pub, key in [(self._pub_torso_base, 'torso_base'), (self._pub_torso_top, 'torso_top')]:
            frame, pos = all_pts[key]
            ps = PointStamped()
            ps.header.stamp = now
            ps.header.frame_id = frame
            ps.point.x = float(pos[0]); ps.point.y = float(pos[1]); ps.point.z = float(pos[2])
            pub.publish(ps)

        # Publish skeleton LINE_STRIP
        for prefix, m_sk, pub_sk in [('left',  self._ml_skel, self._pub_l_skel),
                                      ('right', self._mr_skel, self._pub_r_skel)]:
            skeys = [f'{prefix}_j0', f'{prefix}_j1', f'{prefix}_j3',
                     f'{prefix}_j4', f'{prefix}_j5', f'{prefix}_j6']
            m_sk.header.frame_id = all_pts[skeys[0]][0]
            m_sk.header.stamp = now
            m_sk.points = [Point(x=float(all_pts[k][1][0]),
                                 y=float(all_pts[k][1][1]),
                                 z=float(all_pts[k][1][2])) for k in skeys]
            pub_sk.publish(m_sk)

        # Publish named MarkerArray
        ma = MarkerArray()
        for mid, (name, (frame, pos)) in enumerate(all_pts.items()):
            ma.markers.append(self._sphere_marker(now, frame, name, mid * 2,     pos))
            ma.markers.append(self._text_marker  (now, frame, name, mid * 2 + 1, pos))
        self._pub_named.publish(ma)

    # -------------------------------------------------------------------------
    # Real-time 3D matplotlib plot (background thread)
    # -------------------------------------------------------------------------
    def _run_plot(self):
        SKEL_KEYS = ['j0', 'j1', 'j3', 'j4', 'j5', 'j6']
        X_LIM = (-0.8, 0.8)
        Y_LIM = (-0.8, 0.8)
        Z_LIM = (0.0,  1.6)

        plt.ion()
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('#1a1a1a')
        fig.canvas.manager.set_window_title('FK Joint Visualizer')

        while rclpy.ok():
            with self._pts_lock:
                pts = {k: v.copy() for k, v in self._shared_pts.items()}

            if pts:
                ax.cla()
                ax.set_facecolor('#1a1a1a')
                ax.set_xlim(*X_LIM)
                ax.set_ylim(*Y_LIM)
                ax.set_zlim(*Z_LIM)
                ax.set_xlabel('X [m]', color='white')
                ax.set_ylabel('Y [m]', color='white')
                ax.set_zlabel('Z [m]', color='white')
                ax.set_title('Joint Positions — FK (base_link)', color='white')
                ax.tick_params(colors='white')

                for name, pos in pts.items():
                    c = COLORS_MPL.get(_point_key(name), 'white')
                    ax.scatter(pos[0], pos[1], pos[2], c=c, s=80,
                               edgecolors='gray', depthshade=False, zorder=5)
                    ax.text(pos[0], pos[1], pos[2] + 0.03, name, fontsize=7, color='white')

                for prefix, lc in [('left', 'tomato'), ('right', 'cornflowerblue')]:
                    sk = [f'{prefix}_{k}' for k in SKEL_KEYS]
                    if all(k in pts for k in sk):
                        ax.plot([pts[k][0] for k in sk],
                                [pts[k][1] for k in sk],
                                [pts[k][2] for k in sk],
                                color=lc, linewidth=1.5)

            plt.pause(0.1)

        plt.close()


# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = JointPointsVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
