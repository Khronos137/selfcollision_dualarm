#!/usr/bin/env python3
"""
Sign test of the Gazebo velocity interface (run with Gazebo up and NO controller running).

For every joint of both arms it commands +0.3 rad/s during 0.5 s (all other joints at 0),
waits 0.5 s, measures the position change, and then commands -0.3 rad/s to come back.
Expected result: delta ~ +0.15 rad for every joint. A negative delta means that the
velocity command of that joint moves it in the opposite direction of its measured position.

Usage:  source install/setup.bash && python3 src/dual_manipulator_h/tools/prueba_signos.py
"""
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

VEL, DUR = 0.3, 0.5


class PruebaSignos(Node):
    def __init__(self):
        super().__init__('prueba_signos')
        self.q = {}
        self.create_subscription(JointState, '/joint_states', self.cb, 10)
        self.pub = {'left': self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10),
                    'right': self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)}

    def cb(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.q[n] = p

    def esperar(self, segundos, brazo=None, cmd=None):
        t0 = time.time()
        while time.time() - t0 < segundos:
            if brazo is not None:
                self.pub[brazo].publish(Float64MultiArray(data=cmd))
            rclpy.spin_once(self, timeout_sec=0.01)

    def correr(self):
        self.esperar(1.0)
        if not self.q:
            print('No /joint_states received. Is Gazebo running?')
            return
        print(f'{"joint":>14s} {"q_before":>9s} {"q_after":>9s} {"delta":>8s}  verdict')
        for brazo in ('left', 'right'):
            for j in range(6):
                nombre = f'{brazo}_joint{j+1}'
                cmd = [0.0] * 6
                cmd[j] = VEL
                q0 = self.q[nombre]
                self.esperar(DUR, brazo, cmd)
                self.esperar(0.5, brazo, [0.0] * 6)
                q1 = self.q[nombre]
                d = q1 - q0
                veredicto = 'OK' if d > 0.05 else ('INVERTED' if d < -0.05 else 'NO MOTION')
                print(f'{nombre:>14s} {q0:+9.3f} {q1:+9.3f} {d:+8.3f}  {veredicto}')
                cmd[j] = -VEL
                self.esperar(DUR, brazo, cmd)
                self.esperar(0.3, brazo, [0.0] * 6)


def main():
    rclpy.init()
    n = PruebaSignos()
    try:
        n.correr()
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()