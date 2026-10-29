#!/usr/bin/env python3
"""
ROS 2 node of the BASELINE controller (Lei et al., 2020).

Launch:   ros2 launch dual_manipulator_h_spheres control_sph.launch.py
Gains of the baseline can be overridden as ROS parameters (e.g. V_max:=1.5).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time

from nodo_benchmark import NodoBenchmark
from leiStrategy import ControladorLei, PARAMETROS_LEI
from collision import esferas, RADIOS_INFLADOS


class NodoLei(NodoBenchmark):

    def __init__(self):
        controlador = ControladorLei()
        super().__init__('main_dual_brazo_sph', 'LEI', controlador)
        parametros = dict(PARAMETROS_LEI)
        for nombre, valor in parametros.items():
            self.declare_parameter(nombre, float(valor))
            parametros[nombre] = float(self.get_parameter(nombre).value)
        controlador.p = parametros
        self.pub_vol = self.create_publisher(MarkerArray, '/collision_viz/volumen', 10)
        self.get_logger().info(f'Baseline parameters: {parametros}')

    def visualizar(self, info):
        t0 = Time()
        ma = MarkerArray()
        i = 0
        for lado in ('L', 'R'):
            for k, t, c in esferas(info['P'][lado]):
                m = Marker()
                m.header.frame_id = 'world'; m.header.stamp = t0
                m.ns = f'vol_{lado}'; m.id = i; i += 1
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, c)
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 2.0 * RADIOS_INFLADOS[k]
                m.color.r, m.color.g, m.color.b, m.color.a = 0.3, 0.6, 1.0, 0.25
                ma.markers.append(m)
        self.pub_vol.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoLei()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.publicar(np.zeros(6), np.zeros(6))
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
