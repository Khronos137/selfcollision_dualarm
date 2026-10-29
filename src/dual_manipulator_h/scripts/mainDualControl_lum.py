#!/usr/bin/env python3
"""
ROS 2 node of the PROPOSED controller (dynamically weighted null-space control).

Launch:   ros2 launch dual_manipulator_h control_lum.launch.py [variant:=full]
Variants: full | no_weighting | no_vortex | no_weighting_no_vortex | swapped | symmetric
Any gain of Table 2 can also be overridden as a ROS parameter (e.g. rho_R:=300.0).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Time

from nodo_benchmark import NodoBenchmark
from selfCollisionStrategy import ControladorPropuesto, PARAMETROS_POR_DEFECTO, VARIANTES
from collision import puntos_dict, RADIOS_DICT, RADIO_TORSO
from libreria_rviz import generar_esqueleto_colisiones, generar_flecha_fuerza


class NodoPropuesto(NodoBenchmark):

    def __init__(self):
        controlador = ControladorPropuesto()
        super().__init__('main_dual_brazo_lum', 'LUMELSKY', controlador)

        self.declare_parameter('variant', 'full')
        variante = self.get_parameter('variant').value
        if variante not in VARIANTES:
            raise ValueError(f'Unknown variant {variante}. Options: {list(VARIANTES)}')
        parametros = dict(PARAMETROS_POR_DEFECTO)
        parametros.update(VARIANTES[variante])
        # Numeric overrides. A negative value means "not set" (keep the variant preset),
        # so the launch file can expose gains without overriding the presets.
        for nombre, valor in list(parametros.items()):
            if isinstance(valor, (int, float)) and not isinstance(valor, bool):
                self.declare_parameter(nombre, float(valor),
                                       ParameterDescriptor(dynamic_typing=True))
                v = float(self.get_parameter(nombre).value)
                if v >= 0.0:
                    parametros[nombre] = v
        controlador.p = parametros
        controlador.reiniciar()

        # Label written to the CSV / log folder (can be forced with 'etiqueta')
        self.declare_parameter('etiqueta', 'auto')
        etiqueta = str(self.get_parameter('etiqueta').value).strip()
        if etiqueta and etiqueta != 'auto':
            self.etiqueta = etiqueta
        else:
            self.etiqueta = 'LUMELSKY' if variante == 'full' else f'LUMELSKY_{variante}'
        self.pub_esqueleto = self.create_publisher(MarkerArray, '/colisiones/esqueleto', 10)
        self.pub_fuerzas = self.create_publisher(MarkerArray, '/fuerzas_repulsion', 10)
        self.get_logger().info(f'Label: {self.etiqueta} | variant: {variante} | parameters: {parametros}')

    def visualizar(self, info):
        t0 = Time()
        arr = generar_esqueleto_colisiones(puntos_dict(info['P']['L'], info['P']['R']),
                                           RADIOS_DICT, RADIO_TORSO, 'world')
        for m in arr.markers:
            m.header.stamp = t0
        self.pub_esqueleto.publish(arr)
        flechas = MarkerArray()
        borrar = Marker(); borrar.action = Marker.DELETEALL
        flechas.markers.append(borrar)
        for off, b in ((0, 'R'), (20, 'L')):
            for k, (p, F) in info['fuerzas'][b].items():
                f = generar_flecha_fuerza(p, F, id_flecha=k + off, factor_escala=0.08)
                f.header.stamp = t0
                f.ns = f'fuerza_{b}'
                flechas.markers.append(f)
        self.pub_fuerzas.publish(flechas)


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoPropuesto()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        import numpy as np
        nodo.publicar(np.zeros(6), np.zeros(6))
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()