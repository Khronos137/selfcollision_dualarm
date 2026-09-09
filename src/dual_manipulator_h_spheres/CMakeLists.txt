cmake_minimum_required(VERSION 3.8)
project(dual_manipulator_h_spheres)

find_package(ament_cmake REQUIRED)
find_package(ros2_control REQUIRED)
find_package(ros_gz_sim REQUIRED)

install(
  PROGRAMS
    scripts/mainDualControl_sph.py
#     scripts/mainIzquierda.py
#     scripts/mainDerecha.py
    scripts/collision.py
    scripts/libreria_cinematica_der.py
    scripts/libreria_cinematica_izq.py
    scripts/libreria_rviz.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY description
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
