cmake_minimum_required(VERSION 3.8)
project(dual_manipulator_h)

find_package(ament_cmake REQUIRED)
find_package(ros2_control REQUIRED)
find_package(ros_gz_sim REQUIRED)

# install(
#   PROGRAMS scripts/dual_kinematic_controller.py
#   DESTINATION lib/${PROJECT_NAME}
# )

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY description
  DESTINATION share/${PROJECT_NAME}
)

# install(PROGRAMS
#   scripts/mainDualControl_lum.py
#   scripts/collision.py
#   scripts/libreria_cinematica_der.py
#   scripts/libreria_cinematica_izq.py
#   scripts/libreria_rviz.py
#   DESTINATION lib/${PROJECT_NAME}
# )

# Instalar toda la carpeta de scripts para no perder las librerías
install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

install(DIRECTORY scripts/pyarmor_runtime_000000
  DESTINATION lib/${PROJECT_NAME}
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
