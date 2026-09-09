cmake_minimum_required(VERSION 3.8)
project(dual_manipulator_benchmark)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# 1. Generación del Mensaje
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BezierTrajectory.msg"
)

# 2. Instalación de los scripts de Python
install(PROGRAMS
  scripts/benchmark_director.py
#   scripts/benchmark_logger.py
  scripts/generador_bezier_offline.py
  DESTINATION lib/${PROJECT_NAME}
)

# 3. Instalación de configuraciones
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# 4. Instalación de Launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()