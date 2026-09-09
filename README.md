# Dual Manipulator Evaluation and Control (ROS 2)

This package contains the necessary launch files to initialize the dual manipulator control system, as well as the benchmarking tools for trajectory performance evaluation. 

The system is designed for the ROS 2 Humble environment.

## Prerequisites

Before executing the launch files, ensure that you have built your workspace and sourced the environment.

source /opt/ros/humble/setup.bash
source install/setup.bash

## Running the System

To start the complete environment, execute the following commands in separate terminals. It is recommended to wait for the first process to be fully initialized before proceeding to the next ones.

### 1. Manipulator Initialization
This command brings up the base configuration for the robotic arms, initializing the robot descriptions, state publishers, and, if applicable, the simulation environment (e.g., Gazebo).

ros2 launch dual_manipulator_h dual_manipulator.launch.py

### 2. Controller Execution
Starts the main control node responsible for executing the control law, calculating the required dynamics, and maintaining system stability throughout the movements.

ros2 launch dual_manipulator_h control_lum.launch.py

### 3. Benchmarking and Test Director
Runs the director node designed to coordinate trajectories, record control metrics, and evaluate the performance of the manipulators during operation.

ros2 launch dual_manipulator_benchmark director.launch.py

Devlpd by -khronos :D-