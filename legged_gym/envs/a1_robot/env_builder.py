import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
from legged_gym.envs.a1_robot import locomotion_gym_config
from legged_gym.envs.a1_robot import locomotion_gym_env
from legged_gym.envs.a1_robot.env_wrappers import action_scale_wrapper
# from legged_gym.envs.a1_robot.env_wrappers import reset_task
from legged_gym.envs.a1_robot.env_wrappers import simple_openloop
from legged_gym.envs.a1_robot.env_wrappers import trajectory_generator_wrapper_env
from legged_gym.envs.a1_robot import environment_sensors
from legged_gym.envs.a1_robot import robot_sensors
from legged_gym.envs.a1_robot import a1
from legged_gym.envs.a1_robot import a1_robot
from legged_gym.envs.a1_robot import robot_config


def build_env_isaac(enable_rendering=True,
                    use_real_robot=False,
                    realistic_sim=False):


  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
  sim_params.num_action_repeat = 33
  sim_params.enable_action_filter = False
  sim_params.torque_limits = 40.0
  sim_params.enable_clip_motor_commands = False
  sim_params.enable_action_interpolation = False

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  # robot_kwargs = {"self_collision_enabled": True}
  robot_kwargs = {}

  if use_real_robot:
    robot_class = a1_robot.A1Robot
  else:
    robot_class = a1.A1

  if use_real_robot or realistic_sim:
    robot_kwargs["reset_func_name"] = "_SafeJointsReset"
    robot_kwargs["velocity_source"] = a1.VelocitySource.IMU_FOOT_CONTACT
  else:
    robot_kwargs["reset_func_name"] = "_PybulletReset"
  num_motors = a1.NUM_MOTORS

  # self.base_lin_vel * self.obs_scales.lin_vel,
  # self.base_ang_vel  * self.obs_scales.ang_vel,
  # self.projected_gravity,
  # self.commands[:, :3] * self.commands_scale,
  # (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
  # self.dof_vel * self.obs_scales.dof_vel,
  # self.actions

  sensors = [
    robot_sensors.BaseVelocitySensor(use_real_robot=use_real_robot),
    robot_sensors.ProjectedGravitySensor(),
    robot_sensors.FakeCommandSensor(),
    robot_sensors.MotorAngleSensor(
      noisy_reading=False,
      num_motors=num_motors,
      default_pose=np.array([0.0, 0.9, -1.8] * 4),
      scales=np.array([1.0] * 12)),
    robot_sensors.MotorVelocitySensor(
      noisy_reading=False,
      num_motors=num_motors,
      scales=np.array([0.05] * 12)),
    environment_sensors.LastActionSensor(num_actions=num_motors, scale=0.25, default_pose=np.array([0.0, 0.9, -1.8] * 4))
  ]

  randomizers = []

  robot_kwargs['velocity_source'] = a1.VelocitySource.IMU_FOOT_CONTACT

  env = locomotion_gym_env.LocomotionGymEnv(
      gym_config=gym_config,
      robot_class=robot_class,
      robot_kwargs=robot_kwargs,
      env_randomizers=randomizers,
      robot_sensors=sensors,
      task=None)

  # env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  env = action_scale_wrapper.ActionScaleWrapper(env, 0.25, np.array([0.0, 0.9, -1.8] * 4))
  return env

