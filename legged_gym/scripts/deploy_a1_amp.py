import glob
import sys
import matplotlib.pyplot as plt

import time
import numpy as np
import torch
import pygame

from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.datasets.motion_loader import AMPLoader

from motion_imitation.robots import robot_config
from motion_imitation.envs import env_builder

def play():

    # Initialize gamepad.
    pygame.init()
    p1 = pygame.joystick.Joystick(0)
    p1.init()
    print(p1.get_numaxes())

    # Load policy and discriminator.
    # TODO(aescontrela): Load from configs instead of manually specifying params.
    motion_loader = AMPLoader(
            'cpu', 6 * 0.005,
            motion_files=glob.glob('datasets/mocap_motions_slower/*'),)

    # POLICY_PATH = 'logs/amp_a1_reverse_gainrand/model_30900.pt'
    # POLICY_PATH = 'logs/amp_a1_reverse_gainrand_lowerlerp/model_21550.pt'
    # POLICY_PATH = 'logs/amp_a1_newobs/Feb12_02-00-51_/model_70150.pt'
    # POLICY_PATH = 'logs/a1_simplereward/model_1350.pt'
    # POLICY_PATH = 'logs/a1_rmareward/model_1500.pt'
    # POLICY_PATH = 'logs/amp_a1_newobs_reverse_faster/model_19450.pt'
    POLICY_PATH = 'logs/a1_amp_faster_reverse/model_66700.pt'
    # POLICY_PATH = 'logs/a1_amp_hopturn/model_4150.pt'
    # POLICY_PATH = 'logs/a1_amp_hopturn/model_8350.pt'
    # POLICY_PATH = 'logs/a1_amp_only_forward/model_90800.pt'
    # POLICY_PATH = 'logs/a1_amp_hopturn/model_13600.pt'
    # POLICY_PATH = 'logs/a1_amp_hopturn/model_112450.pt'
    loaded_dict = torch.load(POLICY_PATH, map_location=torch.device('cpu'))

    actor_critic = ActorCritic(
        num_actor_obs=42, num_critic_obs=48, num_actions=12,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation='elu', init_noise_std=1.0, fixed_std=False,)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])

    discriminator = AMPDiscriminator(
        input_dim=motion_loader.observation_dim * 2 + 2, amp_reward_coef=2.0, hidden_layer_sizes=[1024, 512], device='cpu',
        encoder_reward_coef=1.0, use_encoder_head=False, encode_dim=32, task_reward_lerp=0.0)
    discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])

    amp_normalizer = loaded_dict['amp_normalizer']

    # Build A1 env.
    env = env_builder.build_env_isaac(enable_rendering=False, use_real_robot=True)
    robot = env.robot

    # Move the motors slowly to initial position
    robot.ReceiveObservation()
    current_motor_angle = np.array(robot.GetMotorAngles())
    desired_motor_angle = np.array([0., 0.9, -1.8] * 4)

    for t in range(300):
        blend_ratio = np.minimum(t / 200., 1)
        action = (1 - blend_ratio
              ) * current_motor_angle + blend_ratio * desired_motor_angle
        robot.Step(action, robot_config.MotorControlMode.POSITION)
        time.sleep(0.005)


    # input("Continue Operation!")

    obs = env.reset()
    def _obs_dict_to_array(
            obs_dict,
            # order=["BaseVelocity", "ProjectedGravity", "FakeCommand", "MotorAngle", "MotorVelocity", "LastAction"],
            order=["ProjectedGravity", "FakeCommand", "MotorAngle", "MotorVelocity", "LastAction"]):
        return np.concatenate([obs_dict[o] for o in order])
    obs = _obs_dict_to_array(obs)

    ts = []
    cots = []
    commands = []
    base_lin_vels = []
    base_ang_vels = []

    name = 'composite'
    # forward_speed_x = []
    # lateral_speed_x = []
    # up_speed_x = []

    mass = sum(robot.GetBaseMassesFromURDF()) + sum(robot.GetLegMassesFromURDF())


    for _ in range(100000):
        action = actor_critic.act_inference(torch.tensor(obs).float())
        obs, _, _, _ = env.step(action.squeeze().detach().cpu().numpy())

        for event in pygame.event.get():
            norm_ang_vel = -1 * p1.get_axis(0)
            norm_lin_vel = -1 * p1.get_axis(1)
            if p1.get_button(0) == 1:

                np.save(name + '_ts.npy', np.array(ts))
                np.save(name + '_cots.npy', np.array(cots))
                np.save(name + '_commands.npy', np.array(commands))
                np.save(name + '_lin_vels.npy', np.array(base_lin_vels))
                np.save(name + '_ang_vels.npy', np.array(base_ang_vels))

                sys.exit()

        # lin_vel = 1.0 * norm_lin_vel
        # ang_vel = 1.5 * norm_ang_vel

        t = robot.GetTimeSinceReset()
        cot = np.dot(np.abs(robot.GetTrueMotorVelocities()), np.abs(robot.GetTrueMotorTorques())) / (
            np.linalg.norm(robot.GetBaseVelocity()) * 9.81 * mass)
        ts.append(t)
        cots.append(cot)

        print(robot.GetTrueBaseRollPitchYawRate()[2])

        if norm_lin_vel < 0.0:
            lin_vel = 1 * norm_lin_vel
        else:
            lin_vel = 1.75 * norm_lin_vel
        ang_vel = 1.57 * norm_ang_vel

        # if norm_lin_vel < 0.0:
        #     lin_vel = 0 * norm_lin_vel
        # else:
        #     lin_vel = 0 * norm_lin_vel
        # ang_vel = 2.2 * norm_ang_vel

        # if norm_lin_vel < 0.0:
        #     lin_vel = 0 * norm_lin_vel
        # else:
        #     lin_vel = 3 * norm_lin_vel
        # ang_vel = 0.4 * norm_ang_vel
        # if norm_lin_vel < 0.0:
        #     lin_vel = 0 * norm_lin_vel
        # else:
        #     lin_vel = 3 * norm_lin_vel
        # ang_vel = 0.45 * norm_ang_vel

        commands.append(np.array([lin_vel, 0.0, ang_vel]))
        base_lin_vels.append(robot.GetBaseVelocity())
        base_ang_vels.append(robot.GetTrueBaseRollPitchYawRate())

        obs["FakeCommand"] = np.array([lin_vel, 0.0, ang_vel]) * np.array([2.0, 2.0, 0.25])



        # print(obs["FakeCommand"])

        # command_speed_x.append(lin_vel)
        # forward_speed_x.append(robot.GetBaseVelocity()[0])



        # print(robot.GetBaseVelocity())

        obs = _obs_dict_to_array(obs)

    plt.plot(ts, cots)
    plt.grid()
    plt.show()

    plt.plot(ts, command_speed_x)
    plt.plot(ts, forward_speed_x)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    play()
