import matplotlib.pyplot as plt
import numpy as np


def plot_training(results):
    fig, axarr = plt.subplots(3, 2, figsize=(12, 8))

    reward_training(results, axarr[0][0])
    position_training(results['runs'][0], axarr[0][1])
    velocity_axis_training(results['runs'][0], axarr[1][0])
    rotation_axis_training(results['runs'][0], axarr[1][1])
    velocity_angles_training(results['runs'][0], axarr[2][0])
    agent_choices_training(results['runs'][0], axarr[2][1])
    plt.subplots_adjust(hspace=.3)


def reward_training(results, plot):
    plot.set_title("Rewards", fontsize=18)
    plot.plot(results['episode'], results['reward'], label='rewards')
    plot.legend()


def position_training(results, plot):
    plot.set_title("Positions", fontsize=18)
    plot.plot(results['time'], results['x'], label='x')
    plot.plot(results['time'], results['y'], label='y')
    plot.plot(results['time'], results['z'], label='z')
    plot.legend()


def velocity_axis_training(results, plot):
    plot.set_title("Velocity Axis", fontsize=18)
    plot.plot(results['time'], results['x_velocity'], label='x_hat')
    plot.plot(results['time'], results['y_velocity'], label='y_hat')
    plot.plot(results['time'], results['z_velocity'], label='z_hat')
    plot.legend()


def rotation_axis_training(results, plot):
    plot.set_title("Rotation Axis", fontsize=18)
    plot.plot(results['time'], results['phi'], label='phi')
    plot.plot(results['time'], results['theta'], label='theta')
    plot.plot(results['time'], results['psi'], label='psi')
    plot.legend()


def velocity_angles_training(results, plot):
    plot.set_title("Velocity of angles", fontsize=18)
    plot.plot(results['time'], results['phi_velocity'],
              label='phi_velocity')
    plot.plot(results['time'], results['theta_velocity'],
              label='theta_velocity')
    plot.plot(results['time'], results['psi_velocity'],
              label='psi_velocity')
    plot.legend()


def agent_choices_training(results, plot):
    plot.set_title("Choices", fontsize=18)
    plot.plot(results['time'], results['rotor_speed1'],
              label='Rotor 1 revolutions / second')
    plot.plot(results['time'], results['rotor_speed2'],
              label='Rotor 2 revolutions / second')
    plot.plot(results['time'], results['rotor_speed3'],
              label='Rotor 3 revolutions / second')
    plot.plot(results['time'], results['rotor_speed4'],
              label='Rotor 4 revolutions / second')
    plot.legend()
