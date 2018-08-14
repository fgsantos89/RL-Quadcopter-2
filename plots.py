import matplotlib.pyplot as plt
import numpy as np


def plot_training(results):
    fig, axarr = plt.subplots(4, 2, figsize=(22, 10))

    reward_training(results, axarr[0][0])
    position_training_x(results, axarr[0][1])
    position_training_y(results, axarr[1][0])
    position_training_z(results, axarr[1][1])
    velocity_axis_training(results, axarr[2][0])
    rotation_axis_training(results, axarr[2][1])
    velocity_angles_training(results, axarr[3][0])
    agent_choices_training(results, axarr[3][1])
    plt.subplots_adjust(hspace=.3)


def reward_training(results, plot):
    plot.set_title("Rewards", fontsize=18)
    plot.plot(results['episode'], results['reward'], label='rewards')
    plot.legend()


def position_training_x(results, plot):
    plot.set_title("Positions X", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['x'], alpha=0.1, color='red')


def position_training_y(results, plot):
    plot.set_title("Positions Y", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['y'], alpha=0.1, color='red')


def position_training_z(results, plot):
    plot.set_title("Positions Z", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['z'], alpha=0.1, color='red')


def velocity_axis_training(results, plot):
    plot.set_title("Velocity Axis", fontsize=18)
    for run in results['runs']:
        # label='x_hat'
        plot.plot(run['time'], run['x_velocity'], alpha=0.1, color='red')
        # label='y_hat'
        plot.plot(run['time'], run['y_velocity'], alpha=0.1, color='blue')
        # label='z_hat'
        plot.plot(run['time'], run['z_velocity'], alpha=0.1, color='green')


def rotation_axis_training(results, plot):
    plot.set_title("Rotation Axis", fontsize=18)
    for run in results['runs']:
        # phi
        plot.plot(run['time'], run['phi'], alpha=0.1, color='red')
        # theta
        plot.plot(run['time'], run['theta'], alpha=0.1, color='blue')
        # psi
        plot.plot(run['time'], run['psi'], alpha=0.1, color='green')


def velocity_angles_training(results, plot):
    plot.set_title("Velocity of angles", fontsize=18)
    for run in results['runs']:
        # label='phi_velocity'
        plot.plot(run['time'], run['phi_velocity'],
                  alpha=0.1, color='red')
        # label='theta_velocity'
        plot.plot(run['time'], run['theta_velocity'],
                  alpha=0.1, color='blue')
        # label='psi_velocity'
        plot.plot(run['time'], run['psi_velocity'],
                  alpha=0.1, color='green')


def agent_choices_training(results, plot):
    plot.set_title("Choices", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['z'], alpha=0.1, color='red')
        # label='Rotor 1 revolutions / second'
        plot.plot(run['time'], run['rotor_speed1'],
                  alpha=0.1, color='red')
        # label='Rotor 2 revolutions / second'
        plot.plot(run['time'], run['rotor_speed2'],
                  alpha=0.1, color='blue')
        # label='Rotor 3 revolutions / second'
        plot.plot(run['time'], run['rotor_speed3'],
                  alpha=0.1, color='green')
        # label='Rotor 4 revolutions / second'
        plot.plot(run['time'], run['rotor_speed4'],
                  alpha=0.1, color='orange')
