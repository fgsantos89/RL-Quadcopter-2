import matplotlib.pyplot as plt
import numpy as np


def print_rewards(rewards):
    total_start = np.sum([x['start'] for x in rewards])
    total_distance_z = np.sum([x['distance_z'] for x in rewards])
    total_z_axis_valocity = np.sum([x['z_axis_valocity'] for x in rewards])
    total_angular_velocity_sum = np.sum(
        [x['angular_velocity_sum'] for x in rewards])
    total_distance_x_y_sum = np.sum([x['distance_x_y_sum'] for x in rewards])
    total_height = np.sum(
        [x['height'] if 'height' in x else 0 for x in rewards])
    total_crash = np.sum([x['crash'] if 'crash' in x else 0 for x in rewards])
    total_total = np.sum([x['total'] for x in rewards])

    print('total_start={0:.2f}'.format(total_start / total_total))
    print('total_distance_z={0:.2f}'.format(total_distance_z / total_total))
    print('total_z_axis_valocity={0:.2f}'.format(
        total_z_axis_valocity / total_total))
    print('total_angular_velocity_sum={0:.2f}'.format(
        total_angular_velocity_sum / total_total))
    print('total_distance_x_y_sum={0:.2f}'.format(
        total_distance_x_y_sum / total_total))
    print('total_height={0:.2f}'.format(total_height / total_total))
    print('total_crash={0:.2f}'.format(total_crash / total_total))


def plot_results(results, alpha=0.1):
    fig, axarr = plt.subplots(4, 2, figsize=(22, 20))

    reward_results(results, axarr[0][0])
    position_results_x(results, axarr[0][1], alpha)
    position_results_y(results, axarr[1][0], alpha)
    position_results_z(results, axarr[1][1], alpha)
    velocity_axis_results(results, axarr[2][0], alpha)
    rotation_axis_results(results, axarr[2][1], alpha)
    velocity_angles_results(results, axarr[3][0], alpha)
    agent_choices_results(results, axarr[3][1], alpha)
    plt.subplots_adjust(hspace=.3)

    return fig


def reward_results(results, plot):
    plot.set_title("Rewards", fontsize=18)
    plot.plot(results['episode'], results['reward'], label='rewards')
    plot.legend()


def position_results_x(results, plot, alpha):
    plot.set_title("Positions X", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['x'], alpha=alpha, color='red')


def position_results_y(results, plot, alpha):
    plot.set_title("Positions Y", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['y'], alpha=alpha, color='red')


def position_results_z(results, plot, alpha):
    plot.set_title("Positions Z", fontsize=18)
    for run in results['runs']:
        plot.plot(run['time'], run['z'], alpha=alpha, color='red')


def velocity_axis_results(results, plot, alpha):
    plot.set_title("Velocity Axis", fontsize=18)
    for run in results['runs']:
        # label='x_hat'
        plot.plot(run['time'], run['x_velocity'], alpha=alpha, color='red')
        # label='y_hat'
        plot.plot(run['time'], run['y_velocity'], alpha=alpha, color='blue')
        # label='z_hat'
        plot.plot(run['time'], run['z_velocity'], alpha=alpha, color='green')


def rotation_axis_results(results, plot, alpha):
    plot.set_title("Rotation Axis", fontsize=18)
    for run in results['runs']:
        # phi
        plot.plot(run['time'], run['phi'], alpha=alpha, color='red')
        # theta
        plot.plot(run['time'], run['theta'], alpha=alpha, color='blue')
        # psi
        plot.plot(run['time'], run['psi'], alpha=alpha, color='green')


def velocity_angles_results(results, plot, alpha):
    plot.set_title("Velocity of angles", fontsize=18)
    for run in results['runs']:
        # label='phi_velocity'
        plot.plot(run['time'], run['phi_velocity'],
                  alpha=alpha, color='red')
        # label='theta_velocity'
        plot.plot(run['time'], run['theta_velocity'],
                  alpha=alpha, color='blue')
        # label='psi_velocity'
        plot.plot(run['time'], run['psi_velocity'],
                  alpha=alpha, color='green')


def agent_choices_results(results, plot, alpha):
    plot.set_title("Choices", fontsize=18)
    for run in results['runs']:
        # label='Rotor 1 revolutions / second'
        plot.plot(run['time'], run['rotor_speed1'],
                  alpha=alpha, color='red')
        # label='Rotor 2 revolutions / second'
        plot.plot(run['time'], run['rotor_speed2'],
                  alpha=alpha, color='blue')
        # label='Rotor 3 revolutions / second'
        plot.plot(run['time'], run['rotor_speed3'],
                  alpha=alpha, color='green')
        # label='Rotor 4 revolutions / second'
        plot.plot(run['time'], run['rotor_speed4'],
                  alpha=alpha, color='orange')
