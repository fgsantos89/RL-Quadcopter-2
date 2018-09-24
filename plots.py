import matplotlib.pyplot as plt
import numpy as np


def print_rewards(rewards):
    list_distance_z = [x['distance_z'] for x in rewards]
    list_z_axis_valocity = [x['z_axis_velocity'] for x in rewards]
    list_angular_velocity = [x['angular_velocity_sum'] for x in rewards]
    list_distance_x_y = [x['distance_x_y_sum'] for x in rewards]
    list_height = [x['height'] if 'height' in x else 0 for x in rewards]
    list_crash = [x['crash'] if 'crash' in x else 0 for x in rewards]

    print('min_distance_z={0:.4f}'.format(np.min(list_distance_z)))
    print('max_distance_z={0:.4f}'.format(np.amax(list_distance_z)))
    print('mean_distance_z={0:.4f}'.format(np.mean(list_distance_z)))
    print('')

    print('min_z_axis_valocity={0:.4f}'.format(np.min(list_z_axis_valocity)))
    print('max_z_axis_valocity={0:.4f}'.format(np.amax(list_z_axis_valocity)))
    print('mean_z_axis_valocity={0:.4f}'.format(np.mean(list_z_axis_valocity)))
    print('')

    print('min_angular_velocity={0:.4f}'.format(np.min(list_angular_velocity)))
    print('max_angular_velocity={0:.4f}'.format(
        np.amax(list_angular_velocity)))
    print('mean_angular_velocity={0:.4f}'.format(
        np.mean(list_angular_velocity)))
    print('')

    print('min_distance_x_y={0:.4f}'.format(np.min(list_distance_x_y)))
    print('max_distance_x_y={0:.4f}'.format(np.amax(list_distance_x_y)))
    print('mean_distance_x_y={0:.4f}'.format(np.mean(list_distance_x_y)))
    print('')

    print('min_height={0:.4f}'.format(np.min(list_height)))
    print('max_height={0:.4f}'.format(np.amax(list_height)))
    print('mean_height={0:.4f}'.format(np.mean(list_height)))
    print('')

    print('min_crash={0:.4f}'.format(np.min(list_crash)))
    print('max_crash={0:.4f}'.format(np.amax(list_crash)))
    print('mean_crash={0:.4f}'.format(np.mean(list_crash)))
    print('')


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
