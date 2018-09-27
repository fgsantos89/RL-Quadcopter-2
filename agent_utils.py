import sys
import datetime
import os
import json
import inspect

LABELS = ['time',
          'x', 'y', 'z',
          'phi', 'theta', 'psi',
          'x_velocity', 'y_velocity', 'z_velocity',
          'phi_velocity', 'theta_velocity', 'psi_velocity',
          'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']


def save(agent, training_plot, testing_plot):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cur_dir = os.getcwd()
    results_dir_path = cur_dir + '/results'

    if not os.path.isdir(results_dir_path):
        os.mkdir(results_dir_path)

    result_dir_path = results_dir_path + '/' + timestamp
    if not os.path.isdir(result_dir_path):
        os.mkdir(result_dir_path)

    training_plot.savefig(result_dir_path + '/' + 'training.png')
    testing_plot.savefig(result_dir_path + '/' + 'testing.png')

    json_agent = agent_to_dict(agent)
    with open(result_dir_path + '/' + 'agent_params.json', "w") as agent_file:
        json.dump(json_agent, agent_file, indent=4, sort_keys=True)

    with open(result_dir_path + '/' + 'model_summary.txt', "w") as model_file:
        model_file.write('ACTOR LOCAL:\n')
        model_file.write('============\n')
        agent.actor_local.model.summary(
            print_fn=lambda x: model_file.write(x + '\n'))

        model_file.write('\n\nCRITIC LOCAL:\n')
        model_file.write('=============\n')
        agent.critic_local.model.summary(
            print_fn=lambda x: model_file.write(x + '\n'))


def agent_to_dict(agent):
    return {
        'tau': agent.tau,
        'gamma': agent.gamma,
        'best_score': agent.best_score,
        'noise': {
            'exploration_mu': agent.exploration_mu,
            'exploration_theta': agent.exploration_theta,
            'exploration_sigma': agent.exploration_sigma
        },
        'batch_size': agent.batch_size,
        'buffer_size': agent.buffer_size,
        'reward_function': inspect.getsource(agent.task.get_reward)
    }


def testing(agent, num_episodes):
    return run_env(agent, num_episodes, False)


def training(agent, num_episodes):
    return run_env(agent, num_episodes, True)


def run_env(agent, num_episodes, training):
    memory_dict = {'episode': [], 'reward': [], 'runs': []}
    done = False
    task = agent.task
    for i_episode in range(num_episodes):
        state = agent.reset_episode()
        run = {x: [] for x in LABELS}
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            if training:
                agent.step(action, reward, next_state, done)
            state = next_state

            to_write = [task.sim.time] + list(task.sim.pose) + list(
                task.sim.v) + list(task.sim.angular_v) + list(action)
            for ii in range(len(LABELS)):
                run[LABELS[ii]].append(to_write[ii])

            if done:
                memory_dict['runs'].append(run)
                memory_dict['episode'].append(i_episode)
                memory_dict['reward'].append(agent.score)
                print("\rEpisode={:4d},total_reward={:7.3f} score={:7.3f} (best={:7.3f})"
                      .format(i_episode, agent.total_reward, agent.score, agent.best_score))
                sys.stdout.flush()
                break

    return memory_dict
