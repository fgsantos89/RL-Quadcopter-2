import numpy as np
from physics_sim import PhysicsSim


class Task():
    """
    Task (environment) that defines the goal and provides feedback to
    the agent.
    """

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z)
                                            dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in
                                            (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of
                                            the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # 1 - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()

        # tenho que pensar numa reward que seja mais genérica
        # não somente para decolagem

        # recompensa constante para se manter dentro do quadrante válido
        reward = 1.

        # penalizar pela posicao
        # penalizar pela distancia do eixos x, y, z do target
        # distancia maxima é 300 m em cada direção
        distance = self.sim.pose[:3] - self.target_pos[:3]
        log_transform = np.tan(np.power(distance, 2))
        reward -= 0.3 * log_transform.sum()

        # recompensa pela velocidade do eixo
        # (-15, 15) m/s
        # reward += 0.01 * abs(self.sim.v[:3]).sum()

        # penalizar pela instabilidade dos angulos
        # (0, 6)
        # reward -= 0.01 * abs(self.sim.pose[3:]).sum()

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
