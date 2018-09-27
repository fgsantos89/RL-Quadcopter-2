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
        self.action_low = 0  # 0
        self.action_high = 900  # 900
        self.action_size = 4

        # save all the values of the reward to make a plot of the distribution
        # of the parts of the rewards
        self.rewards = []

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10.])

    def get_reward(self):
        """
            Uses current pose of sim to return reward.
        """
        reward_parts = {}

        # penalizar valores muitos altos coodenadas, velocidades

        # initialize a reward based on z-axis distance difference
        # between current and target position
        distance_z = self.sim.pose[2] - self.target_pos[2]

        # (-1, 0)
        factor_distance_z = 0.1 * distance_z
        reward_parts['distance_z'] = factor_distance_z
        reward = factor_distance_z

        # z-axis velocity in to encourage quadcopter to fly towards the target
        # -10 , 20 -> (0.15, 0.3)
        z_axis_velocity = 0.1 * self.sim.v[2]
        reward_parts['z_axis_velocity'] = z_axis_velocity
        reward += z_axis_velocity

        # penalty angular velocity to make sure quadcopter flies straight up
        # 0, 90 -> 0, 0.45
        angular_velocity_sum = 0.2 * abs(self.sim.angular_v).sum()
        reward_parts['angular_velocity_sum'] = angular_velocity_sum
        reward -= angular_velocity_sum

        # subtract the sum of x and y-axis to make goes straight up
        # (-300, 300) -> (-0.3, 0.3)
        distance_x_y_sum = 0.02 * \
            abs(self.target_pos[:2] - self.sim.pose[:2]).sum()
        reward_parts['distance_x_y_sum'] = distance_x_y_sum
        reward -= distance_x_y_sum

        # include some large bonus and penalty rewards also.
        # a bonus on achieving the target height and a penalty on crashing.
        if abs(distance_z) <= 0.1:
            reward_parts['height'] = 100
            reward += 100

        if self.sim.done and self.sim.time < self.sim.runtime:
            reward_parts['crash'] = 100
            reward -= 100

        # clip your final reward between (-1, 1).
        self.rewards.append(reward_parts)
        return np.tanh(reward)

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
