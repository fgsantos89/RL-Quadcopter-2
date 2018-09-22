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
        """
            Uses current pose of sim to return reward.
        """
        # initialize a reward based on z-axis distance difference
        # between current and target position
        # (-10, 0) -> (-1, 1)
        distance_z = self.sim.pose[2] - self.target_pos[2]
        reward = distance_z

        # z-axis velocity in to encourage quadcopter to fly towards the target
        reward += 2.5 * self.sim.v[2]

        # penalty angular velocity to make sure quadcopter flies straight up
        reward -= abs(self.sim.angular_v).sum()

        # subtract the sum of x and y-axis to make goes straight up
        reward -= abs(self.target_pos[:2] - self.sim.pose[:2]).sum()

        # include some large bonus and penalty rewards also.
        # a bonus on achieving the target height and a penalty on crashing.
        if abs(distance_z) <= 0.1:
            reward += 100

        if self.sim.done and self.sim.time < self.sim.runtime:
            reward -= 100

        # clip your final reward between (-1, 1).
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
