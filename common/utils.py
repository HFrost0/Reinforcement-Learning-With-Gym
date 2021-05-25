import gym
import numpy as np


def valid(agent, env, render=False):
    for _ in range(10):
        done = False
        state = env.reset()
        while not done:
            if render:
                env.render()
            action = agent.get_action(state, deter=True)
            state, reward, done, _ = env.step(action)


class NormalizedActions(gym.ActionWrapper):
    def reverse_action(self, action):
        pass

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action
