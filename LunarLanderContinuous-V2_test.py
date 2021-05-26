import gym
from agents import SACAgent
from common.utils import valid, NormalizedActions


env = NormalizedActions(gym.make('LunarLanderContinuous-v2'))
n_obs = env.observation_space.shape[0]
n_action = env.action_space.shape[0]
agent = SACAgent(n_obs, n_action)
agent.load('saved_models/LunarContinuous_sac.pt')
# show results
valid(agent, env, render=True)
