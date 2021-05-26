from collections import deque
import gym
from agents import TD3Agent
from common.utils import NormalizedActions, valid


env = NormalizedActions(gym.make('LunarLanderContinuous-v2'))
episodes = 5000
exploration_noise = 0.1
n_obs = env.observation_space.shape[0]
n_action = env.action_space.shape[0]
agent = TD3Agent(n_obs, n_action)
ma_deque = deque(maxlen=100)
for ep in range(episodes):
    state = env.reset()
    done = False
    i_step = 0
    ep_reward = 0
    while not done:
        i_step += 1
        action = agent.get_action(state, deter=False)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push((state, action, reward, next_state, done))
        state = next_state
        ep_reward += reward
    ma_deque.append(ep_reward)
    average_reward = sum(ma_deque) / len(ma_deque)
    print('Episode:{}/{}, Step:{} Reward:{} Average Reward:{}'.format(ep + 1, episodes, i_step, ep_reward,
                                                                      average_reward))
    if average_reward > 200:
        agent.save('lunar_td3.pt')
        break
    agent.update(i_step)
# valid
valid(agent, env, render=True)
