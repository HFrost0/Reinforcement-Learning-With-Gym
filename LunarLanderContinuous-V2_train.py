from collections import deque
import gym
from agents.sac_agent import SACAgent
from common.utils import valid, NormalizedActions


env = NormalizedActions(gym.make('LunarLanderContinuous-v2'))
episodes = 5000
n_obs = env.observation_space.shape[0]
n_action = env.action_space.shape[0]
agent = SACAgent(n_obs, n_action)
# using moving average todo: use independent valid
ma_deque = deque(maxlen=100)
for ep in range(episodes):
    state = env.reset()
    done = False
    i_step = 0
    ep_reward = 0
    while not done:
        i_step += 1
        action = agent.get_action(state, deter=False)
        action = action.clip(-1., 1.)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push((state, action, reward, next_state, done))
        state = next_state
        ep_reward += reward
    ma_deque.append(ep_reward)
    average_reward = sum(ma_deque) / len(ma_deque)
    print('Episode:{}/{}, Step:{} Reward:{} Average Reward:{}'.format(ep + 1, episodes, i_step, ep_reward,
                                                                      average_reward))
    if average_reward > 200:
        print('done')
        agent.save('saved_models/LunarContinuous_sac.pt')
        break
    agent.update(i_step)
# show results
valid(agent, env, render=True)
