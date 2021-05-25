# Reinforcement-Learning-With-Gym
Simple and efficient [PyTorch](https://pytorch.org/) deep reinforcement algorithm implementation tested by [OpenAI Gym](https://gym.openai.com/). Easy for starters 🥳.
## Step 1 : Setup up environment
To install Gym, you can follow this [instruction](https://github.com/openai/gym#installing-everything).  
Since some environments like `LunarLanderContinuous-V2` need fully installation，you may need to additionally do this:
```
pip install 'gym[box2d]'
```
## Step 2 : Train your agent or test trained agent 🤖
1. By run `LunarLanderContinuous-V2_train.py` you can train a lunar landing robot by using [SAC](https://arxiv.org/abs/1801.01290) algorithm.
2. By run `LunarLanderContinuous-V2_test.py` you can see the performance of my trained robot after about 180 episodes.
![Alt Text](https://github.com/HFrost0/Reinforcement-Learning-With-Gym/blob/master/saved_models/lunar.gif)
## Step 3 : Explore the detail of the algorithm and do your own job👏
This implementation is well arranged and easy to understand. Enjoy your learning of Deep Reinforcement Algorithm.
From more implementations you can see [spinning up](https://spinningup.openai.com/) which I followed.
