import numpy as np
import torch
import torch.nn.functional as F
from common.memory import Memory
from common.models import ActorNet, CriticNet


class TD3Agent:
    def __init__(
            self,
            n_obs,
            n_action,
            device='cpu',
            memory_size=100000,
            batch_size=128,
            gamma=0.99,
            soft_tau=1e-2,
            hidden_size=200,
            policy_noise=0.2,
            exploration_noise=0.1,
            policy_delay=2
    ):
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.hidden_size = hidden_size
        self.soft_tau = soft_tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.device = device

        self.actor = ActorNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.q1 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.q2 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)

        self.target_actor = ActorNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.target_q1 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.target_q2 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_q1.load_state_dict(self.target_q1.state_dict())
        self.target_q2.load_state_dict(self.target_q2.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optimizer_q1 = torch.optim.Adam(self.q1.parameters(), lr=1e-4)
        self.optimizer_q2 = torch.optim.Adam(self.q2.parameters(), lr=1e-4)

        self.memory = Memory(self.memory_size)

    def get_action(self, state, deter):
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(dim=0).to(self.device)
        action = self.actor(state)
        action = action.squeeze(dim=0).cpu().detach().numpy()
        if not deter:
            # add exploration noise
            action += np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action.clip(-1., 1.)
        return action

    def update(self, n_updates):
        if len(self.memory) < self.batch_size:
            return
        for i in range(n_updates):
            data = self.memory.sample(self.batch_size)
            data = (torch.FloatTensor(i).to(self.device) for i in data)
            state, action, reward, next_state, done = data
            reward = reward.unsqueeze(1)
            done = done.unsqueeze(1)
            # critic update
            target_action = self.target_actor(next_state)
            target_action += torch.normal(0, self.policy_noise, size=action.shape, device=self.device)
            target_action = target_action.clamp(-1., 1.)
            target_q = torch.min(self.target_q1(next_state, target_action), self.target_q2(next_state, target_action))
            target_q = reward + (1. - done) * self.gamma * target_q
            target_q = target_q.detach()
            criterion_q1 = F.mse_loss(self.q1(state, action), target_q)
            criterion_q2 = F.mse_loss(self.q2(state, action), target_q)
            # q1 backward
            self.optimizer_q1.zero_grad()
            criterion_q1.backward()
            self.optimizer_q1.step()
            # q2 backward
            self.optimizer_q2.zero_grad()
            criterion_q2.backward()
            self.optimizer_q2.step()
            # actor update
            if i % self.policy_delay == 0:
                criterion_actor = -self.q1(state, self.actor(state)).mean()
                self.optimizer_actor.zero_grad()
                criterion_actor.backward()
                self.optimizer_actor.step()
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1. - self.soft_tau) + self.soft_tau * param.data)
                for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                    target_param.data.copy_(target_param.data * (1. - self.soft_tau) + self.soft_tau * param.data)
                for target_param, param in zip(self.target_q2.parameters(), self.q1.parameters()):
                    target_param.data.copy_(target_param.data * (1. - self.soft_tau) + self.soft_tau * param.data)

    def save(self, file_path):
        data = {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
        }
        torch.save(data, file_path)
        print('model saved')

    def load(self, file_path):
        data = torch.load(file_path)
        self.actor.load_state_dict(data['actor'])
        self.target_actor.load_state_dict(data['target_actor'])
        self.q1.load_state_dict(data['q1'])
        self.q2.load_state_dict(data['q2'])
        self.target_q1.load_state_dict(data['q1'])
        self.target_q2.load_state_dict(data['q2'])
        print('model loaded')
