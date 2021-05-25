import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class CriticNet(nn.Module):
    def __init__(self, n_obs, n_action, hidden_size):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(n_obs + n_action, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

        self.lin3.weight.data.uniform_(-3e-3, 3e-3)
        self.lin3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class ActorNet(nn.Module):
    def __init__(self, n_obs, n_action, hidden_size):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(n_obs, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, n_action)

        self.lin3.weight.data.uniform_(-3e-3, 3e-3)
        self.lin3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        # norm to -1 ~ 1
        x = torch.tanh(self.lin3(x))
        return x


class SquashedGaussianMLPActor(nn.Module):
    """
    Modified by implementations of spinning up
    https://spinningup.openai.com/
    """

    def __init__(self, n_obs, n_action, hidden_size):
        super(SquashedGaussianMLPActor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.lin_mu = nn.Linear(hidden_size, n_action)
        self.lin_std = nn.Linear(hidden_size, n_action)

    def forward(self, state, deter=False, with_log_prob=True):
        out = self.mlp(state)
        mu = self.lin_mu(out)
        log_std = self.lin_std(out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action = mu if deter else dist.rsample()
        if with_log_prob:
            log_prob = dist.log_prob(action).sum(axis=-1)
            log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
            return torch.tanh(action), log_prob.unsqueeze(-1)
        else:
            return torch.tanh(action)
