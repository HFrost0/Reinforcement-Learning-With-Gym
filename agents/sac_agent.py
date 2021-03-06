import torch
import torch.nn.functional as F
from agents import Agent
from common.memory import Memory
from common.models import SquashedGaussianMLPActor, CriticNet


class SACAgent(Agent):
    def __init__(
            self,
            n_obs, n_action, device='cpu',
            memory_size=100000, batch_size=128, hidden_size=200, lr=5e-4,
            gamma=0.99,
            alpha=0.2,
            polyak=0.995,
    ):
        self.device = device
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.lr = lr
        self.polyak = polyak

        self.actor = SquashedGaussianMLPActor(n_obs, n_action, self.hidden_size).to(self.device)
        self.q1 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.q2 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)

        self.target_q1 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.target_q2 = CriticNet(n_obs, n_action, self.hidden_size).to(self.device)
        self.target_q1.load_state_dict(self.target_q1.state_dict())
        self.target_q2.load_state_dict(self.target_q2.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_q1 = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.optimizer_q2 = torch.optim.Adam(self.q2.parameters(), lr=self.lr)

        self.memory = Memory(self.memory_size)

    @torch.no_grad()
    def get_action(self, state, deter):
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(dim=0).to(self.device)
        action = self.actor(state, deter, with_log_prob=False)
        return action.squeeze(dim=0).cpu().numpy()

    def update(self, n_updates):
        if len(self.memory) < self.batch_size:
            return
        for i in range(n_updates):
            state, action, reward, next_state, done = self.memory.sample(self.batch_size, self.device)
            # update critic
            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_state, with_log_prob=True)
                target_q = torch.min(self.target_q1(next_state, next_action), self.target_q2(next_state, next_action))
                target_q = reward + (1. - done) * self.gamma * (target_q - self.alpha * next_log_prob)
            criterion_q1 = F.mse_loss(self.q1(state, action), target_q)
            criterion_q2 = F.mse_loss(self.q2(state, action), target_q)
            self.optimizer_q1.zero_grad()
            criterion_q1.backward()
            self.optimizer_q1.step()
            self.optimizer_q2.zero_grad()
            criterion_q2.backward()
            self.optimizer_q2.step()

            # update actor
            a, log_prob = self.actor(state, with_log_prob=True)
            q = torch.min(self.q1(state, a), self.q2(state, a))
            criterion_actor = (self.alpha * log_prob - q).mean()
            self.optimizer_actor.zero_grad()
            criterion_actor.backward()
            self.optimizer_actor.step()

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for q, target_q in zip([self.q1, self.q2], [self.target_q1, self.target_q2]):
                    for param, target_param in zip(q.parameters(), target_q.parameters()):
                        target_param.data.mul_(self.polyak)
                        target_param.data.add_((1 - self.polyak) * param.data)

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
        self.q1.load_state_dict(data['q1'])
        self.q2.load_state_dict(data['q2'])
        self.target_q1.load_state_dict(data['q1'])
        self.target_q2.load_state_dict(data['q2'])
        print('model loaded')
