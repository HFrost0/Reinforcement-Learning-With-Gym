import random
from collections import deque
import torch


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def sample(self, batch_size, device=None):
        data = random.sample(self.buffer, batch_size)
        data = tuple(zip(*data))
        if device:
            # convert to tensor and to cpu/gpu
            data = self.data2tensor(data, device)
        return data

    def push(self, data):
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    @staticmethod
    def data2tensor(data, device='cpu'):
        data = (torch.FloatTensor(i).to(device) for i in data)
        state, action, reward, next_state, done = data
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        return state, action, reward, next_state, done
