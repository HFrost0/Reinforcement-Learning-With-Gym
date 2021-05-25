import random
from collections import deque


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def sample(self, batch_size):
        data = random.sample(self.buffer, batch_size)
        data = tuple(zip(*data))
        return data

    def push(self, data):
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
