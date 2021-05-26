from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update(self):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass
