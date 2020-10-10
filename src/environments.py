import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from src.representations import Representation

class BaseWorld(ABC):

    class action_space():
        def __init__(self, n_actions):
            self.n = n_actions

        def sample(self, k=1):
            return torch.randint(0, self.n, (k,))

    class observation_space():
        def __init__(self, n_features):
            self.shape = [n_features]

    def __init__(self, n_actions, n_observations):
        self.action_space = self.action_space(n_actions)
        self.observation_space = self.observation_space(n_observations)

    @abstractmethod
    def reset(self, state):
        raise NotImplementedError()

    @abstractmethod
    def get_observation(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()


class GridWorld(BaseWorld):

    def __init__(self, dim=5):
        super().__init__(n_actions=4, n_observations=dim**2)
        self.dim = dim
        self.reset()

    def reset(self, ball_coordinates=None):
        if ball_coordinates is None:
            ball_coordinates = [self.dim // 2, self.dim // 2]
        self.ball_coordinates = ball_coordinates
        self.state = self.get_state()
        return self.get_observation()

    def get_observation(self):
        return self.state.flatten()

    def step(self, action):
        if action == 0:
            self.ball_coordinates[0] = (self.ball_coordinates[0] + 1) % self.dim
        elif action == 1:
            self.ball_coordinates[0] = (self.ball_coordinates[0] - 1) % self.dim
        elif action == 2:
            self.ball_coordinates[1] = (self.ball_coordinates[1] + 1) % self.dim
        elif action == 3:
            self.ball_coordinates[1] = (self.ball_coordinates[1] - 1) % self.dim
        else:
            raise Exception("Invalid action.")

        self.state = self.get_state()

        return self.get_observation()

    def get_state(self, ball_coordinates=None):
        if ball_coordinates is None:
            ball_coordinates = self.ball_coordinates

        state = torch.zeros((self.dim, self.dim))
        state[ball_coordinates[0], ball_coordinates[1]] = 1

        return state

class LatentWorld(BaseWorld):
    def __init__(self,
                 dim=4,
                 n_actions=4,
                 action_reps=None):
        super().__init__(n_actions, dim)
        self.dim = dim

        if action_reps is None:
            self.action_reps = [Representation(dim=self.dim) for _ in range(n_actions)]
        else:
            if len(action_reps) != n_actions:
                raise Exception("Must pass an action representation for every action.")
            if not all([rep.dim==self.dim for rep in self.action_reps]):
                raise Exception("Action representations do not act on the dimension of the latent space.")
            self.action_reps = action_reps

        self.reset()

    def reset(self, state=None):
        if state is None:
            state = torch.tensor([1., 0., 1., 0.]) / np.sqrt(2.)
        self.state = state
        return self.get_observation()

    def get_observation(self):
        return self.state

    def step(self, action):
        self.state = torch.mv(self.action_reps[action].get_matrix(), self.state)
        obs = self.get_observation()
        return obs

    def clear_representations(self):
        for rep in self.action_reps:
            rep.clear_matrix()

    def get_representation_params(self):
        params = []
        for rep in self.action_reps:
            params.append(rep.thetas)
        return params

    def get_representations(self):
        return [rep.thetas for rep in self.action_reps]

    def set_representations(self, representations):
        for rep in self.action_reps:
            rep.set_thetas(representations.pop(0))
