import os
import torch
import numpy as np
from tqdm import tqdm
from utils import CONST_EPS
import minari
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.mini_batch import TransitionMiniBatch



class OnlineReplayBuffer:
    _device: torch.device
    _buffer: ReplayBuffer

    def __init__(
        self, 
        device: torch.device,
        buffer: ReplayBuffer,
    ) -> None:

        self._device = device
        self._size = (buffer.size()-1)

        observations = buffer.observations
        actions = buffer.actions
        self._buffer = buffer
        self._state = observations[:-1, :]
        self._action = actions[:-1, :]
        self._reward = (buffer.rewards)[:-1, :]
        self._next_state = observations[1:,:]
        self._next_action = actions[1:, :]
        self._not_done = 1. - (buffer.done)[:-1, :]
        self._return = np.zeros((self._size, 1), dtype=np.float32)
        self._advantage = np.zeros((self._size, 1), dtype=np.float32)

    def get_max_min_reward(self) -> (float, float):
        return (self._reward.max(), self._reward.min())

    def compute_return(
        self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * self._not_done[i]
            pre_return = self._return[i]


    def compute_advantage(
        self, gamma:float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._state[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * self._not_done[i] - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * self._not_done[i]

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)


    def sample(
        self, batch_size: int
    ) -> TransitionMiniBatch:
        #Expect s, a, r, s', a, not_done, return, advantage
        ind = np.random.randint(0, self._size, size=batch_size)

        return (
                torch.FloatTensor(self._state[ind]).to(self._device),
                torch.FloatTensor(self._action[ind]).to(self._device),
                torch.FloatTensor(self._reward[ind]).to(self._device),
                torch.FloatTensor(self._next_state[ind]).to(self._device),
                torch.FloatTensor(self._next_action[ind]).to(self._device),
                torch.FloatTensor(self._not_done[ind]).to(self._device),
                torch.FloatTensor(self._return[ind]).to(self._device),
                torch.FloatTensor(self._advantage[ind]).to(self._device),
        )



class OfflineReplayBuffer(OnlineReplayBuffer):

    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)
