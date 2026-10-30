import os
import torch
import numpy as np
from tqdm import tqdm
from utils import CONST_EPS
import minari
from einops import rearrange
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.mini_batch import TransitionMiniBatch



class OnlineReplayBuffer:
    _device: torch.device
    _buffer: ReplayBuffer

    def __init__(
        self, 
        buffer: ReplayBuffer,
    ) -> None:

        self._device = device
        self._buffer = buffer

    def store(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_p: np.ndarray,
        a_p: np.ndarray,
        not_done: bool
    ) -> None:

        self._state[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._next_state[self._size] = s_p
        self._next_action[self._size] = a_p
        self._not_done[self._size] = not_done
        self._size += 1


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

        return _buffer.sample_transition_batch(batch_size)



class OfflineReplayBuffer(OnlineReplayBuffer):

    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)
