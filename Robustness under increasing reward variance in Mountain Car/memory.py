import numpy as np
import   random

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.acts_buf = np.zeros([size], dtype=np.float64) 
        self.rews_buf = np.zeros([size], dtype=np.float64)
        self.done_buf = np.zeros(size, dtype=np.float64)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self)  :
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return  self.obs_buf[idxs], self.acts_buf[idxs], self.rews_buf[idxs],  self.next_obs_buf[idxs] ,self.done_buf[idxs] 


    def __len__(self) -> int:
        return self.size