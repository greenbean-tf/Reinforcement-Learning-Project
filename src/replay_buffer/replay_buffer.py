import numpy as np
import torch


class ReplayMemory(object):
    """Fixed-size ring buffer stored in pre-allocated tensors on `device`.

    Same interface as the sample implementation (append / sample / len). Frames are kept as uint8 and the buffer
    can live on the GPU (pass device="cuda"), which makes a large capacity affordable: one transition needs about
    74 KB (two 4x96x96 uint8 frame stacks), i.e. 30,000 transitions ~ 2.2 GB.
    """

    def __init__(self, capacity, device="cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.size = 0
        self.pos = 0
        self.state = None  # tensors are allocated on the first append (shapes are taken from the data)

    def __len__(self):
        return self.size

    def _alloc(self, state, action, reward, done):
        n = self.capacity
        try:
            self.state = torch.empty((n,) + state.shape, dtype=torch.uint8, device=self.device)
            self.next_state = torch.empty_like(self.state)
            self.action = torch.empty((n,) + action.shape, dtype=torch.float32, device=self.device)
            self.reward = torch.empty((n,) + reward.shape, dtype=torch.float32, device=self.device)
            self.done = torch.empty((n,) + done.shape, dtype=torch.float32, device=self.device)
        except RuntimeError as e:  # CUDA out of memory
            self.state = None
            gb = n * (2 * state.nbytes + action.nbytes + reward.nbytes + done.nbytes) / 1e9
            raise RuntimeError(f"cannot allocate a replay buffer of {n} transitions (~{gb:.1f} GB) on {self.device}; "
                               f"use a smaller --buffer") from e

    def append(self, state, action, reward, next_state, done):
        """Saves a transition (overwrites the oldest one when the buffer is full)"""
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        if self.state is None:
            self._alloc(state, action, reward, done)
        i = self.pos
        self.state[i] = torch.from_numpy(state).to(self.device)
        self.next_state[i] = torch.from_numpy(next_state).to(self.device)
        self.action[i] = torch.from_numpy(action).to(self.device)
        self.reward[i] = torch.from_numpy(reward).to(self.device)
        self.done[i] = torch.from_numpy(done).to(self.device)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Sample a batch of transitions: (state, action, reward, next_state, done) as float tensors on `device`"""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch = (self.state[idx].float(), self.action[idx], self.reward[idx],
                 self.next_state[idx].float(), self.done[idx])
        return tuple(x.to(device) for x in batch)
