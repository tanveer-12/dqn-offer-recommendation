"""Minimal DQN implementation for the customer offer environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Transition:
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in idxs]
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])
        return Transition(states, actions, rewards, next_states, dones)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update: int = 1_000,
        min_replay_size: int = 500,
        grad_clip: float = 5.0,
        hidden_dim: int = 128,
        double_dqn: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.min_replay_size = min_replay_size
        self.grad_clip = grad_clip
        self.double_dqn = double_dqn
        self.device = torch.device(device)

        self.online_q = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_q = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.optimizer = optim.Adam(self.online_q.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        self.steps_done += 1
        self._anneal_epsilon()
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_q(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, transition: Transition) -> None:
        self.replay.push(transition)

    def train_step(self) -> float:
        if len(self.replay) < max(self.batch_size, self.min_replay_size):
            return 0.0
        batch = self.replay.sample(self.batch_size)

        states = torch.from_numpy(batch.state).float().to(self.device)
        actions = torch.from_numpy(batch.action).long().to(self.device)
        rewards = torch.from_numpy(batch.reward).float().to(self.device)
        next_states = torch.from_numpy(batch.next_state).float().to(self.device)
        dones = torch.from_numpy(batch.done.astype(np.float32)).to(self.device)

        q_values = self.online_q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online_q(next_states).argmax(1)
                next_q = self.target_q(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_q(next_states).max(1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online_q.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())

        return float(loss.item())

    def _anneal_epsilon(self) -> None:
        frac = max(0.0, 1.0 - (self.steps_done / float(self.epsilon_decay)))
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * frac

    def save(self, path: str) -> None:
        torch.save({"model": self.online_q.state_dict()}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.online_q.load_state_dict(state["model"])
        self.target_q.load_state_dict(self.online_q.state_dict())
