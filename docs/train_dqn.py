from typing import List, Dict, Tuple
from collections import deque
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from game_world import GameState, GameWorld


class ReplayBuffer:
    """
    the experience replay buffer
    args:
        buffer_size: the buffer size
        batch_size: the batch size
    """

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        priority: float = 1.0,
    ) -> None:
        """
        args:
            state (_type_): the game current state
            action (int): the action
            reward (float): the reward
            next_state (_type_): the game next state
            done (bool): the game is done or not
            priority (float, optional): the prioritized score. Defaults to 1.0.
        """
        data = (state, action, reward, next_state, done, priority)
        self.buffer.append(data)
        self.buffer_priorities.append(priority)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.buffer_priorities.clear()

    def get_batch(self) -> Tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
    ]:
        probs = np.array(self.buffer_priorities)
        probs_total = probs.sum()
        probs = probs / probs_total

        selected_index = np.random.choice(
            len(self.buffer), size=self.batch_size, replace=False, p=probs
        )
        data = [self.buffer[idx] for idx in selected_index]

        state = torch.tensor(np.stack([x[0] for x in data]).astype(np.float32))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int32))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]).astype(np.float32))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        priorities = torch.tensor(np.array([x[5] for x in data]).astype(np.float32))

        return state, action, reward, next_state, done, priorities


class DQNet(nn.Module):
    """
    the deep Q-network
    args:
        action_size: the action space size
    """

    def __init__(self, action_size):
        super().__init__()
        self.layer1 = nn.Linear(4, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, action_size)

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DQNAgent:
    """
    the deep Q-network agent
    """

    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.001
        self.epsilon = 0.1
        self.buffer_size = 4000
        self.batch_size = 128
        self.action_size = 3  # {0：move up, 1:move down, 2:stay still}

        self.episode_buffer = deque(maxlen=self.buffer_size)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = DQNet(self.action_size)
        self.qnet_target = DQNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.device = None

    def get_action(self, state, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            qs = self.qnet(state)
            return qs.argmax().item()

    def set_device(self, device) -> None:
        self.device = device
        self.qnet.to(self.device)
        self.qnet_target.to(self.device)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        data = (state, action, reward, next_state, done)
        self.episode_buffer.append(data)

    def sync_buffer(self) -> None:
        modified_reward = 0
        for state, action, reward, next_state, done in reversed(self.episode_buffer):
            priority = 1.0
            modified_reward = reward + modified_reward * self.gamma
            self.replay_buffer.add(state, action, reward, next_state, done, priority)

        self.episode_buffer.clear()

    def clear(self) -> None:
        self.replay_buffer.clear()

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done, priority = (
            self.replay_buffer.get_batch()
        )

        state = state.to(self.device)
        qs = self.qnet(state)
        qs = qs.to(torch.device("cpu"))
        q = qs[range(len(action)), action]

        next_state = next_state.to(self.device)
        next_qs = self.qnet_target(next_state)
        next_qs = next_qs.to(torch.device("cpu"))
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self) -> None:
        self.qnet_target.load_state_dict(self.qnet.state_dict())


if __name__ == "__main__":
    AREA_WIDTH = 1200
    AREA_HEIGHT = 600
    PADDLE_HEIGHT = 50

    episodes = 3600
    sync_interval = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"the device: {device}")

    # the game world
    env = GameWorld(AREA_WIDTH, AREA_HEIGHT, PADDLE_HEIGHT)
    # the DQN agent
    agent = DQNAgent()

    # set the agent device
    agent.set_device(device)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        counter = 0

        state = state.to_normalization(
            x_width=AREA_WIDTH,
            y_height=AREA_HEIGHT,
            angle_extent=180,
            paddle_extent=AREA_HEIGHT,
        )
        while not done:
            epsilon = max(0, 0.1 - episode * 0.0001)
            action = agent.get_action(state, epsilon)

            next_state, reward, done = env.step(action=action, step_num=3)

            reward = reward * 100 if reward > 0 else reward * 5
            next_state = next_state.to_normalization(
                x_width=AREA_WIDTH,
                y_height=AREA_HEIGHT,
                angle_extent=180,
                paddle_extent=AREA_HEIGHT,
            )

            agent.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            counter += 1

        env.destroy_bodies()
        
        agent.sync_buffer()
        print(f"episode:{episode}, got {counter} points, total reward: {total_reward}")

        if episode % 4 == 0:
            for _ in range(30):
                agent.update()
            agent.clear()

        if episode % sync_interval == 0:
            agent.sync_qnet()

    torch.save(agent.qnet.state_dict(), "dqn_model_params.pth")
