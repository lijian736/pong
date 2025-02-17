from typing import List, Dict, Tuple
from collections import deque
import random
import math
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from game_world import GameWorld


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

        selected_index = np.random.choice(len(self.buffer), size=self.batch_size, replace=False, p=probs)
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
        self.layer1 = nn.Linear(5, 64)
        self.layer1.name = "layer1"

        self.layer2 = nn.Linear(64, 32)
        self.layer2.name = "layer2"

        self.layer3 = nn.Linear(32, 32)
        self.layer3.name = "layer3"

        self.layer4 = nn.Linear(32, action_size)
        self.layer4.name = "layer4"

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class DQNAgent:
    """
    the deep Q-network agent
    """

    def __init__(self, learning_rate=None, weights_path=None):
        self.gamma = 0.99
        self.lr = learning_rate if learning_rate is not None else 0.001
        self.epsilon = 0.1
        self.buffer_size = 4000
        self.batch_size = 128
        self.action_size = 2

        self.episode_buffer = deque(maxlen=self.buffer_size)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        self.qnet = DQNet(self.action_size)
        if weights_path is not None:
            self.qnet.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu"), weights_only=True))

        self.qnet_target = DQNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.device = None

    def set_device(self, device) -> None:
        self.device = device
        self.qnet.to(self.device)
        self.qnet_target.to(self.device)

    def get_action(self, state, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            qs = self.qnet(state)
            return qs.argmax().item()

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        data = (state, action, reward, next_state, done)
        self.episode_buffer.append(data)

    def sync_buffer(self, scale: float = 0.9) -> None:
        priority = 1.0

        for state, action, reward, next_state, done in reversed(self.episode_buffer):
            self.replay_buffer.add(state, action, reward, next_state, done, priority)
            priority *= scale

        self.episode_buffer.clear()

    def clear(self) -> None:
        self.replay_buffer.clear()

    def update(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done, priority = self.replay_buffer.get_batch()

        qsd = self.qnet(next_state)
        action_max = torch.argmax(qsd, dim=1, keepdim=False)
        
        next_qs = self.qnet_target(next_state)
        next_q = next_qs[range(len(action_max)), action_max]
        next_q.detach()
        
        qs = self.qnet(state)
        q = qs[range(len(action)), action]

        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self) -> None:
        self.qnet_target.load_state_dict(self.qnet.state_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the pong with Double-DQN algorithm")
    parser.add_argument("-e", "--episode", type=int, default=5000, help="the training episodes number")
    parser.add_argument("-l", "--lr", type=float, default=0.0005, help="the learning rate")
    parser.add_argument("-s", "--priority_scale", type=float, default=0.7, help="the learning rate")

    # parse the command args
    args = parser.parse_args()

    episodes = args.episode
    learning_rate = args.lr
    priority_scale = args.priority_scale

    AREA_WIDTH = 1200
    AREA_HEIGHT = 600
    PADDLE_HEIGHT = 50
    SYNC_INTERVAL = 20

    print(
        f"\nstart to train with episodes [{episodes}], learning rate [{learning_rate}] and priority scale [{priority_scale}]"
    )

    device = torch.device("cpu")
    print(f"the device: {device}")

    if os.path.exists("./double_dqn_model_params.pth"):
        # the DQN agent
        agent = DQNAgent(learning_rate, "./double_dqn_model_params.pth")
    else:
        # the DQN agent
        agent = DQNAgent(learning_rate)

    # set the agent device
    agent.set_device(device)

    start_time = time.perf_counter()

    total_reward = 0
    total_hits = 0

    for episode in range(episodes):
        # the game world
        env = GameWorld(AREA_WIDTH, AREA_HEIGHT)
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            epsilon = max(0, 0.2 - episode * 0.00001)
            action = agent.get_action(state, epsilon)

            next_state, reward, done, hit = env.step(action=action, step_num=8, paddle_height=PADDLE_HEIGHT)

            agent.add(state, action, reward, next_state, done)
            state = next_state

        total_reward += reward
        total_hits += int(hit)

        env.destroy_bodies()
        del env

        agent.sync_buffer(priority_scale)

        if (episode + 1) % SYNC_INTERVAL == 0:
            print(
                f"episode:{episode-SYNC_INTERVAL+1}-{episode}, rewards: {total_reward:.1f}, hits: {total_hits}, duration: {(time.perf_counter() - start_time):.1f} seconds"
            )

            for _ in range(40):
                agent.update()
            agent.clear()
            agent.sync_qnet()

            total_reward = 0
            total_hits = 0
            start_time = time.perf_counter()

    torch.save(agent.qnet.state_dict(), "double_dqn_model_params.pth")
    torch.save(agent.qnet, "double_dqn_model.pth")
