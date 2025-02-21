import os
import math
import time
import copy
import argparse
from collections import deque
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from game_world import GameWorld


class PolicyNet(nn.Module):

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
        x = F.softmax(x, dim=1)

        return x


class ValueNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class Agent:
    """
    the actor-critic agent

    args:
        buffer_size: the buffer size
        policy_learning_rate: the policy learning rate
        value_learning_rate: the value learning rate
        policy_path: the policy model weights path
        value_path: the value model weights path
    """

    def __init__(
        self,
        buffer_size,
        policy_learning_rate=None,
        value_learning_rate=None,
        policy_path=None,
        value_path=None,
    ):
        self.gamma = 0.99
        self.action_size = 2

        self.lr_pi = policy_learning_rate if policy_learning_rate is not None else 0.001
        self.lr_v = value_learning_rate if value_learning_rate is not None else 0.001

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        if policy_path is not None:
            self.pi.load_state_dict(torch.load(policy_path, map_location=torch.device("cpu"), weights_only=True))

        if value_path is not None:
            self.v.load_state_dict(torch.load(value_path, map_location=torch.device("cpu"), weights_only=True))

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.device = None

    def set_device(self, device):
        self.device = device
        self.pi.to(self.device)
        self.v.to(self.device)

    def get_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, probs[action].detach().item()

    def add(self, state, action, action_prob: float, reward: float, next_state, done: bool):
        data = (state, action, action_prob, reward, next_state, done)
        self.buffer.append(data)
        
    def clear(self):
        self.buffer.clear()
        
    def update(self):
        state_v = torch.tensor(np.stack([x[0] for x in self.buffer]), dtype=torch.float32)
        state_pi = torch.tensor(np.stack([x[0] for x in self.buffer]), dtype=torch.float32)
        action = torch.tensor(np.array([x[1] for x in self.buffer]), dtype=torch.int32)
        reward = torch.tensor(np.array([x[3] for x in self.buffer]), dtype=torch.float32)
        next_state = torch.tensor(np.stack([x[4] for x in self.buffer]), dtype=torch.float32)
        done = torch.tensor(np.array([x[5] for x in self.buffer]), dtype=torch.int32)

        next_v = self.v(next_state)
        next_v = torch.flatten(next_v)

        target = reward + self.gamma * next_v * (1 - done)
        target.detach()

        v = self.v(state_v)
        v = torch.flatten(v)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)
        loss_v /= self.buffer_size
        
        delta = target - v
        delta1 = delta.detach().clone()
        
        pis = self.pi(state_pi)
        action_prob = pis[range(len(action)), action]
        
        loss_pi = -torch.log(action_prob) * delta1
        loss_pi = torch.sum(loss_pi)
        loss_pi /= self.buffer_size

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()
            
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the squash using actor-critic algorithm")
    parser.add_argument("-e", "--episode", type=int, default=5000, help="the training episodes number")
    parser.add_argument(
        "-pl",
        "--policy_lr",
        type=float,
        default=0.0001,
        help="the policy learning rate",
    )
    parser.add_argument("-vl", "--value_lr", type=float, default=0.0001, help="the value learning rate")
    parser.add_argument("-s", "--steps", type=int, default=7, help="the learned steps")

    # parse the command args
    args = parser.parse_args()

    AREA_WIDTH = 1200
    AREA_HEIGHT = 600
    PADDLE_HEIGHT = 50
    SYNC_INTERVAL = 100

    episodes = args.episode
    policy_learning_rate = args.policy_lr
    value_learning_rate = args.value_lr
    steps = args.steps

    print(
        f"start to train with episodes [{episodes}] and policy learning rate [{policy_learning_rate}] and value learning rate [{value_learning_rate}]"
    )

    device = torch.device("cpu")
    print(f"the device: {device}")

    torch.autograd.set_detect_anomaly(True)
    
    actor_critic_policy_path = None
    actor_critic_value_path = None
    # the actor-critic agent
    if os.path.exists("./actor_critic_policy_params.pth"):
        actor_critic_policy_path = "./actor_critic_policy_params.pth"

    if os.path.exists("./actor_critic_value_params.pth"):
        actor_critic_value_path = "./actor_critic_value_params.pth"

    agent = Agent(
        steps,
        policy_learning_rate,
        value_learning_rate,
        actor_critic_policy_path,
        actor_critic_value_path,
    )

    # set the agent device
    agent.set_device(device)

    start_time = time.perf_counter()

    total_reward = 0
    total_hits = 0

    for episode in range(episodes):

        env = GameWorld(AREA_WIDTH, AREA_HEIGHT)
        state = env.reset()
        done = False

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, hit = env.step(action=action, step_num=8, paddle_height=PADDLE_HEIGHT)
            
            agent.add(state, action, prob, reward, next_state, done)
            state = copy.deepcopy(next_state)

        total_reward += reward
        total_hits += int(hit)

        env.destroy_bodies()
        del env
        
        agent.update()
        agent.clear()

        if (episode + 1) % SYNC_INTERVAL == 0:
            print(
                f"episode:{episode-SYNC_INTERVAL+1}-{episode}, rewards: {total_reward:.1f}, hits: {total_hits}, duration: {(time.perf_counter() - start_time):.1f} seconds"
            )

            total_reward = 0
            total_hits = 0
            start_time = time.perf_counter()

    torch.save(agent.pi.state_dict(), "actor_critic_policy_params.pth")
    torch.save(agent.v.state_dict(), "actor_critic_value_params.pth")

    torch.save(agent.pi, "actor_critic_policy_model.pth")
    torch.save(agent.v, "actor_critic_value_model.pth")
