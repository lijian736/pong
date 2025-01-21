import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
from game_world import GameState, GameWorld


class PolicyNet(nn.Module):

    def __init__(self, action_size):
        super().__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer1.name = "layer1"

        self.layer2 = nn.Linear(64, 32)
        self.layer2.name = "layer2"

        self.layer3 = nn.Linear(32, action_size)
        self.layer3.name = "layer3"

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.softmax(x, dim=1)

        return x


class ValueNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class Agent:
    """
    the actor-critic agent

    args:
        policy_learning_rate: the policy learning rate
        value_learning_rate: the value learning rate
        policy_path: the policy model weights path
        value_path: the value model weights path
    """

    def __init__(
        self,
        policy_learning_rate=None,
        value_learning_rate=None,
        policy_path=None,
        value_path=None,
    ):
        self.gamma = 0.98
        self.action_size = 3

        self.lr_pi = policy_learning_rate if policy_learning_rate is not None else 0.0005
        self.lr_v = value_learning_rate if value_learning_rate is not None else 0.0005

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        if policy_path is not None:
            self.pi.load_state_dict(
                torch.load(
                    policy_path, map_location=torch.device("cpu"), weights_only=True
                )
            )

        if value_path is not None:
            self.v.load_state_dict(
                torch.load(
                    value_path, map_location=torch.device("cpu"), weights_only=True
                )
            )

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

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
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float32)

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()

        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train the squash in actor-critic algorithm"
    )
    parser.add_argument(
        "-e", "--episode", type=int, default=5000, help="the training episodes number"
    )
    parser.add_argument(
        "-pl",
        "--policy_lr",
        type=float,
        default=0.0001,
        help="the policy learning rate",
    )
    parser.add_argument(
        "-vl", "--value_lr", type=float, default=0.0001, help="the value learning rate"
    )

    # parse the command args
    args = parser.parse_args()

    AREA_WIDTH = 1200
    AREA_HEIGHT = 600
    PADDLE_HEIGHT = 50

    episodes = args.episode
    policy_learning_rate = args.policy_lr
    value_learning_rate = args.value_lr

    print(
        f"start to train with episodes [{episodes}] and policy learning rate [{policy_learning_rate}] and value learning rate [{value_learning_rate}]"
    )

    device = torch.device("cpu")
    print(f"the device: {device}")

    actor_critic_policy_path = None
    actor_critic_value_path = None
    # the actor-critic agent
    if os.path.exists("./actor_critic_policy_params.pth"):
        actor_critic_policy_path = "./actor_critic_policy_params.pth"

    if os.path.exists("./actor_critic_value_params.pth"):
        actor_critic_value_path = "./actor_critic_value_params.pth"

    agent = Agent(
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
        env = GameWorld(AREA_WIDTH, AREA_HEIGHT, PADDLE_HEIGHT)
        state = env.reset()
        done = False

        state = state.to_normalization(
            x_width=AREA_WIDTH,
            y_height=AREA_HEIGHT,
            angle_extent=180,
            paddle_extent=AREA_HEIGHT,
        )

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, hit = env.step(action=action, step_num=3)

            next_state = next_state.to_normalization(
                x_width=AREA_WIDTH,
                y_height=AREA_HEIGHT,
                angle_extent=180,
                paddle_extent=AREA_HEIGHT,
            )

            agent.update(state, prob, reward, next_state, done)
            state = next_state

        total_reward += reward
        total_hits += int(hit)

        if (episode + 1) % 100 == 0:
            print(
                f"episode:{episode-99}-{episode}, rewards: {total_reward}, hits: {total_hits}, duration: {(time.perf_counter() - start_time):.1f} seconds"
            )
            total_reward = 0
            total_hits = 0
            start_time = time.perf_counter()

    torch.save(agent.pi.state_dict(), "actor_critic_policy_params.pth")
    torch.save(agent.v.state_dict(), "actor_critic_value_params.pth")
