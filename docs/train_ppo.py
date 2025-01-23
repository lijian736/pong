import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from game_world import GameWorld


class Policy(nn.Module):

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


class Agent:
    """
    the Proximal Policy Optimization(PPO) agent

    args:
        learning_rate: the learning rate
        policy_path: the policy model weights path
    """

    def __init__(self, learning_rate=None, policy_path=None):
        self.gamma = 0.99
        self.lr = learning_rate if learning_rate is not None else 0.001
        self.action_size = 2

        # the memory for 1 episode
        self.episode_memory = []
        # the memory for all necessary episodes
        self.memory = []

        # the target policy
        self.pi = Policy(self.action_size)
        # the sampling policy
        self.pi_old = Policy(self.action_size)

        if policy_path is not None:
            self.pi_old.load_state_dict(torch.load(policy_path, map_location=torch.device("cpu"), weights_only=True))

        self.optimizer = optim.Adam(self.pi_old.parameters(), lr=self.lr)
        self.device = None

    def set_device(self, device):
        self.device = device
        self.pi.to(self.device)
        self.pi_old.to(self.device)

    def get_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        probs = self.pi_old(state)
        probs = probs[0]
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.episode_memory.append(data)

    def sync_memory(self):
        self.memory.append(self.episode_memory[:])
        self.episode_memory.clear()

    def update(self):
        G, loss = 0, 0
        
        #the reward mean value
        G_mean = 0
        for index, episode_data in enumerate(self.memory):
            reward, _ = episode_data[-1]
            G_mean += (reward - G_mean) / (index + 1)
        
        #REINFORCE with baseline
        for episode_data in self.memory:
            step_mean_reward = G_mean
            for reward, prob in reversed(episode_data):
                G = reward + self.gamma * G
                loss += -torch.log(prob) * (G - step_mean_reward)
                step_mean_reward *= self.gamma

        loss = loss / len(self.memory)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the pong in PPO algorithm")
    parser.add_argument("-e", "--episode", type=int, default=5000, help="the training episodes number")
    parser.add_argument("-l", "--lr", type=float, default=0.0001, help="the learning rate")

    # parse the command args
    args = parser.parse_args()

    episodes = args.episode
    learning_rate = args.lr

    AREA_WIDTH = 1200
    AREA_HEIGHT = 600
    PADDLE_HEIGHT = 50
    SYNC_INTERVAL = 100

    print(f"\nstart to train with episodes [{episodes}] and learning rate [{learning_rate}]")

    device = torch.device("cpu")
    print(f"the device: {device}")

    if os.path.exists("./ppo_model_params.pth"):
        # the PPO agent
        agent = Agent(learning_rate, "./ppo_model_params.pth")
    else:
        agent = Agent(learning_rate)

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

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, hit = env.step(action=action, step_num=3, paddle_height=PADDLE_HEIGHT)
            agent.add(reward, prob)
            state = next_state

        total_reward += reward
        total_hits += int(hit)

        env.destroy_bodies()
        del env

        agent.sync_memory()

        if (episode + 1) % SYNC_INTERVAL == 0:
            print(
                f"episode:{episode-SYNC_INTERVAL+1}-{episode}, rewards: {total_reward:.1f}, hits: {total_hits}, duration: {(time.perf_counter() - start_time):.1f} seconds"
            )
            total_reward = 0
            total_hits = 0
            start_time = time.perf_counter()

            # update the agent
            agent.update()

    torch.save(agent.pi.state_dict(), "ppo_model_params.pth")
    torch.save(agent.pi, "ppo_model.pth")
