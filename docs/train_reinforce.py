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
    the REINFORCE agent
    
    args:
        learning_rate: the learning rate
        policy_path: the policy model weights path
    """
    def __init__(self, learning_rate=None, policy_path=None):
        self.gamma = 0.992
        self.lr = learning_rate if learning_rate is not None else 0.0001
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        
        if policy_path is not None:
            self.pi.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu'), weights_only=True))
        
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
        self.device = None

    def set_device(self, device):
        self.device = device
        self.pi.to(self.device)

    def get_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -torch.log(prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the pong with REINFORCE algorithm")
    parser.add_argument("-e", "--episode", type=int, default=5000, help="the training episodes number")
    parser.add_argument("-l", "--lr", type=float, default=0.0001, help="the learning rate")
    
    #parse the command args
    args = parser.parse_args()
    
    episodes = args.episode
    learning_rate = args.lr
    
    AREA_WIDTH = 1200
    AREA_HEIGHT = 600
    PADDLE_HEIGHT = 50
    
    print(f"start to train with episodes [{episodes}] and learning rate [{learning_rate}]")

    device = torch.device("cpu")
    print(f"the device: {device}")

    if os.path.exists("./reinforce_model_params.pth"):
        # the REINFORCE agent
        agent = Agent(learning_rate, "./reinforce_model_params.pth")
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
        
        agent.update()

        if (episode + 1) % 100 == 0:
            print(
                f"episode:{episode-99}-{episode}, rewards: {total_reward:.1f}, hits: {total_hits}, duration: {(time.perf_counter() - start_time):.1f} seconds"
            )
            total_reward = 0
            total_hits = 0
            start_time = time.perf_counter()

    torch.save(agent.pi.state_dict(), "reinforce_model_params.pth")
    torch.save(agent.pi, "reinforce_model.pth")
