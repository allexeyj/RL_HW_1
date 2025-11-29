import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sumo_rl


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class DQNAgent:

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            lr: float = 3e-4,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.995,
            buffer_size: int = 100000,
            batch_size: int = 64,
            target_update_freq: int = 100,
            double_dqn: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learn_step = 0

        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size)

    def act(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_states).argmax(1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_dqn_experiment():
    hidden_dim = 256
    lr = 3e-4
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    buffer_size = 50000
    batch_size = 64
    target_update_freq = 100
    runs = 3
    episodes = 30

    env = sumo_rl.SumoEnvironment(
        net_file="homeworks/hw1/big-intersection/big-intersection.net.xml",
        route_file="homeworks/hw1/big-intersection/routes.rou.xml",
        use_gui=False,
        num_seconds=5400,
        min_green=5,
        delta_time=5,
        out_csv_name="outputs/big_intersection_dqn",
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for run in range(1, runs + 1):
        print(f"\n=== Run {run}/{runs} ===")

        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            double_dqn=True,
        )

        for ep in range(1, episodes + 1):
            state, info = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store(state, action, reward, next_state, done)
                loss = agent.learn()

                episode_loss += loss
                episode_reward += reward
                state = next_state
                steps += 1

            agent.decay_epsilon()
            avg_loss = episode_loss / max(steps, 1)
            print(f"Episode {ep}/{episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}, Steps: {steps}")

    env.close()


if __name__ == "__main__":
    run_dqn_experiment()