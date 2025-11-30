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


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

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
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay_steps: int = 10000,
            buffer_size: int = 100000,
            batch_size: int = 64,
            target_update_freq: int = 100,
            double_dqn: bool = True,
            use_prioritized: bool = True,
            warmup_steps: int = 1000,
            tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.use_prioritized = use_prioritized
        self.warmup_steps = warmup_steps
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learn_step = 0
        self.total_steps = 0

        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

    def act(self, state: np.ndarray) -> int:
        self.total_steps += 1

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end)
            * self.total_steps / self.epsilon_decay_steps
        )

        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        if len(self.replay_buffer) < self.warmup_steps:
            return 0.0

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        if self.use_prioritized:
            beta = min(1.0, 0.4 + 0.6 * self.learn_step / 10000)
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size, beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

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

        td_errors = current_q - target_q
        loss = (self.loss_fn(current_q, target_q) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        if self.use_prioritized:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.learn_step += 1

        if self.tau > 0:
            self._soft_update()
        elif self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def _soft_update(self):
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']


def run_dqn_experiment():
    hidden_dim = 256
    lr = 1e-4
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 50000
    buffer_size = 100000
    batch_size = 64
    target_update_freq = 100
    warmup_steps = 1000
    tau = 0.005

    runs = 1
    episodes = 100

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file="big-intersection/big-intersection.net.xml",
        route_file="big-intersection/routes.rou.xml",
        use_gui=False,
        num_seconds=5400,
        min_green=5,
        delta_time=5,
        out_csv_name="outputs/big_intersection_dqn",
        single_agent=True
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    all_rewards = []

    for run in range(1, runs + 1):
        print(f"Run {run}/{runs}")

        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            double_dqn=True,
            use_prioritized=True,
            warmup_steps=warmup_steps,
            tau=tau,
        )

        run_rewards = []
        best_reward = float('-inf')

        for ep in range(1, episodes + 1):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, info = reset_result
            else:
                state = reset_result
                info = {}

            episode_reward = 0
            episode_loss = 0
            steps = 0
            done = False
            loss_count = 0

            while not done:
                action = agent.act(state)

                step_result = env.step(action)

                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result

                agent.store(state, action, reward, next_state, done)
                loss = agent.learn()

                if loss > 0:
                    episode_loss += loss
                    loss_count += 1

                episode_reward += reward
                state = next_state
                steps += 1

            run_rewards.append(episode_reward)
            avg_loss = episode_loss / max(loss_count, 1)

            avg_reward_10 = np.mean(run_rewards[-10:]) if len(run_rewards) >= 10 else np.mean(run_rewards)

            print(f"Ep {ep:3d}/{episodes} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg10: {avg_reward_10:8.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Steps: {steps:4d} | "
                  f"Buffer: {len(agent.replay_buffer):6d}")

            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(f"models/dqn_run{run}_best.pth")

        all_rewards.append(run_rewards)

        agent.save(f"models/dqn_run{run}_final.pth")

        np.save(f"outputs/rewards_run{run}.npy", np.array(run_rewards))


    for run in range(runs):
        rewards = all_rewards[run]
        print(f"Run {run + 1}: Mean={np.mean(rewards):.2f}, "
              f"Std={np.std(rewards):.2f}, "
              f"Max={np.max(rewards):.2f}, "
              f"Last10={np.mean(rewards[-10:]):.2f}")

    env.close()


if __name__ == "__main__":
    run_dqn_experiment()